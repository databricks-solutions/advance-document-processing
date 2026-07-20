# Databricks notebook source
# MAGIC %md
# MAGIC # Parsing PDFs with `ai_parse_document` + adaptive `pageRange` routing
# MAGIC
# MAGIC `ai_parse_document` has a hard **500-page-per-call limit**. The `pageRange` option (added in v2.0) lets you parse longer documents by chunking, but it must be a literal (analysis-time foldable) — you can't column-drive it.
# MAGIC
# MAGIC This notebook splits documents into two paths so short PDFs (the common case) aren't blocked by long ones:
# MAGIC
# MAGIC 1. **Bronze** — raw binary + detected `page_count` written to a Delta table.
# MAGIC 2. **Short path** (`page_count <= PAGE_RANGE_LIMIT`): single `ai_parse_document` call, no `pageRange`.
# MAGIC 3. **Long path**  (`page_count > PAGE_RANGE_LIMIT`): generate 500-page chunks, parse each with a literal `pageRange`, merge.
# MAGIC 4. **Silver** — union of both paths with a uniform schema (VARIANT).
# MAGIC 5. **Full text** — reassembled per document for full content.
# MAGIC
# MAGIC Reference: [`ai_parse_document` docs](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document).

# COMMAND ----------

# MAGIC %pip install pypdf

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_folder", "large_pdfs")
dbutils.widgets.text("page_range_limit", "500")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_FOLDER = dbutils.widgets.get("volume_folder")
PAGE_RANGE_LIMIT = int(dbutils.widgets.get("page_range_limit"))

SOURCE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{VOLUME_FOLDER}"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_raw_pdfs"
SILVER_SHORT_TABLE = f"{CATALOG}.{SCHEMA}.silver_parsed_short_docs"
SILVER_LONG_TABLE = f"{CATALOG}.{SCHEMA}.silver_parsed_long_docs"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_parsed_docs"
FULL_TEXT_TABLE = f"{CATALOG}.{SCHEMA}.pagerange_full"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Source:              {SOURCE_PATH}")
print(f"Page limit:          {PAGE_RANGE_LIMIT} pages per ai_parse_document call")
print(f"Bronze table:        {BRONZE_TABLE}")
print(f"Silver short table:  {SILVER_SHORT_TABLE}")
print(f"Silver long table:  {SILVER_LONG_TABLE}")
print(f"Silver table:        {SILVER_TABLE}")
print(f"Full-text:           {FULL_TEXT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load raw PDFs from the volume

# COMMAND ----------

docs_raw = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .option("recursiveFileLookup", True)
    .load(SOURCE_PATH)
)

display(docs_raw.selectExpr("path", "length", "modificationTime"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Detect page count per PDF
# MAGIC
# MAGIC Use `pypdf` with a Pandas UDF. Bad/encrypted files return `-1` so they surface in downstream filters without killing the job.

# COMMAND ----------

import io

import pandas as pd
from pypdf import PdfReader
from pyspark.sql.functions import col, element_at, pandas_udf, split
from pyspark.sql.types import IntegerType


@pandas_udf(IntegerType())
def pdf_page_count(content: pd.Series) -> pd.Series:
    def _count(b: bytes) -> int:
        try:
            return len(PdfReader(io.BytesIO(b)).pages)
        except Exception:
            return -1

    return content.apply(_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Write bronze table (raw binary + page count)
# MAGIC
# MAGIC Persisting the bronze layer means downstream parsing can re-run without re-reading the volume or recomputing page counts. Bad files (`page_count = -1`) stay in bronze for visibility and are filtered before parsing.

# COMMAND ----------

bronze = (
    docs_raw
    .withColumn("file_name", element_at(split(col("path"), "/"), -1))
    .withColumn("page_count", pdf_page_count("content"))
    .select("file_name", "path", "length", "modificationTime", "page_count", "content")
)

(
    bronze.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

display(spark.read.table(BRONZE_TABLE).drop("content"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Route by page count
# MAGIC
# MAGIC - `page_count <= PAGE_RANGE_LIMIT` → **short path**, single call.
# MAGIC - `page_count >  PAGE_RANGE_LIMIT` → **long path**, chunked.
# MAGIC
# MAGIC Both paths read independently from bronze, so a long PDF never blocks the short pipeline.

# COMMAND ----------

bronze_df = spark.read.table(BRONZE_TABLE).filter("page_count > 0")
short_docs = bronze_df.filter(col("page_count") <= PAGE_RANGE_LIMIT)
long_docs = bronze_df.filter(col("page_count") > PAGE_RANGE_LIMIT)

n_short = short_docs.count()
n_long = long_docs.count()
print(f"Short docs (<= {PAGE_RANGE_LIMIT} pages): {n_short}")
print(f"Long docs  (>  {PAGE_RANGE_LIMIT} pages): {n_long}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Short path: single `ai_parse_document` call
# MAGIC
# MAGIC No `pageRange` needed — the document fits in one call.

# COMMAND ----------

from pyspark.sql.functions import expr

short_parsed = (
    short_docs
    .withColumn("parsed", expr("ai_parse_document(content, map('version', '2.0'))"))
    .selectExpr(
        "file_name",
        "page_count",
        "cast(1 AS BIGINT) AS chunk_count",
        "parsed",
        "cast(parsed:error_status AS STRING) AS parse_error",
    )
)

short_ok = short_parsed.filter("parse_error IS NULL").drop("parse_error")
(
    short_ok.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_SHORT_TABLE)
)
display(short_parsed.select("file_name", "page_count", "chunk_count", "parse_error"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Long path: chunk → parse → merge
# MAGIC
# MAGIC ### Why a loop here?
# MAGIC
# MAGIC `ai_parse_document` requires its options map to be a **foldable literal**, so you can't pass `pageRange` as a column reference like `map('pageRange', page_range_col)`. The workaround: enumerate the distinct page-range strings on the driver and build one Spark plan per literal, then `unionByName`. Because chunks are always `1-500`, `501-1000`, ... the distinct values are deterministic — we compute them from the **max** page count in `long_docs` instead of doing a `collect_set` on the data side. This bounds the loop at `ceil(max_pages / PAGE_RANGE_LIMIT)` iterations regardless of how many long PDFs you have.

# COMMAND ----------

# DBTITLE 1,Step 6 — chunk_page_ranges helper
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.functions import expr, udf
from pyspark.sql.types import ArrayType, StringType


def chunk_page_ranges(n_pages: int, chunk_size: int = PAGE_RANGE_LIMIT) -> list[str]:
    if n_pages is None or n_pages <= 0:
        return []
    return [
        f"{start}-{min(start + chunk_size - 1, n_pages)}"
        for start in range(1, n_pages + 1, chunk_size)
    ]


# Driver-side sanity checks
assert chunk_page_ranges(500) == ["1-500"]
assert chunk_page_ranges(750) == ["1-500", "501-750"]
assert chunk_page_ranges(1500) == ["1-500", "501-1000", "1001-1500"]
print("chunk_page_ranges sanity checks passed.")

chunk_ranges_udf = udf(lambda n: chunk_page_ranges(n), ArrayType(StringType()))

# COMMAND ----------

exploded = (
    long_docs
    .withColumn("page_ranges", chunk_ranges_udf("page_count"))
    .selectExpr(
        "file_name",
        "path",
        "content",
        "page_count",
        "posexplode(page_ranges) as (chunk_idx, page_range)",
    )
)

# Bounded set of literal page-range strings, derived from the global max page count
# (no row-side collect_set). Up to ceil(max_pages / PAGE_RANGE_LIMIT) values.
max_pages_row = long_docs.selectExpr("max(page_count) AS m").first()
max_pages = (max_pages_row["m"] or 0) if max_pages_row else 0
literal_ranges = chunk_page_ranges(int(max_pages))
print(f"Literal page-range plans needed: {len(literal_ranges)} -> {literal_ranges}")

if literal_ranges:
    parsed_dfs = []
    for pr in literal_ranges:
        df = (
            exploded
            .filter(col("page_range") == pr)
            .withColumn(
                "parsed",
                expr(f"ai_parse_document(content, map('version', '2.0', 'pageRange', '{pr}'))"),
            )
        )
        parsed_dfs.append(df)
    long_per_chunk = reduce(DataFrame.unionByName, parsed_dfs)
else:
    # No documents exceeded the limit — empty DF with a VARIANT 'parsed' column so
    # the downstream SQL view has a matching schema.
    long_per_chunk = exploded.withColumn("parsed", expr("parse_json(NULL)"))

long_per_chunk = long_per_chunk.selectExpr(
    "file_name",
    "page_count",
    "chunk_idx",
    "page_range",
    "parsed",
    "cast(parsed:error_status AS STRING) AS parse_error",
)
long_per_chunk.createOrReplaceTempView("long_per_chunk")
display(long_per_chunk.select("file_name", "page_count", "chunk_idx", "page_range", "parse_error"))

# COMMAND ----------

# MAGIC %md
# MAGIC Merge the long path's chunks back into one row per document, ordered by `chunk_idx`, and **repackage as a single `VARIANT`** with the same `{ document: { elements, pages }, error_status }` shape that a single `ai_parse_document` call would return. This makes the silver schema identical to the short path — and identical to `pipeline.py`'s bronze layer — so downstream `ai_classify(parsed, ...)` / `ai_extract(parsed, ...)` can treat both paths uniformly.
# MAGIC
# MAGIC If `long_per_chunk` is empty, `long_merged` is also empty with a matching schema and the union in Step 7 simply returns the short-only result.

# COMMAND ----------

# DBTITLE 1,Cell 20
spark.sql(
f"""
CREATE OR REPLACE TABLE {SILVER_LONG_TABLE} AS
SELECT
  file_name,
  page_count,
  COUNT(*) AS chunk_count,
  parse_json(to_json(named_struct(
    'document', named_struct(
      'elements', flatten(transform(
        array_sort(
          collect_list(struct(chunk_idx AS idx, try_cast(parsed:document:elements AS ARRAY<VARIANT>) AS elements)),
          (l, r) -> l.idx - r.idx
        ),
        x -> x.elements
      )),
      'pages', flatten(transform(
        array_sort(
          collect_list(struct(chunk_idx AS idx, try_cast(parsed:document:pages AS ARRAY<VARIANT>) AS pages)),
          (l, r) -> l.idx - r.idx
        ),
        -- Offset page indices so they are globally sequential across chunks:
        -- chunk 0 → indices 0-(PAGE_RANGE_LIMIT-1), chunk 1 → PAGE_RANGE_LIMIT … 2*PAGE_RANGE_LIMIT-1, etc.
        -- NOTE: the named_struct below covers the most common page fields; inspect
        -- parsed:document:pages[0] on a sample row and add any additional fields
        -- your document type exposes before using this in production.
        x -> transform(x.pages, (p, i) ->
          parse_json(to_json(named_struct(
            'page', named_struct(
              'pageNumber', x.idx * {PAGE_RANGE_LIMIT} + i,
              'width',      p:page:width,
              'height',     p:page:height
            )
          )))
        )
      ))
    ),
    'error_status', CAST(NULL AS STRING)
  ))) AS parsed
FROM long_per_chunk
WHERE parse_error IS NULL
GROUP BY file_name, page_count;
"""
)

display(spark.sql(
  f"""
  SELECT
    file_name,
    page_count,
    chunk_count,
    size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS element_count,
    size(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS page_blocks
  FROM {SILVER_LONG_TABLE}
  ORDER BY file_name;
  """
))

# COMMAND ----------

# DBTITLE 1,Step 6 — Partial-chunk failure guard
# ── Partial-chunk failure guard ─────────────────────────────────────────────
# After stitching, every document must have exactly ceil(page_count / PAGE_RANGE_LIMIT)
# chunks. Fewer means at least one chunk parse failed silently and the document
# in SILVER_LONG_TABLE is missing pages.  Print a warning (or raise in prod).

gaps = spark.sql(f"""
    SELECT
        file_name,
        page_count,
        chunk_count                                          AS stitched_chunks,
        ceil(page_count / {PAGE_RANGE_LIMIT})                AS expected_chunks,
        ceil(page_count / {PAGE_RANGE_LIMIT}) - chunk_count  AS missing_chunks
    FROM {SILVER_LONG_TABLE}
    WHERE chunk_count < ceil(page_count / {PAGE_RANGE_LIMIT})
""")

n_gaps = gaps.count()
if n_gaps > 0:
    print(f"WARNING: {n_gaps} document(s) have missing chunk(s) — downstream results will be incomplete.")
    display(gaps)
    # Uncomment to hard-fail in production:
    # raise RuntimeError(f"{n_gaps} long-path document(s) are missing chunks after stitching.")
else:
    total = spark.table(SILVER_LONG_TABLE).count()
    print(f"OK: all {total} long-path document(s) stitched completely (chunk_count == expected).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Silver: union short + long
# MAGIC
# MAGIC Both sides have the same schema: `file_name, page_count, chunk_count, parsed`. Short-path rows have `chunk_count = 1` and `parsed` is the original `ai_parse_document` output. Long-path rows have `chunk_count = ceil(page_count / PAGE_RANGE_LIMIT)` and `parsed` is the chunks reassembled into the same VARIANT shape.

# COMMAND ----------

silver = spark.table(SILVER_SHORT_TABLE).unionByName(spark.table(SILVER_LONG_TABLE))

(
    silver.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

display(
    spark.read.table(SILVER_TABLE)
    .selectExpr(
        "file_name",
        "page_count",
        "chunk_count",
        "size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS element_count",
    )
    .orderBy("file_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Reassemble full text per document
# MAGIC
# MAGIC Concatenate the `content` of every parsed element back into one ordered string per document. Same `full_text` shape downstream `ai_classify` / `ai_extract` already expect (see `pipeline.py`).

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {FULL_TEXT_TABLE} AS
    SELECT
      file_name,
      page_count,
      chunk_count,
      parsed,
      concat_ws(
        '\\n\\n',
        transform(
          try_cast(parsed:document:elements AS ARRAY<STRUCT<content:STRING>>),
          el -> el.content
        )
      ) AS full_text
    FROM {SILVER_TABLE}
    ORDER BY file_name
""")

display(spark.read.table(FULL_TEXT_TABLE).selectExpr("file_name", "page_count", "chunk_count", "length(full_text) AS full_text_len"))

# COMMAND ----------

# DBTITLE 1,Notes & limits
# MAGIC %md
# MAGIC ## Notes & limits
# MAGIC
# MAGIC - **`pageRange` must be a literal.** `ai_parse_document` evaluates its options map at analysis time; `map('pageRange', some_column)` will not work. The long-path loop is the canonical workaround. Iteration count is bounded by `ceil(max_pages / PAGE_RANGE_LIMIT)`, not by the number of long PDFs.
# MAGIC - The 500-page limit is **per call**, not per document. Chunked calls are independent.
# MAGIC - Omitting `pageRange` on a PDF that exceeds 500 pages causes `ai_parse_document` to fail immediately without parsing anything. The router in Step 4 prevents that case from reaching the short path.
# MAGIC - `pypdf` reads page count from the trailer/xref — usually cheap. Encrypted or malformed PDFs surface as `page_count = -1` in bronze and are filtered out before Step 4.
# MAGIC - **Silver `parsed` is a single VARIANT, schema-compatible with `pipeline.py`'s bronze.** Downstream `ai_classify(parsed, ...)` / `ai_extract(parsed, ...)` work identically against short and long-path rows. The reassembled VARIANT preserves `document.elements` and `document.pages` ordered by chunk; if `ai_parse_document` v2.0 adds other top-level fields in the future, extend the `named_struct` in Step 6's SQL to forward them.
# MAGIC - **Page-offset correction for stitched documents.** Without correction the second chunk's `document.pages` entries restart at index 0 instead of `PAGE_RANGE_LIMIT`. The stitching SQL in Step 6 corrects this via `transform(x.pages, (p, i) -> …)` using `x.idx * PAGE_RANGE_LIMIT + i` as the adjusted `pageNumber`. The `named_struct` retains `page.width` and `page.height`; inspect `parsed:document:pages[0]` on a sample row and extend with any additional fields your document type exposes.
# MAGIC - **Partial-chunk failure detection.** If one chunk of a long document fails, the stitching `GROUP BY` silently drops it — producing a document with missing pages and no error signal. The validation cell after Step 6 catches this by comparing each document's `chunk_count` to `ceil(page_count / PAGE_RANGE_LIMIT)` and printing a warning for any gaps. For production, convert the warning to an explicit assertion or a Lakeflow Spark Declarative Pipeline expectation.