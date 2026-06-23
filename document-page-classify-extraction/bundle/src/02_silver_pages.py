# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: page split (stateless)
# MAGIC
# MAGIC Streaming version of NB02. An append-only stream reads the bronze table
# MAGIC and emits **one row per page** with the same two views as the batch
# MAGIC notebook:
# MAGIC
# MAGIC - `page_variant` — page-scoped VARIANT mirroring the parse schema (all
# MAGIC   element types), for `ai_extract`.
# MAGIC - `page_text` — chrome-stripped, figure/table-inlined text, for
# MAGIC   `ai_classify`.
# MAGIC
# MAGIC **Stateless by construction:** both views are built with *within-row*
# MAGIC array higher-order functions over each bronze row's element array — no
# MAGIC `explode` + `GROUP BY`, so no streaming state store / watermark. (The
# MAGIC batch NB02 uses a groupBy because it is a batch query; this rebuild keeps
# MAGIC the stream a pure per-row map.)

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "mortgage_loan_files")
dbutils.widgets.text("table_prefix", "loan_docs_stream")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
volume        = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix  = dbutils.widgets.get("table_prefix")

bronze_table   = f"{catalog}.{schema}.{table_prefix}_bronze_parsed_docs"
pages_table    = f"{catalog}.{schema}.{table_prefix}_silver_pages"
artifacts_root = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt           = f"{artifacts_root}/checkpoints/silver_pages"

print(f"bronze_table = {bronze_table}")
print(f"pages_table  = {pages_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target table exists

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {pages_table} (
    path STRING,
    page_id INT,
    page_variant VARIANT,
    page_text STRING,
    page_text_len INT,
    image_uri STRING,
    has_table BOOLEAN,
    has_figure BOOLEAN,
    element_count INT
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-page expressions (within-row, no groupBy)

# COMMAND ----------

CHROME = "('page_header','page_footer','page_number')"

# All elements whose first bbox lands on this page.
elems_all_expr = (
    "filter(variant_get(parsed,'$.document.elements','array<variant>'), "
    "e -> variant_get(e,'$.bbox[0].page_id','int') = page_obj:id::int) AS elems_all"
)

# Page-scoped VARIANT mirroring the ai_parse_document schema (keeps ALL element
# types — ai_extract benefits from the structure).
page_variant_expr = """
PARSE_JSON(TO_JSON(named_struct(
  'document', named_struct('pages', array(page_obj), 'elements', elems_all),
  'metadata', variant_get(parsed,'$.metadata','variant'),
  'error_status', filter(variant_get(parsed,'$.error_status','array<variant>'),
                         s -> s:page_id::int = page_id)
))) AS page_variant
""".strip()

# Plain-text view: drop chrome, sort by element id, inline figure/table content.
page_text_expr = """
array_join(
  transform(
    array_sort(
      filter(elems_all, e -> NOT (e:type::string IN """ + CHROME + """)),
      (l, r) -> CASE WHEN l:id::int < r:id::int THEN -1
                     WHEN l:id::int > r:id::int THEN  1 ELSE 0 END
    ),
    e -> CASE
           WHEN e:type::string = 'figure' AND e:description::string IS NOT NULL
             THEN concat('[figure] ', e:description::string)
           WHEN e:type::string = 'table' AND e:content::string IS NOT NULL
             THEN concat('[table] ', e:content::string)
           ELSE e:content::string
         END
  ),
  '\\n'
) AS page_text
""".strip()

has_table_expr   = "exists(elems_all, e -> e:type::string = 'table') AS has_table"
has_figure_expr  = "exists(elems_all, e -> e:type::string = 'figure') AS has_figure"
element_count_expr = (
    "size(filter(elems_all, e -> NOT (e:type::string IN " + CHROME + "))) AS element_count"
)

# COMMAND ----------

from pyspark.sql.functions import length, col

pages_stream = (
    spark.readStream.table(bronze_table)
    .where("parsed:document:pages IS NOT NULL AND CAST(parsed:error_status AS STRING) IS NULL")
    # explode pages
    .selectExpr("path", "parsed", "posexplode(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS (pos, page_obj)")
    # per-page keys + the page's element array
    .selectExpr(
        "path",
        "parsed",
        "page_obj",
        "page_obj:id::int        AS page_id",
        "page_obj:image_uri::string AS image_uri",
        elems_all_expr,
    )
    # build the two views + flags
    .selectExpr(
        "path",
        "page_id",
        "image_uri",
        page_variant_expr,
        page_text_expr,
        has_table_expr,
        has_figure_expr,
        element_count_expr,
    )
    .withColumn("page_text_len", length(col("page_text")))
    .select(
        "path", "page_id", "page_variant", "page_text", "page_text_len",
        "image_uri", "has_table", "has_figure", "element_count",
    )
)

query = (
    pages_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", ckpt)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(pages_table)
)

query.awaitTermination()
print("Silver pages write complete.")
print(f"Last progress: {query.lastProgress}")

cnt = spark.table(pages_table).count()
print(f"silver_pages row count = {cnt}")
dbutils.notebook.exit(f"pages={cnt}")
