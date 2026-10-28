"""Parse PDFs longer than ai_parse_document's 500-page-per-call limit.

Paste this whole file into one notebook cell (Genie Code or Databricks notebook),
then call `parse_documents(spark, bronze_df)` -- see SKILL.md. Paste rather than
import: a helper imported from a module and used inside a UDF fails on serverless
unless it is registered with cloudpickle `register_pickle_by_value`.

Pure functions (no Spark imports at module level) are unit-tested in
skills/tests/test_parse_large_pdfs.py.
"""

import io
import json
import math

PAGE_LIMIT = 500  # ai_parse_document hard limit: pages per call


# ---------------------------------------------------------------- pure helpers


def chunk_page_ranges(n_pages, chunk_size=PAGE_LIMIT):
    """['1-500', '501-1000', ...] covering n_pages (1-indexed, inclusive)."""
    if n_pages is None or n_pages <= 0:
        return []
    return [
        f"{start}-{min(start + chunk_size - 1, n_pages)}"
        for start in range(1, n_pages + 1, chunk_size)
    ]


def expected_chunks(n_pages, chunk_size=PAGE_LIMIT):
    return math.ceil(n_pages / chunk_size) if n_pages and n_pages > 0 else 0


def count_pdf_pages(content):
    """Page count from PDF bytes; -1 for encrypted/corrupt files (never raises)."""
    from pypdf import PdfReader

    try:
        return len(PdfReader(io.BytesIO(content)).pages)
    except Exception:
        return -1


def stitch_chunks(chunk_docs):
    """Merge per-chunk ai_parse_document v2.0 outputs into one document.

    chunk_docs: list of (chunk_idx, parsed_dict) in any order.
    Returns a dict with the same top-level shape a single call returns
    ({document: {pages, elements}, metadata, error_status}).

    With pageRange, page ids and element bbox page_ids already come back
    document-global (0-based: physical page 501 -> id 500), so they pass through.
    Element ids restart at 0 in every chunk, so they are renumbered sequentially.
    """
    pages, elements, errors = [], [], []
    metadata = None
    next_element_id = 0
    for _, parsed in sorted(chunk_docs, key=lambda c: c[0]):
        doc = parsed.get("document") or {}
        if metadata is None:
            metadata = parsed.get("metadata")
        pages.extend(doc.get("pages") or [])
        for el in doc.get("elements") or []:
            elements.append({**el, "id": next_element_id})
            next_element_id += 1
        errors.extend(parsed.get("error_status") or [])
    return {
        "document": {"pages": pages, "elements": elements},
        "metadata": metadata,
        "error_status": errors or None,
    }


# ---------------------------------------------------------------- spark pipeline


def add_page_count(df, content_col="content"):
    """Adds `page_count` (pypdf). Install pypdf first: %pip install pypdf."""
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import IntegerType

    @pandas_udf(IntegerType())
    def _page_count(content: pd.Series) -> pd.Series:
        return content.apply(count_pdf_pages)

    return df.withColumn("page_count", _page_count(content_col))


def parse_documents(spark, bronze_df, chunk_size=PAGE_LIMIT, options=None):
    """bronze_df: (path, content, page_count). Returns one row per document:
    (path, page_count, chunk_count, parsed VARIANT) -- same parsed shape for
    short and long documents. Rows with page_count <= 0 are excluded; report them
    from bronze separately.
    """
    from functools import reduce

    from pyspark.sql import DataFrame
    from pyspark.sql.functions import col, expr, udf
    from pyspark.sql.types import ArrayType, StringType

    opts = {"version": "2.0", **(options or {})}

    def _opts_sql(extra=None):
        merged = {**opts, **(extra or {})}
        return ", ".join(f"'{k}', '{v}'" for k, v in merged.items())

    docs = bronze_df.filter("page_count > 0")
    short = docs.filter(col("page_count") <= chunk_size)
    long_ = docs.filter(col("page_count") > chunk_size)

    short_parsed = short.selectExpr(
        "path",
        "page_count",
        "CAST(1 AS BIGINT) AS chunk_count",
        f"ai_parse_document(content, map({_opts_sql()})) AS parsed",
    )

    # pageRange must be a literal: one Spark plan per distinct range string.
    # Ranges are always 1-500, 501-1000, ... so the set is bounded by the longest
    # document, not by how many long documents there are.
    max_pages = long_.selectExpr("max(page_count) AS m").first()["m"] or 0
    if max_pages == 0:
        return short_parsed

    ranges_udf = udf(lambda n: chunk_page_ranges(n, chunk_size), ArrayType(StringType()))
    exploded = long_.withColumn("ranges", ranges_udf("page_count")).selectExpr(
        "path", "page_count", "content", "posexplode(ranges) AS (chunk_idx, page_range)"
    )
    per_chunk = reduce(
        DataFrame.unionByName,
        [
            exploded.filter(col("page_range") == pr).withColumn(
                "parsed", expr(f"ai_parse_document(content, map({_opts_sql({'pageRange': pr})}))")
            )
            for pr in chunk_page_ranges(max_pages, chunk_size)
        ],
    ).selectExpr("path", "page_count", "chunk_idx", "to_json(parsed) AS parsed_json")

    @udf(StringType())
    def _stitch(chunks):
        return json.dumps(
            stitch_chunks(
                [(c["chunk_idx"], json.loads(c["parsed_json"])) for c in chunks if c["parsed_json"]]
            )
        )

    # chunk_count counts only chunks that returned output, so a chunk that came
    # back NULL shows up in incomplete_documents() instead of vanishing.
    long_parsed = (
        per_chunk.groupBy("path", "page_count")
        .agg(
            expr("collect_list(struct(chunk_idx, parsed_json)) AS chunks"),
            expr("count(parsed_json) AS chunk_count"),
        )
        .withColumn("stitched", _stitch("chunks"))
        .selectExpr("path", "page_count", "chunk_count", "parse_json(stitched) AS parsed")
    )
    return short_parsed.unionByName(long_parsed)


def incomplete_documents(parsed_df, chunk_size=PAGE_LIMIT):
    """Long documents missing chunks or carrying chunk errors. Empty = complete."""
    return parsed_df.filter(
        f"chunk_count < ceil(page_count / {chunk_size}) "
        "OR size(try_cast(parsed:error_status AS ARRAY<VARIANT>)) > 0"
    )
