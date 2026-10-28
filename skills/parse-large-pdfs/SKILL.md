---
name: parse-large-pdfs
description: Use when parsing PDFs longer than 500 pages with ai_parse_document, when ai_parse_document fails on long documents, or when a document corpus mixes short files with very long ones (SEC 10-Ks, underwriting submissions, loan files, contracts, regulatory filings) and every document must be parsed in full into one table.
---

# Parsing PDFs Longer Than 500 Pages

## Overview

`ai_parse_document` parses at most **500 pages per call**. On a longer PDF the query
still **succeeds**, but the row holds no content — only
`error_status: [{"error_message": "Document has 620 pages, which exceeds the limit of 500.", ...}]`.
Pipelines that don't check `error_status` lose long documents silently; nothing
is truncated. The fix is to parse long documents in 500-page
`pageRange` chunks and stitch the chunks back into **one VARIANT per document with
the same shape a single call returns**, so downstream `ai_extract` / `ai_classify`
treat short and long documents identically.

For `ai_parse_document` syntax and options, use the `databricks-ai-functions` skill.

## When to use

- Any document in the corpus has more than 500 pages (check with `pypdf`, not by guessing).
- Not needed when every file is ≤ 500 pages — a single call per file is enough.
- A file over **100 MB** hits a separate input-size limit: find these with the
  `binaryFile` `length` column (`length > 100 * 1024 * 1024`), split them into
  smaller PDFs with `pypdf` first, then parse each part.

## Core pattern

1. Read PDFs as `binaryFile`; add `page_count` with `pypdf` (`-1` = unreadable).
2. Route: `page_count <= 500` → one call, no `pageRange`. `> 500` → chunked path.
3. Chunked path: `pageRange` must be a **literal** — build one Spark plan per range
   string (`'1-500'`, `'501-1000'`, …) and `unionByName` them. Ranges come from
   the **max** page count, so plan count is bounded by the longest document.
4. Stitch chunks per document in chunk order. With `pageRange`, page `id` and
   element `bbox.page_id` already come back document-global (0-based: page 501 →
   `500`) — keep them. Element `id` restarts at 0 in every chunk — renumber it.
5. Flag incomplete documents (missing chunk or `error_status`) instead of dropping them.

## Implementation

`scripts/large_pdf_parse.py` implements steps 1–5 and is unit-tested. Copy the
file's contents **unchanged** into one notebook cell, then call its functions from
the next cells as below. (Pasting keeps the tested code intact; importing it as a
module breaks its UDFs on serverless unless registered with `register_pickle_by_value`.)

```python
%pip install pypdf  # own cell, first

bronze = add_page_count(
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf").option("recursiveFileLookup", True)
    .load("/Volumes/<catalog>/<schema>/<volume>/<folder>")
)
bronze.write.mode("overwrite").saveAsTable("<catalog>.<schema>.bronze_pdfs")  # keeps page_count = -1 rows visible

# options={"descriptionElementTypes": ""} cuts cost but drops figure descriptions and the
# built-in chart JSON in figure content; keep the default if charts matter (crop-parsed-elements skill)
parsed = parse_documents(spark, spark.table("<catalog>.<schema>.bronze_pdfs"))
parsed.write.mode("overwrite").saveAsTable("<catalog>.<schema>.silver_parsed")
display(incomplete_documents(spark.table("<catalog>.<schema>.silver_parsed")))  # must be empty
```

Output columns: `path, page_count, chunk_count, parsed` (VARIANT). Feed `parsed`
straight to `ai_extract` / `ai_classify`.

## Common mistakes

| Mistake | Fix |
|---|---|
| One `ai_parse_document` call on a >500-page file | Route by `page_count`; chunk the long path |
| Treating a succeeded query as a successful parse | Check `parsed:error_status IS NULL` per row; it is `null` on success |
| Truncating text (`LEFT(text, 100000)`) to "handle" long docs | Truncation loses pages; chunk and stitch instead |
| `map('pageRange', some_column)` | Fails — options must be literals; one plan per range string |
| Looping once per long document | Loop once per range string, derived from the max page count |
| Flattening to a text column as the only output | Keep the stitched VARIANT; derive text from it if needed |
| Adding a chunk offset to page ids / `bbox.page_id` | They are already document-global; offsetting double-shifts them |
| Concatenating chunk `elements` as-is | Element ids restart at 0 each chunk; renumber them when stitching |
| Rebuilding `pages` as new structs (e.g. `{page: {pageNumber, width}}`) | Pages are `{id, image_uri}`; keep them unchanged so long and short docs match |
| Silently dropping failed chunks | Check `incomplete_documents()`; fail the job or route to review |
| Skipping unreadable PDFs without a trace | Keep `page_count = -1` rows in bronze and report them |
