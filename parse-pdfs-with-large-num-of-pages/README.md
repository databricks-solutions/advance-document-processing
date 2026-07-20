# Parsing PDFs Larger Than the `ai_parse_document` Page Limit

A **single self-contained recipe notebook** — not a workflow. `ai_parse_document`
has a hard **500-page-per-call limit**; this recipe shows how to parse documents
that exceed it by routing long PDFs through chunked calls and stitching the result
back into the same VARIANT shape a single call would produce, so downstream
`ai_classify` / `ai_extract` treat short and long documents identically.

There is no Databricks Asset Bundle here on purpose — the recipe is one technique,
widget-driven, meant to be read and adapted into whichever parse pipeline you
already run. Open it, set the widgets, run top to bottom.

| File | What it does |
|---|---|
| `parse_large_pdfs_with_pagerange.py` | Bronze → route by page count → short/long parse → silver (unified VARIANT) → full text |

## The problem

`ai_parse_document`'s `pageRange` option (v2.0+) lets you parse a slice of a
document, which is the escape hatch for the 500-page limit. But `pageRange` **must
be an analysis-time foldable literal** — you cannot column-drive it
(`map('pageRange', page_range_col)` fails). So a single query can't parse
variable-length long documents; you need one Spark plan per distinct page-range
literal.

## How it works

The recipe splits documents into two paths so short PDFs (the common case) are
never blocked by long ones:

1. **Load** raw PDFs from a UC Volume (`binaryFile`).
2. **Detect page count** with a `pypdf` Pandas UDF. Bad/encrypted files return
   `-1` and surface downstream instead of failing the job.
3. **Bronze** — raw binary + `page_count` persisted to Delta, so parsing can re-run
   without re-reading the volume.
4. **Route** by `page_count`:
   - `<= page_range_limit` → **short path**: one `ai_parse_document` call, no
     `pageRange`.
   - `>  page_range_limit` → **long path**: chunk into `1-500`, `501-1000`, …,
     parse each with a literal `pageRange`, merge.
5. **Silver** — union both paths into one uniform VARIANT schema
   (`file_name, page_count, chunk_count, parsed`).
6. **Full text** — reassemble each document's element `content` into one ordered
   string.

### The long-path loop

Because chunks are always `1-500`, `501-1000`, … the distinct page-range strings
are deterministic. The recipe computes them from the **global max page count**
(`ceil(max_pages / page_range_limit)` values) rather than a row-side `collect_set`,
builds one `ai_parse_document` plan per literal, and `unionByName`s them. Iteration
count is bounded by the longest document, **not** by how many long PDFs you have.

### Stitching long documents

Chunk outputs are merged back into one row per document, ordered by chunk index,
and repackaged as a single VARIANT with the same
`{ document: { elements, pages }, error_status }` shape a single call returns. Page
indices are offset so they stay globally sequential across chunks
(`chunk_idx * page_range_limit + i`). A **partial-chunk failure guard** compares
each document's stitched `chunk_count` to the expected
`ceil(page_count / page_range_limit)` and warns on any gaps (silent chunk drops
would otherwise produce a document missing pages with no error signal).

## Configuration

| Widget | Default | Meaning |
|---|---|---|
| `catalog` | `fins_genai` | Unity Catalog catalog |
| `schema` | `unstructured_documents` | schema for source volume + output tables |
| `volume` | `pdf_examples` | UC Volume holding the PDFs |
| `volume_folder` | `large_pdfs` | subfolder of PDFs under the volume |
| `page_range_limit` | `500` | pages per `ai_parse_document` call (the hard limit) |

### Output tables (under `catalog.schema`)

| Table | Contents |
|---|---|
| `bronze_raw_pdfs` | raw binary + `page_count` per file |
| `silver_parsed_short_docs` | short-path parse output |
| `silver_parsed_long_docs` | long-path stitched output |
| `silver_parsed_docs` | union of both — the table downstream reads |
| `pagerange_full` | per-document reassembled `full_text` |

## Requirements

| Component | Requirement |
|---|---|
| `ai_parse_document` | DBR **17.1+** / serverless env **v3+** (v2.0 for `pageRange`) |
| `pypdf` | installed by the notebook (`%pip install pypdf`) |

## Notes & limits

- **`pageRange` must be a literal** — the long-path loop is the canonical
  workaround; you can't pass it as a column.
- The 500-page limit is **per call**, not per document; chunked calls are
  independent.
- Omitting `pageRange` on a >500-page PDF fails immediately — the Step 4 router
  keeps that case off the short path.
- **Silver `parsed` is schema-compatible** with a normal `ai_parse_document` bronze
  layer, so downstream `ai_classify(parsed, …)` / `ai_extract(parsed, …)` work
  unchanged. If a future `ai_parse_document` version adds top-level fields, extend
  the stitching `named_struct` to forward them.
- For production, convert the partial-chunk **warning** into a hard assertion or a
  Lakeflow Spark Declarative Pipeline expectation.

## Related assets

- [`../document-page-classify-extraction`](../document-page-classify-extraction) and
  [`../ai-extract-word-level-citation`](../ai-extract-word-level-citation) — parse
  pipelines whose bronze VARIANT shape this recipe matches; drop the silver table in
  as their parsed input.
- [`ai_parse_document` docs](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document)
  — the `pageRange` option and page limit.
