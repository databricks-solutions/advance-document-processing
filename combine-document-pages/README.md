# Combine Document Pages — Aggregate Split PDF Pages into One VARIANT

Two standalone recipe notebooks for the case where a single logical document
arrives as **separate single-page PDF files** — for example split upstream by a
scanner, a burst/split tool, or another pipeline. Each page is parsed
individually with
[`ai_parse_document`](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document),
then the per-page results are stitched back into **one document-level VARIANT
that matches the canonical `ai_parse_document` output shape** (with corrected
`page_number` references).

The point is schema compatibility: downstream pipelines that expect *one*
`ai_parse_document` VARIANT per document (bronze layers, `ai_prep_search`,
`ai_extract`, `ai_classify`) can then treat the reassembled document exactly as
if it had been parsed from a single file.

> Complementary to [`../parse-pdfs-with-large-num-of-pages`](../parse-pdfs-with-large-num-of-pages),
> which solves the inverse problem — one very large PDF parsed in chunked
> `pageRange` calls and stitched into a uniform VARIANT. Both produce the same
> unified shape; they differ only in how the input arrives (one big file vs.
> many single-page files).

## The two recipes

| Recipe | Page order comes from | Use when |
|---|---|---|
| [`parse_aggregate_pdf_pages.py`](./parse_aggregate_pdf_pages.py) | The **filename** (regex `page_(\d+)\.pdf`) | The split files encode the page number in their names (`page_1.pdf`, `page_2.pdf`, …) |
| [`parse_aggregate_pdf_infer_page_order.py`](./parse_aggregate_pdf_infer_page_order.py) | The **content** — inferred, with fallbacks | Filenames do **not** encode page order |

**Order-inference fallback chain** (second recipe):

1. **`page_number` element** — the printed page number that `ai_parse_document`
   already extracts as an element of `type = 'page_number'`.
2. **`ai_extract` (v2.1) fallback** — when a page has no `page_number` element,
   ask the model to read the printed page number from the page content.
3. **Row-number fallback** — if both fail, assign order by filename sort as a
   last resort.

## How it works

Both notebooks follow the same shape (serverless **env v5**):

1. **Parse each page** — `READ_FILES(..., format => 'binaryFile')` over the
   source Volume folder, `ai_parse_document(content, MAP('version', '2.0'))` per
   file, into a temp view `parsed_pages`. A verification query shows per-page
   parse status and element/page counts.
2. **Determine page order** — from the filename (recipe 1) or inferred from
   content (recipe 2).
3. **Aggregate into one VARIANT** — a small driver-side Python step collects the
   successfully-parsed pages in order, rewrites each page's and element's
   `page_number` to the resolved value, and reassembles the canonical shape:

   ```
   {
     "document": { "pages": [ … ], "elements": [ … ] },
     "metadata":  { … },        -- carried from the first page
     "error_status": null
   }
   ```

   The result is materialised as a single-row temp view `merged_document` with a
   `parsed` VARIANT column.
4. **Verify** — total page/element counts, no-error flag, and metadata.
5. **(optional)** Preview elements across all pages, and persist the merged
   VARIANT to a UC table.

## Input & output

- **Input**: single-page PDFs in a UC Volume folder. Default source:
  `/Volumes/fins_genai/unstructured_documents/pdf_examples/pdf_pages/` — edit the
  `READ_FILES(...)` path in Step 1 to point at your own split pages.
- **Output**: a single-row `parsed` VARIANT shaped like `ai_parse_document`
  output — `parsed:document:pages`, `parsed:document:elements` (each element
  carries `page_number`, `type`, `content`, `element_index`, …),
  `parsed:metadata`, `parsed:error_status`.

## Prerequisites

- Databricks workspace with **Unity Catalog** and **Serverless** enabled.
- `ai_parse_document` requires **DBR 17.3+ / serverless env v3+**; the inferred-
  order recipe also uses `ai_extract` v2.1 (**DBR 18.2+ / env v3+**). Both
  notebooks pin **environment version 5**.
- A region that supports `ai_parse_document` batch AI inference.

## Run

These are interactive notebooks, not a job/bundle. In the Databricks workspace:

1. Open the recipe that matches your filename convention.
2. Set the source Volume path in **Step 1** to your split-page folder.
3. Run all cells. The merged document lands in the `merged_document` temp view;
   uncomment the final cell to persist it to a table.

## Notes & caveats

- **Driver-side merge**: the aggregation `collect()`s parsed pages to the driver
  and merges in Python — simple and fine for a single document of typical size.
  For very large documents or many documents at once, prefer a SQL/Spark-side
  aggregation instead of collecting to the driver.
- **`page_number` is rewritten** to the resolved order on every page and
  element, so `ORDER BY page_number` is reliable downstream even if the original
  per-file parse numbered every page as page 1.
- **Parse failures are skipped**: only pages with `is_variant_null(parsed:error_status)`
  are merged; check the Step 1 verification query if the merged count is short.
- The `ai_parse_document` output nests content under `parsed:document:elements`
  (a flat array) and `parsed:document:pages`; the `[*]` wildcard is **not**
  valid in `variant_get`, so walk arrays with `try_cast(... AS ARRAY<VARIANT>)`
  + `explode`/`filter`/`transform` (as the notebooks do).
