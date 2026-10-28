---
name: crop-parsed-elements
description: Use when cropping figures, charts, tables, or other elements out of PDFs parsed with ai_parse_document (bbox coord), when extracting chart data from parsed documents or sending chart images to a vision model, or when crops come out misaligned or chart insights have wrong titles, units, or series names.
---

# Cropping ai_parse_document Elements (and Getting Chart Data)

## Overview

Two facts decide everything: **bbox coords are pixels of the page rendered at 200 DPI**,
and **a crop alone loses the chart's title, units, and often its legend** — those are
separate elements on the page. Without them a vision model invents the metric and
drops the scale (observed: "Quarterly Revenue, $" for "Advertising Revenue by User
Geography, In Millions"). For function syntax, use the `databricks-ai-functions` skill.

## First: do you need to crop at all?

| Parse option `descriptionElementTypes` | Chart figure `content` |
|---|---|
| default `'*'` or includes `'figure'` | **Structured chart JSON** (series × categories, values exact on tested decks) + a prose `description` |
| `''` | Raw label text only (`"$50,082\n$46,783…"`) |

- Need chart data and can (re)parse with figures on → read figure `content` JSON
  (`builtin_chart_data`), add page context, done. No crop, no vision call.
- Crop + vision model when: the corpus was parsed with `''` and re-parsing costs more
  than vision calls on figures only; you need your own schema or domain prompt; you
  need the image itself (review UI, OCR, audit); or built-in JSON is missing for a chart.
- Either way, built-in JSON has `"title": "None"`, no units, and may name series
  `"Segment 1"` — always attach page context.

## Verified facts

| | |
|---|---|
| `bbox[0].coord` | `[x0, y0, x1, y1]` int pixels, origin top-left, page rendered at **200 DPI** (px = PDF points × 200/72) |
| Page image | Only with `imageOutputPath` (`pages[].image_uri`, same 200-DPI size); otherwise `null` |
| Without page images | Render with `pypdfium2` at `scale=200/72` — aligns pixel-exact with the parser's image |
| Page | `bbox[0].page_id` = 0-based PDF page index = `pages[].id` (not array position in chunked parses) |
| Box tightness | Boxes touch content and move 2–3 px between parses — pad ~20 px |
| Context | Title/units are `title` + short `text` above the chart; legends may be a `footnote` beside it |

## Implementation

`scripts/crop_elements.py` (unit-tested). Install `pypdfium2` (Apache/BSD; PyMuPDF is
AGPL), copy the file's contents **unchanged** into one cell, then:

```python
%pip install pypdfium2   # own cell, first

els = elements_to_crop(spark.table("<cat>.<sch>.<parsed_table>"), parsed_col="parsed", doc_id_col="path")
display(els.select("doc_id", "page_id", "builtin_chart_json", "context"))   # built-in data + context

crops = crop_elements(spark, els, out_dir="/Volumes/<cat>/<sch>/<vol>/crops")  # doc_id = source PDF path
crops.write.mode("overwrite").saveAsTable("<cat>.<sch>.element_crops")     # status: ok | too_small | bad_page | pdf_not_found

insights = vision_insights(spark, spark.table("<cat>.<sch>.element_crops"), "<vision endpoint>")
```

`vision_insights` calls `ai_query(endpoint, prompt + context, files => crop, failOnError => false)`
(verified with Claude endpoints). With `failOnError => false` the result is a struct with
fields **`result`** and **`errorMessage`** (verified live) — not `response` / `error`. To embed with document text, use `splice_figure_text(elements, {element_id: insight})`
so each chart stays in reading order. `elements_to_crop(..., types=("table",))` crops tables too.

## Common mistakes

| Mistake | Fix |
|---|---|
| Treating coords as PDF points, 72-DPI pixels, or 0–1 | 200-DPI pixels; scale by `render_dpi / 200` |
| Calibrating coords from PDF text blocks | Fails on scanned pages; the scale is a constant |
| Sending only the crop (or only a `caption`) | Pass same-page title, units, and adjacent legend (`figure_context`) |
| Ignoring built-in figure `content` / `description` | Check it first; parse with figures on if chart data is the goal |
| Parsing with `descriptionElementTypes: ''` then wanting chart data | That setting drops the chart JSON; re-parse or crop + vision |
| Mapping page by array position | Use `bbox.page_id` |
| No padding / no size floor | Pad ~20 px; skip tiny figures (logos) with `min_side_px` |
| PyMuPDF in a customer project | AGPL — use `pypdfium2` |
| Embedding vision text *and* running `ai_prep_search` on the original `parsed` | Built-in chart JSON gets embedded too; pick one representation per chart |
