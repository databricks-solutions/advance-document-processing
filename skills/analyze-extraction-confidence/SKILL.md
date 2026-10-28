---
name: analyze-extraction-confidence
description: Use when analyzing, profiling, or thresholding confidence scores from ai_parse_document (element confidence) or ai_extract v2.1 (confidence_score with enableConfidenceScores), when deciding which documents, pages, or extracted fields need human review, or when confidence-based review flags fire on nothing or on everything.
---

# Analyzing ai_parse_document / ai_extract Confidence

## Overview

Confidence from these functions does not look like a smooth 0–1 probability, so
fixed thresholds misfire. **Profile the observed distribution first, then flag
relative to it.** Confidence is a routing signal: flagged items go to a review
queue; nothing is dropped. For function syntax, use the `databricks-ai-functions` skill.
Scoring accuracy against labels is out of scope (only calibration is covered here).

## Shapes and behavior (verified on live output)

| | ai_parse_document v2.0 | ai_extract v2.1 |
|---|---|---|
| Where | `parsed:document:elements[i]:confidence` (page: `bbox[0]:page_id`) | `extracted:response:<path>:{value, confidence_score}` — **leaves only**; objects and arrays have no score |
| Typical spread | Bunched near 1.0 (median ≈ 1.0; min ≈ 0.8 on born-digital PDFs); differs by element `type` | A handful of discrete levels (e.g. 0.95 / 0.98 / 1.0) with rare low outliers (0.1, 0.75) |
| Missing | — | **Absent** field = `{value: null, confidence_score: null}`; **unscored** = value present, score null |

## Core pattern

1. Look at the distribution: `parse_type_profile` (per type) and
   `extract_level_profile` (distinct levels with counts).
2. **Parse:** flag elements below their **own type's** low percentile (default p5).
   An absolute floor like 0.5 or 0.8 flags nothing when scores sit near 1.0.
3. **Extract:** flatten nested leaves recursively, then flag scored leaves strictly
   below the corpus p5 **level**. Discrete levels make the cutoff land on an
   observed value, so ties at the cutoff are not flagged.
4. Keep three leaf states apart: absent (not a review item by itself), unscored
   (review), scored.
5. Roll up per page and per document; a document whose usual level sits below the
   corpus's is a document-level shift worth one look, not 30 field flags.
6. Calibration only with labels: correct rate **per observed level**, never per
   fixed 0.1-wide bins (most bins would be empty).

## Implementation

`scripts/confidence_profile.py` implements steps 1–6 and is unit-tested. Copy the
file's contents **unchanged** into one notebook cell, then:

```python
els = parse_elements(spark.table("<cat>.<sch>.<parsed_table>"), parsed_col="parsed", doc_id_col="path")
display(parse_type_profile(els))                     # look first
flagged_els = flag_parse_elements(els, pct=0.05)     # optional abs_floor= for scanned corpora
display(parse_page_rollup(flagged_els))

leaves = extract_leaves(spark.table("<cat>.<sch>.<extracted_table>"), extracted_col="extracted", doc_id_col="path")
display(extract_level_profile(leaves))               # look first
flagged_leaves, cutoff = flag_extract_leaves(leaves, pct=0.05)
display(extract_doc_summary(flagged_leaves))
display(extract_field_summary(flagged_leaves))

review_queue(flagged_leaves, flagged_els).write.mode("overwrite").saveAsTable("<cat>.<sch>.confidence_review_queue")
# with labels (doc_id, path, is_correct):  display(calibration_by_level(flagged_leaves, labels))
```

Several extraction VARIANT columns in one table (one per document type)? Union them
into one `(path, extracted)` frame first so the corpus cutoff uses every leaf.

## Common mistakes

| Mistake | Fix |
|---|---|
| Absolute parse floor (0.5 / 0.8) | Per-type percentile; read `parse_type_profile` first |
| Pooling all element types | Tables, titles, and `text` score differently; cutoffs per type |
| Reading `response:<field>:confidence_score` only | Nested objects/arrays have no top-level score; flatten to leaves |
| Treating absent fields as low-confidence or "missing score" | `value` null = absent; only value-present-score-null is unscored |
| Dropping unscored leaves (`confidence IS NOT NULL`) | Route them to review; a whole extraction can come back unscored |
| Flagging "below the field's usual level" | Adjacent high levels (0.95 vs 0.98) flood the queue; use the corpus p5 level |
| Fixed-width calibration bins | Group by observed level |
| `calibration_by_level` on a stratified audit sample | It averages labels as given; with stratified labels, weight each row by stratum size / sample size |
| Reading every parse flag as equally urgent | In types whose cutoff is ≥ 0.999 (page headers/numbers) gaps are tiny; review them last, don't drop them |
| `assert IQR > 0` / z-scores on saturated scores | Percentiles with strict `<` handle ties and saturation |
