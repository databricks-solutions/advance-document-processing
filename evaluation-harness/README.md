# Evaluation Harness

Three **standalone recipe notebooks** for measuring the quality of Databricks AI
Functions on documents — the confidence they report and, where you have labels,
whether that confidence is actually right. They are **recipes, not a workflow**:
each is self-contained, widget-driven, and reads a table one of this repo's
pipelines already produced. There is no orchestration, no `config.yaml`, and no
dependency between the three.

Adapted from the [`AI-function-evaluation-harness`](https://github.com/qian-yu-db/AI-function-evaluation-harness)
project (which does this deed-by-deed inside a DAB) — here the domain-specific
pieces are replaced with widgets so the recipes work on any parsed/extracted
table.

## The three recipes

| # | Notebook | Measures | Needs |
|---|---|---|---|
| 01 | `notebooks/01_parse_confidence_stats.py` | `ai_parse_document` **element** confidence — corpus / per-doc / per-page stats + violin/box viz | a table with the parse VARIANT |
| 02 | `notebooks/02_extract_confidence_stats.py` | `ai_extract` v2.1 **per-field** confidence — distribution + review roll-up + box/scatter viz | a table with the raw `ai_extract` VARIANT |
| 03 | `notebooks/03_extract_eval_with_ground_truth.py` | `ai_extract` **accuracy vs ground truth** — hybrid comparator, P/R/F1, `mlflow.genai.evaluate`, calibration | a flat extracted table **+ a ground-truth table** |

All three treat **confidence as descriptive, never a filter** — they surface where
a model was unsure so you can route those items to review, rather than silently
dropping them.

## Input contracts

The recipes are `measure-only` — they do **not** call `ai_parse_document` or
`ai_extract`. Run an extract pipeline first, then point the widgets at its output.
In this repo, [`ai-extract-word-level-citation`](../ai-extract-word-level-citation)
produces exactly the shapes below.

**Recipe 01 — parsed table.** A VARIANT column (default `parsed`) matching
`ai_parse_document` output: `parsed:document:elements` is an array of elements,
each with `type`, `content`, `confidence`, `bbox[0]:page_id`.
Example: `*_bronze_parsed_docs`.

**Recipe 02 — extracted (raw) table.** A VARIANT column (default `extracted`)
matching `ai_extract` **v2.1** run with `enableConfidenceScores`: each field at
`extracted:response:<field>:{value, confidence_score, citation_ids}`.
Example: `*_extracted` (its VARIANT column is named `paystub_extracted` — set the
`extracted_col` widget accordingly).

**Recipe 03 — flat extracted table + ground-truth table.**

- *Flat extracted:* one row per document; per field a value column `<field>` and a
  confidence column `<field><conf_suffix>` (default suffix `_extract_conf`); plus a
  document id column. Example: `*_extracted_flat`.
- *Ground truth:* one row per document; a document-id column (`gt_id_col`) and one
  column per field with the correct value, using `NULL`/empty for fields that are
  genuinely **absent** from the document (absence is scored as a real class).

  The two sides join on a **normalized key** derived from the document id
  (basename → strip extension → lowercase → drop non-alphanumerics), so
  `.../paystub_synth_1.pdf` matches `paystub_synth_1`.

### Sample ground truth (paystubs)

A worked example ships in `sample_data/paystub_ground_truth.csv` — labels for the
four synthetic paystubs in
[`../ai-extract-word-level-citation/sample_data/`](../ai-extract-word-level-citation/sample_data),
generated together by `scripts/generate_sample_paystubs.py` (so the labels match
the documents exactly). Columns use the pipeline's `EXTRACT_FIELDS` names:

```
doc_id             | employee_name | employer_name              | social_security_number | ... | gross__current_amount | net_pay__current_amount
paystub_synth_1.pdf| Jordan Rivera | Metro City Transit Authority| XXX-XX-4821            | ... | 1,884.00              | 1,387.35
paystub_synth_4.pdf| Priya Nair    | Cedar Valley Health System |                        | ... | 2,115.38              | 1,580.63   <- SSN absent (TN/FN case)
```

To use it with recipe 03: load the CSV into a table (or point a `read_files`
view at it), then set `ground_truth_table` to it, `gt_id_col=doc_id`, and
`digit_fields` to the numeric/date fields for strict matching, e.g.
`social_security_number,gross__current_amount,gross__year_to_date_amount,net_pay__current_amount,net_pay__year_to_date_amount`.
The remaining fields (names, marital status) fall through to fuzzy text matching.

## Requirements

| Component | Requirement |
|---|---|
| `ai_parse_document` (upstream) | DBR 17.1+ / serverless env v3+ |
| `ai_extract` 2.1 + confidence (upstream) | DBR 18.2+ / serverless env v3+ |
| Recipes 01 & 02 | `matplotlib`, `numpy`, `pandas` — pre-installed on DBR |
| Recipe 03 | `%pip install "mlflow[databricks]>=3.1.0"` (installed by the notebook); a serving endpoint for the judge model (default `databricks:/databricks-claude-sonnet-4-6`) |

## How to run

Each notebook is independent — open it, set the widgets at the top, and run top to
bottom.

- **01 / 02** need no install; set `catalog` / `schema` / the table name and the
  VARIANT column, then run.
- **03** installs MLflow in the first cell (which restarts Python), so run it from
  the top. Set the flat table and the ground-truth table; `fields` can be left
  **blank** (it auto-resolves to the ground-truth table's columns, skipping
  `_`-prefixed ones like `_rescued_data`). Optionally set `digit_fields` (strict-match
  IDs/amounts) and the `mlflow_experiment` path. Results land in that MLflow experiment.

## Recipe 03 scoring model

- **Hybrid comparator** — `digit_fields` compare **numerically** (decimal-insensitive,
  so `1884.0` == `1884.00` == `"1,884.00"`), with an exact normalized-text fallback
  for non-numeric IDs; every other field uses a normalized Levenshtein ratio
  `>= fuzzy_threshold` (default 0.85).
- **Absence-as-a-class** — each field is classified `TP` / `TN` / `FP` / `FN` /
  `FP_FN`. `FP_FN` (present but wrong) counts against **both** precision and recall.
- **`mlflow.genai.evaluate`** — a built-in `Correctness` LLM judge plus one
  pass-through scorer per field, with the P/R/F1 metrics logged to the same run.
- **Calibration** — bins confidence and checks whether the correctness rate rises
  with confidence (monotonicity), and lists the "confidently wrong" cases a plain
  threshold would wave through.

## Related assets

- [`../ai-extract-word-level-citation`](../ai-extract-word-level-citation) —
  produces the parsed / extracted / flat tables these recipes consume.
- [`../document-page-classify-extraction`](../document-page-classify-extraction) —
  multi-document classify + extract; its `*_bronze_parsed_docs` and per-class
  extract tables also fit recipes 01/02.
