# Evaluation Harness

**Standalone recipe notebooks** for measuring the quality of Databricks AI
Functions on documents — the confidence they report and, where you have labels,
whether that confidence is actually right. They are **recipes, not a workflow**:
each is self-contained, widget-driven, and reads a table one of this repo's
pipelines already produced. There is no orchestration, no `config.yaml`, and no
dependency between the measure-only recipes.

The one exception is **Recipe 00**, a data-prep front-end that runs the AI
Functions once and builds the ground-truth tables Recipes 03 and 04 consume — run
it first if you want the worked paystub example end-to-end, or bring your own
tables and skip it.

Adapted from the [`AI-function-evaluation-harness`](https://github.com/qian-yu-db/AI-function-evaluation-harness)
project (which does this deed-by-deed inside a DAB) — here the domain-specific
pieces are replaced with widgets so the recipes work on any parsed/extracted
table.

> **This is a demonstration, not a prescription.** The paystub example wires up
> one plausible end-to-end path so you can see every step run. On your own data
> the important decisions are *yours to make*: flat vs. nested schema, which fields
> to label, which scoring definition matches how you actually grade extractions,
> and how much you trust model confidence. The recipes lay out the options and the
> trade-offs — see [Choosing an approach](#choosing-an-approach-for-your-data) —
> but the call depends on your documents and your quality bar.

## The recipes

| # | Notebook | Measures | Needs |
|---|---|---|---|
| 00 | `00_parse_extract_prepare_ground_truth.py` | **data prep** — parses PDFs, extracts a flat + a nested schema, builds the ground-truth tables for 03 & 04 by joining the curated answer files | sample PDFs in a UC volume + answer files in `sample_data/` |
| 01 | `01_parse_confidence_stats.py` | `ai_parse_document` **element** confidence — corpus / per-doc / per-page stats + violin/box viz | a table with the parse VARIANT |
| 02 | `02_extract_confidence_stats.py` | `ai_extract` v2.1 **per-field** confidence — distribution + review roll-up + box/scatter viz | a table with the raw `ai_extract` VARIANT |
| 03 | `03_extract_eval_with_ground_truth.py` | `ai_extract` **accuracy vs ground truth (flat)** — fuzzy comparator, per-field P/R/F1, `mlflow.genai.evaluate`, calibration | a flat extracted table **+ a flat ground-truth table** |
| 04 | `04_extract_eval_nested.py` | `ai_extract` **accuracy vs ground truth (nested)** — type-aware scoring (see [the definition](#the-type-aware-eval-definition-recipe-04) below), runs extract live | one eval table (`path`, `input`, `ground_truth`) |

The measure-only recipes (01–04) treat **confidence as descriptive, never a
filter** — they surface where a model was unsure so you can route those items to
review, rather than silently dropping them.

---

## Recipe 00 — parse, extract, and prepare ground truth

Recipe 00 is the **front-end that produces everything the eval recipes read**. It
runs the AI Functions exactly once, then assembles the tables by joining the model
output with the curated answer files. The steps:

1. **Parse** — `ai_parse_document` on the sample paystub PDFs → one parsed VARIANT
   per document (`{prefix}_bronze_parsed`).
2. **Extract twice** — `ai_extract` over that parsed VARIANT with **two schemas**:
   - a **flat** schema (scalars + `gross`/`net_pay`, projected to the 13 columns of
     `paystub_ground_truth.csv`), and
   - a **nested** schema (`PAYSTUB_SCHEMA` — objects *and* arrays).
3. **Flatten** the flat extraction to one value column `<field>` + one confidence
   column `<field>_extract_conf` per field.
4. **Join with answers** — normalize the document id to a join key
   (basename → strip extension → lowercase → drop non-alphanumerics, so
   `.../paystub_synth_1.pdf` matches `paystub_synth_1`) and write the tables below.

### Tables it writes

| Table | Columns | Built from | Consumer |
|---|---|---|---|
| `{prefix}_bronze_parsed` | `path`, `parsed` | `ai_parse_document` | 01 |
| `{prefix}_extracted_flat` | `path`, `<field>`, `<field>_extract_conf` | flat `ai_extract` | **03** (predictions) |
| `{prefix}_ground_truth_simple` | `doc_id`, 13 flat columns | `sample_data/paystub_ground_truth.csv` | **03** (labels) |
| `{prefix}_extracted_nested` | `path`, `input`, `extracted`, `ground_truth` | nested `ai_extract` ⋈ `sample_data/paystub_ground_truth_ai_extract.jsonl` | **04** |

`{prefix}_extracted_nested` bundles the parsed `input`, the nested `extracted`
prediction, and the nested `ground_truth` in one row per document. Recipe 04 can
therefore either **re-extract live** from `input` or read the cached `extracted`
column. The nested labels come from
`sample_data/paystub_ground_truth_ai_extract.jsonl` (no pre-built nested CSV is
needed — the table is assembled by join).

### Flat vs. nested — two shapes for two eval styles

The same documents are extracted under two schemas because the two eval recipes
grade different things:

- **Flat** — every field is a top-level scalar column (`gross__current_amount`,
  `net_pay__current_amount`, …). Simple to store, simple to eyeball, and it is all
  Recipe 03's fuzzy comparator needs. It **cannot represent** repeating rows or
  objects — a paystub's line items are lost.
- **Nested** — fields keep their natural structure: objects (`gross:{current,
  ytd}`), and arrays of objects (`income_item:[{description, rate, amount}, …]`).
  This is what `ai_extract` actually returns, and it is what Recipe 04 scores with
  structure-aware rules.

> In practice you rarely need *both*. The demo extracts both so you can compare 03
> and 04 side by side; on your own data pick the schema that matches your target
> output and run the matching recipe.

### Labelled vs. unlabelled — what each recipe assumes

- **Unlabelled (01, 02)** — you have documents and model output but **no ground
  truth**. These recipes profile *confidence only*: where was the parser or
  extractor unsure? Use them to triage a corpus for review, or as a first look
  before you invest in labelling. They run on any parsed/extracted table.
- **Labelled (03, 04)** — you have a **correct answer** for each field and want to
  measure accuracy against it. This is the expensive part: someone must produce the
  labels. The sample ships hand-curated labels for four paystubs; on your data you
  supply your own, at whatever coverage you can afford (you can label a subset of
  fields — unlabelled fields are simply skipped or treated as absent, per recipe).

---

## Choosing an approach for your data

The demo makes these choices for you. On real data, decide deliberately:

| Decision | Choose **flat / 03** when… | Choose **nested / 04** when… |
|---|---|---|
| **Schema shape** | fields are a fixed, flat set of scalars | output has objects, or **repeating rows** (line items, transactions) |
| **Matching strictness** | near-misses should score *partially* (fuzzy text, decimal-insensitive numbers) | you want the **documented, type-aware definition** — exact/normalized with an LLM-judge fallback for strings |
| **Confidence** | you care whether `ai_extract` confidence is **calibrated** (03 reports it; 04 does not) | calibration is not the question |
| **Absence handling** | absence is a graded class (`TP/TN/FP/FN/FP_FN`) and you want P/R/F1 | absence is scored as accuracy (both-absent = correct) |

A flat schema is just an object of scalar leaves, so **04 can also score the flat
case** — 03 is kept for its fuzzy comparator and confidence-calibration view, which
04 deliberately does not provide. If you only care about the type-aware definition,
04 alone covers everything.

**Both 03 and 04 report per-field metrics** — just under different definitions (P/R/F1
vs. accuracy/object-score/soft-F1). Neither collapses the result to a single number.

---

## The type-aware eval definition (Recipe 04)

Recipe 04 implements a **type-aware** scoring definition: each field is scored by
its **declared type**; a row's score aggregates its fields, and summary metrics
aggregate across rows.

> **Reference.** This mirrors the per-field evaluation Databricks ships in
> [Agent Bricks — Information Extraction](https://docs.databricks.com/aws/en/agents/agent-bricks/info-extraction),
> whose scorecard reports per-field accuracy / precision / recall / F1 against a
> ground-truth JSON that conforms to the `ai_extract`
> [advanced schema](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract).
> The documented UI covers the scalar per-field metrics; the **soft-F1 partial-credit
> matching for arrays/objects** here is a **custom scorer** (implemented in Recipe 04),
> not a documented built-in — Databricks recommends a custom scorer exactly when you
> need partial credit or structure-aware matching.

### Scalar types

| Type | Row rule | Summary metric(s) |
|---|---|---|
| **String** | **normalized match** (lowercase + trim); if that fails, an **LLM judge** decides equivalence | `accuracy` |
| **Number** | exact match (`42.0` == `42`; `41.99` ≠ `42`) | `accuracy` |
| **Integer** | exact match | `accuracy` |
| **Boolean** | exact match | `accuracy` + per-value `precision` / `recall` / `f1` |
| **Enum** | exact match | `accuracy` + per-value P/R/F1 (when ≤ 5 categories) |

`accuracy = correct rows / total rows`, and `overall_score = accuracy`.

**String example** — `{"street": "1 Sesame Street"}` vs. `{"street": "1 Sesame St"}`
fails the normalized match but the LLM judge rules it equivalent → **correct**.
`{"street": "2 Violet Rd"}` vs. `{"street": "2 Orange St"}` → **incorrect**.

**Boolean / enum per-value metrics** — for each value (e.g. `true`, or each enum
category) a confusion matrix gives `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`,
`f1 = 2·P·R/(P+R)`. This surfaces class imbalance a single accuracy hides.

### Composite types

**Object** — score each child by its own type, then average:

```
object_score = mean(child field scores)
```

Summary: `object_score_mean` (average object score across rows) plus each child
field's own summary metrics. Example — predicted `{"name":"Bob","age":14}` vs.
expected `{"name":"Sarah","age":14}` → `(0 + 1) / 2 = 0.5`.

**Array** (of scalars) — find the **optimal 1:1 matching** between predicted and
expected items (each item scored by its type, pairs chosen to maximize total
score), then:

```
precision = matched score / predicted length      (penalizes extra items)
recall    = matched score / expected length        (penalizes missing items)
soft_f1   = 2 · precision · recall / (precision + recall)
```

Summary: `soft_f1_mean`. Example — `["Python","JavaScript","R"]` vs.
`["Python","Java"]`: only `Python` matches → precision `1/3`, recall `1/2`,
soft F1 ≈ **0.4**.

**Array of objects** — same optimal matching, but each candidate pair's similarity
is the **average of its field scores**; the matched object scores then feed the
same precision / recall / soft-F1 formula. Summary: `soft_f1_mean` plus
`soft_f1_<child>_mean` for each child field, so you can see *which* field drags an
array down.

> **Absence** is handled uniformly across all types: both sides absent = `1.0`
> (correctly nothing), one side absent = `0.0`.

### How Recipe 04 runs it

Scoring is **schema-driven and recursive**, in pure Python on the driver
(`scipy.optimize.linear_sum_assignment` for the optimal matching). The hardcoded
`PAYSTUB_SCHEMA` is the single source of truth: `json.dumps` of it is the schema
passed to `ai_extract`, and the same dict's `type` fields select each comparator —
so extraction and scoring can't drift. The built-in MLflow `Correctness` judge is
the string fallback, invoked **only on a normalized-match miss** (so it stays
cheap); set `use_llm_judge=false` for deterministic, normalized-only string
scoring.

> **Sample-data caveat:** the curated ground truth covers only scalar fields plus
> the `gross` / `net_pay` objects. The nested schema's `array` fields
> (`income_item`, `deduction_item`) and address objects are extracted but have **no
> labels**, so Recipe 04's array paths are implemented to spec but **not exercised**
> by this sample until richer ground truth exists.

> **Known limitation — unlabelled fields score 0.0 (fixed in the next iteration).**
> The scorer cannot yet distinguish a field that is *genuinely absent from the
> document* (a true negative) from one that is simply *not labelled in the ground
> truth*. Both are treated as "expected absent," so any field the model **correctly
> extracts but the ground truth does not label** is scored `0.0` (a present
> prediction against an absent expectation). With the shipped sample this shows up
> as `0.0` for the address objects, `income_source_type`, and the `income_item` /
> `deduction_item` arrays — those are **not extraction failures**, just fields the
> curated labels omit. It also skews the confidence-calibration appendix, which
> mixes these false-failures into its high-confidence bins.
>
> For this reason, read Recipe 04's metrics as reliable **only for the labelled
> fields**, and treat the unlabelled-field `0.0`s as noise for now. A future
> iteration will add a "score only fields present in each row's ground truth" mode
> (and a `null`-means-confirmed-absent vs. missing-key-means-unlabelled convention)
> so partial ground truth is handled correctly.

---

## Recipe 03 scoring model

Recipe 03 uses a **fuzzy, absence-aware** definition — deliberately more tolerant
than 04, and paired with confidence calibration:

- **Hybrid comparator** — `digit_fields` compare **numerically** (decimal-insensitive,
  so `1884.0` == `1884.00` == `"1,884.00"`), with an exact normalized-text fallback
  for non-numeric IDs; every other field uses a normalized **Levenshtein ratio**
  `>= fuzzy_threshold` (default 0.85), so small typos still count as matches.
- **Absence-as-a-class** — each field is classified `TP` / `TN` / `FP` / `FN` /
  `FP_FN`. `FP_FN` (present but wrong) counts against **both** precision and recall,
  so a confidently-wrong extraction is penalized twice.
- **`mlflow.genai.evaluate`** — a built-in `Correctness` LLM judge plus one
  pass-through scorer per field, with the P/R/F1 metrics logged to the same run.
- **Calibration** — bins confidence and checks whether the correctness rate rises
  with confidence (monotonicity), and lists the "confidently wrong" cases a plain
  threshold would wave through.

---

## Input contracts

Recipes 01–03 are `measure-only` — they do **not** call `ai_parse_document` or
`ai_extract`. Run Recipe 00 (or an extract pipeline such as
[`ai-extract-word-level-citation`](../ai-extract-word-level-citation)) first, then
point the widgets at its output. Recipe 04 runs `ai_extract` itself on the eval
table's `input` column.

**Recipe 01 — parsed table.** A VARIANT column (default `parsed`) matching
`ai_parse_document` output: `parsed:document:elements` is an array of elements,
each with `type`, `content`, `confidence`, `bbox[0]:page_id`.
Example: `{prefix}_bronze_parsed`.

**Recipe 02 — extracted (raw) table.** A VARIANT column (default `extracted`)
matching `ai_extract` **v2.1** run with `enableConfidenceScores`: each field at
`extracted:response:<field>:{value, confidence_score, citation_ids}`.

**Recipe 03 — flat extracted table + ground-truth table.**

- *Flat extracted:* one row per document; per field a value column `<field>` and a
  confidence column `<field><conf_suffix>` (default suffix `_extract_conf`); plus a
  document id column. Example: `{prefix}_extracted_flat`.
- *Ground truth:* one row per document; a document-id column (`gt_id_col`) and one
  column per field with the correct value, using `NULL`/empty for fields that are
  genuinely **absent** from the document (absence is scored as a real class).
  Example: `{prefix}_ground_truth_simple`.

  The two sides join on the **normalized key** described above.

**Recipe 04 — one eval table.** Columns `path` (id), `input` (parsed-document
VARIANT, what `ai_extract` runs on), and `ground_truth` (nested JSON of correct
values). Example: `{prefix}_extracted_nested` (which also carries a cached `extracted`
column). Read from a Delta table (preferred for the large `input` column) or a
CSV/JSON file.

### Sample documents (paystubs)

The four synthetic paystub **PDFs** are not duplicated here — they live in
[`../ai-extract-word-level-citation/sample_data/`](../ai-extract-word-level-citation/sample_data)
(`paystub_synth_[1-4].pdf`), generated together with the ground-truth labels below
by `scripts/generate_sample_paystubs.py` (repo root) so the two always match.

Recipe 00 reads the PDFs from a **UC volume**, not from the repo, so upload them
first — e.g. with the repo-root helper:

```bash
scripts/upload_pdfs.sh \
  --profile DEFAULT \
  --catalog fins_genai --schema unstructured_documents \
  --volume paystubs --folder "" \
  --source ai-extract-word-level-citation/sample_data
```

then point Recipe 00's `catalog` / `schema` / `volume` / `volume_subdir` widgets at
that location.

### Sample ground truth (paystubs)

Two curated answer files ship in `sample_data/`, both labelling those four paystubs
so the labels match the documents exactly:

- **`paystub_ground_truth.csv`** — the **flat** labels for Recipe 03. Columns use
  the pipeline's field names:

  ```
  doc_id             | employee_name | employer_name               | social_security_number | ... | gross__current_amount | net_pay__current_amount
  paystub_synth_1.pdf| Jordan Rivera | Metro City Transit Authority | XXX-XX-4821            | ... | 1,884.00              | 1,387.35
  paystub_synth_4.pdf| Priya Nair    | Cedar Valley Health System  |                        | ... | 2,115.38              | 1,580.63   <- SSN absent (TN/FN case)
  ```

- **`paystub_ground_truth_ai_extract.jsonl`** — the **nested** labels for Recipe
  04, one JSON object per line (`doc_id` + a nested `ground_truth`). JSONL avoids
  the JSON-in-CSV escaping that Spark's CSV reader mangles. Recipe 00 joins this
  with the extraction output to build `{prefix}_extracted_nested`.

  `build_ai_extract_ground_truth.py` regenerates this JSONL from the flat CSV.

To use the flat labels with Recipe 03 directly (without Recipe 00): load the CSV
into a table, set `ground_truth_table` to it, `gt_id_col=doc_id`, and `digit_fields`
to the numeric/date fields for strict matching, e.g.
`social_security_number,gross__current_amount,gross__year_to_date_amount,net_pay__current_amount,net_pay__year_to_date_amount`.
The remaining fields (names, marital status) fall through to fuzzy text matching.

---

## Requirements

| Component | Requirement |
|---|---|
| `ai_parse_document` (Recipe 00) | DBR 17.1+ / serverless env v3+ |
| `ai_extract` 2.1 + confidence (Recipes 00, 04) | DBR 18.2+ / serverless env v3+ |
| Recipes 01 & 02 | `matplotlib`, `numpy`, `pandas` — pre-installed on DBR |
| Recipes 03 & 04 | `%pip install "mlflow[databricks]>=3.1.0"` (installed by the notebook); Recipe 04 also installs `scipy`; a serving endpoint for the judge model (default `databricks:/databricks-claude-sonnet-4-6`) |

## How to run

Each notebook is independent — open it, set the widgets at the top, and run top to
bottom.

- **00** needs the sample PDFs in a UC volume (widgets `volume` / `volume_subdir`,
  default `/Volumes/fins_genai/unstructured_documents/paystubs`). It
  writes the four tables above; the closing cell prints their row counts.
- **01 / 02** need no install; set `catalog` / `schema` / the table name and the
  VARIANT column, then run.
- **03** installs MLflow in the first cell (which restarts Python), so run it from
  the top. Set the flat table and the ground-truth table; `fields` can be left
  **blank** (it auto-resolves to the ground-truth table's columns, skipping
  `_`-prefixed ones like `_rescued_data`). Optionally set `digit_fields` (strict-match
  IDs/amounts) and the `mlflow_experiment` path.
- **04** installs MLflow + scipy in the first cell. Point `eval_source` at the
  nested eval table (or a CSV/JSON file); set `use_llm_judge` and the
  `mlflow_experiment` path. Results land in that MLflow experiment.

## Related assets

- [`../ai-extract-word-level-citation`](../ai-extract-word-level-citation) —
  produces the parsed / extracted / flat tables these recipes consume; source of
  the `PAYSTUB_SCHEMA` that Recipes 00 and 04 hardcode.
- [`../document-page-classify-extraction`](../document-page-classify-extraction) —
  multi-document classify + extract; its `*_bronze_parsed_docs` and per-class
  extract tables also fit recipes 01/02.

## Databricks documentation

- [`ai_extract`](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract)
  — the SQL function: advanced JSON schema (scalar / object / array types),
  `enableConfidenceScores`, and citations.
- [`ai_parse_document`](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document)
  — document parsing into the elements VARIANT that Recipes 00/01 use.
- [Agent Bricks — Information Extraction](https://docs.databricks.com/aws/en/agents/agent-bricks/info-extraction)
  — Databricks' built-in extraction evaluation and per-field scorecard, the basis
  for Recipe 04's type-aware definition.
- [Evaluate agents with MLflow](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/)
  — `mlflow.genai.evaluate`, built-in scorers (incl. `Correctness`), and custom
  scorers, used by Recipes 03 and 04.
