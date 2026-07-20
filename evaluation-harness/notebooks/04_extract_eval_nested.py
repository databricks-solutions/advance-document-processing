# Databricks notebook source
# MAGIC %md
# MAGIC # Recipe 04 — Type-Aware `ai_extract` Evaluation (Nested / Complex JSON)
# MAGIC
# MAGIC An **end-to-end** recipe: it runs `ai_extract` on a parsed-document eval set and
# MAGIC scores the output against ground truth using **type-aware rules** (documented in
# MAGIC `../README.md`). Unlike Recipe 03 (which requires a *flat* table and applies one
# MAGIC hybrid comparator to every field), this recipe consumes the **raw nested
# MAGIC `ai_extract` VARIANT** and scores each field by its declared type:
# MAGIC
# MAGIC | Type | Row rule | Summary metric |
# MAGIC |---|---|---|
# MAGIC | `string` | normalized match → **LLM-judge fallback** (built-in `Correctness`) | `accuracy` |
# MAGIC | `number` / `integer` | exact match | `accuracy` |
# MAGIC | `boolean` / `enum` | exact match | `accuracy` + per-value `precision`/`recall`/`f1` |
# MAGIC | `object` | recurse per child; score = mean of child scores | `object_score_mean` + per-child metrics |
# MAGIC | `array` | optimal item matching → soft F1 | `soft_f1_mean` |
# MAGIC | `array` of `object` | optimal object matching (sim = mean of field scores) → soft F1 | `soft_f1_mean` + per-child `soft_f1_mean` |
# MAGIC
# MAGIC **Schema is the single source of truth.** `PAYSTUB_SCHEMA` (hardcoded below, copied
# MAGIC from `ai-extract-word-level-citation/notebooks/01_parse_extract_citations.py`)
# MAGIC drives *both* the `ai_extract` call (`json.dumps` → the extraction schema) *and*
# MAGIC scoring (each node's `type` selects the comparator). Matching and recursion run in
# MAGIC **pure Python** on the driver, using `scipy.optimize.linear_sum_assignment` for
# MAGIC optimal array matching.
# MAGIC
# MAGIC **Input contract — one eval table** with columns:
# MAGIC
# MAGIC - `path` (or `id_col`) — document identifier.
# MAGIC - `input` — the **parsed-document VARIANT** (`ai_parse_document` output). `ai_extract`
# MAGIC   runs on this. Stored as VARIANT in a Delta table, or a JSON string when read from CSV.
# MAGIC - `ground_truth` — the nested JSON of correct values (shape produced by
# MAGIC   `../sample_data/build_ai_extract_ground_truth.py`).
# MAGIC
# MAGIC Recipe 00 builds this eval table (`{prefix}_extracted_nested`, the default
# MAGIC `eval_source`) by joining the parsed input + nested `ai_extract` output with the
# MAGIC labels in `../sample_data/paystub_ground_truth_ai_extract.jsonl`. Prefer a **Delta
# MAGIC table** for the large `input` column — CSV ingestion of big JSON strings is fragile.
# MAGIC
# MAGIC > **Sample-data caveat:** the shipped ground truth covers only scalar fields plus
# MAGIC > the `gross` / `net_pay` objects. The `array` / `array-of-object` code paths
# MAGIC > (`income_item`, `deduction_item`, the address objects) are implemented to spec
# MAGIC > but are **not exercised** by this sample until richer ground truth exists.
# MAGIC
# MAGIC Requires **DBR 18.2+ / serverless v3+** for `ai_extract` 2.1 and **MLflow 3** for
# MAGIC the `Correctness` judge.

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]>=3.1.0" scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
# eval_source: a table name (resolved under catalog.schema) OR a file path ending in
# .csv/.json (read via pandas). Must expose id_col / input_col / gt_col.
dbutils.widgets.text("eval_source", "paystub_eval_extracted_nested")
dbutils.widgets.text("id_col", "path")
dbutils.widgets.text("input_col", "input")
dbutils.widgets.text("gt_col", "ground_truth")
dbutils.widgets.text("predictions_table", "paystub_eval_predictions")
dbutils.widgets.text("extract_version", "2.1")
dbutils.widgets.dropdown("use_llm_judge", "true", ["true", "false"])
dbutils.widgets.text("judge_model", "databricks:/databricks-claude-sonnet-4-6")
dbutils.widgets.text("high_conf_threshold", "0.8")
dbutils.widgets.text("mlflow_experiment", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
eval_source = dbutils.widgets.get("eval_source").strip()
id_col = dbutils.widgets.get("id_col")
input_col = dbutils.widgets.get("input_col")
gt_col = dbutils.widgets.get("gt_col")
pred_fqn = f"{catalog}.{schema}.{dbutils.widgets.get('predictions_table')}"
extract_version = dbutils.widgets.get("extract_version")
use_llm_judge = dbutils.widgets.get("use_llm_judge") == "true"
judge_model = dbutils.widgets.get("judge_model")
high_conf_threshold = float(dbutils.widgets.get("high_conf_threshold"))
mlflow_experiment = dbutils.widgets.get("mlflow_experiment").strip()

print(f"eval_source = {eval_source}")
print(f"columns: id={id_col} input={input_col} gt={gt_col}")
print(f"predictions_table = {pred_fqn}")
print(f"use_llm_judge = {use_llm_judge}  judge_model = {judge_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extraction schema (single source of truth)
# MAGIC
# MAGIC Hardcoded copy of `PAYSTUB_SCHEMA` from
# MAGIC `ai-extract-word-level-citation/notebooks/01_parse_extract_citations.py`. The same
# MAGIC parse feeds both notebooks, so this schema is the shared contract. `json.dumps` of
# MAGIC it is the schema passed to `ai_extract`; each node's `type` also selects the
# MAGIC scoring comparator below. `description` keys are used by `ai_extract` and ignored
# MAGIC by scoring. Edit here if the extraction schema changes upstream.

# COMMAND ----------

# DBTITLE 1,PAYSTUB_SCHEMA
PAYSTUB_SCHEMA = {
    "employee_name": {"type": "string", "description": "The employee's full name printed on the paystub."},
    "employer_name": {"type": "string", "description": "The employer / company legal name printed on the paystub."},
    "social_security_number": {"type": "string", "description": "The employee's Social Security Number (taxpayer identifier) as printed."},
    "taxable_marital_status": {"type": "string", "description": "The taxable marital status (e.g. Married | Unmarried | Separated | Single)."},
    "borrower_employee_address": {
        "type": "object",
        "description": "Employee address information.",
        "properties": {
            "address_line": {"type": "string", "description": "The street address line in the employee address block."},
            "city_name": {"type": "string", "description": "The city in the employee address block."},
            "state_code": {"type": "string", "description": "The state code in the employee address block."},
            "zip_code": {"type": "string", "description": "The ZIP / postal code in the employee address block."},
        },
    },
    "borrower_employer_address": {
        "type": "object",
        "description": "Employer address information.",
        "properties": {
            "address_line": {"type": "string", "description": "The street address line in the employer address block."},
            "city_name": {"type": "string", "description": "The city in the employer address block."},
            "state_code": {"type": "string", "description": "The state code in the employer address block."},
            "zip_code": {"type": "string", "description": "The ZIP / postal code in the employer address block."},
        },
    },
    "hire_date": {"type": "string", "description": "The employee hire date as printed."},
    "ets_date": {"type": "string", "description": "The ETS (term-of-service expiration) date as printed, if present."},
    "pay_date": {"type": "string", "description": "The paycheck date as printed."},
    "period_start_date": {"type": "string", "description": "The pay-period start date as printed."},
    "period_end_date": {"type": "string", "description": "The pay-period end date as printed."},
    "payment_frequency": {"type": "string", "description": "The payment frequency type (e.g. Weekly | Biweekly | Monthly)."},
    "income_source_type": {"type": "string", "description": "Whether the employee is paid Hourly or Salaried."},
    "gross": {
        "type": "object",
        "description": "Gross pay amounts.",
        "properties": {
            "current_amount": {"type": "number", "description": "The gross pay for the current pay period."},
            "year_to_date_amount": {"type": "number", "description": "The year-to-date gross pay."},
        },
    },
    "net_pay": {
        "type": "object",
        "description": "Net pay amounts.",
        "properties": {
            "current_amount": {"type": "number", "description": "The net pay for the current pay period."},
            "year_to_date_amount": {"type": "number", "description": "The year-to-date net pay."},
        },
    },
    "adjustments_gross_income": {
        "type": "object",
        "description": "Adjustments to gross income amounts.",
        "properties": {
            "current_amount": {"type": "number", "description": "The current-period adjustments gross income amount."},
            "year_to_date_amount": {"type": "number", "description": "The year-to-date adjustments gross income amount."},
        },
    },
    "income_item": {
        "type": "array",
        "description": "Repeating earning/income rows on the paystub; one object per printed row.",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "The earning item description (e.g. Regular, Overtime)."},
                "period_hours": {"type": "number", "description": "The hours for this earning item this period."},
                "rate": {"type": "number", "description": "The pay rate for this earning item."},
                "current_amount": {"type": "number", "description": "The current-period amount for this earning item."},
                "year_to_date_amount": {"type": "number", "description": "The year-to-date amount for this earning item."},
            },
        },
    },
    "deduction_item": {
        "type": "array",
        "description": "Repeating deduction rows on the paystub; one object per printed row.",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "The deduction item description (e.g. Federal Tax, Medicare)."},
                "period_amount": {"type": "number", "description": "The current-period amount for this deduction item."},
                "year_to_date_amount": {"type": "number", "description": "The year-to-date amount for this deduction item."},
            },
        },
    },
}

INSTRUCTIONS = (
    "The uploaded documents are US consumer paystubs (monthly or weekly). "
    "Extract the named fields. Use the exact text as printed; do not reformat "
    "dates or amounts. Leave a field blank if it is not present. income_item, "
    "deduction_item are repeating rows: emit one object per printed row, "
    "including zero-activity rows that are actually present."
)

# Top-level fields to evaluate = every property in the schema.
FIELDS = list(PAYSTUB_SCHEMA.keys())
print(f"{len(FIELDS)} top-level fields: {FIELDS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Run `ai_extract` on the eval set
# MAGIC
# MAGIC Read the eval source (Delta table or CSV/JSON file), then run `ai_extract` on the
# MAGIC `input` VARIANT. `parse_json(CAST(input AS STRING))` normalizes the column whether
# MAGIC it arrives as a VARIANT (table) or a JSON string (CSV). Predictions are persisted
# MAGIC to `predictions_table` for reuse.

# COMMAND ----------

import json

import pandas as pd
from pyspark.sql import functions as F

# --- read the eval source into a Spark DataFrame with canonical column names ---
if eval_source.lower().endswith((".csv", ".json")):
    # pandas handles quoted, multi-line JSON cells that Spark's CSV reader mangles.
    _pdf = pd.read_json(eval_source) if eval_source.lower().endswith(".json") else pd.read_csv(eval_source)
    _src = spark.createDataFrame(_pdf[[id_col, input_col, gt_col]])
else:
    _fqn = eval_source if "." in eval_source else f"{catalog}.{schema}.{eval_source}"
    _src = spark.table(_fqn).select(id_col, input_col, gt_col)

_src = _src.select(
    F.col(id_col).cast("string").alias("doc_id"),
    F.col(input_col).alias("doc_input"),
    F.col(gt_col).cast("string").alias("ground_truth"),
)
_src.createOrReplaceTempView("_eval_src")
print(f"eval rows: {_src.count()}")

# --- ai_extract over the parsed VARIANT ---
schema_sql = json.dumps(PAYSTUB_SCHEMA).replace("'", "''")
instr_sql = INSTRUCTIONS.replace("'", "''")

spark.sql(f"""
CREATE OR REPLACE TABLE {pred_fqn} AS
SELECT
  doc_id,
  ground_truth,
  ai_extract(
    parse_json(CAST(doc_input AS STRING)),
    '{schema_sql}',
    options => map(
      'version',                '{extract_version}',
      'enableConfidenceScores', 'true',
      'instructions',           '{instr_sql}'
    )
  ) AS extracted
FROM _eval_src
""")

display(spark.sql(f"SELECT doc_id, to_json(extracted) AS extracted_json FROM {pred_fqn}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Normalize predictions & ground truth to nested dicts
# MAGIC
# MAGIC Predictions come from the `ai_extract` VARIANT: read `$.response` as JSON, then
# MAGIC **strip the wrappers** — a leaf `{value, confidence_score, citation_ids}` collapses
# MAGIC to its `value`, recursing through objects and arrays — leaving a plain nested dict
# MAGIC that mirrors the ground-truth shape. Per-leaf `confidence_score` is captured
# MAGIC separately for the calibration appendix. Both sides are keyed by a normalized
# MAGIC document id.
# MAGIC
# MAGIC > The pure-Python normalization + scoring functions below (this cell through the
# MAGIC > recursive scorer) are kept **inline** so the notebook is self-contained. The
# MAGIC > canonical, unit-tested copy lives in `../src/nested_eval.py`
# MAGIC > (`evaluation-harness/tests/test_nested_eval.py`) — keep the two in sync.

# COMMAND ----------

import math
import re

WRAPPER_KEYS = {"value", "confidence_score", "citation_ids"}


def is_wrapper(node):
    return isinstance(node, dict) and "value" in node and set(node).issubset(WRAPPER_KEYS)


def strip_wrappers(node):
    """Collapse ai_extract leaf wrappers to plain values; recurse objects/arrays."""
    if is_wrapper(node):
        return strip_wrappers(node["value"])
    if isinstance(node, dict):
        return {k: strip_wrappers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [strip_wrappers(x) for x in node]
    return node


def walk_confidence(node, prefix, out):
    """Record confidence_score for scalar-leaf wrappers, keyed by dotted path."""
    if is_wrapper(node):
        if node.get("confidence_score") is not None and not isinstance(node["value"], (dict, list)):
            out[prefix] = float(node["confidence_score"])
        return
    if isinstance(node, dict):
        for k, v in node.items():
            walk_confidence(v, f"{prefix}.{k}" if prefix else k, out)
    # arrays intentionally skipped for calibration


def norm_key(s):
    base = re.sub(r"^.*/", "", str(s))       # basename
    base = re.sub(r"\.[^.]+$", "", base)     # strip extension
    return re.sub(r"[^A-Za-z0-9]", "", base).lower()


rows = spark.sql(f"SELECT doc_id, ground_truth, to_json(extracted) AS extracted_json FROM {pred_fqn}").collect()

records = {}  # join_key -> {"doc_id", "pred", "conf", "gt"}
for r in rows:
    resp = (json.loads(r["extracted_json"]) or {}).get("response", {})
    conf = {}
    walk_confidence(resp, "", conf)
    gt = json.loads(r["ground_truth"]) if isinstance(r["ground_truth"], str) else (r["ground_truth"] or {})
    records[norm_key(r["doc_id"])] = {
        "doc_id": r["doc_id"],
        "pred": strip_wrappers(resp),
        "conf": conf,
        "gt": gt,
    }

keys = sorted(records)
print(f"scored documents: {len(keys)}")
assert keys, "No rows — check eval_source and column names."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Type-aware recursive scorer
# MAGIC
# MAGIC `score_node` dispatches on the schema type, records per-field metrics into an
# MAGIC accumulator, and returns a row score in `[0, 1]`. `sim_node` is the **pure**
# MAGIC similarity used to build the cost matrix during array matching (no recording, no
# MAGIC judge). Absence is handled uniformly: both-absent = 1.0, one-absent = 0.0.

# COMMAND ----------

# DBTITLE 1,Scoring primitives
import numpy as np
from scipy.optimize import linear_sum_assignment


def node_type(node):
    t = node.get("type")
    if "enum" in node:
        return "enum"
    if t == "array":
        return "array_object" if node.get("items", {}).get("type") == "object" else "array"
    return t


def is_absent(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def norm_str(v):
    return "" if is_absent(v) else re.sub(r"\s+", " ", str(v).strip().lower())


def to_num(v):
    if is_absent(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^0-9.\-]", "", str(v))
    try:
        return float(s)
    except ValueError:
        return None


def leaf_sim(node, a, b):
    """Pure similarity for a scalar leaf (normalized only, no LLM judge)."""
    aa, ba = is_absent(a), is_absent(b)
    if aa and ba:
        return 1.0
    if aa or ba:
        return 0.0
    t = node_type(node)
    if t in ("number", "integer"):
        na, nb = to_num(a), to_num(b)
        return 1.0 if (na is not None and nb is not None and abs(na - nb) < 1e-9) else 0.0
    return 1.0 if norm_str(a) == norm_str(b) else 0.0


def sim_node(node, a, b):
    """Pure recursive similarity in [0,1] — used to build array-matching matrices."""
    t = node_type(node)
    if t == "object":
        props = node["properties"]
        a, b = a or {}, b or {}
        scores = [sim_node(c, a.get(n), b.get(n)) for n, c in props.items()]
        return sum(scores) / len(scores) if scores else 1.0
    if t in ("array", "array_object"):
        return soft_f1(node["items"], a or [], b or [])
    return leaf_sim(node, a, b)


def optimal_pairs(item_node, A, B):
    """Optimal 1:1 matching maximizing total similarity. Returns (pairs, total)."""
    M = np.zeros((len(A), len(B)))
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            M[i, j] = sim_node(item_node, x, y)
    ri, ci = linear_sum_assignment(-M)
    pairs = [(int(i), int(j), float(M[i, j])) for i, j in zip(ri, ci)]
    return pairs, sum(p[2] for p in pairs)


def soft_f1(item_node, A, B):
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    _, total = optimal_pairs(item_node, A, B)
    p, r = total / len(A), total / len(B)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# COMMAND ----------

# DBTITLE 1,Recursive scorer (records metrics)
from collections import defaultdict


def new_acc():
    return {
        "accuracy": defaultdict(list),      # path -> [0/1]        (string/number/int/enum/bool)
        "confusion": defaultdict(list),     # path -> [(pred,exp)] (enum/bool per-value P/R/F1)
        "object_score": defaultdict(list),  # path -> [float]      (object_score_mean)
        "soft_f1": defaultdict(list),       # path -> [float]      (array & array-of-object)
        "leaf": [],                         # (doc, path, score, conf) for calibration
    }


def score_node(node, pred, exp, path, acc, ctx):
    t = node_type(node)

    if t == "object":
        props = node["properties"]
        pred, exp = pred or {}, exp or {}
        child_scores = [
            score_node(c, pred.get(n), exp.get(n), f"{path}.{n}" if path else n, acc, ctx)
            for n, c in props.items()
        ]
        s = sum(child_scores) / len(child_scores) if child_scores else 1.0
        acc["object_score"][path].append(s)
        return s

    if t == "array":
        s = soft_f1(node["items"], pred or [], exp or [])
        acc["soft_f1"][path].append(s)
        return s

    if t == "array_object":
        item_node, A, B = node["items"], pred or [], exp or []
        if not A and not B:
            s, pairs = 1.0, []
        elif not A or not B:
            s, pairs = 0.0, []
        else:
            pairs, total = optimal_pairs(item_node, A, B)
            p, r = total / len(A), total / len(B)
            s = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        acc["soft_f1"][path].append(s)
        # per-child soft F1 across matched objects
        for cname, cnode in item_node["properties"].items():
            if not A and not B:
                acc["soft_f1"][f"{path}.{cname}"].append(1.0)
                continue
            if not A or not B:
                acc["soft_f1"][f"{path}.{cname}"].append(0.0)
                continue
            ctot = sum(
                sim_node(cnode,
                         A[i].get(cname) if isinstance(A[i], dict) else None,
                         B[j].get(cname) if isinstance(B[j], dict) else None)
                for i, j, _ in pairs
            )
            cp, cr = ctot / len(A), ctot / len(B)
            acc["soft_f1"][f"{path}.{cname}"].append(2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0)
        return s

    # --- scalar leaves ---
    aa, ea = is_absent(pred), is_absent(exp)
    if aa and ea:
        s = 1.0
    elif aa or ea:
        s = 0.0
    elif t in ("number", "integer"):
        s = leaf_sim(node, pred, exp)
    elif t in ("boolean", "enum"):
        s = 1.0 if norm_str(pred) == norm_str(exp) else 0.0
    else:  # string
        if norm_str(pred) == norm_str(exp):
            s = 1.0
        elif ctx.get("judge") is not None:
            s = 1.0 if ctx["judge"](path, pred, exp) else 0.0
        else:
            s = 0.0

    acc["accuracy"][path].append(s)
    if t in ("boolean", "enum"):
        acc["confusion"][path].append((norm_str(pred), norm_str(exp)))
    acc["leaf"].append((ctx["doc_id"], path, s, ctx["conf"].get(path)))
    return s

# COMMAND ----------

# MAGIC %md
# MAGIC ### String LLM-judge fallback (built-in `Correctness`)
# MAGIC
# MAGIC Per the spec, a string field that fails the normalized match is re-checked by an
# MAGIC LLM. We use MLflow's built-in `Correctness` judge, invoked **once per failing
# MAGIC (field, doc) pair** so each call judges exactly one value — the call is naturally
# MAGIC sparse (it fires only on misses). Set `use_llm_judge=false` for deterministic,
# MAGIC normalized-only string scoring.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Correctness, scorer
from mlflow.entities import Feedback

if mlflow_experiment:
    mlflow.set_experiment(mlflow_experiment)

_correctness = Correctness(model=judge_model)


def judge_equiv(path, pred, exp):
    """True if the built-in Correctness judge rules pred equivalent to exp."""
    try:
        fb = _correctness(
            inputs={"field": path},
            outputs=str(pred),
            expectations={"expected_response": str(exp)},
        )
        val = fb.value if isinstance(fb, Feedback) else fb
        return str(val).strip().lower() in ("yes", "true", "1")
    except Exception as e:
        print(f"  judge error on {path!r} ({pred!r} vs {exp!r}): {e} — treating as mismatch")
        return False


judge_fn = judge_equiv if use_llm_judge else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score every document

# COMMAND ----------

acc = new_acc()
row_scores = []       # per (doc, top-field) row score, long form
doc_field_score = {}  # doc -> {field: score} for the MLflow pass-through scorers

for k in keys:
    doc_id = records[k]["doc_id"]
    pred, gt = records[k]["pred"], records[k]["gt"]
    ctx = {"doc_id": doc_id, "conf": records[k]["conf"], "judge": judge_fn}
    doc_field_score[doc_id] = {}
    for field in FIELDS:
        s = score_node(PAYSTUB_SCHEMA[field], pred.get(field), gt.get(field), field, acc, ctx)
        row_scores.append({"doc_id": doc_id, "field": field, "row_score": s})
        doc_field_score[doc_id][field] = s

row_scores_pdf = pd.DataFrame(row_scores)
display(spark.createDataFrame(row_scores_pdf).orderBy("field", "doc_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary metrics (per the type-aware definition in `../README.md`)
# MAGIC
# MAGIC - scalar fields → `<field>_accuracy`
# MAGIC - objects → `<field>_object_score_mean` (+ per-child `<field>.<child>_accuracy`)
# MAGIC - arrays / arrays-of-object → `<field>_soft_f1_mean` (+ per-child `<field>.<child>_soft_f1_mean`)
# MAGIC - boolean / enum → per-value `precision` / `recall` / `f1`

# COMMAND ----------

summary = {}  # metric_name -> value

for path, vals in acc["accuracy"].items():
    summary[f"{path}_accuracy"] = sum(vals) / len(vals)
for path, vals in acc["object_score"].items():
    summary[f"{path}_object_score_mean"] = sum(vals) / len(vals)
for path, vals in acc["soft_f1"].items():
    summary[f"{path}_soft_f1_mean"] = sum(vals) / len(vals)

# per-value precision/recall/f1 for boolean & enum fields
for path, pairs in acc["confusion"].items():
    values = sorted({v for pair in pairs for v in pair if v != ""})
    for v in values:
        tp = sum(1 for p, e in pairs if p == v and e == v)
        fp = sum(1 for p, e in pairs if p == v and e != v)
        fn = sum(1 for p, e in pairs if p != v and e == v)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary[f"{path}_{v}_precision"] = precision
        summary[f"{path}_{v}_recall"] = recall
        summary[f"{path}_{v}_f1"] = f1

summary_pdf = (
    pd.DataFrame(sorted(summary.items()), columns=["metric", "value"])
    .sort_values("metric")
    .reset_index(drop=True)
)
display(spark.createDataFrame(summary_pdf))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log to MLflow
# MAGIC
# MAGIC Logs the summary metrics and runs `mlflow.genai.evaluate` with the built-in
# MAGIC `Correctness` judge (holistic, nested outputs vs. `expected_response`) plus one
# MAGIC pass-through scorer per top-level field surfacing the row score computed above.

# COMMAND ----------

eval_data = []
for k in keys:
    doc_id = records[k]["doc_id"]
    eval_data.append(
        {
            "inputs": {"doc_id": doc_id},
            "outputs": records[k]["pred"],
            "expectations": {"expected_response": records[k]["gt"]},
        }
    )


def make_field_scorer(field):
    @scorer(name=f"{field}_score")
    def _s(inputs, outputs, expectations):
        s = doc_field_score.get(inputs["doc_id"], {}).get(field)
        return Feedback(value=float(s) if s is not None else None, rationale=f"{field} row score")

    return _s


field_scorers = [make_field_scorer(f) for f in FIELDS]

with mlflow.start_run(run_name="ai_extract_eval_nested") as run:
    mlflow.genai.evaluate(
        data=eval_data,
        scorers=[Correctness(model=judge_model), *field_scorers],
    )
    for metric, value in summary.items():
        mlflow.log_metric(metric, float(value))
    print(f"MLflow run: {run.info.run_id}  ({len(summary)} summary metrics logged)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Confidence calibration (appendix)
# MAGIC
# MAGIC `ai_extract`-specific and not part of the eval spec, but useful: bin the per-leaf
# MAGIC `confidence_score` and check whether higher confidence tracks a higher correctness
# MAGIC rate (monotonic = confidence is a reliable review signal). Covers scalar leaves
# MAGIC only (array elements have no single confidence).

# COMMAND ----------

leaf_pdf = pd.DataFrame(acc["leaf"], columns=["doc_id", "field", "score", "conf"])
cal_src = leaf_pdf.dropna(subset=["conf"]).copy()

if cal_src.empty:
    print("No per-leaf confidence scores found — skipping calibration.")
else:
    def conf_bin(c):
        if c >= 0.9:
            return "0.9-1.0"
        if c >= 0.8:
            return "0.8-0.9"
        if c >= 0.7:
            return "0.7-0.8"
        return "<0.7"

    cal_src["conf_bin"] = cal_src["conf"].map(conf_bin)
    cal = (
        cal_src.groupby("conf_bin")
        .agg(n=("score", "size"), correct_rate=("score", "mean"))
        .reset_index()
        .sort_values("conf_bin")
    )
    display(spark.createDataFrame(cal))

    order = ["<0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    rates = [cal.set_index("conf_bin").loc[b, "correct_rate"] for b in order if b in cal["conf_bin"].values]
    monotonic = all(x <= y + 1e-9 for x, y in zip(rates, rates[1:]))
    print(f"correctness rates low→high: {[round(r, 3) for r in rates]}")
    print(f"calibration monotonic (confidence tracks correctness): {monotonic}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confidently wrong (high confidence, still incorrect)

# COMMAND ----------

if not leaf_pdf.empty:
    danger = leaf_pdf[(leaf_pdf["conf"] >= high_conf_threshold) & (leaf_pdf["score"] < 1.0)]
    display(spark.createDataFrame(danger.sort_values("conf", ascending=False)))
