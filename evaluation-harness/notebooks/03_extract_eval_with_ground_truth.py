# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %md
# MAGIC # Recipe 03 — `ai_extract` Evaluation with Ground Truth (MLflow GenAI)
# MAGIC
# MAGIC A **measure-only** recipe. Given a **flat** `ai_extract` table (one column per
# MAGIC field `value`, plus a per-field confidence column) and a **ground-truth** table,
# MAGIC it scores extraction quality per field and logs everything to MLflow:
# MAGIC
# MAGIC 1. **Hybrid comparator** — fuzzy (Levenshtein) match for free-text fields,
# MAGIC    strict normalized-digit equality for ID / date fields.
# MAGIC 2. **Absence-as-a-class** — a `NULL` prediction is a real prediction, so we
# MAGIC    classify every field as `TP` / `TN` / `FP` / `FN` / `FP_FN`.
# MAGIC 3. **Precision / recall / F1** per field (SQL).
# MAGIC 4. **`mlflow.genai.evaluate`** with a built-in `Correctness` LLM judge plus one
# MAGIC    pass-through scorer per field, logging the P/R/F1 metrics to the same run.
# MAGIC 5. **Calibration** — does higher `ai_extract` confidence actually mean a higher
# MAGIC    correctness rate? Bin the confidence and check monotonicity.
# MAGIC
# MAGIC It does **not** call `ai_extract` itself — run an extract stage first (e.g.
# MAGIC `ai-extract-word-level-citation` writes `*_extracted_flat`).
# MAGIC
# MAGIC **Input contracts**
# MAGIC
# MAGIC - *Extracted (flat)* — one row per document; per field a `value` column named
# MAGIC   `<field>` and a confidence column named `<field><conf_suffix>` (default suffix
# MAGIC   `_extract_conf`); plus `doc_id_col`.
# MAGIC   **Schema must be flat** — each field and its confidence must be a top-level
# MAGIC   column (e.g. `employee_name`, `employee_name_extract_conf`). Nested schemas
# MAGIC   such as `results.employee_name.value` are not supported and must be flattened
# MAGIC   in an upstream step before running this notebook.
# MAGIC - *Ground truth* — one row per document; a `gt_id_col` and one column per field
# MAGIC   holding the correct value (or `NULL`/empty when the field is genuinely absent).
# MAGIC
# MAGIC The two sides are joined on a normalized key derived from the document
# MAGIC identifier (basename, extension stripped, lowercased, non-alphanumerics removed).

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]>=3.1.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("extracted_flat_table", "paystub_eval_extracted_flat")
dbutils.widgets.text("ground_truth_table", "paystub_eval_ground_truth_simple")
dbutils.widgets.text("doc_id_col", "path")
dbutils.widgets.text("gt_id_col", "doc_id")
dbutils.widgets.text("conf_suffix", "_extract_conf")
dbutils.widgets.text("fields", "")
dbutils.widgets.text("digit_fields", "")
dbutils.widgets.text("fuzzy_threshold", "0.85")
dbutils.widgets.text("high_conf_threshold", "0.8")
dbutils.widgets.text("judge_model", "databricks:/databricks-claude-sonnet-4-6")
dbutils.widgets.text("mlflow_experiment", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
ext_fqn = f"{catalog}.{schema}.{dbutils.widgets.get('extracted_flat_table')}"
gt_fqn = f"{catalog}.{schema}.{dbutils.widgets.get('ground_truth_table')}"
doc_id_col = dbutils.widgets.get("doc_id_col")
gt_id_col = dbutils.widgets.get("gt_id_col")
conf_suffix = dbutils.widgets.get("conf_suffix")
fuzzy_threshold = float(dbutils.widgets.get("fuzzy_threshold"))
high_conf_threshold = float(dbutils.widgets.get("high_conf_threshold"))
judge_model = dbutils.widgets.get("judge_model")
mlflow_experiment = dbutils.widgets.get("mlflow_experiment").strip()

fields_raw = dbutils.widgets.get("fields").strip()
digit_fields_raw = dbutils.widgets.get("digit_fields").strip()

# Fields default to every non-id column of the ground-truth table.
if fields_raw:
    fields = [f.strip() for f in fields_raw.split(",") if f.strip()]
else:
    fields = [c for c in spark.table(gt_fqn).columns if c not in (gt_id_col, "join_key")]

digit_fields = set(f.strip() for f in digit_fields_raw.split(",") if f.strip())

assert fields, "No fields resolved — set the `fields` widget."
print(f"ext_fqn = {ext_fqn}")
print(f"gt_fqn  = {gt_fqn}")
print(f"{len(fields)} fields: {fields}")
print(f"digit (strict-match) fields: {sorted(digit_fields)}")
print(f"fuzzy_threshold={fuzzy_threshold}  high_conf_threshold={high_conf_threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Join extracted ↔ ground truth
# MAGIC
# MAGIC Both sides are keyed by a normalized document id. We alias predictions as
# MAGIC `pred_<field>` / `conf_<field>` and ground truth as `gt_<field>` so the
# MAGIC comparator below can address them uniformly.

# COMMAND ----------

# DBTITLE 1,Cell 7
from pyspark.sql import functions as F

# exclude Databricks-internal rescue/metadata columns (e.g. _rescued_data)
fields = [f for f in fields if not f.startswith("_")]


def norm_key(col):
    base = F.regexp_replace(col.cast("string"), r"^.*/", "")   # basename
    base = F.regexp_replace(base, r"\.[^.]+$", "")             # strip extension
    return F.lower(F.regexp_replace(base, r"[^A-Za-z0-9]", ""))


# The extracted table must be FLAT: one row per document, with each field as a
# top-level column named <field> and a paired <field><conf_suffix> confidence
# column. Nested schemas (e.g. results.field.value) must be flattened upstream
# before this notebook is run.
ext = (
    spark.table(ext_fqn)
    .select(
        F.col(doc_id_col).alias("doc_id"),
        *[F.col(f"`{f}`").alias(f"pred_{f}") for f in fields],
        *[F.col(f"`{f}{conf_suffix}`").alias(f"conf_{f}") for f in fields],
    )
    .withColumn("join_key", norm_key(F.col("doc_id")))
)

gt = (
    spark.table(gt_fqn)
    .select(
        F.col(gt_id_col).alias("gt_id"),
        *[F.col(f"`{f}`").alias(f"gt_{f}") for f in fields],
    )
    .withColumn("join_key", norm_key(F.col("gt_id")))
)

joined = ext.join(gt, "join_key", "inner")
matched = joined.count()
print(f"joined rows (extracted ∩ ground truth): {matched}")
assert matched > 0, "No rows joined — check doc_id_col / gt_id_col and the normalized key."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Per-field classification (absence-aware, hybrid comparator)
# MAGIC
# MAGIC For each field:
# MAGIC
# MAGIC | ground truth | prediction | match? | class |
# MAGIC |---|---|---|---|
# MAGIC | absent | absent | — | `TN` |
# MAGIC | absent | present | — | `FP` (hallucinated) |
# MAGIC | present | absent | — | `FN` (missed) |
# MAGIC | present | present | yes | `TP` |
# MAGIC | present | present | no | `FP_FN` (wrong value — counts against both P and R) |
# MAGIC
# MAGIC Match rule: **digit fields** compare digits-only equality; **all other fields**
# MAGIC use a normalized Levenshtein ratio `>= fuzzy_threshold`.

# COMMAND ----------

def norm_amount(col):
    # numeric value of the digits (decimal-insensitive), or NULL if non-numeric
    return F.try_cast(F.regexp_replace(col.cast("string"), r"[^0-9.]", ""), "double")


def norm_text(col):
    return F.lower(F.trim(col.cast("string")))


def is_blank(col):
    return col.isNull() | (F.trim(col.cast("string")) == "")


for f in fields:
    pred, gt_col = F.col(f"pred_{f}"), F.col(f"gt_{f}")
    pred_blank, gt_blank = is_blank(pred), is_blank(gt_col)

    if f in digit_fields:
        # Compare numerically so 1884.0 == 1884.00 == "1,884.00"; fall back to exact
        # normalized text when either side has no parseable number (e.g. "N/A"), so two
        # different non-numeric values are not both reduced to "" and called equal.
        pred_amt, gt_amt = norm_amount(pred), norm_amount(gt_col)
        match = F.when(
            pred_amt.isNotNull() & gt_amt.isNotNull(),
            F.abs(pred_amt - gt_amt) < F.lit(0.005),
        ).otherwise(norm_text(pred) == norm_text(gt_col))
    else:
        a, b = norm_text(pred), norm_text(gt_col)
        ratio = F.lit(1.0) - (F.levenshtein(a, b) / F.greatest(F.length(a), F.length(b), F.lit(1)))
        match = ratio >= F.lit(fuzzy_threshold)

    cls = (
        F.when(pred_blank & gt_blank, "TN")
        .when(gt_blank & ~pred_blank, "FP")
        .when(~gt_blank & pred_blank, "FN")
        .when(match, "TP")
        .otherwise("FP_FN")
    )
    joined = joined.withColumn(f"cls_{f}", cls)
    joined = joined.withColumn(f"correct_{f}", F.col(f"cls_{f}").isin("TP", "TN"))


# long form: one row per (doc, field)
cls_structs = [
    F.struct(
        F.lit(f).alias("field"),
        F.col(f"cls_{f}").alias("cls"),
        F.col(f"conf_{f}").alias("conf"),
        F.col(f"correct_{f}").alias("correct"),
    )
    for f in fields
]
long = joined.select("doc_id", F.explode(F.array(*cls_structs)).alias("r")).select("doc_id", "r.*")
display(long.groupBy("field", "cls").count().orderBy("field", "cls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Precision / recall / F1 per field
# MAGIC
# MAGIC `FP_FN` (wrong value) counts in **both** the precision and recall denominators,
# MAGIC so a confidently wrong extraction is penalized twice.

# COMMAND ----------

metrics = (
    long.groupBy("field")
    .agg(
        F.sum((F.col("cls") == "TP").cast("int")).alias("tp"),
        F.sum((F.col("cls") == "TN").cast("int")).alias("tn"),
        F.sum((F.col("cls") == "FP").cast("int")).alias("fp"),
        F.sum((F.col("cls") == "FN").cast("int")).alias("fn"),
        F.sum((F.col("cls") == "FP_FN").cast("int")).alias("fp_fn"),
    )
    .selectExpr(
        "*",
        "tp / nullif(tp + fp + fp_fn, 0) AS precision",
        "tp / nullif(tp + fn + fp_fn, 0) AS recall",
    )
    .selectExpr("*", "2 * precision * recall / nullif(precision + recall, 0) AS f1")
    # A field that is absent-and-correct everywhere has no positives, so P/R/F1 are
    # NULL (undefined), not 0 — sort those last instead of flagging them as "worst".
    .orderBy(F.col("f1").asc_nulls_last())
)
display(metrics)
metrics_rows = metrics.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `mlflow.genai.evaluate` with a Correctness judge
# MAGIC
# MAGIC Each eval row carries the model `outputs` and **absence-aware**
# MAGIC `expected_facts` (`"<field> is not present in the document"` when ground truth
# MAGIC is absent, otherwise `"<field> is <value>"`). We run the built-in `Correctness`
# MAGIC judge plus one pass-through scorer per field (reading the class we computed
# MAGIC above), and log the P/R/F1 metrics into the same run.

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.genai.scorers import scorer, Correctness
from mlflow.entities import Feedback

if mlflow_experiment:
    mlflow.set_experiment(mlflow_experiment)

pdf = joined.toPandas()


def build_expected_facts(row):
    facts = []
    for f in fields:
        g = row[f"gt_{f}"]
        # pd.isna catches both None and NaN (toPandas renders a NULL numeric column as NaN).
        if pd.isna(g) or str(g).strip() == "":
            facts.append(f"{f} is not present in the document")
        else:
            facts.append(f"{f} is {g}")
    return facts


eval_data, cls_lookup = [], {}
for _, row in pdf.iterrows():
    eval_data.append(
        {
            "inputs": {"doc_id": row["doc_id"]},
            "outputs": {f: row[f"pred_{f}"] for f in fields},
            "expectations": {"expected_facts": build_expected_facts(row)},
        }
    )
    cls_lookup[row["doc_id"]] = {f: row[f"cls_{f}"] for f in fields}


def make_field_scorer(field):
    @scorer(name=f"{field}_correct")
    def _s(inputs, outputs, expectations):
        cls = cls_lookup.get(inputs["doc_id"], {}).get(field)
        return Feedback(value=bool(cls in ("TP", "TN")), rationale=f"{field}: {cls}")

    return _s


field_scorers = [make_field_scorer(f) for f in fields]

with mlflow.start_run(run_name="ai_extract_eval_gt") as run:
    results = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[Correctness(model=judge_model), *field_scorers],
    )
    for m in metrics_rows:
        # Log only defined metrics — a NULL (undefined) P/R/F1 should not be recorded
        # as 0.0, which would misrepresent an absent-and-correct field as a failure.
        for _metric in ("precision", "recall", "f1"):
            if m[_metric] is not None:
                mlflow.log_metric(f"{m['field']}_{_metric}", float(m[_metric]))
    print(f"MLflow run: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Confidence calibration
# MAGIC
# MAGIC Bin the per-field `ai_extract` confidence and compute the correctness rate per
# MAGIC bin. A well-calibrated model shows a **monotonically increasing** rate — higher
# MAGIC confidence really is more often right. If not, confidence isn't a reliable
# MAGIC review signal for this field set.

# COMMAND ----------

cal = (
    long.where("conf IS NOT NULL")
    .withColumn(
        "conf_bin",
        F.when(F.col("conf") >= 0.9, "0.9-1.0")
        .when(F.col("conf") >= 0.8, "0.8-0.9")
        .when(F.col("conf") >= 0.7, "0.7-0.8")
        .otherwise("<0.7"),
    )
    .groupBy("conf_bin")
    .agg(F.count("*").alias("n"), F.avg(F.col("correct").cast("double")).alias("correct_rate"))
    .orderBy("conf_bin")
)
display(cal)

# monotonicity verdict (low → high bin)
cal_pd = cal.toPandas().set_index("conf_bin")
bin_order = ["<0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
rates = [cal_pd.loc[b, "correct_rate"] for b in bin_order if b in cal_pd.index]
monotonic = all(x <= y + 1e-9 for x, y in zip(rates, rates[1:]))
print(f"correctness rates low→high: {[round(r, 3) for r in rates]}")
print(f"calibration monotonic (confidence tracks correctness): {monotonic}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confidently wrong (high confidence, still incorrect)
# MAGIC
# MAGIC The dangerous quadrant — the model was sure and still wrong. These are the
# MAGIC cases a confidence threshold alone would let through.

# COMMAND ----------

display(
    long.where((F.col("conf") >= F.lit(high_conf_threshold)) & (~F.col("correct")))
    .select("doc_id", "field", "conf", "cls")
    .orderBy(F.col("conf").desc())
)
