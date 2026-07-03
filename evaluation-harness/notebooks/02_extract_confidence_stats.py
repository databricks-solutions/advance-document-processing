# Databricks notebook source
# MAGIC %md
# MAGIC # Recipe 02 — `ai_extract` Confidence Statistics
# MAGIC
# MAGIC A **measure-only** recipe. Point it at any table that already holds a raw
# MAGIC `ai_extract` **v2.1** VARIANT column (run with `enableConfidenceScores`) and it
# MAGIC profiles the **per-field confidence** distribution, then draws a per-field
# MAGIC box/scatter plot so you can see which fields the model is least sure about.
# MAGIC
# MAGIC It does **not** call `ai_extract` itself. Run one of this repo's extract stages
# MAGIC first — e.g. `ai-extract-word-level-citation` writes `*_extracted` (the raw
# MAGIC `ai_extract` VARIANT, in a column named `paystub_extracted`).
# MAGIC
# MAGIC **Input contract:** a table with a VARIANT column (default `extracted`) whose
# MAGIC shape matches `ai_extract` v2.1 — each field lives at
# MAGIC `extracted:response:<field>:{value, confidence_score, citation_ids}`.
# MAGIC
# MAGIC **Philosophy:** confidence is **descriptive, not a filter** — low-confidence
# MAGIC fields go to a field-level review queue, they are not auto-dropped.
# MAGIC
# MAGIC No extra installs — uses `matplotlib` / `numpy` / `pandas`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC | Widget | Meaning |
# MAGIC |---|---|
# MAGIC | `catalog` / `schema` / `extracted_table` | fully-qualified source table |
# MAGIC | `extracted_col` | name of the raw `ai_extract` VARIANT column |
# MAGIC | `doc_id_col` | column identifying each source document |
# MAGIC | `fields` | comma-separated field list; **blank = auto-discover** the keys under `:response` |
# MAGIC | `review_threshold` | confidence below which a field is counted as "needs review" |

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("extracted_table", "paystub_bbox_stream_extracted")
dbutils.widgets.text("extracted_col", "extracted")
dbutils.widgets.text("doc_id_col", "path")
dbutils.widgets.text("fields", "")
dbutils.widgets.text("review_threshold", "0.7")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
extracted_table = dbutils.widgets.get("extracted_table")
extracted_col = dbutils.widgets.get("extracted_col")
doc_id_col = dbutils.widgets.get("doc_id_col")
REVIEW_THRESHOLD = float(dbutils.widgets.get("review_threshold"))

extracted_fqn = f"{catalog}.{schema}.{extracted_table}"
print(f"extracted_fqn = {extracted_fqn}")
print(f"extracted_col = {extracted_col}  doc_id_col = {doc_id_col}  REVIEW_THRESHOLD = {REVIEW_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve the field list
# MAGIC
# MAGIC If `fields` is blank we discover the extracted fields by reading the keys of
# MAGIC the `response` object (`ai_extract` v2.1 wraps each field there). Casting the
# MAGIC VARIANT object to `MAP<STRING, VARIANT>` lets us `map_keys` it.

# COMMAND ----------

fields_raw = dbutils.widgets.get("fields").strip()
if fields_raw:
    fields = [f.strip() for f in fields_raw.split(",") if f.strip()]
else:
    from pyspark.sql import functions as F

    keys = (
        spark.table(extracted_fqn)
        .selectExpr(f"explode(map_keys(try_cast({extracted_col}:response AS MAP<STRING, VARIANT>))) AS field")
        .distinct()
        .orderBy("field")
    )
    fields = [r["field"] for r in keys.collect()]

assert fields, "No fields found — set the `fields` widget or check `extracted_col`."
print(f"{len(fields)} fields: {fields}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unpack per-field value + confidence
# MAGIC
# MAGIC One row per (document, field) with the extracted `value` and its
# MAGIC `confidence_score`.

# COMMAND ----------

from pyspark.sql import functions as F

select_exprs = [f"{doc_id_col} AS doc_id"]
for f in fields:
    select_exprs.append(f"try_cast({extracted_col}:response:{f}:value AS STRING) AS `{f}`")
    select_exprs.append(f"try_cast({extracted_col}:response:{f}:confidence_score AS DOUBLE) AS `{f}__conf`")

flat = spark.table(extracted_fqn).selectExpr(*select_exprs).withColumn(
    "doc_label", F.element_at(F.split("doc_id", "/"), -1)
)

# long form: one row per (doc, field, confidence)
conf_structs = [
    F.struct(
        F.lit(f).alias("field"),
        F.col(f"`{f}`").alias("value"),
        F.col(f"`{f}__conf`").alias("confidence"),
    )
    for f in fields
]
long = flat.select("doc_id", "doc_label", F.explode(F.array(*conf_structs)).alias("fc")).select(
    "doc_id", "doc_label", "fc.field", "fc.value", "fc.confidence"
)
display(long.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Per-field distribution
# MAGIC
# MAGIC Mean/stddev, the percentile spread, and the `low_conf_rate` (share of documents
# MAGIC where this field landed below `review_threshold`). Sorted least-confident first.

# COMMAND ----------

per_field = (
    long.where("confidence IS NOT NULL")
    .groupBy("field")
    .agg(
        F.count("*").alias("n_docs"),
        F.avg("confidence").alias("mean"),
        F.stddev("confidence").alias("stddev"),
        F.min("confidence").alias("min"),
        F.expr("percentile_approx(confidence, 0.25)").alias("p25"),
        F.expr("percentile_approx(confidence, 0.50)").alias("median"),
        F.avg((F.col("confidence") < F.lit(REVIEW_THRESHOLD)).cast("double")).alias("low_conf_rate"),
    )
    .orderBy("mean")
)
display(per_field)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-document roll-up
# MAGIC
# MAGIC For each document: the weakest field (`min_extract_conf`), how many fields need
# MAGIC review, and which ones. This is the shape a field-level review queue consumes.

# COMMAND ----------

per_doc = (
    long.groupBy("doc_id", "doc_label")
    .agg(
        F.min("confidence").alias("min_extract_conf"),
        F.avg("confidence").alias("mean_extract_conf"),
        F.sum((F.col("confidence") < F.lit(REVIEW_THRESHOLD)).cast("int")).alias("low_conf_field_count"),
        F.collect_list(F.when(F.col("confidence") < F.lit(REVIEW_THRESHOLD), F.col("field"))).alias("low_conf_fields"),
    )
    .orderBy("min_extract_conf")
)
display(per_doc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualize — per-field confidence distribution
# MAGIC
# MAGIC Boxplot (median/IQR) + jittered scatter (one point per document) per field,
# MAGIC fields sorted least-confident on the left. The red line is `review_threshold`.

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

pdf = long.where("confidence IS NOT NULL").select("field", "confidence").toPandas()
gb = pdf.groupby("field")["confidence"]
order = sorted(gb.groups.keys(), key=lambda f: gb.get_group(f).mean())  # least confident first
data = [gb.get_group(f).values for f in order]
xpos = np.arange(1, len(order) + 1)

fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(order)), 5))
ax.boxplot(data, positions=xpos, widths=0.5, showfliers=False)
rng = np.random.default_rng(0)
for i, f in enumerate(order):
    vals = gb.get_group(f).values
    x = rng.normal(xpos[i], 0.05, size=len(vals))
    ax.scatter(x, vals, s=12, alpha=0.4)
ax.axhline(REVIEW_THRESHOLD, color="red", ls="--", lw=1, label=f"review_threshold = {REVIEW_THRESHOLD}")
ax.set_xticks(xpos)
ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("ai_extract confidence_score")
ax.set_title("Per-field ai_extract confidence distribution (least confident on left)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - Route documents with a high `low_conf_field_count` (Section 4) to review.
# MAGIC - Confidence alone tells you what the model *feels* — Recipe 03 checks whether
# MAGIC   it is actually *right*, using ground truth + `mlflow.genai.evaluate`, and
# MAGIC   whether confidence is calibrated to correctness.