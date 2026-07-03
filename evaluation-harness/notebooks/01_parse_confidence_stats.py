# Databricks notebook source
# MAGIC %md
# MAGIC # Recipe 01 — `ai_parse_document` Confidence Statistics
# MAGIC
# MAGIC A **measure-only** recipe. Point it at any table that already holds an
# MAGIC `ai_parse_document` VARIANT column and it profiles the **element-level
# MAGIC confidence** distribution — corpus, per-document, and per-page — then draws a
# MAGIC per-document violin/box plot so you can see which documents parsed poorly.
# MAGIC
# MAGIC It does **not** call `ai_parse_document` itself. Run one of this repo's parse
# MAGIC stages first — e.g. `ai-extract-word-level-citation` writes
# MAGIC `*_bronze_parsed_docs`, and `document-page-classify-extraction` writes the
# MAGIC same.
# MAGIC
# MAGIC **Input contract:** a table with a VARIANT column (default `parsed`) whose
# MAGIC shape matches `ai_parse_document` output —
# MAGIC `parsed:document:elements` is an array of elements, each carrying `type`,
# MAGIC `content`, `confidence`, and `bbox[0]:page_id`.
# MAGIC
# MAGIC **Philosophy:** confidence is **descriptive, not a filter** — this recipe
# MAGIC surfaces where the parser was unsure so you can route those pages for review,
# MAGIC not silently drop them.
# MAGIC
# MAGIC No extra installs — uses `matplotlib` / `numpy` / `pandas` (pre-installed on DBR).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC | Widget | Meaning |
# MAGIC |---|---|
# MAGIC | `catalog` / `schema` / `parsed_table` | fully-qualified source table |
# MAGIC | `parsed_col` | name of the `ai_parse_document` VARIANT column |
# MAGIC | `doc_id_col` | column identifying each source document (e.g. `path`) |
# MAGIC | `abs_floor` | absolute confidence floor — below this an element is flagged `low` |
# MAGIC | `corpus_percentile` | corpus-wide low percentile used as a soft flag |
# MAGIC | `doc_z_threshold` | per-document z-score below which an element looks anomalous |

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("parsed_table", "paystub_bbox_stream_bronze_parsed_docs")
dbutils.widgets.text("parsed_col", "parsed")
dbutils.widgets.text("doc_id_col", "path")
dbutils.widgets.text("abs_floor", "0.5")
dbutils.widgets.text("corpus_percentile", "0.10")
dbutils.widgets.text("doc_z_threshold", "-1.5")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
parsed_table = dbutils.widgets.get("parsed_table")
parsed_col = dbutils.widgets.get("parsed_col")
doc_id_col = dbutils.widgets.get("doc_id_col")
ABS_FLOOR = float(dbutils.widgets.get("abs_floor"))
CORPUS_PERCENTILE = float(dbutils.widgets.get("corpus_percentile"))
DOC_Z_THRESHOLD = float(dbutils.widgets.get("doc_z_threshold"))

parsed_fqn = f"{catalog}.{schema}.{parsed_table}"
print(f"parsed_fqn = {parsed_fqn}")
print(f"parsed_col = {parsed_col}  doc_id_col = {doc_id_col}")
print(f"ABS_FLOOR={ABS_FLOOR}  CORPUS_PERCENTILE={CORPUS_PERCENTILE}  DOC_Z_THRESHOLD={DOC_Z_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Flatten elements → one row per parsed element
# MAGIC
# MAGIC `posexplode_outer` keeps documents that parsed to zero elements visible. Every
# MAGIC field is `try_cast` so a malformed element degrades to `NULL` rather than
# MAGIC failing the whole read.

# COMMAND ----------

# DBTITLE 1,Cell 5
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import expr

elements = (
    spark.table(parsed_fqn)
    .withColumn("elements", expr(f"try_cast({parsed_col}:document:elements AS ARRAY<VARIANT>)"))
    .selectExpr(
        f"{doc_id_col} AS doc_id",
        "posexplode_outer(elements) AS (element_idx, element)",
    )
    .selectExpr(
        "doc_id",
        "element_idx",
        "try_cast(element:bbox[0]:page_id AS INT) AS page_id",
        "try_cast(element:type AS STRING) AS type",
        "try_cast(element:content AS STRING) AS content",
        "try_cast(element:confidence AS DOUBLE) AS confidence",
    )
    .where("confidence IS NOT NULL")
    .withColumn("doc_label", F.element_at(F.split("doc_id", "/"), -1))
)
print(f"elements with a confidence score: {elements.count():,}")
display(elements.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Corpus-level distribution
# MAGIC
# MAGIC The headline numbers: mean/stddev, the percentile spread, and the
# MAGIC `corpus_percentile` value used as a soft flag downstream.

# COMMAND ----------

corpus = elements.agg(
    F.count("*").alias("n_elements"),
    F.avg("confidence").alias("mean"),
    F.stddev("confidence").alias("stddev"),
    F.min("confidence").alias("min"),
    F.expr("percentile_approx(confidence, 0.10)").alias("p10"),
    F.expr("percentile_approx(confidence, 0.25)").alias("p25"),
    F.expr("percentile_approx(confidence, 0.50)").alias("p50"),
    F.expr("percentile_approx(confidence, 0.75)").alias("p75"),
    F.expr("percentile_approx(confidence, 0.90)").alias("p90"),
    F.expr(f"percentile_approx(confidence, {CORPUS_PERCENTILE})").alias("corpus_pctl"),
)
display(corpus)

corpus_row = corpus.first()
corpus_mean = corpus_row["mean"]
corpus_std = corpus_row["stddev"] or 0.0
corpus_pctl = corpus_row["corpus_pctl"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Per-element tiering (corpus + per-document z-scores)
# MAGIC
# MAGIC Three independent flags combine into a layered `conf_tier`:
# MAGIC
# MAGIC - `flag_abs_floor` — below the absolute floor.
# MAGIC - `flag_corpus_pctl` — in the corpus's low tail.
# MAGIC - `flag_doc_z` — unusually low **relative to its own document** (catches a bad
# MAGIC   page inside an otherwise clean doc).
# MAGIC
# MAGIC `low` = below the hard floor; `borderline` = tripped a soft flag; `ok` otherwise.

# COMMAND ----------

w = Window.partitionBy("doc_id")
tiered = (
    elements
    .withColumn("doc_mean", F.avg("confidence").over(w))
    .withColumn("doc_std", F.stddev("confidence").over(w))
    .withColumn("corpus_z", (F.col("confidence") - F.lit(corpus_mean)) / F.lit(corpus_std if corpus_std else 1.0))
    .withColumn(
        "doc_z",
        F.when(F.col("doc_std") > 0, (F.col("confidence") - F.col("doc_mean")) / F.col("doc_std")).otherwise(F.lit(0.0)),
    )
    .withColumn("flag_abs_floor", F.col("confidence") < F.lit(ABS_FLOOR))
    .withColumn("flag_corpus_pctl", F.col("confidence") < F.lit(corpus_pctl))
    .withColumn("flag_doc_z", F.col("doc_z") < F.lit(DOC_Z_THRESHOLD))
    .withColumn(
        "conf_tier",
        F.when(F.col("flag_abs_floor"), "low")
        .when(F.col("flag_corpus_pctl") | F.col("flag_doc_z"), "borderline")
        .otherwise("ok"),
    )
)

print("Element counts by tier:")
display(tiered.groupBy("conf_tier").count().orderBy("conf_tier"))
print("Lowest-confidence elements (review these first):")
display(
    tiered.select("doc_label", "page_id", "type", "confidence", "corpus_z", "doc_z", "conf_tier", "content")
    .orderBy("confidence")
    .limit(25)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-page stats
# MAGIC
# MAGIC A single bad page is easier to act on than a doc-level average. Sorted by
# MAGIC weakest mean page confidence.

# COMMAND ----------

per_page = (
    elements.groupBy("doc_label", "page_id")
    .agg(
        F.count("*").alias("n_elements"),
        F.avg("confidence").alias("mean"),
        F.stddev("confidence").alias("stddev"),
        F.min("confidence").alias("min"),
        F.max("confidence").alias("max"),
    )
    .orderBy("mean")
)
display(per_page)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualize — per-document confidence distribution
# MAGIC
# MAGIC Violin (shape) + boxplot (median/IQR) + jittered scatter (every element),
# MAGIC documents sorted worst-median on top. The red line is the absolute floor, the
# MAGIC orange line the corpus low percentile.

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

pdf = elements.select("doc_label", "confidence").toPandas()
groups = pdf.groupby("doc_label")["confidence"].apply(list)
order = sorted(groups.index, key=lambda d: np.median(groups[d]))  # worst median first
data = [groups[d] for d in order]
ypos = np.arange(1, len(order) + 1)

fig, ax = plt.subplots(figsize=(9, max(3.0, 0.4 * len(order))))
ax.violinplot(data, positions=ypos, vert=False, showextrema=False)
ax.boxplot(data, positions=ypos, vert=False, widths=0.3, showfliers=False)
rng = np.random.default_rng(0)
for i, d in enumerate(order):
    y = rng.normal(ypos[i], 0.04, size=len(groups[d]))
    ax.scatter(groups[d], y, s=6, alpha=0.3)
ax.axvline(ABS_FLOOR, color="red", ls="--", lw=1, label=f"abs_floor = {ABS_FLOOR}")
ax.axvline(corpus_pctl, color="orange", ls=":", lw=1.2, label=f"corpus p{int(CORPUS_PERCENTILE*100)} = {corpus_pctl:.2f}")
ax.set_yticks(ypos)
ax.set_yticklabels(order, fontsize=7)
ax.set_xlabel("element parse confidence")
ax.set_title("Per-document ai_parse_document element confidence (worst median on top)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - Feed the `low` / `borderline` elements into a review queue rather than
# MAGIC   dropping them.
# MAGIC - Recipe 02 does the same profiling for `ai_extract` field confidence.
# MAGIC - Recipe 03 checks whether confidence actually tracks correctness (calibration)
# MAGIC   against ground truth.