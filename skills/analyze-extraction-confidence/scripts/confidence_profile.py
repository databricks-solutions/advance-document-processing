"""Profile ai_parse_document element confidence and ai_extract v2.1 leaf confidence.

Copy this whole file unchanged into one notebook cell (Genie Code or Databricks
notebook), then call the Spark functions below -- see SKILL.md. Pure functions
(no Spark imports at module level) are unit-tested in
skills/tests/test_analyze_extraction_confidence.py.
"""

import json
import math
import re
from collections import Counter

LEAF_SCORED = "scored"      # value present, confidence_score present
LEAF_UNSCORED = "unscored"  # value present, confidence_score null
LEAF_ABSENT = "absent"      # value null (field not found in the document)


# ---------------------------------------------------------------- pure helpers


def flatten_extract_leaves(response):
    """Leaves of an ai_extract v2.1 `response` object.

    Confidence exists only on leaves ({value, confidence_score}); objects and
    arrays carry none. Returns dicts with:
      path   instance path, e.g. 'deduction_item[1].period_amount'
      field  generic path for per-field stats, e.g. 'deduction_item[].period_amount'
      value, confidence, status (LEAF_SCORED | LEAF_UNSCORED | LEAF_ABSENT)
    """
    out = []

    def is_leaf(node):
        return isinstance(node, dict) and "value" in node and "confidence_score" in node

    def holds_leaves(node):
        if is_leaf(node):
            return True
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else []
        return any(holds_leaves(c) for c in children)

    def walk(node, path):
        if is_leaf(node):
            value, conf = node["value"], node["confidence_score"]
            if isinstance(value, (dict, list)) and holds_leaves(value):
                walk(value, path)
                return
            if isinstance(value, (dict, list)):
                value = json.dumps(value)  # one score for a plain list/object value: keep it as one leaf
            status = LEAF_ABSENT if value is None else (LEAF_UNSCORED if conf is None else LEAF_SCORED)
            out.append({
                "path": path,
                "field": re.sub(r"\[\d+\]", "[]", path),
                "value": None if value is None else str(value),
                "confidence": None if conf is None else float(conf),
                "status": status,
            })
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{path}[{i}]")

    walk(response or {}, "")
    return out


def score_levels(confidences):
    """Count of each distinct confidence value (ai_extract scores are a few discrete levels)."""
    return dict(Counter(c for c in confidences if c is not None))


def level_cutoff_from_counts(level_counts, pct=0.05):
    """Nearest-rank `pct` percentile of discrete levels given {level: count}.
    The result is always an observed level; flag strictly below it, so ties are ok."""
    total = sum(level_counts.values())
    if total == 0:
        return None
    rank = max(1, math.ceil(round(pct * total, 9)))  # round() guards float noise like 10.000000000000002
    seen = 0
    for level in sorted(level_counts):
        seen += level_counts[level]
        if seen >= rank:
            return level
    return max(level_counts)


def level_cutoff(confidences, pct=0.05):
    return level_cutoff_from_counts(score_levels(confidences), pct)


def classify_extract_leaf(status, confidence, cutoff):
    """'absent' | 'ok' | 'review:<reason>' for one leaf, relative to the corpus cutoff level."""
    if status == LEAF_ABSENT:
        return "absent"
    if status == LEAF_UNSCORED:
        return "review:unscored"
    if cutoff is None:
        return "review:no_baseline"
    return "review:below_corpus_pct" if confidence < cutoff else "ok"


# ---------------------------------------------------------------- spark: parse


def parse_elements(df, parsed_col="parsed", doc_id_col="path"):
    """One row per ai_parse_document element: doc_id, element_idx, page_id, type, confidence, content."""
    return df.selectExpr(
        f"{doc_id_col} AS doc_id",
        f"posexplode_outer(try_cast({parsed_col}:document:elements AS ARRAY<VARIANT>)) AS (element_idx, el)",
    ).selectExpr(
        "doc_id",
        "element_idx",
        "try_cast(el:bbox[0]:page_id AS INT) AS page_id",
        "try_cast(el:type AS STRING) AS type",
        "try_cast(el:confidence AS DOUBLE) AS confidence",
        "try_cast(el:content AS STRING) AS content",
    )


def parse_type_profile(elements, pct=0.05):
    """Per element type: count, saturation share, percentiles, and the relative cutoff
    (the type's `pct` percentile). Look at this before choosing any threshold."""
    from pyspark.sql import functions as F

    return (
        elements.where("confidence IS NOT NULL")
        .groupBy("type")
        .agg(
            F.count("*").alias("n"),
            F.countDistinct("confidence").alias("n_distinct"),
            F.avg((F.col("confidence") >= 0.999).cast("double")).alias("share_ge_0_999"),
            F.min("confidence").alias("min"),
            F.expr("percentile_approx(confidence, 0.5)").alias("p50"),
            F.expr(f"percentile_approx(confidence, {pct})").alias("cutoff"),
        )
        .orderBy("cutoff")
    )


def flag_parse_elements(elements, pct=0.05, abs_floor=None):
    """Adds `flag`: 'review:below_type_pct' when confidence is below its element type's
    `pct` percentile (and below 1.0), 'review:below_floor' under an optional absolute
    floor, 'review:unscored' when confidence is null, else 'ok'."""
    from pyspark.sql import functions as F

    cutoffs = elements.where("confidence IS NOT NULL").groupBy("type").agg(
        F.expr(f"percentile_approx(confidence, {pct})").alias("type_cutoff")
    )
    flag = F.when(F.col("confidence").isNull(), "review:unscored")
    if abs_floor is not None:
        flag = flag.when(F.col("confidence") < F.lit(abs_floor), "review:below_floor")
    flag = flag.when(
        (F.col("confidence") < F.col("type_cutoff")) & (F.col("confidence") < 1.0), "review:below_type_pct"
    ).otherwise("ok")
    return elements.join(cutoffs, "type", "left").withColumn("flag", flag)


def parse_page_rollup(flagged):
    """Per (doc_id, page_id): element count, min confidence, flagged count. Worst pages first."""
    from pyspark.sql import functions as F

    return (
        flagged.groupBy("doc_id", "page_id")
        .agg(
            F.count("*").alias("n_elements"),
            F.min("confidence").alias("min_confidence"),
            F.sum((F.col("flag") != "ok").cast("int")).alias("n_flagged"),
            F.collect_set(F.when(F.col("flag") != "ok", F.col("type"))).alias("flagged_types"),
        )
        .orderBy(F.desc("n_flagged"), "min_confidence")
    )


# ---------------------------------------------------------------- spark: extract


def extract_leaves(df, extracted_col="extracted", doc_id_col="path"):
    """One row per ai_extract leaf: doc_id, path, field, value, confidence, status."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

    leaf_type = ArrayType(StructType([
        StructField("path", StringType()),
        StructField("field", StringType()),
        StructField("value", StringType()),
        StructField("confidence", DoubleType()),
        StructField("status", StringType()),
    ]))

    @F.udf(leaf_type)
    def _leaves(js):
        return flatten_extract_leaves(json.loads(js)) if js else []

    return (
        df.select(F.col(doc_id_col).alias("doc_id"), F.expr(f"to_json({extracted_col}:response)").alias("js"))
        .select("doc_id", F.explode_outer(_leaves("js")).alias("leaf"))
        .select("doc_id", "leaf.*")
    )


def extract_level_profile(leaves):
    """Distinct confidence levels across all scored leaves, with counts. Look at this first."""
    return leaves.where(f"status = '{LEAF_SCORED}'").groupBy("confidence").count().orderBy("confidence")


def flag_extract_leaves(leaves, pct=0.05):
    """Adds `cutoff` (corpus `pct` percentile level of scored leaves) and `flag` per leaf.
    Returns (flagged_df, cutoff)."""
    from pyspark.sql import functions as F

    counts = {r["confidence"]: r["count"] for r in extract_level_profile(leaves).collect()}
    cutoff = level_cutoff_from_counts(counts, pct)

    @F.udf("string")
    def _flag(status, confidence):
        return classify_extract_leaf(status, confidence, cutoff)

    flagged = leaves.withColumn("cutoff", F.lit(cutoff).cast("double")).withColumn(
        "flag", _flag("status", "confidence")
    )
    return flagged, cutoff


def extract_doc_summary(flagged):
    """Per document: leaf counts by status, review count, and its most common level.
    A document whose usual level sits below the corpus's is a document-level shift."""
    from pyspark.sql import functions as F

    return (
        flagged.groupBy("doc_id")
        .agg(
            F.count("*").alias("n"),
            F.sum((F.col("status") == LEAF_ABSENT).cast("int")).alias("n_absent"),
            F.sum((F.col("status") == LEAF_UNSCORED).cast("int")).alias("n_unscored"),
            F.sum(F.col("flag").startswith("review").cast("int")).alias("n_review"),
            F.expr("mode(confidence)").alias("doc_usual_level"),
            F.min("confidence").alias("min_confidence"),
        )
        .orderBy(F.desc("n_review"), "doc_usual_level")
    )


def extract_field_summary(flagged):
    """Per generic field: scored / unscored / absent counts, review count, min confidence."""
    from pyspark.sql import functions as F

    return (
        flagged.groupBy("field")
        .agg(
            F.count("*").alias("n"),
            F.sum((F.col("status") == LEAF_SCORED).cast("int")).alias("n_scored"),
            F.sum((F.col("status") == LEAF_UNSCORED).cast("int")).alias("n_unscored"),
            F.sum((F.col("status") == LEAF_ABSENT).cast("int")).alias("n_absent"),
            F.sum(F.col("flag").startswith("review").cast("int")).alias("n_review"),
            F.min("confidence").alias("min_confidence"),
        )
        .orderBy(F.desc("n_review"), "field")
    )


def review_queue(flagged_leaves, flagged_elements=None):
    """Everything to route to a human, one row per item; nothing is dropped from the inputs."""
    from pyspark.sql import functions as F

    q = flagged_leaves.where(F.col("flag").startswith("review")).select(
        "doc_id",
        F.lit("extract_leaf").alias("item_type"),
        F.col("path").alias("item"),
        F.lit(None).cast("int").alias("page_id"),
        "value",
        "confidence",
        "flag",
    )
    if flagged_elements is not None:
        q = q.unionByName(
            flagged_elements.where(F.col("flag").startswith("review")).select(
                "doc_id",
                F.lit("parse_element").alias("item_type"),
                F.concat_ws(":", "type", F.col("element_idx").cast("string")).alias("item"),
                "page_id",
                F.substring("content", 1, 200).alias("value"),
                "confidence",
                "flag",
            )
        )
    return q.orderBy("doc_id", "item_type", "confidence")


def calibration_by_level(flagged_leaves, labels, label_col="is_correct"):
    """Only when labels exist. labels: (doc_id, path, <label_col> boolean).
    Correct rate per observed confidence level -- not per fixed-width bin."""
    from pyspark.sql import functions as F

    return (
        flagged_leaves.where(f"status = '{LEAF_SCORED}'")
        .join(labels.select("doc_id", "path", label_col), ["doc_id", "path"])
        .groupBy("confidence")
        .agg(F.count("*").alias("n_labelled"), F.avg(F.col(label_col).cast("double")).alias("correct_rate"))
        .orderBy("confidence")
    )
