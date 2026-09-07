# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Classify Documents: `ai_classify`
# MAGIC
# MAGIC **Stage**: Silver — document-type classification and domain routing.
# MAGIC
# MAGIC Each parsed document is classified into one of five insurance document types using
# MAGIC [`ai_classify`](https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html),
# MAGIC then mapped to a retrieval domain (`reference` or `claims`).
# MAGIC
# MAGIC | Label | Domain |
# MAGIC |---|---|
# MAGIC | `policy_document` | `reference` |
# MAGIC | `endorsement` | `reference` |
# MAGIC | `underwriting_guideline` | `reference` |
# MAGIC | `fnol_claim_form` | `claims` |
# MAGIC | `adjuster_report` | `claims` |
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Input** | `<table_prefix>_bronze_parsed` (parse_error IS NULL rows only) |
# MAGIC | **Output** | `<table_prefix>_silver_classified` |
# MAGIC | **Key columns** | `doc_type`, `domain`, `classification_raw` (VARIANT) |
# MAGIC
# MAGIC **Note**: The label dictionary, instructions, and CASE logic below are inlined from
# MAGIC `bundle/src/insurance_kb_sql_builders.py`. That file is the canonical tested copy.

# COMMAND ----------

import json

# ---------------------------------------------------------------------------
# Inlined from bundle/src/insurance_kb_sql_builders.py
# (canonical, unit-tested version lives there; copied here so the notebook
#  is self-contained and runnable without sys.path manipulation)
# ---------------------------------------------------------------------------

DOC_TYPE_LABELS = {
    "policy_document": (
        "An insurance policy: the declarations page and/or policy wording for a "
        "homeowners or auto policy. Named insured, policy number, policy period, "
        "coverages and coverage limits, deductibles, premiums, and exclusions."
    ),
    "endorsement": (
        "An endorsement or rider that amends an existing base policy — e.g. a "
        "water-backup endorsement or scheduled-personal-property (jewelry) rider. "
        "References a base policy number and states the specific coverage added, "
        "changed, or removed; not a standalone full policy."
    ),
    "underwriting_guideline": (
        "Internal underwriting guidelines: rules for risk acceptance, referral, or "
        "pricing (e.g. maximum roof age, coastal/wildfire exposure, prior-claims "
        "thresholds). A reference rulebook, not tied to one named insured."
    ),
    "fnol_claim_form": (
        "A First Notice of Loss (FNOL) / claim report form. Structured fields: "
        "claim number, claimant, date of loss, policy number, peril/cause of loss, "
        "loss location, and a short loss description."
    ),
    "adjuster_report": (
        "A claims adjuster's inspection or investigation report: narrative findings, "
        "damage assessment, cause-of-loss analysis, photos/estimates references, and "
        "a recommended reserve or settlement."
    ),
}

DOC_TYPE_TO_DOMAIN = {
    "policy_document": "reference",
    "endorsement": "reference",
    "underwriting_guideline": "reference",
    "fnol_claim_form": "claims",
    "adjuster_report": "claims",
}

CLASSIFY_INSTRUCTIONS = (
    "These are insurance documents for an adjuster/underwriter knowledge base. "
    "Each document is a single type (not a mixed packet). Classify by the document's "
    "dominant purpose. Use 'policy_document' for a full policy/declarations, "
    "'endorsement' only when it amends an existing policy, 'underwriting_guideline' "
    "for internal risk/pricing rulebooks, 'fnol_claim_form' for a first-notice-of-loss "
    "report, and 'adjuster_report' for an adjuster's inspection/investigation findings."
)


def labels_json():
    """JSON object string (label -> description) for ai_classify."""
    return json.dumps(DOC_TYPE_LABELS)


def classify_expr(input_col, labels_sql_json, instructions):
    """Build the ai_classify(...) SQL. Doubles single quotes in string literals."""
    labels_sql = labels_sql_json.replace("'", "''")
    instr_sql = instructions.replace("'", "''")
    return f"""
        ai_classify(
            {input_col},
            '{labels_sql}',
            map('instructions', '{instr_sql}')
        )
    """


def domain_case_expr(doc_type_sql):
    """SQL CASE mapping a doc_type SQL expression to its retrieval domain."""
    ref = [k for k, v in DOC_TYPE_TO_DOMAIN.items() if v == "reference"]
    clm = [k for k, v in DOC_TYPE_TO_DOMAIN.items() if v == "claims"]
    ref_in = ", ".join(f"'{k}'" for k in ref)
    clm_in = ", ".join(f"'{k}'" for k in clm)
    return f"""
        CASE
            WHEN {doc_type_sql} IN ({ref_in}) THEN 'reference'
            WHEN {doc_type_sql} IN ({clm_in}) THEN 'claims'
            ELSE 'unknown'
        END
    """

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "insurance_docs")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

bronze_table = f"{catalog}.{schema}.{table_prefix}_bronze_parsed"
silver_table = f"{catalog}.{schema}.{table_prefix}_silver_classified"
print(f"bronze_table = {bronze_table}")
print(f"silver_table = {silver_table}")

# COMMAND ----------

classify_sql = classify_expr("parsed", labels_json(), CLASSIFY_INSTRUCTIONS)
domain_sql   = domain_case_expr("classification_raw:response[0]::string")

spark.sql(f"""
CREATE OR REPLACE TABLE {silver_table} AS
WITH classified AS (
  SELECT
    source_path,
    parsed,
    {classify_sql} AS classification_raw
  FROM {bronze_table}
  WHERE parse_error IS NULL
)
SELECT
  source_path,
  parsed,
  classification_raw,
  classification_raw:response[0]::string AS doc_type,
  {domain_sql} AS domain
FROM classified
""")

from pyspark.sql.functions import col
counts = spark.table(silver_table).groupBy("doc_type", "domain").count()
counts.show(truncate=False)
n = spark.table(silver_table).count()
print(f"classified = {n}")

# COMMAND ----------

spark.table(silver_table).selectExpr("source_path", "doc_type", "domain").display()
