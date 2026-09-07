# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: `ai_classify` document type → domain
# MAGIC Assigns each document one of five labels and maps it to a retrieval
# MAGIC domain (reference / claims). One classify call per document.

# COMMAND ----------

# insurance_kb_sql_builders.py is co-located in src/; Databricks puts the notebook's own
# directory on sys.path, so a plain import works (matches the other bundles).
import insurance_kb_sql_builders as S

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
print(f"silver_table = {silver_table}")

# COMMAND ----------

classify_sql = S.classify_expr("parsed", S.labels_json(), S.CLASSIFY_INSTRUCTIONS)
domain_sql   = S.domain_case_expr("classification_raw:response[0]::string")

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
dbutils.notebook.exit(f"classified={n}")
