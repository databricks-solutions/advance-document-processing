# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Parse Documents: `ai_parse_document`
# MAGIC
# MAGIC **Stage**: Bronze — raw ingestion from the source volume.
# MAGIC
# MAGIC This notebook batch-parses all PDF files from the configured volume sub-directory
# MAGIC into a Delta bronze table using
# MAGIC [`ai_parse_document`](https://docs.databricks.com/en/sql/language-manual/functions/ai_parse_document.html).
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Input** | `/Volumes/<catalog>/<schema>/<volume>/<volume_subdir>/*.pdf` |
# MAGIC | **Output** | `<catalog>.<schema>.<table_prefix>_bronze_parsed` |
# MAGIC | **Columns** | `source_path`, `parsed` (VARIANT), `parse_error`, `ingested_at` |
# MAGIC
# MAGIC Requires Serverless Environment v3+ (or DBR 15.4 LTS+).
# MAGIC After running, scroll down to inspect the parse status for every document.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "insurance_docs")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
volume        = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix  = dbutils.widgets.get("table_prefix")

source_path  = f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}"
bronze_table = f"{catalog}.{schema}.{table_prefix}_bronze_parsed"
print(f"source_path  = {source_path}")
print(f"bronze_table = {bronze_table}")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {bronze_table} AS
WITH parsed AS (
  SELECT
    path AS source_path,
    ai_parse_document(content, map('version', '2.0')) AS parsed
  FROM read_files('{source_path}/', format => 'binaryFile')
  WHERE lower(path) LIKE '%.pdf'
)
SELECT
  source_path,
  parsed,
  variant_get(parsed, '$.error_status', 'STRING') AS parse_error,
  current_timestamp() AS ingested_at
FROM parsed
""")

cnt = spark.table(bronze_table).count()
print(f"bronze rows = {cnt}")

# COMMAND ----------

spark.table(bronze_table).selectExpr("source_path", "parse_error").display()
