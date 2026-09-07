# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Prep: `ai_prep_search` semantic chunking + enrichment
# MAGIC Turns each parsed document into RAG-ready chunks. Embed `chunk_to_embed`,
# MAGIC return `chunk_to_retrieve`. Requires serverless env v3+ / DBR 18.2+.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

silver_table  = f"{catalog}.{schema}.{table_prefix}_silver_classified"
prepped_table = f"{catalog}.{schema}.{table_prefix}_prepped_chunks"
print(f"prepped_table = {prepped_table}")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {prepped_table} AS
WITH prepped AS (
  SELECT source_path, doc_type, domain, ai_prep_search(parsed) AS prep
  FROM {silver_table}
  WHERE doc_type IS NOT NULL AND domain <> 'unknown'
),
chunks AS (
  SELECT
    source_path, doc_type, domain,
    explode(variant_get(prep, '$.chunks', 'ARRAY<VARIANT>')) AS chunk
  FROM prepped
)
SELECT
  variant_get(chunk, '$.chunk_id',          'STRING') AS chunk_id,
  variant_get(chunk, '$.chunk_position',    'INT')    AS chunk_position,
  variant_get(chunk, '$.chunk_to_retrieve', 'STRING') AS chunk_to_retrieve,
  variant_get(chunk, '$.chunk_to_embed',    'STRING') AS chunk_to_embed,
  doc_type, domain, source_path,
  current_timestamp() AS prepped_at
FROM chunks
""")

cnt = spark.table(prepped_table).count()
print(f"prepped chunk rows = {cnt}")
dbutils.notebook.exit(f"chunks={cnt}")
