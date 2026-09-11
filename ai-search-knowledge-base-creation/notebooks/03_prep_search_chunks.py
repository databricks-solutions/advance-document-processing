# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Prep Search Chunks: `ai_prep_search`
# MAGIC
# MAGIC **Stage**: Prep — semantic chunking and RAG enrichment.
# MAGIC
# MAGIC Each classified document is converted into retrieval-ready chunks using
# MAGIC [`ai_prep_search`](https://docs.databricks.com/en/sql/language-manual/functions/ai_prep_search.html).
# MAGIC The function returns a VARIANT with a `chunks` array. Each chunk exposes two key fields:
# MAGIC
# MAGIC | Field | Purpose |
# MAGIC |---|---|
# MAGIC | `chunk_to_embed` | Text sent to the embedding model at index time |
# MAGIC | `chunk_to_retrieve` | Text returned to the caller at query time |
# MAGIC
# MAGIC Only documents with a recognized `doc_type` and a non-`unknown` domain pass through.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Input** | `<table_prefix>_silver_classified` |
# MAGIC | **Output** | `<table_prefix>_prepped_chunks` |
# MAGIC | **Key columns** | `chunk_id`, `chunk_position`, `chunk_to_retrieve`, `chunk_to_embed`, `doc_type`, `domain`, `source_path` |
# MAGIC
# MAGIC Requires Serverless Environment v3+ or DBR 18.2+.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

silver_table  = f"{catalog}.{schema}.{table_prefix}_silver_classified"
prepped_table = f"{catalog}.{schema}.{table_prefix}_prepped_chunks"
print(f"silver_table  = {silver_table}")
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
    -- ai_prep_search nests chunks at $.document.contents (not $.chunks)
    explode(variant_get(prep, '$.document.contents', 'ARRAY<VARIANT>')) AS chunk
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

# COMMAND ----------

spark.table(prepped_table).display()
