# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Route Gold Tables: domain split with CDF enabled
# MAGIC
# MAGIC **Stage**: Gold — route prepped chunks into two domain-specific Delta tables.
# MAGIC
# MAGIC Splits the prepped-chunks table into a `reference` gold table and a `claims`
# MAGIC gold table. Both tables are created with Change Data Feed enabled
# MAGIC (`delta.enableChangeDataFeed = true`) so that Vector Search Delta Sync can
# MAGIC track incremental updates without a full re-embed on each refresh.
# MAGIC
# MAGIC A primary key constraint on `chunk_id` is added after table creation — Vector
# MAGIC Search Delta Sync requires a non-null primary key.
# MAGIC
# MAGIC | Domain | Gold table | Vector Search index (next stage) |
# MAGIC |---|---|---|
# MAGIC | `reference` | `<prefix>_reference_chunks` | `<prefix>_reference_index` |
# MAGIC | `claims` | `<prefix>_claims_chunks` | `<prefix>_claims_index` |
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Input** | `<table_prefix>_prepped_chunks` |
# MAGIC | **Outputs** | `<table_prefix>_reference_chunks`, `<table_prefix>_claims_chunks` |
# MAGIC
# MAGIC **Note**: `gold_table_name` logic is inlined from `bundle/src/insurance_kb_sql_builders.py`.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

prepped_table = f"{catalog}.{schema}.{table_prefix}_prepped_chunks"
print(f"prepped_table = {prepped_table}")

# COMMAND ----------

# gold_table_name inlined from insurance_kb_sql_builders.gold_table_name(prefix, domain)
# → f"{prefix}_{domain}_chunks"

for domain in ("reference", "claims"):
    gold = f"{catalog}.{schema}.{table_prefix}_{domain}_chunks"
    print(f"building {gold} ...")
    spark.sql(f"""
    CREATE OR REPLACE TABLE {gold}
    TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
    SELECT chunk_id, chunk_position, chunk_to_retrieve, chunk_to_embed,
           doc_type, source_path, prepped_at
    FROM {prepped_table}
    WHERE domain = '{domain}'
    """)
    # Vector Search Delta Sync wants a non-null primary key.
    spark.sql(f"ALTER TABLE {gold} ALTER COLUMN chunk_id SET NOT NULL")
    spark.sql(f"ALTER TABLE {gold} ADD CONSTRAINT {table_prefix}_{domain}_pk PRIMARY KEY (chunk_id)")
    print(f"  rows = {spark.table(gold).count()}")

# COMMAND ----------

# Display row counts per gold table
from pyspark.sql import Row

summary_rows = []
for domain in ("reference", "claims"):
    gold = f"{catalog}.{schema}.{table_prefix}_{domain}_chunks"
    n = spark.table(gold).count()
    summary_rows.append(Row(domain=domain, gold_table=gold, row_count=n))

spark.createDataFrame(summary_rows).display()
