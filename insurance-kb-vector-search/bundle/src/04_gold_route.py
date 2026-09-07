# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Gold: route chunks into two domain tables
# MAGIC Splits prepped chunks by domain into reference / claims gold tables,
# MAGIC each Change-Data-Feed enabled for Vector Search Delta Sync.

# COMMAND ----------

# Co-located import (Databricks adds the notebook dir to sys.path).
import insurance_kb_sql_builders as S

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

prepped_table = f"{catalog}.{schema}.{table_prefix}_prepped_chunks"

# COMMAND ----------

for domain in ("reference", "claims"):
    gold = f"{catalog}.{schema}.{S.gold_table_name(table_prefix, domain)}"
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

dbutils.notebook.exit("gold_route_done")
