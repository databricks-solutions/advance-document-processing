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
    # Create once with the schema, a NOT-NULL primary key (Vector Search needs one),
    # and Change Data Feed. Then INSERT OVERWRITE to refresh the data.
    # CREATE-IF-NOT-EXISTS + INSERT OVERWRITE (rather than CREATE OR REPLACE) keeps
    # the table identity and CDF continuity stable across re-runs, so the Delta Sync
    # index keeps syncing instead of failing with DELTA_CHANGE_DATA_FEED_INCOMPATIBLE
    # _DATA_SCHEMA on a replaced table.
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {gold} (
        chunk_id          STRING NOT NULL,
        chunk_position    INT,
        chunk_to_retrieve STRING,
        chunk_to_embed    STRING,
        doc_type          STRING,
        source_path       STRING,
        prepped_at        TIMESTAMP,
        CONSTRAINT {table_prefix}_{domain}_pk PRIMARY KEY (chunk_id)
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    spark.sql(f"""
    INSERT OVERWRITE {gold}
    SELECT chunk_id, chunk_position, chunk_to_retrieve, chunk_to_embed,
           doc_type, source_path, prepped_at
    FROM {prepped_table}
    WHERE domain = '{domain}'
    """)
    print(f"  rows = {spark.table(gold).count()}")

dbutils.notebook.exit("gold_route_done")
