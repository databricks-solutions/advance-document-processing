# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Index: create/refresh Delta Sync Vector Search indexes
# MAGIC One index per gold table, managed embeddings (databricks-gte-large-en),
# MAGIC TRIGGERED sync. Embeds `chunk_to_embed`; return `chunk_to_retrieve` at query time.

# COMMAND ----------

import time
import insurance_kb_sql_builders as S  # co-located in src/ (Databricks adds notebook dir to sys.path)
from databricks.sdk import WorkspaceClient

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")
dbutils.widgets.text("vs_endpoint", "insurance_kb_vs")
dbutils.widgets.text("embedding_model", "databricks-gte-large-en")

catalog         = dbutils.widgets.get("catalog")
schema          = dbutils.widgets.get("schema")
table_prefix    = dbutils.widgets.get("table_prefix")
vs_endpoint     = dbutils.widgets.get("vs_endpoint")
embedding_model = dbutils.widgets.get("embedding_model")

w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md ## Ensure the endpoint exists (create if missing) and is online

# COMMAND ----------

existing = [e.name for e in (w.vector_search_endpoints.list_endpoints() or [])]
if vs_endpoint not in existing:
    print(f"creating endpoint {vs_endpoint} ...")
    w.vector_search_endpoints.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")

# Poll until ONLINE (endpoint creation is asynchronous).
# NOTE: exact attribute names (endpoint_status.state, list_indexes(...).vector_indexes, .name)
# should be confirmed against the installed databricks-sdk version on first run;
# adjust status-poll/list accessors if the SDK differs.
for _ in range(60):
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=vs_endpoint)
    state = ep.endpoint_status.state if ep.endpoint_status else None
    print(f"endpoint state = {state}")
    if str(state) == "ONLINE":
        break
    time.sleep(30)
else:
    raise RuntimeError(f"Vector Search endpoint {vs_endpoint} did not reach ONLINE in time")

# COMMAND ----------
# MAGIC %md ## Create or sync one index per gold table

# COMMAND ----------

for domain in ("reference", "claims"):
    src_table = f"{catalog}.{schema}.{S.gold_table_name(table_prefix, domain)}"
    idx = S.index_name(catalog, schema, table_prefix, domain)
    existing_idx = [i.name for i in (w.vector_search_indexes.list_indexes(endpoint_name=vs_endpoint).vector_indexes or [])]
    if idx in existing_idx:
        print(f"syncing existing index {idx} ...")
        w.vector_search_indexes.sync_index(index_name=idx)
    else:
        print(f"creating index {idx} on {src_table} ...")
        w.vector_search_indexes.create_index(
            name=idx,
            endpoint_name=vs_endpoint,
            primary_key="chunk_id",
            index_type="DELTA_SYNC",
            delta_sync_index_spec={
                "source_table": src_table,
                "embedding_source_columns": [
                    {"name": "chunk_to_embed",
                     "embedding_model_endpoint_name": embedding_model}
                ],
                "pipeline_type": "TRIGGERED",
            },
        )

print("index creation/sync submitted.")
dbutils.notebook.exit("indexes_done")
