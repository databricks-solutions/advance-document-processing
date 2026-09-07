# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Create Vector Search Indexes: Delta Sync
# MAGIC
# MAGIC **Stage**: Index — create or refresh the Vector Search indexes for RAG retrieval.
# MAGIC
# MAGIC One index is created per gold domain table using
# MAGIC [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC Delta Sync with managed embeddings. The pipeline:
# MAGIC
# MAGIC 1. Ensures the Vector Search endpoint exists (creates it if missing).
# MAGIC 2. Polls until the endpoint is `ONLINE`.
# MAGIC 3. For each domain, creates the index (or triggers a sync if it already exists).
# MAGIC
# MAGIC | Field | Role |
# MAGIC |---|---|
# MAGIC | `chunk_to_embed` | Column sent to the embedding model at index time |
# MAGIC | `chunk_to_retrieve` | Column returned to the caller at query time |
# MAGIC
# MAGIC | Domain | Gold source table | Index name |
# MAGIC |---|---|---|
# MAGIC | `reference` | `<prefix>_reference_chunks` | `<prefix>_reference_index` |
# MAGIC | `claims` | `<prefix>_claims_chunks` | `<prefix>_claims_index` |
# MAGIC
# MAGIC **Note**: `gold_table_name` and `index_name` logic are inlined from
# MAGIC `bundle/src/insurance_kb_sql_builders.py`.

# COMMAND ----------

import time
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

# COMMAND ----------
# MAGIC %md ## Create or sync one index per gold table

# COMMAND ----------

# gold_table_name inlined from insurance_kb_sql_builders.gold_table_name(prefix, domain)
# → f"{prefix}_{domain}_chunks"
# index_name inlined from insurance_kb_sql_builders.index_name(catalog, schema, prefix, domain)
# → f"{catalog}.{schema}.{prefix}_{domain}_index"

for domain in ("reference", "claims"):
    src_table = f"{catalog}.{schema}.{table_prefix}_{domain}_chunks"
    idx = f"{catalog}.{schema}.{table_prefix}_{domain}_index"
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

# COMMAND ----------
# MAGIC %md
# MAGIC ## Querying the indexes
# MAGIC
# MAGIC After index creation/sync completes, query each index via the Databricks SDK:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC # Query the reference index (policy docs, endorsements, underwriting guidelines)
# MAGIC results = w.vector_search_indexes.query_index(
# MAGIC     index_name=f"{catalog}.{schema}.{table_prefix}_reference_index",
# MAGIC     columns=["chunk_id", "chunk_to_retrieve", "doc_type", "source_path"],
# MAGIC     query_text="What is the deductible for water damage?",
# MAGIC     num_results=5,
# MAGIC )
# MAGIC
# MAGIC # Query the claims index (FNOL forms, adjuster reports)
# MAGIC results = w.vector_search_indexes.query_index(
# MAGIC     index_name=f"{catalog}.{schema}.{table_prefix}_claims_index",
# MAGIC     columns=["chunk_id", "chunk_to_retrieve", "doc_type", "source_path"],
# MAGIC     query_text="roof damage inspection findings",
# MAGIC     num_results=5,
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC The `chunk_to_retrieve` column contains the full retrieval text for each result.
# MAGIC The pipeline ends here — RAG query patterns live in downstream application notebooks.
