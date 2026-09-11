# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Index: create/refresh Delta Sync Vector Search indexes
# MAGIC One index per gold table, managed embeddings (databricks-gte-large-en),
# MAGIC TRIGGERED sync. Embeds `chunk_to_embed`; return `chunk_to_retrieve` at query time.

# COMMAND ----------

import time
import insurance_kb_sql_builders as S  # co-located in src/ (Databricks adds notebook dir to sys.path)
# databricks-sdk create_index wants typed spec objects (it calls .as_dict() on them),
# not raw dicts — import the vectorsearch dataclasses/enums.
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)
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
    w.vector_search_endpoints.create_endpoint(name=vs_endpoint, endpoint_type=EndpointType.STANDARD)

# Poll until ONLINE (endpoint creation is asynchronous). Accessors below are
# verified against databricks-sdk on this workspace: endpoint_status.state is an
# enum (read .value), and list_indexes(...) yields MiniVectorIndex items directly.
for _ in range(60):
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=vs_endpoint)
    raw_state = ep.endpoint_status.state if ep.endpoint_status else None
    # state is an SDK enum (EndpointStatusState.ONLINE) — its .value is the
    # bare string "ONLINE"; a plain string falls through unchanged.
    state = getattr(raw_state, "value", None) or str(raw_state)
    print(f"endpoint state = {state}")
    if state == "ONLINE":
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
    # list_indexes yields MiniVectorIndex items directly (a generator) — iterate it.
    existing_idx = [i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=vs_endpoint)]
    if idx in existing_idx:
        print(f"syncing existing index {idx} ...")
        w.vector_search_indexes.sync_index(index_name=idx)
    else:
        print(f"creating index {idx} on {src_table} ...")
        w.vector_search_indexes.create_index(
            name=idx,
            endpoint_name=vs_endpoint,
            primary_key="chunk_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=src_table,
                pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name="chunk_to_embed",
                        embedding_model_endpoint_name=embedding_model,
                    )
                ],
            ),
        )

print("index creation/sync submitted.")
dbutils.notebook.exit("indexes_done")
