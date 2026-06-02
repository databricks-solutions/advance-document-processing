# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: Auto Loader + `ai_parse_document`
# MAGIC
# MAGIC Streaming version of notebook 01's bronze step. Picks up new PDF/image
# MAGIC files from the source volume incrementally with Auto Loader and writes a
# MAGIC Delta bronze table keyed by source `path` with the `ai_parse_document`
# MAGIC VARIANT output.
# MAGIC
# MAGIC Requires DBR 17.3+ (serverless env v3+) for `ai_parse_document`.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "complex_documents")
dbutils.widgets.text("table_prefix", "adv_analysis")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
volume        = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix  = dbutils.widgets.get("table_prefix")

source_path     = f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}"
# Streaming artifacts live under a sibling _streaming/ root so they don't
# collide with batch-notebook outputs that share the same source folder.
artifacts_root  = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
image_path      = f"{artifacts_root}/images/{volume_subdir}"
checkpoint_path = f"{artifacts_root}/checkpoints/bronze"
schema_path     = f"{artifacts_root}/schemas/bronze"
bronze_table    = f"{catalog}.{schema}.{table_prefix}_raw_files"

print(f"source_path     = {source_path}")
print(f"image_path      = {image_path}")
print(f"checkpoint_path = {checkpoint_path}")
print(f"bronze_table    = {bronze_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target table exists (so writeStream.toTable can append)

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {bronze_table} (
    path STRING,
    parsed VARIANT
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming read + ai_parse_document + streaming write

# COMMAND ----------

ai_parse_expr = (
    "ai_parse_document(content, "
    "map('version', '2.0', "
    f"    'imageOutputPath', '{image_path}', "
    "    'descriptionElementTypes', '*')"
    ") AS parsed"
)

stream_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", schema_path)
    .option("pathGlobFilter", "*.pdf")  # ignore any image/artifact subdirs under source
    .load(source_path)
    .selectExpr("path", ai_parse_expr)
)

# COMMAND ----------

query = (
    stream_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(bronze_table)
)

query.awaitTermination()
print("Bronze write complete.")
print(f"Last progress: {query.lastProgress}")

# Diagnostic: surface the row count into the Job UI
row_count = spark.table(bronze_table).count()
print(f"Bronze table row count: {row_count}")
dbutils.notebook.exit(f"bronze_rows={row_count}")
