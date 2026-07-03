# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze: Batch `ai_parse_document`
# MAGIC
# MAGIC Reads every paystub PDF currently in the source UC volume, parses each
# MAGIC document with `ai_parse_document`, and rebuilds the bronze table (one row per
# MAGIC source PDF). Page images are written to the workflow artifact root for
# MAGIC downstream OCR.
# MAGIC
# MAGIC Full-batch and idempotent: each run reprocesses the whole volume and overwrites
# MAGIC the bronze table, matching the `CREATE OR REPLACE` behavior of tasks 02-03.
# MAGIC The intended workflow is upload PDFs -> run job -> read the `*_enriched` table.
# MAGIC
# MAGIC Requires serverless env v3+ for `ai_parse_document`.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "paystubs")
dbutils.widgets.text("volume_subdir", "")
dbutils.widgets.text("table_prefix", "paystub_bbox_stream")
dbutils.widgets.dropdown("reset_data", "false", ["true", "false"])

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir").strip("/")
table_prefix = dbutils.widgets.get("table_prefix")
reset_data = dbutils.widgets.get("reset_data") == "true"

volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
source_path = f"{volume_root}/{volume_subdir}" if volume_subdir else volume_root
artifact_scope = volume_subdir if volume_subdir else "_root"
artifacts_root = f"{volume_root}/_artifacts/{table_prefix}"
image_path = f"{artifacts_root}/images/{artifact_scope}"
bronze_table = f"{catalog}.{schema}.{table_prefix}_bronze_parsed_docs"

print(f"source_path  = {source_path}")
print(f"image_path   = {image_path}")
print(f"bronze_table = {bronze_table}")
print(f"reset_data   = {reset_data}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional reset
# MAGIC
# MAGIC When `reset_data=true`, drop every pipeline table and delete the generated
# MAGIC artifact subfolders (images / crops / enriched JSON) under the volume, so the
# MAGIC run starts from a clean state. Source PDFs in the volume are left untouched.

# COMMAND ----------

# All Delta tables the pipeline creates (tasks 01-03), keyed off table_prefix.
PIPELINE_TABLE_SUFFIXES = [
    "bronze_parsed_docs",
    "extracted",
    "extracted_flat",
    "ocr_tokens",
    "localized",
    "enriched",
]

if reset_data:
    for suffix in PIPELINE_TABLE_SUFFIXES:
        table_name = f"{catalog}.{schema}.{table_prefix}_{suffix}"
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        print(f"dropped table {table_name}")

    # Remove the whole artifact root (images/, crops/, enriched_json/ subfolders).
    try:
        dbutils.fs.rm(artifacts_root, True)
        print(f"removed artifacts   {artifacts_root}")
    except Exception as ex:
        print(f"artifacts remove skipped ({artifacts_root}): {ex}")
    print("reset_data complete.")
else:
    print("reset_data=false; skipping cleanup.")

# COMMAND ----------

ai_parse_expr = (
    "ai_parse_document(content, "
    "map('version', '2.0', "
    f"    'imageOutputPath', '{image_path}', "
    "    'descriptionElementTypes', '*')"
    ") AS parsed"
)

bronze_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .option("recursiveFileLookup", "true")
    .load(source_path)
    .where("path NOT LIKE '%/_artifacts/%'")
    .selectExpr("path", ai_parse_expr, "current_timestamp() AS ingested_at")
)

(
    bronze_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)

# COMMAND ----------

row_count = spark.table(bronze_table).count()
print(f"Bronze parse complete. Row count: {row_count}")
dbutils.notebook.exit(f"bronze_rows={row_count}")
