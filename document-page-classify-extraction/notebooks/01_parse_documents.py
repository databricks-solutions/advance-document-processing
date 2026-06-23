# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Parse Documents (Bronze)
# MAGIC
# MAGIC Reads mortgage loan-file PDFs from a UC Volume and parses each with
# MAGIC `ai_parse_document`. A loan file is a single multi-page packet bundling
# MAGIC many distinct sub-documents (1003, pay stubs, bank statements, appraisal,
# MAGIC closing disclosure, …) — the page-level classify + per-class extract flow
# MAGIC in notebooks 03/04 splits and routes them.
# MAGIC
# MAGIC Configuration is driven by **widgets** (top of this notebook). NB01 writes
# MAGIC `config.yaml`, which NB02–05 read — so catalog/schema/volume/table names
# MAGIC are defined once, here, and never hard-coded downstream.
# MAGIC
# MAGIC Requires **DBR 17.1+** (or a Pro/Serverless SQL warehouse) for `ai_parse_document`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up environment

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "mortgage_loan_files")
dbutils.widgets.text("table_prefix", "loan_docs")
dbutils.widgets.dropdown("reset_data", "true", ["true", "false"])

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix = dbutils.widgets.get("table_prefix")
reset_data = dbutils.widgets.get("reset_data") == "true"

print(f"Use Unity Catalog: {catalog}")
print(f"Use Schema: {schema}")
print(f"Use Volume: {volume}")
print(f"Use Volume Subdir: {volume_subdir}")
print(f"Use Table Prefix: {table_prefix}")
print(f"Reset Data: {reset_data}")

# COMMAND ----------

pipeline_config = {
    "source_path":       f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}",
    "image_path":        f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}/images",
    "bronze_table":      f"{catalog}.{schema}.{table_prefix}_bronze_parsed_docs",
    "silver_pages_table":      f"{catalog}.{schema}.{table_prefix}_silver_pages",
    "classified_table":  f"{catalog}.{schema}.{table_prefix}_silver_pages_classified",
    "extracted_table":   f"{catalog}.{schema}.{table_prefix}_silver_pages_extracted",
    "gold_loans_table":  f"{catalog}.{schema}.{table_prefix}_gold_loan_files",
    "gold_income_table": f"{catalog}.{schema}.{table_prefix}_gold_income_docs",
    "gold_bank_table":   f"{catalog}.{schema}.{table_prefix}_gold_bank_statements",
}
print(pipeline_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the config to a yaml file
# MAGIC
# MAGIC Written to the notebook working directory; NB02–05 read it back.

# COMMAND ----------

import yaml

with open("config.yaml", "w") as f:
    yaml.dump({
        "catalog": catalog,
        "schema": schema,
        "volume": volume,
        "volume_subdir": volume_subdir,
        "table_prefix": table_prefix,
        **pipeline_config,
    }, f)

# COMMAND ----------

if reset_data:
    print("Dropping existing pipeline tables ...")
    for key in (
        "bronze_table", "silver_pages_table", "classified_table",
        "extracted_table", "gold_loans_table", "gold_income_table",
        "gold_bank_table",
    ):
        spark.sql(f"DROP TABLE IF EXISTS {pipeline_config[key]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse with `ai_parse_document`

# COMMAND ----------

from pyspark.sql.functions import expr, current_timestamp, col

raw = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "false")
    .option("pathGlobFilter", "*.pdf")
    .load(pipeline_config["source_path"])
    .filter(~col("path").contains("/images/"))
)

parsed = (
    raw.withColumn(
        "parsed",
        expr(
            "ai_parse_document(content, "
            "map('version', '2.0', "
            f"'imageOutputPath', '{pipeline_config['image_path']}', "
            "'descriptionElementTypes', '*'))"
        ),
    )
    .select("path", "parsed")
    .withColumn("ingested_at", current_timestamp())
)

(
    parsed.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(pipeline_config["bronze_table"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"""
SELECT
  path,
  parsed:metadata:file_metadata:file_name::string AS file_name,
  parsed:metadata:version::string                 AS schema_version,
  size(variant_get(parsed, '$.document.pages', 'array<variant>')) AS page_count,
  size(variant_get(parsed, '$.error_status', 'array<variant>'))   AS error_count,
  ingested_at
FROM {pipeline_config['bronze_table']}
ORDER BY file_name
"""))
