# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: explode pages + elements from VARIANT
# MAGIC
# MAGIC Streaming version of notebook 01's silver steps. Two append-only streams
# MAGIC read the bronze table and explode `parsed:document:pages` and
# MAGIC `parsed:document:elements` into per-row Delta tables. Each stream has
# MAGIC its own checkpoint so they can advance independently.

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

bronze_table   = f"{catalog}.{schema}.{table_prefix}_raw_files"
pages_table    = f"{catalog}.{schema}.{table_prefix}_parsed_pages"
elements_table = f"{catalog}.{schema}.{table_prefix}_parsed_elements"
artifacts_root = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt_pages     = f"{artifacts_root}/checkpoints/silver_pages"
ckpt_elements  = f"{artifacts_root}/checkpoints/silver_elements"

print(f"bronze_table   = {bronze_table}")
print(f"pages_table    = {pages_table}")
print(f"elements_table = {elements_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target tables exist

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {pages_table} (
    path STRING,
    page_id INT,
    image_uri STRING
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {elements_table} (
    path STRING,
    element_id INT,
    type STRING,
    bbox ARRAY<INT>,
    page_id INT,
    content STRING,
    description STRING
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pages stream

# COMMAND ----------

pages_stream = (
    spark.readStream.table(bronze_table)
    .where("parsed:document:pages IS NOT NULL AND CAST(parsed:error_status AS STRING) IS NULL")
    .selectExpr(
        "path",
        "posexplode(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS (page_id, page_item)",
    )
    .selectExpr(
        "path",
        "page_id",
        "cast(page_item:image_uri as string) as image_uri",
    )
)

q_pages = (
    pages_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", ckpt_pages)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(pages_table)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Elements stream

# COMMAND ----------

elements_stream = (
    spark.readStream.table(bronze_table)
    .where("parsed:document:elements IS NOT NULL AND CAST(parsed:error_status AS STRING) IS NULL")
    .selectExpr(
        "path",
        "posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (idx, items)",
    )
    .selectExpr(
        "path",
        "cast(items:id as int) as element_id",
        "cast(items:type as string) as type",
        "cast(items:bbox[0]:coord as ARRAY<INT>) as bbox",
        "cast(items:bbox[0]:page_id as int) as page_id",
        """CASE
              WHEN cast(items:type as string) = 'figure'
                THEN cast(items:description as string)
              ELSE cast(items:content as string)
           END as content""",
        "cast(items:description as string) as description",
    )
)

q_elements = (
    elements_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", ckpt_elements)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(elements_table)
)

# COMMAND ----------

# Wait for both streams to complete this AvailableNow run
q_pages.awaitTermination()
q_elements.awaitTermination()
print("Silver writes complete.")
print(f"pages    last progress: {q_pages.lastProgress}")
print(f"elements last progress: {q_elements.lastProgress}")

pages_cnt    = spark.table(pages_table).count()
elements_cnt = spark.table(elements_table).count()
print(f"pages table count    = {pages_cnt}")
print(f"elements table count = {elements_cnt}")
dbutils.notebook.exit(f"pages={pages_cnt},elements={elements_cnt}")
