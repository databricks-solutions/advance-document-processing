# Databricks notebook source
# MAGIC %md
# MAGIC # Ingestion Pattern: Use databricks AI Parse Document UDF
# MAGIC
# MAGIC - Ingest and parse pdf using `ai_parse_document` ([Document](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document?language=Python))
# MAGIC - Perform transformation to create a full document content markdown text
# MAGIC - Add a ground truth column, by reading the json ground truth files in a directory called `ground_truth_fixed` under the same volume as the pdfs.
# MAGIC
# MAGIC Note: 
# MAGIC
# MAGIC `ai_parse_document` signature (requires DBR 17.3+; serverless env v3+ for VARIANT)
# MAGIC
# MAGIC ```sql
# MAGIC SELECT
# MAGIC   ai_parse_document(
# MAGIC       content,
# MAGIC       map('version', '2.0',
# MAGIC           'imageOutputPath', '/path/to/direct_to_store_image',
# MAGIC           'descriptionElementTypes', '*',         -- '' | 'figure' | '*'
# MAGIC           'pageRange', '1,3,5-10')                -- optional, restrict pages
# MAGIC   )
# MAGIC FROM READ_FILES('/path/to/documents', format => 'binaryFile')
# MAGIC ```
# MAGIC
# MAGIC `ai_parse_document` v2.0 Schema
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "document": {
# MAGIC     "pages": [
# MAGIC       {
# MAGIC         "id": INT,
# MAGIC         "image_uri": STRING  // Path to saved page image (if imageOutputPath set)
# MAGIC       }
# MAGIC     ],
# MAGIC     "elements": [
# MAGIC       {
# MAGIC         "id": INT,
# MAGIC         "type": STRING,      // text | table | figure | title | caption
# MAGIC                              //   | section_header | page_header | page_footer
# MAGIC                              //   | page_number | footnote
# MAGIC         "content": STRING,
# MAGIC         "confidence": DOUBLE,  // 0-1
# MAGIC         "bbox": [
# MAGIC           {
# MAGIC             "coord": [ INT ],
# MAGIC             "page_id": INT
# MAGIC           }
# MAGIC         ],
# MAGIC         "description": STRING  // AI-generated description (figures/tables when enabled)
# MAGIC       }
# MAGIC     ]
# MAGIC   },
# MAGIC   "error_status": [
# MAGIC     {
# MAGIC       "error_message": STRING,
# MAGIC       "page_id": INT
# MAGIC     }
# MAGIC   ],
# MAGIC   "metadata": {
# MAGIC     "id": STRING,
# MAGIC     "version": STRING,
# MAGIC     "file_metadata": STRUCT
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Limits: max 500 pages per document, max 100 MB file size.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Environment

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "complex_documents")
dbutils.widgets.text('num_of_files', "2")
dbutils.widgets.text("table_prefix", "adv_analysis")
dbutils.widgets.dropdown("reset_data", "true", ["true", "false"])

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
num_of_files = dbutils.widgets.get("num_of_files")
table_prefix = dbutils.widgets.get("table_prefix")
reset_data = dbutils.widgets.get("reset_data") == "true"

print(f"Use Unit Catalog: {catalog}")
print(f"Use Schema: {schema}")
print(f"Use Volume: {volume}")
print(f"Use Volume Subdir: {volume_subdir}")
print(f"process {num_of_files} files")
print(f"Use Table Prefix: {table_prefix}")
print(f"Reset Data: {reset_data}")

# COMMAND ----------

pipeline_config = {
    "source_path": f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}",
    "raw_files_table_name": f"{catalog}.{schema}.{table_prefix}_raw_files",
    "parsed_pages_table_name": f"{catalog}.{schema}.{table_prefix}_parsed_pages",
    "parsed_elements_table_name": f"{catalog}.{schema}.{table_prefix}_parsed_elements",
    "parsed_images_elements_bbox_table_name": f"{catalog}.{schema}.{table_prefix}_prased_img_elements_bbox",
    "image_path": f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}/images"
}
print(pipeline_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the config to a yaml file

# COMMAND ----------

# write a config yaml file for other notebooks
import yaml

with open("config.yaml", "w") as f:
    yaml.dump({
    "catalog": catalog,
    "schema": schema,
    "volume": volume,
    "volume_subdir": volume_subdir,
    **pipeline_config}, f)

# COMMAND ----------

if reset_data:
    print("Delete tables ...")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['raw_files_table_name']}")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['parsed_pages_table_name']}")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['parsed_elements_table_name']}")
    spark.sql(f"DROP TABLE IF EXISTS {pipeline_config['parsed_images_elements_bbox_table_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Use `ai_parse_document` UDF to Ingest PDF Documents
# MAGIC
# MAGIC * create a bronze table store raw binary
# MAGIC * create a silver table with parsed markdown text and contextual data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create bronze table
# MAGIC
# MAGIC - Ingest from volume and write variant content to bronze table

# COMMAND ----------

bronze_sql_script = f"""
CREATE TABLE {pipeline_config['raw_files_table_name']} AS
WITH files as (
  SELECT
    path,
    content
  FROM
    READ_FILES('{pipeline_config['source_path']}/*.{{pdf,jpg,jpeg,png}}', format => 'binaryFile')
  ORDER BY path ASC
  limit {num_of_files}
)
SELECT
  path,
    ai_parse_document(
        content,
        map('version', '2.0',
            'imageOutputPath', '{pipeline_config['image_path']}',
            'descriptionElementTypes', '*')
    ) as parsed
FROM files
"""

spark.sql(bronze_sql_script)
display(spark.table(pipeline_config['raw_files_table_name']))

# COMMAND ----------

dbutils.fs.ls(pipeline_config['image_path'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create silver tables
# MAGIC
# MAGIC - Page images url
# MAGIC - Elements
# MAGIC - Image-element bbox join (for downstream cropping)
# MAGIC
# MAGIC Note: `ai_parse_document`'s VARIANT output in the bronze table can be consumed
# MAGIC directly by other AI functions (`ai_extract`, `ai_classify`, `ai_summarize`),
# MAGIC so we do not materialize a separate full-text content table.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Page image uri table

# COMMAND ----------

page_img_query = f"""
CREATE OR REPLACE TABLE {pipeline_config['parsed_pages_table_name']} as
SELECT
    path,
    id as page_id,
    cast(image_uri:image_uri as string) as image_uri
FROM
(
    SELECT
        path,
        posexplode(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS (id, image_uri)
    FROM {pipeline_config['raw_files_table_name']}
    WHERE parsed:document:pages IS NOT NULL
    AND CAST(parsed:error_status AS STRING) IS NULL
)
"""

spark.sql(page_img_query)
display(spark.table(f"{pipeline_config['parsed_pages_table_name']}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Elements Table

# COMMAND ----------

elements_query = f"""
CREATE OR REPLACE TABLE {pipeline_config['parsed_elements_table_name']} as 
select
  path,
  cast(items:id as int) as element_id,
  cast(items:type as string) as type,
  cast(items:bbox[0]:coord as ARRAY<INT>) as bbox,
  cast(items:bbox[0]:page_id as int) as page_id,
  CASE 
    WHEN cast(items:type as string) = 'figure' THEN cast(items:description as string)
    ELSE cast(items:content as string)
  END as content,
  cast(items:description as string) as description
from
(
  SELECT
    path,
    posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (idx, items)
  FROM {pipeline_config['raw_files_table_name']}
  WHERE
    parsed:document:elements IS NOT NULL
    AND CAST(parsed:error_status AS STRING) IS NULL
)
"""

spark.sql(elements_query)
display(spark.table(f"{pipeline_config['parsed_elements_table_name']}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an image element bbox tables with image URI
# MAGIC
# MAGIC - This is useful if we would like to crop images using boudning box

# COMMAND ----------

image_bbox_query = f"""
CREATE OR REPLACE TABLE {pipeline_config['parsed_images_elements_bbox_table_name']} as
select
    e.*,
    p.image_uri
from {pipeline_config['parsed_elements_table_name']} e
inner join {pipeline_config['parsed_pages_table_name']} p
on e.path = p.path and e.page_id = p.page_id
where type = 'figure'
"""

spark.sql(image_bbox_query)
display(spark.sql(f"SELECT * FROM {pipeline_config['parsed_images_elements_bbox_table_name']}"))

# COMMAND ----------

