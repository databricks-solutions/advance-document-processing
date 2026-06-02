# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver: ai_classify figures + crop chart images
# MAGIC
# MAGIC Streaming version of notebook 03. Reads the elements stream (filtered to
# MAGIC figures), classifies each as logo / chart / other via `ai_classify`,
# MAGIC filters to charts, then in `foreachBatch`:
# MAGIC
# MAGIC 1. Joins with the pages snapshot to attach `image_uri` per element.
# MAGIC 2. Crops the figure's bbox out of the page image using a pandas UDF —
# MAGIC    writes the JPG to `/Volumes/.../cropped_images/`.
# MAGIC 3. `MERGE`s the `(path, element_id) → cropped_figure_path` mapping row
# MAGIC    into the cropped-mapping table so re-runs are idempotent.

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

elements_table = f"{catalog}.{schema}.{table_prefix}_parsed_elements"
pages_table    = f"{catalog}.{schema}.{table_prefix}_parsed_pages"
mapping_table  = f"{catalog}.{schema}.{table_prefix}_cropped_mapping"
artifacts_root = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt           = f"{artifacts_root}/checkpoints/silver_classify_crop"

print(f"elements_table = {elements_table}")
print(f"pages_table    = {pages_table}")
print(f"mapping_table  = {mapping_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cropping UDF (lifted verbatim from notebook 03)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col, current_timestamp, expr
from pyspark.sql.types import StringType
import pandas as pd
from PIL import Image
import os


@pandas_udf(returnType=StringType())
def crop_figure_udf(
    paths: pd.Series, element_ids: pd.Series, bboxes: pd.Series, image_uris: pd.Series
) -> pd.Series:
    def crop_single_figure(path, element_id, bbox, image_uri):
        try:
            file_name = os.path.basename(path).split(".")[0]
            cropped_image_path = image_uri.replace("/images/", "/cropped_images/")
            cropped_image_path = (
                "/".join(cropped_image_path.split("/")[:-1])
                + f"/{file_name}_{element_id}.jpg"
            )
            os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)
            image = Image.open(image_uri)
            cropped_image = image.crop(bbox)
            cropped_image.save(cropped_image_path)
            return cropped_image_path
        except Exception as e:
            return f"Error: {str(e)}"

    results = []
    for i in range(len(paths)):
        results.append(
            crop_single_figure(
                paths.iloc[i], element_ids.iloc[i], bboxes.iloc[i], image_uris.iloc[i]
            )
        )
    return pd.Series(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target mapping table exists

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {mapping_table} (
    path STRING,
    element_id INT,
    page_id INT,
    bbox ARRAY<INT>,
    image_uri STRING,
    diagram_type STRING,
    cropped_figure_path STRING,
    cropping_timestamp TIMESTAMP
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## foreachBatch: classify → crop → MERGE

# COMMAND ----------

CLASSIFY_LABELS = (
    '{"logo":   "Company logos, brand logos, or corporate identity marks",'
    '  "charts": "Analytic charts: bar chart, line chart, pie chart, combo, area, scatter, etc.",'
    '  "other":  "Figures that are not a chart or a logo"}'
)


def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Defensive dedup: a re-ingested PDF (e.g., after a checkpoint reset)
    # would produce duplicate elements with the same (path, element_id);
    # dedup both sides of the join so the MERGE has a single match per key.
    batch_df = batch_df.dropDuplicates(["path", "element_id"])
    pages_static = spark.table(pages_table).dropDuplicates(["path", "page_id"])
    enriched = batch_df.join(pages_static, ["path", "page_id"], "inner")

    # Classify on the figure description
    classified = enriched.withColumn(
        "diagram_type",
        expr(f"ai_classify(description, '{CLASSIFY_LABELS}'):response[0]::string"),
    ).filter("diagram_type = 'charts'")

    if classified.isEmpty():
        return

    # Crop figures (writes JPG side-effect into the cropped_images dir)
    cropped = (
        classified
        .withColumn(
            "cropped_figure_path",
            crop_figure_udf(col("path"), col("element_id"), col("bbox"), col("image_uri")),
        )
        .withColumn("cropping_timestamp", current_timestamp())
        .select(
            "path",
            "element_id",
            "page_id",
            "bbox",
            "image_uri",
            "diagram_type",
            "cropped_figure_path",
            "cropping_timestamp",
        )
    )

    cropped.createOrReplaceTempView("incoming_mapping")
    spark.sql(f"""
        MERGE INTO {mapping_table} t
        USING incoming_mapping s
          ON t.path = s.path AND t.element_id = s.element_id
        WHEN MATCHED THEN UPDATE SET
            t.page_id             = s.page_id,
            t.bbox                = s.bbox,
            t.image_uri           = s.image_uri,
            t.diagram_type        = s.diagram_type,
            t.cropped_figure_path = s.cropped_figure_path,
            t.cropping_timestamp  = s.cropping_timestamp
        WHEN NOT MATCHED THEN INSERT (
            path, element_id, page_id, bbox,
            image_uri, diagram_type, cropped_figure_path, cropping_timestamp
        ) VALUES (
            s.path, s.element_id, s.page_id, s.bbox,
            s.image_uri, s.diagram_type, s.cropped_figure_path, s.cropping_timestamp
        )
    """)

# COMMAND ----------

stream = (
    spark.readStream.table(elements_table)
    .where("type = 'figure'")
)

query = (
    stream.writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", ckpt)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()
print("Classify/crop write complete.")
print(f"Last progress: {query.lastProgress}")
