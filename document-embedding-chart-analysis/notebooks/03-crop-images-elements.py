# Databricks notebook source
# MAGIC %md
# MAGIC # This Notebook illustrate how to crop image elements from page and perform advance multi-modal analysis
# MAGIC
# MAGIC - Classify figure type based on description
# MAGIC - Perform VLM prompt on the figure that are charts and get insights

# COMMAND ----------

# MAGIC %pip install openai -q
# MAGIC %restart_python

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# COMMAND ----------

# MAGIC %md
# MAGIC # Classify based on image description
# MAGIC
# MAGIC Classes:
# MAGIC - logos
# MAGIC - charts
# MAGIC - other
# MAGIC
# MAGIC Uses `ai_classify` (task-specific function) instead of `ai_query` — preferred per
# MAGIC Databricks AI Functions guidance for fixed-label routing.

# COMMAND ----------

query = f"""
select
    ai_classify(
        description,
        '{{"logo":   "Company logos, brand logos, or corporate identity marks",
          "charts": "Analytic charts: bar chart, line chart, pie chart, combo, area, scatter, etc.",
          "other":  "Figures that are not a chart or a logo"}}'
    ):response[0]::string as diagram_type,
    *
from
(
    select
        *
    from {config['parsed_images_elements_bbox_table_name']}
    where type = 'figure'
)
"""

df_diagram = spark.sql(query)
display(df_diagram)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Charts

# COMMAND ----------

df_selected_diagram = df_diagram.filter(df_diagram['diagram_type'].isin(['charts']))
print(df_selected_diagram.count())
display(df_selected_diagram)

# COMMAND ----------

df_selected_diagram.write \
    .mode("overwrite") \
    .saveAsTable(f"{config['catalog']}.{config['schema']}.adv_analysis_prased_img_elements_bbox_classified_selected")

# COMMAND ----------

# MAGIC %md
# MAGIC # Crop Selected Images

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a image cropping pandas UDF

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col, current_timestamp
from pyspark.sql.types import StringType
import pandas as pd
from PIL import Image
import os

# COMMAND ----------

import shutil

cropped_images_path = f"/Volumes/{config['catalog']}/{config['schema']}/{config['volume']}/{config['volume_subdir']}/cropped_images/"

if os.path.exists(cropped_images_path):
    shutil.rmtree(cropped_images_path)

os.makedirs(cropped_images_path, exist_ok=True)

# COMMAND ----------

@pandas_udf(returnType=StringType())
def crop_figure_udf(
    paths: pd.Series, element_ids: pd.Series, bboxes: pd.Series, image_uris: pd.Series
) -> pd.Series:
    def crop_single_figure(path, element_id, bbox, image_uri):
        try:
            file_name = os.path.basename(path).split(".")[0]

            # Replace 'images' folder with 'cropped_images' in the path
            cropped_image_path = image_uri.replace("/images/", "/cropped_images/")
            cropped_image_path = (
                "/".join(cropped_image_path.split("/")[:-1])
                + f"/{file_name}_{element_id}.jpg"
            )

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)

            # Open and crop image
            image = Image.open(image_uri)
            cropped_image = image.crop(bbox)
            cropped_image.save(cropped_image_path)

            return cropped_image_path
        except Exception as e:
            return f"Error: {str(e)}"

    # Apply the function to each row using vectorized operations
    results = []
    for i in range(len(paths)):
        result = crop_single_figure(
            paths.iloc[i], element_ids.iloc[i], bboxes.iloc[i], image_uris.iloc[i]
        )
        results.append(result)

    return pd.Series(results)

# COMMAND ----------

df_cropped_images = (
    df_selected_diagram.withColumn(
        "cropped_figure_path",
        crop_figure_udf(col("path"), col("element_id"), col("bbox"), col("image_uri")),
    )
    .withColumn("cropping_timestamp", current_timestamp())
)
display(df_cropped_images)

# COMMAND ----------

df_cropped_images.write \
    .mode("overwrite") \
    .saveAsTable(f"{config['catalog']}.{config['schema']}.adv_analysis_prased_img_elements_bbox_cropped_imgs")