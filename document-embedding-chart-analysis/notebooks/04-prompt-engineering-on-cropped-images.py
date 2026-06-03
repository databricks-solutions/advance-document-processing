# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Engineering with VLM

# COMMAND ----------

# MAGIC %pip install openai markdown -q
# MAGIC %restart_python

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("llm_model", "databricks-claude-sonnet-4-5", "LLM model endpoint")
LLM_MODEL = dbutils.widgets.get("llm_model")
print(f"Using LLM model: {LLM_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Engineering on images to get pin details using Pandas UDF

# COMMAND ----------

cropped_images_df = spark.read \
    .format("binaryFile") \
    .load(f"/Volumes/{config['catalog']}/{config['schema']}/{config['volume']}/{config['volume_subdir']}/cropped_images/")
display(cropped_images_df)

# COMMAND ----------

dbutils.widgets.dropdown(
    name='path_dropdown',
    defaultValue=cropped_images_df.select('path').first()['path'],
    choices=[row['path'] for row in cropped_images_df.select('path').distinct().collect()],
    label='Select Image'
)

# COMMAND ----------

selected_img = dbutils.widgets.get('path_dropdown')
print(f"Select {selected_img}")

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType
from typing import Iterator
import base64
from openai import OpenAI
import pandas as pd
import time

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
HOST = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'

client = OpenAI(
    api_key=TOKEN,
    base_url=HOST
)

IMAGE_ANALYSIS_PROMPT = """
You are analyzing charts from a company's earnings call presentation. Please examine the provided chart image and provide a comprehensive analysis following this structure:

## 1. Chart Identification
- Chart type (bar, pie, line, combo, etc.)
- Title and subtitle
- Time period covered
- Data source/footnotes (if visible)

## 2. Quantitative Analysis
- Extract all visible data points with exact values
- Identify key metrics being measured (revenue, margins, growth rates, etc.)
- Calculate period-over-period changes and growth rates
- Note any missing data points or incomplete series

## 3. Trend Analysis
- Primary trends (upward, downward, flat, cyclical)
- Inflection points or significant changes
- Compare actual vs. projections (if shown)
- Seasonality patterns (if applicable)

## 4. Business Insights
- What story is this chart telling about company performance?
- Competitive positioning (if comparison data shown)
- Risk indicators or warning signs
- Positive momentum or achievements
- Segment/product performance differences

## 5. Context & Implications
- How does this relate to company guidance or strategy?
- Market conditions or external factors implied
- Potential investor concerns or highlights
- Forward-looking indicators

## 6. Data Quality Notes
- Any visual clarity issues
- Ambiguous labels or scales
- Potential data presentation concerns

Please format numerical data in tables where appropriate and highlight the top 3 most important insights.
"""


# COMMAND ----------

image_data_raw = cropped_images_df.filter(cropped_images_df['path'] == selected_img).select("content").collect()
image_data = base64.b64encode(image_data_raw[0]['content']).decode("utf-8")
displayHTML(f'<img src="data:image/jpeg;base64,{image_data}" style="max-width: 100%;height: auto; display: block;">')

# COMMAND ----------

completion = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": IMAGE_ANALYSIS_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            ],
        }
    ],
)

# COMMAND ----------

import markdown

displayHTML(markdown.markdown(completion.choices[0].message.content, extensions=['tables']))

#displayHTML(markdown.markdown(completion.choices[0].message.content))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform to all cropped charts using Pandas UDF

# COMMAND ----------

@pandas_udf(StringType())
def extract_chart_insight_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Initialize client once per executor partition
    client = OpenAI(api_key=TOKEN, base_url=HOST)

    for content_series in iterator:
        results = []
        for idx, content in enumerate(content_series):
            try:
                image_data = base64.b64encode(content).decode("utf-8")
                completion = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ],
                    }],
                    timeout=60.0  # Add timeout
                )
                results.append(completion.choices[0].message.content)
            except Exception as e:
                error_msg = f"Error processing image {idx}: {type(e).__name__}: {str(e)}"
                results.append(error_msg)

        yield pd.Series(results)

# COMMAND ----------

chart_insights_df = cropped_images_df \
    .withColumn("chart_insights", extract_chart_insight_udf(col("content")))
display(chart_insights_df)

# COMMAND ----------

chart_insights_df.select(
    "path",
    "modificationTime",
    "length",
    "chart_insights",
).write \
    .mode("overwrite") \
    .saveAsTable(f"{config['catalog']}.{config['schema']}.adv_analysis_chart_insights")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC * Insert insights as context for downstream task
# MAGIC   * Information Extraction
# MAGIC   * RAG Agents
# MAGIC   * Structured Data