# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Chart insights via VLM
# MAGIC
# MAGIC Streaming version of notebook 04. Auto Loader picks up newly-cropped
# MAGIC chart JPGs from `/Volumes/.../cropped_images/`. For each new image we
# MAGIC call the configured VLM serving endpoint (OpenAI-compatible chat
# MAGIC completions) and MERGE the result into `adv_analysis_chart_insights`
# MAGIC keyed by image path.

# COMMAND ----------

# MAGIC %pip install openai -q
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "complex_documents")
dbutils.widgets.text("table_prefix", "adv_analysis")
dbutils.widgets.text("llm_model", "databricks-claude-sonnet-4-5")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
volume        = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix  = dbutils.widgets.get("table_prefix")
LLM_MODEL     = dbutils.widgets.get("llm_model")

chart_insights_table = f"{catalog}.{schema}.{table_prefix}_chart_insights"
artifacts_root       = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
cropped_dir          = f"{artifacts_root}/cropped_images/{volume_subdir}"
ckpt                 = f"{artifacts_root}/checkpoints/chart_insights"
schema_loc           = f"{artifacts_root}/schemas/chart_insights"

print(f"chart_insights_table = {chart_insights_table}")
print(f"cropped_dir          = {cropped_dir}")
print(f"llm_model            = {LLM_MODEL}")

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col, expr
from pyspark.sql.types import StringType
from typing import Iterator
import base64
import pandas as pd
from openai import OpenAI

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
HOST  = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'

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

@pandas_udf(StringType())
def extract_chart_insight_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
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
                    timeout=60.0,
                )
                results.append(completion.choices[0].message.content)
            except Exception as e:
                error_msg = f"Error processing image {idx}: {type(e).__name__}: {str(e)}"
                results.append(error_msg)
        yield pd.Series(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target table exists

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {chart_insights_table} (
    path STRING,
    modificationTime TIMESTAMP,
    length BIGINT,
    chart_insights STRING
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader on cropped_images + foreachBatch MERGE

# COMMAND ----------

def merge_insights(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    # Auto Loader's `path` is prefixed with `dbfs:` while NB03's cropping UDF
    # writes paths without that prefix into the mapping table. Strip it here so
    # downstream stage-5 joins on path work without further normalization.
    transformed = (
        batch_df
        .withColumn("chart_insights", extract_chart_insight_udf(col("content")))
        .withColumn("path", expr("regexp_replace(path, '^dbfs:', '')"))
        .select("path", "modificationTime", "length", "chart_insights")
        .dropDuplicates(["path"])
    )
    transformed.createOrReplaceTempView("incoming_insights")
    spark.sql(f"""
        MERGE INTO {chart_insights_table} t
        USING incoming_insights s
          ON t.path = s.path
        WHEN MATCHED THEN UPDATE SET
            t.modificationTime = s.modificationTime,
            t.length           = s.length,
            t.chart_insights   = s.chart_insights
        WHEN NOT MATCHED THEN INSERT (
            path, modificationTime, length, chart_insights
        ) VALUES (
            s.path, s.modificationTime, s.length, s.chart_insights
        )
    """)

# COMMAND ----------

stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", schema_loc)
    .option("pathGlobFilter", "*.jpg")
    .load(cropped_dir)
)

query = (
    stream.writeStream
    .foreachBatch(merge_insights)
    .option("checkpointLocation", ckpt)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()
print("Chart-insights write complete.")
print(f"Last progress: {query.lastProgress}")
