# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Gold: splice chart insights into per-document text
# MAGIC
# MAGIC Streaming version of notebook 05. Reads the chart-insights stream and in
# MAGIC `foreachBatch`:
# MAGIC
# MAGIC 1. Determines the set of affected document paths via the cropped-mapping
# MAGIC    table (`cropped_figure_path → path`).
# MAGIC 2. Re-runs the NB05 gold SQL — but filtered to only the affected docs —
# MAGIC    to produce `(path, full_text, chart_insights ARRAY<STRUCT>, last_updated)`.
# MAGIC 3. `MERGE INTO adv_analysis_gold_document_text ON path`.
# MAGIC
# MAGIC The gold table has CDF enabled so Vector Search Delta-Sync (or other
# MAGIC incremental consumers) can pick up updates.

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

chart_insights_table = f"{catalog}.{schema}.{table_prefix}_chart_insights"
elements_table       = f"{catalog}.{schema}.{table_prefix}_parsed_elements"
mapping_table        = f"{catalog}.{schema}.{table_prefix}_cropped_mapping"
gold_table           = f"{catalog}.{schema}.{table_prefix}_gold_document_text"
artifacts_root       = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt                 = f"{artifacts_root}/checkpoints/gold_merge"

print(f"chart_insights_table = {chart_insights_table}")
print(f"elements_table       = {elements_table}")
print(f"mapping_table        = {mapping_table}")
print(f"gold_table           = {gold_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure gold target exists with CDF enabled

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {gold_table} (
    path STRING,
    full_text STRING,
    chart_insights ARRAY<STRUCT<element_id: INT, page_id: INT, bbox: ARRAY<INT>, insight: STRING>>,
    last_updated TIMESTAMP
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## foreachBatch: compute affected docs → recompute gold rows → MERGE

# COMMAND ----------

def merge_gold(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Determine the set of affected original-document paths via the mapping table
    batch_df.createOrReplaceTempView("incoming_insights_batch")
    affected_paths_df = spark.sql(f"""
        SELECT DISTINCT m.path
        FROM {mapping_table} m
        INNER JOIN incoming_insights_batch b
          ON b.path = m.cropped_figure_path
    """)
    if affected_paths_df.isEmpty():
        return
    affected_paths_df.createOrReplaceTempView("affected_paths")

    gold_rows = spark.sql(f"""
        WITH affected AS (
          SELECT path FROM affected_paths
        ),
        chart_keyed AS (
          SELECT
              m.path,
              m.element_id,
              m.page_id,
              m.bbox,
              ci.chart_insights AS insight
          FROM {mapping_table} m
          INNER JOIN {chart_insights_table} ci
              ON ci.path = m.cropped_figure_path
          INNER JOIN affected a
              ON a.path = m.path
          WHERE ci.chart_insights IS NOT NULL
            AND ci.chart_insights NOT LIKE 'Error processing image%'
        ),
        elements_filtered AS (
          SELECT e.*
          FROM {elements_table} e
          INNER JOIN affected a ON e.path = a.path
        ),
        elements_spliced AS (
          SELECT
              e.path,
              e.element_id,
              e.type,
              e.page_id,
              e.bbox,
              CASE
                WHEN e.type = 'figure' AND ck.insight IS NOT NULL
                  THEN concat('[CHART(', cast(e.element_id AS string), '): ', ck.insight, ']')
                ELSE e.content
              END AS content
          FROM elements_filtered e
          LEFT JOIN chart_keyed ck
            ON  ck.path       = e.path
            AND ck.element_id = e.element_id
        ),
        full_text_per_doc AS (
          SELECT
              path,
              concat_ws(
                '\\n\\n',
                transform(
                  array_sort(collect_list(struct(element_id, content))),
                  x -> x.content
                )
              ) AS full_text
          FROM elements_spliced
          GROUP BY path
        ),
        structured_insights AS (
          SELECT
              path,
              array_sort(
                collect_list(named_struct(
                  'element_id', element_id,
                  'page_id',    page_id,
                  'bbox',       bbox,
                  'insight',    insight
                ))
              ) AS chart_insights
          FROM chart_keyed
          GROUP BY path
        )
        SELECT
            t.path,
            t.full_text,
            coalesce(s.chart_insights, array()) AS chart_insights,
            current_timestamp() AS last_updated
        FROM full_text_per_doc t
        LEFT JOIN structured_insights s ON s.path = t.path
    """)

    gold_rows.createOrReplaceTempView("gold_rows")
    spark.sql(f"""
        MERGE INTO {gold_table} t
        USING gold_rows s
          ON t.path = s.path
        WHEN MATCHED THEN UPDATE SET
            t.full_text      = s.full_text,
            t.chart_insights = s.chart_insights,
            t.last_updated   = s.last_updated
        WHEN NOT MATCHED THEN INSERT (
            path, full_text, chart_insights, last_updated
        ) VALUES (
            s.path, s.full_text, s.chart_insights, s.last_updated
        )
    """)

# COMMAND ----------

stream = spark.readStream.table(chart_insights_table)

query = (
    stream.writeStream
    .foreachBatch(merge_gold)
    .option("checkpointLocation", ckpt)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()
print("Gold merge complete.")
print(f"Last progress: {query.lastProgress}")
