# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 5: Gold Document Table with Chart Insights Inline
# MAGIC
# MAGIC Build a per-document gold table where:
# MAGIC
# MAGIC - `full_text` is the concatenated content of all parsed elements ordered by
# MAGIC   `element_id`, with **chart figures replaced inline** by their VLM analysis
# MAGIC   produced in notebook 04 (`[CHART(<element_id>): <insight>]`).
# MAGIC - `chart_insights` is the structured array of insights for the document
# MAGIC   (`element_id`, `page_id`, `bbox`, `insight`) — kept alongside the text so
# MAGIC   downstream consumers can either embed the enriched text or query the
# MAGIC   structured slice.
# MAGIC
# MAGIC Output schema:
# MAGIC
# MAGIC | column          | type                                                     |
# MAGIC |-----------------|----------------------------------------------------------|
# MAGIC | path            | STRING (PK)                                              |
# MAGIC | full_text       | STRING                                                   |
# MAGIC | chart_insights  | ARRAY<STRUCT<element_id INT, page_id INT, bbox ARRAY<INT>, insight STRING>> |
# MAGIC | last_updated    | TIMESTAMP                                                |
# MAGIC
# MAGIC See the **Production Streaming Notes** at the bottom for how this batch SQL
# MAGIC maps to a Structured Streaming pipeline.

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inputs and outputs

# COMMAND ----------

elements_table     = config["parsed_elements_table_name"]                                                  # NB01
cropped_map_table  = f"{config['catalog']}.{config['schema']}.adv_analysis_prased_img_elements_bbox_cropped_imgs"  # NB03
chart_insights_tbl = f"{config['catalog']}.{config['schema']}.adv_analysis_chart_insights"                 # NB04
gold_table         = f"{config['catalog']}.{config['schema']}.adv_analysis_gold_document_text"

print(f"elements_table     = {elements_table}")
print(f"cropped_map_table  = {cropped_map_table}")
print(f"chart_insights_tbl = {chart_insights_tbl}")
print(f"gold_table         = {gold_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Re-key chart insights onto `(path, element_id)`
# MAGIC
# MAGIC NB04 writes `adv_analysis_chart_insights` from a `binaryFile` read of the
# MAGIC cropped JPGs, so its `path` is the cropped image filesystem path — not the
# MAGIC original PDF. NB03 already persisted the mapping
# MAGIC (`cropped_figure_path` ↔ original `path`, `element_id`, `page_id`, `bbox`)
# MAGIC in `adv_analysis_prased_img_elements_bbox_cropped_imgs`, so we join on that
# MAGIC rather than parsing filenames.

# COMMAND ----------

chart_keyed_sql = f"""
SELECT
    m.path        AS path,
    m.element_id  AS element_id,
    m.page_id     AS page_id,
    m.bbox        AS bbox,
    ci.chart_insights AS insight
FROM {cropped_map_table}  m
INNER JOIN {chart_insights_tbl} ci
    ON ci.path = m.cropped_figure_path
WHERE ci.chart_insights IS NOT NULL
  AND ci.chart_insights NOT LIKE 'Error processing image%'  -- skip UDF failures
"""

chart_keyed_df = spark.sql(chart_keyed_sql)
display(chart_keyed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Build the gold table in one pass
# MAGIC
# MAGIC - **Element splice**: left-join `parsed_elements` with the keyed insights.
# MAGIC   For `type = 'figure'` rows that have an insight, replace `content` with
# MAGIC   `[CHART(<element_id>): <insight>]`. All other elements pass through.
# MAGIC - **Per-doc full text**: ordered concat via `array_sort(collect_list(struct(element_id, content)))`
# MAGIC   so the ordering is deterministic (a bare `collect_list` is not order-safe).
# MAGIC - **Structured insights**: ordered `collect_list` of `(element_id, page_id, bbox, insight)`
# MAGIC   for the same document.

# COMMAND ----------

gold_sql = f"""
WITH chart_keyed AS (
  SELECT
      m.path,
      m.element_id,
      m.page_id,
      m.bbox,
      ci.chart_insights AS insight
  FROM {cropped_map_table}  m
  INNER JOIN {chart_insights_tbl} ci
      ON ci.path = m.cropped_figure_path
  WHERE ci.chart_insights IS NOT NULL
    AND ci.chart_insights NOT LIKE 'Error processing image%'
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
  FROM {elements_table} e
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
      transform(
        array_sort(
          collect_list(named_struct(
            'element_id', element_id,
            'page_id',    page_id,
            'bbox',       bbox,
            'insight',    insight
          ))
        ),
        x -> x
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
"""

gold_df = spark.sql(gold_sql)
display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Write the gold Delta table
# MAGIC
# MAGIC - `mergeSchema` so the structured `chart_insights` column can evolve.
# MAGIC - `delta.enableChangeDataFeed = true` so a Vector Search Delta-Sync index
# MAGIC   (or downstream consumers) can pick up incremental document updates.

# COMMAND ----------

(
    gold_df.write
    .format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(gold_table)
)

spark.sql(
    f"ALTER TABLE {gold_table} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Sanity check
# MAGIC
# MAGIC Show one document and confirm chart insights are spliced inline.

# COMMAND ----------

sample = spark.table(gold_table).limit(1).collect()
if sample:
    row = sample[0]
    print(f"path: {row['path']}")
    print(f"chart_insights count: {len(row['chart_insights'])}")
    print("-" * 80)
    print("full_text (first 4000 chars):")
    print(row["full_text"][:4000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Streaming Notes
# MAGIC
# MAGIC The batch SQL above maps cleanly onto Structured Streaming. The recommended
# MAGIC layering:
# MAGIC
# MAGIC 1. **Bronze** (NB01): Auto Loader (`cloudFiles`) → `ai_parse_document`.
# MAGIC    Stateless per row, streaming-safe.
# MAGIC
# MAGIC 2. **Silver — elements** (NB01): `explode(parsed:document:elements)`.
# MAGIC    Stateless per row, streaming-safe.
# MAGIC
# MAGIC 3. **Silver — chart insights** (NB03 + NB04): one row per chart figure with
# MAGIC    `(path, element_id, page_id, bbox, insight)`. To make this stateless and
# MAGIC    eliminate the filesystem side effect of cropping JPGs, replace the
# MAGIC    OpenAI client + pandas UDF with:
# MAGIC
# MAGIC    ```sql
# MAGIC    ai_query(
# MAGIC      'databricks-claude-sonnet-4-5',
# MAGIC      concat('Focus on bbox ', to_json(bbox), '. ', :IMAGE_ANALYSIS_PROMPT),
# MAGIC      files => array(image_uri),     -- page image, not cropped
# MAGIC      failOnError => false
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC    No file I/O, no executor-side OpenAI client, retries handled by the engine.
# MAGIC
# MAGIC 4. **Gold** (this notebook): the join + groupBy `path`. The cleanest streaming
# MAGIC    shape is **`foreachBatch` on the silver-elements stream**, which inside the
# MAGIC    micro-batch reads silver-chart-insights as a Delta snapshot, runs the same
# MAGIC    SQL above on the micro-batch's documents, and `MERGE INTO {gold_table}`s by
# MAGIC    `path`. This avoids stateful stream-stream joins and watermark complexity,
# MAGIC    and gives you exactly-once writes via the Delta sink.
# MAGIC
# MAGIC For first production rollout, prefer `trigger=AvailableNow` (triggered batch)
# MAGIC over continuous streaming — same DataFrame API and checkpointing, no always-on
# MAGIC compute, and a natural fit for the per-document grain.
