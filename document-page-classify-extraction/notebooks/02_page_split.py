# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Page Split (Silver)
# MAGIC
# MAGIC Explodes the parsed VARIANT into one row per (path, page_id) and produces
# MAGIC **two views** of each page:
# MAGIC
# MAGIC 1. `page_variant` — sub-document VARIANT mirroring the `ai_parse_document`
# MAGIC    schema (`document.pages`, `document.elements`, `metadata`,
# MAGIC    `error_status`), filtered to elements whose first bbox lands on this
# MAGIC    page. Includes **all** element types (headers, footers, tables,
# MAGIC    figures, confidences, bboxes). Used by `ai_extract` in 04.
# MAGIC 2. `page_text` — concatenated plain-text view, with `page_header`,
# MAGIC    `page_footer`, `page_number` elements stripped and figure descriptions
# MAGIC    inlined as `[figure] <description>`. Used by `ai_classify` in 03.
# MAGIC
# MAGIC Config is read from `config.yaml` (written by NB01).
# MAGIC
# MAGIC - **Input:** bronze parsed-docs table
# MAGIC - **Output:** silver pages table

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {config['silver_pages_table']} AS
WITH page_keys AS (
  -- One row per (path, page); carries the original page VARIANT object
  -- and a reference to the parent parsed VARIANT for later filtering.
  SELECT
    b.path,
    b.parsed,
    p.value:id::int           AS page_id,
    p.value:image_uri::string AS image_uri,
    p.value                   AS page_obj
  FROM {config['bronze_table']} b
  LATERAL VIEW explode(
    variant_get(b.parsed, '$.document.pages', 'array<variant>')
  ) p AS value
),
elements AS (
  -- Flat element rows, used to build page_text below.
  SELECT
    b.path,
    e.value:id::int                                                     AS element_id,
    e.value:type::string                                                AS element_type,
    e.value:content::string                                             AS content,
    e.value:description::string                                         AS description,
    variant_get(e.value, '$.bbox[0].page_id', 'int')                    AS page_id,
    CASE WHEN e.value:type::string = 'table'  THEN 1 ELSE 0 END         AS is_table,
    CASE WHEN e.value:type::string = 'figure' THEN 1 ELSE 0 END         AS is_figure
  FROM {config['bronze_table']} b
  LATERAL VIEW explode(
    variant_get(b.parsed, '$.document.elements', 'array<variant>')
  ) e AS value
),
elements_kept AS (
  -- Drop layout chrome from the text body only.
  SELECT *
  FROM elements
  WHERE element_type NOT IN ('page_header', 'page_footer', 'page_number')
),
page_text AS (
  SELECT
    path,
    page_id,
    concat_ws(
      '\\n',
      array_sort(
        collect_list(
          named_struct(
            'eid', element_id,
            'txt',
            CASE
              WHEN element_type = 'figure' AND description IS NOT NULL
                THEN concat('[figure] ', description)
              WHEN element_type = 'table' AND content IS NOT NULL
                THEN concat('[table] ', content)
              ELSE content
            END
          )
        ),
        (l, r) -> CASE
          WHEN l.eid < r.eid THEN -1
          WHEN l.eid > r.eid THEN  1
          ELSE 0
        END
      ).txt
    )                                            AS page_text,
    COUNT(*)                                     AS element_count,
    MAX(is_table)  = 1                           AS has_table,
    MAX(is_figure) = 1                           AS has_figure
  FROM elements_kept
  GROUP BY path, page_id
)
SELECT
  k.path,
  k.page_id,
  -- Page-scoped sub-document VARIANT.
  -- Structure mirrors ai_parse_document output so it can be passed
  -- directly to ai_extract. Includes ALL element types for this page
  -- (no header/footer stripping — ai_extract benefits from structure).
  PARSE_JSON(TO_JSON(
    named_struct(
      'document', named_struct(
        'pages',    array(k.page_obj),
        'elements', filter(
          variant_get(k.parsed, '$.document.elements', 'array<variant>'),
          e -> variant_get(e, '$.bbox[0].page_id', 'int') = k.page_id
        )
      ),
      'metadata', variant_get(k.parsed, '$.metadata', 'variant'),
      'error_status', filter(
        variant_get(k.parsed, '$.error_status', 'array<variant>'),
        s -> s:page_id::int = k.page_id
      )
    )
  ))                                                      AS page_variant,
  COALESCE(t.page_text, '')                               AS page_text,
  length(COALESCE(t.page_text, ''))                       AS page_text_len,
  k.image_uri                                             AS image_uri,
  COALESCE(t.has_table,    false)                         AS has_table,
  COALESCE(t.has_figure,   false)                         AS has_figure,
  COALESCE(t.element_count, 0)                            AS element_count
FROM page_keys k
LEFT JOIN page_text t
  ON k.path = t.path AND k.page_id = t.page_id
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"""
SELECT path, COUNT(*) AS pages
FROM {config['silver_pages_table']}
GROUP BY path
ORDER BY path
"""))

# COMMAND ----------

# Distribution of page text length — useful sanity check on noise vs. content pages.
display(spark.sql(f"""
SELECT
  CASE
    WHEN page_text_len <  100 THEN '0 — very sparse (<100)'
    WHEN page_text_len <  500 THEN '1 — sparse (100-500)'
    WHEN page_text_len < 2000 THEN '2 — medium (500-2k)'
    ELSE                            '3 — dense (>=2k)'
  END AS density_bucket,
  COUNT(*) AS pages
FROM {config['silver_pages_table']}
GROUP BY 1
ORDER BY 1
"""))

# COMMAND ----------

# Sanity check the page_variant structure on one row.
display(spark.sql(f"""
SELECT
  path,
  page_id,
  size(variant_get(page_variant, '$.document.pages',    'array<variant>')) AS pages_in_variant,
  size(variant_get(page_variant, '$.document.elements', 'array<variant>')) AS elements_in_variant,
  page_variant:document:pages[0]:id::int                                   AS variant_page_id,
  page_variant:metadata:file_metadata:file_name::string                    AS file_name
FROM {config['silver_pages_table']}
ORDER BY path, page_id
LIMIT 5
"""))
