# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Parse & Aggregate Multi-Page PDF — Infer Page Order from Content
# MAGIC
# MAGIC This notebook handles the case where **filenames do not encode page numbers**. After parsing each single-page PDF with `ai_parse_document`, page order is inferred from the **printed page numbers** extracted by the parser (element type `page_number`).
# MAGIC
# MAGIC Fallback chain:
# MAGIC 1. **`page_number` element** — the printed page number extracted by `ai_parse_document`
# MAGIC 2. **`ai_extract` fallback** — asks the LLM to read the page number from the content when no `page_number` element exists
# MAGIC 3. **Row number fallback** — assigns order by filename sort as a last resort
# MAGIC
# MAGIC **Source:** `/Volumes/fins_genai/unstructured_documents/pdf_examples/pdf_pages/`

# COMMAND ----------

# DBTITLE 1,Step 1: Parse each PDF page with ai_parse_document
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW parsed_pages AS
# MAGIC SELECT
# MAGIC   _metadata.file_name,
# MAGIC   ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/fins_genai/unstructured_documents/pdf_examples/pdf_pages/',
# MAGIC   format => 'binaryFile'
# MAGIC );
# MAGIC
# MAGIC SELECT
# MAGIC   file_name,
# MAGIC   is_variant_null(parsed:error_status) AS parse_ok,
# MAGIC   size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS num_elements
# MAGIC FROM parsed_pages
# MAGIC ORDER BY file_name;

# COMMAND ----------

# DBTITLE 1,Step 2: Infer page number from parsed content
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW pages_with_inferred_order AS
# MAGIC WITH page_number_elements AS (
# MAGIC   -- Extract the printed page number from 'page_number' type elements
# MAGIC   SELECT
# MAGIC     file_name,
# MAGIC     parsed,
# MAGIC     try_cast(
# MAGIC       -- Get the first page_number element's content
# MAGIC       try_element_at(
# MAGIC         filter(
# MAGIC           try_cast(parsed:document:elements AS ARRAY<VARIANT>),
# MAGIC           el -> el:type::STRING = 'page_number'
# MAGIC         ),
# MAGIC         1
# MAGIC       ):content::STRING
# MAGIC     AS INT) AS inferred_page_num
# MAGIC   FROM parsed_pages
# MAGIC   WHERE is_variant_null(parsed:error_status)
# MAGIC ),
# MAGIC ai_fallback AS (
# MAGIC   -- For pages with no page_number element, ask ai_extract to read it.
# MAGIC   -- ai_extract v2.1 nests scalars at $.response.<field>.value.
# MAGIC   SELECT
# MAGIC     file_name,
# MAGIC     parsed,
# MAGIC     inferred_page_num,
# MAGIC     CASE
# MAGIC       WHEN inferred_page_num IS NOT NULL THEN inferred_page_num
# MAGIC       ELSE try_cast(
# MAGIC         ai_extract(
# MAGIC           parsed,
# MAGIC           '{"page_number": {"type": "integer", "description": "The printed page number shown on this page, e.g. in the header or footer"}}',
# MAGIC           MAP('version', '2.1')
# MAGIC         ):response.page_number.value::STRING
# MAGIC       AS INT)
# MAGIC     END AS page_num_with_fallback
# MAGIC   FROM page_number_elements
# MAGIC )
# MAGIC SELECT
# MAGIC   file_name,
# MAGIC   inferred_page_num       AS from_page_element,
# MAGIC   page_num_with_fallback  AS from_ai_fallback,
# MAGIC   -- Assign a contiguous 1..N order: pages with a known page number come
# MAGIC   -- first (in that order), then any unresolved pages appended in filename
# MAGIC   -- order. ROW_NUMBER over that sort key avoids the duplicate-page_num
# MAGIC   -- collisions a COALESCE(page_num, ROW_NUMBER()) fallback would produce.
# MAGIC   ROW_NUMBER() OVER (
# MAGIC     ORDER BY COALESCE(page_num_with_fallback, 2147483647), file_name
# MAGIC   ) AS final_page_num,
# MAGIC   parsed
# MAGIC FROM ai_fallback;

# COMMAND ----------

# DBTITLE 1,Step 3: Aggregate parsed pages into a single document VARIANT
import json

# Collect all parsed results, ordered by inferred page number
rows = spark.sql("""
  SELECT
    final_page_num,
    to_json(parsed) AS parsed_json
  FROM pages_with_inferred_order
  ORDER BY final_page_num
""").collect()

assert len(rows) > 0, "No pages parsed successfully — check Step 1 for errors."

# Merge all pages into a single ai_parse_document-shaped VARIANT
all_pages = []
all_elements = []
first_metadata = None

for row in rows:
    parsed = json.loads(row.parsed_json)
    actual_page_num = row.final_page_num

    if first_metadata is None:
        first_metadata = parsed.get("metadata", {})

    # Collect pages with corrected page_number
    for page in parsed.get("document", {}).get("pages", []):
        page["page_number"] = actual_page_num
        all_pages.append(page)

    # Collect elements with corrected page_number
    for element in parsed.get("document", {}).get("elements", []):
        element["page_number"] = actual_page_num
        all_elements.append(element)

# Reconstruct the canonical ai_parse_document output shape
merged_document = {
    "document": {
        "pages": all_pages,
        "elements": all_elements,
    },
    "metadata": first_metadata,
    "error_status": None,
}

merged_json = json.dumps(merged_document)

# Materialise as a single-row DataFrame with a VARIANT column
merged_df = (
    spark.createDataFrame([(merged_json,)], ["json_str"])
    .selectExpr("parse_json(json_str) AS parsed")
)
merged_df.createOrReplaceTempView("merged_document")

print(f"Merged {len(rows)} pages into single document")
print(f"  Total pages:    {len(all_pages)}")
print(f"  Total elements: {len(all_elements)}")

# COMMAND ----------

# DBTITLE 1,Step 4: Verify the merged VARIANT output
# MAGIC %sql
# MAGIC SELECT
# MAGIC   size(try_cast(parsed:document:pages AS ARRAY<VARIANT>))    AS total_pages,
# MAGIC   size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS total_elements,
# MAGIC   is_variant_null(parsed:error_status)                       AS no_errors,
# MAGIC   parsed:metadata                                            AS metadata
# MAGIC FROM merged_document;

# COMMAND ----------

# DBTITLE 1,Step 5 (optional): Preview elements by page
# MAGIC %sql
# MAGIC SELECT
# MAGIC   el:page_number::INT    AS page_number,
# MAGIC   el:type::STRING        AS element_type,
# MAGIC   LEFT(el:content::STRING, 200) AS content_preview
# MAGIC FROM merged_document
# MAGIC LATERAL VIEW explode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) t AS el
# MAGIC ORDER BY page_number, el:element_index::INT
# MAGIC LIMIT 50;