# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Parse & Aggregate Multi-Page PDF with `ai_parse_document`
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads individual PDF page files from a UC Volume
# MAGIC 2. Parses each page with `ai_parse_document` (v2.0)
# MAGIC 3. Aggregates all parsed pages into a **single VARIANT row** matching the `ai_parse_document` output schema — with corrected `page_number` references
# MAGIC
# MAGIC **Source:** `/Volumes/fins_genai/unstructured_documents/pdf_examples/pdf_pages/` (15 single-page PDFs from one large document)

# COMMAND ----------

# DBTITLE 1,Step 1: Parse each PDF page with ai_parse_document
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW parsed_pages AS
# MAGIC SELECT
# MAGIC   _metadata.file_name,
# MAGIC   CAST(regexp_extract(_metadata.file_name, 'page_(\\d+)\\.pdf', 1) AS INT) AS page_num,
# MAGIC   ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/fins_genai/unstructured_documents/pdf_examples/pdf_pages/',
# MAGIC   format => 'binaryFile'
# MAGIC );
# MAGIC
# MAGIC -- Verify: show parse status per page
# MAGIC SELECT
# MAGIC   file_name,
# MAGIC   page_num,
# MAGIC   is_variant_null(parsed:error_status) AS parse_ok,
# MAGIC   size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS num_elements,
# MAGIC   size(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS num_pages
# MAGIC FROM parsed_pages
# MAGIC ORDER BY page_num;

# COMMAND ----------

# DBTITLE 1,Step 2: Aggregate parsed pages into a single document VARIANT
import json

# Collect all parsed results, ordered by page number
rows = spark.sql("""
  SELECT
    page_num,
    to_json(parsed) AS parsed_json
  FROM parsed_pages
  WHERE is_variant_null(parsed:error_status)
  ORDER BY page_num
""").collect()

assert len(rows) > 0, "No pages parsed successfully — check Step 1 for errors."

# Merge all pages into a single ai_parse_document-shaped VARIANT
all_pages = []
all_elements = []
first_metadata = None

for row in rows:
    parsed = json.loads(row.parsed_json)
    actual_page_num = row.page_num

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

# DBTITLE 1,Step 3: Verify the merged VARIANT output
# MAGIC %sql
# MAGIC -- Single-row summary
# MAGIC SELECT
# MAGIC   size(try_cast(parsed:document:pages AS ARRAY<VARIANT>))    AS total_pages,
# MAGIC   size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS total_elements,
# MAGIC   is_variant_null(parsed:error_status)                       AS no_errors,
# MAGIC   parsed:metadata                                            AS metadata
# MAGIC FROM merged_document;

# COMMAND ----------

# DBTITLE 1,Step 4 (optional): Preview elements by page
# MAGIC %sql
# MAGIC -- Explode elements to inspect content across all pages
# MAGIC SELECT
# MAGIC   el:page_number::INT   AS page_number,
# MAGIC   el:type::STRING        AS element_type,
# MAGIC   LEFT(el:content::STRING, 200) AS content_preview
# MAGIC FROM merged_document
# MAGIC LATERAL VIEW explode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) t AS el
# MAGIC ORDER BY page_number, el:element_index::INT
# MAGIC LIMIT 50;

# COMMAND ----------

# DBTITLE 1,Step 5 (optional): Save merged document to a UC table
# MAGIC %sql
# MAGIC -- Uncomment to persist the merged VARIANT as a table
# MAGIC -- CREATE OR REPLACE TABLE fins_genai.unstructured_documents.parsed_full_document AS
# MAGIC -- SELECT parsed FROM merged_document;