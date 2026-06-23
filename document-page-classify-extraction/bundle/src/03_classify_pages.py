# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver: classify pages
# MAGIC
# MAGIC Streaming version of NB03. Append-only stream over the silver pages table;
# MAGIC `ai_classify` assigns each page one of seven labels. Pure per-row map —
# MAGIC stateless, no `foreachBatch`.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "mortgage_loan_files")
dbutils.widgets.text("table_prefix", "loan_docs_stream")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
volume        = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix  = dbutils.widgets.get("table_prefix")

pages_table      = f"{catalog}.{schema}.{table_prefix}_silver_pages"
classified_table = f"{catalog}.{schema}.{table_prefix}_silver_pages_classified"
artifacts_root   = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt             = f"{artifacts_root}/checkpoints/classify_pages"

print(f"pages_table      = {pages_table}")
print(f"classified_table = {classified_table}")

# COMMAND ----------

LABELS_JSON = """
{
  "loan_application": "The Uniform Residential Loan Application (URLA), Fannie Mae Form 1003 / Freddie Mac Form 65, or an equivalent loan application. Structured borrower and co-borrower fields: name, SSN, employment, monthly income, assets and liabilities, the subject property, loan amount, loan purpose, loan type, and borrower declarations/signatures.",
  "income_verification": "Documents that evidence borrower income. Examples: pay stubs / earnings statements (employer, pay period, gross and net pay, year-to-date totals), IRS W-2 wage statements, 1099 forms, a Verification of Employment (VOE) form, or a benefits/award letter. Distinguished from tax_return: these are single-period employer or payer statements, not a filed annual return.",
  "bank_statement": "Depository or asset account statements used to verify funds: checking, savings, money-market, or brokerage statements showing the institution, account holder, masked account number, statement period, beginning/ending balances and transaction activity.",
  "tax_return": "A filed annual income tax return: IRS Form 1040 and its schedules (Schedule C, E, etc.), or a business return. Identified by a tax year, filing status, adjusted gross income, and total tax — an annual filing, not a single pay period.",
  "property_appraisal": "A real-estate appraisal of the subject property, typically the Uniform Residential Appraisal Report (URAR / Fannie Form 1004): appraised value, property characteristics (GLA square footage, year built, bed/bath count), a comparable-sales grid, condition rating, and the appraiser's name/license and effective date.",
  "closing_disclosure": "TRID disclosure forms: the Loan Estimate (LE) or the Closing Disclosure (CD). Standardized loan-terms tables: loan amount, interest rate, monthly principal & interest, APR, loan term, product, total closing costs, and cash to close.",
  "noise": "Pages with no extractable loan data: fax cover sheets with only sender/recipient fields, blank pages, document-separator sheets, signature-only pages, and standalone boilerplate disclosures or legal notices with no borrower-specific values."
}
"""

INSTRUCTIONS = (
    "These pages come from a residential mortgage loan-origination file — a "
    "single packet that bundles many distinct sub-documents. Classify each "
    "page by its dominant content. A page carrying a recognizable form (1003, "
    "W-2, appraisal grid, Closing Disclosure) takes that form's label even if a "
    "fax header or cover band is present. Use `income_verification` for "
    "single-period employer/payer statements (pay stubs, W-2s) and `tax_return` "
    "only for filed annual returns (1040). A page that is mostly empty, a fax "
    "cover, a separator, or a signature without other fields is `noise`."
)

# Escape single quotes for SQL string literals.
labels_sql = LABELS_JSON.strip().replace("'", "''")
instr_sql = INSTRUCTIONS.replace("'", "''")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target table exists

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {classified_table} (
    path STRING,
    page_id INT,
    page_variant VARIANT,
    page_text STRING,
    page_text_len INT,
    image_uri STRING,
    has_table BOOLEAN,
    has_figure BOOLEAN,
    element_count INT,
    classification_raw VARIANT,
    page_class STRING
) USING DELTA
""")

# COMMAND ----------

from pyspark.sql.functions import expr

classify_expr = f"""
ai_classify(
  page_text,
  '{labels_sql}',
  map('instructions', '{instr_sql}')
)
"""

classified_stream = (
    spark.readStream.table(pages_table)
    .withColumn("classification_raw", expr(classify_expr))
    .selectExpr(
        "path", "page_id", "page_variant", "page_text", "page_text_len",
        "image_uri", "has_table", "has_figure", "element_count",
        "classification_raw",
        "classification_raw:response[0]::string AS page_class",
    )
)

query = (
    classified_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", ckpt)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(classified_table)
)

query.awaitTermination()
print("Classify write complete.")
print(f"Last progress: {query.lastProgress}")

cnt = spark.table(classified_table).count()
print(f"classified row count = {cnt}")
dbutils.notebook.exit(f"classified={cnt}")
