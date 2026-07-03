# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Silver: extract fields (per predicted class)
# MAGIC
# MAGIC Streaming version of NB04. Append-only stream over the classified table
# MAGIC (excluding `noise`); routes each page to its class-specific `ai_extract`
# MAGIC schema. Pure per-row map — stateless, no `foreachBatch`.
# MAGIC
# MAGIC Requires `ai_extract` **2.1** (env v3+ / DBR 18.2+) for citations +
# MAGIC confidence scores. Output columns are VARIANT (see NB05 for the shape).

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

classified_table = f"{catalog}.{schema}.{table_prefix}_silver_pages_classified"
extracted_table  = f"{catalog}.{schema}.{table_prefix}_silver_pages_extracted"
artifacts_root   = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt             = f"{artifacts_root}/checkpoints/extract_fields"

# Column fed to ai_extract. Switch to "page_variant" if your ai_extract build
# accepts the parse VARIANT directly.
EXTRACT_INPUT = "page_text"

print(f"classified_table = {classified_table}")
print(f"extracted_table  = {extracted_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-class extraction schemas + instructions

# COMMAND ----------

EXTRACT_SCHEMAS = {
    "loan_application": (
        """{
            "borrower_name":    {"type": "string"},
            "co_borrower_name": {"type": "string"},
            "borrower_ssn":     {"type": "string", "description": "Borrower SSN; often masked/redacted"},
            "loan_amount":      {"type": "number", "description": "Requested loan amount in USD; strip $ and commas"},
            "loan_purpose":     {"type": "enum",   "labels": ["Purchase", "Refinance", "Cash-out Refinance", "Construction", "Other"]},
            "loan_type":        {"type": "enum",   "labels": ["Conventional", "FHA", "VA", "USDA", "Other"]},
            "occupancy":        {"type": "enum",   "labels": ["Primary Residence", "Second Home", "Investment Property"]},
            "property_address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city":   {"type": "string"},
                    "state":  {"type": "string", "description": "Two-letter US state code"},
                    "zip":    {"type": "string"}
                }
            },
            "property_value":          {"type": "number"},
            "interest_rate":           {"type": "number",  "description": "Note rate as a percent, e.g. 6.5 for 6.5%"},
            "loan_term_months":        {"type": "integer"},
            "borrower_employer":       {"type": "string"},
            "borrower_monthly_income": {"type": "number", "description": "Total borrower monthly income in USD"}
        }""",
        "Extract Uniform Residential Loan Application (Form 1003 / URLA) fields. "
        "Amounts are USD — strip $ and commas. Express rates as a number. "
        "Do not infer values that are not present on the page.",
    ),

    "income_verification": (
        """{
            "document_subtype":  {"type": "enum",    "labels": ["Pay Stub", "W-2", "1099", "Verification of Employment", "Award Letter", "Other"]},
            "employer_name":     {"type": "string"},
            "employee_name":     {"type": "string"},
            "pay_period_start":  {"type": "string",  "description": "MM/DD/YYYY"},
            "pay_period_end":    {"type": "string",  "description": "MM/DD/YYYY"},
            "gross_pay_current": {"type": "number",  "description": "Gross pay for the current pay period (pay stubs)"},
            "net_pay_current":   {"type": "number"},
            "ytd_gross_pay":     {"type": "number",  "description": "Year-to-date gross pay"},
            "annual_salary":     {"type": "number"},
            "tax_year":          {"type": "integer", "description": "Tax year for W-2 / 1099 forms"}
        }""",
        "Extract income-evidence fields. Set document_subtype from the form. "
        "Pay-period fields apply to pay stubs; tax_year applies to W-2/1099. "
        "Amounts are USD — strip $ and commas. A field absent on this document "
        "should be NULL, not guessed.",
    ),

    "bank_statement": (
        """{
            "institution_name":      {"type": "string"},
            "account_holder_name":   {"type": "string"},
            "account_number_masked": {"type": "string", "description": "Masked/last-4 account number as printed"},
            "account_type":          {"type": "enum",   "labels": ["Checking", "Savings", "Money Market", "Brokerage", "Other"]},
            "statement_period_start":{"type": "string", "description": "MM/DD/YYYY"},
            "statement_period_end":  {"type": "string", "description": "MM/DD/YYYY"},
            "beginning_balance":     {"type": "number"},
            "ending_balance":        {"type": "number"},
            "total_deposits":        {"type": "number"},
            "total_withdrawals":     {"type": "number"},
            "average_daily_balance": {"type": "number"}
        }""",
        "Extract depository/asset statement summary fields. Amounts are USD — "
        "strip $ and commas. Keep the account number masked exactly as printed.",
    ),

    "tax_return": (
        """{
            "form_type":             {"type": "string",  "description": "e.g. '1040', '1040-SR'"},
            "tax_year":              {"type": "integer"},
            "filer_name":            {"type": "string"},
            "filing_status":         {"type": "enum",    "labels": ["Single", "Married Filing Jointly", "Married Filing Separately", "Head of Household", "Qualifying Surviving Spouse"]},
            "adjusted_gross_income": {"type": "number"},
            "total_income":          {"type": "number"},
            "taxable_income":        {"type": "number"},
            "total_tax":             {"type": "number"},
            "wages_salaries_tips":   {"type": "number"}
        }""",
        "Extract filed annual income-tax-return fields (IRS Form 1040 family). "
        "Amounts are USD — strip $ and commas. This is an annual return, not a "
        "single pay period.",
    ),

    "property_appraisal": (
        """{
            "appraised_value": {"type": "number"},
            "subject_property_address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city":   {"type": "string"},
                    "state":  {"type": "string", "description": "Two-letter US state code"},
                    "zip":    {"type": "string"}
                }
            },
            "property_type":            {"type": "enum",    "labels": ["Single Family", "Condo", "PUD", "Multi-Family", "Manufactured", "Other"]},
            "gross_living_area_sqft":   {"type": "number"},
            "year_built":               {"type": "integer"},
            "bedrooms":                 {"type": "integer"},
            "bathrooms":                {"type": "number",  "description": "e.g. 2.5"},
            "lot_size":                 {"type": "string",  "description": "Verbatim, e.g. '0.25 acres' or '10,890 sqft'"},
            "condition_rating":         {"type": "string",  "description": "URAR condition rating, e.g. 'C3'"},
            "appraisal_effective_date": {"type": "string",  "description": "MM/DD/YYYY"},
            "appraiser_name":           {"type": "string"},
            "comparable_count":         {"type": "integer", "description": "Number of comparable sales in the grid"}
        }""",
        "Extract Uniform Residential Appraisal Report (Form 1004) fields about "
        "the SUBJECT property only — not the comparables. Amounts are USD. Keep "
        "lot_size and condition_rating as printed.",
    ),

    "closing_disclosure": (
        """{
            "document_subtype":               {"type": "enum",    "labels": ["Loan Estimate", "Closing Disclosure"]},
            "loan_id":                         {"type": "string"},
            "loan_amount":                     {"type": "number"},
            "interest_rate":                   {"type": "number",  "description": "Note rate as a percent"},
            "monthly_principal_interest":      {"type": "number"},
            "loan_term_years":                 {"type": "integer"},
            "product":                         {"type": "string",  "description": "e.g. 'Fixed Rate', '5/1 Adjustable Rate'"},
            "apr":                             {"type": "number",  "description": "Annual Percentage Rate as a percent"},
            "total_closing_costs":             {"type": "number"},
            "cash_to_close":                   {"type": "number"},
            "estimated_total_monthly_payment": {"type": "number"}
        }""",
        "Extract TRID Loan Estimate / Closing Disclosure terms. Set "
        "document_subtype from the form title. Amounts are USD — strip $ and "
        "commas. Express rate and APR as numbers.",
    ),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure target table exists
# MAGIC
# MAGIC One VARIANT column per class. `mergeSchema` lets the stream populate them.

# COMMAND ----------

extracted_cols = ",\n    ".join(f"{lbl}_extracted VARIANT" for lbl in EXTRACT_SCHEMAS)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {extracted_table} (
    path STRING,
    page_id INT,
    page_class STRING,
    {extracted_cols}
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Route each page to its class schema (append stream)

# COMMAND ----------

from pyspark.sql.functions import expr

# SQL builder lives in sql_builders.py (co-located, unit-tested); driver-only.
from sql_builders import ai_extract_expr

extract_stream = spark.readStream.table(classified_table).where("page_class <> 'noise'")
for label in EXTRACT_SCHEMAS:
    schema_json, instructions = EXTRACT_SCHEMAS[label]
    extract_stream = extract_stream.withColumn(
        f"{label}_extracted",
        expr(ai_extract_expr(label, schema_json, instructions, EXTRACT_INPUT)),
    )

extract_stream = extract_stream.select(
    "path", "page_id", "page_class",
    *[f"{lbl}_extracted" for lbl in EXTRACT_SCHEMAS],
)

query = (
    extract_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", ckpt)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(extracted_table)
)

query.awaitTermination()
print("Extract write complete.")
print(f"Last progress: {query.lastProgress}")

cnt = spark.table(extracted_table).count()
print(f"extracted row count = {cnt}")
dbutils.notebook.exit(f"extracted={cnt}")
