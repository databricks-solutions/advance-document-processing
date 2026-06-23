# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Extract Fields (per predicted class)
# MAGIC
# MAGIC Routes each classified page to a class-specific `ai_extract` schema. One
# MAGIC output column per content class, populated only on pages carrying that
# MAGIC `page_class`; `noise` pages are dropped first.
# MAGIC
# MAGIC Extraction input is `page_text` (the chrome-stripped, figure/table-inlined
# MAGIC text view built in NB02). NB02 also produced `page_variant` — a page-scoped
# MAGIC VARIANT mirroring the `ai_parse_document` schema. If your workspace's
# MAGIC `ai_extract` build accepts the parse VARIANT directly, set
# MAGIC `EXTRACT_INPUT = "page_variant"` to give the extractor table/checkbox/bbox
# MAGIC structure. We default to `page_text` because it is the broadly-supported
# MAGIC string input.
# MAGIC
# MAGIC Config is read from `config.yaml` (written by NB01).
# MAGIC
# MAGIC - **Input:** silver pages classified table
# MAGIC - **Output:** silver pages extracted table
# MAGIC
# MAGIC Requires `ai_extract` **2.1** (DBR 18.2+ / serverless env v3+) for
# MAGIC citations + confidence scores.

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# Column fed to ai_extract. Switch to "page_variant" if your ai_extract build
# accepts the parse VARIANT directly (see header note).
EXTRACT_INPUT = "page_text"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-class extraction schemas + instructions
# MAGIC
# MAGIC One entry per non-`noise` page class. The key MUST match a `page_class`
# MAGIC label emitted by NB03. Add `credit_report` / `title_insurance` here (plus
# MAGIC the matching label in NB03) to extend coverage.

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

print(f"{len(EXTRACT_SCHEMAS)} per-class extraction schemas defined: {list(EXTRACT_SCHEMAS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Route each page to its class schema
# MAGIC
# MAGIC `CASE WHEN page_class = '<label>'` runs `ai_extract` only on pages of that
# MAGIC class; every other class column is NULL on the row.

# COMMAND ----------

def ai_extract_expr(label: str) -> str:
    """SQL expression: run ai_extract with this label's schema only when the
    page carries `label`, else NULL."""
    schema_json, instructions = EXTRACT_SCHEMAS[label]
    schema_sql = schema_json.replace("'", "''")
    instr_sql = instructions.replace("'", "''")
    return f"""
        CASE WHEN page_class = '{label}' THEN
            ai_extract(
                {EXTRACT_INPUT},
                '{schema_sql}',
                map(
                    'version',                '2.1',
                    'enableCitations',        'true',
                    'enableConfidenceScores', 'true',
                    'instructions',           '{instr_sql}'
                )
            )
        END AS {label}_extracted
    """


select_clauses = [
    "path",
    "page_id",
    "page_class",
] + [ai_extract_expr(lbl) for lbl in EXTRACT_SCHEMAS]

extract_sql = f"""
CREATE OR REPLACE TABLE {config['extracted_table']} AS
SELECT
    {','.join(select_clauses)}
FROM {config['classified_table']}
WHERE page_class <> 'noise'
"""

spark.sql(extract_sql)
display(spark.table(config["extracted_table"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

# Page counts per class actually routed to an extractor.
display(spark.sql(f"""
SELECT page_class, COUNT(*) AS pages
FROM {config['extracted_table']}
GROUP BY page_class
ORDER BY pages DESC
"""))

# COMMAND ----------

# Spot check one extracted struct per class.
display(spark.sql(f"""
SELECT page_class, path, page_id,
       loan_application_extracted,
       income_verification_extracted,
       bank_statement_extracted,
       tax_return_extracted,
       property_appraisal_extracted,
       closing_disclosure_extracted
FROM (
  SELECT *, row_number() OVER (PARTITION BY page_class ORDER BY page_id) AS rn
  FROM {config['extracted_table']}
) t
WHERE rn = 1
ORDER BY page_class
"""))
