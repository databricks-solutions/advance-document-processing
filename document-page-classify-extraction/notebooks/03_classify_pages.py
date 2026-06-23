# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Classify Pages
# MAGIC
# MAGIC Classifies each page of a mortgage loan file into one of seven labels
# MAGIC using `ai_classify`. Single-label: a page is almost always one document
# MAGIC type. Notebook 04 routes each label to its own `ai_extract` schema.
# MAGIC
# MAGIC - `loan_application`   — URLA / Fannie Form 1003 / Freddie Form 65
# MAGIC - `income_verification`— pay stubs, W-2s, 1099s, Verification of Employment
# MAGIC - `bank_statement`     — depository / asset account statements
# MAGIC - `tax_return`         — IRS Form 1040 and schedules
# MAGIC - `property_appraisal` — Uniform Residential Appraisal Report (Form 1004)
# MAGIC - `closing_disclosure` — TRID Loan Estimate (LE) / Closing Disclosure (CD)
# MAGIC - `noise`              — fax covers, blanks, separators, signature-only pages
# MAGIC
# MAGIC Extension points (add a label here + a schema in NB04): `credit_report`
# MAGIC (see the `credit-report-extraction` asset for a full tri-merge schema) and
# MAGIC `title_insurance` (title commitment / homeowners insurance dec page).
# MAGIC
# MAGIC Config is read from `config.yaml` (written by NB01).
# MAGIC
# MAGIC - **Input:** silver pages table
# MAGIC - **Output:** silver pages classified table

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

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

# COMMAND ----------

from pyspark.sql.functions import expr

# Escape single quotes for SQL string literals.
labels_sql = LABELS_JSON.strip().replace("'", "''")
instr_sql = INSTRUCTIONS.replace("'", "''")

classified = (
    spark.table(config["silver_pages_table"])
    .withColumn(
        "classification_raw",
        expr(
            f"""
            ai_classify(
              page_text,
              '{labels_sql}',
              map('instructions', '{instr_sql}')
            )
            """
        ),
    )
    .withColumn("page_class", expr("classification_raw:response[0]::string"))
)

(
    classified.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(config["classified_table"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"""
SELECT page_class, COUNT(*) AS pages
FROM {config['classified_table']}
GROUP BY page_class
ORDER BY pages DESC
"""))

# COMMAND ----------

# Spot check: are the very-sparse pages routing to 'noise' as expected?
display(spark.sql(f"""
SELECT
  page_class,
  CASE
    WHEN page_text_len <  100 THEN '0 — very sparse'
    WHEN page_text_len <  500 THEN '1 — sparse'
    WHEN page_text_len < 2000 THEN '2 — medium'
    ELSE                            '3 — dense'
  END AS density_bucket,
  COUNT(*) AS pages
FROM {config['classified_table']}
GROUP BY 1, 2
ORDER BY 1, 2
"""))

# COMMAND ----------

# Sample rows from each class to eyeball labelling quality.
display(spark.sql(f"""
SELECT page_class, path, page_id, left(page_text, 200) AS preview
FROM (
  SELECT *, row_number() OVER (PARTITION BY page_class ORDER BY page_text_len DESC) AS rn
  FROM {config['classified_table']}
) t
WHERE rn <= 2
ORDER BY page_class
"""))
