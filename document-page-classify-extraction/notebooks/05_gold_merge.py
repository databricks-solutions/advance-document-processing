# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Gold Merge (flat tables)
# MAGIC
# MAGIC Turns the per-page extractions from NB04 into a flat gold layer of **three
# MAGIC CDF-enabled tables**, keyed by `path` (the loan file):
# MAGIC
# MAGIC | Table | Grain | Holds |
# MAGIC |---|---|---|
# MAGIC | `*_gold_loan_files` | one row per loan | singleton-document fields (loan application, tax return, appraisal, closing disclosure) as flat columns, prefixed by class, plus `income_doc_count` / `bank_doc_count` |
# MAGIC | `*_gold_income_docs` | one row per income document | every `income_verification` field, flat (no prefix) |
# MAGIC | `*_gold_bank_statements` | one row per statement | every `bank_statement` field, flat (no prefix) |
# MAGIC
# MAGIC Singleton fields are coalesced across a loan's pages with `MAX(...)` (the
# MAGIC value lands on one page; `MAX` ignores the NULLs from other pages/classes).
# MAGIC Repeating documents are **preserved in full** in their own tables — join
# MAGIC back to the loans table on `path`.
# MAGIC
# MAGIC Config is read from `config.yaml` (written by NB01).
# MAGIC
# MAGIC - **Input:** silver pages extracted table
# MAGIC - **Output:** gold loans / income / bank tables
# MAGIC
# MAGIC > **`ai_extract` v2.1 output shape (confirmed).** Each `*_extracted` column
# MAGIC > is a **VARIANT**, not a STRUCT, so fields are read with `variant_get`,
# MAGIC > not dot access. With citations + confidence enabled, the value of each
# MAGIC > field lives at `$.response.<field>.value` (nested objects wrap at the
# MAGIC > leaf, e.g. `$.response.property_address.street.value`). Confidence is at
# MAGIC > `$.response.<field>.confidence_score` if you want to carry it too.

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# JSON path prefix/suffix around each field inside the extracted VARIANT.
# Confirmed for ai_extract v2.1 with enableCitations + enableConfidenceScores.
RESPONSE_ROOT = "response"   # e.g. "$.response.<field>.value"
VALUE_LEAF = "value"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the extracted VARIANT shape (verify field paths)
# MAGIC
# MAGIC `printSchema` shows the columns are VARIANT; `to_json` reveals the
# MAGIC internal structure the field paths below rely on.

# COMMAND ----------

spark.table(config["extracted_table"]).printSchema()

display(spark.sql(f"""
SELECT page_class, to_json(loan_application_extracted) AS sample_json
FROM {config['extracted_table']}
WHERE page_class = 'loan_application' AND loan_application_extracted IS NOT NULL
LIMIT 1
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Field map
# MAGIC
# MAGIC Declarative spec of every gold column as `(field_path, sql_type)`. Edit
# MAGIC here to add/remove fields or classes — the SQL is generated from this map.
# MAGIC `field_path` uses dots for nested objects (e.g. `property_address.street`).

# COMMAND ----------

# Singleton classes -> output column prefix in the gold_loan_files table.
SINGLETON_PREFIX = {
    "loan_application":   "loan_application",
    "tax_return":         "tax",
    "property_appraisal": "appraisal",
    "closing_disclosure": "closing_disclosure",
}

SINGLETON_FIELDS = {
    "loan_application": [
        ("borrower_name", "STRING"), ("co_borrower_name", "STRING"),
        ("borrower_ssn", "STRING"), ("loan_amount", "DOUBLE"),
        ("loan_purpose", "STRING"), ("loan_type", "STRING"),
        ("occupancy", "STRING"),
        ("property_address.street", "STRING"), ("property_address.city", "STRING"),
        ("property_address.state", "STRING"), ("property_address.zip", "STRING"),
        ("property_value", "DOUBLE"), ("interest_rate", "DOUBLE"),
        ("loan_term_months", "INT"), ("borrower_employer", "STRING"),
        ("borrower_monthly_income", "DOUBLE"),
    ],
    "tax_return": [
        ("form_type", "STRING"), ("tax_year", "INT"), ("filer_name", "STRING"),
        ("filing_status", "STRING"), ("adjusted_gross_income", "DOUBLE"),
        ("total_income", "DOUBLE"), ("taxable_income", "DOUBLE"),
        ("total_tax", "DOUBLE"), ("wages_salaries_tips", "DOUBLE"),
    ],
    "property_appraisal": [
        ("appraised_value", "DOUBLE"),
        ("subject_property_address.street", "STRING"),
        ("subject_property_address.city", "STRING"),
        ("subject_property_address.state", "STRING"),
        ("subject_property_address.zip", "STRING"),
        ("property_type", "STRING"), ("gross_living_area_sqft", "DOUBLE"),
        ("year_built", "INT"), ("bedrooms", "INT"), ("bathrooms", "DOUBLE"),
        ("lot_size", "STRING"), ("condition_rating", "STRING"),
        ("appraisal_effective_date", "STRING"), ("appraiser_name", "STRING"),
        ("comparable_count", "INT"),
    ],
    "closing_disclosure": [
        ("document_subtype", "STRING"), ("loan_id", "STRING"),
        ("loan_amount", "DOUBLE"), ("interest_rate", "DOUBLE"),
        ("monthly_principal_interest", "DOUBLE"), ("loan_term_years", "INT"),
        ("product", "STRING"), ("apr", "DOUBLE"),
        ("total_closing_costs", "DOUBLE"), ("cash_to_close", "DOUBLE"),
        ("estimated_total_monthly_payment", "DOUBLE"),
    ],
}

# Repeating classes -> their own flat table (one row per document, no prefix).
REPEATING = {
    "income_verification": config["gold_income_table"],
    "bank_statement":      config["gold_bank_table"],
}

REPEATING_FIELDS = {
    "income_verification": [
        ("document_subtype", "STRING"), ("employer_name", "STRING"),
        ("employee_name", "STRING"), ("pay_period_start", "STRING"),
        ("pay_period_end", "STRING"), ("gross_pay_current", "DOUBLE"),
        ("net_pay_current", "DOUBLE"), ("ytd_gross_pay", "DOUBLE"),
        ("annual_salary", "DOUBLE"), ("tax_year", "INT"),
    ],
    "bank_statement": [
        ("institution_name", "STRING"), ("account_holder_name", "STRING"),
        ("account_number_masked", "STRING"), ("account_type", "STRING"),
        ("statement_period_start", "STRING"), ("statement_period_end", "STRING"),
        ("beginning_balance", "DOUBLE"), ("ending_balance", "DOUBLE"),
        ("total_deposits", "DOUBLE"), ("total_withdrawals", "DOUBLE"),
        ("average_daily_balance", "DOUBLE"),
    ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate and run the SQL

# COMMAND ----------

EXTRACTED_TABLE = config["extracted_table"]


def value_expr(cls: str, field: str, sql_type: str) -> str:
    """Typed read of a field's value out of this class's extracted VARIANT."""
    json_path = f"$.{RESPONSE_ROOT}.{field}.{VALUE_LEAF}"
    return f"variant_get({cls}_extracted, '{json_path}', '{sql_type}')"


def gold_name(prefix: str, field: str) -> str:
    return f"{prefix}_{field.replace('.', '_')}"


def cdf_create(table: str) -> str:
    return f"CREATE OR REPLACE TABLE {table}\nTBLPROPERTIES (delta.enableChangeDataFeed = true) AS"


# --- gold_loan_files: one row per loan, singleton fields coalesced ----------
singleton_aggs = [
    f"MAX({value_expr(cls, fld, typ)}) AS {gold_name(SINGLETON_PREFIX[cls], fld)}"
    for cls, fields in SINGLETON_FIELDS.items() for (fld, typ) in fields
]
count_aggs = [
    f"COUNT_IF(page_class = '{cls}') AS {cls.split('_')[0]}_doc_count"
    for cls in REPEATING
]
loans_sql = (
    f"{cdf_create(config['gold_loans_table'])}\n"
    f"SELECT path,\n  "
    + ",\n  ".join(singleton_aggs + count_aggs)
    + f"\nFROM {EXTRACTED_TABLE}\nGROUP BY path"
)

# --- companion tables: one row per repeating document -----------------------
def companion_sql(cls: str, table: str) -> str:
    cols = [
        f"{value_expr(cls, fld, typ)} AS {fld.replace('.', '_')}"
        for (fld, typ) in REPEATING_FIELDS[cls]
    ]
    return (
        f"{cdf_create(table)}\n"
        f"SELECT path, page_id,\n  "
        + ",\n  ".join(cols)
        + f"\nFROM {EXTRACTED_TABLE}\nWHERE page_class = '{cls}'"
    )

print("=== gold_loan_files ===")
print(loans_sql)
spark.sql(loans_sql)

for cls, table in REPEATING.items():
    sql = companion_sql(cls, table)
    print(f"\n=== {table} ===")
    print(sql)
    spark.sql(sql)

# COMMAND ----------

display(spark.table(config["gold_loans_table"]))

# COMMAND ----------

display(spark.table(config["gold_income_table"]))

# COMMAND ----------

display(spark.table(config["gold_bank_table"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

# Row counts per gold table.
display(spark.sql(f"""
SELECT 'gold_loan_files'      AS tbl, COUNT(*) AS rows FROM {config['gold_loans_table']}
UNION ALL
SELECT 'gold_income_docs'     AS tbl, COUNT(*) AS rows FROM {config['gold_income_table']}
UNION ALL
SELECT 'gold_bank_statements' AS tbl, COUNT(*) AS rows FROM {config['gold_bank_table']}
"""))

# COMMAND ----------

# Loan-level view with its repeating docs rolled up — confirms the join key
# works and counts line up with the companion tables.
display(spark.sql(f"""
SELECT
  l.path,
  l.loan_application_borrower_name,
  l.loan_application_loan_amount,
  l.closing_disclosure_apr,
  l.appraisal_appraised_value,
  l.income_doc_count,
  COUNT(DISTINCT i.page_id) AS income_rows,
  l.bank_doc_count,
  COUNT(DISTINCT b.page_id) AS bank_rows
FROM {config['gold_loans_table']} l
LEFT JOIN {config['gold_income_table']} i USING (path)
LEFT JOIN {config['gold_bank_table']}   b USING (path)
GROUP BY ALL
ORDER BY l.path
"""))
