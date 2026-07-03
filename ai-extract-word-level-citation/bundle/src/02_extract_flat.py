# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver: Extract Paystub Fields + Flatten Citations
# MAGIC
# MAGIC Runs `ai_extract` 2.1 with citations and confidence scores against the
# MAGIC parsed document VARIANT, then flattens the response into the shape consumed
# MAGIC by the localization task.
# MAGIC
# MAGIC Requires DBR 18.2+ / serverless env v3+ for `ai_extract` 2.1.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "paystubs")
dbutils.widgets.text("volume_subdir", "")
dbutils.widgets.text("table_prefix", "paystub_bbox_stream")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir").strip("/")
table_prefix = dbutils.widgets.get("table_prefix")

bronze_table = f"{catalog}.{schema}.{table_prefix}_bronze_parsed_docs"
extracted_table = f"{catalog}.{schema}.{table_prefix}_extracted"
flat_table = f"{catalog}.{schema}.{table_prefix}_extracted_flat"

print(f"bronze_table    = {bronze_table}")
print(f"extracted_table = {extracted_table}")
print(f"flat_table      = {flat_table}")

# COMMAND ----------

EXTRACT_INPUT = "parsed"

PAYSTUB_SCHEMA = """{
  "employee_name": {
    "type": "string",
    "description": "The employee's full name printed on the paystub."
  },
  "employer_name": {
    "type": "string",
    "description": "The employer / company legal name printed on the paystub."
  },
  "social_security_number": {
    "type": "string",
    "description": "The employee's Social Security Number (taxpayer identifier) as printed."
  },
  "taxable_marital_status": {
    "type": "string",
    "description": "The taxable marital status (e.g. Married | Unmarried | Separated | Single)."
  },
  "borrower_employee_address": {
    "type": "object",
    "description": "Employee address information.",
    "properties": {
      "address_line": {
        "type": "string",
        "description": "The street address line in the employee address block."
      },
      "city_name": {
        "type": "string",
        "description": "The city in the employee address block."
      },
      "state_code": {
        "type": "string",
        "description": "The state code in the employee address block."
      },
      "zip_code": {
        "type": "string",
        "description": "The ZIP / postal code in the employee address block."
      }
    }
  },
  "borrower_employer_address": {
    "type": "object",
    "description": "Employer address information.",
    "properties": {
      "address_line": {
        "type": "string",
        "description": "The street address line in the employer address block."
      },
      "city_name": {
        "type": "string",
        "description": "The city in the employer address block."
      },
      "state_code": {
        "type": "string",
        "description": "The state code in the employer address block."
      },
      "zip_code": {
        "type": "string",
        "description": "The ZIP / postal code in the employer address block."
      }
    }
  },
  "hire_date": {
    "type": "string",
    "description": "The employee hire date as printed."
  },
  "ets_date": {
    "type": "string",
    "description": "The ETS (term-of-service expiration) date as printed, if present."
  },
  "pay_date": {
    "type": "string",
    "description": "The paycheck date as printed."
  },
  "period_start_date": {
    "type": "string",
    "description": "The pay-period start date as printed."
  },
  "period_end_date": {
    "type": "string",
    "description": "The pay-period end date as printed."
  },
  "payment_frequency": {
    "type": "string",
    "description": "The payment frequency type (e.g. Weekly | Biweekly | Monthly)."
  },
  "income_source_type": {
    "type": "string",
    "description": "Whether the employee is paid Hourly or Salaried."
  },
  "gross": {
    "type": "object",
    "description": "Gross pay amounts.",
    "properties": {
      "current_amount": {
        "type": "number",
        "description": "The gross pay for the current pay period."
      },
      "year_to_date_amount": {
        "type": "number",
        "description": "The year-to-date gross pay."
      }
    }
  },
  "net_pay": {
    "type": "object",
    "description": "Net pay amounts.",
    "properties": {
      "current_amount": {
        "type": "number",
        "description": "The net pay for the current pay period."
      },
      "year_to_date_amount": {
        "type": "number",
        "description": "The year-to-date net pay."
      }
    }
  },
  "adjustments_gross_income": {
    "type": "object",
    "description": "Adjustments to gross income amounts.",
    "properties": {
      "current_amount": {
        "type": "number",
        "description": "The current-period adjustments gross income amount."
      },
      "year_to_date_amount": {
        "type": "number",
        "description": "The year-to-date adjustments gross income amount."
      }
    }
  },
  "income_item": {
    "type": "array",
    "description": "Repeating earning/income rows on the paystub; one object per printed row.",
    "items": {
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "The earning item description (e.g. Regular, Overtime)."
        },
        "period_hours": {
          "type": "number",
          "description": "The hours for this earning item this period."
        },
        "rate": {
          "type": "number",
          "description": "The pay rate for this earning item."
        },
        "current_amount": {
          "type": "number",
          "description": "The current-period amount for this earning item."
        },
        "year_to_date_amount": {
          "type": "number",
          "description": "The year-to-date amount for this earning item."
        }
      }
    }
  },
  "deduction_item": {
    "type": "array",
    "description": "Repeating deduction rows on the paystub; one object per printed row.",
    "items": {
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "The deduction item description (e.g. Federal Tax, Medicare)."
        },
        "period_amount": {
          "type": "number",
          "description": "The current-period amount for this deduction item."
        },
        "year_to_date_amount": {
          "type": "number",
          "description": "The year-to-date amount for this deduction item."
        }
      }
    }
  }
}"""

INSTRUCTIONS = (
    "The uploaded documents are US consumer paystubs (monthly or weekly). "
    "Extract the named fields. Use the exact text as printed; do not reformat "
    "dates or amounts. Leave a field blank if it is not present. income_item, "
    "deduction_item are repeating rows: emit one object per printed row, "
    "including zero-activity rows that are actually present."
)

FIELD_SPECS = [
    ("employee_name", "employee_name", "string"),
    ("employer_name", "employer_name", "string"),
    ("social_security_number", "social_security_number", "string"),
    ("taxable_marital_status", "taxable_marital_status", "string"),
    ("borrower_employee_address__address_line", "borrower_employee_address.address_line", "string"),
    ("borrower_employee_address__city_name", "borrower_employee_address.city_name", "string"),
    ("borrower_employee_address__state_code", "borrower_employee_address.state_code", "string"),
    ("borrower_employee_address__zip_code", "borrower_employee_address.zip_code", "string"),
    ("borrower_employer_address__address_line", "borrower_employer_address.address_line", "string"),
    ("borrower_employer_address__city_name", "borrower_employer_address.city_name", "string"),
    ("borrower_employer_address__state_code", "borrower_employer_address.state_code", "string"),
    ("borrower_employer_address__zip_code", "borrower_employer_address.zip_code", "string"),
    ("hire_date", "hire_date", "string"),
    ("ets_date", "ets_date", "string"),
    ("pay_date", "pay_date", "string"),
    ("period_start_date", "period_start_date", "string"),
    ("period_end_date", "period_end_date", "string"),
    ("payment_frequency", "payment_frequency", "string"),
    ("income_source_type", "income_source_type", "string"),
    ("gross__current_amount", "gross.current_amount", "double"),
    ("gross__year_to_date_amount", "gross.year_to_date_amount", "double"),
    ("net_pay__current_amount", "net_pay.current_amount", "double"),
    ("net_pay__year_to_date_amount", "net_pay.year_to_date_amount", "double"),
    ("adjustments_gross_income__current_amount", "adjustments_gross_income.current_amount", "double"),
    ("adjustments_gross_income__year_to_date_amount", "adjustments_gross_income.year_to_date_amount", "double"),
]

ARRAY_FIELDS = ["income_item", "deduction_item"]
RESPONSE_ROOT = "response"

# COMMAND ----------

schema_sql = PAYSTUB_SCHEMA.replace("'", "''")
instr_sql = INSTRUCTIONS.replace("'", "''")

spark.sql(f"""
CREATE OR REPLACE TABLE {extracted_table} AS
SELECT
  path,
  variant_get(parsed, '$.document.pages[0].image_uri', 'string') AS image_uri,
  variant_get(parsed, '$.document.elements', 'variant') AS parsed_elements,
  ai_extract(
    {EXTRACT_INPUT},
    '{schema_sql}',
    options => map(
      'version',                '2.1',
      'enableCitations',        'true',
      'enableConfidenceScores', 'true',
      'instructions',           '{instr_sql}'
    )
  ) AS paystub_extracted
FROM {bronze_table}
""")

extracted_count = spark.table(extracted_table).count()
print(f"Extracted rows: {extracted_count}")

# COMMAND ----------

select_clauses = [
    "path",
    "image_uri",
    "parsed_elements",
    "COALESCE(element_at(split(image_uri, '/'), -1), element_at(split(path, '/'), -1)) AS image_name",
]

for col_name, variant_path, sql_type in FIELD_SPECS:
    select_clauses.append(
        f"variant_get(paystub_extracted, '$.{RESPONSE_ROOT}.{variant_path}.value', '{sql_type}') AS {col_name}"
    )
    select_clauses.append(
        f"variant_get(paystub_extracted, '$.{RESPONSE_ROOT}.{variant_path}.confidence_score', 'double') AS {col_name}_extract_conf"
    )
    select_clauses.append(
        f"variant_get(paystub_extracted, '$.{RESPONSE_ROOT}.{variant_path}.citation_ids', 'array<int>') AS {col_name}_citation_ids"
    )

for array_field in ARRAY_FIELDS:
    select_clauses.append(
        f"variant_get(paystub_extracted, '$.{RESPONSE_ROOT}.{array_field}', 'variant') AS {array_field}"
    )

select_clauses.append("variant_get(paystub_extracted, '$.metadata.citations', 'variant') AS citations")
select_clauses.append("variant_get(paystub_extracted, '$.metadata.pages', 'variant') AS citation_pages")

flat_sql = f"""
CREATE OR REPLACE TABLE {flat_table} AS
SELECT
  {",".join(select_clauses)}
FROM {extracted_table}
"""
spark.sql(flat_sql)

flat_count = spark.table(flat_table).count()
print(f"Flat rows: {flat_count}")
dbutils.notebook.exit(f"extracted={extracted_count};flat={flat_count}")
