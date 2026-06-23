# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Gold: flat tables (foreachBatch + MERGE)
# MAGIC
# MAGIC Streaming version of NB05. The only stage that aggregates across rows, so
# MAGIC it uses `foreachBatch`: each micro-batch is an ordinary batch DataFrame,
# MAGIC the per-loan `GROUP BY` runs in batch scope (no state store / watermark),
# MAGIC and `MERGE` makes re-processing idempotent. Effectively stateless — the
# MAGIC document is the atomic unit, so a loan's pages never split across batches.
# MAGIC
# MAGIC Three CDF-enabled gold tables, keyed by `path`:
# MAGIC
# MAGIC - `*_gold_loan_files`     — one row per loan (singleton fields + counts)
# MAGIC - `*_gold_income_docs`    — one row per income document
# MAGIC - `*_gold_bank_statements`— one row per statement
# MAGIC
# MAGIC `ai_extract` v2.1 returns VARIANT; values are read with
# MAGIC `variant_get(col, '$.response.<field>.value', '<TYPE>')`.

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

extracted_table = f"{catalog}.{schema}.{table_prefix}_silver_pages_extracted"
gold_loans      = f"{catalog}.{schema}.{table_prefix}_gold_loan_files"
gold_income     = f"{catalog}.{schema}.{table_prefix}_gold_income_docs"
gold_bank       = f"{catalog}.{schema}.{table_prefix}_gold_bank_statements"
artifacts_root  = f"/Volumes/{catalog}/{schema}/{volume}/_streaming/{table_prefix}"
ckpt            = f"{artifacts_root}/checkpoints/gold_merge"

RESPONSE_ROOT = "response"
VALUE_LEAF = "value"

print(f"extracted_table = {extracted_table}")
print(f"gold_loans      = {gold_loans}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Field map (field_path, sql_type)

# COMMAND ----------

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
        ("loan_purpose", "STRING"), ("loan_type", "STRING"), ("occupancy", "STRING"),
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

REPEATING = {
    "income_verification": gold_income,
    "bank_statement":      gold_bank,
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
# MAGIC ## Helpers + ensure target tables exist (CDF on)

# COMMAND ----------

def value_expr(cls, field, sql_type):
    return f"variant_get({cls}_extracted, '$.{RESPONSE_ROOT}.{field}.{VALUE_LEAF}', '{sql_type}')"


def gold_name(prefix, field):
    return f"{prefix}_{field.replace('.', '_')}"


# gold_loan_files columns: (name, type)
loan_cols = [
    (gold_name(SINGLETON_PREFIX[cls], fld), typ)
    for cls, fields in SINGLETON_FIELDS.items() for (fld, typ) in fields
] + [("income_doc_count", "BIGINT"), ("bank_doc_count", "BIGINT")]

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {gold_loans} (
    path STRING,
    {", ".join(f"{n} {t}" for n, t in loan_cols)}
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

for cls, table in REPEATING.items():
    cols = ", ".join(f"{fld.replace('.', '_')} {typ}" for fld, typ in REPEATING_FIELDS[cls])
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        path STRING,
        page_id INT,
        {cols}
    ) USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## foreachBatch: recompute affected loans/docs → MERGE

# COMMAND ----------

def build_merge(target, source_view, key_cols, all_cols):
    on = " AND ".join(f"t.{c} = s.{c}" for c in key_cols)
    set_clause = ", ".join(f"t.{c} = s.{c}" for c in all_cols if c not in key_cols)
    insert_cols = ", ".join(all_cols)
    insert_vals = ", ".join(f"s.{c}" for c in all_cols)
    return f"""
        MERGE INTO {target} t
        USING {source_view} s ON {on}
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """


# Pre-build the recompute SELECTs (filtered to affected paths at run time).
singleton_select = [
    f"MAX({value_expr(cls, fld, typ)}) AS {gold_name(SINGLETON_PREFIX[cls], fld)}"
    for cls, fields in SINGLETON_FIELDS.items() for (fld, typ) in fields
]
count_select = [
    f"COUNT_IF(page_class = '{cls}') AS {cls.split('_')[0]}_doc_count"
    for cls in REPEATING
]
loan_all_cols = ["path"] + [n for n, _ in loan_cols]


def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    batch_df.select("path").distinct().createOrReplaceTempView("affected_paths")

    # --- gold_loan_files: recompute affected loans, MERGE on path
    loans_recomputed = spark.sql(f"""
        SELECT path,
          {", ".join(singleton_select + count_select)}
        FROM {extracted_table}
        WHERE path IN (SELECT path FROM affected_paths)
        GROUP BY path
    """)
    loans_recomputed.createOrReplaceTempView("loans_recomputed")
    spark.sql(build_merge(gold_loans, "loans_recomputed", ["path"], loan_all_cols))

    # --- companion tables: recompute affected per-document rows, MERGE on (path, page_id)
    for cls, table in REPEATING.items():
        cols = [f"{value_expr(cls, fld, typ)} AS {fld.replace('.', '_')}"
                for fld, typ in REPEATING_FIELDS[cls]]
        recomputed = spark.sql(f"""
            SELECT path, page_id,
              {", ".join(cols)}
            FROM {extracted_table}
            WHERE page_class = '{cls}' AND path IN (SELECT path FROM affected_paths)
        """)
        view = f"{cls}_recomputed"
        recomputed.createOrReplaceTempView(view)
        all_cols = ["path", "page_id"] + [fld.replace('.', '_') for fld, _ in REPEATING_FIELDS[cls]]
        spark.sql(build_merge(table, view, ["path", "page_id"], all_cols))


query = (
    spark.readStream.table(extracted_table)
    .writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", ckpt)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()
print("Gold merge complete.")
print(f"Last progress: {query.lastProgress}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"""
SELECT 'gold_loan_files'      AS tbl, COUNT(*) AS rows FROM {gold_loans}
UNION ALL
SELECT 'gold_income_docs'     AS tbl, COUNT(*) AS rows FROM {gold_income}
UNION ALL
SELECT 'gold_bank_statements' AS tbl, COUNT(*) AS rows FROM {gold_bank}
"""))
