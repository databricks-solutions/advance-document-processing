"""Pure, unit-testable SQL-string builders for the classify-extract pipeline.

Lifted verbatim from ``bundle/src/05_gold_merge.py`` (``value_expr``, ``gold_name``,
``build_merge``) and ``bundle/src/04_extract_fields.py`` (``ai_extract_expr``).

These run on the **driver** — they assemble SQL text that ``spark.sql`` / ``expr``
then executes — so the notebooks import this module directly (no executor
shipping). Keeping them here lets the SQL construction be tested without Spark.

``ai_extract_expr`` takes its schema / instructions / input column as arguments
(the notebook still owns the domain-specific ``EXTRACT_SCHEMAS`` config) so the
builder itself stays pure and testable.
"""

RESPONSE_ROOT = "response"
VALUE_LEAF = "value"


def value_expr(cls, field, sql_type, response_root=RESPONSE_ROOT, value_leaf=VALUE_LEAF):
    return f"variant_get({cls}_extracted, '$.{response_root}.{field}.{value_leaf}', '{sql_type}')"


def gold_name(prefix, field):
    return f"{prefix}_{field.replace('.', '_')}"


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


def ai_extract_expr(label, schema_json, instructions, extract_input="page_text"):
    schema_sql = schema_json.replace("'", "''")
    instr_sql = instructions.replace("'", "''")
    return f"""
        CASE WHEN page_class = '{label}' THEN
            ai_extract(
                {extract_input},
                '{schema_sql}',
                map(
                    'version',                '2.1',
                    'enableCitations',        'true',
                    'enableConfidenceScores', 'true',
                    'instructions',           '{instr_sql}'
                )
            )
        END
    """
