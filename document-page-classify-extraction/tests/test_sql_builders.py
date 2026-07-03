"""Unit tests for the classify-extract SQL-string builders (bundle/src/sql_builders.py).

Pure string construction — assert the generated SQL, not Spark execution.
"""

import sql_builders as S


# --- value_expr -------------------------------------------------------------

def test_value_expr_builds_variant_get():
    assert S.value_expr("income_verification", "gross.amount", "DOUBLE") == (
        "variant_get(income_verification_extracted, "
        "'$.response.gross.amount.value', 'DOUBLE')"
    )


def test_value_expr_respects_root_overrides():
    out = S.value_expr("loan", "x", "STRING", response_root="resp", value_leaf="v")
    assert "'$.resp.x.v'" in out


# --- gold_name --------------------------------------------------------------

def test_gold_name_replaces_dots_with_underscores():
    assert S.gold_name("loan", "borrower.address.city") == "loan_borrower_address_city"


def test_gold_name_plain_field():
    assert S.gold_name("income", "employer") == "income_employer"


# --- build_merge ------------------------------------------------------------

def test_build_merge_single_key():
    sql = S.build_merge("cat.sch.gold", "src_view", ["path"], ["path", "a", "b"])
    assert "MERGE INTO cat.sch.gold t" in sql
    assert "USING src_view s ON t.path = s.path" in sql
    # key column excluded from SET, non-keys included
    assert "UPDATE SET t.a = s.a, t.b = s.b" in sql
    assert "t.path = s.path" in sql and "SET t.path" not in sql
    # INSERT lists all columns
    assert "INSERT (path, a, b) VALUES (s.path, s.a, s.b)" in sql


def test_build_merge_composite_key():
    sql = S.build_merge("t", "v", ["path", "page_id"], ["path", "page_id", "val"])
    assert "ON t.path = s.path AND t.page_id = s.page_id" in sql
    # both key cols excluded from SET → only val remains
    assert "UPDATE SET t.val = s.val" in sql
    assert "t.page_id = s.page_id" in sql  # present in ON
    assert "SET t.path" not in sql and "SET t.page_id" not in sql


# --- ai_extract_expr --------------------------------------------------------

def test_ai_extract_expr_gates_on_label_and_uses_input_col():
    sql = S.ai_extract_expr("bank_statement", '{"balance":"..."}', "Be precise.", "page_text")
    assert "CASE WHEN page_class = 'bank_statement' THEN" in sql
    assert "ai_extract(\n                page_text," in sql
    assert "'version',                '2.1'" in sql
    assert "'enableCitations',        'true'" in sql
    assert "'enableConfidenceScores', 'true'" in sql


def test_ai_extract_expr_escapes_single_quotes():
    # single quotes in label/schema/instructions must be doubled for SQL string literals
    sql = S.ai_extract_expr("won't_match", "{'k':'v'}", "don't guess", "page_text")
    assert "page_class = 'won''t_match'" in sql
    assert "{''k'':''v''}" in sql
    assert "don''t guess" in sql


def test_ai_extract_expr_default_input_column():
    sql = S.ai_extract_expr("x", "{}", "instr")
    assert "ai_extract(\n                page_text," in sql
