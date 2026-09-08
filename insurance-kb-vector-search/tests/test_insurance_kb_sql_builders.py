"""Unit tests for the insurance-KB SQL/config builders (bundle/src/insurance_kb_sql_builders.py).

Pure string / dict construction — assert generated SQL and maps, not Spark."""

import json

import insurance_kb_sql_builders as S


# --- label set + domain map ------------------------------------------------

def test_five_labels_defined():
    assert set(S.DOC_TYPE_LABELS) == {
        "policy_document", "endorsement", "underwriting_guideline",
        "fnol_claim_form", "adjuster_report",
    }


def test_label_set_matches_domain_map_keys():
    assert set(S.DOC_TYPE_LABELS) == set(S.DOC_TYPE_TO_DOMAIN)


def test_domains_partition_the_labels_exactly():
    # every label maps to a valid domain; both domains are non-empty; no others
    assert set(S.DOC_TYPE_TO_DOMAIN.values()) == {"reference", "claims"}
    ref = {k for k, v in S.DOC_TYPE_TO_DOMAIN.items() if v == "reference"}
    clm = {k for k, v in S.DOC_TYPE_TO_DOMAIN.items() if v == "claims"}
    assert ref == {"policy_document", "endorsement", "underwriting_guideline"}
    assert clm == {"fnol_claim_form", "adjuster_report"}


def test_labels_json_is_valid_json_with_descriptions():
    obj = json.loads(S.labels_json())
    assert set(obj) == set(S.DOC_TYPE_LABELS)
    assert all(isinstance(v, str) and v for v in obj.values())


def test_domain_for_known_and_unknown():
    assert S.domain_for("policy_document") == "reference"
    assert S.domain_for("adjuster_report") == "claims"
    assert S.domain_for("not_a_type") == "unknown"


# --- classify_expr ---------------------------------------------------------

def test_classify_expr_uses_input_col_and_instructions():
    sql = S.classify_expr("parsed", '{"a":"b"}', "Insurance docs.")
    assert "ai_classify(" in sql
    assert "parsed," in sql
    assert "'instructions'" in sql
    assert "Insurance docs." in sql


def test_classify_expr_escapes_single_quotes():
    sql = S.classify_expr("parsed", "{'k':'v'}", "don't guess")
    assert "{''k'':''v''}" in sql
    assert "don''t guess" in sql


# --- domain_case_expr ------------------------------------------------------

def test_domain_case_expr_maps_both_domains():
    sql = S.domain_case_expr("dt")
    assert "CASE" in sql and "END" in sql
    assert "'policy_document'" in sql and "'underwriting_guideline'" in sql
    assert "THEN 'reference'" in sql
    assert "'fnol_claim_form'" in sql and "THEN 'claims'" in sql
    assert "ELSE 'unknown'" in sql


# --- doc_text_expr ---------------------------------------------------------

def test_doc_text_expr_default_args():
    sql = S.doc_text_expr()
    assert sql.startswith("substr(")
    assert "array_join(" in sql
    # transform over the elements array (no [*] wildcard — variant_get rejects it)
    assert "transform(" in sql
    assert ":document:elements" in sql
    assert "e:content" in sql
    assert "[*]" not in sql
    assert "1, 12000" in sql


def test_doc_text_expr_default_parsed_col():
    sql = S.doc_text_expr()
    # default parsed_col is "parsed"
    assert "parsed:document:elements" in sql


def test_doc_text_expr_custom_col():
    sql = S.doc_text_expr(parsed_col="my_col")
    assert "my_col:document:elements" in sql
    assert "1, 12000" in sql


def test_doc_text_expr_custom_max_chars():
    sql = S.doc_text_expr(max_chars=5000)
    assert "1, 5000" in sql
    assert "12000" not in sql


def test_doc_text_expr_custom_col_and_max_chars():
    sql = S.doc_text_expr(parsed_col="raw", max_chars=8192)
    assert "raw:document:elements" in sql
    assert "1, 8192" in sql
    assert "12000" not in sql


# --- name builders ---------------------------------------------------------

def test_gold_table_name():
    assert S.gold_table_name("insurance_kb", "reference") == "insurance_kb_reference_chunks"
    assert S.gold_table_name("insurance_kb", "claims") == "insurance_kb_claims_chunks"


def test_index_name_fully_qualified():
    assert S.index_name("fins_genai", "unstructured_documents", "insurance_kb", "claims") == \
        "fins_genai.unstructured_documents.insurance_kb_claims_index"
