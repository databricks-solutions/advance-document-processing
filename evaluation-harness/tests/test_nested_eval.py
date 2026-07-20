"""Unit tests for the type-aware nested scorer (src/nested_eval.py).

Pure functions only — no Spark, no MLflow, no LLM judge. Locks down the
bug-prone parts: number exact-match, object mean, array/array-of-object soft F1,
absence handling, ai_extract wrapper stripping, and the summary rollup. These
mirror the inline scorer in notebooks/04_extract_eval_nested.py — keep in sync.
"""

import nested_eval as N

STR = {"type": "string"}
NUM = {"type": "number"}
OBJ = {"type": "object", "properties": {"current_amount": NUM, "year_to_date_amount": NUM}}
ARR_STR = {"type": "array", "items": STR}
ARR_OBJ = {"type": "array", "items": {"type": "object", "properties": {"description": STR, "current_amount": NUM}}}


def _score(node, pred, exp):
    """Score one field in isolation, returning (row_score, accumulator)."""
    acc = N.new_acc()
    ctx = {"doc_id": "d", "conf": {}, "judge": None}
    s = N.score_node(node, pred, exp, "f", acc, ctx)
    return s, acc


# --- wrapper stripping ------------------------------------------------------

def test_strip_wrappers_scalar_object_array():
    raw = {
        "employee_name": {"value": "Jordan", "confidence_score": 0.9, "citation_ids": [1]},
        "gross": {"current_amount": {"value": 1884, "confidence_score": 0.8, "citation_ids": [2]}},
        "income_item": [{"description": {"value": "Reg", "confidence_score": 0.7, "citation_ids": [3]}}],
    }
    assert N.strip_wrappers(raw) == {
        "employee_name": "Jordan",
        "gross": {"current_amount": 1884},
        "income_item": [{"description": "Reg"}],
    }


def test_walk_confidence_records_scalar_leaves_only():
    raw = {
        "employee_name": {"value": "Jordan", "confidence_score": 0.9, "citation_ids": [1]},
        "gross": {"current_amount": {"value": 1884, "confidence_score": 0.8, "citation_ids": [2]}},
    }
    out = {}
    N.walk_confidence(raw, "", out)
    assert out == {"employee_name": 0.9, "gross.current_amount": 0.8}


def test_norm_key_matches_path_to_basename():
    assert N.norm_key("dbfs:/Volumes/x/y/paystub_synth_1.pdf") == "paystubsynth1"
    assert N.norm_key("paystub_synth_1.pdf") == "paystubsynth1"


# --- scalar leaves ----------------------------------------------------------

def test_string_normalized_match():
    assert _score(STR, " Jordan Rivera ", "jordan rivera")[0] == 1.0
    assert _score(STR, "Jordan", "Marcus")[0] == 0.0


def test_number_exact_match_is_strict():
    assert _score(NUM, 1884.0, 1884)[0] == 1.0          # 1884.0 == 1884
    assert _score(NUM, 1884.01, 1884)[0] == 0.0         # off by a cent → fail
    assert _score(NUM, "1,884.00", 1884)[0] == 1.0      # formatted string parses


def test_absence_both_absent_is_correct_one_absent_is_wrong():
    assert _score(STR, None, None)[0] == 1.0
    assert _score(STR, "", None)[0] == 1.0
    assert _score(STR, "Jordan", None)[0] == 0.0        # hallucinated
    assert _score(STR, None, "Jordan")[0] == 0.0        # missed


# --- object -----------------------------------------------------------------

def test_object_score_is_mean_of_children():
    s, acc = _score(OBJ, {"current_amount": 100, "year_to_date_amount": 999},
                         {"current_amount": 100, "year_to_date_amount": 200})
    assert s == 0.5                                     # one child right, one wrong
    assert acc["object_score"]["f"] == [0.5]


def test_object_all_correct():
    s, _ = _score(OBJ, {"current_amount": 100, "year_to_date_amount": 200},
                       {"current_amount": 100, "year_to_date_amount": 200})
    assert s == 1.0


# --- array of scalars -------------------------------------------------------

def test_array_soft_f1_partial():
    # 1 exact match of 3 predicted / 2 expected → P=1/3, R=1/2, F1≈0.4
    s, _ = _score(ARR_STR, ["Python", "JavaScript", "R"], ["Python", "Java"])
    assert round(s, 3) == 0.4


def test_array_empty_cases():
    assert _score(ARR_STR, [], [])[0] == 1.0
    assert _score(ARR_STR, [], ["x"])[0] == 0.0
    assert _score(ARR_STR, ["x"], [])[0] == 0.0


# --- array of objects -------------------------------------------------------

def test_array_object_soft_f1_and_per_child():
    A = [{"description": "Regular", "current_amount": 100}, {"description": "Overtime", "current_amount": 50}]
    B = [{"description": "Regular", "current_amount": 100}, {"description": "Bonus", "current_amount": 50}]
    s, acc = _score(ARR_OBJ, A, B)
    assert round(s, 3) == 0.75                          # obj1 sim 1.0, obj2 sim 0.5
    assert round(acc["soft_f1"]["f.description"][0], 3) == 0.5   # 1 of 2 descriptions match
    assert acc["soft_f1"]["f.current_amount"][0] == 1.0          # both amounts match


def test_array_object_optimal_matching_beats_positional():
    # Reversed order must still match optimally (Bob↔Bob, Alice↔Alice).
    A = [{"description": "Bob", "current_amount": 2}, {"description": "Alice", "current_amount": 1}]
    B = [{"description": "Alice", "current_amount": 1}, {"description": "Bob", "current_amount": 2}]
    assert _score(ARR_OBJ, A, B)[0] == 1.0


# --- summary rollup ---------------------------------------------------------

def test_summarize_rolls_up_accuracy_object_and_softf1():
    acc = N.new_acc()
    ctx = {"doc_id": "d", "conf": {}, "judge": None}
    # two docs: string right then wrong → accuracy 0.5
    N.score_node(STR, "a", "a", "name", acc, ctx)
    N.score_node(STR, "x", "a", "name", acc, ctx)
    N.score_node(OBJ, {"current_amount": 1, "year_to_date_amount": 2},
                      {"current_amount": 1, "year_to_date_amount": 2}, "gross", acc, ctx)
    summary = N.summarize(acc)
    assert summary["name_accuracy"] == 0.5
    assert summary["gross_object_score_mean"] == 1.0
    assert summary["gross.current_amount_accuracy"] == 1.0
