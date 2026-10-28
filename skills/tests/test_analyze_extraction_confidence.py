from confidence_profile import (
    LEAF_ABSENT,
    LEAF_SCORED,
    LEAF_UNSCORED,
    classify_extract_leaf,
    flatten_extract_leaves,
    level_cutoff,
    score_levels,
)

# Shape observed live from ai_extract v2.1 (enableConfidenceScores) on a nested schema:
# confidence lives only on leaves; objects and arrays carry no score of their own;
# absent fields are {"value": None, "confidence_score": None}.
RESPONSE = {
    "employee_name": {"value": "Jordan Rivera", "confidence_score": 0.95},
    "ets_date": {"value": None, "confidence_score": None},
    "hire_date": {"value": "14-Mar-2021", "confidence_score": None},
    "gross": {
        "current_amount": {"value": "1884", "confidence_score": 0.98},
        "year_to_date_amount": {"value": "43332", "confidence_score": 0.95},
    },
    "deduction_item": [
        {"description": {"value": "401k", "confidence_score": 0.95},
         "period_amount": {"value": "142.1", "confidence_score": 0.1}},
        {"description": {"value": "Medical", "confidence_score": 0.95},
         "period_amount": {"value": "88.0", "confidence_score": 0.95}},
    ],
}


def _by_path(leaves):
    return {leaf["path"]: leaf for leaf in leaves}


def test_flatten_reaches_object_and_array_leaves():
    leaves = _by_path(flatten_extract_leaves(RESPONSE))
    assert set(leaves) == {
        "employee_name", "ets_date", "hire_date",
        "gross.current_amount", "gross.year_to_date_amount",
        "deduction_item[0].description", "deduction_item[0].period_amount",
        "deduction_item[1].description", "deduction_item[1].period_amount",
    }
    assert leaves["deduction_item[1].period_amount"]["field"] == "deduction_item[].period_amount"
    assert leaves["gross.current_amount"]["confidence"] == 0.98


def test_flatten_classifies_absent_unscored_scored():
    leaves = _by_path(flatten_extract_leaves(RESPONSE))
    assert leaves["ets_date"]["status"] == LEAF_ABSENT
    assert leaves["hire_date"]["status"] == LEAF_UNSCORED
    assert leaves["employee_name"]["status"] == LEAF_SCORED


def test_flatten_handles_scalar_arrays_and_empty():
    assert flatten_extract_leaves(None) == []
    leaves = flatten_extract_leaves({"tags": [{"value": "a", "confidence_score": 0.9}]})
    assert leaves == [{"path": "tags[0]", "field": "tags[]", "value": "a",
                       "confidence": 0.9, "status": LEAF_SCORED}]


def test_score_levels():
    assert score_levels([0.95, 0.98, 0.95, None, 0.1]) == {0.95: 2, 0.98: 1, 0.1: 1}


def test_level_cutoff_lands_on_observed_levels():
    # level counts observed live: paystubs 0.90 x4, 0.95 x100, 0.98 x97
    paystub = [0.90] * 4 + [0.95] * 100 + [0.98] * 97
    assert level_cutoff(paystub, 0.05) == 0.95
    # loan docs: rare low outliers below a 0.95/0.98/1.0 plateau
    loan = [0.1, 0.75, 0.85, 0.9, 0.9, 0.93, 0.93, 0.93] + [0.95] * 111 + [0.98] * 179 + [0.99] * 9 + [1.0] * 30
    assert level_cutoff(loan, 0.05) == 0.95
    assert sum(c < level_cutoff(loan, 0.05) for c in loan) == 8
    # one dominant level and nothing below it -> nothing can be "below"
    assert level_cutoff([0.98] * 50, 0.05) == 0.98
    assert level_cutoff([None], 0.05) is None


def test_classify_extract_leaf():
    assert classify_extract_leaf(LEAF_ABSENT, None, 0.95) == "absent"
    assert classify_extract_leaf(LEAF_UNSCORED, None, 0.95) == "review:unscored"
    assert classify_extract_leaf(LEAF_SCORED, 0.1, 0.95) == "review:below_corpus_pct"
    assert classify_extract_leaf(LEAF_SCORED, 0.95, 0.95) == "ok"   # ties at the cutoff are ok
    assert classify_extract_leaf(LEAF_SCORED, 0.98, 0.95) == "ok"
    assert classify_extract_leaf(LEAF_SCORED, 0.9, None) == "review:no_baseline"


def test_flatten_keeps_leaf_whose_value_is_a_plain_list_or_object():
    # A leaf may wrap a structured value with one score; it must not vanish.
    leaves = flatten_extract_leaves({
        "tags": {"value": ["a", "b"], "confidence_score": 0.9},
        "meta": {"value": {"k": 1}, "confidence_score": 0.8},
    })
    assert leaves == [
        {"path": "tags", "field": "tags", "value": '["a", "b"]', "confidence": 0.9, "status": LEAF_SCORED},
        {"path": "meta", "field": "meta", "value": '{"k": 1}', "confidence": 0.8, "status": LEAF_SCORED},
    ]


def test_flatten_still_recurses_when_wrapped_value_holds_leaves():
    leaves = flatten_extract_leaves({
        "items": {"value": [{"name": {"value": "x", "confidence_score": 0.95}}], "confidence_score": 0.9},
    })
    assert [(l["path"], l["confidence"]) for l in leaves] == [("items[0].name", 0.95)]
