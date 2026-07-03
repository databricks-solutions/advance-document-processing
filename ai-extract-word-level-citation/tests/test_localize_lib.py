"""Unit tests for the word-level-citation localize helpers (bundle/src/localize_lib.py).

Pure functions only — no Spark, no OCR, no PIL. The coordinate math and the
OCR/value string-matching are the bug-prone parts these tests lock down.
"""

import localize_lib as L


# --- normalization ----------------------------------------------------------

def test_norm_strips_non_alphanumeric_and_lowercases():
    assert L.norm("Acme Corp., Inc.") == "acmecorpinc"
    assert L.norm("  Hello-World_123 ") == "helloworld123"


def test_norm_num_currency_and_commas():
    assert L.norm_num("$3,450.00") == "3450.00"
    assert L.norm_num("3450") == "3450.00"


def test_norm_num_parentheses_are_negative():
    assert L.norm_num("(1,200.50)") == "-1200.50"


def test_norm_num_non_numeric_returns_empty():
    assert L.norm_num("N/A") == ""
    assert L.norm_num("") == ""


# --- geometry ---------------------------------------------------------------

def test_normalize_box_sorts_corners():
    assert L.normalize_box([30, 40, 10, 20]) == [10.0, 20.0, 30.0, 40.0]


def test_normalize_box_rejects_short_or_empty():
    assert L.normalize_box([1, 2, 3]) is None
    assert L.normalize_box(None) is None


def test_box_area():
    assert L.box_area([0, 0, 10, 5]) == 50.0
    assert L.box_area(None) == 0.0
    assert L.box_area([10, 10, 0, 0]) == 0.0  # degenerate → clamped to 0


def test_intersection_area_overlap_and_disjoint():
    assert L.intersection_area([0, 0, 10, 10], [5, 5, 15, 15]) == 25.0
    assert L.intersection_area([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_overlap_score_identical_boxes_is_one():
    box = [0, 0, 10, 10]
    assert L.overlap_score(box, box) == 1.0


def test_overlap_score_contained_box_uses_coverage():
    # small box fully inside large → coverage 1.0 dominates IoU
    assert L.overlap_score([0, 0, 100, 100], [10, 10, 20, 20]) == 1.0


def test_overlap_score_disjoint_is_zero():
    assert L.overlap_score([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_union_box():
    assert L.union_box([[0, 0, 5, 5], [3, 4, 10, 8]]) == [0, 0, 10, 8]


def test_to_page_coords_adds_origin():
    assert L.to_page_coords([1, 2, 3, 4], (100, 200)) == [101, 202, 103, 204]


# --- citation → element join ------------------------------------------------

def test_first_coord_returns_first_valid_normalized_box():
    citation = {"bbox": [{"coord": [30, 40, 10, 20], "page_id": 2}]}
    coord, page_id = L.first_coord(citation)
    assert coord == [10.0, 20.0, 30.0, 40.0]
    assert page_id == 2


def test_first_coord_empty():
    assert L.first_coord({"bbox": []}) == (None, None)


def test_element_box_on_page_matches_page_id():
    element = {"bbox": [
        {"coord": [0, 0, 5, 5], "page_id": 0},
        {"coord": [1, 1, 9, 9], "page_id": 1},
    ]}
    assert L.element_box_on_page(element, 1) == [1.0, 1.0, 9.0, 9.0]
    assert L.element_box_on_page(element, 3) is None


def test_text_from_element_strips_html_and_whitespace():
    element = {"content": "Gross   Pay: <b>$3,450.00</b>\n"}
    assert L.text_from_element(element) == "Gross Pay: $3,450.00"


def test_text_from_element_empty():
    assert L.text_from_element(None) == ""
    assert L.text_from_element({}) == ""


def test_element_for_citation_picks_best_overlap():
    citation = {"bbox": [{"coord": [10, 10, 20, 20], "page_id": 0}]}
    elements = [
        {"content": "far away", "bbox": [{"coord": [100, 100, 110, 110], "page_id": 0}]},
        {"content": "the match", "bbox": [{"coord": [10, 10, 20, 20], "page_id": 0}]},
        {"content": "wrong page", "bbox": [{"coord": [10, 10, 20, 20], "page_id": 1}]},
    ]
    element, text, score = L.element_for_citation(citation, elements)
    assert text == "the match"
    assert score == 1.0


def test_element_for_citation_no_coord_returns_empty():
    assert L.element_for_citation({"bbox": []}, [{"content": "x"}]) == (None, "", 0.0)


# --- target extraction + token matching -------------------------------------

def test_numeric_targets_finds_matching_tokens_from_text():
    text = "Net pay 2,610.55 and gross 3,450.00 this period"
    targets = L.numeric_targets_from_element_text(text, "3450.00")
    assert "3,450.00" in targets  # raw on-page form preserved
    assert "3450.00" in targets   # the value itself is always appended


def test_numeric_targets_empty_for_non_numeric_value():
    assert L.numeric_targets_from_element_text("some text", "N/A") == []


def test_text_targets_includes_direct_substring_hit():
    text = "Employer: Acme Corporation, LLC"
    targets = L.text_targets_from_element_text(text, "Acme Corporation")
    assert "Acme Corporation" in targets


def test_dedupe_keep_order():
    assert L.dedupe_keep_order(["a", "b", "a", " ", "b", "c"]) == ["a", "b", "c"]


def test_match_tokens_numeric_exact_scores_100():
    tokens = [
        {"text": "$3,450.00", "box": [0, 0, 10, 5]},
        {"text": "gross", "box": [20, 0, 30, 5]},
    ]
    box, score, matched = L.match_tokens(tokens, "3450.00", "numeric")
    assert score == 100.0
    assert box == [0, 0, 10, 5]


def test_match_tokens_multiword_text_unions_boxes():
    tokens = [
        {"text": "Acme", "box": [0, 0, 10, 5]},
        {"text": "Corporation", "box": [12, 0, 30, 5]},
    ]
    box, score, matched = L.match_tokens(tokens, "Acme Corporation", "text")
    assert score >= 85
    assert box == [0, 0, 30, 5]  # union of both token boxes


def test_match_tokens_empty_returns_zero():
    assert L.match_tokens([], "anything", "text") == (None, 0.0, "")


def test_match_best_target_picks_highest_scoring_target():
    tokens = [{"text": "$3,450.00", "box": [0, 0, 10, 5]}]
    box, score, matched, target = L.match_best_target(tokens, ["999.99", "3450.00"], "numeric")
    assert target == "3450.00"
    assert score == 100.0


def test_match_targets_dispatches_on_kind():
    assert L.match_targets("3450.00", "numeric", "gross 3,450.00") == \
        L.numeric_targets_from_element_text("gross 3,450.00", "3450.00")
    assert L.match_targets("Acme", "text", "Acme Corp") == \
        L.text_targets_from_element_text("Acme Corp", "Acme")
