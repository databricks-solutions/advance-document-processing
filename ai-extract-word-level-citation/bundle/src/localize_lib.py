"""Pure, unit-testable helpers for the word-level-citation localize stage.

Extracted verbatim from ``bundle/src/03_localize.py`` so the coordinate math and
OCR string-matching can be tested without Spark / dbutils / pytesseract / PIL.

Everything here is a pure function (no side effects, no I/O), which is why it is
import-safe on the driver, on executors (03_localize registers this module for
cloudpickle pickle-by-value so the pandas UDFs serialize these functions into
their closures — serverless is Spark Connect and has no ``sparkContext`` /
``addPyFile``), and locally under pytest. The impure OCR/crop helpers
(``crop_element``, ``ocr_tokens``) deliberately stay in the notebook.
"""

import json
import re
from html import unescape

from rapidfuzz import fuzz

_NUMERIC_TOKEN = re.compile(r"[$(+-]?\d[\d,]*(?:\.\d+)?\)?")


def _loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def normalize_box(coord):
    if not coord or len(coord) < 4:
        return None
    x0, x1 = sorted((float(coord[0]), float(coord[2])))
    y0, y1 = sorted((float(coord[1]), float(coord[3])))
    return [x0, y0, x1, y1]


def first_coord(citation):
    for bbox in citation.get("bbox", []) or []:
        coord = normalize_box(bbox.get("coord"))
        if coord:
            return coord, bbox.get("page_id")
    return None, None


def box_area(box):
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a, b):
    if not a or not b:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def overlap_score(a, b):
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    iou = inter / union if union else 0.0
    coverage = inter / min(box_area(a), box_area(b))
    return max(iou, coverage)


def element_box_on_page(element, page_id):
    for bbox in element.get("bbox", []) or []:
        if bbox.get("page_id") is None or int(bbox.get("page_id")) != int(page_id):
            continue
        coord = normalize_box(bbox.get("coord"))
        if coord:
            return coord
    return None


def text_from_element(element):
    if not element:
        return ""
    raw = element.get("content") or element.get("description") or element.get("text") or ""
    text = unescape(str(raw))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def element_for_citation(citation, parsed_elements):
    coord, page_id = first_coord(citation)
    if coord is None or page_id is None:
        return None, "", 0.0

    best = (None, "", 0.0)
    for element in parsed_elements or []:
        elem_coord = element_box_on_page(element, page_id)
        score = overlap_score(coord, elem_coord)
        if score > best[2]:
            best = (element, text_from_element(element), score)
    return best


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def norm_num(value):
    raw = str(value)
    negative = raw.strip().startswith("(") and raw.strip().endswith(")")
    digits = re.sub(r"[^0-9.]", "", raw)
    try:
        number = float(digits)
        return f"{-number if negative else number:.2f}"
    except ValueError:
        return ""


def union_box(boxes):
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def match_tokens(tokens, target, kind):
    if not tokens:
        return None, 0.0, ""
    target_norm = norm_num(target) if kind == "numeric" else norm(target)
    if not target_norm:
        return None, 0.0, ""

    best = (None, 0.0, "")
    target_words = re.findall(r"\S+", str(target))
    max_window = min(len(tokens), max(6, min(20, len(target_words) + 4)))
    for i in range(len(tokens)):
        for width in range(1, max_window + 1):
            run = tokens[i:i + width]
            if not run:
                continue
            joined = "".join(token["text"] for token in run)
            candidate = norm_num(joined) if kind == "numeric" else norm(joined)
            if not candidate:
                continue
            score = 100.0 if kind == "numeric" and candidate == target_norm else fuzz.ratio(candidate, target_norm)
            if score > best[1]:
                best = (union_box([token["box"] for token in run]), score, " ".join(token["text"] for token in run))
    return best


def to_page_coords(local_box, origin):
    cx0, cy0 = origin
    return [
        local_box[0] + cx0,
        local_box[1] + cy0,
        local_box[2] + cx0,
        local_box[3] + cy0,
    ]


def dedupe_keep_order(values):
    seen = set()
    out = []
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def numeric_targets_from_element_text(element_text, value):
    target = norm_num(value)
    if not target:
        return []

    candidates = []
    for match in _NUMERIC_TOKEN.finditer(element_text or ""):
        raw = match.group(0)
        if norm_num(raw) == target:
            candidates.append(raw)
    candidates.append(str(value))
    return dedupe_keep_order(candidates)


def text_targets_from_element_text(element_text, value):
    target = norm(value)
    if not target:
        return []

    candidates = []
    direct = re.search(re.escape(str(value)), element_text or "", flags=re.IGNORECASE)
    if direct:
        candidates.append((100.0, element_text[direct.start():direct.end()]))

    words = re.findall(r"\S+", element_text or "")
    value_words = re.findall(r"\S+", str(value))
    if words:
        min_window = max(1, len(value_words) - 2)
        max_window = min(len(words), max(3, min(20, len(value_words) + 4)))
        for i in range(len(words)):
            for width in range(min_window, max_window + 1):
                run = words[i:i + width]
                if len(run) != width:
                    continue
                text = " ".join(run)
                score = fuzz.ratio(norm(text), target)
                if score >= 75:
                    candidates.append((score, text))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return dedupe_keep_order([text for _, text in candidates] + [str(value)])


def match_targets(value, kind, element_text):
    if kind == "numeric":
        return numeric_targets_from_element_text(element_text, value)
    return text_targets_from_element_text(element_text, value)


def match_best_target(tokens, targets, kind):
    best = (None, 0.0, "", "")
    for target in targets:
        local_box, score, matched = match_tokens(tokens, target, kind)
        if score > best[1]:
            best = (local_box, score, matched, target)
    return best
