# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Localize + Visualize Second-Level BBoxes (Paystubs)
# MAGIC
# MAGIC POC notebook 2 of 2. Takes the element-level citation bboxes from NB01 and
# MAGIC pinpoints **where inside the cited element** each entity actually sits — the
# MAGIC specific word(s) of a paragraph or cell of a table — using **Tesseract OCR word
# MAGIC boxes + string matching** (the recommendation's approach: OCR provides the
# MAGIC geometry; no VLM emits coordinates).
# MAGIC
# MAGIC Per-entity flow:
# MAGIC
# MAGIC 1. Look up the field's citation → element-level bbox (page-pixel coords).
# MAGIC 2. Join that bbox back to the matching `ai_parse_document` element to recover
# MAGIC    source text/context. Bbox citations do not carry a raw snippet.
# MAGIC 3. **Crop** that element from the saved page image (reuses the
# MAGIC    `document-embedding-chart-analysis` crop logic, in-memory).
# MAGIC 4. **Tesseract OCR** the crop → tokens + crop-local boxes.
# MAGIC 5. Derive raw match candidates from the parsed element text, then
# MAGIC    **match** the best candidate to OCR token box(es); union if multi-word.
# MAGIC 6. **Transform** the crop-local box back to page coords (additive offset —
# MAGIC    OCR runs at the page image's own DPI, so no scaling).
# MAGIC 7. **Draw** the element box + the inner second-level box; score the result.
# MAGIC
# MAGIC Run NB01 first (it writes `config.yaml` and the `*_extracted_flat` table).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies
# MAGIC
# MAGIC Uses **pytesseract** (Tesseract OCR is pre-installed on the Databricks
# MAGIC runtime) and **rapidfuzz** for fuzzy string matching. Restarts Python after install.

# COMMAND ----------

# MAGIC %pip install -q pytesseract rapidfuzz
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
print(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pull entities to localize
# MAGIC
# MAGIC One entity = (document, field) that has a value and at least one citation.
# MAGIC We cast both citation VARIANTs to JSON strings and parse

# COMMAND ----------

# DBTITLE 1,Pull entities to localize
import json
from pyspark.sql import functions as F

# Scalar field column names — must match NB01 cell 14 FIELD_SPECS
EXTRACT_FIELDS = [
    "employee_name", "employer_name", "social_security_number",
    "taxable_marital_status",
    "borrower_employee_address__address_line", "borrower_employee_address__city_name",
    "borrower_employee_address__state_code", "borrower_employee_address__zip_code",
    "borrower_employer_address__address_line", "borrower_employer_address__city_name",
    "borrower_employer_address__state_code", "borrower_employer_address__zip_code",
    "hire_date", "ets_date", "pay_date", "period_start_date", "period_end_date",
    "payment_frequency", "income_source_type",
    "gross__current_amount", "gross__year_to_date_amount",
    "net_pay__current_amount", "net_pay__year_to_date_amount",
    "adjustments_gross_income__current_amount", "adjustments_gross_income__year_to_date_amount",
]
NUMBER_FIELDS = {
    "gross__current_amount", "gross__year_to_date_amount",
    "net_pay__current_amount", "net_pay__year_to_date_amount",
    "adjustments_gross_income__current_amount", "adjustments_gross_income__year_to_date_amount",
}

# Array fields stored as VARIANT in the flat table
ARRAY_FIELDS = ["income_item", "deduction_item"]
ARRAY_SUB_FIELDS = {
    "income_item": [
        ("description",         "string"),
        ("period_hours",        "double"),
        ("rate",                "double"),
        ("current_amount",      "double"),
        ("year_to_date_amount", "double"),
    ],
    "deduction_item": [
        ("description",         "string"),
        ("period_amount",       "double"),
        ("year_to_date_amount", "double"),
    ],
}

flat = spark.table(config["flat_table"])

rows = (
    flat.selectExpr(
        "image_name",
        "image_uri",
        "cast(parsed_elements AS string) AS parsed_elements_json",
        *[c for f in EXTRACT_FIELDS for c in (f, f"{f}_citation_ids")],
        # Array fields stored as VARIANT; citations live inside each element's sub-fields
        *[f"cast({af} AS string) AS {af}_json" for af in ARRAY_FIELDS],
        "cast(citations AS string)      AS citations_json",
        "cast(citation_pages AS string) AS citation_pages_json",
    )
    .collect()
)
print(f"{len(rows)} document rows in {config['flat_table']}")

# COMMAND ----------

# DBTITLE 1,Build entity work-list

def _loads(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


# Flatten into an entity work-list: one item per (document, field, citation_id).
# Tag each entity 'numeric' vs 'text' so we can sample across both localization cases.
entities = []
for r in rows:
    citations = _loads(r["citations_json"]) or []
    cit_by_id = {int(c.get("id", -1)): c for c in citations}
    cit_pages = _loads(r["citation_pages_json"]) or []
    page_uri_by_id = {int(p.get("id", -1)): p.get("image_uri") for p in cit_pages}
    parsed_elements = _loads(r["parsed_elements_json"]) or []

    # --- Scalar fields ---
    for f in EXTRACT_FIELDS:
        value = r[f]
        cids = r[f"{f}_citation_ids"]
        if value is None or not cids:
            continue
        for cid in cids:
            cit = cit_by_id.get(int(cid))
            if not cit:
                continue
            entities.append({
                "image_name": r["image_name"],
                "row_image_uri": r["image_uri"],
                "field": f,
                "value": value,
                "kind": "numeric" if f in NUMBER_FIELDS else "text",
                "citation_id": int(cid),
                "citation": cit,
                "page_uri_by_id": page_uri_by_id,
                "parsed_elements": parsed_elements,
            })

    # --- Array fields (income_item, deduction_item) ---
    # Each element's sub-fields have per-sub-field {value, confidence_score, citation_ids}
    for af in ARRAY_FIELDS:
        array_data = _loads(r[f"{af}_json"]) or []
        sub_fields = ARRAY_SUB_FIELDS[af]

        for idx, item in enumerate(array_data):
            if not isinstance(item, dict):
                continue
            for sub_name, sub_type in sub_fields:
                raw = item.get(sub_name)
                if raw is None:
                    continue

                # Sub-fields are structured: {value, confidence_score, citation_ids}
                if isinstance(raw, dict) and "value" in raw:
                    value = raw["value"]
                    sub_cids = raw.get("citation_ids") or []
                else:
                    # Fallback for unexpected flat values (no citations available)
                    value = raw
                    sub_cids = []

                if value is None or not sub_cids:
                    continue

                field_label = f"{af}[{idx}].{sub_name}"
                kind = "numeric" if sub_type == "double" else "text"

                for cid in sub_cids:
                    cit = cit_by_id.get(int(cid))
                    if not cit:
                        continue
                    entities.append({
                        "image_name": r["image_name"],
                        "row_image_uri": r["image_uri"],
                        "field": field_label,
                        "value": value,
                        "kind": kind,
                        "citation_id": int(cid),
                        "citation": cit,
                        "page_uri_by_id": page_uri_by_id,
                        "parsed_elements": parsed_elements,
                    })

n_num = sum(1 for e in entities if e["kind"] == "numeric")
n_arr = sum(1 for e in entities if "[" in e["field"])
print(f"{len(entities)} citable entities ({n_num} numeric / {len(entities) - n_num} text)")
print(f"  └─ {n_arr} from array fields, {len(entities) - n_arr} from scalar fields")

# COMMAND ----------

# DBTITLE 1,Sample entities for localization
# Process ALL entities (no sampling — full-scale localization run).
full_sample = entities

n_arr = sum(1 for e in full_sample if "[" in e["field"])
n_scalar_num = sum(1 for e in full_sample if "[" not in e["field"] and e["kind"] == "numeric")
n_scalar_txt = sum(1 for e in full_sample if "[" not in e["field"] and e["kind"] == "text")
print(f"Processing {len(full_sample)} entities: {n_arr} array, {n_scalar_num} scalar-numeric, {n_scalar_txt} scalar-text")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Crop + OCR + match helpers
# MAGIC
# MAGIC - **Crop:** `Image.open(uri).crop(bbox)`
# MAGIC   with `bbox=[x1,y1,x2,y2]` page-pixel coords — but in-memory for OCR.
# MAGIC - **OCR:** Tesseract returns word-level boxes in **crop-local** pixels as
# MAGIC   `[left, top, width, height]`; we convert to `[x0,y0,x1,y1]`.
# MAGIC - **Element join:** citation bbox → parsed element by page + bbox overlap.
# MAGIC - **Match target:** derive raw candidates from parsed element text. For
# MAGIC   numeric fields, prefer the on-page number formatting that normalizes to the
# MAGIC   extracted value; for text, use the closest element-text window.
# MAGIC - **OCR match:** normalization-tolerant. For numbers we compare the numeric
# MAGIC   value; for text we slide a window over OCR tokens and take the best fuzzy
# MAGIC   match.

# COMMAND ----------

# DBTITLE 1,Cell 10
import re
from html import unescape

from PIL import Image
from rapidfuzz import fuzz
import pytesseract


def normalize_box(coord):
    """Databricks bbox coord -> [x0,y0,x1,y1] with sorted corners."""
    if not coord or len(coord) < 4:
        return None
    x0, x1 = sorted((float(coord[0]), float(coord[2])))
    y0, y1 = sorted((float(coord[1]), float(coord[3])))
    return [x0, y0, x1, y1]


def first_coord(citation):
    """First (element-level) bbox of a citation → ([x1,y1,x2,y2], page_id)."""
    for b in citation.get("bbox", []) or []:
        coord = normalize_box(b.get("coord"))
        if coord:
            return coord, b.get("page_id")
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
    """Score matching boxes by IoU plus containment tolerance."""
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    iou = inter / union if union else 0.0
    coverage = inter / min(box_area(a), box_area(b))
    return max(iou, coverage)


def element_box_on_page(element, page_id):
    for b in element.get("bbox", []) or []:
        if b.get("page_id") is None or int(b.get("page_id")) != int(page_id):
            continue
        coord = normalize_box(b.get("coord"))
        if coord:
            return coord
    return None


def text_from_element(element):
    """Parsed element content as readable text for matching."""
    if not element:
        return ""
    raw = (
        element.get("content")
        or element.get("description")
        or element.get("text")
        or ""
    )
    text = unescape(str(raw))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def element_for_citation(citation, parsed_elements):
    """Best parsed element for the citation bbox, using page_id + bbox overlap."""
    coord, page_id = first_coord(citation)
    if coord is None or page_id is None:
        return None, "", 0.0

    best = (None, "", 0.0)
    for element in parsed_elements or []:
        elem_coord = element_box_on_page(element, page_id)
        score = overlap_score(coord, elem_coord)
        if score <= best[2]:
            continue
        best = (element, text_from_element(element), score)
    return best


def crop_element(image_uri, coord):
    """In-memory crop of the cited element. Returns (PIL image, origin (cx0,cy0))."""
    img = Image.open(image_uri).convert("RGB")
    cx0, cy0, cx1, cy1 = (int(round(c)) for c in coord)
    return img.crop((cx0, cy0, cx1, cy1)), (cx0, cy0)


OCR_SCALE = 2   # upscale factor before OCR (improves small-font recognition)
OCR_PSM = 11    # PSM 11 = sparse text, find as much text as possible in no particular order


def ocr_tokens(pil_crop):
    """Tesseract → list of {text, box:[x0,y0,x1,y1] crop-local, conf}.

    Upscales by OCR_SCALE before OCR to improve recognition on small text,
    then scales boxes back to original crop coordinates.
    Uses PSM 6 (uniform text block) which handles wide multi-column layouts
    better than the default PSM 3.
    """
    # Upscale for better OCR accuracy
    w, h = pil_crop.size
    scaled = pil_crop.resize((w * OCR_SCALE, h * OCR_SCALE), Image.LANCZOS)

    config = f"--psm {OCR_PSM}"
    data = pytesseract.image_to_data(scaled, config=config, output_type=pytesseract.Output.DICT)
    tokens = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        # Scale box coordinates back to original crop space
        x = data["left"][i] / OCR_SCALE
        y = data["top"][i] / OCR_SCALE
        bw = data["width"][i] / OCR_SCALE
        bh = data["height"][i] / OCR_SCALE
        tokens.append({
            "text": txt,
            "box": [x, y, x + bw, y + bh],
            "conf": conf / 100.0,
        })
    return tokens


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def norm_num(s):
    raw = str(s)
    negative = raw.strip().startswith("(") and raw.strip().endswith(")")
    digits = re.sub(r"[^0-9.]", "", raw)
    try:
        value = float(digits)
        return f"{-value if negative else value:.2f}"
    except ValueError:
        return ""


def union_box(boxes):
    xs0 = [b[0] for b in boxes]; ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]; ys1 = [b[3] for b in boxes]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def match_tokens(tokens, target, kind):
    """Best contiguous token run matching `target`. Returns (union_box, score, matched_text)."""
    if not tokens:
        return None, 0.0, ""
    tgt = norm_num(target) if kind == "numeric" else norm(target)
    if not tgt:
        return None, 0.0, ""

    best = (None, 0.0, "")
    target_words = re.findall(r"\S+", str(target))
    max_window = min(len(tokens), max(6, min(20, len(target_words) + 4)))
    for i in range(len(tokens)):
        for w in range(1, max_window + 1):
            run = tokens[i:i + w]
            if not run:
                continue
            joined = "".join(t["text"] for t in run)
            cand = norm_num(joined) if kind == "numeric" else norm(joined)
            if not cand:
                continue
            # numerics: exact normalized value wins; else fuzzy ratio.
            if kind == "numeric":
                score = 100.0 if cand == tgt else fuzz.ratio(cand, tgt)
            else:
                score = fuzz.ratio(cand, tgt)
            if score > best[1]:
                best = (union_box([t["box"] for t in run]), score,
                        " ".join(t["text"] for t in run))
    return best


def to_page_coords(local_box, origin):
    """Crop-local box → page-pixel box (additive offset; same DPI ⇒ no scaling)."""
    cx0, cy0 = origin
    return [local_box[0] + cx0, local_box[1] + cy0,
            local_box[2] + cx0, local_box[3] + cy0]


_NUMERIC_TOKEN = re.compile(r"[$(+-]?\d[\d,]*(?:\.\d+)?\)?")


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
    for m in _NUMERIC_TOKEN.finditer(element_text or ""):
        raw = m.group(0)
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
            for w in range(min_window, max_window + 1):
                run = words[i:i + w]
                if len(run) != w:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Localize each entity

# COMMAND ----------

# DBTITLE 1,Localize entities (ThreadPool + OCR cache)
# ══════════════════════════════════════════════════════════════════════════════
# ThreadpoolExcecutor
# Use this cell for small sample data
# It runs 16 threads
# ══════════════════════════════════════════════════════════════════════════════

# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from threading import Lock

# MATCH_THRESHOLD = {"numeric": 95.0, "text": 85.0}
# MAX_WORKERS = 16  # Tesseract releases the GIL; threads parallelize well

# # ── OCR cache: avoid redundant Tesseract calls for entities sharing the same crop ──
# # Key = (image_uri, element_coord tuple) since multiple entities can share a citation bbox.
# _ocr_cache = {}
# _ocr_cache_lock = Lock()


# def _cached_ocr(image_uri, coord):
#     """Crop + OCR with caching. Thread-safe."""
#     cache_key = (image_uri, tuple(coord))
#     with _ocr_cache_lock:
#         if cache_key in _ocr_cache:
#             return _ocr_cache[cache_key]
#     # Outside lock: do the expensive work
#     crop, origin = crop_element(image_uri, coord)
#     tokens = ocr_tokens(crop)
#     result = (tokens, origin)
#     with _ocr_cache_lock:
#         _ocr_cache[cache_key] = result
#     return result


# def _localize_entity(e):
#     """Localize a single entity. Returns the result dict."""
#     coord, page_id = first_coord(e["citation"])
#     image_uri = e["page_uri_by_id"].get(page_id) or e["row_image_uri"]
#     rec = {**{k: e[k] for k in ("image_name", "field", "value", "kind", "citation_id")},
#            "image_uri": image_uri, "element_coord": coord, "page_id": page_id}
#     try:
#         element, element_text, element_overlap = element_for_citation(
#             e["citation"], e["parsed_elements"]
#         )
#         if element is None:
#             raise ValueError("No parsed element matched the citation bbox")
#         if not element_text:
#             raise ValueError("Matched parsed element has no text/content")

#         tokens, origin = _cached_ocr(image_uri, coord)
#         targets = match_targets(e["value"], e["kind"], element_text)
#         local_box, score, matched, target = match_best_target(tokens, targets, e["kind"])
#         threshold = MATCH_THRESHOLD[e["kind"]]
#         rec.update({
#             "element_id": element.get("id"),
#             "element_overlap": round(element_overlap, 3),
#             "element_text": element_text,
#             "target_candidates": targets,
#             "match_target": target,
#             "threshold": threshold,
#             "n_tokens": len(tokens),
#             "match_score": round(score, 1),
#             "matched_text": matched,
#             "second_level_coord": to_page_coords(local_box, origin) if local_box else None,
#             "ok": bool(local_box) and score >= threshold,
#             "error": None,
#         })
#     except Exception as ex:
#         rec.update({"ok": False, "error": str(ex), "match_score": 0.0,
#                     "second_level_coord": None, "threshold": MATCH_THRESHOLD[e["kind"]]})
#     return rec


# # ── Run with ThreadPoolExecutor ──────────────────────────────────────────────────────
# print(f"Localizing {len(full_sample)} entities with {MAX_WORKERS} threads...")
# t0 = time.time()

# results = [None] * len(full_sample)
# with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     futures = {executor.submit(_localize_entity, e): idx
#                for idx, e in enumerate(full_sample)}
#     done_count = 0
#     for future in as_completed(futures):
#         idx = futures[future]
#         results[idx] = future.result()
#         done_count += 1
#         if done_count % 500 == 0:
#             elapsed = time.time() - t0
#             rate = done_count / elapsed
#             eta = (len(full_sample) - done_count) / rate
#             print(f"  {done_count}/{len(full_sample)} done  "
#                   f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {rate:.1f} entities/s)")

# elapsed = time.time() - t0
# print(f"\nCompleted in {elapsed:.1f}s ({len(full_sample)/elapsed:.1f} entities/s)")
# print(f"OCR cache: {len(_ocr_cache)} unique crops (saved {len(full_sample) - len(_ocr_cache)} redundant OCR calls)")

# import pandas as pd
# summary = pd.DataFrame([{k: r.get(k) for k in
#     ("image_name", "field", "kind", "value", "match_target", "matched_text",
#      "match_score", "threshold", "element_id", "element_overlap", "ok", "error")}
#     for r in results])
# summary["value"] = summary["value"].astype(str)
# display(spark.createDataFrame(summary))

# hit_rate = sum(1 for r in results if r["ok"]) / max(1, len(results))
# print(f"\nLocalization hit-rate: {hit_rate:.0%} ({sum(1 for r in results if r['ok'])}/{len(results)})")

# COMMAND ----------

# DBTITLE 1,Alternative: Spark-distributed localization (for large datasets)
# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE PIPELINE (serverless-compatible)
# Use cells 13 + 14 INSTEAD of cell 12 for large datasets.
#
# Stage 1 (this cell): Deduplicate crops → OCR each unique crop exactly once
#   - Each Spark task processes ONE image crop (~100-500MB memory)
#   - Embarrassingly parallel: ~1500-2000 independent tasks
#   - Results written to Delta table (reusable across reruns)
#
# Stage 2 (next cell): Join entities with OCR tokens → fuzzy match
#   - Pure CPU, no images — trivially parallel, sub-second per entity
#   - 10K+ independent tasks, no skew
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import time
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, ArrayType, MapType
)

# ── Step 1a: Extract unique crops from the entity list ──────────────────────
unique_crops = {}  # (image_uri, coord_tuple) → {image_name, page_id}
for e in full_sample:
    coord, page_id = first_coord(e["citation"])
    if coord is None:
        continue
    image_uri = e["page_uri_by_id"].get(page_id) or e["row_image_uri"]
    key = (image_uri, tuple(coord))
    if key not in unique_crops:
        unique_crops[key] = {"image_name": e["image_name"], "page_id": page_id}

print(f"Unique crops to OCR: {len(unique_crops)} (from {len(full_sample)} entities)")
print(f"Cache savings: {len(full_sample) - len(unique_crops)} redundant OCR calls avoided")

# ── Step 1b: Build crops DataFrame for Spark distribution ───────────────────
crop_records = []
for (image_uri, coord_tuple), meta in unique_crops.items():
    crop_records.append({
        "image_uri": image_uri,
        "coord_x0": coord_tuple[0],
        "coord_y0": coord_tuple[1],
        "coord_x1": coord_tuple[2],
        "coord_y1": coord_tuple[3],
        "image_name": meta["image_name"],
    })

crops_df = spark.createDataFrame(pd.DataFrame(crop_records))
# Repartition so each task handles a small batch (~10 crops) — fits serverless memory
num_partitions = max(16, len(crop_records) // 10)
crops_df = crops_df.repartition(num_partitions)
print(f"Repartitioned into {num_partitions} partitions")

# ── Step 1c: OCR function (processes crops one at a time) ───────────────────
# Crop images saved here (path from config.yaml, created in NB01)
CROPS_VOLUME_PATH = config['crops_path']

ocr_result_schema = StructType([
    StructField("image_uri", StringType()),
    StructField("coord_x0", DoubleType()),
    StructField("coord_y0", DoubleType()),
    StructField("coord_x1", DoubleType()),
    StructField("coord_y1", DoubleType()),
    StructField("tokens_json", StringType()),  # JSON-serialized list of {text, box, conf}
    StructField("origin_x", DoubleType()),
    StructField("origin_y", DoubleType()),
    StructField("n_tokens", DoubleType()),
    StructField("crop_path", StringType()),  # Volume path where crop image is saved
    StructField("error", StringType()),
])


def ocr_batch(pdf_iter):
    """mapInPandas function: OCR each crop row sequentially.
    Saves each crop image to UC Volume, then runs Tesseract.
    Only ONE image in memory at a time → fits serverless memory."""
    import json
    import os
    # Ensure crops directory exists (FUSE mount requires explicit mkdir)
    os.makedirs(CROPS_VOLUME_PATH, exist_ok=True)
    for pdf in pdf_iter:
        results = []
        for _, row in pdf.iterrows():
            coord = [row["coord_x0"], row["coord_y0"], row["coord_x1"], row["coord_y1"]]
            # Deterministic filename: {image_hash}_{x0}_{y0}_{x1}_{y1}.png
            img_hash = os.path.basename(row["image_uri"]).rsplit(".", 1)[0]
            crop_fname = f"{img_hash}_{int(coord[0])}_{int(coord[1])}_{int(coord[2])}_{int(coord[3])}.png"
            crop_path = f"{CROPS_VOLUME_PATH}/{crop_fname}"
            out = {
                "image_uri": row["image_uri"],
                "coord_x0": row["coord_x0"],
                "coord_y0": row["coord_y0"],
                "coord_x1": row["coord_x1"],
                "coord_y1": row["coord_y1"],
                "tokens_json": "[]",
                "origin_x": 0.0,
                "origin_y": 0.0,
                "n_tokens": 0.0,
                "crop_path": None,
                "error": None,
            }
            try:
                crop, origin = crop_element(row["image_uri"], coord)
                # OCR first (the critical path — must not be blocked by save)
                tokens = ocr_tokens(crop)
                out["tokens_json"] = json.dumps(tokens)
                out["origin_x"] = float(origin[0])
                out["origin_y"] = float(origin[1])
                out["n_tokens"] = float(len(tokens))
                # Save crop to Volume (best-effort, non-blocking)
                try:
                    crop.save(crop_path, format="PNG")
                    out["crop_path"] = crop_path
                except Exception:
                    pass  # Volume write failure doesn't affect OCR results
                # Free memory immediately
                del crop
            except Exception as ex:
                out["error"] = str(ex)
            results.append(out)
        yield pd.DataFrame(results)


# ── Step 1d: Run distributed OCR ────────────────────────────────────────────
OCR_TABLE = f"{config['catalog']}.{config['schema']}.{config['table_prefix']}_ocr_tokens"

t0 = time.time()
ocr_results_df = crops_df.mapInPandas(ocr_batch, schema=ocr_result_schema)

# Write to Delta table (persists across reruns; skip re-OCR on next run)
ocr_results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OCR_TABLE)

ocr_elapsed = time.time() - t0
ocr_count = spark.table(OCR_TABLE).count()
ocr_errors = spark.table(OCR_TABLE).filter("error IS NOT NULL").count()

ocr_saved_crops = spark.table(OCR_TABLE).filter("crop_path IS NOT NULL").count()

print(f"\nStage 1 complete: {ocr_count} crops OCR'd in {ocr_elapsed:.1f}s")
print(f"  Throughput: {ocr_count/ocr_elapsed:.1f} crops/s")
print(f"  Errors: {ocr_errors}")
print(f"  Crops saved: {ocr_saved_crops} images → {CROPS_VOLUME_PATH}")
print(f"  OCR tokens saved to: {OCR_TABLE}")

# COMMAND ----------

# DBTITLE 1,Stage 2: Distributed matching (serverless-compatible)
# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Fuzzy matching — pure CPU, no images, trivially parallel.
#
# Joins each entity with its pre-computed OCR tokens from Stage 1,
# then runs match_targets + match_best_target per entity.
# Memory: ~1MB per task (just strings). Zero I/O.
# ══════════════════════════════════════════════════════════════════════════════

import time
import pandas as pd
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, BooleanType, LongType
)

MATCH_THRESHOLD = {"numeric": 95.0, "text": 85.0}

# ── Step 2a: Build entities DataFrame with citation coord + element context ──
entity_records = []
for e in full_sample:
    coord, page_id = first_coord(e["citation"])
    if coord is None:
        continue
    image_uri = e["page_uri_by_id"].get(page_id) or e["row_image_uri"]
    # Find matching element text (needed for match_targets)
    element, element_text, element_overlap = element_for_citation(
        e["citation"], e["parsed_elements"]
    )
    entity_records.append({
        "image_name": e["image_name"],
        "image_uri": image_uri,
        "field": e["field"],
        "value": str(e["value"]),
        "kind": e["kind"],
        "coord_x0": coord[0],
        "coord_y0": coord[1],
        "coord_x1": coord[2],
        "coord_y1": coord[3],
        "element_id": int(element.get("id", -1)) if element else -1,
        "element_text": element_text or "",
        "element_overlap": round(element_overlap, 3),
    })

entities_df = spark.createDataFrame(pd.DataFrame(entity_records))
print(f"Entities to match: {len(entity_records)}")

# ── Step 2b: Read OCR tokens from Stage 1 table ──
OCR_TABLE = f"{config['catalog']}.{config['schema']}.{config['table_prefix']}_ocr_tokens"
ocr_tokens_df = spark.table(OCR_TABLE).select(
    "image_uri", "coord_x0", "coord_y0", "coord_x1", "coord_y1",
    "tokens_json", "origin_x", "origin_y", "error"
)

# ── Step 2c: Join entities with their OCR tokens ──
join_keys = ["image_uri", "coord_x0", "coord_y0", "coord_x1", "coord_y1"]
matched_df = entities_df.join(ocr_tokens_df, on=join_keys, how="left")

# Repartition for parallelism: ~50 entities per task
num_match_partitions = max(16, len(entity_records) // 50)
matched_df = matched_df.repartition(num_match_partitions)

# ── Step 2d: Matching function (pure CPU) ──
match_result_schema = StructType([
    StructField("image_name", StringType()),
    StructField("field", StringType()),
    StructField("kind", StringType()),
    StructField("value", StringType()),
    StructField("image_uri", StringType()),
    StructField("element_id", LongType()),
    StructField("element_overlap", DoubleType()),
    StructField("element_text", StringType()),
    StructField("match_target", StringType()),
    StructField("matched_text", StringType()),
    StructField("match_score", DoubleType()),
    StructField("threshold", DoubleType()),
    StructField("n_tokens", LongType()),
    StructField("ok", BooleanType()),
    StructField("error", StringType()),
    StructField("second_level_x0", DoubleType()),
    StructField("second_level_y0", DoubleType()),
    StructField("second_level_x1", DoubleType()),
    StructField("second_level_y1", DoubleType()),
    StructField("element_coord_x0", DoubleType()),
    StructField("element_coord_y0", DoubleType()),
    StructField("element_coord_x1", DoubleType()),
    StructField("element_coord_y1", DoubleType()),
])


def match_batch(pdf_iter):
    """mapInPandas: run fuzzy matching per entity. Pure CPU, no images."""
    import json
    for pdf in pdf_iter:
        out_rows = []
        for _, row in pdf.iterrows():
            out = {
                "image_name": row["image_name"],
                "field": row["field"],
                "kind": row["kind"],
                "value": row["value"],
                "image_uri": row["image_uri"],
                "element_id": int(row["element_id"]) if row["element_id"] != -1 else None,
                "element_overlap": row["element_overlap"],
                "element_text": row["element_text"],
                "match_target": "",
                "matched_text": "",
                "match_score": 0.0,
                "threshold": MATCH_THRESHOLD[row["kind"]],
                "n_tokens": 0,
                "ok": False,
                "error": row.get("error"),  # OCR error from Stage 1
                "second_level_x0": None,
                "second_level_y0": None,
                "second_level_x1": None,
                "second_level_y1": None,
                "element_coord_x0": row["coord_x0"],
                "element_coord_y0": row["coord_y0"],
                "element_coord_x1": row["coord_x1"],
                "element_coord_y1": row["coord_y1"],
            }
            try:
                if row.get("error"):
                    raise ValueError(f"OCR failed: {row['error']}")
                if not row["element_text"]:
                    raise ValueError("No element text for matching")

                tokens = json.loads(row["tokens_json"]) if row["tokens_json"] else []
                out["n_tokens"] = len(tokens)
                origin = (row["origin_x"], row["origin_y"])

                targets = match_targets(row["value"], row["kind"], row["element_text"])
                local_box, score, matched, target = match_best_target(tokens, targets, row["kind"])
                threshold = MATCH_THRESHOLD[row["kind"]]
                page_box = to_page_coords(local_box, origin) if local_box else None

                out["match_target"] = target
                out["match_score"] = round(score, 1)
                out["matched_text"] = matched
                out["ok"] = bool(local_box) and score >= threshold

                if page_box:
                    out["second_level_x0"] = page_box[0]
                    out["second_level_y0"] = page_box[1]
                    out["second_level_x1"] = page_box[2]
                    out["second_level_y1"] = page_box[3]
            except Exception as ex:
                out["error"] = str(ex)
            out_rows.append(out)
        yield pd.DataFrame(out_rows)


# ── Step 2e: Run distributed matching ──
t0 = time.time()
results_df = matched_df.mapInPandas(match_batch, schema=match_result_schema)

# Collect results (pure CPU results are small — safe to toPandas)
results_pdf = results_df.toPandas()
match_elapsed = time.time() - t0

total = len(results_pdf)
ok_count = int(results_pdf["ok"].sum())
hit_rate = ok_count / max(1, total)

print(f"\nStage 2 complete: {total} entities matched in {match_elapsed:.1f}s")
print(f"  Throughput: {total/match_elapsed:.0f} entities/s")
print(f"  Hit-rate: {ok_count}/{total} ({hit_rate:.0%})")
display(spark.createDataFrame(results_pdf))

# Convert to list of dicts for section 4/5 compatibility
results = []
for _, row in results_pdf.iterrows():
    r = row.to_dict()
    # Reconstruct element_coord and second_level_coord for visualization
    r["element_coord"] = [r["element_coord_x0"], r["element_coord_y0"],
                          r["element_coord_x1"], r["element_coord_y1"]]
    if r["second_level_x0"] is not None:
        r["second_level_coord"] = [r["second_level_x0"], r["second_level_y0"],
                                   r["second_level_x1"], r["second_level_y1"]]
    else:
        r["second_level_coord"] = None
    results.append(r)

print(f"\n{'='*60}")
try:
    print(f"TOTAL PIPELINE: Stage 1 + Stage 2 = {ocr_elapsed + match_elapsed:.1f}s")
except NameError:
    print(f"Stage 2 only: {match_elapsed:.1f}s (Stage 1 results read from table)")
print(f"{'='*60}")

# COMMAND ----------

# DBTITLE 1,Save localization results to Delta table
# ══════════════════════════════════════════════════════════════════════════════
# Save Stage 2 localization results to a persistent Delta table.
# This enables downstream joins without re-running the localization pipeline.
# ══════════════════════════════════════════════════════════════════════════════

LOCALIZED_TABLE = f"{config['catalog']}.{config['schema']}.{config['table_prefix']}_localized"

# Write the full results (both ok=true and ok=false for audit)
localized_df = spark.createDataFrame(results_pdf)
localized_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(LOCALIZED_TABLE)

print(f"Localization results saved to: {LOCALIZED_TABLE}")
print(f"  Total rows: {localized_df.count()}")
print(f"  Localized (ok=true): {localized_df.filter('ok = true').count()}")

# COMMAND ----------

# DBTITLE 1,Join localization back to extraction flat table
# ══════════════════════════════════════════════════════════════════════════════
# Join localization results back to the EXTRACTED table (preserves the original
# ai_extract() VARIANT column). Adds a 'localizations' array of structs with
# word-level bounding boxes alongside the unflattened VARIANT output.
#
# Output schema:
#   - path                   : source document path
#   - paystub_extracted      : original ai_extract() VARIANT (untouched)
#   - image_uri              : page image path
#   - image_name             : image filename (join key)
#   - localizations          : array<struct> with per-entity bbox enrichment
#
# Each struct in 'localizations' contains:
#   - field, localized, match_score, matched_text
#   - word_bbox_x0/y0/x1/y1   (precise word-level, null if not localized)
#   - element_bbox_x0/y0/x1/y1 (original citation element-level bbox)
#   - best_bbox_x0/y0/x1/y1    (COALESCE: word > element, always non-null)
# ══════════════════════════════════════════════════════════════════════════════

from pyspark.sql import functions as F

LOCALIZED_TABLE = f"{config['catalog']}.{config['schema']}.{config['table_prefix']}_localized"
ENRICHED_TABLE = f"{config['catalog']}.{config['schema']}.{config['table_prefix']}_enriched"

# ── Prepare localization results with best_bbox ──
localized = spark.table(LOCALIZED_TABLE).select(
    "image_name", "field",
    F.col("value").alias("extracted_value"),
    F.col("kind").alias("value_type"),
    F.col("ok").alias("localized"),
    F.col("match_score"),
    F.col("matched_text"),
    # Bounding boxes as arrays: [x0, y0, x1, y1]
    F.array(
        F.col("second_level_x0"), F.col("second_level_y0"),
        F.col("second_level_x1"), F.col("second_level_y1")
    ).alias("word_bbox"),
    F.array(
        F.col("element_coord_x0"), F.col("element_coord_y0"),
        F.col("element_coord_x1"), F.col("element_coord_y1")
    ).alias("element_bbox"),
)

# best_bbox: word-level when localized (non-null), element-level as fallback
localized = localized.withColumn(
    "best_bbox",
    F.when(F.col("word_bbox")[0].isNotNull(), F.col("word_bbox"))
     .otherwise(F.col("element_bbox"))
)

# ── Group localizations by document ──
localized_grouped = (
    localized
    .groupBy("image_name")
    .agg(
        F.collect_list(
            F.struct(
                "field", "extracted_value", "value_type",
                "localized", "match_score", "matched_text",
                "word_bbox", "element_bbox", "best_bbox",
            )
        ).alias("localizations")
    )
)

# ── Read the EXTRACTED table (has the original VARIANT column) ──
extracted = spark.table(config['extracted_table'])

# Derive image_name from image_uri for join key (matches flat table convention)
extracted = extracted.withColumn(
    "image_name",
    F.element_at(F.split(F.col("image_uri"), "/"), -1)
)

# ── Join: extracted table + localizations array ──
enriched = extracted.join(localized_grouped, on="image_name", how="left")

# Write enriched table
enriched.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(ENRICHED_TABLE)

enriched_count = spark.table(ENRICHED_TABLE).count()
print(f"Enriched table saved to: {ENRICHED_TABLE}")
print(f"  Documents: {enriched_count}")
print(f"  Preserves: original ai_extract() VARIANT column (paystub_extracted)")
print(f"  Adds: 'localizations' array of structs with word-level bboxes")
print(f"")
print(f"  Each struct in 'localizations':")
print(f"    field                      (entity key, e.g. 'employee_name', 'income_item[2].rate')")
print(f"    extracted_value            (original value from ai_extract)")
print(f"    value_type                 ('numeric' or 'text')")
print(f"    localized, match_score, matched_text")
print(f"    word_bbox    array<double> [x0, y0, x1, y1] (precise, nulls if not localized)")
print(f"    element_bbox array<double> [x0, y0, x1, y1] (original citation bbox)")
print(f"    best_bbox    array<double> [x0, y0, x1, y1] (word if localized, else element)")
print(f"")
print(f"Example queries:")
print(f"  -- Get all localized entities with their word-level bbox")
print(f"  SELECT image_name, loc.*")
print(f"  FROM {ENRICHED_TABLE}")
print(f"  LATERAL VIEW explode(localizations) AS loc")
print(f"  WHERE loc.localized = true")
print(f"")
print(f"  -- Access original VARIANT (unflattened ai_extract output)")
print(f"  SELECT image_name,")
print(f"    paystub_extracted:response:employee_name:value::string AS employee_name")
print(f"  FROM {ENRICHED_TABLE}")

# COMMAND ----------

# DBTITLE 1,Export enriched results as JSON files (one per document)
# ══════════════════════════════════════════════════════════════════════════════
# Export the enriched table as individual JSON files (one per PDF document)
# to a UC Volume for easy downloading.
#
# Uses Spark-distributed writes via mapInPandas for scalability.
# Each worker writes its partition's documents in parallel via FUSE.
#
# Output: /Volumes/.../paystubs/enriched_json/<image_name>.json
# Each JSON contains:
#   - path: source PDF
#   - image_uri: page image path
#   - paystub_extracted: original ai_extract() response (nested)
#   - localizations: array of {field, extracted_value, bbox, ...}
# ══════════════════════════════════════════════════════════════════════════════

import time
from pyspark.sql.types import StructType, StructField, StringType, LongType

ENRICHED_TABLE = f"{config['catalog']}.{config['schema']}.{config['table_prefix']}_enriched"
JSON_OUTPUT_PATH = f"/Volumes/{config['catalog']}/{config['schema']}/{config['volume']}/enriched_json"

# Prepare: convert VARIANT columns to JSON strings in Spark (pushes serialization to workers)
export_df = spark.sql(f"""
    SELECT
        image_name,
        path,
        image_uri,
        to_json(paystub_extracted) AS paystub_extracted_json,
        to_json(localizations) AS localizations_json
    FROM {ENRICHED_TABLE}
""")

# Repartition for parallelism (~10 docs per task)
doc_count = export_df.count()
num_export_partitions = max(4, doc_count // 10)
export_df = export_df.repartition(num_export_partitions)

# Volume path captured for closure
JSON_VOL = JSON_OUTPUT_PATH

export_result_schema = StructType([
    StructField("output_path", StringType()),
    StructField("status", StringType()),
    StructField("bytes_written", LongType()),
])


def write_json_batch(pdf_iter):
    """mapInPandas: write JSON files in parallel from each Spark task."""
    import json
    import os
    import pandas as pd

    os.makedirs(JSON_VOL, exist_ok=True)

    for pdf in pdf_iter:
        results = []
        for _, row in pdf.iterrows():
            fname = row["image_name"].rsplit(".", 1)[0] + ".json"
            output_path = f"{JSON_VOL}/{fname}"
            try:
                doc = {
                    "image_name": row["image_name"],
                    "path": row["path"],
                    "image_uri": row["image_uri"],
                    "paystub_extracted": json.loads(row["paystub_extracted_json"]) if row["paystub_extracted_json"] else None,
                    "localizations": json.loads(row["localizations_json"]) if row["localizations_json"] else [],
                }
                content = json.dumps(doc, indent=2)
                with open(output_path, "w") as f:
                    f.write(content)
                results.append({
                    "output_path": output_path,
                    "status": "ok",
                    "bytes_written": len(content.encode("utf-8")),
                })
            except Exception as ex:
                results.append({
                    "output_path": output_path,
                    "status": f"error: {ex}",
                    "bytes_written": 0,
                })
        yield pd.DataFrame(results)


# Run distributed export
t0 = time.time()
export_results = export_df.mapInPandas(write_json_batch, schema=export_result_schema)
export_pdf = export_results.toPandas()
export_elapsed = time.time() - t0

ok_count = (export_pdf["status"] == "ok").sum()
error_count = len(export_pdf) - ok_count
total_bytes = export_pdf["bytes_written"].sum()

print(f"Exported {ok_count} JSON files in {export_elapsed:.1f}s (parallel)")
print(f"  Output: {JSON_OUTPUT_PATH}/")
print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")
print(f"  Throughput: {ok_count / export_elapsed:.1f} files/s")
if error_count:
    print(f"  Errors: {error_count}")
    print(export_pdf[export_pdf["status"] != "ok"][["output_path", "status"]].to_string())

# COMMAND ----------

# DBTITLE 1,Section 4 header
# MAGIC %md
# MAGIC ## 4. Score / decision gate
# MAGIC
# MAGIC Record the hit-rate. The recommendation expects OCR alone
# MAGIC to localize **~80%+** of numeric/text entities. If it clears that bar,
# MAGIC proceed to pipeline-ization (batch notebooks → streaming DAB) mirroring the
# MAGIC `document-page-classify-extraction` blueprint. If not, add the VLM
# MAGIC **disambiguation** fallback (“which of these OCR tokens is X?” — a text
# MAGIC answer, never coordinates) and re-score.

# COMMAND ----------

# DBTITLE 1,Score summary
print(f"Total entities   : {len(results)}")
print(f"Thresholds       : {MATCH_THRESHOLD}")
print(f"Localized        : {sum(1 for r in results if r['ok'])}")
print(f"Hit-rate         : {hit_rate:.0%}")

# Breakdown by kind
for kind in ("numeric", "text"):
    kind_results = [r for r in results if r["kind"] == kind]
    kind_ok = sum(1 for r in kind_results if r["ok"])
    kind_rate = kind_ok / max(1, len(kind_results))
    print(f"  {kind:8s}: {kind_ok}/{len(kind_results)} ({kind_rate:.0%})")

# Breakdown by source (scalar vs array)
scalar_results = [r for r in results if "[" not in r["field"]]
array_results = [r for r in results if "[" in r["field"]]
print(f"  scalar  : {sum(1 for r in scalar_results if r['ok'])}/{len(scalar_results)} ({sum(1 for r in scalar_results if r['ok'])/max(1,len(scalar_results)):.0%})")
print(f"  array   : {sum(1 for r in array_results if r['ok'])}/{len(array_results)} ({sum(1 for r in array_results if r['ok'])/max(1,len(array_results)):.0%})")

if any(not r["ok"] for r in results):
    print(f"\nFailures/low:")
    for r in results:
        if not r["ok"]:
            print(f"  {r['field']:45s} {r['image_name'][:20]}... score={r['match_score']}  err={r.get('error') or ''}")

# COMMAND ----------

# DBTITLE 1,Section 5 header (optional visualization)
# MAGIC %md
# MAGIC ## 5. Visualize element box (blue) + target box (red/amber) — Optional
# MAGIC
# MAGIC One panel per localized entity: the full page with the element-level
# MAGIC citation box in **blue** and the pinpointed second-level box in **red**.
# MAGIC Low-confidence best guesses are shown in **amber dashed** style so they are
# MAGIC visible for debugging but not confused with accepted localizations.
# MAGIC
# MAGIC > **Note:** This section is optional — skip if not needed for debugging.

# COMMAND ----------

# DBTITLE 1,Visualize localized bboxes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_entity(rec):
    if rec.get("error"):
        print(f"SKIP {rec['field']} ({rec['image_name']}): {rec['error']}")
        return
    img = Image.open(rec["image_uri"]).convert("RGB")
    ex0, ey0, ex1, ey1 = rec["element_coord"]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 8), dpi=72)
    fig.suptitle(
        f"{rec['field']} = {rec['value']!r}   "
        f"[match {rec['match_score']} / threshold {rec['threshold']:.0f} · "
        f"{'OK' if rec['ok'] else 'LOW'}]   "
        f"target: {rec.get('match_target')!r}   matched: {rec['matched_text']!r}",
        fontsize=11,
    )

    ax_full.imshow(img)
    ax_full.add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0,
                                fill=False, edgecolor="#1f77ff", lw=2, label="element citation"))
    if rec["second_level_coord"]:
        sx0, sy0, sx1, sy1 = rec["second_level_coord"]
        target_color = "#e41a1c" if rec["ok"] else "#ff9800"
        target_style = "-" if rec["ok"] else "--"
        target_label = "target" if rec["ok"] else "best match below threshold"
        ax_full.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0,
                                    fill=False, edgecolor=target_color, lw=2,
                                    linestyle=target_style, label=target_label))
    ax_full.set_title("page", fontsize=9)
    ax_full.legend(loc="upper right", fontsize=8)
    ax_full.axis("off")

    # Zoom on the cited element; second-level box rebased to element-local coords.
    ax_zoom.imshow(img.crop((int(ex0), int(ey0), int(ex1), int(ey1))))
    if rec["second_level_coord"]:
        sx0, sy0, sx1, sy1 = rec["second_level_coord"]
        target_color = "#e41a1c" if rec["ok"] else "#ff9800"
        target_style = "-" if rec["ok"] else "--"
        ax_zoom.add_patch(Rectangle((sx0 - ex0, sy0 - ey0), sx1 - sx0, sy1 - sy0,
                                    fill=False, edgecolor=target_color, lw=2,
                                    linestyle=target_style))
    ax_zoom.set_title("cited element (zoom)", fontsize=9)
    ax_zoom.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)  # free memory; avoids accumulating figures across reruns


# ── Document picker ─────────────────────────────────────────────────────────────────
doc_names = sorted({r["image_name"] for r in results})

try:
    dbutils.widgets.remove("vis_doc")
except Exception:
    pass

dbutils.widgets.dropdown(
    name="vis_doc",
    defaultValue=doc_names[0] if doc_names else "",
    choices=doc_names if doc_names else [""],
    label="Visualize document",
)

selected_doc = dbutils.widgets.get("vis_doc")
doc_results = [r for r in results if r["image_name"] == selected_doc]

print(f"Document : {selected_doc}")
print(f"Entities : {len(doc_results)}  "
      f"({sum(1 for r in doc_results if r['ok'])} OK / "
      f"{sum(1 for r in doc_results if not r['ok'])} low/fail)")

for rec in doc_results:
    show_entity(rec)