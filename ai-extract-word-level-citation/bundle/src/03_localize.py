# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Localize Entities to Word-Level Boxes
# MAGIC
# MAGIC Builds citable entities from the flat extraction table, deduplicates element
# MAGIC crops, OCRs each crop with Tesseract, matches extracted values to OCR token
# MAGIC boxes, and writes:
# MAGIC
# MAGIC - `*_ocr_tokens`
# MAGIC - `*_localized`
# MAGIC - `*_enriched`
# MAGIC
# MAGIC The `*_enriched` table is the pipeline's final output.

# COMMAND ----------

# MAGIC %pip install -q pytesseract rapidfuzz
# MAGIC dbutils.library.restartPython()

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

volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
artifact_scope = volume_subdir if volume_subdir else "_root"
artifacts_root = f"{volume_root}/_artifacts/{table_prefix}"
crops_path = f"{artifacts_root}/crops/{artifact_scope}"

flat_table = f"{catalog}.{schema}.{table_prefix}_extracted_flat"
extracted_table = f"{catalog}.{schema}.{table_prefix}_extracted"
ocr_table = f"{catalog}.{schema}.{table_prefix}_ocr_tokens"
localized_table = f"{catalog}.{schema}.{table_prefix}_localized"
enriched_table = f"{catalog}.{schema}.{table_prefix}_enriched"

print(f"flat_table      = {flat_table}")
print(f"ocr_table       = {ocr_table}")
print(f"localized_table = {localized_table}")
print(f"enriched_table  = {enriched_table}")
print(f"crops_path      = {crops_path}")

# COMMAND ----------

import json
import os
import re
import time
from html import unescape

import pandas as pd
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

EXTRACT_FIELDS = [
    "employee_name",
    "employer_name",
    "social_security_number",
    "taxable_marital_status",
    "borrower_employee_address__address_line",
    "borrower_employee_address__city_name",
    "borrower_employee_address__state_code",
    "borrower_employee_address__zip_code",
    "borrower_employer_address__address_line",
    "borrower_employer_address__city_name",
    "borrower_employer_address__state_code",
    "borrower_employer_address__zip_code",
    "hire_date",
    "ets_date",
    "pay_date",
    "period_start_date",
    "period_end_date",
    "payment_frequency",
    "income_source_type",
    "gross__current_amount",
    "gross__year_to_date_amount",
    "net_pay__current_amount",
    "net_pay__year_to_date_amount",
    "adjustments_gross_income__current_amount",
    "adjustments_gross_income__year_to_date_amount",
]

NUMBER_FIELDS = {
    "gross__current_amount",
    "gross__year_to_date_amount",
    "net_pay__current_amount",
    "net_pay__year_to_date_amount",
    "adjustments_gross_income__current_amount",
    "adjustments_gross_income__year_to_date_amount",
}

ARRAY_FIELDS = ["income_item", "deduction_item"]
ARRAY_SUB_FIELDS = {
    "income_item": [
        ("description", "string"),
        ("period_hours", "double"),
        ("rate", "double"),
        ("current_amount", "double"),
        ("year_to_date_amount", "double"),
    ],
    "deduction_item": [
        ("description", "string"),
        ("period_amount", "double"),
        ("year_to_date_amount", "double"),
    ],
}

MATCH_THRESHOLD = {"numeric": 95.0, "text": 85.0}
OCR_SCALE = 2
OCR_PSM = 11

# COMMAND ----------


# Pure helpers live in localize_lib.py (co-located, unit-tested). Serverless compute
# is Spark Connect — spark.sparkContext (and addPyFile) is unavailable — so instead
# of shipping the file we register the module for pickle-by-value. cloudpickle then
# serializes these functions directly into the pandas-UDF closures below, so the
# executors don't need to import the module. We register on both the standalone and
# the pyspark-vendored cloudpickle so whichever copy Spark Connect uses picks it up.
import sys

sys.path.insert(0, os.getcwd())
import localize_lib  # noqa: E402

import cloudpickle
import pyspark.cloudpickle as _pyspark_cloudpickle

cloudpickle.register_pickle_by_value(localize_lib)
_pyspark_cloudpickle.register_pickle_by_value(localize_lib)

from localize_lib import (  # noqa: E402
    _loads,
    normalize_box,
    first_coord,
    box_area,
    intersection_area,
    overlap_score,
    element_box_on_page,
    text_from_element,
    element_for_citation,
    norm,
    norm_num,
    union_box,
    match_tokens,
    to_page_coords,
    dedupe_keep_order,
    numeric_targets_from_element_text,
    text_targets_from_element_text,
    match_targets,
    match_best_target,
)


def crop_element(image_uri, coord):
    img = Image.open(image_uri).convert("RGB")
    cx0, cy0, cx1, cy1 = (int(round(c)) for c in coord)
    return img.crop((cx0, cy0, cx1, cy1)), (cx0, cy0)


def ocr_tokens(pil_crop):
    width, height = pil_crop.size
    scaled = pil_crop.resize((width * OCR_SCALE, height * OCR_SCALE), Image.LANCZOS)
    data = pytesseract.image_to_data(
        scaled,
        config=f"--psm {OCR_PSM}",
        output_type=pytesseract.Output.DICT,
    )

    tokens = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        x = data["left"][i] / OCR_SCALE
        y = data["top"][i] / OCR_SCALE
        width = data["width"][i] / OCR_SCALE
        height = data["height"][i] / OCR_SCALE
        tokens.append(
            {
                "text": text,
                "box": [x, y, x + width, y + height],
                "conf": conf / 100.0,
            }
        )
    return tokens


# COMMAND ----------

flat = spark.table(flat_table)
rows = (
    flat.selectExpr(
        "path",
        "image_name",
        "image_uri",
        "cast(parsed_elements AS string) AS parsed_elements_json",
        *[column for field in EXTRACT_FIELDS for column in (field, f"{field}_citation_ids")],
        *[f"cast({array_field} AS string) AS {array_field}_json" for array_field in ARRAY_FIELDS],
        "cast(citations AS string)      AS citations_json",
        "cast(citation_pages AS string) AS citation_pages_json",
    )
    .collect()
)
print(f"{len(rows)} document rows in {flat_table}")

entities = []
for row in rows:
    citations = _loads(row["citations_json"]) or []
    citation_by_id = {int(citation.get("id", -1)): citation for citation in citations}
    citation_pages = _loads(row["citation_pages_json"]) or []
    page_uri_by_id = {int(page.get("id", -1)): page.get("image_uri") for page in citation_pages}
    parsed_elements = _loads(row["parsed_elements_json"]) or []

    for field in EXTRACT_FIELDS:
        value = row[field]
        citation_ids = row[f"{field}_citation_ids"]
        if value is None or not citation_ids:
            continue
        for citation_id in citation_ids:
            citation = citation_by_id.get(int(citation_id))
            if not citation:
                continue
            entities.append(
                {
                    "path": row["path"],
                    "image_name": row["image_name"],
                    "row_image_uri": row["image_uri"],
                    "field": field,
                    "value": value,
                    "kind": "numeric" if field in NUMBER_FIELDS else "text",
                    "citation_id": int(citation_id),
                    "citation": citation,
                    "page_uri_by_id": page_uri_by_id,
                    "parsed_elements": parsed_elements,
                }
            )

    for array_field in ARRAY_FIELDS:
        array_data = _loads(row[f"{array_field}_json"]) or []
        for idx, item in enumerate(array_data):
            if not isinstance(item, dict):
                continue
            for sub_name, sub_type in ARRAY_SUB_FIELDS[array_field]:
                raw = item.get(sub_name)
                if raw is None:
                    continue
                if isinstance(raw, dict) and "value" in raw:
                    value = raw["value"]
                    citation_ids = raw.get("citation_ids") or []
                else:
                    value = raw
                    citation_ids = []
                if value is None or not citation_ids:
                    continue
                for citation_id in citation_ids:
                    citation = citation_by_id.get(int(citation_id))
                    if not citation:
                        continue
                    entities.append(
                        {
                            "path": row["path"],
                            "image_name": row["image_name"],
                            "row_image_uri": row["image_uri"],
                            "field": f"{array_field}[{idx}].{sub_name}",
                            "value": value,
                            "kind": "numeric" if sub_type == "double" else "text",
                            "citation_id": int(citation_id),
                            "citation": citation,
                            "page_uri_by_id": page_uri_by_id,
                            "parsed_elements": parsed_elements,
                        }
                    )

numeric_count = sum(1 for entity in entities if entity["kind"] == "numeric")
print(f"{len(entities)} citable entities ({numeric_count} numeric / {len(entities) - numeric_count} text)")

# COMMAND ----------

unique_crops = {}
for entity in entities:
    coord, page_id = first_coord(entity["citation"])
    if coord is None:
        continue
    image_uri = entity["page_uri_by_id"].get(page_id) or entity["row_image_uri"]
    key = (image_uri, tuple(coord))
    unique_crops.setdefault(key, {"image_name": entity["image_name"], "page_id": page_id})

crop_records = [
    {
        "image_uri": image_uri,
        "coord_x0": coord_tuple[0],
        "coord_y0": coord_tuple[1],
        "coord_x1": coord_tuple[2],
        "coord_y1": coord_tuple[3],
        "image_name": meta["image_name"],
    }
    for (image_uri, coord_tuple), meta in unique_crops.items()
]

print(f"Unique crops to OCR: {len(crop_records)}")

ocr_result_schema = StructType(
    [
        StructField("image_uri", StringType()),
        StructField("coord_x0", DoubleType()),
        StructField("coord_y0", DoubleType()),
        StructField("coord_x1", DoubleType()),
        StructField("coord_y1", DoubleType()),
        StructField("tokens_json", StringType()),
        StructField("origin_x", DoubleType()),
        StructField("origin_y", DoubleType()),
        StructField("n_tokens", DoubleType()),
        StructField("crop_path", StringType()),
        StructField("error", StringType()),
    ]
)


def ocr_batch(pdf_iter):
    import json
    import os

    os.makedirs(crops_path, exist_ok=True)
    for pdf in pdf_iter:
        results = []
        for _, row in pdf.iterrows():
            coord = [row["coord_x0"], row["coord_y0"], row["coord_x1"], row["coord_y1"]]
            image_base = os.path.basename(row["image_uri"]).rsplit(".", 1)[0]
            crop_name = f"{image_base}_{int(coord[0])}_{int(coord[1])}_{int(coord[2])}_{int(coord[3])}.png"
            crop_path = f"{crops_path}/{crop_name}"
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
                tokens = ocr_tokens(crop)
                out["tokens_json"] = json.dumps(tokens)
                out["origin_x"] = float(origin[0])
                out["origin_y"] = float(origin[1])
                out["n_tokens"] = float(len(tokens))
                try:
                    crop.save(crop_path, format="PNG")
                    out["crop_path"] = crop_path
                except Exception:
                    pass
                del crop
            except Exception as ex:
                out["error"] = str(ex)
            results.append(out)
        yield pd.DataFrame(results)


if crop_records:
    crops_df = spark.createDataFrame(pd.DataFrame(crop_records))
    crops_df = crops_df.repartition(max(16, len(crop_records) // 10))
    t0 = time.time()
    ocr_df = crops_df.mapInPandas(ocr_batch, schema=ocr_result_schema)
    ocr_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(ocr_table)
    ocr_elapsed = time.time() - t0
else:
    empty_ocr = spark.createDataFrame([], schema=ocr_result_schema)
    empty_ocr.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(ocr_table)
    ocr_elapsed = 0.0

ocr_count = spark.table(ocr_table).count()
ocr_errors = spark.table(ocr_table).filter("error IS NOT NULL").count()
print(f"OCR complete: {ocr_count} crops in {ocr_elapsed:.1f}s; errors={ocr_errors}")

# COMMAND ----------

entity_records = []
for entity in entities:
    coord, page_id = first_coord(entity["citation"])
    if coord is None:
        continue
    image_uri = entity["page_uri_by_id"].get(page_id) or entity["row_image_uri"]
    element, element_text, element_overlap = element_for_citation(
        entity["citation"],
        entity["parsed_elements"],
    )
    entity_records.append(
        {
            "path": entity["path"],
            "image_name": entity["image_name"],
            "image_uri": image_uri,
            "field": entity["field"],
            "value": str(entity["value"]),
            "kind": entity["kind"],
            "citation_id": int(entity["citation_id"]),
            "page_id": int(page_id) if page_id is not None else None,
            "coord_x0": coord[0],
            "coord_y0": coord[1],
            "coord_x1": coord[2],
            "coord_y1": coord[3],
            "element_id": int(element.get("id", -1)) if element and element.get("id") is not None else -1,
            "element_text": element_text or "",
            "element_overlap": round(element_overlap, 3),
        }
    )

print(f"Entities to match: {len(entity_records)}")

match_result_schema = StructType(
    [
        StructField("path", StringType()),
        StructField("image_name", StringType()),
        StructField("field", StringType()),
        StructField("kind", StringType()),
        StructField("value", StringType()),
        StructField("image_uri", StringType()),
        StructField("citation_id", LongType()),
        StructField("page_id", LongType()),
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
    ]
)


def match_batch(pdf_iter):
    import json
    import math

    for pdf in pdf_iter:
        out_rows = []
        for _, row in pdf.iterrows():
            ocr_error = row.get("error")
            if isinstance(ocr_error, float) and math.isnan(ocr_error):
                ocr_error = None

            out = {
                "path": row["path"],
                "image_name": row["image_name"],
                "field": row["field"],
                "kind": row["kind"],
                "value": row["value"],
                "image_uri": row["image_uri"],
                "citation_id": int(row["citation_id"]),
                "page_id": int(row["page_id"]) if row["page_id"] == row["page_id"] else None,
                "element_id": int(row["element_id"]) if row["element_id"] != -1 else None,
                "element_overlap": row["element_overlap"],
                "element_text": row["element_text"],
                "match_target": "",
                "matched_text": "",
                "match_score": 0.0,
                "threshold": MATCH_THRESHOLD[row["kind"]],
                "n_tokens": 0,
                "ok": False,
                "error": ocr_error,
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
                if ocr_error:
                    raise ValueError(f"OCR failed: {ocr_error}")
                if not row["element_text"]:
                    raise ValueError("No element text for matching")

                tokens = json.loads(row["tokens_json"]) if row["tokens_json"] else []
                out["n_tokens"] = len(tokens)
                origin = (row["origin_x"], row["origin_y"])
                targets = match_targets(row["value"], row["kind"], row["element_text"])
                local_box, score, matched, target = match_best_target(tokens, targets, row["kind"])
                page_box = to_page_coords(local_box, origin) if local_box else None

                out["match_target"] = target
                out["matched_text"] = matched
                out["match_score"] = round(score, 1)
                out["ok"] = bool(local_box) and score >= MATCH_THRESHOLD[row["kind"]]

                if page_box:
                    out["second_level_x0"] = page_box[0]
                    out["second_level_y0"] = page_box[1]
                    out["second_level_x1"] = page_box[2]
                    out["second_level_y1"] = page_box[3]
            except Exception as ex:
                out["error"] = str(ex)
            out_rows.append(out)
        yield pd.DataFrame(out_rows)


if entity_records:
    entities_df = spark.createDataFrame(pd.DataFrame(entity_records))
    ocr_tokens_df = spark.table(ocr_table).select(
        "image_uri",
        "coord_x0",
        "coord_y0",
        "coord_x1",
        "coord_y1",
        "tokens_json",
        "origin_x",
        "origin_y",
        "error",
    )
    matched_df = entities_df.join(
        ocr_tokens_df,
        on=["image_uri", "coord_x0", "coord_y0", "coord_x1", "coord_y1"],
        how="left",
    ).repartition(max(16, len(entity_records) // 50))

    t0 = time.time()
    results_df = matched_df.mapInPandas(match_batch, schema=match_result_schema)
    results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(localized_table)
    match_elapsed = time.time() - t0
else:
    empty_localized = spark.createDataFrame([], schema=match_result_schema)
    empty_localized.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(localized_table)
    match_elapsed = 0.0

localized_df = spark.table(localized_table)
total = localized_df.count()
ok_count = localized_df.filter("ok = true").count()
print(f"Localization complete: {ok_count}/{total} ok in {match_elapsed:.1f}s")

# COMMAND ----------

localized = spark.table(localized_table).select(
    "path",
    "image_name",
    "field",
    F.col("value").alias("extracted_value"),
    F.col("kind").alias("value_type"),
    F.col("ok").alias("localized"),
    "match_score",
    "matched_text",
    F.array(
        F.col("second_level_x0"),
        F.col("second_level_y0"),
        F.col("second_level_x1"),
        F.col("second_level_y1"),
    ).alias("word_bbox"),
    F.array(
        F.col("element_coord_x0"),
        F.col("element_coord_y0"),
        F.col("element_coord_x1"),
        F.col("element_coord_y1"),
    ).alias("element_bbox"),
)

localized = localized.withColumn(
    "best_bbox",
    F.when(F.col("word_bbox")[0].isNotNull(), F.col("word_bbox")).otherwise(F.col("element_bbox")),
)

localized_grouped = (
    localized.groupBy("path", "image_name")
    .agg(
        F.collect_list(
            F.struct(
                "field",
                "extracted_value",
                "value_type",
                "localized",
                "match_score",
                "matched_text",
                "word_bbox",
                "element_bbox",
                "best_bbox",
            )
        ).alias("localizations")
    )
)

extracted = spark.table(extracted_table).withColumn(
    "image_name",
    F.element_at(F.split(F.col("image_uri"), "/"), -1),
)

enriched = extracted.join(localized_grouped, on=["path", "image_name"], how="left")
enriched.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(enriched_table)

enriched_count = spark.table(enriched_table).count()
print(f"Enriched documents: {enriched_count}")
dbutils.notebook.exit(f"localized_ok={ok_count};localized_total={total};enriched={enriched_count}")
