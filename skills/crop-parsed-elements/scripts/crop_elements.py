"""Crop ai_parse_document v2.0 elements (figures, tables, ...) to images and attach page context.

Copy this whole file unchanged into one notebook cell (Genie Code or Databricks
notebook), after `%pip install pypdfium2` in its own cell, then call the Spark
functions below -- see SKILL.md. Pure functions (no Spark / pypdfium2 at module level)
are unit-tested in skills/tests/test_crop_parsed_elements.py.

Coordinates (verified live): bbox coord = [x0, y0, x1, y1] integer pixels, origin top-left,
in the page rendered at 200 DPI -- the same space as the page image ai_parse_document
writes with imageOutputPath. Without imageOutputPath, render the PDF page at 200 DPI
(pypdfium2 scale = 200/72) and the boxes line up pixel-exact.
"""

import hashlib
import json
import math
import os

PARSE_DPI = 200
CONTEXT_TYPES = ("title", "section_header", "caption")
NEARBY_TYPES = ("text", "footnote")  # legends are often parsed as `footnote` next to the chart
MAX_CONTEXT_TEXT_CHARS = 80          # short text anywhere on the page: units, subtitles
ADJACENT_GAP_PX = 150                # nearby text within this gap (parse px) counts as context ...
MAX_ADJACENT_TEXT_CHARS = 400        # ... unless it is long body text


# ---------------------------------------------------------------- pure helpers


def pixel_box(coord, page_w_px, page_h_px, pad_px=20, render_dpi=PARSE_DPI):
    """Crop box (left, top, right, bottom) in pixels of a page rendered at `render_dpi`.
    `coord` and `pad_px` are in parse pixels (200 DPI); the result is clamped to the page."""
    x0, x1 = sorted((coord[0], coord[2]))
    y0, y1 = sorted((coord[1], coord[3]))
    s = render_dpi / PARSE_DPI
    return (
        max(0, math.floor((x0 - pad_px) * s)),
        max(0, math.floor((y0 - pad_px) * s)),
        min(page_w_px, math.ceil((x1 + pad_px) * s)),
        min(page_h_px, math.ceil((y1 + pad_px) * s)),
    )


def _page_of(el):
    boxes = el.get("bbox") or []
    return boxes[0].get("page_id") if boxes else None


def _top_left(el):
    c = el["bbox"][0]["coord"]
    return (min(c[1], c[3]), min(c[0], c[2]))


def _adjacent(a, b, gap):
    """Boxes overlap on one axis and are within `gap` pixels on the other."""
    ax0, ax1 = sorted((a[0], a[2])); ay0, ay1 = sorted((a[1], a[3]))
    bx0, bx1 = sorted((b[0], b[2])); by0, by1 = sorted((b[1], b[3]))
    x_overlap, y_overlap = ax0 < bx1 and bx0 < ax1, ay0 < by1 and by0 < ay1
    x_gap = max(bx0 - ax1, ax0 - bx1, 0)
    y_gap = max(by0 - ay1, ay0 - by1, 0)
    return (y_overlap and x_gap <= gap) or (x_overlap and y_gap <= gap)


def figure_context(elements, figure):
    """Text a vision model needs but the crop usually lacks, in reading order: same-page
    titles, section headers, captions; short text anywhere on the page (units such as
    'In Millions'); and text/footnotes adjacent to the figure (legends)."""
    page = _page_of(figure)
    fig_box = figure["bbox"][0]["coord"]
    picked = []
    for el in elements:
        if el is figure or _page_of(el) != page:
            continue
        text = (el.get("content") or "").strip()
        if not text:
            continue
        kind = el.get("type")
        if (
            kind in CONTEXT_TYPES
            or (kind == "text" and len(text) <= MAX_CONTEXT_TEXT_CHARS)
            or (kind in NEARBY_TYPES and len(text) <= MAX_ADJACENT_TEXT_CHARS
                and _adjacent(fig_box, el["bbox"][0]["coord"], ADJACENT_GAP_PX))
        ):
            picked.append(el)
    picked.sort(key=_top_left)
    return "\n".join(el["content"].strip() for el in picked)


def builtin_chart_data(figure):
    """The figure's built-in structured chart data (a dict), or None.
    ai_parse_document fills figure `content` with chart JSON only when figure descriptions
    are on (descriptionElementTypes default '*' or including 'figure'); with '' it is raw label text."""
    content = (figure.get("content") or "").strip()
    if not content.startswith("{"):
        return None
    try:
        data = json.loads(content)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def list_elements(parsed, types=("figure",)):
    """Elements of the given types from one parsed document, each with its array index,
    page id, first bbox coord, built-in description/chart data, and page context."""
    elements = (parsed.get("document") or {}).get("elements") or []
    out = []
    for idx, el in enumerate(elements):
        if el.get("type") not in types or not el.get("bbox"):
            continue
        chart = builtin_chart_data(el)
        out.append({
            "element_idx": idx,
            "element_id": el.get("id"),
            "type": el.get("type"),
            "page_id": _page_of(el),
            "coord": [int(v) for v in el["bbox"][0]["coord"]],
            "n_boxes": len(el["bbox"]),
            "description": el.get("description"),
            "builtin_chart_json": json.dumps(chart) if chart is not None else None,
            "context": figure_context(elements, el),
        })
    return out


def splice_figure_text(elements, replacements):
    """Document text in element order, with figure content replaced by
    replacements[element id] (e.g. a chart insight) so embeddings keep the chart in place."""
    lines = []
    for el in elements:
        if el.get("type") == "figure" and el.get("id") in replacements:
            lines.append(replacements[el["id"]])
        elif (el.get("content") or "").strip():
            lines.append(el["content"].strip())
    return "\n".join(lines)


def sql_str(value):
    """Spark SQL string literal with backslashes and single quotes escaped."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def crop_file_name(doc_id, element_idx, page_id):
    stem = os.path.splitext(os.path.basename(doc_id))[0]
    digest = hashlib.sha1(doc_id.encode()).hexdigest()[:10]
    return f"{stem}_{digest}/p{page_id:04d}_e{element_idx:05d}.png"


# ---------------------------------------------------------------- spark


def elements_to_crop(parsed_df, parsed_col="parsed", doc_id_col="path", types=("figure",)):
    """One row per element of `types`: doc_id, element_idx, element_id, type, page_id, coord,
    n_boxes, description, builtin_chart_json, context."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

    row_type = ArrayType(StructType([
        StructField("element_idx", IntegerType()),
        StructField("element_id", IntegerType()),
        StructField("type", StringType()),
        StructField("page_id", IntegerType()),
        StructField("coord", ArrayType(IntegerType())),
        StructField("n_boxes", IntegerType()),
        StructField("description", StringType()),
        StructField("builtin_chart_json", StringType()),
        StructField("context", StringType()),
    ]))
    wanted = tuple(types)

    @F.udf(row_type)
    def _list(js):
        return list_elements(json.loads(js), wanted) if js else []

    return (
        parsed_df.select(F.col(doc_id_col).alias("doc_id"), F.expr(f"to_json({parsed_col})").alias("js"))
        .select("doc_id", F.explode(_list("js")).alias("e"))
        .select("doc_id", "e.*")
    )


def crop_elements(spark, elements_df, out_dir, pad_px=20, min_side_px=150, render_dpi=PARSE_DPI):
    """Render each needed PDF page once (pypdfium2, `render_dpi`), crop every element, write PNGs
    under `out_dir` (a /Volumes path). doc_id must be the source PDF's /Volumes path.
    Returns elements_df columns + crop_path, status
    (ok | too_small | bad_page | pdf_not_found | bad_pdf).
    Only the first bbox of each element is cropped (n_boxes shows when there are more)."""
    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    def _norm(c):
        return F.regexp_replace(c, "^dbfs:", "")

    # Which referenced PDFs exist: list the parent folders' paths only (binaryFile skips reading
    # `content` when it is not selected), then read just those files. Loading the doc_id list
    # directly raises PATH_NOT_FOUND on any missing file.
    wanted = {r["doc_id"].removeprefix("dbfs:") for r in elements_df.select("doc_id").distinct().collect()}
    listed = spark.read.format("binaryFile").option("pathGlobFilter", "*.pdf").load(
        sorted({p.rsplit("/", 1)[0] for p in wanted})
    ).select(_norm(F.col("path")).alias("p"))
    present = [r["p"] for r in listed.collect() if r["p"] in wanted]
    pdfs = (
        spark.read.format("binaryFile").load(present).select(_norm(F.col("path")).alias("pdf_path"), "content")
        if present else spark.createDataFrame([], "pdf_path string, content binary")
    )

    # One row per document (its elements collected), joined to the PDF bytes once -- joining
    # bytes onto every element row would hold one copy of the PDF per element.
    el_cols = elements_df.columns
    per_doc = elements_df.groupBy("doc_id").agg(F.collect_list(F.struct(*el_cols)).alias("els"))
    # left join: a doc_id that matches no PDF gets status pdf_not_found instead of vanishing
    work = per_doc.withColumn("pdf_path", _norm(F.col("doc_id"))).join(pdfs, "pdf_path", "left")
    out_schema = StructType(
        list(elements_df.schema.fields) + [StructField("crop_path", StringType()), StructField("status", StringType())]
    )
    cols = [f.name for f in out_schema.fields]

    def _crop_doc(doc_df):
        import pypdfium2 as pdfium

        content = doc_df["content"].iloc[0]
        records = [dict(e) for e in doc_df["els"].iloc[0]]
        if content is None:
            return pd.DataFrame([{**r, "crop_path": None, "status": "pdf_not_found"} for r in records], columns=cols)
        try:
            pdf = pdfium.PdfDocument(content)
        except pdfium.PdfiumError:  # corrupt or encrypted PDF: mark its rows, don't fail the job
            return pd.DataFrame([{**r, "crop_path": None, "status": "bad_pdf"} for r in records], columns=cols)
        current_page, img, rows = None, None, []
        # page order + one rendered page in memory at a time (a 200-DPI page is ~30-50 MB)
        records.sort(key=lambda rec: (rec["page_id"] is None, rec["page_id"] or 0))
        for r in records:
            box_w = abs(r["coord"][2] - r["coord"][0])
            box_h = abs(r["coord"][3] - r["coord"][1])
            if min(box_w, box_h) < min_side_px:
                rows.append({**r, "crop_path": None, "status": "too_small"})
                continue
            if r["page_id"] is None or not 0 <= r["page_id"] < len(pdf):
                rows.append({**r, "crop_path": None, "status": "bad_page"})
                continue
            if r["page_id"] != current_page:
                img = pdf[r["page_id"]].render(scale=render_dpi / 72).to_pil().convert("RGB")
                current_page = r["page_id"]
            crop_path = f"{out_dir.rstrip('/')}/{crop_file_name(r['doc_id'], r['element_idx'], r['page_id'])}"
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            img.crop(pixel_box(list(r["coord"]), img.width, img.height, pad_px, render_dpi)).save(crop_path)
            rows.append({**r, "crop_path": crop_path, "status": "ok"})
        return pd.DataFrame(rows, columns=cols)

    return work.groupBy("doc_id").applyInPandas(_crop_doc, schema=out_schema)


DEFAULT_CHART_PROMPT = (
    "Extract this chart's data. Use the page context below for the title and units; do not invent "
    "either. Return only JSON: {\"is_chart\": bool, \"title\": str|null, \"units\": str|null, "
    "\"categories\": [str], \"series\": [{\"name\": str, \"values\": [number|null]}]}. "
    "Use null for any value you cannot read."
)


def vision_insights(spark, crops_df, endpoint, prompt=DEFAULT_CHART_PROMPT):
    """ai_query(endpoint, prompt + page context, files => crop PNG) for every ok crop.
    Returns crops_df columns + response (string, code fences stripped) + error."""
    from pyspark.sql import functions as F

    ok = crops_df.where("status = 'ok'")
    paths = [r["crop_path"] for r in ok.select("crop_path").collect()]
    if not paths:  # nothing to send: same columns, no rows
        return ok.withColumn("response", F.lit(None).cast("string")).withColumn("error", F.lit(None).cast("string"))
    images = spark.read.format("binaryFile").load(paths).select(
        F.regexp_replace("path", "^dbfs:", "").alias("crop_path"), F.col("content").alias("image")
    )
    full_prompt = F.concat(
        F.lit(prompt + "\n\nPage context:\n"), F.coalesce("context", F.lit("")),
        F.lit("\n\nParser description (may be generic or wrong):\n"), F.coalesce("description", F.lit("")),
    )
    q = ok.join(images, "crop_path").withColumn("full_prompt", full_prompt).withColumn(
        "r", F.expr(f"ai_query({sql_str(endpoint)}, full_prompt, files => image, failOnError => false)")
    )
    return q.drop("image", "full_prompt").withColumn(
        "response", F.regexp_replace("r.result", r"(?s)^\s*```(?:json)?\s*|\s*```\s*$", "")
    ).withColumn("error", F.col("r.errorMessage")).drop("r")
