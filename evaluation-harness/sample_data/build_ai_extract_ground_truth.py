"""Unflatten paystub_ground_truth.csv into the nested ai_extract shape.

Reads the flattened ground-truth CSV (one column per field, `__` marking
object nesting) and emits JSONL (one JSON object per line) with:

    {"doc_id": ..., "ground_truth": {...}}

JSONL is used instead of CSV because `ground_truth` is itself JSON — CSV would
force double-quote escaping that Spark/Delta ingestion tends to mangle. Read it
directly with `spark.read.json(path)`; `ground_truth` lands as a struct (no
second parse). Recipe 00 joins this JSONL with the extraction output to build the
nested eval table.

`ground_truth` mirrors the entities `ai_extract` would produce (see
ai-extract-word-level-citation/notebooks/01_parse_extract_citations.py):
nested objects for `gross`/`net_pay`, numeric fields cast to numbers, and
blank fields dropped.

Usage:
    uv run build_ai_extract_ground_truth.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
FLAT_CSV = HERE / "paystub_ground_truth.csv"
OUT_JSONL = HERE / "paystub_ground_truth_ai_extract.jsonl"

# Numeric fields (typed `double` in the notebook's FIELD_SPECS).
NUMBER_FIELDS = {
    "gross__current_amount",
    "gross__year_to_date_amount",
    "net_pay__current_amount",
    "net_pay__year_to_date_amount",
}


def to_number(raw: str) -> float:
    """Parse '1,884.00' -> 1884.0; return int when the value is whole."""
    value = float(raw.replace(",", ""))
    return int(value) if value.is_integer() else value


def unflatten(row: dict) -> dict:
    """Turn a flat row (minus doc_id) into a nested entity dict."""
    nested: dict = {}
    for col, raw in row.items():
        if col == "doc_id" or raw is None or raw.strip() == "":
            continue  # skip key column and blank fields (matches "leave blank")
        value = to_number(raw) if col in NUMBER_FIELDS else raw
        parts = col.split("__")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def main() -> None:
    with FLAT_CSV.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("doc_id")]  # drop blank lines

    records = [{"doc_id": r["doc_id"], "ground_truth": unflatten(r)} for r in rows]

    # JSONL — read natively with spark.read.json(path); ground_truth is a struct.
    with OUT_JSONL.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
