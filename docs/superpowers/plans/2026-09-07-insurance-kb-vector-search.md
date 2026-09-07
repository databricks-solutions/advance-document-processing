# Insurance KB Vector-Search Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reference/demo pipeline that ingests insurance PDFs from a UC Volume and produces two Databricks Vector Search indexes via `ai_parse_document → ai_classify → ai_prep_search → CREATE INDEX`.

**Architecture:** Batch medallion pipeline shipped in two flavors (interactive notebooks + a batch Databricks Asset Bundle), mirroring `ai-extract-word-level-citation/`. A single document-level `ai_classify` decision routes each document's chunks into one of two gold Delta tables (`reference` / `claims`), each backed by its own Delta Sync index with managed embeddings. Pure SQL-string/config helpers are extracted into a co-located module and unit-tested with pytest; Databricks stage files are verified with `databricks bundle validate`.

**Tech Stack:** Databricks AI Functions (`ai_parse_document`, `ai_classify`, `ai_prep_search`), Databricks Vector Search (Delta Sync, managed embeddings), Databricks Asset Bundles (serverless env 5), PySpark/Spark SQL, `databricks-sdk` (WorkspaceClient VS APIs), reportlab (synthetic PDFs), pytest (`uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-09-07-insurance-kb-vector-search-design.md`

## Global Constraints

*(Every task's requirements implicitly include this section. Values are verbatim.)*

- **Project dir:** `insurance-kb-vector-search/`
- **UC target:** catalog `fins_genai`, schema `unstructured_documents`, volume `pdf_examples`, volume subdir `insurance_docs`, table prefix `insurance_kb`
- **Source PDFs path:** `/Volumes/fins_genai/unstructured_documents/pdf_examples/insurance_docs/`
- **CLI profile for ALL `databricks` commands:** `fevm-classic-stable`
- **Compute:** serverless environment version **5** (`ai_prep_search` needs DBR 18.2+/env v3+; `ai_parse_document` needs 17.3+)
- **Vector Search:** endpoint `insurance_kb_vs`, type `STANDARD`, `pipeline_type` `TRIGGERED`, embedding model `databricks-gte-large-en`, embed column `chunk_to_embed`, primary key `chunk_id`, return column `chunk_to_retrieve`
- **Five doc types → domain:** `policy_document`→reference, `endorsement`→reference, `underwriting_guideline`→reference, `fnol_claim_form`→claims, `adjuster_report`→claims
- **Batch only** (no streaming/Auto Loader); pipeline **stops at index creation** (no query smoke test, no `ai_extract`, no accuracy-scoring stage)
- **Synthetic data:** fully fictional; **15 PDFs** (3 per type); deterministic (seeded)
- **Python:** always `uv run`; keep `pyproject.toml` minimal (do not add reportlab as a project dep — invoke the generator with `uv run --with reportlab`)
- **Style:** no excessive try/except; match the surrounding repo's naming and idioms

**TDD note:** Task 1 (pure helpers) uses full red→green pytest TDD. The synthetic generator (Task 2) and the Databricks stage files (Tasks 3–7) cannot be exercised by local pytest; their verification steps are `uv run --with reportlab python ...` output assertions and `databricks bundle validate --profile fevm-classic-stable` respectively. A real workspace run is the user's call at execution time (note the default CLI token is expired; `fevm-classic-stable` must be authenticated).

---

## File Structure

```
insurance-kb-vector-search/
├── README.md                         # Task 8
├── notebooks/                        # Task 7 (interactive/illustrative)
│   ├── 01_parse_documents.py
│   ├── 02_classify_documents.py
│   ├── 03_prep_search_chunks.py
│   ├── 04_route_gold_tables.py
│   └── 05_create_vector_indexes.py
├── bundle/                           # Tasks 3–6 (production batch DAB)
│   ├── databricks.yml                # Task 3
│   ├── resources/
│   │   └── insurance_kb.job.yml      # Task 3
│   └── src/
│       ├── sql_builders.py           # Task 1 (pure helpers)
│       ├── 01_bronze_parse.py        # Task 4
│       ├── 02_silver_classify.py     # Task 4
│       ├── 03_prep_search.py         # Task 5
│       ├── 04_gold_route.py          # Task 5
│       └── 05_create_indexes.py      # Task 6
└── tests/
    └── test_sql_builders.py          # Task 1

scripts/generate_sample_insurance_docs.py   # Task 2
pyproject.toml                               # Task 1 (add testpath + pythonpath)
README.md                                    # Task 8 (add project-table row)
```

---

### Task 1: Pure helpers + unit tests

The testable core: the classification label set, the `doc_type→domain` routing map, the `ai_classify` SQL builder (with quote escaping), and the gold-table / index name builders. Everything else imports this module. Full TDD.

**Files:**
- Create: `insurance-kb-vector-search/bundle/src/sql_builders.py`
- Create: `insurance-kb-vector-search/tests/test_sql_builders.py`
- Modify: `pyproject.toml` (add testpath + pythonpath entries)

**Interfaces:**
- Produces (consumed by Tasks 4, 5, 6, 7):
  - `DOC_TYPE_LABELS: dict[str, str]` — label → description (5 entries)
  - `DOC_TYPE_TO_DOMAIN: dict[str, str]` — label → `"reference"`|`"claims"` (5 entries)
  - `CLASSIFY_INSTRUCTIONS: str`
  - `labels_json() -> str` — JSON object string of `DOC_TYPE_LABELS` for `ai_classify`
  - `domain_for(doc_type: str) -> str` — returns domain, or `"unknown"` for unrecognized
  - `classify_expr(input_col: str, labels_sql_json: str, instructions: str) -> str` — builds the `ai_classify(...)` SQL, doubling single quotes in the two string literals
  - `domain_case_expr(doc_type_sql: str) -> str` — builds a SQL `CASE` mapping a doc_type SQL expression to its domain
  - `gold_table_name(prefix: str, domain: str) -> str` → `f"{prefix}_{domain}_chunks"`
  - `index_name(catalog: str, schema: str, prefix: str, domain: str) -> str` → `f"{catalog}.{schema}.{prefix}_{domain}_index"`

- [ ] **Step 1: Write the failing tests**

Create `insurance-kb-vector-search/tests/test_sql_builders.py`:

```python
"""Unit tests for the insurance-KB SQL/config builders (bundle/src/sql_builders.py).

Pure string / dict construction — assert generated SQL and maps, not Spark."""

import json

import sql_builders as S


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


# --- name builders ---------------------------------------------------------

def test_gold_table_name():
    assert S.gold_table_name("insurance_kb", "reference") == "insurance_kb_reference_chunks"
    assert S.gold_table_name("insurance_kb", "claims") == "insurance_kb_claims_chunks"


def test_index_name_fully_qualified():
    assert S.index_name("fins_genai", "unstructured_documents", "insurance_kb", "claims") == \
        "fins_genai.unstructured_documents.insurance_kb_claims_index"
```

- [ ] **Step 2: Wire pytest discovery — modify `pyproject.toml`**

In `[tool.pytest.ini_options]`, add the new project to both lists (keep existing entries):

```toml
testpaths = [
    "ai-extract-word-level-citation/tests",
    "document-page-classify-extraction/tests",
    "evaluation-harness/tests",
    "insurance-kb-vector-search/tests",
]
pythonpath = [
    "ai-extract-word-level-citation/bundle/src",
    "document-page-classify-extraction/bundle/src",
    "evaluation-harness/src",
    "insurance-kb-vector-search/bundle/src",
]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest insurance-kb-vector-search/tests/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sql_builders'` (module not yet created).

- [ ] **Step 4: Write minimal implementation**

Create `insurance-kb-vector-search/bundle/src/sql_builders.py`:

```python
"""Pure, unit-testable builders for the insurance-KB vector-search pipeline.

These run on the driver — they assemble the classification label set, the
doc_type→domain routing map, and SQL-string / name fragments that the stage
notebooks feed to spark.sql / expr. Keeping them here lets the construction be
tested without Spark (see tests/test_sql_builders.py)."""

import json

# Five document types, each with a description to sharpen ai_classify accuracy.
DOC_TYPE_LABELS = {
    "policy_document": (
        "An insurance policy: the declarations page and/or policy wording for a "
        "homeowners or auto policy. Named insured, policy number, policy period, "
        "coverages and coverage limits, deductibles, premiums, and exclusions."
    ),
    "endorsement": (
        "An endorsement or rider that amends an existing base policy — e.g. a "
        "water-backup endorsement or scheduled-personal-property (jewelry) rider. "
        "References a base policy number and states the specific coverage added, "
        "changed, or removed; not a standalone full policy."
    ),
    "underwriting_guideline": (
        "Internal underwriting guidelines: rules for risk acceptance, referral, or "
        "pricing (e.g. maximum roof age, coastal/wildfire exposure, prior-claims "
        "thresholds). A reference rulebook, not tied to one named insured."
    ),
    "fnol_claim_form": (
        "A First Notice of Loss (FNOL) / claim report form. Structured fields: "
        "claim number, claimant, date of loss, policy number, peril/cause of loss, "
        "loss location, and a short loss description."
    ),
    "adjuster_report": (
        "A claims adjuster's inspection or investigation report: narrative findings, "
        "damage assessment, cause-of-loss analysis, photos/estimates references, and "
        "a recommended reserve or settlement."
    ),
}

# The single routing decision the pipeline makes.
DOC_TYPE_TO_DOMAIN = {
    "policy_document": "reference",
    "endorsement": "reference",
    "underwriting_guideline": "reference",
    "fnol_claim_form": "claims",
    "adjuster_report": "claims",
}

CLASSIFY_INSTRUCTIONS = (
    "These are insurance documents for an adjuster/underwriter knowledge base. "
    "Each document is a single type (not a mixed packet). Classify by the document's "
    "dominant purpose. Use 'policy_document' for a full policy/declarations, "
    "'endorsement' only when it amends an existing policy, 'underwriting_guideline' "
    "for internal risk/pricing rulebooks, 'fnol_claim_form' for a first-notice-of-loss "
    "report, and 'adjuster_report' for an adjuster's inspection/investigation findings."
)


def labels_json():
    """JSON object string (label -> description) for ai_classify."""
    return json.dumps(DOC_TYPE_LABELS)


def domain_for(doc_type):
    """Retrieval domain for a doc_type; 'unknown' if unrecognized."""
    return DOC_TYPE_TO_DOMAIN.get(doc_type, "unknown")


def classify_expr(input_col, labels_sql_json, instructions):
    """Build the ai_classify(...) SQL. Doubles single quotes in string literals."""
    labels_sql = labels_sql_json.replace("'", "''")
    instr_sql = instructions.replace("'", "''")
    return f"""
        ai_classify(
            {input_col},
            '{labels_sql}',
            map('instructions', '{instr_sql}')
        )
    """


def domain_case_expr(doc_type_sql):
    """SQL CASE mapping a doc_type SQL expression to its retrieval domain."""
    ref = [k for k, v in DOC_TYPE_TO_DOMAIN.items() if v == "reference"]
    clm = [k for k, v in DOC_TYPE_TO_DOMAIN.items() if v == "claims"]
    ref_in = ", ".join(f"'{k}'" for k in ref)
    clm_in = ", ".join(f"'{k}'" for k in clm)
    return f"""
        CASE
            WHEN {doc_type_sql} IN ({ref_in}) THEN 'reference'
            WHEN {doc_type_sql} IN ({clm_in}) THEN 'claims'
            ELSE 'unknown'
        END
    """


def gold_table_name(prefix, domain):
    """Unqualified gold chunk-table name for a domain."""
    return f"{prefix}_{domain}_chunks"


def index_name(catalog, schema, prefix, domain):
    """Fully-qualified Vector Search index name for a domain."""
    return f"{catalog}.{schema}.{prefix}_{domain}_index"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest insurance-kb-vector-search/tests/ -v`
Expected: PASS (all tests green). Also run the full suite to confirm no regression: `uv run pytest -q`.

- [ ] **Step 6: Commit**

```bash
git add insurance-kb-vector-search/bundle/src/sql_builders.py \
        insurance-kb-vector-search/tests/test_sql_builders.py pyproject.toml
git commit -m "feat(insurance-kb): pure SQL/config helpers + unit tests"
```

---

### Task 2: Synthetic insurance PDF generator + sample data

Generate 15 fictional insurance PDFs (3 per type) into `sample_data/`, plus a ground-truth CSV. Mirror the reportlab plumbing (styles, footer, `SimpleDocTemplate`) from `scripts/generate_sample_loan_files.py` — read that file first and reuse its patterns.

**Files:**
- Create: `scripts/generate_sample_insurance_docs.py`
- Create (generated output): `insurance-kb-vector-search/sample_data/*.pdf`, `insurance-kb-vector-search/sample_data/ground_truth_doc_types.csv`

**Interfaces:**
- Produces: 15 PDFs consumed at runtime by the pipeline (Task 4+), and `ground_truth_doc_types.csv` with header `filename,doc_type,domain`.
- Consumes: `sql_builders.DOC_TYPE_TO_DOMAIN` is **not** imported here (scripts stay standalone/reportlab-only). The domain column is written from a local literal map that must match the spec's 5-type mapping.

- [ ] **Step 1: Read the reference generator**

Read `scripts/generate_sample_loan_files.py` in full to reuse: the `reportlab` imports, `styles`/`ParagraphStyle` block, `money()`/`pct()` helpers, the page footer/header (`onPage` callback), and the `SimpleDocTemplate(...).build(flowables)` call.

- [ ] **Step 2: Write the generator**

Create `scripts/generate_sample_insurance_docs.py` with this structure (fill the per-type flowable builders using the loan generator's style helpers; content specs below):

```python
"""Generate synthetic, fully fictional insurance PDFs for the
insurance-kb-vector-search pipeline.

3 documents per type x 5 types = 15 PDFs, plus ground_truth_doc_types.csv.
Deterministic (seeded) so filenames are stable across regenerations.

Run (from repo root):
    uv run --with reportlab python scripts/generate_sample_insurance_docs.py
Output: insurance-kb-vector-search/sample_data/<doc_type>_<n>_<slug>.pdf
        insurance-kb-vector-search/sample_data/ground_truth_doc_types.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# reportlab imports (mirror generate_sample_loan_files.py)
# from reportlab.lib import colors ... SimpleDocTemplate, Paragraph, Table, ...

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "insurance-kb-vector-search"
    / "sample_data"
)

# Local literal map (scripts stay standalone; must match the spec's mapping).
DOC_TYPE_TO_DOMAIN = {
    "policy_document": "reference",
    "endorsement": "reference",
    "underwriting_guideline": "reference",
    "fnol_claim_form": "claims",
    "adjuster_report": "claims",
}

# Fully fictional pools (no real carriers/people/addresses).
CARRIERS = ["Harbor Mutual Insurance", "Summit Ridge Casualty", "Blue Meridian Insurance Group"]
INSUREDS = [ ... ]   # 3+ fictional named insureds w/ address, policy no.
PERILS = ["Water damage (plumbing)", "Wind/hail", "Fire", "Theft", "Vehicle collision"]

DOC_BUILDERS = {
    "policy_document": build_policy_document,
    "endorsement": build_endorsement,
    "underwriting_guideline": build_underwriting_guideline,
    "fnol_claim_form": build_fnol_claim_form,
    "adjuster_report": build_adjuster_report,
}


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def build_manifest(seed: int = 7):
    """Return an ordered list of (filename, doc_type, domain) — 3 per type."""
    rng = random.Random(seed)
    rows = []
    for doc_type in DOC_TYPE_TO_DOMAIN:
        for n in range(1, 4):
            tag = slug(rng.choice(CARRIERS if "policy" in doc_type or "endorse" in doc_type
                                  else PERILS))[:18]
            fname = f"{doc_type}_{n}_{tag}.pdf"
            rows.append((fname, doc_type, DOC_TYPE_TO_DOMAIN[doc_type]))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    rng = random.Random(7)
    for fname, doc_type, _domain in manifest:
        flowables = DOC_BUILDERS[doc_type](rng)   # returns a list of reportlab flowables
        # SimpleDocTemplate(str(OUT_DIR / fname), pagesize=LETTER, ...).build(flowables)
    with open(OUT_DIR / "ground_truth_doc_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "doc_type", "domain"])
        w.writerows(manifest)
    print(f"Wrote {len(manifest)} PDFs + ground_truth_doc_types.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
```

Per-type content (each builder returns a list of flowables; 1–3 pages; type-consistent so classification is non-trivial):

- **`build_policy_document`** — carrier header; "HOMEOWNERS POLICY DECLARATIONS"; named insured + address + policy number + policy period; a coverages table (Dwelling / Other Structures / Personal Property / Liability / Medical) with limits and deductibles; a premium summary; a short exclusions paragraph.
- **`build_endorsement`** — "POLICY ENDORSEMENT"; references a base policy number; endorsement number + effective date; a paragraph describing the added/changed coverage (water-backup sublimit, or scheduled jewelry with itemized values table); "all other terms unchanged" boilerplate.
- **`build_underwriting_guideline`** — "UNDERWRITING GUIDELINES — Personal Lines"; numbered rule sections (Eligibility, Roof Age, Coastal/Wildfire Exposure, Prior Claims, Referral Triggers) as bullet/rule lists; no named insured.
- **`build_fnol_claim_form`** — "FIRST NOTICE OF LOSS"; a form-style two-column field grid (Claim No., Policy No., Claimant, Date of Loss, Reported Date, Peril, Loss Location) + a free-text "Description of Loss" box.
- **`build_adjuster_report`** — "CLAIM INSPECTION REPORT"; claim/adjuster header; narrative sections (Summary, Site Inspection Findings, Cause of Loss, Damage Assessment, Recommended Reserve) with a small line-item damage/estimate table.

- [ ] **Step 3: Generate the artifacts**

Run: `uv run --with reportlab python scripts/generate_sample_insurance_docs.py`
Expected: prints `Wrote 15 PDFs + ground_truth_doc_types.csv ...`.

- [ ] **Step 4: Verify counts and ground-truth CSV**

Run:
```bash
ls insurance-kb-vector-search/sample_data/*.pdf | wc -l          # expect 15
tail -n +2 insurance-kb-vector-search/sample_data/ground_truth_doc_types.csv | wc -l   # expect 15
cut -d, -f2 insurance-kb-vector-search/sample_data/ground_truth_doc_types.csv | tail -n +2 | sort | uniq -c   # expect 3 each x 5 types
```
Expected: 15 PDFs; CSV has 15 data rows; exactly 3 rows per doc_type. Open one PDF per type to eyeball it renders and reads as that type.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_sample_insurance_docs.py insurance-kb-vector-search/sample_data/
git commit -m "feat(insurance-kb): synthetic insurance PDF generator + sample data"
```

---

### Task 3: Bundle configuration (databricks.yml + job) + validate

Scaffold the batch DAB so later stage files have a job to attach to. Manual-trigger job (no schedule), serverless env 5, five sequential tasks.

**Files:**
- Create: `insurance-kb-vector-search/bundle/databricks.yml`
- Create: `insurance-kb-vector-search/bundle/resources/insurance_kb.job.yml`

**Interfaces:**
- Produces: bundle variables `catalog, schema, volume, volume_subdir, table_prefix, vs_endpoint, embedding_model` consumed as `base_parameters` by every stage notebook (Tasks 4–6). Task keys: `bronze_parse → silver_classify → prep_search → gold_route → create_indexes`.

- [ ] **Step 1: Write `databricks.yml`**

Create `insurance-kb-vector-search/bundle/databricks.yml`:

```yaml
bundle:
  name: insurance_kb_vector_search

variables:
  catalog:
    description: Unity Catalog name
    default: fins_genai
  schema:
    description: Schema name
    default: unstructured_documents
  volume:
    description: Volume name containing source PDFs
    default: pdf_examples
  volume_subdir:
    description: Subfolder under the volume that holds the source insurance PDFs
    default: insurance_docs
  table_prefix:
    description: Prefix applied to all bronze/silver/gold tables and indexes
    default: insurance_kb
  vs_endpoint:
    description: Vector Search endpoint name (shared by both indexes)
    default: insurance_kb_vs
  embedding_model:
    description: Databricks-managed embedding model endpoint
    default: databricks-gte-large-en

include:
  - resources/*.yml

targets:
  dev:
    mode: development
    default: true
  prod:
    mode: production
```

- [ ] **Step 2: Write the job resource**

Create `insurance-kb-vector-search/bundle/resources/insurance_kb.job.yml`:

```yaml
resources:
  jobs:
    insurance_kb_vector_search:
      name: "[${bundle.target}] Insurance KB Parse-Classify-Prep-Index"

      max_concurrent_runs: 1

      # Serverless notebook environment pinned to env version 5
      # (ai_parse_document needs env v3+; ai_prep_search needs env v3+ / DBR 18.2+)
      environments:
        - environment_key: serverless_v5
          spec:
            client: "5"

      tasks:
        - task_key: bronze_parse
          environment_key: serverless_v5
          notebook_task:
            notebook_path: ../src/01_bronze_parse.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              volume_subdir: ${var.volume_subdir}
              table_prefix: ${var.table_prefix}

        - task_key: silver_classify
          environment_key: serverless_v5
          depends_on:
            - task_key: bronze_parse
          notebook_task:
            notebook_path: ../src/02_silver_classify.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              volume_subdir: ${var.volume_subdir}
              table_prefix: ${var.table_prefix}

        - task_key: prep_search
          environment_key: serverless_v5
          depends_on:
            - task_key: silver_classify
          notebook_task:
            notebook_path: ../src/03_prep_search.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              volume_subdir: ${var.volume_subdir}
              table_prefix: ${var.table_prefix}

        - task_key: gold_route
          environment_key: serverless_v5
          depends_on:
            - task_key: prep_search
          notebook_task:
            notebook_path: ../src/04_gold_route.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              volume_subdir: ${var.volume_subdir}
              table_prefix: ${var.table_prefix}

        - task_key: create_indexes
          environment_key: serverless_v5
          depends_on:
            - task_key: gold_route
          notebook_task:
            notebook_path: ../src/05_create_indexes.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              table_prefix: ${var.table_prefix}
              vs_endpoint: ${var.vs_endpoint}
              embedding_model: ${var.embedding_model}
```

- [ ] **Step 3: Validate the bundle**

Run: `cd insurance-kb-vector-search/bundle && databricks bundle validate --profile fevm-classic-stable`
Expected: validation succeeds (it will warn/succeed even though the `../src/*.py` notebooks don't exist yet — validation checks config, not notebook presence; if it errors on missing notebooks, that resolves after Tasks 4–6). Record any error text.

- [ ] **Step 4: Commit**

```bash
git add insurance-kb-vector-search/bundle/databricks.yml \
        insurance-kb-vector-search/bundle/resources/insurance_kb.job.yml
git commit -m "feat(insurance-kb): batch DAB config + job resource"
```

---

### Task 4: Bundle src — bronze parse + silver classify

The first two production stages: parse PDFs to a bronze VARIANT table, then classify each document and derive its domain. Verified with `bundle validate`.

**Files:**
- Create: `insurance-kb-vector-search/bundle/src/01_bronze_parse.py`
- Create: `insurance-kb-vector-search/bundle/src/02_silver_classify.py`

**Interfaces:**
- Consumes: `sql_builders` (`labels_json`, `classify_expr`, `domain_case_expr`, `CLASSIFY_INSTRUCTIONS`); bundle `base_parameters`.
- Produces tables: `{prefix}_bronze_parsed` (cols `source_path STRING, parsed VARIANT, parse_error STRING, ingested_at TIMESTAMP`) and `{prefix}_silver_classified` (cols `source_path, parsed, classification_raw VARIANT, doc_type STRING, domain STRING`).

- [ ] **Step 1: Write `01_bronze_parse.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: `ai_parse_document`
# MAGIC Batch parse of insurance PDFs from the source volume subdir into a bronze
# MAGIC VARIANT table, keyed by source path. Requires serverless env v3+.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "insurance_docs")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog       = dbutils.widgets.get("catalog")
schema        = dbutils.widgets.get("schema")
volume        = dbutils.widgets.get("volume")
volume_subdir = dbutils.widgets.get("volume_subdir")
table_prefix  = dbutils.widgets.get("table_prefix")

source_path  = f"/Volumes/{catalog}/{schema}/{volume}/{volume_subdir}"
bronze_table = f"{catalog}.{schema}.{table_prefix}_bronze_parsed"
print(f"source_path  = {source_path}")
print(f"bronze_table = {bronze_table}")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {bronze_table} AS
WITH parsed AS (
  SELECT
    path AS source_path,
    ai_parse_document(content, map('version', '2.0')) AS parsed
  FROM read_files('{source_path}/', format => 'binaryFile')
  WHERE lower(path) LIKE '%.pdf'
)
SELECT
  source_path,
  parsed,
  variant_get(parsed, '$.error_status', 'STRING') AS parse_error,
  current_timestamp() AS ingested_at
FROM parsed
""")

cnt = spark.table(bronze_table).count()
print(f"bronze rows = {cnt}")
dbutils.notebook.exit(f"bronze_rows={cnt}")
```

- [ ] **Step 2: Write `02_silver_classify.py`**

Classify on the parsed VARIANT directly (documented; demo docs are small so the 128k-token cap is not a concern), then derive `domain` with the tested `domain_case_expr` over the extracted label.

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: `ai_classify` document type → domain
# MAGIC Assigns each document one of five labels and maps it to a retrieval
# MAGIC domain (reference / claims). One classify call per document.

# COMMAND ----------

import sys
sys.path.append("../")  # so `import sql_builders` resolves when run from src/
import sql_builders as S

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("volume", "pdf_examples")
dbutils.widgets.text("volume_subdir", "insurance_docs")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

bronze_table = f"{catalog}.{schema}.{table_prefix}_bronze_parsed"
silver_table = f"{catalog}.{schema}.{table_prefix}_silver_classified"
print(f"silver_table = {silver_table}")

# COMMAND ----------

classify_sql = S.classify_expr("parsed", S.labels_json(), S.CLASSIFY_INSTRUCTIONS)
domain_sql   = S.domain_case_expr("classification_raw:response[0]::string")

spark.sql(f"""
CREATE OR REPLACE TABLE {silver_table} AS
WITH classified AS (
  SELECT
    source_path,
    parsed,
    {classify_sql} AS classification_raw
  FROM {bronze_table}
  WHERE parse_error IS NULL
)
SELECT
  source_path,
  parsed,
  classification_raw,
  classification_raw:response[0]::string AS doc_type,
  {domain_sql} AS domain
FROM classified
""")

from pyspark.sql.functions import col
counts = spark.table(silver_table).groupBy("doc_type", "domain").count()
counts.show(truncate=False)
n = spark.table(silver_table).count()
dbutils.notebook.exit(f"classified={n}")
```

> **Note on `sys.path.append("../")`:** the bundle uploads `src/` as a workspace directory; `sql_builders.py` sits beside the stage notebooks, so `../` from the notebook's run dir resolves to `src/`. If the executor finds imports don't resolve in the serverless run, the fallback is `sys.path.append(os.path.dirname(...))` using the notebook path — validate during the first real run.

- [ ] **Step 3: Validate the bundle**

Run: `cd insurance-kb-vector-search/bundle && databricks bundle validate --profile fevm-classic-stable`
Expected: succeeds.

- [ ] **Step 4: Commit**

```bash
git add insurance-kb-vector-search/bundle/src/01_bronze_parse.py \
        insurance-kb-vector-search/bundle/src/02_silver_classify.py
git commit -m "feat(insurance-kb): bronze parse + silver classify stages"
```

---

### Task 5: Bundle src — prep_search + gold route

Chunk with `ai_prep_search`, then route chunks into two CDF-enabled gold tables by domain.

**Files:**
- Create: `insurance-kb-vector-search/bundle/src/03_prep_search.py`
- Create: `insurance-kb-vector-search/bundle/src/04_gold_route.py`

**Interfaces:**
- Consumes: `{prefix}_silver_classified`; `sql_builders.gold_table_name`.
- Produces: `{prefix}_prepped_chunks` (cols `chunk_id STRING, chunk_position INT, chunk_to_retrieve STRING, chunk_to_embed STRING, doc_type STRING, domain STRING, source_path STRING, prepped_at TIMESTAMP`), and two gold tables `{prefix}_reference_chunks` / `{prefix}_claims_chunks` (same columns minus `domain`, CDF enabled, `chunk_id` NOT NULL + PK).

- [ ] **Step 1: Write `03_prep_search.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Prep: `ai_prep_search` semantic chunking + enrichment
# MAGIC Turns each parsed document into RAG-ready chunks. Embed `chunk_to_embed`,
# MAGIC return `chunk_to_retrieve`. Requires serverless env v3+ / DBR 18.2+.

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

silver_table  = f"{catalog}.{schema}.{table_prefix}_silver_classified"
prepped_table = f"{catalog}.{schema}.{table_prefix}_prepped_chunks"
print(f"prepped_table = {prepped_table}")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {prepped_table} AS
WITH prepped AS (
  SELECT source_path, doc_type, domain, ai_prep_search(parsed) AS prep
  FROM {silver_table}
  WHERE doc_type IS NOT NULL AND domain <> 'unknown'
),
chunks AS (
  SELECT
    source_path, doc_type, domain,
    explode(variant_get(prep, '$.chunks', 'ARRAY<VARIANT>')) AS chunk
  FROM prepped
)
SELECT
  variant_get(chunk, '$.chunk_id',          'STRING') AS chunk_id,
  variant_get(chunk, '$.chunk_position',    'INT')    AS chunk_position,
  variant_get(chunk, '$.chunk_to_retrieve', 'STRING') AS chunk_to_retrieve,
  variant_get(chunk, '$.chunk_to_embed',    'STRING') AS chunk_to_embed,
  doc_type, domain, source_path,
  current_timestamp() AS prepped_at
FROM chunks
""")

cnt = spark.table(prepped_table).count()
print(f"prepped chunk rows = {cnt}")
dbutils.notebook.exit(f"chunks={cnt}")
```

- [ ] **Step 2: Write `04_gold_route.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Gold: route chunks into two domain tables
# MAGIC Splits prepped chunks by domain into reference / claims gold tables,
# MAGIC each Change-Data-Feed enabled for Vector Search Delta Sync.

# COMMAND ----------

import sys
sys.path.append("../")
import sql_builders as S

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
table_prefix = dbutils.widgets.get("table_prefix")

prepped_table = f"{catalog}.{schema}.{table_prefix}_prepped_chunks"

# COMMAND ----------

for domain in ("reference", "claims"):
    gold = f"{catalog}.{schema}.{S.gold_table_name(table_prefix, domain)}"
    print(f"building {gold} ...")
    spark.sql(f"""
    CREATE OR REPLACE TABLE {gold}
    TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
    SELECT chunk_id, chunk_position, chunk_to_retrieve, chunk_to_embed,
           doc_type, source_path, prepped_at
    FROM {prepped_table}
    WHERE domain = '{domain}'
    """)
    # Vector Search Delta Sync wants a non-null primary key.
    spark.sql(f"ALTER TABLE {gold} ALTER COLUMN chunk_id SET NOT NULL")
    spark.sql(f"ALTER TABLE {gold} ADD CONSTRAINT {table_prefix}_{domain}_pk PRIMARY KEY (chunk_id)")
    print(f"  rows = {spark.table(gold).count()}")

dbutils.notebook.exit("gold_route_done")
```

- [ ] **Step 3: Validate the bundle**

Run: `cd insurance-kb-vector-search/bundle && databricks bundle validate --profile fevm-classic-stable`
Expected: succeeds.

- [ ] **Step 4: Commit**

```bash
git add insurance-kb-vector-search/bundle/src/03_prep_search.py \
        insurance-kb-vector-search/bundle/src/04_gold_route.py
git commit -m "feat(insurance-kb): prep_search chunking + gold domain routing"
```

---

### Task 6: Bundle src — create Vector Search indexes

Final production stage: ensure the shared endpoint exists, then create/refresh one Delta Sync index per gold table (managed embeddings, TRIGGERED).

**Files:**
- Create: `insurance-kb-vector-search/bundle/src/05_create_indexes.py`

**Interfaces:**
- Consumes: `{prefix}_reference_chunks`, `{prefix}_claims_chunks`; `sql_builders.gold_table_name`, `sql_builders.index_name`; params `vs_endpoint`, `embedding_model`.
- Produces: endpoint `insurance_kb_vs`; indexes `{prefix}_reference_index`, `{prefix}_claims_index`.

- [ ] **Step 1: Write `05_create_indexes.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Index: create/refresh Delta Sync Vector Search indexes
# MAGIC One index per gold table, managed embeddings (databricks-gte-large-en),
# MAGIC TRIGGERED sync. Embeds `chunk_to_embed`; return `chunk_to_retrieve` at query time.

# COMMAND ----------

import sys, time
sys.path.append("../")
import sql_builders as S
from databricks.sdk import WorkspaceClient

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_prefix", "insurance_kb")
dbutils.widgets.text("vs_endpoint", "insurance_kb_vs")
dbutils.widgets.text("embedding_model", "databricks-gte-large-en")

catalog         = dbutils.widgets.get("catalog")
schema          = dbutils.widgets.get("schema")
table_prefix    = dbutils.widgets.get("table_prefix")
vs_endpoint     = dbutils.widgets.get("vs_endpoint")
embedding_model = dbutils.widgets.get("embedding_model")

w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md ## Ensure the endpoint exists (create if missing) and is online

# COMMAND ----------

existing = [e.name for e in (w.vector_search_endpoints.list_endpoints() or [])]
if vs_endpoint not in existing:
    print(f"creating endpoint {vs_endpoint} ...")
    w.vector_search_endpoints.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")

# Poll until ONLINE (endpoint creation is asynchronous).
for _ in range(60):
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=vs_endpoint)
    state = ep.endpoint_status.state if ep.endpoint_status else None
    print(f"endpoint state = {state}")
    if str(state) == "ONLINE":
        break
    time.sleep(30)

# COMMAND ----------
# MAGIC %md ## Create or sync one index per gold table

# COMMAND ----------

for domain in ("reference", "claims"):
    src_table = f"{catalog}.{schema}.{S.gold_table_name(table_prefix, domain)}"
    idx = S.index_name(catalog, schema, table_prefix, domain)
    existing_idx = [i.name for i in (w.vector_search_indexes.list_indexes(endpoint_name=vs_endpoint).vector_indexes or [])]
    if idx in existing_idx:
        print(f"syncing existing index {idx} ...")
        w.vector_search_indexes.sync_index(index_name=idx)
    else:
        print(f"creating index {idx} on {src_table} ...")
        w.vector_search_indexes.create_index(
            name=idx,
            endpoint_name=vs_endpoint,
            primary_key="chunk_id",
            index_type="DELTA_SYNC",
            delta_sync_index_spec={
                "source_table": src_table,
                "embedding_source_columns": [
                    {"name": "chunk_to_embed",
                     "embedding_model_endpoint_name": embedding_model}
                ],
                "pipeline_type": "TRIGGERED",
            },
        )

print("index creation/sync submitted.")
dbutils.notebook.exit("indexes_done")
```

> **API note for the executor:** the exact attribute names on the SDK response objects (`endpoint_status.state`, `list_indexes(...).vector_indexes`, `.name`) should be confirmed against the installed `databricks-sdk` version during the first run; adjust the status-poll/list accessors if the SDK differs. The `create_index` call signature above matches the databricks-vector-search skill reference.

- [ ] **Step 2: Validate the bundle**

Run: `cd insurance-kb-vector-search/bundle && databricks bundle validate --profile fevm-classic-stable`
Expected: succeeds, and now references all five existing `../src/*.py` notebooks.

- [ ] **Step 3: Commit**

```bash
git add insurance-kb-vector-search/bundle/src/05_create_indexes.py
git commit -m "feat(insurance-kb): create/refresh Delta Sync vector indexes"
```

---

### Task 7: Interactive notebooks (illustrative flavor)

Port each src stage into a step notebook under `notebooks/` — the repo's illustrative/interactive counterpart to the production bundle. Same SQL/logic as the corresponding `bundle/src/` file (copy the SQL strings verbatim), but with markdown narration and `.display()` calls instead of `dbutils.notebook.exit(...)` chaining, and without the bundle `base_parameters` wiring (use `dbutils.widgets` defaults directly).

**Files:**
- Create: `insurance-kb-vector-search/notebooks/01_parse_documents.py`
- Create: `insurance-kb-vector-search/notebooks/02_classify_documents.py`
- Create: `insurance-kb-vector-search/notebooks/03_prep_search_chunks.py`
- Create: `insurance-kb-vector-search/notebooks/04_route_gold_tables.py`
- Create: `insurance-kb-vector-search/notebooks/05_create_vector_indexes.py`

**Interfaces:**
- Consumes the same tables/params as Tasks 4–6. For the helper import in notebooks 02/04/05, copy the small helper functions inline OR `sys.path.append` to `../bundle/src`; prefer inline copies of the 2–3 needed strings so the notebooks are self-contained demos (the canonical, tested versions live in `bundle/src/sql_builders.py`).

- [ ] **Step 1: Write `01_parse_documents.py`**

Same `CREATE OR REPLACE TABLE {bronze_table} AS ...` SQL as `bundle/src/01_bronze_parse.py` Step 1. Add a leading `%md` cell explaining the parse stage, and after the write add:
```python
spark.table(bronze_table).selectExpr("source_path", "parse_error").display()
```
Drop the `dbutils.notebook.exit(...)` line.

- [ ] **Step 2: Write `02_classify_documents.py`**

Copy the `classify_sql`/`domain_sql` construction and the silver `CREATE OR REPLACE TABLE` from `bundle/src/02_silver_classify.py`, inlining the label JSON + instructions + `domain_case_expr` logic (copy the literal strings/CASE from `sql_builders.py`). Add a `%md` cell and end with:
```python
spark.table(silver_table).selectExpr("source_path", "doc_type", "domain").display()
```

- [ ] **Step 3: Write `03_prep_search_chunks.py`**

Same prep SQL as `bundle/src/03_prep_search.py`. End with `spark.table(prepped_table).display()`.

- [ ] **Step 4: Write `04_route_gold_tables.py`**

Same two-table CDF routing loop as `bundle/src/04_gold_route.py` (inline the `gold_table_name` f-string). End by displaying row counts per gold table.

- [ ] **Step 5: Write `05_create_vector_indexes.py`**

Same WorkspaceClient endpoint-ensure + per-domain create/sync loop as `bundle/src/05_create_indexes.py` (inline the `index_name`/`gold_table_name` f-strings). End with a `%md` cell noting how to query each index (referencing `w.vector_search_indexes.query_index(...)`), but do not add an executable query cell (pipeline stops at index creation per spec).

- [ ] **Step 6: Commit**

```bash
git add insurance-kb-vector-search/notebooks/
git commit -m "feat(insurance-kb): interactive step notebooks"
```

---

### Task 8: Documentation (project README + root table row)

**Files:**
- Create: `insurance-kb-vector-search/README.md`
- Modify: `README.md` (root project table + repo-layout tree)

**Interfaces:** none (docs).

- [ ] **Step 1: Write `insurance-kb-vector-search/README.md`**

Follow the structure of `ai-extract-word-level-citation/README.md`. Sections:
- **Title + one-paragraph overview** — parse → classify → prep_search → two Delta Sync indexes; adjuster/underwriter KB.
- **Architecture** — the Stage 1–5 medallion flow and the two-domain routing (reuse the spec's data-flow diagram).
- **Document types** — the 5 types → 2 domains table.
- **Defaults** — the Global-Constraints values (catalog/schema/volume/subdir/prefix, endpoint, embedding model).
- **Prerequisites** — serverless env 5 / DBR 18.2+; UC + Vector Search enabled; profile `fevm-classic-stable`; region support.
- **Quickstart** —
  1. `uv run --with reportlab python scripts/generate_sample_insurance_docs.py`
  2. Upload: `scripts/upload_pdfs.sh -p fevm-classic-stable -c fins_genai -s unstructured_documents -v pdf_examples -f insurance_docs -i insurance-kb-vector-search/sample_data`
  3. `cd insurance-kb-vector-search/bundle && databricks bundle deploy -t dev --profile fevm-classic-stable`
  4. `databricks bundle run insurance_kb_vector_search -t dev --profile fevm-classic-stable`
- **Testing** — `uv run pytest insurance-kb-vector-search/tests/ -v`.
- **Notebooks vs bundle** — note the bundle `src/` is canonical/tested; notebooks illustrate.

- [ ] **Step 2: Update the root `README.md`**

Add a row to the Projects table:
```
| [`insurance-kb-vector-search/`](./insurance-kb-vector-search/) | **Insurance knowledge-base RAG-prep** pipeline. Parses insurance PDFs with `ai_parse_document`, classifies each document type with `ai_classify`, chunks with `ai_prep_search`, routes chunks by retrieval domain into two gold tables, and builds a Delta Sync **Vector Search** index per domain (reference / claims). Worked example: an adjuster/underwriter knowledge base. |
```
Add to the repo-layout tree:
```
├── insurance-kb-vector-search/          # Parse→classify→prep→two vector indexes (notebooks + batch DAB)
```
Add `generate_sample_insurance_docs.py` to the `scripts/` sub-tree listing. If the Prerequisites section enumerates DBR floors, add the `ai_prep_search` DBR 18.2+ requirement note (it is already implied by env 5).

- [ ] **Step 3: Commit**

```bash
git add insurance-kb-vector-search/README.md README.md
git commit -m "docs(insurance-kb): project README + root table entry"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- §3 doc types + routing map → Task 1 (helpers/tests), Task 2 (synthetic content)
- §4 Stage 1 bronze → Task 4; Stage 2 silver classify → Task 4; Stage 3 prep → Task 5; Stage 4 gold route (2 tables, CDF) → Task 5; Stage 5 indexes → Task 6
- §5 UC naming/variables → Task 3 (bundle vars) + Global Constraints
- §6 deliverables/file tree → Tasks 1–8 (notebooks Task 7, bundle Tasks 3–6, tests Task 1, sample_data Task 2, READMEs Task 8)
- §7 synthetic generator (15 PDFs, ground-truth CSV, seeded) → Task 2
- §8 testing (pure helpers, `uv run pytest`) → Task 1
- §9 requirements (env 5, profile) → Global Constraints, Task 3 job env, Task 8 README
- §10 risks (token cap → classify approach; embed correct column; CDF; idempotent endpoint/index; region) → addressed in Tasks 4/5/6 code + notes
- §11 out-of-scope (no smoke test/extract/streaming) → honored (no such tasks)

**2. Placeholder scan** — the only non-literal region is Task 2's per-type reportlab flowable builders, which are specified by concrete content requirements + explicit reuse of the existing `generate_sample_loan_files.py` scaffolding (an in-repo file to read), and the fictional data pools (`INSUREDS = [...]`) which are intentionally author-filled fictional content, not logic. All pipeline logic, SQL, config, and tests are given in full.

**3. Type consistency** — helper names are used identically across tasks: `labels_json()`, `classify_expr(input_col, labels_sql_json, instructions)`, `domain_case_expr(doc_type_sql)`, `gold_table_name(prefix, domain)`, `index_name(catalog, schema, prefix, domain)`, `domain_for(doc_type)`. Table names are consistent: `{prefix}_bronze_parsed`, `{prefix}_silver_classified`, `{prefix}_prepped_chunks`, `{prefix}_{domain}_chunks`, `{prefix}_{domain}_index`. Column names (`chunk_id`, `chunk_position`, `chunk_to_retrieve`, `chunk_to_embed`, `doc_type`, `domain`, `source_path`, `prepped_at`) match between prep (Task 5), gold (Task 5), and index PK/embed columns (Task 6).
