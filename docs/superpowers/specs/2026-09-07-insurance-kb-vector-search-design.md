# Insurance Knowledge Base — Parse → Classify → Prep → Index

**Design spec** · 2026-09-07 · project dir: `insurance-kb-vector-search/`

## 1. Overview

A new reference/demo pipeline for the `advance-document-processing` repo that
ingests insurance PDFs from a Unity Catalog Volume and turns them into
retrieval-ready Databricks Vector Search indexes, entirely with built-in AI
Functions. It is the first pipeline in the repo whose terminal output is a
**Vector Search index** rather than a gold extraction table — it completes the
repo's story from "parse & extract" into "parse & retrieve (RAG)".

The end-to-end flow:

```
ai_parse_document → ai_classify → ai_prep_search → CREATE VECTOR SEARCH INDEX
```

Target use case: a **knowledge base for an insurance adjuster / underwriter AI
assistant**. Documents split into two retrieval domains that mirror how those
personas actually search:

- **reference** — authoritative, relatively stable policy knowledge
  (underwriter: "what does this form say about water damage?")
- **claims** — transactional, per-claim evidence
  (adjuster: "what did the inspection find on this loss?")

## 2. Goals / Non-Goals

**Goals**
- Demonstrate the full `parse → classify → prep_search → index` chain with AI
  Functions, no custom model endpoints.
- Ship both flavors the repo standardizes on: interactive **notebooks** and a
  **Databricks Asset Bundle** (batch/manual-trigger, like
  `ai-extract-word-level-citation`).
- Show **classify-driven routing**: one classification decision sends each
  document's chunks to one of two gold tables, each backed by its own index.
- Provide **synthetic, fully fictional** insurance PDFs so the demo runs
  out-of-the-box, plus a ground-truth CSV of document-type labels.
- Carry the repo's engineering conventions: `table_prefix` isolation,
  centralized bundle variables, pure-Python helpers with `pytest` unit tests.

**Non-Goals**
- No retrieval/serving/assistant layer, no Agent Bricks Knowledge Assistant
  (explicitly out of scope — this is the ingestion+index half).
- No query smoke test and no classification-accuracy scoring step (per decision;
  the ground-truth CSV ships for the user's own optional checking, not as a
  pipeline stage).
- No streaming/Auto Loader (batch only).
- No field extraction (`ai_extract`) — classification + chunking only.
- No page-level splitting (documents are single-type, unlike the existing
  mortgage-packet pipeline).

## 3. Personas & Document Types

Five document types, each mapped to one retrieval domain:

| `doc_type` | Domain | Persona | Synthetic content |
|---|---|---|---|
| `policy_document` | `reference` | Underwriter | Dec page + homeowners/auto policy wording: coverages, limits, deductibles, exclusions |
| `endorsement` | `reference` | Underwriter | Rider amending a base policy (e.g. water-backup, scheduled jewelry) |
| `underwriting_guideline` | `reference` | Underwriter | Internal risk-acceptance / pricing rules (e.g. roof age, coastal exposure) |
| `fnol_claim_form` | `claims` | Adjuster | First Notice of Loss — structured fields (claimant, date of loss, peril, description) |
| `adjuster_report` | `claims` | Adjuster | Narrative inspection findings + damage assessment |

The `doc_type → domain` map is the single routing decision the pipeline makes.
It lives in a pure-Python helper (`sql_builders.py`) so it is unit-testable.

## 4. Architecture

Batch medallion pipeline. Each stage is one notebook (illustrative) and one
bundle `src/` file (production).

### Stage 1 — Bronze: parse
Read PDFs from the Volume subdir as `binaryFile`, run `ai_parse_document`, keep
the raw VARIANT and any per-document parse error.

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{prefix}_bronze_parsed AS
SELECT
  path AS source_path,
  ai_parse_document(content) AS parsed,
  variant_get(ai_parse_document(content), '$.error_status', 'STRING') AS parse_error
FROM read_files('/Volumes/{catalog}/{schema}/{volume}/{subdir}/', format => 'binaryFile');
```
(Implementation computes `ai_parse_document` once via a CTE rather than twice;
shown inline here for clarity.)

### Stage 2 — Silver: classify document type
Derive a document-level text slice from the parsed VARIANT (concatenated element
contents), then classify into exactly one of the five labels. Classifying on a
bounded text slice — not the whole VARIANT — keeps input under the `ai_classify`
128k-token cap and avoids embedding layout metadata in the prompt.

- Labels are supplied **with descriptions** plus an insurance-domain
  `instructions` string for accuracy (per skill guidance).
- `ai_classify` returns `VARIANT {"response": ["label"], "error_message": null}`;
  the label is read via `variant_get(..., '$.response[0]', 'STRING')`.
- `domain` is derived from `doc_type` via the routing map.

```
silver_classified: source_path, parsed, classification_raw, doc_type, domain
```
(note: `error_message` is derivable from `classification_raw:error_message`; no separate error column)

Rows that fail to parse or classify are filtered out before Stage 3.

### Stage 3 — Prep: chunk for retrieval
`ai_prep_search(parsed)` performs semantic chunking + context enrichment, then
`explode` the chunk array. Carry `doc_type` and `domain` onto every chunk row.

Chunk contract (from `ai_prep_search`):
- `chunk_id` — unique per chunk → **primary key**
- `chunk_position` — ordinal within the document
- `chunk_to_retrieve` — raw chunk text → **returned to the LLM at query time**
- `chunk_to_embed` — context-enriched text → **embedding source**

### Stage 4 — Gold: route into two tables
Split the prepped chunks by `domain` into two gold tables. Two physical tables
(not one table + filtered views) because a Vector Search Delta Sync index syncs
from a whole Delta table. **Change Data Feed is enabled** on both so the indexes
can sync.

Both tables share the schema:

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | STRING | PK |
| `chunk_position` | INT | |
| `chunk_to_retrieve` | STRING | returned to LLM |
| `chunk_to_embed` | STRING | embedding source |
| `doc_type` | STRING | filterable metadata |
| `source_path` | STRING | provenance |
| `prepped_at` | TIMESTAMP | |

```
{prefix}_reference_chunks   ← policy_document, endorsement, underwriting_guideline
{prefix}_claims_chunks      ← fnol_claim_form, adjuster_report
```

### Stage 5 — Index: one Delta Sync index per gold table
Ensure a shared Vector Search endpoint exists, then create/refresh one
Delta Sync index per gold table with **Databricks-managed embeddings**.

- Endpoint: `{vs_endpoint}` (type `STANDARD`), shared by both indexes.
- Embedding model: `databricks-gte-large-en`.
- Per index: primary key `chunk_id`, embedding source column `chunk_to_embed`,
  sync mode **TRIGGERED**.
- `doc_type` is synced as a metadata column for optional filtered queries.

```
{prefix}_reference_index  ← {prefix}_reference_chunks
{prefix}_claims_index     ← {prefix}_claims_chunks
```

The pipeline terminates here (no query step).

### Data-flow diagram

```
/Volumes/.../pdf_examples/insurance_docs/*.pdf
   │  ai_parse_document
   ▼
bronze_parsed (parsed VARIANT)
   │  ai_classify → doc_type → domain
   ▼
silver_classified
   │  ai_prep_search → explode chunks (+doc_type,+domain)
   ▼
prepped chunks
   │  route by domain
   ├────────────► reference_chunks ──► reference_index
   └────────────► claims_chunks    ──► claims_index
```

## 5. Unity Catalog objects & naming

Centralized as bundle variables (defaults shown):

| Variable | Default |
|---|---|
| `catalog` | `fins_genai` |
| `schema` | `unstructured_documents` |
| `volume` | `pdf_examples` |
| `volume_subdir` | `insurance_docs` |
| `table_prefix` | `insurance_kb` |
| `vs_endpoint` | `insurance_kb_vs` |
| `embedding_model` | `databricks-gte-large-en` |

Source PDFs: `/Volumes/fins_genai/unstructured_documents/pdf_examples/insurance_docs/`

## 6. Deliverables & file tree

```
insurance-kb-vector-search/
├── README.md                         # architecture, defaults, quickstart
├── notebooks/                        # interactive/illustrative flavor
│   ├── 01_parse_documents.py
│   ├── 02_classify_documents.py
│   ├── 03_prep_search_chunks.py
│   ├── 04_route_gold_tables.py
│   └── 05_create_vector_indexes.py
├── bundle/                           # production flavor (batch DAB)
│   ├── databricks.yml
│   ├── resources/
│   │   └── insurance_kb.job.yml      # manual-trigger batch job
│   └── src/
│       ├── 01_bronze_parse.py
│       ├── 02_silver_classify.py
│       ├── 03_prep_search.py
│       ├── 04_gold_route.py
│       ├── 05_create_indexes.py
│       └── sql_builders.py           # pure helpers: labels, doc_type→domain map, DDL builders
├── tests/
│   └── test_sql_builders.py          # pytest, no Databricks required
└── sample_data/                      # ~15 committed synthetic PDFs
    ├── ground_truth_doc_types.csv    # filename → doc_type (for optional user checking)
    └── *.pdf
```

Plus repo-level changes:
- `scripts/generate_sample_insurance_docs.py` — synthetic PDF generator.
- New row in the root `README.md` project table describing this pipeline.
- Reuse existing `scripts/upload_pdfs.sh` for uploading PDFs to the Volume.

## 7. Synthetic data generator

`scripts/generate_sample_insurance_docs.py`, following the existing
`generate_sample_loan_files.py` approach (HTML → PDF via the repo's PDF
generation method):

- **3 documents per type × 5 types = 15 PDFs**, all fully fictional (no real
  people, policy numbers, carriers, or addresses).
- Varied but type-consistent layouts so classification is non-trivial:
  dec-page tables, form fields, narrative reports, guideline rule lists.
- Emits `sample_data/ground_truth_doc_types.csv` mapping each filename to its
  intended `doc_type`, for the user's own optional accuracy checking.
- Deterministic (seeded) so regenerating yields stable filenames.

## 8. Testing

Pure-Python unit tests in `tests/test_sql_builders.py`, run from repo root with
`uv run pytest` (matches the two existing tested pipelines; no Databricks
required):

- `doc_type → domain` routing map: every label maps to a valid domain; the two
  domains partition the five labels exactly (no orphan, no overlap).
- Label-set integrity: the classification label set equals the routing map keys.
- DDL/config builders (index spec, gold-table names) produce the expected
  fully-qualified names from given variables.

## 9. Requirements

- DBR **18.2+** or **serverless environment version 5** (repo standard) — required
  for `ai_prep_search`; `ai_parse_document` needs 17.3+.
- Unity Catalog + Serverless enabled; Vector Search enabled in the workspace.
- CLI profile **`fevm-classic-stable`** for all `bundle`/`fs` commands.
- Region supporting `ai_parse_document` / `ai_prep_search` batch AI inference.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Large doc exceeds `ai_classify` 128k-token cap | Classify on a bounded doc-level text slice, not the full VARIANT |
| Embedding the wrong column | Embed `chunk_to_embed`; return `chunk_to_retrieve` (documented + tested convention) |
| Delta Sync index can't pick up changes | Enable Change Data Feed on both gold tables |
| VS endpoint / index creation not idempotent across reruns | Create-if-not-exists for endpoint; create-or-sync for indexes |
| `ai_prep_search` region-restricted | Documented in README prereqs; fail fast with a clear message |

## 11. Out of scope (possible follow-ups)

- Retrieval + assistant layer (Agent Bricks Knowledge Assistant or a serving
  endpoint) wired onto the two indexes.
- Classification-accuracy evaluation using the existing `evaluation-harness/`.
- Streaming/Auto Loader flavor for a continuously-updating KB.
- `ai_extract` of structured claim/policy fields as index metadata.
