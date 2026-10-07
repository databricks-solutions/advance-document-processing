# FINS Unstructured Data Processing — Industry Pattern Glossary

## Purpose & Vision

This glossary catalogs **document / unstructured data processing patterns** across
Financial Services BU+1 sub-verticals. Each pattern is described in business terms a
customer would recognize, mapped to the Databricks AI Functions pipeline that
implements it, and tagged with the right component skills. The goal is to feed this
glossary into an **agent skill** that can:

1. **Identify** which pattern(s) a customer's use case maps to
2. **Design** the architecture of the workflow (which AI Functions, in what order, with what orchestration)
3. **Select** the right AI Functions and component skills
4. **Generate** an implementation plan that Genie Code can execute

---

## Reference Architecture — The Universal Pipeline

All FINS document processing patterns share a common backbone:

```
Ingest → Parse → Classify (optional) → Extract → Enrich/Match → Orchestrate → Structured Output
```

| Stage | Databricks Component | Function / Product |
|---|---|---|
| **Ingest** | Lakeflow Connect / Auto Loader → UC Volumes | `READ_FILES(..., format='binaryFile')` |
| **Parse** | AI Functions | `ai_parse_document(content, options)` |
| **Classify** | AI Functions | `ai_classify(text, labels)` |
| **Extract** | AI Functions | `ai_extract(text, schema)` |
| **Chunk (RAG)** | AI Functions | `ai_prep_search(text)` |
| **Match/Resolve** | Vector Search / AI Functions | `ai_similarity()`, `vector_search()` |
| **Redact** | AI Functions | `ai_mask(text, categories)` |
| **Orchestrate** | Lakeflow SDP / Workflows / DABs | Declarative Pipelines, Jobs |
| **Govern** | Unity Catalog | Lineage, ACLs, audit trail |
| **Serve** | Agent Bricks / Genie | Knowledge Assistants, downstream agents |

### Common Composition Patterns

| Goal | Pipeline | Example |
|---|---|---|
| Extract fields from documents | `ai_parse_document` → `ai_classify` (optional) → `ai_extract` | Parse invoices, classify by type, extract line items |
| Build search / RAG | `ai_parse_document` → `ai_prep_search` → Vector Search | Parse contracts, chunk for embedding, power Q&A agent |
| Classify text already in a table | `ai_classify` (directly on text column) | Route support tickets by intent |
| Custom reasoning / multi-step | `ai_query` with prompt engineering | Summarize earnings calls, generate risk assessments |
| PII redaction pipeline | `ai_parse_document` → `ai_mask` → `ai_extract` | Redact PHI from clinical notes before extraction |

---

## Banking & Payments Patterns

### Pattern B1: Loan Document Intake & Processing

**Business Context:** Banks process millions of loan applications annually. Each
application package contains 10–30 documents (applications, pay stubs, tax returns,
bank statements, W-2s). Manual review takes 30–60 minutes per package and is
error-prone.

**Document Types:** Loan applications, pay stubs, W-2 forms, tax returns (1040), bank
statements, employment verification letters, property appraisals, title reports

**Pipeline:**

```
UC Volumes (raw PDFs) → ai_parse_document → ai_classify (route by doc type)
  → ai_extract (borrower info, income, assets per doc type schema) → Delta Gold table
  → downstream: credit decisioning model, compliance checks
```

**Key AI Functions:**

- `ai_parse_document` — OCR + layout extraction from scanned/digital PDFs
- `ai_classify` — Route documents by type (application vs. pay stub vs. tax return)
- `ai_extract` — Pull borrower name, SSN, income, employer, loan amount with typed schema per doc type

**Complexity Factors:** Mixed scanned/digital, handwritten signatures, multi-page tables, income verification across multiple doc types

**Customer Signals:** "We process thousands of loan apps per day and our OCR is brittle" · "We need to extract borrower info from 15 different document types"

### Pattern B2: KYC / AML Compliance Document Processing

**Business Context:** Know Your Customer (KYC) and Anti-Money Laundering (AML)
regulations require banks to verify customer identity, assess risk, and monitor for
suspicious activity. Document verification is the bottleneck — each onboarding can
require 20+ document checks.

**Document Types:** Government-issued IDs (passports, driver's licenses), utility bills,
corporate filings, beneficial ownership forms, sanctions screening results, PEP
declarations

**Pipeline:**

```
Customer upload → ai_parse_document → ai_extract (identity fields, addresses, dates)
  → ai_similarity / vector_search (match against watchlists, sanctions lists)
  → ai_classify (risk tier: low/medium/high)
  → Delta table → compliance dashboard + case management
```

**Key AI Functions:**

- `ai_parse_document` — Extract from IDs, utility bills, corporate filings
- `ai_extract` — Pull name, DOB, address, ID number, expiry date
- `ai_similarity` + `vector_search` — Fuzzy match against watchlists and sanctions databases
- `ai_classify` — Risk-tier classification

**Complexity Factors:** Multi-language documents, poor scan quality on IDs, cross-referencing across document types, real-time requirements for digital onboarding

### Pattern B3: Check Processing & Payment Reconciliation

**Business Context:** Banks process millions of checks daily. Automated check
processing requires reading handwritten amounts, payee names, dates, and MICR lines,
then reconciling against account records.

**Document Types:** Personal checks, business checks, cashier's checks, money orders, deposit slips

**Pipeline:**

```
Check images → ai_parse_document → ai_extract (payee, amount, date, MICR, memo)
  → ai_classify (check type, exception flag)
  → reconciliation engine (match to account records)
```

**Key AI Functions:**

- `ai_parse_document` — Handle handwritten + printed content on check images
- `ai_extract` — Pull amount (written + numeric), payee, date, account/routing numbers
- `ai_classify` — Flag exceptions (stale-dated, amount mismatch, suspicious)

### Pattern B4: Regulatory Reporting & Compliance Document Analysis

**Business Context:** Banks must file and analyze regulatory reports (Call Reports,
stress test submissions, BSA/AML reports). Compliance teams also need to interpret new
regulatory guidance and assess impact.

**Document Types:** Call Reports, CCAR/DFAST submissions, BSA/AML filings, regulatory
guidance documents, consent orders, examination reports

**Pipeline:**

```
Regulatory PDFs → ai_parse_document → ai_prep_search → Vector Search
  → Knowledge Assistant (Q&A agent for compliance team)
  + ai_extract (specific metrics, deadlines, requirements)
```

**Key AI Functions:**

- `ai_parse_document` — Handle dense regulatory PDFs with complex tables
- `ai_prep_search` — Chunk for RAG-based compliance Q&A
- `ai_extract` — Pull specific requirements, deadlines, thresholds
- `ai_query` — Summarize regulatory changes, assess impact

### Pattern B5: Merchant Description Enrichment

**Business Context:** Transaction descriptions from payment networks are cryptic
abbreviations. Banks and fintechs need to resolve these to canonical merchant names for
analytics, categorization, and customer-facing apps.

**Document Types:** Transaction records (structured text, not PDFs), merchant registry databases

**Pipeline:**

```
Transaction table → ai_extract (merchant name from description)
  → ai_similarity / vector_search (fuzzy match against merchant registry)
  → enriched transaction table
```

**Key AI Functions:**

- `ai_extract` — Parse merchant name from transaction description string
- `ai_similarity` + `vector_search` — Match against canonical merchant registry

---

## Insurance Patterns

### Pattern I1: Claims Document Intake & Triage

**Business Context:** Insurance claims arrive as multi-document packages: claim forms,
police reports, medical records, photos, repair estimates. Manual triage takes hours
per claim. Automating intake can reduce administrative expenses by 20–60%.

**Document Types:** First Notice of Loss (FNOL) forms, police/incident reports, medical
records, repair estimates, photos (vehicle damage, property damage), receipts, invoices

**Pipeline:**

```
Claims submission → ai_parse_document (forms + reports)
  → ai_classify (claim type: auto/property/liability/workers comp)
  → ai_extract (claimant info, incident details, damage description, amounts)
  → ai_classify (severity: minor/moderate/major/catastrophic)
  → routing engine → adjuster assignment + fraud scoring
```

**Key AI Functions:**

- `ai_parse_document` — Handle mixed document types including handwritten forms
- `ai_classify` — Multi-level classification (claim type → severity → fraud risk)
- `ai_extract` — Pull claimant, policy number, incident date/location, damage amounts
- `ai_query` — Summarize claim narrative for adjuster review

**Complexity Factors:** Handwritten forms, photos mixed with documents, multi-page medical records, PII/PHI requiring redaction

**Customer Signals:** "We have terabytes of claims documents and can't do anything with them" · "Our claims triage takes 3 days — we need it in hours"

### Pattern I2: Underwriting Document Analysis

**Business Context:** Underwriters evaluate risk by reviewing applications, financial
statements, loss runs, inspection reports, and third-party data. A commercial
underwriting package can be 200+ pages.

**Document Types:** Insurance applications, financial statements, loss run reports,
inspection reports, MVRs (Motor Vehicle Reports), CLUE reports, building appraisals

**Pipeline:**

```
Submission package → ai_parse_document → ai_classify (doc type routing)
  → ai_extract (risk factors per doc type: revenue, loss history, building specs)
  → risk scoring model → underwriter workbench
  + ai_prep_search → Knowledge Assistant (underwriting guidelines Q&A)
```

**Key AI Functions:**

- `ai_parse_document` — Handle large multi-page packages (200+ pages)
- `ai_classify` — Route by document type within the submission package
- `ai_extract` — Pull risk-relevant fields: revenue, employee count, loss history, building construction type
- `ai_prep_search` — Chunk underwriting guidelines for RAG

### Pattern I3: Policy Document Processing & Servicing

**Business Context:** Policy documents, endorsements, and declarations pages need to be
parsed for servicing, renewal, and compliance. Insurers manage millions of active
policies with frequent amendments.

**Document Types:** Policy declarations pages, endorsements, coverage schedules, exclusion riders, renewal notices, cancellation notices

**Pipeline:**

```
Policy documents → ai_parse_document → ai_extract (coverage limits, deductibles,
  effective dates, named insureds, exclusions)
  → Delta policy master table → servicing agents + renewal automation
```

### Pattern I4: Catastrophe Modeling & Damage Assessment

**Business Context:** After a catastrophe event (hurricane, wildfire, flood), insurers
must rapidly assess damage across thousands of claims using aerial imagery, adjuster
photos, and field reports.

**Document Types:** Aerial/satellite imagery, adjuster field photos, damage assessment reports, weather data, property records

**Pipeline:**

```
Images + reports → ai_parse_document (reports) + VLM prompt engineering (images)
  → ai_extract (damage severity, affected structures, estimated loss)
  → ai_classify (damage category: total loss / major / minor / cosmetic)
  → geospatial aggregation → catastrophe response dashboard
```

---

## Capital Markets Patterns

### Pattern CM1: SEC Filing Analysis & Financial Intelligence

**Business Context:** Analysts review thousands of SEC filings (10-Ks, 10-Qs, proxy
statements, 8-Ks) to extract financial metrics, risk factors, and material changes.
Manual review of a single 10-K can take 4–8 hours.

**Document Types:** 10-K annual reports, 10-Q quarterly reports, 8-K current reports,
proxy statements (DEF 14A), prospectuses, earnings transcripts

**Pipeline:**

```
SEC EDGAR filings → ai_parse_document (handle 200-500 page filings)
  → ai_extract (revenue, EPS, risk factors, material changes, executive comp)
  → ai_prep_search → Vector Search (analyst Q&A agent)
  + ai_query (summarize risk factor changes YoY, generate investment memos)
```

**Key AI Functions:**

- `ai_parse_document` — Handle very large documents (500+ pages) with dense tables
- `ai_extract` — Pull financial metrics, risk factors, executive compensation
- `ai_prep_search` — Chunk for RAG-based analyst research assistant
- `ai_query` — Comparative analysis, YoY change detection, memo generation

**Complexity Factors:** Very large documents, complex nested tables, footnotes with critical information, cross-referencing across filings

### Pattern CM2: Trade Finance Document Processing

**Business Context:** Trade finance involves letters of credit, bills of lading,
invoices, and certificates of origin. Processing is manual, paper-heavy, and
error-prone. A single trade can involve 20+ documents across multiple counterparties.

**Document Types:** Letters of credit, bills of lading, commercial invoices, packing
lists, certificates of origin, insurance certificates, inspection certificates

**Pipeline:**

```
Trade documents → ai_parse_document → ai_classify (doc type)
  → ai_extract (counterparties, amounts, terms, dates, goods description)
  → ai_similarity (match counterparties against known entities)
  → automated settlement / compliance check
```

### Pattern CM3: M&A Due Diligence Document Processing

**Business Context:** M&A due diligence requires reviewing thousands of documents in
virtual data rooms — contracts, financial statements, IP filings, litigation records.
Speed and accuracy directly impact deal outcomes.

**Document Types:** Contracts, financial statements, IP patents, litigation records,
corporate governance documents, employment agreements, real estate leases

**Pipeline:**

```
Data room documents → ai_parse_document → ai_classify (doc category)
  → ai_extract (key terms, obligations, risk clauses, financial metrics)
  → ai_prep_search → Due Diligence Q&A Agent
  + ai_query (flag red flags, summarize material contracts)
```

### Pattern CM4: Portfolio Performance & Trade Reconciliation

**Business Context:** Asset managers and custodians reconcile trade confirmations,
account statements, and NAV reports daily. Discrepancies must be identified and resolved
before market open.

**Document Types:** Trade confirmations, account statements, NAV reports, corporate action notices, margin calls

**Pipeline:**

```
Broker/custodian statements → ai_parse_document → ai_extract (trade details,
  positions, NAV, corporate actions)
  → reconciliation engine (match against internal records)
  → exception queue for operations team
```

---

## Cross-Vertical Patterns

### Pattern XV1: Invoice Processing & AP Automation

**Business Context:** Every financial institution processes vendor invoices. AP
automation extracts vendor, amount, line items, and PO numbers to feed ERP systems.
Applies across all FINS sub-verticals.

**Document Types:** Vendor invoices, purchase orders, delivery receipts, credit memos

**Pipeline:**

```
Invoices → ai_parse_document → ai_extract (vendor, invoice #, date, line items,
  amounts, PO #, tax)
  → ai_similarity (match vendor against master vendor list)
  → ERP integration (3-way match: PO ↔ receipt ↔ invoice)
```

### Pattern XV2: Contract Analysis & Obligation Tracking

**Business Context:** Financial institutions manage thousands of contracts (vendor,
customer, partnership, regulatory). Extracting key terms, obligations, and renewal
dates is critical for risk management and compliance.

**Document Types:** Service agreements, NDAs, licensing agreements, regulatory consent orders, partnership agreements

**Pipeline:**

```
Contracts → ai_parse_document → ai_extract (parties, effective dates,
  termination clauses, obligations, SLAs, penalties)
  → ai_prep_search → Contract Q&A Agent
  + obligation tracking table → automated alerts
```

### Pattern XV3: Customer Communication Intelligence

**Business Context:** Financial institutions receive millions of customer
communications (emails, letters, chat transcripts, call transcripts). Extracting intent,
sentiment, and actionable items drives service quality and compliance.

**Document Types:** Customer emails, complaint letters, chat transcripts, call transcripts, social media posts

**Pipeline:**

```
Communications → ai_classify (intent: complaint/inquiry/request/feedback)
  → ai_analyze_sentiment (positive/negative/neutral)
  → ai_extract (account #, issue description, requested action)
  → routing engine → case management + compliance monitoring
```

**Key AI Functions:**

- `ai_classify` — Intent classification (up to 500 labels)
- `ai_analyze_sentiment` — Sentiment scoring
- `ai_extract` — Pull structured fields from free text
- `ai_query` — Summarize conversation, generate response draft

### Pattern XV4: Document Anonymization & PII Redaction

**Business Context:** Regulated financial institutions must redact PII/PCI/PHI before
sharing documents with third parties, auditors, or for analytics. This is a prerequisite
pattern that often precedes other processing.

**Document Types:** Any document containing PII — customer records, claims, applications, correspondence

**Pipeline:**

```
Source documents → ai_parse_document → ai_mask (PII categories: names, SSN,
  account numbers, addresses, DOB)
  → anonymized Delta table → safe for analytics / third-party sharing
```

---

## Complexity & Decision Framework

When scoping a customer's document processing use case, assess these dimensions:

| Dimension | Low Complexity | High Complexity |
|---|---|---|
| **Document variety** | Single doc type, consistent format | 10+ doc types, mixed formats |
| **Scan quality** | Born-digital PDFs | Scanned, faxed, handwritten |
| **Page count** | 1–10 pages per document | 100–500+ pages per document |
| **Volume** | < 10K docs/month | 1M+ docs/month |
| **Extraction schema** | Flat, < 10 fields | Nested, 50+ fields, line items |
| **Accuracy requirement** | 85%+ acceptable | 95%+ required (regulatory) |
| **Real-time need** | Batch (hours/days) | Near-real-time (minutes) |
| **PII sensitivity** | No PII | PII/PHI/PCI requiring redaction |
| **Ground truth** | Labeled data available | No ground truth, must create |

---

## AI Functions Quick Reference

| Function | What It Does | When to Use | Input | Output |
|---|---|---|---|---|
| `ai_parse_document` | OCR + layout extraction | Raw PDFs, images, DOCX, PPTX → structured variant | Binary file content | VARIANT (elements, tables, text, bounding boxes) |
| `ai_extract` | Structured field extraction | Pull specific fields from text/parsed docs | Text + JSON schema | STRUCT matching schema |
| `ai_classify` | Text classification | Route/categorize documents or text | Text + label array (up to 500) | Predicted label(s) |
| `ai_prep_search` | Chunking for RAG | Prepare documents for vector search | Parsed text | Chunks with metadata |
| `ai_query` | Custom LLM inference | Custom reasoning, multi-step logic, specific model needs | Text + prompt | Free-form or structured response |
| `ai_mask` | PII redaction | Remove sensitive data before processing/sharing | Text + PII categories | Redacted text |
| `ai_similarity` | Semantic similarity | Entity matching, deduplication | Two text inputs | Similarity score |
| `ai_analyze_sentiment` | Sentiment analysis | Customer communication analysis | Text | Sentiment label + score |
| `ai_translate` | Translation | Multi-language document processing | Text + target language | Translated text |
| `ai_forecast` | Time series forecasting | Predict document volumes, processing costs | Time series data | Forecast values |

---

## Next Steps: From Glossary → Agent Skill → Implementation

### Phase 1: Pattern Identification Skill

Build a Genie skill that takes a customer's use case description and maps it to one or
more patterns from this glossary. Input: natural language description of the use case.
Output: matched pattern(s), recommended pipeline, complexity assessment.

### Phase 2: Architecture Design Skill

Build a skill that takes the matched pattern(s) and customer-specific parameters
(document types, volume, accuracy requirements) and generates a detailed architecture
diagram with specific AI Functions, orchestration approach, and infrastructure sizing.

### Phase 3: Implementation Plan Generator

Build a skill that takes the architecture and generates a step-by-step implementation
plan with:

- Notebook templates for each pipeline stage
- Schema definitions for `ai_extract`
- Label taxonomies for `ai_classify`
- Evaluation harness configuration
- Lakeflow SDP pipeline definitions
- DABs deployment configuration

### Phase 4: Genie Code Execution

The implementation plan feeds into Genie Code, which can execute the notebooks, create
the tables, configure the pipelines, and deploy the solution.

---

*Report generated by Databricks Genie*
