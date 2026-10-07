# Project Plan — Document-Processing Skills + Solution-Architect Meta-Skill

**Branch:** `feat/doc-processing-skills-and-architect`
**Status:** Draft for review
**Last updated:** 2026-10-07

---

## 1. Goal

Turn the reusable patterns in this repo into **agent skills** that run on both
**Genie Code** and **Claude Code**, and ship a **solution-architect meta-skill**
that interviews a customer, profiles their sample documents, and produces an
**implementation plan** (architecture + skill-by-skill task list) — *without*
implementing it. A coding agent (Genie Code, Claude Code) then executes that plan.

The repo's seven worked examples are both the **source material** for the skills
and the **regression suite** for the meta-skill.

---

## 2. Locked decisions

| # | Decision | Choice |
|---|---|---|
| Platforms | Where skills run | **Genie Code + Claude Code** (both) |
| Home | Where skills live | **This repo** (component + meta), published outward |
| Primary user | Who uses the meta-skill | **Customer** (prescriptive, implementation-ready) |
| Workspace | Profiler access | **Yes** — profiler may run on the customer's workspace |
| Input model | How scenario input is gathered | **Doc-first journey**: thin business interview + sample-doc profiling drives the plan; domain packs are a consulted asset; the technical pattern catalog is the engine |
| Form factor | Meta-component shape | **Orchestrating meta-skill** (portable; invokes component skills; no autonomous loop/harness) |
| Distribution | Skill resolution | **Repo-authored + UC Gateway publish**: plan references skills by stable name; resolver prefers the customer's UC Gateway when present, falls back to repo/plugin. Includes a spike to confirm UC agent-skill registration |

**Design principle (applies throughout):** component skills *build on top of*
`databricks-ai-functions` — they reference it for AI-function syntax and spend
their own content on the **pattern + gotchas**, so they do not go stale.

---

## 3. Architecture

```
                        ┌─────────────────────────────────────────┐
                        │   solution-architect  (meta-skill)       │
                        │   phased journey, invokes component skills│
                        └───────────────┬─────────────────────────┘
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
   ┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
   │ pattern catalog │        │  domain packs    │        │ component skills │
   │ (reqs→pattern→  │        │ mortgage /       │        │ profiler, large- │
   │  skill table)   │        │ insurance / HR   │        │ pdf, confidence… │
   └─────────────────┘        └──────────────────┘        └────────┬─────────┘
                                                                    ▼
                                                   references `databricks-ai-functions`
                                                   + AI functions governed in UC Gateway
```

### 3.1 Skill-reference contract (enables "both" distribution)

- Every component skill has a **stable `name`** (its frontmatter `name`).
- The meta-skill's plan output and its internal composition reference skills **by name only**.
- A thin **resolver** maps `name → location`, in priority order:
  1. Customer's **UC Gateway** governed skill (if registered + accessible)
  2. Locally installed skill (`.claude/skills/` or `/Workspace/.../.assistant/skills/`)
  3. Repo reference-implementation path (fallback link in the plan)
- The plan therefore stays **portable**: it names what each step needs; resolution happens wherever it runs.

---

## 4. Component skills

### 4.1 Priority (build first)

| Skill (`name`) | Source | What it owns | Workspace? |
|---|---|---|---|
| `document-complexity-profiler` | *(new)* | **Keystone.** Two-tier profiling: (a) cheap local `pypdf` pass — page count, text-layer vs. scan, image density, split-page hints; (b) optional deep pass — `ai_parse_document` on a sample → element types, tables/figures per page, mixed doc-types per packet, handwriting. **Output = profile + recommended pattern(s)**, not raw numbers. | Tier-b yes |
| `parse-large-pdfs` | `parse-pdfs-with-large-num-of-pages/` | Parsing beyond the 500-page/call limit via chunked `pageRange` + stitching into one uniform VARIANT (bronze-compatible). | Yes |
| `analyze-extraction-confidence` | `evaluation-harness/` | Two modes: (1) profile `ai_parse_document`/`ai_extract` confidence distributions; (2) score against ground truth (flat fuzzy P/R/F1 + nested type-aware) via `mlflow.genai.evaluate`. | Yes |

The profiler is the keystone because it **grounds the meta-skill in evidence** (customers mis-describe their own docs) and links the component skills to the meta-skill.

### 4.2 Later (one per remaining repo pattern)

| Skill (`name`) | Source |
|---|---|
| `page-classify-route-extract` | `document-page-classify-extraction/` |
| `word-level-citation` | `ai-extract-word-level-citation/` |
| `chart-figure-analysis` | `document-embedding-chart-analysis/` |
| `combine-split-pages` | `combine-document-pages/` |
| `rag-knowledge-base` | `ai-search-knowledge-base-creation/` |

### 4.3 Skill authoring conventions (match existing repo skills)

- `SKILL.md` with YAML frontmatter: `name`, `description` (trigger-optimized).
- Progressive disclosure: supporting `N-topic.md` files, loaded on demand.
- Keep AI-function syntax out of scope → defer to `databricks-ai-functions`.
- Lead each skill with the **gotchas** (seed from memory: `variant_get` paths, no `[*]`,
  `addPyFile` on serverless, never `CREATE OR REPLACE` a Delta Sync source table).

---

## 5. Pattern catalog (the meta-skill engine)

Domain-agnostic table: **requirement → pattern → skill(s) → reference impl**. Seed rows:

| Requirement signal | Pattern | Skill(s) | Reference impl |
|---|---|---|---|
| Multi-document packet, mixed types | page classify → route → extract | `page-classify-route-extract` | `document-page-classify-extraction/` |
| Needs evidence / audit trail | citations (optionally word-level) | `word-level-citation` | `ai-extract-word-level-citation/` |
| > 500 pages per doc | chunked parse + stitch | `parse-large-pdfs` | `parse-pdfs-with-large-num-of-pages/` |
| Charts/figures carry meaning | vision-LLM branch | `chart-figure-analysis` | `document-embedding-chart-analysis/` |
| Docs arrive as split single pages | reassemble to one VARIANT | `combine-split-pages` | `combine-document-pages/` |
| Downstream is Q&A / retrieval | `ai_prep_search` + Vector Search | `rag-knowledge-base` | `ai-search-knowledge-base-creation/` |
| Streaming vs. batch ingest | Auto Loader + AvailableNow / batch DAB | (bundle choice) | respective bundles |
| Any extraction at all | confidence profiling + eval | `analyze-extraction-confidence` | `evaluation-harness/` |

This catalog is **mostly writing, not code** — cheapest high-value step; build it early so we can tell whether the meta-skill has enough to work with.

---

## 6. Meta-skill: `solution-architect`

**Form:** orchestrating skill (portable). **Journey phases:**

1. **Intake** — thin *business* interview: what decision does the output feed, volume,
   latency (batch/streaming), compliance/PII, target consumers. (Few high-signal questions;
   customer-answerable.)
2. **Profile** — invoke `document-complexity-profiler` on uploaded samples. Technical
   reality from evidence, not self-report.
3. **Match** — map profile + answers against the **pattern catalog**; consult matching
   **domain pack** to enrich/validate; web search only as optional add-on for uncovered domains.
4. **Draft plan** — emit the plan (section 7).
5. **Validate** — self-check plan against regression expectations (section 9); surface open questions.

**Deliverable: the plan does not implement.** It is handed to a coding agent.

### Plan output template
- Scenario summary + assumptions
- Document profile results (from profiler)
- Architecture diagram (Mermaid)
- Medallion stages — each with: pattern, skill(s) by **name**, rationale
- Evaluation plan — ground truth, confidence thresholds, human-review gate
- Open questions
- Ordered implementation task list (coding-agent-ready; each task names its skill + fallback reference-impl link)

---

## 7. Domain packs

Reference files consulted by the journey — **not** the driver. Seed cheaply from repo
worked examples; grow by demand. Each pack: document types, typical extraction schemas,
regulatory/PII concerns.

- **Mortgage / lending** → `document-page-classify-extraction/`, `ai-extract-word-level-citation/`
- **Insurance** → `ai-search-knowledge-base-creation/`
- **HR / payroll** → paystub example in `ai-extract-word-level-citation/`

One meta-skill, **pluggable packs** — never one meta-skill per industry. Defer broad
industry research until a pack is demanded.

---

## 8. Distribution & UC Gateway

- **Source of truth:** this repo. Reuse the existing install paths (`.claude/skills/install_skills.sh`,
  `install_genie_code_skills.py` → `/Workspace/Users/<user>/.assistant/skills/`).
- **UC Gateway publish:** register component + meta skills as governed assets so the
  customer's workspace resolves them from *their* gateway (preferred resolver).
- **AI functions** already UC-governed — no change; skills just consume them.
- **⚠️ Spike (blocking for the UC-resolver path):** confirm the current mechanism for
  registering *agent skills* (not just AI functions) as UC Gateway assets, access-control
  model, and how Genie Code / Claude Code resolve them. Until confirmed, ship via
  repo/plugin install (resolver priority 2–3) and treat UC Gateway as an additive channel.

---

## 9. Evaluation / regression suite

The repo's own projects are the test cases. For each scenario, assert the meta-skill's
plan lands on the expected pattern:

| Scenario fed to meta-skill | Expected plan centers on |
|---|---|
| Mortgage loan-file packet | `page-classify-route-extract` |
| Paystub audit, evidence required | `word-level-citation` |
| 800-page contract | `parse-large-pdfs` |
| Report with charts for RAG | `chart-figure-analysis` + `rag-knowledge-base` |
| Adjuster/underwriter KB | `rag-knowledge-base` (two domains) |

Use `skill-creator` eval tooling to run these as a regression set with variance analysis.

---

## 10. Phases & milestones

| Phase | Deliverable | Depends on |
|---|---|---|
| **P0 — Confirm** | This plan reviewed; UC-Gateway spike scheduled | — |
| **P1 — Pattern catalog** | `docs/pattern-catalog.md` (section 5), fleshed out | P0 |
| **P2 — Component skills** | `document-complexity-profiler`, `parse-large-pdfs`, `analyze-extraction-confidence` (SKILL.md + gotchas + tests where applicable) | P1 |
| **P3 — Meta-skill** | `solution-architect` skill with phased journey + plan template + 2 domain packs | P2 |
| **P4 — Evals** | Regression suite (section 9) green on seeded scenarios | P3 |
| **P5 — Distribute** | Install scripts updated; UC Gateway publish (gated on spike) | P4, spike |
| **P6 — Expand** | Remaining component skills (4.2) + more domain packs, by demand | P5 |

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Meta-skill improvises generic architectures | Pattern catalog is a hard asset the journey must consult |
| Customers mis-describe documents | Doc-first: profiler extracts ground truth from samples |
| Component skills drift from AI-function syntax | Defer syntax to `databricks-ai-functions`; own only pattern + gotchas |
| Plan not portable across platforms | Reference skills by stable name + resolver + fallback links |
| UC agent-skill hosting less mature than assumed | Spike before committing; repo/plugin install always works |
| Scope creep into one-skill-per-industry | One meta-skill, pluggable packs, demand-driven growth |

---

## 12. Open questions

1. UC Gateway agent-skill registration mechanism + access model (spike, section 8).
2. Profiler sample size / cost ceiling for the tier-b deep pass on large corpora.
3. Minimum business-question set for intake that stays customer-answerable yet high-signal.
4. Whether the plan output should also emit a starter DAB skeleton, or stay pure markdown.
