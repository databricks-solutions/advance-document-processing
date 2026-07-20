# Advanced Document Processing

Reference implementations for advanced document-processing pipelines on
Databricks — combining Unity Catalog, `ai_parse_document`, task-specific AI
Functions (`ai_classify`, `ai_extract`), vision LLMs, Auto Loader, and
Databricks Asset Bundles.

## Projects

| Project | Description |
|---|---|
| [`document-embedding-chart-analysis/`](./document-embedding-chart-analysis/) | End-to-end PDF **chart-analysis** pipeline. Parses PDFs with `ai_parse_document`, classifies and crops chart figures, runs a vision LLM on each chart, and splices the resulting insights back into a per-document gold table — ready for embedding/RAG. |
| [`document-page-classify-extraction/`](./document-page-classify-extraction/) | **Page-level classify-then-extract** pipeline. Splits a multi-document PDF packet into pages, classifies each page by document type with `ai_classify`, routes each to a type-specific `ai_extract` schema, and flattens the results into per-entity gold tables. Worked example: a residential mortgage loan file. |
| [`ai-extract-word-level-citation/`](./ai-extract-word-level-citation/) | **Word-level citation** pipeline. Parses PDFs with `ai_parse_document`, extracts fields with `ai_extract` 2.1 citations, then crops each cited element and uses **Tesseract OCR + string-matching** to localize every entity to word-level bounding boxes. Worked example: paystubs. |
| [`evaluation-harness/`](./evaluation-harness/) | **Evaluation recipe notebooks** (standalone, not a workflow). A data-prep front-end plus notebooks that profile `ai_parse_document` / `ai_extract` confidence distributions and score `ai_extract` against ground truth (flat fuzzy P/R/F1 and nested type-aware scoring) with `mlflow.genai.evaluate`. |
| [`parse-pdfs-with-large-num-of-pages/`](./parse-pdfs-with-large-num-of-pages/) | **Single recipe notebook** (standalone, not a workflow). Parses PDFs beyond `ai_parse_document`'s 500-page-per-call limit by routing long documents through chunked `pageRange` calls and stitching the result back into one uniform VARIANT — schema-compatible with the parse pipelines' bronze layer. |

The three pipelines each ship in two flavors — interactive **batch notebooks** and
a **Databricks Asset Bundle** (DAB). The chart-analysis and page-classify-extraction
bundles are **streaming** (Auto Loader + `Trigger.AvailableNow`, scheduled); the
word-level-citation bundle is **batch** (manual trigger). `evaluation-harness/` and
`parse-pdfs-with-large-num-of-pages/` are notebooks only. Each project's own
`README.md` covers its architecture, defaults, and quickstart.

## Repo layout

```
advance-document-processing/
├── document-embedding-chart-analysis/   # PDF chart-analysis pipeline (notebooks + streaming DAB)
├── document-page-classify-extraction/   # Page classify + extract pipeline (notebooks + streaming DAB)
├── ai-extract-word-level-citation/      # Word-level citation pipeline (notebooks + batch DAB)
├── evaluation-harness/                  # AI-function evaluation recipe notebooks
├── parse-pdfs-with-large-num-of-pages/  # Parse >500-page PDFs via chunked pageRange (recipe notebook)
├── scripts/                             # Cross-project helper scripts
│   ├── upload_pdfs.sh                   # Upload local PDFs to a UC Volume via the CLI
│   ├── generate_sample_loan_files.py    # Generate synthetic mortgage loan-file PDFs
│   └── generate_sample_paystubs.py      # Generate synthetic paystub PDFs + ground-truth CSV
├── pyproject.toml                       # pytest config for the bundles' unit tests (uv run pytest)
├── .github/CODEOWNERS
├── LICENSE.md
├── NOTICE.md
└── SECURITY.md
```

The two pipelines that extracted their bundle's pure helpers into importable
modules (`ai-extract-word-level-citation`, `document-page-classify-extraction`)
carry a `tests/` folder; run the whole suite from the repo root with
`uv run pytest` (pure Python, no Databricks required).

## Prerequisites

- Databricks workspace with Unity Catalog and Serverless Jobs enabled
- DBR **17.3+** (or serverless environment version **3+**) for `ai_parse_document`;
  serverless env **5** is recommended (used by the streaming bundles)
- DBR **18.2+** (serverless env **3+**) for `ai_extract` 2.1 with citations +
  confidence scores (page classify-extract and word-level-citation pipelines)
- Databricks CLI **v0.205+** (the unified CLI) for `bundle` and `fs` commands
- A multimodal serving endpoint for chart analysis (default
  `databricks-claude-sonnet-4-5`) — required only by the chart-analysis project
- Tesseract OCR (pre-installed on DBR) for the word-level-citation pipeline
- `uv` + `pytest` for the bundles' unit tests (`uv run pytest` from the repo root;
  no Databricks needed). A judge serving endpoint (default
  `databricks-claude-sonnet-4-6`) for `evaluation-harness` recipes 03 & 04.
- `pypdf` (installed by the notebook) for `parse-pdfs-with-large-num-of-pages`

## How to get help

Databricks support doesn't cover this content. For questions or bugs, please
open a GitHub issue and the team will help on a best-effort basis.

## License

&copy; 2026 Databricks, Inc. All rights reserved. The source in this notebook is
provided subject to the Databricks License
[https://databricks.com/db-license-source]. All included or referenced third
party libraries are subject to the licenses set forth below.

| library | description | license | source |
|---|---|---|---|
| pillow | Image cropping in the chart and word-level pipelines | HPND | https://pypi.org/project/pillow/ |
| openai | OpenAI-compatible client for the Databricks serving endpoint | Apache 2.0 | https://pypi.org/project/openai/ |
| markdown | Render VLM markdown output in notebooks | BSD | https://pypi.org/project/markdown/ |
| pyyaml | Cross-notebook config exchange | MIT | https://pypi.org/project/pyyaml/ |
| reportlab | Generate synthetic loan-file and paystub sample PDFs | BSD | https://pypi.org/project/reportlab/ |
| pytesseract | Tesseract OCR wrapper (word-level pipeline) | Apache 2.0 | https://pypi.org/project/pytesseract/ |
| rapidfuzz | OCR/value string-matching (word-level pipeline) | MIT | https://pypi.org/project/rapidfuzz/ |
| mlflow | GenAI evaluation in the evaluation-harness | Apache 2.0 | https://pypi.org/project/mlflow/ |
| matplotlib | Confidence-distribution plots (evaluation-harness) | matplotlib (BSD-style) | https://pypi.org/project/matplotlib/ |
| pypdf | PDF page-count detection (large-PDF parse recipe) | BSD | https://pypi.org/project/pypdf/ |
| scipy | Optimal array matching in nested eval (evaluation-harness) | BSD | https://pypi.org/project/scipy/ |
