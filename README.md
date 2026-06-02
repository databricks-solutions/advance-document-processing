# Advanced Document Processing

Reference implementations for advanced document-processing pipelines on
Databricks — combining Unity Catalog, `ai_parse_document`, task-specific AI
Functions, vision LLMs, Auto Loader, and Databricks Asset Bundles.

## Projects

| Project | Description |
|---|---|
| [`document-embedding-chart-analysis/`](./document-embedding-chart-analysis/) | End-to-end PDF chart-analysis pipeline. Parses PDFs with `ai_parse_document`, classifies and crops chart figures, runs a vision LLM on each chart, and splices the resulting insights back into a per-document gold table. Ships as both interactive batch notebooks and a streaming Databricks Asset Bundle. |

Each project's own `README.md` covers its architecture, defaults, and quickstart.

## Repo layout

```
advance-document-processing/
├── document-embedding-chart-analysis/   # PDF chart-analysis pipeline (notebooks + DAB)
├── scripts/                             # Cross-project helper scripts
│   └── upload_pdfs.sh                   # Upload local PDFs to a UC Volume via the CLI
├── CODEOWNERS.txt
├── LICENSE.md
├── NOTICE.md
└── SECURITY.md
```

## Prerequisites

- Databricks workspace with Unity Catalog and Serverless Jobs enabled
- DBR **17.3+** (or serverless environment version **3+**) for `ai_parse_document`;
  serverless env **5** is recommended (used by the streaming bundle)
- Databricks CLI **v0.205+** (the unified CLI) for `bundle` and `fs` commands
- A multimodal serving endpoint for chart analysis (default
  `databricks-claude-sonnet-4-5`)

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
| pillow | Image cropping in the chart pipeline | HPND | https://pypi.org/project/pillow/ |
| openai | OpenAI-compatible client for the Databricks serving endpoint | Apache 2.0 | https://pypi.org/project/openai/ |
| markdown | Render VLM markdown output in notebooks | BSD | https://pypi.org/project/markdown/ |
| pyyaml | Cross-notebook config exchange | MIT | https://pypi.org/project/pyyaml/ |
