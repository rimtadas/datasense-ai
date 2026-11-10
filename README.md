---
title: DataSense AI
emoji: 📚
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
python_version: 3.13
app_file: app.py
short_description: Intelligent Data Discovery & Analytics Agent
---

# DataSense AI

**From dataset profiling to autonomous analytics.**

DataSense AI is an evolving data intelligence application designed to help users understand datasets and explore business questions using natural language.

It combines deterministic Python-based data analysis with LLM-powered agent orchestration. The goal is to let users investigate datasets, ask follow-up questions and receive evidence-backed analytical explanations.

## Current Project Status

DataSense AI currently has two development components:

1. **Dataset profiling application:** A Gradio application that accepts CSV and Parquet uploads and generates automated profiling reports.
2. **Conversational analytics agent:** A separate working prototype that uses an LLM to select analytical tools, execute Python calculations and explain results.

The conversational agent has not yet been integrated into the Gradio application.

### Agent evaluation status

The agent has completed its initial 15-question analytical benchmark and a separate cross-dataset portability test.

| Evaluation | Result |
|---|---:|
| Analytical accuracy | 51/51 checks (100%) |
| Strict tool selection | 11/15 (73.33%) |
| Functional tool coverage | 15/15 (100%) |
| Tool argument correctness | 52/52 checks (100%) |
| Tool result sufficiency | 35/35 checks (100%) |
| Final answer quality | 196/196 checks (100%) |
| Second-dataset portability | 6/6 questions completed |

The tool-selection deviations were valid alternative tool choices rather than failed analytical answers. One efficiency improvement remains: Q10 should request a top-five distribution directly instead of retrieving all country values.

See [Day 4 Evaluation Report](evaluation/DAY_4_EVALUATION_REPORT.md) for the methodology, detailed results, deviations and limitations.

## Current Features

### Dataset Profiling

- Upload CSV and Parquet datasets.
- Display dataset dimensions and duplicate row counts.
- Identify missing and blank values.
- Detect column data types.
- Calculate unique value counts.
- Generate numeric statistics, including minimum, maximum, mean and median.
- Display frequent categorical values.
- Identify date ranges and categorical distributions.

### Conversational Analytics Agent

The separate agent prototype currently supports:

- Natural-language analytical questions.
- Dynamic dataset schema discovery and context generation.
- Six reusable analytical tools.
- Deterministic Python and Pandas calculations.
- Categorical discovery and value distributions.
- Filtering, grouping and aggregation.
- Comparisons between groups.
- Configurable semantic metadata for identifiers, monetary measures and date columns.
- Currency-aware aggregation safeguards for configured monetary fields.
- Excluded-column enforcement.
- Structured tool execution results.
- Multiple tool rounds within an agent run.

### Analytics Tools

| Tool | Purpose |
|---|---|
| `get_dataset_overview` | Dataset dimensions, schema and overview |
| `get_column_summary` | Column-level statistics and missing values |
| `get_categorical_values` | Discover distinct categorical values |
| `get_value_distribution` | Category counts and percentages |
| `group_and_aggregate` | Filtered grouping and aggregation |
| `compare_groups` | Compare metrics across groups |

The tools are designed to operate on different DataFrame schemas rather than depend on fixed e-commerce column names.

## Architecture

The project separates data ingestion, dataset understanding, analytical execution and agent orchestration.

```text
DataSense AI
|
|-- Dataset Ingestion
|   |-- CSV / Parquet
|   `-- Hugging Face datasets
|
|-- Dataset Understanding
|   |-- Dataset profiling
|   |-- Dynamic schema discovery
|   |-- Data type detection
|   `-- Configurable semantic metadata
|
|-- Analytics Engine
|   |-- Dataset overview
|   |-- Column summaries
|   |-- Category discovery
|   |-- Value distributions
|   |-- Grouping and aggregation
|   `-- Group comparisons
|
|-- Agent Orchestration
|   |-- LLM reasoning
|   |-- Tool selection
|   |-- Deterministic Python execution
|   `-- Evidence-based responses
|
`-- Evaluation
    |-- Independent expected results
    |-- Analytical accuracy
    |-- Tool selection
    |-- Tool arguments
    |-- Answer quality
    `-- Cross-dataset portability
