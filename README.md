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

**From dataset profiling to conversational analytics.**

DataSense AI is an AI-powered data exploration application that helps users understand datasets and investigate business questions using natural language.

It combines deterministic Python-based analytics with LLM-powered agent orchestration to generate evidence-backed answers.

## Features

### Dataset profiling
- Upload CSV and Parquet datasets.
- Display dataset dimensions and duplicate row counts.
- Identify missing and blank values.
- Detect column data types.
- Calculate unique value counts.
- Generate numerical statistics, including minimum, maximum, mean and median.
- Display frequent categorical values.
- Identify date ranges and categorical distributions.

### Conversational analytics
- Ask analytical questions in natural language.
- Explore dataset schemas dynamically.
- Discover categorical values and distributions.
- Filter, group and aggregate data.
- Compare metrics across groups.
- Ask follow-up questions within a conversation.
- Use configurable semantic metadata for identifiers, monetary measures and date columns.
- Apply currency-aware aggregation safeguards.
- Enforce excluded-column access restrictions.

### Analytics engine

The application uses reusable analytical tools:

| Tool | Purpose |
|---|---|
| `get_dataset_overview` | Dataset dimensions and schema |
| `get_column_summary` | Column statistics and missing values |
| `get_categorical_values` | Discover distinct categorical values |
| `get_value_distribution` | Category counts and percentages |
| `group_and_aggregate` | Filtered grouping and aggregation |
| `compare_groups` | Compare metrics across groups |

Calculations are performed using Python and Pandas. The language model selects tools and explains their results.

## Architecture

The application separates:

1. Data ingestion — CSV, Parquet and Hugging Face datasets.
2. Dataset understanding — profiling, schema discovery and semantic metadata.
3. Analytics engine — deterministic calculations and comparisons.
4. Agent orchestration — LLM reasoning and tool execution.
5. Gradio interface — dataset upload, profiling and conversational analytics.

## Evaluation

The current MVP has been tested using an independent benchmark and a second dataset with a different schema.

| Evaluation | Result |
|---|---:|
| Analytical benchmark questions | 15/15 passed |
| Numerical accuracy checks | 53/53 passed |
| Cross-dataset portability questions | 6 executed |
| Regression tests | 36/36 passed |
| Prompt-injection tests | 6/6 passed |

These results reflect the tested datasets and scenarios, not a guarantee of correctness for every possible dataset or question.

## Technology stack

- Python
- Gradio
- Pandas and NumPy
- Hugging Face Hub and Datasets
- LLM-powered tool orchestration
- Deterministic analytical functions

## Project status

DataSense AI is an evolving MVP. Current development focuses on analytical reliability, dataset portability, security safeguards and improving the user experience.