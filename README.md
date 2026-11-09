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

The project combines deterministic Python-based data analysis with LLM-powered agent orchestration. Its longer-term goal is to let users investigate datasets, ask follow-up questions and receive evidence-backed analytical explanations.

## Current Project Status

DataSense AI currently has two development components:

1. **Dataset profiling application:** A Gradio application that accepts CSV and Parquet uploads and generates automated profiling reports.
2. **Conversational analytics agent:** A separate working prototype that uses an LLM to select analytical tools, execute Python calculations and explain results.

The conversational agent has not yet been integrated into the Gradio application.

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

### Conversational Analytics Agent (Development Prototype)

- Interpret natural-language analytical questions.
- Discover available dataset columns and their analytical roles.
- Select from six reusable analytical tools.
- Calculate results using Python and Pandas.
- Support filtering, grouping, aggregation and group comparisons.
- Calculate categorical distributions and percentages.
- Apply configurable semantic metadata, including identifiers, monetary measures and date columns.
- Retain structured tool outputs for evaluation.
- Support multi-step tool execution and follow-up investigations within an agent run.

### Evaluation Foundation

- 15 predefined analytical evaluation questions.
- Independent Pandas-based expected-result calculations.
- Structured tool execution logging.
- Selective question execution.
- Incremental evaluation result storage.

Automated numerical accuracy scoring and tool-selection scoring are planned for Day 4.

## Architecture

The project separates dataset understanding, analytics and agent orchestration.

```text
DataSense AI
│
├── Dataset Ingestion
│   ├── CSV / Parquet uploads
│   └── Hugging Face datasets
│
├── Dataset Understanding
│   ├── Dataset profiling
│   ├── Schema discovery
│   ├── Data type detection
│   └── Semantic metadata
│
├── Analytics Engine
│   ├── Dataset overview
│   ├── Column summaries
│   ├── Category discovery
│   ├── Value distributions
│   ├── Grouping and aggregation
│   └── Group comparisons
│
├── Agent Orchestration
│   ├── LLM reasoning
│   ├── Tool selection
│   ├── Python tool execution
│   └── Evidence-based responses
│
└── Evaluation
    ├── Exam questions
    ├── Expected results
    └── Execution and accuracy evaluation
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Data processing | Pandas, NumPy |
| Application interface | Gradio |
| LLM inference | Hugging Face Inference API |
| Current model | `openai/gpt-oss-120b:fastest` |
| Dataset ecosystem | Hugging Face |
| Data formats | CSV, Parquet |
| Version control | Git, GitHub |

## Project Structure

```text
datasense-ai/
├── agent/
│   ├── __init__.py
│   └── runner.py
├── analytics/
│   ├── __init__.py
│   └── tools.py
├── data_connector/
│   ├── __init__.py
│   └── huggingface_loader.py
├── dataset_context/
│   ├── __init__.py
│   └── builder.py
├── profiler/
│   ├── __init__.py
│   └── engine.py
├── evaluation/
│   ├── __init__.py
│   ├── exam_questions.py
│   ├── expected_results.py
│   ├── run_exam.py
│   └── baseline_results.csv
├── app.py
├── test_agent.py
├── test_expected_results.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

Additional test scripts are maintained in the project root during development.

## Dataset

The initial analytics-agent prototype uses a public synthetic e-commerce dataset:

[ davidev07/synthetic-ecommerce-dataset ](https://huggingface.co/datasets/davidev07/synthetic-ecommerce-dataset)

The transactions table contains 1,000 rows and 18 columns.

The project is designed to evolve beyond this initial dataset. Core analytics functions are intended to work across different schemas, with dataset-specific semantic metadata supplied separately.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/rimtadas/datasense-ai.git
cd datasense-ai
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure Hugging Face inference

Create a `.env` file in the project root:

```text
HF_TOKEN=your_huggingface_inference_token
```

Keep this file out of version control. Never commit API tokens.

### 6. Run the profiling application

```powershell
python app.py
```

Open the local Gradio URL displayed in the terminal, usually:

`http://127.0.0.1:7860`

### 7. Run the analytics agent prototype

```powershell
python test_agent.py
```

### 8. Run the evaluation exam

Run all questions:

```powershell
python -m evaluation.run_exam
```

Run selected questions:

```powershell
python -m evaluation.run_exam --question-ids Q13
```

Evaluation results are saved incrementally in `evaluation/baseline_results.csv`.

## Development Roadmap

### Day 1–2: Dataset Profiling and Analytics Foundation
- Dataset ingestion and profiling.
- Reusable analytical functions.
- Dataset schema and context generation.
- Support for configurable semantic metadata.

### Day 3: Agent Foundation and Initial Evaluation
- LLM-powered analytical tool selection.
- Integration of six deterministic Python analytics tools.
- Multi-step agent execution.
- Fifteen evaluation questions.
- Independent expected-result calculations.
- Structured tool-result retention.
- Incremental evaluation baseline.

### Day 4: Agent Evaluation and Optimization
- Automated numerical accuracy evaluation.
- Tool-selection accuracy.
- Execution reliability scoring.
- Evaluation-driven prompt and tool improvements.
- Agent efficiency and context optimization.

### Upcoming Development
- Integrate the conversational agent into the Gradio interface.
- Support multiple datasets and user-uploaded data.
- Improve automatic semantic-role detection.
- Enable multi-step business investigations.
- Add visualizations and analytical summaries.
- Support evidence-backed trend and anomaly investigations.
- Expand to related datasets and cross-table analysis.

## Current Limitations

- The conversational agent is a development prototype, separate from the deployed profiling interface.
- The initial agent evaluation uses one synthetic e-commerce transactions table.
- Numerical accuracy has not yet been fully benchmarked across all 15 questions.
- Dataset-specific metadata currently requires configuration.
- Cross-table joins and multi-dataset investigations are not yet implemented.
- Descriptive analytics can identify patterns but cannot establish causation on their own.

## Project Goal

The long-term goal is to build an autonomous analytics assistant that can understand unfamiliar datasets, select appropriate analytical methods, investigate business questions through multiple steps and communicate findings with transparent numerical evidence.

DataSense AI aims to make data exploration more accessible without sacrificing the reliability of Python-based analytical calculations.