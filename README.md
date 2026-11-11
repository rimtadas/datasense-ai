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

DataSense AI is an AI-powered data exploration application that helps users understand datasets and investigate analytical questions using natural language.

It combines deterministic Python-based analytics with LLM-powered agent orchestration. The language model interprets questions and selects analytical tools; Python and Pandas perform the actual calculations.

**Live demo:** https://huggingface.co/spaces/rimtadas/datasense-ai

> This is an evolving MVP. Please do not upload confidential or sensitive production data.

## Features

### 1. Dataset profiling

- Upload CSV and Parquet datasets.
- Load CSV and Parquet files from Hugging Face dataset repositories.
- Automatically profile datasets after loading.
- Display dataset dimensions and duplicate row counts.
- Identify missing and blank values.
- Detect column data types.
- Calculate unique value counts.
- Generate numerical statistics, including minimum, maximum, mean and median.
- Display frequent categorical values.
- Identify date ranges and categorical distributions.

### 2. Conversational analytics

- Ask analytical questions in natural language.
- Explore dataset schemas dynamically.
- Discover categorical values and distributions.
- Filter, group and aggregate data.
- Compare metrics across groups.
- Ask follow-up questions within a conversation.
- Use configurable semantic metadata for identifiers, monetary measures and date columns.
- Apply currency-aware aggregation safeguards.
- Enforce configured excluded-column access restrictions.

### 3. Session management

- Create and switch between multiple analysis sessions.
- Maintain separate datasets and conversation histories across sessions.
- Restore saved sessions.
- Delete individual sessions.
- Persist uploaded datasets and Hugging Face dataset references locally.

Session persistence depends on the configured storage location. The default local storage directory is `./data`. Persistent storage must be configured separately when deploying to an environment with ephemeral filesystem storage.

### 4. Analytics engine

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

The application separates five main responsibilities:

1. **Data ingestion:** CSV, Parquet and Hugging Face dataset loading.
2. **Dataset understanding:** Profiling, schema discovery and semantic metadata.
3. **Analytics engine:** Deterministic calculations, filtering, grouping and comparisons.
4. **Agent orchestration:** LLM reasoning, tool selection, validation and execution.
5. **User interface and persistence:** Gradio interface, conversation management and SQLite-backed sessions.

## Technology stack

- Python 3.13
- Gradio 6.31
- Pandas and NumPy
- PyArrow
- Hugging Face Hub and Datasets
- Hugging Face Inference API
- SQLite
- LLM-powered tool orchestration

## Running locally

### Prerequisites

- Python 3.13
- A Hugging Face account and access token with permission to use the configured inference model.

### 1. Clone the repository

```bash
git clone https://github.com/rimtadas/datasense-ai.git
cd datasense-ai
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_hugging_face_token
```

Never commit your `.env` file or expose your access token.

### 5. Start the application

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal.

## Dataset and resource limits

The current MVP applies the following limits:

| Resource | Limit |
|---|---:|
| Uploaded file size | 100 MB |
| Dataset rows | 1,000,000 |
| Dataset columns | 200 |
| Dataset memory | 512 MB |
| Sessions per workspace | 10 |
| Messages per session | 200 |
| Dataset storage per session | 120 MB |
| Workspace storage | 500 MB |

Supported file formats are CSV and Parquet (`.parquet` and `.pq`).

Hugging Face loading is restricted to supported dataset files and uses a resolved repository revision so a saved session can refer to the same dataset version.

## Evaluation

The MVP has been evaluated using an independent analytical benchmark, a second dataset with a different schema, and regression and security-focused tests.

| Evaluation | Result |
|---|---:|
| Analytical benchmark questions | 15/15 passed |
| Numerical accuracy checks | 53/53 passed |
| Cross-dataset portability questions | 6/6 passed |
| Regression tests | 42/42 passed |
| Prompt-injection tests | 6/6 passed |

These results reflect the tested datasets and scenarios. They are not a guarantee of correctness for every possible dataset or analytical question.

## Security and privacy considerations

- API credentials are supplied through environment variables or deployment secrets.
- Excluded columns are omitted from agent context and protected by analytical tool access checks.
- Analytical tool arguments and column references are validated.
- Transient inference API failures are handled with bounded retries.
- Application logs focus on operational metadata rather than dataset contents or conversation text.

**Important limitations:**

- Sensitive-column exclusion must be configured; automatic PII detection is not implemented.
- Browser workspace identifiers are not user authentication.
- The public demo is not designed for confidential or regulated datasets.
- Persistent cloud storage requires separately configured persistent storage.
- Security checks cover the tested MVP scenarios and are not a comprehensive security certification.

## Project status

DataSense AI is an evolving MVP focused on:

- Reliable, evidence-backed analytical answers.
- Dataset-agnostic analytics.
- Reusable analytical tools.
- Session-based exploration.
- Practical privacy and validation safeguards.

The goal is to make data exploration more accessible while keeping analytical calculations transparent and reproducible.