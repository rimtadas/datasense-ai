---
title: DataSense AI
emoji: 📚
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
python_version: 3.13
app_file: app.py
short_description: Intelligent Data Discovery & Profiling Assistant
---

# DataSense AI

### Intelligent Data Discovery & Profiling Assistant

DataSense AI is a data profiling application designed to help users quickly understand the structure, quality and statistical characteristics of their datasets.

## Current Features

- Upload CSV and Parquet datasets.
- View dataset dimensions (rows and columns).
- Identify missing and blank values.
- Detect duplicate rows.
- Identify column data types.
- Calculate unique value counts.
- Generate numeric statistics such as minimum, maximum, mean and median.
- Display the most frequent categorical values.

## Tech Stack

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Interface:** Gradio
- **Storage / Dataset Ecosystem:** Hugging Face
- **Version Control:** Git and GitHub

## Project Structure

```text
datasense-ai/
├── app.py
├── profiler/
│   ├── __init__.py
│   └── engine.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/rimtadas/datasense-ai.git
cd datasense-ai
```

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal, usually:

`http://127.0.0.1:7860`

## Usage

1. Upload a CSV or Parquet file.
2. Click **Analyze Dataset**.
3. Review the dataset overview.
4. Explore the column profiling report.

## Development Roadmap

- Improve automatic data type and date detection.
- Expand data quality checks.
- Add data visualisations.
- Introduce natural-language data exploration.
- Integrate LLM-powered analytical capabilities.
- Develop agentic workflows for investigating business metrics and explaining changes.

## Project Status

Initial prototype: dataset upload and automated profiling functionality implemented. Further development is