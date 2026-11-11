# DataSense AI — Architecture

## 1. Overview

DataSense AI uses a modular architecture to separate dataset ingestion, profiling, analytical execution, LLM orchestration, and session persistence.

The central design principle is that **the LLM interprets questions and selects tools, while Python performs the actual analytical calculations.**

This keeps numerical results grounded in the underlying dataset rather than relying on generated arithmetic.

## 2. Application components

| Component | Responsibility |
|---|---|
| `app.py` | Gradio interface, dataset workflows, chat interaction and application state |
| `profiler.py` | Dataset profiling and descriptive statistics |
| `data_connector/huggingface_loader.py` | Hugging Face repository discovery, validation and dataset loading |
| `dataset_context/builder.py` | Builds dataset context and applies excluded-column configuration |
| `agent/runner.py` | LLM interaction, tool selection, argument validation, dispatch and retry handling |
| `analytics/tools.py` | Deterministic Pandas-based analytical functions |
| `session_store.py` | SQLite-backed session metadata, conversation history and dataset persistence |
| `exceptions.py` | User-facing application exceptions |
| `evaluation/` | Benchmark execution and analytical accuracy evaluation |

## 3. Data flow

### Local dataset workflow

1. User uploads a CSV or Parquet file.
2. The application validates file type, size, dimensions and dataset structure.
3. The dataset is loaded and profiled.
4. Dataset metadata and a persisted dataset copy are associated with the active session.
5. The user can ask analytical questions.

### Hugging Face dataset workflow

1. User provides a dataset repository.
2. The application lists supported CSV and Parquet files.
3. The loader performs a remote file-size and revision preflight check.
4. The selected dataset is downloaded using the resolved repository revision.
5. The dataset is validated, loaded and profiled.
6. The repository, filename and resolved revision are saved in the session.

### Conversational analytics workflow

1. User submits a natural-language question.
2. The agent receives the question, dataset context and conversation history.
3. The LLM selects an available analytical tool and supplies its arguments.
4. The application validates the requested columns, filters and operations.
5. The selected Python analytical function executes against the active DataFrame.
6. Tool results are returned to the LLM for explanation.
7. The final response and conversation messages are displayed and persisted.

## 4. Analytical tools

The analytics layer provides reusable functions for:

- Dataset overview and schema discovery.
- Column summaries.
- Categorical value discovery.
- Value distributions.
- Filtering, grouping and aggregation.
- Comparison of metrics across groups.

The functions are designed to operate on different dataset schemas rather than relying on a single fixed business dataset.

Semantic metadata can be supplied to identify identifiers, monetary measures, datetime columns and excluded columns.

## 5. Agent orchestration

The agent runner manages interactions with the configured Hugging Face Inference API model.

Its responsibilities include:

- Constructing model messages and tool definitions.
- Handling tool selection and execution.
- Validating analytical arguments.
- Maintaining conversation context.
- Retrying selected transient API failures with bounded attempts.
- Collecting operational metrics such as execution time and token usage.

Analytical tool executions are not automatically repeated when an inference API request fails.

## 6. Session persistence

Session persistence uses SQLite for session metadata and conversation messages, with dataset files stored separately.

The application supports:

- Multiple sessions within a browser workspace.
- Separate datasets and conversation histories per session.
- Session restoration and deletion.
- Persisted local uploads.
- Hugging Face dataset references pinned to a repository revision.

The workspace identifier is maintained through browser state. It is an opaque identifier, not an authentication mechanism.

The storage directory defaults to `./data` and can be configured using:

```env
DATA_SENSE_STORAGE_DIR=/path/to/persistent/storage
```

For deployments with ephemeral filesystems, persistent storage must be configured separately.

## 7. Security and validation boundaries

The MVP includes:

- File type, size and dataset dimension checks.
- Analytical argument and column validation.
- Configurable excluded-column restrictions.
- Bounded retries for selected transient inference failures.
- Environment-based API credential configuration.
- Operational logging that avoids recording dataset contents and conversation text.

Excluded-column protection depends on configuration. Automatic PII detection is not implemented.

Browser workspace identifiers do not provide user authentication or strong isolation.

## 8. Evaluation approach

The application has been evaluated using:

- An independent analytical benchmark.
- Numerical checks against expected results.
- A second dataset with a different schema.
- Regression tests.
- Prompt-injection-focused tests.

Results are documented in the project README. They represent tested scenarios and should not be interpreted as a guarantee of correctness or comprehensive security.

## 9. Current scope

DataSense AI is an MVP. It focuses on reliable dataset exploration and reusable analytical tools rather than production-scale infrastructure.

It does not currently provide user authentication, collaboration, automatic PII detection, comprehensive audit logging or production-grade rate limiting.