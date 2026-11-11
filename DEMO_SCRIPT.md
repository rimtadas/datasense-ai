# DataSense AI — MVP Demo

## 1. Introduction

DataSense AI helps users explore datasets, understand data quality and investigate analytical questions using natural language.

It combines deterministic Python-based analytical tools with LLM-powered tool orchestration. The model selects analytical operations, while Python and Pandas calculate the results.

**Live demo:** https://huggingface.co/spaces/rimtadas/datasense-ai

## 2. Dataset profiling

Upload the sample e-commerce dataset.

Demonstrate:

- Automatic dataset profiling.
- Number of rows and columns.
- Column data types.
- Missing and blank values.
- Duplicate rows.
- Numerical and categorical summaries.

Explain that the profiling workflow is designed to work with different dataset schemas.

## 3. Analytical question — filtering and distribution

Ask:

> Among cancelled transactions, what is the payment method distribution?

Demonstrate how DataSense AI filters the dataset and calculates category counts and percentages.

Show the analytical result table.

## 4. Analytical question — aggregation and semantic metadata

Ask:

> What is the average transaction amount for delivered orders, broken down by currency?

Explain how the application filters delivered transactions, groups by currency and calculates the average.

Highlight the currency-aware aggregation safeguards.

## 5. Dataset portability

Load the customer-support dataset.

Ask:

> What is the average resolution time in hours for each support team, considering only Resolved tickets?

Explain that the analytical tools operate on the active dataset schema rather than depending on fixed e-commerce columns.

## 6. Session management

Demonstrate:

- Creating a new session.
- Loading a different dataset.
- Switching between sessions.
- Restoring a previous conversation.
- Deleting a session.

Explain that sessions retain their dataset reference or uploaded dataset and conversation history.

## 7. Closing

Summarise the main capabilities:

- Dataset-driven profiling.
- Natural-language analytical questions.
- Tool-based calculations.
- Filtering, grouping and comparisons.
- Dataset portability.
- Session-based exploration.
- Configurable column-access safeguards.

## 8. Evaluation evidence

| Evaluation | Result |
|---|---:|
| Analytical benchmark questions | 15/15 passed |
| Numerical accuracy checks | 53/53 passed |
| Cross-dataset portability questions | 6/6 passed |
| Regression tests | 42/42 passed |
| Prompt-injection tests | 6/6 passed |

These results demonstrate the tested MVP capabilities. They are not a guarantee of correctness for every dataset or every possible question.