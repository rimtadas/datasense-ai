# DataSense AI — Day 4 Evaluation Report

**Date:** 10 November 2026
**Scope:** Analytical accuracy, tool selection, tool arguments, answer quality and cross-dataset portability.

## 1. Executive Summary

Day 4 focused on validating the reliability of the DataSense AI analytical agent and testing whether its core analytics functionality can operate on a dataset with a different schema.

The evaluation covered:
- A 1,000-row synthetic e-commerce transactions dataset with 18 columns.
- A separate 12-row synthetic customer-support dataset with 10 columns.
- Six reusable analytical tools.
- Fifteen predefined e-commerce benchmark questions.
- Six additional customer-support portability questions.

### Evaluation results

| Evaluation stage | Result | Score |
|---|---:|---:|
| Analytical accuracy | 51/51 checks | 100% |
| Strict tool selection | 11/15 questions | 73.33% |
| Functional tool coverage | 15/15 questions | 100% |
| Analytical stage coverage | 16/16 stages | 100% |
| Tool argument correctness | 52/52 checks | 100% |
| Tool result sufficiency | 35/35 checks | 100% |
| Final answer quality | 196/196 checks | 100% |
| Final answer question coverage | 15/15 questions | 100% |

All 15 benchmark questions completed successfully. No unexpected tool calls or failed tool executions were recorded in the tool-selection evaluation.

The strict tool-selection score is lower than the functional coverage score because four questions used valid alternative tools instead of their predefined preferred tools.

## 2. Analytical Accuracy

**Result: 51/51 checks passed (100%).**

The analytical accuracy evaluation compared agent-generated numerical results against independently calculated Pandas reference values.

The 15 questions covered:
- Dataset overview and schema.
- Date ranges.
- Numeric statistics.
- Categorical values and distributions.
- Filtered counts.
- Grouped aggregations.
- Comparisons between groups.
- Currency-aware transaction analysis.

The agent returned the expected analytical results for all 15 questions.

## 3. Tool Selection

### Results

| Metric | Result |
|---|---:|
| Strict adherence to expected tools | 11/15 (73.33%) |
| Functional tool coverage | 15/15 (100%) |
| Analytical stage coverage | 16/16 (100%) |
| Unexpected tool calls | 0 |
| Successful tool calls | 16 |
| Failed tool calls | 0 |

### Valid alternative tool selections

Four questions deviated from the predefined preferred tool sequence.

| Question | Preferred tool | Actual tool | Explanation |
|---|---|---|---|
| Q05 | `get_categorical_values` | `get_value_distribution` | The distribution tool returned unique categories, counts and percentages. |
| Q09 | `group_and_aggregate` | `get_value_distribution` | The distribution tool directly returned counts for each order status. |
| Q10 | `group_and_aggregate` | `get_value_distribution` | The distribution tool returned country transaction counts, from which the top five were identified. |
| Q14 | `get_categorical_values` + `compare_groups` | `get_value_distribution` + `compare_groups` | The distribution tool identified the two most common statuses; the comparison tool completed the currency-level analysis. |

These selections did not cause analytical failures. They demonstrate that the agent can sometimes choose a different but functionally appropriate analytical path.

### Efficiency improvement

Q10 contains one identified efficiency issue.

The question requests the five countries with the highest transaction counts. The agent called `get_value_distribution` without setting `limit=5`, retrieving all ten countries.

The answer was correct, but the tool call could have been more efficient by explicitly requesting the top five values.

## 4. Tool Argument and Result Sufficiency

**Results:**
- Argument correctness: 52/52 checks passed (100%).
- Result sufficiency: 35/35 checks passed (100%).
- Fully passed questions: 15/15.

The evaluation checked whether the agent supplied appropriate arguments for its selected tools and whether the returned tool results contained sufficient information to answer each question.

The evaluation also covered filters, grouping dimensions, metrics, aggregation methods, comparison groups and currency breakdowns.

One efficiency warning was identified for Q10, as described above.

## 5. Final Answer Quality

**Results:**
- Factual and completeness checks: 196/196 passed (100%).
- Questions fully passed: 15/15.

The final-answer evaluation checked whether responses included the required numerical results and facts from the benchmark.

The agent successfully communicated:
- Numerical results and category counts.
- Relevant filters and grouping dimensions.
- Currency-specific comparisons.
- Percentage differences where required.
- Explanations of the returned analytical results.

**Evaluation limitation:** These checks are deterministic and benchmark-specific. They do not constitute a comprehensive assessment of writing quality, every possible unsupported claim, subtle semantic errors or causal reasoning.

## 6. Cross-Dataset Portability

A second synthetic dataset was introduced to test the agent on a schema unrelated to the original e-commerce transactions dataset.

### Customer-support dataset

| Attribute | Value |
|---|---|
| Dataset | Customer support tickets |
| Rows | 12 |
| Physical columns | 10 |
| Agent-accessible columns | 9 |
| Excluded column | `internal_notes` |

The dataset included ticket identifiers, creation dates, support teams, priorities, regions, channels, resolution times, satisfaction scores and ticket statuses.

Dataset-specific metadata configured the ticket identifier, date column, column descriptions and excluded field.

### Portability questions

| Question | Analytical task | Tool exercised |
|---|---|---|
| P01 | Dataset overview and accessible columns | `get_dataset_overview` |
| P02 | Creation date range | `get_column_summary` |
| P03 | Ticket counts by support team | `get_categorical_values` |
| P04 | Priority distribution in the West region | `get_value_distribution` |
| P05 | Mean resolution time by support team for resolved tickets | `group_and_aggregate` |
| P06 | Satisfaction comparison between resolved and open tickets | `compare_groups` |

All six questions completed successfully without execution errors.

### Independent reference results

- Date range: 1–12 January 2026.
- Support team counts: Billing 5, Tech 4, Returns 3.
- West-region priority counts: High 3, Medium 2, Low 1.
- Mean resolution hours for resolved tickets: Billing 3.0, Returns 5.5, Tech 7.0.
- Mean satisfaction: Resolved 4.38, Open 2.00.

The agent responses matched the independently calculated reference results.

### Safeguard checks

Additional checks were performed for missing-value profiling and excluded-column enforcement.

| Check | Result |
|---|---|
| Missing values in `resolution_hours` | 1 detected |
| Missing values in `satisfaction_score` | 1 detected |
| Access to excluded `internal_notes` | Correctly blocked |

The exclusion check returned:

`Column 'internal_notes' is excluded from agent access.`

### Portability conclusion

The test demonstrates that all six analytical tools can operate on a different dataset schema without changing their core implementations.

It also demonstrates configurable dataset metadata, dynamic date handling, filtering, grouping, comparisons, missing-value profiling and excluded-column enforcement.

**Scope limitation:** This is a small synthetic portability test, not a comprehensive validation across many datasets, formats, data volumes or real-world edge cases. The results were checked against independent reference calculations during test execution; this is not yet a broad automated cross-dataset benchmark.

## 7. Architecture and Reliability Safeguards

The current agent architecture separates:
- Dataset loading.
- Dataset context and schema discovery.
- Reusable analytical functions.
- LLM-based tool selection.
- Deterministic Python execution.
- Evaluation and result logging.

The core analytics tools are designed to use the active DataFrame and its metadata rather than assume a fixed e-commerce schema.

Monetary measures can be configured through semantic metadata. When a monetary measure has a configured currency column, the analytics tools enforce currency-aware aggregation safeguards.

The LLM selects analytical tools, while Python and Pandas perform the calculations.

## 8. Current Limitations

The evaluation does not establish that the agent is production-ready.

Current limitations include:

1. The main benchmark uses one synthetic e-commerce dataset.
2. The portability test uses a small, in-memory synthetic customer-support dataset.
3. The runner still loads the default e-commerce dataset when the agent module is imported.
4. User-facing dataset selection and uploaded-data integration have not yet been connected to the conversational agent.
5. The Gradio profiling application and conversational agent remain separate components.
6. Dataset-specific semantic metadata may require configuration.
7. Cross-table joins and multi-dataset investigations are not implemented.
8. Deterministic answer-quality checks do not cover every semantic or reasoning failure.
9. Descriptive comparisons do not establish causation.

## 9. Day 4 Conclusion

The Day 4 evaluation demonstrates that the current agent can complete the 15-question benchmark with correct numerical results, appropriate analytical coverage and sufficient tool outputs.

The second-dataset test provides initial evidence that the analytics engine is not tied to the original e-commerce schema.

The main improvement identified is more consistent tool selection and more efficient use of tool parameters, particularly the top-five limit in Q10.

The next development stage should focus on integrating the agent into the Gradio application, supporting user-selected datasets and enabling reliable follow-up investigations.

---

**Overall assessment:** The analytical benchmark passed, all six analytical tools were exercised successfully on a second schema, and the principal remaining work is product integration, broader validation and improved agent efficiency.
