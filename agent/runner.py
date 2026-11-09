
import os
import json

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from data_connector.huggingface_loader import (
    load_huggingface_dataset
)

from dataset_context import build_dataset_context

from analytics.tools import (
    get_dataset_overview,
    get_column_summary,
    get_categorical_values,
    get_value_distribution,
    group_and_aggregate,
    compare_groups,
)


# --------------------------------------------------
# 1. INITIALIZATION
# --------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from the .env file.")

client = InferenceClient(token=HF_TOKEN)

MODEL = "openai/gpt-oss-120b:fastest"

REPO_ID = "davidev07/synthetic-ecommerce-dataset"
DATASET_FILE = "transactions.csv"


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename=DATASET_FILE,
)

# Parse date column for this dataset.
df["purchase_date"] = pd.to_datetime(
    df["purchase_date"],
    errors="coerce",
)

# Dataset-specific metadata.
# The analytics tools remain dataset-agnostic.
dataset_config = {
    "dataset_description": (
        "Synthetic e-commerce transaction dataset."
    ),
    "semantic_types": {
        "transaction_id": "identifier",
        "user_id": "identifier",
        "product_id": "identifier",
        "purchase_date": "datetime",
        "unit_price": "monetary",
        "subtotal": "monetary",
        "discount_amount": "monetary",
        "total_amount": "monetary",
    },
    "currency_column": "currency",
    "column_descriptions": {
        "total_amount": "Final transaction amount.",
        "quantity": "Number of product units purchased.",
        "payment_status": "Status of the transaction payment.",
        "country": "Country associated with the transaction.",
        "purchase_date": "Date and time of the purchase.",
    },
    "exclude_from_context": [
        "shipping_address"
    ],
}

print("Dataset loaded:", len(df), "rows")


# --------------------------------------------------
# 3. BUILD DATASET CONTEXT
# --------------------------------------------------

dataset_context = build_dataset_context(
    df=df,
    dataset_name=DATASET_FILE,
    dataset_config=dataset_config,
)

context_json = json.dumps(
    dataset_context,
    indent=2,
    default=str,
)

print("Dataset context generated.")

excluded_columns = set(
    dataset_config.get("exclude_from_context", [])
)


# --------------------------------------------------
# 4. DEFINE ANALYTICAL TOOLS
# --------------------------------------------------

tools = [

    # TOOL 1: DATASET OVERVIEW

    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": (
                "Get the dataset overview, including row count, "
                "column count, available columns and duplicate row count. "
                "Excluded columns are not exposed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    # TOOL 2: COLUMN SUMMARY

    {
        "type": "function",
        "function": {
            "name": "get_column_summary",
            "description": (
                "Get a summary of a dataset column. "
                "Returns data type, semantic type, missing values, "
                "unique values and relevant statistics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Exact column name from the dataset context."
                        ),
                    },
                },
                "required": ["column"],
            },
        },
    },

    # TOOL 3: CATEGORY DISCOVERY

    {
        "type": "function",
        "function": {
            "name": "get_categorical_values",
            "description": (
                "Discover the actual unique values and their frequencies "
                "for a categorical or Boolean column. Use this tool when "
                "the user asks what categories, values, statuses, types "
                "or options exist in a column, or needs to verify category "
                "values before filtering or comparing groups. "
                "Returns category values, counts, unique count and "
                "whether the results were truncated. "
                "For filtered distributions or questions requiring "
                "percentages, use get_value_distribution instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Exact categorical column name from the dataset context."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of category values to return. "
                            "Defaults to 30."
                        ),
                    },
                },
                "required": ["column"],
            },
        },
    },

    # TOOL 4: VALUE DISTRIBUTION

    {
        "type": "function",
        "function": {
            "name": "get_value_distribution",
            "description": (
                "Calculate a complete frequency distribution for a "
                "categorical or Boolean column. Returns each category's "
                "count and percentage, total rows after filtering, "
                "missing-value count and percentage, unique category "
                "count and truncation status. Supports filters on other "
                "columns, such as filtering order_status to Cancelled "
                "before calculating payment_method distribution. "
                "Percentages are calculated in Python using all filtered "
                "rows as the denominator, including rows with missing "
                "values. Use this tool whenever the user asks for category "
                "counts and percentages, proportions, distributions, "
                "or most common values. Do not calculate these percentages "
                "yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Categorical column whose value distribution "
                            "should be calculated."
                        ),
                    },
                    "filters": {
                        "type": ["object", "null"],
                        "description": (
                            "Optional filters applied before calculating "
                            "the distribution. Example: "
                            "{'order_status': 'Cancelled'}."
                        ),
                        "additionalProperties": True,
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": (
                            "Optional maximum number of categories to return. "
                            "Use null for all categories."
                        ),
                    },
                },
                "required": ["column"],
            },
        },
    },

    # TOOL 5: GROUP AND AGGREGATE

    {
        "type": "function",
        "function": {
            "name": "group_and_aggregate",
            "description": (
                "Group dataset records and calculate an aggregation. "
                "Can also filter, sort and limit the results. "
                "For monetary metrics, respect the dataset currency "
                "configuration: group by currency or filter to one currency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": ["string", "null"],
                        "description": (
                            "Column used to group the data. "
                            "Use null for an overall aggregation."
                        ),
                    },
                    "metric": {
                        "type": ["string", "null"],
                        "description": (
                            "Column to aggregate. Use null for row counts."
                        ),
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": [
                            "count",
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                        ],
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional filters expressed as column-value pairs."
                        ),
                        "additionalProperties": True,
                    },
                    "sort_by": {
                        "type": "string",
                        "description": (
                            "Sort by 'result' or a grouping column."
                        ),
                    },
                    "ascending": {
                        "type": "boolean",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "Maximum number of result rows.",
                    },
                },
                "required": ["aggregation"],
            },
        },
    },

    # TOOL 6: COMPARE GROUPS

    {
        "type": "function",
        "function": {
            "name": "compare_groups",
            "description": (
                "Compare the same aggregation for two values in a column. "
                "Use this for questions comparing groups, such as Delivered "
                "versus Cancelled. It calculates both group values, the "
                "absolute difference (group_b minus group_a), and percentage "
                "difference in Python. Use group_by to compare separately "
                "across another dimension such as currency. Monetary metrics "
                "must be grouped by currency or filtered to one currency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "comparison_column": {
                        "type": "string",
                        "description": (
                            "Column containing the two groups to compare."
                        ),
                    },
                    "group_a": {
                        "type": ["string", "number", "boolean"],
                        "description": (
                            "First group; used as the comparison baseline."
                        ),
                    },
                    "group_b": {
                        "type": ["string", "number", "boolean"],
                        "description": (
                            "Second group; compared against group_a."
                        ),
                    },
                    "metric": {
                        "type": ["string", "null"],
                        "description": (
                            "Numeric column to aggregate. Use null for row counts."
                        ),
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": [
                            "count",
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                        ],
                    },
                    "group_by": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional column to break comparisons down by, "
                            "for example currency. Use null for one overall comparison."
                        ),
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional additional filters applied to both groups."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": [
                    "comparison_column",
                    "group_a",
                    "group_b",
                    "aggregation",
                ],
            },
        },
    },
]


# --------------------------------------------------
# 5. VALIDATE COLUMN ACCESS
# --------------------------------------------------

def validate_allowed_column(column):

    if column is None:
        return

    if column in excluded_columns:
        raise ValueError(
            f"Column '{column}' is excluded from agent access."
        )


def validate_filter_columns(filters):

    if not filters:
        return

    if not isinstance(filters, dict):
        raise ValueError("Filters must be a dictionary.")

    for column in filters:
        validate_allowed_column(column)


# --------------------------------------------------
# 6. MAP TOOL NAMES TO PYTHON FUNCTIONS
# --------------------------------------------------

def execute_tool(function_name, arguments):

    if function_name == "get_dataset_overview":

        return get_dataset_overview(
            df,
            dataset_config=dataset_config,
        )

    elif function_name == "get_column_summary":

        column = arguments["column"]

        validate_allowed_column(column)

        return get_column_summary(
            df,
            column=column,
            dataset_config=dataset_config,
        )

    elif function_name == "get_categorical_values":

        column = arguments["column"]

        validate_allowed_column(column)

        return get_categorical_values(
            df,
            column=column,
            limit=arguments.get("limit", 30),
        )

    elif function_name == "get_value_distribution":

        column = arguments["column"]
        filters = arguments.get("filters")

        validate_allowed_column(column)
        validate_filter_columns(filters)

        return get_value_distribution(
            df=df,
            column=column,
            filters=filters,
            limit=arguments.get("limit"),
            dataset_config=dataset_config,
        )

    elif function_name == "group_and_aggregate":

        group_by = arguments.get("group_by")
        metric = arguments.get("metric")
        filters = arguments.get("filters")

        validate_allowed_column(group_by)
        validate_allowed_column(metric)
        validate_filter_columns(filters)

        return group_and_aggregate(
            df,
            group_by=group_by,
            metric=metric,
            aggregation=arguments["aggregation"],
            filters=filters,
            sort_by=arguments.get("sort_by", "result"),
            ascending=arguments.get("ascending", False),
            limit=arguments.get("limit"),
            dataset_config=dataset_config,
        )

    elif function_name == "compare_groups":

        comparison_column = arguments["comparison_column"]
        group_by = arguments.get("group_by")
        metric = arguments.get("metric")
        filters = arguments.get("filters")

        validate_allowed_column(comparison_column)
        validate_allowed_column(group_by)
        validate_allowed_column(metric)
        validate_filter_columns(filters)

        return compare_groups(
            df=df,
            comparison_column=comparison_column,
            group_a=arguments["group_a"],
            group_b=arguments["group_b"],
            metric=metric,
            aggregation=arguments["aggregation"],
            group_by=group_by,
            filters=filters,
            dataset_config=dataset_config,
        )

    else:

        raise ValueError(
            f"Unknown tool: {function_name}"
        )

# --------------------------------------------------
# 7. DEFINE SYSTEM PROMPT
# --------------------------------------------------

system_prompt = f"""
You are DataSense AI, an autonomous analytical assistant.

Your job is to answer questions about the currently loaded dataset.

You have access to analytical tools that execute real Python calculations.

IMPORTANT RULES:

1. Use the dataset context to understand available columns,
   data types, descriptions and semantic roles.

2. Never invent column names. Use only columns present in the context.

3. Never access columns listed under excluded_columns.

4. Use analytical tools to retrieve actual data and calculate results.
   Never invent statistics or calculate dataset metrics yourself.

5. Use get_dataset_overview for questions about dataset size
   and general overview.

6. Use get_column_summary to investigate individual columns.

7. Use get_categorical_values when:
   - The user asks what values or categories exist in a column.
   - The user asks how many distinct categories exist.
   - You need to verify category values before comparing groups.

8. Use get_value_distribution when:
   - The user asks for category counts and percentages.
   - The user asks which categories are most or least common.
   - The user asks for a frequency distribution or proportions.
   - The user requests a distribution after applying filters.
   - The user asks how a categorical variable is distributed
     within a particular subset of the dataset.

   This tool calculates counts and percentages in Python.
   Do not calculate percentages yourself.

9. Use group_and_aggregate for grouping, filtering and aggregations.

10. Use compare_groups when the user asks to compare two groups.
    Do not calculate group differences or percentage differences yourself.

11. For compare_groups, set group_a as the baseline and group_b as
    the group being compared. The tool defines the difference as
    group_b minus group_a.

12. Respect monetary column and currency metadata.
    Never combine different currencies into one monetary result.
    For monetary comparisons across currencies, use group_by with
    the configured currency column.

13. Category filters are case-insensitive.
    Use category values shown in the dataset context or discovered
    through get_categorical_values.

14. If a category is invalid, use the error message and its suggestions
    to correct the category before retrying.

15. Distinguish configured semantic roles from inferred
    analytical role suggestions.

16. If a question requires multiple investigations, you may call
    multiple tools or make further tool calls based on earlier results.

17. Reuse existing tool results when possible.
    Do not repeat a calculation if the required result is already
    available in the conversation history.

18. Explain findings clearly and use actual tool results as evidence.

19. Do not confuse average transaction value with the likelihood
    of an event occurring. A higher average transaction amount among
    cancelled orders does not establish that expensive orders
    are more likely to be cancelled.

20. Clearly distinguish observed patterns from possible explanations.
    Do not present hypotheses as established causes.

21. Do not claim causation based only on correlation or descriptive
    statistics.

DATASET CONTEXT:

{context_json}
"""


# --------------------------------------------------
# 8. REUSABLE AGENT FUNCTION
# --------------------------------------------------

def _json_safe(value):
    """
    Convert a Python value into JSON-compatible data.

    This supports reliable storage of tool outputs,
    including NumPy and Pandas values.
    """
    return json.loads(
        json.dumps(value, default=str)
    )


def run_agent(
    question,
    max_tool_rounds=5,
    verbose=True,
):
    """
    Run DataSense AI for one natural-language question.

    Returns:
        question:
            Original user question.

        final_answer:
            Final natural-language response.

        selected_tools:
            All selected tools and their arguments.

        tool_results:
            Structured results returned by Python tools,
            including successful and failed executions.

        tool_call_count:
            Total number of tool calls.

        rounds_used:
            Number of LLM rounds used.

        execution_errors:
            List of tool execution errors.

        completed:
            Whether the agent produced a final answer.
    """

    # --------------------------------------------------
    # 1. INPUT VALIDATION
    # --------------------------------------------------

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if (
        isinstance(max_tool_rounds, bool)
        or not isinstance(max_tool_rounds, int)
        or max_tool_rounds < 1
    ):
        raise ValueError(
            "max_tool_rounds must be a positive integer."
        )

    # --------------------------------------------------
    # 2. INITIALIZE CONVERSATION
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    selected_tools = []
    tool_results = []

    tool_call_count = 0
    execution_errors = []

    final_answer = None
    rounds_used = 0
    completed = False

    # --------------------------------------------------
    # 3. AGENT EXECUTION LOOP
    # --------------------------------------------------

    for round_number in range(max_tool_rounds):

        rounds_used = round_number + 1

        if verbose:
            print(f"\nAgent round: {rounds_used}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1500,
        )

        assistant_message = response.choices[0].message

        # --------------------------------------------------
        # 4. CHECK WHETHER THE AGENT HAS FINISHED
        # --------------------------------------------------

        if not assistant_message.tool_calls:

            final_answer = assistant_message.content or ""

            messages.append({
                "role": "assistant",
                "content": final_answer,
            })

            completed = True

            if verbose:
                print("\nDataSense AI:")
                print(final_answer)

            break

        # --------------------------------------------------
        # 5. PRESERVE ASSISTANT TOOL-CALL REQUEST
        # --------------------------------------------------

        assistant_tool_calls = []

        for tool_call in assistant_message.tool_calls:

            assistant_tool_calls.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            })

        messages.append({
            "role": "assistant",
            "tool_calls": assistant_tool_calls,
        })

        # --------------------------------------------------
        # 6. EXECUTE REQUESTED PYTHON TOOLS
        # --------------------------------------------------

        for tool_call in assistant_message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            tool_call_count += 1

            selected_tools.append({
                "tool": function_name,
                "arguments": _json_safe(arguments),
                "round": rounds_used,
            })

            if verbose:
                print("\nSelected tool:", function_name)
                print("Arguments:", arguments)

            try:

                result = execute_tool(
                    function_name,
                    arguments,
                )

                # Convert the result into JSON-safe data.
                safe_result = _json_safe(result)

                # Retain the actual structured Python output.
                tool_results.append({
                    "tool": function_name,
                    "arguments": _json_safe(arguments),
                    "round": rounds_used,
                    "success": True,
                    "result": safe_result,
                    "error": None,
                })

                if verbose:
                    print("Python result:", safe_result)

                tool_result = {
                    "success": True,
                    "result": safe_result,
                }

            except Exception as error:

                error_message = (
                    f"{function_name}: "
                    f"{type(error).__name__}: {error}"
                )

                execution_errors.append(error_message)

                # Retain failed executions too.
                tool_results.append({
                    "tool": function_name,
                    "arguments": _json_safe(arguments),
                    "round": rounds_used,
                    "success": False,
                    "result": None,
                    "error": error_message,
                })

                if verbose:
                    print("Tool error:", error_message)

                tool_result = {
                    "success": False,
                    "error": str(error),
                }

            # Send the tool output back to the LLM.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    tool_result,
                    default=str,
                ),
            })

    # --------------------------------------------------
    # 7. EXECUTION SUMMARY
    # --------------------------------------------------

    if not completed and verbose:
        print(
            "\nAgent stopped: maximum tool-call rounds reached."
        )

    # --------------------------------------------------
    # 8. RETURN STRUCTURED EXECUTION RESULTS
    # --------------------------------------------------

    return {
        "question": question,
        "final_answer": final_answer,
        "selected_tools": selected_tools,
        "tool_results": tool_results,
        "tool_call_count": tool_call_count,
        "rounds_used": rounds_used,
        "execution_errors": execution_errors,
        "completed": completed,
    }
