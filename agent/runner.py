
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

from dataset_context.dataset_configs import (
    ECOMMERCE_CONFIG,
    ECOMMERCE_DATETIME_COLUMNS,
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

REPO_ID = os.getenv(
    "DATASENSE_DATASET_REPO_ID",
    "davidev07/synthetic-ecommerce-dataset",
)

DATASET_FILE = os.getenv(
    "DATASENSE_DATASET_FILE",
    "transactions.csv",
)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename=DATASET_FILE,
)

# Parse configured datetime columns.
DATETIME_COLUMNS = ECOMMERCE_DATETIME_COLUMNS

for column in DATETIME_COLUMNS:
    if column not in df.columns:
        raise ValueError(
            f"Configured datetime column '{column}' "
            "does not exist in the dataset."
        )

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
    )

# Dataset-specific metadata is maintained separately.
dataset_config = ECOMMERCE_CONFIG

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
                "Optional filters applied before calculating "
                "the distribution. Provide column names and values "
                "that define the subset of records to analyse."
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
                            "the distribution. Provide column names and values "
                            "that define the subset of records to analyse."
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
                "Use this for questions comparing any two groups, categories, "
                "or conditions. It calculates both group values, the "
                "absolute difference (group_b minus group_a), and percentage "
                "difference in Python. Use group_by to compare separately "
                "across another dimension. Monetary metrics must be grouped "
                "by the configured currency column or filtered to one currency."
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

def validate_allowed_column(column, active_excluded_columns=None):

    if active_excluded_columns is None:
        active_excluded_columns = excluded_columns

    if column is None:
        return

    if column in active_excluded_columns:
        raise ValueError(
            f"Column '{column}' is excluded from agent access."
        )


def validate_filter_columns(
    filters,
    active_excluded_columns=None,
):

    if not filters:
        return

    if not isinstance(filters, dict):
        raise ValueError("Filters must be a dictionary.")

    for column in filters:
        validate_allowed_column(
            column,
            active_excluded_columns,
        )


# --------------------------------------------------
# 6. MAP TOOL NAMES TO PYTHON FUNCTIONS
# --------------------------------------------------

def execute_tool(
    function_name,
    arguments,
    active_df=None,
    active_dataset_config=None,
    active_excluded_columns=None,
):
    """
    Execute an analytical tool against the supplied dataset.

    If no dataset is supplied, retain the existing
    module-level defaults for backward compatibility.
    """

    if active_df is None:
        active_df = df

    if active_dataset_config is None:
        active_dataset_config = dataset_config

    if active_excluded_columns is None:
        active_excluded_columns = excluded_columns

    if function_name == "get_dataset_overview":

        return get_dataset_overview(
            active_df,
            dataset_config=active_dataset_config,
        )

    elif function_name == "get_column_summary":

        column = arguments["column"]

        validate_allowed_column(
            column,
            active_excluded_columns,
        )

        return get_column_summary(
            active_df,
            column=column,
            dataset_config=active_dataset_config,
        )

    elif function_name == "get_categorical_values":

        column = arguments["column"]

        validate_allowed_column(
            column,
            active_excluded_columns,
        )

        return get_categorical_values(
            active_df,
            column=column,
            limit=arguments.get("limit", 30),
        )

    elif function_name == "get_value_distribution":

        column = arguments["column"]
        filters = arguments.get("filters")

        validate_allowed_column(
            column,
            active_excluded_columns,
        )

        validate_filter_columns(
            filters,
            active_excluded_columns,
        )

        return get_value_distribution(
            df=active_df,
            column=column,
            filters=filters,
            limit=arguments.get("limit"),
            dataset_config=active_dataset_config,
        )

    elif function_name == "group_and_aggregate":

        group_by = arguments.get("group_by")
        metric = arguments.get("metric")
        filters = arguments.get("filters")

        validate_allowed_column(
            group_by,
            active_excluded_columns,
        )

        validate_allowed_column(
            metric,
            active_excluded_columns,
        )

        validate_filter_columns(
            filters,
            active_excluded_columns,
        )

        return group_and_aggregate(
            active_df,
            group_by=group_by,
            metric=metric,
            aggregation=arguments["aggregation"],
            filters=filters,
            sort_by=arguments.get("sort_by", "result"),
            ascending=arguments.get("ascending", False),
            limit=arguments.get("limit"),
            dataset_config=active_dataset_config,
        )

    elif function_name == "compare_groups":

        comparison_column = arguments["comparison_column"]
        group_by = arguments.get("group_by")
        metric = arguments.get("metric")
        filters = arguments.get("filters")

        validate_allowed_column(
            comparison_column,
            active_excluded_columns,
        )

        validate_allowed_column(
            group_by,
            active_excluded_columns,
        )

        validate_allowed_column(
            metric,
            active_excluded_columns,
        )

        validate_filter_columns(
            filters,
            active_excluded_columns,
        )

        return compare_groups(
            df=active_df,
            comparison_column=comparison_column,
            group_a=arguments["group_a"],
            group_b=arguments["group_b"],
            metric=metric,
            aggregation=arguments["aggregation"],
            group_by=group_by,
            filters=filters,
            dataset_config=active_dataset_config,
        )

    else:

        raise ValueError(
            f"Unknown tool: {function_name}"
        )

# --------------------------------------------------
# 7. DEFINE SYSTEM PROMPT
# --------------------------------------------------


def build_system_prompt(active_context_json):

    return f"""
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

19. Clearly distinguish observed patterns from possible explanations.
    Do not present hypotheses as established causes.

20. Do not claim causation based only on correlation or descriptive
    statistics.

DATASET CONTEXT:

{active_context_json}
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
    df=None,
    dataset_config=None,
    dataset_name=None,
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
    # 1A. INITIALIZE ACTIVE DATASET
    # --------------------------------------------------

    using_default_dataset = df is None

    if using_default_dataset:
        active_df = globals()["df"]
    else:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a Pandas DataFrame."
            )

        active_df = df

    if active_df.empty:
        raise ValueError("The dataset is empty.")

    if dataset_config is None:
        if using_default_dataset:
            active_dataset_config = globals()["dataset_config"]
        else:
            active_dataset_config = {}
    else:
        if not isinstance(dataset_config, dict):
            raise TypeError(
                "dataset_config must be a dictionary."
            )

        active_dataset_config = dataset_config

    active_dataset_name = (
        dataset_name
        or (
            DATASET_FILE
            if using_default_dataset
            else "custom dataset"
        )
    )

    # Work on a copy so the caller's DataFrame is not modified.
    active_df = active_df.copy()

    # Parse datetime columns configured through semantic metadata.
    semantic_types = active_dataset_config.get(
        "semantic_types", {}
    )

    datetime_columns = [
        column
        for column, semantic_type in semantic_types.items()
        if semantic_type == "datetime"
    ]

    for column in datetime_columns:

        if column not in active_df.columns:
            raise ValueError(
                f"Configured datetime column '{column}' "
                "does not exist in the dataset."
            )

        active_df[column] = pd.to_datetime(
            active_df[column],
            errors="coerce",
        )

    # Build context specifically for this dataset.
    active_dataset_context = build_dataset_context(
        df=active_df,
        dataset_name=active_dataset_name,
        dataset_config=active_dataset_config,
    )

    active_context_json = json.dumps(
        active_dataset_context,
        indent=2,
        default=str,
    )

    active_system_prompt = build_system_prompt(
        active_context_json
    )

    active_excluded_columns = set(
        active_dataset_config.get(
            "exclude_from_context", []
        )
    )

    # --------------------------------------------------
    # 2. INITIALIZE CONVERSATION
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": active_system_prompt,
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
                    active_df=active_df,
                    active_dataset_config=active_dataset_config,
                    active_excluded_columns=active_excluded_columns,
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
