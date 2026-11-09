
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASELINE_FILE = Path(__file__).parent / "baseline_results.csv"

EXPECTED_QUESTIONS = 15

VALID_STATUSES = {
    "completed",
    "completed_with_errors",
    "incomplete",
    "failed",
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_number(value):
    """Convert a value to a number, treating missing values as unavailable."""
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_structured_results(value):
    """
    Check whether a baseline row contains valid, non-empty
    structured tool results.
    """
    if pd.isna(value) or not str(value).strip():
        return False

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False

    return isinstance(parsed, list) and len(parsed) > 0


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def generate_summary():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}"
        )

    df = pd.read_csv(BASELINE_FILE)

    if df.empty:
        print("The evaluation baseline is empty.")
        return

    required_columns = {
        "question_id",
        "status",
        "tool_call_count",
        "successful_tool_calls",
        "failed_tool_calls",
        "tool_results",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Baseline is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Avoid counting duplicate question IDs more than once.
    df = df.drop_duplicates(
        subset=["question_id"],
        keep="last"
    )

    # --------------------------------------------------
    # 1. QUESTION COVERAGE
    # --------------------------------------------------

    total_questions = len(df)

    executed_questions = df["status"].isin(
        VALID_STATUSES
    ).sum()

    completed_questions = (
        df["status"] == "completed"
    ).sum()

    completed_with_errors = (
        df["status"] == "completed_with_errors"
    ).sum()

    incomplete_questions = (
        df["status"] == "incomplete"
    ).sum()

    failed_questions = (
        df["status"] == "failed"
    ).sum()

    not_executed_questions = (
        total_questions - executed_questions
    )

    coverage_percentage = (
        executed_questions / EXPECTED_QUESTIONS * 100
    )

    # --------------------------------------------------
    # 2. TOOL EXECUTION COUNTS
    # --------------------------------------------------

    tool_call_counts = df["tool_call_count"].apply(
        safe_number
    )

    successful_counts = df["successful_tool_calls"].apply(
        safe_number
    )

    failed_counts = df["failed_tool_calls"].apply(
        safe_number
    )

    total_tool_calls = tool_call_counts.sum(
        min_count=1
    )

    successful_tool_calls = successful_counts.sum(
        min_count=1
    )

    failed_tool_calls = failed_counts.sum(
        min_count=1
    )

    # Number of questions for which tool execution counts exist.
    questions_with_tool_counts = tool_call_counts.notna().sum()

    # --------------------------------------------------
    # 3. STRUCTURED RESULT COVERAGE
    # --------------------------------------------------

    structured_result_available = df["tool_results"].apply(
        has_structured_results
    )

    structured_questions = structured_result_available.sum()

    structured_coverage = (
        structured_questions / EXPECTED_QUESTIONS * 100
    )

    missing_structured_questions = (
        total_questions - structured_questions
    )

    # --------------------------------------------------
    # 4. REPORT
    # --------------------------------------------------

    def display_number(value):
        if pd.isna(value):
            return "Unavailable"
        return f"{value:,.0f}"

    print("\n" + "=" * 55)
    print("DATASENSE AI - EVALUATION BASELINE SUMMARY")
    print("=" * 55)

    print("\nQUESTION COVERAGE")
    print("-" * 55)
    print(f"Expected questions:          {EXPECTED_QUESTIONS}")
    print(f"Questions in baseline:       {total_questions}")
    print(f"Executed questions:          {executed_questions}")
    print(f"Not executed / unknown:      {not_executed_questions}")
    print(f"Execution coverage:          {coverage_percentage:.1f}%")

    print("\nEXECUTION STATUS")
    print("-" * 55)
    print(f"Completed:                   {completed_questions}")
    print(f"Completed with errors:       {completed_with_errors}")
    print(f"Incomplete:                  {incomplete_questions}")
    print(f"Failed:                      {failed_questions}")

    print("\nTOOL EXECUTION")
    print("-" * 55)
    print(f"Questions with tool counts:  {questions_with_tool_counts}")
    print(f"Total tool calls:            {display_number(total_tool_calls)}")
    print(f"Successful tool calls:       {display_number(successful_tool_calls)}")
    print(f"Failed tool calls:           {display_number(failed_tool_calls)}")

    print("\nSTRUCTURED RESULT COVERAGE")
    print("-" * 55)
    print(f"Questions with results:      {structured_questions}")
    print(f"Questions without results:   {missing_structured_questions}")
    print(f"Structured result coverage:  {structured_coverage:.1f}%")

    print("\nQUESTION-LEVEL DETAILS")
    print("-" * 55)

    detail_columns = [
        "question_id",
        "status",
        "tool_call_count",
        "successful_tool_calls",
        "failed_tool_calls",
    ]

    details = df[detail_columns].copy()

    details["structured_results"] = (
        structured_result_available.map({
            True: "Available",
            False: "Unavailable",
        })
    )

    print(details.to_string(index=False))

    print("\n" + "=" * 55)
    print("SUMMARY COMPLETE")
    print("=" * 55)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    generate_summary()
