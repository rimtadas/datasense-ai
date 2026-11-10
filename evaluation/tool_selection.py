
import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.exam_questions import EXAM_QUESTIONS


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASELINE_FILE = Path(__file__).parent / "baseline_results.csv"

# Each list represents a required analytical step.
# A step can be satisfied by any one of its accepted tools.
#
# Questions not listed here use their exam_questions.py
# expected_tools as their required sequence.

ACCEPTABLE_TOOL_STAGES = {
    "Q05": [
        {"get_categorical_values", "get_value_distribution"},
    ],
    "Q09": [
        {"group_and_aggregate", "get_value_distribution"},
    ],
    "Q10": [
        {"group_and_aggregate", "get_value_distribution"},
    ],
    "Q14": [
        {"get_categorical_values", "get_value_distribution"},
        {"compare_groups"},
    ],
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def parse_json_list(value):
    """Parse a JSON list safely."""

    if pd.isna(value) or not str(value).strip():
        return None

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    return parsed if isinstance(parsed, list) else None


def get_selected_tool_names(value):
    """Extract tool names from the recorded selected_tools field."""

    parsed = parse_json_list(value)

    if parsed is None:
        return None

    names = []

    for item in parsed:

        if isinstance(item, str):
            names.append(item)

        elif isinstance(item, dict) and isinstance(
            item.get("tool"), str
        ):
            names.append(item["tool"])

        else:
            return None

    return names


def get_execution_counts(value):
    """Count successful and failed structured tool executions."""

    results = parse_json_list(value)

    if results is None:
        return None

    successful = sum(
        1 for item in results
        if isinstance(item, dict) and item.get("success") is True
    )

    failed = sum(
        1 for item in results
        if isinstance(item, dict) and item.get("success") is False
    )

    return successful, failed


def get_expected_tools(question):
    """Return the original expected tool sequence."""

    return question["expected_tools"]


def get_acceptable_stages(question):
    """Return acceptable tools for each required analytical step."""

    question_id = question["id"]

    if question_id in ACCEPTABLE_TOOL_STAGES:
        return ACCEPTABLE_TOOL_STAGES[question_id]

    return [
        {tool}
        for tool in get_expected_tools(question)
    ]


def evaluate_tool_selection(selected_tools, stages):
    """
    Match selected calls to required analytical stages in order.

    Returns stage coverage and unexpected calls.
    """

    matched_stages = []
    unexpected_calls = []

    next_stage = 0

    for tool in selected_tools:

        if next_stage < len(stages) and tool in stages[next_stage]:

            matched_stages.append(tool)
            next_stage += 1

        else:

            unexpected_calls.append(tool)

    missing_stages = len(stages) - len(matched_stages)

    return {
        "matched_stages": matched_stages,
        "missing_stages": missing_stages,
        "unexpected_calls": unexpected_calls,
        "functional_pass": (
            missing_stages == 0
            and len(unexpected_calls) == 0
        ),
        "stage_coverage": (
            len(matched_stages) / len(stages) * 100
            if stages else 100.0
        ),
    }


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(
    question_ids=None,
):

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}"
        )

    df = pd.read_csv(BASELINE_FILE)

    required_columns = {
        "question_id",
        "selected_tools",
        "tool_results",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Baseline is missing columns: {sorted(missing_columns)}"
        )

    # Keep the latest result for each question.
    df = df.drop_duplicates(
        subset=["question_id"],
        keep="last",
    )

    baseline = {
        str(row["question_id"]): row
        for _, row in df.iterrows()
    }

    questions = EXAM_QUESTIONS

    if question_ids:
        unknown_ids = set(question_ids) - {
            question["id"] for question in questions
        }

        if unknown_ids:
            raise ValueError(
                f"Unknown question IDs: {sorted(unknown_ids)}"
            )

        questions = [
            question for question in questions
            if question["id"] in question_ids
        ]

    reports = []

    for question in questions:

        question_id = question["id"]
        expected_tools = get_expected_tools(question)
        stages = get_acceptable_stages(question)

        row = baseline.get(question_id)

        if row is None:

            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
            })

            continue

        selected_tools = get_selected_tool_names(
            row["selected_tools"]
        )

        execution_counts = get_execution_counts(
            row["tool_results"]
        )

        if selected_tools is None:

            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
            })

            continue

        selection = evaluate_tool_selection(
            selected_tools,
            stages,
        )

        strict_match = selected_tools == expected_tools

        if execution_counts is None:
            successful_calls = None
            failed_calls = None
        else:
            successful_calls, failed_calls = execution_counts

        reports.append({
            "question_id": question_id,
            "status": "EVALUATED",
            "expected_tools": expected_tools,
            "selected_tools": selected_tools,
            "strict_match": strict_match,
            "functional_pass": selection["functional_pass"],
            "stage_coverage": selection["stage_coverage"],
            "missing_stages": selection["missing_stages"],
            "unexpected_calls": selection["unexpected_calls"],
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
        })

    return reports


# --------------------------------------------------
# REPORT
# --------------------------------------------------

def print_report(reports):

    evaluated = [
        report for report in reports
        if report["status"] == "EVALUATED"
    ]

    not_evaluated = [
        report for report in reports
        if report["status"] == "NOT EVALUATED"
    ]

    if not evaluated:
        print("No questions could be evaluated.")
        return

    strict_passes = sum(
        report["strict_match"]
        for report in evaluated
    )

    functional_passes = sum(
        report["functional_pass"]
        for report in evaluated
    )

    total_stages = sum(
        len(get_acceptable_stages(question))
        for question in EXAM_QUESTIONS
        if question["id"] in {
            report["question_id"] for report in evaluated
        }
    )

    covered_stages = sum(
        report["stage_coverage"] / 100 *
        len(get_acceptable_stages(
            next(
                q for q in EXAM_QUESTIONS
                if q["id"] == report["question_id"]
            )
        ))
        for report in evaluated
    )

    successful_calls = sum(
        report["successful_calls"] or 0
        for report in evaluated
    )

    failed_calls = sum(
        report["failed_calls"] or 0
        for report in evaluated
    )

    calls_with_execution_data = sum(
        report["successful_calls"] is not None
        for report in evaluated
    )

    unexpected_calls = sum(
        len(report["unexpected_calls"])
        for report in evaluated
    )

    print("\n" + "=" * 65)
    print("DATASENSE AI - TOOL SELECTION EVALUATION")
    print("=" * 65)

    print("\nOVERALL RESULTS")
    print("-" * 65)

    print(f"Questions evaluated:          {len(evaluated)}")
    print(f"Questions not evaluated:      {len(not_evaluated)}")

    print(
        f"Strict tool adherence:        "
        f"{strict_passes}/{len(evaluated)} "
        f"({strict_passes / len(evaluated) * 100:.2f}%)"
    )

    print(
        f"Functional tool coverage:     "
        f"{functional_passes}/{len(evaluated)} "
        f"({functional_passes / len(evaluated) * 100:.2f}%)"
    )

    print(
        f"Analytical stage coverage:    "
        f"{covered_stages:.0f}/{total_stages} "
        f"({covered_stages / total_stages * 100:.2f}%)"
    )

    print(f"Unexpected tool calls:        {unexpected_calls}")

    print("\nTOOL EXECUTION")
    print("-" * 65)

    if calls_with_execution_data == len(evaluated):

        print(f"Successful calls:             {successful_calls}")
        print(f"Failed calls:                 {failed_calls}")

    else:

        print(
            "Execution data:               "
            "Incomplete for some questions"
        )

        print(f"Successful calls recorded:    {successful_calls}")
        print(f"Failed calls recorded:        {failed_calls}")

    print("\nQUESTION-LEVEL RESULTS")
    print("-" * 65)

    for report in reports:

        question_id = report["question_id"]

        if report["status"] == "NOT EVALUATED":
            print(f"{question_id}: NOT EVALUATED")
            continue

        strict_status = (
            "PASS" if report["strict_match"] else "DEVIATION"
        )

        functional_status = (
            "PASS" if report["functional_pass"] else "FAIL"
        )

        print(f"\n{question_id}")
        print(f"  Expected:       {report['expected_tools']}")
        print(f"  Selected:       {report['selected_tools']}")
        print(f"  Strict:         {strict_status}")
        print(f"  Functional:     {functional_status}")
        print(
            f"  Stage coverage: "
            f"{report['stage_coverage']:.1f}%"
        )

        if report["unexpected_calls"]:
            print(
                f"  Unexpected:     "
                f"{report['unexpected_calls']}"
            )

        if report["missing_stages"]:
            print(
                f"  Missing stages: "
                f"{report['missing_stages']}"
            )

        if report["failed_calls"] is not None:
            print(
                f"  Execution:      "
                f"{report['successful_calls']} successful, "
                f"{report['failed_calls']} failed"
            )

    print("\n" + "=" * 65)
    print("TOOL SELECTION EVALUATION COMPLETE")
    print("=" * 65)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate DataSense AI tool selection."
    )

    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Optional question IDs to evaluate.",
    )

    args = parser.parse_args()

    results = evaluate(
        question_ids=args.question_ids
    )

    print_report(results)
