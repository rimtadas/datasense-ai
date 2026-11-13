
"""
DataSense AI - Evaluation Exam Runner

Runs evaluation questions through the reusable agent
and saves structured execution results to CSV.

Supports:
- Full exam execution
- Selective question execution
- Incremental saving
- Preserving previous results during targeted reruns
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from agent.runner import run_agent
from evaluation.exam_questions import (
    EXAM_QUESTIONS,
    validate_exam_questions,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

OUTPUT_FILE = (
    Path(__file__).parent / "evaluation_results.csv"
)

MAX_TOOL_ROUNDS = 5


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def json_string(value):
    """Convert a Python object into a JSON string."""

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def build_result_record(question_data, agent_result):
    """
    Build one evaluation result record from agent execution.
    """

    selected_tools = agent_result["selected_tools"]
    tool_results = agent_result.get("tool_results", [])

    tool_names = [
        item["tool"]
        for item in selected_tools
    ]

    successful_tool_calls = sum(
        1 for item in tool_results
        if item["success"]
    )

    failed_tool_calls = sum(
        1 for item in tool_results
        if not item["success"]
    )

    if not agent_result["completed"]:
        status = "incomplete"

    elif agent_result["execution_errors"]:
        status = "completed_with_errors"

    else:
        status = "completed"

    return {
        "question_id": question_data["id"],
        "category": question_data["category"],
        "difficulty": question_data["difficulty"],
        "question": question_data["question"],

        "expected_tools": json_string(
            question_data["expected_tools"]
        ),

        "selected_tools": json_string(
            tool_names
        ),

        "tool_details": json_string(
            selected_tools
        ),

        "tool_results": json_string(
            tool_results
        ),

        "tool_call_count": agent_result["tool_call_count"],

        "successful_tool_calls": successful_tool_calls,

        "failed_tool_calls": failed_tool_calls,

        "rounds_used": agent_result["rounds_used"],

        # Agent performance metrics
        "execution_time_seconds": agent_result.get(
            "execution_time_seconds"
        ),

        "api_attempts": agent_result.get(
            "api_attempts"
        ),

        "prompt_tokens": agent_result.get(
            "prompt_tokens"
        ),

        "completion_tokens": agent_result.get(
            "completion_tokens"
        ),

        "total_tokens": (
            agent_result["prompt_tokens"]
            + agent_result["completion_tokens"]
            if (
                agent_result.get("prompt_tokens") is not None
                and agent_result.get("completion_tokens") is not None
            )
            else None
        ),

        "final_answer": agent_result["final_answer"],

        "execution_errors": json_string(
            agent_result["execution_errors"]
        ),

        "status": status,
    }


def build_failed_result(question_data, error):
    """Build a result record when the question execution fails."""

    error_message = (
        f"{type(error).__name__}: {error}"
    )

    return {
        "question_id": question_data["id"],
        "category": question_data["category"],
        "difficulty": question_data["difficulty"],
        "question": question_data["question"],

        "expected_tools": json_string(
            question_data["expected_tools"]
        ),

        "selected_tools": "[]",
        "tool_details": "[]",
        "tool_results": "[]",

        "tool_call_count": 0,
        "successful_tool_calls": 0,
        "failed_tool_calls": 0,

        "rounds_used": 0,

        "execution_time_seconds": None,
        "api_attempts": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "final_answer": "",

        "execution_errors": json_string([
            error_message
        ]),

        "status": "failed",
    }


# --------------------------------------------------
# 3. SAVE RESULTS WITHOUT LOSING PREVIOUS QUESTIONS
# --------------------------------------------------

def save_results_incrementally(new_results):
    """
    Insert or update results by question_id.

    Existing results for questions not included in the
    current execution are preserved.
    """

    if OUTPUT_FILE.exists():

        existing_df = pd.read_csv(
            OUTPUT_FILE,
            encoding="utf-8-sig",
        )

        # Support older baseline files that lack new columns.
        combined_df = pd.concat(
            [
                existing_df,
                pd.DataFrame(new_results),
            ],
            ignore_index=True,
            sort=False,
        )

        # Keep the newest result for each question.
        combined_df = combined_df.drop_duplicates(
            subset=["question_id"],
            keep="last",
        )

    else:

        combined_df = pd.DataFrame(new_results)

    # Preserve the original exam question ordering.
    question_order = {
        question["id"]: index
        for index, question in enumerate(EXAM_QUESTIONS)
    }

    combined_df["_order"] = combined_df[
        "question_id"
    ].map(question_order)

    combined_df = combined_df.sort_values(
        "_order",
        na_position="last",
    ).drop(columns=["_order"])

    # Write to a temporary file before replacing the baseline.
    temporary_file = OUTPUT_FILE.with_suffix(".tmp.csv")

    combined_df.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_file.replace(OUTPUT_FILE)


# --------------------------------------------------
# 4. RUN THE EXAM
# --------------------------------------------------

def run_exam(question_ids=None):

    validate_exam_questions()

    # Select questions if IDs were supplied.
    if question_ids:

        requested_ids = set(question_ids)

        available_ids = {
            question["id"]
            for question in EXAM_QUESTIONS
        }

        invalid_ids = requested_ids - available_ids

        if invalid_ids:
            raise ValueError(
                f"Unknown question IDs: {sorted(invalid_ids)}"
            )

        selected_questions = [
            question
            for question in EXAM_QUESTIONS
            if question["id"] in requested_ids
        ]

    else:

        selected_questions = EXAM_QUESTIONS

    total_questions = len(selected_questions)

    print("\n" + "=" * 60)
    print("DATASENSE AI - EVALUATION EXAM")
    print("=" * 60)

    print("Questions selected:", total_questions)
    print("Question IDs:", [
        question["id"]
        for question in selected_questions
    ])
    print("Output file:", OUTPUT_FILE)

    results = []

    # --------------------------------------------------
    # EXECUTE QUESTIONS
    # --------------------------------------------------

    for index, question_data in enumerate(
        selected_questions,
        start=1,
    ):

        question_id = question_data["id"]
        question_text = question_data["question"]

        print("\n" + "=" * 60)
        print(
            f"QUESTION {index}/{total_questions}: {question_id}"
        )
        print("=" * 60)

        print(question_text)

        try:

            agent_result = run_agent(
                question=question_text,
                max_tool_rounds=MAX_TOOL_ROUNDS,
                verbose=True,
            )

            result = build_result_record(
                question_data,
                agent_result,
            )

        except Exception as error:

            print("\nQuestion execution failed:")
            print(type(error).__name__, str(error))

            result = build_failed_result(
                question_data,
                error,
            )

        results.append(result)

        # Save after every question.
        save_results_incrementally([result])

        print("\nQuestion status:", result["status"])
        print("Tool calls:", result["tool_call_count"])
        print(
            "Successful tool calls:",
            result["successful_tool_calls"],
        )
        print(
            "Failed tool calls:",
            result["failed_tool_calls"],
        )

    # --------------------------------------------------
    # 5. FINAL SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    status_counts = results_df["status"].value_counts()

    completed_count = int(
        status_counts.get("completed", 0)
    )

    completed_with_errors_count = int(
        status_counts.get("completed_with_errors", 0)
    )

    incomplete_count = int(
        status_counts.get("incomplete", 0)
    )

    failed_count = int(
        status_counts.get("failed", 0)
    )

    total_tool_calls = int(
        results_df["tool_call_count"].sum()
    )

    successful_tool_calls = int(
        results_df["successful_tool_calls"].sum()
    )

    failed_tool_calls = int(
        results_df["failed_tool_calls"].sum()
    )

    print("\n" + "=" * 60)
    print("EXAM SUMMARY")
    print("=" * 60)

    print("Questions executed:", total_questions)
    print("Completed:", completed_count)
    print("Completed with errors:", completed_with_errors_count)
    print("Incomplete:", incomplete_count)
    print("Failed:", failed_count)

    print("\nTotal tool calls:", total_tool_calls)
    print("Successful tool calls:", successful_tool_calls)
    print("Failed tool calls:", failed_tool_calls)

    print("\nResults saved to:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# 6. ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run DataSense AI evaluation questions."
    )

    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help=(
            "Optional question IDs to execute. "
            "Example: --question-ids Q13 Q15"
        ),
    )

    args = parser.parse_args()

    run_exam(
        question_ids=args.question_ids
    )
