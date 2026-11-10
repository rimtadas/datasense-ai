
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from data_connector.huggingface_loader import load_huggingface_dataset
from evaluation.expected_results import (
    calculate_q01_expected,
    calculate_q02_expected,
    calculate_q03_expected,
    calculate_q04_expected,
    calculate_q05_expected,
    calculate_q06_expected,
    calculate_q07_expected,
    calculate_q08_expected,
    calculate_q09_expected,
    calculate_q10_expected,
    calculate_q11_expected,
    calculate_q12_expected,
    calculate_q13_expected,
    calculate_q14_expected,
    calculate_q15_expected,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

REPO_ID = "davidev07/synthetic-ecommerce-dataset"
DATASET_FILE = "transactions.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = PROJECT_ROOT / "evaluation" / "baseline_results.csv"

SUPPORTED_QUESTIONS = [
    f"Q{i:02d}" for i in range(1, 16)
]

DATASET_CONFIG = {
    "exclude_from_context": ["shipping_address"],
}

REL_TOL = 1e-9
ABS_TOL = 1e-9


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset():

    print("Loading dataset...")

    df = load_huggingface_dataset(
        repo_id=REPO_ID,
        filename=DATASET_FILE,
    )

    df["purchase_date"] = pd.to_datetime(
        df["purchase_date"],
        errors="coerce",
    )

    print("Dataset loaded:", len(df), "rows")

    return df


# --------------------------------------------------
# 3. LOAD BASELINE
# --------------------------------------------------

def load_baseline():

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}"
        )

    baseline = pd.read_csv(BASELINE_FILE)

    required_columns = {
        "question_id",
        "status",
        "tool_results",
    }

    missing_columns = required_columns - set(baseline.columns)

    if missing_columns:
        raise ValueError(
            f"Baseline is missing columns: {sorted(missing_columns)}"
        )

    return baseline


# --------------------------------------------------
# 4. PARSE STRUCTURED TOOL RESULTS
# --------------------------------------------------

def parse_tool_results(value):

    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    return parsed


# --------------------------------------------------
# 5. VALUE COMPARISON
# --------------------------------------------------

def values_match(expected, actual):

    if expected is None or actual is None:
        return expected is None and actual is None

    try:
        expected_number = float(expected)
        actual_number = float(actual)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(expected_number):
        return False

    if not math.isfinite(actual_number):
        return False

    return math.isclose(
        expected_number,
        actual_number,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )


def structured_values_match(expected, actual):

    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual

    if expected is None or actual is None:
        return expected is None and actual is None

    if isinstance(expected, (int, float)):
        return values_match(expected, actual)

    if isinstance(expected, dict):

        if not isinstance(actual, dict):
            return False

        if set(expected.keys()) != set(actual.keys()):
            return False

        return all(
            structured_values_match(
                expected[key],
                actual[key],
            )
            for key in expected
        )

    if isinstance(expected, list):

        if not isinstance(actual, list):
            return False

        if len(expected) != len(actual):
            return False

        return all(
            structured_values_match(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual)
        )

    return expected == actual


# --------------------------------------------------
# 6. CHECK BUILDER
# --------------------------------------------------

def build_checks(comparisons):

    checks = []

    for name, expected, actual in comparisons:

        checks.append({
            "metric": name,
            "expected": expected,
            "actual": actual,
            "passed": structured_values_match(
                expected,
                actual,
            ),
        })

    return checks


def build_result(question_id, checks):

    passed_count = sum(
        check["passed"] for check in checks
    )

    return {
        "question_id": question_id,
        "status": (
            "PASS"
            if checks and passed_count == len(checks)
            else "FAIL"
        ),
        "passed_checks": passed_count,
        "total_checks": len(checks),
        "checks": checks,
    }


# --------------------------------------------------
# 7. FIND TOOL EXECUTION
# --------------------------------------------------

def get_tool_result(
    tool_results,
    expected_tool,
    expected_arguments,
):

    matching_calls = [
        item
        for item in tool_results
        if isinstance(item, dict)
        and item.get("tool") == expected_tool
    ]

    if not matching_calls:
        return None, "Expected tool execution was not found."

    if len(matching_calls) != 1:
        return None, (
            f"Expected exactly one {expected_tool} execution, "
            f"but found {len(matching_calls)}."
        )

    tool_call = matching_calls[0]

    if tool_call.get("success") is not True:
        return None, f"The {expected_tool} tool execution failed."

    arguments = tool_call.get("arguments", {})

    if arguments != expected_arguments:
        return None, (
            f"Incorrect tool arguments. "
            f"Expected: {expected_arguments}. "
            f"Actual: {arguments}."
        )

    result = tool_call.get("result")

    if not isinstance(result, dict):
        return None, "Tool result is missing or invalid."

    return result, None


# --------------------------------------------------
# 8. RESULT NORMALISATION
# --------------------------------------------------

def extract_category_counts(category_values):

    if not isinstance(category_values, list):
        return None

    counts = {}

    for item in category_values:

        if not isinstance(item, dict):
            return None

        if "value" not in item or "count" not in item:
            return None

        category = str(item["value"])

        if category in counts:
            return None

        counts[category] = item["count"]

    return counts


def extract_category_distribution(category_values):

    if not isinstance(category_values, list):
        return None

    distribution = {}

    for item in category_values:

        if not isinstance(item, dict):
            return None

        if "value" not in item:
            return None

        if "count" not in item or "percentage" not in item:
            return None

        category = str(item["value"])

        if category in distribution:
            return None

        distribution[category] = {
            "count": item["count"],
            "percentage": item["percentage"],
        }

    return distribution


def extract_currency_results(results):

    if not isinstance(results, list):
        return None

    currency_results = {}

    for item in results:

        if not isinstance(item, dict):
            return None

        currency = item.get("currency")

        if not currency:
            return None

        if currency in currency_results:
            return None

        currency_results[currency] = item

    return currency_results


# --------------------------------------------------
# 9. EVALUATE Q01-Q12 AND Q15
# --------------------------------------------------

def evaluate_question(df, question_id, baseline_row):

    tool_results = parse_tool_results(
        baseline_row["tool_results"]
    )

    if tool_results is None:
        return {
            "question_id": question_id,
            "status": "NOT EVALUATED",
            "reason": "Structured tool results are missing or invalid.",
            "checks": [],
        }

    # Q01: Dataset overview
    if question_id == "Q01":

        expected = calculate_q01_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_dataset_overview",
            {},
        )

        if error:
            return failed_result(question_id, error)

        comparisons = [
            ("Row count", expected["row_count"], result.get("row_count")),
            ("Column count", expected["column_count"], result.get("column_count")),
            ("Duplicate row count", expected["duplicate_rows"], result.get("duplicate_rows")),
        ]

    # Q02: Available and excluded columns
    elif question_id == "Q02":

        expected = calculate_q02_expected(
            df,
            DATASET_CONFIG,
        )

        result, error = get_tool_result(
            tool_results,
            "get_dataset_overview",
            {},
        )

        if error:
            return failed_result(question_id, error)

        actual_excluded = result.get("excluded_columns")

        comparisons = [
            ("Available columns", expected["available_columns"], result.get("columns")),
            ("Excluded columns", expected["excluded_columns"], actual_excluded),
            ("Available column count", expected["available_column_count"], result.get("context_column_count")),
            (
                "Excluded column count",
                expected["excluded_column_count"],
                len(actual_excluded) if isinstance(actual_excluded, list) else None,
            ),
        ]

    # Q03: Date range
    elif question_id == "Q03":

        expected = calculate_q03_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_column_summary",
            {"column": "purchase_date"},
        )

        if error:
            return failed_result(question_id, error)

        expected_missing = len(df) - expected["valid_date_count"]

        comparisons = [
            ("Column", expected["column"], result.get("column")),
            (
                "Valid date count",
                expected["valid_date_count"],
                result.get("row_count", 0) - result.get("missing_count", 0),
            ),
            ("Missing date count", expected_missing, result.get("missing_count")),
            ("Earliest date", expected["min_date"], result.get("min_date")),
            ("Latest date", expected["max_date"], result.get("max_date")),
        ]

    # Q04: Numeric column summary
    elif question_id == "Q04":

        expected = calculate_q04_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_column_summary",
            {"column": "quantity"},
        )

        if error:
            return failed_result(question_id, error)

        comparisons = [
            ("Column", expected["column"], result.get("column")),
            ("Minimum", expected["min"], result.get("min")),
            ("Maximum", expected["max"], result.get("max")),
            ("Mean", expected["mean"], result.get("mean")),
            ("Median", expected["median"], result.get("median")),
        ]

    # Q05: Order status distribution
    elif question_id == "Q05":

        expected = calculate_q05_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_value_distribution",
            {"column": "order_status"},
        )

        if error:
            return failed_result(question_id, error)

        actual_counts = extract_category_counts(
            result.get("category_values")
        )

        comparisons = [
            ("Unique category count", expected["unique_count"], result.get("unique_count")),
            ("Rows after filtering", len(df), result.get("rows_after_filtering")),
            ("Category counts", expected["category_counts"], actual_counts),
        ]

    # Q06: Payment method categories
    elif question_id == "Q06":

        expected = calculate_q06_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_categorical_values",
            {"column": "payment_method"},
        )

        if error:
            return failed_result(question_id, error)

        actual_counts = extract_category_counts(
            result.get("category_values")
        )

        comparisons = [
            ("Unique category count", expected["unique_count"], result.get("unique_count")),
            ("Category counts", expected["category_counts"], actual_counts),
        ]

    # Q07: Payment method distribution
    elif question_id == "Q07":

        expected = calculate_q07_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_value_distribution",
            {"column": "payment_method"},
        )

        if error:
            return failed_result(question_id, error)

        actual_distribution = extract_category_distribution(
            result.get("category_values")
        )

        comparisons = [
            ("Total rows", expected["total_rows"], result.get("rows_after_filtering")),
            ("Unique category count", expected["unique_count"], result.get("unique_count")),
            ("Payment method distribution", expected["distribution"], actual_distribution),
        ]

    # Q08: Payment method distribution among cancelled transactions
    elif question_id == "Q08":

        expected = calculate_q08_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_value_distribution",
            {
                "column": "payment_method",
                "filters": {"order_status": "Cancelled"},
                "limit": None,
            },
        )

        if error:
            return failed_result(question_id, error)

        actual_distribution = extract_category_distribution(
            result.get("category_values")
        )

        comparisons = [
            ("Cancelled transaction count", expected["total_rows"], result.get("rows_after_filtering")),
            ("Payment method distribution", expected["distribution"], actual_distribution),
        ]

    # Q09: Order status counts
    elif question_id == "Q09":

        expected = calculate_q09_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_value_distribution",
            {"column": "order_status"},
        )

        if error:
            return failed_result(question_id, error)

        actual_counts = extract_category_counts(
            result.get("category_values")
        )

        comparisons = [
            ("Order status counts", expected["category_counts"], actual_counts),
        ]

    # Q10: Top five countries
    elif question_id == "Q10":

        expected = calculate_q10_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_value_distribution",
            {"column": "country"},
        )

        if error:
            return failed_result(question_id, error)

        categories = result.get("category_values")
        actual_top_countries = None

        if isinstance(categories, list):

            actual_top_countries = [
                {
                    "country": item.get("value"),
                    "count": item.get("count"),
                    "percentage": item.get("percentage"),
                }
                for item in categories[:5]
                if isinstance(item, dict)
            ]

            if len(actual_top_countries) != min(5, len(categories)):
                actual_top_countries = None

        comparisons = [
            ("Top five countries", expected["top_countries"], actual_top_countries),
        ]

    # Q11: Delivered average transaction amount by currency
    elif question_id == "Q11":

        expected = calculate_q11_expected(df)

        result, error = get_tool_result(
            tool_results,
            "group_and_aggregate",
            {
                "group_by": "currency",
                "metric": "total_amount",
                "aggregation": "mean",
                "filters": {"order_status": "Delivered"},
                "sort_by": "result",
                "ascending": False,
            },
        )

        if error:
            return failed_result(question_id, error)

        currency_results = extract_currency_results(
            result.get("results")
        )

        actual_means = None

        if currency_results is not None:
            actual_means = {
                currency: item.get("result")
                for currency, item in currency_results.items()
            }

        comparisons = [
            ("Aggregation", expected["aggregation"], result.get("aggregation")),
            ("Metric", expected["metric"], result.get("metric")),
            ("Filter", expected["filter"], result.get("filters")),
            (
                "Delivered transaction count",
                int((df["order_status"] == "Delivered").sum()),
                result.get("rows_after_filtering"),
            ),
            ("Average amount by currency", expected["results_by_currency"], actual_means),
        ]

    # Q12: Delivered versus Cancelled average amount by currency
    elif question_id == "Q12":

        expected = calculate_q12_expected(df)

        result, error = get_tool_result(
            tool_results,
            "compare_groups",
            {
                "comparison_column": "order_status",
                "group_a": "Delivered",
                "group_b": "Cancelled",
                "metric": "total_amount",
                "aggregation": "mean",
                "group_by": "currency",
            },
        )

        if error:
            return failed_result(question_id, error)

        currency_results = extract_currency_results(
            result.get("results")
        )

        actual_comparison = None

        if currency_results is not None:

            actual_comparison = {}

            for currency, item in currency_results.items():

                actual_comparison[currency] = {
                    "delivered_mean": item.get("group_a_value"),
                    "cancelled_mean": item.get("group_b_value"),
                    "difference_b_minus_a": item.get("difference_b_minus_a"),
                    "percentage_difference_vs_delivered": item.get("percentage_difference_vs_a"),
                }

        actual_rows = result.get("rows_after_filtering", {})

        comparisons = [
            ("Comparison definition", expected["comparison"], "Delivered vs Cancelled"),
            (
                "Delivered row count",
                int((df["order_status"] == "Delivered").sum()),
                actual_rows.get("group_a") if isinstance(actual_rows, dict) else None,
            ),
            (
                "Cancelled row count",
                int((df["order_status"] == "Cancelled").sum()),
                actual_rows.get("group_b") if isinstance(actual_rows, dict) else None,
            ),
            ("Currency comparison results", expected["results_by_currency"], actual_comparison),
        ]

    # Q15: Highest country among cancelled transactions
    elif question_id == "Q15":

        expected = calculate_q15_expected(df)

        result, error = get_tool_result(
            tool_results,
            "get_value_distribution",
            {
                "column": "country",
                "filters": {"order_status": "Cancelled"},
                "limit": None,
            },
        )

        if error:
            return failed_result(question_id, error)

        categories = result.get("category_values", [])

        highest_country = None
        highest_count = None
        highest_percentage = None

        if isinstance(categories, list) and categories:

            first = categories[0]

            if isinstance(first, dict):
                highest_country = first.get("value")
                highest_count = first.get("count")
                highest_percentage = first.get("percentage")

        comparisons = [
            ("Total cancelled transactions", expected["total_cancelled"], result.get("rows_after_filtering")),
            ("Highest country", expected["highest_country"], highest_country),
            ("Highest country transaction count", expected["transaction_count"], highest_count),
            ("Percentage of cancelled transactions", expected["percentage"], highest_percentage),
        ]

    else:
        raise ValueError(
            f"Unsupported question: {question_id}"
        )

    return build_result(
        question_id,
        build_checks(comparisons),
    )


# --------------------------------------------------
# 10. COMMON FAILURE RESULT
# --------------------------------------------------

def failed_result(question_id, error):

    return {
        "question_id": question_id,
        "status": "FAIL",
        "reason": error,
        "checks": [],
    }


# --------------------------------------------------
# 11. EVALUATE Q13
# --------------------------------------------------

def evaluate_q13(df, baseline_row):

    expected = calculate_q13_expected(df)

    tool_results = parse_tool_results(
        baseline_row["tool_results"]
    )

    if tool_results is None:
        return {
            "question_id": "Q13",
            "status": "NOT EVALUATED",
            "reason": "Structured tool results are missing or invalid.",
            "checks": [],
        }

    tool_call, error = get_tool_result(
        tool_results,
        "compare_groups",
        {
            "comparison_column": "payment_status",
            "group_a": "Completed",
            "group_b": "Failed",
            "metric": None,
            "aggregation": "count",
            "group_by": None,
            "filters": {},
        },
    )

    if error:
        return failed_result("Q13", error)

    rows = tool_call.get("rows_after_filtering", {})
    result_rows = tool_call.get("results", [])

    if not isinstance(rows, dict) or not isinstance(result_rows, list):
        return failed_result(
            "Q13",
            "Unexpected tool result structure.",
        )

    if len(result_rows) != 1 or not isinstance(result_rows[0], dict):
        return failed_result(
            "Q13",
            "Expected exactly one comparison result.",
        )

    actual = result_rows[0]

    comparisons = [
        ("Completed rows after filtering", expected["group_a_count"], rows.get("group_a")),
        ("Failed rows after filtering", expected["group_b_count"], rows.get("group_b")),
        ("Completed count", expected["group_a_count"], actual.get("group_a_value")),
        ("Failed count", expected["group_b_count"], actual.get("group_b_value")),
        ("Difference (Failed - Completed)", expected["difference_b_minus_a"], actual.get("difference_b_minus_a")),
        ("Percentage difference", expected["percentage_difference_vs_a"], actual.get("percentage_difference_vs_a")),
    ]

    return build_result(
        "Q13",
        build_checks(comparisons),
    )


# --------------------------------------------------
# 12. EVALUATE Q14 - MULTI-TOOL QUESTION
# --------------------------------------------------

def evaluate_q14(df, baseline_row):

    expected = calculate_q14_expected(df)

    tool_results = parse_tool_results(
        baseline_row["tool_results"]
    )

    if tool_results is None:
        return {
            "question_id": "Q14",
            "status": "NOT EVALUATED",
            "reason": "Structured tool results are missing or invalid.",
            "checks": [],
        }

    distribution_result, error = get_tool_result(
        tool_results,
        "get_value_distribution",
        {"column": "order_status"},
    )

    if error:
        return failed_result(
            "Q14",
            f"First tool call: {error}",
        )

    # Verify that the distribution tool was called before comparison.
    distribution_index = next(
        (
            index
            for index, item in enumerate(tool_results)
            if isinstance(item, dict)
            and item.get("tool") == "get_value_distribution"
        ),
        None,
    )

    comparison_index = next(
        (
            index
            for index, item in enumerate(tool_results)
            if isinstance(item, dict)
            and item.get("tool") == "compare_groups"
        ),
        None,
    )

    if (
        distribution_index is None
        or comparison_index is None
        or distribution_index >= comparison_index
    ):
        return failed_result(
            "Q14",
            "The two tools were not executed in the expected sequence.",
        )

    actual_categories = distribution_result.get("category_values")

    actual_top_statuses = None

    if isinstance(actual_categories, list):

        actual_top_statuses = [
            {
                "status": item.get("value"),
                "count": item.get("count"),
            }
            for item in actual_categories[:2]
            if isinstance(item, dict)
        ]

        if len(actual_top_statuses) != 2:
            actual_top_statuses = None

    status_a = expected["top_two_statuses"][0]["status"]
    status_b = expected["top_two_statuses"][1]["status"]

    comparison_result, error = get_tool_result(
        tool_results,
        "compare_groups",
        {
            "comparison_column": "order_status",
            "group_a": status_a,
            "group_b": status_b,
            "metric": "total_amount",
            "aggregation": "mean",
            "group_by": "currency",
        },
    )

    if error:
        return failed_result(
            "Q14",
            f"Second tool call: {error}",
        )

    currency_results = extract_currency_results(
        comparison_result.get("results")
    )

    actual_comparison = None

    if currency_results is not None:

        actual_comparison = {}

        for currency, item in currency_results.items():

            actual_comparison[currency] = {
                "group_a_mean": item.get("group_a_value"),
                "group_b_mean": item.get("group_b_value"),
                "difference_b_minus_a": item.get("difference_b_minus_a"),
                "percentage_difference_vs_a": item.get("percentage_difference_vs_a"),
            }

    comparisons = [
        (
            "Top two order statuses",
            expected["top_two_statuses"],
            actual_top_statuses,
        ),
        (
            "Comparison definition",
            expected["comparison"],
            f"{status_a} vs {status_b}",
        ),
        (
            "Currency comparison results",
            expected["results_by_currency"],
            actual_comparison,
        ),
    ]

    return build_result(
        "Q14",
        build_checks(comparisons),
    )


# --------------------------------------------------
# 13. EVALUATION REPORT
# --------------------------------------------------

def print_report(result):

    print("\n" + "=" * 55)
    print("DATASENSE AI - ACCURACY EVALUATION")
    print("=" * 55)

    print("Question:", result["question_id"])
    print("Status:", result["status"])

    if result.get("reason"):
        print("Reason:", result["reason"])

    if result.get("checks"):

        print(
            "\nChecks passed:",
            f'{result["passed_checks"]}/{result["total_checks"]}',
        )

        print("\nDetailed comparison")
        print("-" * 55)

        for check in result["checks"]:

            status = "PASS" if check["passed"] else "FAIL"

            print(f"\n{check['metric']}: {status}")
            print("  Expected:", check["expected"])
            print("  Actual:  ", check["actual"])

    print("\n" + "=" * 55)


# --------------------------------------------------
# 14. MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate DataSense AI analytical accuracy."
    )

    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=SUPPORTED_QUESTIONS,
        help="Question IDs to evaluate.",
    )

    args = parser.parse_args()

    question_ids = [
        question_id.upper()
        for question_id in args.question_ids
    ]

    unsupported = [
        question_id
        for question_id in question_ids
        if question_id not in SUPPORTED_QUESTIONS
    ]

    if unsupported:
        parser.error(
            "Unsupported question IDs: "
            + ", ".join(unsupported)
        )

    baseline = load_baseline()
    df = load_dataset()

    all_results = []

    for question_id in question_ids:

        matching_rows = baseline[
            baseline["question_id"] == question_id
        ]

        if matching_rows.empty:

            result = {
                "question_id": question_id,
                "status": "NOT EVALUATED",
                "reason": "Question not found in baseline.",
                "checks": [],
            }

        elif len(matching_rows) > 1:

            result = {
                "question_id": question_id,
                "status": "NOT EVALUATED",
                "reason": "Multiple baseline rows found for this question.",
                "checks": [],
            }

        else:

            baseline_row = matching_rows.iloc[0]

            if question_id == "Q13":

                result = evaluate_q13(
                    df=df,
                    baseline_row=baseline_row,
                )

            elif question_id == "Q14":

                result = evaluate_q14(
                    df=df,
                    baseline_row=baseline_row,
                )

            else:

                result = evaluate_question(
                    df=df,
                    question_id=question_id,
                    baseline_row=baseline_row,
                )

        all_results.append(result)
        print_report(result)

    # Overall summary
    total_passed = sum(
        result["status"] == "PASS"
        for result in all_results
    )

    total_failed = sum(
        result["status"] == "FAIL"
        for result in all_results
    )

    total_not_evaluated = sum(
        result["status"] == "NOT EVALUATED"
        for result in all_results
    )

    total_checks_passed = sum(
        result.get("passed_checks", 0)
        for result in all_results
    )

    total_checks = sum(
        result.get("total_checks", 0)
        for result in all_results
    )

    print("\n")
    print("=" * 55)
    print("OVERALL ACCURACY EVALUATION SUMMARY")
    print("=" * 55)

    print("Questions evaluated:", len(all_results))
    print("Questions passed:", total_passed)
    print("Questions failed:", total_failed)
    print("Questions not evaluated:", total_not_evaluated)

    print(
        "Accuracy checks:",
        f"{total_checks_passed}/{total_checks}",
    )

    if total_checks:
        print(
            "Check accuracy:",
            f"{(total_checks_passed / total_checks) * 100:.2f}%",
        )

    print("=" * 55)


if __name__ == "__main__":
    main()
