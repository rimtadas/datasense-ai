
import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE_FILE = Path(__file__).parent / "baseline_results.csv"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def parse_json_list(value):
    if pd.isna(value) or not str(value).strip():
        return None

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None

    return parsed if isinstance(parsed, list) else None


def load_tool_results(value):
    results = parse_json_list(value)

    if results is None:
        return None

    return [
        item for item in results
        if isinstance(item, dict)
    ]


def get_calls(results, tool_name=None):
    if results is None:
        return []

    if tool_name is None:
        return results

    return [
        item for item in results
        if item.get("tool") == tool_name
    ]


def get_arguments(call):
    arguments = call.get("arguments", {})
    return arguments if isinstance(arguments, dict) else {}


def get_result(call):
    result = call.get("result")
    return result if isinstance(result, dict) else {}


def is_successful(call):
    return call.get("success") is True


def normalise(value):
    if isinstance(value, str):
        return value.casefold()
    return value


def same_value(actual, expected):
    return normalise(actual) == normalise(expected)


def get_filter_value(filters, column):
    """
    Read a simple equality filter.

    Supports:
        {"column": "value"}
        {"column": {"eq": "value"}}
    """

    if not isinstance(filters, dict) or column not in filters:
        return None

    condition = filters[column]

    if isinstance(condition, dict):
        if len(condition) != 1:
            return None

        operator, value = next(iter(condition.items()))

        if operator != "eq":
            return None

        return value

    return condition


def category_values(result):
    values = result.get("category_values", [])

    return values if isinstance(values, list) else []


def category_name(item):
    return item.get("value") if isinstance(item, dict) else None


def category_count(item):
    if not isinstance(item, dict):
        return None

    return item.get("count")


def add_check(checks, name, passed, detail=""):
    checks.append({
        "name": name,
        "passed": bool(passed),
        "detail": detail,
    })


def add_argument_check(checks, name, passed, detail=""):
    add_check(checks, name, passed, detail)


def add_result_check(checks, name, passed, detail=""):
    add_check(checks, name, passed, detail)


def first_call(results, tool_name):
    calls = get_calls(results, tool_name)
    return calls[0] if calls else None


def successful_tool_calls(results):
    if results is None:
        return []

    return [
        call for call in results
        if is_successful(call)
    ]


def has_complete_categories(result):
    values = category_values(result)

    return (
        bool(values)
        and result.get("category_values_truncated") is False
        and result.get("unique_count") == len(values)
    )


def has_distribution_percentages(result):
    values = category_values(result)

    return (
        bool(values)
        and all(
            isinstance(item, dict)
            and "count" in item
            and "percentage" in item
            for item in values
        )
    )


# --------------------------------------------------
# QUESTION EVALUATORS
# --------------------------------------------------

def evaluate_question(question_id, results):
    argument_checks = []
    result_checks = []
    warnings = []

    calls = successful_tool_calls(results)

    def require_call(tool_name):
        call = first_call(calls, tool_name)

        add_argument_check(
            argument_checks,
            f"Uses {tool_name}",
            call is not None,
            "Found successful call." if call else "No successful call found.",
        )

        return call

    def require_column(call, expected):
        actual = get_arguments(call).get("column") if call else None

        add_argument_check(
            argument_checks,
            f"Column is {expected}",
            same_value(actual, expected),
            f"Actual: {actual}",
        )

    # --------------------------------------------------
    # Q01 - DATASET OVERVIEW
    # --------------------------------------------------

    if question_id == "Q01":

        call = require_call("get_dataset_overview")

        if call:
            add_argument_check(
                argument_checks,
                "No unnecessary arguments",
                not get_arguments(call),
                f"Actual: {get_arguments(call)}",
            )

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Contains row count",
            isinstance(result.get("row_count"), int),
        )

        add_result_check(
            result_checks,
            "Contains column count",
            isinstance(result.get("column_count"), int),
        )

        add_result_check(
            result_checks,
            "Contains duplicate row count",
            isinstance(result.get("duplicate_rows"), int),
        )

    # --------------------------------------------------
    # Q02 - AVAILABLE AND EXCLUDED COLUMNS
    # --------------------------------------------------

    elif question_id == "Q02":

        call = require_call("get_dataset_overview")
        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Contains available columns",
            isinstance(result.get("columns"), list)
            and len(result["columns"]) > 0,
        )

        add_result_check(
            result_checks,
            "Contains excluded columns",
            isinstance(result.get("excluded_columns"), list),
        )

        add_result_check(
            result_checks,
            "Contains visible column count",
            isinstance(result.get("context_column_count"), int),
        )

    # --------------------------------------------------
    # Q03 - DATE RANGE
    # --------------------------------------------------

    elif question_id == "Q03":

        call = require_call("get_column_summary")
        require_column(call, "purchase_date")

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Contains minimum date",
            bool(result.get("min_date")),
        )

        add_result_check(
            result_checks,
            "Contains maximum date",
            bool(result.get("max_date")),
        )

    # --------------------------------------------------
    # Q04 - NUMERIC COLUMN SUMMARY
    # --------------------------------------------------

    elif question_id == "Q04":

        call = require_call("get_column_summary")
        require_column(call, "quantity")

        result = get_result(call) if call else {}

        for field in ("min", "max", "mean", "median"):

            add_result_check(
                result_checks,
                f"Contains {field}",
                result.get(field) is not None,
            )

    # --------------------------------------------------
    # Q05 - CATEGORY VALUES AND COUNTS
    # --------------------------------------------------

    elif question_id == "Q05":

        call = (
            first_call(calls, "get_categorical_values")
            or first_call(calls, "get_value_distribution")
        )

        add_argument_check(
            argument_checks,
            "Uses a category discovery or distribution tool",
            call is not None,
        )

        require_column(call, "order_status")

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Returns all order statuses",
            has_complete_categories(result),
        )

        add_result_check(
            result_checks,
            "Returns category counts",
            bool(category_values(result))
            and all(
                category_count(item) is not None
                for item in category_values(result)
            ),
        )

    # --------------------------------------------------
    # Q06 - AVAILABLE PAYMENT METHODS
    # --------------------------------------------------

    elif question_id == "Q06":

        call = (
            first_call(calls, "get_categorical_values")
            or first_call(calls, "get_value_distribution")
        )

        add_argument_check(
            argument_checks,
            "Uses a category discovery or distribution tool",
            call is not None,
        )

        require_column(call, "payment_method")

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Returns all payment methods",
            has_complete_categories(result),
        )

    # --------------------------------------------------
    # Q07 - PAYMENT METHOD DISTRIBUTION
    # --------------------------------------------------

    elif question_id == "Q07":

        call = require_call("get_value_distribution")
        require_column(call, "payment_method")

        args = get_arguments(call) if call else {}

        add_argument_check(
            argument_checks,
            "Does not apply unnecessary filters",
            not args.get("filters"),
            f"Actual filters: {args.get('filters')}",
        )

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Returns counts and percentages",
            has_distribution_percentages(result),
        )

        add_result_check(
            result_checks,
            "Returns complete distribution",
            result.get("category_values_truncated") is False,
        )

    # --------------------------------------------------
    # Q08 - FILTERED PAYMENT METHOD DISTRIBUTION
    # --------------------------------------------------

    elif question_id == "Q08":

        call = require_call("get_value_distribution")
        require_column(call, "payment_method")

        args = get_arguments(call) if call else {}
        filters = args.get("filters")

        filter_value = get_filter_value(
            filters,
            "order_status",
        )

        add_argument_check(
            argument_checks,
            "Filters order_status to Cancelled",
            same_value(filter_value, "Cancelled"),
            f"Actual filters: {filters}",
        )

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Returns filtered row count",
            isinstance(result.get("rows_after_filtering"), int),
        )

        add_result_check(
            result_checks,
            "Returns counts and percentages",
            has_distribution_percentages(result),
        )

        add_result_check(
            result_checks,
            "Returns complete payment method distribution",
            result.get("category_values_truncated") is False,
        )

    # --------------------------------------------------
    # Q09 - TRANSACTION COUNTS BY STATUS
    # --------------------------------------------------

    elif question_id == "Q09":

        distribution_call = first_call(
            calls, "get_value_distribution"
        )

        aggregation_call = first_call(
            calls, "group_and_aggregate"
        )

        call = distribution_call or aggregation_call

        add_argument_check(
            argument_checks,
            "Uses distribution or aggregation",
            call is not None,
        )

        if distribution_call:

            require_column(call, "order_status")

            args = get_arguments(call)

            add_argument_check(
                argument_checks,
                "Does not apply restrictive filters",
                not args.get("filters"),
            )

        elif aggregation_call:

            args = get_arguments(call)

            add_argument_check(
                argument_checks,
                "Groups by order_status",
                same_value(args.get("group_by"), "order_status"),
                f"Actual: {args.get('group_by')}",
            )

            add_argument_check(
                argument_checks,
                "Uses count aggregation",
                same_value(args.get("aggregation"), "count"),
                f"Actual: {args.get('aggregation')}",
            )

            add_argument_check(
                argument_checks,
                "Does not apply unnecessary filters",
                not args.get("filters"),
            )

        result = get_result(call) if call else {}

        add_result_check(
            result_checks,
            "Returns counts for each status",
            bool(result.get("results"))
            or bool(category_values(result)),
        )

    # --------------------------------------------------
    # Q10 - TOP FIVE COUNTRIES
    # --------------------------------------------------

    elif question_id == "Q10":

        distribution_call = first_call(
            calls, "get_value_distribution"
        )

        aggregation_call = first_call(
            calls, "group_and_aggregate"
        )

        call = distribution_call or aggregation_call

        add_argument_check(
            argument_checks,
            "Uses distribution or aggregation",
            call is not None,
        )

        args = get_arguments(call) if call else {}

        if distribution_call:

            require_column(call, "country")

            add_argument_check(
                argument_checks,
                "Does not apply unnecessary filters",
                not args.get("filters"),
            )

            if args.get("limit") != 5:
                warnings.append(
                    "Q10: Prefer limit=5 to avoid retrieving all countries."
                )

            result = get_result(call) if call else {}
            values = category_values(result)

        elif aggregation_call:

            add_argument_check(
                argument_checks,
                "Groups by country",
                same_value(args.get("group_by"), "country"),
            )

            add_argument_check(
                argument_checks,
                "Uses count aggregation",
                same_value(args.get("aggregation"), "count"),
            )

            add_argument_check(
                argument_checks,
                "Sorts descending by count",
                args.get("ascending", False) is False
                and args.get("sort_by", "result") == "result",
            )

            add_argument_check(
                argument_checks,
                "Requests top five results",
                args.get("limit") == 5,
                f"Actual limit: {args.get('limit')}",
            )

            result = get_result(call) if call else {}
            values = result.get("results", [])

        else:
            result = {}
            values = []

        add_result_check(
            result_checks,
            "Returns at least five countries",
            len(values) >= 5,
        )

        counts = [
            category_count(item)
            if "value" in item
            else item.get("result")
            for item in values
            if isinstance(item, dict)
        ]

        add_result_check(
            result_checks,
            "Countries are ordered by descending count",
            len(counts) >= 2
            and all(
                counts[i] >= counts[i + 1]
                for i in range(len(counts) - 1)
            ),
        )

    # --------------------------------------------------
    # Q11 - DELIVERED MEAN AMOUNT BY CURRENCY
    # --------------------------------------------------

    elif question_id == "Q11":

        call = require_call("group_and_aggregate")
        args = get_arguments(call) if call else {}

        add_argument_check(
            argument_checks,
            "Groups by currency",
            same_value(args.get("group_by"), "currency"),
        )

        add_argument_check(
            argument_checks,
            "Uses total_amount metric",
            same_value(args.get("metric"), "total_amount"),
        )

        add_argument_check(
            argument_checks,
            "Uses mean aggregation",
            same_value(args.get("aggregation"), "mean"),
        )

        filter_value = get_filter_value(
            args.get("filters"),
            "order_status",
        )

        add_argument_check(
            argument_checks,
            "Filters to Delivered",
            same_value(filter_value, "Delivered"),
        )

        result = get_result(call) if call else {}
        rows = result.get("results", [])

        add_result_check(
            result_checks,
            "Returns currency-level results",
            bool(rows)
            and all("currency" in row for row in rows),
        )

        add_result_check(
            result_checks,
            "Returns numeric mean values",
            bool(rows)
            and all(
                isinstance(row.get("result"), (int, float))
                for row in rows
            ),
        )

    # --------------------------------------------------
    # Q12 - DELIVERED VS CANCELLED
    # --------------------------------------------------

    elif question_id == "Q12":

        call = require_call("compare_groups")
        args = get_arguments(call) if call else {}

        add_argument_check(
            argument_checks,
            "Compares order_status",
            same_value(
                args.get("comparison_column"),
                "order_status",
            ),
        )

        add_argument_check(
            argument_checks,
            "Group A is Delivered",
            same_value(args.get("group_a"), "Delivered"),
        )

        add_argument_check(
            argument_checks,
            "Group B is Cancelled",
            same_value(args.get("group_b"), "Cancelled"),
        )

        add_argument_check(
            argument_checks,
            "Uses total_amount metric",
            same_value(args.get("metric"), "total_amount"),
        )

        add_argument_check(
            argument_checks,
            "Uses mean aggregation",
            same_value(args.get("aggregation"), "mean"),
        )

        add_argument_check(
            argument_checks,
            "Groups by currency",
            same_value(args.get("group_by"), "currency"),
        )

        result = get_result(call) if call else {}
        rows = result.get("results", [])

        add_result_check(
            result_checks,
            "Returns currency-level comparisons",
            bool(rows)
            and all("currency" in row for row in rows),
        )

        add_result_check(
            result_checks,
            "Returns absolute differences",
            bool(rows)
            and all(
                "difference_b_minus_a" in row
                for row in rows
            ),
        )

        add_result_check(
            result_checks,
            "Returns percentage differences",
            bool(rows)
            and all(
                "percentage_difference_vs_a" in row
                for row in rows
            ),
        )

    # --------------------------------------------------
    # Q13 - COMPLETED VS FAILED COUNTS
    # --------------------------------------------------

    elif question_id == "Q13":

        call = require_call("compare_groups")
        args = get_arguments(call) if call else {}

        add_argument_check(
            argument_checks,
            "Compares payment_status",
            same_value(
                args.get("comparison_column"),
                "payment_status",
            ),
        )

        add_argument_check(
            argument_checks,
            "Group A is Completed",
            same_value(args.get("group_a"), "Completed"),
        )

        add_argument_check(
            argument_checks,
            "Group B is Failed",
            same_value(args.get("group_b"), "Failed"),
        )

        add_argument_check(
            argument_checks,
            "Uses count aggregation",
            same_value(args.get("aggregation"), "count"),
        )

        add_argument_check(
            argument_checks,
            "Does not aggregate a numeric metric",
            args.get("metric") is None,
        )

        result = get_result(call) if call else {}
        rows = result.get("results", [])

        add_result_check(
            result_checks,
            "Returns both group counts",
            len(rows) > 0
            and isinstance(
                result.get("rows_after_filtering"), dict
            )
            and all(
                key in result["rows_after_filtering"]
                for key in ("group_a", "group_b")
            ),
        )

        add_result_check(
            result_checks,
            "Returns comparison difference",
            bool(rows)
            and "difference_b_minus_a" in rows[0],
        )

    # --------------------------------------------------
    # Q14 - DISCOVER TOP TWO, THEN COMPARE
    # --------------------------------------------------

    elif question_id == "Q14":

        discovery_call = (
            first_call(calls, "get_categorical_values")
            or first_call(calls, "get_value_distribution")
        )

        compare_call = first_call(calls, "compare_groups")

        add_argument_check(
            argument_checks,
            "Uses category discovery",
            discovery_call is not None,
        )

        add_argument_check(
            argument_checks,
            "Uses group comparison",
            compare_call is not None,
        )

        discovery_args = (
            get_arguments(discovery_call)
            if discovery_call else {}
        )

        add_argument_check(
            argument_checks,
            "Discovers order_status",
            same_value(
                discovery_args.get("column"),
                "order_status",
            ),
        )

        discovery_result = (
            get_result(discovery_call)
            if discovery_call else {}
        )

        discovered = category_values(discovery_result)

        top_two = sorted(
            discovered,
            key=lambda item: (
                category_count(item) is None,
                -(category_count(item) or 0),
            ),
        )[:2]

        top_two_values = [
            category_name(item)
            for item in top_two
        ]

        compare_args = (
            get_arguments(compare_call)
            if compare_call else {}
        )

        actual_groups = {
            normalise(compare_args.get("group_a")),
            normalise(compare_args.get("group_b")),
        }

        expected_groups = {
            normalise(value)
            for value in top_two_values
        }

        add_argument_check(
            argument_checks,
            "Compares the two most common discovered statuses",
            len(expected_groups) == 2
            and actual_groups == expected_groups,
            f"Discovered: {top_two_values}; "
            f"compared: {compare_args.get('group_a')}, "
            f"{compare_args.get('group_b')}",
        )

        add_argument_check(
            argument_checks,
            "Compares order_status",
            same_value(
                compare_args.get("comparison_column"),
                "order_status",
            ),
        )

        add_argument_check(
            argument_checks,
            "Uses total_amount metric",
            same_value(
                compare_args.get("metric"),
                "total_amount",
            ),
        )

        add_argument_check(
            argument_checks,
            "Uses mean aggregation",
            same_value(
                compare_args.get("aggregation"),
                "mean",
            ),
        )

        add_argument_check(
            argument_checks,
            "Groups by currency",
            same_value(
                compare_args.get("group_by"),
                "currency",
            ),
        )

        compare_result = (
            get_result(compare_call)
            if compare_call else {}
        )

        comparison_rows = compare_result.get("results", [])

        add_result_check(
            result_checks,
            "Discovery returns category counts",
            len(top_two_values) == 2
            and all(
                category_count(item) is not None
                for item in top_two
            ),
        )

        add_result_check(
            result_checks,
            "Comparison returns currency-level results",
            bool(comparison_rows)
            and all(
                "currency" in row
                for row in comparison_rows
            ),
        )

    # --------------------------------------------------
    # Q15 - CANCELLED TRANSACTIONS BY COUNTRY
    # --------------------------------------------------

    elif question_id == "Q15":

        call = require_call("get_value_distribution")
        require_column(call, "country")

        args = get_arguments(call) if call else {}

        filter_value = get_filter_value(
            args.get("filters"),
            "order_status",
        )

        add_argument_check(
            argument_checks,
            "Filters to Cancelled",
            same_value(filter_value, "Cancelled"),
            f"Actual filters: {args.get('filters')}",
        )

        result = get_result(call) if call else {}
        values = category_values(result)

        add_result_check(
            result_checks,
            "Returns country counts and percentages",
            has_distribution_percentages(result),
        )

        add_result_check(
            result_checks,
            "Returns at least one country",
            bool(values),
        )

        add_result_check(
            result_checks,
            "Returns highest-count country first",
            len(values) >= 1
            and all(
                category_count(values[i])
                >= category_count(values[i + 1])
                for i in range(len(values) - 1)
            ),
        )

    return {
        "question_id": question_id,
        "argument_checks": argument_checks,
        "result_checks": result_checks,
        "warnings": warnings,
    }


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(question_ids=None):

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}"
        )

    df = pd.read_csv(BASELINE_FILE)

    required_columns = {
        "question_id",
        "tool_results",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Baseline is missing columns: {sorted(missing)}"
        )

    df = df.drop_duplicates(
        subset=["question_id"],
        keep="last",
    )

    baseline = {
        str(row["question_id"]): row
        for _, row in df.iterrows()
    }

    all_question_ids = [
        f"Q{i:02d}" for i in range(1, 16)
    ]

    if question_ids:

        unknown = set(question_ids) - set(all_question_ids)

        if unknown:
            raise ValueError(
                f"Unknown question IDs: {sorted(unknown)}"
            )

        selected_ids = question_ids

    else:
        selected_ids = all_question_ids

    reports = []

    for question_id in selected_ids:

        row = baseline.get(question_id)

        if row is None:
            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
            })
            continue

        results = load_tool_results(row["tool_results"])

        if results is None:
            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
            })
            continue

        report = evaluate_question(
            question_id,
            results,
        )

        report["status"] = "EVALUATED"
        reports.append(report)

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

    all_argument_checks = [
        check
        for report in evaluated
        for check in report["argument_checks"]
    ]

    all_result_checks = [
        check
        for report in evaluated
        for check in report["result_checks"]
    ]

    argument_passes = sum(
        check["passed"] for check in all_argument_checks
    )

    result_passes = sum(
        check["passed"] for check in all_result_checks
    )

    argument_questions_passed = sum(
        bool(report["argument_checks"])
        and all(
            check["passed"]
            for check in report["argument_checks"]
        )
        for report in evaluated
    )

    result_questions_passed = sum(
        bool(report["result_checks"])
        and all(
            check["passed"]
            for check in report["result_checks"]
        )
        for report in evaluated
    )

    fully_passed = sum(
        bool(report["argument_checks"])
        and bool(report["result_checks"])
        and all(
            check["passed"]
            for check in (
                report["argument_checks"]
                + report["result_checks"]
            )
        )
        for report in evaluated
    )

    warnings = [
        warning
        for report in evaluated
        for warning in report["warnings"]
    ]

    failed_arguments = [
        check
        for check in all_argument_checks
        if not check["passed"]
    ]

    failed_results = [
        check
        for check in all_result_checks
        if not check["passed"]
    ]

    print("\n" + "=" * 65)
    print("DATASENSE AI - TOOL ARGUMENT EVALUATION")
    print("=" * 65)

    print("\nOVERALL RESULTS")
    print("-" * 65)

    print(f"Questions evaluated:          {len(evaluated)}")
    print(f"Questions not evaluated:      {len(not_evaluated)}")

    print(
        f"Argument correctness:         "
        f"{argument_passes}/{len(all_argument_checks)} "
        f"({argument_passes / len(all_argument_checks) * 100:.2f}%)"
        if all_argument_checks else
        "Argument correctness:         N/A"
    )

    print(
        f"Result sufficiency:            "
        f"{result_passes}/{len(all_result_checks)} "
        f"({result_passes / len(all_result_checks) * 100:.2f}%)"
        if all_result_checks else
        "Result sufficiency:            N/A"
    )

    print(
        f"Questions with correct args:   "
        f"{argument_questions_passed}/{len(evaluated)}"
    )

    print(
        f"Questions with sufficient data:"
        f" {result_questions_passed}/{len(evaluated)}"
    )

    print(
        f"Questions fully passed:        "
        f"{fully_passed}/{len(evaluated)}"
    )

    print(f"Argument failures:             {len(failed_arguments)}")
    print(f"Result sufficiency failures:   {len(failed_results)}")
    print(f"Efficiency warnings:            {len(warnings)}")

    print("\nQUESTION-LEVEL RESULTS")
    print("-" * 65)

    for report in reports:

        question_id = report["question_id"]

        if report["status"] == "NOT EVALUATED":
            print(f"\n{question_id}: NOT EVALUATED")
            continue

        args = report["argument_checks"]
        results = report["result_checks"]

        args_passed = sum(
            check["passed"] for check in args
        )

        results_passed = sum(
            check["passed"] for check in results
        )

        print(f"\n{question_id}")
        print(
            f"  Arguments: {args_passed}/{len(args)} "
            f"{'PASS' if args_passed == len(args) else 'FAIL'}"
        )

        for check in args:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"    [{status}] {check['name']}")

            if not check["passed"] and check["detail"]:
                print(f"           {check['detail']}")

        print(
            f"  Results:   {results_passed}/{len(results)} "
            f"{'PASS' if results_passed == len(results) else 'FAIL'}"
        )

        for check in results:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"    [{status}] {check['name']}")

        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")

    print("\n" + "=" * 65)
    print("TOOL ARGUMENT EVALUATION COMPLETE")
    print("=" * 65)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate DataSense AI tool arguments and results."
    )

    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Optional question IDs to evaluate.",
    )

    args = parser.parse_args()

    reports = evaluate(
        question_ids=args.question_ids
    )

    print_report(reports)
