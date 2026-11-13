
import argparse
import math
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

from data_connector.huggingface_loader import load_huggingface_dataset

from dataset_context.dataset_configs import (
    ECOMMERCE_CONFIG,
    ECOMMERCE_DATETIME_COLUMNS,
)

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
# CONFIGURATION
# --------------------------------------------------

BASELINE_FILE = Path(__file__).parent / "baseline_results.csv"

DATASET_REPO = os.getenv(
    "DATASENSE_DATASET_REPO_ID",
    "davidev07/synthetic-ecommerce-dataset",
)

DATASET_FILE = os.getenv(
    "DATASENSE_DATASET_FILE",
    "transactions.csv",
)

DATASET_CONFIG = ECOMMERCE_CONFIG


EXPECTED_FUNCTIONS = {
    "Q01": calculate_q01_expected,
    "Q02": calculate_q02_expected,
    "Q03": calculate_q03_expected,
    "Q04": calculate_q04_expected,
    "Q05": calculate_q05_expected,
    "Q06": calculate_q06_expected,
    "Q07": calculate_q07_expected,
    "Q08": calculate_q08_expected,
    "Q09": calculate_q09_expected,
    "Q10": calculate_q10_expected,
    "Q11": calculate_q11_expected,
    "Q12": calculate_q12_expected,
    "Q13": calculate_q13_expected,
    "Q14": calculate_q14_expected,
    "Q15": calculate_q15_expected,
}


# --------------------------------------------------
# TEXT AND NUMBER HELPERS
# --------------------------------------------------

def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()
    value = value.replace("\u2212", "-")
    value = value.replace("\u2013", "-")
    value = value.replace("\u2014", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def contains_text(answer, expected):
    return normalize_text(expected) in normalize_text(answer)


def extract_numbers(answer):
    """
    Extract numeric values from an answer.

    Supports:
      1,000
      1.659
      -18.41
      15.55%
      Unicode minus and dash characters
    """

    answer = unicodedata.normalize("NFKC", str(answer))

    # Normalise Unicode dash/minus characters to ASCII minus.
    for dash in (
        "\u2212",
        "\u2010",
        "\u2011",
        "\u2012",
        "\u2013",
        "\u2014",
    ):
        answer = answer.replace(dash, "-")

    # Remove spaces between a sign and its number.
    answer = re.sub(r"(?<=[+-])\s+(?=\d)", "", answer)

    pattern = r"(?<![A-Za-z])[-+]?\d[\d,\s]*(?:\.\d+)?"

    values = []

    for match in re.findall(pattern, answer):
        try:
            normalized = re.sub(r"[\s,]", "", match)
            values.append(float(normalized))
        except ValueError:
            continue

    return values


def contains_number(answer, expected, tolerance=None):
    """
    Check whether an expected number appears in the answer.

    Integer values are checked with a small tolerance.
    Decimal values allow normal display rounding.
    """

    if expected is None:
        return False

    if tolerance is None:
        if isinstance(expected, float) and not expected.is_integer():
            tolerance = 0.0051
        else:
            tolerance = 1e-9

    numbers = extract_numbers(answer)

    return any(
        math.isclose(
            actual,
            float(expected),
            rel_tol=1e-9,
            abs_tol=tolerance,
        )
        for actual in numbers
    )


def add_check(checks, name, passed, detail=""):
    checks.append({
        "name": name,
        "passed": bool(passed),
        "detail": detail,
    })


def check_text(checks, answer, name, expected):
    add_check(
        checks,
        name,
        contains_text(answer, expected),
        f"Expected text: {expected}",
    )


def check_number(
    checks,
    answer,
    name,
    expected,
    tolerance=None,
):
    add_check(
        checks,
        name,
        contains_number(answer, expected, tolerance),
        f"Expected number: {expected}",
    )


def check_categories(checks, answer, categories):
    for category in categories:
        check_text(
            checks,
            answer,
            f"Mentions category: {category}",
            category,
        )


# --------------------------------------------------
# EXPECTED RESULTS
# --------------------------------------------------

def calculate_expected_results(df):

    results = {}

    for question_id, function in EXPECTED_FUNCTIONS.items():

        if question_id == "Q02":
            expected = function(df, DATASET_CONFIG)
        else:
            expected = function(df)

        results[question_id] = expected

    return results


# --------------------------------------------------
# QUESTION EVALUATORS
# --------------------------------------------------

def evaluate_answer(question_id, answer, expected):

    checks = []

    # --------------------------------------------------
    # Q01 - DATASET OVERVIEW
    # --------------------------------------------------

    if question_id == "Q01":

        check_number(
            checks,
            answer,
            "Correct row count",
            expected["row_count"],
        )

        check_number(
            checks,
            answer,
            "Correct column count",
            expected["column_count"],
        )

        no_duplicates_phrases = (
            "no duplicate rows",
            "no duplicates",
            "zero duplicate rows",
        )

        if (
            expected["duplicate_rows"] == 0
            and any(
                phrase in normalize_text(answer)
                for phrase in no_duplicates_phrases
            )
        ):
            add_check(
                checks,
                "Correct duplicate row count",
                True,
                "The answer explicitly states there are no duplicates.",
            )
        else:
            check_number(
                checks,
                answer,
                "Correct duplicate row count",
                expected["duplicate_rows"],
            )

    # --------------------------------------------------
    # Q02 - AVAILABLE AND EXCLUDED COLUMNS
    # --------------------------------------------------

    elif question_id == "Q02":

        check_categories(
            checks,
            answer,
            expected["available_columns"],
        )

        check_categories(
            checks,
            answer,
            expected["excluded_columns"],
        )

    # --------------------------------------------------
    # Q03 - DATE RANGE
    # --------------------------------------------------

    elif question_id == "Q03":

        # Accept both YYYY-MM-DD and YYYYMMDD formats.
        answer_digits = re.sub(r"\D", "", str(answer))

        expected_min = re.sub(
            r"\D",
            "",
            str(expected["min_date"]),
        )

        expected_max = re.sub(
            r"\D",
            "",
            str(expected["max_date"]),
        )

        add_check(
            checks,
            "Correct earliest date",
            expected_min in answer_digits,
            f"Expected date: {expected['min_date']}",
        )

        add_check(
            checks,
            "Correct latest date",
            expected_max in answer_digits,
            f"Expected date: {expected['max_date']}",
        )

    # --------------------------------------------------
    # Q04 - QUANTITY STATISTICS
    # --------------------------------------------------

    elif question_id == "Q04":

        for field in ("min", "max", "mean", "median"):

            check_number(
                checks,
                answer,
                f"Correct {field}",
                expected[field],
            )

    # --------------------------------------------------
    # Q05 - ORDER STATUS VALUES AND COUNTS
    # --------------------------------------------------

    elif question_id == "Q05":

        actual_unique_count = sum(
            1
            for category in expected["category_counts"]
            if contains_text(answer, category)
        )

        add_check(
            checks,
            "Correct unique status count",
            actual_unique_count == expected["unique_count"],
            f"Expected {expected['unique_count']}, found {actual_unique_count}",
        )

        for category, count in expected["category_counts"].items():

            check_text(
                checks,
                answer,
                f"Mentions status {category}",
                category,
            )

            check_number(
                checks,
                answer,
                f"Correct count for {category}",
                count,
            )

    # --------------------------------------------------
    # Q06 - PAYMENT METHODS
    # --------------------------------------------------

    elif question_id == "Q06":

        count_is_correct = (
            contains_number(answer, expected["unique_count"])
            or (
                expected["unique_count"] == 6
                and bool(
                    re.search(
                        r"\bsix\b",
                        normalize_text(answer),
                    )
                )
            )
        )

        add_check(
            checks,
            "Correct unique payment method count",
            count_is_correct,
            f"Expected count: {expected['unique_count']}",
        )

        check_categories(
            checks,
            answer,
            expected["category_counts"].keys(),
        )

    # --------------------------------------------------
    # Q07 - PAYMENT METHOD DISTRIBUTION
    # --------------------------------------------------

    elif question_id == "Q07":

        check_number(
            checks,
            answer,
            "Correct total transaction count",
            expected["total_rows"],
        )

        for category, values in expected["distribution"].items():

            check_text(
                checks,
                answer,
                f"Mentions {category}",
                category,
            )

            check_number(
                checks,
                answer,
                f"Correct count for {category}",
                values["count"],
            )

            check_number(
                checks,
                answer,
                f"Correct percentage for {category}",
                values["percentage"],
            )

    # --------------------------------------------------
    # Q08 - CANCELLED PAYMENT METHODS
    # --------------------------------------------------

    elif question_id == "Q08":

        check_number(
            checks,
            answer,
            "Correct cancelled transaction total",
            expected["total_rows"],
        )

        for category, values in expected["distribution"].items():

            check_text(
                checks,
                answer,
                f"Mentions {category}",
                category,
            )

            check_number(
                checks,
                answer,
                f"Correct count for {category}",
                values["count"],
            )

            check_number(
                checks,
                answer,
                f"Correct percentage for {category}",
                values["percentage"],
            )

    # --------------------------------------------------
    # Q09 - TRANSACTION COUNTS BY STATUS
    # --------------------------------------------------

    elif question_id == "Q09":

        for category, count in expected["category_counts"].items():

            check_text(
                checks,
                answer,
                f"Mentions status {category}",
                category,
            )

            check_number(
                checks,
                answer,
                f"Correct count for {category}",
                count,
            )

    # --------------------------------------------------
    # Q10 - TOP FIVE COUNTRIES
    # --------------------------------------------------

    elif question_id == "Q10":

        for rank, item in enumerate(
            expected["top_countries"],
            start=1,
        ):

            country = item["country"]

            check_text(
                checks,
                answer,
                f"Mentions top-five country {country}",
                country,
            )

            check_number(
                checks,
                answer,
                f"Correct transaction count for {country}",
                item["count"],
            )

            check_number(
                checks,
                answer,
                f"Correct percentage for {country}",
                item["percentage"],
            )

    # --------------------------------------------------
    # Q11 - DELIVERED MEAN BY CURRENCY
    # --------------------------------------------------

    elif question_id == "Q11":

        for currency, value in expected["results_by_currency"].items():

            check_text(
                checks,
                answer,
                f"Mentions currency {currency}",
                currency,
            )

            check_number(
                checks,
                answer,
                f"Correct mean for {currency}",
                value,
            )

    # --------------------------------------------------
    # Q12 - DELIVERED VS CANCELLED
    # --------------------------------------------------

    elif question_id == "Q12":

        check_text(
            checks,
            answer,
            "Identifies Delivered group",
            "Delivered",
        )

        check_text(
            checks,
            answer,
            "Identifies Cancelled group",
            "Cancelled",
        )

        for currency, values in expected["results_by_currency"].items():

            check_text(
                checks,
                answer,
                f"Mentions currency {currency}",
                currency,
            )

            check_number(
                checks,
                answer,
                f"Correct Delivered mean for {currency}",
                values["delivered_mean"],
            )

            check_number(
                checks,
                answer,
                f"Correct Cancelled mean for {currency}",
                values["cancelled_mean"],
            )

            check_number(
                checks,
                answer,
                f"Correct absolute difference for {currency}",
                values["difference_b_minus_a"],
            )

            check_number(
                checks,
                answer,
                f"Correct percentage difference for {currency}",
                values["percentage_difference_vs_delivered"],
            )

    # --------------------------------------------------
    # Q13 - COMPLETED VS FAILED
    # --------------------------------------------------

    elif question_id == "Q13":

        check_text(
            checks,
            answer,
            "Identifies Completed group",
            "Completed",
        )

        check_text(
            checks,
            answer,
            "Identifies Failed group",
            "Failed",
        )

        check_number(
            checks,
            answer,
            "Correct Completed count",
            expected["group_a_count"],
        )

        check_number(
            checks,
            answer,
            "Correct Failed count",
            expected["group_b_count"],
        )

        negative_direction = bool(
            re.search(
                r"\b(fewer|less|lower)\b",
                normalize_text(answer),
            )
        )

        difference = expected["difference_b_minus_a"]

        difference_is_correct = (
            contains_number(answer, difference)
            or (
                difference < 0
                and negative_direction
                and contains_number(answer, abs(difference))
            )
        )

        add_check(
            checks,
            "Correct absolute difference",
            difference_is_correct,
            f"Expected difference: {difference}",
        )

        percentage_difference = expected["percentage_difference_vs_a"]

        percentage_is_correct = (
            contains_number(
                answer,
                percentage_difference,
                tolerance=0.051,
            )
            or (
                percentage_difference < 0
                and negative_direction
                and contains_number(
                    answer,
                    abs(percentage_difference),
                    tolerance=0.051,
                )
            )
        )

        add_check(
            checks,
            "Correct percentage difference",
            percentage_is_correct,
            f"Expected percentage difference: {percentage_difference}",
        )

    # --------------------------------------------------
    # Q14 - TOP TWO STATUSES AND COMPARISON
    # --------------------------------------------------

    elif question_id == "Q14":

        for item in expected["top_two_statuses"]:

            check_text(
                checks,
                answer,
                f"Mentions status {item['status']}",
                item["status"],
            )

            check_number(
                checks,
                answer,
                f"Correct count for {item['status']}",
                item["count"],
            )

        for currency, values in expected["results_by_currency"].items():

            check_text(
                checks,
                answer,
                f"Mentions currency {currency}",
                currency,
            )

            check_number(
                checks,
                answer,
                f"Correct group A mean for {currency}",
                values["group_a_mean"],
            )

            check_number(
                checks,
                answer,
                f"Correct group B mean for {currency}",
                values["group_b_mean"],
            )

            check_number(
                checks,
                answer,
                f"Correct difference for {currency}",
                values["difference_b_minus_a"],
            )

            check_number(
                checks,
                answer,
                f"Correct percentage difference for {currency}",
                values["percentage_difference_vs_a"],
                tolerance=0.011,
            )

    # --------------------------------------------------
    # Q15 - HIGHEST CANCELLED COUNTRY
    # --------------------------------------------------

    elif question_id == "Q15":

        check_text(
            checks,
            answer,
            "Correct highest country",
            expected["highest_country"],
        )

        check_number(
            checks,
            answer,
            "Correct highest country transaction count",
            expected["transaction_count"],
        )

        check_number(
            checks,
            answer,
            "Correct total cancelled transactions",
            expected["total_cancelled"],
        )

        check_number(
            checks,
            answer,
            "Correct percentage of cancelled transactions",
            expected["percentage"],
            tolerance=0.051,
        )

    return checks


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(question_ids=None, results_file=None):
    results_file = Path(results_file) if results_file else BASELINE_FILE

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}"
        )

    baseline = pd.read_csv(results_file)

    required_columns = {
        "question_id",
        "final_answer",
        "status",
    }

    missing = required_columns - set(baseline.columns)

    if missing:
        raise ValueError(
            f"Baseline is missing columns: {sorted(missing)}"
        )

    baseline = baseline.drop_duplicates(
        subset=["question_id"],
        keep="last",
    )

    # Load the dataset independently of the agent.
    print("Loading dataset for independent expected results...")

    df = load_huggingface_dataset(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
    )

    for column in ECOMMERCE_DATETIME_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"Configured datetime column '{column}' "
                "does not exist in the dataset."
            )

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    print(f"Dataset loaded: {len(df)} rows")

    expected_results = calculate_expected_results(df)

    baseline_rows = {
        str(row["question_id"]): row
        for _, row in baseline.iterrows()
    }

    all_ids = [
        f"Q{i:02d}" for i in range(1, 16)
    ]

    if question_ids:

        unknown = set(question_ids) - set(all_ids)

        if unknown:
            raise ValueError(
                f"Unknown question IDs: {sorted(unknown)}"
            )

        selected_ids = question_ids

    else:
        selected_ids = all_ids

    reports = []

    for question_id in selected_ids:

        row = baseline_rows.get(question_id)

        if row is None:
            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
                "checks": [],
            })
            continue

        answer = row.get("final_answer")

        if not isinstance(answer, str) or not answer.strip():
            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
                "checks": [],
            })
            continue

        if str(row.get("status", "")).casefold() != "completed":
            reports.append({
                "question_id": question_id,
                "status": "NOT EVALUATED",
                "checks": [],
            })
            continue

        checks = evaluate_answer(
            question_id,
            answer,
            expected_results[question_id],
        )

        reports.append({
            "question_id": question_id,
            "status": "EVALUATED",
            "checks": checks,
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

    all_checks = [
        check
        for report in evaluated
        for check in report["checks"]
    ]

    passed = sum(
        check["passed"]
        for check in all_checks
    )

    failed = len(all_checks) - passed

    questions_passed = sum(
        bool(report["checks"])
        and all(
            check["passed"]
            for check in report["checks"]
        )
        for report in evaluated
    )

    print("\n" + "=" * 65)
    print("DATASENSE AI - FINAL ANSWER QUALITY EVALUATION")
    print("=" * 65)

    print("\nOVERALL RESULTS")
    print("-" * 65)

    print(f"Questions evaluated:          {len(evaluated)}")
    print(f"Questions not evaluated:      {len(not_evaluated)}")

    if all_checks:
        print(
            f"Factual/completeness checks:  "
            f"{passed}/{len(all_checks)} "
            f"({passed / len(all_checks) * 100:.2f}%)"
        )
    else:
        print("Factual/completeness checks:  N/A")

    print(f"Questions fully passed:       {questions_passed}/{len(evaluated)}")
    print(f"Failed checks:                {failed}")

    print("\nQUESTION-LEVEL RESULTS")
    print("-" * 65)

    for report in reports:

        question_id = report["question_id"]

        if report["status"] == "NOT EVALUATED":
            print(f"\n{question_id}: NOT EVALUATED")
            continue

        checks = report["checks"]

        question_passed = sum(
            check["passed"] for check in checks
        )

        print(
            f"\n{question_id}: "
            f"{question_passed}/{len(checks)} "
            f"{'PASS' if question_passed == len(checks) else 'FAIL'}"
        )

        for check in checks:

            status = "PASS" if check["passed"] else "FAIL"

            print(f"  [{status}] {check['name']}")

            if not check["passed"]:
                print(f"         {check['detail']}")

    print("\n" + "=" * 65)
    print("ANSWER QUALITY EVALUATION COMPLETE")
    print("=" * 65)

    print(
        "\nNote: This deterministic evaluator checks required facts "
        "and expected numerical values. It does not fully assess "
        "unsupported narrative claims, writing quality, or causal "
        "reasoning."
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate DataSense AI final answer quality."
    )

    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Optional question IDs to evaluate.",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=BASELINE_FILE,
        help="CSV file containing evaluation results.",
    )

    args = parser.parse_args()

    reports = evaluate(
        question_ids=args.question_ids,
        results_file=args.results_file,
    )

    print_report(reports)
