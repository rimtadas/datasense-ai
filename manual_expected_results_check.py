
from data_connector.huggingface_loader import (
    load_huggingface_dataset
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
# 1. LOAD DATASET
# --------------------------------------------------

REPO_ID = "davidev07/synthetic-ecommerce-dataset"
DATASET_FILE = "transactions.csv"

print("Loading dataset...")

df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename=DATASET_FILE,
)

print(f"Dataset loaded: {len(df)} rows")


# --------------------------------------------------
# 2. Q01: DATASET OVERVIEW
# --------------------------------------------------

print("\nQ01: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q01 = calculate_q01_expected(df)

for key, value in q01.items():
    print(f"{key}: {value}")

assert q01["row_count"] == 1000
assert q01["column_count"] == 18
assert q01["duplicate_rows"] == 0

print("Q01 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# 3. Q02: AVAILABLE AND EXCLUDED COLUMNS
# --------------------------------------------------

dataset_config = {
    "exclude_from_context": ["shipping_address"]
}

print("\nQ02: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q02 = calculate_q02_expected(df, dataset_config)

print("Available columns:", q02["available_columns"])
print("Available column count:", q02["available_column_count"])
print("Excluded columns:", q02["excluded_columns"])
print("Excluded column count:", q02["excluded_column_count"])

assert q02["available_column_count"] == 17
assert q02["excluded_columns"] == ["shipping_address"]
assert q02["excluded_column_count"] == 1

print("Q02 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# 4. Q03: PURCHASE DATE RANGE
# --------------------------------------------------

print("\nQ03: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q03 = calculate_q03_expected(df)

for key, value in q03.items():
    print(f"{key}: {value}")

assert q03["valid_date_count"] == 1000
assert q03["min_date"].startswith("2025-09-23")
assert q03["max_date"].startswith("2026-09-22")

print("Q03 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q04: QUANTITY SUMMARY
# --------------------------------------------------

print("\nQ04: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q04 = calculate_q04_expected(df)

for key, value in q04.items():
    print(f"{key}: {value}")

assert q04["min"] == 1
assert q04["max"] == 5
assert q04["mean"] == 1.659
assert q04["median"] == 1

print("Q04 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q05: ORDER STATUS DISTRIBUTION
# --------------------------------------------------

print("\nQ05: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q05 = calculate_q05_expected(df)

print("Unique categories:", q05["unique_count"])
print("Category counts:", q05["category_counts"])

assert q05["unique_count"] == 5
assert sum(q05["category_counts"].values()) == 1000

print("Q05 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q06: PAYMENT METHOD CATEGORIES
# --------------------------------------------------

print("\nQ06: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q06 = calculate_q06_expected(df)

print("Unique categories:", q06["unique_count"])
print("Category counts:", q06["category_counts"])

assert q06["unique_count"] == 6
assert sum(q06["category_counts"].values()) == 1000

print("Q06 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q07: PAYMENT METHOD DISTRIBUTION
# --------------------------------------------------

print("\nQ07: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q07 = calculate_q07_expected(df)

for category, result in q07["distribution"].items():
    print(
        f"{category}: {result['count']} "
        f"({result['percentage']}%)"
    )

assert q07["total_rows"] == 1000
assert q07["unique_count"] == 6
assert sum(
    item["count"] for item in q07["distribution"].values()
) == 1000
assert round(
    sum(item["percentage"] for item in q07["distribution"].values()),
    2
) == 100.0

print("Q07 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q08: CANCELLED PAYMENT METHOD DISTRIBUTION
# --------------------------------------------------

print("\nQ08: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q08 = calculate_q08_expected(df)

print("Cancelled transactions:", q08["total_rows"])

for category, result in q08["distribution"].items():
    print(
        f"{category}: {result['count']} "
        f"({result['percentage']}%)"
    )

assert q08["total_rows"] == 283
assert sum(
    item["count"] for item in q08["distribution"].values()
) == 283
assert round(
    sum(item["percentage"] for item in q08["distribution"].values()),
    2
) == 100.0

print("Q08 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q09: ORDER STATUS COUNTS
# --------------------------------------------------

print("\nQ09: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q09 = calculate_q09_expected(df)

print("Category counts:", q09["category_counts"])

assert sum(q09["category_counts"].values()) == 1000
assert q09["category_counts"]["Delivered"] == 330
assert q09["category_counts"]["Cancelled"] == 283

print("Q09 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q10: TOP FIVE COUNTRIES
# --------------------------------------------------

print("\nQ10: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q10 = calculate_q10_expected(df)

for item in q10["top_countries"]:
    print(
        f"{item['country']}: {item['count']} "
        f"({item['percentage']}%)"
    )

assert len(q10["top_countries"]) == 5
assert q10["top_countries"][0]["country"] == "Italy"
assert q10["top_countries"][0]["count"] == 146

print("Q10 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q11: DELIVERED AVERAGE AMOUNT BY CURRENCY
# --------------------------------------------------

print("\nQ11: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q11 = calculate_q11_expected(df)

for currency, mean in q11["results_by_currency"].items():
    print(f"{currency}: {mean:.4f}")

assert len(q11["results_by_currency"]) == 6
assert abs(q11["results_by_currency"]["AUD"] - 954.3006896551725) < 0.000001

print("Q11 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q12: DELIVERED VS CANCELLED COMPARISON
# --------------------------------------------------

print("\nQ12: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q12 = calculate_q12_expected(df)

for currency, result in q12["results_by_currency"].items():
    print(f"\nCurrency: {currency}")
    for key, value in result.items():
        print(f"  {key}: {value}")

assert len(q12["results_by_currency"]) == 6
assert abs(
    q12["results_by_currency"]["AUD"]["difference_b_minus_a"]
    - 42.6426
) < 0.001

print("\nQ12 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q14: TOP TWO STATUSES AND COMPARISON
# --------------------------------------------------

print("\nQ14: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q14 = calculate_q14_expected(df)

print("Top two statuses:", q14["top_two_statuses"])
print("Comparison:", q14["comparison"])

for currency, result in q14["results_by_currency"].items():
    print(f"\nCurrency: {currency}")
    for key, value in result.items():
        print(f"  {key}: {value}")

assert q14["top_two_statuses"][0]["status"] == "Delivered"
assert q14["top_two_statuses"][1]["status"] == "Cancelled"
assert len(q14["results_by_currency"]) == 6

print("\nQ14 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# Q15: HIGHEST COUNTRY BY CANCELLED TRANSACTIONS
# --------------------------------------------------

print("\nQ15: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q15 = calculate_q15_expected(df)

for key, value in q15.items():
    print(f"{key}: {value}")

assert q15["total_cancelled"] == 283
assert q15["highest_country"] == "United States"
assert q15["transaction_count"] == 44
assert q15["percentage"] == 15.55

print("\nQ15 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# 3. Q13: PAYMENT STATUS COMPARISON
# --------------------------------------------------

print("\nQ13: INDEPENDENT EXPECTED RESULT")
print("--------------------------------")

q13 = calculate_q13_expected(df)

for key, value in q13.items():
    print(f"{key}: {value}")

assert q13["group_a_count"] == 750
assert q13["group_b_count"] == 103
assert q13["difference_b_minus_a"] == -647

print("Q13 EXPECTED RESULT: PASSED")


# --------------------------------------------------
# 4. SUMMARY
# --------------------------------------------------

print("\nALL EXPECTED RESULT TESTS PASSED")
