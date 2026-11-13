
import pandas as pd

from data_connector.huggingface_loader import (
    load_huggingface_dataset
)

from analytics.tools import (
    get_dataset_overview,
    get_column_summary,
    group_and_aggregate
)


REPO_ID = "davidev07/synthetic-ecommerce-dataset"


# --------------------------------------------------
# DATASET 1: E-COMMERCE
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET 1: E-COMMERCE")
print("=" * 60)

df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename="transactions.csv"
)

df["purchase_date"] = pd.to_datetime(
    df["purchase_date"],
    errors="coerce"
)

ecommerce_config = {
    "semantic_types": {
        "unit_price": "monetary",
        "subtotal": "monetary",
        "discount_amount": "monetary",
        "total_amount": "monetary",
        "transaction_id": "identifier",
        "user_id": "identifier",
        "product_id": "identifier",
        "purchase_date": "datetime"
    },
    "currency_column": "currency"
}


# TEST 1: Dataset overview

print("\nTEST 1: DATASET OVERVIEW")
print("------------------------")

overview = get_dataset_overview(df)

for key, value in overview.items():
    print(f"{key}: {value}")


# TEST 2: Numeric column summary

print("\nTEST 2: TOTAL AMOUNT SUMMARY BY CURRENCY")
print("----------------------------------------")

summary = get_column_summary(
    df,
    "total_amount",
    dataset_config=ecommerce_config
)

for key, value in summary.items():
    print(f"{key}: {value}")


# TEST 3: Categorical column summary

print("\nTEST 3: PAYMENT STATUS SUMMARY")
print("--------------------------------")

summary = get_column_summary(
    df,
    "payment_status",
    dataset_config=ecommerce_config
)

for key, value in summary.items():
    print(f"{key}: {value}")


# TEST 4: Date column summary

print("\nTEST 4: PURCHASE DATE SUMMARY")
print("-----------------------------")

summary = get_column_summary(
    df,
    "purchase_date",
    dataset_config=ecommerce_config
)

for key, value in summary.items():
    print(f"{key}: {value}")


# TEST 5: Transaction count by country

print("\nTEST 5: TRANSACTION COUNT BY COUNTRY")
print("------------------------------------")

result = group_and_aggregate(
    df,
    group_by="country",
    aggregation="count"
)

for row in result["results"]:
    print(row)


# TEST 6: Average amount by currency

print("\nTEST 6: AVERAGE AMOUNT BY CURRENCY")
print("----------------------------------")

result = group_and_aggregate(
    df,
    group_by="currency",
    metric="total_amount",
    aggregation="mean",
    dataset_config=ecommerce_config
)

for row in result["results"]:
    print(row)


# TEST 7: Filtering and sorting

print("\nTEST 7: FILTERED TRANSACTIONS")
print("-----------------------------")

result = group_and_aggregate(
    df,
    group_by="country",
    aggregation="count",
    filters={
        "payment_status": "Completed"
    },
    sort_by="result",
    ascending=False,
    limit=5
)

for row in result["results"]:
    print(row)


# TEST 8: Currency safeguard

print("\nTEST 8: CURRENCY SAFEGUARD")
print("--------------------------")

try:
    group_and_aggregate(
        df,
        metric="total_amount",
        aggregation="mean",
        dataset_config=ecommerce_config
    )

except ValueError as error:
    print("Expected safeguard:", error)


# --------------------------------------------------
# DATASET 2: HR
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET 2: HR (SYNTHETIC TEST DATA)")
print("=" * 60)

hr_df = pd.DataFrame({
    "employee_id": [101, 102, 103, 104, 105, 106],
    "department": [
        "Analytics",
        "Engineering",
        "Analytics",
        "HR",
        "Engineering",
        "HR"
    ],
    "salary": [
        12000,
        15000,
        14000,
        9000,
        16000,
        10000
    ],
    "currency_code": [
        "AED",
        "AED",
        "AED",
        "EUR",
        "AED",
        "EUR"
    ],
    "years_experience": [5, 8, 6, 3, 10, 4]
})

hr_config = {
    "semantic_types": {
        "employee_id": "identifier",
        "salary": "monetary"
    },
    "currency_column": "currency_code"
}


# TEST 9: HR dataset overview

print("\nTEST 9: HR DATASET OVERVIEW")
print("---------------------------")

overview = get_dataset_overview(hr_df)

for key, value in overview.items():
    print(f"{key}: {value}")


# TEST 10: Employee count by department

print("\nTEST 10: EMPLOYEES BY DEPARTMENT")
print("--------------------------------")

result = group_and_aggregate(
    hr_df,
    group_by="department",
    aggregation="count",
    sort_by="result",
    ascending=False
)

for row in result["results"]:
    print(row)


# TEST 11: Average salary by currency

print("\nTEST 11: AVERAGE SALARY BY CURRENCY")
print("-----------------------------------")

result = group_and_aggregate(
    hr_df,
    group_by="currency_code",
    metric="salary",
    aggregation="mean",
    dataset_config=hr_config
)

for row in result["results"]:
    print(row)


# TEST 12: Average salary in AED

print("\nTEST 12: AVERAGE SALARY IN AED")
print("------------------------------")

result = group_and_aggregate(
    hr_df,
    metric="salary",
    aggregation="mean",
    filters={
        "currency_code": "AED"
    },
    dataset_config=hr_config
)

for row in result["results"]:
    print(row)


print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
