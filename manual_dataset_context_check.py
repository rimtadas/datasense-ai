
import json
import pandas as pd

from data_connector.huggingface_loader import (
    load_huggingface_dataset
)

from dataset_context import build_dataset_context


REPO_ID = "davidev07/synthetic-ecommerce-dataset"


# --------------------------------------------------
# TEST 1: E-COMMERCE DATASET
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 1: E-COMMERCE DATASET CONTEXT")
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
    },
    "exclude_from_context": [
        "shipping_address"
    ],
}

context = build_dataset_context(
    df=df,
    dataset_name="E-commerce Transactions",
    dataset_config=ecommerce_config,
)

print("\nDATASET OVERVIEW")
print(json.dumps(
    context["dataset_overview"],
    indent=4
))

print("\nDATASET DESCRIPTION")
print(context["dataset_description"])

print("\nCURRENCY COLUMN")
print(context["currency_column"])

print("\nCOLUMN CONTEXT")

for column in context["columns"]:
    print(
        column["column_name"],
        "| Type:", column["data_type"],
        "| Semantic:", column["semantic_type"],
        "| Role:", column["analytical_role"],
        "| Description:", column["description"],
    )

print("\nEXCLUDED COLUMNS")
print(context["excluded_columns"])


# Confirm the context can be serialized as JSON.
json.dumps(context)

print("\nE-commerce context generated successfully.")


# --------------------------------------------------
# TEST 2: HR DATASET
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 2: HR DATASET CONTEXT")
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
    "dataset_description": "Synthetic employee dataset.",
    "semantic_types": {
        "employee_id": "identifier",
        "salary": "monetary",
    },
    "currency_column": "currency_code",
    "column_descriptions": {
        "salary": "Employee salary.",
        "department": "Employee department.",
    },
}

hr_context = build_dataset_context(
    df=hr_df,
    dataset_name="HR Employees",
    dataset_config=hr_config,
)

print("\nDATASET OVERVIEW")
print(json.dumps(
    hr_context["dataset_overview"],
    indent=4
))

print("\nCOLUMN CONTEXT")

for column in hr_context["columns"]:
    print(
        column["column_name"],
        "| Type:", column["data_type"],
        "| Semantic:", column["semantic_type"],
        "| Role:", column["analytical_role"],
    )

json.dumps(hr_context)

print("\nHR context generated successfully.")

print("\n" + "=" * 60)
print("ALL DATASET CONTEXT TESTS COMPLETED")
print("=" * 60)
