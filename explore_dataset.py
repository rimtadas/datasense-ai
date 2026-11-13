from data_connector.huggingface_loader import (
    load_huggingface_dataset
)

import pandas as pd


REPO_ID = "davidev07/synthetic-ecommerce-dataset"

df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename="transactions.csv"
)

df["purchase_date"] = pd.to_datetime(
    df["purchase_date"],
    errors="coerce"
)

print("\nCATEGORICAL VALUE DISTRIBUTIONS")
print("--------------------------------")

categorical_columns = [
    "currency",
    "country",
    "payment_method",
    "payment_status",
    "order_status",
    "device_type"
]

for column in categorical_columns:
    print(f"\n{column.upper()}")
    print(df[column].value_counts(dropna=False).to_string())


print("\nNUMERIC COLUMN SUMMARY")
print("----------------------")

numeric_columns = [
    "quantity",
    "unit_price",
    "subtotal",
    "discount_percentage",
    "discount_amount",
    "total_amount"
]

print(df[numeric_columns].describe().to_string())


print("\nDATE RANGE")
print("----------")
print("Minimum:", df["purchase_date"].min())
print("Maximum:", df["purchase_date"].max())


print("\nPAYMENT STATUS VS ORDER STATUS")
print("--------------------------------")

print(
    pd.crosstab(
        df["payment_status"],
        df["order_status"]
    ).to_string()
)