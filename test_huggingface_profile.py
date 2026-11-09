import pandas as pd

from data_connector.huggingface_loader import (
    load_huggingface_dataset
)
from profiler import profile_dataset


REPO_ID = "davidev07/synthetic-ecommerce-dataset"

# Load transactions from Hugging Face
df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename="transactions.csv"
)

# Convert purchase_date into a proper datetime column
df["purchase_date"] = pd.to_datetime(
    df["purchase_date"],
    errors="coerce"
)

# Run the existing profiling engine
profile = profile_dataset(df)

# Display all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nDATASET OVERVIEW")
print("----------------")
print("Dataset:", REPO_ID)
print("Table: transactions")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nCOLUMN NAMES")
print("------------")
print(df.columns.tolist())

print("\nCOLUMN PROFILING RESULTS")
print("-----------------------")
print(profile.to_string(index=False))