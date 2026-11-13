
from data_connector.huggingface_loader import (
    load_huggingface_dataset
)

REPO_ID = "davidev07/synthetic-ecommerce-dataset"

df = load_huggingface_dataset(
    repo_id=REPO_ID,
    filename="transactions.csv"
)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())
