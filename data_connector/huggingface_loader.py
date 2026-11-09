
import pandas as pd
from huggingface_hub import hf_hub_download


def load_huggingface_dataset(
    repo_id: str,
    filename: str
) -> pd.DataFrame:
    """
    Load a CSV or Parquet file from a Hugging Face dataset repository.

    Parameters:
        repo_id: Hugging Face dataset repository ID.
        filename: Name of the CSV or Parquet file.

    Returns:
        A pandas DataFrame.
    """

    supported_formats = (".csv", ".parquet")

    if not filename.lower().endswith(supported_formats):
        raise ValueError(
            "Only CSV and Parquet files are currently supported."
        )

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)

    if df.empty:
        raise ValueError("The selected dataset is empty.")

    return df
