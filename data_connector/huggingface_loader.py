
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# DATASET SAFEGUARD CONFIGURATION
# --------------------------------------------------

MAX_FILE_SIZE_MB = 100
MAX_ROWS = 1_000_000
MAX_COLUMNS = 200
MAX_DATASET_MEMORY_MB = 512


# --------------------------------------------------
# LOAD HUGGING FACE DATASET
# --------------------------------------------------

def load_huggingface_dataset(
    repo_id: str,
    filename: str,
) -> pd.DataFrame:
    """
    Download and safely load a CSV or Parquet file
    from a Hugging Face dataset repository.
    """

    if not repo_id or not repo_id.strip():
        raise ValueError("Please provide a Hugging Face dataset repository ID.")

    if not filename or not filename.strip():
        raise ValueError("Please provide a dataset filename.")

    suffix = Path(filename).suffix.lower()

    if suffix not in (".csv", ".parquet", ".pq"):
        raise ValueError(
            "Only CSV and Parquet files are currently supported."
        )

    # Download the selected file.
    file_path = hf_hub_download(
        repo_id=repo_id.strip(),
        filename=filename.strip(),
        repo_type="dataset",
        token=True,
    )

    file_path = Path(file_path)

    # Check downloaded file size.
    file_size_mb = file_path.stat().st_size / (1024 ** 2)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File size is {file_size_mb:.2f} MB. "
            f"The maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    # ----------------------------------------------
    # CSV VALIDATION AND LOADING
    # ----------------------------------------------

    if suffix == ".csv":

        header = pd.read_csv(file_path, nrows=0)
        column_count = len(header.columns)

        if column_count > MAX_COLUMNS:
            raise ValueError(
                f"The dataset contains {column_count:,} columns. "
                f"The maximum allowed is {MAX_COLUMNS:,}."
            )

        df = pd.read_csv(
            file_path,
            nrows=MAX_ROWS + 1,
        )

        if len(df) > MAX_ROWS:
            raise ValueError(
                f"The dataset exceeds the maximum allowed "
                f"of {MAX_ROWS:,} rows."
            )

    # ----------------------------------------------
    # PARQUET VALIDATION AND LOADING
    # ----------------------------------------------

    else:

        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata

        if metadata.num_rows > MAX_ROWS:
            raise ValueError(
                f"The dataset contains {metadata.num_rows:,} rows. "
                f"The maximum allowed is {MAX_ROWS:,}."
            )

        if metadata.num_columns > MAX_COLUMNS:
            raise ValueError(
                f"The dataset contains {metadata.num_columns:,} columns. "
                f"The maximum allowed is {MAX_COLUMNS:,}."
            )

        df = pd.read_parquet(file_path)

    # ----------------------------------------------
    # FINAL VALIDATION
    # ----------------------------------------------

    if df.empty:
        raise ValueError("The selected dataset is empty.")

    memory_mb = (
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    )

    if memory_mb > MAX_DATASET_MEMORY_MB:
        raise ValueError(
            f"The loaded dataset uses {memory_mb:.2f} MB of memory. "
            f"The maximum allowed is {MAX_DATASET_MEMORY_MB} MB."
        )

    return df
