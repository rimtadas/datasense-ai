
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from huggingface_hub import hf_hub_download
from exceptions import UserFacingError

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
    return_metadata: bool = False,
    revision: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Download and safely load a CSV or Parquet file
    from a Hugging Face dataset repository.

    When return_metadata=True, also return the verified
    repository ID, filename, commit revision and file size.
    """

    if not repo_id or not repo_id.strip():
        raise UserFacingError("Please provide a Hugging Face dataset repository ID.")

    if not filename or not filename.strip():
        raise UserFacingError("Please provide a dataset filename.")

    suffix = Path(filename).suffix.lower()

    if suffix not in (".csv", ".parquet", ".pq"):
        raise UserFacingError(
            "Only CSV and Parquet files are currently supported."
        )


    # ----------------------------------------------
    # REMOTE FILE SIZE PREFLIGHT
    # ----------------------------------------------

    preflight = hf_hub_download(
        repo_id=repo_id.strip(),
        filename=filename.strip(),
        repo_type="dataset",
        token=True,
        dry_run=True,
    )

    # Fail safely if remote file metadata is unavailable.
    if (
        preflight.file_size is None
        or preflight.commit_hash is None
    ):
        raise UserFacingError(
            "Unable to verify the dataset file size or revision. "
            "Please try again."
        )

    file_size_mb = (
        preflight.file_size / (1024 ** 2)
    )

    if preflight.file_size > MAX_FILE_SIZE_MB * (1024 ** 2):
        raise UserFacingError(
            f"File size is {file_size_mb:.2f} MB. "
            f"The maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    # ----------------------------------------------
    # DOWNLOAD VERIFIED FILE REVISION
    # ----------------------------------------------

    file_path = hf_hub_download(
        repo_id=repo_id.strip(),
        filename=filename.strip(),
        repo_type="dataset",
        revision=preflight.commit_hash,
        token=True,
    )

    file_path = Path(file_path)

    # Verify the downloaded file size again.
    actual_file_size = file_path.stat().st_size

    if actual_file_size > MAX_FILE_SIZE_MB * (1024 ** 2):
        raise UserFacingError(
            f"Downloaded file exceeds the maximum allowed "
            f"size of {MAX_FILE_SIZE_MB} MB."
        )

    # ----------------------------------------------
    # CSV VALIDATION AND LOADING
    # ----------------------------------------------

    if suffix == ".csv":

        header = pd.read_csv(file_path, nrows=0)
        column_count = len(header.columns)

        if column_count > MAX_COLUMNS:
            raise UserFacingError(
                f"The dataset contains {column_count:,} columns. "
                f"The maximum allowed is {MAX_COLUMNS:,}."
            )

        df = pd.read_csv(
            file_path,
            nrows=MAX_ROWS + 1,
        )

        if len(df) > MAX_ROWS:
            raise UserFacingError(
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
            raise UserFacingError(
                f"The dataset contains {metadata.num_rows:,} rows. "
                f"The maximum allowed is {MAX_ROWS:,}."
            )

        if metadata.num_columns > MAX_COLUMNS:
            raise UserFacingError(
                f"The dataset contains {metadata.num_columns:,} columns. "
                f"The maximum allowed is {MAX_COLUMNS:,}."
            )

        df = pd.read_parquet(file_path)

    # ----------------------------------------------
    # FINAL VALIDATION
    # ----------------------------------------------

    if df.empty:
        raise UserFacingError("The selected dataset is empty.")

    memory_mb = (
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    )

    if memory_mb > MAX_DATASET_MEMORY_MB:
        raise UserFacingError(
            f"The loaded dataset uses {memory_mb:.2f} MB of memory. "
            f"The maximum allowed is {MAX_DATASET_MEMORY_MB} MB."
        )

    if return_metadata:
        metadata = {
            "repo_id": repo_id.strip(),
            "filename": filename.strip(),
            "revision": preflight.commit_hash,
            "file_size_bytes": preflight.file_size,
        }

        return df, metadata

    return df
