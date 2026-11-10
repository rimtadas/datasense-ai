
import re
from datetime import datetime

import pandas as pd


# Common, explicit date and datetime formats.
COMMON_DATETIME_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y%m%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


DATE_COLUMN_NAME_PATTERN = re.compile(
    r"(^|[_\s-])(date|datetime|time|timestamp|created|"
    r"updated|modified|signup|purchase|birth|expiry)([_\s-]|$)",
    re.IGNORECASE,
)


def _is_date_candidate(column):
    """Identify columns whose names suggest date/time information."""
    return bool(DATE_COLUMN_NAME_PATTERN.search(str(column)))


def _infer_datetime_format(series):
    """
    Infer a date format from the largest successfully parsed subset.

    Invalid values do not prevent format discovery, but competing
    formats that produce different interpretations are rejected.

    Returns:
        (format, status)
    """

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return None, "no_non_missing_values"

    # Use distinct values to reduce repeated parsing work.
    sample_values = values.drop_duplicates().head(100).tolist()

    matching_formats = []

    for date_format in COMMON_DATETIME_FORMATS:
        parsed_values = {}

        for value in sample_values:
            try:
                parsed_values[value] = datetime.strptime(
                    value,
                    date_format,
                )
            except ValueError:
                continue

        if parsed_values:
            matching_formats.append(
                (date_format, parsed_values)
            )

    if not matching_formats:
        return None, "no_supported_format"

    # Find the formats that successfully parse the largest
    # number of distinct sampled values.
    max_matches = max(
        len(parsed_values)
        for _, parsed_values in matching_formats
    )

    best_formats = [
        (date_format, parsed_values)
        for date_format, parsed_values in matching_formats
        if len(parsed_values) == max_matches
    ]

    # If equally successful formats produce different interpretations
    # or parse different subsets, the format is ambiguous.
    reference = best_formats[0][1]

    for _, parsed_values in best_formats[1:]:
        if parsed_values != reference:
            return None, "ambiguous_format"

    return best_formats[0][0], "inferred"


def prepare_datetime_columns(df, dataset_config=None):
    """
    Return a copy of the DataFrame with confidently identified
    datetime columns parsed.

    Explicit configuration:

        {
            "semantic_types": {
                "signup_date": "datetime"
            },
            "datetime_formats": {
                "signup_date": "%d-%m-%Y"
            }
        }

    Automatic inference is restricted to columns whose names
    suggest date/time information.

    Returns:
        (prepared_dataframe, parsing_report)
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if dataset_config is None:
        dataset_config = {}

    if not isinstance(dataset_config, dict):
        raise TypeError("dataset_config must be a dictionary.")

    semantic_types = dataset_config.get("semantic_types", {})
    datetime_formats = dataset_config.get("datetime_formats", {})

    if not isinstance(semantic_types, dict):
        raise TypeError("semantic_types must be a dictionary.")

    if not isinstance(datetime_formats, dict):
        raise TypeError("datetime_formats must be a dictionary.")

    for column in datetime_formats:
        if column not in df.columns:
            raise ValueError(
                f"Datetime format references unknown column '{column}'."
            )

    for column, date_format in datetime_formats.items():
        if not isinstance(date_format, str) or not date_format:
            raise ValueError(
                f"Datetime format for '{column}' must be a non-empty string."
            )

    configured_columns = {
        column
        for column, semantic_type in semantic_types.items()
        if semantic_type == "datetime"
    }

    configured_columns.update(datetime_formats.keys())

    candidate_columns = set(configured_columns)

    for column in df.columns:
        if _is_date_candidate(column):
            candidate_columns.add(column)

    prepared_df = df.copy()
    report = {}

    for column in df.columns:
        if column not in candidate_columns:
            continue

        series = prepared_df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            report[column] = {
                "status": "already_datetime",
                "format": None,
                "missing_count": int(series.isna().sum()),
                "invalid_count": 0,
            }
            continue

        # Missing values include actual nulls and empty strings.
        non_missing = (
            series.notna()
            & series.astype(str).str.strip().ne("")
        )

        missing_count = int((~non_missing).sum())

        explicit_format = datetime_formats.get(column)

        if explicit_format is not None:
            date_format = explicit_format
            status = "configured"

        else:
            date_format, inference_status = _infer_datetime_format(
                series
            )

            if date_format is None:
                report[column] = {
                    "status": inference_status,
                    "format": None,
                    "missing_count": missing_count,
                    "invalid_count": None,
                }
                continue

            status = inference_status

        parsed = pd.to_datetime(
            series,
            format=date_format,
            errors="coerce",
        )

        invalid_count = int(
            (non_missing & parsed.isna()).sum()
        )

        # Do not partially convert automatically inferred columns.
        # The user must clarify inconsistent values or configure
        # the intended format explicitly.
        if invalid_count > 0 and explicit_format is None:
            report[column] = {
                "status": "inconsistent_values",
                "format": date_format,
                "missing_count": missing_count,
                "invalid_count": invalid_count,
            }
            continue

        prepared_df[column] = parsed

        report[column] = {
            "status": status,
            "format": date_format,
            "missing_count": missing_count,
            "invalid_count": invalid_count,
        }

    return prepared_df, report
