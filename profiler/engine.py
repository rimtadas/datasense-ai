
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality and statistical profile
    for every column in a pandas DataFrame.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        A DataFrame containing profiling results for each column.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty and len(df.columns) == 0:
        raise ValueError("The input dataset has no rows or columns.")

    profile = []

    for column in df.columns:
        series = df[column]
        dtype = str(series.dtype)

        # Identify string/object columns
        is_string = (
            pd.api.types.is_string_dtype(series)
            or series.dtype == "object"
        )

        # Identify blank and missing values
        missing_count = int(series.isna().sum())

        if is_string:
            blank_mask = (
                series.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

            blank_count = int(blank_mask.sum())

            # Exclude whitespace-only values from distinct counts
            clean_series = series.mask(blank_mask)

        else:
            blank_count = 0
            clean_series = series

        # Distinct values (excluding missing and blank values)
        unique_count = int(clean_series.nunique(dropna=True))

        # Identify logical column type
        if pd.api.types.is_bool_dtype(series):
            column_type = "Boolean"

        elif pd.api.types.is_integer_dtype(series):
            column_type = "Integer"

        elif pd.api.types.is_float_dtype(series):
            column_type = "Float"

        elif pd.api.types.is_datetime64_any_dtype(series):
            column_type = "Date/Time"

        elif pd.api.types.is_numeric_dtype(series):
            column_type = "Numeric"

        else:
            column_type = "String/Categorical"

        # Base column profile
        column_profile = {
            "column": column,
            "data_type": dtype,
            "column_type": column_type,
            "row_count": len(df),
            "missing_count": missing_count,
            "blank_count": blank_count,
            "unique_count": unique_count,
        }

        # Numeric statistics
        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            column_profile.update({
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
            })

        # Date statistics
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_profile.update({
                "min_date": series.min(),
                "max_date": series.max(),
            })

        # Most frequent categorical values
        elif column_type == "String/Categorical":
            top_values = (
                clean_series.value_counts()
                .head(5)
            )

            column_profile["top_values"] = top_values.to_dict()

        profile.append(column_profile)

    return pd.DataFrame(profile)
