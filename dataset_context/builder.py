
import pandas as pd


SUPPORTED_SEMANTIC_TYPES = {
    "identifier",
    "monetary",
    "datetime",
}


def _validate_dataset_config(df, dataset_config):
    """Validate optional dataset metadata."""

    if dataset_config is None:
        dataset_config = {}

    if not isinstance(dataset_config, dict):
        raise TypeError("dataset_config must be a dictionary.")

    semantic_types = dataset_config.get("semantic_types", {})
    column_descriptions = dataset_config.get(
        "column_descriptions", {}
    )
    excluded_columns = dataset_config.get(
        "exclude_from_context", []
    )

    if not isinstance(semantic_types, dict):
        raise TypeError("semantic_types must be a dictionary.")

    if not isinstance(column_descriptions, dict):
        raise TypeError("column_descriptions must be a dictionary.")

    if not isinstance(excluded_columns, (list, tuple, set)):
        raise TypeError(
            "exclude_from_context must be a list, tuple or set."
        )

    for column, semantic_type in semantic_types.items():
        if column not in df.columns:
            raise ValueError(
                f"Configured column '{column}' does not exist."
            )

        if semantic_type not in SUPPORTED_SEMANTIC_TYPES:
            raise ValueError(
                f"Unsupported semantic type '{semantic_type}' "
                f"for column '{column}'."
            )

    for column in column_descriptions:
        if column not in df.columns:
            raise ValueError(
                f"Description references unknown column '{column}'."
            )

    for column in excluded_columns:
        if column not in df.columns:
            raise ValueError(
                f"Excluded column '{column}' does not exist."
            )

    currency_column = dataset_config.get("currency_column")

    if currency_column is not None:
        if currency_column not in df.columns:
            raise ValueError(
                f"Currency column '{currency_column}' does not exist."
            )

    return {
        "semantic_types": semantic_types,
        "column_descriptions": column_descriptions,
        "excluded_columns": set(excluded_columns),
        "currency_column": currency_column,
        "dataset_description": dataset_config.get(
            "dataset_description"
        ),
    }


def _infer_data_type(series):
    """Infer a general data type from the Pandas dtype."""

    if pd.api.types.is_bool_dtype(series):
        return "Boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"

    if pd.api.types.is_integer_dtype(series):
        return "Integer"

    if pd.api.types.is_float_dtype(series):
        return "Float"

    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"

    return "String"


def _infer_analytical_role(series, data_type, semantic_type):
    """Suggest an analytical role based on metadata and data characteristics."""

    if semantic_type is not None:
        return {
            "role": semantic_type,
            "basis": "configured",
        }

    if data_type == "Date/Time":
        return {
            "role": "time_dimension",
            "basis": "inferred_from_dtype",
        }

    if data_type in {"Integer", "Float", "Numeric"}:
        return {
            "role": "measure_candidate",
            "basis": "inferred_from_dtype",
        }

    if data_type == "Boolean":
        return {
            "role": "dimension_candidate",
            "basis": "inferred_from_dtype",
        }

    non_missing_count = int(series.notna().sum())
    unique_count = int(series.nunique(dropna=True))

    if non_missing_count == 0:
        role = "unknown"
    elif unique_count <= min(50, max(10, non_missing_count * 0.5)):
        role = "dimension_candidate"
    else:
        role = "text_or_identifier_candidate"

    return {
        "role": role,
        "basis": "inferred_from_cardinality",
    }


def _safe_value(value):
    """Convert Pandas/NumPy values into serializable Python values."""

    if pd.isna(value):
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    return value


def _build_column_context(
    df,
    column,
    config,
    include_sample_values=False,
    category_limit=30,
):
    """Build context for one column."""

    series = df[column]

    data_type = _infer_data_type(series)
    semantic_type = config["semantic_types"].get(column)

    missing_count = int(series.isna().sum())
    row_count = len(df)
    non_missing_count = row_count - missing_count
    unique_count = int(series.nunique(dropna=True))

    missing_percentage = (
        round((missing_count / row_count) * 100, 2)
        if row_count > 0
        else 0
    )

    unique_percentage = (
        round((unique_count / non_missing_count) * 100, 2)
        if non_missing_count > 0
        else 0
    )

    analytical_role = _infer_analytical_role(
        series,
        data_type,
        semantic_type,
    )

    column_context = {
        "column_name": column,
        "pandas_dtype": str(series.dtype),
        "data_type": data_type,
        "semantic_type": semantic_type,
        "description": config["column_descriptions"].get(column),
        "analytical_role": analytical_role["role"],
        "role_basis": analytical_role["basis"],
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "unique_count": unique_count,
        "unique_percentage": unique_percentage,
    }

    if semantic_type == "datetime":
        datetime_ready = pd.api.types.is_datetime64_any_dtype(series)

        column_context["datetime_ready"] = datetime_ready

        if not datetime_ready:
            column_context["datetime_note"] = (
                "This column is configured as datetime, but its values "
                "have not been converted to a Pandas datetime dtype. "
                "Check datetime_parsing metadata before performing "
                "date-based analysis."
            )

    if (
        semantic_type == "monetary"
        and (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        )
    ):
        raise ValueError(
            f"Monetary column '{column}' must have a numeric dtype."
        )



    if data_type == "Date/Time":
        column_context["date_range"] = {
            "min": _safe_value(series.min()),
            "max": _safe_value(series.max()),
        }

    if include_sample_values and data_type == "String":
        sample_values = (
            series.dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        column_context["sample_values"] = sample_values

    # Discover actual values for low-cardinality dimensions.
    if (
        analytical_role["role"] == "dimension_candidate"
        and data_type in {"String", "Boolean"}
    ):
        value_counts = series.value_counts(dropna=True)

        column_context["category_values"] = [
            {
                "value": _safe_value(value),
                "count": int(count),
            }
            for value, count in value_counts.head(category_limit).items()
        ]

        column_context["category_values_truncated"] = (
            len(value_counts) > category_limit
        )

    return column_context


def build_dataset_context(
    df,
    dataset_name=None,
    dataset_config=None,
    include_sample_values=False,
    category_limit=30,
):
    """
    Build a structured, dataset-agnostic context dictionary.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    if not df.columns.is_unique:
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Rename them before building context."
        )

    if (
        not isinstance(category_limit, int)
        or isinstance(category_limit, bool)
        or category_limit < 1
    ):
        raise ValueError("category_limit must be a positive integer.")

    config = _validate_dataset_config(df, dataset_config)

    columns = []
    excluded_columns = []

    for column in df.columns:
        if column in config["excluded_columns"]:
            excluded_columns.append(column)
            continue

        columns.append(
            _build_column_context(
                df=df,
                column=column,
                config=config,
                include_sample_values=include_sample_values,
                category_limit=category_limit,
            )
        )

    return {
        "dataset_name": dataset_name,
        "dataset_description": config["dataset_description"],
        "dataset_overview": {
            "row_count": len(df),
            "column_count": len(df.columns),
            "context_column_count": len(columns),
            "excluded_column_count": len(excluded_columns),
            "duplicate_row_count": int(df.duplicated().sum()),
        },
        "currency_column": config["currency_column"],
        "columns": columns,
        "excluded_columns": excluded_columns,
    }
