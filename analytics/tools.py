
import difflib

import pandas as pd


SUPPORTED_AGGREGATIONS = {
    "count",
    "sum",
    "mean",
    "median",
    "min",
    "max",
}

SUPPORTED_FILTER_OPERATORS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "in", "not_in", "contains", "is_null", "not_null",
}

SUPPORTED_DATE_GRANULARITIES = {
    "day": "D",
    "week": "W-SUN",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}


def _casefold_value(value):
    """Normalize strings for matching without changing source data."""
    return value.casefold() if isinstance(value, str) else value


def _case_insensitive_equality(series, value, negate=False):
    """Match string values case-insensitively."""

    if isinstance(value, str):
        left = series.map(_casefold_value)
        right = value.casefold()
    else:
        left = series
        right = value

    return left != right if negate else left == right


def _case_insensitive_membership(series, values, negate=False):
    """Case-insensitive membership matching."""

    values = list(values)

    if any(isinstance(value, str) for value in values):
        left = series.map(_casefold_value)
        right = [_casefold_value(value) for value in values]
    else:
        left = series
        right = values

    mask = left.isin(right)
    return ~mask if negate else mask


def _validate_categorical_filter_value(
    original_series,
    value,
    column,
):
    """
    Validate requested string values against actual dataset categories.

    Case differences are accepted. Unknown values raise an error
    with close-match suggestions.
    """

    if not isinstance(value, str):
        return

    is_categorical = (
        pd.api.types.is_object_dtype(original_series)
        or pd.api.types.is_string_dtype(original_series)
        or isinstance(original_series.dtype, pd.CategoricalDtype)
    )

    if not is_categorical:
        return

    actual_values = original_series.dropna().unique().tolist()

    string_values = [
        item for item in actual_values
        if isinstance(item, str)
    ]

    if not string_values:
        return

    actual_by_normalized = {
        item.casefold(): item
        for item in string_values
    }

    if value.casefold() in actual_by_normalized:
        return

    suggestions = difflib.get_close_matches(
        value,
        string_values,
        n=3,
        cutoff=0.5,
    )

    message = (
        f"Unknown category '{value}' for column '{column}'. "
    )

    if suggestions:
        message += f"Did you mean: {suggestions}?"
    else:
        message += (
            f"Available categories include: {string_values[:15]}."
        )

    raise ValueError(message)


def _comparison_group_key(value):
    """Create stable keys for case-insensitive comparison values."""

    if pd.isna(value):
        return ("missing",)

    if isinstance(value, str):
        return ("string", value.casefold())

    try:
        hash(value)
        return (type(value).__name__, value)
    except TypeError:
        return (type(value).__name__, repr(value))


def _group_value_key(value):
    """Align grouped results without merging distinct source labels."""

    if pd.isna(value):
        return ("missing",)

    try:
        hash(value)
        return (type(value).__name__, value)
    except TypeError:
        return (type(value).__name__, repr(value))


# ==================================================
# DATASET CONFIGURATION
# ==================================================

def _validate_dataset_config(
    df: pd.DataFrame,
    dataset_config: dict | None
) -> dict:

    if dataset_config is None:
        return {
            "semantic_types": {},
            "currency_column": None,
        }

    if not isinstance(dataset_config, dict):
        raise TypeError("dataset_config must be a dictionary.")

    semantic_types = dataset_config.get("semantic_types", {})
    currency_column = dataset_config.get("currency_column")

    if not isinstance(semantic_types, dict):
        raise TypeError("semantic_types must be a dictionary.")

    supported_semantic_types = {
        "identifier",
        "monetary",
        "datetime",
    }

    for column, semantic_type in semantic_types.items():

        if column not in df.columns:
            raise ValueError(
                f"Configured column '{column}' does not exist in the dataset."
            )

        if semantic_type not in supported_semantic_types:
            raise ValueError(
                f"Unsupported semantic type '{semantic_type}' "
                f"for column '{column}'. "
                f"Supported types: {sorted(supported_semantic_types)}"
            )

    if currency_column is not None:

        if not isinstance(currency_column, str):
            raise TypeError("currency_column must be a string.")

        if currency_column not in df.columns:
            raise ValueError(
                f"Configured currency column '{currency_column}' "
                "does not exist in the dataset."
            )

    return {
        "semantic_types": semantic_types,
        "currency_column": currency_column,
    }


# ==================================================
# CATEGORICAL VALUE DISCOVERY
# ==================================================

def get_categorical_values(
    df: pd.DataFrame,
    column: str,
    limit: int = 30,
) -> dict:
    """Discover actual values and frequencies in a categorical column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):
        raise ValueError("limit must be a positive integer.")

    series = df[column]

    is_categorical = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    if not is_categorical:
        raise ValueError(
            f"Column '{column}' is not a categorical, string "
            "or Boolean column."
        )

    counts = series.value_counts(dropna=True)

    values = [
        {
            "value": value.item() if hasattr(value, "item") else value,
            "count": int(count),
        }
        for value, count in counts.head(limit).items()
    ]

    return {
        "column": column,
        "unique_count": int(series.nunique(dropna=True)),
        "category_values": values,
        "category_values_truncated": len(counts) > limit,
    }


# ==================================================
# VALUE DISTRIBUTION
# ==================================================

def get_value_distribution(
    df: pd.DataFrame,
    column: str,
    filters: dict = None,
    limit: int = None,
    dataset_config: dict = None,
) -> dict:
    """
    Return the frequency distribution of a categorical column.

    Supports optional filters, missing-value reporting,
    percentages and result truncation.

    Percentages use all rows remaining after filtering,
    including rows with missing values.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyse an empty dataset.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    if limit is not None:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("limit must be a positive integer.")

    series = df[column]

    is_categorical = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    if not is_categorical:
        raise ValueError(
            f"Column '{column}' is not a categorical, "
            "string or Boolean column."
        )

    grouped_result = group_and_aggregate(
        df=df,
        group_by=column,
        metric=None,
        aggregation="count",
        filters=filters,
        sort_by="result",
        ascending=False,
        limit=None,
        dataset_config=dataset_config,
    )

    total_rows = grouped_result["rows_after_filtering"]

    missing_count = 0
    category_counts = []

    for row in grouped_result["results"]:

        value = row[column]
        count = int(row["result"])

        if pd.isna(value):
            missing_count = count
            continue

        if hasattr(value, "item"):
            value = value.item()

        percentage = (
            (count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        category_counts.append({
            "value": value,
            "count": count,
            "percentage": round(percentage, 2),
        })

    total_unique_values = len(category_counts)

    if limit is not None:
        displayed_values = category_counts[:limit]
    else:
        displayed_values = category_counts

    missing_percentage = (
        (missing_count / total_rows) * 100
        if total_rows > 0
        else 0.0
    )

    return {
        "column": column,
        "filters": filters or {},
        "rows_after_filtering": total_rows,
        "unique_count": total_unique_values,
        "missing_count": missing_count,
        "missing_percentage": round(missing_percentage, 2),
        "category_values": displayed_values,
        "category_values_truncated": (
            len(displayed_values) < total_unique_values
        ),
    }


# ==================================================
# DATASET OVERVIEW
# ==================================================

def get_dataset_overview(
    df: pd.DataFrame,
    dataset_config: dict = None
) -> dict:
    """Return a high-level dataset overview, respecting excluded columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    excluded_columns = []

    if dataset_config is not None:
        if not isinstance(dataset_config, dict):
            raise TypeError("dataset_config must be a dictionary.")

        excluded_columns = dataset_config.get(
            "exclude_from_context", []
        )

        if not isinstance(excluded_columns, (list, tuple, set)):
            raise TypeError(
                "exclude_from_context must be a list, tuple or set."
            )

        unknown_columns = set(excluded_columns) - set(df.columns)

        if unknown_columns:
            raise ValueError(
                f"Excluded columns do not exist: {sorted(unknown_columns)}"
            )

    visible_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    # Calculate duplicates using only columns available to the agent.
    # Excluded columns must not influence the duplicate-row count.
    visible_df = df[visible_columns]

    duplicate_rows = (
        int(visible_df.duplicated().sum())
        if visible_columns
        else 0
    )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "context_column_count": len(visible_columns),
        "columns": visible_columns,
        "excluded_columns": list(excluded_columns),
        "duplicate_rows": duplicate_rows,
    }


# ==================================================
# COLUMN SUMMARY
# ==================================================

def get_column_summary(
    df: pd.DataFrame,
    column: str,
    dataset_config: dict | None = None
) -> dict:
    """Return a summary of a specific dataset column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    config = _validate_dataset_config(df, dataset_config)

    semantic_types = config["semantic_types"]
    currency_column = config["currency_column"]

    series = df[column]
    semantic_type = semantic_types.get(column)

    result = {
        "column": column,
        "data_type": str(series.dtype),
        "semantic_type": semantic_type,
        "row_count": len(series),
        "missing_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
    }

    if semantic_type == "identifier":

        result["description"] = (
            "Identifier column. Numeric statistics are not applicable."
        )

    elif semantic_type == "monetary":

        if currency_column is None:
            raise ValueError(
                f"'{column}' is marked as monetary, but no "
                "currency_column is configured in dataset_config."
            )

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                f"Monetary column '{column}' must be numeric and not Boolean."
            )

        statistics_by_currency = {}

        for currency, group in df.groupby(
            currency_column,
            dropna=False
        ):

            values = group[column].dropna()

            statistics_by_currency[str(currency)] = {
                "count": int(len(values)),
                "min": float(values.min()) if not values.empty else None,
                "max": float(values.max()) if not values.empty else None,
                "mean": float(values.mean()) if not values.empty else None,
                "median": float(values.median()) if not values.empty else None,
            }

        result["statistics_by_currency"] = statistics_by_currency

    elif (
        semantic_type == "datetime"
        or pd.api.types.is_datetime64_any_dtype(series)
    ):

        if not pd.api.types.is_datetime64_any_dtype(series):
            raise ValueError(
                f"Column '{column}' is marked as datetime, "
                "but its Pandas dtype is not datetime. "
                "Convert it using pd.to_datetime() first."
            )

        result.update({
            "min_date": (
                str(series.min()) if series.notna().any() else None
            ),
            "max_date": (
                str(series.max()) if series.notna().any() else None
            ),
        })

    elif (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):

        result.update({
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "mean": float(series.mean()) if series.notna().any() else None,
            "median": float(series.median()) if series.notna().any() else None,
        })

    else:

        result["top_values"] = (
            series.value_counts(dropna=False)
            .head(5)
            .to_dict()
        )

    return result


# ==================================================
# FILTER, GROUP AND AGGREGATE
# ==================================================

def group_and_aggregate(
    df: pd.DataFrame,
    group_by: str = None,
    metric: str = None,
    aggregation: str = "count",
    filters: dict = None,
    sort_by: str = "result",
    ascending: bool = False,
    limit: int = None,
    dataset_config: dict = None,
    validate_filter_values: bool = True,
    date_granularity: str = None,
    include_empty_periods: bool = False,
) -> dict:
    """
    Filter, group and aggregate dataset records.

    String equality and membership filters are case-insensitive.
    Unknown string categories are rejected by default with suggestions.

    Optional date_granularity groups a parsed datetime column by:
    day, week, month, quarter or year.

    When include_empty_periods is True, missing periods between the
    first and last valid dates are included with a count of zero.
    This option is supported only for date-based count aggregation.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyse an empty dataset.")

    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Choose from: {sorted(SUPPORTED_AGGREGATIONS)}"
        )

    if group_by is not None and group_by not in df.columns:
        raise ValueError(
            f"Grouping column '{group_by}' does not exist."
        )

    if date_granularity is not None:

        if not isinstance(date_granularity, str):
            raise TypeError("date_granularity must be a string or None.")

        date_granularity = date_granularity.lower()

        if date_granularity not in SUPPORTED_DATE_GRANULARITIES:
            raise ValueError(
                "date_granularity must be one of: "
                f"{sorted(SUPPORTED_DATE_GRANULARITIES)}."
            )

        if group_by is None:
            raise ValueError(
                "A grouping column is required when "
                "date_granularity is specified."
            )

        if not pd.api.types.is_datetime64_any_dtype(df[group_by]):
            raise ValueError(
                f"Column '{group_by}' must be a parsed datetime "
                "column to use date_granularity."
            )
    if not isinstance(include_empty_periods, bool):
        raise TypeError("include_empty_periods must be Boolean.")

    if include_empty_periods and date_granularity is None:
        raise ValueError(
            "include_empty_periods requires date_granularity."
        )

    if include_empty_periods and aggregation != "count":
        raise ValueError(
            "include_empty_periods is supported only "
            "with count aggregation."
        )

    if aggregation != "count" and metric is None:
        raise ValueError(
            "A metric column is required for this aggregation."
        )

    if metric is not None and metric not in df.columns:
        raise ValueError(
            f"Metric column '{metric}' does not exist."
        )

    if sort_by not in {"result", group_by, None}:
        raise ValueError(
            "sort_by must be 'result', the grouping column, or None."
        )

    if limit is not None:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("limit must be a positive integer.")

    if not isinstance(validate_filter_values, bool):
        raise TypeError("validate_filter_values must be Boolean.")

    config = _validate_dataset_config(df, dataset_config)

    semantic_types = config["semantic_types"]
    currency_column = config["currency_column"]

    filtered_df = df.copy()

    applied_filters = {} if filters is None else filters

    if not isinstance(applied_filters, dict):
        raise TypeError("Filters must be supplied as a dictionary.")


    for column, condition in applied_filters.items():

        if column not in filtered_df.columns:
            raise ValueError(
                f"Filter column '{column}' does not exist."
            )

        series = filtered_df[column]

        if isinstance(condition, dict):

            if not condition:
                raise ValueError(
                    f"Filter for '{column}' cannot be empty."
                )

            combined_mask = None

            for operator, value in condition.items():

                if operator not in SUPPORTED_FILTER_OPERATORS:
                    raise ValueError(
                        f"Unsupported filter operator '{operator}'."
                    )

                if operator in {"in", "not_in"}:
                    if (
                        isinstance(value, (str, bytes))
                        or not isinstance(value, (list, tuple, set))
                    ):
                        raise ValueError(
                            f"'{operator}' requires a list of values."
                        )

                if operator == "contains" and not isinstance(value, str):
                    raise ValueError(
                        "'contains' requires a string value."
                    )

                if operator in {"is_null", "not_null"} and value is not True:
                    raise ValueError(
                        f"'{operator}' must be set to True."
                    )

                if validate_filter_values:

                    validation_values = []

                    if operator in {"eq", "ne"}:
                        validation_values = [value]

                    elif operator in {"in", "not_in"}:
                        validation_values = list(value)

                    for requested_value in validation_values:
                        _validate_categorical_filter_value(
                            original_series=df[column],
                            value=requested_value,
                            column=column,
                        )

                if operator == "eq":
                    mask = _case_insensitive_equality(series, value)

                elif operator == "ne":
                    mask = _case_insensitive_equality(
                        series, value, negate=True
                    )

                elif operator == "gt":
                    mask = series > value

                elif operator == "gte":
                    mask = series >= value

                elif operator == "lt":
                    mask = series < value

                elif operator == "lte":
                    mask = series <= value

                elif operator == "in":
                    mask = _case_insensitive_membership(series, value)

                elif operator == "not_in":
                    mask = _case_insensitive_membership(
                        series, value, negate=True
                    )

                elif operator == "contains":
                    mask = series.astype("string").str.contains(
                        value,
                        case=False,
                        na=False,
                        regex=False,
                    )

                elif operator == "is_null":
                    mask = series.isna()

                else:
                    mask = series.notna()

                combined_mask = (
                    mask
                    if combined_mask is None
                    else combined_mask & mask
                )

            filtered_df = filtered_df.loc[
                combined_mask.fillna(False)
            ].copy()

        elif isinstance(condition, (list, tuple, set)):

            if validate_filter_values:
                for requested_value in condition:
                    _validate_categorical_filter_value(
                        original_series=df[column],
                        value=requested_value,
                        column=column,
                    )

            mask = _case_insensitive_membership(series, condition)

            filtered_df = filtered_df.loc[
                mask.fillna(False)
            ].copy()

        else:

            if validate_filter_values:
                _validate_categorical_filter_value(
                    original_series=df[column],
                    value=condition,
                    column=column,
                )

            mask = _case_insensitive_equality(series, condition)

            filtered_df = filtered_df.loc[
                mask.fillna(False)
            ].copy()

    # Validate metric
    if metric is not None and aggregation != "count":

        if not pd.api.types.is_numeric_dtype(filtered_df[metric]):
            raise ValueError(
                f"Metric '{metric}' must be numeric."
            )

        if pd.api.types.is_bool_dtype(filtered_df[metric]):
            raise ValueError(
                f"Metric '{metric}' cannot be Boolean."
            )

    # Currency safeguards
    is_monetary = (
        metric is not None
        and semantic_types.get(metric) == "monetary"
        and aggregation != "count"
    )

    currency_values = None

    if is_monetary:

        if currency_column is None:
            raise ValueError(
                f"'{metric}' is marked as monetary, but no "
                "currency_column is configured in dataset_config."
            )

        currency_filter = currency_column in applied_filters
        currency_grouping = group_by == currency_column

        if not currency_filter and not currency_grouping:
            raise ValueError(
                f"'{metric}' is monetary. "
                f"Add a '{currency_column}' filter or group by "
                f"'{currency_column}' to avoid mixing currencies."
            )

        currencies = filtered_df[currency_column].dropna().unique()

        if not currency_grouping and len(currencies) > 1:
            raise ValueError(
                "The selected filters include multiple currencies. "
                "Filter to one currency or group by currency."
            )

        if (
            not currency_grouping
            and filtered_df[currency_column].isna().any()
        ):
            raise ValueError(
                "Some monetary records have missing currency values."
            )

        currency_values = currencies.tolist()

    # Prepare grouping data.
    aggregation_df = filtered_df.copy()
    grouping_column = group_by

    if date_granularity is not None:

        frequency = SUPPORTED_DATE_GRANULARITIES[date_granularity]

        period_values = (
            filtered_df[group_by]
            .dt.to_period(frequency)
        )

        grouping_column = "_datasense_date_group"

        # Keep missing dates as missing groups.
        aggregation_df[grouping_column] = (
            period_values.astype("string")
        )

    # Aggregation
    if aggregation == "count":

        if group_by is None:
            results = [{"result": len(filtered_df)}]

        else:
            grouped = (
                aggregation_df.groupby(
                    grouping_column,
                    dropna=False,
                )
                .size()
                .reset_index(name="result")
            )

            if date_granularity is not None:
                grouped = grouped.rename(
                    columns={grouping_column: group_by}
                )

            results = grouped.to_dict(orient="records")

    else:

        if group_by is None:

            if aggregation == "sum":
                value = filtered_df[metric].sum()
            else:
                value = getattr(filtered_df[metric], aggregation)()

            results = [{
                "result": (
                    float(value) if pd.notna(value) else None
                )
            }]

        else:

            grouped = (
                aggregation_df.groupby(
                    grouping_column,
                    dropna=False,
                )[metric]
                .agg(aggregation)
                .reset_index(name="result")
            )

            if date_granularity is not None:
                grouped = grouped.rename(
                    columns={grouping_column: group_by}
                )

            results = grouped.to_dict(orient="records")

    # Include missing calendar periods when explicitly requested.
    if include_empty_periods:

        valid_periods = period_values.dropna()

        if not valid_periods.empty:

            period_counts = (
                aggregation_df.loc[filtered_df[group_by].notna()]
                .groupby(grouping_column)
                .size()
            )

            complete_periods = pd.period_range(
                start=valid_periods.min(),
                end=valid_periods.max(),
                freq=frequency,
            )

            results = [
                {
                    group_by: str(period),
                    "result": int(period_counts.get(str(period), 0)),
                }
                for period in complete_periods
            ]

        else:
            results = []

        # Keep records with missing dates separate from calendar periods.
        missing_date_count = int(filtered_df[group_by].isna().sum())

        if missing_date_count:
            results.append({
                group_by: None,
                "result": missing_date_count,
            })

    # Sorting
    if sort_by is not None and results:

        sort_column = (
            group_by if sort_by == group_by else "result"
        )

        if sort_column is not None:

            non_null_results = [
                row for row in results
                if pd.notna(row.get(sort_column))
            ]

            null_results = [
                row for row in results
                if pd.isna(row.get(sort_column))
            ]

            non_null_results = sorted(
                non_null_results,
                key=lambda row: row[sort_column],
                reverse=not ascending
            )

            results = non_null_results + null_results

    if limit is not None:
        results = results[:limit]

    return {
        "group_by": group_by,
        "date_granularity": date_granularity,
        "include_empty_periods": include_empty_periods,
        "metric": metric,
        "aggregation": aggregation,
        "filters": applied_filters,
        "rows_after_filtering": len(filtered_df),
        "result_count": len(results),
        "currency": currency_values,
        "results": results,
    }


# ==================================================
# COMPARE TWO GROUPS
# ==================================================

def compare_groups(
    df: pd.DataFrame,
    comparison_column: str,
    group_a,
    group_b,
    metric: str = None,
    aggregation: str = "mean",
    group_by: str = None,
    filters: dict = None,
    dataset_config: dict = None,
) -> dict:
    """
    Compare an aggregation between two values of a categorical column.

    Difference = group_b minus group_a.
    Percentage difference = difference / abs(group_a) * 100.

    If group_by is supplied, comparisons are performed separately
    for each value of that column.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyse an empty dataset.")

    if comparison_column not in df.columns:
        raise ValueError(
            f"Comparison column '{comparison_column}' does not exist."
        )

    if group_by is not None and group_by not in df.columns:
        raise ValueError(
            f"Grouping column '{group_by}' does not exist."
        )

    if group_by == comparison_column:
        raise ValueError(
            "group_by and comparison_column must be different columns."
        )

    if isinstance(group_a, (dict, list, tuple, set)) or isinstance(
        group_b, (dict, list, tuple, set)
    ):
        raise TypeError("group_a and group_b must each be a single value.")

    if _comparison_group_key(group_a) == _comparison_group_key(group_b):
        raise ValueError("group_a and group_b must be different values.")

    if filters is None:
        base_filters = {}
    elif isinstance(filters, dict):
        base_filters = dict(filters)
    else:
        raise TypeError("Filters must be supplied as a dictionary.")

    if comparison_column in base_filters:
        raise ValueError(
            f"Do not include '{comparison_column}' in filters; "
            "use group_a and group_b for that column."
        )

    filters_a = {**base_filters, comparison_column: group_a}
    filters_b = {**base_filters, comparison_column: group_b}

    result_a = group_and_aggregate(
        df=df,
        group_by=group_by,
        metric=metric,
        aggregation=aggregation,
        filters=filters_a,
        sort_by=None,
        dataset_config=dataset_config,
    )

    result_b = group_and_aggregate(
        df=df,
        group_by=group_by,
        metric=metric,
        aggregation=aggregation,
        filters=filters_b,
        sort_by=None,
        dataset_config=dataset_config,
    )

    def index_results(result):

        indexed = {}

        for row in result["results"]:

            if group_by is None:
                key = ("all",)
                label = "all"

            else:
                label = row[group_by]
                key = _group_value_key(label)

            indexed[key] = {
                "label": label,
                "value": row.get("result"),
            }

        return indexed

    indexed_a = index_results(result_a)
    indexed_b = index_results(result_b)

    all_keys = list(indexed_a)
    all_keys.extend(
        key for key in indexed_b
        if key not in indexed_a
    )

    comparisons = []

    for key in all_keys:

        a = indexed_a.get(key)
        b = indexed_b.get(key)

        label = a["label"] if a is not None else b["label"]

        value_a = (
            a["value"]
            if a is not None and result_a["rows_after_filtering"] > 0
            else None
        )

        value_b = (
            b["value"]
            if b is not None and result_b["rows_after_filtering"] > 0
            else None
        )

        if value_a is None or value_b is None:

            difference = None
            percentage_difference = None

        else:

            difference = float(value_b - value_a)

            percentage_difference = (
                None
                if value_a == 0
                else float((difference / abs(value_a)) * 100)
            )

        row = {
            "group_a_value": value_a,
            "group_b_value": value_b,
            "difference_b_minus_a": difference,
            "percentage_difference_vs_a": percentage_difference,
        }

        if group_by is not None:
            row[group_by] = label

        comparisons.append(row)

    if group_by is not None:

        comparisons.sort(
            key=lambda row: (
                row[group_by] is None,
                str(row[group_by]).casefold()
            )
        )

    return {
        "comparison_column": comparison_column,
        "group_a": group_a,
        "group_b": group_b,
        "metric": metric,
        "aggregation": aggregation,
        "group_by": group_by,
        "difference_definition": "group_b minus group_a",
        "percentage_definition": (
            "difference divided by absolute group_a, multiplied by 100"
        ),
        "rows_after_filtering": {
            "group_a": result_a["rows_after_filtering"],
            "group_b": result_b["rows_after_filtering"],
        },
        "result_count": len(comparisons),
        "results": comparisons,
    }
