
import pandas as pd


# --------------------------------------------------
# Q01: DATASET OVERVIEW
# --------------------------------------------------

def calculate_q01_expected(df: pd.DataFrame) -> dict:
    """
    Independently calculate the expected dataset overview:
    row count, column count, and duplicate row count.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    return {
        "question_id": "Q01",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
    }

# --------------------------------------------------
# Q02: AVAILABLE AND EXCLUDED COLUMNS
# --------------------------------------------------

def calculate_q02_expected(df: pd.DataFrame, dataset_config: dict) -> dict:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    excluded_columns = dataset_config.get("exclude_from_context", [])

    available_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    return {
        "question_id": "Q02",
        "available_columns": available_columns,
        "available_column_count": len(available_columns),
        "excluded_columns": excluded_columns,
        "excluded_column_count": len(excluded_columns),
    }


# --------------------------------------------------
# Q03: DATE RANGE
# --------------------------------------------------

def calculate_q03_expected(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if "purchase_date" not in df.columns:
        raise ValueError("Column 'purchase_date' not found.")

    dates = pd.to_datetime(df["purchase_date"], errors="coerce")

    return {
        "question_id": "Q03",
        "column": "purchase_date",
        "valid_date_count": int(dates.notna().sum()),
        "min_date": str(dates.min()),
        "max_date": str(dates.max()),
    }


# --------------------------------------------------
# Q04: NUMERIC COLUMN SUMMARY
# --------------------------------------------------

def calculate_q04_expected(df: pd.DataFrame) -> dict:
    if "quantity" not in df.columns:
        raise ValueError("Column 'quantity' not found.")

    values = pd.to_numeric(df["quantity"], errors="coerce").dropna()

    return {
        "question_id": "Q04",
        "column": "quantity",
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


# --------------------------------------------------
# Q05: ORDER STATUS DISTRIBUTION
# --------------------------------------------------

def calculate_q05_expected(df: pd.DataFrame) -> dict:
    if "order_status" not in df.columns:
        raise ValueError("Column 'order_status' not found.")

    counts = df["order_status"].value_counts(dropna=False)

    return {
        "question_id": "Q05",
        "unique_count": int(df["order_status"].nunique()),
        "category_counts": {
            str(category): int(count)
            for category, count in counts.items()
        },
    }

# --------------------------------------------------
# Q06: PAYMENT METHOD CATEGORIES
# --------------------------------------------------

def calculate_q06_expected(df: pd.DataFrame) -> dict:
    if "payment_method" not in df.columns:
        raise ValueError("Column 'payment_method' not found.")

    counts = df["payment_method"].value_counts(dropna=False)

    return {
        "question_id": "Q06",
        "unique_count": int(df["payment_method"].nunique()),
        "category_counts": {
            str(category): int(count)
            for category, count in counts.items()
        },
    }


# --------------------------------------------------
# Q07: PAYMENT METHOD DISTRIBUTION
# --------------------------------------------------

def calculate_q07_expected(df: pd.DataFrame) -> dict:
    if "payment_method" not in df.columns:
        raise ValueError("Column 'payment_method' not found.")

    counts = df["payment_method"].value_counts(dropna=False)
    total_rows = len(df)

    distribution = {
        str(category): {
            "count": int(count),
            "percentage": round(count / total_rows * 100, 2)
        }
        for category, count in counts.items()
    }

    return {
        "question_id": "Q07",
        "total_rows": total_rows,
        "unique_count": int(df["payment_method"].nunique()),
        "distribution": distribution,
    }


# --------------------------------------------------
# Q08: PAYMENT METHOD DISTRIBUTION AMONG CANCELLED
# --------------------------------------------------

def calculate_q08_expected(df: pd.DataFrame) -> dict:
    filtered = df[df["order_status"] == "Cancelled"]

    counts = filtered["payment_method"].value_counts()

    return {
        "question_id": "Q08",
        "total_rows": len(filtered),
        "distribution": {
            str(category): {
                "count": int(count),
                "percentage": round(count / len(filtered) * 100, 2)
            }
            for category, count in counts.items()
        },
    }


# --------------------------------------------------
# Q09: TRANSACTION COUNTS BY ORDER STATUS
# --------------------------------------------------

def calculate_q09_expected(df: pd.DataFrame) -> dict:
    counts = df["order_status"].value_counts()

    return {
        "question_id": "Q09",
        "category_counts": {
            str(category): int(count)
            for category, count in counts.items()
        },
    }


# --------------------------------------------------
# Q10: TOP FIVE COUNTRIES BY TRANSACTION VOLUME
# --------------------------------------------------

def calculate_q10_expected(df: pd.DataFrame) -> dict:
    counts = df["country"].value_counts().head(5)

    return {
        "question_id": "Q10",
        "top_countries": [
            {
                "country": str(country),
                "count": int(count),
                "percentage": round(count / len(df) * 100, 2)
            }
            for country, count in counts.items()
        ],
    }


# --------------------------------------------------
# Q11: DELIVERED AVERAGE AMOUNT BY CURRENCY
# --------------------------------------------------

def calculate_q11_expected(df: pd.DataFrame) -> dict:
    filtered = df[df["order_status"] == "Delivered"]

    grouped = (
        filtered
        .groupby("currency")["total_amount"]
        .mean()
        .sort_index()
    )

    return {
        "question_id": "Q11",
        "aggregation": "mean",
        "metric": "total_amount",
        "filter": {"order_status": "Delivered"},
        "results_by_currency": {
            str(currency): float(mean)
            for currency, mean in grouped.items()
        },
    }


# --------------------------------------------------
# Q12: COMPARE DELIVERED VS CANCELLED BY CURRENCY
# --------------------------------------------------

def calculate_q12_expected(df: pd.DataFrame) -> dict:
    delivered = (
        df[df["order_status"] == "Delivered"]
        .groupby("currency")["total_amount"]
        .mean()
    )

    cancelled = (
        df[df["order_status"] == "Cancelled"]
        .groupby("currency")["total_amount"]
        .mean()
    )

    results = {}

    for currency in sorted(set(delivered.index) & set(cancelled.index)):
        a = float(delivered[currency])
        b = float(cancelled[currency])
        difference = b - a

        results[str(currency)] = {
            "delivered_mean": a,
            "cancelled_mean": b,
            "difference_b_minus_a": difference,
            "percentage_difference_vs_delivered": (
                difference / abs(a) * 100 if a != 0 else None
            ),
        }

    return {
        "question_id": "Q12",
        "comparison": "Delivered vs Cancelled",
        "results_by_currency": results,
    }


# --------------------------------------------------
# Q14: TOP TWO ORDER STATUSES AND AMOUNT COMPARISON
# --------------------------------------------------

def calculate_q14_expected(df: pd.DataFrame) -> dict:
    status_counts = df["order_status"].value_counts()
    top_two_statuses = status_counts.head(2)

    status_a = top_two_statuses.index[0]
    status_b = top_two_statuses.index[1]

    mean_a = (
        df[df["order_status"] == status_a]
        .groupby("currency")["total_amount"]
        .mean()
    )

    mean_b = (
        df[df["order_status"] == status_b]
        .groupby("currency")["total_amount"]
        .mean()
    )

    results = {}

    for currency in sorted(set(mean_a.index) & set(mean_b.index)):
        a = float(mean_a[currency])
        b = float(mean_b[currency])
        difference = b - a

        results[str(currency)] = {
            "group_a_mean": a,
            "group_b_mean": b,
            "difference_b_minus_a": difference,
            "percentage_difference_vs_a": (
                difference / abs(a) * 100 if a != 0 else None
            ),
        }

    return {
        "question_id": "Q14",
        "top_two_statuses": [
            {
                "status": str(status),
                "count": int(count),
            }
            for status, count in top_two_statuses.items()
        ],
        "comparison": f"{status_a} vs {status_b}",
        "results_by_currency": results,
    }


# --------------------------------------------------
# Q15: HIGHEST COUNTRY BY CANCELLED TRANSACTIONS
# --------------------------------------------------

def calculate_q15_expected(df: pd.DataFrame) -> dict:
    cancelled = df[df["order_status"] == "Cancelled"]

    country_counts = cancelled["country"].value_counts()

    if country_counts.empty:
        raise ValueError("No cancelled transactions found.")

    top_country = country_counts.index[0]
    top_count = int(country_counts.iloc[0])
    total_cancelled = len(cancelled)

    return {
        "question_id": "Q15",
        "total_cancelled": total_cancelled,
        "highest_country": str(top_country),
        "transaction_count": top_count,
        "percentage": round(
            top_count / total_cancelled * 100, 2
        ),
    }


# --------------------------------------------------
# Q13: PAYMENT STATUS COMPARISON
# --------------------------------------------------

def calculate_q13_expected(df: pd.DataFrame) -> dict:
    """
    Independently calculate Completed versus Failed
    transaction counts using Pandas directly.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if "payment_status" not in df.columns:
        raise ValueError("Column 'payment_status' not found.")

    completed_count = int(
        (df["payment_status"] == "Completed").sum()
    )

    failed_count = int(
        (df["payment_status"] == "Failed").sum()
    )

    difference = failed_count - completed_count

    percentage_difference = (
        difference / abs(completed_count) * 100
        if completed_count != 0
        else None
    )

    return {
        "question_id": "Q13",
        "group_a": "Completed",
        "group_a_count": completed_count,
        "group_b": "Failed",
        "group_b_count": failed_count,
        "difference_b_minus_a": difference,
        "percentage_difference_vs_a": percentage_difference,
    }
