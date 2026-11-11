
import pandas as pd
import pytest

from analytics.tools import group_and_aggregate


def test_monthly_count_includes_missing_periods():
    df = pd.DataFrame({
        "signup_date": pd.to_datetime([
            "2025-01-10",
            "2025-03-20",
            "2025-03-25",
        ])
    })

    result = group_and_aggregate(
        df=df,
        group_by="signup_date",
        aggregation="count",
        date_granularity="month",
        include_empty_periods=True,
        sort_by="signup_date",
        ascending=True,
    )

    counts = {
        row["signup_date"]: row["result"]
        for row in result["results"]
    }

    assert counts == {
        "2025-01": 1,
        "2025-02": 0,
        "2025-03": 2,
    }


def test_missing_dates_are_reported_separately():
    df = pd.DataFrame({
        "signup_date": pd.to_datetime([
            "2025-01-10",
            "2025-03-20",
            None,
        ])
    })

    result = group_and_aggregate(
        df=df,
        group_by="signup_date",
        aggregation="count",
        date_granularity="month",
        include_empty_periods=True,
    )

    counts = {
        row["signup_date"]: row["result"]
        for row in result["results"]
    }

    assert counts == {
        "2025-01": 1,
        "2025-02": 0,
        "2025-03": 1,
        None: 1,
    }


def test_date_filter_limits_period_range():
    df = pd.DataFrame({
        "signup_date": pd.to_datetime([
            "2024-12-15",
            "2025-01-10",
            "2025-03-20",
            "2025-04-15",
        ])
    })

    result = group_and_aggregate(
        df=df,
        group_by="signup_date",
        aggregation="count",
        filters={
            "signup_date": {
                "gte": "2025-01-01",
                "lt": "2025-04-01",
            }
        },
        date_granularity="month",
        include_empty_periods=True,
        sort_by="signup_date",
        ascending=True,
    )

    counts = {
        row["signup_date"]: row["result"]
        for row in result["results"]
    }

    assert counts == {
        "2025-01": 1,
        "2025-02": 0,
        "2025-03": 1,
    }


def test_empty_periods_requires_date_granularity():
    df = pd.DataFrame({"city": ["Dubai", "Berlin"]})

    with pytest.raises(ValueError, match="date_granularity"):
        group_and_aggregate(
            df=df,
            group_by="city",
            aggregation="count",
            include_empty_periods=True,
        )


def test_empty_periods_only_supports_count_aggregation():
    df = pd.DataFrame({
        "signup_date": pd.to_datetime([
            "2025-01-10",
            "2025-03-20",
        ]),
        "revenue": [100, 200],
    })

    with pytest.raises(ValueError, match="count aggregation"):
        group_and_aggregate(
            df=df,
            group_by="signup_date",
            metric="revenue",
            aggregation="sum",
            date_granularity="month",
            include_empty_periods=True,
        )
