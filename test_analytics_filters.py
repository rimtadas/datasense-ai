import pandas as pd

from analytics.tools import group_and_aggregate


def test_date_range_filter_uses_both_boundaries():
    df = pd.DataFrame({
        "city": ["Dubai", "Dubai", "Berlin", "Berlin", "London"],
        "signup_date": pd.to_datetime([
            "2025-01-10",
            "2025-06-15",
            "2025-03-20",
            "2026-01-01",
            "2024-12-31",
        ]),
    })

    result = group_and_aggregate(
        df,
        group_by="city",
        aggregation="count",
        filters={
            "signup_date": {
                "gte": "2025-01-01",
                "lt": "2026-01-01",
            }
        },
        sort_by="result",
        ascending=False,
    )

    actual = {
        row["city"]: row["result"]
        for row in result["results"]
    }

    assert actual == {"Dubai": 2, "Berlin": 1}