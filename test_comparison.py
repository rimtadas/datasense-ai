
import pandas as pd

from analytics.tools import (
    compare_groups,
    group_and_aggregate,
)


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

df = pd.DataFrame({
    "order_status": [
        "Delivered",
        "delivered",
        "Cancelled",
        "CANCELLED",
        "Delivered",
        "Cancelled",
    ],
    "currency": [
        "USD", "USD", "USD", "USD", "EUR", "EUR"
    ],
    "total_amount": [
        100, 200, 200, 300, 100, 80
    ],
})

dataset_config = {
    "semantic_types": {
        "total_amount": "monetary",
    },
    "currency_column": "currency",
}


# --------------------------------------------------
# TEST 1: CASE-INSENSITIVE FILTERING
# --------------------------------------------------

print("\nTEST 1: CASE-INSENSITIVE FILTERING")

result = group_and_aggregate(
    df=df,
    group_by="currency",
    metric="total_amount",
    aggregation="mean",
    filters={"order_status": "dElIvErEd"},
    dataset_config=dataset_config,
)

assert result["rows_after_filtering"] == 3

print("Rows matched:", result["rows_after_filtering"])
print("Results:", result["results"])


# --------------------------------------------------
# TEST 2: COMPARE TWO GROUPS
# --------------------------------------------------

print("\nTEST 2: COMPARE DELIVERED VS CANCELLED")

result = compare_groups(
    df=df,
    comparison_column="order_status",
    group_a="delivered",
    group_b="CANCELLED",
    metric="total_amount",
    aggregation="mean",
    group_by="currency",
    dataset_config=dataset_config,
)

comparison_df = pd.DataFrame(result["results"])

print(comparison_df.to_string(index=False))

assert result["rows_after_filtering"]["group_a"] == 3
assert result["rows_after_filtering"]["group_b"] == 3
assert result["result_count"] == 2


# --------------------------------------------------
# TEST 3: VERIFY CALCULATIONS
# --------------------------------------------------

print("\nTEST 3: VERIFY DIFFERENCES")

usd = comparison_df[
    comparison_df["currency"] == "USD"
].iloc[0]

eur = comparison_df[
    comparison_df["currency"] == "EUR"
].iloc[0]

assert usd["group_a_value"] == 150
assert usd["group_b_value"] == 250
assert usd["difference_b_minus_a"] == 100

assert round(
    usd["percentage_difference_vs_a"], 2
) == 66.67

assert eur["group_a_value"] == 100
assert eur["group_b_value"] == 80
assert eur["difference_b_minus_a"] == -20

assert eur["percentage_difference_vs_a"] == -20


# --------------------------------------------------
# TEST 4: CURRENCY SAFEGUARD
# --------------------------------------------------

print("\nTEST 4: CURRENCY SAFEGUARD")

try:
    compare_groups(
        df=df,
        comparison_column="order_status",
        group_a="Delivered",
        group_b="Cancelled",
        metric="total_amount",
        aggregation="mean",
        dataset_config=dataset_config,
    )

    raise AssertionError(
        "Expected currency safeguard to reject mixed currencies."
    )

except ValueError as error:
    assert "currency" in str(error).lower()
    print("Expected safeguard:", error)


print("\n" + "=" * 55)
print("ALL COMPARISON TESTS PASSED")
print("=" * 55)
