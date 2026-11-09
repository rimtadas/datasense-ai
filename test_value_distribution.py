
import pandas as pd

from analytics.tools import get_value_distribution


# --------------------------------------------------
# TEST 1: BASIC DISTRIBUTION
# --------------------------------------------------

df = pd.DataFrame({
    "payment_method": [
        "Debit Card",
        "Credit Card",
        "Debit Card",
        "PayPal",
        "Credit Card",
        "Debit Card",
        "PayPal",
        "Debit Card",
    ],
    "order_status": [
        "Cancelled",
        "Cancelled",
        "Delivered",
        "Cancelled",
        "Delivered",
        "Cancelled",
        "Delivered",
        "Cancelled",
    ],
})

print("\nTEST 1: BASIC DISTRIBUTION")
print("--------------------------")

result = get_value_distribution(
    df=df,
    column="payment_method",
)

print(result)


# --------------------------------------------------
# TEST 2: FILTERED DISTRIBUTION
# --------------------------------------------------

print("\nTEST 2: FILTERED DISTRIBUTION")
print("-----------------------------")

result = get_value_distribution(
    df=df,
    column="payment_method",
    filters={
        "order_status": "Cancelled"
    },
)

print(result)


# --------------------------------------------------
# TEST 3: CASE-INSENSITIVE FILTERING
# --------------------------------------------------

print("\nTEST 3: CASE-INSENSITIVE FILTERING")
print("----------------------------------")

result = get_value_distribution(
    df=df,
    column="payment_method",
    filters={
        "order_status": "cancelled"
    },
)

print(result)


# --------------------------------------------------
# TEST 4: RESULT LIMIT
# --------------------------------------------------

print("\nTEST 4: RESULT LIMIT")
print("--------------------")

result = get_value_distribution(
    df=df,
    column="payment_method",
    limit=2,
)

print(result)


# --------------------------------------------------
# TEST 5: MISSING VALUES
# --------------------------------------------------

print("\nTEST 5: MISSING VALUES")
print("----------------------")

df_missing = pd.DataFrame({
    "department": [
        "Sales",
        "HR",
        "Sales",
        None,
        "Finance",
        None,
    ]
})

result = get_value_distribution(
    df=df_missing,
    column="department",
)

print(result)


# --------------------------------------------------
# TEST 6: INVALID CATEGORY
# --------------------------------------------------

print("\nTEST 6: INVALID CATEGORY")
print("------------------------")

try:
    get_value_distribution(
        df=df,
        column="payment_method",
        filters={
            "order_status": "Canceld"
        },
    )

except ValueError as error:
    print("Expected validation error:")
    print(error)


print("\nALL TESTS COMPLETED")
