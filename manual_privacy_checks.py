
import pandas as pd

from dataset_context.builder import build_dataset_context
from agent.runner import execute_tool


# Synthetic data only — never use real personal information here.
df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "email": [
        "alice@example.com",
        "bob@example.com",
        "charlie@example.com",
    ],
    "city": ["Dubai", "Berlin", "Munich"],
    "internal_notes": [
        "Customer complaint about billing",
        "Account escalation",
        "Refund discussion",
    ],
    "revenue": [100, 200, 150],
})

config = {
    "exclude_from_context": ["email", "internal_notes"]
}

excluded = {"email", "internal_notes"}


# --------------------------------------------------
# TEST 1: EXCLUDED COLUMNS NOT IN CONTEXT
# --------------------------------------------------

print("\nTEST 1: EXCLUDED COLUMNS NOT IN CONTEXT")

context = build_dataset_context(
    df=df,
    dataset_name="synthetic_privacy_test",
    dataset_config=config,
)

available_columns = [
    column["column_name"] for column in context["columns"]
]

print("Available columns:", available_columns)
print("Excluded columns:", context["excluded_columns"])

assert "email" not in available_columns
assert "internal_notes" not in available_columns

print("PASSED")


# --------------------------------------------------
# TEST 2: DIRECT ACCESS TO EMAIL IS BLOCKED
# --------------------------------------------------

print("\nTEST 2: DIRECT ACCESS TO EMAIL IS BLOCKED")

try:
    execute_tool(
        "get_column_summary",
        {"column": "email"},
        active_df=df,
        active_dataset_config=config,
        active_excluded_columns=excluded,
    )

    raise AssertionError("Email access was not blocked.")

except ValueError as error:
    print("Blocked:", error)
    assert "excluded" in str(error).lower()

print("PASSED")


# --------------------------------------------------
# TEST 3: DIRECT ACCESS TO INTERNAL NOTES IS BLOCKED
# --------------------------------------------------

print("\nTEST 3: DIRECT ACCESS TO INTERNAL NOTES IS BLOCKED")

try:
    execute_tool(
        "get_column_summary",
        {"column": "internal_notes"},
        active_df=df,
        active_dataset_config=config,
        active_excluded_columns=excluded,
    )

    raise AssertionError("Internal notes access was not blocked.")

except ValueError as error:
    print("Blocked:", error)
    assert "excluded" in str(error).lower()

print("PASSED")


# --------------------------------------------------
# TEST 4: FILTERING ON AN EXCLUDED COLUMN IS BLOCKED
# --------------------------------------------------

print("\nTEST 4: FILTERING ON AN EXCLUDED COLUMN IS BLOCKED")

try:
    execute_tool(
        "get_value_distribution",
        {
            "column": "city",
            "filters": {
                "email": "alice@example.com"
            },
        },
        active_df=df,
        active_dataset_config=config,
        active_excluded_columns=excluded,
    )

    raise AssertionError("Filtering on email was not blocked.")

except ValueError as error:
    print("Blocked:", error)
    assert "excluded" in str(error).lower()

print("PASSED")


# --------------------------------------------------
# TEST 5: NORMAL ANALYTICS STILL WORK
# --------------------------------------------------

print("\nTEST 5: NORMAL ANALYTICS STILL WORK")

result = execute_tool(
    "get_column_summary",
    {"column": "revenue"},
    active_df=df,
    active_dataset_config=config,
    active_excluded_columns=excluded,
)

print("Revenue summary:", result)

assert result is not None

print("PASSED")


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 50)
print("ALL FIVE PRIVACY TESTS COMPLETED SUCCESSFULLY")
print("=" * 50)
