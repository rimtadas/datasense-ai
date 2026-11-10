import argparse

import pandas as pd

from agent.runner import run_agent, execute_tool


def build_support_dataset():
    """Create a dataset unrelated to the existing e-commerce schema."""

    data = [
        [ "T001", "2026-01-01", "Billing", "High",   "West",  "Email",  4.0,  5, "Resolved", "Internal note A"],
        [ "T002", "2026-01-02", "Tech",    "Medium", "East",  "Chat",   8.0,  4, "Resolved", "Internal note B"],
        [ "T003", "2026-01-03", "Billing", "Low",    "West",  "Email",  2.0,  5, "Resolved", "Internal note C"],
        [ "T004", "2026-01-04", "Tech",    "High",   "North", "Phone", 24.0,  2, "Open",     "Internal note D"],
        [ "T005", "2026-01-05", "Returns", "Medium", "West",  "Chat",   6.0,  4, "Resolved", "Internal note E"],
        [ "T006", "2026-01-06", "Billing", "High",   "East",  "Email", 12.0,  3, "Open",     "Internal note F"],
        [ "T007", "2026-01-07", "Tech",    "Low",    "South", "Chat",   3.0,  5, "Resolved", "Internal note G"],
        [ "T008", "2026-01-08", "Returns", "High",   "West",  "Phone", 18.0, None, "Open",   "Internal note H"],
        [ "T009", "2026-01-09", "Billing", "Medium", "North", "Email", None, 4, "Resolved", "Internal note I"],
        [ "T010", "2026-01-10", "Tech",    "Medium", "West",  "Chat",  10.0,  3, "Resolved", "Internal note J"],
        [ "T011", "2026-01-11", "Returns", "Low",    "East",  "Email",  5.0,  5, "Resolved", "Internal note K"],
        [ "T012", "2026-01-12", "Billing", "High",   "West",  "Phone", 20.0,  1, "Open",     "Internal note L"],
    ]

    columns = [
        "ticket_id",
        "created_at",
        "support_team",
        "priority",
        "region",
        "channel",
        "resolution_hours",
        "satisfaction_score",
        "status",
        "internal_notes",
    ]

    df = pd.DataFrame(data, columns=columns)

    df["created_at"] = pd.to_datetime(df["created_at"])

    return df


DATASET_CONFIG = {
    "dataset_description": (
        "Customer support ticket dataset for testing "
        "dataset-agnostic analytics."
    ),
    "semantic_types": {
        "ticket_id": "identifier",
        "created_at": "datetime",
    },
    "column_descriptions": {
        "resolution_hours": "Time taken to resolve a support ticket, in hours.",
        "satisfaction_score": "Customer satisfaction score.",
        "support_team": "Team responsible for handling the ticket.",
        "status": "Current ticket status.",
    },
    "exclude_from_context": ["internal_notes"],
}

def run_portability_safeguard_checks(df):
    """Verify missing-value profiling and excluded-column enforcement."""

    print("\nPORTABILITY SAFEGUARD CHECKS")
    print("-" * 65)

    # Check 1: Missing values are correctly detected.
    resolution_summary = execute_tool(
        "get_column_summary",
        {"column": "resolution_hours"},
        active_df=df,
        active_dataset_config=DATASET_CONFIG,
        active_excluded_columns={"internal_notes"},
    )

    satisfaction_summary = execute_tool(
        "get_column_summary",
        {"column": "satisfaction_score"},
        active_df=df,
        active_dataset_config=DATASET_CONFIG,
        active_excluded_columns={"internal_notes"},
    )

    print("Missing-value profiling:")
    print("resolution_hours:", resolution_summary)
    print("satisfaction_score:", satisfaction_summary)

    # Check 2: Excluded columns cannot be accessed through tools.
    try:
        execute_tool(
            "get_column_summary",
            {"column": "internal_notes"},
            active_df=df,
            active_dataset_config=DATASET_CONFIG,
            active_excluded_columns={"internal_notes"},
        )

        raise AssertionError(
            "Excluded column was accessible through execute_tool."
        )

    except ValueError as error:
        if "excluded from agent access" not in str(error):
            raise

        print("\nExcluded-column enforcement: PASSED")
        print("Validation message:", error)

QUESTIONS = {
    1: [
        (
            "P01",
            "How many rows and columns are in this dataset, "
            "and which columns are available to the agent?"
        ),
        (
            "P02",
            "What is the date range of the created_at column?"
        ),
        (
            "P03",
            "What are the distinct support_team values and "
            "their record counts?"
        ),
        (
            "P04",
            "Show the distribution of ticket priorities for "
            "tickets in the West region."
        ),
    ],
    2: [
        (
            "P05",
            "What is the average resolution time in hours "
            "for each support team, considering only Resolved tickets?"
        ),
        (
            "P06",
            "Compare the average customer satisfaction score "
            "between Resolved and Open tickets."
        ),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=int,
        choices=[1, 2],
        default=1,
        help="Run batch 1 (four questions) or batch 2 (two questions).",
    )
    args = parser.parse_args()

    df = build_support_dataset()
    run_portability_safeguard_checks(df)

    print("=" * 65)
    print("DATASENSE AI - CROSS-DATASET PORTABILITY TEST")
    print("=" * 65)
    print("Dataset: customer_support_tickets")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Batch:", args.batch)
    print("Excluded column: internal_notes")
    print()

    # Independent reference calculations.
    print("INDEPENDENT REFERENCE CHECKS")
    print("-" * 65)
    print("Date range:", df["created_at"].min(), "to", df["created_at"].max())
    print("\nSupport team counts:")
    print(df["support_team"].value_counts().to_string())

    west = df[df["region"] == "West"]
    print("\nPriority counts in West:")
    print(west["priority"].value_counts().to_string())

    resolved = df[df["status"] == "Resolved"]
    print("\nAverage resolution hours by team (Resolved only):")
    print(
        resolved.groupby("support_team")["resolution_hours"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\nAverage satisfaction by status:")
    print(
        df.groupby("status")["satisfaction_score"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\n" + "=" * 65)
    print("AGENT RESULTS")
    print("=" * 65)

    for question_id, question in QUESTIONS[args.batch]:
        print(f"\n{question_id}: {question}")

        result = run_agent(
            question,
            verbose=False,
            df=df,
            dataset_config=DATASET_CONFIG,
            dataset_name="customer_support_tickets",
        )

        print("\nAnswer:")
        print(result["final_answer"])

        print("\nSelected tools:")
        print(result["selected_tools"])

        print("Tool calls:", result["tool_call_count"])
        print("Completed:", result["completed"])
        print("Execution errors:", result["execution_errors"])

        print("-" * 65)


if __name__ == "__main__":
    main()
