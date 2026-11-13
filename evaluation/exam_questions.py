
"""
DataSense AI - Initial Evaluation Exam Set

15 questions covering different analytical capabilities.

The expected_tools field describes the expected tool family.
It is not necessarily the exact sequence of tool calls.
"""

EXAM_QUESTIONS = [

    # --------------------------------------------------
    # A. DATASET UNDERSTANDING
    # --------------------------------------------------

    {
        "id": "Q01",
        "category": "dataset_understanding",
        "question": (
            "How many rows and columns are in the dataset? "
            "Are there any duplicate rows?"
        ),
        "expected_tools": [],
        "difficulty": "easy",
    },

    {
        "id": "Q02",
        "category": "dataset_understanding",
        "question": (
            "Which columns are available for analysis, and "
            "which columns have been excluded from agent access?"
        ),
        "expected_tools": [],
        "difficulty": "easy",
    },

    # --------------------------------------------------
    # B. COLUMN PROFILING
    # --------------------------------------------------

    {
        "id": "Q03",
        "category": "column_profiling",
        "question": (
            "What is the date range covered by the purchase_date column?"
        ),
        "expected_tools": [],
        "difficulty": "easy",
    },

    {
        "id": "Q04",
        "category": "column_profiling",
        "question": (
            "What are the minimum, maximum, mean and median "
            "values of quantity?"
        ),
        "expected_tools": ["get_column_summary"],
        "difficulty": "easy",
    },

    # --------------------------------------------------
    # C. CATEGORY DISCOVERY
    # --------------------------------------------------

    {
        "id": "Q05",
        "category": "category_discovery",
        "question": (
            "What unique values exist in the order_status column, "
            "and how many transactions belong to each status?"
        ),
        "expected_tools": ["get_categorical_values"],
        "difficulty": "easy",
    },

    {
        "id": "Q06",
        "category": "category_discovery",
        "question": (
            "What payment methods are available in the dataset?"
        ),
        "expected_tools": ["get_categorical_values"],
        "difficulty": "easy",
    },

    # --------------------------------------------------
    # D. DISTRIBUTION ANALYSIS
    # --------------------------------------------------

    {
        "id": "Q07",
        "category": "distribution_analysis",
        "question": (
            "What is the distribution of payment methods "
            "across all transactions? Show counts and percentages."
        ),
        "expected_tools": ["get_value_distribution"],
        "difficulty": "medium",
    },

    {
        "id": "Q08",
        "category": "distribution_analysis",
        "question": (
            "Among cancelled transactions, which payment methods "
            "are most common? Show counts and percentages."
        ),
        "expected_tools": ["get_value_distribution"],
        "difficulty": "medium",
    },

    # --------------------------------------------------
    # E. AGGREGATION AND FILTERING
    # --------------------------------------------------

    {
        "id": "Q09",
        "category": "aggregation_and_filtering",
        "question": (
            "How many transactions are there for each order status?"
        ),
        "expected_tools": ["group_and_aggregate"],
        "difficulty": "medium",
    },

    {
        "id": "Q10",
        "category": "aggregation_and_filtering",
         "question": (
            "Which five countries have the highest number "
            "of transactions? For each country, report "
            "the transaction count and its percentage "
            "of all transactions."
        ),
        "expected_tools": ["get_value_distribution"],
        "difficulty": "medium",
    },

    {
        "id": "Q11",
        "category": "aggregation_and_filtering",
        "question": (
            "What is the average total_amount for delivered "
            "transactions, broken down by currency?"
        ),
        "expected_tools": ["group_and_aggregate"],
        "difficulty": "medium",
    },

    # --------------------------------------------------
    # F. GROUP COMPARISON
    # --------------------------------------------------

    {
        "id": "Q12",
        "category": "group_comparison",
        "question": (
            "Compare the average total_amount of Delivered "
            "and Cancelled transactions, separately for each "
            "currency. Show the absolute and percentage differences."
        ),
        "expected_tools": ["compare_groups"],
        "difficulty": "hard",
    },

    {
        "id": "Q13",
        "category": "group_comparison",
        "question": (
            "Compare the number of transactions with payment_status "
            "equal to Completed versus Failed."
        ),
        "expected_tools": ["compare_groups"],
        "difficulty": "medium",
    },

    # --------------------------------------------------
    # G. MULTI-STEP REASONING
    # --------------------------------------------------

    {
        "id": "Q14",
        "category": "multi_step_reasoning",
        "question": (
            "Identify the two most common order statuses. "
            "Then compare their average transaction amounts "
            "separately by currency. For each currency, report "
            "both group averages, the absolute difference "
            "(second status minus first status), and the "
            "percentage difference relative to the first "
            "status average."
        ),
        "expected_tools": [
            {"get_value_distribution"},
            {"compare_groups"},
        ],
        "difficulty": "hard",
    },

    {
        "id": "Q15",
        "category": "multi_step_reasoning",
        "question": (
            "Which country has the highest number of cancelled "
            "transactions? What percentage of all cancelled "
            "transactions does that country represent?"
        ),
        "expected_tools": [
            "group_and_aggregate",
        ],
        "difficulty": "hard",
    },

]


# --------------------------------------------------
# EVALUATION METADATA
# --------------------------------------------------

EXAM_METADATA = {
    "project": "DataSense AI",
    "exam_version": "1.0",
    "dataset": "davidev07/synthetic-ecommerce-dataset",
    "dataset_file": "transactions.csv",
    "total_questions": len(EXAM_QUESTIONS),
    "categories": [
        "dataset_understanding",
        "column_profiling",
        "category_discovery",
        "distribution_analysis",
        "aggregation_and_filtering",
        "group_comparison",
        "multi_step_reasoning",
    ],
}


# --------------------------------------------------
# BASIC VALIDATION
# --------------------------------------------------

def validate_exam_questions():

    question_ids = [
        question["id"]
        for question in EXAM_QUESTIONS
    ]

    assert len(question_ids) == len(set(question_ids)), (
        "Duplicate question IDs found."
    )

    assert len(EXAM_QUESTIONS) == EXAM_METADATA["total_questions"], (
        "Exam question count does not match metadata."
    )

    for question in EXAM_QUESTIONS:

        required_fields = {
            "id",
            "category",
            "question",
            "expected_tools",
            "difficulty",
        }

        assert required_fields.issubset(question.keys()), (
            f"Missing fields in {question.get('id')}"
        )

        assert isinstance(question["expected_tools"], list), (
            f"expected_tools must be a list for {question['id']}"
        )

    return True


if __name__ == "__main__":

    validate_exam_questions()

    print("DataSense AI Evaluation Exam")
    print("----------------------------")

    print(
        "Total questions:",
        EXAM_METADATA["total_questions"]
    )

    print(
        "Categories:",
        len(EXAM_METADATA["categories"])
    )

    print("\nQuestions:")

    for question in EXAM_QUESTIONS:

        print(
            f"{question['id']} | "
            f"{question['category']} | "
            f"{question['difficulty']} | "
            f"{question['question']}"
        )

    print("\nExam validation: PASSED")
