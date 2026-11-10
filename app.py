
import re
from copy import deepcopy
from pathlib import Path

import gradio as gr
import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfApi

from profiler import profile_dataset
from agent.runner import run_agent
from data_connector.huggingface_loader import load_huggingface_dataset
from dataset_context.dataset_configs import ECOMMERCE_CONFIG


# --------------------------------------------------
# DATASET SAFEGUARD CONFIGURATION
# --------------------------------------------------

MAX_UPLOAD_SIZE_MB = 100
MAX_ROWS = 1_000_000
MAX_COLUMNS = 200
MAX_DATASET_MEMORY_MB = 512

# Apply this configuration only to this known repository.
KNOWN_ECOMMERCE_REPOSITORY = (
    "davidev07/synthetic-ecommerce-dataset"
)


# --------------------------------------------------
# 1. DATASET CONFIGURATION
# --------------------------------------------------

def get_dataset_config(df, source_type=None, repo_id=None):
    """
    Return a configuration appropriate for the loaded dataset.

    Unknown datasets receive an empty, generic configuration.
    The known e-commerce repository receives its matching
    configuration, filtered to columns actually present.
    """

    if (
        source_type != "huggingface"
        or repo_id != KNOWN_ECOMMERCE_REPOSITORY
    ):
        return {}

    available_columns = set(df.columns)

    config = deepcopy(ECOMMERCE_CONFIG)

    # Keep only metadata that references existing columns.
    config["semantic_types"] = {
        column: semantic_type
        for column, semantic_type
        in config.get("semantic_types", {}).items()
        if column in available_columns
    }

    config["column_descriptions"] = {
        column: description
        for column, description
        in config.get("column_descriptions", {}).items()
        if column in available_columns
    }

    config["exclude_from_context"] = [
        column
        for column in config.get("exclude_from_context", [])
        if column in available_columns
    ]

    currency_column = config.get("currency_column")

    if currency_column not in available_columns:
        config.pop("currency_column", None)

    return config


# --------------------------------------------------
# 2. RESET ACTIVE DATASET
# --------------------------------------------------

def clear_dataset():
    """
    Clear all active dataset-related session state.

    This includes the DataFrame, dataset name, configuration,
    conversation history, chatbot and analytical results.
    """

    return (
        "",
        pd.DataFrame(),
        None,
        None,
        {},
        [],
        [],
        pd.DataFrame(),
    )


# --------------------------------------------------
# 3. SHARED DATASET VALIDATION AND PROFILING
# --------------------------------------------------

def load_and_profile_dataset(
    df,
    dataset_name,
    file_size_mb=None,
    dataset_config=None,
):
    """Validate and profile a loaded DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "The loaded dataset is not a valid DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The selected dataset is empty. "
            "Please select a dataset containing data rows."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Rename them before loading."
        )

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"The dataset exceeds the maximum allowed "
            f"of {MAX_ROWS:,} rows."
        )

    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(
            f"The dataset contains {len(df.columns):,} columns. "
            f"The maximum allowed is {MAX_COLUMNS:,}."
        )

    memory_mb = (
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    )

    if memory_mb > MAX_DATASET_MEMORY_MB:
        raise ValueError(
            f"The loaded dataset uses {memory_mb:.2f} MB "
            f"of memory. The maximum allowed is "
            f"{MAX_DATASET_MEMORY_MB} MB."
        )

    # Keep a separate copy of the configuration for this session.
    active_config = deepcopy(dataset_config or {})

    # Profile the validated dataset.
    profile = profile_dataset(df)

    display_profile = profile.copy()

    if "top_values" in display_profile.columns:
        display_profile["top_values"] = (
            display_profile["top_values"].apply(
                lambda x: str(x) if isinstance(x, dict) else x
            )
        )

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    if file_size_mb is None:
        file_size_display = "Hugging Face download"
    else:
        file_size_display = f"{file_size_mb:.2f} MB"

    excluded_count = len(
        active_config.get("exclude_from_context", [])
    )

    summary = f"""
### Dataset Overview

- **Dataset:** {dataset_name}
- **Rows:** {len(df):,}
- **Columns:** {len(df.columns):,}
- **File size:** {file_size_display}
- **Memory usage:** {memory_mb:.2f} MB
- **Total missing values:** {missing_values:,}
- **Duplicate rows:** {duplicate_rows:,}
- **Configured excluded columns:** {excluded_count}

Your dataset passed validation and is ready for analysis.
"""

    return (
        summary,
        display_profile,
        df,
        dataset_name,
        active_config,
        [],
        [],
        pd.DataFrame(),
    )


# --------------------------------------------------
# 4. LOAD LOCAL CSV OR PARQUET
# --------------------------------------------------

def analyze_file(file_path):
    """Load, validate and profile an uploaded dataset."""

    if file_path is None:
        raise gr.Error(
            "Please upload a CSV or Parquet file."
        )

    file_path = Path(file_path)

    try:
        suffix = file_path.suffix.lower()

        if suffix not in [".csv", ".parquet", ".pq"]:
            raise ValueError(
                "Unsupported file format. Upload CSV or Parquet."
            )

        file_size_mb = (
            file_path.stat().st_size / (1024 ** 2)
        )

        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"File size is {file_size_mb:.2f} MB. "
                f"The maximum allowed upload size is "
                f"{MAX_UPLOAD_SIZE_MB} MB."
            )

        # CSV
        if suffix == ".csv":

            header = pd.read_csv(file_path, nrows=0)

            column_count = len(header.columns)

            if column_count > MAX_COLUMNS:
                raise ValueError(
                    f"The dataset contains {column_count:,} columns. "
                    f"The maximum allowed is {MAX_COLUMNS:,}."
                )

            df = pd.read_csv(
                file_path,
                nrows=MAX_ROWS + 1,
            )

            if len(df) > MAX_ROWS:
                raise ValueError(
                    f"The dataset exceeds the maximum allowed "
                    f"of {MAX_ROWS:,} rows."
                )

        # PARQUET
        else:

            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata

            if metadata.num_rows > MAX_ROWS:
                raise ValueError(
                    f"The dataset contains {metadata.num_rows:,} rows. "
                    f"The maximum allowed is {MAX_ROWS:,}."
                )

            if metadata.num_columns > MAX_COLUMNS:
                raise ValueError(
                    f"The dataset contains {metadata.num_columns:,} columns. "
                    f"The maximum allowed is {MAX_COLUMNS:,}."
                )

            df = pd.read_parquet(file_path)

        # Local uploads use generic configuration by default.
        dataset_config = get_dataset_config(
            df,
            source_type="local",
        )

        return load_and_profile_dataset(
            df=df,
            dataset_name=file_path.stem,
            file_size_mb=file_size_mb,
            dataset_config=dataset_config,
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(
            f"Unable to load or profile dataset: {str(e)}"
        )


# --------------------------------------------------
# 5. LIST HUGGING FACE REPOSITORY FILES
# --------------------------------------------------

def list_huggingface_files(repo_id):
    """List supported CSV and Parquet files in a dataset repository."""

    if not repo_id or not repo_id.strip():
        raise gr.Error(
            "Please enter a Hugging Face dataset repository ID."
        )

    try:
        api = HfApi()

        files = api.list_repo_files(
            repo_id=repo_id.strip(),
            repo_type="dataset",
        )

        supported_files = sorted(
            filename
            for filename in files
            if filename.lower().endswith(
                (".csv", ".parquet", ".pq")
            )
        )

        if not supported_files:
            raise gr.Error(
                "No supported CSV or Parquet files were found "
                "in this repository."
            )

        return gr.update(
            choices=supported_files,
            value=supported_files[0],
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(
            f"Unable to list Hugging Face repository files: {str(e)}"
        )


# --------------------------------------------------
# 6. LOAD HUGGING FACE DATASET
# --------------------------------------------------

def analyze_huggingface(repo_id, filename):
    """Download, validate and profile a Hugging Face dataset."""

    if not repo_id or not repo_id.strip():
        raise gr.Error(
            "Please enter a Hugging Face dataset repository ID."
        )

    if not filename or not filename.strip():
        raise gr.Error(
            "Please select a CSV or Parquet filename."
        )

    try:
        repo_id = repo_id.strip()
        filename = filename.strip()

        df = load_huggingface_dataset(
            repo_id=repo_id,
            filename=filename,
        )

        dataset_name = f"{repo_id}/{Path(filename).name}"

        dataset_config = get_dataset_config(
            df,
            source_type="huggingface",
            repo_id=repo_id,
        )

        return load_and_profile_dataset(
            df=df,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(
            f"Unable to load Hugging Face dataset: {str(e)}"
        )


# --------------------------------------------------
# 7. BUILD DETERMINISTIC ANALYTICAL RESULTS
# --------------------------------------------------

def build_result_table(result, question=""):
    """
    Build a presentation-friendly DataFrame from the latest
    successful group_and_aggregate tool result.

    All displayed values come directly from Python.
    """

    tool_results = result.get("tool_results", [])

    for tool_result in reversed(tool_results):

        if (
            tool_result.get("tool") != "group_and_aggregate"
            or not tool_result.get("success")
        ):
            continue

        python_result = tool_result.get("result", {})
        rows = python_result.get("results", [])

        if not rows:
            return pd.DataFrame()

        table = pd.DataFrame(rows)
        arguments = tool_result.get("arguments", {})

        group_by = arguments.get("group_by")
        granularity = arguments.get("date_granularity")
        aggregation = arguments.get("aggregation")
        metric = arguments.get("metric")

        # FRIENDLY GROUPING COLUMN NAME
        if group_by in table.columns:

            if granularity:

                group_label = granularity.capitalize()

                if group_by == "signup_date":
                    group_label = f"Registration {group_label}"

                elif group_by == "created_at":
                    group_label = f"Created {group_label}"

                elif group_by == "purchase_date":
                    group_label = f"Purchase {group_label}"

            else:
                group_label = group_by.replace("_", " ").title()

            table = table.rename(
                columns={group_by: group_label}
            )

        # FRIENDLY RESULT COLUMN NAME
        if "result" in table.columns:

            question_lower = question.lower()

            if aggregation == "count" and metric is None:

                if (
                    "user" in question_lower
                    and (
                        "registr" in question_lower
                        or "signup" in question_lower
                    )
                ):
                    result_label = "User Registrations"

                elif "transaction" in question_lower:
                    result_label = "Transactions"

                elif "order" in question_lower:
                    result_label = "Orders"

                else:
                    result_label = "Count"

            elif metric:
                result_label = (
                    f"{str(aggregation).title()} "
                    f"{metric.replace('_', ' ').title()}"
                )

            else:
                result_label = "Result"

            table = table.rename(
                columns={"result": result_label}
            )

        return table

    return pd.DataFrame()


# --------------------------------------------------
# 8. REMOVE LLM-GENERATED TABLE CONTENT
# --------------------------------------------------

def remove_llm_table_content(answer, result):
    """
    Remove LLM-generated Markdown tables and date-based
    CSV-style result rows.

    The actual results are displayed in a separate DataFrame.
    """

    if not answer:
        return ""

    tool_results = result.get("tool_results", [])

    has_date_aggregation = any(
        item.get("tool") == "group_and_aggregate"
        and item.get("success")
        and item.get("arguments", {}).get("date_granularity")
        for item in tool_results
    )

    cleaned_lines = []

    date_result_pattern = re.compile(
        r"^\s*(?:"
        r"\d{4}-\d{2}(?:-\d{2})?"
        r"|\d{4}Q[1-4]"
        r"|\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}"
        r")\s*,\s*-?\d+(?:\.\d+)?\s*$"
    )

    for line in answer.splitlines():

        stripped = line.strip()

        if stripped.startswith("|"):
            continue

        if has_date_aggregation and date_result_pattern.match(line):
            continue

        cleaned_lines.append(line)

    cleaned_answer = "\n".join(cleaned_lines)

    cleaned_answer = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_answer,
    )

    return cleaned_answer.strip()


# --------------------------------------------------
# 9. ANALYTICS CHATBOT
# --------------------------------------------------

def chat_with_dataset(
    question,
    chat_history,
    active_df,
    conversation_history,
    dataset_name,
    dataset_config,
):
    """Answer questions using the currently loaded dataset."""

    if active_df is None:
        raise gr.Error(
            "Please upload or load a dataset first."
        )

    if not question or not question.strip():
        raise gr.Error("Please enter a question.")

    chat_history = chat_history or []
    conversation_history = conversation_history or []
    dataset_config = dataset_config or {}

    try:
        result = run_agent(
            question=question.strip(),
            df=active_df,
            dataset_config=deepcopy(dataset_config),
            dataset_name=dataset_name or "Loaded dataset",
            conversation_history=conversation_history,
            verbose=False,
        )

        if not result["completed"] or not result["final_answer"]:
            raise gr.Error(
                "The agent could not complete the analysis. "
                "Please try rephrasing your question."
            )

        result_table = build_result_table(
            result,
            question=question,
        )

        answer = result["final_answer"]

        if not result_table.empty:
            answer = remove_llm_table_content(
                answer,
                result,
            )

        updated_chat = chat_history + [
            {
                "role": "user",
                "content": question.strip(),
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        updated_conversation_history = (
            result["conversation_history"]
        )

        if (
            updated_conversation_history
            and updated_conversation_history[-1]["role"] == "assistant"
        ):
            updated_conversation_history[-1] = {
                "role": "assistant",
                "content": answer,
            }

        return (
            updated_chat,
            updated_conversation_history,
            "",
            result_table,
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(
            f"Unable to answer your question: {str(e)}"
        )


# --------------------------------------------------
# 10. GRADIO APPLICATION
# --------------------------------------------------

with gr.Blocks(title="DataSense AI") as app:

    # Session-specific state.
    dataset_state = gr.State(value=None)
    dataset_name_state = gr.State(value=None)
    dataset_config_state = gr.State(value={})
    conversation_state = gr.State(value=[])

    gr.Markdown(
        """
        # DataSense AI
        ### Intelligent Data Discovery & Analytics Assistant

        Load a dataset to explore its structure, discover
        data quality issues and ask questions using natural language.
        """
    )

    # DATASET EXPLORER
    with gr.Tab("Dataset Explorer"):

        with gr.Tab("Upload File"):

            gr.Markdown("### Upload CSV or Parquet")

            file_input = gr.File(
                label="Upload CSV or Parquet",
                file_types=[".csv", ".parquet", ".pq"],
                type="filepath",
            )

            analyze_button = gr.Button(
                "Analyze Dataset",
                variant="primary",
            )

        with gr.Tab("Hugging Face Dataset"):

            gr.Markdown(
                """
                ### Load a dataset from Hugging Face

                Enter a public Hugging Face dataset repository ID.
                List its available CSV and Parquet files, then select
                the file you want to analyze.

                Example repository:
                `davidev07/synthetic-ecommerce-dataset`
                """
            )

            hf_repo_input = gr.Textbox(
                label="Hugging Face Dataset Repository ID",
                placeholder="username/dataset-name",
            )

            hf_list_button = gr.Button(
                "List Available Files",
            )

            hf_filename_input = gr.Dropdown(
                label="Select Dataset File",
                choices=[],
                value=None,
                interactive=True,
            )

            hf_load_button = gr.Button(
                "Load Hugging Face Dataset",
                variant="primary",
            )

        summary_output = gr.Markdown(
            label="Dataset Overview"
        )

        profile_output = gr.Dataframe(
            label="Column Profiling Report",
            interactive=False,
            wrap=True,
        )

    # ANALYTICS CHATBOT
    with gr.Tab("Ask DataSense AI"):

        gr.Markdown(
            """
            ### Ask questions about your dataset

            Ask questions about trends, distributions,
            comparisons, aggregations and other analytical insights.
            """
        )

        chatbot = gr.Chatbot(
            label="DataSense AI",
            height=450,
        )

        result_table_output = gr.Dataframe(
            label="Analytical Results",
            interactive=False,
            wrap=True,
        )

        with gr.Row():

            question_input = gr.Textbox(
                label="Your question",
                placeholder=(
                    "e.g. Which team has the highest average resolution time?"
                ),
                lines=2,
                scale=5,
            )

            send_button = gr.Button(
                "Send",
                variant="primary",
                scale=1,
            )

    # SHARED DATASET AND UI OUTPUTS
    dataset_outputs = [
        summary_output,
        profile_output,
        dataset_state,
        dataset_name_state,
        dataset_config_state,
        conversation_state,
        chatbot,
        result_table_output,
    ]

    # EVENT: LOCAL FILE UPLOAD
    analyze_button.click(
        fn=clear_dataset,
        inputs=[],
        outputs=dataset_outputs,
    ).then(
        fn=analyze_file,
        inputs=[file_input],
        outputs=dataset_outputs,
    )

    # EVENT: LIST HUGGING FACE FILES
    hf_list_button.click(
        fn=list_huggingface_files,
        inputs=[hf_repo_input],
        outputs=[hf_filename_input],
    )

    # EVENT: LOAD HUGGING FACE DATASET
    hf_load_button.click(
        fn=clear_dataset,
        inputs=[],
        outputs=dataset_outputs,
    ).then(
        fn=analyze_huggingface,
        inputs=[
            hf_repo_input,
            hf_filename_input,
        ],
        outputs=dataset_outputs,
    )

    # EVENT: SEND CHAT MESSAGE
    chat_inputs = [
        question_input,
        chatbot,
        dataset_state,
        conversation_state,
        dataset_name_state,
        dataset_config_state,
    ]

    chat_outputs = [
        chatbot,
        conversation_state,
        question_input,
        result_table_output,
    ]

    send_button.click(
        fn=chat_with_dataset,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )

    question_input.submit(
        fn=chat_with_dataset,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )


# --------------------------------------------------
# 11. LAUNCH APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.launch()
