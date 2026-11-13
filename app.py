import uuid
import re
from copy import deepcopy
from pathlib import Path
import logging
import gradio as gr
import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfApi

from profiler import profile_dataset
from agent.runner import run_agent
from data_connector.huggingface_loader import load_huggingface_dataset
from dataset_context.dataset_configs import ECOMMERCE_CONFIG

from exceptions import UserFacingError
import session_store

# --------------------------------------------------
# APPLICATION LOGGING
# --------------------------------------------------

logger = logging.getLogger("datasense_ai.app")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False
# --------------------------------------------------
# SESSION STORAGE
# --------------------------------------------------

session_store.initialize_db()

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
        raise UserFacingError(
            "The loaded dataset is not a valid DataFrame."
        )

    if df.empty:
        raise UserFacingError(
            "The selected dataset is empty. "
            "Please select a dataset containing data rows."
        )

    if not df.columns.is_unique:
        raise UserFacingError(
            "The dataset contains duplicate column names. "
            "Rename them before loading."
        )

    if len(df) > MAX_ROWS:
        raise UserFacingError(
            f"The dataset exceeds the maximum allowed "
            f"of {MAX_ROWS:,} rows."
        )

    if len(df.columns) > MAX_COLUMNS:
        raise UserFacingError(
            f"The dataset contains {len(df.columns):,} columns. "
            f"The maximum allowed is {MAX_COLUMNS:,}."
        )

    memory_mb = (
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    )

    if memory_mb > MAX_DATASET_MEMORY_MB:
        raise UserFacingError(
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

def analyze_file(file_path, workspace_id, session_id):
    """Load, validate and profile an uploaded dataset."""

    if file_path is None:
        raise gr.Error(
            "Please upload a CSV or Parquet file."
        )
    if not workspace_id or not session_id:
        raise gr.Error(
            "Please initialize a session before loading a dataset."
        )

    file_path = Path(file_path)

    try:
        suffix = file_path.suffix.lower()

        if suffix not in [".csv", ".parquet", ".pq"]:
            raise UserFacingError(
                "Unsupported file format. Upload CSV or Parquet."
            )

        file_size_mb = (
            file_path.stat().st_size / (1024 ** 2)
        )

        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            raise UserFacingError(
                f"File size is {file_size_mb:.2f} MB. "
                f"The maximum allowed upload size is "
                f"{MAX_UPLOAD_SIZE_MB} MB."
            )

        # CSV
        if suffix == ".csv":

            header = pd.read_csv(file_path, nrows=0)

            column_count = len(header.columns)

            if column_count > MAX_COLUMNS:
                raise UserFacingError(
                    f"The dataset contains {column_count:,} columns. "
                    f"The maximum allowed is {MAX_COLUMNS:,}."
                )

            df = pd.read_csv(
                file_path,
                nrows=MAX_ROWS + 1,
            )

            if len(df) > MAX_ROWS:
                raise UserFacingError(
                    f"The dataset exceeds the maximum allowed "
                    f"of {MAX_ROWS:,} rows."
                )

        # PARQUET
        else:

            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata

            if metadata.num_rows > MAX_ROWS:
                raise UserFacingError(
                    f"The dataset contains {metadata.num_rows:,} rows. "
                    f"The maximum allowed is {MAX_ROWS:,}."
                )

            if metadata.num_columns > MAX_COLUMNS:
                raise UserFacingError(
                    f"The dataset contains {metadata.num_columns:,} columns. "
                    f"The maximum allowed is {MAX_COLUMNS:,}."
                )

            df = pd.read_parquet(file_path)

        # Local uploads use generic configuration by default.
        dataset_config = get_dataset_config(
            df,
            source_type="local",
        )

        outputs = load_and_profile_dataset(
            df=df,
            dataset_name=file_path.stem,
            file_size_mb=file_size_mb,
            dataset_config=dataset_config,
        )

        # Persist the validated upload in the active session.
        try:
            session_store.save_uploaded_dataset(
                workspace_id=workspace_id,
                session_id=session_id,
                source_path=file_path,
                dataset_config=outputs[4],
            )
        except ValueError as e:
            raise gr.Error(str(e)) from None

        return outputs

    except UserFacingError as e:
        raise gr.Error(str(e)) from None

    except gr.Error:
        raise

    except Exception:
        logger.exception(
            "Unexpected error while loading local dataset."
        )
        raise gr.Error(
            "Unable to load or profile the dataset due to an "
            "unexpected error. Please check your file and try again."
        ) from None


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
            value=None,
        )

    except gr.Error:
        raise

    except Exception:
        logger.exception(
            "Unexpected error while listing Hugging Face repository files."
        )
        raise gr.Error(
            "Unable to list files from the Hugging Face repository. "
            "Please verify the repository ID and try again."
        ) from None
# --------------------------------------------------
# 6. LOAD HUGGING FACE DATASET
# --------------------------------------------------


def analyze_huggingface(
    repo_id,
    filename,
    workspace_id,
    session_id,
):
    """Download, validate and profile a Hugging Face dataset."""

    if not repo_id or not repo_id.strip():
        raise gr.Error(
            "Please enter a Hugging Face dataset repository ID."
        )

    if not filename or not filename.strip():
        raise gr.Error(
            "Please select a CSV or Parquet filename."
        )

    if not workspace_id or not session_id:
        raise gr.Error(
            "Please initialize a session before loading a dataset."
        )

    try:
        repo_id = repo_id.strip()
        filename = filename.strip()

        df, metadata = load_huggingface_dataset(
            repo_id=repo_id,
            filename=filename,
            return_metadata=True,
        )

        dataset_name = f"{repo_id}/{Path(filename).name}"

        dataset_config = get_dataset_config(
            df,
            source_type="huggingface",
            repo_id=repo_id,
        )

        outputs = load_and_profile_dataset(
            df=df,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )

        # Persist the exact Hugging Face repository revision.
        try:
            session_store.save_huggingface_reference(
                workspace_id=workspace_id,
                session_id=session_id,
                metadata=metadata,
                dataset_config=outputs[4],
            )
        except ValueError as e:
            raise gr.Error(str(e)) from None

        return outputs

    except UserFacingError as e:
        raise gr.Error(str(e)) from None

    except gr.Error:
        raise

    except Exception:
        logger.exception(
            "Unexpected error while loading Hugging Face dataset."
        )
        raise gr.Error(
            "Unable to load the Hugging Face dataset. "
            "Please verify the repository and selected file, "
            "then try again."
        ) from None



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


def build_ranking_summary(question, result_table):
    """Generate a deterministic summary for ranking questions."""

    if result_table is None or result_table.empty:
        return ""

    ranking_terms = (
        "top",
        "highest",
        "most",
        "best",
        "maximum",
        "largest",
    )

    question_lower = question.lower()

    if not any(term in question_lower for term in ranking_terms):
        return ""

    numeric_columns = result_table.select_dtypes(
        include="number"
    ).columns.tolist()

    # Only summarize unambiguous single-metric results.
    if len(numeric_columns) != 1:
        return ""

    result_column = numeric_columns[0]

    group_columns = [
        column
        for column in result_table.columns
        if column != result_column
    ]

    if len(group_columns) != 1:
        return ""

    group_column = group_columns[0]

    valid_results = result_table.dropna(
        subset=[result_column, group_column]
    )

    if valid_results.empty:
        return ""

    maximum = valid_results[result_column].max()

    winners = valid_results.loc[
        valid_results[result_column] == maximum,
        group_column,
    ].tolist()

    def format_group(value):
        value = str(value)

        # Convert YYYY-MM into a readable month.
        if re.fullmatch(r"\d{4}-\d{2}", value):
            try:
                return pd.to_datetime(
                    value,
                    format="%Y-%m",
                ).strftime("%B %Y")
            except ValueError:
                pass

        return value

    formatted_maximum = (
        f"{maximum:,.2f}"
        if isinstance(maximum, float)
        else f"{maximum:,}"
    )

    metric_label = result_column.lower()

    if len(winners) == 1:
        winner = format_group(winners[0])

        return (
            f"**Highest {group_column}:** "
            f"{winner} — **{formatted_maximum} "
            f"{metric_label}.**"
        )

    if len(winners) == len(valid_results):
        return (
            f"**No single top {group_column.lower()}:** "
            f"all {len(winners)} groups are tied at "
            f"**{formatted_maximum} {metric_label} each.**"
        )

    displayed_winners = [
        format_group(value)
        for value in winners[:5]
    ]

    winner_text = ", ".join(displayed_winners)

    if len(winners) > 5:
        winner_text += f", and {len(winners) - 5} more"

    return (
        f"**Highest {group_column}:** "
        f"{winner_text} — tied at "
        f"**{formatted_maximum} {metric_label}.**"
    )


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
    workspace_id,
    session_id,
):
    """Answer questions using the currently loaded dataset."""
    if not workspace_id or not session_id:
        raise gr.Error(
            "Please initialize a session before sending a question."
        )
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

        logger.info(
            "Agent performance | Duration: %.3fs | "
            "API attempts: %d | Prompt tokens: %s | "
            "Completion tokens: %s | Tool calls: %d",
            result["execution_time_seconds"],
            result["api_attempts"],
            result["prompt_tokens"],
            result["completion_tokens"],
            result["tool_call_count"],
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

        ranking_summary = build_ranking_summary(
            question,
            result_table,
        )

        if ranking_summary and not answer:
            answer = ranking_summary

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
        # Persist the successful exchange in the active session.
        session_store.add_message(
            workspace_id=workspace_id,
            session_id=session_id,
            role="user",
            content=question.strip(),
        )

        session_store.add_message(
            workspace_id=workspace_id,
            session_id=session_id,
            role="assistant",
            content=answer,
        )
        return (
            updated_chat,
            updated_conversation_history,
            "",
            result_table,
            gr.update(
                choices=get_session_choices(workspace_id),
                value=session_id,
            ),
        )

    except UserFacingError as e:
        raise gr.Error(str(e)) from None

    except gr.Error:
        raise

    except Exception:
        logger.exception(
            "Unexpected error during dataset question processing."
        )
        raise gr.Error(
            "Unable to answer your question due to an unexpected "
            "error. Please try again."
        ) from None

def get_session_choices(workspace_id):
    """Return the 10 most recent sessions for the sidebar."""

    sessions = session_store.list_sessions(workspace_id)

    return [
        (
            f"{item['name']} · {item['message_count']} messages",
            item["session_id"],
        )
        for item in sessions[:10]
    ]


def initialize_workspace(workspace_id):
    if not workspace_id:
        workspace_id = uuid.uuid4().hex

    sessions = session_store.list_sessions(workspace_id)

    if not sessions:
        active_session_id = session_store.create_session(
            workspace_id
        )
    else:
        active_session_id = sessions[0]["session_id"]

    choices = get_session_choices(workspace_id)

    return (
        workspace_id,
        active_session_id,
        gr.update(
            choices=choices,
            value=active_session_id,
        ),
    )

def load_session(workspace_id, session_id):
    """Restore a session's saved dataset and chat history."""

    if not workspace_id or not session_id:
        raise gr.Error(
            "Please select a valid session."
        )

    try:
        session = session_store.get_session(
            workspace_id,
            session_id,
        )

        if session is None:
            raise gr.Error(
                "The selected session could not be found."
            )

        # Restore the persisted user and assistant messages.
        stored_messages = session_store.get_messages(
            workspace_id,
            session_id,
        )

        messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in stored_messages
        ]

        outputs = list(clear_dataset())

        source_type = session.get("source_type")
        dataset_config = session.get("dataset_config") or {}

        # Restore a locally uploaded dataset.
        if source_type == "upload":
            dataset_path = session_store.get_dataset_path(
                workspace_id,
                session_id,
            )

            if dataset_path is None or not dataset_path.exists():
                raise UserFacingError(
                    "The saved dataset file for this session "
                    "could not be found."
                )

            suffix = dataset_path.suffix.lower()

            if suffix == ".csv":
                df = pd.read_csv(dataset_path)
            elif suffix in (".parquet", ".pq"):
                df = pd.read_parquet(dataset_path)
            else:
                raise UserFacingError(
                    "The saved dataset has an unsupported file format."
                )

            file_size_mb = (
                dataset_path.stat().st_size / (1024 ** 2)
            )

            outputs = load_and_profile_dataset(
                df=df,
                dataset_name=dataset_path.stem,
                file_size_mb=file_size_mb,
                dataset_config=dataset_config,
            )

        # Restore a Hugging Face dataset at its saved revision.
        elif source_type == "huggingface":
            repo_id = session.get("repo_id")
            filename = session.get("filename")
            revision = session.get("revision")

            if not repo_id or not filename:
                raise UserFacingError(
                    "The saved Hugging Face dataset reference "
                    "is incomplete."
                )

            df = load_huggingface_dataset(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )

            outputs = load_and_profile_dataset(
                df=df,
                dataset_name=f"{repo_id}/{Path(filename).name}",
                dataset_config=dataset_config,
            )

        elif source_type not in (None, ""):
            raise UserFacingError(
                "This session uses an unsupported dataset source."
            )

        # Restore conversation state and visible chat.
        outputs = list(outputs)
        outputs[5] = messages
        outputs[6] = messages

        # Analytical results are not persisted yet.
        outputs[7] = pd.DataFrame()

        return session_id, *outputs

    except UserFacingError as e:
        raise gr.Error(str(e)) from None

    except gr.Error:
        raise

    except FileNotFoundError:
        logger.exception("Saved session dataset file was not found.")
        raise gr.Error(
            "The saved dataset file is unavailable. "
            "Please reload the dataset."
        ) from None

    except Exception:
        logger.exception(
            "Unexpected error while restoring a session."
        )
        raise gr.Error(
            "Unable to restore this session. Please try again."
        ) from None

def create_new_session(workspace_id):
    """Create a new session and switch the current tab to it."""

    if not workspace_id:
        raise gr.Error(
            "Workspace is not initialized. Please refresh the page."
        )

    try:
        new_session_id = session_store.create_session(workspace_id)

        choices = get_session_choices(workspace_id)

        cleared_outputs = clear_dataset()

        return (
            new_session_id,
            gr.update(
                choices=choices,
                value=new_session_id,
            ),
            *cleared_outputs,
        )

    except ValueError as e:
        raise gr.Error(str(e)) from None

    except Exception:
        logger.exception("Unexpected error while creating a new session.")
        raise gr.Error(
            "Unable to create a new session. Please try again."
        ) from None

def delete_current_session(workspace_id, session_id):
    """Delete the active session and switch to another session."""

    if not workspace_id or not session_id:
        raise gr.Error("Please select a valid session.")

    try:
        session_store.delete_session(workspace_id, session_id)

        remaining_sessions = session_store.list_sessions(workspace_id)

        if remaining_sessions:
            next_session_id = remaining_sessions[0]["session_id"]
            restored = load_session(workspace_id, next_session_id)
            dataset_outputs_restored = restored[1:]
        else:
            next_session_id = session_store.create_session(workspace_id)
            dataset_outputs_restored = clear_dataset()

        choices = get_session_choices(workspace_id)

        return (
            next_session_id,
            gr.update(
                choices=choices,
                value=next_session_id,
            ),
            *dataset_outputs_restored,
            gr.update(visible=False),
        )

    except gr.Error:
        raise

    except Exception:
        logger.exception("Unexpected error while deleting a session.")
        raise gr.Error(
            "Unable to delete this session. Please try again."
        ) from None


# --------------------------------------------------
# 10. GRADIO APPLICATION
# --------------------------------------------------

APP_CSS = """
.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
}

.workspace-layout {
    align-items: flex-start !important;
    gap: 18px !important;
}

.session-sidebar {
    flex: 0 0 260px !important;
    max-width: 260px !important;
    min-width: 220px !important;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px !important;
}

.main-panel {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

.source-options {
    align-items: stretch !important;
}

.source-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px !important;
    min-width: 0 !important;
}

@media (max-width: 768px) {
    .workspace-layout {
        flex-direction: column !important;
        gap: 12px !important;
    }

    .session-sidebar,
    .main-panel,
    .source-card {
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    .source-options {
        flex-direction: column !important;
    }

    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
}
"""

# Enter sends the question.
# Ctrl+Enter inserts a new line.
QUESTION_KEYBOARD_JS = r"""
<script>
document.addEventListener("keydown", (event) => {
    const textarea = event.target;

    if (!(textarea instanceof HTMLTextAreaElement)) return;
    if (!textarea.closest("#question-input")) return;
    if (event.key !== "Enter") return;

    if (event.ctrlKey) {
        event.preventDefault();

        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;

        textarea.setRangeText("\n", start, end, "end");
        textarea.dispatchEvent(
            new Event("input", { bubbles: true })
        );

        return;
    }

    if (!event.shiftKey) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();

        const sendContainer = document.querySelector(
            "#send-question"
        );

        const sendButton = sendContainer?.matches("button")
            ? sendContainer
            : sendContainer?.querySelector("button");

        if (sendButton && !sendButton.disabled) {
            sendButton.click();
        }
    }
}, true);
</script>
"""


with gr.Blocks(
    title="DataSense AI",
) as app:

    # --------------------------------------------------
    # SESSION AND DATASET STATE
    # --------------------------------------------------

    workspace_state = gr.BrowserState(
        default_value=None,
        storage_key="datasense_ai_workspace",
    )

    active_session_state = gr.State(value=None)

    dataset_state = gr.State(value=None)
    dataset_name_state = gr.State(value=None)
    dataset_config_state = gr.State(value={})
    conversation_state = gr.State(value=[])

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    gr.Markdown(
        """
        # DataSense AI

        Discover, understand and analyze your data using natural language.
        """
    )

    # --------------------------------------------------
    # MAIN WORKSPACE
    # --------------------------------------------------

    with gr.Row(elem_classes=["workspace-layout"]):

        # ==============================================
        # LEFT SIDEBAR: SESSION HISTORY
        # ==============================================

        with gr.Column(
            scale=0,
            min_width=220,
            elem_classes=["session-sidebar"],
        ):

            gr.Markdown("### Your workspace")

            new_session_button = gr.Button(
                "+ New Session",
                variant="primary",
                size="lg",
            )

            session_selector = gr.Radio(
                label="Recent sessions",
                choices=[],
                value=None,
                interactive=True,
            )

            gr.Markdown(
                "Select a session to restore its dataset and conversation."
            )

            gr.Markdown("---")

            delete_session_button = gr.Button(
                "Delete current session",
                variant="stop",
            )

            with gr.Row(
                visible=False,
                elem_id="delete-confirmation",
            ) as confirm_delete_group:

                confirm_delete_button = gr.Button(
                    "Confirm",
                    variant="stop",
                    size="sm",
                )

                cancel_delete_button = gr.Button(
                    "Cancel",
                    variant="secondary",
                    size="sm",
                )

        # ==============================================
        # RIGHT PANEL: DATA WORKSPACE
        # ==============================================

        with gr.Column(
            scale=1,
            min_width=300,
            elem_classes=["main-panel"],
        ):

            # ------------------------------------------
            # 1. DATA SOURCE
            # ------------------------------------------

            gr.Markdown("## Connect your data")

            with gr.Row(elem_classes=["source-options"]):

                # LOCAL UPLOAD
                with gr.Column(
                    scale=1,
                    min_width=250,
                    elem_classes=["source-card"],
                ):

                    gr.Markdown("### Upload a file")

                    gr.Markdown(
                        "Upload a CSV or Parquet dataset. "
                        "Profiling starts automatically."
                    )

                    file_input = gr.File(
                        label="Choose dataset",
                        file_types=[".csv", ".parquet", ".pq"],
                        type="filepath",
                    )

                    upload_status = gr.Markdown(
                        "No local dataset loaded."
                    )

                # HUGGING FACE
                with gr.Column(
                    scale=1,
                    min_width=250,
                    elem_classes=["source-card"],
                ):

                    gr.Markdown("### Hugging Face")

                    gr.Markdown(
                        "Load a dataset from a Hugging Face repository."
                    )

                    hf_repo_input = gr.Textbox(
                        label="Repository ID",
                        placeholder="username/dataset-name",
                    )

                    hf_list_button = gr.Button(
                        "Find available files",
                    )

                    hf_filename_input = gr.Dropdown(
                        label="Select a dataset file",
                        choices=[],
                        value=None,
                        interactive=True,
                    )

                    hf_status = gr.Markdown(
                        "No Hugging Face dataset loaded."
                    )

            # ------------------------------------------
            # 2. DATASET OVERVIEW
            # ------------------------------------------

            gr.Markdown("## Dataset overview")

            summary_output = gr.Markdown(
                "Upload a file or select a Hugging Face dataset to begin."
            )

            profile_output = gr.Dataframe(
                label="Column profiling report",
                interactive=False,
                wrap=True,
            )

            # ------------------------------------------
            # 3. CONVERSATION
            # ------------------------------------------

            gr.Markdown("## Ask DataSense AI")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
            )

            question_input = gr.Textbox(
                label="Your question",
                placeholder="Ask anything about your dataset...",
                lines=3,
                elem_id="question-input",
            )

            gr.Examples(
                examples=[
                    ["What are the main data quality issues in this dataset?"],
                    ["Which columns have the most missing values?"],
                    ["Show me the monthly trend over time."],
                    ["Which category has the highest average value?"],
                    ["Are there any months with zero registrations?"],
                    ["Compare the performance of different groups."],
                ],
                inputs=question_input,
                label="Try an example question",
            )

            send_button = gr.Button(
                "Ask DataSense AI",
                variant="primary",
                size="lg",
                elem_id="send-question",
            )

            result_table_output = gr.Dataframe(
                label="Analytical results",
                interactive=False,
                wrap=True,
            )

    # --------------------------------------------------
    # SHARED DATASET OUTPUTS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # STATUS OUTPUTS
    # --------------------------------------------------

    # Upload and Hugging Face status are separate from
    # the shared dataset outputs.

    # --------------------------------------------------
    # EVENT: LOCAL FILE UPLOAD
    # --------------------------------------------------

    file_input.upload(
        fn=analyze_file,
        inputs=[
            file_input,
            workspace_state,
            active_session_state,
        ],
        outputs=dataset_outputs,
    ).then(
        fn=lambda: "Local dataset loaded and profiled.",
        inputs=None,
        outputs=[upload_status],
    )

    # --------------------------------------------------
    # EVENT: LIST HUGGING FACE FILES
    # --------------------------------------------------

    hf_list_button.click(
        fn=list_huggingface_files,
        inputs=[hf_repo_input],
        outputs=[hf_filename_input],
    )

    # --------------------------------------------------
    # EVENT: SELECT HUGGING FACE FILE
    # --------------------------------------------------

    hf_filename_input.input(
        fn=analyze_huggingface,
        inputs=[
            hf_repo_input,
            hf_filename_input,
            workspace_state,
            active_session_state,
        ],
        outputs=dataset_outputs,
    ).then(
        fn=lambda: "Hugging Face dataset loaded and profiled.",
        inputs=None,
        outputs=[hf_status],
    )

    # --------------------------------------------------
    # EVENT: CHAT
    # --------------------------------------------------

    chat_inputs = [
        question_input,
        chatbot,
        dataset_state,
        conversation_state,
        dataset_name_state,
        dataset_config_state,
        workspace_state,
        active_session_state,
    ]

    chat_outputs = [
        chatbot,
        conversation_state,
        question_input,
        result_table_output,
        session_selector,
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
    # EVENT: SWITCH SESSION
    # --------------------------------------------------

    session_selector.input(
        fn=load_session,
        inputs=[
            workspace_state,
            session_selector,
        ],
        outputs=[
            active_session_state,
            *dataset_outputs,
        ],
    )

    # --------------------------------------------------
    # EVENT: INITIALIZE WORKSPACE
    # --------------------------------------------------

    app.load(
        fn=initialize_workspace,
        inputs=[workspace_state],
        outputs=[
            workspace_state,
            active_session_state,
            session_selector,
        ],
    ).then(
        fn=load_session,
        inputs=[
            workspace_state,
            active_session_state,
        ],
        outputs=[
            active_session_state,
            *dataset_outputs,
        ],
    )

    # --------------------------------------------------
    # EVENT: NEW SESSION
    # --------------------------------------------------

    new_session_button.click(
        fn=create_new_session,
        inputs=[workspace_state],
        outputs=[
            active_session_state,
            session_selector,
            *dataset_outputs,
        ],
    ).then(
        fn=lambda: (
            "No local dataset loaded.",
            "No Hugging Face dataset loaded.",
        ),
        inputs=None,
        outputs=[upload_status, hf_status],
    )

    # --------------------------------------------------
    # EVENT: SHOW DELETE CONFIRMATION
    # --------------------------------------------------

    delete_session_button.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=[confirm_delete_group],
    )

    # --------------------------------------------------
    # EVENT: CANCEL DELETE
    # --------------------------------------------------

    cancel_delete_button.click(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=[confirm_delete_group],
    )

    # --------------------------------------------------
    # EVENT: CONFIRM DELETE
    # --------------------------------------------------

    confirm_delete_button.click(
        fn=delete_current_session,
        inputs=[
            workspace_state,
            active_session_state,
        ],
        outputs=[
            active_session_state,
            session_selector,
            *dataset_outputs,
            confirm_delete_group,
        ],
    ).then(
        fn=lambda: (
            "No local dataset loaded.",
            "No Hugging Face dataset loaded.",
        ),
        inputs=None,
        outputs=[upload_status, hf_status],
    )


# --------------------------------------------------
# 11. LAUNCH APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.queue(
        default_concurrency_limit=3
    ).launch(
        css=APP_CSS,
        head=QUESTION_KEYBOARD_JS,
    )
