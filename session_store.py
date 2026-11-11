
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil

# MVP workspace limits
MAX_SESSIONS_PER_WORKSPACE = 10
MAX_MESSAGES_PER_SESSION = 200
MAX_DATASET_SIZE_BYTES = 100 * 1024 * 1024
MAX_SESSION_SIZE_BYTES = 120 * 1024 * 1024
MAX_WORKSPACE_SIZE_BYTES = 500 * 1024 * 1024

PROJECT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(
    os.getenv("DATA_SENSE_STORAGE_DIR", PROJECT_DIR / "data")
).resolve()
DB_PATH = STORAGE_DIR / "datasense.sqlite3"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _validate_workspace_id(workspace_id):
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("A valid workspace ID is required.")

    if len(workspace_id) > 128:
        raise ValueError("Invalid workspace ID.")

    return workspace_id.strip()


@contextmanager
def _connection():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_db():
    with _connection() as connection:
        connection.execute("PRAGMA journal_mode = WAL")

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                dataset_name TEXT,
                source_type TEXT,
                repo_id TEXT,
                filename TEXT,
                revision TEXT,
                dataset_size_bytes INTEGER NOT NULL DEFAULT 0,
                dataset_path TEXT,
                dataset_config_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_workspace
            ON sessions(workspace_id);

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id)
                    REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, message_id);
            """

        )
        # Add dataset_path to databases created before this change.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(sessions)"
            ).fetchall()
        }

        if "dataset_path" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN dataset_path TEXT"
            )


def _get_session(connection, workspace_id, session_id):
    row = connection.execute(
        """
        SELECT * FROM sessions
        WHERE workspace_id = ? AND session_id = ?
        """,
        (workspace_id, session_id),
    ).fetchone()

    if row is None:
        raise ValueError("Session not found.")

    return row


def _message_bytes(connection, session_id):
    row = connection.execute(
        """
        SELECT COALESCE(SUM(LENGTH(CAST(content AS BLOB))), 0)
        AS total
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    return row["total"]


def _workspace_usage(connection, workspace_id):
    dataset_row = connection.execute(
        """
        SELECT COALESCE(SUM(dataset_size_bytes), 0) AS total
        FROM sessions
        WHERE workspace_id = ?
        """,
        (workspace_id,),
    ).fetchone()

    message_row = connection.execute(
        """
        SELECT COALESCE(SUM(LENGTH(CAST(m.content AS BLOB))), 0)
        AS total
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE s.workspace_id = ?
        """,
        (workspace_id,),
    ).fetchone()

    return dataset_row["total"] + message_row["total"]


def create_session(workspace_id, name="New session"):
    workspace_id = _validate_workspace_id(workspace_id)
    name = str(name).strip() or "New session"

    initialize_db()

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM sessions
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        ).fetchone()

        if row["total"] >= MAX_SESSIONS_PER_WORKSPACE:
            raise ValueError(
                f"You can have up to {MAX_SESSIONS_PER_WORKSPACE} "
                "sessions in one workspace. Delete a session to continue."
            )

        session_id = uuid.uuid4().hex
        timestamp = _now()

        connection.execute(
            """
            INSERT INTO sessions (
                session_id, workspace_id, name,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, workspace_id, name, timestamp, timestamp),
        )

    return session_id


def list_sessions(workspace_id):
    workspace_id = _validate_workspace_id(workspace_id)
    initialize_db()

    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT
                s.*,
                (
                    SELECT COUNT(*)
                    FROM messages m
                    WHERE m.session_id = s.session_id
                ) AS message_count
            FROM sessions s
            WHERE s.workspace_id = ?
            ORDER BY s.updated_at DESC
            """,
            (workspace_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def rename_session(workspace_id, session_id, name):
    workspace_id = _validate_workspace_id(workspace_id)
    name = str(name).strip()

    if not name:
        raise ValueError("Session name cannot be empty.")

    initialize_db()

    with _connection() as connection:
        _get_session(connection, workspace_id, session_id)

        connection.execute(
            """
            UPDATE sessions
            SET name = ?, updated_at = ?
            WHERE workspace_id = ? AND session_id = ?
            """,
            (name, _now(), workspace_id, session_id),
        )


def delete_session(workspace_id, session_id):
    workspace_id = _validate_workspace_id(workspace_id)
    initialize_db()

    dataset_path = None

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        session = _get_session(connection, workspace_id, session_id)
        dataset_path = session["dataset_path"]

        connection.execute(
            """
            DELETE FROM sessions
            WHERE workspace_id = ? AND session_id = ?
            """,
            (workspace_id, session_id),
        )

    if dataset_path:
        path = (STORAGE_DIR / dataset_path).resolve()

        if path.is_relative_to(STORAGE_DIR):
            path.unlink(missing_ok=True)

            parent = path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

def get_messages(workspace_id, session_id):
    workspace_id = _validate_workspace_id(workspace_id)
    initialize_db()

    with _connection() as connection:
        _get_session(connection, workspace_id, session_id)

        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY message_id
            """,
            (session_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def add_message(workspace_id, session_id, role, content):
    workspace_id = _validate_workspace_id(workspace_id)

    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be user or assistant.")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")

    message_size = len(content.encode("utf-8"))
    initialize_db()

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        session = _get_session(connection, workspace_id, session_id)

        message_count = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()["total"]

        if message_count >= MAX_MESSAGES_PER_SESSION:
            raise ValueError(
                f"This session has reached the "
                f"{MAX_MESSAGES_PER_SESSION}-message limit. "
                "Start a new session to continue."
            )

        session_usage = (
            session["dataset_size_bytes"]
            + _message_bytes(connection, session_id)
        )

        if session_usage + message_size > MAX_SESSION_SIZE_BYTES:
            raise ValueError(
                "This session has reached its 120 MB storage limit. "
                "Start a new session or delete this one."
            )

        if (
            _workspace_usage(connection, workspace_id) + message_size
            > MAX_WORKSPACE_SIZE_BYTES
        ):
            raise ValueError(
                "This workspace has reached its 500 MB storage limit. "
                "Delete an existing session to free space."
            )

        connection.execute(
            """
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, _now()),
        )

        connection.execute(
            """
            UPDATE sessions SET updated_at = ?
            WHERE session_id = ?
            """,
            (_now(), session_id),
        )


def update_dataset_metadata(
    workspace_id,
    session_id,
    *,
    dataset_name,
    source_type,
    dataset_size_bytes,
    dataset_config=None,
    repo_id=None,
    filename=None,
    revision=None,
):
    workspace_id = _validate_workspace_id(workspace_id)

    if not isinstance(dataset_size_bytes, int) or dataset_size_bytes < 0:
        raise ValueError("Dataset size must be a non-negative integer.")

    if dataset_size_bytes > MAX_DATASET_SIZE_BYTES:
        raise ValueError("Dataset exceeds the 100 MB per-session limit.")

    config_json = json.dumps(dataset_config or {})

    initialize_db()

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        session = _get_session(connection, workspace_id, session_id)

        existing_message_bytes = _message_bytes(
            connection, session_id
        )

        if (
            dataset_size_bytes + existing_message_bytes
            > MAX_SESSION_SIZE_BYTES
        ):
            raise ValueError("Dataset would exceed the 120 MB session limit.")

        workspace_usage = _workspace_usage(connection, workspace_id)
        updated_usage = (
            workspace_usage
            - session["dataset_size_bytes"]
            + dataset_size_bytes
        )

        if updated_usage > MAX_WORKSPACE_SIZE_BYTES:
            raise ValueError(
                "Dataset would exceed the 500 MB workspace limit."
            )

        connection.execute(
            """
            UPDATE sessions
            SET dataset_name = ?,
                source_type = ?,
                repo_id = ?,
                filename = ?,
                revision = ?,
                dataset_size_bytes = ?,
                dataset_config_json = ?,
                updated_at = ?
            WHERE workspace_id = ? AND session_id = ?
            """,
            (
                dataset_name,
                source_type,
                repo_id,
                filename,
                revision,
                dataset_size_bytes,
                config_json,
                _now(),
                workspace_id,
                session_id,
            ),
        )
def save_uploaded_dataset(
    workspace_id,
    session_id,
    source_path,
    dataset_config=None,
):
    workspace_id = _validate_workspace_id(workspace_id)
    source_path = Path(source_path).resolve()

    if not source_path.is_file():
        raise ValueError("The uploaded dataset file could not be found.")

    suffix = source_path.suffix.lower()

    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError("Only CSV and Parquet files are supported.")

    source_size = source_path.stat().st_size

    if source_size == 0:
        raise ValueError("The uploaded dataset file is empty.")

    if source_size > MAX_DATASET_SIZE_BYTES:
        raise ValueError("Dataset exceeds the 100 MB per-session limit.")

    initialize_db()

    stored_path = None
    old_dataset_path = None

    try:
        with _connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = _get_session(
                connection, workspace_id, session_id
            )

            old_dataset_path = session["dataset_path"]

            session_message_bytes = _message_bytes(
                connection, session_id
            )

            if (
                source_size + session_message_bytes
                > MAX_SESSION_SIZE_BYTES
            ):
                raise ValueError(
                    "Dataset would exceed the 120 MB session limit."
                )

            workspace_usage = _workspace_usage(
                connection, workspace_id
            )

            new_workspace_usage = (
                workspace_usage
                - session["dataset_size_bytes"]
                + source_size
            )

            if new_workspace_usage > MAX_WORKSPACE_SIZE_BYTES:
                raise ValueError(
                    "Dataset would exceed the 500 MB workspace limit."
                )

            # Use the generated session ID, never a user-supplied path.
            dataset_dir = STORAGE_DIR / "datasets" / session_id
            dataset_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{uuid.uuid4().hex}{suffix}"
            stored_path = dataset_dir / filename

            shutil.copyfile(source_path, stored_path)

            actual_size = stored_path.stat().st_size

            if actual_size != source_size:
                raise ValueError(
                    "The uploaded dataset changed while being saved."
                )

            if actual_size > MAX_DATASET_SIZE_BYTES:
                raise ValueError(
                    "Dataset exceeds the 100 MB per-session limit."
                )

            relative_path = stored_path.relative_to(
                STORAGE_DIR
            ).as_posix()

            connection.execute(
                """
                UPDATE sessions
                SET dataset_name = ?,
                    source_type = 'upload',
                    repo_id = NULL,
                    filename = NULL,
                    revision = NULL,
                    dataset_size_bytes = ?,
                    dataset_path = ?,
                    dataset_config_json = ?,
                    updated_at = ?
                WHERE workspace_id = ? AND session_id = ?
                """,
                (
                    source_path.name,
                    actual_size,
                    relative_path,
                    json.dumps(dataset_config or {}),
                    _now(),
                    workspace_id,
                    session_id,
                ),
            )

        # Remove the previous dataset only after the database update succeeds.
        if old_dataset_path:
            old_path = (STORAGE_DIR / old_dataset_path).resolve()

            if old_path.is_relative_to(STORAGE_DIR):
                old_path.unlink(missing_ok=True)

        return stored_path

    except Exception:
        if stored_path and stored_path.exists():
            stored_path.unlink(missing_ok=True)
        raise


def get_dataset_path(workspace_id, session_id):
    workspace_id = _validate_workspace_id(workspace_id)
    initialize_db()

    with _connection() as connection:
        session = _get_session(
            connection, workspace_id, session_id
        )

        relative_path = session["dataset_path"]

    if not relative_path:
        return None

    path = (STORAGE_DIR / relative_path).resolve()

    if not path.is_relative_to(STORAGE_DIR):
        raise ValueError("Invalid stored dataset path.")

    if not path.is_file():
        raise FileNotFoundError(
            "The saved dataset file is missing."
        )

    return path


def get_session(workspace_id, session_id):
    workspace_id = _validate_workspace_id(workspace_id)
    initialize_db()

    with _connection() as connection:
        session = _get_session(
            connection, workspace_id, session_id
        )

        result = dict(session)
        result["dataset_config"] = json.loads(
            result.pop("dataset_config_json")
        )

        return result


def save_huggingface_reference(
    workspace_id,
    session_id,
    metadata,
    dataset_config=None,
):
    workspace_id = _validate_workspace_id(workspace_id)

    if not isinstance(metadata, dict):
        raise ValueError("Invalid Hugging Face dataset metadata.")

    repo_id = metadata.get("repo_id")
    filename = metadata.get("filename")
    revision = metadata.get("revision")
    dataset_size_bytes = metadata.get("file_size_bytes")

    if not all(
        isinstance(value, str) and value.strip()
        for value in (repo_id, filename, revision)
    ):
        raise ValueError("Incomplete Hugging Face dataset reference.")

    if (
        not isinstance(dataset_size_bytes, int)
        or isinstance(dataset_size_bytes, bool)
        or dataset_size_bytes < 0
    ):
        raise ValueError("Invalid Hugging Face dataset file size.")

    if dataset_size_bytes > MAX_DATASET_SIZE_BYTES:
        raise ValueError("Dataset exceeds the 100 MB per-session limit.")

    config_json = json.dumps(dataset_config or {})
    initialize_db()

    old_dataset_path = None

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        session = _get_session(
            connection, workspace_id, session_id
        )

        old_dataset_path = session["dataset_path"]

        message_bytes = _message_bytes(connection, session_id)

        if (
            dataset_size_bytes + message_bytes
            > MAX_SESSION_SIZE_BYTES
        ):
            raise ValueError(
                "Dataset would exceed the 120 MB session limit."
            )

        workspace_usage = _workspace_usage(
            connection, workspace_id
        )

        updated_usage = (
            workspace_usage
            - session["dataset_size_bytes"]
            + dataset_size_bytes
        )

        if updated_usage > MAX_WORKSPACE_SIZE_BYTES:
            raise ValueError(
                "Dataset would exceed the 500 MB workspace limit."
            )

        connection.execute(
            """
            UPDATE sessions
            SET dataset_name = ?,
                source_type = 'huggingface',
                repo_id = ?,
                filename = ?,
                revision = ?,
                dataset_size_bytes = ?,
                dataset_path = NULL,
                dataset_config_json = ?,
                updated_at = ?
            WHERE workspace_id = ? AND session_id = ?
            """,
            (
                filename,
                repo_id,
                filename,
                revision,
                dataset_size_bytes,
                config_json,
                _now(),
                workspace_id,
                session_id,
            ),
        )

    # Remove an old uploaded dataset only after the reference is saved.
    if old_dataset_path:
        old_path = (STORAGE_DIR / old_dataset_path).resolve()

        if old_path.is_relative_to(STORAGE_DIR):
            old_path.unlink(missing_ok=True)

            parent = old_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
