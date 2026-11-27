
import os
import sys
import re
import json
import time
import sqlite3
import hashlib
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Any, Optional

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
RESULTS_DB = BASE_DIR / "evaluation" / "bird_results.db"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import your existing backend utilities
from server.util import get_llm, clean_sql_output

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DEFAULT_REQUEST_INTERVAL = 2
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CALLS_PER_DAY = 200

DB_LOCK = threading.Lock()

# ---------------------------------------------------------
# SQLITE CONNECTION
# ---------------------------------------------------------

def get_results_connection():
    RESULTS_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(RESULTS_DB),
        timeout=30
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def migrate_results_db():
    """Add missing columns to existing evaluation tables."""

    migrations = {
        "evaluation_runs": {
            "total_questions": "INTEGER DEFAULT 0",
            "completed_questions": "INTEGER DEFAULT 0",
            "correct_questions": "INTEGER DEFAULT 0",
            "execution_accuracy": "REAL DEFAULT 0",
            "started_at": "TEXT",
            "finished_at": "TEXT",
            "updated_at": "TEXT",
        },
        "evaluation_results": {
            "execution_match": "INTEGER DEFAULT 0",
            "exact_match": "INTEGER DEFAULT 0",
            "llm_time": "REAL DEFAULT 0",
            "execution_time": "REAL DEFAULT 0",
            "error": "TEXT",
        }
    }

    with get_results_connection() as conn:
        for table, columns in migrations.items():
            existing = {
                row["name"]
                for row in conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }

            for column, column_type in columns.items():
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} "
                        f"ADD COLUMN {column} {column_type}"
                    )

        conn.commit()


def init_results_db():
    with get_results_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            total_questions INTEGER DEFAULT 0,
            completed_questions INTEGER DEFAULT 0,
            correct_questions INTEGER DEFAULT 0,
            execution_accuracy REAL DEFAULT 0,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            db_id TEXT,
            question TEXT,
            gold_sql TEXT,
            generated_sql TEXT,
            gold_result TEXT,
            predicted_result TEXT,
            status TEXT NOT NULL,
            exact_match INTEGER DEFAULT 0,
            execution_match INTEGER DEFAULT 0,
            llm_time REAL DEFAULT 0,
            execution_time REAL DEFAULT 0,
            error TEXT,
            created_at TEXT,
            UNIQUE(run_id, question_id),
            FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            question_id TEXT,
            call_date TEXT NOT NULL,
            model TEXT,
            status TEXT,
            duration REAL DEFAULT 0,
            error TEXT,
            created_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_results_run
        ON evaluation_results(run_id);

        CREATE INDEX IF NOT EXISTS idx_api_calls_date
        ON api_calls(call_date);
        """)
        conn.commit()

    migrate_results_db()


# ---------------------------------------------------------
# RUN MANAGEMENT
# ---------------------------------------------------------

def create_or_get_run(run_id: str, total_questions: int):
    now = datetime.now().isoformat(timespec="seconds")

    with get_results_connection() as conn:
        existing = conn.execute(
            "SELECT * FROM evaluation_runs WHERE run_id = ?",
            (run_id,)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE evaluation_runs
                SET total_questions = ?, updated_at = ?
                WHERE run_id = ?
            """, (total_questions, now, run_id))
            conn.commit()
            return dict(existing)

        conn.execute("""
            INSERT INTO evaluation_runs (
                run_id, status, total_questions,
                started_at, updated_at
            )
            VALUES (?, 'running', ?, ?, ?)
        """, (run_id, total_questions, now, now))

        conn.commit()

    return {
        "run_id": run_id,
        "status": "running",
        "total_questions": total_questions
    }


def update_run_status(run_id: str, status: str):
    now = datetime.now().isoformat(timespec="seconds")

    with get_results_connection() as conn:
        conn.execute("""
            UPDATE evaluation_runs
            SET status = ?,
                updated_at = ?,
                finished_at = CASE
                    WHEN ? IN ('completed', 'failed', 'stopped')
                    THEN ?
                    ELSE finished_at
                END
            WHERE run_id = ?
        """, (status, now, status, now, run_id))
        conn.commit()


def update_run_statistics(run_id: str):
    with get_results_connection() as conn:
        stats = conn.execute("""
            SELECT
                COUNT(*) AS completed,
                SUM(CASE WHEN execution_match = 1 THEN 1 ELSE 0 END)
                    AS correct,
                AVG(llm_time) AS avg_llm_time,
                AVG(execution_time) AS avg_execution_time
            FROM evaluation_results
            WHERE run_id = ?
              AND status IN ('success', 'incorrect')
        """, (run_id,)).fetchone()

        completed = stats["completed"] or 0
        correct = stats["correct"] or 0
        accuracy = (correct / completed * 100) if completed else 0

        conn.execute("""
            UPDATE evaluation_runs
            SET completed_questions = ?,
                correct_questions = ?,
                execution_accuracy = ?,
                updated_at = ?
            WHERE run_id = ?
        """, (
            completed,
            correct,
            accuracy,
            datetime.now().isoformat(timespec="seconds"),
            run_id
        ))
        conn.commit()


# ---------------------------------------------------------
# API CALL TRACKING AND DAILY LIMIT
# ---------------------------------------------------------

def get_daily_call_count():
    today = date.today().isoformat()

    with get_results_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total
            FROM api_calls
            WHERE call_date = ?
        """, (today,)).fetchone()

    return row["total"] or 0


def save_api_call(
    run_id: str,
    question_id: str,
    status: str,
    duration: float,
    error: Optional[str] = None
):
    today = date.today().isoformat()

    with get_results_connection() as conn:
        conn.execute("""
            INSERT INTO api_calls (
                run_id, question_id, call_date,
                model, status, duration, error, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            question_id,
            today,
            os.getenv("GEMINI_MODEL", "configured-backend-model"),
            status,
            duration,
            error,
            datetime.now().isoformat(timespec="seconds")
        ))
        conn.commit()


# ---------------------------------------------------------
# QUESTION HELPERS
# ---------------------------------------------------------

def get_question_value(item: dict, *keys, default=None):
    if not isinstance(item, dict):
        return default
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return default


def normalize_sql(sql: str) -> str:
    sql = (sql or "").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.rstrip(";").strip().lower()


def serialize_result(rows: list) -> str:
    return json.dumps(
        [list(row) for row in rows],
        ensure_ascii=False,
        default=str
    )


def execute_sql(db_path: str, sql: str):
    """
    Execute read-only SQL against the selected SQLite database.
    Returns rows as a list of tuples.
    """

    if not sql:
        raise ValueError("Generated SQL is empty.")

    # Open the database in read-only mode.
    absolute_path = Path(db_path).resolve()
    uri = absolute_path.as_uri() + "?mode=ro"

    conn = sqlite3.connect(uri, uri=True, timeout=30)

    try:
        conn.execute("PRAGMA query_only = ON")

        cursor = conn.cursor()
        cursor.execute(sql)

        rows = cursor.fetchall()
        return rows

    finally:
        conn.close()


# ---------------------------------------------------------
# GEMINI SQL GENERATION
# ---------------------------------------------------------

def generate_sql(question: str, db_path: str):
    """
    Generate SQL using the existing backend LLM configuration.
    The database schema is read from SQLite and passed to Gemini.
    """

    conn = sqlite3.connect(
        f"file:{Path(db_path).resolve()}?mode=ro",
        uri=True
    )

    try:
        cursor = conn.cursor()

        tables = cursor.execute("""
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
        """).fetchall()

        schema_parts = []

        for table_name, create_sql in tables:
            schema_parts.append(
                f"Table: {table_name}\n{create_sql}"
            )

            try:
                foreign_keys = cursor.execute(
                    f'PRAGMA foreign_key_list("{table_name}")'
                ).fetchall()

                if foreign_keys:
                    schema_parts.append(
                        f"Foreign keys for {table_name}: {foreign_keys}"
                    )
            except sqlite3.Error:
                pass

        schema = "\n\n".join(schema_parts)

    finally:
        conn.close()

    prompt = f"""
You are an expert SQLite SQL query generator.

Generate one valid SQLite SELECT query for the user's question.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

RULES:
1. Use only tables and columns in the provided schema.
2. Generate SQLite-compatible SQL.
3. Only generate read-only SELECT or WITH queries.
4. Do not generate INSERT, UPDATE, DELETE, DROP, ALTER,
   CREATE, REPLACE, ATTACH, DETACH, or PRAGMA statements.
5. Do not explain the answer.
6. Return only SQL, without Markdown code fences.

SQL:
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(
            (part.get("text", "") or "")
            if isinstance(part, dict)
            else (str(part) if part is not None else "")
            for part in content
        )

    sql = clean_sql_output(str(content))
    return sql


# ---------------------------------------------------------
# GOLD SQL EXECUTION
# ---------------------------------------------------------

def get_gold_sql(item: dict):
    return get_question_value(
        item,
        "SQL",
        "sql",
        "gold_sql",
        "query"
    )


def get_question_text(item: dict):
    return get_question_value(
        item,
        "question",
        "Question",
        "natural_language_question",
        default=""
    )


def get_question_id(item: dict, index: int):
    return str(
        get_question_value(
            item,
            "question_id",
            "id",
            default=index
        )
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_question(
    run_id: str,
    question_id: str,
    item: dict,
    db_path: str,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    daily_limit: int = DEFAULT_MAX_CALLS_PER_DAY
):
    """Evaluate one question and always return a dictionary for runner.py."""

    if not isinstance(item, dict):
        return {
            "status": "error", "generated_sql": "",
            "llm_time": 0.0, "execution_time": 0.0,
            "error": f"Dataset record must be an object, got {type(item).__name__}."
        }

    question = get_question_text(item)
    gold_sql = get_gold_sql(item)

    if not question or not gold_sql:
        message = (
            f"Question {question_id} is missing "
            f"{'question' if not question else 'gold SQL'} in its dataset record."
        )
        return {
            "status": "error", "generated_sql": "",
            "llm_time": 0.0, "execution_time": 0.0, "error": message
        }

    llm_time = 0.0
    execution_time = 0.0
    generated_sql = ""
    gold_result = None
    predicted_result = None
    error = None
    qid = str(question_id)

    # Execute the reference SQL first.
    try:
        started = time.time()
        gold_rows = execute_sql(db_path, gold_sql)
        gold_result = serialize_result(gold_rows)
        execution_time += time.time() - started
    except Exception as exc:
        error = f"Gold SQL execution failed: {exc}"
        save_result(run_id, qid, item, "", None, None, "error", 0, 0,
                    llm_time, execution_time, error)
        update_run_statistics(run_id)
        return {
            "status": "error", "generated_sql": "",
            "llm_time": llm_time, "execution_time": execution_time,
            "error": error
        }

    if get_daily_call_count() >= daily_limit:
        raise RuntimeError("DAILY_BUDGET_REACHED")

    if request_interval > 0:
        time.sleep(request_interval)

    api_success = False
    for attempt in range(max_retries + 1):
        if get_daily_call_count() >= daily_limit:
            raise RuntimeError("DAILY_BUDGET_REACHED")

        started = time.time()
        try:
            generated_sql = generate_sql(question, db_path)
            duration = time.time() - started
            llm_time += duration
            save_api_call(run_id, qid, "success", duration)
            api_success = True
            break
        except Exception as exc:
            duration = time.time() - started
            llm_time += duration
            error_text = str(exc)
            save_api_call(run_id, qid, "error", duration, error_text)

            retryable = any(
                token in error_text.lower()
                for token in (
                    "429", "rate limit", "resource exhausted",
                    "503", "500", "timeout", "temporarily unavailable"
                )
            )
            if not retryable or attempt >= max_retries:
                error = error_text
                break

            time.sleep(min(60, max(request_interval, 1) * (2 ** (attempt + 1))))

    if not api_success:
        error = error or "SQL generation failed."
        save_result(run_id, qid, item, generated_sql, gold_result, None,
                    "error", 0, 0, llm_time, execution_time, error)
        update_run_statistics(run_id)
        return {
            "status": "error", "generated_sql": generated_sql,
            "llm_time": llm_time, "execution_time": execution_time,
            "error": error
        }

    try:
        started = time.time()
        predicted_rows = execute_sql(db_path, generated_sql)
        execution_time += time.time() - started
        predicted_result = serialize_result(predicted_rows)

        gold_normalized = sorted(
            json.loads(gold_result),
            key=lambda row: json.dumps(row, sort_keys=True, default=str)
        )
        predicted_normalized = sorted(
            json.loads(predicted_result),
            key=lambda row: json.dumps(row, sort_keys=True, default=str)
        )

        execution_match = int(gold_normalized == predicted_normalized)
        exact_match = int(normalize_sql(generated_sql) == normalize_sql(gold_sql))
        status = "success" if execution_match else "incorrect"

    except Exception as exc:
        execution_match = 0
        exact_match = int(normalize_sql(generated_sql) == normalize_sql(gold_sql))
        status = "error"
        error = f"Generated SQL execution failed: {exc}"

    save_result(
        run_id, qid, item, generated_sql, gold_result, predicted_result,
        status, exact_match, execution_match, llm_time, execution_time, error
    )
    update_run_statistics(run_id)

    return {
        "status": status,
        "generated_sql": generated_sql,
        "llm_time": round(llm_time, 4),
        "execution_time": round(execution_time, 4),
        "exact_match": exact_match,
        "execution_match": execution_match,
        "error": error
    }


def save_result(
    run_id: str,
    question_id: str,
    item: dict,
    generated_sql: str,
    gold_result: Optional[str],
    predicted_result: Optional[str],
    status: str,
    exact_match: int,
    execution_match: int,
    llm_time: float,
    execution_time: float,
    error: Optional[str]
):
    question = get_question_text(item)
    gold_sql = get_gold_sql(item)
    db_id = get_question_value(item, "db_id", "database_id")

    now = datetime.now().isoformat(timespec="seconds")

    with get_results_connection() as conn:
        conn.execute("""
            INSERT INTO evaluation_results (
                run_id,
                question_id,
                db_id,
                question,
                gold_sql,
                generated_sql,
                gold_result,
                predicted_result,
                status,
                exact_match,
                execution_match,
                llm_time,
                execution_time,
                error,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(run_id, question_id)
            DO UPDATE SET
                generated_sql = excluded.generated_sql,
                gold_result = excluded.gold_result,
                predicted_result = excluded.predicted_result,
                status = excluded.status,
                exact_match = excluded.exact_match,
                execution_match = excluded.execution_match,
                llm_time = excluded.llm_time,
                execution_time = excluded.execution_time,
                error = excluded.error,
                created_at = excluded.created_at
        """, (
            run_id,
            question_id,
            db_id,
            question,
            gold_sql,
            generated_sql,
            gold_result,
            predicted_result,
            status,
            exact_match,
            execution_match,
            llm_time,
            execution_time,
            error,
            now
        ))
        conn.commit()
