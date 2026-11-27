import os
import sqlite3
from flask import request, jsonify
from flask_cors import cross_origin
from server import app
from server.util import connect_to_sql, execute_and_explain
from server.sample_db import get_sample_db_path, SAMPLE_QUESTIONS
from langchain_community.utilities.sql_database import SQLDatabase

# Global connection storage (keyed by session/connection id or default active)
# Default initializes to sample DB so demos work without setup!
active_connections = {}

def get_or_create_default_db():
    if "default" not in active_connections:
        sample_path = get_sample_db_path()
        active_connections["default"] = SQLDatabase.from_uri(f"sqlite:///{sample_path}")
    return active_connections["default"]

@app.route('/health', methods=['GET'])
@cross_origin()
def health():
    return jsonify({
        "status": "healthy",
        "service": "SQL-AI Backend",
        "has_api_key": bool(os.getenv("GOOGLE_API_KEY"))
    }), 200

@app.route('/sample-info', methods=['GET'])
@cross_origin()
def sample_info():
    """Provides sample DB schema metadata, table lists, and suggested questions."""
    try:
        db = get_or_create_default_db()
        tables = db.get_usable_table_names()
        return jsonify({
            "status": "success",
            "db_type": "SQLite E-Commerce Demo",
            "tables": tables,
            "sample_questions": SAMPLE_QUESTIONS
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload-db', methods=['POST'])
@cross_origin()
def upload_db():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    if not filename:
        return jsonify({"status": "error", "message": "No file selected"}), 400

    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ['db', 'sqlite', 'sqlite3', 'csv']:
        return jsonify({"status": "error", "message": "Unsupported file format. Please upload .db, .sqlite, or .csv files."}), 400

    session_id = request.form.get("session_id", "default")
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, filename)
    uploaded_file.save(file_path)

    try:
        if ext == 'csv':
            import pandas as pd
            import sqlite3
            table_name = os.path.splitext(filename)[0].lower().replace(" ", "_").replace("-", "_")
            db_path = os.path.join(uploads_dir, f"{table_name}_converted.db")
            conn = sqlite3.connect(db_path)
            df = pd.read_csv(file_path)
            # Clean column names
            df.columns = [str(c).strip().replace(" ", "_").replace("-", "_") for c in df.columns]
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()
            db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        else:
            db = SQLDatabase.from_uri(f"sqlite:///{file_path}")

        active_connections[session_id] = db
        tables = db.get_usable_table_names()
        return jsonify({
            "status": "success",
            "message": f"Successfully loaded database from {filename}",
            "filename": filename,
            "tables": tables
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to process file: {str(e)}"
        }), 400

@app.route('/connect', methods=['POST'])
@cross_origin()
def connect():
    data = request.get_json() or {}
    connection_type = data.get("type", "custom")
    session_id = data.get("session_id", "default")

    try:
        if connection_type == "sample":
            sample_path = get_sample_db_path()
            db = SQLDatabase.from_uri(f"sqlite:///{sample_path}")
            active_connections[session_id] = db
            tables = db.get_usable_table_names()
            return jsonify({
                "status": "success",
                "message": "Connected to Sample E-Commerce Database",
                "is_sample": True,
                "tables": tables
            }), 200
        else:
            form_data = data.get('form_data')
            sql_link = data.get('sql_link', '')
            db = connect_to_sql(db_stats=form_data, url=sql_link)
            active_connections[session_id] = db
            tables = db.get_usable_table_names()
            return jsonify({
                "status": "success",
                "message": "Connected to custom database successfully",
                "is_sample": False,
                "tables": tables
            }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Connection failed: {str(e)}"
        }), 400

@app.route('/ask', methods=['POST'])
@cross_origin()
def ask():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    session_id = data.get('session_id', 'default')

    if not query:
        return jsonify({"status": "error", "message": "Please provide a valid question."}), 400

    # Ensure we have a database connection ready
    db = active_connections.get(session_id)
    if not db:
        # Fallback to sample DB for frictionless demo
        db = get_or_create_default_db()
        active_connections[session_id] = db

    try:
        result = execute_and_explain(db, query)
        return jsonify({
            "status": "success",
            "answer": result["summary"],
            "sql": result["sql"],
            "columns": result["columns"],
            "rows": result["rows"],
            "total_rows": result["total_rows"],
            "execution_time_sec": result["execution_time_sec"]
        }), 200
    except ValueError as ve:
        # e.g., missing API key
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 401
    except PermissionError as pe:
        # Safety guardrail blocked query
        return jsonify({
            "status": "security_block",
            "message": str(pe)
        }), 403
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Execution error: {str(e)}"
        }), 500

@app.route("/evaluation-results", methods=["GET"])
@cross_origin()
def get_evaluation_results():
    BIRD_DB_PATH = "../evaluation/bird_results.db"
    try:
        conn = sqlite3.connect(BIRD_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Overall statistics
        cursor.execute("""
            SELECT
                COUNT(*) AS total_questions,
                SUM(CASE WHEN status = 'correct' THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN status = 'incorrect' THEN 1 ELSE 0 END) AS incorrect,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN execution_match = 1 THEN 1 ELSE 0 END) AS execution_matches,
                SUM(CASE WHEN exact_match = 1 THEN 1 ELSE 0 END) AS exact_matches,
                AVG(llm_time) AS avg_llm_time,
                AVG(execution_time) AS avg_execution_time
            FROM evaluation_results
        """)
        stats = dict(cursor.fetchone())

        total = stats["total_questions"] or 0
        correct = stats["correct"] or 0

        stats["accuracy"] = round(
            (correct / total) * 100, 2
        ) if total else 0

        stats["correct"] = correct
        stats["incorrect"] = stats["incorrect"] or 0
        stats["errors"] = stats["errors"] or 0
        stats["execution_matches"] = stats["execution_matches"] or 0
        stats["exact_matches"] = stats["exact_matches"] or 0

        # Evaluation runs
        cursor.execute("""
            SELECT *
            FROM evaluation_runs
            ORDER BY started_at DESC
        """)
        runs = [dict(row) for row in cursor.fetchall()]

        # Detailed evaluation results
        cursor.execute("""
            SELECT
                id, run_id, question_id, db_id, question,
                status, exact_match, execution_match,
                llm_time, execution_time, error, created_at
            FROM evaluation_results
            ORDER BY created_at DESC
            LIMIT 200
        """)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return jsonify({
            "status": "success",
            "stats": stats,
            "runs": runs,
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return jsonify({
        "message": "SQL AI Agent API is running",
        "endpoints": ["/health", "/sample-info", "/connect", "/ask","/evaluation-results"]
    }), 200