
import sys
import json
import argparse

from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

# backend directory
BASE_DIR = Path(__file__).resolve().parents[1]

# NLP/minidev directory
DATASET_DIR = BASE_DIR.parent / "minidev"

JSON_FILE = DATASET_DIR / "mini_dev_sqlite.jsonl"

DB_DIR = DATASET_DIR /"MINIDEV"/"dev_databases"
BACKEND_DIR = BASE_DIR / "backend"



# ============================================================
# PYTHON IMPORT PATH
# Allows direct execution:
# python evaluation/runner.py
# ============================================================

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from evaluator import (
    init_results_db,
    create_or_get_run,
    update_run_status,
    update_run_statistics,
    get_results_connection,
    evaluate_question,
    DEFAULT_REQUEST_INTERVAL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_CALLS_PER_DAY
)


# ============================================================
# LOAD BIRD DATASET
# ============================================================


def load_dataset():
    if not JSON_FILE.exists():
        raise FileNotFoundError(
            f"Dataset JSONL not found: {JSON_FILE}"
        )

    questions = []

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                questions.append(item)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

    print(f"Loaded {len(questions)} questions from dataset.")

    return questions



# ============================================================
# FIND SQLITE DATABASE
# ============================================================

def get_database_path(db_id):

    # Standard BIRD directory structure.
    expected_path = (
        DB_DIR
        / db_id
        / f"{db_id}.sqlite"
    )

    if expected_path.exists():
        return expected_path

    # Support alternative extracted layouts.
    matches = []

    for extension in [".sqlite", ".db", ".sqlite3"]:

        matches.extend(
            DB_DIR.rglob(f"{db_id}{extension}")
        )

    if not matches:

        raise FileNotFoundError(
            f"Database not found for db_id: {db_id}\n"
            f"Searched directory: {DB_DIR}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple databases found for {db_id}: {matches}"
        )

    return matches[0]


# ============================================================
# GET COMPLETED QUESTIONS
# ============================================================

def get_completed_questions(run_id):

    with get_results_connection() as conn:

        rows = conn.execute(
            """
            SELECT question_id
            FROM evaluation_results
            WHERE run_id = ?
            AND status IN ('success', 'incorrect')
            """,
            (run_id,)
        ).fetchall()

        return {
            str(row["question_id"])
            for row in rows
        }


# ============================================================
# DISPLAY QUESTION PROGRESS
# ============================================================

def print_progress(
    index,
    total,
    question_id,
    result
):

    if not isinstance(result, dict):
        result = {
            "status": "error",
            "error": f"Evaluator returned {type(result).__name__} instead of a result dictionary."
        }

    status = result.get("status", "unknown")

    if status == "success":
        label = "CORRECT"

    elif status == "incorrect":
        label = "INCORRECT"

    else:
        label = "ERROR"

    print("\n" + "-" * 65)

    print(
        f"[{index}/{total}] Question ID: {question_id}"
    )

    print(f"Status: {label}")

    if status in ["success", "incorrect"]:

        print(
            f"LLM generation time: "
            f"{result.get('llm_time', 0)} seconds"
        )

        print(
            f"SQL execution time: "
            f"{result.get('execution_time', 0)} seconds"
        )

        print(
            f"Generated SQL:\n"
            f"{result.get('generated_sql', '')}"
        )

    else:

        print(
            f"Error: {result.get('error', 'Unknown error')}"
        )


# ============================================================
# FINAL EVALUATION STATISTICS
# ============================================================

def show_statistics(run_id, total_questions):

    with get_results_connection() as conn:

        stats = conn.execute(
            """
            SELECT

                COUNT(*) AS attempted,

                SUM(
                    CASE
                        WHEN status IN ('success', 'incorrect')
                        THEN 1
                        ELSE 0
                    END
                ) AS evaluated,

                SUM(
                    CASE
                        WHEN execution_match = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS correct,

                SUM(
                    CASE
                        WHEN status = 'incorrect'
                        THEN 1
                        ELSE 0
                    END
                ) AS incorrect,

                SUM(
                    CASE
                        WHEN status = 'error'
                        THEN 1
                        ELSE 0
                    END
                ) AS errors,

                AVG(llm_time) AS avg_llm_time,

                AVG(execution_time) AS avg_execution_time

            FROM evaluation_results

            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()

        api_calls = conn.execute(
            """
            SELECT COUNT(*)
            FROM api_calls
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()[0]

    evaluated = stats["evaluated"] or 0

    correct = stats["correct"] or 0

    incorrect = stats["incorrect"] or 0

    errors = stats["errors"] or 0

    remaining = max(
        total_questions - evaluated,
        0
    )

    accuracy = (
        (correct / evaluated) * 100
        if evaluated > 0
        else 0
    )

    print("\n")
    print("=" * 65)
    print("                 BIRD EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Run ID:                 {run_id}")

    print(f"Total questions:        {total_questions}")

    print(f"Evaluated questions:    {evaluated}")

    print(f"Correct questions:      {correct}")

    print(f"Incorrect questions:   {incorrect}")

    print(f"Errors:                 {errors}")

    print(f"Remaining questions:    {remaining}")

    print("-" * 65)

    print(
        f"Execution Accuracy:     {accuracy:.2f}%"
    )

    print(
        f"Average LLM time:       "
        f"{stats['avg_llm_time'] or 0:.2f} seconds"
    )

    print(
        f"Average SQL time:       "
        f"{stats['avg_execution_time'] or 0:.4f} seconds"
    )

    print(
        f"Total API attempts:     {api_calls}"
    )

    print("=" * 65)


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def run_evaluation(

    run_id,

    limit=None,

    request_interval=DEFAULT_REQUEST_INTERVAL,

    max_retries=DEFAULT_MAX_RETRIES,

    daily_limit=DEFAULT_MAX_CALLS_PER_DAY

):

    # Initialize evaluation SQLite database.
    init_results_db()

    # Load BIRD dataset.
    questions = load_dataset()

    total_dataset_questions = len(questions)

    # Limit questions for testing.
    if limit is not None:

        questions = questions[:limit]

    total_questions = len(questions)

    # Create or resume existing evaluation.
    create_or_get_run(
        run_id,
        total_questions
    )

    completed_questions = get_completed_questions(
        run_id
    )

    print("\n")
    print("=" * 65)
    print("              BIRD-MINI-DEV EVALUATION")
    print("=" * 65)

    print(f"Run ID:             {run_id}")

    print(f"Dataset:            {JSON_FILE}")

    print(f"Database folder:    {DB_DIR}")

    print(f"Total dataset:      {total_dataset_questions}")

    print(f"Questions selected: {total_questions}")

    print(f"Already completed:  {len(completed_questions)}")

    print(
        f"Remaining:          "
        f"{max(total_questions - len(completed_questions), 0)}"
    )

    print(f"Request interval:   {request_interval} seconds")

    print(f"Maximum retries:    {max_retries}")

    print(f"Daily API budget:   {daily_limit}")

    print("=" * 65)

    try:

        for index, item in enumerate(
            questions,
            start=1
        ):

            # Use BIRD question ID if available.
            if not isinstance(item, dict):
                print(f"Skipping invalid dataset record at position {index}: expected JSON object.")
                continue

            question_id = str(item.get("question_id", index - 1))

            # Skip completed questions.
            if question_id in completed_questions:

                print(
                    f"[{index}/{total_questions}] "
                    f"Skipping question {question_id} "
                    f"(already evaluated)"
                )

                continue

            db_id = item["db_id"]

            question = item["question"]

            print("\n" + "=" * 65)

            print(f"Question ID: {question_id}")

            print(f"Database: {db_id}")

            print(f"Question: {question}")

            print("=" * 65)

            try:

                # Find the correct BIRD SQLite database.
                db_path = get_database_path(db_id)

                # Evaluate question using evaluator.py.
                result = evaluate_question(

                    run_id=run_id,

                    question_id=question_id,

                    item=item,

                    db_path=db_path,

                    request_interval=request_interval,

                    max_retries=max_retries,

                    daily_limit=daily_limit

                )

                print_progress(
                    index,
                    total_questions,
                    question_id,
                    result
                )

                # Update run progress.
                update_run_statistics(run_id)

                if result.get("status") in [
                    "success",
                    "incorrect"
                ]:

                    completed_questions.add(question_id)

            except Exception as error:

                if str(error) == "DAILY_BUDGET_REACHED":

                    print("\nDaily Gemini request budget reached.")

                    print(
                        "Evaluation paused safely."
                    )

                    break

                print(
                    f"\nQuestion {question_id} failed: {error}"
                )
                traceback.print_exc()

                # Continue to the next question.
                # Failed questions can be retried on resume.

    except KeyboardInterrupt:

        print("\nEvaluation interrupted by user.")

        print("Saving progress...")

    except Exception as error:

        print(
            f"\nUnexpected evaluation error: {error}"
        )

    finally:

        # Check actual database progress.
        completed_questions = get_completed_questions(
            run_id
        )

        if len(completed_questions) >= total_questions:

            final_status = "completed"

        else:

            final_status = "paused"

        update_run_status(
            run_id,
            final_status
        )

        update_run_statistics(
            run_id
        )

        show_statistics(
            run_id,
            total_questions
        )

        print(
            f"\nEvaluation status: {final_status}"
        )

        print(
            "Results saved at:"
        )

        print(
            BASE_DIR / "evaluation" / "bird_results.db"
        )


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "BIRD-Mini-Dev NL2SQL Evaluation Runner"
        )

    )

    parser.add_argument(

        "--run-id",

        type=str,

        default="bird-gemini-baseline",

        help="Unique evaluation run ID"

    )

    parser.add_argument(

        "--limit",

        type=int,

        default=None,

        help="Number of questions to evaluate"

    )

    parser.add_argument(

        "--interval",

        type=float,

        default=DEFAULT_REQUEST_INTERVAL,

        help="Minimum seconds between Gemini requests"

    )

    parser.add_argument(

        "--retries",

        type=int,

        default=DEFAULT_MAX_RETRIES,

        help="Maximum retries for rate-limit errors"

    )

    parser.add_argument(

        "--daily-limit",

        type=int,

        default=DEFAULT_MAX_CALLS_PER_DAY,

        help="Maximum Gemini API attempts per UTC day"

    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    if args.interval < 0:
        parser.error("--interval cannot be negative")

    if args.retries < 1:
        parser.error("--retries must be at least 1")

    if args.daily_limit < 1:
        parser.error("--daily-limit must be at least 1")

    run_evaluation(

        run_id=args.run_id,

        limit=args.limit,

        request_interval=args.interval,

        max_retries=args.retries,

        daily_limit=args.daily_limit

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
