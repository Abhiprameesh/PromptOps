import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from app.evaluation.results import EvaluationResult

DB_PATH = Path("data/evaluation.db")


class Database:

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self.create_tables()
        self.migrate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_tables(self):
        """
        Creates base evaluation tables if they do not exist.
        """
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs(
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prompt_version TEXT,
            model TEXT,
            total_cases INTEGER,
            passed_cases INTEGER,
            failed_cases INTEGER,
            accuracy REAL
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS case_results(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            case_id TEXT,
            expected_category TEXT,
            predicted_category TEXT,
            passed INTEGER,
            error TEXT,
            FOREIGN KEY(run_id)
            REFERENCES evaluation_runs(run_id)
        )
        """)

        self.conn.commit()

    def migrate(self):
        """
        Idempotently inspects and adds missing columns for Evaluation Engine V2.
        Uses PRAGMA table_info to ensure columns are only added if missing.
        """
        # Evaluation Engine V2 columns for case_results
        case_results_columns = {
            "expected_priority": "TEXT",
            "predicted_priority": "TEXT",
            "category_passed": "INTEGER",
            "priority_passed": "INTEGER",
            "latency_ms": "REAL",
            "raw_response": "TEXT",
        }

        # Evaluation Engine V2 columns for evaluation_runs
        evaluation_runs_columns = {
            "category_accuracy": "REAL",
            "priority_accuracy": "REAL",
            "composite_accuracy": "REAL",
            "avg_latency_ms": "REAL",
        }

        self._add_missing_columns("case_results", case_results_columns)
        self._add_missing_columns("evaluation_runs", evaluation_runs_columns)
        self.conn.commit()

    def _add_missing_columns(self, table_name: str, column_defs: dict):
        """
        Inspects existing columns in table_name and executes ALTER TABLE ADD COLUMN
        for any column not currently present.
        """
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in self.cursor.fetchall()}

        for col_name, col_type in column_defs.items():
            if col_name not in existing_columns:
                self.cursor.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                )

    def save_run(
        self,
        result: EvaluationResult,
        prompt_version: str,
        model: str,
    ) -> int:
        """
        Saves an evaluation run and its case results using explicit column names
        and atomic transaction handling.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute(
                """
                INSERT INTO evaluation_runs(
                    timestamp,
                    prompt_version,
                    model,
                    total_cases,
                    passed_cases,
                    failed_cases,
                    accuracy,
                    category_accuracy,
                    priority_accuracy,
                    composite_accuracy,
                    avg_latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    prompt_version,
                    model,
                    result.total_cases,
                    result.passed_cases,
                    result.failed_cases,
                    result.accuracy,
                    result.category_accuracy,
                    result.priority_accuracy,
                    result.composite_accuracy,
                    result.avg_latency_ms,
                ),
            )

            run_id = self.cursor.lastrowid

            for case in result.case_results:
                self.cursor.execute(
                    """
                    INSERT INTO case_results(
                        run_id,
                        case_id,
                        expected_category,
                        predicted_category,
                        passed,
                        error,
                        expected_priority,
                        predicted_priority,
                        category_passed,
                        priority_passed,
                        latency_ms,
                        raw_response
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        case.case_id,
                        case.expected_category,
                        case.predicted_category,
                        int(case.passed),
                        case.error,
                        case.expected_priority,
                        case.predicted_priority,
                        int(case.category_passed)
                        if case.category_passed is not None
                        else None,
                        int(case.priority_passed)
                        if case.priority_passed is not None
                        else None,
                        case.latency_ms,
                        case.raw_response,
                    ),
                )

            self.conn.commit()
            return run_id
        except Exception:
            self.conn.rollback()
            raise

    # ---------------------------------------------------------
    # Legacy Query Methods - Guaranteed Positional Tuple Contracts
    # ---------------------------------------------------------

    def get_run_by_id(self, run_id):
        self.cursor.execute("""
            SELECT
                run_id,
                timestamp,
                prompt_version,
                model,
                total_cases,
                passed_cases,
                failed_cases,
                accuracy
            FROM evaluation_runs
            WHERE run_id = ?
        """, (run_id,))
        return self.cursor.fetchone()

    def get_case_results(self, run_id):
        self.cursor.execute("""
            SELECT
                case_id,
                expected_category,
                predicted_category,
                passed,
                error
            FROM case_results
            WHERE run_id = ?
            ORDER BY case_id
        """, (run_id,))
        return self.cursor.fetchall()

    def get_all_runs(self):
        self.cursor.execute("""
            SELECT
                run_id,
                timestamp,
                prompt_version,
                model,
                accuracy,
                passed_cases,
                failed_cases
            FROM evaluation_runs
            ORDER BY run_id DESC
        """)
        return self.cursor.fetchall()

    def get_latest_run(self):
        self.cursor.execute("""
            SELECT
                run_id,
                timestamp,
                prompt_version,
                model,
                total_cases,
                passed_cases,
                failed_cases,
                accuracy
            FROM evaluation_runs
            ORDER BY run_id DESC
            LIMIT 1
        """)
        return self.cursor.fetchone()

    def get_run_summary(self, run_id):
        self.cursor.execute("""
            SELECT
                run_id,
                timestamp,
                prompt_version,
                model,
                total_cases,
                passed_cases,
                failed_cases,
                accuracy
            FROM evaluation_runs
            WHERE run_id = ?
        """, (run_id,))
        return self.cursor.fetchone()

    def get_run_details(self, run_id):
        self.cursor.execute("""
            SELECT
                case_id,
                expected_category,
                predicted_category,
                passed,
                error
            FROM case_results
            WHERE run_id = ?
            ORDER BY case_id
        """, (run_id,))
        return self.cursor.fetchall()

    def get_last_two_runs(self):
        self.cursor.execute("""
            SELECT
                run_id,
                timestamp,
                prompt_version,
                model,
                accuracy
            FROM evaluation_runs
            ORDER BY run_id DESC
            LIMIT 2
        """)
        return self.cursor.fetchall()

    # ---------------------------------------------------------
    # V2 Extended Query Methods (Non-breaking Additions)
    # ---------------------------------------------------------

    def get_extended_run_by_id(self, run_id):
        self.cursor.execute("""
            SELECT
                run_id,
                timestamp,
                prompt_version,
                model,
                total_cases,
                passed_cases,
                failed_cases,
                accuracy,
                category_accuracy,
                priority_accuracy,
                composite_accuracy,
                avg_latency_ms
            FROM evaluation_runs
            WHERE run_id = ?
        """, (run_id,))
        return self.cursor.fetchone()

    def get_extended_case_results(self, run_id):
        self.cursor.execute("""
            SELECT
                case_id,
                expected_category,
                predicted_category,
                passed,
                error,
                expected_priority,
                predicted_priority,
                category_passed,
                priority_passed,
                latency_ms,
                raw_response
            FROM case_results
            WHERE run_id = ?
            ORDER BY case_id
        """, (run_id,))
        return self.cursor.fetchall()

    def close(self):
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
