import sqlite3
from datetime import datetime
from pathlib import Path

from app.evaluation.results import EvaluationResult

DB_PATH = Path("data/evaluation.db")


class Database:
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

    def __init__(self):

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()

        self.create_tables()

    def create_tables(self):

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

    def save_run(
        self,
        result: EvaluationResult,
        prompt_version: str,
        model: str,
    ):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute(
            """
            INSERT INTO evaluation_runs(
                timestamp,
                prompt_version,
                model,
                total_cases,
                passed_cases,
                failed_cases,
                accuracy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                prompt_version,
                model,
                result.total_cases,
                result.passed_cases,
                result.failed_cases,
                result.accuracy,
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
                    error
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    case.case_id,
                    case.expected_category,
                    case.predicted_category,
                    int(case.passed),
                    case.error,
                ),
            )

        self.conn.commit()

        return run_id

    def close(self):
        self.conn.close()