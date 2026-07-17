import sqlite3
from pathlib import Path


DB_PATH = Path("data/evaluation.db")


class Database:

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

    def close(self):
        self.conn.close()