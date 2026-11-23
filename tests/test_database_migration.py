import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.evaluation.results import CaseResult, EvaluationResult
from app.storage.database import Database


class TestDatabaseMigrationV2(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and copy the original backup DB for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test_evaluation.db"

        backup_path = Path("data/evaluation_backup.db")
        if backup_path.exists():
            shutil.copyfile(backup_path, self.test_db_path)
        else:
            # Fallback to current DB if backup isn't present
            shutil.copyfile("data/evaluation.db", self.test_db_path)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_adds_new_columns_and_preserves_historical_records(self):
        """Verify migration adds all V2 columns and preserves existing runs/cases."""
        with Database(self.test_db_path) as db:
            # Verify 5 historical runs exist intact
            runs = db.get_all_runs()
            self.assertEqual(len(runs), 5, "All 5 historical runs must be preserved")

            # Verify latest run data matches expected values
            latest_run = db.get_run_by_id(5)
            self.assertIsNotNone(latest_run)
            self.assertEqual(latest_run[0], 5)  # run_id
            self.assertEqual(latest_run[2], "v1")  # prompt_version
            self.assertEqual(latest_run[3], "gemma3:4b")  # model
            self.assertEqual(latest_run[7], 40.0)  # accuracy

            # Inspect evaluation_runs columns via PRAGMA
            db.cursor.execute("PRAGMA table_info(evaluation_runs)")
            eval_cols = [r[1] for r in db.cursor.fetchall()]
            for expected_col in [
                "category_accuracy",
                "priority_accuracy",
                "composite_accuracy",
                "avg_latency_ms",
            ]:
                self.assertIn(expected_col, eval_cols)

            # Inspect case_results columns via PRAGMA
            db.cursor.execute("PRAGMA table_info(case_results)")
            case_cols = [r[1] for r in db.cursor.fetchall()]
            for expected_col in [
                "expected_priority",
                "predicted_priority",
                "category_passed",
                "priority_passed",
                "latency_ms",
                "raw_response",
            ]:
                self.assertIn(expected_col, case_cols)

    def test_migration_idempotency(self):
        """Verify running migration multiple times causes zero errors and no duplicate columns."""
        with Database(self.test_db_path) as db:
            # Run migrate multiple times
            db.migrate()
            db.migrate()

            db.cursor.execute("PRAGMA table_info(evaluation_runs)")
            eval_cols = [r[1] for r in db.cursor.fetchall()]
            self.assertEqual(
                len(eval_cols),
                len(set(eval_cols)),
                "Columns in evaluation_runs must be unique",
            )

            db.cursor.execute("PRAGMA table_info(case_results)")
            case_cols = [r[1] for r in db.cursor.fetchall()]
            self.assertEqual(
                len(case_cols),
                len(set(case_cols)),
                "Columns in case_results must be unique",
            )

    def test_legacy_positional_tuple_contracts(self):
        """Verify legacy queries return the exact same column counts and positional indices."""
        with Database(self.test_db_path) as db:
            # get_run_by_id must return 8 elements
            run = db.get_run_by_id(5)
            self.assertEqual(len(run), 8)
            run_id, ts, prompt, model, total, passed, failed, accuracy = run
            self.assertEqual(run_id, 5)
            self.assertEqual(accuracy, 40.0)

            # get_all_runs must return 7 elements per row
            all_runs = db.get_all_runs()
            for r in all_runs:
                self.assertEqual(len(r), 7)

            # get_last_two_runs must return 5 elements per row
            last_two = db.get_last_two_runs()
            self.assertEqual(len(last_two), 2)
            for r in last_two:
                self.assertEqual(len(r), 5)

            # get_case_results must return 5 elements per row
            cases = db.get_case_results(5)
            self.assertEqual(len(cases), 5)
            for c in cases:
                self.assertEqual(len(c), 5)
                case_id, expected_cat, pred_cat, passed_val, err = c
                self.assertIn(passed_val, (0, 1))

    def test_save_run_with_legacy_and_v2_results(self):
        """Verify saving runs with legacy (None) and populated V2 fields works correctly."""
        with Database(self.test_db_path) as db:
            # 1. Save run with legacy fields (V2 fields are None)
            legacy_case = CaseResult(
                case_id="GH_LEGACY",
                expected_category="bug",
                predicted_category="bug",
                passed=True,
            )
            legacy_eval = EvaluationResult(
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
                accuracy=100.0,
                case_results=[legacy_case],
            )
            legacy_run_id = db.save_run(legacy_eval, "v1", "test-model")
            self.assertGreater(legacy_run_id, 5)

            # Check legacy query
            saved_legacy = db.get_run_by_id(legacy_run_id)
            self.assertEqual(saved_legacy[7], 100.0)

            # Extended query should show NULL for V2 fields
            extended_legacy = db.get_extended_run_by_id(legacy_run_id)
            self.assertIsNone(extended_legacy[8])  # category_accuracy is None

            extended_legacy_case = db.get_extended_case_results(legacy_run_id)[0]
            self.assertIsNone(extended_legacy_case[7])  # category_passed is None
            self.assertIsNone(extended_legacy_case[8])  # priority_passed is None

            # 2. Save run with full V2 fields populated
            v2_case = CaseResult(
                case_id="GH_V2",
                expected_category="bug",
                predicted_category="bug",
                passed=True,
                expected_priority="high",
                predicted_priority="high",
                category_passed=True,
                priority_passed=False,
                latency_ms=85.4,
                raw_response='{"category": "bug", "priority": "high"}',
            )
            v2_eval = EvaluationResult(
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
                accuracy=100.0,
                case_results=[v2_case],
                category_accuracy=100.0,
                priority_accuracy=0.0,
                composite_accuracy=0.0,
                avg_latency_ms=85.4,
                difficulty_breakdown={"easy": 100.0},
            )
            v2_run_id = db.save_run(v2_eval, "v2", "test-model-v2")

            extended_v2 = db.get_extended_run_by_id(v2_run_id)
            self.assertEqual(extended_v2[8], 100.0)  # category_accuracy
            self.assertEqual(extended_v2[9], 0.0)  # priority_accuracy
            self.assertEqual(extended_v2[10], 0.0)  # composite_accuracy
            self.assertEqual(extended_v2[11], 85.4)  # avg_latency_ms

            extended_case = db.get_extended_case_results(v2_run_id)[0]
            self.assertEqual(extended_case[5], "high")  # expected_priority
            self.assertEqual(extended_case[6], "high")  # predicted_priority
            self.assertEqual(extended_case[7], 1)  # category_passed (True -> 1)
            self.assertEqual(extended_case[8], 0)  # priority_passed (False -> 0)
            self.assertEqual(extended_case[9], 85.4)  # latency_ms
            self.assertEqual(
                extended_case[10], '{"category": "bug", "priority": "high"}'
            )


if __name__ == "__main__":
    unittest.main()
