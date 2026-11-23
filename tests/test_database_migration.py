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
                "category_metrics",
                "priority_metrics",
                "difficulty_breakdown",
                "tag_breakdown",
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
                "summary_valid",
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

            # Extended query should show NULL for all V2 fields and deserialized None
            extended_legacy = db.get_extended_run_by_id(legacy_run_id)
            self.assertIsNone(extended_legacy[8])   # category_accuracy
            self.assertIsNone(extended_legacy[12])  # category_metrics
            self.assertIsNone(extended_legacy[13])  # priority_metrics
            self.assertIsNone(extended_legacy[14])  # difficulty_breakdown
            self.assertIsNone(extended_legacy[15])  # tag_breakdown

            extended_legacy_case = db.get_extended_case_results(legacy_run_id)[0]
            self.assertIsNone(extended_legacy_case[7])  # category_passed is None
            self.assertIsNone(extended_legacy_case[8])  # priority_passed is None
            self.assertIsNone(extended_legacy_case[9])  # summary_valid is None

            # 2. Save run with full V2 fields populated (including JSON analytics & summary_valid)
            case_valid = CaseResult(
                case_id="GH_V2_1",
                expected_category="bug",
                predicted_category="bug",
                passed=True,
                expected_priority="high",
                predicted_priority="high",
                category_passed=True,
                priority_passed=True,
                summary_valid=True,  # Should persist as 1
                latency_ms=85.4,
                raw_response='{"category": "bug", "priority": "high", "summary": "Valid summary"}',
            )
            case_invalid_summary = CaseResult(
                case_id="GH_V2_2",
                expected_category="ui",
                predicted_category="ui",
                passed=False,
                expected_priority="medium",
                predicted_priority="medium",
                category_passed=True,
                priority_passed=True,
                summary_valid=False,  # Should persist as 0
                latency_ms=45.2,
                raw_response='{"category": "ui", "priority": "medium", "summary": ""}',
            )
            case_none_summary = CaseResult(
                case_id="GH_V2_3",
                expected_category="backend",
                predicted_category=None,
                passed=False,
                error="Timeout error",
                summary_valid=None,  # Should persist as NULL
                latency_ms=10.0,
            )

            sample_cat_metrics = {
                "accuracy": 66.7,
                "per_class": {"bug": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
            }
            sample_pri_metrics = {
                "accuracy": 66.7,
                "per_class": {"high": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
            }
            sample_diff_breakdown = {"easy": 100.0, "hard": 50.0}
            sample_tag_breakdown = {"auth": 100.0, "ui": 0.0}

            v2_eval = EvaluationResult(
                total_cases=3,
                passed_cases=1,
                failed_cases=2,
                accuracy=33.3,
                case_results=[case_valid, case_invalid_summary, case_none_summary],
                category_accuracy=66.7,
                priority_accuracy=66.7,
                composite_accuracy=33.3,
                avg_latency_ms=46.87,
                difficulty_breakdown=sample_diff_breakdown,
                tag_breakdown=sample_tag_breakdown,
                category_metrics=sample_cat_metrics,
                priority_metrics=sample_pri_metrics,
            )
            v2_run_id = db.save_run(v2_eval, "v2", "test-model-v2")

            extended_v2 = db.get_extended_run_by_id(v2_run_id)
            self.assertEqual(extended_v2[8], 66.7)   # category_accuracy
            self.assertEqual(extended_v2[9], 66.7)   # priority_accuracy
            self.assertEqual(extended_v2[10], 33.3)  # composite_accuracy
            self.assertEqual(extended_v2[11], 46.87) # avg_latency_ms

            # Test JSON deserialization into native dictionaries
            self.assertEqual(extended_v2[12]["accuracy"], 66.7)
            self.assertEqual(extended_v2[12]["per_class"]["bug"]["f1"], 1.0)
            self.assertEqual(extended_v2[13]["accuracy"], 66.7)
            self.assertEqual(extended_v2[14]["easy"], 100.0)
            self.assertEqual(extended_v2[14]["hard"], 50.0)
            self.assertEqual(extended_v2[15]["auth"], 100.0)
            self.assertEqual(extended_v2[15]["ui"], 0.0)

            # Test case_results retrieval and summary_valid values (1, 0, NULL)
            extended_cases = db.get_extended_case_results(v2_run_id)
            self.assertEqual(len(extended_cases), 3)

            # Case 1: summary_valid = True -> 1
            self.assertEqual(extended_cases[0][0], "GH_V2_1")
            self.assertEqual(extended_cases[0][7], 1)   # category_passed
            self.assertEqual(extended_cases[0][8], 1)   # priority_passed
            self.assertEqual(extended_cases[0][9], 1)   # summary_valid
            self.assertEqual(extended_cases[0][10], 85.4)

            # Case 2: summary_valid = False -> 0
            self.assertEqual(extended_cases[1][0], "GH_V2_2")
            self.assertEqual(extended_cases[1][9], 0)   # summary_valid

            # Case 3: summary_valid = None -> None (NULL)
            self.assertEqual(extended_cases[2][0], "GH_V2_3")
            self.assertIsNone(extended_cases[2][9])     # summary_valid


if __name__ == "__main__":
    unittest.main()
