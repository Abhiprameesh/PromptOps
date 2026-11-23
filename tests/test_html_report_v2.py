import os
import re
import tempfile
import unittest
from pathlib import Path

from app.evaluation.results import CaseResult, EvaluationResult
from app.reporting.html_report import HTMLReportGenerator
from app.reporting.regression import RegressionDetector, RegressionThresholds


class TestHTMLReportV2(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.orig_report_dir = HTMLReportGenerator.REPORT_DIR
        HTMLReportGenerator.REPORT_DIR = Path(self.temp_dir.name)

        # Complete V2 EvaluationResult fixture
        self.v2_cases = [
            CaseResult(
                case_id="GH001",
                expected_category="bug",
                predicted_category="bug",
                passed=True,
                expected_priority="high",
                predicted_priority="high",
                category_passed=True,
                priority_passed=True,
                summary_valid=True,
                latency_ms=125.4,
            ),
            CaseResult(
                case_id="GH002",
                expected_category="feature",
                predicted_category="bug",
                passed=False,
                expected_priority="low",
                predicted_priority="low",
                category_passed=False,
                priority_passed=True,
                summary_valid=True,
                latency_ms=180.2,
                error="Category mismatch",
            ),
        ]

        self.v2_result = EvaluationResult(
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            accuracy=50.0,
            category_accuracy=50.0,
            priority_accuracy=100.0,
            composite_accuracy=50.0,
            avg_latency_ms=152.8,
            category_metrics={
                "accuracy": 50.0,
                "per_class": {
                    "bug": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1},
                    "feature": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
                },
                "macro_avg": {"precision": 0.25, "recall": 0.5, "f1": 0.3333},
                "weighted_avg": {"precision": 0.25, "recall": 0.5, "f1": 0.3333},
                "confusion_matrix": {
                    "bug": {"bug": 1, "feature": 0},
                    "feature": {"bug": 1, "feature": 0},
                },
            },
            priority_metrics={
                "accuracy": 100.0,
                "per_class": {
                    "high": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
                    "low": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
                },
                "macro_avg": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "weighted_avg": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "confusion_matrix": {
                    "high": {"high": 1, "low": 0},
                    "low": {"high": 0, "low": 1},
                },
            },
            difficulty_breakdown={
                "easy": {"total": 1, "passed": 1, "accuracy": 100.0},
                "hard": {"total": 1, "passed": 0, "accuracy": 0.0},
            },
            tag_breakdown={
                "core": {"total": 2, "passed": 1, "accuracy": 50.0},
            },
            case_results=self.v2_cases,
        )

        # Historical V1 EvaluationResult fixture
        self.v1_cases = [
            CaseResult(
                case_id="GH001",
                expected_category="bug",
                predicted_category="bug",
                passed=True,
            ),
            CaseResult(
                case_id="GH002",
                expected_category="feature",
                predicted_category="ui",
                passed=False,
                error="Legacy error",
            ),
        ]
        self.v1_result = EvaluationResult(
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            accuracy=50.0,
            case_results=self.v1_cases,
        )

    def tearDown(self):
        HTMLReportGenerator.REPORT_DIR = self.orig_report_dir
        self.temp_dir.cleanup()

    # 1. Complete V2 HTML report generation
    def test_complete_v2_html_report_generation(self):
        report_path = HTMLReportGenerator.generate(
            run_id=101,
            result=self.v2_result,
            prompt_version="v2",
            model="gemma3:4b",
        )
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")

        # Overview checks
        self.assertIn("Evaluation & Regression Report", content)
        self.assertIn("Overall Accuracy", content)
        self.assertIn("50.00%", content)
        self.assertIn("100.00%", content)
        self.assertIn("152.80 ms", content)

        # Classification checks
        self.assertIn("Category Performance", content)
        self.assertIn("Priority Performance", content)
        self.assertIn("Macro Average", content)
        self.assertIn("Weighted Average", content)

        # Slices checks
        self.assertIn("Difficulty Breakdown", content)
        self.assertIn("Tag Breakdown", content)

        # Case explorer
        self.assertIn("GH001", content)
        self.assertIn("GH002", content)
        self.assertIn("Category mismatch", content)

    # 2. Historical V1 HTML report generation
    def test_historical_v1_html_report_generation(self):
        report_path = HTMLReportGenerator.generate(
            run_id=102,
            result=self.v1_result,
            prompt_version="v1",
            model="gemma3:4b",
        )
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")

        # Legacy accuracy must be present
        self.assertIn("50.00%", content)
        # Missing V2 metrics MUST be N/A, never 0.00%
        self.assertIn(">N/A<", content)
        # Missing classification metrics should show unavailable banner, not crash
        self.assertIn("unavailable for this run (legacy V1 run)", content)

    # 3. Missing optional metrics
    def test_missing_optional_metrics(self):
        partial_result = EvaluationResult(
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            accuracy=100.0,
            category_accuracy=100.0,
            # priority_accuracy, composite_accuracy, latency omitted/None
            case_results=self.v1_cases,
        )
        report_path = HTMLReportGenerator.generate(
            run_id=103,
            result=partial_result,
            prompt_version="v2",
            model="gemma3:4b",
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("100.00%", content)
        self.assertIn(">N/A<", content)

    # 4. Missing classification metrics
    def test_missing_classification_metrics(self):
        result_no_class = EvaluationResult(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            accuracy=100.0,
            category_metrics={},
            priority_metrics=None,
            case_results=self.v1_cases,
        )
        report_path = HTMLReportGenerator.generate(
            run_id=104,
            result=result_no_class,
            prompt_version="v2",
            model="gemma3:4b",
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("classification metrics are unavailable", content)

    # 5. Confusion matrix rendering
    def test_confusion_matrix_rendering(self):
        report_path = HTMLReportGenerator.generate(
            run_id=105,
            result=self.v2_result,
            prompt_version="v2",
            model="gemma3:4b",
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Confusion Matrix", content)
        self.assertIn("True: bug", content)
        self.assertIn("Pred: bug", content)
        self.assertIn("cm-match", content)

    # 6. Regression report integration
    def test_regression_report_integration(self):
        prev_run = (1, "2026-08-01", "v1", "gemma3:4b", 2, 2, 0, 100.0)
        curr_run = (2, "2026-08-02", "v2", "gemma3:4b", 2, 1, 1, 50.0)

        regression = RegressionDetector.compare(prev_run, curr_run)
        report_path = HTMLReportGenerator.generate(
            run_id=106,
            result=self.v2_result,
            prompt_version="v2",
            model="gemma3:4b",
            regression_report=regression,
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Regression Detection & CI/CD Gate", content)
        self.assertIn("CRITICAL", content)
        self.assertIn("Recommended Action:", content)
        self.assertIn("Key Metrics Comparison", content)

    # 7. Class-level regression rendering
    def test_class_level_regression_rendering(self):
        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[12] = {"per_class": {"bug": {"f1": 0.85}}}

        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[12] = {"per_class": {"bug": {"f1": 0.65}}}

        regression = RegressionDetector.compare(prev_run, curr_run)
        report_path = HTMLReportGenerator.generate(
            run_id=107,
            result=self.v2_result,
            prompt_version="v2",
            model="m",
            regression_report=regression,
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Degraded Metrics & Classes", content)
        self.assertIn("category_class_bug_f1", content)

    # 8. Case churn rendering
    def test_case_churn_rendering(self):
        prev_cases = [("C1", "bug", "bug", 1, None), ("C2", "ui", "ui", 1, None)]
        curr_cases = [
            ("C1", "bug", "ui", 0, None),     # newly failed
            ("C3", "db", "db", 1, None),      # added
        ]
        prev_run = (1, "t1", "v1", "m", 2, 2, 0, 100.0)
        curr_run = (2, "t2", "v2", "m", 2, 1, 1, 50.0)

        regression = RegressionDetector.compare(
            prev_run, curr_run, previous_cases=prev_cases, current_cases=curr_cases
        )
        report_path = HTMLReportGenerator.generate(
            run_id=108,
            result=self.v2_result,
            prompt_version="v2",
            model="m",
            regression_report=regression,
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Newly Failed", content)
        self.assertIn("Added Cases", content)
        self.assertIn("Removed Cases", content)
        self.assertIn("C1", content)
        self.assertIn("C3", content)

    # 9. Empty metadata breakdowns
    def test_empty_metadata_breakdowns(self):
        res = EvaluationResult(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            accuracy=100.0,
            difficulty_breakdown={},
            tag_breakdown=None,
            case_results=self.v1_cases,
        )
        report_path = HTMLReportGenerator.generate(
            run_id=109,
            result=res,
            prompt_version="v1",
            model="m",
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Difficulty breakdown is unavailable", content)
        self.assertIn("Tag breakdown is unavailable", content)

    # 10. Report generation without a baseline
    def test_report_generation_without_a_baseline(self):
        report_path = HTMLReportGenerator.generate(
            run_id=110,
            result=self.v2_result,
            prompt_version="v2",
            model="gemma3:4b",
            regression_report=None,
        )
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Standalone Run:", content)
        self.assertNotIn("Key Metrics Comparison", content)

    # 11. Standalone HTML asset verification
    def test_standalone_html_asset_verification(self):
        report_path = HTMLReportGenerator.generate(
            run_id=111,
            result=self.v2_result,
            prompt_version="v2",
            model="gemma3:4b",
        )
        content = report_path.read_text(encoding="utf-8")

        # Must not load external scripts or external stylesheets
        self.assertFalse(
            bool(re.search(r'<script[^>]+src=["\']http', content, re.IGNORECASE)),
            "Found external script tag in HTML report",
        )
        self.assertFalse(
            bool(re.search(r'<link[^>]+rel=["\']stylesheet["\'][^>]+href=["\']http', content, re.IGNORECASE)),
            "Found external stylesheet in HTML report",
        )
        # Must contain embedded <style>
        self.assertIn("<style>", content)
        self.assertIn("</style>", content)


if __name__ == "__main__":
    unittest.main()
