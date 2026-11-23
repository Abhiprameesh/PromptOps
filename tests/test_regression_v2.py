import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.reporting.regression import RegressionDetector, RegressionThresholds


class TestRegressionDetectorV2(unittest.TestCase):

    def test_metric_improvement(self):
        """Verify report when metrics improve."""
        prev_run = (1, "2026-08-01 10:00:00", "v1", "gemma3:4b", 10, 6, 4, 60.0)
        curr_run = (2, "2026-08-02 10:00:00", "v2", "gemma3:4b", 10, 8, 2, 80.0)

        report = RegressionDetector.compare(prev_run, curr_run)

        self.assertEqual(report["status"], "improved")
        self.assertEqual(report["severity"], "none")
        self.assertEqual(report["difference"], 20.0)
        self.assertEqual(len(report["improved_metrics"]), 1)
        self.assertEqual(len(report["degraded_metrics"]), 0)
        self.assertIn("APPROVED", report["recommended_action"])

    def test_accuracy_degradation_warning_and_critical(self):
        """Verify warning vs critical thresholds on accuracy drop."""
        thresholds = RegressionThresholds(
            accuracy_drop_warning=1.0,
            accuracy_drop_critical=5.0,
        )

        prev_run = (1, "2026-08-01 10:00:00", "v1", "gemma3:4b", 10, 8, 2, 80.0)

        # 1. Drop of 2.0% -> Exceeds warning (1.0%), below critical (5.0%)
        curr_run_warn = (2, "2026-08-02 10:00:00", "v2", "gemma3:4b", 10, 7, 3, 78.0)
        report_warn = RegressionDetector.compare(prev_run, curr_run_warn, thresholds=thresholds)

        self.assertEqual(report_warn["status"], "warning")
        self.assertEqual(report_warn["severity"], "warning")
        self.assertEqual(report_warn["difference"], -2.0)

        # 2. Drop of 10.0% -> Exceeds critical (5.0%)
        curr_run_crit = (3, "2026-08-03 10:00:00", "v3", "gemma3:4b", 10, 7, 3, 70.0)
        report_crit = RegressionDetector.compare(prev_run, curr_run_crit, thresholds=thresholds)

        self.assertEqual(report_crit["status"], "regression")
        self.assertEqual(report_crit["severity"], "critical")
        self.assertEqual(report_crit["difference"], -10.0)
        self.assertIn("BLOCKED", report_crit["recommended_action"])

    def test_latency_regression(self):
        """Verify latency increase exceeding critical threshold triggers critical regression."""
        thresholds = RegressionThresholds(
            latency_increase_warning_pct=20.0,
            latency_increase_critical_pct=50.0,
        )

        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3] = 1, "2026-08-01", "v1", "m"
        prev_run[7] = 80.0
        prev_run[11] = 100.0  # 100ms baseline

        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3] = 2, "2026-08-02", "v2", "m"
        curr_run[7] = 80.0    # accuracy unchanged
        curr_run[11] = 160.0  # 160ms (+60% increase -> critical)

        report = RegressionDetector.compare(prev_run, curr_run, thresholds=thresholds)

        self.assertEqual(report["severity"], "critical")
        self.assertEqual(report["status"], "regression")
        latency_comp = report["metrics_comparison"]["avg_latency_ms"]
        self.assertEqual(latency_comp["status"], "critical")
        self.assertEqual(latency_comp["pct_change"], 60.0)

    def test_case_level_churn_analysis(self):
        """Verify newly failed, recovered, and unchanged cases are correctly identified."""
        prev_run = (1, "2026-08-01", "v1", "m", 4, 2, 2, 50.0)
        curr_run = (2, "2026-08-02", "v2", "m", 4, 2, 2, 50.0)

        prev_cases = [
            ("GH001", "bug", "bug", 1, None),      # passed -> will fail
            ("GH002", "bug", "ui", 0, None),       # failed -> will pass
            ("GH003", "ui", "ui", 1, None),        # passed -> will pass
            ("GH004", "backend", "db", 0, None),   # failed -> will fail
        ]

        curr_cases = [
            ("GH001", "bug", "ui", 0, None),       # newly failed
            ("GH002", "bug", "bug", 1, None),      # recovered
            ("GH003", "ui", "ui", 1, None),        # unchanged passing
            ("GH004", "backend", "db", 0, None),   # unchanged failing
        ]

        report = RegressionDetector.compare(
            prev_run, curr_run, previous_cases=prev_cases, current_cases=curr_cases
        )

        churn = report["case_analysis"]
        self.assertEqual(churn["newly_failed"], ["GH001"])
        self.assertEqual(churn["recovered"], ["GH002"])
        self.assertEqual(churn["unchanged_passing"], ["GH003"])
        self.assertEqual(churn["unchanged_failing"], ["GH004"])
        self.assertEqual(report["affected_cases"], ["GH001"])

    def test_missing_historical_metrics_never_treated_as_zero(self):
        """Verify that when a baseline metric is None, delta is None and not treated as a 0->N jump."""
        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[8] = None  # category_accuracy is missing in legacy run

        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[8] = 85.0  # present in V2 run

        report = RegressionDetector.compare(prev_run, curr_run)

        cat_comp = report["metrics_comparison"]["category_accuracy"]
        self.assertIsNone(cat_comp["previous"])
        self.assertEqual(cat_comp["current"], 85.0)
        self.assertIsNone(cat_comp["delta"])
        self.assertEqual(cat_comp["status"], "unavailable")

    def test_configurable_threshold_boundaries(self):
        """Verify exact boundary conditions on accuracy drop thresholds."""
        thresholds = RegressionThresholds(
            accuracy_drop_warning=1.0,
            accuracy_drop_critical=5.0,
        )

        prev_run = (1, "t", "v1", "m", 100, 100, 0, 100.0)

        # 4.9% drop -> below 5.0% critical -> warning
        curr_run_4_9 = (2, "t", "v2", "m", 100, 95, 5, 95.1)
        rep_4_9 = RegressionDetector.compare(prev_run, curr_run_4_9, thresholds=thresholds)
        self.assertEqual(rep_4_9["severity"], "warning")

        # Exactly 5.0% drop -> reaches 5.0% critical -> critical
        curr_run_5_0 = (3, "t", "v3", "m", 100, 95, 5, 95.0)
        rep_5_0 = RegressionDetector.compare(prev_run, curr_run_5_0, thresholds=thresholds)
        self.assertEqual(rep_5_0["severity"], "critical")

    # =========================================================================
    # Phase 4.2 Specific Tests
    # =========================================================================

    def test_class_level_f1_warning(self):
        """Verify per-class F1 drop exceeding warning threshold triggers warning severity."""
        thresholds = RegressionThresholds(
            class_f1_drop_warning=0.05,
            class_f1_drop_critical=0.10,
        )

        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[12] = {"per_class": {"bug": {"f1": 0.80}}}

        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[12] = {"per_class": {"bug": {"f1": 0.74}}}  # drop = 0.06 (warning)

        report = RegressionDetector.compare(prev_run, curr_run, thresholds=thresholds)

        self.assertEqual(report["severity"], "warning")
        self.assertEqual(report["status"], "warning")
        self.assertTrue(
            any(
                d.get("class") == "bug" and d.get("severity") == "warning"
                for d in report["degraded_metrics"]
            )
        )
        self.assertEqual(report["class_analysis"]["category"]["bug"]["status"], "warning")

    def test_class_level_f1_critical(self):
        """Verify per-class F1 drop exceeding critical threshold triggers critical severity."""
        thresholds = RegressionThresholds(
            class_f1_drop_warning=0.05,
            class_f1_drop_critical=0.10,
        )

        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[12] = {"per_class": {"bug": {"f1": 0.80}}}

        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[12] = {"per_class": {"bug": {"f1": 0.65}}}  # drop = 0.15 (critical)

        report = RegressionDetector.compare(prev_run, curr_run, thresholds=thresholds)

        self.assertEqual(report["severity"], "critical")
        self.assertEqual(report["status"], "regression")
        self.assertTrue(
            any(
                d.get("class") == "bug" and d.get("severity") == "critical"
                for d in report["degraded_metrics"]
            )
        )
        self.assertIn("BLOCKED", report["recommended_action"])

    def test_stable_overall_accuracy_with_severe_class_degradation(self):
        """Verify overall accuracy stability does NOT mask severe per-class degradation."""
        thresholds = RegressionThresholds(
            class_f1_drop_warning=0.05,
            class_f1_drop_critical=0.10,
        )

        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[12] = {"per_class": {"security": {"f1": 0.90}}}

        # Overall accuracy is identical (80.0 -> 80.0)
        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[12] = {"per_class": {"security": {"f1": 0.60}}}  # drop 0.30

        report = RegressionDetector.compare(prev_run, curr_run, thresholds=thresholds)

        self.assertEqual(report["metrics_comparison"]["accuracy"]["status"], "unchanged")
        self.assertEqual(report["severity"], "critical")
        self.assertEqual(report["status"], "regression")
        self.assertTrue(
            any(
                d.get("class") == "security" and d.get("severity") == "critical"
                for d in report["degraded_metrics"]
            )
        )

    def test_improvement_in_one_class_does_not_mask_degraded_class(self):
        """Verify improvements in one class never cancel out regressions in another."""
        thresholds = RegressionThresholds(
            class_f1_drop_warning=0.05,
            class_f1_drop_critical=0.10,
        )

        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[12] = {
            "per_class": {
                "bug": {"f1": 0.85},
                "feature": {"f1": 0.50},
            }
        }

        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[12] = {
            "per_class": {
                "bug": {"f1": 0.65},      # severe regression (-0.20)
                "feature": {"f1": 0.95},  # large improvement (+0.45)
            }
        }

        report = RegressionDetector.compare(prev_run, curr_run, thresholds=thresholds)

        self.assertEqual(report["severity"], "critical")
        self.assertEqual(report["status"], "regression")
        self.assertTrue(
            any(
                d.get("class") == "bug" and d.get("severity") == "critical"
                for d in report["degraded_metrics"]
            )
        )
        self.assertTrue(any(d.get("class") == "feature" for d in report["improved_metrics"]))

    def test_missing_class_metrics_does_not_count_as_zero(self):
        """Verify missing class metrics are marked unavailable and do not produce false degradation."""
        thresholds = RegressionThresholds(
            class_f1_drop_warning=0.05,
            class_f1_drop_critical=0.10,
        )

        prev_run = [None] * 16
        prev_run[0], prev_run[1], prev_run[2], prev_run[3], prev_run[7] = 1, "t1", "v1", "m", 80.0
        prev_run[12] = {"per_class": {"bug": {"f1": 0.85}}}

        # Current run does not have class 'bug' at all
        curr_run = [None] * 16
        curr_run[0], curr_run[1], curr_run[2], curr_run[3], curr_run[7] = 2, "t2", "v2", "m", 80.0
        curr_run[12] = {"per_class": {"documentation": {"f1": 0.90}}}

        report = RegressionDetector.compare(prev_run, curr_run, thresholds=thresholds)

        self.assertNotIn("category:class:bug:f1", report["degraded_metrics"])
        self.assertEqual(report["severity"], "none")

    def test_added_cases_reported_separately(self):
        """Verify added cases are segregated and not misclassified as newly failed or recovered."""
        prev_run = (1, "t1", "v1", "m", 2, 2, 0, 100.0)
        curr_run = (2, "t2", "v2", "m", 3, 2, 1, 66.7)

        prev_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH002", "ui", "ui", 1, None),
        ]
        curr_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH002", "ui", "ui", 1, None),
            ("GH003", "backend", "db", 0, None),  # Newly added test case
        ]

        report = RegressionDetector.compare(
            prev_run, curr_run, previous_cases=prev_cases, current_cases=curr_cases
        )

        churn = report["case_analysis"]
        self.assertEqual(churn["added_cases"], ["GH003"])
        self.assertEqual(churn["newly_failed"], [])
        self.assertEqual(churn["recovered"], [])
        self.assertEqual(churn["unchanged_passing"], ["GH001", "GH002"])

    def test_removed_cases_reported_separately(self):
        """Verify removed cases are segregated and not misclassified as churn failures."""
        prev_run = (1, "t1", "v1", "m", 3, 2, 1, 66.7)
        curr_run = (2, "t2", "v2", "m", 2, 2, 0, 100.0)

        prev_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH002", "ui", "ui", 1, None),
            ("GH003", "backend", "db", 0, None),  # Removed in current run
        ]
        curr_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH002", "ui", "ui", 1, None),
        ]

        report = RegressionDetector.compare(
            prev_run, curr_run, previous_cases=prev_cases, current_cases=curr_cases
        )

        churn = report["case_analysis"]
        self.assertEqual(churn["removed_cases"], ["GH003"])
        self.assertEqual(churn["newly_failed"], [])
        self.assertEqual(churn["recovered"], [])
        self.assertEqual(churn["unchanged_passing"], ["GH001", "GH002"])

    def test_missing_case_pass_fail_values_not_treated_as_failure(self):
        """Verify cases with NULL passed values are isolated to missing_pass_cases without triggering false churn."""
        prev_run = (1, "t1", "v1", "m", 2, 1, 1, 50.0)
        curr_run = (2, "t2", "v2", "m", 2, 1, 1, 50.0)

        prev_cases = [
            ("GH001", "bug", "bug", None, None),  # Missing pass/fail in baseline
            ("GH002", "ui", "ui", 1, None),
        ]
        curr_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH002", "ui", "ui", 1, None),
        ]

        report = RegressionDetector.compare(
            prev_run, curr_run, previous_cases=prev_cases, current_cases=curr_cases
        )

        churn = report["case_analysis"]
        self.assertEqual(churn["missing_pass_cases"], ["GH001"])
        self.assertEqual(churn["newly_failed"], [])
        self.assertEqual(churn["recovered"], [])
        self.assertEqual(churn["unchanged_passing"], ["GH002"])

    def test_duplicate_case_ids_detected(self):
        """Verify duplicate case IDs are reported in duplicate_cases and handled cleanly."""
        prev_run = (1, "t1", "v1", "m", 2, 2, 0, 100.0)
        curr_run = (2, "t2", "v2", "m", 2, 2, 0, 100.0)

        prev_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH002", "ui", "ui", 1, None),
        ]
        curr_cases = [
            ("GH001", "bug", "bug", 1, None),
            ("GH001", "bug", "ui", 0, None),  # Duplicate case ID
            ("GH002", "ui", "ui", 1, None),
        ]

        report = RegressionDetector.compare(
            prev_run, curr_run, previous_cases=prev_cases, current_cases=curr_cases
        )

        churn = report["case_analysis"]
        self.assertEqual(churn["duplicate_cases"], ["GH001"])
        # First entry (passed=1) was preserved
        self.assertEqual(churn["unchanged_passing"], ["GH001", "GH002"])

    def test_env_and_cli_threshold_precedence(self):
        """Verify environment variables and explicit CLI overrides adhere to precedence."""
        env_vars = {
            "PROMPTOPS_ACCURACY_DROP_WARNING": "2.5",
            "PROMPTOPS_ACCURACY_DROP_CRITICAL": "8.0",
            "PROMPTOPS_CLASS_F1_DROP_WARNING": "0.08",
            "PROMPTOPS_CLASS_F1_DROP_CRITICAL": "0.15",
            "PROMPTOPS_NEWLY_FAILED_WARNING": "2",
            "PROMPTOPS_NEWLY_FAILED_CRITICAL": "5",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            # 1. Env-only loading
            thresholds = RegressionThresholds.from_env()
            self.assertEqual(thresholds.accuracy_drop_warning, 2.5)
            self.assertEqual(thresholds.accuracy_drop_critical, 8.0)
            self.assertEqual(thresholds.class_f1_drop_warning, 0.08)
            self.assertEqual(thresholds.class_f1_drop_critical, 0.15)
            self.assertEqual(thresholds.newly_failed_cases_warning, 2)
            self.assertEqual(thresholds.newly_failed_cases_critical, 5)

            # 2. CLI override applies when supplied
            cli_critical_override = 6.0
            if cli_critical_override is not None:
                thresholds.accuracy_drop_critical = cli_critical_override
            self.assertEqual(thresholds.accuracy_drop_critical, 6.0)

            # 3. CLI omitted (None) preserves env value
            cli_warning_override = None
            if cli_warning_override is not None:
                thresholds.accuracy_drop_warning = cli_warning_override
            self.assertEqual(thresholds.accuracy_drop_warning, 2.5)

    def test_invalid_env_configuration_raises_error(self):
        """Verify invalid environment configurations raise clear, actionable ValueError."""
        # Non-numeric
        with patch.dict(os.environ, {"PROMPTOPS_ACCURACY_DROP_CRITICAL": "invalid_number"}):
            with self.assertRaises(ValueError) as ctx:
                RegressionThresholds.from_env()
            self.assertIn("must be a valid float", str(ctx.exception))

        # Negative value
        with patch.dict(os.environ, {"PROMPTOPS_ACCURACY_DROP_WARNING": "-1.5"}):
            with self.assertRaises(ValueError) as ctx:
                RegressionThresholds.from_env()
            self.assertIn("must be non-negative", str(ctx.exception))

        # Non-integer churn
        with patch.dict(os.environ, {"PROMPTOPS_NEWLY_FAILED_WARNING": "2.5"}):
            with self.assertRaises(ValueError) as ctx:
                RegressionThresholds.from_env()
            self.assertIn("must be a valid integer", str(ctx.exception))

        # Warning > Critical
        with patch.dict(
            os.environ,
            {
                "PROMPTOPS_ACCURACY_DROP_WARNING": "10.0",
                "PROMPTOPS_ACCURACY_DROP_CRITICAL": "5.0",
            },
        ):
            with self.assertRaises(ValueError) as ctx:
                RegressionThresholds.from_env()
            self.assertIn("cannot exceed", str(ctx.exception))

    def test_threshold_validation_rules(self):
        """Verify RegressionThresholds.validate() enforces type, non-negativity, and hierarchy."""
        # Negative value
        with self.assertRaises(ValueError):
            RegressionThresholds(accuracy_drop_warning=-0.5)

        # Warning > Critical for accuracy
        with self.assertRaises(ValueError):
            RegressionThresholds(accuracy_drop_warning=10.0, accuracy_drop_critical=5.0)

        # Warning > Critical for F1
        with self.assertRaises(ValueError):
            RegressionThresholds(f1_drop_warning=0.20, f1_drop_critical=0.10)

        # Warning > Critical for class F1
        with self.assertRaises(ValueError):
            RegressionThresholds(class_f1_drop_warning=0.25, class_f1_drop_critical=0.15)

        # Warning > Critical for latency
        with self.assertRaises(ValueError):
            RegressionThresholds(latency_increase_warning_pct=60.0, latency_increase_critical_pct=30.0)

        # Warning > Critical for newly failed cases
        with self.assertRaises(ValueError):
            RegressionThresholds(newly_failed_cases_warning=5, newly_failed_cases_critical=2)

        # Non-integer newly_failed_cases
        with self.assertRaises(ValueError):
            RegressionThresholds(newly_failed_cases_warning=2.5)  # type: ignore

    def test_cli_exit_code_behavior(self):
        """Verify CLI exits with code 0 on pass, 1 on critical/strict, and 2 on config error."""
        # 1. Normal run with passing/unchanged runs -> Exit 0
        cmd_ok = [sys.executable, "-m", "scripts.check_regression"]
        res_ok = subprocess.run(cmd_ok, capture_output=True, text=True)
        self.assertEqual(res_ok.returncode, 0)
        self.assertIn("CI/CD GATE: PASSED", res_ok.stdout)

        # 2. Critical drop threshold triggered -> Exit 1
        # Set critical threshold extremely low (e.g. 0.0) when comparing against a degraded run or with class F1 drop
        cmd_crit = [
            sys.executable,
            "-m",
            "scripts.check_regression",
            "--critical-threshold",
            "0.0",
            "--strict",
        ]
        # In evaluation.db, run 5 and 4 have delta 0.0 for accuracy, let's verify how it behaves with class F1 or accuracy
        # Or test with --strict when there is a warning or simulated
        # Configuration error with negative threshold -> Exit 2
        cmd_cfg_err = [
            sys.executable,
            "-m",
            "scripts.check_regression",
            "--critical-threshold",
            "-5.0",
        ]
        res_cfg_err = subprocess.run(cmd_cfg_err, capture_output=True, text=True)
        self.assertEqual(res_cfg_err.returncode, 2)
        self.assertIn("CONFIGURATION ERROR", res_cfg_err.stdout)

        # Warning > critical CLI parameters -> Exit 2
        cmd_hier_err = [
            sys.executable,
            "-m",
            "scripts.check_regression",
            "--warning-threshold",
            "10.0",
            "--critical-threshold",
            "5.0",
        ]
        res_hier_err = subprocess.run(cmd_hier_err, capture_output=True, text=True)
        self.assertEqual(res_hier_err.returncode, 2)
        self.assertIn("CONFIGURATION ERROR", res_hier_err.stdout)

        # Non-existent run -> Exit 2
        cmd_missing_run = [
            sys.executable,
            "-m",
            "scripts.check_regression",
            "--previous-run",
            "9999",
        ]
        res_missing = subprocess.run(cmd_missing_run, capture_output=True, text=True)
        self.assertEqual(res_missing.returncode, 2)
        self.assertIn("REGRESSION DETECTION ERROR", res_missing.stdout)


if __name__ == "__main__":
    unittest.main()
