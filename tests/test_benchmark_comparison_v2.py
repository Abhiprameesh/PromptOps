import csv
import json
import sqlite3
import unittest
from pathlib import Path

from app.reporting.comparison import (
    BENCHMARK_RUN_MAPPING,
    SUPPORTED_CATEGORIES,
    SUPPORTED_PRIORITIES,
    BenchmarkComparisonEngine,
)


class TestBenchmarkComparisonV2(unittest.TestCase):
    """
    Automated verification suite for Phase 6.3 Multi-Model Comparative Analysis.
    Tests aggregations, metric calculations, confusion matrices, latency statistics,
    artifact validity, and database isolation.
    """

    @classmethod
    def setUpClass(cls):
        cls.db_path = Path("data/validation.db")
        cls.manifest_path = Path("reports/benchmarks/v2/benchmark_manifest.json")
        cls.results_path = Path("reports/benchmarks/v2/benchmark_results.json")
        cls.output_dir = Path("reports/benchmarks/v2/comparison")

        cls.engine = BenchmarkComparisonEngine(
            db_path=cls.db_path,
            manifest_path=cls.manifest_path,
            results_path=cls.results_path,
            output_dir=cls.output_dir,
        )
        cls.audit_res = cls.engine.load_and_audit()
        cls.metrics = cls.engine.compute_comparative_metrics()

    def test_audit_and_aggregation_completeness(self):
        """Verify exactly 3 models, 9 runs, 50 cases per run, and 450 total measured cases."""
        self.assertEqual(self.audit_res["total_models"], 3)
        self.assertEqual(self.audit_res["total_runs"], 9)
        self.assertEqual(self.audit_res["total_cases"], 450)

        # Verify raw case counts per run
        for run_id in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
            self.assertIn(run_id, self.engine.raw_cases_by_run)
            self.assertEqual(len(self.engine.raw_cases_by_run[run_id]), 50)

    def test_warmup_isolation(self):
        """Verify warm-up requests are recorded separately in manifest and excluded from benchmark runs."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("warmups", manifest)
        self.assertEqual(len(manifest["warmups"]), 3)
        self.assertIn("gemma3_4b_warmup", manifest["warmups"])
        self.assertIn("mistral_latest_warmup", manifest["warmups"])
        self.assertIn("llama3_latest_warmup", manifest["warmups"])

        # Confirm warm-up latencies are positive
        for w_key, w_data in manifest["warmups"].items():
            self.assertGreater(w_data["latency_ms"], 0)
            self.assertTrue(w_data["parsed_successfully"])

        # Confirm benchmark run IDs (10-18) contain zero warm-up cases
        for run_id in range(10, 19):
            cases = self.engine.raw_cases_by_run[run_id]
            for c in cases:
                self.assertNotEqual(c["case_id"], "warmup")
                self.assertTrue(c["case_id"].startswith("GH"))

    def test_recalculated_metrics_accuracy(self):
        """Verify recalculated metrics match ground truth and composite accuracy is non-arithmetic."""
        models = self.metrics["models"]

        # Gemma 3 4B
        gemma = models["gemma3:4b"]
        self.assertEqual(gemma["composite_accuracy"], 32.00)
        self.assertEqual(gemma["category_accuracy"], 48.00)
        self.assertEqual(gemma["priority_accuracy"], 52.00)
        # Verify composite is not arithmetic mean of cat and pri: (48 + 52)/2 = 50 != 32
        self.assertNotEqual(gemma["composite_accuracy"], (gemma["category_accuracy"] + gemma["priority_accuracy"]) / 2.0)

        # Mistral 7B
        mistral = models["mistral:latest"]
        self.assertEqual(mistral["composite_accuracy"], 24.00)
        self.assertEqual(mistral["category_accuracy"], 70.00)
        self.assertEqual(mistral["priority_accuracy"], 38.00)
        self.assertNotEqual(mistral["composite_accuracy"], (mistral["category_accuracy"] + mistral["priority_accuracy"]) / 2.0)

        # Llama 3 8B
        llama = models["llama3:latest"]
        self.assertEqual(llama["composite_accuracy"], 26.00)
        self.assertEqual(llama["category_accuracy"], 62.00)
        self.assertEqual(llama["priority_accuracy"], 44.00)
        self.assertNotEqual(llama["composite_accuracy"], (llama["category_accuracy"] + llama["priority_accuracy"]) / 2.0)

    def test_per_class_and_macro_f1_calculations(self):
        """Verify per-class metrics and macro F1 calculations across all 8 categories."""
        for m_name, m_info in self.metrics["models"].items():
            cat_metrics = m_info["category_metrics"]
            per_class = cat_metrics["per_class"]

            # All 8 supported categories must be present
            for cat in SUPPORTED_CATEGORIES:
                self.assertIn(cat, per_class, f"Category '{cat}' missing from {m_name}")
                self.assertIn("precision", per_class[cat])
                self.assertIn("recall", per_class[cat])
                self.assertIn("f1", per_class[cat])
                self.assertIn("support", per_class[cat])
                self.assertGreater(per_class[cat]["support"], 0)

            # Macro F1 must be unweighted arithmetic mean of per-class F1s
            f1_values = [per_class[cat]["f1"] for cat in SUPPORTED_CATEGORIES]
            expected_macro_f1 = round(sum(f1_values) / len(f1_values), 4)
            self.assertAlmostEqual(cat_metrics["macro_avg"]["f1"], expected_macro_f1, places=3)

    def test_confusion_matrices(self):
        """Verify confusion matrix row sums match class support."""
        for m_name, m_info in self.metrics["models"].items():
            cm_cat = m_info["category_metrics"]["confusion_matrix"]
            per_class = m_info["category_metrics"]["per_class"]

            for true_c in SUPPORTED_CATEGORIES:
                row_sum = sum(cm_cat[true_c].values())
                expected_support = per_class[true_c]["support"]
                self.assertEqual(row_sum, expected_support, f"CM row sum mismatch for {m_name} class {true_c}")

            # Priority CM check
            cm_pri = m_info["priority_metrics"]["confusion_matrix"]
            for true_p in SUPPORTED_PRIORITIES:
                p_sum = sum(cm_pri[true_p].values())
                p_support = m_info["priority_metrics"]["per_class"][true_p]["support"]
                self.assertEqual(p_sum, p_support)

    def test_latency_aggregation_and_variance(self):
        """Verify latency statistics (mean, median, p95, min, max, std dev) adhere to mathematical bounds."""
        for m_name, m_info in self.metrics["models"].items():
            lat = m_info["aggregate_latency"]
            self.assertEqual(lat["total_samples"], 150)
            self.assertGreater(lat["mean_ms"], 0)
            self.assertGreater(lat["median_ms"], 0)
            self.assertGreater(lat["p95_ms"], 0)
            self.assertGreater(lat["min_ms"], 0)
            self.assertGreater(lat["max_ms"], lat["min_ms"])
            self.assertGreaterEqual(lat["max_ms"], lat["p95_ms"])
            self.assertGreaterEqual(lat["p95_ms"], lat["median_ms"])
            self.assertGreaterEqual(lat["median_ms"], lat["min_ms"])
            self.assertGreater(lat["variance_ms"], 0)

    def test_repetition_consistency(self):
        """Verify repetition consistency tracking and deterministic temperature 0.0 behavior."""
        for m_name, m_info in self.metrics["models"].items():
            cons = m_info["consistency"]
            self.assertTrue(cons["identical_predictions_across_repetitions"])
            self.assertEqual(cons["inconsistent_case_count"], 0)
            self.assertEqual(cons["accuracy_std_across_reps"], 0.0)

    def test_reliability_denominators(self):
        """Verify reliability denominators: exactly 150 inferences per model, 100% success rate."""
        for m_name, m_info in self.metrics["models"].items():
            rel = m_info["reliability"]
            self.assertEqual(rel["total_inferences"], 150)
            self.assertEqual(rel["successful_inferences"], 150)
            self.assertEqual(rel["failed_inferences"], 0)
            self.assertEqual(rel["invalid_json_responses"], 0)
            self.assertEqual(rel["success_rate_pct"], 100.0)
            self.assertEqual(rel["invalid_json_rate_pct"], 0.0)

    def test_csv_and_json_artifacts_validity(self):
        """Verify all generated CSV and JSON deliverable files exist, are readable, and non-empty."""
        expected_files = [
            "comparison_summary.json",
            "model_comparison.csv",
            "per_class_comparison.csv",
            "latency_comparison.csv",
            "repetition_consistency.json",
            "reliability_comparison.json",
            "confusion_matrices.json",
            "comparative_report.html",
            "README.md",
        ]

        for fname in expected_files:
            fpath = self.output_dir / fname
            self.assertTrue(fpath.exists(), f"Deliverable {fname} does not exist")
            self.assertGreater(fpath.stat().st_size, 0, f"Deliverable {fname} is empty")

        # Verify model_comparison.csv row count (header + 3 models)
        with open(self.output_dir / "model_comparison.csv", "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 4)

        # Verify per_class_comparison.csv row count (header + 3 models * 11 classes = 34)
        with open(self.output_dir / "per_class_comparison.csv", "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 34)

    def test_html_report_validity(self):
        """Verify comparative HTML report is self-contained with no external CDN links."""
        html_path = self.output_dir / "comparative_report.html"
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("PromptOps Multi-Model Comparative Benchmark Report", content)
        self.assertIn("<svg", content)
        self.assertIn("gemma3:4b", content)
        self.assertIn("mistral:latest", content)
        self.assertIn("llama3:latest", content)

        # Verify no external CDN scripts or stylesheet links
        self.assertNotIn("http://", content)
        self.assertNotIn("https://cdn", content)
        self.assertNotIn("https://cdnjs", content)

    def test_missing_run_error_handling(self):
        """Verify engine raises ValueError when a required run is missing from the database."""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_db_path = Path(temp_dir) / "empty_valid_test.db"
        from app.storage.database import Database
        with Database(temp_db_path) as db:
            pass  # creates tables and migrations cleanly

        try:
            bad_engine = BenchmarkComparisonEngine(
                db_path=temp_db_path,
                manifest_path=self.manifest_path,
                results_path=self.results_path,
            )
            with self.assertRaises(ValueError) as ctx:
                bad_engine.load_and_audit()
            self.assertIn("missing from database", str(ctx.exception))
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_v1_database_untouched(self):
        """Verify production database data/evaluation.db remains strictly unmodified."""
        eval_db = Path("data/evaluation.db")
        conn = sqlite3.connect(eval_db)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM evaluation_runs")
        run_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM case_results")
        case_count = cur.fetchone()[0]
        conn.close()

        self.assertEqual(run_count, 5, "data/evaluation.db runs must remain exactly 5")
        self.assertEqual(case_count, 25, "data/evaluation.db cases must remain exactly 25")


if __name__ == "__main__":
    unittest.main()
