import unittest
from app.reporting.regression import RegressionDetector


class TestRegressionDetector(unittest.TestCase):
    def test_compare_improvement(self):
        # Tuple format: (run_id, timestamp, prompt_version, model, accuracy, passed, failed)
        prev = (1, "2026-07-18", "v1", "gemma3:4b", 50.0, 5, 5)
        curr = (2, "2026-07-19", "v1", "gemma3:4b", 70.0, 7, 3)

        report = RegressionDetector.compare(prev, curr)
        self.assertEqual(report["status"], "improved")
        self.assertEqual(report["difference"], 20.0)

    def test_compare_regression(self):
        prev = (1, "2026-07-18", "v1", "gemma3:4b", 80.0, 8, 2)
        curr = (2, "2026-07-19", "v1", "gemma3:4b", 75.0, 7, 3)

        report = RegressionDetector.compare(prev, curr)
        self.assertEqual(report["status"], "regression")
        self.assertEqual(report["difference"], -5.0)

    def test_compare_unchanged(self):
        prev = (1, "2026-07-18", "v1", "gemma3:4b", 60.0, 6, 4)
        curr = (2, "2026-07-19", "v1", "gemma3:4b", 60.0, 6, 4)

        report = RegressionDetector.compare(prev, curr)
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(report["difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
