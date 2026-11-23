import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from app.dashboard.components import (
    prepare_case_churn_summary,
    prepare_overview_metrics,
    render_breakdown_section,
    render_case_churn_view,
    render_classification_analysis,
    render_header_metrics,
    render_latency_section,
    render_metadata,
    render_regression_section,
)
from app.dashboard.tables import (
    prepare_breakdown_tables,
    prepare_case_comparison_data,
    prepare_case_explorer_data,
    prepare_classification_table,
    prepare_confusion_matrix_df,
    render_case_comparison_table,
    render_case_explorer_table,
    render_classification_table,
    render_history_table,
    render_run_comparison_table,
)
from app.reporting.regression import RegressionDetector


def fake_columns(spec, *args, **kwargs):
    count = spec if isinstance(spec, int) else len(spec)
    return [MagicMock() for _ in range(count)]


class TestDashboardV2(unittest.TestCase):

    def setUp(self):
        # Sample complete V2 run tuple (16 elements)
        self.v2_run = (
            2,                              # 0: run_id
            "2026-08-18 12:00:00",          # 1: timestamp
            "v2",                           # 2: prompt_version
            "gemma3:4b",                    # 3: model
            10,                             # 4: total_cases
            8,                              # 5: passed_cases
            2,                              # 6: failed_cases
            80.0,                           # 7: accuracy
            90.0,                           # 8: category_accuracy
            85.0,                           # 9: priority_accuracy
            75.0,                           # 10: composite_accuracy
            145.5,                          # 11: avg_latency_ms
            {                               # 12: category_metrics
                "accuracy": 90.0,
                "per_class": {
                    "bug": {"precision": 0.95, "recall": 0.90, "f1": 0.9245, "support": 5},
                    "feature": {"precision": 0.85, "recall": 0.80, "f1": 0.8242, "support": 5},
                },
                "macro_avg": {"precision": 0.90, "recall": 0.85, "f1": 0.8744},
                "weighted_avg": {"precision": 0.90, "recall": 0.85, "f1": 0.8744},
                "confusion_matrix": {
                    "bug": {"bug": 4, "feature": 1},
                    "feature": {"bug": 0, "feature": 5},
                },
            },
            {                               # 13: priority_metrics
                "accuracy": 85.0,
                "per_class": {
                    "high": {"precision": 1.0, "recall": 0.8, "f1": 0.8889, "support": 5},
                    "low": {"precision": 0.8, "recall": 1.0, "f1": 0.8889, "support": 5},
                },
                "macro_avg": {"precision": 0.9, "recall": 0.9, "f1": 0.8889},
                "weighted_avg": {"precision": 0.9, "recall": 0.9, "f1": 0.8889},
                "confusion_matrix": {
                    "high": {"high": 4, "low": 1},
                    "low": {"high": 0, "low": 5},
                },
            },
            {                               # 14: difficulty_breakdown
                "easy": {"total": 4, "passed": 4, "accuracy": 100.0},
                "medium": {"total": 4, "passed": 3, "accuracy": 75.0},
                "hard": {"total": 2, "passed": 1, "accuracy": 50.0},
            },
            {                               # 15: tag_breakdown
                "api": {"total": 6, "passed": 5, "accuracy": 83.33},
                "ui": {"total": 4, "passed": 3, "accuracy": 75.0},
            },
        )

        # Sample complete V2 case results (12 elements)
        self.v2_cases = [
            ("C01", "bug", "bug", 1, None, "high", "high", 1, 1, 1, 120.4, '{"category":"bug"}'),
            ("C02", "bug", "feature", 0, "Mismatch", "high", "low", 0, 0, 1, 160.8, '{"category":"feature"}'),
        ]

        # Sample historical legacy V1 run tuple (8 elements)
        self.v1_run = (
            1,
            "2026-08-17 10:00:00",
            "v1",
            "gemma3:4b",
            5,
            2,
            3,
            40.0,
        )

        # Sample legacy V1 case results (5 elements)
        self.v1_cases = [
            ("C01", "bug", "bug", 1, None),
            ("C02", "bug", "ui", 0, "Failed prediction"),
        ]

    # -------------------------------------------------------------------------
    # Test 1: Dashboard rendering with a complete V2 evaluation run
    # -------------------------------------------------------------------------
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    @patch("streamlit.tabs")
    @patch("streamlit.dataframe")
    @patch("streamlit.plotly_chart")
    def test_v2_run_data_preparation_and_rendering(
        self, mock_plotly, mock_df, mock_tabs, mock_metric, mock_cols
    ):
        mock_cols.side_effect = fake_columns
        mock_tabs.return_value = [MagicMock(), MagicMock()]

        # 1. Overview metrics preparation
        metrics = prepare_overview_metrics(self.v2_run)
        self.assertEqual(metrics["accuracy"], "80.00%")
        self.assertEqual(metrics["category_accuracy"], "90.00%")
        self.assertEqual(metrics["priority_accuracy"], "85.00%")
        self.assertEqual(metrics["composite_accuracy"], "75.00%")
        self.assertEqual(metrics["avg_latency_ms"], "145.50 ms")
        self.assertEqual(metrics["category_macro_f1"], "0.8744")
        self.assertEqual(metrics["priority_macro_f1"], "0.8889")

        # 2. Classification table & confusion matrix
        cat_df = prepare_classification_table(self.v2_run[12])
        self.assertIsNotNone(cat_df)
        self.assertEqual(len(cat_df), 4)  # 2 classes + macro avg + weighted avg
        self.assertIn("Class", cat_df.columns)
        self.assertIn("F1 Score", cat_df.columns)

        cm_df = prepare_confusion_matrix_df(self.v2_run[12])
        self.assertIsNotNone(cm_df)
        self.assertEqual(cm_df.loc["bug", "bug"], 4)
        self.assertEqual(cm_df.loc["bug", "feature"], 1)

        # 3. Breakdowns
        diff_df, tag_df = prepare_breakdown_tables(self.v2_run[14], self.v2_run[15])
        self.assertIsNotNone(diff_df)
        self.assertEqual(len(diff_df), 3)
        self.assertIsNotNone(tag_df)
        self.assertEqual(len(tag_df), 2)

        # 4. Case explorer DataFrame
        cases_df = prepare_case_explorer_data(self.v2_cases)
        self.assertIn("Exp Cat", cases_df.columns)
        self.assertIn("Exp Pri", cases_df.columns)
        self.assertIn("Latency", cases_df.columns)

        # 5. Rendering without error
        render_header_metrics(self.v2_run)
        render_classification_analysis(self.v2_run[12], self.v2_run[13])
        render_breakdown_section(self.v2_run[14], self.v2_run[15])
        render_latency_section(self.v2_run, self.v2_cases)
        render_case_explorer_table(self.v2_cases)

    # -------------------------------------------------------------------------
    # Test 2: Dashboard rendering with a historical V1 run
    # -------------------------------------------------------------------------
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    @patch("streamlit.dataframe")
    @patch("streamlit.info")
    def test_v1_historical_run_compatibility(
        self, mock_info, mock_df, mock_metric, mock_cols
    ):
        mock_cols.side_effect = fake_columns

        # 1. Overview metrics preparation for V1 (8-element tuple)
        metrics = prepare_overview_metrics(self.v1_run)
        self.assertEqual(metrics["accuracy"], "40.00%")
        # Missing V2 metrics MUST be 'N/A' and never '0.00%' or 0
        self.assertEqual(metrics["category_accuracy"], "N/A")
        self.assertEqual(metrics["priority_accuracy"], "N/A")
        self.assertEqual(metrics["composite_accuracy"], "N/A")
        self.assertEqual(metrics["avg_latency_ms"], "N/A")
        self.assertEqual(metrics["category_macro_f1"], "N/A")

        # 2. Classification tables for V1 are None
        self.assertIsNone(prepare_classification_table(None))
        self.assertIsNone(prepare_confusion_matrix_df(None))

        # 3. Breakdown tables for V1 are (None, None)
        diff_df, tag_df = prepare_breakdown_tables(None, None)
        self.assertIsNone(diff_df)
        self.assertIsNone(tag_df)

        # 4. Case explorer produces legacy columns
        cases_df = prepare_case_explorer_data(self.v1_cases)
        self.assertIn("Expected", cases_df.columns)
        self.assertIn("Predicted", cases_df.columns)
        self.assertNotIn("Latency", cases_df.columns)

        # 5. Rendering does not crash
        render_header_metrics(self.v1_run)
        render_classification_analysis(None, None)
        render_breakdown_section(None, None)
        render_latency_section(self.v1_run, self.v1_cases)
        render_case_explorer_table(self.v1_cases)

    # -------------------------------------------------------------------------
    # Test 3: Missing optional metrics
    # -------------------------------------------------------------------------
    def test_missing_optional_metrics(self):
        """Verify partially populated V2 runs display available metrics and mark missing as N/A."""
        partial_v2_run = list(self.v2_run)
        partial_v2_run[8] = 88.0   # category_accuracy present
        partial_v2_run[9] = None   # priority_accuracy missing
        partial_v2_run[10] = None  # composite_accuracy missing
        partial_v2_run[11] = None  # latency missing

        metrics = prepare_overview_metrics(tuple(partial_v2_run))
        self.assertEqual(metrics["category_accuracy"], "88.00%")
        self.assertEqual(metrics["priority_accuracy"], "N/A")
        self.assertEqual(metrics["composite_accuracy"], "N/A")
        self.assertEqual(metrics["avg_latency_ms"], "N/A")

    # -------------------------------------------------------------------------
    # Test 4: Missing classification metrics
    # -------------------------------------------------------------------------
    @patch("streamlit.info")
    def test_missing_classification_metrics(self, mock_info):
        """Verify empty classification metrics are handled safely."""
        self.assertIsNone(prepare_classification_table({}))
        self.assertIsNone(prepare_classification_table({"per_class": {}}))
        self.assertIsNone(prepare_confusion_matrix_df({}))

        render_classification_table(None)
        mock_info.assert_called_with("Classification metrics are unavailable for this run.")

    # -------------------------------------------------------------------------
    # Test 5: Empty difficulty and tag breakdowns
    # -------------------------------------------------------------------------
    @patch("streamlit.columns")
    @patch("streamlit.info")
    def test_empty_difficulty_and_tag_breakdowns(self, mock_info, mock_cols):
        mock_cols.side_effect = fake_columns

        diff_df, tag_df = prepare_breakdown_tables({}, {})
        self.assertIsNone(diff_df)
        self.assertIsNone(tag_df)

        render_breakdown_section({}, {})
        self.assertTrue(mock_info.called)

    # -------------------------------------------------------------------------
    # Test 6: Regression comparison integration
    # -------------------------------------------------------------------------
    @patch("streamlit.error")
    @patch("streamlit.warning")
    @patch("streamlit.dataframe")
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    def test_regression_comparison_integration(
        self, mock_metric, mock_cols, mock_df, mock_warn, mock_err
    ):
        mock_cols.side_effect = fake_columns

        report = RegressionDetector.compare(
            previous_run=self.v1_run,
            current_run=self.v2_run,
            previous_cases=self.v1_cases,
            current_cases=self.v2_cases,
        )

        self.assertIn("severity", report)
        self.assertIn("metrics_comparison", report)
        self.assertIn("case_analysis", report)

        # Render regression section without error
        render_regression_section(report)
        self.assertTrue(mock_df.called)

    # -------------------------------------------------------------------------
    # Test 7: Missing baseline or current run
    # -------------------------------------------------------------------------
    @patch("streamlit.info")
    def test_missing_baseline_or_current_run(self, mock_info):
        render_run_comparison_table(self.v2_run, None, 2, 1)
        mock_info.assert_called_with("Select a run to compare with.")

        render_regression_section({})
        mock_info.assert_called_with("No regression comparison report available.")

    # -------------------------------------------------------------------------
    # Test 8: Empty case-level churn sections
    # -------------------------------------------------------------------------
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    @patch("streamlit.expander")
    def test_empty_case_level_churn_sections(self, mock_expander, mock_metric, mock_cols):
        mock_cols.side_effect = fake_columns

        empty_churn = {
            "newly_failed": [],
            "recovered": [],
            "unchanged_passing": [],
            "unchanged_failing": [],
            "added_cases": [],
            "removed_cases": [],
            "missing_pass_cases": [],
            "duplicate_cases": [],
        }

        summary = prepare_case_churn_summary(empty_churn)
        for k, v in summary.items():
            self.assertEqual(len(v), 0)

        render_case_churn_view(empty_churn)
        # When all churn categories are empty, no expander should be created
        self.assertFalse(mock_expander.called)

    # -------------------------------------------------------------------------
    # Test 9: Duplicate and missing case information
    # -------------------------------------------------------------------------
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    @patch("streamlit.expander")
    def test_duplicate_and_missing_case_information(
        self, mock_expander, mock_metric, mock_cols
    ):
        mock_cols.side_effect = fake_columns
        mock_exp = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_exp

        populated_churn = {
            "newly_failed": ["GH01"],
            "recovered": ["GH02"],
            "unchanged_passing": ["GH03"],
            "unchanged_failing": ["GH04"],
            "added_cases": ["GH05"],
            "removed_cases": ["GH06"],
            "missing_pass_cases": ["GH07"],
            "duplicate_cases": ["GH08"],
        }

        summary = prepare_case_churn_summary(populated_churn)
        self.assertEqual(summary["duplicate_cases"], ["GH08"])
        self.assertEqual(summary["missing_pass_cases"], ["GH07"])
        self.assertEqual(summary["added_cases"], ["GH05"])
        self.assertEqual(summary["removed_cases"], ["GH06"])

        # Check case comparison status for missing info
        cases_prev = [("GH07", "bug", "bug", None, None)]
        cases_curr = [("GH07", "bug", "bug", 1, None)]
        comp_rows = prepare_case_comparison_data(cases_curr, cases_prev, 2, 1)
        self.assertEqual(len(comp_rows), 1)
        self.assertEqual(comp_rows[0]["Change"], "⚠️ Missing Info")

        # Rendering displays expanders for non-empty lists
        render_case_churn_view(populated_churn)
        self.assertTrue(mock_expander.called)


if __name__ == "__main__":
    unittest.main()
