import unittest
from app.evaluation.results import CaseResult, EvaluationResult


class TestResultsSchemaV2(unittest.TestCase):

    def test_case_result_legacy_v1_instantiation(self):
        """Verify CaseResult can be instantiated with only legacy V1 fields."""
        case = CaseResult(
            case_id="GH001",
            expected_category="bug",
            predicted_category="bug",
            passed=True,
        )
        self.assertEqual(case.case_id, "GH001")
        self.assertEqual(case.expected_category, "bug")
        self.assertEqual(case.predicted_category, "bug")
        self.assertTrue(case.passed)
        self.assertIsNone(case.error)

        # Ensure all V2 fields default to None for backward compatibility
        self.assertIsNone(case.expected_priority)
        self.assertIsNone(case.predicted_priority)
        self.assertIsNone(case.category_passed)
        self.assertIsNone(case.priority_passed)
        self.assertIsNone(case.latency_ms)
        self.assertIsNone(case.raw_response)

    def test_case_result_legacy_v1_error_instantiation(self):
        """Verify CaseResult handles error instantiation as done in runner.py."""
        case = CaseResult(
            case_id="GH002",
            expected_category="feature-request",
            predicted_category=None,
            passed=False,
            error="Connection timed out",
        )
        self.assertEqual(case.case_id, "GH002")
        self.assertFalse(case.passed)
        self.assertEqual(case.error, "Connection timed out")
        self.assertIsNone(case.predicted_priority)

    def test_case_result_v2_full_fields(self):
        """Verify CaseResult correctly stores all V2 fields when provided."""
        case = CaseResult(
            case_id="GH003",
            expected_category="bug",
            predicted_category="bug",
            passed=True,
            expected_priority="high",
            predicted_priority="high",
            category_passed=True,
            priority_passed=True,
            latency_ms=142.5,
            raw_response='{"category": "bug", "priority": "high", "summary": "Login failure"}',
        )
        self.assertEqual(case.expected_priority, "high")
        self.assertEqual(case.predicted_priority, "high")
        self.assertTrue(case.category_passed)
        self.assertTrue(case.priority_passed)
        self.assertEqual(case.latency_ms, 142.5)
        self.assertIn("Login failure", case.raw_response)

    def test_evaluation_result_legacy_v1_instantiation(self):
        """Verify EvaluationResult works seamlessly with only V1 arguments."""
        case = CaseResult(
            case_id="GH001",
            expected_category="bug",
            predicted_category="bug",
            passed=True,
        )
        result = EvaluationResult(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            accuracy=100.0,
            case_results=[case],
        )
        self.assertEqual(result.total_cases, 1)
        self.assertEqual(result.passed_cases, 1)
        self.assertEqual(result.failed_cases, 0)
        self.assertEqual(result.accuracy, 100.0)
        self.assertEqual(len(result.case_results), 1)

        # Ensure all V2 fields default to None for backward compatibility
        self.assertIsNone(result.category_accuracy)
        self.assertIsNone(result.priority_accuracy)
        self.assertIsNone(result.composite_accuracy)
        self.assertIsNone(result.avg_latency_ms)
        self.assertIsNone(result.difficulty_breakdown)

    def test_evaluation_result_v2_full_fields(self):
        """Verify EvaluationResult works with all V2 metrics populated."""
        case = CaseResult(
            case_id="GH001",
            expected_category="bug",
            predicted_category="bug",
            passed=True,
            expected_priority="high",
            predicted_priority="high",
            category_passed=True,
            priority_passed=True,
            latency_ms=120.0,
        )
        result = EvaluationResult(
            total_cases=10,
            passed_cases=8,
            failed_cases=2,
            accuracy=80.0,
            case_results=[case],
            category_accuracy=90.0,
            priority_accuracy=80.0,
            composite_accuracy=70.0,
            avg_latency_ms=135.4,
            difficulty_breakdown={"easy": 100.0, "medium": 75.0, "hard": 50.0},
        )
        self.assertEqual(result.category_accuracy, 90.0)
        self.assertEqual(result.priority_accuracy, 80.0)
        self.assertEqual(result.composite_accuracy, 70.0)
        self.assertEqual(result.avg_latency_ms, 135.4)
        self.assertEqual(result.difficulty_breakdown["easy"], 100.0)

    def test_serialization_and_deserialization(self):
        """Verify Pydantic model serialization and deserialization retains V1 & V2 fields."""
        case = CaseResult(
            case_id="GH001",
            expected_category="bug",
            predicted_category="bug",
            passed=True,
            latency_ms=50.0,
        )
        data = case.model_dump()
        restored = CaseResult.model_validate(data)
        self.assertEqual(restored.case_id, "GH001")
        self.assertEqual(restored.latency_ms, 50.0)
        self.assertIsNone(restored.priority_passed)


if __name__ == "__main__":
    unittest.main()
