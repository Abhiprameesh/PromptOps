import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.core.config import PromptConfig, PromptMetadata
from app.core.schemas import GitHubIssue, IssueClassification
from app.evaluation.results import CaseResult, EvaluationResult
from app.evaluation.runner import (
    EvaluationRunner,
    _normalize_text,
    compute_classification_metrics,
)
from app.evaluation.schema import (
    DatasetMetadata,
    GoldenDataset,
    GoldenDatasetCase,
)


def create_sample_case(
    case_id: str,
    category: str = "bug",
    priority: str = "high",
    difficulty: str = "easy",
    tags: list = None,
) -> GoldenDatasetCase:
    return GoldenDatasetCase(
        id=case_id,
        input=GitHubIssue(
            title=f"Sample issue {case_id}",
            body="Sample issue description body.",
        ),
        expected=IssueClassification(
            category=category,
            priority=priority,
            summary="Expected summary text.",
        ),
        metadata=DatasetMetadata(
            difficulty=difficulty,
            tags=tags or ["auth"],
            notes="Test case notes",
        ),
    )


def create_sample_config() -> PromptConfig:
    return PromptConfig(
        version="v1",
        name="Test Classifier",
        model="test-model",
        temperature=0.0,
        max_tokens=150,
        system_prompt="Classify issues.",
        user_prompt_template="Title: {title}\nBody: {body}",
        output_schema={"category": "string", "priority": "string", "summary": "string"},
        metadata=PromptMetadata(
            author="Test",
            description="Test prompt config",
            created_at="2026-11-23",
        ),
    )


class TestEvaluationRunnerV2(unittest.TestCase):

    def test_text_normalization(self):
        """Verify _normalize_text handles whitespace, mixed case, and None."""
        self.assertEqual(_normalize_text("  Bug  "), "bug")
        self.assertEqual(_normalize_text("FEATURE-REQUEST"), "feature-request")
        self.assertEqual(_normalize_text(None), "")
        self.assertEqual(_normalize_text(""), "")

    def test_multi_field_evaluation_all_correct(self):
        """Verify composite pass when category, priority, and summary are all valid."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        mock_service.client.generate = AsyncMock(
            return_value='{"category": "bug", "priority": "high", "summary": "Valid summary."}'
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        self.assertEqual(result.total_cases, 1)
        self.assertEqual(result.passed_cases, 1)
        self.assertEqual(result.accuracy, 100.0)

        case = result.case_results[0]
        self.assertTrue(case.category_passed)
        self.assertTrue(case.priority_passed)
        self.assertTrue(case.summary_valid)
        self.assertTrue(case.passed)
        self.assertGreater(case.latency_ms, 0)
        self.assertIsNotNone(case.raw_response)

    def test_multi_field_evaluation_incorrect_category(self):
        """Verify failure when category is wrong even if priority and summary are valid."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        mock_service.client.generate = AsyncMock(
            return_value='{"category": "ui", "priority": "high", "summary": "Valid summary."}'
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        case = result.case_results[0]
        self.assertFalse(case.category_passed)
        self.assertTrue(case.priority_passed)
        self.assertTrue(case.summary_valid)
        self.assertFalse(case.passed)
        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.category_accuracy, 0.0)
        self.assertEqual(result.priority_accuracy, 100.0)

    def test_multi_field_evaluation_incorrect_priority(self):
        """Verify failure when priority is wrong even if category and summary are valid."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        mock_service.client.generate = AsyncMock(
            return_value='{"category": "bug", "priority": "low", "summary": "Valid summary."}'
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        case = result.case_results[0]
        self.assertTrue(case.category_passed)
        self.assertFalse(case.priority_passed)
        self.assertTrue(case.summary_valid)
        self.assertFalse(case.passed)
        self.assertEqual(result.category_accuracy, 100.0)
        self.assertEqual(result.priority_accuracy, 0.0)

    def test_whitespace_and_case_normalization(self):
        """Verify category and priority matching handles casing and surrounding whitespace."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        mock_service.client.generate = AsyncMock(
            return_value='{"category": "  BUG  ", "priority": "  HIGH  ", "summary": "Valid summary."}'
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        case = result.case_results[0]
        self.assertTrue(case.category_passed)
        self.assertTrue(case.priority_passed)
        self.assertTrue(case.passed)

    def test_empty_or_whitespace_summary_fails(self):
        """Verify empty or blank summary fails composite pass."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        # Summary is only whitespace
        mock_service.client.generate = AsyncMock(
            return_value='{"category": "bug", "priority": "high", "summary": "   "}'
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        case = result.case_results[0]
        self.assertTrue(case.category_passed)
        self.assertTrue(case.priority_passed)
        self.assertFalse(case.summary_valid)
        self.assertFalse(case.passed)

    def test_invalid_json_preserves_raw_response_and_does_not_crash(self):
        """Verify invalid JSON is captured, raw response preserved, and pipeline does not fail."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        mock_service.client.generate = AsyncMock(
            return_value="I am an LLM and I forgot to output JSON."
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        self.assertEqual(result.total_cases, 1)
        self.assertEqual(result.passed_cases, 0)
        self.assertEqual(result.failed_cases, 1)

        case = result.case_results[0]
        self.assertFalse(case.passed)
        self.assertIsNotNone(case.error)
        self.assertEqual(case.raw_response, "I am an LLM and I forgot to output JSON.")
        self.assertIsNone(case.predicted_category)

    def test_inference_exception_handling(self):
        """Verify inference exceptions are safely caught without crashing the run."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()
        mock_service.client.generate = AsyncMock(
            side_effect=RuntimeError("Connection timeout to Ollama")
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[create_sample_case("GH001", category="bug", priority="high")],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        self.assertEqual(result.total_cases, 1)
        case = result.case_results[0]
        self.assertFalse(case.passed)
        self.assertIn("Connection timeout", case.error)
        self.assertGreater(case.latency_ms, 0)

    def test_classification_metrics_computation(self):
        """Verify per-class precision, recall, F1, macro/weighted averages, and confusion matrix."""
        y_true = ["bug", "bug", "feature-request", "ui"]
        # 1 correct bug, 1 bug predicted as ui, 1 correct feature-request, 1 ui had an error
        y_pred = ["bug", "ui", "feature-request", None]

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertIn("per_class", metrics)
        self.assertIn("bug", metrics["per_class"])
        self.assertIn("feature-request", metrics["per_class"])
        self.assertIn("ui", metrics["per_class"])

        # For feature-request: TP=1, FP=0, FN=0 -> Precision=1.0, Recall=1.0, F1=1.0
        self.assertEqual(metrics["per_class"]["feature-request"]["precision"], 1.0)
        self.assertEqual(metrics["per_class"]["feature-request"]["recall"], 1.0)
        self.assertEqual(metrics["per_class"]["feature-request"]["f1"], 1.0)

        # For bug: TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5, F1=0.6667
        self.assertEqual(metrics["per_class"]["bug"]["precision"], 1.0)
        self.assertEqual(metrics["per_class"]["bug"]["recall"], 0.5)

        # For ui: TP=0, FP=1, FN=1 (error counted as FN) -> Precision=0.0, Recall=0.0
        self.assertEqual(metrics["per_class"]["ui"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["ui"]["recall"], 0.0)

        # Confusion matrix checks
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["bug"]["bug"], 1)
        self.assertEqual(cm["bug"]["ui"], 1)
        self.assertEqual(cm["ui"]["<error>"], 1)

        # Macro and weighted averages exist
        self.assertIn("macro_avg", metrics)
        self.assertIn("weighted_avg", metrics)
        self.assertGreater(metrics["macro_avg"]["f1"], 0.0)

    def test_difficulty_and_tag_breakdowns(self):
        """Verify difficulty and tag accuracy breakdowns across multiple cases."""
        mock_service = MagicMock()
        mock_service.client = MagicMock()

        # Responses for 3 cases: easy passes, medium fails, hard passes
        mock_service.client.generate = AsyncMock(
            side_effect=[
                '{"category": "bug", "priority": "high", "summary": "Case 1"}',
                '{"category": "wrong", "priority": "low", "summary": "Case 2"}',
                '{"category": "feature-request", "priority": "medium", "summary": "Case 3"}',
            ]
        )

        dataset = GoldenDataset(
            version="v1",
            description="Test dataset",
            created_at="2026-11-23",
            cases=[
                create_sample_case("GH1", category="bug", priority="high", difficulty="easy", tags=["auth"]),
                create_sample_case("GH2", category="bug", priority="high", difficulty="medium", tags=["auth", "ui"]),
                create_sample_case("GH3", category="feature-request", priority="medium", difficulty="hard", tags=["ui"]),
            ],
        )

        runner = EvaluationRunner(service=mock_service)
        result = asyncio.run(runner.evaluate(dataset, create_sample_config()))

        self.assertEqual(result.total_cases, 3)
        self.assertEqual(result.passed_cases, 2)

        # Difficulty breakdown: easy = 100%, medium = 0%, hard = 100%
        self.assertEqual(result.difficulty_breakdown["easy"], 100.0)
        self.assertEqual(result.difficulty_breakdown["medium"], 0.0)
        self.assertEqual(result.difficulty_breakdown["hard"], 100.0)

        # Tag breakdown: auth (2 cases, 1 passed -> 50%), ui (2 cases, 1 passed -> 50%)
        self.assertEqual(result.tag_breakdown["auth"], 50.0)
        self.assertEqual(result.tag_breakdown["ui"], 50.0)


if __name__ == "__main__":
    unittest.main()
