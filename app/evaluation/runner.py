import asyncio
import time
from typing import Any, Dict, List, Optional

from app.core.config import PromptConfig
from app.evaluation.results import CaseResult, EvaluationResult
from app.evaluation.schema import GoldenDataset
from app.llm.formatter import build_user_prompt
from app.llm.parser import parse_llm_response
from app.services.inference import InferenceService


def _normalize_text(val: Optional[str]) -> str:
    """Normalize label using whitespace trimming and lowercasing."""
    if val is None:
        return ""
    return str(val).strip().lower()


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[Optional[str]],
) -> Dict[str, Any]:
    """
    Computes precision, recall, F1, macro/weighted averages, and a confusion matrix
    for classification tasks without silently excluding missing or error predictions.
    """
    norm_true = [_normalize_text(t) for t in y_true]
    norm_pred = [
        _normalize_text(p) if p is not None else "<error>"
        for p in y_pred
    ]

    target_classes = sorted(list(set(norm_true)))
    all_observed = sorted(list(set(norm_true).union(set(norm_pred))))
    total_cases = len(norm_true)

    if total_cases == 0:
        return {
            "accuracy": 0.0,
            "per_class": {},
            "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "weighted_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "confusion_matrix": {},
        }

    per_class = {}
    for c in target_classes:
        tp = sum(1 for yt, yp in zip(norm_true, norm_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(norm_true, norm_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(norm_true, norm_pred) if yt == c and yp != c)
        support = sum(1 for yt in norm_true if yt == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro averages (unweighted average across target classes)
    num_classes = len(target_classes)
    if num_classes > 0:
        macro_precision = round(
            sum(per_class[c]["precision"] for c in target_classes) / num_classes, 4
        )
        macro_recall = round(
            sum(per_class[c]["recall"] for c in target_classes) / num_classes, 4
        )
        macro_f1 = round(
            sum(per_class[c]["f1"] for c in target_classes) / num_classes, 4
        )
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    # Weighted averages (weighted by class support)
    total_support = sum(per_class[c]["support"] for c in target_classes)
    if total_support > 0:
        weighted_precision = round(
            sum(per_class[c]["precision"] * per_class[c]["support"] for c in target_classes)
            / total_support,
            4,
        )
        weighted_recall = round(
            sum(per_class[c]["recall"] * per_class[c]["support"] for c in target_classes)
            / total_support,
            4,
        )
        weighted_f1 = round(
            sum(per_class[c]["f1"] * per_class[c]["support"] for c in target_classes)
            / total_support,
            4,
        )
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    # Confusion matrix: rows = true classes, columns = all observed predictions
    confusion_matrix = {}
    for true_cls in target_classes:
        confusion_matrix[true_cls] = {}
        for pred_cls in all_observed:
            count = sum(
                1
                for yt, yp in zip(norm_true, norm_pred)
                if yt == true_cls and yp == pred_cls
            )
            confusion_matrix[true_cls][pred_cls] = count

    correct_total = sum(1 for yt, yp in zip(norm_true, norm_pred) if yt == yp)
    accuracy = round((correct_total / total_cases) * 100.0, 2)

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
        "confusion_matrix": confusion_matrix,
    }


class EvaluationRunner:

    def __init__(self, service: Optional[InferenceService] = None):
        self.service = service if service is not None else InferenceService()

    async def evaluate(
        self,
        dataset: GoldenDataset,
        prompt_config: PromptConfig,
    ) -> EvaluationResult:

        # Concurrency limit to prevent overloading Ollama
        semaphore = asyncio.Semaphore(5)

        async def evaluate_case(case) -> CaseResult:
            start_time = time.perf_counter()
            raw_response = None
            prediction = None
            error = None
            latency_ms = 0.0

            async with semaphore:
                try:
                    if hasattr(self.service, "client") and hasattr(
                        self.service.client, "generate"
                    ):
                        user_prompt = build_user_prompt(case.input, prompt_config)
                        raw_response = await self.service.client.generate(
                            prompt_config, user_prompt
                        )
                        latency_ms = round(
                            (time.perf_counter() - start_time) * 1000.0, 2
                        )
                        prediction = parse_llm_response(raw_response)
                    else:
                        prediction = await self.service.infer(
                            case.input, prompt_config
                        )
                        latency_ms = round(
                            (time.perf_counter() - start_time) * 1000.0, 2
                        )
                except Exception as e:
                    latency_ms = round(
                        (time.perf_counter() - start_time) * 1000.0, 2
                    )
                    error = str(e)

            expected_category = case.expected.category
            expected_priority = getattr(case.expected, "priority", None)

            if prediction is not None:
                predicted_category = prediction.category
                predicted_priority = getattr(prediction, "priority", None)
                predicted_summary = getattr(prediction, "summary", None)

                # 1. Category correctness (case-insensitive + trimmed)
                cat_pred_norm = _normalize_text(predicted_category)
                cat_exp_norm = _normalize_text(expected_category)
                category_passed = bool(
                    cat_pred_norm and cat_pred_norm == cat_exp_norm
                )

                # 2. Priority correctness (case-insensitive + trimmed)
                if expected_priority is not None:
                    pri_pred_norm = _normalize_text(predicted_priority)
                    pri_exp_norm = _normalize_text(expected_priority)
                    priority_passed = bool(
                        pri_pred_norm and pri_pred_norm == pri_exp_norm
                    )
                else:
                    priority_passed = True

                # 3. Summary validity (presence and non-empty content)
                summary_valid = bool(
                    predicted_summary and str(predicted_summary).strip()
                )

                # Composite pass requires category, priority, and valid summary
                composite_passed = (
                    category_passed and priority_passed and summary_valid
                )
            else:
                predicted_category = None
                predicted_priority = None
                category_passed = False
                priority_passed = False
                summary_valid = False
                composite_passed = False

            return CaseResult(
                case_id=case.id,
                expected_category=expected_category,
                predicted_category=predicted_category,
                passed=composite_passed,
                error=error,
                expected_priority=expected_priority,
                predicted_priority=predicted_priority,
                category_passed=category_passed,
                priority_passed=priority_passed,
                summary_valid=summary_valid,
                latency_ms=latency_ms,
                raw_response=raw_response,
            )

        # Run all test cases concurrently subject to the semaphore limit
        tasks = [evaluate_case(case) for case in dataset.cases]
        case_results = await asyncio.gather(*tasks)

        total = len(dataset.cases)
        passed = sum(1 for r in case_results if r.passed)
        failed = total - passed

        composite_accuracy = (
            round((passed / total) * 100.0, 2) if total > 0 else 0.0
        )
        category_passed_count = sum(1 for r in case_results if r.category_passed)
        category_accuracy = (
            round((category_passed_count / total) * 100.0, 2)
            if total > 0
            else 0.0
        )
        priority_passed_count = sum(1 for r in case_results if r.priority_passed)
        priority_accuracy = (
            round((priority_passed_count / total) * 100.0, 2)
            if total > 0
            else 0.0
        )

        valid_latencies = [
            r.latency_ms for r in case_results if r.latency_ms is not None
        ]
        avg_latency_ms = (
            round(sum(valid_latencies) / len(valid_latencies), 2)
            if valid_latencies
            else 0.0
        )

        # Difficulty breakdown
        difficulty_breakdown = {}
        for diff in ["easy", "medium", "hard"]:
            diff_cases = [
                r
                for c, r in zip(dataset.cases, case_results)
                if getattr(c, "metadata", None)
                and getattr(c.metadata, "difficulty", None) == diff
            ]
            if diff_cases:
                passed_diff = sum(1 for r in diff_cases if r.passed)
                difficulty_breakdown[diff] = round(
                    (passed_diff / len(diff_cases)) * 100.0, 2
                )

        # Tag breakdown
        tag_breakdown = {}
        all_tags = sorted(
            list(
                {
                    t
                    for c in dataset.cases
                    if getattr(c, "metadata", None)
                    and hasattr(c.metadata, "tags")
                    for t in (c.metadata.tags or [])
                }
            )
        )
        for tag in all_tags:
            tagged_cases = [
                r
                for c, r in zip(dataset.cases, case_results)
                if getattr(c, "metadata", None)
                and tag in (c.metadata.tags or [])
            ]
            if tagged_cases:
                passed_tag = sum(1 for r in tagged_cases if r.passed)
                tag_breakdown[tag] = round(
                    (passed_tag / len(tagged_cases)) * 100.0, 2
                )

        # Classification metrics
        category_metrics = compute_classification_metrics(
            [c.expected.category for c in dataset.cases],
            [r.predicted_category for r in case_results],
        )
        priority_metrics = compute_classification_metrics(
            [c.expected.priority for c in dataset.cases],
            [r.predicted_priority for r in case_results],
        )

        return EvaluationResult(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            accuracy=composite_accuracy,
            case_results=case_results,
            category_accuracy=category_accuracy,
            priority_accuracy=priority_accuracy,
            composite_accuracy=composite_accuracy,
            avg_latency_ms=avg_latency_ms,
            difficulty_breakdown=difficulty_breakdown,
            tag_breakdown=tag_breakdown,
            category_metrics=category_metrics,
            priority_metrics=priority_metrics,
        )
