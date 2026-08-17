import asyncio
from app.core.config import PromptConfig
from app.evaluation.results import (
    CaseResult,
    EvaluationResult,
)
from app.evaluation.schema import GoldenDataset
from app.services.inference import InferenceService


class EvaluationRunner:

    def __init__(self):
        self.service = InferenceService()

    async def evaluate(
        self,
        dataset: GoldenDataset,
        prompt_config: PromptConfig,
    ) -> EvaluationResult:

        # Concurrency limit to prevent overloading local Ollama instance
        semaphore = asyncio.Semaphore(5)

        async def evaluate_case(case) -> CaseResult:
            async with semaphore:
                try:
                    prediction = await self.service.infer(
                        case.input,
                        prompt_config,
                    )

                    success = (
                        prediction.category
                        == case.expected.category
                    )

                    return CaseResult(
                        case_id=case.id,
                        expected_category=case.expected.category,
                        predicted_category=prediction.category,
                        passed=success,
                    )

                except Exception as e:
                    return CaseResult(
                        case_id=case.id,
                        expected_category=case.expected.category,
                        predicted_category=None,
                        passed=False,
                        error=str(e),
                    )

        # Run all test cases concurrently subject to the semaphore limit
        tasks = [evaluate_case(case) for case in dataset.cases]
        case_results = await asyncio.gather(*tasks)

        passed = sum(1 for r in case_results if r.passed)
        total = len(dataset.cases)

        return EvaluationResult(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            accuracy=(passed / total) * 100 if total > 0 else 0,
            case_results=case_results,
        )