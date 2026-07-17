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

        results = []
        passed = 0

        for case in dataset.cases:

            try:
                prediction = await self.service.infer(
                    case.input,
                    prompt_config,
                )


                success = (
                    prediction.category
                    == case.expected.category
                )

                if success:
                    passed += 1

                
                results.append(
                    CaseResult(
                        case_id=case.id,
                        expected_category=case.expected.category,
                        predicted_category=prediction.category,
                        passed=success,
                    )
                )

            except Exception as e:

               
                results.append(
                    CaseResult(
                        case_id=case.id,
                        expected_category=case.expected.category,
                        predicted_category=None,
                        passed=False,
                        error=str(e),
                    )
                )

        total = len(dataset.cases)

        return EvaluationResult(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            accuracy=(passed / total) * 100 if total > 0 else 0,
            case_results=results,
        )