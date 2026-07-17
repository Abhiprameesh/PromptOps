import asyncio

from app.core.config import load_prompt_config
from app.evaluation.loader import load_dataset
from app.evaluation.runner import EvaluationRunner


async def main():

    config = load_prompt_config(r"prompt\v1.yaml")

    dataset = load_dataset(
        r"datasets\v1\github_issues.json"
    )

    runner = EvaluationRunner()

    result = await runner.evaluate(
        dataset,
        config,
    )

    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    print(f"Total Cases : {result.total_cases}")
    print(f"Passed      : {result.passed_cases}")
    print(f"Failed      : {result.failed_cases}")
    print(f"Accuracy    : {result.accuracy:.2f}%")

    print("\nPer Case Results")
    print("-" * 80)

    for case in result.case_results:

        status = "PASS" if case.passed else "FAIL"

        prediction = (
            case.predicted_category
            if case.predicted_category
            else "N/A"
        )

        print(
            f"{case.case_id:<8}"
            f"{status:<8}"
            f"{prediction:<20}"
            f"{case.expected_category}"
        )

        if case.error:
            print(f"        Error: {case.error}")


if __name__ == "__main__":
    asyncio.run(main())