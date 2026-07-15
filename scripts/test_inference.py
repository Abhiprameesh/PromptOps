import asyncio

from app.core.schemas import GitHubIssue
from app.services.inference import InferenceService


async def main():
    service = InferenceService()

    issue = GitHubIssue(
        title="Cannot login",
        body="After resetting my password I cannot login anymore."
    )

    result = await service.infer(issue)

    print("=" * 50)
    print("Inference Result")
    print("=" * 50)
    print(result.model_dump_json(indent=4))


if __name__ == "__main__":
    asyncio.run(main())