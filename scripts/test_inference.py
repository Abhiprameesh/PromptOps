import asyncio

from app.core.config import load_prompt_config
from app.core.schemas import GitHubIssue
from app.services.inference import InferenceService


async def main():

    config = load_prompt_config(r"prompt\v1.yaml")

    service = InferenceService()

    issue = GitHubIssue(
        title="Cannot login",
        body="After resetting my password I cannot login anymore."
    )

    result = await service.infer(
        issue,
        config,
    )

    print(result.model_dump_json(indent=4))


if __name__ == "__main__":
    asyncio.run(main())