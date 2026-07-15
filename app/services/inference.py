from app.core.config import load_prompt_config
from app.core.schemas import GitHubIssue, IssueClassification
from app.llm.client import OllamaClient
from app.llm.formatter import build_user_prompt
from app.llm.parser import parse_llm_response


class InferenceService:
    """
    Coordinates the complete inference pipeline.
    """

    def __init__(self):
        self.client = OllamaClient()
        self.config = load_prompt_config(r"prompt/v1.yaml")

    async def infer(
        self,
        issue: GitHubIssue,
    ) -> IssueClassification:

        user_prompt = build_user_prompt(
            issue,
            self.config,
        )

        raw_response = await self.client.generate(
            prompt_config=self.config,
            user_input=user_prompt,
        )

        return parse_llm_response(raw_response)