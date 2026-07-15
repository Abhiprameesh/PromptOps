from app.core.config import PromptConfig
from app.core.schemas import GitHubIssue, IssueClassification
from app.llm.client import OllamaClient
from app.llm.formatter import build_user_prompt
from app.llm.parser import parse_llm_response


class InferenceService:

    def __init__(self):
        self.client = OllamaClient()

    async def infer(
        self,
        issue: GitHubIssue,
        prompt_config: PromptConfig,
    ) -> IssueClassification:

        user_prompt = build_user_prompt(
            issue,
            prompt_config,
        )

        raw_response = await self.client.generate(
            prompt_config,
            user_prompt,
        )

        return parse_llm_response(raw_response)