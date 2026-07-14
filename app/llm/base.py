from abc import ABC, abstractmethod

from app.core.config import PromptConfig


class LLMClient(ABC):
    """
    Base interface for all LLM providers.
    """

    @abstractmethod
    async def generate(
        self,
        prompt_config: PromptConfig,
        user_input: str,
    ) -> str:
        """
        Send a prompt to an LLM and return the raw text response.
        """
        pass