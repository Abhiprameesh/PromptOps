import ollama

from app.core.config import PromptConfig
from app.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self):
        self.client = ollama.AsyncClient()

    async def generate(
        self,
        prompt_config: PromptConfig,
        user_input: str,
    ) -> str:

        response = await self.client.chat(
            model=prompt_config.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_config.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            options={
                "temperature": prompt_config.temperature,
                "num_predict": prompt_config.max_tokens,
            },
        )

        return response["message"]["content"]