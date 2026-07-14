import asyncio

from app.core.config import load_prompt_config
from app.llm.client import OllamaClient


async def main():
    config = load_prompt_config(r"prompt\v1.yaml")

    client = OllamaClient()

    response = await client.generate(
        prompt_config=config,
        user_input="""
Title:
Cannot login

Description:
After resetting my password I cannot login anymore.
""",
    )

    print("=" * 50)
    print("LLM Response")
    print("=" * 50)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())