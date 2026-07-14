from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel


class PromptMetadata(BaseModel):
    author: str
    description: str
    created_at: str


class PromptConfig(BaseModel):
    version: str
    name: str

    model: str
    temperature: float
    max_tokens: int

    system_prompt: str
    user_prompt_template: str

    output_schema: Dict[str, str]

    metadata: PromptMetadata

def load_prompt_config(path: str) -> PromptConfig:
    """
    Load and validate a prompt configuration from YAML.
    """

    with open(Path(path), "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return PromptConfig(**data)