import json

from app.core.schemas import IssueClassification


def parse_llm_response(raw_response: str) -> IssueClassification:
    """
    Parse and validate the raw response returned by the LLM.
    """

    cleaned = raw_response.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json")

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")

    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```")

    cleaned = cleaned.strip()

    data = json.loads(cleaned)

    return IssueClassification(**data)