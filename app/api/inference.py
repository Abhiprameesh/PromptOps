from fastapi import APIRouter

from app.core.config import load_prompt_config
from app.core.schemas import GitHubIssue
from app.services.inference import InferenceService

router = APIRouter(prefix="/infer", tags=["Inference"])

service = InferenceService()
config = load_prompt_config("prompts/v1.yaml")


@router.post("/")
async def infer(issue: GitHubIssue):
    return await service.infer(
        issue,
        config,
    )