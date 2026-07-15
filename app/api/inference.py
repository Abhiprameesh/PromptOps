from fastapi import APIRouter

from app.core.schemas import GitHubIssue, IssueClassification
from app.services.inference import InferenceService

router = APIRouter(prefix="/infer", tags=["Inference"])

service = InferenceService()


@router.post("/", response_model=IssueClassification)
async def infer(issue: GitHubIssue):
    return await service.infer(issue)