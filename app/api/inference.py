from fastapi import APIRouter

from app.core.schemas import GitHubIssue, IssueClassification

router = APIRouter(prefix="/infer", tags=["Inference"])


@router.post("/", response_model=IssueClassification)
async def infer(issue: GitHubIssue):
    return IssueClassification(
        category="authentication",
        priority="high",
        summary="Users cannot log in after resetting passwords."
    )