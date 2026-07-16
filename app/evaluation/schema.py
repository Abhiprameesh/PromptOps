from typing import List, Literal

from pydantic import BaseModel

from app.core.schemas import GitHubIssue, IssueClassification


class DatasetMetadata(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    tags: List[str]
    notes: str


class GoldenDatasetCase(BaseModel):
    id: str
    input: GitHubIssue
    expected: IssueClassification
    metadata: DatasetMetadata


class GoldenDataset(BaseModel):
    version: str
    description: str
    created_at: str
    cases: List[GoldenDatasetCase]