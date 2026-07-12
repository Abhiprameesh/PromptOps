from pydantic import BaseModel


class GitHubIssue(BaseModel):
    title: str
    body: str


class IssueClassification(BaseModel):
    category: str
    priority: str
    summary: str