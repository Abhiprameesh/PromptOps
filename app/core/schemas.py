from typing import Literal

from pydantic import BaseModel


class GitHubIssue(BaseModel):
    title: str
    body: str


class IssueClassification(BaseModel):
    category: Literal[
        "authentication",
        "ui",
        "backend",
        "database",
        "performance",
        "documentation",
        "bug",
        "feature-request",
    ]

    priority: Literal[
        "low",
        "medium",
        "high",
    ]

    summary: str