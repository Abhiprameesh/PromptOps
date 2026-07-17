from typing import List, Optional

from pydantic import BaseModel


class CaseResult(BaseModel):
    case_id: str

    expected_category: str
    predicted_category: Optional[str] = None

    passed: bool

    error: Optional[str] = None


class EvaluationResult(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int

    accuracy: float

    case_results: List[CaseResult]