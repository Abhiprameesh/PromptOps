from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CaseResult(BaseModel):
    # Existing V1 fields
    case_id: str
    expected_category: str
    predicted_category: Optional[str] = None
    passed: bool
    error: Optional[str] = None

    # Evaluation Engine V2 optional fields (backward-compatible)
    expected_priority: Optional[str] = None
    predicted_priority: Optional[str] = None
    category_passed: Optional[bool] = None
    priority_passed: Optional[bool] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[str] = None


class EvaluationResult(BaseModel):
    # Existing V1 fields
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float
    case_results: List[CaseResult]

    # Evaluation Engine V2 optional fields (backward-compatible)
    category_accuracy: Optional[float] = None
    priority_accuracy: Optional[float] = None
    composite_accuracy: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    difficulty_breakdown: Optional[Dict[str, Any]] = None
