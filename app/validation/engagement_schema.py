"""
Pydantic schemas for employee engagement evaluation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any

class EngagementInput(BaseModel):
    """Input payload for engagement scoring."""
    employee_id: int = Field(..., description="Employee identifier")
    performance_score: float = Field(..., ge=0.0, le=100.0, description="Performance score 0-100")
    kpi_score: float = Field(..., ge=0.0, le=100.0, description="KPI attainment score 0-100")
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Attendance rate percentage 0-100")
    peer_rating: float = Field(..., ge=1.0, le=5.0, description="Peer rating 1.0 to 5.0")
    task_completion_pct: float = Field(..., ge=0.0, le=100.0, description="Task completion percentage 0-100")
    work_hours_logged: float = Field(..., ge=0.0, le=100.0, description="Weekly hours logged")
    manager_feedback: float = Field(..., ge=1.0, le=5.0, description="Manager feedback score 1.0 to 5.0")
    training_hours: float = Field(0.0, ge=0.0, le=100.0, description="Training hours logged")

class EngagementScoreResponse(BaseModel):
    """Response schema for engagement score."""
    employee_id: int
    engagement_score: float
    engagement_tier: str
    breakdown: Dict[str, float]
