"""
Engagement service computing multi-factor engagement scores and analytics.
"""

from typing import Dict, Any, List
import pandas as pd
from app.utils.config import ENGAGEMENT_PROCESSED_PATH
from app.validation.engagement_schema import EngagementInput, EngagementScoreResponse
from app.utils.logger import logger

class EngagementService:
    """Service handling employee engagement analytics."""
    
    @staticmethod
    def calculate_engagement(input_data: EngagementInput) -> EngagementScoreResponse:
        """Calculates multi-factor composite engagement score."""
        score = (
            0.25 * input_data.attendance_pct +
            0.25 * input_data.task_completion_pct +
            0.20 * input_data.kpi_score +
            0.15 * (input_data.peer_rating / 5.0 * 100.0) +
            0.15 * (input_data.manager_feedback / 5.0 * 100.0)
        )
        score = round(score, 2)
        
        if score >= 85.0:
            tier = "Highly Engaged"
        elif score >= 75.0:
            tier = "Moderately Engaged"
        elif score >= 65.0:
            tier = "Passively Engaged"
        else:
            tier = "Disengaged / At Risk"
            
        breakdown = {
            "attendance_contribution": round(0.25 * input_data.attendance_pct, 2),
            "task_completion_contribution": round(0.25 * input_data.task_completion_pct, 2),
            "kpi_contribution": round(0.20 * input_data.kpi_score, 2),
            "peer_rating_contribution": round(0.15 * (input_data.peer_rating / 5.0 * 100.0), 2),
            "manager_feedback_contribution": round(0.15 * (input_data.manager_feedback / 5.0 * 100.0), 2)
        }
        
        return EngagementScoreResponse(
            employee_id=input_data.employee_id,
            engagement_score=score,
            engagement_tier=tier,
            breakdown=breakdown
        )

    @staticmethod
    def get_department_engagement() -> List[Dict[str, Any]]:
        """Returns department-level engagement summary statistics."""
        if not ENGAGEMENT_PROCESSED_PATH.exists():
            return []
        df = pd.read_csv(ENGAGEMENT_PROCESSED_PATH)
        dept_summary = df.groupby('department').agg(
            employee_count=('employee_id', 'count'),
            avg_engagement=('engagement_score', 'mean'),
            avg_kpi=('kpi_score', 'mean'),
            avg_attendance=('attendance_pct', 'mean')
        ).round(2).reset_index()
        return dept_summary.to_dict(orient="records")

engagement_service = EngagementService()
