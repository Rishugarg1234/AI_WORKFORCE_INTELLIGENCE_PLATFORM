"""
Dashboard Service aggregating cross-functional KPIs, department analytics, and drilldowns.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH, ORGANIZATION_SKILL_GAPS_PATH, EMPLOYEE_RECOMMENDATIONS_PATH
from app.utils.logger import logger

class DashboardService:
    """Service providing aggregated workforce intelligence data for UI and APIs."""

    @staticmethod
    def get_summary_metrics(department: Optional[str] = None) -> Dict[str, Any]:
        """Calculates executive KPI cards."""
        if not EMPLOYEE_INTELLIGENCE_PATH.exists():
            return {
                "total_employees": 0,
                "high_risk_employees": 0,
                "medium_risk_employees": 0,
                "low_risk_employees": 0,
                "avg_engagement": 0.0,
                "avg_readiness": 0.0,
                "total_departments": 0
            }

        df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
        if department and department != "All":
            df = df[df['department'].str.lower() == department.lower()]

        total = len(df)
        if total == 0:
            return {
                "total_employees": 0,
                "high_risk_employees": 0,
                "medium_risk_employees": 0,
                "low_risk_employees": 0,
                "avg_engagement": 0.0,
                "avg_readiness": 0.0,
                "total_departments": 0
            }

        high_risk = int((df['attrition_risk_level'] == 'High').sum())
        med_risk = int((df['attrition_risk_level'] == 'Medium').sum())
        low_risk = int((df['attrition_risk_level'] == 'Low').sum())
        avg_eng = round(float(df['engagement_score'].mean()), 2)
        avg_read = round(float(df['readiness_score'].mean()), 2)
        depts = int(df['department'].nunique())

        return {
            "total_employees": total,
            "high_risk_employees": high_risk,
            "high_risk_pct": round(high_risk / total * 100.0, 2),
            "medium_risk_employees": med_risk,
            "low_risk_employees": low_risk,
            "avg_engagement": avg_eng,
            "avg_readiness": avg_read,
            "total_departments": depts
        }

    @staticmethod
    def get_attrition_by_department() -> List[Dict[str, Any]]:
        """Returns attrition metrics aggregated by department."""
        if not EMPLOYEE_INTELLIGENCE_PATH.exists():
            return []

        df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
        dept_group = df.groupby('department').agg(
            total_employees=('employee_id', 'count'),
            high_risk_count=('attrition_risk_level', lambda x: (x == 'High').sum()),
            medium_risk_count=('attrition_risk_level', lambda x: (x == 'Medium').sum()),
            low_risk_count=('attrition_risk_level', lambda x: (x == 'Low').sum()),
            avg_attrition_probability=('attrition_probability', 'mean'),
            avg_engagement_score=('engagement_score', 'mean'),
            avg_readiness_score=('readiness_score', 'mean')
        ).round(3).reset_index()

        dept_group['high_risk_rate_pct'] = (dept_group['high_risk_count'] / dept_group['total_employees'] * 100.0).round(2)
        return dept_group.to_dict(orient="records")

    @staticmethod
    def get_organization_skill_gaps(limit: int = 15) -> List[Dict[str, Any]]:
        """Returns ranked organization-wide missing skills."""
        if not ORGANIZATION_SKILL_GAPS_PATH.exists():
            return []
        df = pd.read_csv(ORGANIZATION_SKILL_GAPS_PATH)
        return df.head(limit).to_dict(orient="records")

    @staticmethod
    def get_top_recommendations(limit: int = 20) -> List[Dict[str, Any]]:
        """Returns sample of prioritized upskilling recommendations."""
        if not EMPLOYEE_RECOMMENDATIONS_PATH.exists():
            return []
        df = pd.read_csv(EMPLOYEE_RECOMMENDATIONS_PATH)
        return df.head(limit).to_dict(orient="records")

    @staticmethod
    def get_employee_detail(employee_id: int) -> Optional[Dict[str, Any]]:
        """Returns full 360-degree Employee Intelligence View."""
        if not EMPLOYEE_INTELLIGENCE_PATH.exists():
            return None
        df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
        match = df[df['employee_id'] == employee_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

dashboard_service = DashboardService()
