"""
Workforce Intelligence and Executive Dashboard API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import Dict, Any, List, Optional
from app.services.dashboard_service import dashboard_service
from app.utils.logger import logger

router = APIRouter(tags=["Workforce Dashboard & Employees"])

@router.get(
    "/dashboard/summary",
    summary="Get Executive Workforce KPIs",
    description="Returns aggregate headcount, high-risk employee count, average engagement score, and average skill readiness."
)
async def get_dashboard_summary(department: Optional[str] = Query(None, description="Filter by department")):
    try:
        return dashboard_service.get_summary_metrics(department=department)
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/dashboard/attrition-by-department",
    summary="Get Attrition Distribution by Department",
    description="Returns departmental breakdown of employee count, risk tiers, and average attrition probability."
)
async def get_attrition_by_department():
    try:
        return dashboard_service.get_attrition_by_department()
    except Exception as e:
        logger.error(f"Error fetching department attrition stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/dashboard/skill-gaps",
    summary="Get Organization-Wide Critical Skill Gaps",
    description="Returns ranked list of missing skills across the enterprise along with severity classifications."
)
async def get_organization_skill_gaps(limit: int = Query(15, ge=1, le=100)):
    try:
        return dashboard_service.get_organization_skill_gaps(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching organization skill gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/dashboard/recommendations",
    summary="Get Upskilling Recommendations",
    description="Returns sample of prioritized learning paths, courses, and certifications mapped to missing competencies."
)
async def get_upskilling_recommendations(limit: int = Query(20, ge=1, le=100)):
    try:
        return dashboard_service.get_top_recommendations(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/employees/{employee_id}",
    summary="Get 360-Degree Employee Intelligence Record",
    description="Returns granular profile for an individual employee including attrition risk, engagement, matched/missing skills, and recommendations."
)
async def get_employee_record(employee_id: int):
    try:
        record = dashboard_service.get_employee_detail(employee_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee ID {employee_id} not found in enterprise records."
            )
        return record
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
