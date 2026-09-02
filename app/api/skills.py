"""
Skills and Role Intelligence API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import pandas as pd
from app.utils.config import ROLE_SKILL_MATRIX_PATH, ORGANIZATION_SKILL_GAPS_PATH
from app.services.skill_gap_service import skill_gap_service
from app.utils.logger import logger

router = APIRouter(tags=["Skills & Role Intelligence"])

@router.get(
    "/skills/role-matrix",
    summary="Get Role-Skill Archetype Matrix",
    description="Returns required core and technical competencies defined for each organizational role."
)
async def get_role_skill_matrix(role: str = Query(None, description="Optional job role filter")):
    try:
        if not ROLE_SKILL_MATRIX_PATH.exists():
            return []
        df = pd.read_csv(ROLE_SKILL_MATRIX_PATH)
        if role:
            df = df[df['job_role'].str.lower() == role.lower()]
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching role skill matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/skills/gap-analysis",
    summary="Calculate On-Demand Skill Gap",
    description="Calculates matched skills, missing skills, and readiness percentage for a given role and list of current skills."
)
async def calculate_custom_skill_gap(role: str, current_skills: List[str]):
    try:
        return skill_gap_service.calculate_skill_gap(role, current_skills)
    except Exception as e:
        logger.error(f"Error calculating skill gap: {e}")
        raise HTTPException(status_code=500, detail=str(e))
