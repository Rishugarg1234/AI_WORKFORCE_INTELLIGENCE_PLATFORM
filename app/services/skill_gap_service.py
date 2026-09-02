"""
Skill gap service calculating matched skills, missing skills, and readiness scores.
"""

from typing import Dict, Any, List, Set
import pandas as pd
from app.utils.config import ROLE_SKILL_MATRIX_PATH, EMPLOYEE_SKILLS_PATH, EMPLOYEE_SKILL_GAPS_PATH
from app.utils.logger import logger

class SkillGapService:
    """Service handling skill gap analysis and workforce capability metrics."""
    
    @staticmethod
    def calculate_skill_gap(role: str, employee_skills: List[str]) -> Dict[str, Any]:
        """Calculates set-based skill gap for given role and employee skill set."""
        if not ROLE_SKILL_MATRIX_PATH.exists():
            return {
                "role": role,
                "matched_skills": employee_skills,
                "missing_skills": [],
                "skill_gap_percentage": 0.0,
                "readiness_score": 100.0
            }
            
        df_role = pd.read_csv(ROLE_SKILL_MATRIX_PATH)
        role_skills_series = df_role[df_role['job_role'].str.lower() == role.lower()]['required_skill'].str.strip()
        
        if role_skills_series.empty:
            role_skills_list = ["Problem Solving", "Communication", "Project Management"]
        else:
            role_skills_list = role_skills_series.tolist()

        # Build case-insensitive lookup mapping: lower_skill -> canonical_name
        role_lookup = {s.strip().lower(): s.strip() for s in role_skills_list}
        emp_skills_lower = {s.strip().lower() for s in employee_skills}

        matched_lower = set(role_lookup.keys()).intersection(emp_skills_lower)
        missing_lower = set(role_lookup.keys()) - emp_skills_lower

        matched_canonical = [role_lookup[k] for k in matched_lower]
        missing_canonical = [role_lookup[k] for k in missing_lower]

        total_req = len(role_lookup)
        gap_pct = round((len(missing_canonical) / total_req) * 100.0, 2) if total_req > 0 else 0.0
        readiness = round((len(matched_canonical) / total_req) * 100.0, 2) if total_req > 0 else 100.0

        return {
            "role": role,
            "matched_skills": sorted(matched_canonical),
            "missing_skills": sorted(missing_canonical),
            "total_required_skills": total_req,
            "total_matched_skills": len(matched_canonical),
            "skill_gap_percentage": gap_pct,
            "readiness_score": readiness
        }

    @staticmethod
    def get_employee_skill_profile(employee_id: int) -> Dict[str, Any]:
        """Retrieves verified skill gap profile for a stored employee."""
        if not EMPLOYEE_SKILL_GAPS_PATH.exists():
            return {}
        df = pd.read_csv(EMPLOYEE_SKILL_GAPS_PATH)
        match = df[df['employee_id'] == employee_id]
        if match.empty:
            return {}
        return match.iloc[0].to_dict()

skill_gap_service = SkillGapService()
