"""
Pydantic schemas for employee data and attrition prediction request/response.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

class EmployeeAttritionInput(BaseModel):
    """Schema for individual employee attrition prediction input."""
    employee_id: Optional[int] = Field(None, description="Optional employee identifier")
    age: int = Field(..., ge=18, le=100, description="Age between 18 and 100")
    gender: str = Field(..., description="Gender (e.g. Male, Female, Other)")
    department: str = Field(..., description="Department name (e.g. IT, Sales, HR, Finance, Marketing, Support)")
    job_role: str = Field(..., description="Job role title (e.g. Developer, Sales Executive, Auditor)")
    education_level: int = Field(2, ge=1, le=5, description="Education level tier (1 to 5)")
    monthly_salary: float = Field(..., gt=0, description="Monthly salary in USD")
    overtime_hours_per_month: int = Field(0, ge=0, le=160, description="Monthly overtime hours (0 to 160)")
    leaves_taken: int = Field(0, ge=0, le=365, description="Annual leaves taken (0 to 365)")
    projects_handled: int = Field(1, ge=0, le=50, description="Number of projects handled")
    training_hours: int = Field(0, ge=0, le=200, description="Annual training hours")
    customer_satisfaction: Optional[float] = Field(None, ge=1.0, le=10.0, description="Customer/Stakeholder satisfaction (1-10)")
    last_promotion_year: int = Field(2022, ge=1990, le=2026, description="Year of last promotion")
    years_at_company: int = Field(1, ge=0, le=50, description="Tenure at company in years")
    work_life_balance_score: float = Field(3.0, ge=1.0, le=5.5, description="Work life balance rating (1.0 to 5.5)")
    performance_rating: int = Field(3, ge=1, le=5, description="Performance rating score (1 to 5)")
    country: str = Field("USA", description="Country location")
    leave_day_name: str = Field("Friday", description="Most frequent leave day")

    @field_validator('department')
    @classmethod
    def validate_department(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Department cannot be empty")
        return v_clean

    @field_validator('job_role')
    @classmethod
    def validate_job_role(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Job role cannot be empty")
        return v_clean

    model_config = {
        "json_schema_extra": {
            "example": {
                "employee_id": 101,
                "age": 28,
                "gender": "Female",
                "department": "Sales",
                "job_role": "Sales Executive",
                "education_level": 2,
                "monthly_salary": 54000,
                "overtime_hours_per_month": 25,
                "leaves_taken": 2,
                "projects_handled": 12,
                "training_hours": 35,
                "customer_satisfaction": 7.5,
                "last_promotion_year": 2020,
                "years_at_company": 6,
                "work_life_balance_score": 2.1,
                "performance_rating": 2,
                "country": "USA",
                "leave_day_name": "Friday"
            }
        }
    }

class AttritionPredictionResponse(BaseModel):
    """Prediction output schema."""
    employee_id: Optional[int] = None
    attrition_probability: float
    attrition_risk_level: str
    risk_interpretation: str
    model_version: str
    timestamp: str

class EmployeeRecordResponse(BaseModel):
    """Full 360-degree Employee Intelligence View schema."""
    employee_id: int
    name: str
    department: str
    job_role: str
    age: Optional[int] = None
    monthly_salary: Optional[float] = None
    years_experience: Optional[int] = None
    attrition_probability: float
    attrition_risk_level: str
    engagement_score: float
    matched_skills: str
    missing_skills: str
    skill_gap_percentage: float
    readiness_score: float
    top_recommendations: str
    top_certifications: str
