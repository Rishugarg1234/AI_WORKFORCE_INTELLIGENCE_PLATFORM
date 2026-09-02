"""
Unit tests for Pydantic input validation schemas.
"""

import pytest
from pydantic import ValidationError
from app.validation.employee_schema import EmployeeAttritionInput
from app.validation.engagement_schema import EngagementInput

def test_valid_employee_attrition_input():
    """Tests valid employee input passes schema validation."""
    valid_data = {
        "age": 30,
        "gender": "Male",
        "department": "IT",
        "job_role": "Developer",
        "education_level": 3,
        "monthly_salary": 80000,
        "overtime_hours_per_month": 10,
        "leaves_taken": 5,
        "projects_handled": 8,
        "training_hours": 20,
        "customer_satisfaction": 8.0,
        "last_promotion_year": 2021,
        "years_at_company": 5,
        "work_life_balance_score": 3.5,
        "performance_rating": 4,
        "country": "USA",
        "leave_day_name": "Monday"
    }
    emp = EmployeeAttritionInput(**valid_data)
    assert emp.age == 30
    assert emp.monthly_salary == 80000

def test_invalid_age_under_18():
    """Tests age under 18 raises ValidationError."""
    data = {
        "age": 16,  # Invalid
        "gender": "Female",
        "department": "HR",
        "job_role": "HR Manager",
        "monthly_salary": 60000
    }
    with pytest.raises(ValidationError):
        EmployeeAttritionInput(**data)

def test_invalid_age_over_100():
    """Tests age over 100 raises ValidationError."""
    data = {
        "age": 105,  # Invalid
        "gender": "Female",
        "department": "HR",
        "job_role": "HR Manager",
        "monthly_salary": 60000
    }
    with pytest.raises(ValidationError):
        EmployeeAttritionInput(**data)

def test_invalid_negative_salary():
    """Tests negative salary raises ValidationError."""
    data = {
        "age": 35,
        "gender": "Male",
        "department": "Finance",
        "job_role": "Auditor",
        "monthly_salary": -5000  # Invalid
    }
    with pytest.raises(ValidationError):
        EmployeeAttritionInput(**data)

def test_missing_required_department():
    """Tests missing department raises ValidationError."""
    data = {
        "age": 40,
        "gender": "Male",
        "job_role": "Developer",
        "monthly_salary": 70000
    }
    with pytest.raises(ValidationError):
        EmployeeAttritionInput(**data)

def test_valid_engagement_input():
    """Tests valid engagement payload."""
    data = {
        "employee_id": 1002,
        "performance_score": 85.0,
        "kpi_score": 90.0,
        "attendance_pct": 95.0,
        "peer_rating": 4.5,
        "task_completion_pct": 92.0,
        "work_hours_logged": 45.0,
        "manager_feedback": 4.2,
        "training_hours": 15.0
    }
    eng = EngagementInput(**data)
    assert eng.employee_id == 1002
    assert eng.attendance_pct == 95.0

def test_invalid_engagement_score_out_of_bounds():
    """Tests out of bound engagement scores."""
    data = {
        "employee_id": 1002,
        "performance_score": 150.0,  # Invalid (>100)
        "kpi_score": 90.0,
        "attendance_pct": 95.0,
        "peer_rating": 6.5,          # Invalid (>5)
        "task_completion_pct": 92.0,
        "work_hours_logged": 45.0,
        "manager_feedback": 4.2
    }
    with pytest.raises(ValidationError):
        EngagementInput(**data)
