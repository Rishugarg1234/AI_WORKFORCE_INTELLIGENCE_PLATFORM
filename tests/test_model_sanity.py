"""
Sanity and Behavioral Consistency Tests for the Attrition Risk ML Model.
Verifies that the inference engine responds logically and robustly across distinct simulated HR profiles.
"""

import pytest
from app.ml.predictor import predict_attrition
from app.validation.employee_schema import EmployeeAttritionInput

def test_scenario_1_low_risk_employee():
    """Scenario 1: Highly satisfied, low overtime, strong work-life balance."""
    payload = EmployeeAttritionInput(
        age=34,
        gender="Female",
        department="IT",
        job_role="Software Engineer",
        education_level=3,
        monthly_salary=95000,
        overtime_hours_per_month=2,       # Minimal overtime
        leaves_taken=12,                  # Healthy leave utilization
        projects_handled=6,
        training_hours=30,
        customer_satisfaction=9.0,        # High satisfaction
        last_promotion_year=2024,         # Recent promotion
        years_at_company=3,
        work_life_balance_score=4.8,      # Excellent work-life balance
        performance_rating=4,
        country="USA",
        leave_day_name="Friday"
    )
    result = predict_attrition(payload)

    assert 0.0 <= result.attrition_probability <= 1.0
    assert result.attrition_risk_level in ["Low", "Medium", "High"]
    # Low flight risk profile should receive low probability
    assert result.attrition_probability < 0.35
    assert result.attrition_risk_level == "Low"

def test_scenario_2_medium_risk_employee():
    """Scenario 2: Moderate satisfaction, average overtime and tenure."""
    payload = EmployeeAttritionInput(
        age=38,
        gender="Male",
        department="Finance",
        job_role="Financial Analyst",
        education_level=3,
        monthly_salary=75000,
        overtime_hours_per_month=18,
        leaves_taken=7,
        projects_handled=9,
        training_hours=15,
        customer_satisfaction=6.0,
        last_promotion_year=2021,
        years_at_company=6,
        work_life_balance_score=3.0,
        performance_rating=3,
        country="USA",
        leave_day_name="Wednesday"
    )
    result = predict_attrition(payload)

    assert 0.0 <= result.attrition_probability <= 1.0
    assert result.attrition_risk_level in ["Low", "Medium", "High"]

def test_scenario_3_high_risk_employee():
    """Scenario 3: Severe overtime, low satisfaction, stagnation, poor work-life balance."""
    payload = EmployeeAttritionInput(
        age=27,
        gender="Male",
        department="Sales",
        job_role="Sales Executive",
        education_level=2,
        monthly_salary=42000,
        overtime_hours_per_month=48,      # Excessive overtime
        leaves_taken=1,                   # Severe leave suppression
        projects_handled=18,
        training_hours=40,
        customer_satisfaction=3.5,        # Low satisfaction
        last_promotion_year=2015,         # Prolonged promotion delay (9+ years)
        years_at_company=10,
        work_life_balance_score=1.2,      # Poor work-life balance
        performance_rating=1,
        country="Canada",
        leave_day_name="Friday"
    )
    result = predict_attrition(payload)

    assert 0.0 <= result.attrition_probability <= 1.0
    assert result.attrition_risk_level in ["Low", "Medium", "High"]
    # Severe burnout indicators should yield high probability
    assert result.attrition_probability >= 0.70
    assert result.attrition_risk_level == "High"

def test_monotonicity_risk_shift():
    """Verifies that increasing overtime and reducing work-life balance shifts probability upward."""
    base_good = {
        "age": 30,
        "gender": "Female",
        "department": "IT",
        "job_role": "Developer",
        "education_level": 2,
        "monthly_salary": 70000,
        "overtime_hours_per_month": 5,
        "leaves_taken": 10,
        "projects_handled": 5,
        "training_hours": 20,
        "customer_satisfaction": 8.0,
        "last_promotion_year": 2023,
        "years_at_company": 3,
        "work_life_balance_score": 4.5,
        "performance_rating": 4,
        "country": "USA",
        "leave_day_name": "Friday"
    }

    base_stressed = base_good.copy()
    base_stressed["overtime_hours_per_month"] = 45
    base_stressed["work_life_balance_score"] = 1.2
    base_stressed["last_promotion_year"] = 2014

    res_good = predict_attrition(base_good)
    res_stressed = predict_attrition(base_stressed)

    assert res_stressed.attrition_probability >= res_good.attrition_probability
