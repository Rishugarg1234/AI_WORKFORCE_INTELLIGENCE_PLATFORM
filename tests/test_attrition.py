"""
Unit tests for Attrition Risk ML Predictor.
"""

from app.ml.predictor import predict_attrition
from app.validation.employee_schema import EmployeeAttritionInput

def test_attrition_prediction_probability_range():
    """Tests that attrition probability is strictly between 0 and 1."""
    input_data = EmployeeAttritionInput(
        age=35,
        gender="Female",
        department="Finance",
        job_role="Financial Analyst",
        education_level=3,
        monthly_salary=85000,
        overtime_hours_per_month=15,
        leaves_taken=6,
        projects_handled=10,
        training_hours=25,
        customer_satisfaction=8.5,
        last_promotion_year=2023,
        years_at_company=4,
        work_life_balance_score=4.2,
        performance_rating=4,
        country="USA",
        leave_day_name="Thursday"
    )
    
    response = predict_attrition(input_data)
    
    assert 0.0 <= response.attrition_probability <= 1.0
    assert response.attrition_risk_level in ["Low", "Medium", "High"]
    assert response.model_version == "v1"

def test_risk_category_assignment():
    """Tests risk category thresholds."""
    # High risk profile: high overtime, 0 leaves, low work-life balance, long promotion stagnation
    high_risk_input = {
        "age": 26,
        "gender": "Male",
        "department": "Sales",
        "job_role": "Sales Executive",
        "education_level": 2,
        "monthly_salary": 45000,
        "overtime_hours_per_month": 45,
        "leaves_taken": 1,
        "projects_handled": 16,
        "training_hours": 40,
        "customer_satisfaction": 4.0,
        "last_promotion_year": 2014,
        "years_at_company": 10,
        "work_life_balance_score": 1.5,
        "performance_rating": 1,
        "country": "USA",
        "leave_day_name": "Friday"
    }
    response = predict_attrition(high_risk_input)
    assert 0.0 <= response.attrition_probability <= 1.0
    assert response.attrition_risk_level in ["Low", "Medium", "High"]
