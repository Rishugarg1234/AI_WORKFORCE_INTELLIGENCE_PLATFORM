"""
Comprehensive API integration test suite using FastAPI TestClient.
Tests status codes, response schemas, error handling, parameter validation, and edge cases.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1. Health Endpoint Tests
def test_health_check_success():
    """Tests GET /health returns 200 and operational health metadata."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_version" in data
    assert "model_algorithm" in data
    assert "timestamp" in data

# 2. Dashboard Summary Endpoint Tests
def test_dashboard_summary_unfiltered():
    """Tests GET /dashboard/summary returns valid executive KPIs."""
    res = client.get("/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_employees"] == 5500
    assert data["high_risk_employees"] >= 0
    assert 0.0 <= data["avg_engagement"] <= 100.0
    assert 0.0 <= data["avg_readiness"] <= 100.0
    assert data["total_departments"] >= 5

def test_dashboard_summary_with_department_filter():
    """Tests GET /dashboard/summary with valid department query."""
    res = client.get("/dashboard/summary?department=IT")
    assert res.status_code == 200
    data = res.json()
    assert data["total_employees"] > 0
    assert data["total_employees"] < 5500

# 3. Departmental Attrition Profile Tests
def test_attrition_by_department_endpoint():
    """Tests GET /dashboard/attrition-by-department returns structured array."""
    res = client.get("/dashboard/attrition-by-department")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    first = data[0]
    for key in ["department", "total_employees", "high_risk_count", "medium_risk_count", "low_risk_count"]:
        assert key in first

# 4. Organization Skill Gaps Endpoint Tests
def test_organization_skill_gaps_endpoint():
    """Tests GET /dashboard/skill-gaps with pagination limit."""
    res = client.get("/dashboard/skill-gaps?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) <= 10
    if len(data) > 0:
        assert "missing_skill" in data[0]
        assert "employees_missing" in data[0]
        assert "severity" in data[0]

# 5. Recommendations Endpoint Tests
def test_recommendations_endpoint():
    """Tests GET /dashboard/recommendations."""
    res = client.get("/dashboard/recommendations?limit=15")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) <= 15
    if len(data) > 0:
        assert "recommended_course" in data[0]
        assert "recommended_certification" in data[0]

# 6. Employee Detail Endpoint Tests
def test_employee_detail_existing():
    """Tests GET /employees/{id} for valid employee ID."""
    res = client.get("/employees/1")
    assert res.status_code == 200
    data = res.json()
    assert data["employee_id"] == 1
    assert "name" in data
    assert "department" in data
    assert "job_role" in data
    assert "attrition_risk_level" in data
    assert "engagement_score" in data
    assert "readiness_score" in data

def test_employee_detail_non_existent():
    """Tests GET /employees/{id} for non-existent ID returns 404."""
    res = client.get("/employees/99999999")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()

# 7. Role Matrix and Skill Gap Analysis Tests
def test_skills_role_matrix():
    """Tests GET /skills/role-matrix."""
    res = client.get("/skills/role-matrix?role=Developer")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert any("python" in item["required_skill"].lower() for item in data)

def test_skills_custom_gap_analysis():
    """Tests POST /skills/gap-analysis."""
    payload = ["Python", "SQL"]
    res = client.post("/skills/gap-analysis?role=Developer", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "matched_skills" in data
    assert "missing_skills" in data
    assert "readiness_score" in data
    assert "Python" in data["matched_skills"]

# 8. Attrition Prediction Endpoint Tests
def test_predict_attrition_valid_payload():
    """Tests POST /predict/attrition with valid payload."""
    payload = {
        "age": 32,
        "gender": "Female",
        "department": "IT",
        "job_role": "Developer",
        "education_level": 3,
        "monthly_salary": 80000,
        "overtime_hours_per_month": 15,
        "leaves_taken": 6,
        "projects_handled": 7,
        "training_hours": 25,
        "customer_satisfaction": 7.5,
        "last_promotion_year": 2022,
        "years_at_company": 4,
        "work_life_balance_score": 3.5,
        "performance_rating": 3,
        "country": "USA",
        "leave_day_name": "Friday"
    }
    res = client.post("/predict/attrition", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["attrition_probability"] <= 1.0
    assert data["attrition_risk_level"] in ["Low", "Medium", "High"]
    assert "risk_interpretation" in data
    assert data["model_version"] == "v1"

def test_predict_attrition_invalid_payload():
    """Tests POST /predict/attrition with invalid/missing required fields returns 422."""
    invalid_payload = {
        "age": 14,                # Invalid age (<18)
        "monthly_salary": -500    # Invalid negative salary
    }
    res = client.post("/predict/attrition", json=invalid_payload)
    assert res.status_code == 422
