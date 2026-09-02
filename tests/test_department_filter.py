"""
Automated unit tests for Streamlit Department Filter functionality.
Verifies that filtering by department properly isolates department cohorts,
calculates department-specific KPIs, and that 'All' returns the full workforce.
"""

import pytest
import pandas as pd
from pathlib import Path
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH
from app.services.dashboard_service import dashboard_service

@pytest.fixture
def intelligence_df():
    assert EMPLOYEE_INTELLIGENCE_PATH.exists(), f"Missing dataset: {EMPLOYEE_INTELLIGENCE_PATH}"
    return pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)

def test_filter_all_returns_complete_dataset(intelligence_df):
    """Verifies that selecting 'All' yields the entire workforce."""
    summary_all = dashboard_service.get_summary_metrics(department="All")
    assert summary_all["total_employees"] == len(intelligence_df)
    assert summary_all["total_employees"] == 5500
    assert summary_all["high_risk_employees"] == int((intelligence_df["attrition_risk_level"] == "High").sum())

def test_department_filters_produce_accurate_subsets(intelligence_df):
    """Tests each department individually to verify exact isolation and metrics."""
    departments = sorted(intelligence_df["department"].unique().tolist())
    assert len(departments) >= 5, f"Expected at least 5 departments, found: {departments}"

    total_filtered_count = 0
    total_high_risk_count = 0

    for dept in departments:
        dept_subset = intelligence_df[intelligence_df["department"] == dept]
        summary_dept = dashboard_service.get_summary_metrics(department=dept)

        actual_count = len(dept_subset)
        actual_high_risk = int((dept_subset["attrition_risk_level"] == "High").sum())
        actual_avg_eng = round(float(dept_subset["engagement_score"].mean()), 2)
        actual_avg_read = round(float(dept_subset["readiness_score"].mean()), 2)

        assert summary_dept["total_employees"] == actual_count, f"Mismatch in count for department {dept}"
        assert summary_dept["high_risk_employees"] == actual_high_risk, f"Mismatch in high risk count for {dept}"
        assert abs(summary_dept["avg_engagement"] - actual_avg_eng) < 0.05, f"Mismatch in engagement for {dept}"
        assert abs(summary_dept["avg_readiness"] - actual_avg_read) < 0.05, f"Mismatch in readiness for {dept}"

        total_filtered_count += actual_count
        total_high_risk_count += actual_high_risk

    # The sum of all individual departments must equal the full dataset total
    assert total_filtered_count == len(intelligence_df)
    assert total_high_risk_count == int((intelligence_df["attrition_risk_level"] == "High").sum())

def test_invalid_department_handling():
    """Verifies graceful handling when an unknown department is requested."""
    summary_invalid = dashboard_service.get_summary_metrics(department="NonExistentDept")
    assert summary_invalid["total_employees"] == 0
    assert summary_invalid["high_risk_employees"] == 0
    assert summary_invalid["avg_engagement"] == 0.0
