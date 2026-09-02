"""
Data Quality and Integrity Test Suite for the Enterprise HR AI platform.
Enforces strict schema constraints, ID uniqueness, valid ranges, categorical validity,
and data cleanliness across all processed pipeline artifacts.
"""

import pytest
import pandas as pd
import numpy as np
from app.utils.config import (
    EMPLOYEE_INTELLIGENCE_PATH,
    EMPLOYEE_ATTRITION_PROCESSED_PATH,
    ENGAGEMENT_PROCESSED_PATH,
    ROLE_SKILL_MATRIX_PATH,
    EMPLOYEE_SKILLS_PATH
)

def test_employee_id_uniqueness():
    """Validates that Employee IDs are 100% unique in unified intelligence and source cohorts."""
    df_intel = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    assert df_intel["employee_id"].nunique() == len(df_intel), "Duplicate Employee IDs detected in employee_intelligence.csv!"

    df_attr = pd.read_csv(EMPLOYEE_ATTRITION_PROCESSED_PATH)
    assert df_attr["EmployeeID"].nunique() == len(df_attr), "Duplicate Employee IDs detected in employee_attrition_processed.csv!"

    df_eng = pd.read_csv(ENGAGEMENT_PROCESSED_PATH)
    assert df_eng["employee_id"].nunique() == len(df_eng), "Duplicate Employee IDs detected in engagement_processed.csv!"

def test_required_columns_presence():
    """Validates that all critical business intelligence columns are present."""
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    expected_cols = [
        "employee_id", "name", "department", "job_role", "age",
        "monthly_salary", "years_experience", "attrition_probability",
        "attrition_risk_level", "engagement_score", "matched_skills",
        "missing_skills", "skill_gap_percentage", "readiness_score",
        "top_recommendations", "top_certifications"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: '{col}'"

def test_numerical_ranges_validity():
    """Validates sensible numerical boundaries across workforce metrics."""
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)

    # Probabilities strictly between 0 and 1
    assert (df["attrition_probability"] >= 0.0).all()
    assert (df["attrition_probability"] <= 1.0).all()

    # Readiness score strictly between 0 and 100
    assert (df["readiness_score"] >= 0.0).all()
    assert (df["readiness_score"] <= 100.0).all()

    # Skill gap percentage strictly between 0 and 100
    assert (df["skill_gap_percentage"] >= 0.0).all()
    assert (df["skill_gap_percentage"] <= 100.0).all()

    # Monthly salary strictly positive
    valid_salaries = df["monthly_salary"].dropna()
    assert (valid_salaries > 0).all()

def test_categorical_validity():
    """Validates allowed categorical values for departments and risk tiers."""
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)

    allowed_risks = {"Low", "Medium", "High"}
    actual_risks = set(df["attrition_risk_level"].dropna().unique())
    assert actual_risks.issubset(allowed_risks), f"Unexpected risk tiers: {actual_risks - allowed_risks}"

    allowed_depts = {"Finance", "HR", "IT", "Marketing", "Sales", "Support"}
    actual_depts = set(df["department"].dropna().unique())
    assert actual_depts.issubset(allowed_depts), f"Unexpected departments: {actual_depts - allowed_depts}"

def test_skill_matrix_integrity():
    """Validates that canonical role-skill matrix contains non-empty roles and skills."""
    df_matrix = pd.read_csv(ROLE_SKILL_MATRIX_PATH)
    assert len(df_matrix) > 0
    assert df_matrix["job_role"].isnull().sum() == 0
    assert df_matrix["required_skill"].isnull().sum() == 0

def test_missing_values_handled():
    """Ensures no critical fields have unhandled NaN values in the unified layer."""
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    critical_fields = ["employee_id", "name", "department", "job_role", "attrition_probability", "attrition_risk_level", "readiness_score"]
    for field in critical_fields:
        assert df[field].isnull().sum() == 0, f"Critical field '{field}' contains null values"
