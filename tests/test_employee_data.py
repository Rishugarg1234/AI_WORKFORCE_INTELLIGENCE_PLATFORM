"""
Data integrity and consistency tests for the Employee 360° Drill Down view.
Verifies that individual employee records retrieved via API exactly match
the underlying processed employee intelligence dataset across all attributes.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH

client = TestClient(app)

@pytest.fixture
def intelligence_df():
    assert EMPLOYEE_INTELLIGENCE_PATH.exists(), f"Dataset not found at {EMPLOYEE_INTELLIGENCE_PATH}"
    return pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)

def test_employee_360_sample_integrity(intelligence_df):
    """Samples 15 diverse employees across cohorts and validates API response fidelity."""
    # Seeded sample for reproducibility
    sample_ids = intelligence_df["employee_id"].sample(15, random_state=42).tolist()

    # Explicitly include edge cases (first from attrition cohort, first from engagement cohort)
    sample_ids.extend([1, 100021])

    for emp_id in set(sample_ids):
        res = client.get(f"/employees/{emp_id}")
        assert res.status_code == 200, f"Failed to retrieve employee_id {emp_id}"
        api_data = res.json()

        df_row = intelligence_df[intelligence_df["employee_id"] == emp_id].iloc[0]

        # Verify identity
        assert api_data["employee_id"] == int(df_row["employee_id"])
        assert api_data["name"] == str(df_row["name"])

        # Verify organizational placement
        assert api_data["department"] == str(df_row["department"])
        assert api_data["job_role"] == str(df_row["job_role"])

        # Verify risk and engagement scores
        assert abs(api_data["attrition_probability"] - float(df_row["attrition_probability"])) < 1e-3
        assert api_data["attrition_risk_level"] == str(df_row["attrition_risk_level"])
        assert abs(api_data["engagement_score"] - float(df_row["engagement_score"])) < 1e-2

        # Verify skill capabilities and gaps
        assert abs(api_data["readiness_score"] - float(df_row["readiness_score"])) < 1e-2
        assert abs(api_data["skill_gap_percentage"] - float(df_row["skill_gap_percentage"])) < 1e-2

        if pd.notnull(df_row["matched_skills"]):
            assert api_data["matched_skills"] == str(df_row["matched_skills"])
        if pd.notnull(df_row["missing_skills"]):
            assert api_data["missing_skills"] == str(df_row["missing_skills"])

def test_employee_dossier_cross_contamination_check(intelligence_df):
    """Ensures records belonging to different employees do not share or cross-leak unique names."""
    emp_a_row = intelligence_df.iloc[0]
    emp_b_row = intelligence_df.iloc[100]

    res_a = client.get(f"/employees/{emp_a_row['employee_id']}").json()
    res_b = client.get(f"/employees/{emp_b_row['employee_id']}").json()

    assert res_a["employee_id"] != res_b["employee_id"]
    assert res_a["name"] != res_b["name"]
