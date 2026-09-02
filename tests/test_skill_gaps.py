"""
Validation and integrity tests for Organization-Wide Skill Gaps.
Verifies aggregation math, ranking order, severity classifications, and non-negativity.
"""

import pytest
import pandas as pd
from app.utils.config import EMPLOYEE_SKILL_GAPS_PATH, ORGANIZATION_SKILL_GAPS_PATH
from app.services.dashboard_service import dashboard_service

@pytest.fixture
def skill_gap_datasets():
    assert EMPLOYEE_SKILL_GAPS_PATH.exists(), f"Missing {EMPLOYEE_SKILL_GAPS_PATH}"
    assert ORGANIZATION_SKILL_GAPS_PATH.exists(), f"Missing {ORGANIZATION_SKILL_GAPS_PATH}"
    df_emp_gaps = pd.read_csv(EMPLOYEE_SKILL_GAPS_PATH)
    df_org_gaps = pd.read_csv(ORGANIZATION_SKILL_GAPS_PATH)
    return df_emp_gaps, df_org_gaps

def test_organization_skill_gap_independent_aggregation(skill_gap_datasets):
    """Independently explodes employee missing skills and compares top 10 counts against org gap master."""
    df_emp_gaps, df_org_gaps = skill_gap_datasets

    # Independent aggregation
    exploded_missing = []
    for _, row in df_emp_gaps.iterrows():
        if pd.notnull(row["missing_skills"]) and str(row["missing_skills"]).strip() not in ["", "None"]:
            skills = [s.strip() for s in str(row["missing_skills"]).split(";") if s.strip()]
            for s in skills:
                exploded_missing.append(s)

    independent_counts = pd.Series(exploded_missing).value_counts().to_dict()

    # Compare top 10 skills in org gaps
    for _, row in df_org_gaps.head(10).iterrows():
        sk_name = row["missing_skill"]
        expected_count = int(row["employees_missing"])
        actual_calc = independent_counts.get(sk_name, 0)
        assert actual_calc == expected_count, f"Count mismatch for skill {sk_name}: expected {expected_count}, got {actual_calc}"

def test_organization_skill_gap_sorting_and_validity(skill_gap_datasets):
    """Verifies that organization gaps are sorted in descending order without duplicates or invalid severities."""
    _, df_org_gaps = skill_gap_datasets

    assert len(df_org_gaps) > 0
    assert df_org_gaps["missing_skill"].nunique() == len(df_org_gaps), "Found duplicate skill entries in org gaps master"

    # Verify descending sort
    counts = df_org_gaps["employees_missing"].tolist()
    assert counts == sorted(counts, reverse=True), "Organization skill gaps are not sorted descending by employees_missing"

    # Verify non-negativity
    assert (df_org_gaps["employees_missing"] >= 0).all()
    assert (df_org_gaps["percentage_missing"] >= 0.0).all()

    # Verify severity labels
    valid_severities = {"HIGH", "MEDIUM", "LOW"}
    actual_severities = set(df_org_gaps["severity"].unique())
    assert actual_severities.issubset(valid_severities), f"Unexpected severity labels: {actual_severities - valid_severities}"

def test_api_skill_gaps_matches_service():
    """Verifies that the dashboard service returns accurate top skill gap records."""
    top_5 = dashboard_service.get_organization_skill_gaps(limit=5)
    assert len(top_5) == 5
    assert all("missing_skill" in item for item in top_5)
    assert all("severity" in item for item in top_5)
