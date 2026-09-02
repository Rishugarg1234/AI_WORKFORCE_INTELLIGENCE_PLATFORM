"""
Unit tests for the Skill Gap Analysis engine.
"""

from app.services.skill_gap_service import skill_gap_service

def test_full_skill_match():
    """Tests employee possessing 100% of required role skills."""
    role = "Developer"
    # Developer skills: ['Python', 'SQL', 'Git', 'Docker', 'REST APIs', 'System Design']
    skills = ['Python', 'SQL', 'Git', 'Docker', 'REST APIs', 'System Design']
    
    result = skill_gap_service.calculate_skill_gap(role, skills)
    
    assert result['total_matched_skills'] == len(result['matched_skills'])
    assert len(result['missing_skills']) == 0
    assert result['readiness_score'] == 100.0
    assert result['skill_gap_percentage'] == 0.0

def test_partial_skill_gap():
    """Tests employee missing some required skills."""
    role = "Developer"
    skills = ['Python', 'SQL']
    
    result = skill_gap_service.calculate_skill_gap(role, skills)
    
    assert 'Python' in result['matched_skills']
    assert 'SQL' in result['matched_skills']
    assert len(result['missing_skills']) > 0
    assert result['readiness_score'] < 100.0
    assert result['skill_gap_percentage'] > 0.0
    assert result['readiness_score'] + result['skill_gap_percentage'] == 100.0

def test_zero_skill_match():
    """Tests employee possessing none of the required skills."""
    role = "Accountant"
    # Accountant skills: ['Financial Accounting', 'Tax Compliance', 'Excel', 'QuickBooks', 'Auditing', 'ERP Systems']
    skills = ['Rust', 'Cobol', 'Clojure']
    
    result = skill_gap_service.calculate_skill_gap(role, skills)
    
    assert len(result['matched_skills']) == 0
    assert result['readiness_score'] == 0.0
    assert result['skill_gap_percentage'] == 100.0
