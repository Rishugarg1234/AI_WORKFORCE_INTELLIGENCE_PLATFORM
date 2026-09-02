"""
Upskilling and course recommendation service.
"""

from typing import Dict, Any, List
import pandas as pd
from app.utils.config import EMPLOYEE_RECOMMENDATIONS_PATH, ORGANIZATION_SKILL_GAPS_PATH

class RecommendationService:
    """Service providing targeted upskilling pathways."""
    
    _catalog = {
        'Python': {
            'course': 'Advanced Python for Enterprise Systems',
            'certification': 'PCEP / PCAP Certified Associate Python Programmer',
            'project': 'Build an Automated Asynchronous ETL Data Pipeline',
            'type': 'Technical Courseware'
        },
        'Docker': {
            'course': 'Enterprise Containerization with Docker & Kubernetes',
            'certification': 'Docker Certified Associate (DCA)',
            'project': 'Containerize Core Microservices and Implement Docker Compose',
            'type': 'DevOps Hands-on'
        },
        'Kubernetes': {
            'course': 'Cloud Native Orchestration & Microservices',
            'certification': 'Certified Kubernetes Administrator (CKA)',
            'project': 'Deploy Zero-Downtime Multi-Cluster Deployment',
            'type': 'Cloud Certification'
        },
        'Sql': {
            'course': 'Advanced SQL Analytics and Query Optimization',
            'certification': 'Oracle / PostgreSQL Certified Professional',
            'project': 'Optimize High-Throughput Relational Database Queries',
            'type': 'Data Engineering'
        },
        'Financial Modeling': {
            'course': 'Advanced Corporate Financial Modeling & Valuation',
            'certification': 'FMVA (Financial Modeling & Valuation Analyst)',
            'project': 'Construct 3-Statement Forecasting and Scenario Model',
            'type': 'Finance Masterclass'
        },
        'Seo': {
            'course': 'Technical SEO & Modern Web Visibility Masterclass',
            'certification': 'Google Analytics & Advanced SEO Certification',
            'project': 'Conduct Technical Site Audit and Core Web Vitals Optimization',
            'type': 'Digital Marketing'
        },
        'Talent Acquisition': {
            'course': 'Strategic Talent Acquisition & Competency-Based Sourcing',
            'certification': 'SHRM-CP / AIRS Certified Diversity Recruiter',
            'project': 'Design High-Retention Executive Hiring Funnel',
            'type': 'HR Leadership'
        },
        'Ci/Cd': {
            'course': 'Continuous Integration & Continuous Deployment with GitHub Actions',
            'certification': 'DevOps Institute CI/CD Foundation',
            'project': 'Architect End-to-End Automated Testing & Deployment Pipeline',
            'type': 'DevOps Hands-on'
        }
    }
    
    @classmethod
    def get_recommendations_for_skills(cls, missing_skills: List[str]) -> List[Dict[str, Any]]:
        """Generates actionable recommendations for a given list of missing skills."""
        results = []
        for sk in missing_skills:
            sk_clean = sk.strip().title()
            cat = cls._catalog.get(sk_clean, {
                'course': f'Mastery Course in {sk_clean}',
                'certification': f'Professional Certificate in {sk_clean}',
                'project': f'Applied Hands-on Project for {sk_clean}',
                'type': 'Professional Development'
            })
            results.append({
                "missing_skill": sk_clean,
                "recommended_course": cat['course'],
                "recommended_certification": cat['certification'],
                "recommended_project": cat['project'],
                "recommendation_type": cat['type'],
                "priority": "High"
            })
        return results

    @staticmethod
    def get_employee_recommendations(employee_id: int) -> List[Dict[str, Any]]:
        """Retrieves stored recommendations for an employee."""
        if not EMPLOYEE_RECOMMENDATIONS_PATH.exists():
            return []
        df = pd.read_csv(EMPLOYEE_RECOMMENDATIONS_PATH)
        match = df[df['employee_id'] == employee_id]
        return match.to_dict(orient="records")

recommendation_service = RecommendationService()
