"""
Application configuration module.
Defines file paths, model versions, and runtime configuration settings.
"""

from pathlib import Path
import os

# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PREDICTIONS_DIR = DATA_DIR / "predictions"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

DATA_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Processed dataset file paths
EMPLOYEE_ATTRITION_PROCESSED_PATH = DATA_PROCESSED_DIR / "employee_attrition_processed.csv"
ENGAGEMENT_PROCESSED_PATH = DATA_PROCESSED_DIR / "engagement_processed.csv"
OCCUPATION_MASTER_PATH = DATA_PROCESSED_DIR / "occupation_master.csv"
ESSENTIAL_SKILLS_PATH = DATA_PROCESSED_DIR / "essential_skills_processed.csv"
SOFTWARE_SKILLS_PATH = DATA_PROCESSED_DIR / "software_skills_processed.csv"
ROLE_SKILL_MATRIX_PATH = DATA_PROCESSED_DIR / "role_skill_matrix.csv"
EMPLOYEE_SKILLS_PATH = DATA_PROCESSED_DIR / "employee_skills.csv"
EMPLOYEE_SKILL_GAPS_PATH = DATA_PROCESSED_DIR / "employee_skill_gaps.csv"
ORGANIZATION_SKILL_GAPS_PATH = DATA_PROCESSED_DIR / "organization_skill_gaps.csv"
EMPLOYEE_RECOMMENDATIONS_PATH = DATA_PROCESSED_DIR / "employee_recommendations.csv"
EMPLOYEE_INTELLIGENCE_PATH = DATA_PROCESSED_DIR / "employee_intelligence.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_V1_DIR = MODELS_DIR / "v1"
MODEL_PIPELINE_PATH = MODEL_V1_DIR / "attrition_pipeline.joblib"
MODEL_METADATA_PATH = MODEL_V1_DIR / "metadata.json"

# Logging paths
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "app.log"

# Prediction audit log path
PREDICTIONS_LOG_PATH = DATA_PREDICTIONS_DIR / "predictions_log.csv"

# API Configuration
API_TITLE = "Enterprise HR AI — Workforce Intelligence API"
API_DESCRIPTION = "Production AI API for Attrition Prediction, Engagement Intelligence, Skill Gap Analysis & Upskilling Recommendations"
API_VERSION = "1.0.0"
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 8000))
