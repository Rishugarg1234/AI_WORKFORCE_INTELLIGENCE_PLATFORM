"""
Attrition Risk Predictor and Inference Engine.
Handles feature transformations, model inference, risk classification, and audit logging.
"""

import datetime
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from app.ml.model_loader import get_attrition_pipeline, get_model_metadata
from app.validation.employee_schema import EmployeeAttritionInput, AttritionPredictionResponse
from app.utils.config import PREDICTIONS_LOG_PATH
from app.utils.logger import logger

def _log_prediction(timestamp: str, employee_id: Union[int, str, None], model_version: str, probability: float, risk_level: str):
    """Appends prediction details to audit log CSV without overwriting."""
    try:
        file_exists = PREDICTIONS_LOG_PATH.exists()
        with open(PREDICTIONS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "employee_id", "model_version", "prediction_probability", "risk_level"])
            writer.writerow([timestamp, employee_id if employee_id is not None else "N/A", model_version, round(probability, 4), risk_level])
    except Exception as e:
        logger.error(f"Failed to write prediction log: {e}")

def prepare_features(data: Dict[str, Any]) -> pd.DataFrame:
    """Applies exact feature engineering logic matching pipeline training."""
    # Normalize dictionary keys
    age = data.get("age", data.get("Age", 30))
    gender = data.get("gender", data.get("Gender", "Male"))
    department = data.get("department", data.get("Department", "IT"))
    job_role = data.get("job_role", data.get("JobRole", "Developer"))
    education_level = data.get("education_level", data.get("EducationLevel", 2))
    monthly_salary = float(data.get("monthly_salary", data.get("MonthlySalary", 60000)))
    overtime_hours = float(data.get("overtime_hours_per_month", data.get("OvertimeHoursPerMonth", 0)))
    leaves_taken = float(data.get("leaves_taken", data.get("LeavesTaken", 5)))
    projects_handled = int(data.get("projects_handled", data.get("ProjectsHandled", 5)))
    training_hours = float(data.get("training_hours", data.get("TrainingHours", 20)))
    
    raw_sat = data.get("customer_satisfaction", data.get("CustomerSatisfaction", None))
    sat_missing = 1 if raw_sat is None or pd.isna(raw_sat) else 0
    sat_val = 7.0 if sat_missing else float(raw_sat)
    
    last_promo = int(data.get("last_promotion_year", data.get("LastPromotionYear", 2022)))
    years_at_company = float(data.get("years_at_company", data.get("YearsAtCompany", 3)))
    wlb_score = float(data.get("work_life_balance_score", data.get("WorkLifeBalanceScore", 3.0)))
    perf_rating = int(data.get("performance_rating", data.get("PerformanceRating", 3)))
    country = data.get("country", data.get("Country", "USA"))
    leave_day = data.get("leave_day_name", data.get("LeaveDayName", "Friday"))

    # Feature Engineering
    income_per_year = monthly_salary * 12.0 / (years_at_company + 1.0)
    promotion_gap = (2026.0 - last_promo) / (years_at_company + 1.0)
    overtime_ratio = overtime_hours / 160.0
    leave_utilization = leaves_taken / 20.0
    work_life_satisfaction = wlb_score * sat_val

    feature_dict = {
        "Age": [age],
        "Gender": [gender],
        "Department": [department],
        "JobRole": [job_role],
        "EducationLevel": [education_level],
        "MonthlySalary": [monthly_salary],
        "OvertimeHoursPerMonth": [overtime_hours],
        "LeavesTaken": [leaves_taken],
        "ProjectsHandled": [projects_handled],
        "TrainingHours": [training_hours],
        "CustomerSatisfaction": [sat_val],
        "LastPromotionYear": [last_promo],
        "YearsAtCompany": [years_at_company],
        "WorkLifeBalanceScore": [wlb_score],
        "PerformanceRating": [perf_rating],
        "CustomerSatisfaction_missing": [sat_missing],
        "income_per_year_at_company": [income_per_year],
        "promotion_gap_ratio": [promotion_gap],
        "overtime_ratio": [overtime_ratio],
        "leave_utilization": [leave_utilization],
        "work_life_satisfaction": [work_life_satisfaction],
        "Country": [country],
        "LeaveDayName": [leave_day]
    }
    return pd.DataFrame(feature_dict)

def predict_attrition(input_data: Union[EmployeeAttritionInput, Dict[str, Any]]) -> AttritionPredictionResponse:
    """Runs inference on employee input and returns probability and risk level."""
    data_dict = input_data.model_dump() if isinstance(input_data, EmployeeAttritionInput) else input_data
    emp_id = data_dict.get("employee_id", None)
    
    logger.info(f"Received attrition prediction request for employee_id: {emp_id}")
    
    pipeline = get_attrition_pipeline()
    metadata = get_model_metadata()
    version = metadata.get("version", "v1")

    df_features = prepare_features(data_dict)
    
    # Run model prediction
    proba = float(pipeline.predict_proba(df_features)[0, 1])
    
    # Risk categorization
    if proba >= 0.70:
        risk_level = "High"
        interpretation = "High risk of departure. Immediate retention and engagement intervention recommended."
    elif proba >= 0.30:
        risk_level = "Medium"
        interpretation = "Moderate attrition risk. Monitor workload, compensation trajectory, and career progression."
    else:
        risk_level = "Low"
        interpretation = "Low attrition risk. Employee exhibits healthy organizational engagement signals."

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Audit logging
    _log_prediction(timestamp, emp_id, version, proba, risk_level)
    logger.info(f"Prediction completed: employee_id={emp_id}, proba={proba:.4f}, risk={risk_level}")

    return AttritionPredictionResponse(
        employee_id=emp_id,
        attrition_probability=round(proba, 4),
        attrition_risk_level=risk_level,
        risk_interpretation=interpretation,
        model_version=version,
        timestamp=timestamp
    )
