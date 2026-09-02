"""
Model Evaluation Script for Enterprise HR AI Platform.
Evaluates the saved production model on the holdout test set (Stratified 80/20, random_state=42).
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "v1" / "attrition_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "v1" / "metadata.json"
RESULTS_JSON_PATH = BASE_DIR / "models" / "v1" / "evaluation_results.json"
CONFUSION_MATRIX_IMG_PATH = BASE_DIR / "docs" / "confusion_matrix.png"

def evaluate():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Processed attrition dataset not found at {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found at {MODEL_PATH}")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Feature Engineering (consistent with training and predictor)
    df_feat = df.copy()
    df_feat['income_per_year_at_company'] = df_feat['MonthlySalary'] * 12.0 / (df_feat['YearsAtCompany'] + 1.0)
    df_feat['promotion_gap_ratio'] = (2026.0 - df_feat['LastPromotionYear']) / (df_feat['YearsAtCompany'] + 1.0)
    df_feat['overtime_ratio'] = df_feat['OvertimeHoursPerMonth'] / 160.0
    df_feat['leave_utilization'] = df_feat['LeavesTaken'] / 20.0
    df_feat['work_life_satisfaction'] = df_feat['WorkLifeBalanceScore'] * df_feat['CustomerSatisfaction']

    target_col = 'AttritionRisk'
    y = df_feat[target_col].map({'Yes': 1, 'No': 0}).values

    drop_cols = ['EmployeeID', 'Name', 'PhoneNumber', 'JoiningDate', 'LastLeaveDate', target_col, 'CountryCode']
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # Holdout Test Split (Exact 80/20 Stratified Split matching training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load trained model pipeline
    pipeline = joblib.load(MODEL_PATH)

    # Read metadata if exists
    meta_info = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta_info = json.load(f)

    model_name = meta_info.get("model_name", "attrition_risk_classifier")
    algo_name = meta_info.get("algorithm", "Logistic Regression")
    ver_name = meta_info.get("version", "v1")
    threshold = meta_info.get("decision_threshold", 0.5)

    # Predictions & Probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate actual metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    
    clf_report = classification_report(y_test, y_pred, target_names=['Active (No)', 'Attrition (Yes)'], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print Formatted Results
    print("========================================")
    print("MODEL EVALUATION RESULTS")
    print("========================================")
    print(f"Model Name:          {model_name}")
    print(f"Algorithm:           {algo_name}")
    print(f"Model Version:       {ver_name}")
    print(f"Decision Threshold:  {threshold}")
    print("")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print("")
    print("Classification Report:")
    print(clf_report)
    print("Confusion Matrix:")
    print(f"  TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
    print(f"  FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
    print("========================================")

if __name__ == "__main__":
    evaluate()
