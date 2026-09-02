"""
Simple, Academically Defensible ML Model Comparison and Training Pipeline.
============================================================================
Enterprise HR AI — Attrition Risk Classifier v1

Designed for a 3rd-year engineering project:
  - 3 Classical Models: Logistic Regression, Random Forest, XGBoost
  - Built-in class imbalance handling (class_weight='balanced' / scale_pos_weight)
  - Standard Stratified 80/20 train/test split (random_state=42)
  - Model selection based on F1 Score & ROC-AUC balance (not raw accuracy alone)
  - Clean sklearn Pipeline with Standard Scaler + Imputer + OneHotEncoder
  - No complex GridSearch, SMOTE, or exotic over-engineering
"""

import json
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "v1" / "attrition_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "v1" / "metadata.json"
EVAL_JSON_PATH = BASE_DIR / "models" / "v1" / "evaluation_results.json"
CM_IMG_PATH = BASE_DIR / "docs" / "confusion_matrix.png"

RANDOM_STATE = 42

def build_features(df: pd.DataFrame):
    """Domain feature engineering matching predictor.py."""
    d = df.copy()
    d["income_per_year_at_company"] = d["MonthlySalary"] * 12.0 / (d["YearsAtCompany"] + 1.0)
    d["promotion_gap_ratio"] = (2026.0 - d["LastPromotionYear"]) / (d["YearsAtCompany"] + 1.0)
    d["overtime_ratio"] = d["OvertimeHoursPerMonth"] / 160.0
    d["leave_utilization"] = d["LeavesTaken"] / 20.0
    d["work_life_satisfaction"] = d["WorkLifeBalanceScore"] * d["CustomerSatisfaction"]

    y = d["AttritionRisk"].map({"Yes": 1, "No": 0}).values
    drop_cols = ["EmployeeID", "Name", "PhoneNumber", "JoiningDate",
                 "LastLeaveDate", "AttritionRisk", "CountryCode"]
    X = d.drop(columns=[c for c in drop_cols if c in d.columns])
    return X, y

def make_pipeline(classifier, num_cols, cat_cols) -> Pipeline:
    """Builds a standard, leakage-free sklearn pipeline."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

def eval_metrics(pipe, X_test, y_test, threshold: float = 0.5) -> dict:
    """Calculates evaluation metrics at a specific probability threshold."""
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "y_pred": y_pred,
        "proba": proba,
    }

def main():
    print("=" * 60)
    print("ENTERPRISE HR AI - 3-MODEL COMPARISON & SELECTION")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(DATASET_PATH)
    X, y = build_features(df)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # Step 1: Check Class Balance
    yes_count = int(y.sum())
    no_count = int((y == 0).sum())
    total_count = len(y)
    print("\n[STEP 1] Class Distribution:")
    print(f"  - AttritionRisk = No  (Active):    {no_count:4d} ({no_count/total_count*100:.1f}%)")
    print(f"  - AttritionRisk = Yes (Attrition): {yes_count:4d} ({yes_count/total_count*100:.1f}%)")
    print(f"  - Imbalance Ratio: {no_count/yes_count:.1f}x (Moderate imbalance)")
    print("  - Action: Using built-in class weighting (class_weight='balanced' / scale_pos_weight)")

    # Step 2: Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n[STEP 2] Stratified 80/20 Train/Test Split (random_state={RANDOM_STATE}):")
    print(f"  - Training Set:   {len(X_train)} samples ({y_train.sum()} Attrition, {(y_train==0).sum()} Active)")
    print(f"  - Test Set:       {len(X_test)} samples ({y_test.sum()} Attrition, {(y_test==0).sum()} Active)")

    # Step 3: Candidate Models with Imbalance Handling
    spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            max_depth=6,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="logloss",
            max_depth=3,
            n_estimators=100,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE
        )
    }

    # Step 4: Model Comparison
    print("\n[STEP 3 & 4] Model Evaluation & Comparison (Threshold = 0.5):")
    print("-" * 65)
    print(f"{'Model':<22} {'Accuracy':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 65)

    comparison_results = {}
    fitted_pipelines = {}

    for name, clf in candidate_models.items():
        pipe = make_pipeline(clf, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        metrics = eval_metrics(pipe, X_test, y_test, threshold=0.5)
        comparison_results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name:<22} {metrics['accuracy']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics['roc_auc']:>9.4f}")
    print("-" * 65)

    # Rank by composite balance (F1 + ROC-AUC)
    ranked = sorted(
        comparison_results.keys(),
        key=lambda n: comparison_results[n]["f1"] + comparison_results[n]["roc_auc"],
        reverse=True
    )
    best_name = ranked[0]
    best_pipe = fitted_pipelines[best_name]
    best_metrics = comparison_results[best_name]
    print(f"\n[RANKING] Best Balanced Model: {best_name}")
    for idx, n in enumerate(ranked, 1):
        f1_val = comparison_results[n]['f1']
        auc_val = comparison_results[n]['roc_auc']
        print(f"  #{idx} {n}: F1={f1_val:.4f}, ROC-AUC={auc_val:.4f} (Score={f1_val+auc_val:.4f})")

    # Step 5: Simple Threshold Check
    chosen_threshold = 0.5
    final_metrics = best_metrics

    # Test alternative threshold 0.40 only if recall < 0.40
    if best_metrics["recall"] < 0.40:
        print("\n[STEP 5] Threshold Analysis (Recall < 0.40 at standard 0.5 threshold):")
        alt_metrics = eval_metrics(best_pipe, X_test, y_test, threshold=0.40)
        print(f"  - Standard Threshold 0.50: Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}")
        print(f"  - Adjusted Threshold 0.40: Precision={alt_metrics['precision']:.4f}, Recall={alt_metrics['recall']:.4f}, F1={alt_metrics['f1']:.4f}")
        if alt_metrics["f1"] >= best_metrics["f1"]:
            print("  - Rationale: Threshold 0.40 improves recall for flight-risk detection with superior/equal F1. Adopting 0.40.")
            chosen_threshold = 0.40
            final_metrics = alt_metrics
        else:
            print("  - Keeping standard threshold 0.50 as it provides higher F1.")
    else:
        print(f"\n[STEP 5] Standard Threshold 0.50 maintains strong Recall ({best_metrics['recall']:.4f}). Retaining 0.50.")

    # Step 6: Final Evaluation
    y_pred_final = final_metrics["y_pred"]
    cm = confusion_matrix(y_test, y_pred_final)
    clf_report_str = classification_report(
        y_test, y_pred_final,
        target_names=["Active (No)", "Attrition (Yes)"],
        zero_division=0
    )
    clf_report_dict = classification_report(
        y_test, y_pred_final,
        target_names=["Active (No)", "Attrition (Yes)"],
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("FINAL SELECTED MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Selected Model:       {best_name}")
    print(f"Decision Threshold:   {chosen_threshold}")
    print(f"Accuracy:             {final_metrics['accuracy']:.4f}")
    print(f"Precision:            {final_metrics['precision']:.4f}")
    print(f"Recall:               {final_metrics['recall']:.4f}")
    print(f"F1 Score:             {final_metrics['f1']:.4f}")
    print(f"ROC-AUC:              {final_metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(clf_report_str)
    print("Confusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0,0]:3d} | False Positives (FP): {cm[0,1]:3d}")
    print(f"  False Negatives (FN): {cm[1,0]:3d} | True Positives (TP):  {cm[1,1]:3d}")

    # 5-fold Cross-Validation
    print("\n5-Fold Stratified Cross-Validation:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
    cv_f1 = cross_val_score(best_pipe, X, y, cv=cv, scoring="f1")
    cv_roc = cross_val_score(best_pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"  - CV F1 per fold:  {[round(v, 4) for v in cv_f1]}")
    print(f"  - CV F1 Mean:      {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  - CV ROC-AUC Mean: {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")

    # Persist Production Artifacts
    print("\n[STEP 7] Persisting Model Artifacts:")
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"  [OK] Saved Pipeline: {MODEL_PATH}")

    metadata = {
        "model_name": "attrition_risk_classifier",
        "version": "v1",
        "algorithm": best_name,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "employee_attrition_processed.csv",
        "target_column": "AttritionRisk",
        "positive_class": "Yes (1)",
        "negative_class": "No (0)",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(len(X.columns)),
        "decision_threshold": chosen_threshold,
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1_score": final_metrics["f1"],
        "roc_auc": final_metrics["roc_auc"],
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_roc_mean": round(float(cv_roc.mean()), 4),
        "random_state": RANDOM_STATE,
        "imbalance_handling": "Built-in class weights (balanced/scale_pos_weight)",
        "model_comparison": {
            n: {
                "accuracy": comparison_results[n]["accuracy"],
                "precision": comparison_results[n]["precision"],
                "recall": comparison_results[n]["recall"],
                "f1": comparison_results[n]["f1"],
                "roc_auc": comparison_results[n]["roc_auc"]
            } for n in candidate_models
        }
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  [OK] Saved Metadata: {METADATA_PATH}")

    eval_results = {
        "model_name": "attrition_risk_classifier",
        "algorithm": best_name,
        "model_version": "v1",
        "evaluation_dataset": "employee_attrition_processed.csv",
        "split_type": "Stratified Holdout Test Split (20%)",
        "decision_threshold": chosen_threshold,
        "test_sample_count": int(len(y_test)),
        "class_distribution": {
            "negative_active_count": int((y_test == 0).sum()),
            "positive_attrition_count": int((y_test == 1).sum())
        },
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1_score": final_metrics["f1"],
        "roc_auc": final_metrics["roc_auc"],
        "cross_validation": {
            "strategy": "StratifiedKFold(5, shuffle=True, random_state=99)",
            "f1_per_fold": [round(v, 4) for v in cv_f1],
            "f1_mean": round(float(cv_f1.mean()), 4),
            "f1_std": round(float(cv_f1.std()), 4),
            "roc_per_fold": [round(v, 4) for v in cv_roc],
            "roc_mean": round(float(cv_roc.mean()), 4),
            "roc_std": round(float(cv_roc.std()), 4)
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        },
        "model_comparison": {
            n: {k: v for k, v in comparison_results[n].items() if k not in ("y_pred", "proba")}
            for n in candidate_models
        },
        "classification_report": clf_report_dict
    }
    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"  [OK] Saved Evaluation Results: {EVAL_JSON_PATH}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Active", "Predicted Attrition"],
        yticklabels=["Actual Active", "Actual Attrition"]
    )
    plt.title(f"Confusion Matrix - {best_name}\n(Threshold: {chosen_threshold}, F1: {final_metrics['f1']:.4f}, ROC-AUC: {final_metrics['roc_auc']:.4f})", fontsize=11)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(CM_IMG_PATH, dpi=150)
    plt.close()
    print(f"  [OK] Saved Confusion Matrix Plot: {CM_IMG_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
