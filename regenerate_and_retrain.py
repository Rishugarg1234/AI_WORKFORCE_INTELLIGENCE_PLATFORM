"""
Probabilistic Attrition Dataset Regeneration and Model Retraining Pipeline
===========================================================================
This script replaces the deterministic rule-based AttritionRisk label generation
with a realistic multi-factor probabilistic model. It then retrains the XGBoost
pipeline, evaluates it, and updates all artifacts.

Methodology:
    1. Read the existing raw dataset (all columns preserved, schema unchanged).
    2. Compute a multi-factor risk score using meaningful HR signals.
    3. Apply sigmoid transformation to convert score to probability.
    4. Add controlled Gaussian noise to prevent perfect predictability.
    5. Sample AttritionRisk probabilistically (Bernoulli draw).
    6. Preserve ~10-12% base rate to keep class imbalance realistic.
    7. Overwrite ONLY the AttritionRisk column in data/raw/employee_attrition.csv.
    8. Re-run the cleaning pipeline to produce employee_attrition_processed.csv.
    9. Retrain the XGBoost pipeline with identical architecture.
   10. Evaluate on holdout test set and run 5-fold cross-validation.
   11. Update models/v1/attrition_pipeline.joblib, metadata.json, evaluation_results.json.
   12. Regenerate confusion_matrix.png.

IMPORTANT: No target leakage. No hardcoded accuracy target. Results reflect actual model fit.
"""

import json
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "employee_attrition.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "v1" / "attrition_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "v1" / "metadata.json"
EVAL_JSON_PATH = BASE_DIR / "models" / "v1" / "evaluation_results.json"
CM_IMG_PATH = BASE_DIR / "docs" / "confusion_matrix.png"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# STEP 1: Probabilistic Label Generation
# ─────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Handle degenerate constant series gracefully."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def generate_probabilistic_labels(df_raw: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Replaces AttritionRisk with probabilistically generated labels.

    Risk Score Composition (all factors normalized to [0,1] before weighting):
      Factor                    Direction   Weight   Rationale
      ─────────────────────────────────────────────────────────────
      OvertimeHoursPerMonth     ↑ risk      0.25     Burnout driver
      PerformanceRating         ↓ risk      0.20     Low perf → more at-risk
      WorkLifeBalanceScore      ↓ risk      0.18     Poor WLB → higher flight
      PromotionGap (years)      ↑ risk      0.15     Stagnation increases risk
      CustomerSatisfaction      ↓ risk      0.10     Low satisfaction → higher risk
      MonthlySalary             ↓ risk      0.07     Low pay → higher risk
      LeavesTaken               ↑ risk      0.05     Disengagement proxy

    Noise: Gaussian N(0, 0.35) added to logit to prevent determinism.
    Intercept: -2.2 applied so base attrition rate ≈ 10-13%.
    """
    rng = np.random.default_rng(seed)
    d = df_raw.copy()

    # Compute promotion gap in years (higher = longer since last promotion = more risk)
    d['_promo_gap'] = (2026 - pd.to_numeric(d['LastPromotionYear'], errors='coerce')).clip(lower=0)

    # Normalize each factor to [0, 1]
    ot_norm   = _minmax_norm(pd.to_numeric(d['OvertimeHoursPerMonth'], errors='coerce').fillna(0))
    perf_norm = _minmax_norm(pd.to_numeric(d['PerformanceRating'], errors='coerce').fillna(3))
    wlb_norm  = _minmax_norm(pd.to_numeric(d['WorkLifeBalanceScore'], errors='coerce').fillna(d['WorkLifeBalanceScore'].median()))
    promo_gap_norm = _minmax_norm(d['_promo_gap'].fillna(5))
    sat_norm  = _minmax_norm(pd.to_numeric(d['CustomerSatisfaction'], errors='coerce').fillna(5.0))
    sal_norm  = _minmax_norm(pd.to_numeric(d['MonthlySalary'], errors='coerce').fillna(d['MonthlySalary'].median()))
    leave_norm = _minmax_norm(pd.to_numeric(d['LeavesTaken'], errors='coerce').fillna(d['LeavesTaken'].median()))

    # Multi-factor risk logit (positive = increases risk, negative = decreases risk)
    logit = (
        + 0.25 * ot_norm           # high overtime → risk up
        - 0.20 * perf_norm         # high performance → risk down
        - 0.18 * wlb_norm          # good WLB → risk down
        + 0.15 * promo_gap_norm    # long promotion gap → risk up
        - 0.10 * sat_norm          # high satisfaction → risk down
        - 0.07 * sal_norm          # high salary → risk down
        + 0.05 * leave_norm        # more leaves taken → risk up (disengagement)
    )

    # Intercept: shifts base rate to ~10-12%
    INTERCEPT = -2.2
    logit_scaled = (logit - logit.mean()) / (logit.std() + 1e-9) * 1.5 + INTERCEPT

    # Controlled Gaussian noise — prevents any single feature from being perfectly predictive
    NOISE_STD = 0.35
    noise = rng.normal(0, NOISE_STD, size=len(d))
    logit_noisy = logit_scaled + noise

    # Convert to probability via sigmoid
    prob = _sigmoid(logit_noisy)

    # Bernoulli draw — stochastic label assignment
    labels = rng.binomial(1, prob).astype(int)

    actual_rate = labels.mean()
    print(f"  Attrition base rate after probabilistic generation: {actual_rate:.3f} ({labels.sum()} Yes / {(1-labels).sum()} No)")
    if actual_rate < 0.05 or actual_rate > 0.30:
        print(f"  WARNING: Base rate {actual_rate:.3f} is outside typical HR range [0.05, 0.30]. Adjust INTERCEPT if needed.")

    # Map 1/0 back to Yes/No
    df_raw_out = df_raw.copy()
    df_raw_out.drop(columns=['_promo_gap'], errors='ignore', inplace=True)
    df_raw_out['AttritionRisk'] = np.where(labels == 1, 'Yes', 'No')
    return df_raw_out, prob


# ─────────────────────────────────────────────
# STEP 2: Cleaning Pipeline (matches notebook 03)
# ─────────────────────────────────────────────

def clean_attrition_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Replicates the cleaning logic from notebook 03_data_cleaning.ipynb."""
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    df = df.drop_duplicates()

    # CustomerSatisfaction imputation + missing indicator
    cs = pd.to_numeric(df['CustomerSatisfaction'].replace('nan', np.nan), errors='coerce')
    med_sat = cs.median()
    df['CustomerSatisfaction_missing'] = cs.isnull().astype(int)
    df['CustomerSatisfaction'] = cs.fillna(med_sat)

    # Ensure numeric types
    num_cols = ['Age', 'EducationLevel', 'MonthlySalary', 'OvertimeHoursPerMonth',
                'LeavesTaken', 'ProjectsHandled', 'TrainingHours',
                'LastPromotionYear', 'YearsAtCompany', 'PerformanceRating']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['WorkLifeBalanceScore'] = pd.to_numeric(df['WorkLifeBalanceScore'], errors='coerce')

    return df


# ─────────────────────────────────────────────
# STEP 3: Feature Engineering + Training Pipeline
# ─────────────────────────────────────────────

def build_feature_matrix(df_proc: pd.DataFrame):
    """Applies domain feature engineering consistent with the production predictor."""
    df = df_proc.copy()
    df['income_per_year_at_company'] = df['MonthlySalary'] * 12.0 / (df['YearsAtCompany'] + 1.0)
    df['promotion_gap_ratio'] = (2026.0 - df['LastPromotionYear']) / (df['YearsAtCompany'] + 1.0)
    df['overtime_ratio'] = df['OvertimeHoursPerMonth'] / 160.0
    df['leave_utilization'] = df['LeavesTaken'] / 20.0
    df['work_life_satisfaction'] = df['WorkLifeBalanceScore'] * df['CustomerSatisfaction']

    target_col = 'AttritionRisk'
    y = df[target_col].map({'Yes': 1, 'No': 0}).values

    drop_cols = ['EmployeeID', 'Name', 'PhoneNumber', 'JoiningDate',
                 'LastLeaveDate', target_col, 'CountryCode']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y


def train_pipeline(X_train, y_train, num_cols, cat_cols):
    """Builds and fits the same sklearn Pipeline used in original training."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    clf = XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_SEED
    )
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    return pipe


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PROBABILISTIC DATASET REGENERATION & MODEL RETRAINING")
    print("=" * 60)

    # --- 1. Load raw data ---
    print("\n[1/9] Loading raw dataset...")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"  Loaded {len(df_raw)} rows from {RAW_PATH}")
    orig_rate = (df_raw['AttritionRisk'] == 'Yes').mean()
    print(f"  Original deterministic attrition rate: {orig_rate:.3f}")

    # --- 2. Generate probabilistic labels ---
    print("\n[2/9] Generating probabilistic AttritionRisk labels...")
    df_relabelled, attrition_probs = generate_probabilistic_labels(df_raw, seed=RANDOM_SEED)

    # Verify: confirm no single simple rule reconstructs the new labels
    new_labels_yes = (df_relabelled['AttritionRisk'] == 'Yes')
    old_rule_yes = (df_raw['PerformanceRating'] <= 2) & (df_raw['OvertimeHoursPerMonth'] >= 21)
    rule_match_pct = (old_rule_yes == new_labels_yes).mean()
    print(f"  Old deterministic rule agreement with new labels: {rule_match_pct:.3f} (was 1.000)")
    assert rule_match_pct < 0.98, "New labels still suspiciously match the old rule!"

    # --- 3. Overwrite raw AttritionRisk column ONLY ---
    print("\n[3/9] Overwriting AttritionRisk in data/raw/employee_attrition.csv...")
    df_raw_updated = df_raw.copy()
    df_raw_updated['AttritionRisk'] = df_relabelled['AttritionRisk']
    df_raw_updated.to_csv(RAW_PATH, index=False)
    print(f"  Saved: {RAW_PATH}")
    print(f"  New distribution: {df_raw_updated['AttritionRisk'].value_counts().to_dict()}")

    # --- 4. Re-run cleaning pipeline ---
    print("\n[4/9] Re-running data cleaning pipeline...")
    df_proc = clean_attrition_data(df_raw_updated)
    df_proc.to_csv(PROCESSED_PATH, index=False)
    print(f"  Saved cleaned dataset: {PROCESSED_PATH} ({len(df_proc)} rows, {len(df_proc.columns)} cols)")

    # --- 5. Build feature matrix and split ---
    print("\n[5/9] Engineering features and splitting dataset...")
    X, y = build_feature_matrix(df_proc)
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")
    print(f"  Feature count: {len(X.columns)}")

    # --- 6. Train model ---
    print("\n[6/9] Training XGBoost pipeline...")
    pipe = train_pipeline(X_train, y_train, num_cols, cat_cols)
    print("  Training complete.")

    # --- 7. Evaluate on holdout test set ---
    print("\n[7/9] Evaluating on holdout test set (stratified 80/20)...")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    roc  = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0
    clf_report_str  = classification_report(y_test, y_pred, target_names=['Active (No)', 'Attrition (Yes)'], zero_division=0)
    clf_report_dict = classification_report(y_test, y_pred, target_names=['Active (No)', 'Attrition (Yes)'], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*40}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"Algorithm:     XGBoost")
    print(f"Model Version: v1 (retrained)")
    print(f"Evaluation:    Stratified Holdout Test Split (20%)")
    print(f"Test Samples:  {len(y_test)}")
    print()
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"ROC-AUC:       {roc:.4f}")
    print()
    print("Classification Report:")
    print(clf_report_str)
    print(f"Confusion Matrix:")
    print(f"  TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]} | TP: {cm[1,1]}")
    print(f"{'='*40}")

    # --- 8. Cross-validation ---
    print("\n[8/9] Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
    cv_f1  = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
    cv_roc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    print(f"  F1    per fold: {[round(v,4) for v in cv_f1]}")
    print(f"  F1    mean={cv_f1.mean():.4f}  std={cv_f1.std():.4f}")
    print(f"  ROC   per fold: {[round(v,4) for v in cv_roc]}")
    print(f"  ROC   mean={cv_roc.mean():.4f}  std={cv_roc.std():.4f}")

    # --- 9. Save all artifacts ---
    print("\n[9/9] Saving model artifacts...")

    # Save pipeline
    joblib.dump(pipe, MODEL_PATH)
    print(f"  Saved pipeline: {MODEL_PATH}")

    # Update metadata.json
    metadata = {
        "model_name": "attrition_risk_classifier",
        "version": "v1",
        "algorithm": "XGBoost",
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "employee_attrition_processed.csv",
        "target_column": "AttritionRisk",
        "positive_class": "Yes (1)",
        "negative_class": "No (0)",
        "target_generation": "probabilistic_multi_factor",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(len(X.columns)),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "test_size": 0.2,
        "random_state": RANDOM_SEED
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  Saved metadata: {METADATA_PATH}")

    # Update evaluation_results.json
    eval_results = {
        "model_name": "attrition_risk_classifier",
        "algorithm": "XGBoost",
        "model_version": "v1",
        "evaluation_dataset": "employee_attrition_processed.csv",
        "target_generation_methodology": "probabilistic_multi_factor_sigmoid",
        "split_type": "Stratified Holdout Test Split (20%)",
        "test_sample_count": int(len(y_test)),
        "class_distribution": {
            "negative_active_count": int((y_test == 0).sum()),
            "positive_attrition_count": int((y_test == 1).sum())
        },
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(roc, 4),
        "cross_validation": {
            "strategy": "StratifiedKFold(5, shuffle=True, random_state=99)",
            "f1_per_fold":    [round(v, 4) for v in cv_f1],
            "f1_mean":        round(float(cv_f1.mean()), 4),
            "f1_std":         round(float(cv_f1.std()), 4),
            "roc_per_fold":   [round(v, 4) for v in cv_roc],
            "roc_mean":       round(float(cv_roc.mean()), 4),
            "roc_std":        round(float(cv_roc.std()), 4)
        },
        "confusion_matrix": {
            "true_negatives":  int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives":  int(cm[1, 1])
        },
        "classification_report": clf_report_dict
    }
    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"  Saved evaluation results: {EVAL_JSON_PATH}")

    # Regenerate confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=['Predicted Active', 'Predicted Attrition'],
        yticklabels=['Actual Active', 'Actual Attrition']
    )
    plt.title(f"Confusion Matrix — XGBoost v1 (Probabilistic Dataset)", fontsize=12)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(CM_IMG_PATH, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {CM_IMG_PATH}")

    print("\n" + "=" * 60)
    print("REGENERATION AND RETRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Metrics (Holdout Test Set):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"\nCross-Validation F1:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"Cross-Validation ROC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    print("\nNext steps:")
    print("  1. Run: pytest -v")
    print("  2. Run: uvicorn app.main:app --reload")
    print("  3. Run: streamlit run frontend/dashboard.py")


if __name__ == "__main__":
    main()
