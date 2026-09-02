import pandas as pd
import json
import joblib
from pathlib import Path

BASE = Path(".")
raw = BASE / "data" / "raw"
processed = BASE / "data" / "processed"
models = BASE / "models" / "v1"

print("=== RAW DATASETS ===")
for f in sorted(raw.iterdir()):
    print(f"  [OK] {f.name} ({f.stat().st_size//1024}KB)")

print("")
print("=== PROCESSED DATASETS ===")
for f in sorted(processed.iterdir()):
    print(f"  [OK] {f.name} ({f.stat().st_size//1024}KB)")

print("")
print("=== MODEL FILES ===")
for f in sorted(models.iterdir()):
    print(f"  [OK] {f.name} ({f.stat().st_size//1024}KB)")

print("")
print("=== LOADING MODEL ===")
pipe = joblib.load(models / "attrition_pipeline.joblib")
print(f"  Model loaded: {type(pipe).__name__}")
meta = json.loads((models / "metadata.json").read_text())
print(f"  Algorithm:    {meta.get('algorithm')}")
print(f"  Version:      {meta.get('version')}")

eval_res = json.loads((models / "evaluation_results.json").read_text())
print("")
print("=== EVALUATION RESULTS (Holdout Test Split) ===")
for k, v in eval_res.items():
    if k not in ("classification_report", "confusion_matrix", "class_distribution"):
        print(f"  {k}: {v}")

print("")
print("=== EMPLOYEE INTELLIGENCE DATASET ===")
df = pd.read_csv(processed / "employee_intelligence.csv")
print(f"  Shape: {df.shape}")
print(f"  Unique Employee IDs: {df['employee_id'].nunique()}")
print(f"  Departments: {sorted(df['department'].unique().tolist())}")
print(f"  Risk Tiers: {df['attrition_risk_level'].value_counts().to_dict()}")
print(f"  Avg Engagement Score: {df['engagement_score'].mean():.2f}")
print(f"  Avg Skill Readiness:  {df['readiness_score'].mean():.2f}%")

print("")
print("=== TOP 5 ORG-WIDE SKILL GAPS ===")
sg = pd.read_csv(processed / "organization_skill_gaps.csv")
for _, r in sg.head(5).iterrows():
    print(f"  Rank: {r['missing_skill']} — {int(r['employees_missing'])} employees ({r['severity']})")

print("")
print("=== VERIFICATION ARTIFACTS ===")
artifacts = [
    "verify_dashboard_data.py",
    "evaluate_model.py",
    "data/processed/dashboard_verification_report.txt",
    "models/v1/evaluation_results.json",
    "docs/confusion_matrix.png",
    "docs/TESTING.md",
    "tests/test_api.py",
    "tests/test_attrition.py",
    "tests/test_data_quality.py",
    "tests/test_department_filter.py",
    "tests/test_employee_data.py",
    "tests/test_model_sanity.py",
    "tests/test_skill_gap.py",
    "tests/test_skill_gaps.py",
    "tests/test_validation.py",
]
for a in artifacts:
    p = BASE / a
    status = "[OK]" if p.exists() else "[MISSING]"
    print(f"  {status} {a}")
