# Machine Learning Model Evaluation & Selection Report
## Enterprise HR AI — Attrition Risk Classifier v1

---

## 1. Executive Summary

This report documents the evaluation, benchmarking, and selection of the Machine Learning model for the **Enterprise HR AI** platform.

Initially, the model achieved **100% Accuracy, Precision, Recall, and ROC-AUC**. An audit revealed that this was caused by a synthetic deterministic rule in the target label generation (`PerformanceRating <= 2 AND OvertimeHoursPerMonth >= 21`). 

To create an academically defensible, realistic engineering project, the target was regenerated using a **multi-factor probabilistic model** incorporating multiple HR dimensions and controlled stochastic noise. Three standard classifiers were benchmarked on a stratified 80/20 train/test split with built-in class balancing. **Logistic Regression** was selected as the champion model for its superior recall (72.2%), strong discrimination (ROC-AUC 0.7981), and transparent interpretability.

---

## 2. Dataset & Class Distribution

The attrition dataset contains **500 employee records** across 23 domain features.

| Class | Count | Percentage | Business Meaning |
|---|---|---|---|
| **No (0)** | 409 | 81.8% | Active employees who stayed |
| **Yes (1)** | 91 | 18.2% | Employees who left (Attrition) |
| **Total** | 500 | 100.0% | Moderate class imbalance (4.5 : 1) |

> **Imbalance Handling Strategy:**  
> Because the positive class is smaller (~18.2%), standard default models tend to favor the majority class. Rather than using complex synthetic oversampling (like SMOTE), we employ simple, built-in class weighting:
> - `class_weight='balanced'` for Logistic Regression and Random Forest.
> - `scale_pos_weight = N_neg / N_pos` for XGBoost.

---

## 3. Why the 100% Accuracy Occurred Initially vs. Probabilistic Fix

### The Initial Deterministic Problem
In the initial version, every employee with `PerformanceRating <= 2` AND `OvertimeHoursPerMonth >= 21` had `AttritionRisk = Yes`, and all others had `No`. 
- Tree-based models easily discovered this exact two-feature threshold rule.
- While mathematically correct on that specific synthetic file, **100% accuracy is unrealistic for human workforce behavior**.

### The Probabilistic Multi-Factor Solution
Real-world employee attrition is driven by multiple competing factors rather than a single hard-coded `IF` condition. The target was updated to use a weighted risk logit transformed via a sigmoid function:

$$\text{Risk Logit} = 0.25(\text{Overtime}) - 0.20(\text{Performance}) - 0.18(\text{WLB}) + 0.15(\text{PromotionGap}) - 0.10(\text{Satisfaction}) - 0.07(\text{Salary}) + 0.05(\text{Leaves}) + \epsilon$$

where $\epsilon \sim \mathcal{N}(0, 0.35)$ represents unobserved personal and market variance. This creates realistic probabilistic boundary overlap.

---

## 4. 3-Model Comparison & Benchmarking

All models were evaluated on the **same stratified 80/20 test split (100 holdout samples: 82 Active, 18 Attrition)** with identical sklearn preprocessing (StandardScaler + SimpleImputer + OneHotEncoder):

| Model | Class Imbalance Method | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | `class_weight='balanced'` | **0.6600** | **0.3095** | **0.7222** | **0.4333** | **0.7981** |
| **Random Forest** | `class_weight='balanced'` | 0.6800 | 0.2941 | 0.5556 | 0.3846 | 0.6809 |
| **XGBoost** | `scale_pos_weight=4.48` | 0.7100 | 0.2609 | 0.3333 | 0.2927 | 0.7154 |

---

## 5. Model Selection Rationale

### Champion Model: **Logistic Regression (`class_weight='balanced'`)**

1. **Highest Recall (72.22%):** In enterprise HR retention, **False Negatives are far more costly than False Positives**. Catching 13 out of 18 flight-risk employees allows HR teams to intervene early.
2. **Highest Discrimination (ROC-AUC = 0.7981):** Strong ranking capability across varying probability thresholds.
3. **Highest F1 Score (0.4333):** Best harmonic balance between Precision and Recall.
4. **Academically Defensible & Explainable:** Linear coefficients provide transparent, easily explainable risk factor weights during technical viva and stakeholder reviews.

---

## 6. Final Evaluation Metrics & Confusion Matrix

**Holdout Test Evaluation (100 samples, Decision Threshold = 0.5):**

```text
========================================
SELECTED MODEL: LOGISTIC REGRESSION (v1)
========================================
Accuracy:   0.6600 (66.0%)
Precision:  0.3095 (31.0%)
Recall:     0.7222 (72.2%)
F1 Score:   0.4333
ROC-AUC:    0.7981

5-Fold Cross-Validation:
  - F1 Mean:      0.4727 +/- 0.0752
  - ROC-AUC Mean: 0.7765 +/- 0.0482
```

### Confusion Matrix Breakdown

```
                       Predicted Active (0)    Predicted Attrition (1)
Actual Active (0):             53 (TN)                 29 (FP)
Actual Attrition (1):           5 (FN)                 13 (TP)
```

- **True Positives (13):** At-risk employees correctly flagged for retention outreach.
- **True Negatives (53):** Stable employees correctly identified.
- **False Positives (29):** Stable employees flagged for check-in (low business cost).
- **False Negatives (5):** Only 5 departures missed out of 18.

---

## 7. Metrics Explained for Project Viva / Presentation

- **Accuracy (66%):** Overall fraction of correct predictions. In imbalanced datasets, accuracy alone can be misleading; hence we focus on F1 and Recall.
- **Recall (72.2%):** $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{13}{13 + 5}$. Measures our ability to catch employees who are actually leaving.
- **Precision (31.0%):** $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{13}{13 + 29}$. When the model flags an employee, how often are they truly at risk.
- **F1 Score (0.4333):** The harmonic mean of Precision and Recall.
- **ROC-AUC (0.7981):** Area under the ROC curve measuring how well the model ranks positive instances above negative instances. A score close to 0.80 represents strong real-world predictive utility.

---

## 8. Real-World Limitations

1. **Synthetic Data Characteristics:** Synthetic datasets capture predefined statistical relationships but cannot fully capture complex real-world variables like managerial conflicts, family relocation, or macroeconomic job markets.
2. **Sample Size:** 500 records represents a mid-sized enterprise MVP. Production deployments benefit from multi-year historical logs.
3. **Threshold Tuning:** The 0.5 default probability cutoff prioritizes retention safety (high recall). In cost-constrained environments, the threshold can be adjusted to balance intervention budget.
