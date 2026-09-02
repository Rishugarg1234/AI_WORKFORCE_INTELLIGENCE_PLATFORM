# Enterprise HR AI — Feature Engineering Documentation

This document describes all base and domain-engineered features used for predicting employee attrition risk.

## Target Variable
- **Column**: `AttritionRisk`
- **Encoding**: `Yes` -> 1 (Attrition Risk), `No` -> 0 (Active/Low Risk)
- **Positive Class Prevalence**: ~11.0% (55 positive cases out of 500 records)

## Dropped Identifier / Leakage Columns
- `EmployeeID`: Unique employee identifier (prevent overfitting)
- `Name`: Free-text name (high cardinality, PII)
- `PhoneNumber`: Contact identifier
- `JoiningDate`: Date string (tenure captured via `YearsAtCompany`)
- `LastLeaveDate`: Date string (leave pattern captured via `LeavesTaken` and `LeaveDayName`)
- `CountryCode`: Redundant with `Country`
- `AttritionRisk`: Excluded to prevent direct target leakage

## Domain Engineered Features

| Feature Name | Formulation | Business Rationale |
|---|---|---|
| `income_per_year_at_company` | `MonthlySalary * 12 / (YearsAtCompany + 1)` | Measures annual compensation velocity relative to tenure. Stagnant earnings over long tenure elevate departure risk. |
| `promotion_gap_ratio` | `(2026 - LastPromotionYear) / (YearsAtCompany + 1)` | Ratio of years without promotion relative to company tenure. High values flag career stagnation. |
| `overtime_ratio` | `OvertimeHoursPerMonth / 160.0` | Overtime burden relative to standard full-time capacity. High overtime induces burnout. |
| `leave_utilization` | `LeavesTaken / 20.0` | Proportion of annual leave allowance utilized. Unused leaves or extreme leave spikes indicate disengagement or stress. |
| `work_life_satisfaction` | `WorkLifeBalanceScore * CustomerSatisfaction` | Interaction term measuring composite workplace well-being. |

## Preprocessing Pipeline Architecture
- **Numerical Pipeline**: Median Imputation -> Standard Scaling
- **Categorical Pipeline**: Most Frequent Imputation -> One-Hot Encoding (`handle_unknown='ignore'`)
- **Total Input Features**: 23 (Transformed matrix width: 53)
