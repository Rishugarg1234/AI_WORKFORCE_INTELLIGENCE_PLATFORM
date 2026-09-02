# Enterprise HR AI — Data Relationships and Schema Integration

This document defines the relationships, keys, overlap statistics, and integration architecture across all datasets in the platform.

## Relationship Matrix

| Dataset A | Dataset B | Join Key | Relationship Type | Overlap Evidence | Purpose | Confidence |
|---|---|---|---|---|---|---|
| `employee_attrition_processed.csv` | `engagement_processed.csv` | `EmployeeID <-> employee_id` | None (Disjoint Cohorts) | 0 matching IDs (0% overlap, 1-500 vs 100021-999957) | Attrition study cohort vs Organization-wide engagement monitoring cohort | Definite (Disjoint ID spaces) |
| `employee_attrition_processed.csv` | `engagement_processed.csv` | `Department, JobRole <-> department, job_role` | Many-to-Many (Categorical Taxonomy) | 5 common departments ({'HR', 'Marketing', 'Sales', 'Finance', 'IT'}), 5 exact common roles ({'HR Manager', 'Account Manager', 'Sales Executive', 'Auditor', 'Accountant'}) | Cross-cohort departmental benchmarking and role modeling | High |
| `occupation_master.csv` | `essential_skills_processed.csv` | `O*NET-SOC Code` | One-to-Many | 910 / 1016 occupations matched (89.6%) | Map baseline core soft/cognitive competencies to standardized occupations | High |
| `occupation_master.csv` | `software_skills_processed.csv` | `O*NET-SOC Code` | One-to-Many | 923 / 1016 occupations matched (90.8%) | Map technical software/tools requirements to standardized occupations | High |
| `essential_skills_processed.csv` | `software_skills_processed.csv` | `O*NET-SOC Code` | Many-to-Many | 910 shared O*NET codes | Provide dual-dimension skill matrix (essential + technical tools) per occupation | High |

## Key Architectural Decisions

1. **Employee ID Disjoint Cohorts**:
   - `employee_attrition_processed.csv` holds Employee IDs `1` to `500`.
   - `engagement_processed.csv` holds 5,000 distinct employees with 6-digit IDs (`100021` to `999957`).
   - These are separate employee datasets: one represents an attrition risk analysis study cohort with rich compensation, tenure, and exit signals; the other represents enterprise-wide performance and engagement tracking.
   - For unified employee intelligence, they are modeled harmoniously with consistent schema standards and role/department taxonomies.

2. **O*NET SOC Skill Taxonomy**:
   - `occupation_master.csv` serves as the canonical role dictionary (1,016 occupations).
   - `essential_skills_processed.csv` maps each occupation to 10 core competencies (Importance and Level ratings).
   - `software_skills_processed.csv` maps each occupation to specialized technical software, flagged with `Hot Technology` and `In Demand`.
