# Enterprise HR AI — Employee Skills Synthesis & Assumption Documentation

## 1. Context & Dataset Inspection
During Phase 1 (Data Understanding & Validation), all raw datasets were systematically inspected:
- `employee_attrition.csv`: Contains demographics, tenure, salary, satisfaction, and exit target.
- `hr_performance_engagement.csv`: Contains performance, ratings, hours, and attendance.
- `occupation_data.csv`, `essential_skills.csv`, `software_skills.csv`: Contains standard O*NET occupational taxonomies and required skills per SOC code.

None of the raw operational datasets contained a pre-existing granular table of individual employee skill inventories (e.g. employee X holds skills A, B, C).

## 2. Controlled MVP Synthesis Rationale
To enable the Skill Gap Engine (Step 13), Organization-Wide Skill Gap Intelligence (Step 14), and Personalized Recommendation Engine (Step 15), a deterministic, controlled MVP dataset was created: `data/processed/employee_skills.csv`.

## 3. Deterministic Rules Applied
1. **Real Employee IDs**: All 5,500 real employee IDs from the attrition and engagement datasets were utilized.
2. **Role-Grounded Skill Association**: Skills assigned to employees are derived strictly from the canonical O*NET role-skill matrix corresponding to their real assigned `job_role` and `department`. No random unrelated skills (e.g. assigning medical surgery skills to a software developer) were allowed.
3. **Deterministic Experience & Performance Weighting**: The number and proficiency of acquired skills correlate deterministically with employee tenure (`years_experience`) and `performance`, seeded reproducibly by `employee_id`.
4. **Reproducibility**: All assignments use explicit pseudo-random seeds (`np.random.RandomState(emp_id)`), ensuring exact byte-for-byte reproducibility across runs.
