"""
Streamlit Executive Dashboard for Enterprise HR AI — Workforce Intelligence & Upskilling Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path

# Setup page layout
st.set_page_config(
    page_title="AI Workforce Intelligence Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #2563EB;
        text-align: left;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.3rem;
    }
    .severity-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .severity-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .severity-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# API & Local Data Fallback Handler
API_BASE_URL = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

@st.cache_data(ttl=60)
def fetch_api_data(endpoint: str):
    """Fetches data from FastAPI backend with fallback to processed CSVs."""
    try:
        res = requests.get(f"{API_BASE_URL}{endpoint}", timeout=2.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def load_fallback_intelligence():
    """Direct disk fallback if API is offline."""
    path = PROCESSED_DIR / "employee_intelligence.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_fallback_org_gaps():
    path = PROCESSED_DIR / "organization_skill_gaps.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_fallback_recommendations():
    path = PROCESSED_DIR / "employee_recommendations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# Load primary intelligence dataset
df_intel = load_fallback_intelligence()
df_org_gaps = load_fallback_org_gaps()
df_recs = load_fallback_recommendations()

# Header
st.markdown('<div class="main-header">👥 AI WORKFORCE INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified Enterprise Analytics for Attrition Risk, Engagement Diagnostics, Skill Gap Optimization & Personalized Upskilling</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Navigation & Filters")

all_departments = ["All"] + sorted(df_intel['department'].unique().tolist()) if not df_intel.empty else ["All"]
selected_dept = st.sidebar.selectbox("Filter by Department", all_departments)

view_mode = st.sidebar.radio(
    "Select Intelligence View",
    ["📊 Executive Overview", "🎯 Skill Gap & Upskilling", "🔍 Employee 360° Drill Down", "⚡ Interactive Risk Simulator"]
)

# Apply Filter
df_filtered = df_intel.copy()
if selected_dept != "All" and not df_filtered.empty:
    df_filtered = df_filtered[df_filtered['department'] == selected_dept]

# -------------------------------------------------------------
# 1. EXECUTIVE OVERVIEW VIEW
# -------------------------------------------------------------
if view_mode == "📊 Executive Overview":
    st.subheader(f"Executive Metrics — {selected_dept if selected_dept != 'All' else 'Enterprise Wide'}")

    # Top KPI Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_emp = len(df_filtered)
    high_risk_count = int((df_filtered['attrition_risk_level'] == 'High').sum()) if not df_filtered.empty else 0
    avg_eng = round(df_filtered['engagement_score'].mean(), 1) if not df_filtered.empty else 0.0
    avg_read = round(df_filtered['readiness_score'].mean(), 1) if not df_filtered.empty else 0.0

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2563EB;">
            <div class="kpi-title">Total Workforce</div>
            <div class="kpi-value">{total_emp:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #DC2626;">
            <div class="kpi-title">High Flight Risk</div>
            <div class="kpi-value">{high_risk_count:,} <span style="font-size: 0.9rem; color: #DC2626;">({round(high_risk_count/max(1, total_emp)*100, 1)}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #059669;">
            <div class="kpi-title">Average Engagement</div>
            <div class="kpi-value">{avg_eng} <span style="font-size: 0.9rem; color: #059669;">/ 100</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #D97706;">
            <div class="kpi-title">Average Skill Readiness</div>
            <div class="kpi-value">{avg_read}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 📌 Attrition Risk Distribution")
        if not df_filtered.empty:
            risk_counts = df_filtered['attrition_risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Tier', 'Count']
            color_map = {'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
            fig_risk = px.pie(
                risk_counts,
                values='Count',
                names='Risk Tier',
                color='Risk Tier',
                color_discrete_map=color_map,
                hole=0.45
            )
            fig_risk.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        st.markdown("##### 🏢 Departmental Attrition Risk Profile")
        if not df_intel.empty:
            dept_risk = df_intel.groupby(['department', 'attrition_risk_level']).size().reset_index(name='count')
            fig_dept = px.bar(
                dept_risk,
                x='department',
                y='count',
                color='attrition_risk_level',
                color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'},
                barmode='stack',
                labels={'count': 'Headcount', 'department': 'Department', 'attrition_risk_level': 'Risk Tier'}
            )
            fig_dept.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_dept, use_container_width=True)

    # Charts Row 2: Engagement vs Skill Readiness Scatter
    st.markdown("##### 🎯 Engagement vs. Capability Readiness Scatter")
    if not df_filtered.empty:
        fig_scatter = px.scatter(
            df_filtered.sample(min(400, len(df_filtered)), random_state=42),
            x='engagement_score',
            y='readiness_score',
            color='attrition_risk_level',
            color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'},
            hover_data=['name', 'job_role', 'department'],
            labels={'engagement_score': 'Engagement Score (0-100)', 'readiness_score': 'Role Readiness Score (%)', 'attrition_risk_level': 'Attrition Risk'}
        )
        fig_scatter.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# 2. SKILL GAP & UPSKILLING VIEW
# -------------------------------------------------------------
elif view_mode == "🎯 Skill Gap & Upskilling":
    st.subheader("Organization-Wide Skill Gap & Upskilling Intelligence")

    col_sg1, col_sg2 = st.columns([3, 2])

    with col_sg1:
        st.markdown("##### 📊 Top Enterprise Skill Deficiencies")
        if not df_org_gaps.empty:
            top_gaps = df_org_gaps.head(12)
            fig_bar = px.bar(
                top_gaps,
                x='employees_missing',
                y='missing_skill',
                orientation='h',
                color='severity',
                color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'},
                labels={'employees_missing': 'Employees Lacking Competency', 'missing_skill': 'Missing Skill', 'severity': 'Criticality Severity'}
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=420, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_sg2:
        st.markdown("##### 🚨 Critical Skill Severity Breakdown")
        if not df_org_gaps.empty:
            st.dataframe(
                df_org_gaps[['missing_skill', 'employees_missing', 'percentage_missing', 'severity']],
                use_container_width=True,
                height=420
            )

    st.markdown("---")
    st.markdown("##### 📚 High-Impact Targeted Recommendations")
    if not df_recs.empty:
        search_skill = st.text_input("Filter Recommendations by Skill / Role", "")
        recs_view = df_recs.copy()
        if search_skill:
            recs_view = recs_view[recs_view['missing_skill'].str.contains(search_skill, case=False, na=False)]

        st.dataframe(
            recs_view[['employee_id', 'missing_skill', 'recommended_course', 'recommended_certification', 'recommended_project', 'priority']].head(50),
            use_container_width=True,
            height=320
        )

# -------------------------------------------------------------
# 3. EMPLOYEE 360° DRILL DOWN VIEW
# -------------------------------------------------------------
elif view_mode == "🔍 Employee 360° Drill Down":
    st.subheader("Employee 360-Degree Workforce Intelligence Dossier")

    if not df_intel.empty:
        emp_options = df_intel['employee_id'].astype(str) + " - " + df_intel['name'] + " (" + df_intel['job_role'] + ")"
        selected_emp_str = st.selectbox("Search & Select Employee Dossier", emp_options)
        selected_id = int(selected_emp_str.split(" - ")[0])

        emp_record = df_intel[df_intel['employee_id'] == selected_id].iloc[0]

        st.markdown("<br>", unsafe_allow_html=True)
        col_prof1, col_prof2, col_prof3 = st.columns([1, 1.2, 1.5])

        with col_prof1:
            st.markdown(f"### {emp_record['name']}")
            st.markdown(f"**Employee ID:** `{emp_record['employee_id']}`")
            st.markdown(f"**Department:** `{emp_record['department']}`")
            st.markdown(f"**Role:** `{emp_record['job_role']}`")
            if pd.notnull(emp_record.get('age')):
                st.markdown(f"**Age:** {int(emp_record['age'])}")
            if pd.notnull(emp_record.get('years_experience')):
                st.markdown(f"**Tenure:** {int(emp_record['years_experience'])} years")

        with col_prof2:
            st.markdown("#### Risk & Engagement")
            risk_color = "#EF4444" if emp_record['attrition_risk_level'] == 'High' else ("#F59E0B" if emp_record['attrition_risk_level'] == 'Medium' else "#10B981")
            st.markdown(f"**Attrition Risk:** <span style='color:{risk_color}; font-weight:700; font-size:1.2rem;'>{emp_record['attrition_risk_level']} ({emp_record['attrition_probability']*100:.1f}%)</span>", unsafe_allow_html=True)
            st.progress(float(emp_record['attrition_probability']))

            st.markdown(f"**Engagement Score:** `{emp_record['engagement_score']} / 100`")
            st.progress(float(emp_record['engagement_score']) / 100.0)

            st.markdown(f"**Role Readiness Score:** `{emp_record['readiness_score']}%`")
            st.progress(float(emp_record['readiness_score']) / 100.0)

        with col_prof3:
            st.markdown("#### Skills Portfolio & Gaps")
            st.markdown(f"**✅ Current Matched Skills:**")
            st.info(emp_record['matched_skills'])

            st.markdown(f"**⚠️ Missing Role Competencies:**")
            st.warning(emp_record['missing_skills'])

        st.markdown("---")
        st.markdown("#### 🎯 Personalized Actionable Upskilling Pathways")
        emp_recs = df_recs[df_recs['employee_id'] == selected_id]
        if not emp_recs.empty:
            for idx, r in emp_recs.iterrows():
                st.markdown(f"""
                - **Skill Gap: `{r['missing_skill']}`** ({r['priority']} Priority)
                  - 📖 **Course:** {r['recommended_course']}
                  - 📜 **Certification:** {r['recommended_certification']}
                  - 🛠️ **Project:** {r['recommended_project']}
                """)
        else:
            st.success("🎉 Employee has zero missing role competencies. Ready for role expansion or promotion!")

# -------------------------------------------------------------
# 4. INTERACTIVE RISK SIMULATOR
# -------------------------------------------------------------
elif view_mode == "⚡ Interactive Risk Simulator":
    st.subheader("Real-Time Attrition Risk Simulation Engine")
    st.markdown("Adjust employee compensation, work-life metrics, and tenure parameters to evaluate real-time ML flight risk probability.")

    with st.form("risk_simulator_form"):
        sim_c1, sim_c2, sim_c3 = st.columns(3)

        with sim_c1:
            sim_age = st.number_input("Employee Age", min_value=18, max_value=80, value=30)
            sim_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            sim_dept = st.selectbox("Department", ["IT", "Sales", "Finance", "HR", "Marketing", "Support"])
            sim_role = st.selectbox("Job Role", [
                "Developer", "Software Engineer", "Sales Executive", "Accountant", "Auditor",
                "Financial Analyst", "HR Manager", "HR Executive", "Content Lead", "SEO Analyst",
                "Support Engineer", "Tester", "Helpdesk", "Cybersecurity Specialist", "Data Analyst"
            ])
            sim_salary = st.number_input("Monthly Salary ($)", min_value=10000, max_value=200000, value=65000, step=1000)

        with sim_c2:
            sim_overtime = st.slider("Overtime Hours / Month", 0, 80, 15)
            sim_leaves = st.slider("Annual Leaves Taken", 0, 30, 8)
            sim_projects = st.slider("Projects Handled", 1, 30, 8)
            sim_training = st.slider("Training Hours", 0, 100, 25)
            sim_sat = st.slider("Satisfaction Score (1-10)", 1.0, 10.0, 7.0, 0.5)

        with sim_c3:
            sim_wlb = st.slider("Work-Life Balance Score (1-5)", 1.0, 5.0, 3.2, 0.1)
            sim_perf = st.slider("Performance Rating (1-5)", 1, 5, 3)
            sim_tenure = st.number_input("Years at Company", min_value=0, max_value=40, value=4)
            sim_last_promo = st.number_input("Last Promotion Year", min_value=1995, max_value=2026, value=2022)
            sim_country = st.selectbox("Country", ["USA", "India", "Canada", "UK", "Germany", "France"])

        submit_btn = st.form_submit_button("🚀 Run Live ML Inference")

    if submit_btn:
        payload = {
            "age": int(sim_age),
            "gender": sim_gender,
            "department": sim_dept,
            "job_role": sim_role,
            "education_level": 2,
            "monthly_salary": float(sim_salary),
            "overtime_hours_per_month": int(sim_overtime),
            "leaves_taken": int(sim_leaves),
            "projects_handled": int(sim_projects),
            "training_hours": int(sim_training),
            "customer_satisfaction": float(sim_sat),
            "last_promotion_year": int(sim_last_promo),
            "years_at_company": int(sim_tenure),
            "work_life_balance_score": float(sim_wlb),
            "performance_rating": int(sim_perf),
            "country": sim_country,
            "leave_day_name": "Friday"
        }

        # Try API or local predictor
        res_data = None
        try:
            r = requests.post(f"{API_BASE_URL}/predict/attrition", json=payload, timeout=2.0)
            if r.status_code == 200:
                res_data = r.json()
        except Exception:
            pass

        if not res_data:
            from app.ml.predictor import predict_attrition
            pred_obj = predict_attrition(payload)
            res_data = pred_obj.model_dump()

        st.markdown("### 🎯 Inference Results")
        res_c1, res_c2 = st.columns([1, 2])

        proba = res_data['attrition_probability']
        risk_lvl = res_data['attrition_risk_level']

        with res_c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': f"Flight Risk: {risk_lvl}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if risk_lvl == 'High' else ("#F59E0B" if risk_lvl == 'Medium' else "#10B981")},
                    'steps': [
                        {'range': [0, 30], 'color': "#D1FAE5"},
                        {'range': [30, 70], 'color': "#FEF3C7"},
                        {'range': [70, 100], 'color': "#FEE2E2"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_c2:
            st.markdown(f"**Risk Level:** `{risk_lvl}`")
            st.markdown(f"**Model Probability:** `{proba:.4f}`")
            st.info(res_data['risk_interpretation'])
            st.caption(f"Model version: `{res_data['model_version']}` | Timestamp: `{res_data['timestamp']}`")
