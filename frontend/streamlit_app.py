import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="HEARTCARE AI | Clinical Risk Assessment",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --hc-bg: #0a1018;
    --hc-border: rgba(148,163,184,.16);
    --hc-text: #f5f7fa;
    --hc-muted: #9aa9bb;
    --hc-blue: #38bdf8;
    --hc-teal: #2dd4bf;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(circle at 8% 0%, rgba(56,189,248,.07), transparent 28%),
                radial-gradient(circle at 92% 10%, rgba(45,212,191,.055), transparent 25%),
                var(--hc-bg);
    color: var(--hc-text);
}
[data-testid="stAppViewContainer"] { background: transparent; }
.main .block-container { max-width: 1450px; padding: 2rem 3rem 3.5rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1723 0%, #0a111a 100%);
    border-right: 1px solid var(--hc-border);
}
section[data-testid="stSidebar"] > div { padding-top: 1.35rem; }
.hc-nav-label{color:#64748b;font-size:.67rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin:1.1rem 0 .55rem}
section[data-testid="stSidebar"] .stButton>button{width:100%!important;text-align:left!important;justify-content:flex-start!important;min-height:42px!important;padding:.55rem .72rem!important;margin:.15rem 0!important;border-radius:11px!important;background:transparent!important;color:#9aa9bb!important;border:1px solid transparent!important;box-shadow:none!important;font-weight:650!important}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(56,189,248,.07)!important;border-color:rgba(56,189,248,.16)!important;transform:translateX(2px);color:#e8f7ff!important}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:linear-gradient(90deg,rgba(14,165,233,.16),rgba(45,212,191,.08))!important;color:#f5f7fa!important;border-color:rgba(56,189,248,.22)!important;box-shadow:inset 3px 0 0 #22d3ee,0 8px 22px rgba(14,165,233,.08)!important}
.hc-sidebar-status{margin-top:1rem;padding:.8rem .85rem;border:1px solid rgba(148,163,184,.13);border-radius:14px;background:rgba(15,25,38,.62)}
.hc-sidebar-status .dot{color:#4ade80;margin-right:.35rem}.hc-sidebar-status .row{color:#9aa9bb;font-size:.72rem;line-height:1.85}
.hc-brand-sub{color:#64748b;font-size:.66rem;line-height:1.45;letter-spacing:.09em;text-transform:uppercase}
.hc-stepper{display:flex;align-items:center;gap:.55rem;margin:.4rem 0 1rem}.hc-step{flex:1;text-align:center;padding:.65rem .4rem;border:1px solid rgba(148,163,184,.13);border-radius:12px;background:rgba(17,27,40,.74)}.hc-step .n{color:#38bdf8;font-size:.66rem;font-weight:800;letter-spacing:.08em}.hc-step .t{color:#d7e2ec;font-size:.76rem;font-weight:650;margin-top:.15rem}.hc-step-arrow{color:#475569;font-size:.8rem}
.hc-result{border-radius:22px;border:1px solid rgba(56,189,248,.2);background:radial-gradient(circle at 85% 0%,rgba(45,212,191,.11),transparent 28%),linear-gradient(145deg,#111d2b,#0c1520);padding:1.35rem 1.5rem;box-shadow:0 20px 46px rgba(0,0,0,.22);margin-top:.9rem}
.hc-result-label{color:#67e8f9;font-size:.7rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.hc-result-risk{font-size:2.7rem;line-height:1.05;font-weight:850;margin:.42rem 0 .2rem}.hc-result-number{font-size:2.15rem;font-weight:800;color:#fff}
.hc-meter{position:relative;height:10px;border-radius:999px;background:linear-gradient(90deg,#22c55e 0 40%,#f59e0b 40% 70%,#ef4444 70% 100%);margin:1rem 0 .35rem}.hc-meter-marker{position:absolute;top:50%;width:16px;height:16px;border-radius:50%;border:3px solid #071018;background:#fff;transform:translate(-50%,-50%);box-shadow:0 0 0 2px rgba(255,255,255,.45),0 0 18px rgba(255,255,255,.28)}.hc-meter-labels{display:flex;justify-content:space-between;color:#64748b;font-size:.67rem;font-weight:700;letter-spacing:.08em}
.hc-factor{padding:.62rem .72rem;border:1px solid rgba(148,163,184,.11);border-radius:11px;background:rgba(15,25,38,.55);margin:.35rem 0}.hc-factor-name{color:#e2e8f0;font-weight:700;font-size:.78rem}.hc-factor-val{color:#94a3b8;font-size:.72rem;margin-top:.15rem}
.hc-footer{border-top:1px solid var(--hc-border);margin-top:2rem;padding:1.15rem 0 .25rem;color:#64748b;font-size:.7rem;text-align:center}
h1,h2,h3 { color: var(--hc-text) !important; letter-spacing: -.02em; }
h1 { font-size: clamp(2.1rem, 4vw, 3rem) !important; font-weight: 700 !important; }
h2 { font-size: 1.55rem !important; font-weight: 650 !important; }
h3 { font-size: 1.05rem !important; font-weight: 650 !important; }
.stCaption,[data-testid="stCaptionContainer"] { color: var(--hc-muted) !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(21,34,51,.97), rgba(14,25,38,.97));
    border: 1px solid var(--hc-border); border-radius: 16px; padding: 1.05rem 1.15rem;
    min-height: 104px; box-shadow: 0 12px 30px rgba(0,0,0,.18); transition: .18s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(56,189,248,.35); }
div[data-testid="stMetricLabel"] { color: var(--hc-muted) !important; font-size: .81rem !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: var(--hc-text) !important; font-size: 1.72rem !important; font-weight: 700 !important; }
.stButton > button,.stDownloadButton > button {
    border-radius: 12px !important; min-height: 46px; font-weight: 700 !important;
    border: 1px solid rgba(56,189,248,.28) !important; transition: .18s ease !important;
}
.stButton > button[kind="primary"],.stDownloadButton > button {
    background: linear-gradient(135deg,#0284c7 0%,#0f766e 100%) !important; color: white !important;
}
.stButton > button:hover,.stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(14,116,144,.22); }
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #0f1a27 !important; border: 1px solid var(--hc-border) !important; border-radius: 11px !important; color: var(--hc-text) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(145deg, rgba(17,27,40,.96), rgba(12,20,30,.96));
    border: 1px dashed rgba(56,189,248,.38) !important; border-radius: 16px !important;
}
div[data-testid="stPlotlyChart"], div[data-testid="stPyplot"] {
    background: linear-gradient(145deg, rgba(17,27,40,.82), rgba(12,20,31,.88));
    border: 1px solid var(--hc-border); border-radius: 18px; padding: .35rem; box-shadow: 0 14px 34px rgba(0,0,0,.12);
}
div[data-testid="stDataFrame"] { border: 1px solid var(--hc-border); border-radius: 14px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 13px !important; border: 1px solid var(--hc-border) !important; }
[data-testid="stExpander"] { background: rgba(17,27,40,.62); border: 1px solid var(--hc-border); border-radius: 14px; }
hr { border-color: var(--hc-border) !important; margin: 1.3rem 0 !important; }
.hc-hero {
    border: 1px solid rgba(56,189,248,.16);
    background: radial-gradient(circle at 80% 0%, rgba(45,212,191,.12), transparent 24%), linear-gradient(135deg,rgba(17,27,40,.98),rgba(12,21,33,.98));
    border-radius: 22px; padding: 1.45rem 1.6rem; margin-bottom: 1.15rem; box-shadow: 0 22px 50px rgba(0,0,0,.20);
}
.hc-kicker { color: var(--hc-teal); font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
.hc-subtitle { color: var(--hc-muted); font-size: .98rem; line-height: 1.55; margin-top: .45rem; }
.hc-status { display:inline-block; margin-top:.72rem; padding:.36rem .68rem; border-radius:999px; border:1px solid rgba(52,211,153,.24); background:rgba(52,211,153,.07); color:#86efac; font-size:.7rem; font-weight:800; letter-spacing:.08em; }
.hc-card { background:linear-gradient(145deg,rgba(17,27,40,.98),rgba(12,21,32,.98)); border:1px solid var(--hc-border); border-radius:18px; padding:1.15rem 1.25rem; box-shadow:0 16px 38px rgba(0,0,0,.15); }
.hc-section { color:var(--hc-teal); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin:.15rem 0 .5rem; }
.hc-disclaimer { margin-top:.85rem; padding:.85rem 1rem; border-radius:14px; background:rgba(251,191,36,.055); border:1px solid rgba(251,191,36,.20); color:#f7d98a; font-size:.82rem; line-height:1.5; }

</style>
""", unsafe_allow_html=True)

def hc_header(kicker, title, subtitle, status=None):
    status_html = f'<div class="hc-status">● {status}</div>' if status else ''
    st.markdown(
        f'<div class="hc-hero"><div class="hc-kicker">{kicker}</div>'
        f'<div style="font-size:2.3rem;font-weight:750;line-height:1.16;color:#f5f7fa;">{title}</div>'
        f'<div class="hc-subtitle">{subtitle}</div>{status_html}</div>',
        unsafe_allow_html=True
    )

def hc_disclaimer(text):
    st.markdown(f'<div class="hc-disclaimer"><b>Clinical use notice</b><br>{text}</div>', unsafe_allow_html=True)

def style_plotly(fig, height=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe5ee"), margin=dict(l=24,r=24,t=56,b=28)
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def render_nav():
    pages = [
        ("⌂", "Overview", "🏠 Dashboard"),
        ("◉", "Risk Assessment", "👤 Individual Prediction"),
        ("✦", "AI Explainability", "🔍 Explainable Prediction"),
        ("◫", "Population Analytics", "📊 Population Analytics"),
        ("▣", "Model Performance", "📈 Model Evaluation"),
        ("⇧", "Batch Assessment", "📁 Batch Prediction"),
    ]
    labels = [x[2] for x in pages]
    if "page" not in st.session_state:
        st.session_state.page = labels[0]
    current = st.session_state.page

    st.sidebar.markdown("""
    <div style="padding:0 0 .35rem;">
      <div style="display:flex;align-items:center;gap:.65rem;">
        <div style="width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0ea5e9,#0f766e);color:#fff;font-size:22px;">♥</div>
        <div>
          <div style="color:#f8fafc;font-size:1.08rem;font-weight:850;">HEARTCARE AI</div>
          <div class="hc-brand-sub">Clinical Intelligence Platform</div>
        </div>
      </div>
      <div style="margin-top:.65rem;color:#64748b;font-size:.61rem;font-weight:800;letter-spacing:.12em;">AI-POWERED CARDIOVASCULAR RISK ASSESSMENT</div>
      <div class="hc-status" style="margin-top:.6rem;">● SYSTEM ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="hc-nav-label">Workspace</div>', unsafe_allow_html=True)
    for icon, title, stored in pages:
        if st.sidebar.button(f"{icon}  {title}", key=f"nav_{title}", type="primary" if current == stored else "secondary"):
            st.session_state.page = stored
            st.rerun()

    st.sidebar.markdown("""
    <div class="hc-sidebar-status">
      <div style="color:#dbeafe;font-size:.72rem;font-weight:800;margin-bottom:.25rem;letter-spacing:.08em;">SYSTEM STATUS</div>
      <div class="row"><span class="dot">●</span>API Connected</div>
      <div class="row"><span class="dot">●</span>ML Model Active</div>
      <div class="row"><span class="dot">●</span>SHAP Enabled</div>
    </div>
    <div style="margin-top:.75rem;color:#64748b;font-size:.68rem;line-height:1.5;">Educational clinical decision support.<br>Not a medical diagnosis.</div>
    """, unsafe_allow_html=True)
    return current


def render_footer():
    st.markdown("""
    <div class="hc-footer">
      <div style="color:#94a3b8;font-weight:750;">HEARTCARE AI · Intelligent Healthcare Disease Risk Prediction</div>
      <div style="margin-top:.2rem;">Educational Clinical Decision Support System · Model: Logistic Regression · Explainability: SHAP</div>
      <div style="margin-top:.35rem;color:#a78bfa;">⚠ Not a medical diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)


def risk_meter(probability):
    pct = max(0.0, min(1.0, float(probability)))
    return f'<div class="hc-meter"><div class="hc-meter-marker" style="left:{pct*100:.2f}%;"></div></div><div class="hc-meter-labels"><span>LOW</span><span>MEDIUM</span><span>HIGH</span></div>'


API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# Session state
# -----------------------------

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/framingham_stage2_engineered.csv"

try:
    df = pd.read_csv(DATA_PATH)
except Exception:
    df = None


# -----------------------------
# Helper functions
# -----------------------------

def get_bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obese"


def get_age_group(age):

    if age < 30:
        return "Young"

    elif age < 50:
        return "Middle Age"

    else:
        return "Senior"


def calculate_risk_factor_count(
    current_smoker,
    prevalent_hyp,
    diabetes,
    systolic_bp,
    diastolic_bp,
    total_chol,
    bmi
):

    count = 0

    if systolic_bp >= 140 or diastolic_bp >= 90 or prevalent_hyp == 1:
        count += 1

    if total_chol >= 240:
        count += 1

    if current_smoker == 1:
        count += 1

    if diabetes == 1:
        count += 1

    if bmi >= 30:
        count += 1

    return count


# -----------------------------
# Professional sidebar navigation
# -----------------------------

page = render_nav()


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    hc_header(
        "HEARTCARE AI",
        "Intelligent Heart Disease Risk Assessment",
        "AI-powered estimation of 10-year coronary heart disease risk using clinical and lifestyle factors.",
        "AI MODEL ACTIVE"
    )


    # -----------------------------
    # Dataset statistics
    # -----------------------------

    if df is not None:

        total_records = len(df)

        positive_cases = int(
            df["TenYearCHD"].sum()
        )

        negative_cases = total_records - positive_cases

        positive_percentage = (
            positive_cases / total_records * 100
        )

        # -----------------------------
        # Metrics
        # -----------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Records",
                f"{total_records:,}"
            )

        with col2:
            st.metric(
                "Heart Disease Cases",
                f"{positive_cases:,}"
            )

        with col3:
            st.metric(
                "Negative Cases",
                f"{negative_cases:,}"
            )

        with col4:
            st.metric(
                "Positive Rate",
                f"{positive_percentage:.2f}%"
            )

        st.divider()

        # -----------------------------
        # Dataset distribution
        # -----------------------------

        st.markdown('<div class="hc-section">Outcome overview</div><h3 style="margin-top:0;">Disease Distribution</h3>', unsafe_allow_html=True)

        distribution_df = pd.DataFrame({
            "Outcome": [
                "No Heart Disease",
                "Heart Disease"
            ],
            "Count": [
                negative_cases,
                positive_cases
            ]
        })

        fig = px.pie(
            distribution_df,
            names="Outcome",
            values="Count",
            title="10-Year CHD Distribution"
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Dataset could not be loaded."
        )


    # -----------------------------
    # Model information
    # -----------------------------

    st.divider()

    st.markdown('<div class="hc-section">How HEARTCARE AI works</div><h3 style="margin-top:0;">From patient data to explainable insight</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hc-stepper">'
        '<div class="hc-step"><div class="n">01</div><div class="t">Patient Data</div></div>'
        '<div class="hc-step-arrow">→</div>'
        '<div class="hc-step"><div class="n">02</div><div class="t">Risk Analysis</div></div>'
        '<div class="hc-step-arrow">→</div>'
        '<div class="hc-step"><div class="n">03</div><div class="t">AI Prediction</div></div>'
        '<div class="hc-step-arrow">→</div>'
        '<div class="hc-step"><div class="n">04</div><div class="t">Explainable Result</div></div>'
        '</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="hc-section">Model overview</div><h3 style="margin-top:0;">Clinical AI Stack</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**Final Model**\n\n"
            "Logistic Regression"
        )

    with col2:

        st.info(
            "**Disease**\n\n"
            "10-Year Coronary Heart Disease"
        )

    with col3:

        st.info(
            "**Explainability**\n\n"
            "SHAP"
        )


    st.divider()

    st.markdown('<div class="hc-section">System purpose</div><h3 style="margin-top:0;">Clinical Decision Support</h3>', unsafe_allow_html=True)

    st.write(
        "The system uses patient clinical and lifestyle "
        "parameters to estimate 10-year coronary heart "
        "disease risk and provide an explainable prediction."
    )

    st.warning(
        "This application is intended for educational "
        "and decision-support purposes only and is not "
        "a medical diagnosis."
    )


# =========================================================
# INDIVIDUAL PREDICTION
# =========================================================

elif page == "👤 Individual Prediction":

    hc_header("RISK ASSESSMENT", "Individual Risk Assessment", "Enter patient parameters to estimate the 10-year coronary heart disease risk.")


    st.markdown(
        '<div class="hc-stepper">'
        '<div class="hc-step"><div class="n">01</div><div class="t">Patient Profile</div></div>'
        '<div class="hc-step-arrow">→</div>'
        '<div class="hc-step"><div class="n">02</div><div class="t">Lifestyle</div></div>'
        '<div class="hc-step-arrow">→</div>'
        '<div class="hc-step"><div class="n">03</div><div class="t">Clinical Data</div></div>'
        '<div class="hc-step-arrow">→</div>'
        '<div class="hc-step"><div class="n">04</div><div class="t">Assessment</div></div>'
        '</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="hc-section">Clinical parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        male = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50
        )

        education = st.selectbox(
            "Education Level",
            [1, 2, 3, 4],
            index=1
        )

        current_smoker = st.selectbox(
            "Current Smoker",
            ["No", "Yes"]
        )

        cigs_per_day = st.number_input(
            "Cigarettes Per Day",
            min_value=0,
            max_value=100,
            value=0
        )

        bpm_eds = st.selectbox(
            "Blood Pressure Medication",
            ["No", "Yes"]
        )

        prevalent_stroke = st.selectbox(
            "Previous Stroke",
            ["No", "Yes"]
        )

    with col2:

        prevalent_hyp = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )

        total_chol = st.number_input(
            "Total Cholesterol",
            min_value=100,
            max_value=500,
            value=200
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure",
            min_value=80,
            max_value=250,
            value=120
        )

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure",
            min_value=40,
            max_value=150,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

        heart_rate = st.number_input(
            "Heart Rate",
            min_value=40,
            max_value=200,
            value=75
        )

        glucose = st.number_input(
            "Glucose",
            min_value=40,
            max_value=400,
            value=100
        )


    # -----------------------------
    # Convert values
    # -----------------------------

    male_value = 1 if male == "Male" else 0
    smoker_value = 1 if current_smoker == "Yes" else 0
    bpmeds_value = 1 if bpm_eds == "Yes" else 0
    stroke_value = 1 if prevalent_stroke == "Yes" else 0
    hyp_value = 1 if prevalent_hyp == "Yes" else 0
    diabetes_value = 1 if diabetes == "Yes" else 0


    # -----------------------------
    # Engineered features
    # -----------------------------

    bmi_category = get_bmi_category(bmi)

    age_group = get_age_group(age)

    risk_factor_count = calculate_risk_factor_count(
        smoker_value,
        hyp_value,
        diabetes_value,
        systolic_bp,
        diastolic_bp,
        total_chol,
        bmi
    )


    with st.expander("🔧 Automatically Calculated Features"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(
                "**BMI Category:**",
                bmi_category
            )

        with col2:
            st.write(
                "**Age Group:**",
                age_group
            )

        with col3:
            st.write(
                "**Risk Factor Count:**",
                risk_factor_count
            )


    # -----------------------------
    # Prediction
    # -----------------------------

    st.divider()

    if st.button(
        "Assess 10-Year CHD Risk",
        type="primary",
        use_container_width=True
    ):

        patient_data = {

            "male": male_value,
            "age": age,
            "education": education,
            "currentSmoker": smoker_value,
            "cigsPerDay": cigs_per_day,
            "BPMeds": bpmeds_value,
            "prevalentStroke": stroke_value,
            "prevalentHyp": hyp_value,
            "diabetes": diabetes_value,
            "totChol": total_chol,
            "sysBP": systolic_bp,
            "diaBP": diastolic_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": glucose,

            "BMI_Category": bmi_category,
            "Age_Group": age_group,
            "Risk_Factor_Count": risk_factor_count
        }

        try:

            response = requests.post(
                API_URL,
                json=patient_data,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

# Store latest prediction for Explainable Prediction page
                st.session_state.prediction_result = result

                prediction = result["prediction"]
                probability = result["probability"]
                risk_level = result["risk_level"]
                top_factors = result["top_factors"]


                st.success(
                    "Prediction completed successfully!"
                )


                risk_text = "HIGH RISK" if risk_level == "HIGH" else ("MEDIUM RISK" if risk_level == "MEDIUM" else "LOW RISK")
                risk_color = "#f87171" if risk_level == "HIGH" else ("#fbbf24" if risk_level == "MEDIUM" else "#4ade80")
                st.markdown(
                    f"""
                    <div class="hc-result">
                        <div class="hc-result-label">Risk assessment complete</div>
                        <div class="hc-result-risk" style="color:{risk_color};">{risk_text}</div>
                        <div style="color:#94a3b8;font-size:.78rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;">Estimated 10-Year CHD Risk</div>
                        <div class="hc-result-number">{probability * 100:.2f}%</div>
                        <div style="margin-top:.45rem;color:#94a3b8;font-size:.78rem;">Risk Level · <b style="color:{risk_color};">{risk_level}</b></div>
                        {risk_meter(probability)}
                    </div>
                    """, unsafe_allow_html=True
                )

                if risk_level == "HIGH":

                    st.error(
                    "🔴 High Risk: The model estimates a relatively "
                    "higher probability of 10-year CHD."
                )

                elif risk_level == "MEDIUM":

                    st.warning(
                    "🟠 Medium Risk: The model estimates an intermediate "
                    "probability of 10-year CHD."
                )

                else:

                    st.success(
                    "🟢 Low Risk: The model estimates a relatively "
                    "lower probability of 10-year CHD."
                )


                # -----------------------------
                # SHAP Explanation
                # -----------------------------

                st.divider()

                st.header(
                    "🔍 Prediction Explanation"
                )

                st.write(
                    "Top factors influencing the model prediction:"
                )


                shap_data = []

                for item in top_factors:

                    feature = item["feature"]

                    contribution = item["contribution"]

                    feature = feature.replace(
                        "num__", ""
                    )

                    feature = feature.replace(
                        "cat__", ""
                    )

                    shap_data.append({
                        "Feature": feature,
                        "Contribution": contribution
                    })


                shap_df = pd.DataFrame(
                    shap_data
                )


                positive_factors = shap_df[
                    shap_df["Contribution"] > 0
                ]

                negative_factors = shap_df[
                    shap_df["Contribution"] < 0
                ]


                col1, col2 = st.columns(2)


                with col1:

                    st.subheader(
                        "⬆️ Increasing Risk"
                    )

                    if len(positive_factors) > 0:

                        for _, row in positive_factors.iterrows():

                            st.write(
                                f"**{row['Feature']}** "
                                f"(+{row['Contribution']:.4f})"
                            )

                    else:

                        st.write(
                            "No major positive factors."
                        )


                with col2:

                    st.subheader(
                        "⬇️ Reducing Risk"
                    )

                    if len(negative_factors) > 0:

                        for _, row in negative_factors.iterrows():

                            st.write(
                                f"**{row['Feature']}** "
                                f"({row['Contribution']:.4f})"
                            )

                    else:

                        st.write(
                            "No major negative factors."
                        )


                st.subheader(
                    "📊 SHAP Feature Contributions"
                )


                chart_df = shap_df.sort_values(
                    "Contribution"
                )


                fig = px.bar(
                    chart_df,
                    x="Contribution",
                    y="Feature",
                    orientation="h",
                    title="SHAP Feature Contributions"
                )


                fig.update_layout(
                    height=450
                )


                style_plotly(fig)
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                st.info(
                    "⚠️ This system is an educational "
                    "clinical decision support tool and "
                    "is not a medical diagnosis."
                )


            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.write(response.text)


        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Please make sure the backend is running."
            )


        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )


# =========================================================
# EXPLAINABLE PREDICTION
# =========================================================

# =========================================================
# EXPLAINABLE PREDICTION
# =========================================================

elif page == "🔍 Explainable Prediction":

    hc_header("AI EXPLAINABILITY", "Why did the model make this prediction?", "Review the latest assessment and understand which features influenced the model.")


    # -----------------------------------------
    # Check whether prediction exists
    # -----------------------------------------

    if st.session_state.prediction_result is None:

        st.info(
            "👤 Please go to Individual Prediction, "
            "enter patient information, and generate a "
            "prediction first."
        )

        st.warning(
            "The explanation is generated for the most "
            "recent individual prediction."
        )

    else:

        result = st.session_state.prediction_result

        prediction = result["prediction"]
        probability = result["probability"]
        risk_level = result["risk_level"]
        top_factors = result["top_factors"]


        # -----------------------------------------
        # Prediction Summary
        # -----------------------------------------

        st.subheader("🫀 Prediction Summary")

        risk_text = "HIGH RISK" if risk_level == "HIGH" else ("MEDIUM RISK" if risk_level == "MEDIUM" else "LOW RISK")
        risk_color = "#f87171" if risk_level == "HIGH" else ("#fbbf24" if risk_level == "MEDIUM" else "#4ade80")
        st.markdown(
            f"""
            <div class="hc-result">
                <div class="hc-result-label">Latest assessment</div>
                <div style="display:flex;gap:2.5rem;align-items:flex-end;flex-wrap:wrap;">
                    <div><div style="color:#64748b;font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;">Prediction</div><div style="font-size:1.6rem;font-weight:800;color:#f8fafc;">{prediction}</div></div>
                    <div><div style="color:#64748b;font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;">Probability</div><div style="font-size:1.6rem;font-weight:800;color:#f8fafc;">{probability * 100:.2f}%</div></div>
                    <div><div style="color:#64748b;font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;">Risk Level</div><div style="font-size:1.6rem;font-weight:850;color:{risk_color};">{risk_level}</div></div>
                </div>
                {risk_meter(probability)}
            </div>
            """, unsafe_allow_html=True
        )

        if risk_level == "HIGH":

            st.error(
                "🔴 High Risk: The model estimates a relatively "
                "higher probability of 10-year CHD."
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "🟠 Medium Risk: The model estimates an intermediate "
                "probability of 10-year CHD."
            )

        else:

            st.success(
                "🟢 Low Risk: The model estimates a relatively "
                "lower probability of 10-year CHD."
            )


        # -----------------------------------------
        # Explanation
        # -----------------------------------------

        st.divider()

        st.subheader(
            "🧠 Why did the model make this prediction?"
        )

        st.write(
            "SHAP (SHapley Additive exPlanations) shows "
            "how individual features influenced the model's "
            "prediction."
        )


        # -----------------------------------------
        # Prepare SHAP data
        # -----------------------------------------

        shap_data = []

        for item in top_factors:

            feature = item["feature"]

            contribution = item["contribution"]

            feature = feature.replace(
                "num__",
                ""
            )

            feature = feature.replace(
                "cat__",
                ""
            )

            shap_data.append({
                "Feature": feature,
                "Contribution": contribution
            })


        shap_df = pd.DataFrame(
            shap_data
        )


        # -----------------------------------------
        # Increasing / Reducing Factors
        # -----------------------------------------

        increasing = shap_df[
            shap_df["Contribution"] > 0
        ].sort_values(
            "Contribution",
            ascending=False
        )

        reducing = shap_df[
            shap_df["Contribution"] < 0
        ].sort_values(
            "Contribution"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "⬆️ Increasing Model Risk"
            )

            if len(increasing) > 0:

                for _, row in increasing.iterrows():

                    st.write(
                        f"🔴 **{row['Feature']}**  \n"
                        f"Contribution: "
                        f"+{row['Contribution']:.4f}"
                    )

            else:

                st.write(
                    "No major increasing factors."
                )


        with col2:

            st.subheader(
                "⬇️ Reducing Model Risk"
            )

            if len(reducing) > 0:

                for _, row in reducing.iterrows():

                    st.write(
                        f"🟢 **{row['Feature']}**  \n"
                        f"Contribution: "
                        f"{row['Contribution']:.4f}"
                    )

            else:

                st.write(
                    "No major reducing factors."
                )


        # -----------------------------------------
        # SHAP Chart
        # -----------------------------------------

        st.divider()

        st.subheader(
            "📊 Feature Contribution"
        )

        chart_df = shap_df.sort_values(
            "Contribution"
        )


        fig = px.bar(
            chart_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="SHAP Feature Contributions"
        )


        fig.update_layout(
            height=450,
            xaxis_title="SHAP Contribution",
            yaxis_title="Feature"
        )


        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Interpretation Guide
        # -----------------------------------------

        st.divider()

        st.subheader(
            "📖 How to Interpret the Explanation"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                "🟢 **Negative SHAP value**\n\n"
                "The feature pushed the model away from "
                "the positive CHD prediction."
            )

        with col2:

            st.error(
                "🔴 **Positive SHAP value**\n\n"
                "The feature pushed the model toward "
                "the positive CHD prediction."
            )


        st.info(
            "SHAP values explain the behavior of the machine "
            "learning model. They do not represent medical "
            "causal relationships."
        )


        # -----------------------------------------
        # Disclaimer
        # -----------------------------------------

        st.warning(
            "⚠️ This system is an educational clinical "
            "decision support tool and is not a medical diagnosis."
        )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "📁 Batch Prediction":

    hc_header("BATCH ASSESSMENT", "Batch Risk Assessment", "Upload patient records and generate automated risk estimates.")


    st.subheader("CSV Patient Upload")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    with st.expander("Required fields", expanded=False):
        st.caption(
            "male, age, education, currentSmoker, cigsPerDay, BPMeds, "
            "prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, "
            "BMI, heartRate, glucose"
        )


    if uploaded_file is not None:

        try:

            preview_df = pd.read_csv(
                uploaded_file
            )

            st.subheader("Data Preview")

            st.dataframe(
                preview_df.head(10),
                use_container_width=True
            )

            st.write(
                f"Total records: **{len(preview_df)}**"
            )


            if st.button(
                "🚀 Generate Batch Predictions",
                type="primary",
                use_container_width=True
            ):

                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                }

                try:

                    response = requests.post(
                        "http://127.0.0.1:8000/batch-predict",
                        files=files,
                        timeout=60
                    )

                    if response.status_code == 200:

                        batch_result = response.json()

                        results_df = pd.DataFrame(
                            batch_result["results"]
                        )

                        st.success(
                            f"Successfully predicted "
                            f"{batch_result['total_records']} "
                            f"patients!"
                        )


                        # -----------------------------
                        # Summary
                        # -----------------------------

                        st.subheader(
                            "Batch Assessment Summary"
                        )

                        high_risk_count = len(results_df[results_df["risk_level"] == "HIGH"])
                        medium_risk_count = len(results_df[results_df["risk_level"] == "MEDIUM"])
                        low_risk_count = len(results_df[results_df["risk_level"] == "LOW"])

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Patients", batch_result["total_records"])
                        with col2:
                            st.metric("High Risk", high_risk_count)
                        with col3:
                            st.metric("Medium Risk", medium_risk_count)
                        with col4:
                            st.metric("Low Risk", low_risk_count)


                        # -----------------------------
                        # Results
                        # -----------------------------

                        st.divider()

                        st.subheader(
                            "Prediction Results"
                        )

                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )


                        # -----------------------------
                        # Download
                        # -----------------------------

                        csv_data = results_df.to_csv(
                            index=False
                        )

                        st.download_button(
                            label="Download Prediction Results CSV",
                            data=csv_data,
                            file_name="heart_disease_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


                    else:

                        st.error(
                            f"API Error: "
                            f"{response.status_code}"
                        )

                        st.write(
                            response.text
                        )


                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to FastAPI. "
                        "Please make sure the backend is running."
                    )


        except Exception as e:

            st.error(
                f"Unable to read CSV: {str(e)}"
            )


# =========================================================
# POPULATION ANALYTICS
# =========================================================

# =========================================================
# POPULATION ANALYTICS
# =========================================================

elif page == "📊 Population Analytics":

    hc_header("POPULATION ANALYTICS", "Population Health Analytics", "Explore patterns and risk characteristics across the dataset.")


    if df is None:

        st.error("Dataset could not be loaded.")

    else:

        # -----------------------------------------
        # Filters
        # -----------------------------------------

        st.markdown('<div class="hc-section">Analysis filters</div><h3 style="margin-top:0;">Population Filters</h3>', unsafe_allow_html=True)

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:

            gender_filter = st.selectbox(
                "Gender",
                ["All", "Female", "Male"]
            )

        with filter_col2:

            smoker_filter = st.selectbox(
                "Smoking Status",
                ["All", "Non-Smoker", "Smoker"]
            )

        with filter_col3:

            diabetes_filter = st.selectbox(
                "Diabetes",
                ["All", "No Diabetes", "Diabetes"]
            )


        # -----------------------------------------
        # Apply filters
        # -----------------------------------------

        analytics_df = df.copy()

        if gender_filter == "Female":
            analytics_df = analytics_df[
                analytics_df["male"] == 0
            ]

        elif gender_filter == "Male":
            analytics_df = analytics_df[
                analytics_df["male"] == 1
            ]


        if smoker_filter == "Non-Smoker":
            analytics_df = analytics_df[
                analytics_df["currentSmoker"] == 0
            ]

        elif smoker_filter == "Smoker":
            analytics_df = analytics_df[
                analytics_df["currentSmoker"] == 1
            ]


        if diabetes_filter == "No Diabetes":
            analytics_df = analytics_df[
                analytics_df["diabetes"] == 0
            ]

        elif diabetes_filter == "Diabetes":
            analytics_df = analytics_df[
                analytics_df["diabetes"] == 1
            ]


        # -----------------------------------------
        # Filtered statistics
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Filtered population</div><h3 style="margin-top:0;">Population Snapshot</h3>', unsafe_allow_html=True)

        total = len(analytics_df)

        disease_cases = int(
            analytics_df["TenYearCHD"].sum()
        )

        no_disease = total - disease_cases

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Records",
                f"{total:,}"
            )

        with col2:
            st.metric(
                "CHD Cases",
                f"{disease_cases:,}"
            )

        with col3:
            st.metric(
                "No CHD",
                f"{no_disease:,}"
            )


        # -----------------------------------------
        # Disease Distribution
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Outcome overview</div><h3 style="margin-top:0;">Disease Distribution</h3>', unsafe_allow_html=True)

        disease_df = pd.DataFrame({
            "Outcome": [
                "No Heart Disease",
                "Heart Disease"
            ],
            "Count": [
                no_disease,
                disease_cases
            ]
        })

        fig = px.pie(
            disease_df,
            names="Outcome",
            values="Count",
            title="10-Year CHD Distribution"
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Age Group Analysis
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Age segmentation</div><h3 style="margin-top:0;">CHD Rate by Age Group</h3>', unsafe_allow_html=True)

        age_df = (
            analytics_df
            .groupby("Age_Group")["TenYearCHD"]
            .mean()
            .reset_index()
        )

        age_df["Risk Percentage"] = (
            age_df["TenYearCHD"] * 100
        )

        fig = px.bar(
            age_df,
            x="Age_Group",
            y="Risk Percentage",
            title="CHD Rate by Age Group",
            text="Risk Percentage"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="CHD Rate (%)",
            xaxis_title="Age Group"
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # BMI Category
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">BMI analysis</div><h3 style="margin-top:0;">CHD Rate by BMI Category</h3>', unsafe_allow_html=True)

        bmi_df = (
            analytics_df
            .groupby("BMI_Category")["TenYearCHD"]
            .mean()
            .reset_index()
        )

        bmi_df["Risk Percentage"] = (
            bmi_df["TenYearCHD"] * 100
        )

        fig = px.bar(
            bmi_df,
            x="BMI_Category",
            y="Risk Percentage",
            title="CHD Rate by BMI Category",
            text="Risk Percentage"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="CHD Rate (%)",
            xaxis_title="BMI Category"
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Glucose Distribution
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Glucose profile</div><h3 style="margin-top:0;">Glucose Distribution</h3>', unsafe_allow_html=True)

        fig = px.histogram(
            analytics_df,
            x="glucose",
            color="TenYearCHD",
            nbins=30,
            title="Glucose Distribution by CHD Outcome",
            labels={
                "TenYearCHD": "CHD Outcome"
            }
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Blood Pressure Distribution
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Blood pressure</div><h3 style="margin-top:0;">Systolic vs Diastolic BP</h3>', unsafe_allow_html=True)

        fig = px.scatter(
            analytics_df,
            x="sysBP",
            y="diaBP",
            color="TenYearCHD",
            title="Systolic vs Diastolic Blood Pressure",
            labels={
                "sysBP": "Systolic BP",
                "diaBP": "Diastolic BP",
                "TenYearCHD": "CHD Outcome"
            }
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Risk Factor Distribution
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Risk factors</div><h3 style="margin-top:0;">CHD Rate by Risk Factor Count</h3>', unsafe_allow_html=True)

        risk_df = (
            analytics_df
            .groupby("Risk_Factor_Count")["TenYearCHD"]
            .mean()
            .reset_index()
        )

        risk_df["Risk Percentage"] = (
            risk_df["TenYearCHD"] * 100
        )

        fig = px.bar(
            risk_df,
            x="Risk_Factor_Count",
            y="Risk Percentage",
            title="CHD Rate by Number of Risk Factors",
            text="Risk Percentage"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Risk Factor Count",
            yaxis_title="CHD Rate (%)"
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Feature Correlation
        # -----------------------------------------

        st.divider()

        st.markdown('<div class="hc-section">Clinical relationships</div><h3 style="margin-top:0;">Feature Correlation Matrix</h3>', unsafe_allow_html=True)

        numeric_df = analytics_df.select_dtypes(
            include="number"
        )

        correlation = numeric_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Feature Correlation Matrix"
        )

        style_plotly(fig)
        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------
        # Disclaimer
        # -----------------------------------------

        st.info(
            "ℹ️ These charts describe patterns in the historical "
            "dataset. They do not establish medical causation "
            "or provide individual medical advice."
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

# =========================================================
# MODEL EVALUATION
# =========================================================

# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "📈 Model Evaluation":

    hc_header("MODEL PERFORMANCE", "Model Performance & Validation", "Evaluate the final Logistic Regression model using the held-out test dataset.")


    # -----------------------------------------
    # Load evaluation results
    # -----------------------------------------

    evaluation_path = "outputs/evaluation_results.csv"

    try:

        evaluation_df = pd.read_csv(
            evaluation_path
        )

        metrics = evaluation_df.iloc[0]

        # -----------------------------------------
        # Performance Metrics
        # -----------------------------------------

        st.markdown('<div class="hc-section">Validation metrics</div><h3 style="margin-top:0;">Model Performance</h3>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{metrics['Accuracy'] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{metrics['Precision'] * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{metrics['Recall'] * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{metrics['F1 Score'] * 100:.2f}%"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "ROC-AUC",
                f"{metrics['ROC-AUC'] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Sensitivity",
                f"{metrics['Sensitivity'] * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Specificity",
                f"{metrics['Specificity'] * 100:.2f}%"
            )


        # -----------------------------------------
        # Load model
        # -----------------------------------------

        MODEL_PATH = (
            "models/heart_disease_logistic_pipeline.pkl"
        )

        evaluation_model = joblib.load(
            MODEL_PATH
        )


        # -----------------------------------------
        # Prepare dataset
        # -----------------------------------------

        target_col = "TenYearCHD"

        redundant_flag_cols = [
            "High_BP_Flag",
            "High_Chol_Flag",
            "Smoker_Flag",
            "Diabetes_Flag",
            "High_BMI_Flag"
        ]

        X = df.drop(
            columns=[
                target_col
            ] + redundant_flag_cols
        )

        y = df[target_col]


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )


        y_pred = evaluation_model.predict(
            X_test
        )

        y_probability = evaluation_model.predict_proba(
            X_test
        )[:, 1]


        # =========================================
        # ROC CURVE
        # =========================================

        st.divider()

        st.markdown('<div class="hc-section">Discrimination</div><h3 style="margin-top:0;">ROC Curve</h3>', unsafe_allow_html=True)

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_probability
        )

        roc_fig = go.Figure()

        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="Logistic Regression"
            )
        )

        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(
                    dash="dash"
                )
            )
        )

        roc_fig.update_layout(
            title=f"ROC Curve (AUC = {metrics['ROC-AUC']:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        style_plotly(roc_fig)
        st.plotly_chart(
            roc_fig,
            use_container_width=True
        )


        # =========================================
        # PRECISION-RECALL CURVE
        # =========================================

        st.divider()

        st.subheader(
            "🎯 Precision-Recall Curve"
        )

        precision_values, recall_values, pr_thresholds = (
            precision_recall_curve(
                y_test,
                y_probability
            )
        )

        pr_fig = go.Figure()

        pr_fig.add_trace(
            go.Scatter(
                x=recall_values,
                y=precision_values,
                mode="lines",
                name="Logistic Regression"
            )
        )

        pr_fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        style_plotly(pr_fig)
        st.plotly_chart(
            pr_fig,
            use_container_width=True
        )


        # =========================================
        # FEATURE IMPORTANCE
        # =========================================

        st.divider()

        st.subheader(
            "🔎 Feature Importance"
        )

        preprocessor = (
            evaluation_model
            .named_steps["preprocessor"]
        )

        classifier = (
            evaluation_model
            .named_steps["classifier"]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        coefficients = classifier.coef_[0]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coefficients),
            "Coefficient": coefficients
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )

        importance_df = importance_df.sort_values(
            "Importance"
        )

        importance_fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Most Important Features"
        )

        style_plotly(importance_fig)
        st.plotly_chart(
            importance_fig,
            use_container_width=True
        )


        # =========================================
        # SHAP SUMMARY PLOT
        # =========================================

        st.divider()

        st.subheader(
            "🧠 SHAP Summary Plot"
        )

        st.write(
            "This plot shows the overall influence of features "
            "across the test population."
        )

        background = X_train.sample(
            n=min(100, len(X_train)),
            random_state=42
        )

        background_transformed = (
            preprocessor.transform(background)
        )

        X_test_transformed = (
            preprocessor.transform(X_test)
        )

        if hasattr(
            background_transformed,
            "toarray"
        ):
            background_transformed = (
                background_transformed.toarray()
            )

        if hasattr(
            X_test_transformed,
            "toarray"
        ):
            X_test_transformed = (
                X_test_transformed.toarray()
            )

        shap_explainer = shap.LinearExplainer(
            classifier,
            background_transformed
        )

        shap_values = shap_explainer(
            X_test_transformed
        )

        # Limit plot to first 500 test samples
        plot_samples = min(
            500,
            X_test_transformed.shape[0]
        )

        fig = plt.figure()

        shap.summary_plot(
            shap_values.values[:plot_samples],
            X_test_transformed[:plot_samples],
            feature_names=feature_names,
            show=False
        )

        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

        st.caption(
            "Positive SHAP values indicate movement toward "
            "the positive CHD class, while negative values "
            "indicate movement away from it."
        )


        # =========================================
        # MODEL SUMMARY
        # =========================================

        st.divider()

        st.subheader(
            "🤖 Final Model Summary"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                "**Algorithm:** Logistic Regression"
            )

        with col2:

            st.info(
                "**Target:** Ten-Year Coronary Heart Disease"
            )


        st.success(
            "The final model was selected using multiple "
            "evaluation metrics rather than accuracy alone."
        )

        st.warning(
            "⚠️ Evaluation results are based on the held-out "
            "test dataset. This application is an educational "
            "clinical decision-support tool and is not a "
            "medical diagnosis."
        )


    except Exception as e:

        st.error(
            f"Unable to generate evaluation results: {str(e)}"
        )

        st.markdown("---")

st.markdown(
    """
    <div style="margin-top:2rem;padding:.95rem 1rem;text-align:center;border-top:1px solid rgba(148,163,184,.16);
                color:#94a3b8;font-size:.75rem;">
        <b style="color:#dbeafe;">HEARTCARE AI</b> · Educational Clinical Decision Support System ·
        Model-based risk estimation only · Not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

render_footer()
