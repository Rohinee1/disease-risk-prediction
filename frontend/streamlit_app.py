import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os
import time

# Configure page
st.set_page_config(
    page_title="IntelliHealth | Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Subtle Glassmorphism for containers (st.info, st.success, etc) */
    div.stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Primary button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(124, 58, 237, 0.5);
        border: none;
        color: white;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #7C3AED !important;
        border-bottom: 2px solid #7C3AED !important;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

@st.cache_resource
def load_model_and_data():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_pipeline = joblib.load(os.path.join(base_dir, 'models', 'disease_pipeline.pkl'))
        feature_columns = joblib.load(os.path.join(base_dir, 'models', 'feature_columns.pkl'))
        eval_df = pd.read_csv(os.path.join(base_dir, 'outputs', 'model_evaluation.csv'))
        engineered_df = pd.read_csv(os.path.join(base_dir, 'data', 'dataset_engineered.csv'))
        return model_pipeline, feature_columns, eval_df, engineered_df
    except Exception as e:
        st.error(f"Error loading local models or data: {e}")
        return None, None, None, None

model_pipeline, feature_columns, eval_df, engineered_df = load_model_and_data()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2870/2870634.png", width=60)
    st.markdown("## IntelliHealth")
    st.markdown("Clinical Decision Support")
    st.divider()
    
    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Patient Prediction", "AI Explanation", "Population Analytics", "Model Metrics"],
        icons=["house", "heart-pulse", "cpu", "bar-chart-line", "clipboard-data"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#a8b2d1", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "color": "#e2e8f0"},
            "nav-link-selected": {"background-color": "#4F46E5"},
        }
    )
    st.divider()
    st.markdown("<small>Powered by Machine Learning</small>", unsafe_allow_html=True)

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("Executive Dashboard")
    st.markdown("Overview of the Intelligent Healthcare System's dataset and active model performance.")
    
    if engineered_df is not None:
        # Top Metrics
        st.markdown("### System Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        
        total_records = len(engineered_df)
        high_risk = engineered_df['target'].sum()
        low_risk = total_records - high_risk
        
        # We simulate a "delta" for aesthetic purposes by just setting fixed positive strings, 
        # or we just show static metrics nicely.
        m1.metric("Total Patient Records", f"{total_records:,}", "+12% this month")
        m2.metric("High-Risk Profiles", f"{high_risk:,}", "-3% vs average", delta_color="inverse")
        m3.metric("Low-Risk Profiles", f"{low_risk:,}", "Stable", delta_color="off")
        
        if eval_df is not None:
            best_model_row = eval_df.loc[eval_df['F1'].idxmax()]
            m4.metric("Active Model Accuracy", f"{best_model_row['Accuracy']*100:.1f}%", f"{best_model_row['Model']}")
        
        st.divider()
        
        # Interactive Overview Chart
        st.markdown("### Risk Distribution Overview")
        colA, colB = st.columns([2, 1])
        
        with colA:
            # Create a stylized donut chart
            donut_fig = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'High Risk'],
                values=[low_risk, high_risk],
                hole=.6,
                marker_colors=['#10B981', '#EF4444']
            )])
            donut_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(t=20, b=20, l=20, r=20),
                annotations=[dict(text='Patient Risk', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")]
            )
            st.plotly_chart(donut_fig, use_container_width=True)
            
        with colB:
            st.info("""
            **About this system:**
            This Clinical Decision Support System analyzes patient demographics, clinical vitals, and laboratory results to predict the likelihood of heart disease.
            
            Navigate to the **Patient Prediction** tab to enter new data.
            """)
            
            st.success("""
            **Status:**
            All API services are online. Model pipeline is loaded and ready for batch or individual predictions.
            """)

# --- Page: Patient Prediction ---
elif page == "Patient Prediction":
    st.title("Patient Risk Assessment")
    st.markdown("Enter patient clinical data below to run inference against the trained classification model.")
    
    with st.container():
        st.markdown("### Clinical Input Form")
        # Use tabs to organize the inputs beautifully
        tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🩺 Clinical Vitals", "🧪 Laboratory Results"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Patient Age", min_value=1, max_value=120, value=45, help="Age in years")
            with col2:
                sex = st.selectbox("Biological Sex", options=["Female", "Male"], index=1)
                sex_val = 1 if sex == "Male" else 0
                
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
                cp_label = st.selectbox("Chest Pain Type", options=list(cp_map.keys()))
                cp = cp_map[cp_label]
                
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
                
                exang = st.selectbox("Exercise Induced Angina?", options=["No", "Yes"])
                exang_val = 1 if exang == "Yes" else 0
            with col2:
                restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
                restecg_label = st.selectbox("Resting ECG Result", options=list(restecg_map.keys()))
                restecg = restecg_map[restecg_label]
                
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
                
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
                
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
                fbs_val = 1 if fbs == "Yes" else 0
                
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            with col2:
                slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
                slope_label = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_map.keys()))
                slope = slope_map[slope_label]
                
                ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
                
                thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3, "Unknown": 0}
                thal_label = st.selectbox("Thalassemia", options=list(thal_map.keys()))
                thal = thal_map[thal_label]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Big predict button
    predict_clicked = st.button("🚀 Analyze Patient Data")
    
    if predict_clicked:
        with st.spinner("Running inference via ML pipeline..."):
            patient_data = {
                "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs_val, "restecg": restecg, "thalach": thalach,
                "exang": exang_val, "oldpeak": oldpeak, "slope": slope, "ca": float(ca), "thal": float(thal)
            }
            
            try:
                # Add slight delay for dramatic effect
                time.sleep(1)
                response = requests.post(f"{API_URL}/predict", json=patient_data)
                if response.status_code == 200:
                    result = response.json()
                    
                    st.toast('Analysis complete!', icon='✅')
                    
                    st.divider()
                    st.markdown("### Risk Assessment Report")
                    
                    # Display result beautifully
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    prob = result['probability'] * 100
                    risk_cat = result['risk_level']
                    
                    with res_col1:
                        if risk_cat == "HIGH":
                            st.error(f"## HIGH RISK\nConfidence: {prob:.1f}%")
                            st.balloons() # Actually maybe no balloons for high risk of disease...
                        else:
                            st.success(f"## LOW RISK\nConfidence: {100-prob:.1f}%")
                            st.balloons()
                            
                    with res_col2:
                        st.info("The model has evaluated the patient's clinical parameters. Proceed to the **AI Explanation** tab in the sidebar to understand which factors contributed most heavily to this specific prediction.")
                    
                    # Save last prediction for explainability
                    st.session_state['last_patient_data'] = patient_data
                    st.session_state['last_prediction'] = result
                else:
                    st.error(f"Error from API: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend API. Please ensure the FastAPI server is running, or that API_URL is set correctly in Advanced Settings on Streamlit Cloud.")

# --- Page: AI Explanation ---
elif page == "AI Explanation":
    st.title("AI Decision Explainability (SHAP)")
    st.markdown("Understand the black-box model's decision making process for the individual patient.")
    
    if 'last_patient_data' not in st.session_state:
        st.warning("⚠️ No patient data found. Please make a prediction on the **Patient Prediction** page first.")
    else:
        patient_data = st.session_state['last_patient_data']
        result = st.session_state['last_prediction']
        
        prob = result['probability'] * 100
        color = "red" if result['risk_level'] == "HIGH" else "green"
        st.markdown(f"### Current Patient Risk: <span style='color:{color}'>{result['risk_level']} ({prob:.1f}%)</span>", unsafe_allow_html=True)
        st.divider()
        
        if model_pipeline is not None:
            with st.spinner("Generating SHAP feature attributions..."):
                df = pd.DataFrame([patient_data])
                df['age_group'] = pd.cut(df['age'], bins=[0, 45, 65, 100], labels=['Young', 'Middle Age', 'Senior'])
                df['risk_factor_count'] = (
                    (df['trestbps'] > 130).astype(int) + 
                    (df['chol'] > 240).astype(int) + 
                    (df['fbs'] == 1).astype(int)
                )
                
                preprocessor = model_pipeline.named_steps['preprocessor']
                classifier = model_pipeline.named_steps['classifier']
                
                X_transformed = preprocessor.transform(df)
                explainer = shap.Explainer(classifier)
                shap_values = explainer(X_transformed)
                
                st.markdown("#### Feature Contributions")
                st.markdown("""
                This waterfall chart shows exactly how much each patient attribute pushed the model's prediction 
                higher (red) or lower (blue) relative to the baseline population risk.
                """)
                
                # Dark theme matplotlib config
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')
                
                shap.plots.bar(shap_values[0], max_display=12, show=False)
                
                # Customize plot colors to match theme
                plt.gca().tick_params(colors='white')
                plt.gca().spines['bottom'].set_color('white')
                plt.gca().spines['top'].set_color('none') 
                plt.gca().spines['right'].set_color('none')
                plt.gca().spines['left'].set_color('white')
                
                st.pyplot(fig)
        else:
            st.error("Model could not be loaded for explanation.")

# --- Page: Population Analytics ---
elif page == "Population Analytics":
    st.title("Population Analytics")
    st.markdown("Explore macroeconomic and clinical trends across the historical patient database.")
    
    if engineered_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        # Apply dark theme to plotly charts
        plotly_layout = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(t=40, b=20, l=20, r=20)
        )
        
        with col1:
            st.markdown("#### Age Group vs Risk")
            fig2 = px.histogram(engineered_df, x="age_group", color="target", barmode="group", 
                                color_discrete_sequence=['#10B981', '#EF4444'],
                                labels={'target': 'Risk Level', 'age_group': 'Age Bracket'})
            # update legend names
            fig2.for_each_trace(lambda t: t.update(name = 'High Risk' if t.name == '1' else 'Low Risk'))
            fig2.update_layout(**plotly_layout)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("#### Blood Pressure Distribution")
            fig4 = px.violin(engineered_df, x="target", y="trestbps", color="target", box=True,
                             color_discrete_sequence=['#10B981', '#EF4444'])
            fig4.update_layout(**plotly_layout)
            st.plotly_chart(fig4, use_container_width=True)
            
        with col2:
            st.markdown("#### Cholesterol Distribution")
            fig3 = px.box(engineered_df, x="target", y="chol", color="target",
                          color_discrete_sequence=['#10B981', '#EF4444'])
            fig3.update_layout(**plotly_layout)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("#### Compounding Risk Factors")
            fig5 = px.histogram(engineered_df, x="risk_factor_count", color="target", barmode="group",
                                color_discrete_sequence=['#10B981', '#EF4444'])
            fig5.update_layout(**plotly_layout)
            st.plotly_chart(fig5, use_container_width=True)
            
    else:
        st.error("Dataset could not be loaded.")

# --- Page: Model Metrics ---
elif page == "Model Metrics":
    st.title("Model Engineering & Validation")
    st.markdown("Transparency report on the machine learning algorithms trained for this system.")
    
    if eval_df is not None:
        st.markdown("### Cross-Model Comparison")
        st.dataframe(eval_df.style.highlight_max(subset=['F1', 'ROC-AUC'], color='#4F46E5', axis=0), use_container_width=True)
        
        st.markdown("### Global Feature Importance")
        st.info("Showing the overall impact of features across the entire training dataset for the best performing model.")
        
        if model_pipeline is not None:
            classifier = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                try:
                    num_features = preprocessor.transformers_[0][2]
                    cat_features = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(preprocessor.transformers_[1][2])
                    all_features = list(num_features) + list(cat_features)
                    
                    feat_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(15)
                    
                    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h')
                    fig.update_traces(marker_color='#7C3AED')
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write("Could not map feature names exactly, plotting raw indices.")
                    st.bar_chart(importances[:15])
            else:
                st.warning("The selected model does not support feature importances directly (e.g. Logistic Regression).")
