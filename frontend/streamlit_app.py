import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Disease Risk Prediction", layout="wide")

# API URL - Can be overridden by environment variable for deployment
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

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Dashboard", 
                         "Individual Prediction", 
                         "Explainable Prediction", 
                         "Population Analytics", 
                         "Model Evaluation"])

if page == "Dashboard":
    st.title("Project Dashboard")
    st.markdown("### Intelligent Healthcare Disease Risk Prediction System")
    
    if engineered_df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(engineered_df))
        
        positive_class = engineered_df['target'].sum()
        col2.metric("High-Risk Count (Positive)", positive_class)
        col3.metric("Lower-Risk Count (Negative)", len(engineered_df) - positive_class)
        
        st.subheader("Model Overview")
        if eval_df is not None:
            best_model_row = eval_df.loc[eval_df['F1'].idxmax()]
            st.write(f"**Best Model Selected:** {best_model_row['Model']}")
            st.write(f"**Accuracy:** {best_model_row['Accuracy']:.4f}")
            st.write(f"**F1 Score:** {best_model_row['F1']:.4f}")
            st.write(f"**ROC-AUC:** {best_model_row['ROC-AUC']:.4f}")

elif page == "Individual Prediction":
    st.title("Individual Risk Prediction")
    st.markdown("Enter patient information to predict heart disease risk.")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
            chol = st.number_input("Serum Cholestoral in mg/dl", value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1])
        with col2:
            restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
            thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
            exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1])
            oldpeak = st.number_input("ST depression induced by exercise", value=1.0)
            slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
            ca = st.selectbox("Number of major vessels (0-4) colored by flourosopy", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])
            
        submitted = st.form_submit_button("Predict Disease Risk")
        
    if submitted:
        patient_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": float(ca), "thal": float(thal)
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=patient_data)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Risk Assessment")
                st.write(f"**Prediction:** {result['prediction']}")
                st.write(f"**Probability:** {result['probability'] * 100:.2f}%")
                st.write(f"**Risk Category:** {result['risk_level']}")
                
                # Save last prediction for explainability
                st.session_state['last_patient_data'] = patient_data
                st.session_state['last_prediction'] = result
            else:
                st.error(f"Error from API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend API. Is it running?")

elif page == "Explainable Prediction":
    st.title("Explainable Prediction")
    
    if 'last_patient_data' not in st.session_state:
        st.info("Please make an Individual Prediction first to see its explanation.")
    else:
        st.subheader("Why did the model produce this prediction?")
        patient_data = st.session_state['last_patient_data']
        result = st.session_state['last_prediction']
        
        st.write(f"**Current Prediction:** {result['risk_level']} RISK ({result['probability']*100:.1f}%)")
        
        if model_pipeline is not None:
            # Reconstruct the feature engineering for this single instance
            df = pd.DataFrame([patient_data])
            df['age_group'] = pd.cut(df['age'], bins=[0, 45, 65, 100], labels=['Young', 'Middle Age', 'Senior'])
            df['risk_factor_count'] = (
                (df['trestbps'] > 130).astype(int) + 
                (df['chol'] > 240).astype(int) + 
                (df['fbs'] == 1).astype(int)
            )
            
            # The model_pipeline is an ImbPipeline: [preprocessor, smote, classifier]
            preprocessor = model_pipeline.named_steps['preprocessor']
            classifier = model_pipeline.named_steps['classifier']
            
            # Transform data
            X_transformed = preprocessor.transform(df)
            
            # Generate SHAP values
            # Using TreeExplainer for Random Forest / XGBoost, or KernelExplainer for Logistic Regression
            explainer = shap.Explainer(classifier)
            shap_values = explainer(X_transformed)
            
            st.write("Feature Importance for this Prediction:")
            # We can use shap.plots.waterfall but matplotlib integration is easier with summary_plot or bar
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # We need feature names from the preprocessor to make it readable
            # Since categorical encoding changes the number of columns, getting exact names can be tricky
            # We will just plot standard SHAP bar for the instance
            shap.plots.bar(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)
            
            st.markdown("""
            **How to read this chart:**
            - **Red bars (positive values)** push the prediction higher (increase risk).
            - **Blue bars (negative values)** push the prediction lower (decrease risk).
            """)
            
        else:
            st.error("Model could not be loaded for explanation.")

elif page == "Population Analytics":
    st.title("Population Analytics")
    if engineered_df is not None:
        
        st.subheader("Disease Distribution")
        fig1 = px.pie(engineered_df, names='target', title="Overall Disease Risk Distribution (0=Low, 1=High)")
        st.plotly_chart(fig1)
        
        st.subheader("Risk by Age Group")
        fig2 = px.histogram(engineered_df, x="age_group", color="target", barmode="group")
        st.plotly_chart(fig2)
        
        st.subheader("Cholesterol Distribution")
        fig3 = px.box(engineered_df, x="target", y="chol", color="target")
        st.plotly_chart(fig3)
        
        st.subheader("Blood Pressure Distribution")
        fig4 = px.violin(engineered_df, x="target", y="trestbps", color="target", box=True)
        st.plotly_chart(fig4)
        
        # Risk factor distribution
        st.subheader("Risk Factors Present vs Target")
        fig5 = px.histogram(engineered_df, x="risk_factor_count", color="target", barmode="group")
        st.plotly_chart(fig5)
        
    else:
        st.error("Dataset could not be loaded.")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    
    if eval_df is not None:
        st.subheader("Metrics")
        st.dataframe(eval_df)
        
        st.subheader("Feature Importance (Global)")
        if model_pipeline is not None:
            classifier = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get feature names from preprocessor if possible, else use indices
                try:
                    num_features = preprocessor.transformers_[0][2]
                    cat_features = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(preprocessor.transformers_[1][2])
                    all_features = list(num_features) + list(cat_features)
                    
                    feat_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(15)
                    
                    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top 15 Feature Importances")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig)
                except Exception as e:
                    st.write("Could not map feature names exactly, plotting raw indices.")
                    st.bar_chart(importances[:15])
            else:
                st.info("The selected model does not support feature importances directly (e.g. Logistic Regression).")
