# Intelligent Healthcare Disease Risk Prediction & Clinical Decision Support System

This project is a full-stack machine learning system that predicts the risk of Heart Disease using clinical and lifestyle parameters. It functions as a clinical decision-support application providing predictions, probability scores, model explanations (via SHAP), and structured risk assessments.

## Architecture
- **Dataset:** UCI Heart Disease Dataset
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Backend API:** FastAPI
- **Frontend Dashboard:** Streamlit
- **Explainable AI:** SHAP

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download Dataset:
   ```bash
   python download_data.py
   ```

3. Train Model:
   ```bash
   cd training
   python train.py
   cd ..
   ```

## Running the Application

To run the application, you need to start both the FastAPI backend and the Streamlit frontend.

**Terminal 1 (Backend):**
```bash
cd backend
python app.py
```
The API will be available at `http://localhost:8000`.

**Terminal 2 (Frontend):**
```bash
cd frontend
streamlit run streamlit_app.py
```
The Streamlit app will open in your browser, typically at `http://localhost:8501`.

## Features
- **Dashboard:** Overview of dataset and model performance metrics.
- **Individual Prediction:** Input patient parameters to receive a prediction and risk category.
- **Explainable Prediction:** See exactly which features contributed to a patient's risk score using SHAP.
- **Population Analytics:** Interactive Plotly charts showing distributions of disease, age, cholesterol, etc.
- **Model Evaluation:** View accuracy, precision, recall, F1, ROC-AUC, and feature importance.
