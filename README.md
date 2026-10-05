# ❤️ Intelligent Healthcare Disease Risk Prediction & Clinical Decision Support System

## 1. Project Overview

The Intelligent Healthcare Disease Risk Prediction & Clinical Decision Support System is a machine-learning-based application designed to estimate the 10-year risk of coronary heart disease.

The system accepts patient clinical and lifestyle information and provides:

- Heart disease risk prediction
- Risk probability
- Risk level
- Explainable AI using SHAP
- Population-level analytics
- Model evaluation
- Batch prediction through CSV upload

This system is intended for educational and decision-support purposes only and is not a medical diagnosis.

---

## 2. Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Logistic Regression
- Decision Tree

### Explainable AI

- SHAP

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit
- Plotly

### Model Storage

- Joblib
- Pickle model pipeline

---

## 3. Dataset

The project uses the Framingham Heart Study dataset.

Target variable:

`TenYearCHD`

Target meaning:

- 0 = No 10-year CHD event
- 1 = 10-year CHD event

The processed dataset is stored in:

`data/framingham_stage2_engineered.csv`

---

## 4. Machine Learning Pipeline

The machine learning workflow includes:

1. Data loading
2. Data cleaning
3. Missing value handling
4. Exploratory Data Analysis
5. Feature engineering
6. Categorical encoding
7. Feature scaling
8. Train-test split
9. SMOTE for class imbalance
10. Model training
11. Model evaluation
12. Model comparison
13. Final model selection

---

## 5. Feature Engineering

The system uses engineered features including:

- BMI Category
- Age Group
- Risk Factor Count

These features are automatically calculated before prediction.

---

## 6. Models

The project compares:

### Logistic Regression

Used as the final model because it provided stronger recall and ROC-AUC performance for the project.

### Decision Tree

Used as the comparison model.

---

## 7. Final Model

The selected final model is:

**Logistic Regression**

The trained model is stored as:

`models/heart_disease_logistic_pipeline.pkl`

The feature order is stored as:

`models/heart_disease_features.pkl`

---

## 8. Model Performance

The Logistic Regression model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 66.04% |
| Precision | 23.93% |
| Recall | 56.59% |
| F1 Score | 33.64% |
| ROC-AUC | 68.56% |
| Sensitivity | 56.59% |
| Specificity | 67.73% |

### Confusion Matrix

| | Predicted No CHD | Predicted CHD |
|---|---:|---:|
| Actual No CHD | 487 | 232 |
| Actual CHD | 56 | 73 |

---

## 9. Explainable AI

SHAP is used to explain individual model predictions.

The system identifies:

- Factors increasing model risk
- Factors reducing model risk
- Feature contribution values
- SHAP feature contribution charts
- SHAP summary visualization

Positive SHAP values indicate movement toward the positive CHD class, while negative SHAP values indicate movement away from it.

SHAP explanations describe model behavior and should not be interpreted as medical causal relationships.

---

## 10. Application Features

### 🏠 Dashboard

Provides:

- Dataset statistics
- Disease distribution
- Model information
- System purpose

### 👤 Individual Prediction

Users can enter patient information and receive:

- Prediction
- Probability
- Risk level
- SHAP explanation

### 🔍 Explainable Prediction

Provides detailed explanation of the latest prediction using SHAP.

### 📊 Population Analytics

Provides:

- Disease distribution
- Risk by age group
- Risk by BMI category
- Glucose distribution
- Blood pressure analysis
- Risk factor distribution
- Feature correlation analysis
- Population filters

### 📈 Model Evaluation

Provides:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Sensitivity
- Specificity
- Confusion matrix
- ROC curve
- Precision-recall curve
- Feature importance
- SHAP summary plot

### 📁 Batch Prediction

Users can upload a CSV containing multiple patient records and receive:

- Patient ID
- Prediction
- Probability
- Risk level

The prediction results can also be downloaded as a CSV file.

---

## 11. Backend API

FastAPI provides the following endpoints:

### GET `/`

Checks whether the API is running.

### GET `/health`

Returns API health status.

### GET `/model-info`

Returns model information.

### POST `/predict`

Generates an individual heart disease risk prediction and SHAP explanation.

### POST `/batch-predict`

Generates predictions for multiple patients uploaded through a CSV file.

---

## 12. Project Structure

```text
Disease prediction/
│
├── backend/
│   ├── app.py
│   └── explain.py
│
├── data/
│   └── framingham_stage2_engineered.csv
│
├── frontend/
│   └── streamlit_app.py
│
├── models/
│   ├── heart_disease_logistic_pipeline.pkl
│   └── heart_disease_features.pkl
│
├── outputs/
│   ├── evaluation.py
│   └── evaluation_results.csv
│
├── requirements.txt
├── README.md
└── .gitignore