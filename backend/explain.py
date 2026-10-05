import os
import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "heart_disease_logistic_pipeline.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "framingham_stage2_engineered.csv"
)

# -----------------------------
# Load model
# -----------------------------
pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# -----------------------------
# Load processed dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

target_col = "TenYearCHD"

redundant_flag_cols = [
    "High_BP_Flag",
    "High_Chol_Flag",
    "Smoker_Flag",
    "Diabetes_Flag",
    "High_BMI_Flag"
]

X = df.drop(columns=[target_col] + redundant_flag_cols)
y = df[target_col]

# -----------------------------
# Same train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Background data for SHAP
# -----------------------------
background = X_train.sample(
    n=min(200, len(X_train)),
    random_state=42
)

background_transformed = preprocessor.transform(background)

# -----------------------------
# Patient to explain
# -----------------------------
patient = pd.DataFrame([{
    "male": 1,
    "age": 55,
    "education": 2,
    "currentSmoker": 1,
    "cigsPerDay": 10,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 240,
    "sysBP": 150,
    "diaBP": 90,
    "BMI": 28.5,
    "heartRate": 80,
    "glucose": 85,
    "BMI_Category": "Overweight",
    "Age_Group": "Middle Age",
    "Risk_Factor_Count": 4
}])

# -----------------------------
# Prediction
# -----------------------------
prediction = pipeline.predict(patient)[0]
probability = pipeline.predict_proba(patient)[0][1]

print("Prediction:", prediction)
print("Probability:", probability)

# -----------------------------
# Transform patient
# -----------------------------
patient_transformed = preprocessor.transform(patient)

# Convert sparse matrix to dense
if hasattr(patient_transformed, "toarray"):
    patient_transformed = patient_transformed.toarray()

if hasattr(background_transformed, "toarray"):
    background_transformed = background_transformed.toarray()

# -----------------------------
# Feature names
# -----------------------------
feature_names = preprocessor.get_feature_names_out()

print("\nNumber of transformed features:", len(feature_names))
print("Number of transformed values:", patient_transformed.shape[1])

# -----------------------------
# SHAP Linear Explainer
# -----------------------------
explainer = shap.LinearExplainer(
    classifier,
    background_transformed
)

shap_values = explainer(patient_transformed)

# -----------------------------
# Display SHAP values
# -----------------------------
print("\nSHAP explanation created successfully!\n")

print("Feature contributions:")

for feature, value in zip(feature_names, shap_values.values[0]):
    print(f"{feature}: {value:.4f}")