import io
import os
import joblib
import pandas as pd
import shap

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_disease_logistic_pipeline.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_disease_features.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "framingham_stage2_engineered.csv"
)

# --------------------------------------------------
# Load model and features
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURE_PATH)

# --------------------------------------------------
# SHAP setup
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

df = pd.read_csv(DATA_PATH)

target_col = "TenYearCHD"

redundant_flag_cols = [
    "High_BP_Flag",
    "High_Chol_Flag",
    "Smoker_Flag",
    "Diabetes_Flag",
    "High_BMI_Flag"
]

X = df.drop(
    columns=[target_col] + redundant_flag_cols
)

y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

background = X_train.sample(
    n=min(200, len(X_train)),
    random_state=42
)

background_transformed = preprocessor.transform(background)

if hasattr(background_transformed, "toarray"):
    background_transformed = background_transformed.toarray()

explainer = shap.LinearExplainer(
    classifier,
    background_transformed
)

feature_names = preprocessor.get_feature_names_out()

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Heart Disease Risk Prediction and Clinical Decision Support System",
    version="1.0"
)

# --------------------------------------------------
# Patient data
# --------------------------------------------------

class PatientData(BaseModel):

    male: int
    age: float
    education: float
    currentSmoker: int
    cigsPerDay: float
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

    BMI_Category: str
    Age_Group: str
    Risk_Factor_Count: int


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Heart Disease Risk Prediction API is running"
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Model information
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model": "Logistic Regression",
        "disease": "Heart Disease",
        "target": "TenYearCHD",
        "features": len(feature_order)
    }


# --------------------------------------------------
# Prediction + SHAP explanation
# --------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    patient = pd.DataFrame([data.model_dump()])

    # Keep exact feature order
    patient = patient[feature_order]

    # Prediction
    prediction = int(model.predict(patient)[0])

    probability = float(
        model.predict_proba(patient)[0][1]
    )

    # Risk level
    if probability < 0.30:
        risk_level = "LOW"
    elif probability < 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # --------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------

    patient_transformed = preprocessor.transform(patient)

    if hasattr(patient_transformed, "toarray"):
        patient_transformed = patient_transformed.toarray()

    shap_result = explainer(patient_transformed)

    shap_values = shap_result.values[0]

    # Create feature contribution list
    contributions = []

    for feature, value in zip(feature_names, shap_values):

        contributions.append({
            "feature": feature,
            "contribution": round(float(value), 4)
        })

    # Sort by absolute contribution
    contributions = sorted(
        contributions,
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    # Top 5 factors
    top_factors = contributions[:5]

    return {
        "prediction": (
            "HIGH_RISK"
            if prediction == 1
            else "LOW_RISK"
        ),

        "probability": round(probability, 4),

        "risk_level": risk_level,

        "top_factors": top_factors
    }


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):

    # Check file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        # Read uploaded CSV
        contents = await file.read()

        batch_df = pd.read_csv(
            io.BytesIO(contents)
        )

        # Required original columns
        required_columns = [
            "male",
            "age",
            "education",
            "currentSmoker",
            "cigsPerDay",
            "BPMeds",
            "prevalentStroke",
            "prevalentHyp",
            "diabetes",
            "totChol",
            "sysBP",
            "diaBP",
            "BMI",
            "heartRate",
            "glucose"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in batch_df.columns
        ]

        if missing_columns:

            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing required columns.",
                    "missing_columns": missing_columns
                }
            )

        # --------------------------------------------------
        # Automatically create engineered features
        # --------------------------------------------------

        def bmi_category(bmi):

            if bmi < 18.5:
                return "Underweight"

            elif bmi < 25:
                return "Normal"

            elif bmi < 30:
                return "Overweight"

            else:
                return "Obese"


        def age_group(age):

            if age < 30:
                return "Young"

            elif age < 50:
                return "Middle Age"

            else:
                return "Senior"


        def risk_count(row):

            count = 0

            if (
                row["sysBP"] >= 140
                or row["diaBP"] >= 90
                or row["prevalentHyp"] == 1
            ):
                count += 1

            if row["totChol"] >= 240:
                count += 1

            if row["currentSmoker"] == 1:
                count += 1

            if row["diabetes"] == 1:
                count += 1

            if row["BMI"] >= 30:
                count += 1

            return count


        batch_df["BMI_Category"] = batch_df[
            "BMI"
        ].apply(bmi_category)

        batch_df["Age_Group"] = batch_df[
            "age"
        ].apply(age_group)

        batch_df["Risk_Factor_Count"] = batch_df.apply(
            risk_count,
            axis=1
        )

        # --------------------------------------------------
        # Keep model feature order
        # --------------------------------------------------

        model_input = batch_df[
            feature_order
        ]

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        predictions = model.predict(
            model_input
        )

        probabilities = model.predict_proba(
            model_input
        )[:, 1]

        # --------------------------------------------------
        # Create results
        # --------------------------------------------------

        results = pd.DataFrame()

        if "patient_id" in batch_df.columns:

            results["patient_id"] = batch_df[
                "patient_id"
            ]

        else:

            results["patient_id"] = [
                f"P{i+1:04d}"
                for i in range(len(batch_df))
            ]


        results["prediction"] = [
            "HIGH_RISK" if p == 1
            else "LOW_RISK"
            for p in predictions
        ]


        results["probability"] = [
            round(float(p), 4)
            for p in probabilities
        ]


        results["risk_level"] = [
            "LOW" if p < 0.30
            else "MEDIUM" if p < 0.60
            else "HIGH"
            for p in probabilities
        ]


        return {
            "total_records": len(results),
            "results": results.to_dict(
                orient="records"
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )