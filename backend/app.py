from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import sys
import os

from backend.schemas import PatientData, PredictionResponse
from typing import List

app = FastAPI(title="Disease Risk Prediction API")

# Load model pipeline
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_pipeline = joblib.load(os.path.join(base_dir, 'models', 'disease_pipeline.pkl'))
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

def preprocess_input(patient: PatientData) -> pd.DataFrame:
    df = pd.DataFrame([patient.dict()])
    
    # Engineer features as done in training
    df['age_group'] = pd.cut(df['age'], bins=[0, 45, 65, 100], labels=['Young', 'Middle Age', 'Senior'])
    
    # Risk Factor Count
    df['risk_factor_count'] = (
        (df['trestbps'] > 130).astype(int) + 
        (df['chol'] > 240).astype(int) + 
        (df['fbs'] == 1).astype(int)
    )
    return df

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.get("/model-info")
def model_info():
    if model_pipeline:
        return {"model": str(model_pipeline.steps[-1][1])}
    return {"error": "Model not loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded")
        
    try:
        input_df = preprocess_input(patient)
        prediction = model_pipeline.predict(input_df)[0]
        
        if hasattr(model_pipeline, "predict_proba"):
            probability = model_pipeline.predict_proba(input_df)[0][1]
        else:
            probability = float(prediction)
            
        risk_level = "HIGH" if probability >= 0.5 else "LOW"
        pred_label = "HIGH_RISK" if prediction == 1 else "LOW_RISK"
        
        return PredictionResponse(
            prediction=pred_label,
            probability=round(probability, 4),
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict", response_model=List[PredictionResponse])
def batch_predict(patients: List[PatientData]):
    results = []
    for patient in patients:
        results.append(predict(patient))
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
