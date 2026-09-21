from pydantic import BaseModel
from typing import List

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
