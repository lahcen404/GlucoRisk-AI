from fastapi import FastAPI
from api.schemas import PatientRequest
import os
import mlflow
from api.model import load_model
import pandas as pd


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientRequest):
    
    
    model = load_model()
    patient_data = patient.model_dump()
    data = pd.DataFrame([patient_data]).astype(float)
    prediction = model.predict(data)[0]
    
    return{
        "predictiooon": str(prediction)
    }
    
    
@app.get("/model")
def check_model():
    model = load_model()

    return {
        "status": "loaded",
        "model": str(model)
    }
    
   
    
@app.get("/mlflow")
def check_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(tracking_uri)

    experiments = mlflow.search_experiments()

    return {
        "tracking_uri": tracking_uri,
        "experiments_count": len(experiments)
    }