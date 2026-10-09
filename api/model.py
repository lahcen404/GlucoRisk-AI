import os
import mlflow

MODEL_URI = "models:/GlucoRiskAI@production"


def load_model():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    mlflow.set_tracking_uri(tracking_uri)

    model = mlflow.pyfunc.load_model(MODEL_URI)

    return model