from datetime import datetime
from pathlib import Path
import os
import subprocess

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


# Project paths inside the Airflow container
PROJECT_DIR = Path("/opt/airflow")

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "diabetes_preprocessed.csv"
)

TRAIN_SCRIPT = PROJECT_DIR / "src" / "train.py"

MODEL_PATH = (
    PROJECT_DIR / "models" / "glucoriskai_svm_pipeline.joblib"
)


# Task 1: Validate the dataset
def check_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(f"Dataset found: {DATA_PATH}")


# Task 2: Execute the real training script
def train_model():
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT}"
        )

    # Remove an old artifact so verification checks
    # that this run actually produced a new model.
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    env["PYTHONPATH"] = str(PROJECT_DIR)

    subprocess.run(
        ["python", str(TRAIN_SCRIPT)],
        cwd=str(PROJECT_DIR),
        env=env,
        check=True,
        text=True,
    )

    print("Model training script completed.")


# Task 3: Verify the trained model
def verify_training():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model was not saved: {MODEL_PATH}"
        )

    if MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("The saved model file is empty.")

    print(f"Trained model verified: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size} bytes")


# Define the actual GlucoRiskAI pipeline
with DAG(
    dag_id="glucoriskai_training_pipeline",
    description="Validate diabetes data, train the SVM model, and verify the artifact",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["glucoriskai", "machine-learning", "mlflow"],
) as dag:

    dataset_task = PythonOperator(
        task_id="check_dataset",
        python_callable=check_dataset,
    )

    training_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    verification_task = PythonOperator(
        task_id="verify_training",
        python_callable=verify_training,
    )

    # Execution order
    dataset_task >> training_task >> verification_task