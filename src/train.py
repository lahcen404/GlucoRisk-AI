# src/train.py

import json
import os
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import imblearn
import mlflow as mlflow_package

from mlflow.models import infer_signature

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    confusion_matrix,
    silhouette_score,
    ConfusionMatrixDisplay,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# Paths and configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "diabetes_preprocessed.csv"
)

CLUSTERED_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "diabetes_clustered.csv"
)

MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

EXPERIMENT_NAME = "GlucoRiskAI"
REGISTERED_MODEL_NAME = "GlucoRiskAI"

RANDOM_STATE = 42
FINAL_K = 3


CLINICAL_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# ============================================================
# Data functions
# ============================================================

def load_data():
    """Load the preprocessed dataset."""

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def prepare_features(df):
    """Select clinical features."""

    X = df[CLINICAL_COLUMNS].copy()

    print(f"Processed shape: {X.shape}")

    return X


# ============================================================
# K-Means functions
# ============================================================

def evaluate_kmeans(X_scaled):
    """Evaluate K-Means for different values of K."""

    results = []

    print("\nK-Means evaluation:")

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        results.append(
            {
                "K": k,
                "Inertia": inertia,
                "Silhouette": silhouette,
            }
        )

        print(
            f"K={k} | "
            f"Inertia={inertia:.2f} | "
            f"Silhouette={silhouette:.4f}"
        )

    return pd.DataFrame(results)


def fit_final_kmeans(X_scaled, k):
    """Train final K-Means."""

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def create_risk_category(df):
    """
    Map clusters to project-defined risk categories.

    Cluster 1 = High Risk
    Cluster 0 and 2 = Low Risk
    """

    df = df.copy()

    df["risk_category"] = df["Cluster"].map(
        {
            0: "Low Risk",
            1: "High Risk",
            2: "Low Risk",
        }
    )

    return df


# ============================================================
# Evaluation functions
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label="High Risk",
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label="High Risk",
            zero_division=0,
        ),

        "F1-score": f1_score(
            y_true,
            y_pred,
            pos_label="High Risk",
            zero_division=0,
        ),
    }


def create_confusion_matrix_plot(
    y_true,
    y_pred,
    output_path,
):
    """Save confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "Low Risk",
            "High Risk",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(5, 4),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Low Risk",
            "High Risk",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def create_kmeans_plots(
    evaluation_df,
    output_dir,
):
    """Create K-Means evaluation plots."""

    inertia_path = (
        output_dir
        / "kmeans_inertia.png"
    )

    silhouette_path = (
        output_dir
        / "kmeans_silhouette.png"
    )

    # Elbow plot

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        evaluation_df["K"],
        evaluation_df["Inertia"],
        marker="o",
    )

    plt.xlabel(
        "Number of clusters (K)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        "K-Means Elbow Method"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        inertia_path,
        dpi=150,
    )

    plt.close()

    # Silhouette plot

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        evaluation_df["K"],
        evaluation_df["Silhouette"],
        marker="o",
    )

    plt.xlabel(
        "Number of clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "K-Means Silhouette Score"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        silhouette_path,
        dpi=150,
    )

    plt.close()

    return (
        inertia_path,
        silhouette_path,
    )


# ============================================================
# Classification functions
# ============================================================

def create_models():
    """Create baseline models."""

    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),

        "KNN": KNeighborsClassifier(),

        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
        ),

        "SVM": SVC(
            random_state=RANDOM_STATE,
        ),
    }


def train_baseline_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train a baseline model."""

    pipeline = ImbPipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                ),
            ),

            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    return (
        pipeline,
        predictions,
        metrics,
    )


def tune_svm(
    X_train,
    y_train,
):
    """Tune SVM with GridSearchCV."""

    pipeline = ImbPipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                ),
            ),

            (
                "model",
                SVC(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    parameter_grid = {
        "model__C": [
            0.1,
            1,
            10,
            100,
        ],

        "model__kernel": [
            "linear",
            "rbf",
        ],

        "model__gamma": [
            "scale",
            "auto",
        ],
    }

    # High Risk is the positive class.
    f1_scorer = make_scorer(
        f1_score,
        pos_label="High Risk",
        zero_division=0,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


# ============================================================
# MLflow setup
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

print(
    f"\nMLflow Tracking URI: "
    f"{MLFLOW_TRACKING_URI}"
)

print(
    f"MLflow Experiment: "
    f"{EXPERIMENT_NAME}"
)


# ============================================================
# Load data
# ============================================================

df = load_data()

X = prepare_features(df)


# ============================================================
# Scale data for clustering
# ============================================================

clustering_scaler = StandardScaler()

X_scaled = clustering_scaler.fit_transform(
    X
)


# ============================================================
# K-Means evaluation
# ============================================================

evaluation_df = evaluate_kmeans(
    X_scaled
)

evaluation_csv = (
    ARTIFACT_DIR
    / "kmeans_evaluation.csv"
)

evaluation_df.to_csv(
    evaluation_csv,
    index=False,
)


(
    inertia_plot,
    silhouette_plot,
) = create_kmeans_plots(
    evaluation_df,
    ARTIFACT_DIR,
)


# ============================================================
# Final K-Means
# ============================================================

final_kmeans, cluster_labels = (
    fit_final_kmeans(
        X_scaled,
        FINAL_K,
    )
)


df_clustered = df.copy()

df_clustered["Cluster"] = (
    cluster_labels
)


# ============================================================
# Cluster profiles
# ============================================================

cluster_profile = (
    df_clustered
    .groupby("Cluster")[CLINICAL_COLUMNS]
    .mean()
    .round(2)
)

print(
    "\nCluster profiles:"
)

print(
    cluster_profile
)


cluster_profile_path = (
    ARTIFACT_DIR
    / "cluster_profile.csv"
)

cluster_profile.to_csv(
    cluster_profile_path
)


# ============================================================
# Risk categories
# ============================================================

df_clustered = create_risk_category(
    df_clustered
)

print(
    "\nRisk distribution:"
)

print(
    df_clustered[
        "risk_category"
    ].value_counts()
)


df_clustered.to_csv(
    CLUSTERED_DATA_PATH,
    index=False,
)


# ============================================================
# Clustering MLflow run
# ============================================================

with mlflow.start_run(
    run_name="Clustering_K3"
):

    mlflow.log_param(
        "k",
        FINAL_K,
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE,
    )

    mlflow.log_param(
        "scaler",
        "StandardScaler",
    )

    mlflow.log_param(
        "n_features",
        len(CLINICAL_COLUMNS),
    )

    mlflow.log_param(
        "dataset_rows",
        len(df),
    )

    mlflow.log_metric(
        "inertia",
        final_kmeans.inertia_,
    )

    final_silhouette = silhouette_score(
        X_scaled,
        cluster_labels,
    )

    mlflow.log_metric(
        "silhouette_score",
        final_silhouette,
    )

    mlflow.log_text(
        "\n".join(CLINICAL_COLUMNS),
        "feature_list.txt",
    )

    mlflow.log_artifact(
        str(evaluation_csv)
    )

    mlflow.log_artifact(
        str(inertia_plot)
    )

    mlflow.log_artifact(
        str(silhouette_plot)
    )

    mlflow.log_artifact(
        str(cluster_profile_path)
    )

    kmeans_model_path = (
        ARTIFACT_DIR
        / "kmeans_model.joblib"
    )

    kmeans_scaler_path = (
        ARTIFACT_DIR
        / "clustering_scaler.joblib"
    )

    joblib.dump(
        final_kmeans,
        kmeans_model_path,
    )

    joblib.dump(
        clustering_scaler,
        kmeans_scaler_path,
    )

    mlflow.log_artifact(
        str(kmeans_model_path)
    )

    mlflow.log_artifact(
        str(kmeans_scaler_path)
    )

    mlflow.set_tag(
        "project",
        "GlucoRiskAI",
    )

    mlflow.set_tag(
        "task",
        "unsupervised_clustering",
    )

    mlflow.set_tag(
        "risk_definition",
        "cluster_derived",
    )


print(
    "\nClustering MLflow run completed."
)


# ============================================================
# Classification data
# ============================================================

X_classification = (
    df_clustered[
        CLINICAL_COLUMNS
    ]
)

y = (
    df_clustered[
        "risk_category"
    ]
)


print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X_classification,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


print(
    f"\nTrain shape: "
    f"{X_train.shape}"
)

print(
    f"Test shape: "
    f"{X_test.shape}"
)


# ============================================================
# Check SMOTE
# ============================================================

check_scaler = StandardScaler()

X_train_scaled = (
    check_scaler.fit_transform(
        X_train
    )
)

smote_check = SMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = (
    smote_check.fit_resample(
        X_train_scaled,
        y_train,
    )
)


print(
    f"\nAfter SMOTE: "
    f"{X_train_smote.shape}"
)

print(
    y_train_smote.value_counts()
)


# ============================================================
# Baseline models
# ============================================================

models = create_models()

baseline_results = []


for model_name, model in models.items():

    (
        pipeline,
        predictions,
        metrics,
    ) = train_baseline_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print(
        f"\n{model_name}: "
        f"Accuracy={metrics['Accuracy']:.4f}, "
        f"Precision={metrics['Precision']:.4f}, "
        f"Recall={metrics['Recall']:.4f}, "
        f"F1={metrics['F1-score']:.4f}"
    )

    baseline_results.append(
        {
            "Model": model_name,
            **metrics,
        }
    )

    run_name = (
        "Classification_"
        + model_name.replace(
            " ",
            "_",
        )
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "model",
            model_name,
        )

        mlflow.log_param(
            "scaler",
            "StandardScaler",
        )

        mlflow.log_param(
            "sampling",
            "SMOTE",
        )

        mlflow.log_param(
            "test_size",
            0.2,
        )

        for metric_name, value in (
            metrics.items()
        ):

            mlflow.log_metric(
                metric_name,
                value,
            )

        mlflow.log_text(
            "\n".join(
                CLINICAL_COLUMNS
            ),
            "feature_list.txt",
        )

        mlflow.set_tag(
            "project",
            "GlucoRiskAI",
        )

        mlflow.set_tag(
            "task",
            "classification",
        )

        mlflow.set_tag(
            "risk_target",
            "cluster_derived_binary_classification",
        )


baseline_results_df = pd.DataFrame(
    baseline_results
)

baseline_results_path = (
    ARTIFACT_DIR
    / "baseline_results.csv"
)

baseline_results_df.to_csv(
    baseline_results_path,
    index=False,
)


# ============================================================
# SVM tuning
# ============================================================

print(
    "\nTuning SVM..."
)

search = tune_svm(
    X_train,
    y_train,
)


print(
    "\nBest SVM parameters:"
)

print(
    search.best_params_
)

print(
    f"Best CV High-Risk F1: "
    f"{search.best_score_:.4f}"
)


# ============================================================
# Final pipeline
# ============================================================

# GridSearchCV already fitted the best pipeline.
#
# Contains:
# StandardScaler -> SMOTE -> SVC

final_pipeline = (
    search.best_estimator_
)


# ============================================================
# Final evaluation
# ============================================================

final_predictions = (
    final_pipeline.predict(
        X_test
    )
)

final_metrics = calculate_metrics(
    y_test,
    final_predictions,
)


print(
    "\nFinal model results:"
)

print(
    f"Accuracy:  "
    f"{final_metrics['Accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{final_metrics['Precision']:.4f}"
)

print(
    f"Recall:    "
    f"{final_metrics['Recall']:.4f}"
)

print(
    f"F1 Score:  "
    f"{final_metrics['F1-score']:.4f}"
)


# ============================================================
# Confusion matrix
# ============================================================

print(
    "\nConfusion matrix:"
)

final_cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=[
        "Low Risk",
        "High Risk",
    ],
)


final_cm_df = pd.DataFrame(
    final_cm,
    index=[
        "Low Risk",
        "High Risk",
    ],
    columns=[
        "Low Risk",
        "High Risk",
    ],
)

print(
    final_cm_df
)


confusion_matrix_path = (
    ARTIFACT_DIR
    / "final_confusion_matrix.png"
)

create_confusion_matrix_plot(
    y_test,
    final_predictions,
    confusion_matrix_path,
)


# ============================================================
# Save SVM parameters
# ============================================================

parameter_grid_path = (
    ARTIFACT_DIR
    / "svm_parameter_grid.json"
)

with open(
    parameter_grid_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "model__C": [
                0.1,
                1,
                10,
                100,
            ],

            "model__kernel": [
                "linear",
                "rbf",
            ],

            "model__gamma": [
                "scale",
                "auto",
            ],
        },
        file,
        indent=4,
    )


best_params_path = (
    ARTIFACT_DIR
    / "best_svm_parameters.json"
)

with open(
    best_params_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        search.best_params_,
        file,
        indent=4,
    )


# ============================================================
# Dataset information
# ============================================================

dataset_info_path = (
    ARTIFACT_DIR
    / "dataset_information.txt"
)

with open(
    dataset_info_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        f"Dataset shape: {df.shape}\n"
    )

    file.write(
        f"Classification features: "
        f"{len(CLINICAL_COLUMNS)}\n"
    )

    file.write(
        "Features:\n"
    )

    for column in CLINICAL_COLUMNS:

        file.write(
            f"- {column}\n"
        )

    file.write(
        "\nTarget: risk_category\n"
    )

    file.write(
        "Target definition: "
        "cluster-derived "
        "High Risk / Low Risk\n"
    )

    file.write(
        f"\nTrain shape: "
        f"{X_train.shape}\n"
    )

    file.write(
        f"Test shape: "
        f"{X_test.shape}\n"
    )


# ============================================================
# Library versions
# ============================================================

library_versions_path = (
    ARTIFACT_DIR
    / "library_versions.txt"
)

with open(
    library_versions_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        f"Python: "
        f"{platform.python_version()}\n"
    )

    file.write(
        f"pandas: "
        f"{pd.__version__}\n"
    )

    file.write(
        f"numpy: "
        f"{np.__version__}\n"
    )

    file.write(
        f"scikit-learn: "
        f"{sklearn.__version__}\n"
    )

    file.write(
        f"imbalanced-learn: "
        f"{imblearn.__version__}\n"
    )

    file.write(
        f"MLflow: "
        f"{mlflow_package.__version__}\n"
    )


# ============================================================
# Save local model
# ============================================================

local_model_path = (
    MODEL_DIR
    / "glucoriskai_svm_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    local_model_path,
)


# ============================================================
# Final MLflow production run
# ============================================================

with mlflow.start_run(
    run_name="Final_SVM_Production"
) as run:

    # Parameters

    mlflow.log_param(
        "model",
        "SVM",
    )

    mlflow.log_param(
        "kernel",
        search.best_params_[
            "model__kernel"
        ],
    )

    mlflow.log_param(
        "C",
        search.best_params_[
            "model__C"
        ],
    )

    mlflow.log_param(
        "gamma",
        search.best_params_[
            "model__gamma"
        ],
    )

    mlflow.log_param(
        "scaler",
        "StandardScaler",
    )

    mlflow.log_param(
        "sampling",
        "SMOTE",
    )

    mlflow.log_param(
        "cv",
        5,
    )

    mlflow.log_param(
        "scoring",
        "f1_high_risk",
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE,
    )

    # Metrics

    mlflow.log_metric(
        "cv_f1_high_risk",
        search.best_score_,
    )

    for metric_name, value in (
        final_metrics.items()
    ):

        mlflow.log_metric(
            metric_name,
            value,
        )

    # Tags

    mlflow.set_tag(
        "project",
        "GlucoRiskAI",
    )

    mlflow.set_tag(
        "model_type",
        "SVM",
    )

    mlflow.set_tag(
        "validation_status",
        "passed",
    )

    mlflow.set_tag(
        "risk_task",
        "cluster_derived_binary_classification",
    )

    mlflow.set_tag(
        "production_alias",
        "production",
    )

    # Features

    mlflow.log_text(
        "\n".join(
            CLINICAL_COLUMNS
        ),
        "feature_list.txt",
    )

    # Artifacts

    mlflow.log_artifact(
        str(dataset_info_path)
    )

    mlflow.log_artifact(
        str(library_versions_path)
    )

    mlflow.log_artifact(
        str(parameter_grid_path)
    )

    mlflow.log_artifact(
        str(best_params_path)
    )

    mlflow.log_artifact(
        str(confusion_matrix_path)
    )

    mlflow.log_artifact(
        str(baseline_results_path)
    )

    # Input example

    input_example = (
        X_train.head(3)
    )

    # Signature

    signature = infer_signature(
        X_train,
        final_pipeline.predict(
            X_train.head(3)
        ),
    )

    # ========================================================
    # Register model
    # ========================================================

    print(
        "\nRegistering model in MLflow..."
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        name="glucoriskai_pipeline",
        signature=signature,
        input_example=input_example,
        registered_model_name=(
            REGISTERED_MODEL_NAME
        ),
        serialization_format="skops",

        # These are the types reported
        # by skops as trusted.

        skops_trusted_types=[
            "imblearn.over_sampling._smote.base.SMOTE",
            "imblearn.pipeline.Pipeline",
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._kd_tree.KDTree",
        ],

        tags={
            "project": "GlucoRiskAI",
            "model_type": "SVM",
            "validation_status": "passed",
            "risk_task": (
                "cluster_derived_binary_classification"
            ),
        },
    )

    print(
        "\nModel registered successfully."
    )

    print(
        f"Model URI: "
        f"{model_info.model_uri}"
    )


# ============================================================
# Set production alias
# ============================================================

try:

    client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    latest_versions = (
        client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )
    )

    if latest_versions:

        latest_version = max(
            latest_versions,
            key=lambda version: int(
                version.version
            ),
        )

        version_number = (
            latest_version.version
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "production",
            version_number,
        )

        print(
            "\nProduction alias created:"
        )

        print(
            f"{REGISTERED_MODEL_NAME}"
            f"@production "
            f"-> version "
            f"{version_number}"
        )

except Exception as error:

    print(
        "\nWarning: Could not create "
        "production alias."
    )

    print(error)


# ============================================================
# Final summary
# ============================================================

print(
    "\n===================================="
)

print(
    "GlucoRiskAI training completed."
)

print(
    "===================================="
)

print(
    "Model: SVM"
)

print(
    f"Best parameters: "
    f"{search.best_params_}"
)

print(
    f"Accuracy: "
    f"{final_metrics['Accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{final_metrics['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{final_metrics['Recall']:.4f}"
)

print(
    f"F1-score: "
    f"{final_metrics['F1-score']:.4f}"
)

print(
    f"Local model: "
    f"{local_model_path}"
)

print(
    f"MLflow model: "
    f"{REGISTERED_MODEL_NAME}"
    f"@production"
)