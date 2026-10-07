from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_and_apply_smote(X_train, X_test, y_train):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    return (
        X_train_scaled,
        X_test_scaled,
        X_train_smote,
        y_train_smote,
        scaler
    )


def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(
            random_state=42
        ),
        "SVM": SVC(
            random_state=42
        ),
    }


def train_models(models, features, target):
    for model in models.values():
        model.fit(features, target)

    return models


def tune_svm(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", SVC(random_state=42)),
    ])

    parameter_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    f1_scorer = make_scorer(
        f1_score,
        pos_label="High Risk"
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    return search


def create_final_pipeline(best_model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", best_model),
    ])


def train_final_pipeline(pipeline, X_train, y_train):
    pipeline.fit(
        X_train,
        y_train
    )