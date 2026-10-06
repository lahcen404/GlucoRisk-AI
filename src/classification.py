import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def split_data(features, target):
	return train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)


def scale_and_apply_smote(X_train, X_test, y_train):
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	smote = SMOTE(random_state=42)
	X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
	return X_train_scaled, X_test_scaled, X_train_smote, y_train_smote, scaler


def create_models():
	return {
		"Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
		"KNN": KNeighborsClassifier(),
		"Random Forest": RandomForestClassifier(random_state=42),
		"SVM": SVC(random_state=42),
	}


def train_models(models, features, target):
	for model in models.values():
		model.fit(features, target)
	return models


def tune_svm(features, target):
	parameter_grid = {
		"C": [0.1, 1, 10, 100],
		"kernel": ["linear", "rbf"],
		"gamma": ["scale", "auto"],
	}
	f1_scorer = make_scorer(f1_score, pos_label="High Risk")
	search = GridSearchCV(
		estimator=SVC(random_state=42),
		param_grid=parameter_grid,
		cv=5,
		scoring=f1_scorer,
	)
	search.fit(features, target)
	return search


def create_final_pipeline(best_model):
	return Pipeline([
		("scaler", StandardScaler()),
		("smote", SMOTE(random_state=42)),
		("model", best_model),
	])


def train_final_pipeline(pipeline, X_train, y_train):
	pipeline.fit(X_train, y_train)
	return pipeline
