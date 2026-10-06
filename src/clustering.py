import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_for_clustering(data, clinical_columns):
	scaler = StandardScaler()
	return scaler.fit_transform(data[list(clinical_columns)]), scaler


def evaluate_kmeans(scaled_data, k_values=range(2, 11)):
	inertias = []
	silhouette_scores = []
	for cluster_count in k_values:
		model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
		model.fit(scaled_data)
		inertias.append(model.inertia_)

		model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
		labels = model.fit_predict(scaled_data)
		silhouette_scores.append(silhouette_score(scaled_data, labels))
	return inertias, silhouette_scores


def fit_kmeans(scaled_data, n_clusters=3):
	model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	return model, model.fit_predict(scaled_data)


def add_cluster_labels(data, labels, cluster_column="Cluster"):
	result = data.copy()
	result[cluster_column] = labels
	return result


def add_risk_categories(data, cluster_column="Cluster", risk_column="risk_category"):
	result = data.copy()
	result[risk_column] = result[cluster_column].map({
		0: "Low Risk",
		1: "High Risk",
		2: "Low Risk",
	})
	return result
