import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


def classification_metrics(actual, predicted, positive_label="High Risk"):
	return {
		"Accuracy": accuracy_score(actual, predicted),
		"Precision": precision_score(actual, predicted, pos_label=positive_label),
		"Recall": recall_score(actual, predicted, pos_label=positive_label),
		"F1-score": f1_score(actual, predicted, pos_label=positive_label),
	}


def classification_confusion_matrix(actual, predicted, labels=("Low Risk", "High Risk")):
	matrix = confusion_matrix(actual, predicted, labels=list(labels))
	return pd.DataFrame(matrix, index=list(labels), columns=list(labels))
