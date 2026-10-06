import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


ZERO_AS_MISSING_COLUMNS = [
	"Glucose",
	"BloodPressure",
	"SkinThickness",
	"Insulin",
	"BMI",
]
OUTLIER_CAPPING_COLUMNS = ["SkinThickness", "Insulin"]


def replace_zero_values(data, columns=ZERO_AS_MISSING_COLUMNS):
	result = data.copy()
	result[list(columns)] = result[list(columns)].replace(0, np.nan)
	return result


def scale_features(data):
	scaler = StandardScaler()
	return scaler.fit_transform(data), scaler


def impute_scaled_features(scaled_data, n_neighbors=5):
	imputer = KNNImputer(n_neighbors=n_neighbors)
	return imputer.fit_transform(scaled_data), imputer


def inverse_scale_features(scaled_data, scaler):
	return scaler.inverse_transform(scaled_data)


def cap_iqr_outliers(data, columns=OUTLIER_CAPPING_COLUMNS):
	result = data.copy()
	selected = list(columns)
	first_quartile = result[selected].quantile(0.25)
	third_quartile = result[selected].quantile(0.75)
	iqr = third_quartile - first_quartile
	lower_bound = first_quartile - 1.5 * iqr
	upper_bound = third_quartile + 1.5 * iqr

	for column in selected:
		result[column] = result[column].clip(
			lower=lower_bound[column],
			upper=upper_bound[column],
		)
	return result


def preprocess_data(data, clinical_columns):
	prepared = replace_zero_values(data)
	scaled_data, scaler = scale_features(prepared[list(clinical_columns)])
	imputed_scaled_data, _ = impute_scaled_features(scaled_data)
	imputed_data = pd.DataFrame(
		inverse_scale_features(imputed_scaled_data, scaler),
		columns=list(clinical_columns),
		index=data.index,
	)
	capped_data = cap_iqr_outliers(imputed_data)
	return capped_data
