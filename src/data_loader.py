import pandas as pd


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


def load_data(path):
	return pd.read_csv(path)
