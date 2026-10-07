import joblib
import pandas as pd


def save_dataframe(data, path):
	data.to_csv(path, index=False)


def save_model(model, path):
	joblib.dump(model, path)


def load_model(path):
	return joblib.load(path)
