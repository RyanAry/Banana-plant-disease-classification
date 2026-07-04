from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from training.visualization_utils import save_training_visualization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


RANDOM_STATE = 42


def load_split_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
	data_directory = PROJECT_ROOT / "extracted_dataset"
	train_frame = pd.read_csv(data_directory / "train.csv")
	test_frame = pd.read_csv(data_directory / "test.csv")
	return train_frame, test_frame


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
	return [column for column in dataframe.columns if column not in {"label", "source_path"}]


def build_model() -> GridSearchCV:
	model = RandomForestClassifier(
		random_state=RANDOM_STATE,
		n_jobs=-1,
	)

	param_grid = {
		"n_estimators": [200, 300, 500],
		"max_depth": [None, 20, 30],
		"min_samples_split": [2, 5],
		"min_samples_leaf": [1, 2],
		"max_features": ["sqrt", "log2"],
		"class_weight": [None, "balanced"],
		"bootstrap": [True],
		"criterion": ["gini", "entropy"],
	}

	cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

	return GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		scoring="f1_macro",
		cv=cross_validation,
		n_jobs=-1,
		verbose=1,
		error_score="raise",
	)


def evaluate_model(model: RandomForestClassifier, features: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
	predictions = model.predict(features)

	report = classification_report(labels, predictions, output_dict=True, zero_division=0)
	accuracy = accuracy_score(labels, predictions)
	precision = report["macro avg"]["precision"]
	recall = report["macro avg"]["recall"]
	f1_macro = f1_score(labels, predictions, average="macro")

	metrics = {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1_macro,
	}

	print(f"Test accuracy: {accuracy:.4f}")
	print(f"Test macro precision: {precision:.4f}")
	print(f"Test macro recall: {recall:.4f}")
	print(f"Test macro F1: {f1_macro:.4f}")
	print("\nClassification report:\n")
	print(classification_report(labels, predictions, digits=4))
	print("Confusion matrix:\n")
	print(confusion_matrix(labels, predictions))

	return predictions, metrics


def save_metrics(metrics: dict[str, float], best_params: dict, cv_score: float, output_path: Path) -> None:
	"""Save training metrics to JSON for cross-model comparison."""
	result = {
		"model_name": "Random Forest",
		"best_params": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) for k, v in best_params.items()},
		"best_cv_f1": cv_score,
		"test_metrics": metrics,
	}
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)
	print(f"Saved metrics to: {output_path}")


def main() -> None:
	train_frame, test_frame = load_split_dataset()
	feature_columns = get_feature_columns(train_frame)

	x_train = train_frame[feature_columns]
	y_train = train_frame["label"]
	x_test = test_frame[feature_columns]
	y_test = test_frame["label"]

	grid_search = build_model()
	print("Starting Random Forest tuning...")
	grid_search.fit(x_train, y_train)

	best_model: RandomForestClassifier = grid_search.best_estimator_
	print("\nBest parameters:")
	for key, value in grid_search.best_params_.items():
		print(f"- {key}: {value}")
	print(f"Best cross-validated macro F1: {grid_search.best_score_:.4f}")

	print("\nEvaluating on test set...")
	y_pred, metrics = evaluate_model(best_model, x_test, y_test)

	model_directory = PROJECT_ROOT / "models"
	model_directory.mkdir(parents=True, exist_ok=True)
	model_path = model_directory / "random_forest_model.pkl"
	visualization_path = model_directory / "random_forest_training_results.png"
	metrics_path = model_directory / "random_forest_metrics.json"

	joblib.dump(
		{
			"model": best_model,
			"feature_columns": feature_columns,
			"classes": list(best_model.classes_),
		},
		model_path,
	)

	save_training_visualization(
		"Random Forest",
		y_test.to_numpy(),
		y_pred,
		list(best_model.classes_),
		metrics,
		visualization_path,
	)

	save_metrics(metrics, grid_search.best_params_, grid_search.best_score_, metrics_path)

	print(f"\nSaved trained model to: {model_path}")
	print(f"Saved training visualization to: {visualization_path}")


if __name__ == "__main__":
	main()