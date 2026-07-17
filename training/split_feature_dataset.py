from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_STATE = 42


def load_feature_dataset() -> pd.DataFrame:
	data_directory = PROJECT_ROOT / "extracted_dataset"
	feature_files = [
		data_directory / "banana_features_original.csv",
		data_directory / "banana_features_augmented.csv",
	]
	frames = [pd.read_csv(path) for path in feature_files]
	return pd.concat(frames, ignore_index=True)


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
	return [column for column in dataframe.columns if column not in {"label", "source_path"}]


def normalize_features(dataframe: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
	"""Normalize feature columns to 0-1 range using MinMaxScaler."""
	scaler = MinMaxScaler()
	normalized_dataframe = dataframe.copy()
	normalized_dataframe[feature_columns] = scaler.fit_transform(dataframe[feature_columns])
	return normalized_dataframe, scaler


def split_dataset(feature_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	train_frame, test_frame = train_test_split(
		feature_dataset,
		test_size=TEST_RATIO,
		train_size=TRAIN_RATIO,
		stratify=feature_dataset["label"],
		random_state=RANDOM_STATE,
	)
	return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def print_class_distribution(name: str, dataframe: pd.DataFrame) -> None:
	print(f"\n{name.upper()} CLASS DISTRIBUTION")
	print(dataframe["label"].value_counts().sort_index().to_string())
	print(f"Total images: {len(dataframe)}")


def main() -> None:
	feature_dataset = load_feature_dataset()
	feature_columns = get_feature_columns(feature_dataset)

	print(f"Loaded feature dataset: {len(feature_dataset)} images")
	print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

	# Normalisasi fitur ke rentang 0-1
	print("\nNormalizing features using MinMaxScaler...")
	normalized_dataset, scaler = normalize_features(feature_dataset, feature_columns)

	print("Before normalization (sample range):")
	print(f"  area:    min={feature_dataset['area'].min():.2f}, max={feature_dataset['area'].max():.2f}")
	print(f"  mean_r:  min={feature_dataset['mean_r'].min():.2f}, max={feature_dataset['mean_r'].max():.2f}")
	print(f"  energy:  min={feature_dataset['energy'].min():.6f}, max={feature_dataset['energy'].max():.6f}")
	print("After normalization (sample range):")
	print(f"  area:    min={normalized_dataset['area'].min():.2f}, max={normalized_dataset['area'].max():.2f}")
	print(f"  mean_r:  min={normalized_dataset['mean_r'].min():.2f}, max={normalized_dataset['mean_r'].max():.2f}")
	print(f"  energy:  min={normalized_dataset['energy'].min():.6f}, max={normalized_dataset['energy'].max():.6f}")

	# Split dataset 80% training, 20% testing
	train_frame, test_frame = split_dataset(normalized_dataset)

	output_directory = PROJECT_ROOT / "extracted_dataset"
	train_path = output_directory / "train.csv"
	test_path = output_directory / "test.csv"
	scaler_path = output_directory / "scaler.pkl"

	train_frame.to_csv(train_path, index=False)
	test_frame.to_csv(test_path, index=False)
	joblib.dump(scaler, scaler_path)

	print_class_distribution("train", train_frame)
	print_class_distribution("test", test_frame)
	print(f"\nSaved training split to: {train_path}")
	print(f"Saved testing split to: {test_path}")
	print(f"Saved scaler to: {scaler_path}")


if __name__ == "__main__":
	main()