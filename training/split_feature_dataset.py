from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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
	train_frame, test_frame = split_dataset(feature_dataset)

	output_directory = PROJECT_ROOT / "extracted_dataset"
	train_path = output_directory / "train.csv"
	test_path = output_directory / "test.csv"

	train_frame.to_csv(train_path, index=False)
	test_frame.to_csv(test_path, index=False)

	print(f"Loaded feature dataset: {len(feature_dataset)} images")
	print_class_distribution("train", train_frame)
	print_class_distribution("test", test_frame)
	print(f"\nSaved training split to: {train_path}")
	print(f"Saved testing split to: {test_path}")


if __name__ == "__main__":
	main()