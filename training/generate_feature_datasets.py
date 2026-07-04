from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.services.feature_vector_service import extract_feature_vector
from app.services.preprocessing_service import preprocess_image_path


@dataclass(frozen=True)
class DatasetGenerationResult:
	output_path: Path
	total_files: int
	processed_files: int
	skipped_files: int


def _iter_image_files(dataset_root: Path) -> list[Path]:
	valid_extensions = {".jpg", ".jpeg", ".png"}
	return [path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in valid_extensions]


def _get_label(dataset_root: Path, image_path: Path) -> str:
	return image_path.relative_to(dataset_root).parts[0]


def generate_feature_dataset(dataset_root: Path, output_path: Path) -> DatasetGenerationResult:
	image_paths = _iter_image_files(dataset_root)
	rows: list[dict[str, float | str]] = []
	skipped_files = 0

	print(f"Scanning {dataset_root} ...")
	print(f"Found {len(image_paths)} image files")

	for index, image_path in enumerate(image_paths, start=1):
		label = _get_label(dataset_root, image_path)
		print(f"[{index}/{len(image_paths)}] Processing {image_path.relative_to(dataset_root)}", end="")
		try:
			preprocessed_image = preprocess_image_path(image_path)
			feature_vector = extract_feature_vector(preprocessed_image)
			row: dict[str, float | str] = {
				"label": label,
				"source_path": str(image_path.relative_to(dataset_root)),
			}
			row.update(feature_vector)
			rows.append(row)
			print(" - ok")
		except Exception as error:
			skipped_files += 1
			print(f" - skipped ({error})")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataframe = pd.DataFrame(rows)
	if not dataframe.empty:
		column_order = ["label", "source_path"] + [column for column in dataframe.columns if column not in {"label", "source_path"}]
		dataframe = dataframe[column_order]
	dataframe.to_csv(output_path, index=False)

	print(f"Saved {len(rows)} rows to {output_path}")
	print(f"Skipped {skipped_files} invalid image(s)")

	return DatasetGenerationResult(
		output_path=output_path,
		total_files=len(image_paths),
		processed_files=len(rows),
		skipped_files=skipped_files,
	)


def main() -> None:
	dataset_mappings = {
		PROJECT_ROOT / "dataset" / "OriginalSet": PROJECT_ROOT / "extracted_dataset" / "banana_features_original.csv",
		PROJECT_ROOT / "dataset" / "AugmentedSet": PROJECT_ROOT / "extracted_dataset" / "banana_features_augmented.csv",
	}

	for dataset_root, output_path in dataset_mappings.items():
		if not dataset_root.exists():
			print(f"Skipping missing dataset root: {dataset_root}")
			continue
		generate_feature_dataset(dataset_root, output_path)


if __name__ == "__main__":
	main()