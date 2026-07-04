from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


MODEL_REGISTRY: dict[str, str] = {
	"SVM": "svm_model.pkl",
	"Random Forest": "random_forest_model.pkl",
}


def load_all_metrics(models_dir: Path) -> list[dict]:
	"""Load all *_metrics.json files from the models directory."""
	metrics_files = sorted(models_dir.glob("*_metrics.json"))
	results = []
	for path in metrics_files:
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			data["_source_file"] = path.name
			results.append(data)
		except Exception as e:
			print(f"Warning: Could not load {path.name}: {e}")
	return results


def print_comparison_table(all_metrics: list[dict]) -> None:
	"""Print a formatted comparison table of all models."""
	header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Macro':>10} {'CV F1':>10}"
	separator = "-" * len(header)

	print("\n" + separator)
	print("MODEL COMPARISON RESULTS")
	print(separator)
	print(header)
	print(separator)

	for entry in all_metrics:
		name = entry.get("model_name", "Unknown")
		tm = entry.get("test_metrics", {})
		cv_f1 = entry.get("best_cv_f1", 0.0)
		print(
			f"{name:<20} "
			f"{tm.get('accuracy', 0.0):>10.4f} "
			f"{tm.get('precision', 0.0):>10.4f} "
			f"{tm.get('recall', 0.0):>10.4f} "
			f"{tm.get('f1', 0.0):>10.4f} "
			f"{cv_f1:>10.4f}"
		)

	print(separator)


def find_best_model(all_metrics: list[dict]) -> dict:
	"""Return the model entry with the highest test accuracy."""
	return max(all_metrics, key=lambda m: m.get("test_metrics", {}).get("accuracy", 0.0))


def update_env_file(env_path: Path, model_pkl_filename: str) -> None:
	"""Update MODEL_PATH in the .env file to point to the best model."""
	new_model_path = f"models/{model_pkl_filename}"

	if env_path.exists():
		lines = env_path.read_text(encoding="utf-8").splitlines()
		updated = False
		new_lines = []
		for line in lines:
			if line.startswith("MODEL_PATH="):
				new_lines.append(f"MODEL_PATH={new_model_path}")
				updated = True
			else:
				new_lines.append(line)
		if not updated:
			new_lines.append(f"MODEL_PATH={new_model_path}")
		env_path.write_text("\r\n".join(new_lines) + "\r\n", encoding="utf-8")
	else:
		env_path.write_text(f"MODEL_PATH={new_model_path}\r\n", encoding="utf-8")


def main() -> None:
	models_dir = PROJECT_ROOT / "models"
	all_metrics = load_all_metrics(models_dir)

	if not all_metrics:
		print("No metrics files found in models/. Please train models first.")
		print("  python -m training.train_svm")
		print("  python -m training.train_random_forest")
		return

	print_comparison_table(all_metrics)

	best = find_best_model(all_metrics)
	best_name = best.get("model_name", "Unknown")
	best_acc = best.get("test_metrics", {}).get("accuracy", 0.0)

	print(f"\n>>> Best model: {best_name} (accuracy: {best_acc:.4f})")

	pkl_filename = MODEL_REGISTRY.get(best_name)
	if pkl_filename is None:
		print(f"Warning: No .pkl mapping found for model '{best_name}'. Skipping .env update.")
		return

	model_file = models_dir / pkl_filename
	if not model_file.exists():
		print(f"Warning: Model file {model_file} not found. Skipping .env update.")
		return

	env_path = PROJECT_ROOT / ".env"
	update_env_file(env_path, pkl_filename)
	print(f">>> Updated .env: MODEL_PATH=models/{pkl_filename}")

	# Print best parameters
	best_params = best.get("best_params", {})
	if best_params:
		print(f"\nBest parameters for {best_name}:")
		for key, value in best_params.items():
			print(f"  {key}: {value}")


if __name__ == "__main__":
	main()
