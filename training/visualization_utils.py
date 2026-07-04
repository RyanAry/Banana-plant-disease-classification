from __future__ import annotations

from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_training_visualization(
	model_name: str,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_labels: list[str],
	metrics: dict[str, float],
	output_path: Path,
) -> None:
	cm = confusion_matrix(y_true, y_pred, labels=class_labels)
	figure, axes = plt.subplots(1, 2, figsize=(16, 6))
	figure.suptitle(f"{model_name} Training Results", fontsize=16, fontweight="bold")

	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(
		ax=axes[0],
		cmap="Blues",
		colorbar=False,
		values_format="d",
	)
	axes[0].set_title("Confusion Matrix")
	axes[0].tick_params(axis="x", rotation=45)

	metric_names = list(metrics.keys())
	metric_values = [metrics[name] for name in metric_names]
	metric_colors = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed"]
	axes[1].bar(metric_names, metric_values, color=metric_colors[: len(metric_names)])
	axes[1].set_ylim(0, 1.0)
	axes[1].set_title("Evaluation Metrics")
	axes[1].set_ylabel("Score")
	for index, value in enumerate(metric_values):
		axes[1].text(index, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=10)

	figure.tight_layout(rect=(0, 0, 1, 0.95))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(figure)