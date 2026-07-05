from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services.feature_vector_service import extract_feature_vector
from app.services.preprocessing_service import preprocess_image_path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
	disease_label: str
	disease_name: str
	confidence: float
	description: str
	recommendation: str
	class_probabilities: dict[str, float]


class PredictionServiceError(Exception):
	"""Base class for prediction pipeline errors."""


class MissingModelError(PredictionServiceError):
	"""Raised when the trained model file is missing or unreadable."""


class FeatureExtractionError(PredictionServiceError):
	"""Raised when preprocessing or feature extraction fails."""


class PredictionFailureError(PredictionServiceError):
	"""Raised when the model cannot produce a prediction."""


DISEASE_METADATA: dict[str, dict[str, str]] = {
	"healthy": {
		"disease_name": "Sehat",
		"description": "Daun pisang terlihat sehat dan tidak menunjukkan tanda-tanda penyakit yang dilacak.",
		"recommendation": "Lanjutkan pemantauan rutin, irigasi, dan sanitasi lahan untuk menjaga kesehatan tanaman.",
	},
	"cordana": {
		"disease_name": "Cordana",
		"description": "Daun menunjukkan tanda-tanda yang konsisten dengan bercak daun Cordana, yang dapat menurunkan kualitas daun seiring waktu.",
		"recommendation": "Buang daun yang terinfeksi berat, tingkatkan sirkulasi udara, dan terapkan pengelolaan fungisida yang sesuai jika diperlukan.",
	},
	"pestalotiopsis": {
		"disease_name": "Pestalotiopsis",
		"description": "Gambar konsisten dengan infeksi Pestalotiopsis, yang sering dikaitkan dengan bercak daun dan kerusakan jaringan.",
		"recommendation": "Buang sisa tanaman yang terkena, hindari kelembaban daun berlebihan, dan pantau tanaman di sekitarnya untuk penyebaran.",
	},
	"sigatoka": {
		"disease_name": "Black Sigatoka",
		"description": "Model mendeteksi pola yang konsisten dengan Black Sigatoka, penyakit daun pisang yang serius.",
		"recommendation": "Buang daun yang terinfeksi, jaga sanitasi lahan, dan ikuti program fungisida preventif.",
	},
}

DISEASE_DISPLAY_NAMES: dict[str, str] = {
	"healthy": "Sehat",
	"cordana": "Cordana",
	"pestalotiopsis": "Pestalotiopsis",
	"sigatoka": "Black Sigatoka",
}


# ---------- hash-based model cache ----------

_model_cache: dict[str, tuple[str, dict[str, Any]]] = {}


def _file_hash(path: Path) -> str:
	"""Compute a fast hash of the model file for cache invalidation."""
	hasher = hashlib.md5()
	with open(path, "rb") as f:
		# Read in 64KB chunks to avoid loading the entire file into memory
		while chunk := f.read(65_536):
			hasher.update(chunk)
	return hasher.hexdigest()


def load_model_artifact(model_path: str) -> dict[str, Any]:
	"""Load and cache a model artifact. Automatically reloads when the file changes."""
	model_file = Path(model_path)
	if not model_file.exists():
		raise MissingModelError("File model terlatih tidak ditemukan. Silakan latih model terlebih dahulu.")

	try:
		current_hash = _file_hash(model_file)
	except OSError as error:
		raise MissingModelError("File model terlatih tidak dapat dibaca.") from error

	cache_key = str(model_file.resolve())
	cached = _model_cache.get(cache_key)

	if cached is not None:
		cached_hash, cached_artifact = cached
		if cached_hash == current_hash:
			logger.debug("Model cache hit for %s", model_path)
			return cached_artifact
		logger.info("Model file changed on disk, reloading: %s", model_path)

	try:
		logger.info("Loading model from %s", model_path)
		artifact = joblib.load(model_file)
	except Exception as error:
		raise MissingModelError("File model terlatih tidak dapat dimuat. Silakan latih ulang model.") from error

	_model_cache[cache_key] = (current_hash, artifact)
	return artifact


def _get_model_classes(model: Any) -> list:
	"""Extract class labels from a model, handling both Pipeline and standalone estimators."""
	if hasattr(model, "classes_"):
		return list(model.classes_)
	# Pipeline: walk named_steps to find the estimator with classes_
	if hasattr(model, "named_steps"):
		for step_name, step_estimator in model.named_steps.items():
			if hasattr(step_estimator, "classes_"):
				return list(step_estimator.classes_)
	return []


def _compute_all_probabilities(model: Any, features: pd.DataFrame) -> dict[str, float]:
	"""Compute prediction probabilities for ALL classes, returning a dict of {label: probability}."""
	classes = _get_model_classes(model)
	if not classes:
		logger.warning("No classes found in model, returning empty probabilities")
		return {}

	# 1) predict_proba — Random Forest, SVM with probability=True, etc.
	if hasattr(model, "predict_proba"):
		try:
			probabilities = model.predict_proba(features)
			return {str(cls): float(probabilities[0, i]) for i, cls in enumerate(classes)}
		except Exception:
			logger.warning("predict_proba failed, falling back to decision_function")

	# 2) decision_function — SVM without probability=True
	if hasattr(model, "decision_function"):
		try:
			scores = model.decision_function(features)
			scores = np.asarray(scores, dtype=np.float64)
			if scores.ndim == 1:
				scores = scores.reshape(1, -1)
			# softmax
			shifted = scores - np.max(scores, axis=1, keepdims=True)
			exp_values = np.exp(shifted)
			probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
			return {str(cls): float(probabilities[0, i]) for i, cls in enumerate(classes)}
		except Exception:
			logger.warning("decision_function failed")

	# 3) Fallback — equal distribution
	logger.warning("No probability method available, returning uniform distribution")
	uniform_prob = 1.0 / len(classes) if classes else 0.0
	return {str(cls): uniform_prob for cls in classes}


def _compute_confidence(model: Any, features: pd.DataFrame, predicted_label: str) -> float:
	"""Compute prediction confidence supporting predict_proba, decision_function, or fallback."""
	all_probs = _compute_all_probabilities(model, features)
	return all_probs.get(predicted_label, 0.0)


def predict_image(image_path: str | Path, model_path: str) -> PredictionResult:
	logger.info("Starting prediction for image: %s", image_path)

	artifact = load_model_artifact(model_path)
	model = artifact["model"]
	feature_columns = artifact["feature_columns"]

	try:
		logger.info("Preprocessing and extracting features...")
		preprocessed_image = preprocess_image_path(image_path)
		feature_vector = extract_feature_vector(preprocessed_image)
		features = pd.DataFrame([[feature_vector[column] for column in feature_columns]], columns=feature_columns)
	except Exception as error:
		logger.error("Feature extraction failed: %s", error)
		raise FeatureExtractionError("Ekstraksi fitur gagal untuk gambar yang diunggah.") from error

	try:
		logger.info("Running model prediction...")
		predicted_label = model.predict(features)[0]
	except Exception as error:
		logger.error("Prediction failed: %s", error)
		raise PredictionFailureError("Prediksi gagal untuk gambar yang diunggah.") from error

	metadata = DISEASE_METADATA.get(predicted_label, {
		"disease_name": str(predicted_label).title(),
		"description": "Model menghasilkan prediksi untuk gambar daun pisang yang diunggah.",
		"recommendation": "Tinjau gambar dengan pengamatan lapangan dan pertimbangkan verifikasi ahli jika diperlukan.",
	})

	all_probabilities = _compute_all_probabilities(model, features)
	confidence = all_probabilities.get(predicted_label, 0.0)

	# Map raw class labels to display names for the UI
	class_probabilities = {}
	for label, prob in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True):
		display_name = DISEASE_DISPLAY_NAMES.get(label, str(label).title())
		class_probabilities[display_name] = round(prob * 100, 2)

	logger.info("Prediction: %s (confidence: %.4f)", predicted_label, confidence)
	logger.info("All probabilities: %s", class_probabilities)

	return PredictionResult(
		disease_label=str(predicted_label),
		disease_name=metadata["disease_name"],
		confidence=confidence,
		description=metadata["description"],
		recommendation=metadata["recommendation"],
		class_probabilities=class_probabilities,
	)