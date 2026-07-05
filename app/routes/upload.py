from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, render_template, request, send_from_directory

from app.services.prediction_service import (
	FeatureExtractionError,
	MissingModelError,
	PredictionFailureError,
	predict_image,
)
from app.services.upload_service import save_uploaded_image, validate_image_upload


logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


@bp.route("/", methods=["GET", "POST"])
def index():
	context = {
		"error_message": None,
		"success_message": None,
		"uploaded_image": None,
		"prediction_result": None,
	}

	def set_error(message: str, status_code: int = 400):
		context["error_message"] = message
		return render_template("index.html", **context), status_code

	if request.method == "POST":
		logger.info("Received POST request for image upload")
		file_storage = request.files.get("image")
		is_valid, error_message = validate_image_upload(file_storage) if file_storage is not None else (False, "Silakan pilih file gambar.")

		if not is_valid:
			logger.warning("Validation failed: %s", error_message)
			return set_error(error_message or "Silakan pilih file gambar.")

		image_path = None
		try:
			filename = save_uploaded_image(file_storage, current_app.config["UPLOAD_FOLDER"])
			image_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
			logger.info("Image saved: %s", image_path)

			prediction_result = predict_image(image_path, current_app.config["MODEL_PATH"])
			logger.info("Prediction complete: %s (confidence: %.4f)", prediction_result.disease_label, prediction_result.confidence)
		except MissingModelError as error:
			logger.error("Missing model error: %s", error)
			_cleanup_uploaded_file(image_path)
			return set_error(str(error), 500)
		except FeatureExtractionError as error:
			logger.error("Feature extraction error: %s", error)
			_cleanup_uploaded_file(image_path)
			return set_error(str(error), 422)
		except PredictionFailureError as error:
			logger.error("Prediction failure: %s", error)
			_cleanup_uploaded_file(image_path)
			return set_error(str(error), 500)
		except Exception:
			logger.exception("Unexpected error during prediction")
			_cleanup_uploaded_file(image_path)
			return set_error("Terjadi kesalahan tak terduga saat memproses gambar.", 500)

		context["success_message"] = "Gambar berhasil diunggah."
		context["uploaded_image"] = filename
		context["prediction_result"] = prediction_result

	return render_template("index.html", **context)


@bp.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
	return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


def _cleanup_uploaded_file(image_path: Path | None) -> None:
	"""Remove the uploaded file on prediction error to avoid orphaned files."""
	if image_path is not None and image_path.exists():
		try:
			image_path.unlink()
			logger.info("Cleaned up uploaded file: %s", image_path)
		except OSError:
			logger.warning("Failed to clean up uploaded file: %s", image_path)