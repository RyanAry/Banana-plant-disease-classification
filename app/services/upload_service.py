from __future__ import annotations

from pathlib import Path
from typing import Final
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS: Final[set[str]] = {"jpg", "jpeg", "png"}
ALLOWED_MIME_TYPES: Final[set[str]] = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_FORMATS: Final[set[str]] = {"JPEG", "PNG"}


def is_allowed_extension(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_allowed_mimetype(mimetype: str | None) -> bool:
	return mimetype in ALLOWED_MIME_TYPES


def validate_image_upload(file_storage: FileStorage) -> tuple[bool, str | None]:
	if not file_storage.filename:
		return False, "Please select an image file."

	if not is_allowed_extension(file_storage.filename):
		return False, "Only JPG, JPEG, and PNG files are allowed."

	if not is_allowed_mimetype(file_storage.mimetype):
		return False, "Invalid MIME type. Please upload a JPEG or PNG image."

	try:
		image = Image.open(file_storage.stream)
		image_format = image.format
		image.verify()
		file_storage.stream.seek(0)
	except UnidentifiedImageError:
		file_storage.stream.seek(0)
		return False, "The uploaded file is not a valid image."
	except Exception:
		file_storage.stream.seek(0)
		return False, "Unable to read the uploaded image."

	if image_format not in ALLOWED_IMAGE_FORMATS:
		return False, "The uploaded image format is not supported."

	return True, None


def save_uploaded_image(file_storage: FileStorage, upload_folder: str) -> str:
	filename = secure_filename(file_storage.filename or "")
	extension = Path(filename).suffix.lower()
	base_name = Path(filename).stem or "image"
	safe_filename = f"{base_name}_{uuid4().hex}{extension}"
	upload_path = Path(upload_folder)
	upload_path.mkdir(parents=True, exist_ok=True)
	file_storage.save(upload_path / safe_filename)
	return safe_filename