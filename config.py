from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str, default_path: Path) -> str:
	path = Path(value)
	if path.is_absolute():
		return str(path)

	return str((BASE_DIR / path).resolve())


class Config:
	SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
	UPLOAD_FOLDER = _resolve_path(os.getenv("UPLOAD_FOLDER", "uploads"), BASE_DIR / "uploads")
	MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(5 * 1024 * 1024)))
	MODEL_PATH = _resolve_path(os.getenv("MODEL_PATH", "models/svm_model.pkl"), BASE_DIR / "models" / "svm_model.pkl")
	TEMPLATES_FOLDER = _resolve_path(os.getenv("TEMPLATES_FOLDER", "app/templates"), BASE_DIR / "app" / "templates")
	STATIC_FOLDER = _resolve_path(os.getenv("STATIC_FOLDER", "app/static"), BASE_DIR / "app" / "static")
	STATIC_URL_PATH = os.getenv("STATIC_URL_PATH", "/static")
