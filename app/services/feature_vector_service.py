from __future__ import annotations

from app.services.feature_extraction_service import extract_color_features
from app.services.preprocessing_service import PreprocessingResult
from app.services.shape_extraction_service import extract_shape_features
from app.services.texture_extraction_service import extract_texture_features


def extract_feature_vector(preprocessed_image: PreprocessingResult) -> dict[str, float]:
	feature_vector: dict[str, float] = {}
	feature_vector.update(extract_color_features(preprocessed_image))
	feature_vector.update(extract_texture_features(preprocessed_image))
	feature_vector.update(extract_shape_features(preprocessed_image))
	return feature_vector