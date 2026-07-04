from __future__ import annotations

from typing import Final

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from app.services.preprocessing_service import PreprocessingResult


DEFAULT_DISTANCES: Final[tuple[int, ...]] = (1,)
DEFAULT_ANGLES: Final[tuple[float, ...]] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_LEVELS: Final[int] = 256


def extract_texture_features(preprocessed_image: PreprocessingResult) -> dict[str, float]:
	grayscale_image = preprocessed_image.grayscale
	gray_levels = grayscale_image.astype(np.uint8)

	glcm = graycomatrix(
		gray_levels,
		distances=DEFAULT_DISTANCES,
		angles=DEFAULT_ANGLES,
		levels=DEFAULT_LEVELS,
		symmetric=True,
		normed=True,
	)

	return {
		"contrast": float(graycoprops(glcm, "contrast").mean()),
		"correlation": float(graycoprops(glcm, "correlation").mean()),
		"energy": float(graycoprops(glcm, "energy").mean()),
		"homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
		"asm": float(graycoprops(glcm, "ASM").mean()),
		"dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
	}