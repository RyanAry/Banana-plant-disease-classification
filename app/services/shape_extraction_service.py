from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from app.services.preprocessing_service import PreprocessingResult


DEFAULT_THRESHOLD_FLAGS: Final[int] = cv2.THRESH_BINARY + cv2.THRESH_OTSU


def _find_largest_contour(binary_image: np.ndarray) -> np.ndarray | None:
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	return max(contours, key=cv2.contourArea)


def extract_shape_features(preprocessed_image: PreprocessingResult) -> dict[str, float]:
	grayscale_image = preprocessed_image.grayscale
	_, binary_image = cv2.threshold(grayscale_image, 0, 255, DEFAULT_THRESHOLD_FLAGS)
	contour = _find_largest_contour(binary_image)

	if contour is None:
		return {
			"area": 0.0,
			"perimeter": 0.0,
			"aspect_ratio": 0.0,
			"circularity": 0.0,
			"solidity": 0.0,
			"extent": 0.0,
		}

	area = float(cv2.contourArea(contour))
	perimeter = float(cv2.arcLength(contour, True))
	x, y, width, height = cv2.boundingRect(contour)
	aspect_ratio = float(width / height) if height > 0 else 0.0
	convex_hull = cv2.convexHull(contour)
	convex_hull_area = float(cv2.contourArea(convex_hull))
	bbox_area = float(width * height)
	circularity = float((4.0 * np.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
	solidity = float(area / convex_hull_area) if convex_hull_area > 0 else 0.0
	extent = float(area / bbox_area) if bbox_area > 0 else 0.0

	return {
		"area": area,
		"perimeter": perimeter,
		"aspect_ratio": aspect_ratio,
		"circularity": circularity,
		"solidity": solidity,
		"extent": extent,
	}