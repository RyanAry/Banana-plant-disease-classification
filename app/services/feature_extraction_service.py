from __future__ import annotations

from typing import Final

import numpy as np

from app.services.preprocessing_service import PreprocessingResult


COLOR_CHANNEL_NAMES: Final[tuple[str, str, str]] = ("r", "g", "b")
HSV_CHANNEL_NAMES: Final[tuple[str, str, str]] = ("h", "s", "v")


def _channel_mean(channel: np.ndarray) -> float:
	return float(np.mean(channel))


def _channel_std(channel: np.ndarray) -> float:
	return float(np.std(channel))


def extract_color_features(preprocessed_image: PreprocessingResult) -> dict[str, float]:
	rgb_image = preprocessed_image.rgb
	hsv_image = preprocessed_image.hsv

	features: dict[str, float] = {}

	for index, channel_name in enumerate(COLOR_CHANNEL_NAMES):
		channel = rgb_image[:, :, index]
		features[f"mean_{channel_name}"] = _channel_mean(channel)
		features[f"std_{channel_name}"] = _channel_std(channel)

	for index, channel_name in enumerate(HSV_CHANNEL_NAMES):
		channel = hsv_image[:, :, index]
		features[f"mean_{channel_name}"] = _channel_mean(channel)

	return features