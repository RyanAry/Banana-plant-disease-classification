from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (256, 256)
DEFAULT_BLUR_KERNEL_SIZE: Final[tuple[int, int]] = (5, 5)


@dataclass(frozen=True)
class PreprocessingResult:
	original: np.ndarray
	resized: np.ndarray
	rgb: np.ndarray
	hsv: np.ndarray
	grayscale: np.ndarray
	blurred: np.ndarray
	normalized: np.ndarray


def load_image(image_path: str | Path) -> np.ndarray:
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f'Unable to load image: {image_path}')

	return image


def resize_image(image: np.ndarray, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
	return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: tuple[int, int] = DEFAULT_BLUR_KERNEL_SIZE) -> np.ndarray:
	return cv2.GaussianBlur(image, kernel_size, 0)


def normalize_image(image: np.ndarray) -> np.ndarray:
	return image.astype(np.float32) / 255.0


def preprocess_image_array(
	image: np.ndarray,
	image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
	blur_kernel_size: tuple[int, int] = DEFAULT_BLUR_KERNEL_SIZE,
) -> PreprocessingResult:
	resized = resize_image(image, image_size)
	rgb = convert_to_rgb(resized)
	hsv = convert_to_hsv(rgb)
	grayscale = convert_to_grayscale(rgb)
	blurred = apply_gaussian_blur(rgb, blur_kernel_size)
	normalized = normalize_image(blurred)

	return PreprocessingResult(
		original=image,
		resized=resized,
		rgb=rgb,
		hsv=hsv,
		grayscale=grayscale,
		blurred=blurred,
		normalized=normalized,
	)


def preprocess_image_path(
	image_path: str | Path,
	image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
	blur_kernel_size: tuple[int, int] = DEFAULT_BLUR_KERNEL_SIZE,
) -> PreprocessingResult:
	image = load_image(image_path)
	return preprocess_image_array(image, image_size=image_size, blur_kernel_size=blur_kernel_size)