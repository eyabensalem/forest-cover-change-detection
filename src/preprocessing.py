"""Utilitaires partages pour charger les images et extraire les features."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy import ndimage as ndi

EPSILON = 1e-6
BASE_FEATURE_NAMES = ("r", "g", "b")


def load_image(image_path: str | Path) -> np.ndarray:
    """Charge une image RGB et la normalise dans [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_green_ratio(image: np.ndarray) -> np.ndarray:
    """Calcule G / (R + G + B) pixel par pixel."""
    _validate_rgb_image(image)
    channel_sum = np.clip(image.sum(axis=2), EPSILON, None)
    return image[:, :, 1] / channel_sum


def compute_local_statistics(image: np.ndarray, window_size: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Retourne la moyenne locale et la variance locale sur l'intensite."""
    _validate_rgb_image(image)
    if window_size < 1:
        raise ValueError("window_size doit etre >= 1.")

    intensity = image.mean(axis=2)
    local_mean = ndi.uniform_filter(intensity, size=window_size, mode="reflect")
    local_sq_mean = ndi.uniform_filter(intensity**2, size=window_size, mode="reflect")
    local_variance = np.clip(local_sq_mean - local_mean**2, 0.0, None)
    return local_mean.astype(np.float32), local_variance.astype(np.float32)


def extract_features(
    image: np.ndarray,
    include_hsv: bool = True,
    include_green_ratio: bool = True,
    include_local_stats: bool = True,
    window_size: int = 7,
    flatten: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Extrait un tenseur de features coherent pour le pipeline.

    Contrat recommande pour le projet:
    - entree: image RGB normalisee de shape (H, W, 3)
    - sortie: features de shape (H, W, F) ou (H*W, F)
    """
    _validate_rgb_image(image)

    feature_blocks: list[np.ndarray] = [image.astype(np.float32)]
    feature_names = list(BASE_FEATURE_NAMES)

    if include_hsv:
        hsv = rgb_to_hsv(np.clip(image, 0.0, 1.0)).astype(np.float32)
        feature_blocks.append(hsv)
        feature_names.extend(["h", "s", "v"])

    if include_green_ratio:
        green_ratio = compute_green_ratio(image).astype(np.float32)[..., None]
        feature_blocks.append(green_ratio)
        feature_names.append("green_ratio")

    if include_local_stats:
        local_mean, local_variance = compute_local_statistics(image, window_size=window_size)
        feature_blocks.append(local_mean[..., None])
        feature_blocks.append(local_variance[..., None])
        feature_names.extend(["local_mean", "local_variance"])

    features = np.concatenate(feature_blocks, axis=2).astype(np.float32)
    if flatten:
        features = features.reshape(-1, features.shape[-1])

    return features, feature_names


def describe_feature_set(feature_names: Sequence[str]) -> str:
    """Retourne une description courte utile pour logs et README."""
    return ", ".join(feature_names)


def _validate_rgb_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit etre de shape (H, W, 3).")
