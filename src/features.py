import cv2
import numpy as np


def rgb_features(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def hsv_features(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image_255 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_255 = image.copy()

    hsv = cv2.cvtColor(image_255, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return hsv


def green_ratio(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    return g / (r + g + b + 1e-6)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image_255 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_255 = image.copy()

    gray = cv2.cvtColor(image_255, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def local_mean(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.blur(gray_image.astype(np.float32), (kernel_size, kernel_size))


def local_variance(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    gray = gray_image.astype(np.float32)
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    return mean_sq - mean ** 2


def build_feature_stack(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    rgb = rgb_features(image)
    hsv = hsv_features(image)
    g_ratio = green_ratio(image)

    gray = rgb_to_gray(image)
    mean_local = local_mean(gray, kernel_size=kernel_size)
    var_local = local_variance(gray, kernel_size=kernel_size)

    feature_stack = np.dstack([
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        g_ratio,
        mean_local,
        var_local
    ])

    return feature_stack