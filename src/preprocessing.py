import cv2
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def denoise_median(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Léger débruitage qui préserve mieux les contours qu'un flou gaussien.
    """
    if image.dtype != np.uint8:
        image_255 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_255 = image.copy()

    return cv2.medianBlur(image_255, ksize)


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8)
) -> np.ndarray:
    """
    Amélioration du contraste local avec CLAHE.
    Très utile ici pour mieux faire ressortir végétation, sol et structures urbaines.
    """
    if image.dtype != np.uint8:
        image_255 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_255 = image.copy()

    lab = cv2.cvtColor(image_255, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return result


def basic_preprocessing(
    image: np.ndarray,
    apply_denoise: bool = True,
    apply_contrast: bool = True
) -> np.ndarray:
    """
    Prétraitement justifiable :
    1. débruitage léger
    2. amélioration du contraste local
    """
    output = image.copy()

    if apply_denoise:
        output = denoise_median(output, ksize=3)

    if apply_contrast:
        output = enhance_contrast_clahe(
            output,
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )

    return output
