from pathlib import Path
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb