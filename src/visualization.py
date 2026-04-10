from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _save_current_figure(output_path=None):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def show_image(image: np.ndarray, title: str = "Image", figsize=(6, 6), output_path=None) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    _save_current_figure(output_path)


def show_two_images(
    img1: np.ndarray,
    img2: np.ndarray,
    title1="t0",
    title2="t1",
    figsize=(12, 6),
    output_path=None
) -> None:
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis("off")

    _save_current_figure(output_path)


def plot_rgb_histogram(image: np.ndarray, title: str = "Histogramme RGB", output_path=None) -> None:
    plt.figure(figsize=(8, 5))

    colors = ["r", "g", "b"]
    labels = ["Rouge", "Vert", "Bleu"]

    for i, (color, label) in enumerate(zip(colors, labels)):
        hist, _ = np.histogram(image[:, :, i].ravel(), bins=256, range=(0, 255))
        plt.plot(hist, color=color, label=label)

    plt.title(title)
    plt.xlabel("Intensité")
    plt.ylabel("Nombre de pixels")
    plt.legend()
    plt.grid(alpha=0.3)

    _save_current_figure(output_path)


def show_map(gray_or_mask: np.ndarray, title: str = "Carte", cmap: str = "gray", figsize=(6, 6), output_path=None) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(gray_or_mask, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    _save_current_figure(output_path)