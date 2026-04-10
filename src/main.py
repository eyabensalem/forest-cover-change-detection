from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import numpy as np
import cv2
import matplotlib.pyplot as plt

from io_utils import load_image
from visualization import show_two_images, plot_rgb_histogram, show_map
from preprocessing import basic_preprocessing
from features import (
    green_ratio,
    rgb_to_gray,
    local_mean,
    local_variance,
    build_feature_stack,
)


def save_figure(array, output_path, cmap=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    if cmap:
        plt.imshow(array, cmap=cmap)
    else:
        plt.imshow(array)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def filter_small_components(mask: np.ndarray, min_size: int = 150) -> np.ndarray:
    """
    Supprime les petites composantes connexes.
    mask doit être en uint8 avec valeurs 0 ou 255.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean_mask[labels == i] = 255

    return clean_mask


def main():
    base_dir = Path(__file__).resolve().parent.parent

    t0_path = base_dir / "data" / "t0.png"
    t1_path = base_dir / "data" / "t1.png"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement des images...")
    img_t0 = load_image(str(t0_path))
    img_t1 = load_image(str(t1_path))

    print("Sauvegarde des images initiales...")
    show_two_images(
        img_t0,
        img_t1,
        title1="Image t0",
        title2="Image t1",
        output_path=outputs_dir / "01_images_initiales.png"
    )

    print("Sauvegarde des histogrammes RGB...")
    plot_rgb_histogram(
        img_t0,
        title="Histogramme RGB - t0",
        output_path=outputs_dir / "02_histogramme_t0.png"
    )
    plot_rgb_histogram(
        img_t1,
        title="Histogramme RGB - t1",
        output_path=outputs_dir / "03_histogramme_t1.png"
    )

    print("Prétraitement utile : débruitage léger + contraste local...")
    img_t0_prep = basic_preprocessing(img_t0, apply_denoise=True, apply_contrast=True)
    img_t1_prep = basic_preprocessing(img_t1, apply_denoise=True, apply_contrast=True)

    show_two_images(
        img_t0_prep,
        img_t1_prep,
        title1="t0 prétraitée",
        title2="t1 prétraitée",
        output_path=outputs_dir / "04_images_pretraitees.png"
    )

    print("Calcul du ratio vert...")
    ratio_t0 = green_ratio(img_t0_prep)
    ratio_t1 = green_ratio(img_t1_prep)

    show_map(
        ratio_t0,
        title="Ratio vert - t0",
        cmap="Greens",
        output_path=outputs_dir / "05_ratio_vert_t0.png"
    )
    show_map(
        ratio_t1,
        title="Ratio vert - t1",
        cmap="Greens",
        output_path=outputs_dir / "06_ratio_vert_t1.png"
    )

    print("Calcul des features locales...")
    gray_t0 = rgb_to_gray(img_t0_prep)
    gray_t1 = rgb_to_gray(img_t1_prep)

    mean_t0 = local_mean(gray_t0, kernel_size=5)
    mean_t1 = local_mean(gray_t1, kernel_size=5)

    var_t0 = local_variance(gray_t0, kernel_size=5)
    var_t1 = local_variance(gray_t1, kernel_size=5)

    show_map(
        mean_t0,
        title="Moyenne locale - t0",
        cmap="gray",
        output_path=outputs_dir / "07_moyenne_locale_t0.png"
    )
    show_map(
        mean_t1,
        title="Moyenne locale - t1",
        cmap="gray",
        output_path=outputs_dir / "08_moyenne_locale_t1.png"
    )

    show_map(
        var_t0,
        title="Variance locale - t0",
        cmap="gray",
        output_path=outputs_dir / "09_variance_locale_t0.png"
    )
    show_map(
        var_t1,
        title="Variance locale - t1",
        cmap="gray",
        output_path=outputs_dir / "10_variance_locale_t1.png"
    )

    print("Construction du feature stack...")
    features_t0 = build_feature_stack(img_t0_prep, kernel_size=5)
    features_t1 = build_feature_stack(img_t1_prep, kernel_size=5)

    print(f"Shape features t0 : {features_t0.shape}")
    print(f"Shape features t1 : {features_t1.shape}")

    print("Calcul de la carte de changement...")
    diff_ratio = np.abs(ratio_t1 - ratio_t0)
    diff_mean = np.abs(mean_t1 - mean_t0)
    diff_var = np.abs(var_t1 - var_t0)

    diff_ratio = normalize_map(diff_ratio)
    diff_mean = normalize_map(diff_mean)
    diff_var = normalize_map(diff_var)

    # Le ratio vert est le signal le plus utile sur tes images.
    # On réduit le poids de la variance car elle est plus bruitée.
    change_map = 0.6 * diff_ratio + 0.2 * diff_mean + 0.2 * diff_var

    # Léger lissage pour éviter une carte trop granuleuse
    change_map = cv2.GaussianBlur(change_map, (5, 5), 0)
    change_map = normalize_map(change_map)

    show_map(
        change_map,
        title="Carte de changement globale",
        cmap="hot",
        output_path=outputs_dir / "11_change_map.png"
    )

    print("Création du masque de changement...")
    threshold = np.percentile(change_map, 90)
    change_mask = (change_map > threshold).astype(np.uint8) * 255

    show_map(
        change_mask,
        title="Masque de changement brut",
        cmap="gray",
        output_path=outputs_dir / "12_change_mask_brut.png"
    )

    # Nettoyage morphologique plus fort
    kernel = np.ones((5, 5), np.uint8)
    change_mask_clean = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
    change_mask_clean = cv2.morphologyEx(change_mask_clean, cv2.MORPH_CLOSE, kernel)

    # Suppression des petits objets isolés
    change_mask_clean = filter_small_components(change_mask_clean, min_size=80)

    show_map(
        change_mask_clean,
        title="Masque de changement nettoyé",
        cmap="gray",
        output_path=outputs_dir / "13_change_mask_nettoye.png"
    )

    changed_pixels = np.sum(change_mask_clean > 0)
    total_pixels = change_mask_clean.shape[0] * change_mask_clean.shape[1]
    changed_ratio = changed_pixels / total_pixels

    print(f"Pixels changés : {changed_pixels}")
    print(f"Proportion de changement : {changed_ratio:.4f}")

    print("Sauvegarde complémentaire...")
    save_figure(img_t0, outputs_dir / "14_t0_original.png")
    save_figure(img_t1, outputs_dir / "15_t1_original.png")
    save_figure(img_t0_prep, outputs_dir / "16_t0_preprocessed.png")
    save_figure(img_t1_prep, outputs_dir / "17_t1_preprocessed.png")

    print("Pipeline première partie + changement terminé avec succès.")
    print(f"Résultats enregistrés dans : {outputs_dir}")


if __name__ == "__main__":
    main()