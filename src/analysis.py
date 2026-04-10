"""Analyse de changement entre deux dates et generation des sorties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.clustering import save_dendrogram, segment_image
    from src.preprocessing import describe_feature_set, extract_features, load_image
else:
    from .clustering import save_dendrogram, segment_image
    from .preprocessing import describe_feature_set, extract_features, load_image

DEFAULT_T0_PATH = Path("outputs/preprocessed_images/16_t0_preprocessed.png")
DEFAULT_T1_PATH = Path("outputs/preprocessed_images/17_t1_preprocessed.png")
DEFAULT_OUTPUT_DIR = Path("outputs")


def clean_mask(
    mask: np.ndarray,
    opening_radius: int = 2,
    closing_radius: int = 3,
    min_component_size: int = 64,
) -> np.ndarray:
    """Nettoie un masque binaire avec ouverture, fermeture et suppression de petits composants."""
    structure_open = _disk(opening_radius)
    structure_close = _disk(closing_radius)

    cleaned = ndi.binary_opening(mask, structure=structure_open)
    cleaned = ndi.binary_closing(cleaned, structure=structure_close)
    cleaned = ndi.binary_fill_holes(cleaned)
    cleaned = _remove_small_components(cleaned, min_component_size=min_component_size)
    return cleaned.astype(bool)


def compute_change_metrics(mask_t0: np.ndarray, mask_t1: np.ndarray) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Calcule les indicateurs de changement entre deux dates."""
    if mask_t0.shape != mask_t1.shape:
        raise ValueError("Les deux masques doivent avoir la meme shape.")

    total_pixels = int(mask_t0.size)
    veg_t0 = int(mask_t0.sum())
    veg_t1 = int(mask_t1.sum())

    loss_mask = mask_t0 & ~mask_t1
    gain_mask = ~mask_t0 & mask_t1

    net_loss_ratio = float((veg_t0 - veg_t1) / veg_t0) if veg_t0 else 0.0
    direct_loss_ratio = float(loss_mask.sum() / veg_t0) if veg_t0 else 0.0

    metrics: dict[str, float | int] = {
        "total_pixels": total_pixels,
        "vegetation_pixels_t0": veg_t0,
        "vegetation_pixels_t1": veg_t1,
        "vegetation_fraction_t0": float(veg_t0 / total_pixels),
        "vegetation_fraction_t1": float(veg_t1 / total_pixels),
        "lost_pixels": int(loss_mask.sum()),
        "gained_pixels": int(gain_mask.sum()),
        "net_change_pixels": int(veg_t1 - veg_t0),
        "net_loss_ratio_formula": net_loss_ratio,
        "direct_loss_ratio": direct_loss_ratio,
    }
    return metrics, loss_mask, gain_mask


def create_change_map(mask_t0: np.ndarray, mask_t1: np.ndarray) -> np.ndarray:
    """Construit une carte RGB des changements."""
    stable_vegetation = mask_t0 & mask_t1
    loss_mask = mask_t0 & ~mask_t1
    gain_mask = ~mask_t0 & mask_t1

    change_map = np.zeros(mask_t0.shape + (3,), dtype=np.float32)
    change_map[stable_vegetation] = (0.18, 0.62, 0.24)
    change_map[loss_mask] = (0.86, 0.2, 0.2)
    change_map[gain_mask] = (0.2, 0.45, 0.85)
    return change_map


def run_change_analysis(
    image_path_t0: str | Path = DEFAULT_T0_PATH,
    image_path_t1: str | Path = DEFAULT_T1_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_clusters: int = 4,
    include_hsv: bool = True,
    include_green_ratio: bool = True,
    include_local_stats: bool = True,
    window_size: int = 7,
    dendrogram_sample_size: int = 1200,
    opening_radius: int = 2,
    closing_radius: int = 3,
    min_component_size: int = 64,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Execute tout le pipeline sur les deux images et sauvegarde les sorties."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_t0 = load_image(image_path_t0)
    image_t1 = load_image(image_path_t1)

    clustering_t0 = segment_image(
        image_t0,
        n_clusters=n_clusters,
        include_hsv=include_hsv,
        include_green_ratio=include_green_ratio,
        include_local_stats=include_local_stats,
        window_size=window_size,
        random_state=random_state,
    )
    clustering_t1 = segment_image(
        image_t1,
        n_clusters=n_clusters,
        include_hsv=include_hsv,
        include_green_ratio=include_green_ratio,
        include_local_stats=include_local_stats,
        window_size=window_size,
        random_state=random_state,
    )

    cleaned_mask_t0 = clean_mask(
        clustering_t0.vegetation_mask,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        min_component_size=min_component_size,
    )
    cleaned_mask_t1 = clean_mask(
        clustering_t1.vegetation_mask,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        min_component_size=min_component_size,
    )

    metrics, loss_mask, gain_mask = compute_change_metrics(cleaned_mask_t0, cleaned_mask_t1)
    change_map = create_change_map(cleaned_mask_t0, cleaned_mask_t1)

    feature_matrix_t0, feature_names = extract_features(
        image_t0,
        include_hsv=include_hsv,
        include_green_ratio=include_green_ratio,
        include_local_stats=include_local_stats,
        window_size=window_size,
        flatten=True,
    )
    feature_matrix_t1, _ = extract_features(
        image_t1,
        include_hsv=include_hsv,
        include_green_ratio=include_green_ratio,
        include_local_stats=include_local_stats,
        window_size=window_size,
        flatten=True,
    )

    save_dendrogram(
        feature_matrix_t0,
        output_dir / "dendrogram_t0.png",
        sample_size=dendrogram_sample_size,
        random_state=random_state,
    )
    save_dendrogram(
        feature_matrix_t1,
        output_dir / "dendrogram_t1.png",
        sample_size=dendrogram_sample_size,
        random_state=random_state,
    )

    _save_rgb_image(clustering_t0.segmented_rgb, output_dir / "segmentation_t0.png")
    _save_rgb_image(clustering_t1.segmented_rgb, output_dir / "segmentation_t1.png")
    _save_mask(cleaned_mask_t0, output_dir / "vegetation_mask_t0.png")
    _save_mask(cleaned_mask_t1, output_dir / "vegetation_mask_t1.png")
    _save_rgb_image(change_map, output_dir / "deforestation_map.png")

    summary_path = output_dir / "analysis_summary.png"
    _save_summary_figure(
        image_t0=image_t0,
        image_t1=image_t1,
        segmentation_t0=clustering_t0.segmented_rgb,
        segmentation_t1=clustering_t1.segmented_rgb,
        mask_t0=cleaned_mask_t0,
        mask_t1=cleaned_mask_t1,
        change_map=change_map,
        output_path=summary_path,
    )

    metrics.update(
        {
            "image_t0": _normalize_output_path(image_path_t0),
            "image_t1": _normalize_output_path(image_path_t1),
            "t0_label": "etat avant",
            "t1_label": "etat apres",
            "feature_set": describe_feature_set(feature_names),
            "vegetation_cluster_t0": int(clustering_t0.vegetation_cluster),
            "vegetation_cluster_t1": int(clustering_t1.vegetation_cluster),
            "vegetation_scores_t0": [float(x) for x in clustering_t0.vegetation_scores],
            "vegetation_scores_t1": [float(x) for x in clustering_t1.vegetation_scores],
            "n_clusters": int(n_clusters),
            "summary_figure": _normalize_output_path(summary_path),
            "interpretation_note": "La carte de changement est interpretable uniquement si t0 et t1 couvrent bien la meme zone et sont spatialement comparables.",
        }
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _save_summary_figure(
    image_t0: np.ndarray,
    image_t1: np.ndarray,
    segmentation_t0: np.ndarray,
    segmentation_t1: np.ndarray,
    mask_t0: np.ndarray,
    mask_t1: np.ndarray,
    change_map: np.ndarray,
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    panels = [
        (image_t0, "Image t0"),
        (image_t1, "Image t1"),
        (segmentation_t0, "Segmentation t0"),
        (segmentation_t1, "Segmentation t1"),
        (mask_t0, "Vegetation t0"),
        (mask_t1, "Vegetation t1"),
        (change_map, "Carte de changement"),
    ]

    for ax, (panel, title) in zip(axes.flat, panels):
        cmap = "gray" if panel.ndim == 2 else None
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    legend_ax = axes[1, 3]
    legend_ax.axis("off")
    legend_ax.text(
        0.0,
        0.92,
        "Legende",
        fontsize=14,
        fontweight="bold",
        transform=legend_ax.transAxes,
    )
    legend_ax.text(0.0, 0.72, "Rouge : vegetation perdue", fontsize=11, transform=legend_ax.transAxes)
    legend_ax.text(0.0, 0.56, "Bleu : vegetation gagnee", fontsize=11, transform=legend_ax.transAxes)
    legend_ax.text(0.0, 0.40, "Vert : vegetation stable", fontsize=11, transform=legend_ax.transAxes)
    legend_ax.text(
        0.0,
        0.16,
        "Hypothese cle : les deux images doivent representer la meme zone pour que la quantification soit valable.",
        fontsize=10,
        wrap=True,
        transform=legend_ax.transAxes,
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_rgb_image(image: np.ndarray, output_path: str | Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(output_path)


def _save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_path)


def _remove_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    labels, num_components = ndi.label(mask)
    if num_components == 0:
        return mask

    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= min_component_size
    keep[0] = False
    return keep[labels]


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (xx**2 + yy**2) <= radius**2


def _normalize_output_path(path: str | Path) -> str:
    return Path(path).as_posix()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de detection de changement de couverture forestiere.")
    parser.add_argument("--t0", default=str(DEFAULT_T0_PATH), help="Chemin de l'image avant.")
    parser.add_argument("--t1", default=str(DEFAULT_T1_PATH), help="Chemin de l'image apres.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Dossier de sortie.")
    parser.add_argument("--clusters", type=int, default=4, help="Nombre de clusters pour K-means.")
    parser.add_argument("--window-size", type=int, default=7, help="Taille de fenetre pour les stats locales.")
    parser.add_argument("--opening-radius", type=int, default=2, help="Rayon de l'ouverture morphologique.")
    parser.add_argument("--closing-radius", type=int, default=3, help="Rayon de la fermeture morphologique.")
    parser.add_argument("--min-component-size", type=int, default=64, help="Taille minimale des composants gardes.")
    parser.add_argument("--dendrogram-sample-size", type=int, default=1200, help="Nombre de pixels echantillonnes pour la CAH.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    metrics = run_change_analysis(
        image_path_t0=args.t0,
        image_path_t1=args.t1,
        output_dir=args.output_dir,
        n_clusters=args.clusters,
        window_size=args.window_size,
        opening_radius=args.opening_radius,
        closing_radius=args.closing_radius,
        min_component_size=args.min_component_size,
        dendrogram_sample_size=args.dendrogram_sample_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
