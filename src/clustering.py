"""Clustering et selection automatique de la vegetation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.preprocessing import extract_features
else:
    from .preprocessing import extract_features


@dataclass(slots=True)
class ClusteringResult:
    labels: np.ndarray
    cluster_centers: np.ndarray
    feature_names: list[str]
    vegetation_cluster: int
    vegetation_scores: np.ndarray
    vegetation_mask: np.ndarray
    segmented_rgb: np.ndarray


def sample_feature_matrix(
    feature_matrix: np.ndarray,
    sample_size: int = 1500,
    random_state: int = 42,
) -> np.ndarray:
    """Echantillonne des pixels pour la CAH."""
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix doit etre de shape (N, F).")

    if feature_matrix.shape[0] <= sample_size:
        return feature_matrix

    rng = np.random.default_rng(random_state)
    indices = rng.choice(feature_matrix.shape[0], size=sample_size, replace=False)
    return feature_matrix[indices]


def save_dendrogram(
    feature_matrix: np.ndarray,
    output_path: str | Path,
    sample_size: int = 1500,
    method: str = "ward",
    random_state: int = 42,
) -> Path:
    """Sauvegarde un dendrogramme pour justifier le choix de k."""
    output_path = Path(output_path)
    sample = sample_feature_matrix(feature_matrix, sample_size=sample_size, random_state=random_state)
    linkage_matrix = linkage(sample, method=method)

    plt.figure(figsize=(11, 4.5))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        show_contracted=True,
        leaf_rotation=45,
        leaf_font_size=9,
    )
    plt.title("CAH sur un echantillon de pixels")
    plt.xlabel("Groupes fusionnes")
    plt.ylabel("Distance")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return output_path


def run_kmeans(
    features: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique K-means sur tous les pixels."""
    feature_matrix, image_shape = _flatten_features(features)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(feature_matrix)
    return labels.reshape(image_shape), model.cluster_centers_


def choose_vegetation_cluster(
    cluster_centers: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[int, np.ndarray]:
    """
    Identifie automatiquement la vegetation.

    Regle principale alignee avec le PDF:
    cluster avec le ratio vert moyen le plus eleve.
    """
    feature_names = list(feature_names)
    if "green_ratio" in feature_names:
        scores = cluster_centers[:, feature_names.index("green_ratio")]
    elif all(name in feature_names for name in ("r", "g", "b")):
        red = cluster_centers[:, feature_names.index("r")]
        green = cluster_centers[:, feature_names.index("g")]
        blue = cluster_centers[:, feature_names.index("b")]
        scores = green - 0.5 * red - 0.25 * blue
    else:
        raise ValueError("Impossible d'identifier la vegetation sans features colorees.")

    vegetation_cluster = int(np.argmax(scores))
    return vegetation_cluster, np.asarray(scores, dtype=np.float32)


def labels_to_segmented_rgb(
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Reconstruit une image segmentee a partir des centres de clusters."""
    feature_names = list(feature_names)
    if all(name in feature_names for name in ("r", "g", "b")):
        rgb_indices = [feature_names.index("r"), feature_names.index("g"), feature_names.index("b")]
        palette = np.clip(cluster_centers[:, rgb_indices], 0.0, 1.0)
    else:
        cmap = plt.get_cmap("tab10", cluster_centers.shape[0])
        palette = cmap(np.arange(cluster_centers.shape[0]))[:, :3]

    return palette[labels]


def segment_image(
    image: np.ndarray,
    n_clusters: int = 4,
    include_hsv: bool = True,
    include_green_ratio: bool = True,
    include_local_stats: bool = True,
    window_size: int = 7,
    random_state: int = 42,
) -> ClusteringResult:
    """Pipeline de segmentation complet pour une image."""
    features, feature_names = extract_features(
        image,
        include_hsv=include_hsv,
        include_green_ratio=include_green_ratio,
        include_local_stats=include_local_stats,
        window_size=window_size,
        flatten=False,
    )
    labels, cluster_centers = run_kmeans(features, n_clusters=n_clusters, random_state=random_state)
    vegetation_cluster, vegetation_scores = choose_vegetation_cluster(cluster_centers, feature_names)
    segmented_rgb = labels_to_segmented_rgb(labels, cluster_centers, feature_names)
    vegetation_mask = labels == vegetation_cluster

    return ClusteringResult(
        labels=labels,
        cluster_centers=cluster_centers,
        feature_names=feature_names,
        vegetation_cluster=vegetation_cluster,
        vegetation_scores=vegetation_scores,
        vegetation_mask=vegetation_mask,
        segmented_rgb=segmented_rgb,
    )


def _flatten_features(features: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    if features.ndim != 3:
        raise ValueError("features doit etre de shape (H, W, F).")
    image_shape = features.shape[:2]
    return features.reshape(-1, features.shape[-1]), image_shape
