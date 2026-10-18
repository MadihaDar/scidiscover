"""
clustering.py

HDBSCAN over an embedding (typically UMAP or PCA output). HDBSCAN
rather than k-means because scientific data rarely has a known
cluster count in advance, and HDBSCAN's noise label (-1) is itself
informative: "these samples don't fit any discovered pattern" is a
legitimate scientific statement, not a failure of the algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray  # -1 = noise / outlier
    n_clusters: int
    n_noise_points: int
    probabilities: np.ndarray | None  # membership strength per point


def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> ClusterResult:
    import hdbscan  # noqa: WPS433

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(embedding)

    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise_points=n_noise,
        probabilities=getattr(clusterer, "probabilities_", None),
    )


def cluster_summary_stats(
    df, labels: np.ndarray, numeric_columns: list[str]
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Per-cluster mean/std for each numeric column, relative to the
    overall dataset mean. This is the raw material the report
    generator turns into sentences like "Cluster 3 has elevated
    feature X relative to the rest of the dataset."
    """
    import pandas as pd

    working = df[numeric_columns].copy()
    working["__cluster__"] = labels

    overall_mean = working[numeric_columns].mean()
    overall_std = working[numeric_columns].std().replace(0, np.nan)

    summary: dict[int, dict[str, dict[str, float]]] = {}
    for cluster_id, group in working.groupby("__cluster__"):
        col_stats = {}
        for col in numeric_columns:
            cluster_mean = group[col].mean()
            z = (cluster_mean - overall_mean[col]) / overall_std[col]
            col_stats[col] = {
                "cluster_mean": float(cluster_mean),
                "overall_mean": float(overall_mean[col]),
                "z_relative_to_overall": float(z) if not np.isnan(z) else 0.0,
            }
        summary[int(cluster_id)] = col_stats

    return summary
