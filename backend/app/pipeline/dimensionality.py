"""
dimensionality.py

PCA and UMAP wrappers. Kept separate from clustering.py so each stage
of the pipeline is independently testable and independently callable
from the API (a researcher may want the embedding without clustering,
or vice versa).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class EmbeddingResult:
    coordinates: np.ndarray  # (n_samples, n_components)
    explained_variance_ratio: list[float] | None  # None for UMAP
    method: str
    n_components: int


def run_pca(
    X: np.ndarray, n_components: int = 2, standardize: bool = True
) -> EmbeddingResult:
    """
    PCA is the first thing we run on any dataset, not just because
    it's fast, but because the explained-variance curve itself is
    scientifically informative (e.g. Stephens et al. 2008 showing 4
    components explain 95% of C. elegans postural variance -- that
    curve IS a finding, not just a preprocessing step).
    """
    if standardize:
        X = StandardScaler().fit_transform(X)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    return EmbeddingResult(
        coordinates=coords,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        method="pca",
        n_components=n_components,
    )


def explained_variance_curve(X: np.ndarray, standardize: bool = True) -> list[float]:
    """Full explained-variance-by-component curve, for the 'how many
    dimensions actually matter here' plot -- the eigenworm-style
    result generalized to any dataset."""
    if standardize:
        X = StandardScaler().fit_transform(X)
    max_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=max_components)
    pca.fit(X)
    return pca.explained_variance_ratio_.tolist()


def run_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    standardize: bool = True,
    random_state: int = 42,
) -> EmbeddingResult:
    """
    UMAP for visualization / clustering input. Imported lazily since
    umap-learn is a heavier dependency and some deployments may only
    need PCA.
    """
    import umap  # noqa: WPS433 (intentional lazy import)

    if standardize:
        X = StandardScaler().fit_transform(X)

    n_neighbors = min(n_neighbors, max(X.shape[0] - 1, 2))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)

    return EmbeddingResult(
        coordinates=coords,
        explained_variance_ratio=None,
        method="umap",
        n_components=n_components,
    )
