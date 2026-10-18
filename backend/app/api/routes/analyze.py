from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from app.pipeline import dimensionality, clustering, interpretable_ml
from app.services.dataset_store import store
from app.models.schemas import (
    EmbeddingRequest, EmbeddingResponse,
    ClusterRequest, ClusterResponse,
    ModelRequest, ModelResponse,
)

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/embedding", response_model=EmbeddingResponse)
def compute_embedding(req: EmbeddingRequest):
    try:
        df = store.get_frame(req.dataset_id)
        profile = store.get_profile(req.dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    columns = req.columns or profile.numeric_columns()
    if not columns:
        raise HTTPException(400, "No numeric columns available for embedding.")

    X = df[columns].to_numpy()

    if req.method == "umap":
        result = dimensionality.run_umap(X, n_components=req.n_components)
    else:
        result = dimensionality.run_pca(X, n_components=req.n_components)

    return EmbeddingResponse(
        method=result.method,
        coordinates=result.coordinates.tolist(),
        explained_variance_ratio=result.explained_variance_ratio,
    )


@router.post("/cluster", response_model=ClusterResponse)
def compute_clusters(req: ClusterRequest):
    embedding = np.array(req.embedding_coordinates)
    result = clustering.run_hdbscan(embedding, min_cluster_size=req.min_cluster_size)

    return ClusterResponse(
        labels=result.labels.tolist(),
        n_clusters=result.n_clusters,
        n_noise_points=result.n_noise_points,
    )


@router.post("/model", response_model=ModelResponse)
def train_model(req: ModelRequest):
    try:
        df = store.get_frame(req.dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    missing = [c for c in req.feature_columns + [req.target_column] if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found in dataset: {missing}")

    result = interpretable_ml.train_and_explain(
        df, feature_columns=req.feature_columns,
        target_column=req.target_column, task=req.task,
    )

    return ModelResponse(
        task=result.task,
        performance=result.performance,
        feature_importance=result.feature_importance,
    )
