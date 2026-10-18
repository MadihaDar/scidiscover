from __future__ import annotations

from pydantic import BaseModel


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    n_rows: int
    n_columns: int
    columns: list[dict]


class EmbeddingRequest(BaseModel):
    dataset_id: str
    method: str = "pca"  # "pca" | "umap"
    n_components: int = 2
    columns: list[str] | None = None  # defaults to all numeric columns


class EmbeddingResponse(BaseModel):
    method: str
    coordinates: list[list[float]]
    explained_variance_ratio: list[float] | None = None


class ClusterRequest(BaseModel):
    dataset_id: str
    embedding_coordinates: list[list[float]]
    min_cluster_size: int = 5


class ClusterResponse(BaseModel):
    labels: list[int]
    n_clusters: int
    n_noise_points: int


class ModelRequest(BaseModel):
    dataset_id: str
    feature_columns: list[str]
    target_column: str
    task: str | None = None


class ModelResponse(BaseModel):
    task: str
    performance: dict[str, float]
    feature_importance: dict[str, float]


class ReportResponse(BaseModel):
    markdown: str
    hypotheses: list[dict] = []
