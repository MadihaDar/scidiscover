"""
dataset_store.py

In-memory store keyed by dataset_id. This is intentionally the
simplest thing that works: a dict. For a portfolio project this is
fine and honest -- swapping it for Redis/Postgres later is a clearly
isolated change because every other module only talks to this
interface (get/put), never to the storage mechanism directly.
"""
from __future__ import annotations

import uuid

import pandas as pd

from app.pipeline.data_loader import DatasetProfile


class DatasetStore:
    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._profiles: dict[str, DatasetProfile] = {}

    def put(self, df: pd.DataFrame, profile: DatasetProfile) -> str:
        dataset_id = str(uuid.uuid4())
        self._frames[dataset_id] = df
        self._profiles[dataset_id] = profile
        return dataset_id

    def get_frame(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self._frames:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        return self._frames[dataset_id]

    def get_profile(self, dataset_id: str) -> DatasetProfile:
        if dataset_id not in self._profiles:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        return self._profiles[dataset_id]


# module-level singleton, imported by route handlers
store = DatasetStore()
