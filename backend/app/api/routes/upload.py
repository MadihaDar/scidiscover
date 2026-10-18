from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, UploadFile, HTTPException

from app.pipeline.data_loader import profile_dataset, clean_dataset
from app.services.dataset_store import store
from app.models.schemas import DatasetProfileResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetProfileResponse)
async def upload_dataset(file: UploadFile):
    name = file.filename.lower()
    if not (name.endswith(".csv") or name.endswith(".ts")):
        raise HTTPException(400, "Supported uploads: .csv, .ts (UEA time-series format).")

    raw_bytes = await file.read()
    try:
        if name.endswith(".ts"):
            import tempfile
            from app.pipeline.ts_format import read_ts_file
            from app.pipeline.features import extract_timeseries_features

            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as tmp:
                tmp.write(raw_bytes)
            series, labels = read_ts_file(tmp.name)
            df = extract_timeseries_features(series)
            df["label"] = labels
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(400, "Uploaded file has no rows.")

    profile = profile_dataset(df)
    cleaned = clean_dataset(df, profile)

    dataset_id = store.put(cleaned, profile)

    return DatasetProfileResponse(
        dataset_id=dataset_id,
        n_rows=profile.n_rows,
        n_columns=profile.n_columns,
        columns=[c.__dict__ | {"kind": c.kind.value} for c in profile.columns],
    )
