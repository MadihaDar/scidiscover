"""
ts_format.py

Minimal pure-Python reader for the UEA/UCR ".ts" time-series format
(no sktime dependency). Handles equal-length multivariate series, which
covers EigenWorms. Returns (list of (n_channels, n_timesteps) arrays, labels).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ts_file(path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    series_list: list[np.ndarray] = []
    labels: list[str] = []
    in_data = False
    # UEA header comments contain non-UTF8 bytes, so decode as latin-1
    with open(path, "r", encoding="latin-1") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            # each line: ch1_values:ch2_values:...:label
            parts = line.split(":")
            label = parts[-1].strip()
            channels = [
                np.array([float(v) if v != "?" else np.nan for v in ch.split(",")])
                for ch in parts[:-1]
            ]
            series_list.append(np.stack(channels))
            labels.append(label)
    if not series_list:
        raise ValueError(f"No data rows found in {path}")
    return series_list, np.array(labels)
