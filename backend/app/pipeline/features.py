"""
features.py

Automatic feature engineering for multichannel time series: turns each
(channels x timesteps) sample into one interpretable row. Every feature
has a physical meaning (amplitude, power, dominant frequency, ...), which
is what keeps the downstream SHAP ranking readable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dominant_freq(signal: np.ndarray) -> float:
    centered = signal - np.nanmean(signal)
    spectrum = np.abs(np.fft.rfft(np.nan_to_num(centered)))
    if len(spectrum) < 2:
        return 0.0
    return float(np.argmax(spectrum[1:]) + 1) / len(signal)


def _autocorr_lag(signal: np.ndarray, lag: int) -> float:
    s = np.nan_to_num(signal - np.nanmean(signal))
    denom = float(np.dot(s, s))
    if denom == 0 or lag >= len(s):
        return 0.0
    return float(np.dot(s[:-lag], s[lag:]) / denom)


def extract_timeseries_features(
    series_list: list[np.ndarray], channel_names: list[str] | None = None
) -> pd.DataFrame:
    n_channels = series_list[0].shape[0]
    names = channel_names or [f"ch{c}" for c in range(n_channels)]
    rows = []
    for series in series_list:
        row: dict[str, float] = {}
        for c, name in enumerate(names):
            x = series[c]
            row[f"{name}_mean"] = float(np.nanmean(x))
            row[f"{name}_std"] = float(np.nanstd(x))
            row[f"{name}_energy"] = float(np.nanmean(x ** 2))
            row[f"{name}_zero_cross_rate"] = float(
                np.sum(np.diff(np.sign(np.nan_to_num(x))) != 0) / len(x)
            )
            row[f"{name}_dominant_freq"] = _dominant_freq(x)
            row[f"{name}_autocorr_lag10"] = _autocorr_lag(x, 10)
            row[f"{name}_mean_abs_diff"] = float(np.nanmean(np.abs(np.diff(x))))
        rows.append(row)
    return pd.DataFrame(rows)
