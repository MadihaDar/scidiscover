"""
data_loader.py

Handles ingestion of arbitrary CSV / time-series data and automatic
profiling: which columns are numeric, categorical, time-series, or
mostly missing. This is what lets the rest of the pipeline stay
domain-agnostic (eigenworm channels today, microbiome OTU counts or
stress-strain curves tomorrow) without any per-domain branching.

Design note: we keep this file free of anything eigenworm-specific.
Domain knowledge belongs in notebooks/scripts that *call* this module,
not inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIME_SERIES = "time_series"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    name: str
    kind: ColumnKind
    missing_fraction: float
    n_unique: int
    notes: str = ""


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == ColumnKind.NUMERIC]

    def categorical_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == ColumnKind.CATEGORICAL]

    def time_series_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == ColumnKind.TIME_SERIES]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": [
                {
                    "name": c.name,
                    "kind": c.kind.value,
                    "missing_fraction": round(c.missing_fraction, 4),
                    "n_unique": c.n_unique,
                    "notes": c.notes,
                }
                for c in self.columns
            ],
        }


def load_csv(path: str | Path) -> pd.DataFrame:
    """Thin wrapper today; the seam where format-specific loaders
    (e.g. .ts files from the UEA archive, .biom for microbiome data)
    get added later without touching the profiling logic below."""
    return pd.read_csv(path)


def profile_dataset(
    df: pd.DataFrame,
    categorical_uniqueness_threshold: float = 0.05,
    time_series_prefix_hint: str | None = None,
) -> DatasetProfile:
    """
    Infer a per-column profile so downstream steps (dimensionality
    reduction, clustering, interpretable ML) know what they're
    looking at without a human specifying it column-by-column.

    Heuristics (deliberately simple and explainable, not ML-based):
      - low-cardinality non-numeric / low-cardinality integer columns
        -> categorical
      - columns whose names share a common numeric-suffix pattern
        (e.g. t_0, t_1, ... or repeated prefix + index) -> time_series
      - everything else numeric -> numeric
      - >90% missing -> flagged in notes regardless of kind
    """
    n_rows = len(df)
    profiles: list[ColumnProfile] = []

    # crude grouping of columns that look like repeated time-indexed
    # measurements, e.g. "eigenworm1_t0..t17983" style wide time series
    ts_groups = _detect_time_series_groups(df.columns, hint=time_series_prefix_hint)

    for col in df.columns:
        series = df[col]
        missing_fraction = series.isna().mean()
        n_unique = series.nunique(dropna=True)

        if col in ts_groups:
            kind = ColumnKind.TIME_SERIES
            notes = f"part of time-series group '{ts_groups[col]}'"
        elif n_unique <= 1:
            kind = ColumnKind.IDENTIFIER if n_unique == n_rows else ColumnKind.UNKNOWN
            notes = "constant or ID-like column"
        elif pd.api.types.is_numeric_dtype(series):
            uniqueness_ratio = n_unique / max(n_rows, 1)
            non_null = series.dropna()
            integer_valued = bool(len(non_null)) and bool(np.all(np.equal(np.mod(non_null, 1), 0)))
            # only integer-valued columns can be codes/labels; a float column with
            # few distinct values (e.g. a quantized frequency) is still a measurement
            if integer_valued and (uniqueness_ratio <= categorical_uniqueness_threshold or n_unique <= 10) and n_unique <= 20:
                kind = ColumnKind.CATEGORICAL
                notes = "integer-valued, low-cardinality; treated as categorical"
            else:
                kind = ColumnKind.NUMERIC
                notes = ""
        else:
            if n_unique == n_rows:
                kind = ColumnKind.IDENTIFIER
                notes = "unique per row; likely an identifier"
            else:
                kind = ColumnKind.CATEGORICAL
                notes = ""

        if missing_fraction > 0.9:
            notes = (notes + "; " if notes else "") + "mostly missing (>90%)"

        profiles.append(
            ColumnProfile(
                name=col,
                kind=kind,
                missing_fraction=missing_fraction,
                n_unique=n_unique,
                notes=notes,
            )
        )

    return DatasetProfile(n_rows=n_rows, n_columns=len(df.columns), columns=profiles)


def _detect_time_series_groups(
    columns: list[str], hint: str | None = None
) -> dict[str, str]:
    """
    Detect columns that are really one long time series split across
    many wide columns (common export format: signal_0, signal_1, ...).
    Returns {column_name: group_name}.
    """
    import re

    groups: dict[str, list[str]] = {}
    pattern = re.compile(r"^(.*?)[_\-]?(\d+)$")

    for col in columns:
        match = pattern.match(str(col))
        if not match:
            continue
        prefix = match.group(1) or "series"
        if hint and hint not in prefix:
            continue
        groups.setdefault(prefix, []).append(col)

    # only treat as a time-series group if there are enough columns
    # to plausibly be a signal rather than a handful of coincidentally
    # numbered columns
    result: dict[str, str] = {}
    for prefix, cols in groups.items():
        if len(cols) >= 10:
            for c in cols:
                result[c] = prefix
    return result


def clean_dataset(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """
    Minimal, explainable cleaning: median-impute numeric columns,
    mode-impute categorical columns, drop columns that are >90%
    missing. No silent row-dropping — scientific data is often small
    enough that every row matters, so missingness is imputed and
    flagged in the report rather than discarded.
    """
    cleaned = df.copy()
    dropped_columns = []

    for col_profile in profile.columns:
        col = col_profile.name
        if "mostly missing" in col_profile.notes:
            cleaned.drop(columns=[col], inplace=True, errors="ignore")
            dropped_columns.append(col)
            continue

        if col not in cleaned.columns:
            continue

        if col_profile.kind == ColumnKind.NUMERIC:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        elif col_profile.kind == ColumnKind.CATEGORICAL:
            mode = cleaned[col].mode(dropna=True)
            if not mode.empty:
                cleaned[col] = cleaned[col].fillna(mode.iloc[0])

    if dropped_columns:
        cleaned.attrs["dropped_columns"] = dropped_columns
    return cleaned
