"""
02_second_dataset_wine.py

Generality check: the SAME pipeline (no code changes) on a different
domain - chemical profiles of wines from three cultivars (UCI Wine, real
data bundled with scikit-learn). Purpose is to show the tool is not
eigenworm-specific; it is not a claim about wine science.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sklearn.datasets import load_wine

from app.cli import analyze
from plotting import make_figures

df = load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})
res = analyze(df, target="cultivar")
print(res["report"].to_markdown())
print("Clusters found:", res["clusters"].n_clusters, "| noise points:", res["clusters"].n_noise_points)
print("Holdout accuracy:", res["model"].performance)
make_figures(res, df, "cultivar", ROOT / "docs" / "figures", "wine")
