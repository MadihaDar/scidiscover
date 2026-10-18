"""Figure helpers shared by the analysis scripts (matplotlib, static PNGs for the README)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figures(res, df, target, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = np.array(res["curve"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(curve) + 1), np.cumsum(curve), "o-")
    ax.axhline(0.95, ls="--", c="gray", lw=1)
    ax.set(xlabel="Principal components", ylabel="Cumulative variance explained",
           title="How low-dimensional is the data?")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_variance.png", dpi=150); plt.close(fig)

    xy = res["embedding"].coordinates
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(xy[:, 0], xy[:, 1], c=res["clusters"].labels, cmap="tab10", s=18)
    axes[0].set_title("Discovered clusters (HDBSCAN; -1 = noise)")
    cats = df[target].astype(str).to_numpy()
    for i, cat in enumerate(sorted(set(cats))):
        m = cats == cat
        axes[1].scatter(xy[m, 0], xy[m, 1], color=plt.cm.tab10(i), s=18, label=str(cat))
    axes[1].legend(title=target, fontsize=7, title_fontsize=8)
    axes[1].set_title(f"Same embedding, colored by known '{target}'")
    for a in axes: a.set(xlabel=f"{res['embedding'].method} 1", ylabel=f"{res['embedding'].method} 2")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_embedding.png", dpi=150); plt.close(fig)

    imp = sorted(res["model"].feature_importance.items(), key=lambda x: x[1])[-10:]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([k for k, _ in imp], [v for _, v in imp])
    ax.set(xlabel="Importance (mean |SHAP| or model importance)", title="Top predictors")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_importance.png", dpi=150); plt.close(fig)
