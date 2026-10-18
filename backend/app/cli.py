"""
Command-line runner: `python -m app.cli data.csv --target label --out report.md`
Runs the full pipeline with no server, for reproducible analyses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.pipeline import clustering, dimensionality, report_generator
from app.pipeline.data_loader import clean_dataset, profile_dataset
from app.pipeline.interpretable_ml import train_and_explain


def analyze(df: pd.DataFrame, target: str | None = None, train_idx=None, test_idx=None):
    profile = profile_dataset(df)
    cleaned = clean_dataset(df, profile)
    numeric = [c for c in profile.numeric_columns() if c != target]
    X = cleaned[numeric].to_numpy()
    curve = dimensionality.explained_variance_curve(X)
    emb = dimensionality.run_umap(X) if len(X) > 30 else dimensionality.run_pca(X)
    clus = clustering.run_hdbscan(emb.coordinates, min_cluster_size=max(5, len(X) // 25))
    summary = clustering.cluster_summary_stats(cleaned, clus.labels, numeric)
    model = None
    if target:
        model = train_and_explain(cleaned, numeric, target, train_idx=train_idx, test_idx=test_idx)
    agreement = None
    if target:
        from sklearn.metrics import adjusted_rand_score
        agreement = {"label": target,
                     "adjusted_rand_index": float(adjusted_rand_score(cleaned[target], clus.labels))}
    report = report_generator.generate_report(
        profile, curve, summary,
        model.feature_importance if model else None,
        model.performance if model else None,
        label_agreement=agreement,
        importance_method=model.importance_method if model else "shap",
        eval_note="official train/test split" if train_idx is not None else "random 25% holdout",
    )
    return {"profile": profile, "cleaned": cleaned, "numeric": numeric, "curve": curve,
            "embedding": emb, "clusters": clus, "summary": summary,
            "model": model, "report": report}


def main():
    p = argparse.ArgumentParser(description="SciDiscover command-line analysis")
    p.add_argument("csv")
    p.add_argument("--target", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = analyze(pd.read_csv(args.csv), args.target)
    md = result["report"].to_markdown()
    if args.out:
        Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
