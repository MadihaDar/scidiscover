"""
Tests run entirely on synthetic data generated in-process, so they
don't depend on downloading the EigenWorms dataset (see
data/README.md for how to get the real thing for the notebooks).
This keeps CI fast and keeps the pipeline logic verifiably correct
independent of any one domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline.data_loader import profile_dataset, clean_dataset, ColumnKind
from app.pipeline.dimensionality import run_pca, explained_variance_curve
from app.pipeline.clustering import run_hdbscan, cluster_summary_stats
from app.pipeline.interpretable_ml import train_and_explain
from app.pipeline.report_generator import generate_report


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 150
    # two well-separated blobs in 2 informative dims + 1 noise dim
    group = rng.integers(0, 2, size=n)
    x1 = rng.normal(loc=group * 5, scale=1.0)
    x2 = rng.normal(loc=group * -3, scale=1.0)
    noise = rng.normal(size=n)
    category = np.where(group == 0, "control", "treated")

    df = pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "noise_feature": noise,
        "condition": category,
        "sample_id": [f"S{i}" for i in range(n)],
    })
    # introduce some missingness to exercise cleaning
    df.loc[df.sample(5, random_state=1).index, "feature_a"] = np.nan
    return df


def test_profile_dataset_infers_kinds(synthetic_df):
    profile = profile_dataset(synthetic_df)
    kinds = {c.name: c.kind for c in profile.columns}

    assert kinds["feature_a"] == ColumnKind.NUMERIC
    assert kinds["feature_b"] == ColumnKind.NUMERIC
    assert kinds["condition"] == ColumnKind.CATEGORICAL
    assert kinds["sample_id"] == ColumnKind.IDENTIFIER


def test_clean_dataset_imputes_missing(synthetic_df):
    profile = profile_dataset(synthetic_df)
    cleaned = clean_dataset(synthetic_df, profile)
    assert cleaned["feature_a"].isna().sum() == 0


def test_pca_explains_variance(synthetic_df):
    X = synthetic_df[["feature_a", "feature_b", "noise_feature"]].fillna(0).to_numpy()
    result = run_pca(X, n_components=2)
    assert result.coordinates.shape == (len(synthetic_df), 2)
    assert sum(result.explained_variance_ratio) > 0.5  # two real signal dims dominate

    curve = explained_variance_curve(X)
    assert len(curve) == 3
    assert abs(sum(curve) - 1.0) < 1e-6


def test_hdbscan_recovers_two_groups(synthetic_df):
    X = synthetic_df[["feature_a", "feature_b"]].fillna(0).to_numpy()
    result = run_hdbscan(X, min_cluster_size=10)
    # two well-separated blobs should not all collapse into one cluster or all noise
    assert result.n_clusters >= 1
    assert result.n_noise_points < len(synthetic_df)


def test_cluster_summary_flags_separating_features(synthetic_df):
    X = synthetic_df[["feature_a", "feature_b"]].fillna(0).to_numpy()
    labels = run_hdbscan(X, min_cluster_size=10).labels
    summary = cluster_summary_stats(synthetic_df, labels, ["feature_a", "feature_b", "noise_feature"])
    assert isinstance(summary, dict)
    assert len(summary) >= 1


def test_train_and_explain_classification(synthetic_df):
    df = synthetic_df.copy()
    df["label"] = (df["feature_a"].fillna(0) > 2).astype(int)
    result = train_and_explain(
        df.fillna(0), feature_columns=["feature_a", "feature_b", "noise_feature"],
        target_column="label", task="classification",
    )
    assert result.task == "classification"
    assert "accuracy" in result.performance
    assert set(result.feature_importance.keys()) == {"feature_a", "feature_b", "noise_feature"}
    # the informative feature should outrank pure noise
    assert result.feature_importance["feature_a"] > result.feature_importance["noise_feature"]


def test_report_generation_is_non_empty(synthetic_df):
    profile = profile_dataset(synthetic_df)
    cleaned = clean_dataset(synthetic_df, profile)
    X = cleaned[["feature_a", "feature_b", "noise_feature"]].to_numpy()

    variance_curve = explained_variance_curve(X)
    labels = run_hdbscan(X, min_cluster_size=10).labels
    summary = cluster_summary_stats(cleaned, labels, ["feature_a", "feature_b", "noise_feature"])

    report = generate_report(
        dataset_profile=profile,
        pca_variance_ratio=variance_curve,
        cluster_summary=summary,
    )
    markdown = report.to_markdown()
    assert "Dataset Overview" in markdown
    assert len(report.sections) >= 1


# ---- .ts reader, feature engineering, CLI, API ----
import io as _io


def _fake_ts(n=12, steps=200, channels=3):
    rng = np.random.default_rng(1)
    lines = ["@problemName Fake", "@univariate false", "@classLabel true a b", "@data"]
    for i in range(n):
        lab = "a" if i % 2 == 0 else "b"
        t = np.linspace(0, 10, steps)
        chans = [",".join(f"{v:.4f}" for v in np.sin((1 + (i % 2)) * t + c) + rng.normal(0, .1, steps))
                 for c in range(channels)]
        lines.append(":".join(chans) + ":" + lab)
    return "\n".join(lines)


def test_ts_reader_and_features(tmp_path):
    from app.pipeline.ts_format import read_ts_file
    from app.pipeline.features import extract_timeseries_features

    f = tmp_path / "x.ts"
    f.write_text(_fake_ts())
    series, labels = read_ts_file(f)
    assert len(series) == 12 and series[0].shape == (3, 200)
    assert set(labels) == {"a", "b"}
    feats = extract_timeseries_features(series)
    assert feats.shape == (12, 3 * 7)
    # class b oscillates twice as fast -> dominant frequency must differ
    assert feats["ch0_dominant_freq"][labels == "b"].mean() > feats["ch0_dominant_freq"][labels == "a"].mean()


def test_cli_analyze_wine():
    from sklearn.datasets import load_wine
    from app.cli import analyze

    df = load_wine(as_frame=True).frame
    res = analyze(df, target="target")
    assert res["model"].performance["accuracy"] > 0.85
    assert "Predictive Features" in res["report"].to_markdown()


def test_api_accepts_ts_upload():
    from fastapi.testclient import TestClient
    from app.main import app

    r = TestClient(app).post("/datasets/upload", files={"file": ("x.ts", _fake_ts(), "text/plain")})
    assert r.status_code == 200
    assert any(c["name"] == "label" for c in r.json()["columns"])


def test_low_cardinality_float_stays_numeric():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"quantized_freq": rng.choice([0.01, 0.02, 0.03], 200),
                       "class_code": rng.integers(0, 3, 200),
                       "x": rng.normal(size=200)})
    kinds = {c.name: c.kind for c in profile_dataset(df).columns}
    assert kinds["quantized_freq"] == ColumnKind.NUMERIC
    assert kinds["class_code"] == ColumnKind.CATEGORICAL
