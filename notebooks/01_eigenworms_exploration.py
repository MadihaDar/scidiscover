"""
01_eigenworms_exploration.py

Runs SciDiscover on EigenWorms (UEA archive; see data/README.md).
  - Real data present  -> trains on the official TRAIN split, reports accuracy
    on the official TEST split, plus unsupervised structure on all worms.
  - Real data missing  -> runs on loudly-labeled SYNTHETIC data, only to
    exercise the code path. Never quote its numbers as results.
Outputs figures to docs/figures/eigenworms_*.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.cli import analyze
from app.pipeline.features import extract_timeseries_features
from app.pipeline.interpretable_ml import train_and_explain
from app.pipeline.ts_format import read_ts_file
from plotting import make_figures

DATA = ROOT / "data" / "eigenworms"
CHANNELS = [f"eig{i+1}" for i in range(6)]


def synthetic():
    rng = np.random.default_rng(0)
    out, labels = [], []
    for _ in range(80):
        s = int(rng.integers(0, 3))
        t = np.linspace(0, 40, 3000)
        out.append(np.stack([(1 + .2 * s) * np.sin((1 + .3 * s) * t + ph) + rng.normal(0, .2, t.size)
                             for ph in np.linspace(0, np.pi, 6)]))
        labels.append(str(s))
    return out, np.array(labels)


def main():
    real = (DATA / "EigenWorms_TRAIN.ts").exists() and (DATA / "EigenWorms_TEST.ts").exists()
    if real:
        tr_s, tr_y = read_ts_file(DATA / "EigenWorms_TRAIN.ts")
        te_s, te_y = read_ts_file(DATA / "EigenWorms_TEST.ts")
        tag = "eigenworms"
    else:
        print("!! REAL DATA NOT FOUND - using SYNTHETIC placeholder (not a result). See data/README.md\n")
        s, y = synthetic()
        tr_s, tr_y, te_s, te_y = s[:55], y[:55], s[55:], y[55:]
        tag = "eigenworms_SYNTHETIC"

    tr = extract_timeseries_features(tr_s, CHANNELS); tr["strain"] = tr_y
    te = extract_timeseries_features(te_s, CHANNELS); te["strain"] = te_y
    full = pd.concat([tr, te], ignore_index=True)
    train_idx = np.arange(len(tr)); test_idx = np.arange(len(tr), len(full))

    # Unsupervised parts see all worms (no labels used); the supervised model
    # is trained on TRAIN only and scored on TEST only.
    res = analyze(full, target="strain", train_idx=train_idx, test_idx=test_idx)
    print(res["report"].to_markdown())

    feats = [c for c in tr.columns if c != "strain"]
    Xtr, ytr, Xte, yte = tr[feats], tr["strain"], te[feats], te["strain"]
    out = {"dataset": tag, "n_train": len(tr), "n_test": len(te),
           "class_counts_test": yte.value_counts().to_dict(),
           "majority_baseline": float(yte.value_counts(normalize=True).max())}

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # 1) several model families on the official split
    models = {
        "random_forest": lambda seed: RandomForestClassifier(300, random_state=seed),
        "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
        "logistic_regression": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }
    out["official_split_accuracy"] = {}
    for name, mk in models.items():
        accs = [mk(seed).fit(Xtr, ytr).score(Xte, yte) for seed in range(5)]
        out["official_split_accuracy"][name] = {"mean": float(np.mean(accs)), "min": float(np.min(accs)), "max": float(np.max(accs))}

    # 2) cross-validation on TRAIN only (independent of the test set)
    cv = cross_val_score(RandomForestClassifier(300, random_state=0), Xtr, ytr,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out["train_5fold_cv_random_forest"] = {"mean": float(cv.mean()), "std": float(cv.std())}

    # 3) permutation control: shuffle TRAIN labels; accuracy should fall to ~chance
    rng = np.random.default_rng(0)
    perm = [RandomForestClassifier(300, random_state=0).fit(Xtr, rng.permutation(ytr.to_numpy())).score(Xte, yte)
            for _ in range(20)]
    out["permuted_label_accuracy"] = {"mean": float(np.mean(perm)), "max": float(np.max(perm))}

    # 4) per-class view
    rf = RandomForestClassifier(300, random_state=0).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    labels = sorted(yte.unique())
    out["confusion_matrix_labels"] = labels
    out["confusion_matrix"] = confusion_matrix(yte, pred, labels=labels).tolist()

    # 5) which channels carry the signal? (accuracy using one eigenworm at a time)
    out["single_channel_accuracy"] = {}
    for ch in CHANNELS:
        cols = [c for c in feats if c.startswith(ch + "_")]
        out["single_channel_accuracy"][ch] = float(
            RandomForestClassifier(300, random_state=0).fit(Xtr[cols], ytr).score(Xte[cols], yte))

    # 6) variance carried by each eigenworm amplitude across all frames
    # (the six channels are already projections onto eigenworms, so this checks
    # how concentrated postural variance is in the leading modes - NOT a re-derivation
    # of the eigenworms themselves, which needs raw body-posture data)
    frames = np.concatenate([x for x in tr_s + te_s], axis=1)
    var = frames.var(axis=1)
    out["eigenworm_amplitude_variance_share"] = [float(v) for v in var / var.sum()]
    out["first4_modes_variance_share"] = float(var[:4].sum() / var.sum())

    import json
    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    make_figures(res, full, "strain", ROOT / "docs" / "figures", tag)


if __name__ == "__main__":
    main()
