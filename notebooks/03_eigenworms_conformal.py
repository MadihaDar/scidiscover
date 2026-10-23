"""
03_eigenworms_conformal.py

Question: SciDiscover reports ~0.93 test accuracy on EigenWorms, but which
individual predictions can be trusted? Split conformal prediction gives each
worm a SET of plausible strains with a coverage guarantee. Here we test
whether that guarantee actually holds on this small dataset.

Protocol (fixed before looking at TEST; the official TEST split is used only
for the final evaluation, never for tuning):
  1. Split the 128 TRAIN worms (stratified) into proper-train (88) and
     calibration (40). Repeat over N_REPEATS random splits.
  2. Fit a random forest on proper-train; build conformal sets for TEST at
     alpha = 0.10 and 0.20.
  3. Measure coverage, set size, per-strain behaviour, and compare with an
     uncalibrated baseline that trusts the forest's probabilities as-is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline import conformal as cf
from app.pipeline.features import extract_timeseries_features
from app.pipeline.ts_format import read_ts_file

DATA = ROOT / "data" / "eigenworms"
CHANNELS = [f"eig{i+1}" for i in range(6)]
N_CAL, N_REPEATS, ALPHAS = 40, 200, (0.10, 0.20)
FIG, RES = ROOT / "docs" / "figures", ROOT / "docs" / "results"


def main():
    tr_s, tr_y = read_ts_file(DATA / "EigenWorms_TRAIN.ts")
    te_s, te_y = read_ts_file(DATA / "EigenWorms_TEST.ts")
    Xtr = extract_timeseries_features(tr_s, CHANNELS).to_numpy()
    Xte = extract_timeseries_features(te_s, CHANNELS).to_numpy()
    classes = sorted(set(tr_y))
    ytr = np.array([classes.index(v) for v in tr_y])
    yte = np.array([classes.index(v) for v in te_y])
    K = len(classes)
    print("train class counts:", dict(zip(classes, np.bincount(ytr))),
          "| test:", dict(zip(classes, np.bincount(yte))))

    # ---- calibration diagnostics: forest trained on ALL 128 training worms ----
    full = RandomForestClassifier(300, random_state=0).fit(Xtr, ytr)
    p_full = full.predict_proba(Xte)
    diag = {
        "test_accuracy_full_train_forest": float((p_full.argmax(1) == yte).mean()),
        "ece_top_label_10bins": cf.expected_calibration_error(p_full, yte),
        "brier": cf.brier_score(p_full, yte),
        "mean_top_probability": float(p_full.max(1).mean()),
        "reliability_bins(conf,acc,n)": cf.reliability_curve(p_full, yte, 5),
    }

    # ---- repeated split-conformal runs ----
    sss = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=N_CAL, random_state=0)
    out = {a: {"cov": [], "size": [], "single": [], "empty": [], "single_acc": [],
               "cls_cov": [], "cls_size": [], "naive_cov": [], "naive_size": []} for a in ALPHAS}
    accs, cal_accs = [], []
    for i, (fit_idx, cal_idx) in enumerate(sss.split(Xtr, ytr)):
        rf = RandomForestClassifier(300, random_state=i).fit(Xtr[fit_idx], ytr[fit_idx])
        p_cal, p_te = rf.predict_proba(Xtr[cal_idx]), rf.predict_proba(Xte)
        accs.append(float((p_te.argmax(1) == yte).mean()))
        cal_accs.append(float((p_cal.argmax(1) == ytr[cal_idx]).mean()))
        for a in ALPHAS:
            r = out[a]
            q = cf.conformal_quantile(p_cal, ytr[cal_idx], a)
            S = cf.prediction_sets(p_te, q)
            sz = S.sum(1)
            r["cov"].append(cf.coverage(S, yte)); r["size"].append(float(sz.mean()))
            r["single"].append(float((sz == 1).mean())); r["empty"].append(float((sz == 0).mean()))
            one = sz == 1
            r["single_acc"].append(float(S[one][np.arange(one.sum()), yte[one]].mean()) if one.any() else np.nan)
            r["cls_cov"].append(cf.per_class_coverage(S, yte, K))
            r["cls_size"].append([float(sz[yte == k].mean()) for k in range(K)])
            N = cf.naive_cumulative_sets(p_te, 1 - a)
            r["naive_cov"].append(cf.coverage(N, yte)); r["naive_size"].append(float(N.sum(1).mean()))

    def band(x):
        x = np.asarray(x, float)
        return {"mean": float(np.nanmean(x)), "p5": float(np.nanpercentile(x, 5)),
                "p95": float(np.nanpercentile(x, 95)), "min": float(np.nanmin(x))}

    summary = {"n_proper_train": int(len(Xtr) - N_CAL), "n_calibration": N_CAL, "n_test": int(len(Xte)),
               "n_repeats": N_REPEATS, "classes": classes,
               "mean_test_accuracy_proper_train_forest": float(np.mean(accs)),
               "mean_calibration_accuracy_same_forests": float(np.mean(cal_accs)),
               "calibration_diagnostics": diag, "by_alpha": {}}
    for a in ALPHAS:
        r = out[a]
        summary["by_alpha"][str(a)] = {
            "target_coverage": 1 - a,
            "coverage": band(r["cov"]),
            "fraction_runs_below_target": float(np.mean(np.array(r["cov"]) < 1 - a)),
            "avg_set_size": band(r["size"]),
            "fraction_singleton_sets": band(r["single"]),
            "fraction_empty_sets": band(r["empty"]),
            "accuracy_when_singleton": band(r["single_acc"]),
            "per_strain_coverage_mean": [float(v) for v in np.nanmean(r["cls_cov"], axis=0)],
            "per_strain_set_size_mean": [float(v) for v in np.nanmean(r["cls_size"], axis=0)],
            "naive_uncalibrated_coverage": band(r["naive_cov"]),
            "naive_uncalibrated_set_size": band(r["naive_size"]),
        }
    # ---- exchangeability controls: does the guarantee hold when calibration and
    # evaluation data are exchangeable BY CONSTRUCTION (random splits of one pool)? ----
    def control(X, y, n_fit, n_cal, n_eval, reps=200, seed=1):
        rng = np.random.default_rng(seed)
        res = {a: [] for a in ALPHAS}
        for i in range(reps):
            idx = rng.permutation(len(y))[: n_fit + n_cal + n_eval]
            f, c, e = idx[:n_fit], idx[n_fit:n_fit + n_cal], idx[n_fit + n_cal:]
            rf = RandomForestClassifier(300, random_state=i).fit(X[f], y[f])
            pc, pe = rf.predict_proba(X[c]), rf.predict_proba(X[e])
            for a in ALPHAS:
                res[a].append(cf.coverage(cf.prediction_sets(pe, cf.conformal_quantile(pc, y[c], a)), y[e]))
        return {str(a): float(np.mean(v)) for a, v in res.items()}

    Xall, yall = np.vstack([Xtr, Xte]), np.concatenate([ytr, yte])
    summary["controls"] = {
        "within_TRAIN_only (fit 68 / cal 30 / eval 30)": control(Xtr, ytr, 68, 30, 30),
        "pooled_train+test_random_split (fit 129 / cal 40 / eval 90)": control(Xall, yall, 129, 40, 90),
        "theory_range_for_n_cal=40": {"alpha_0.1": [0.9, round(0.9 + 1 / 41, 3)], "alpha_0.2": [0.8, round(0.8 + 1 / 41, 3)]},
    }
    print(json.dumps(summary["controls"], indent=2))
    print("cal acc", summary["mean_calibration_accuracy_same_forests"], "test acc", summary["mean_test_accuracy_proper_train_forest"])
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    (RES / "eigenworms_conformal.json").write_text(json.dumps(summary, indent=2))

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, a in zip(axes, ALPHAS):
        ax.hist(out[a]["cov"], bins=20, color="#3a7bd5")
        ax.axvline(1 - a, color="k", ls="--", label=f"target {1-a:.0%}")
        ax.axvline(np.mean(out[a]["cov"]), color="r", label="mean")
        ax.set(title=f"alpha={a}: coverage on 131 test worms", xlabel="coverage", ylabel="# calibration splits")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "eigenworms_conformal_coverage.png", dpi=150); plt.close(fig)

    a = 0.10
    cs, cc = np.nanmean(out[a]["cls_size"], axis=0), np.nanmean(out[a]["cls_cov"], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(classes, cs, color="#3a7bd5"); axes[0].axhline(1, c="gray", lw=1)
    axes[0].set(title="Avg set size by true strain (alpha=0.10)", xlabel="strain", ylabel="classes per set")
    axes[1].bar(classes, cc, color="#d5a13a"); axes[1].axhline(0.9, c="k", ls="--")
    axes[1].set(title="Coverage by true strain (alpha=0.10)", xlabel="strain", ylabel="coverage", ylim=(0, 1.05))
    fig.tight_layout(); fig.savefig(FIG / "eigenworms_conformal_by_strain.png", dpi=150); plt.close(fig)

    rel = cf.reliability_curve(p_full, yte, 5)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot([r[0] for r in rel], [r[1] for r in rel], "o-")
    for r in rel: ax.annotate(f"n={r[2]}", (r[0], r[1]), fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax.set(xlabel="mean predicted top-class probability", ylabel="fraction correct",
           title="Forest reliability on test (5 bins)", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout(); fig.savefig(FIG / "eigenworms_conformal_reliability.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
