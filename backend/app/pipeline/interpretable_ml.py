"""
interpretable_ml.py

Trains a classifier/regressor on request and returns SHAP-based
feature importance alongside predictions -- the point isn't just
"what's the accuracy" but "which variables actually drive the
outcome, and how confident is the model." Gradient-boosted trees
(not a neural net) are the default model because SHAP's TreeExplainer
is fast and exact, and because a second-year portfolio project is
better served by a model whose behavior you can fully explain than by
a bigger model you can't.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


@dataclass
class ModelResult:
    task: str  # "classification" | "regression"
    performance: dict[str, float]
    feature_importance: dict[str, float]  # mean |SHAP value| per feature
    predictions: np.ndarray
    confidence: np.ndarray | None  # class probability margin, or None for regression
    importance_method: str = "shap"  # "shap" or "model_importance" (fallback)
    model_name: str = ""
    test_index: np.ndarray | None = None
    y_true: np.ndarray | None = None


def train_and_explain(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    task: str | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    train_idx=None,
    test_idx=None,
) -> ModelResult:
    X = df[feature_columns]
    y = df[target_column]

    if task is None:
        task = "classification" if y.nunique() <= 20 and not pd.api.types.is_float_dtype(y) else "regression"

    if train_idx is not None and test_idx is not None:
        # explicit split (e.g. a dataset's official train/test partition)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if task == "classification" and y.nunique() > 1 else None,
        )

    if task == "classification":
        # SHAP's TreeExplainer does not support multiclass GradientBoostingClassifier,
        # so multiclass problems use a random forest (exact tree SHAP supported).
        if y_train.nunique() > 2:
            model = RandomForestClassifier(n_estimators=300, random_state=random_state)
        else:
            model = GradientBoostingClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        performance = {"accuracy": float(accuracy_score(y_test, preds))}
        probs = model.predict_proba(X_test)
        confidence = np.max(probs, axis=1)
    else:
        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        performance = {"r2": float(r2_score(y_test, preds))}
        confidence = None

    feature_importance, method = _shap_importance(model, X_test)

    return ModelResult(
        task=task,
        performance=performance,
        feature_importance=feature_importance,
        predictions=preds,
        confidence=confidence,
        importance_method=method,
        model_name=type(model).__name__,
        test_index=np.asarray(X_test.index),
        y_true=np.asarray(y_test),
    )


def _shap_importance(model, X_test: pd.DataFrame) -> tuple[dict[str, float], str]:
    """
    Mean absolute SHAP value per feature. Falls back to the model's
    built-in feature_importances_ if shap isn't installed, so the
    rest of the pipeline degrades gracefully rather than hard-failing
    -- a scientist evaluating this tool shouldn't hit an ImportError
    wall on the one line that mattered most.
    """
    try:
        import shap  # noqa: WPS433

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):  # multi-class classification
            stacked = np.mean([np.abs(v) for v in shap_values], axis=0)
        elif np.ndim(shap_values) == 3:
            # some sklearn/shap version combos return a single
            # (n_samples, n_features, n_classes) array for multiclass
            # instead of a list -- average |SHAP| over the class axis
            stacked = np.abs(shap_values).mean(axis=2)
        else:
            stacked = np.abs(shap_values)

        mean_abs = stacked.mean(axis=0)
        return dict(zip(X_test.columns, [float(v) for v in mean_abs])), "shap"
    except Exception:  # noqa: BLE001
        # SHAP's TreeExplainer has real gaps (e.g. it does not support
        # multiclass GradientBoostingClassifier at all -- this isn't a
        # hypothetical, it reproducibly fails on 3+ class problems).
        # Rather than crash the whole pipeline over an explainability
        # nicety, fall back to the model's built-in importances and
        # keep going -- a researcher should get *a* ranked feature
        # list even when exact SHAP values aren't available for their
        # model/task combination.
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return {}, "none"
        return dict(zip(X_test.columns, [float(v) for v in importances])), "model_importance"
