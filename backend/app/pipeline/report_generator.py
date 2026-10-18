"""
report_generator.py

Turns the numeric output of the pipeline (cluster stats, SHAP
importances, PCA variance curve) into plain-English sentences.
Deliberately template-based rather than LLM-based: every sentence
here is a direct, traceable statement about a computed number, with
no possibility of hallucination. The LLM only enters at the
hypothesis-generation stage (hypothesis_generator.py), and even there
its output is clearly separated from these grounded statements.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReportSection:
    title: str
    statements: list[str] = field(default_factory=list)


@dataclass
class ScientificReport:
    sections: list[ReportSection] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Automated Analysis Report\n"]
        for section in self.sections:
            lines.append(f"## {section.title}\n")
            for statement in section.statements:
                lines.append(f"- {statement}")
            lines.append("")
        return "\n".join(lines)


def generate_report(
    dataset_profile,
    pca_variance_ratio: list[float] | None = None,
    cluster_summary: dict | None = None,
    feature_importance: dict[str, float] | None = None,
    model_performance: dict[str, float] | None = None,
    z_score_threshold: float = 1.0,
    top_k_features: int = 5,
    label_agreement: dict | None = None,
    importance_method: str = "shap",
    eval_note: str = "holdout test set",
) -> ScientificReport:
    report = ScientificReport()

    # --- Section 1: dataset shape ---
    overview = ReportSection(title="Dataset Overview")
    overview.statements.append(
        f"{dataset_profile.n_rows} samples across {dataset_profile.n_columns} columns "
        f"({len(dataset_profile.numeric_columns())} numeric, "
        f"{len(dataset_profile.categorical_columns())} categorical, "
        f"{len(dataset_profile.time_series_columns())} time-series)."
    )
    report.sections.append(overview)

    # --- Section 2: dimensionality ---
    if pca_variance_ratio:
        dim = ReportSection(title="Dimensionality")
        cumulative = 0.0
        n_for_95 = None
        for i, v in enumerate(pca_variance_ratio, start=1):
            cumulative += v
            if n_for_95 is None and cumulative >= 0.95:
                n_for_95 = i
        n_features = len(pca_variance_ratio)
        if n_for_95:
            ratio = n_for_95 / n_features
            if ratio <= 0.34:
                verdict = "strongly low-dimensional structure"
            elif ratio <= 0.67:
                verdict = "moderately low-dimensional structure"
            else:
                verdict = "little dimensionality reduction (variance is spread across most features)"
            dim.statements.append(
                f"{n_for_95} of {n_features} principal components are needed to explain "
                f"95% of total variance, indicating {verdict}."
            )
        dim.statements.append(
            f"Component 1 alone explains {pca_variance_ratio[0]*100:.1f}% of variance."
        )
        report.sections.append(dim)

    # --- Section 3: clusters ---
    if cluster_summary:
        cluster_section = ReportSection(title="Cluster Findings")
        for cluster_id, stats in cluster_summary.items():
            if cluster_id == -1:
                continue
            notable = [
                (col, s["z_relative_to_overall"])
                for col, s in stats.items()
                if abs(s["z_relative_to_overall"]) >= z_score_threshold
            ]
            notable.sort(key=lambda x: -abs(x[1]))
            if not notable:
                continue
            descriptors = [
                f"{'high' if z > 0 else 'low'} {col}" for col, z in notable[:3]
            ]
            cluster_section.statements.append(
                f"Cluster {cluster_id} is characterized by " + ", ".join(descriptors) + "."
            )
        if cluster_section.statements:
            report.sections.append(cluster_section)

    if label_agreement:
        agree = ReportSection(title="Do Discovered Clusters Match Known Labels?")
        ari = label_agreement["adjusted_rand_index"]
        strength = "strong" if ari >= 0.6 else "partial" if ari >= 0.3 else "weak"
        agree.statements.append(
            f"Adjusted Rand index between unsupervised clusters and '{label_agreement['label']}' "
            f"is {ari:.2f} ({strength} agreement; 0 = chance, 1 = identical)."
        )
        report.sections.append(agree)

    # --- Section 4: predictive features ---
    if feature_importance:
        pred_section = ReportSection(title="Predictive Features")
        ranked = sorted(feature_importance.items(), key=lambda x: -x[1])[:top_k_features]
        for name, score in ranked:
            label = "mean |SHAP|" if importance_method == "shap" else "model-based importance (SHAP unavailable)"
            pred_section.statements.append(
                f"'{name}' ranks among the strongest predictors ({label} = {score:.4f})."
            )
        if model_performance:
            metric_name, metric_value = next(iter(model_performance.items()))
            pred_section.statements.append(
                f"Model {metric_name}: {metric_value:.3f} ({eval_note})."
            )
        report.sections.append(pred_section)

    return report
