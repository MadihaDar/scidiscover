from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.pipeline import dimensionality, clustering, report_generator, hypothesis_generator
from app.services.dataset_store import store
from app.models.schemas import ReportResponse

router = APIRouter(prefix="/report", tags=["report"])


class ReportRequest(BaseModel):
    dataset_id: str
    target_column: str | None = None
    include_hypotheses: bool = False
    domain_context: str = ""


@router.post("", response_model=ReportResponse)
def build_report(req: ReportRequest):
    try:
        df = store.get_frame(req.dataset_id)
        profile = store.get_profile(req.dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    numeric_cols = profile.numeric_columns()
    if not numeric_cols:
        raise HTTPException(400, "No numeric columns to analyze.")

    X = df[numeric_cols].to_numpy()
    pca_result = dimensionality.run_pca(X, n_components=min(5, len(numeric_cols)))
    variance_curve = dimensionality.explained_variance_curve(X)

    cluster_result = clustering.run_hdbscan(pca_result.coordinates)
    cluster_summary = clustering.cluster_summary_stats(df, cluster_result.labels, numeric_cols)

    feature_importance = None
    model_performance = None
    if req.target_column and req.target_column in df.columns:
        from app.pipeline.interpretable_ml import train_and_explain
        feature_cols = [c for c in numeric_cols if c != req.target_column]
        if feature_cols:
            model_result = train_and_explain(df, feature_cols, req.target_column)
            feature_importance = model_result.feature_importance
            model_performance = model_result.performance

    report = report_generator.generate_report(
        dataset_profile=profile,
        pca_variance_ratio=variance_curve,
        cluster_summary=cluster_summary,
        feature_importance=feature_importance,
        model_performance=model_performance,
    )

    hypotheses = []
    if req.include_hypotheses:
        all_findings = [s for section in report.sections for s in section.statements]
        generated = hypothesis_generator.generate_hypotheses(
            report_findings=all_findings, domain_context=req.domain_context,
        )
        hypotheses = [
            {"text": h.text, "based_on": h.based_on, "label": h.label} for h in generated
        ]

    return ReportResponse(markdown=report.to_markdown(), hypotheses=hypotheses)
