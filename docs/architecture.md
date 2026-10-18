# Architecture

```
                     ┌─────────────────────────┐
                     │   React frontend (Vite)  │
                     │  upload → embed → report │
                     └────────────┬─────────────┘
                                  │ REST (JSON)
                     ┌────────────▼─────────────┐
                     │      FastAPI backend      │
                     │  /datasets  /analyze  /report
                     └────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                    │
    ┌─────────▼────────┐ ┌────────▼────────┐ ┌─────────▼─────────┐
    │  data_loader.py   │ │ dimensionality.py│ │ interpretable_ml.py│
    │  profiling/clean  │ │  PCA / UMAP      │ │  GBM + SHAP        │
    └───────────────────┘ └──────────────────┘ └────────────────────┘
              │                   │                    │
              └───────────────────┼────────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │    clustering.py           │
                     │    HDBSCAN + cluster stats │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  report_generator.py       │
                     │  numbers → grounded prose  │
                     └────────────┬─────────────┘
                                  │ (only if requested)
                     ┌────────────▼─────────────┐
                     │ hypothesis_generator.py    │
                     │  LLM: labeled speculation  │
                     └───────────────────────────┘
```

## Why it's shaped this way

**Every pipeline module is domain-agnostic and independently callable.**
`data_loader.py` never imports anything eigenworm-specific; domain knowledge
lives in `notebooks/`, which call the same pipeline modules the API uses.
This is what makes "run the same tool on microbiome data" a real claim
rather than a README promise — swapping datasets means writing a new
notebook, not touching pipeline code.

**Grounded statements and AI speculation are architecturally separated.**
`report_generator.py` only ever describes numbers it was handed — no LLM
call happens inside it. `hypothesis_generator.py` is the only module that
calls an LLM, and its output type (`Hypothesis`) always carries a
`label` field the frontend renders visibly, so a reader can never mistake
a computed statistic for an AI guess.

**Storage is a swappable interface, not a design commitment.** `dataset_store.py`
is an in-memory dict today. Every route talks to it through `get_frame` /
`get_profile` / `put`, so replacing it with Postgres or Redis later touches
one file, not the routes or pipeline.

## Known limitations (v1)

- CSV upload only; no native `.ts` (sktime) or `.biom` (microbiome) support
  yet in the API — those formats are handled in notebooks for now.
- `dataset_store` is in-memory and single-process; restarting the backend
  clears uploaded datasets.
- SHAP's `TreeExplainer` does not support multiclass `GradientBoostingClassifier`
  in some sklearn/shap version combinations; `interpretable_ml.py` falls
  back to the model's built-in `feature_importances_` in that case
  (see the exception handling in `_shap_importance`).
