# SciDiscover

A domain-agnostic pipeline for scientific pattern discovery: upload a
dataset, get back dimensionality structure, discovered clusters, the
features that actually predict an outcome, and a plain-language report —
with any AI-generated hypotheses clearly separated from what the data
directly shows.

## Why this exists

Most "AI dashboard" tools stop at prediction. This one is built around a
scientific question instead: **what structure is in this dataset, which variables
drive it, and what would be worth testing next?** — with every claim traceable
to a computed number.

The demo dataset is [EigenWorms](https://www.timeseriesclassification.com/description.php?Dataset=EigenWorms)
(UEA archive; Brown et al. 2013, Yemini et al. 2013): 259 C. elegans recordings,
each 17,984 frames of six eigenworm amplitudes (the six base shapes from the
eigenworm decomposition of Stephens et al. 2008), labelled wild-type (N2) or one
of four mutant strains. The question: **do the dynamics of body posture carry
enough information to tell strains apart, and which dynamical features carry it?**

Note what this is *not*: the channels are already eigenworm projections, so this
project does not re-derive the eigenworms. It analyses the dynamics of the amplitudes.

The pipeline itself has no eigenworm-specific logic anywhere in it — see
[`docs/architecture.md`](docs/architecture.md) for how domain-agnosticism
is enforced, not just claimed.

## What it does

1. **Automatic dataset understanding** — upload a CSV, get back inferred
   column types (numeric / categorical / time-series / identifier) and a
   cleaning pass, with no manual schema required.
2. **Pattern discovery** — PCA and UMAP embeddings; HDBSCAN clustering
   (noise points are a legitimate output, not a failure mode).
3. **Interpretable ML** — tree-based classification/regression (random forest for
   multiclass, gradient boosting otherwise) with SHAP feature importance and
   confidence estimates; the report states when SHAP is unavailable and a fallback is used.
4. **Automatic report** — plain-language findings generated directly from
   the computed statistics (template-based, not LLM-based — every sentence
   traces back to a specific number).
5. **Hypothesis generation** — an optional, clearly-labeled LLM pass that
   proposes testable explanations for the discovered patterns. Never mixed
   into the factual report.

## Quickstart

Requires Python 3.12 and Node 20 (frontend only).

````bash
# one-time setup (from the repo root)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
````

**Run the analyses (no server needed)**
````bash
# put EigenWorms_TRAIN.ts and EigenWorms_TEST.ts in data/eigenworms/ (see data/README.md)
python notebooks/01_eigenworms_exploration.py      # writes docs/results + docs/figures
python notebooks/02_second_dataset_wine.py         # second dataset, no download needed
cd backend && python -m app.cli your.csv --target label --out report.md
````

**Run the web app**
````bash
cd backend && uvicorn app.main:app --reload         # API on :8000, docs at /docs
cd frontend && npm install && npm run dev           # UI on :5173
````
Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to enable
AI-generated hypotheses (clearly labelled as unvalidated).

**Docker**
````bash
docker compose up --build
````

**Tests**
````bash
cd backend && pytest -v        # 11 tests, all on synthetic or bundled data
````

## Results

### EigenWorms (real data, official split: 128 train / 131 test, 5 strains)

Each worm is summarised by 42 interpretable features (7 per eigenworm channel).
Everything supervised is trained on the TRAIN split only and scored on TEST only.
Full numbers: [`docs/results/eigenworms.json`](docs/results/eigenworms.json).

| Check | Result |
|---|---|
| Majority-class baseline (test) | 0.42 |
| Random forest, 5 seeds | 0.925 mean (0.90 – 0.95) |
| Gradient boosting, 5 seeds | 0.911 mean |
| Logistic regression | 0.870 |
| 5-fold CV on TRAIN only (random forest) | 0.82 ± 0.08 |
| Control: shuffled training labels | 0.40 mean, 0.48 max (≈ chance) |
| Single eigenworm channel alone | 0.70 – 0.85 (eig5 best) |
| Unsupervised clusters vs. strain (ARI) | 0.22 (weak) |

How to read this honestly:
- Strain is predictable from posture dynamics well above chance, and the
  shuffled-label control confirms the pipeline is not leaking labels.
- Test accuracy (0.93) sits above CV on the small training set (0.82 ± 0.08);
  with 128 training and 131 test worms, a few points of this gap is plausibly sampling noise.
- Errors are concentrated: strain 2 is sometimes called 1 (5 of 22), and strains 4 and 5
  are confused with each other (7 of 36). Strain 3 is perfectly separated.
- Unsupervised structure only partly follows strain (ARI 0.22): the dominant
  variation in these summary features is not strain identity.
- The first four eigenworm amplitudes carry 78% of the variance in these six
  channels. Stephens et al. report ~95% in four modes on different data and
  preprocessing, so this is **not** a reproduction of that figure.
- Two further caveats (possible split leakage, unmapped strain labels) are
  under [Known limitations](#known-limitations).

Figures: `docs/figures/eigenworms_*.png`.

### UCI Wine (generality check; same code path)

| Check | Result |
|---|---|
| Components for 95% variance | 10 of 13 (little low-dimensional structure; the report says so) |
| Unsupervised clusters vs. cultivar | adjusted Rand index 0.80 |
| Holdout accuracy (random forest) | 1.00 on a 45-sample random split; too small to distinguish from ~0.95-0.98, indicative only |
| Top predictors (SHAP) | color intensity, flavanoids, proline |

Wine only shows the pipeline is not eigenworm-specific.

Reproduce: `python notebooks/01_eigenworms_exploration.py` (after placing the
`.ts` files per `data/README.md`), `python notebooks/02_second_dataset_wine.py`,
or `cd backend && PYTHONPATH=. python -m app.cli your.csv --target label`.

## Project structure

````
scidiscover/
├── backend/
│   ├── app/
│   │   ├── cli.py          # run the full pipeline from the command line
│   │   ├── pipeline/       # data_loader, ts_format, features, dimensionality,
│   │   │                   # clustering, interpretable_ml, report_generator,
│   │   │                   # hypothesis_generator — domain-agnostic core
│   │   ├── api/routes/     # upload, analyze, report endpoints
│   │   ├── services/       # dataset storage
│   │   └── models/         # pydantic schemas
│   └── tests/
├── frontend/               # React + Vite + Plotly
├── notebooks/              # 01 EigenWorms, 02 Wine, plotting helpers
├── data/                   # raw data (gitignored) — see data/README.md
└── docs/                   # architecture.md, figures/, results/
````

## Status / roadmap

- [x] Core pipeline (profiling, PCA/UMAP, HDBSCAN, SHAP, report gen)
- [x] FastAPI backend, native `.ts` + CSV upload, command-line runner
- [x] React frontend (upload → embed → report flow)
- [x] Time-series feature engineering (7 interpretable features per channel)
- [x] Second dataset (UCI Wine) through the identical pipeline — see Results
- [x] 11 unit/integration tests, Docker support
- [x] Real EigenWorms analysis with validation controls (see Results)

## Known limitations

- **No per-strain biology.** Strain labels are numbered 1–5 and the archive does not
  say which number is which of N2, goa-1, unc-1, unc-38 and unc-63. Results are
  reported per numbered class only, and the hypothesis generator should not be
  asked to name strains.
- **Not a re-derivation of the eigenworms.** The channels are already eigenworm
  projections; the project analyses their dynamics, not raw body posture.
- **Small samples.** 128 training and 131 test worms (Wine: 45 test samples), so
  accuracies carry a few points of sampling noise.
- **Summary-feature approach.** Each worm is reduced to 42 summary statistics, which
  discards temporal structure that sequence models could use.
- **In-memory storage.** Uploaded datasets in the API live in process memory and are
  lost on restart.

## Citations

- Stephens, G.J., Johnson-Kerner, B., Bialek, W., Ryu, W.S. (2008).
  Dimensionality and dynamics in the behavior of C. elegans. *PLOS
  Computational Biology*.
- Brown, A., Yemini, E., Grundy, L., Jucikas, T., Schafer, W. (2013).
  A dictionary of behavioral motifs reveals clusters of genes affecting
  *Caenorhabditis elegans* locomotion. *PNAS*.
- Yemini, E. et al. (2013). A database of C. elegans behavioral phenotypes.
  *Nature Methods*.
- Bagnall, A. et al. (2018). The UEA multivariate time series
  classification archive.
````
````