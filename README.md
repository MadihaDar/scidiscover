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

Note what this is *not*: the channels are already eigenworm projections, so this project does not re-derive the eigenworms. It analyses the dynamics of the amplitudes.

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

```bash
# one-time setup (from the repo root)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

**Run the analyses (no server needed)**
```bash
# put EigenWorms_TRAIN.ts and EigenWorms_TEST.ts in data/eigenworms/ (see data/README.md)
python notebooks/01_eigenworms_exploration.py      # writes docs/results + docs/figures
python notebooks/02_second_dataset_wine.py         # second dataset, no download needed
python notebooks/03_eigenworms_conformal.py        # per-worm uncertainty (about a minute)
cd backend && python -m app.cli your.csv --target label --out report.md
```

**Run the web app**
```bash
cd backend && uvicorn app.main:app --reload         # API on :8000, docs at /docs
cd frontend && npm install && npm run dev           # UI on :5173
```
Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to enable
AI-generated hypotheses (clearly labelled as unvalidated).

**Docker**
```bash
docker compose up --build
```

**Tests**
```bash
cd backend && pytest -v        # 14 tests, all on synthetic or bundled data
```

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

### Which individual predictions can be trusted? (conformal prediction)

Accuracy is an average. To say something about individual worms,
`pipeline/conformal.py` implements split conformal prediction: each worm gets a
*set* of plausible strains, built so that the true strain is inside the set at least
`1 − α` of the time **if calibration and test worms are exchangeable**.
Protocol (`notebooks/03_eigenworms_conformal.py`): the 128 TRAIN worms are split
(stratified) into 88 proper-train and 40 calibration worms, repeated over 200 random
splits; the official TEST split is used only for evaluation. Results:
[`docs/results/eigenworms_conformal.json`](docs/results/eigenworms_conformal.json).

| Check (131 test worms, 200 calibration splits) | α = 0.10 (target 90%) | α = 0.20 (target 80%) |
|---|---|---|
| Coverage, mean (5th–95th percentile) | 0.981 (0.961 – 0.992) | 0.913 (0.832 – 0.969) |
| Splits with coverage below target | 0 of 200 | 3 of 200 |
| Average set size (strains per worm) | 1.65 | 1.10 |
| Worms with a single-strain set | 51% | 84% |
| Accuracy on those single-strain worms | 0.99 | 0.94 |
| Empty sets | 0.2% | 3% |

What this shows, and what it does not:
- **The guarantee holds, but it is conservative here, and the reason matters.**
  For exchangeable data and 40 calibration worms, theory puts mean coverage between
  0.90 and about 0.92 at α = 0.10. Observed coverage (0.98) is well above that.
  Two controls where exchangeability holds by construction land close to theory:
  random splits of the pooled 259 worms give 0.93 (α = 0.10) and 0.84 (α = 0.20);
  random splits within TRAIN only give 0.94 and 0.85 (slightly conservative, as
  expected with discrete forest scores). So the method behaves as it should, and the
  official TEST split looks *easier* than held-out TRAIN worms: the same forests
  score 0.89 on TEST but 0.81 on the calibration worms. This is consistent with the
  possible session-overlap leakage flagged below, but it does not prove it.
- **Set size tracks difficulty only partly.** At α = 0.10, strains 4 and 5 (the
  classifier's confused pair) get the widest sets (about 2.0), as hoped. But strain 3,
  which is never misclassified, also gets wide sets (1.9), and strain 2, which is
  confused with strain 1, gets narrow ones (1.4). Set size reflects how peaked the
  forest's probabilities are (which depends on class size and overlap), not only
  where errors occur.
- **Coverage is marginal, not per strain.** At α = 0.20 per-strain coverage ranges
  from 0.86 (strains 2, 5) to 0.94 (strain 1), so the 80% promise does not hold
  strain by strain. With 40 calibration worms spread over five strains (about 4 for
  strain 5), class-conditional calibration is not feasible here.
- **Raw forest probabilities are not the problem; they are under-confident.**
  Mean top probability is 0.73 against 0.90 accuracy (ECE 0.17, 5-bin reliability
  plot in `docs/figures/eigenworms_conformal_reliability.png`). Taking the forest's
  probabilities at face value gives sets that over-cover (0.99) and are wider (2.3
  strains vs 1.65 at α = 0.10). Conformal calibration roughly halves the excess width.
- Test accuracy here (0.89 for forests trained on 88 worms) is lower than the 0.925
  reported above for forests trained on all 128.

Figures: `docs/figures/eigenworms_conformal_*.png`.

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

```
scidiscover/
├── backend/
│   ├── app/
│   │   ├── cli.py          # run the full pipeline from the command line
│   │   ├── pipeline/       # data_loader, ts_format, features, dimensionality,
│   │   │                   # clustering, interpretable_ml, conformal,
│   │   │                   # report_generator,
│   │   │                   # hypothesis_generator — domain-agnostic core
│   │   ├── api/routes/     # upload, analyze, report endpoints
│   │   ├── services/       # dataset storage
│   │   └── models/         # pydantic schemas
│   └── tests/
├── frontend/               # React + Vite + Plotly
├── notebooks/              # 01 EigenWorms, 02 Wine, 03 conformal prediction, plotting
├── data/                   # raw data (gitignored) — see data/README.md
└── docs/                   # architecture.md, figures/, results/
```

## Status / roadmap

- [x] Core pipeline (profiling, PCA/UMAP, HDBSCAN, SHAP, report gen)
- [x] FastAPI backend, native `.ts` + CSV upload, command-line runner
- [x] React frontend (upload → embed → report flow)
- [x] Time-series feature engineering (7 interpretable features per channel)
- [x] Second dataset (UCI Wine) through the identical pipeline — see Results
- [x] 14 unit/integration tests, Docker support
- [x] Conformal prediction on EigenWorms with exchangeability controls (see Results)
- [x] Real EigenWorms analysis with validation controls (see Results)
- [ ] Persistent dataset storage (currently in-memory)

## Known limitations

- **Possible train/test leakage in EigenWorms.** The `.ts` files carry no recording
  or plate IDs, so I could not check whether worms from the same recording session
  appear on both sides of the archive's split. If they do, test accuracy is
  optimistic. The 5-fold CV on the training set (0.82 ± 0.08) is a more
  conservative reference point.
- **No per-strain biology.** Strain labels are numbered 1–5 and the archive does not
  say which number is which of N2, goa-1, unc-1, unc-38 and unc-63. Results are
  reported per numbered class only, and the hypothesis generator should not be
  asked to name strains.
- **Not a re-derivation of the eigenworms.** The channels are already eigenworm
  projections; the project analyses their dynamics, not raw body posture.
- **Conformal guarantees assume exchangeability**, which the EigenWorms split may
  violate (see above): observed coverage is far above nominal, which makes the sets
  valid but wider than necessary. A 40-worm calibration set also makes coverage vary
  between splits (reported as a spread), and the 131-worm test set limits how
  precisely coverage can be measured. The guarantee is marginal, not per strain or
  per worm. This is a tabular summary-feature setting, not a spatiotemporal model.
- **Small samples.** 128 training and 131 test worms (Wine: 45 test samples), so
  accuracies carry a few points of sampling noise.
- **Summary-feature approach.** Each worm is reduced to 42 summary statistics, which
  discards temporal structure that sequence models could use.
- **In-memory storage.** Uploaded datasets in the API live in process memory and are
  lost on restart.
- **AI hypotheses are unvalidated.** They are suggestions for experiments, never findings.

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
