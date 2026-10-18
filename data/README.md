# Getting the EigenWorms dataset

This project's primary demo dataset is **EigenWorms**, from the UEA/UCR
Multivariate Time Series Classification Archive. It's built directly on the
eigenworm postural decomposition introduced by Stephens et al. (2008) and
extended by Yemini et al. (2013) — 6 eigenworm-projection channels recorded
over ~18,000 timesteps per worm, labeled by strain (wild-type vs. four
mutant strains).

## Download

1. Go to https://www.timeseriesclassification.com/description.php?Dataset=EigenWorms
2. Download the `.ts` format archive (train + test splits).
3. Unzip into `data/eigenworms/` (this folder is gitignored — do not commit
   raw data to the repo; that's what this README is for).

Expected layout after unzipping:
```
data/eigenworms/
├── EigenWorms_TRAIN.ts
└── EigenWorms_TEST.ts
```

## Loading into pandas

The `.ts` format is sktime's native format. The cleanest path is:

```python
from sktime.datasets import load_from_tsfile

X_train, y_train = load_from_tsfile("data/eigenworms/EigenWorms_TRAIN.ts")
X_test, y_test = load_from_tsfile("data/eigenworms/EigenWorms_TEST.ts")
```

`X_train` is a DataFrame of nested Series (one cell per channel per sample).
`notebooks/01_eigenworms_exploration.py` shows how to flatten this into the
wide numeric format the SciDiscover pipeline expects (fixed-length windows
or summary-statistic features per channel — raw 18,000-step series are too
long to feed directly into PCA/clustering without windowing first).

## Citation

If referenced in the README/report, cite:
- Stephens, G. J., Johnson-Kerner, B., Bialek, W., & Ryu, W. S. (2008).
  Dimensionality and dynamics in the behavior of C. elegans. *PLOS
  Computational Biology*.
- Yemini, E., et al. (2013). A database of C. elegans behavioral phenotypes.
  *Nature Methods*.

## Notes from use
- Official split is 128 train / 131 test (the archive's text says the reverse).
- The `.ts` header comments contain non-UTF-8 bytes; `ts_format.py` reads with latin-1.
- Files are ~130 MB each and gitignored.
