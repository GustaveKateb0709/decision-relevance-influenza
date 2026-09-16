# Auditing model-based antiviral recommendations: a decision-relevance diagnostic for seasonal influenza surveillance

Research code and results accompanying the manuscript of the same title
(submitted to PLOS Computational Biology).

## The idea in three lines

1. Surveillance reports only **reported** cases, so the reporting rate `rho` and the
   initial epidemic size `i0` enter the observations **only through their product**
   — an exact multiplicative ridge in the likelihood.
2. The *decision* (how much antiviral to commit) depends on the absolute burden,
   which is **not** invariant along that ridge.
3. We quantify how much of the decision margin's uncertainty comes from directions
   the data cannot see. The naive (geometric) version of that diagnostic gives a
   **dangerous false negative**; the variance-weighted score proposed in the paper
   separates decision-sensitive from decision-insensitive settings on both a
   synthetic grid and a decade of real CDC FluView seasons.

## Layout

```
src/
  model.py                    SEIRS + negative-binomial reporting + antiviral actions (numba RK4)
  calibrate.py                outcome-independent calibration of the intervention cost ratio
  verify_ridges.py            numerical verification of the (rho, i0) ridge
  identifiability_laplace.py  Fisher/Laplace decision-identifiability diagnostics
  add_fisher_columns.py       enrich cells.csv with Fisher/Laplace quantities
  npe.py                      hand-written NPE: GRU summary + MAF (MADE), APT loss
  train_bounded_w6.py         bounded-support NPE training (W = 6)
  run_experiment.py           full evaluation grid -> results/cells.csv
  add_snr_npe.py              NPE-path margin SNR columns
  lambda_sensitivity.py       cost-ratio sensitivity sweep -> results/lambda_sensitivity.csv
  ppf_boxweighted.py          robustness of PPF to posterior over-dispersion
  external_validity.py        same diagnostics on real CDC FluView seasons
  make_figures.py             publication figures
  verify_figures.py           figure verification
  make_docx.py                manuscript build tooling
checkpoints/                  pre-trained bounded NPEs (load with npe.load_npe)
results/                      machine-readable outputs behind every reported number
data/                         DATA_NOTES.md + fetch/build scripts (raw CSVs not committed)
```

## Reproduce

```bash
pip install -r requirements.txt

# synthetic-grid analyses (CPU; the full NPE training takes hours,
# pre-trained checkpoints are committed under checkpoints/)
cd src
python calibrate.py
python verify_ridges.py
python identifiability_laplace.py
python run_experiment.py --quick    # fast end-to-end check; drop --quick for the full grid
python lambda_sensitivity.py
python ppf_boxweighted.py

# real-data external validity (two public sources, no registration required)
cd ../data
python fetch_data.py                # WHO FluNet + CDC FluView via CMU DELPHI Epidata
python build_seasons.py             # -> data/seasons.csv
cd ../src
python external_validity.py
```

Everything is seeded. Reference runtime: Python 3.13.12, torch 2.13.0, numba 0.66.0,
single-threaded CPU, 8 GB unified memory.

## Data

Real surveillance data come from two public sources, both fetched without
registration:

- **WHO FluNet** — `https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv`
- **CDC FluView** via the CMU DELPHI Epidata API — `https://api.delphi.cmu.edu/epidata/fluview/`

See `data/DATA_NOTES.md` for access dates, field definitions, the season definition,
the screening rules, and known pitfalls (ISO vs MMWR week alignment, data revisions,
uneven FluNet coverage).

## A note on the cost-ratio constant

`model.py:LAMBDA_COST` is set by a **declared, outcome-independent rule**
(`src/calibrate.py`): the cost per unit coverage equals the prior-median
uncontrolled per-capita burden divided by the maximum coverage level. This makes
the intervention cost-neutral at the median burden. An earlier version of that
calibration double-counted `i0` and was ~113,000× too small, which degenerated the
decision; the fix and its consequences are documented in the docstring of
`calibrate.py:neutral_lambda` and in the manuscript.

## License

MIT — see `LICENSE`.
