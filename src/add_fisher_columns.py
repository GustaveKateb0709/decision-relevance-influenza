"""Add Fisher/Laplace-geometry DRS columns to results/cells.csv.

The NPE-based DRS columns are kept untouched (they are the methodological
negative control).  This script adds, per cell:

    * Fisher/Laplace DRS evaluated at the plug-in point theta_hat
      (drs_var_fisher, drs_geo_fisher, + _control)          -> decision point
    * the same evaluated at the TRUE theta
      (drs_var_fisher_true, drs_geo_fisher_true, + _control) -> comparable with
      results/laplace.csv
    * margin / margin_sd / margin_snr at the true theta, and peak_count
      (max expected reported count in the observation window)

Geometry follows identifiability_laplace.drs_variants (expected Fisher
information + direction-specific prior/posterior sloppiness).

Writes: results/cells.csv (columns appended), results/fisher_summary.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = "<PROJECT_ROOT>"
SRC = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")
sys.path.insert(0, SRC)

from model import PRIOR_HIGH, PRIOR_LOW, PHI_DISP, prior_sample, sample_reports  # noqa: E402
from npe import load_npe, sample_posterior                                        # noqa: E402
from identifiability_laplace import drs_variants                                  # noqa: E402
from verify_ridges import reported_mean_natural                                   # noqa: E402

SEED = 0
N_REP = 8
N_SAMPLES = 2000
PHIS = [2.0, 10.0, 50.0]
WS = [6, 10]
RHOS = list(np.exp(np.linspace(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4]), 4)))


def build_seasons(n_rep, seed):
    rng = np.random.default_rng(seed)
    out = []
    for rho in RHOS:
        for _ in range(n_rep):
            u = prior_sample(1, rng)[0]
            u[4] = np.log(rho)
            out.append((rho, np.exp(u)))
    return out


def main():
    df = pd.read_csv(os.path.join(RES, "cells.csv"))
    models = {W: load_npe(os.path.join(CKPT, f"npe_W{W}.pt"), double=True) for W in WS}
    seasons = build_seasons(N_REP, SEED + 7)
    rng = np.random.default_rng(SEED + 999)

    rows = []
    k = 0
    for (rho, theta) in seasons:
        for phi in PHIS:
            for W in WS:
                x = sample_reports(theta, n_weeks=W, rng=rng, phi=phi)
                Us = sample_posterior(models[W], x, n=N_SAMPLES, seed=SEED + k)
                theta_hat = np.exp(Us.mean(0))

                out = {"cell_id": k}
                # --- PRIMARY: Fisher/Laplace at the TRUE theta -------------
                # (same convention as results/laplace.csv; the version whose
                #  control arm behaves)
                r_t, _, _ = drs_variants(theta, n_weeks=W, control=False)
                rc_t, _, _ = drs_variants(theta, n_weeks=W, control=True)
                out["drs_var_fisher"] = r_t["drs_var"]
                out["drs_geo_fisher"] = r_t["drs_geo"]
                out["drs_var_fisher_control"] = rc_t["drs_var"]
                out["drs_geo_fisher_control"] = rc_t["drs_geo"]
                out["n_sloppy_fisher"] = r_t["n_sloppy"]
                # --- Fisher/Laplace at the plug-in point theta_hat ----------
                r_hat, _, _ = drs_variants(theta_hat, n_weeks=W, control=False)
                rc_hat, _, _ = drs_variants(theta_hat, n_weeks=W, control=True)
                out["drs_var_fisher_plug"] = r_hat["drs_var"]
                out["drs_geo_fisher_plug"] = r_hat["drs_geo"]
                out["drs_var_fisher_plug_control"] = rc_hat["drs_var"]
                out["drs_geo_fisher_plug_control"] = rc_hat["drs_geo"]
                out["n_sloppy_fisher_plug"] = r_hat["n_sloppy"]
                out["margin_snr_fisher_plug"] = r_hat["margin_snr"]
                # --- decision margin / SNR / peak count (true theta) --------
                out["margin_true"] = r_t["margin"]
                out["margin_sd_true"] = r_t["margin_sd"]
                out["margin_snr_true"] = r_t["margin_snr"]
                out["peak_count"] = float(reported_mean_natural(theta, W).max())
                rows.append(out)
                k += 1
                if k % 48 == 0:
                    print(f"  ... {k}/{len(seasons)*len(PHIS)*len(WS)}", flush=True)

    add = pd.DataFrame(rows)
    assert len(add) == len(df), (len(add), len(df))
    assert (add["cell_id"].values == df["cell_id"].values).all(), "cell_id misalignment"
    # drop any pre-existing copies of these columns, then append
    for c in add.columns:
        if c != "cell_id" and c in df.columns:
            df = df.drop(columns=[c])
    df = pd.concat([df, add.drop(columns=["cell_id"])], axis=1)
    df.to_csv(os.path.join(RES, "cells.csv"), index=False)
    print(f"wrote {os.path.join(RES,'cells.csv')}  shape={df.shape}", flush=True)

    # ---------------- cross-validation numbers -------------------------
    summ = {"n_cells": int(len(df))}
    for tag in ("fisher", "fisher_plug"):
        m = df[f"drs_var_{tag}"].to_numpy()
        c = df[f"drs_var_{tag}_control"].to_numpy()
        gm = df[f"drs_geo_{tag}"].to_numpy()
        gc = df[f"drs_geo_{tag}_control"].to_numpy()
        summ[f"drs_var_{tag}_median"] = float(np.nanmedian(m))
        summ[f"drs_var_{tag}_control_median"] = float(np.nanmedian(c))
        summ[f"frac_drs_var_{tag}_gt_control"] = float(np.mean(m > c))
        summ[f"drs_geo_{tag}_median"] = float(np.nanmedian(gm))
        summ[f"drs_geo_{tag}_control_median"] = float(np.nanmedian(gc))
        summ[f"frac_drs_geo_{tag}_gt_control"] = float(np.mean(gm > gc))
    # margin_snr vs peak_count correlations
    snr = df["margin_snr_true"].to_numpy()
    pk = df["peak_count"].to_numpy()
    ok = np.isfinite(snr) & np.isfinite(pk)
    sp = spearmanr(snr[ok], pk[ok])
    pe = pearsonr(snr[ok], pk[ok])
    summ["margin_snr_true_median"] = float(np.nanmedian(snr))
    summ["frac_margin_snr_lt_1"] = float(np.mean(snr < 1))
    summ["frac_margin_snr_lt_0p5"] = float(np.mean(snr < 0.5))
    summ["spearman_snr_peakcount"] = float(sp.statistic)
    summ["spearman_p"] = float(sp.pvalue)
    summ["pearson_snr_peakcount"] = float(pe.statistic)
    summ["pearson_p"] = float(pe.pvalue)
    summ["n_used"] = int(ok.sum())
    json.dump(summ, open(os.path.join(RES, "fisher_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
