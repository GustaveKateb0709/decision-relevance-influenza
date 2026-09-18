"""Add an NPE-posterior-based margin-SNR column (independent of the Fisher path).

    margin_snr_npe = margin(theta_hat) / sqrt( g' Sigma_npe g )

with margin/g evaluated at the plug-in point and Sigma_npe the NPE posterior
covariance in u-space.  This is a genuinely independent uncertainty estimate
from the Fisher/Laplace path, so the SNR-vs-peak_count correlation can be
cross-validated between the two.

Appends margin_snr_npe / margin_npe / margin_sd_npe to results/cells.csv and
reports the correlations (overall + W=6 subset).
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")

from model import PRIOR_HIGH, PRIOR_LOW, PHI_DISP, prior_sample, sample_reports  # noqa: E402
from npe import load_npe, sample_posterior                                        # noqa: E402
from verify_ridges import loss_grad_u, marginal_action_pair                       # noqa: E402
from model import loss                                                            # noqa: E402

SEED, N_REP, N_SAMPLES = 0, 8, 2000
PHIS, WS = [2.0, 10.0, 50.0], [6, 10]
RHOS = list(np.exp(np.linspace(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4]), 4)))


def build_seasons(n_rep, seed):
    rng = np.random.default_rng(seed)
    out = []
    for rho in RHOS:
        for _ in range(n_rep):
            u = prior_sample(1, rng)[0]
            u[4] = np.log(rho)
            out.append(np.exp(u))
    return out


def main():
    df = pd.read_csv(os.path.join(RES, "cells.csv"))
    models = {W: load_npe(os.path.join(CKPT, f"npe_W{W}.pt"), double=True) for W in WS}
    seasons = build_seasons(N_REP, SEED + 7)
    rng = np.random.default_rng(SEED + 999)

    rows, k = [], 0
    for theta in seasons:
        for phi in PHIS:
            for W in WS:
                x = sample_reports(theta, n_weeks=W, rng=rng, phi=phi)
                Us = sample_posterior(models[W], x, n=N_SAMPLES, seed=SEED + k)
                uhat = Us.mean(0)
                th_hat = np.exp(uhat)
                Sig = np.cov(Us.T)
                a_hi, a_lo = marginal_action_pair(th_hat)
                m = float(loss(th_hat, a_hi) - loss(th_hat, a_lo))
                g = loss_grad_u(th_hat, a_hi, a_lo, control=False)
                var = float(g @ Sig @ g)
                sd = float(np.sqrt(max(var, 0.0)))
                rows.append({"cell_id": k, "margin_npe": m, "margin_sd_npe": sd,
                             "margin_snr_npe": (m / sd) if sd > 0 else np.inf})
                k += 1
    add = pd.DataFrame(rows)
    assert (add["cell_id"].values == df["cell_id"].values).all()
    for c in add.columns:
        if c != "cell_id" and c in df.columns:
            df = df.drop(columns=[c])
    df = pd.concat([df, add.drop(columns=["cell_id"])], axis=1)
    df.to_csv(os.path.join(RES, "cells.csv"), index=False)
    print("wrote cells.csv shape", df.shape, flush=True)

    rep = {}
    for tag, d in (("all", df), ("W6", df[df.W == 6]), ("W10", df[df.W == 10])):
        for col in ("margin_snr_true", "margin_snr_npe"):
            if col not in d:
                continue
            s = d[col].to_numpy(); p = d["peak_count"].to_numpy()
            ok = np.isfinite(s) & np.isfinite(p)
            sp = spearmanr(s[ok], p[ok]); pe = pearsonr(s[ok], p[ok])
            rep[f"{tag}_{col}"] = {
                "n": int(ok.sum()),
                "median_snr": float(np.nanmedian(s)),
                "frac_snr_lt_1": float(np.mean(s < 1)),
                "frac_snr_lt_0p5": float(np.mean(s < 0.5)),
                "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
                "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue),
            }
    json.dump(rep, open(os.path.join(RES, "snr_correlation_summary.json"), "w"), indent=2)
    for kk, vv in rep.items():
        print(f"{kk:28s} n={vv['n']:3d} med={vv['median_snr']:.3f} "
              f"<1={vv['frac_snr_lt_1']:.3f} sp={vv['spearman']:+.3f} "
              f"(p={vv['spearman_p']:.1e}) pe={vv['pearson']:+.3f} (p={vv['pearson_p']:.1e})")


if __name__ == "__main__":
    main()
