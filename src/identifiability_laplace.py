"""
Decisive diagnostic: does the *decision* margin inherit its uncertainty from the
structurally sloppy direction, once the posterior covariance is taken into
account?

Framework (Laplace, no NPE needed):
    F        = J^T W J                      expected Fisher information (u-space)
    C_prior  = diag((log hi - log lo)^2/12) prior covariance (uniform in log space)
    Sigma    = (F + C_prior^{-1})^{-1}      Laplace posterior covariance
    g        = grad_u [ l(a_hi) - l(a_lo) ] decision margin gradient
    DRS_geo  = ||P_S g||^2 / ||g||^2                    (pure geometry)
    DRS_var  = sum_{v in S} (v'g)^2 var_v  /  g' Sigma g (variance-weighted)

DRS_var is the honest question: "what fraction of the decision margin's
posterior variance comes from the weakly-identified directions?"

Run:  python src/identifiability_laplace.py
Writes: results/laplace.csv, results/laplace_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import PRIOR_HIGH, PRIOR_LOW, TAU, loss, prior_sample, scale_invariant_loss
from verify_ridges import (fisher_from_J, loss_grad_u, marginal_action_pair,
                           reported_mean_natural, sensitivity_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
T_OBS = 6
PARAM_NAMES = ("R0", "sigma", "gamma", "omega", "rho", "i0")


def prior_cov() -> np.ndarray:
    return np.diag(((np.log(PRIOR_HIGH) - np.log(PRIOR_LOW)) ** 2) / 12.0)


def laplace_posterior(theta, n_weeks=T_OBS):
    """Returns (F, Sigma, shrinkage) in u-space."""
    mu = reported_mean_natural(theta, n_weeks)
    J = sensitivity_matrix(theta, n_weeks)
    F = fisher_from_J(J, mu)
    Cp = prior_cov()
    Sigma = np.linalg.inv(F + np.linalg.inv(Cp))
    lam_p, V = np.linalg.eigh(Cp)
    lam_s, _ = np.linalg.eigh(Sigma)
    # per-direction prior/posterior sd ratio (ascending order matches V)
    shrink = np.sqrt(lam_p / np.clip(lam_s, 1e-300, None))
    return F, Sigma, shrink


def drs_variants(theta, n_weeks=T_OBS, control=False):
    """Decision-relevant sloppiness with a DIRECTION-SPECIFIC sloppy criterion.

    The prior variances differ by orders of magnitude across parameters
    (log i0 spans log(50)=3.9, log R0 spans log(1.30)=0.26), so a single global
    variance threshold misclassifies directions. We compare each posterior
    eigen-direction against the prior variance *along that same direction*.
    """
    F = fisher_at(theta, n_weeks)
    Cp = prior_cov()
    M = np.nan_to_num(F + np.linalg.inv(Cp), nan=0.0, posinf=0.0, neginf=0.0)
    Sigma = np.linalg.pinv(M)
    Sigma = (Sigma + Sigma.T) / 2.0
    lam, V = np.linalg.eigh(Sigma)
    lam = np.clip(lam, 0.0, None)
    prior_var_dir = np.einsum("ij,ij->j", V, Cp @ V)
    sloppy = lam >= 0.5 * prior_var_dir
    S = V[:, sloppy]
    a_hi, a_lo = marginal_action_pair(theta)
    g = loss_grad_u(theta, a_hi, a_lo, control=control)
    gn2 = float(g @ g)
    out = {"n_sloppy": int(sloppy.sum()), "a_hi": a_hi, "a_lo": a_lo,
           "g_norm": float(np.sqrt(gn2))}
    if gn2 < 1e-300:
        out.update(drs_geo=np.nan, drs_var=np.nan)
        return out, lam, V
    out["drs_geo"] = float(np.sum((S.T @ g) ** 2) / gn2)
    tot = float(g @ Sigma @ g)
    contrib = sum((V[:, j] @ g) ** 2 * lam[j] for j in np.where(sloppy)[0])
    out["drs_var"] = float(contrib / tot) if tot > 0 else np.nan
    out["margin"] = float(loss(theta, a_hi) - loss(theta, a_lo))
    out["margin_sd"] = float(np.sqrt(max(tot, 0.0)))
    out["margin_snr"] = (out["margin"] / out["margin_sd"]
                         if out["margin_sd"] > 0 else np.inf)
    if not control:
        out["margin"] = float(loss(theta, a_hi) - loss(theta, a_lo))
    return out, lam, V


def fisher_at(theta, n_weeks=T_OBS):
    mu = reported_mean_natural(theta, n_weeks)
    J = sensitivity_matrix(theta, n_weeks)
    return fisher_from_J(J, mu)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(11)
    N = 200
    U = prior_sample(N, rng)
    rows = []
    for i in range(N):
        th = np.exp(U[i])
        r, lam, V = drs_variants(th)
        rc, _, _ = drs_variants(th, control=True)
        sh = np.sqrt(np.diag(prior_cov())) / np.sqrt(np.clip(np.diag(
            laplace_posterior(th)[1]), 1e-300, None))
        rows.append({
            "R0": th[0], "rho": th[4], "i0": th[5],
            "peak_count": float(reported_mean_natural(th, T_OBS).max()),
            "n_sloppy": r["n_sloppy"],
            "drs_geo": r["drs_geo"], "drs_var": r["drs_var"],
            "drs_geo_control": rc["drs_geo"], "drs_var_control": rc["drs_var"],
            "margin": r["margin"], "margin_sd": r["margin_sd"],
            "margin_snr": r["margin_snr"],
            "shrink_max": float(sh.max()), "shrink_min": float(sh.min()),
        })
        if (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{N}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "laplace.csv"), index=False)
    print(f"\nwrote results/laplace.csv ({len(df)} rows)")

    summ = {
        "n_cells": len(df),
        "n_sloppy_counts": df["n_sloppy"].value_counts().to_dict(),
        "drs_geo_median": float(df["drs_geo"].median()),
        "drs_geo_control_median": float(df["drs_geo_control"].median()),
        "drs_var_median": float(df["drs_var"].median()),
        "drs_var_control_median": float(df["drs_var_control"].median()),
        "frac_margin_snr_lt_1": float(np.mean(df["margin_snr"] < 1)),
        "frac_margin_snr_lt_0p5": float(np.mean(df["margin_snr"] < 0.5)),
        "shrink_max_median": float(df["shrink_max"].median()),
        "shrink_min_median": float(df["shrink_min"].median()),
    }
    print("\n=== SUMMARY ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")
    with open(os.path.join(RESULTS, "laplace_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
