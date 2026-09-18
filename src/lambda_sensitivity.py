r"""
lambda (cost-ratio) sensitivity analysis for the decision indices.

Motivation: a reviewer will ask "do your conclusions depend on the calibrated
cost ratio lambda?  What if it is 10x larger?"  This script answers that by
recomputing the decision indices on the SAME 192-cell grid with the SAME seeds,
for lambda in {0.1x, 0.3x, 1x, 3x, 10x} x LAMBDA_COST.

No retraining: the BOUNDED NPE checkpoints (checkpoints/npe_W6.pt, npe_W10.pt)
are loaded and only the decision layer is recomputed.

Loss convention (identical to indices._loss_lam / model.loss):

    loss(a | theta) = cumulative_incidence(theta, beta_scale = 1 - EPS_BETA*tau_a)
                      + lambda * tau_a

Consequences exploited here (all verified numerically below):

  * the burden term burden_a(theta) = cumulative_incidence(...) does NOT depend
    on lambda  ->  it is computed ONCE per cell by calling the validated numba
    kernel _batch_actions(..., lambda_cost=0.0) and then any lambda is applied
    analytically as burden + lambda*tau_a;
  * the posterior covariance Sigma (Laplace/Fisher) does NOT depend on lambda;
  * the margin gradient g = grad_u[burden_hi - burden_lo] does NOT depend on
    lambda (the cost term lambda*tau_a is constant in u-space), so
    verify_ridges.loss_grad_u is valid for every lambda;
  * the action PAIR and the optimal action DO depend on lambda and are
    recomputed at every lambda.

Pair convention follows verify_ridges.marginal_action_pair: with
vals = burden + lambda*tau, order = argsort(vals), the pair is
(a_hi, a_lo) = (order[1], order[0]) and margin = vals[a_hi] - vals[a_lo].

Outputs
    results/lambda_sensitivity.csv
    results/lambda_sensitivity_summary.json   (incl. a validation block that
    checks the lambda=1x row against the published cells.csv ppf / margin_snr_true)

Run:  python src/lambda_sensitivity.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    EPS_BETA, LAMBDA_COST, PRIOR_HIGH, PRIOR_LOW, prior_sample, sample_reports,
)
from npe import load_npe, sample_posterior                 # noqa: E402
from indices import _batch_actions                         # noqa: E402
from identifiability_laplace import fisher_at, prior_cov   # noqa: E402
from verify_ridges import loss_grad_u                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")

SEED = 0
N_REP = 8
N_SAMPLES = 2000
PHIS = [2.0, 10.0, 50.0]
WS = [6, 10]
RHOS = list(np.exp(np.linspace(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4]), 4)))
FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]
LAM0 = float(LAMBDA_COST)
TAU = (0.10, 0.30, 0.60)          # model.TAU


def build_seasons(n_rep: int, seed: int):
    """Identical to run_experiment.build_seasons (same rng consumption)."""
    rng = np.random.default_rng(seed)
    seasons = []
    for rho in RHOS:
        for _ in range(n_rep):
            u = prior_sample(1, rng)[0]
            u[4] = np.log(rho)
            seasons.append((rho, np.exp(u)))
    return seasons


def sigma_for(theta: np.ndarray, n_weeks: int) -> np.ndarray:
    """Laplace posterior covariance, identical to identifiability_laplace.drs_variants."""
    F = fisher_at(theta, n_weeks)
    Cp = prior_cov()
    M = np.nan_to_num(F + np.linalg.inv(Cp), nan=0.0, posinf=0.0, neginf=0.0)
    S = np.linalg.pinv(M)
    return (S + S.T) / 2.0


def pair_at(burden: np.ndarray, lam: float):
    """(a_hi, a_lo, vals) with the marginal_action_pair convention."""
    vals = burden + lam * np.asarray(TAU)
    order = np.argsort(vals)
    return int(order[1]), int(order[0]), vals


def main():
    t0 = time.time()
    models = {}
    for W in WS:
        m = load_npe(os.path.join(CKPT, f"npe_W{W}.pt"), double=True)
        meta = getattr(m, "meta", None)
        if meta is not None and not bool(meta.get("bounded", False)):
            raise RuntimeError(f"checkpoint npe_W{W}.pt is NOT the bounded model")
        models[W] = m
    print("[load] bounded checkpoints ok", flush=True)

    seasons = build_seasons(N_REP, SEED + 7)
    rng = np.random.default_rng(SEED + 999)
    lam_values = [f * LAM0 for f in FACTORS]
    n_lam = len(FACTORS)

    # per-cell records, one list per lambda
    per_lam = [{"ppf": [], "snr": [], "best_true": [], "best_plug": [],
                "margin": [], "sd": []} for _ in range(n_lam)]

    total = len(seasons) * len(PHIS) * len(WS)
    k = 0
    for (rho, theta) in seasons:
        for phi in PHIS:
            for W in WS:
                x = sample_reports(theta, n_weeks=W, rng=rng, phi=phi)
                Us = sample_posterior(models[W], x, n=N_SAMPLES, seed=SEED + k)
                THs = np.exp(Us)
                that = np.exp(Us.mean(0))

                # burden (lambda-independent), computed once per cell
                _, _, B_s = _batch_actions(THs, np.asarray(TAU), 0.0, EPS_BETA)
                _, _, b_h = _batch_actions(theta[None, :], np.asarray(TAU), 0.0, EPS_BETA)
                _, _, b_p = _batch_actions(that[None, :], np.asarray(TAU), 0.0, EPS_BETA)
                b_t = b_h[0]
                b_hat = b_p[0]

                Sig = sigma_for(theta, W)          # lambda-independent
                for j, f in enumerate(FACTORS):
                    lam = f * LAM0
                    tau = np.asarray(TAU)

                    acts = np.argmin(B_s + lam * tau[None, :], axis=1)
                    a_hat = int(np.argmin(b_hat + lam * tau))
                    ppf = float(np.mean(acts != a_hat))

                    a_hi, a_lo, vals = pair_at(b_t, lam)
                    margin = float(vals[a_hi] - vals[a_lo])
                    g = loss_grad_u(theta, a_hi, a_lo)      # lambda-independent
                    sd = float(np.sqrt(max(float(g @ Sig @ g), 0.0)))
                    snr = (margin / sd) if sd > 0 else np.inf

                    per_lam[j]["ppf"].append(ppf)
                    per_lam[j]["snr"].append(snr)
                    per_lam[j]["best_true"].append(int(np.argmin(vals)))
                    per_lam[j]["best_plug"].append(a_hat)
                    per_lam[j]["margin"].append(margin)
                    per_lam[j]["sd"].append(sd)

                k += 1
                if k % 24 == 0 or k == total:
                    print(f"[grid] {k}/{total}  elapsed={time.time()-t0:.1f}s",
                          flush=True)

    # ---------------- aggregate ----------------------------------------
    rows = []
    for j, f in enumerate(FACTORS):
        d = per_lam[j]
        ppf = np.asarray(d["ppf"], dtype=float)
        snr = np.asarray(d["snr"], dtype=float)
        bt = np.asarray(d["best_true"])
        bp = np.asarray(d["best_plug"])
        n = len(ppf)
        rows.append({
            "lambda_factor": f,
            "lambda_value": f * LAM0,
            "ppf_median": float(np.median(ppf)),
            "ppf_mean": float(np.mean(ppf)),
            "ppf_iqr25": float(np.percentile(ppf, 25)),
            "ppf_iqr75": float(np.percentile(ppf, 75)),
            "frac_snr_lt_1": float(np.mean(snr < 1)),
            "snr_median": float(np.median(snr)),
            "best_action_0_pct": float(100.0 * np.mean(bt == 0)),
            "best_action_1_pct": float(100.0 * np.mean(bt == 1)),
            "best_action_2_pct": float(100.0 * np.mean(bt == 2)),
            "best_plug_0_pct": float(100.0 * np.mean(bp == 0)),
            "best_plug_1_pct": float(100.0 * np.mean(bp == 1)),
            "best_plug_2_pct": float(100.0 * np.mean(bp == 2)),
            "frac_best_plug_ne_true": float(np.mean(bp != bt)),
            "n_cells": n,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "lambda_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, flush=True)

    # ---------------- validation against the published grid -------------
    val = {}
    try:
        cells = pd.read_csv(os.path.join(RESULTS, "cells.csv"))
        i1 = FACTORS.index(1.0)
        # ppf at 1x must reproduce cells.csv ppf (same seeds / same loss)
        # we recompute the per-cell ppf at 1x here for the comparison
        ppf1 = np.asarray(per_lam[i1]["ppf"], dtype=float)
        if len(ppf1) == len(cells):
            d = np.abs(ppf1 - cells["ppf"].to_numpy())
            val["ppf_1x_max_abs_diff_vs_cells"] = float(np.nanmax(d))
        snr1 = np.asarray(per_lam[i1]["snr"], dtype=float)
        if len(snr1) == len(cells):
            m = np.isfinite(snr1) & np.isfinite(cells["margin_snr_true"].to_numpy())
            val["snr_1x_max_abs_diff_vs_margin_snr_true"] = float(
                np.max(np.abs(snr1[m] - cells["margin_snr_true"].to_numpy()[m])))
            val["snr_1x_median_vs_cells_margin_snr_true"] = (
                float(np.median(snr1)), float(np.median(cells["margin_snr_true"])))
    except Exception as e:  # pragma: no cover
        val["error"] = repr(e)

    summary = {
        "lambda_reference": LAM0,
        "lambda_factors": FACTORS,
        "lambda_values": [f * LAM0 for f in FACTORS],
        "n_cells": int(sum(len(d["ppf"]) for d in per_lam) // n_lam),
        "n_samples_per_cell": N_SAMPLES,
        "npe": "bounded (checkpoints/npe_W6.pt, npe_W10.pt)",
        "seeds": {"seasons": SEED + 7, "observations": SEED + 999,
                  "posterior": "SEED + cell_id"},
        "note": ("burden and posterior covariance are lambda-independent; "
                 "only the action pair / optimal action / margin are recomputed "
                 "per lambda.  Bounded posterior => every draw is inside the "
                 "prior box, so raw PPF == box-reweighted PPF and ESS == 1."),
        "validation": val,
        "table": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    out_json = os.path.join(RESULTS, "lambda_sensitivity_summary.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print("wrote", out_json, flush=True)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False))
    print("validation:", json.dumps(val, indent=2))


if __name__ == "__main__":
    main()
