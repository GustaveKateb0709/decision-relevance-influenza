r"""
Full experiment driver for the synthetic-grid evaluation.

    1. train two NPEs  (W = 6 and W = 10),  checkpoint -> checkpoints/
    2. evaluate the decision-relevance indices on a crossed grid
           rho  log-spaced over the prior [0.05, 0.60]  (4 levels, 12-fold span)
           phi  in {2, 10, 50}             (over-dispersion stress)
           W    in {6, 10}
       with a fixed number of "true seasons" per rho level
    3. SBC-style posterior coverage on 200 seasons
    4. write results/cells.csv and results/npe_meta.json

Run:  python src/run_experiment.py --quick      # fast end-to-end check
      python src/run_experiment.py              # full run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    LAMBDA_COST,
    N_POP,
    PARAM_NAMES,
    PHI_DISP,
    PRIOR_HIGH,
    PRIOR_LOW,
    T_OBS,
    best_action,
    best_action_control,
    prior_sample,
    sample_reports,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")

# NOTE: the study brief specified rho in {0.02, 0.05, 0.15, 0.50}, which matched
# an earlier version of model.py whose rho-prior was [0.02, 0.50].  model.py was
# later revised (rho-prior -> [0.05, 0.60]); to stay inside the prior support we
# use four log-spaced levels spanning the *current* prior box.  Override with
# --rhos if a different grid is wanted.
RHOS = list(np.exp(np.linspace(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4]), 4)))
PHIS = [2.0, 10.0, 50.0]
WS = [6, 10]


def _sha1(path: str) -> str:
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return ""


def train_models(args):
    import torch
    torch.set_num_threads(int(os.environ.get("NPE_THREADS", "1")))
    from npe import train_npe

    os.makedirs(CKPT_DIR, exist_ok=True)
    metas = {}
    # one shared prior draw so W=6 and W=10 see the same seasons
    rng = np.random.default_rng(args.seed)
    U_shared = prior_sample(args.n_sim, rng)

    for W in WS:
        out = os.path.join(CKPT_DIR, f"npe_W{W}.pt")
        rngW = np.random.default_rng(args.seed + 100 + W)
        TH = np.exp(U_shared)
        X = np.empty((args.n_sim, W))
        for i in range(args.n_sim):
            X[i] = sample_reports(TH[i], n_weeks=W, rng=rngW, phi=args.phi_train)
        print(f"[train] W={W}: simulating {args.n_sim} seasons done", flush=True)
        res = train_npe(W, n_sim=args.n_sim, epochs=args.epochs, batch=args.batch,
                        lr=args.lr, seed=args.seed, out_path=out,
                        data=(U_shared, X), phi=args.phi_train, verbose=True,
                        embed=args.embed, maf_hidden=args.maf_hidden,
                        standardize=not args.no_standardize,
                        bounded=args.bounded)
        metas[f"W{W}"] = res["meta"]
    return metas


def build_seasons(n_rep: int, seed: int):
    """Return list of (rho, theta_true) -- n_rep seasons at each rho level."""
    rng = np.random.default_rng(seed)
    seasons = []
    for rho in RHOS:
        for _ in range(n_rep):
            u = prior_sample(1, rng)[0]
            u[4] = np.log(rho)
            seasons.append((rho, np.exp(u)))
    return seasons


def run_grid(args, metas):
    from npe import load_npe
    from indices import posterior_indices

    seeds = {W: load_npe(os.path.join(CKPT_DIR, f"npe_W{W}.pt"), double=True)
             for W in WS}
    seasons = build_seasons(args.n_rep, seed=args.seed + 7)

    rows = []
    rng = np.random.default_rng(args.seed + 999)
    t0 = time.time()
    total = len(seasons) * len(PHIS) * len(WS)
    k = 0
    per_cell_times = []
    for (rho, theta) in seasons:
        for phi in PHIS:
            for W in WS:
                tc = time.time()
                x = sample_reports(theta, n_weeks=W, rng=rng, phi=phi)
                rec = posterior_indices(seeds[W], x, theta, phi=phi,
                                        n_samples=args.n_samples,
                                        seed=args.seed + k, W=W,
                                        lam=args.lam)
                rec["rho_level"] = rho
                rec["phi_train"] = args.phi_train
                rec["cell_id"] = k
                rows.append(rec)
                per_cell_times.append(time.time() - tc)
                k += 1
                if k % 10 == 0 or k == total:
                    el = time.time() - t0
                    print(f"[grid] {k}/{total} cells  elapsed={el:.1f}s  "
                          f"last_cell={per_cell_times[-1]:.2f}s", flush=True)

    print(f"[grid] done: {len(rows)} cells in {time.time() - t0:.1f}s", flush=True)
    return rows, float(np.mean(per_cell_times)), float(np.median(per_cell_times))


def sbc_coverage(args, metas, n_seasons=200, n_samples=1000):
    """Posterior coverage of the TRUE parameters (SBC-style)."""
    from npe import load_npe, sample_posterior

    models = {W: load_npe(os.path.join(CKPT_DIR, f"npe_W{W}.pt"), double=True)
              for W in WS}
    rng = np.random.default_rng(args.seed + 2024)
    out = {}
    for W in WS:
        zs = {nm: [] for nm in PARAM_NAMES}
        cov90 = []
        ppf_list = []
        pred_cov = []
        for _ in range(n_seasons):
            u = prior_sample(1, rng)[0]
            # spread the true rho evenly across the *current* prior range
            rho = float(np.exp(rng.uniform(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4]))))
            u[4] = np.log(rho)
            theta = np.exp(u)
            x = sample_reports(theta, n_weeks=W, rng=rng, phi=PHI_DISP)
            Us = sample_posterior(models[W], x, n=n_samples, seed=int(rng.integers(1e9)))
            um, us = Us.mean(0), Us.std(0)
            z = (u - um) / np.where(us > 1e-12, us, np.nan)
            for j, nm in enumerate(PARAM_NAMES):
                zs[nm].append(float(z[j]))
            cov90.append(float(np.mean(np.abs(z) < 1.6448536269514722)))
            # prediction coverage (mean only)
            from indices import _reported_mean_batch
            mu = _reported_mean_batch(np.exp(Us), W, N_POP)
            lo = mu.mean(0) - 1.6448536269514722 * mu.std(0)
            hi = mu.mean(0) + 1.6448536269514722 * mu.std(0)
            pred_cov.append(float(np.mean((x >= lo) & (x <= hi))))
        out[f"W{W}"] = {
            "n_seasons": n_seasons,
            "mean_cov90_param": float(np.mean(cov90)),
            "mean_pred_cov90_meanonly": float(np.mean(pred_cov)),
            "z_mean": {nm: float(np.mean(zs[nm])) for nm in PARAM_NAMES},
            "z_std": {nm: float(np.std(zs[nm])) for nm in PARAM_NAMES},
            "cov90_per_dim": {
                nm: float(np.mean(np.abs(np.array(zs[nm])) < 1.6448536269514722))
                for nm in PARAM_NAMES
            },
        }
        print(f"[sbc] W={W}: cov90_param={out[f'W{W}']['mean_cov90_param']:.3f} "
              f"pred_cov90={out[f'W{W}']['mean_pred_cov90_meanonly']:.3f}", flush=True)
    return out


def smoke_test(args, metas):
    from npe import load_npe
    from indices import posterior_indices, decision_gradient, decision_gradient_torch

    W = 6
    model = load_npe(os.path.join(CKPT_DIR, f"npe_W{W}.pt"), double=True)
    rng = np.random.default_rng(args.seed + 31)
    u = prior_sample(1, rng)[0]
    u[4] = np.log(0.05)
    theta = np.exp(u)
    x = sample_reports(theta, n_weeks=W, rng=rng, phi=PHI_DISP)
    rec = posterior_indices(model, x, theta, phi=PHI_DISP,
                            n_samples=args.n_samples, seed=0, W=W,
                            compute_grad_check=True, lam=args.lam)
    keep = ["W", "rho_true", "i0_true", "theta_hat_rho", "theta_hat_i0",
            "best_true", "best_plug", "ppf", "er_rel", "drs", "drs_avg_pairs",
            "drs_control", "drs_control_avg_pairs", "n_sloppy", "relsd_max",
            "pred_cov90", "grad_check_relerr", "marginal_pair", "lam",
            "margin_marginal"]
    print("[smoke] observation x =", np.asarray(x).tolist())
    for k in keep:
        print(f"[smoke] {k:24s} = {rec[k]}")
    # grad check on a couple more points
    gc = []
    for _ in range(3):
        uu = prior_sample(1, rng)[0]
        th = np.exp(uu)
        g_fd = decision_gradient(th, 0, 1)
        g_ad = decision_gradient_torch(th, 0, 1)
        gc.append(float(np.linalg.norm(g_fd - g_ad) / max(np.linalg.norm(g_ad), 1e-30)))
    print("[smoke] FD-vs-autograd gradient rel-err (3 pts):", gc)
    return {"smoke": {k: (rec[k] if not isinstance(rec[k], np.floating) else float(rec[k]))
                      for k in keep},
            "grad_check_extra": gc}


def write_cells(rows, path):
    # drop non-scalar helpers
    keys = [k for k in rows[0].keys() if k != "eigvals"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})


def summarize_rows(rows):
    """Overall + grouped medians for the headline indices."""
    import statistics as st

    def med(vals):
        vals = [v for v in vals if v == v and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def grp(key):
        return {r[key] if not isinstance(r[key], float) else round(r[key], 6) for r in rows}

    cols = ["ppf", "er_rel", "drs", "drs_var", "drs_control", "drs_control_var",
            "drs_1em4", "drs_control_1em4", "cos_g_ridge", "n_sloppy",
            "relsd_max", "pred_cov90", "cov90_param"]
    out = {"n_cells": len(rows)}
    out["overall"] = {c: med([r.get(c, np.nan) for r in rows]) for c in cols}
    # fraction of cells whose control DRS is smaller than the main DRS
    out["frac_drs_var_gt_control"] = float(np.mean(
        [r["drs_var"] > r["drs_control_var"] for r in rows
         if np.isfinite(r.get("drs_var", np.nan)) and np.isfinite(r.get("drs_control_var", np.nan))]))
    out["frac_drs_geo_gt_control"] = float(np.mean(
        [r["drs"] > r["drs_control"] for r in rows]))
    out["frac_ppf_gt_0.2"] = float(np.mean([r["ppf"] > 0.2 for r in rows]))
    out["frac_best_plug_ne_true"] = float(np.mean(
        [r["best_plug"] != r["best_true"] for r in rows]))
    # grouped by W, by phi, by rho level
    for key, field in (("W", "W"), ("phi", "phi"), ("rho_level", "rho_level")):
        groups = {}
        for r in rows:
            groups.setdefault(r[field], []).append(r)
        groups = {k: {
            "n": len(v),
            "ppf": med([x["ppf"] for x in v]),
            "er_rel": med([x["er_rel"] for x in v]),
            "drs": med([x["drs"] for x in v]),
            "drs_var": med([x["drs_var"] for x in v]),
            "drs_control": med([x["drs_control"] for x in v]),
            "drs_control_var": med([x["drs_control_var"] for x in v]),
        } for k, v in sorted(groups.items(), key=lambda kv: kv[0])}
        out[f"by_{key}"] = groups
    return out


def _decision_degeneracy(lam=None, n: int = 4000, seed: int = 1) -> dict:
    """Diagnose whether the decision problem carries any information.

    If the burden term dominates the cost term for *every* parameter in the
    prior, all draws select the maximal tier and PPF is trivially 0 -- the
    decision is degenerate.  This helper reports the action distribution over
    the prior and the implied flip threshold in i0, so the issue is visible
    directly in npe_meta.json.
    """
    import math
    from model import EPS_BETA, PRIOR_HIGH, PRIOR_LOW, TAU, prior_sample
    from indices import _batch_actions

    lam = LAMBDA_COST if lam is None else float(lam)
    rng = np.random.default_rng(seed)
    TH = np.exp(prior_sample(n, rng))
    acts, acts_c, losses = _batch_actions(TH, TAU, lam, EPS_BETA)
    dtau02 = float(TAU[2] - TAU[0])
    m02 = losses[:, 0] - losses[:, 2]          # l(a=0) - l(a=2)
    m01 = losses[:, 0] - losses[:, 1]
    # P_a contains i0; dividing by i0 gives the per-seed amplification diff.
    amp02 = float(np.median((m02 + lam * dtau02) / np.maximum(TH[:, 5], 1e-300)))
    i0_med_prior = float(math.exp(0.5 * (math.log(PRIOR_LOW[5]) + math.log(PRIOR_HIGH[5]))))
    i0_star = float(lam * dtau02 / amp02) if amp02 > 0 else float("inf")
    lam_needed = float(i0_med_prior * amp02 / dtau02) if amp02 > 0 else float("nan")
    return {
        "lam": lam,
        "frac_best_action_0": float(np.mean(acts == 0)),
        "frac_best_action_1": float(np.mean(acts == 1)),
        "frac_best_action_2": float(np.mean(acts == 2)),
        "frac_control_action_2": float(np.mean(acts_c == 2)),
        "median_loss_margin_02": float(np.median(m02)),
        "median_loss_margin_01": float(np.median(m01)),
        "median_amplification_diff_02": amp02,
        "i0_star_02": i0_star,
        "i0_prior_range": [float(PRIOR_LOW[5]), float(PRIOR_HIGH[5])],
        "i0_med_prior": i0_med_prior,
        "lam_for_flip_at_prior_median_i0": lam_needed,
        "lam_ratio_needed_over_current": (lam_needed / lam) if lam > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sim", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_rep", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--phi_train", type=float, default=PHI_DISP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--rhos", type=str, default=None,
                    help="comma separated rho levels (default: adaptive to prior)")
    ap.add_argument("--embed", type=int, default=48)
    ap.add_argument("--maf_hidden", type=int, default=96)
    ap.add_argument("--no_standardize", action="store_true",
                    help="disable affine u-space standardisation (it is ON by "
                         "default: without it the NPE is badly miscalibrated)")
    ap.add_argument("--bounded", action="store_true",
                    help="train with the sigmoid bounded-support bijector "
                         "(posterior confined to the prior box; replaces "
                         "standardisation).  See npe.NPE(bounded=True).")
    ap.add_argument("--lam", type=float, default=None,
                    help="override the cost ratio LAMBDA_COST (model.py). "
                         "Default: use model.LAMBDA_COST.  See the "
                         "decision-degeneracy note in npe_meta.json.")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output files, e.g. --tag _lamfix -> "
                         "cells_lamfix.csv / npe_meta_lamfix.json")
    args = ap.parse_args()

    global RHOS
    if args.rhos:
        RHOS = [float(x) for x in args.rhos.split(",")]

    if args.quick:
        args.n_sim = 2000
        args.epochs = 5
        args.batch = 128
        args.n_rep = 1
        args.n_samples = 500

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    t0 = time.time()
    metas = {}
    if not args.skip_train:
        metas = train_models(args)
    else:
        for W in WS:
            p = os.path.join(CKPT_DIR, f"npe_W{W}.pt")
            if os.path.exists(p):
                import torch
                metas[f"W{W}"] = torch.load(p, map_location="cpu",
                                            weights_only=False)["meta"]

    rows, mean_cell, med_cell = run_grid(args, metas)
    cells_path = os.path.join(RESULTS, f"cells{args.tag}.csv")
    meta_path = os.path.join(RESULTS, f"npe_meta{args.tag}.json")
    write_cells(rows, cells_path)
    print(f"[write] {cells_path}  ({len(rows)} rows)", flush=True)

    sbc = sbc_coverage(args, metas, n_seasons=(20 if args.quick else 200),
                       n_samples=(200 if args.quick else 1000))
    smoke = smoke_test(args, metas)

    # summary stats
    summary = summarize_rows(rows)
    summary["mean_cell_seconds"] = mean_cell
    summary["median_cell_seconds"] = med_cell
    summary["wall_seconds"] = time.time() - t0

    meta_out = {
        "config": {**vars(args), "RHOS": RHOS, "PHIS": PHIS, "WS": WS},
        "constants": {"LAMBDA_COST": LAMBDA_COST,
                      "lam_used": float(args.lam) if args.lam is not None else float(LAMBDA_COST),
                      "PHI_DISP": PHI_DISP,
                      "N_POP": N_POP, "T_OBS": T_OBS,
                      "PRIOR_LOW": list(map(float, PRIOR_LOW)),
                      "PRIOR_HIGH": list(map(float, PRIOR_HIGH))},
        "model_py_sha1": _sha1(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "model.py")),
        "train_meta": metas,
        "sbc": sbc,
        "smoke": smoke,
        "summary": summary,
    }
    summary["lam_used"] = meta_out["constants"]["lam_used"]
    summary["decision_degeneracy"] = _decision_degeneracy(args.lam)
    meta_out["summary"] = summary
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    print(f"[write] {meta_path}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
