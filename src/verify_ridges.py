"""
MVP gate: numerical verification of the structural non-identifiability ridge
(R1: rho <-> i0) and a first, noise-free evaluation of the decision-relevant
sloppiness index (DRS).

Sloppy subspace definition used here -- robust, deterministic, PSD by
construction:

    J[u]_{t,k} = d E[reported_t] / d u_k          (n_weeks x 6 parameters)
    F          = J^T W J ,  W_tt = 1 / Var(NB_t)   (expected Fisher information)
    S          = span{ eigenvectors of F with eigenvalue <= kappa * lambda_max }

That is the structural statement "the observations are insensitive along S".
It needs no optimisation, no boundary conditions and no neural approximation,
so it can adjudicate the design before any NPE is trained.

Run:  python src/verify_ridges.py
Writes: results/ridges.csv, results/ridge_profile.csv, results/ridge_example.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import nbinom

sys.path.insert(0, os.path.dirname(__file__))
from model import (N_POP, PHI_DISP, PRIOR_HIGH, PRIOR_LOW, TAU,
                   incidence_by_week, loss, prior_sample,
                   scale_invariant_loss)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PARAM_NAMES = ("R0", "sigma", "gamma", "omega", "rho", "i0")
T_OBS = 6
KAPPA = 1e-4          # relative eigenvalue cut defining the sloppy subspace


# ---------------------------------------------------------------------------
# Observables and likelihood (window-only integration: ~3x faster)
# ---------------------------------------------------------------------------
def reported_mean_natural(theta: np.ndarray, n_weeks: int) -> np.ndarray:
    inc = incidence_by_week(theta, beta_scale=1.0,
                            T_full=float(n_weeks))[:n_weeks]
    return theta[4] * N_POP * inc


def loglik_natural(theta: np.ndarray, y: np.ndarray) -> float:
    mu = np.maximum(reported_mean_natural(theta, len(y)), 1e-12)
    p = PHI_DISP / (PHI_DISP + mu)
    return float(np.sum(nbinom.logpmf(y, PHI_DISP, p)))


def generate_data(theta_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mu = np.maximum(reported_mean_natural(theta_true, T_OBS), 1e-12)
    p = PHI_DISP / (PHI_DISP + mu)
    return rng.negative_binomial(PHI_DISP, p).astype(float)


# ---------------------------------------------------------------------------
# R1 ridge profile (exact likelihood flatness)
# ---------------------------------------------------------------------------
def ridge_profile(theta_true: np.ndarray, y: np.ndarray, k_grid: np.ndarray):
    base = loglik_natural(theta_true, y)
    vals = np.array([
        loglik_natural(np.concatenate([theta_true[:4],
                                       [theta_true[4] / k, theta_true[5] * k]]), y)
        for k in k_grid])
    return vals - base


# ---------------------------------------------------------------------------
# Identifiability geometry
# ---------------------------------------------------------------------------
def sensitivity_matrix(theta: np.ndarray, n_weeks: int, h: float = 1e-4):
    u = np.log(theta)
    J = np.zeros((n_weeks, len(u)))
    for k in range(len(u)):
        e = np.zeros(len(u)); e[k] = h
        mp = reported_mean_natural(np.exp(u + e), n_weeks)
        mm = reported_mean_natural(np.exp(u - e), n_weeks)
        J[:, k] = (mp - mm) / (2 * h)
    return J


def fisher_from_J(J, mu, phi=PHI_DISP):
    var = np.maximum(mu + mu ** 2 / phi, 1e-12)
    Jw = J * np.sqrt(1.0 / var)[:, None]
    return Jw.T @ Jw


def sloppy_subspace(F, kappa=KAPPA):
    lam, V = np.linalg.eigh(F)              # ascending
    lam = np.clip(lam, 0.0, None)
    lam_max = lam.max() if lam.max() > 0 else 1.0
    return lam, V, (lam <= kappa * lam_max)


def loss_grad_u(theta, a_hi, a_lo, h=1e-5, control=False):
    f = scale_invariant_loss if control else loss
    u = np.log(theta)
    g = np.zeros(len(u))
    for i in range(len(u)):
        e = np.zeros(len(u)); e[i] = h
        g[i] = (f(np.exp(u + e), a_hi) - f(np.exp(u + e), a_lo)
                - f(np.exp(u - e), a_hi) + f(np.exp(u - e), a_lo)) / (2 * h)
    return g


def marginal_action_pair(theta):
    vals = np.array([loss(theta, a) for a in range(len(TAU))])
    order = np.argsort(vals)
    return int(order[1]), int(order[0])


def analyse(theta, y=None, n_weeks=T_OBS, kappa=KAPPA):
    mu = reported_mean_natural(theta, n_weeks)
    J = sensitivity_matrix(theta, n_weeks)
    F = fisher_from_J(J, mu)
    lam, V, sloppy = sloppy_subspace(F, kappa)
    S = V[:, sloppy]
    a_hi, a_lo = marginal_action_pair(theta)
    out = {"theta": theta, "peak_count": float(mu.max()),
           "n_sloppy": int(sloppy.sum()), "lam": lam,
           "lam_rel_min": float(lam.min() / max(lam.max(), 1e-300)),
           "a_hi": a_hi, "a_lo": a_lo}
    for tag, ctrl in (("drs", False), ("drs_control", True)):
        g = loss_grad_u(theta, a_hi, a_lo, control=ctrl)
        gn = np.linalg.norm(g)
        out[tag] = float(np.sum((S.T @ g) ** 2) / gn ** 2) if gn > 1e-30 else np.nan
    v = V[:, 0]
    rd = np.zeros(len(theta)); rd[4] = 1 / np.sqrt(2); rd[5] = -1 / np.sqrt(2)
    out["cos_vmin_ridge"] = float(abs(v @ rd))
    if y is not None:
        kg = np.exp(np.linspace(0, np.log(PRIOR_HIGH[4] / PRIOR_LOW[4]), 30))
        prof = ridge_profile(theta, y, kg)
        out["ridge_max_abs_dLL"] = float(np.abs(prof).max())
        out["ridge_dLL_span"] = float(prof.max() - prof.min())
        out["_profile"] = (kg, prof)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)

    rng = np.random.default_rng(7)
    th = np.exp(prior_sample(1, rng)[0])
    y = generate_data(th, rng)
    print("theta_true :", dict(zip(PARAM_NAMES, np.round(th, 6))))
    print("observed   :", y)
    r = analyse(th, y)
    print(f"\npeak expected count      : {r['peak_count']:.1f}")
    print(f"R1 ridge max|dLL|        : {r['ridge_max_abs_dLL']:.4f}")
    print(f"information eigenvalues  : {np.array2string(r['lam'], precision=4)}")
    print(f"lambda_min/lambda_max    : {r['lam_rel_min']:.3e}")
    print(f"|cos(v_min, ridge dir)|  : {r['cos_vmin_ridge']:.4f}")
    print(f"n_sloppy (kappa={KAPPA}) : {r['n_sloppy']}")
    print(f"marginal action pair     : hi={r['a_hi']} lo={r['a_lo']}")
    print(f"DRS (burden loss)        : {r['drs']:.4f}")
    print(f"DRS (scale-inv control)  : {r['drs_control']:.4f}")

    N = 300
    rng = np.random.default_rng(11)
    U = prior_sample(N, rng)
    rows = []
    for i in range(N):
        t = np.exp(U[i])
        yy = generate_data(t, rng)
        rr = analyse(t, yy)
        rows.append({
            "R0": t[0], "sigma": t[1], "gamma": t[2], "omega": t[3],
            "rho": t[4], "i0": t[5],
            "peak_count": rr["peak_count"],
            "ridge_max_abs_dLL": rr["ridge_max_abs_dLL"],
            "lam_rel_min": rr["lam_rel_min"],
            "cos_vmin_ridge": rr["cos_vmin_ridge"],
            "n_sloppy": rr["n_sloppy"],
            "drs": rr["drs"], "drs_control": rr["drs_control"],
            "a_hi": rr["a_hi"], "a_lo": rr["a_lo"],
        })
        if (i + 1) % 60 == 0:
            print(f"  ... {i+1}/{N}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ridges.csv"), index=False)
    print(f"\nwrote results/ridges.csv ({len(df)} rows)")
    print("\n--- summary ---")
    print(f"|cos(v_min, ridge)| : median={df['cos_vmin_ridge'].median():.3f} "
          f"p25={df['cos_vmin_ridge'].quantile(.25):.3f}")
    print(f"n_sloppy            : {df['n_sloppy'].value_counts().to_dict()}")
    print(f"DRS                 : median={df['drs'].median():.3f} "
          f"mean={df['drs'].mean():.3f} frac>0.5={np.mean(df['drs']>0.5):.3f}")
    print(f"DRS_control         : median={df['drs_control'].median():.3f} "
          f"mean={df['drs_control'].mean():.3f}")
    print(f"ridge max|dLL|      : median={df['ridge_max_abs_dLL'].median():.3f} "
          f"p90={df['ridge_max_abs_dLL'].quantile(.9):.3f}")

    kg, prof = r["_profile"]
    np.savetxt(os.path.join(RESULTS, "ridge_profile.csv"),
               np.column_stack([kg, prof]), delimiter=",", header="k,dLL",
               comments="")
    with open(os.path.join(RESULTS, "ridge_example.json"), "w") as f:
        json.dump({"theta_true": th.tolist(), "observed": y.tolist(),
                   "lam": r["lam"].tolist(), "drs": r["drs"],
                   "drs_control": r["drs_control"],
                   "cos_vmin_ridge": r["cos_vmin_ridge"]}, f, indent=2)


if __name__ == "__main__":
    main()
