"""
External-validity arm on REAL influenza seasons.

Why the scale is handled the way it is
--------------------------------------
The model predicts  mu_t = rho * N_POP * incidence_t(theta).  For real
surveillance neither the reporting rate nor the catchment population is known,
and the observed weekly counts (thousands to hundreds of thousands) cannot be
produced by the synthetic prior on rho (0.05-0.60) with N_POP = 1 at all.

So we estimate the combined surveillance multiplier `rho` in two passes:

  pass 1  wide log-uniform prior (1e-2 .. 1e12)  -> locate the scale
  pass 2  log-uniform prior centred on the pass-1 estimate, with width
          `n_decades` -> measure how decision identifiability responds to the
          assumed uncertainty about the reporting multiplier

The question this arm answers is therefore:

    on real seasons, how much of the DECISION margin's posterior variance comes
    from weakly-identified directions, and how does that depend on how much you
    admit not knowing about the reporting multiplier?

Run:  python src/external_validity.py
Writes: results/external_validity.csv, results/external_validity_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

sys.path.insert(0, os.path.dirname(__file__))
from model import (N_POP, PRIOR_HIGH, PRIOR_LOW, TAU, incidence_by_week, loss,
                   scale_invariant_loss)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
W_EXT = 6
DECADES_GRID = (1.0, 2.0, 3.0)
PHI = 10.0
RHO_WIDE = (1e-2, 1e12)

BASE_LO = PRIOR_LOW.copy()
BASE_HI = PRIOR_HIGH.copy()


def mean_counts(theta, n_weeks):
    inc = incidence_by_week(theta, beta_scale=1.0, T_full=float(n_weeks))[:n_weeks]
    return theta[4] * N_POP * inc


def nll(y, mu):
    mu = np.maximum(mu, 1e-12)
    return -float(np.sum(nbinom.logpmf(y, PHI, PHI / (PHI + mu))))


def make_bounds(rho_centre, n_decades):
    lo, hi = BASE_LO.copy(), BASE_HI.copy()
    lo[4] = rho_centre / 10 ** (n_decades / 2)
    hi[4] = rho_centre * 10 ** (n_decades / 2)
    return np.log(lo), np.log(hi)


def fit(y, lo, hi, u0=None, n_start=4, seed=0):
    rng = np.random.default_rng(seed)
    starts = [u0] + [rng.uniform(lo, hi) for _ in range(n_start)] if u0 is not None \
        else [rng.uniform(lo, hi) for _ in range(n_start + 1)]
    best, bestv = None, np.inf
    for s in starts:
        if s is None:
            continue
        s = np.clip(s, lo + 1e-9, hi - 1e-9)
        r = minimize(lambda u: nll(y, mean_counts(np.exp(u), len(y))),
                     s, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options={"maxiter": 600})
        if r.fun < bestv:
            best, bestv = r.x, r.fun
    return np.clip(best, lo, hi), bestv


def fisher_expected(theta, n_weeks, h=1e-4):
    """F = J^T W J, PSD by construction."""
    u = np.log(theta)
    d = len(u)
    J = np.zeros((n_weeks, d))
    for k in range(d):
        e = np.zeros(d); e[k] = h
        J[:, k] = (mean_counts(np.exp(u + e), n_weeks)
                   - mean_counts(np.exp(u - e), n_weeks)) / (2 * h)
    mu = np.maximum(mean_counts(theta, n_weeks), 1e-12)
    var = np.maximum(mu + mu ** 2 / PHI, 1e-12)
    Jw = J * np.sqrt(1.0 / var)[:, None]
    return np.nan_to_num(Jw.T @ Jw, nan=0.0, posinf=0.0, neginf=0.0)


def loss_grad_nat(theta, a_hi, a_lo, h=1e-5, control=False):
    f = scale_invariant_loss if control else loss
    u = np.log(theta)
    g = np.zeros(len(u))
    for i in range(len(u)):
        e = np.zeros(len(u)); e[i] = h
        g[i] = (f(np.exp(u + e), a_hi) - f(np.exp(u + e), a_lo)
                - f(np.exp(u - e), a_hi) + f(np.exp(u - e), a_lo)) / (2 * h)
    return g


def margin_pair(theta):
    vals = np.array([loss(theta, a) for a in range(len(TAU))])
    o = np.argsort(vals)
    return int(o[1]), int(o[0])


def analyse_series(y, n_decades, seed=0):
    # pass 1: locate the scale
    lo1, hi1 = BASE_LO.copy(), BASE_HI.copy()
    lo1[4], hi1[4] = RHO_WIDE
    u1, ll1 = fit(y, np.log(lo1), np.log(hi1), seed=seed)

    # pass 2: prior centred on the pass-1 scale estimate
    lo2, hi2 = make_bounds(np.exp(u1[4]), n_decades)
    u2, ll2 = fit(y, lo2, hi2, u0=u1, seed=seed)
    th = np.exp(u2)

    F = fisher_expected(th, len(y))
    Cp = np.diag(((hi2 - lo2) ** 2) / 12.0)
    M = np.nan_to_num(F + np.linalg.inv(Cp), nan=0.0, posinf=0.0, neginf=0.0)
    Sig = np.linalg.pinv(M)
    Sig = (Sig + Sig.T) / 2.0
    lam, V = np.linalg.eigh(Sig)
    lam = np.clip(lam, 0.0, None)
    pv = np.einsum("ij,ij->j", V, Cp @ V)
    sloppy = lam >= 0.5 * pv

    a_hi, a_lo = margin_pair(th)
    out = {"R0": th[0], "rho": th[4], "i0": th[5], "n_sloppy": int(sloppy.sum()),
           "a_hi": a_hi, "a_lo": a_lo, "nll": float(ll2),
           "rho_pass1": float(np.exp(u1[4])),
           "mean_count": float(y.mean())}
    for tag, ctrl in (("", False), ("_control", True)):
        g = loss_grad_nat(th, a_hi, a_lo, control=ctrl)
        gn2 = float(g @ g)
        tot = float(g @ Sig @ g)
        contrib = sum((V[:, j] @ g) ** 2 * lam[j] for j in np.where(sloppy)[0])
        out["drs_var" + tag] = float(contrib / tot) if tot > 0 else np.nan
        out["drs_geo" + tag] = (float(np.sum((V[:, sloppy].T @ g) ** 2) / gn2)
                                if gn2 > 0 else np.nan)
        if not ctrl:
            out["margin"] = float(loss(th, a_hi) - loss(th, a_lo))
            out["margin_sd"] = float(np.sqrt(max(tot, 0.0)))
            out["margin_snr"] = (out["margin"] / out["margin_sd"]
                                 if out["margin_sd"] > 0 else np.inf)
    return out


def main() -> None:
    df = pd.read_csv(os.path.join(DATA, "seasons.csv"))
    fv = df[(df["source"] == "fluview") & df["ili_count"].notna()]
    seqs = []
    for (reg, sea), g in fv.groupby(["region", "season"]):
        y = g.sort_values("week_index")["ili_count"].to_numpy()[:W_EXT]
        if len(y) == W_EXT and np.all(y > 0):
            seqs.append((reg, sea, y.astype(float)))
    print(f"usable real FluView series: {len(seqs)}")

    rows = []
    for n_dec in DECADES_GRID:
        snrs, drs, drsc = [], [], []
        for i, (reg, sea, y) in enumerate(seqs):
            r = analyse_series(y, n_dec, seed=i)
            r.update(region=reg, season=sea, n_decades=n_dec)
            rows.append(r)
            snrs.append(r["margin_snr"]); drs.append(r["drs_var"])
            drsc.append(r["drs_var_control"])
        snrs = np.array([s for s in snrs if np.isfinite(s)])
        print(f"  decades={n_dec}: n={len(seqs)}  "
              f"SNR median={np.median(snrs):.3f}  frac(SNR<1)={np.mean(snrs<1):.3f}  "
              f"drs_var={np.nanmedian(drs):.3f}  drs_var_ctrl={np.nanmedian(drsc):.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "external_validity.csv"), index=False)
    summ = {}
    for n_dec in DECADES_GRID:
        s = out[out["n_decades"] == n_dec]
        sn = s["margin_snr"].replace([np.inf, -np.inf], np.nan).dropna()
        summ[str(n_dec)] = {
            "n": int(len(s)),
            "margin_snr_median": float(sn.median()) if len(sn) else None,
            "frac_margin_snr_lt_1": float(np.mean(sn < 1)) if len(sn) else None,
            "drs_var_median": float(s["drs_var"].median()),
            "drs_var_control_median": float(s["drs_var_control"].median()),
            "drs_geo_median": float(s["drs_geo"].median()),
            "drs_geo_control_median": float(s["drs_geo_control"].median()),
            "rho_median": float(s["rho"].median()),
        }
    with open(os.path.join(RESULTS, "external_validity_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("\n=== SUMMARY (real FluView seasons) ===")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
