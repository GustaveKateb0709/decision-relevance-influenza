"""Robustness probe: PPF under a box-truncated (prior-reweighted) posterior.

The NPE flow places ~99.95% of its mass outside the prior box (verified),
which inflates PPF.  Here we recompute PPF using the prior as a proposal and
reweighting by the flow density q(theta|x) (uniform prior => the reweighted
distribution is the flow posterior restricted to the prior box).

Writes results/ppf_boxweighted.csv  (cell_id, W, phi, rho_level,
ppf_flow, ppf_boxw, ess_frac).
Read-only w.r.t. everything else.
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # this file lives in <root>/src/
SRC = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")
sys.path.insert(0, SRC)

from model import (PHI_DISP, PRIOR_HIGH, PRIOR_LOW, TAU, EPS_BETA,
                   prior_sample, sample_reports)          # noqa: E402
from npe import load_npe, sample_posterior                 # noqa: E402
from indices import _batch_actions, _loss_vector_lam       # noqa: E402

LAM = 0.00982770365058944
SEED = 0
N_REP = 8
N_SAMPLES = 2000
N_PRIOR = 6000
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
    seeds = {W: load_npe(os.path.join(CKPT, f"npe_W{W}.pt"), double=True) for W in WS}
    seasons = build_seasons(N_REP, seed=SEED + 7)
    rng = np.random.default_rng(SEED + 999)        # reproduces cells.csv draws
    rng_prior = np.random.default_rng(SEED + 424242)   # separate stream for IS
    rows = []
    k = 0
    for (rho, theta) in seasons:
        for phi in PHIS:
            for W in WS:
                x = sample_reports(theta, n_weeks=W, rng=rng, phi=phi)
                m = seeds[W]
                Us = sample_posterior(m, x, n=N_SAMPLES, seed=SEED + k)
                a, _, _ = _batch_actions(np.exp(Us), TAU, LAM, EPS_BETA)
                uhat = Us.mean(0)
                ahat = int(np.argmin(_loss_vector_lam(np.exp(uhat), LAM)))
                ppf_flow = float(np.mean(a != ahat))
                # support check: with the bounded bijector every draw is in-box,
                # so the box-truncated reweighting is a no-op and ESS == 1.
                lo_b, hi_b = np.log(PRIOR_LOW), np.log(PRIOR_HIGH)
                frac_in_box = float(np.mean(np.all((Us >= lo_b) & (Us <= hi_b), axis=1)))
                if frac_in_box > 0.999:
                    ppf_boxw, ess = ppf_flow, 1.0
                else:
                    Up = prior_sample(N_PRIOR, rng_prior)
                    with torch.no_grad():
                        lq = m.log_prob(
                            torch.tensor(Up, dtype=torch.float64),
                            torch.tensor(x, dtype=torch.float64).reshape(1, -1).expand(N_PRIOR, -1),
                        ).numpy()
                    w = np.exp(lq - np.nanmax(lq))
                    w = np.nan_to_num(w, nan=0.0)
                    if w.sum() <= 0:
                        w = np.ones(N_PRIOR)
                    w = w / w.sum()
                    ap, _, _ = _batch_actions(np.exp(Up), TAU, LAM, EPS_BETA)
                    ppf_boxw = float(np.sum(w * (ap != ahat)))
                    ess = float(1.0 / np.sum(w ** 2) / N_PRIOR)
                rows.append({"cell_id": k, "W": W, "phi": phi, "rho_level": rho,
                             "ppf_flow": ppf_flow, "ppf_boxw": ppf_boxw,
                             "ess_frac": ess, "frac_in_box": frac_in_box})
                k += 1
                if k % 24 == 0:
                    print(f"{k}/{len(seasons)*len(PHIS)*len(WS)}", flush=True)
    out = os.path.join(RES, "ppf_boxweighted.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    pf = np.array([r["ppf_flow"] for r in rows])
    pb = np.array([r["ppf_boxw"] for r in rows])
    es = np.array([r["ess_frac"] for r in rows])
    fb = np.array([r["frac_in_box"] for r in rows])
    print(f"median ppf_flow={np.median(pf):.4f}  median ppf_boxw={np.median(pb):.4f}")
    print(f"ratio boxw/flow median={np.median(pb/pf):.3f}  median ESS_frac={np.median(es):.3f}")
    print(f"median frac_in_box={np.median(fb):.4f}  (1.0 => bounded model, reweighting is a no-op)")
    print("wrote", out)


if __name__ == "__main__":
    main()
