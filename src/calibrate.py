"""
Neutral calibration of the intervention cost ratio LAMBDA_COST.

Rule (declared, not tuned to any outcome):
    lambda = median_over_prior( i0 * Phi_0 ) / max(tau)

i.e. the cost per unit antiviral coverage is set so that the intervention is
cost-neutral at the *prior-median* uncontrolled burden. This is a purely
epidemiological calibration of a policy cost ratio; it makes no reference to
PPF, DRS, or any downstream result.

Run:  python src/calibrate.py
Writes: results/calibration.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import (PRIOR_HIGH, PRIOR_LOW, TAU, EPS_BETA, cumulative_incidence,
                   prior_sample)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def phi_vector(theta: np.ndarray) -> np.ndarray:
    """Per-seed burden amplification under each action."""
    return np.array([cumulative_incidence(theta, beta_scale=1.0 - EPS_BETA * t)
                     for t in TAU])


def neutral_lambda(n: int = 2000, seed: int = 1) -> float:
    """lambda = median(per-capita uncontrolled burden) / max(tau).

    NOTE (bug fixed 2026-09-16): `cumulative_incidence` already returns the
    per-capita burden for the given theta -- it is proportional to i0 because
    the ODE is integrated from that theta's own initial condition. An earlier
    version multiplied by TH[:,5] (= i0) a second time, which made lambda
    ~1.1e5 times too small; the cost term then vanished and the decision
    degenerated to "always maximum coverage" (PPF identically 0).
    See results/lambda_calibration_diagnosis.json.
    """
    rng = np.random.default_rng(seed)
    TH = np.exp(prior_sample(n, rng))
    PA = np.array([phi_vector(th) for th in TH])
    burden0 = PA[:, 0]                      # per-capita burden already includes i0
    return float(np.median(burden0) / TAU.max())


def ridge_ppf(lam: float, n: int = 400, seed: int = 1, n_ridge: int = 25) -> dict:
    """Decision-flip fraction along the exact (rho, i0) ridge.

    Walks log(rho) over its full prior span; i0 is compensated so the
    observables are (to numerical accuracy) unchanged.

    NOTE: PA[i] is the per-capita burden for theta_i and ALREADY contains that
    theta's own i0.  Along the ridge i0 changes with rho, so we must rescale
    PA by the i0 ratio -- the earlier version multiplied by theta's i0 again.
    """
    rng = np.random.default_rng(seed)
    TH = np.exp(prior_sample(n, rng))
    PA = np.array([phi_vector(th) for th in TH])
    grid = np.linspace(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4]), n_ridge)
    ppf = np.empty(n)
    for i, th in enumerate(TH):
        base = th[4] * th[5]                       # invariant along the ridge
        ref_i0 = th[5]
        acts = []
        for s in grid:
            rho = np.exp(s)
            i0 = base / rho
            burden_a = PA[i] * (i0 / ref_i0)       # rescale to the i0 at this point
            acts.append(int(np.argmin(burden_a + lam * TAU)))
        a_plug = int(np.argmin(PA[i] + lam * TAU))
        ppf[i] = np.mean([a != a_plug for a in acts])
    return {
        "ppf_mean": float(ppf.mean()),
        "ppf_median": float(np.median(ppf)),
        "frac_ppf_gt_0.2": float(np.mean(ppf > 0.2)),
        "frac_ppf_gt_0.4": float(np.mean(ppf > 0.4)),
    }


def main() -> None:
    lam = neutral_lambda()
    print(f"neutral lambda = {lam:.6e}")
    diag = ridge_ppf(lam)
    print("ridge PPF:", json.dumps(diag, indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "calibration.json"), "w") as f:
        json.dump({"lambda_cost": lam, "rule": "median(i0*Phi_0)/max(tau)",
                   "ridge_ppf": diag}, f, indent=2)
    print("wrote results/calibration.json")


if __name__ == "__main__":
    main()
