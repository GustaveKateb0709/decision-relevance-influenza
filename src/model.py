"""
Model for decision-relevant non-identifiability in seasonal influenza
antiviral commitment.

Design (why it works):
  * Observables: weekly *reported* cases = rho * (true weekly infections).
    Because nothing anchors the absolute count, rho and the initial seed i0
    enter the observables ONLY through the product rho * i0.
    => exact multiplicative ridge (R1) in log-space along (log rho, log i0).
  * Decision: choose an antiviral commitment level tau in {0.10, 0.30, 0.60}.
    Loss  l(a, theta) = i0 * Phi_a(theta) + lambda * tau_a      (per capita)
    where Phi_a is the seed-amplification factor of the TRUE burden under
    action a. Because the cost term does NOT scale with i0, the optimal
    action flips at  i0* = (lambda * dtau) / (Phi_low - Phi_high).
  * Therefore: two parameter vectors sitting on the same ridge (identical
    observables) can yield DIFFERENT optimal actions.

Units: time in weeks, population fractions (N_POP = 1e6 for counts).
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# Constants of the study design (declared, not estimated)
# ----------------------------------------------------------------------------
N_POP = 1.0e6          # population size (known)
T_FULL = 20.0          # weeks: burden horizon (rest of season)
T_OBS = 10             # weeks: surveillance observation window (default)
DT = 0.02              # weeks: RK4 step
EPS_BETA = 0.50        # antiviral efficacy on transmission (per unit coverage)
TAU = np.array([0.10, 0.30, 0.60])   # antiviral commitment levels (action set)
LAMBDA_COST = 9.827704e-03   # cost per unit coverage, per-capita infection units.
                       # Neutral calibration: lambda = median(i0 * Phi_0) / tau_max,
                       # i.e. the cost ratio at which the intervention is
                       # cost-neutral at the prior-median burden.
                       # See src/calibrate.py (reproducible).
PHI_DISP = 10.0        # negative-binomial dispersion (declared known)

# Parameters, in natural units.  Unknown vector u = log(theta).
PARAM_NAMES = ("R0", "sigma", "gamma", "omega", "rho", "i0")
PRIOR_LOW = np.array([1.15, 2.00, 1.00, 0.005, 0.05, 1.0e-6])
PRIOR_HIGH = np.array([1.50, 7.00, 2.50, 0.040, 0.60, 5.0e-5])
DIM = len(PARAM_NAMES)

# Seed split: fraction of the seed that starts exposed vs infectious
E0_FRAC = 0.30


def u_to_theta(u: np.ndarray) -> np.ndarray:
    """log-space -> natural parameters."""
    return np.exp(np.asarray(u, dtype=float))


def theta_to_u(theta: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(theta, dtype=float))


def prior_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform prior on the log-box. Returns (n, DIM) in u-space."""
    lo, hi = np.log(PRIOR_LOW), np.log(PRIOR_HIGH)
    return rng.uniform(lo, hi, size=(n, DIM))


def prior_logpdf_u(u: np.ndarray) -> np.ndarray:
    """log prior density in u-space (uniform box). Supports (n, DIM) or (DIM,)."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    lo, hi = np.log(PRIOR_LOW), np.log(PRIOR_HIGH)
    inside = np.all((u >= lo) & (u <= hi), axis=1)
    vol = np.sum(hi - lo)
    out = np.where(inside, -vol, -np.inf)
    return out


# ----------------------------------------------------------------------------
# Integrator (numba). Returns weekly incidence and cumulative incidence for one
# action at weekly resolution up to T_FULL.
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _rk4_weekly(R0, sigma, gamma, omega, i0, beta_scale, T_full, dt):
    beta = R0 * gamma * beta_scale
    S = 1.0 - i0
    E = i0 * E0_FRAC
    I = i0 * (1.0 - E0_FRAC)
    R = 0.0
    n_steps = int(round(T_full / dt))
    n_weeks = int(np.floor(T_full + 1e-9))
    inc_week = np.zeros(n_weeks)
    cum = 0.0
    wk = 0
    acc = 0.0
    dt_1 = dt * 0.5
    dt_2 = dt * 0.5
    dt_3 = dt
    dt_6 = dt / 6.0
    for k in range(n_steps):
        # derivative
        dS1 = -beta * S * I + omega * R
        dE1 = beta * S * I - sigma * E
        dI1 = sigma * E - gamma * I
        dR1 = gamma * I - omega * R
        new1 = beta * S * I

        S2 = S + dt_1 * dS1; E2 = E + dt_1 * dE1; I2 = I + dt_1 * dI1; R2 = R + dt_1 * dR1
        dS2 = -beta * S2 * I2 + omega * R2
        dE2 = beta * S2 * I2 - sigma * E2
        dI2 = sigma * E2 - gamma * I2
        dR2 = gamma * I2 - omega * R2
        new2 = beta * S2 * I2

        S3 = S + dt_1 * dS2; E3 = E + dt_1 * dE2; I3 = I + dt_1 * dI2; R3 = R + dt_1 * dR2
        dS3 = -beta * S3 * I3 + omega * R3
        dE3 = beta * S3 * I3 - sigma * E3
        dI3 = sigma * E3 - gamma * I3
        dR3 = gamma * I3 - omega * R3
        new3 = beta * S3 * I3

        S4 = S + dt_3 * dS3; E4 = E + dt_3 * dE3; I4 = I + dt_3 * dI3; R4 = R + dt_3 * dR3
        dS4 = -beta * S4 * I4 + omega * R4
        dE4 = beta * S4 * I4 - sigma * E4
        dI4 = sigma * E4 - gamma * I4
        dR4 = gamma * I4 - omega * R4
        new4 = beta * S4 * I4

        S += dt_6 * (dS1 + 2.0 * dS2 + 2.0 * dS3 + dS4)
        E += dt_6 * (dE1 + 2.0 * dE2 + 2.0 * dE3 + dE4)
        I += dt_6 * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)
        R += dt_6 * (dR1 + 2.0 * dR2 + 2.0 * dR3 + dR4)
        newint = dt_6 * (new1 + 2.0 * new2 + 2.0 * new3 + new4)

        acc += newint
        cum += newint
        if (k + 1) * dt >= (wk + 1) - 1e-9 and wk < n_weeks:
            inc_week[wk] = acc
            acc = 0.0
            wk += 1
    return inc_week, cum


def incidence_by_week(theta: np.ndarray, beta_scale: float = 1.0,
                      T_full: float = T_FULL, dt: float = DT) -> np.ndarray:
    """Weekly new infections (fraction of N_POP) for one parameter vector."""
    R0, sigma, gamma, omega, rho, i0 = theta
    inc, _cum = _rk4_weekly(R0, sigma, gamma, omega, i0, beta_scale, T_full, dt)
    return inc


def cumulative_incidence(theta: np.ndarray, beta_scale: float = 1.0,
                         T_full: float = T_FULL, dt: float = DT) -> float:
    """Cumulative new infections (fraction of N_POP) over [0, T_full]."""
    R0, sigma, gamma, omega, rho, i0 = theta
    _inc, cum = _rk4_weekly(R0, sigma, gamma, omega, i0, beta_scale, T_full, dt)
    return cum


def incidence_batch(U: np.ndarray, beta_scale: float = 1.0,
                    T_full: float = T_FULL, dt: float = DT,
                    n_weeks: int | None = None):
    """(n, DIM) u-space -> (n, n_weeks) weekly incidence, (n,) cumulative."""
    n = U.shape[0]
    nw = int(np.floor(T_full + 1e-9)) if n_weeks is None else n_weeks
    out = np.zeros((n, nw))
    cum = np.zeros(n)
    for i in range(n):
        th = np.exp(U[i])
        inc, c = _rk4_weekly(th[0], th[1], th[2], th[3], th[5],
                             beta_scale, T_full, dt)
        out[i] = inc[:nw]
        cum[i] = c
    return out, cum


# ----------------------------------------------------------------------------
# Observation model: reported cases
# ----------------------------------------------------------------------------
def reported_mean(theta: np.ndarray, n_weeks: int = T_OBS) -> np.ndarray:
    """Expected reported cases per week, weeks 1..n_weeks (baseline, a = none)."""
    inc = incidence_by_week(theta, beta_scale=1.0)[:n_weeks]
    return theta[4] * N_POP * inc          # rho * true infections


def sample_reports(theta: np.ndarray, n_weeks: int = T_OBS,
                   rng: np.random.Generator = None,
                   phi: float = PHI_DISP) -> np.ndarray:
    """Draw weekly reported counts ~ NegBinomial(mean=mu, dispersion=phi)."""
    if rng is None:
        rng = np.random.default_rng()
    mu = reported_mean(theta, n_weeks)
    mu = np.maximum(mu, 1e-12)
    n = phi
    p = phi / (phi + mu)
    return rng.negative_binomial(n, p).astype(float)


# ----------------------------------------------------------------------------
# Burden, loss, decision
# ----------------------------------------------------------------------------
def burden(theta: np.ndarray, action: int) -> float:
    """TRUE total infections over [0, T_FULL] under action `action` (absolute)."""
    beta_scale = 1.0 - EPS_BETA * TAU[action]
    return N_POP * cumulative_incidence(theta, beta_scale=beta_scale)


def loss(theta: np.ndarray, action: int) -> float:
    """Per-capita loss: true burden + intervention cost. Cost does NOT scale i0."""
    return burden(theta, action) / N_POP + LAMBDA_COST * TAU[action]


def best_action(theta: np.ndarray) -> int:
    vals = [loss(theta, a) for a in range(len(TAU))]
    return int(np.argmin(vals))


def loss_vector(theta: np.ndarray) -> np.ndarray:
    return np.array([loss(theta, a) for a in range(len(TAU))])


def scale_invariant_loss(theta: np.ndarray, action: int) -> float:
    """CONTROL loss: only the *relative* reduction matters (no i0, no absolute
    burden). This is the DRS -> 0 control arm of the design."""
    cum0 = cumulative_incidence(theta, beta_scale=1.0)
    cuma = cumulative_incidence(theta, beta_scale=1.0 - EPS_BETA * TAU[action])
    rel_reduction = (cum0 - cuma) / max(cum0, 1e-300)
    return (1.0 - rel_reduction) + LAMBDA_COST * TAU[action]


def best_action_control(theta: np.ndarray) -> int:
    vals = [scale_invariant_loss(theta, a) for a in range(len(TAU))]
    return int(np.argmin(vals))
