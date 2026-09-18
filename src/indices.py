r"""
Decision-relevance indices for the antiviral-commitment audit.

Four quantities are computed for a single (true season, observation) pair, all
in the *u-space* (u = log of the natural parameters):

    PPF         posterior-probability-of-flip: fraction of posterior draws
                whose optimal action differs from the plug-in action.
    ER_rel      relative excess risk of the plug-in rule  (dimensionless).
    DRS         Decision-Relevance Score: fraction of the squared decision
                gradient that lives in the *sloppy* subspace of the posterior
                Hessian  H = -grad^2_u log q(u | x).
    DRS_control same quantity for the scale-invariant control loss (no i0),
                which by design should be ~ 0.

Plus parameter-level (relative sds, entropy) and prediction-level (posterior
predictive RMSE / coverage) diagnostics used for the "divergence quadrant".
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    DT,
    DIM,
    EPS_BETA,
    LAMBDA_COST,
    N_POP,
    PARAM_NAMES,
    PHI_DISP,
    PRIOR_HIGH,
    PRIOR_LOW,
    TAU,
    T_FULL,
    _rk4_weekly,
    best_action,
    best_action_control,
    cumulative_incidence,
    loss,
    loss_vector,
    reported_mean,
    scale_invariant_loss,
)

PAIRS = [(0, 1), (0, 2), (1, 2)]
PRIOR_VAR_MEAN = float(np.mean((np.log(PRIOR_HIGH) - np.log(PRIOR_LOW)) ** 2 / 12.0))


# ---------------------------------------------------------------------------
# loss / decision helpers that take an explicit cost ratio ``lam``
# (model.loss uses the module-level LAMBDA_COST; here we allow a corrected one)
# ---------------------------------------------------------------------------
def _loss_lam(theta: np.ndarray, a: int, lam: float) -> float:
    return float(cumulative_incidence(theta, beta_scale=1.0 - EPS_BETA * TAU[a])
                 + lam * TAU[a])


def _loss_vector_lam(theta: np.ndarray, lam: float) -> np.ndarray:
    return np.array([_loss_lam(theta, a, lam) for a in range(len(TAU))])


def _best_action_lam(theta: np.ndarray, lam: float) -> int:
    return int(np.argmin(_loss_vector_lam(theta, lam)))


def _control_loss_lam(theta: np.ndarray, a: int, lam: float) -> float:
    cum0 = cumulative_incidence(theta, beta_scale=1.0)
    cuma = cumulative_incidence(theta, beta_scale=1.0 - EPS_BETA * TAU[a])
    rel = (cum0 - cuma) / max(cum0, 1e-300)
    return float((1.0 - rel) + lam * TAU[a])


def _control_vector_lam(theta: np.ndarray, lam: float) -> np.ndarray:
    return np.array([_control_loss_lam(theta, a, lam) for a in range(len(TAU))])


def is_lambda_degenerate(lam: float, n: int = 2000, seed: int = 1) -> dict:
    """How often the optimal action is the maximal tier over the prior."""
    from model import prior_sample
    rng = np.random.default_rng(seed)
    TH = np.exp(prior_sample(n, rng))
    acts, acts_c, _ = _batch_actions(TH, TAU, lam, EPS_BETA)
    return {"frac_best_is_2": float(np.mean(acts == 2)),
            "frac_best_ctrl_is_2": float(np.mean(acts_c == 2))}


# ============================================================================
# vectorised action / loss computation (numba)
# ============================================================================
@njit(cache=True, fastmath=True)
def _batch_actions(TH, taus, lambda_cost, eps_beta):
    """Optimal action + control action + loss matrix for a batch of theta."""
    n = TH.shape[0]
    na = taus.shape[0]
    actions = np.empty(n, dtype=np.int64)
    actions_control = np.empty(n, dtype=np.int64)
    losses = np.empty((n, na))
    for i in range(n):
        R0 = TH[i, 0]
        sg = TH[i, 1]
        gm = TH[i, 2]
        om = TH[i, 3]
        i0 = TH[i, 5]
        cum0 = 0.0
        for a in range(na):
            bs = 1.0 - eps_beta * taus[a]
            _inc, cum = _rk4_weekly(R0, sg, gm, om, i0, bs, T_FULL, DT)
            losses[i, a] = cum + lambda_cost * taus[a]
            if a == 0:
                cum0 = cum
        bi = 0
        bv = losses[i, 0]
        for a in range(1, na):
            if losses[i, a] < bv:
                bv = losses[i, a]
                bi = a
        actions[i] = bi
        bc = 0
        bcv = 1.0e300
        for a in range(na):
            bs = 1.0 - eps_beta * taus[a]
            _inc, cum = _rk4_weekly(R0, sg, gm, om, i0, bs, T_FULL, DT)
            cl = (1.0 - (cum0 - cum) / cum0) + lambda_cost * taus[a]
            if cl < bcv:
                bcv = cl
                bc = a
        actions_control[i] = bc
    return actions, actions_control, losses


@njit(cache=True, fastmath=True)
def _reported_mean_batch(TH, W, n_pop):
    """(n, W) expected reported counts for a batch of theta."""
    n = TH.shape[0]
    out = np.empty((n, W))
    for i in range(n):
        inc, _cum = _rk4_weekly(TH[i, 0], TH[i, 1], TH[i, 2], TH[i, 3],
                                TH[i, 5], 1.0, T_FULL, DT)
        for w in range(W):
            out[i, w] = TH[i, 4] * n_pop * inc[w]
    return out


# ============================================================================
# decision gradients (finite differences in u-space, consistent with model.py)
# ============================================================================
def decision_gradient(theta: np.ndarray, a1: int, a2: int, h: float = 1e-5,
                      lam: float | None = None) -> np.ndarray:
    """grad_u [ l(a1, theta) - l(a2, theta) ]  (u = log theta).

    ``lam`` overrides the module-level cost ratio (default: model.LAMBDA_COST).
    """
    lam = LAMBDA_COST if lam is None else float(lam)
    u = np.log(np.asarray(theta, dtype=float))
    g = np.empty(DIM)
    for k in range(DIM):
        up = u.copy(); up[k] += h
        um = u.copy(); um[k] -= h
        fp = _loss_lam(np.exp(up), a1, lam) - _loss_lam(np.exp(up), a2, lam)
        fm = _loss_lam(np.exp(um), a1, lam) - _loss_lam(np.exp(um), a2, lam)
        g[k] = (fp - fm) / (2.0 * h)
    return g


def control_gradient(theta: np.ndarray, a1: int, a2: int, h: float = 1e-5,
                     lam: float | None = None) -> np.ndarray:
    """grad_u [ l_control(a1, theta) - l_control(a2, theta) ]."""
    lam = LAMBDA_COST if lam is None else float(lam)
    u = np.log(np.asarray(theta, dtype=float))
    g = np.empty(DIM)
    for k in range(DIM):
        up = u.copy(); up[k] += h
        um = u.copy(); um[k] -= h
        fp = _control_loss_lam(np.exp(up), a1, lam) - _control_loss_lam(np.exp(up), a2, lam)
        fm = _control_loss_lam(np.exp(um), a1, lam) - _control_loss_lam(np.exp(um), a2, lam)
        g[k] = (fp - fm) / (2.0 * h)
    return g


def decision_gradient_torch(theta: np.ndarray, a1: int, a2: int,
                            dt: float = DT) -> np.ndarray:
    """Exact autograd version of :func:`decision_gradient` (torch RK4).

    Used only to validate the finite-difference gradient; the FD version is
    the one used in the pipeline for speed.
    """
    th = torch.tensor(np.asarray(theta, dtype=np.float64), dtype=torch.float64)
    u = torch.log(th).requires_grad_(True)

    def cum(beta_scale):
        th = torch.exp(u)
        R0, sg, gm, om, i0 = th[0], th[1], th[2], th[3], th[5]
        beta = R0 * gm * beta_scale
        S = 1.0 - i0
        E = i0 * 0.30
        I = i0 * 0.70
        R = torch.zeros_like(i0)
        n_steps = int(round(T_FULL / dt))
        c = 0.0
        for _ in range(n_steps):
            dS1 = -beta * S * I + om * R
            dE1 = beta * S * I - sg * E
            dI1 = sg * E - gm * I
            dR1 = gm * I - om * R
            n1 = beta * S * I
            S2 = S + 0.5 * dt * dS1; E2 = E + 0.5 * dt * dE1
            I2 = I + 0.5 * dt * dI1; R2 = R + 0.5 * dt * dR1
            dS2 = -beta * S2 * I2 + om * R2; dE2 = beta * S2 * I2 - sg * E2
            dI2 = sg * E2 - gm * I2; dR2 = gm * I2 - om * R2
            n2 = beta * S2 * I2
            S3 = S + 0.5 * dt * dS2; E3 = E + 0.5 * dt * dE2
            I3 = I + 0.5 * dt * dI2; R3 = R + 0.5 * dt * dR2
            dS3 = -beta * S3 * I3 + om * R3; dE3 = beta * S3 * I3 - sg * E3
            dI3 = sg * E3 - gm * I3; dR3 = gm * I3 - om * R3
            n3 = beta * S3 * I3
            S4 = S + dt * dS3; E4 = E + dt * dE3
            I4 = I + dt * dI3; R4 = R + dt * dR3
            dS4 = -beta * S4 * I4 + om * R4; dE4 = beta * S4 * I4 - sg * E4
            dI4 = sg * E4 - gm * I4; dR4 = gm * I4 - om * R4
            n4 = beta * S4 * I4
            S = S + dt / 6.0 * (dS1 + 2 * dS2 + 2 * dS3 + dS4)
            E = E + dt / 6.0 * (dE1 + 2 * dE2 + 2 * dE3 + dE4)
            I = I + dt / 6.0 * (dI1 + 2 * dI2 + 2 * dI3 + dI4)
            R = R + dt / 6.0 * (dR1 + 2 * dR2 + 2 * dR3 + dR4)
            c = c + dt / 6.0 * (n1 + 2 * n2 + 2 * n3 + n4)
        return c

    f = cum(1.0 - EPS_BETA * TAU[a1]) - cum(1.0 - EPS_BETA * TAU[a2])
    g, = torch.autograd.grad(f, u)
    return g.numpy()


# ============================================================================
# posterior Hessian and DRS
# ============================================================================
def hessian_logq(npe, x_obs: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    """Exact H = -grad^2_u log q(u | x) at u_hat (6x6, symmetric).

    ``npe`` must be a *double* precision model.
    """
    u = torch.tensor(np.asarray(u_hat, dtype=np.float64), dtype=torch.float64,
                     requires_grad=True)
    xt = torch.tensor(np.asarray(x_obs, dtype=np.float64),
                      dtype=torch.float64).reshape(1, -1)

    def f(uu: torch.Tensor) -> torch.Tensor:
        return -npe.log_prob(uu.reshape(1, -1), xt).sum()

    H = torch.autograd.functional.hessian(f, u)
    H = 0.5 * (H + H.transpose(0, 1))
    return H.detach().numpy()


GAMMAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]


def _gtag(gg: float) -> str:
    return f"1em{int(round(-math.log10(gg)))}"


RIDGE_DIR = np.zeros(DIM)
RIDGE_DIR[4] = 1.0 / math.sqrt(2.0)
RIDGE_DIR[5] = -1.0 / math.sqrt(2.0)


def spectral_decomposition(H: np.ndarray):
    """Symmetric eigen-decomposition of the (observed-information) matrix H."""
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)                      # ascending eigenvalues
    lmax = float(w.max())
    if lmax <= 1e-12:                             # degenerate guard
        lmax = 1e-12
    return w, V, lmax


def drs_geometric(V, w, g, gamma: float) -> tuple:
    """DRS = ||P_S g||^2 / ||g||^2 for S = span{v_i : lam_i <= gamma*lam_max}."""
    lmax = float(w.max()) if w.max() > 1e-12 else 1e-12
    mask = w <= gamma * lmax
    den = float(np.sum(g ** 2))
    if not mask.any() or den <= 0:
        return 0.0, mask
    coords = V[:, mask].T @ g
    return float(np.sum(coords ** 2) / den), mask


def drs_variance_weighted(Sigma: np.ndarray, g: np.ndarray,
                          prior_var_mean: float, shrink_thresh: float = 0.5) -> float:
    """Variance-weighted DRS (the metric that *does* separate the two arms).

    Decompose the posterior covariance Sigma; a direction is "sloppy" when the
    data barely shrunk it, i.e. posterior variance >= ``shrink_thresh`` x the
    mean prior variance.  DRS_var is the share of the decision margin's
    posterior variance  g' Sigma g  coming from those directions.
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    lam, V = np.linalg.eigh(Sigma)
    sloppy = lam >= shrink_thresh * prior_var_mean
    tot = float(g @ Sigma @ g)
    if tot <= 0 or not sloppy.any():
        return float("nan")
    contrib = 0.0
    for j in np.where(sloppy)[0]:
        v = V[:, j]
        contrib += (v @ g) ** 2 * lam[j]
    return float(min(contrib / tot, 1.0))


def marginal_pair(lvec: np.ndarray) -> tuple:
    diffs = [abs(lvec[a] - lvec[b]) for a, b in PAIRS]
    return PAIRS[int(np.argmin(diffs))]


# ============================================================================
# main per-cell routine
# ============================================================================
def posterior_indices(npe, x_obs: np.ndarray, theta_true: np.ndarray,
                      phi: float = PHI_DISP, n_samples: int = 2000,
                      seed: int = 0, gamma: float = 1e-2,
                      W: int | None = None, compute_grad_check: bool = False,
                      lam: float | None = None) -> dict:
    """All indices for one (true season, observation) cell.

    ``npe`` must already be a double-precision :class:`npe.NPE`.
    ``lam`` overrides the module-level cost ratio (default model.LAMBDA_COST);
    it only enters the loss/decision quantities, never the posterior itself.
    Returns a flat dict.
    """
    from npe import sample_posterior  # local import to avoid cycles

    lam = LAMBDA_COST if lam is None else float(lam)

    x_obs = np.asarray(x_obs, dtype=float).ravel()
    theta_true = np.asarray(theta_true, dtype=float).ravel()
    if W is None:
        W = x_obs.shape[0]

    # ---- posterior samples -------------------------------------------------
    U_s = sample_posterior(npe, x_obs, n=n_samples, seed=seed)
    TH_s = np.exp(U_s)
    u_mean = U_s.mean(axis=0)
    u_sd = U_s.std(axis=0)
    u_true = np.log(theta_true)
    # central estimate: u-space posterior mean mapped to natural units.  (The
    # naive natural-space mean E[exp(u)] is unstable when the flow has heavy
    # tails -- see the calibration notes in results/npe_meta.json.)
    theta_hat = np.exp(u_mean)
    theta_hat_natural_mean = TH_s.mean(axis=0)
    u_hat = u_mean
    # fraction of posterior draws inside the declared prior box
    lo_b, hi_b = np.log(PRIOR_LOW), np.log(PRIOR_HIGH)
    frac_post_in_box = float(np.mean(np.all((U_s >= lo_b) & (U_s <= hi_b), axis=1)))

    # ---- actions and losses ------------------------------------------------
    acts, acts_ctrl, losses_s = _batch_actions(TH_s, TAU, lam, EPS_BETA)
    l_hat = _loss_vector_lam(theta_hat, lam)
    a_hat = int(np.argmin(l_hat))
    l_ctrl_hat = _control_vector_lam(theta_hat, lam)
    a_ctrl_hat = int(np.argmin(l_ctrl_hat))

    # ---- PPF ---------------------------------------------------------------
    ppf = float(np.mean(acts != a_hat))
    ppf_control = float(np.mean(acts_ctrl != a_ctrl_hat))

    # ---- ER_rel ------------------------------------------------------------
    plug_losses = losses_s[:, a_hat]
    oracle_losses = losses_s[np.arange(acts.shape[0]), acts]
    den_er = float(np.nanmean(oracle_losses))
    if np.isfinite(den_er) and den_er > 0:
        er_rel = float((np.nanmean(plug_losses) - den_er) / den_er)
    else:
        er_rel = float("nan")
    er_rel = max(er_rel, 0.0) if np.isfinite(er_rel) else er_rel

    # ---- true-parameter decision ------------------------------------------
    best_true = _best_action_lam(theta_true, lam)
    best_true_control = int(np.argmin(_control_vector_lam(theta_true, lam)))
    loss_true = _loss_vector_lam(theta_true, lam)
    # regret of the plug-in rule evaluated at the TRUE parameters
    regret_true = float(_loss_lam(theta_true, a_hat, lam) - loss_true[best_true])

    # ---- posterior Hessian & DRS ------------------------------------------
    H = hessian_logq(npe, x_obs, u_hat)
    w, V, lmax = spectral_decomposition(H)
    Sigma_u = np.cov(U_s.T)                       # posterior covariance (u-space)

    # marginal decision pair from the posterior-mean loss
    mp = marginal_pair(l_hat)
    g_marg = decision_gradient(theta_hat, mp[0], mp[1], lam=lam)

    # geometric DRS at several thresholds (primary = gamma)
    drs_by_gamma = {}
    nsl_by_gamma = {}
    for gg in GAMMAS:
        d, mask = drs_geometric(V, w, g_marg, gg)
        drs_by_gamma[gg] = d
        nsl_by_gamma[gg] = int(mask.sum())
    drs = drs_by_gamma[gamma]
    n_sloppy = nsl_by_gamma[gamma]
    drs_var = drs_variance_weighted(Sigma_u, g_marg, PRIOR_VAR_MEAN)

    # average over all action pairs (geometric, at the primary gamma)
    drs_pairs = []
    drs_per_pair = {}
    for (a, b) in PAIRS:
        gg = decision_gradient(theta_hat, a, b, lam=lam)
        d, _ = drs_geometric(V, w, gg, gamma)
        drs_pairs.append(d)
        drs_per_pair[f"drs_pair_{a}{b}"] = d
    drs_avg = float(np.mean(drs_pairs))

    # control arm (scale-invariant loss)
    cmp = marginal_pair(l_ctrl_hat)
    g_c_marg = control_gradient(theta_hat, cmp[0], cmp[1], lam=lam)
    drs_c_by_gamma = {}
    for gg in GAMMAS:
        d, _ = drs_geometric(V, w, g_c_marg, gg)
        drs_c_by_gamma[gg] = d
    drs_c = drs_c_by_gamma[gamma]
    drs_c_var = drs_variance_weighted(Sigma_u, g_c_marg, PRIOR_VAR_MEAN)
    drs_c_pairs = []
    drs_c_per_pair = {}
    for (a, b) in PAIRS:
        gg = control_gradient(theta_hat, a, b, lam=lam)
        d, _ = drs_geometric(V, w, gg, gamma)
        drs_c_pairs.append(d)
        drs_c_per_pair[f"drs_control_pair_{a}{b}"] = d
    drs_c_avg = float(np.mean(drs_c_pairs))

    # alignment of the decision gradient with the structural ridge direction
    def _cos_ridge(g):
        n = np.linalg.norm(g)
        return float(abs(g @ RIDGE_DIR) / n) if n > 0 else 0.0
    cos_g_ridge = _cos_ridge(g_marg)
    cos_g_ridge_control = _cos_ridge(g_c_marg)
    g_norm = float(np.linalg.norm(g_marg))

    # ---- parameter-level diagnostics --------------------------------------
    relsd_theta = TH_s.std(axis=0) / np.maximum(np.abs(TH_s.mean(axis=0)), 1e-300)
    relsd_u = u_sd / np.maximum(np.abs(u_mean), 1e-300)

    # posterior entropy of the u-space flow estimate
    with torch.no_grad():
        logq = npe.log_prob(
            torch.tensor(U_s, dtype=torch.float64),
            torch.tensor(x_obs, dtype=torch.float64).reshape(1, -1).expand(n_samples, -1),
        ).numpy()
    entropy_u = float(-np.mean(logq))

    # parameter coverage based on the true u (SBC-style z score)
    u_true_sd = np.where(u_sd > 1e-12, u_sd, np.nan)
    z_u = (u_true - u_mean) / u_true_sd
    cov90_param = float(np.mean(np.abs(z_u) < 1.6448536269514722))

    # ---- posterior predictive calibration ---------------------------------
    mu_s = _reported_mean_batch(TH_s, W, N_POP)          # (S, W)
    mu_s = np.nan_to_num(mu_s, nan=0.0, posinf=1e12, neginf=0.0)
    pred_mean = mu_s.mean(axis=0)
    pred_rmse = float(np.sqrt(np.mean((pred_mean - x_obs) ** 2)))
    # relative RMSE (scale-free)
    pred_rrmse = float(pred_rmse / np.maximum(np.abs(x_obs).mean(), 1e-12))

    rng = np.random.default_rng(seed + 777)
    mu_clip = np.clip(np.nan_to_num(mu_s, nan=1e12, posinf=1e12), 1e-12, 1e12)
    p_nb = np.clip(phi / (phi + mu_clip), 1e-12, 1.0 - 1e-12)
    y_rep = rng.negative_binomial(phi, p_nb)              # (S, W) posterior predictive draws
    lo = np.quantile(y_rep, 0.05, axis=0)
    hi = np.quantile(y_rep, 0.95, axis=0)
    pred_cov90 = float(np.mean((x_obs >= lo) & (x_obs <= hi)))
    # mean-only (latent) coverage
    lo_m = mu_s.mean(axis=0) - 1.6448536269514722 * mu_s.std(axis=0)
    hi_m = mu_s.mean(axis=0) + 1.6448536269514722 * mu_s.std(axis=0)
    pred_cov90_meanonly = float(np.mean((x_obs >= lo_m) & (x_obs <= hi_m)))

    # ---- optional gradient sanity check -----------------------------------
    grad_check = np.nan
    if compute_grad_check:
        g_t = decision_gradient_torch(theta_hat, mp[0], mp[1])
        grad_check = float(np.linalg.norm(g_marg - g_t) / max(np.linalg.norm(g_t), 1e-30))

    # ---- pack --------------------------------------------------------------
    rec = {
        "W": int(W),
        "phi": float(phi),
        "lam": float(lam),
        "n_samples": int(n_samples),
        "frac_post_in_box": frac_post_in_box,
        "rho_true": float(theta_true[4]),
        "i0_true": float(theta_true[5]),
        "best_true": best_true,
        "best_plug": a_hat,
        "best_true_control": best_true_control,
        "best_plug_control": a_ctrl_hat,
        "ppf": ppf,
        "ppf_control": ppf_control,
        "er_rel": er_rel,
        "regret_true": regret_true,
        "drs": drs,
        "drs_avg_pairs": drs_avg,
        "drs_var": drs_var,
        "drs_control": drs_c,
        "drs_control_avg_pairs": drs_c_avg,
        "drs_control_var": drs_c_var,
        "n_sloppy": n_sloppy,
        "lambda_max": lmax,
        "gamma": gamma,
        "cos_g_ridge": cos_g_ridge,
        "cos_g_ridge_control": cos_g_ridge_control,
        "g_norm": g_norm,
        "drs_control_over_drs": (drs_c / drs) if drs > 0 else float("nan"),
        "grad_check_relerr": grad_check,
        "entropy_u": entropy_u,
        "relsd_max": float(np.nanmax(relsd_theta)),
        "relsd_max_u": float(np.nanmax(relsd_u)),
        "pred_rmse": pred_rmse,
        "pred_rrmse": pred_rrmse,
        "pred_cov90": pred_cov90,
        "pred_cov90_meanonly": pred_cov90_meanonly,
        "cov90_param": cov90_param,
        "marginal_pair": f"{mp[0]}{mp[1]}",
        "margin_marginal": float(abs(l_hat[mp[0]] - l_hat[mp[1]])),
        "loss_hat_0": float(l_hat[0]),
        "loss_hat_1": float(l_hat[1]),
        "loss_hat_2": float(l_hat[2]),
    }
    for i, nm in enumerate(PARAM_NAMES):
        rec[f"theta_hat_{nm}"] = float(theta_hat[i])
        rec[f"theta_sd_{nm}"] = float(TH_s.std(axis=0)[i])
        rec[f"u_hat_{nm}"] = float(u_hat[i])
        rec[f"u_sd_{nm}"] = float(u_sd[i])
        rec[f"relsd_{nm}"] = float(relsd_theta[i])
        rec[f"z_{nm}"] = float(z_u[i]) if np.isfinite(z_u[i]) else np.nan
    rec.update(drs_per_pair)
    rec.update(drs_c_per_pair)
    for gg in GAMMAS:
        tag = _gtag(gg)
        rec[f"drs_{tag}"] = drs_by_gamma[gg]
        rec[f"drs_control_{tag}"] = drs_c_by_gamma[gg]
        rec[f"n_sloppy_{tag}"] = nsl_by_gamma[gg]
    rec["eigvals"] = list(map(float, w))
    return rec


# ============================================================================
# smoke test
# ============================================================================
if __name__ == "__main__":
    import argparse
    from npe import load_npe, simulate_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n_samples", type=int, default=2000)
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", f"npe_W{args.W}.pt")
    model = load_npe(ckpt, double=True)

    U, X = simulate_dataset(20, args.W, seed=123, phi=PHI_DISP)
    i = 3
    rec = posterior_indices(model, X[i], np.exp(U[i]), phi=PHI_DISP,
                            n_samples=args.n_samples, seed=0,
                            compute_grad_check=True)
    for k in ("W", "theta_hat_rho", "theta_hat_i0", "best_true", "best_plug",
              "ppf", "er_rel", "drs", "drs_avg_pairs", "drs_control",
              "drs_control_avg_pairs", "n_sloppy", "relsd_max",
              "pred_cov90", "grad_check_relerr"):
        print(f"{k:26s} {rec[k]}")
