r"""
Neural Posterior Estimation (NPE) for Paper 16 -- hand-written, no ``sbi``.

Pipeline
--------
    x (weekly reported counts, length W)  --GRU-->  c  (summary embedding)
    u ~ q(u | x)  modelled by a conditional Masked Autoregressive Flow (MAF)
        * MADE (masked autoencoder) provides the autoregressive conditioners
        * K affine autoregressive layers with fixed random permutations
    training loss  =  APT / SNPE-C  (Greenberg et al. 2019):
        for a batch of N pairs (u_j, x_j) treat  log q(u_j | x_i)  as
        unnormalised logits and minimise the cross-entropy with target j = i.

Key requirement for the downstream DRS index: ``log_prob(u, x)`` must be
twice differentiable w.r.t. ``u`` so that ``torch.autograd.functional.hessian``
returns an exact 6x6 Hessian.  Everything here is plain ``torch`` so that
holds by construction.

All computations are CPU-only (``torch.set_num_threads(1)``).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    DIM,
    PARAM_NAMES,
    PHI_DISP,
    PRIOR_HIGH,
    PRIOR_LOW,
    T_OBS,
    prior_sample,
    sample_reports,
)

# CPU-only by design.  The study brief suggested 1 thread; we expose an env
# override (NPE_THREADS) so the wall-clock can be tuned on the 8-core box
# without any change to numerical results.
torch.set_num_threads(int(os.environ.get("NPE_THREADS", "1")))

_LOG2PI = math.log(2.0 * math.pi)
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_FILE_DIR)
CKPT_DIR = os.path.join(_ROOT, "checkpoints")


# ============================================================================
# Summary network (GRU over the observation window)
# ============================================================================
class SummaryNet(nn.Module):
    """GRU encoder mapping a length-W count series to a summary embedding."""

    def __init__(self, seq_len: int, hidden: int = 64, embed: int = 48):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.embed = embed
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W) raw non-negative counts
        z = torch.log1p(x).unsqueeze(-1)          # (B, W, 1)
        out, h = self.gru(z)
        h = h[-1]                                  # (B, hidden)
        return self.head(h)                        # (B, embed)


# ============================================================================
# MADE -- masked autoencoder producing (mu, log_alpha) for each u-dimension
# ============================================================================
class MADE(nn.Module):
    """Masked autoencoder enforcing the autoregressive property.

    Output row ``d`` (both ``mu_d`` and ``log_alpha_d``) depends only on
    ``u_{<d}`` and on the whole context ``c``.
    """

    def __init__(self, dim: int, ctx_dim: int, hidden: int = 96,
                 n_layers: int = 4, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.ctx_dim = ctx_dim
        self.hidden = hidden
        self.n_layers = n_layers

        g = torch.Generator().manual_seed(seed)
        # degree of every hidden unit in {0, ..., dim-1}; a unit of degree m
        # "sees" the first m coordinates of u  (u_0 .. u_{m-1})  plus all context.
        degrees = torch.randint(0, dim, (hidden,), generator=g)

        in_dim = dim + ctx_dim
        # ---- mask: input -> first hidden -----------------------------------
        m1 = torch.zeros(hidden, in_dim)
        for h in range(hidden):
            m = int(degrees[h])
            for j in range(dim):
                m1[h, j] = 1.0 if j < m else 0.0
            m1[h, dim:] = 1.0                       # context always connected
        self.register_buffer("mask1", m1)

        # ---- masks: hidden -> hidden ---------------------------------------
        masks = []
        for _ in range(n_layers - 2):
            mm = torch.zeros(hidden, hidden)
            for a in range(hidden):                # a = current unit
                ma = int(degrees[a])
                for b in range(hidden):            # b = previous unit
                    mm[a, b] = 1.0 if int(degrees[b]) <= ma else 0.0
            masks.append(mm)
        self._hidden_masks = masks

        # ---- mask: hidden -> output ----------------------------------------
        # rows 0..dim-1 -> mu, rows dim..2dim-1 -> log_alpha ; degree(row)=d
        mo = torch.zeros(2 * dim, hidden)
        for d in range(dim):
            for b in range(hidden):
                mo[d, b] = 1.0 if int(degrees[b]) <= d else 0.0
                mo[dim + d, b] = 1.0 if int(degrees[b]) <= d else 0.0
        self.register_buffer("mask_out", mo)

        # ---- parameters ----------------------------------------------------
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lins = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers - 2)])
        self.lin_out = nn.Linear(hidden, 2 * dim)

    def forward(self, u: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([u, c], dim=-1)
        h = F.linear(inp, self.lin1.weight * self.mask1, self.lin1.bias)
        h = torch.tanh(h)
        for i, lin in enumerate(self.lins):
            h = F.linear(h, lin.weight * self._hidden_masks[i], lin.bias)
            h = torch.tanh(h)
        out = F.linear(h, self.lin_out.weight * self.mask_out, self.lin_out.bias)
        mu = out[:, : self.dim]
        log_alpha = out[:, self.dim:]
        return mu, log_alpha


# ============================================================================
# Autoregressive affine layer  +  stacked conditional MAF
# ============================================================================
class MAFLayer(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, hidden: int, n_layers: int,
                 seed: int):
        super().__init__()
        self.dim = dim
        self.made = MADE(dim, ctx_dim, hidden, n_layers, seed=seed)
        g = torch.Generator().manual_seed(seed + 10_000)
        perm = torch.randperm(dim, generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv)

    # forward: u -> z,  z_d = (u_d - mu_d(u_<d)) * exp(-alpha_d(u_<d))
    def forward(self, u: torch.Tensor, c: torch.Tensor):
        mu, log_alpha = self.made(u, c)
        z = (u - mu) * torch.exp(-log_alpha)
        logdet = -log_alpha.sum(dim=-1)
        return z, logdet

    # inverse: z -> u  (sequential over dimension due to autoregressive structure)
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        u = torch.zeros_like(z)
        for d in range(D):
            mu, log_alpha = self.made(u, c)         # dim d only depends on u_<d
            u = u.clone()
            u[:, d] = mu[:, d] + torch.exp(log_alpha[:, d]) * z[:, d]
        return u


class ConditionalMAF(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, hidden: int = 96,
                 n_made_layers: int = 4, n_flow: int = 5, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            MAFLayer(dim, ctx_dim, hidden, n_made_layers, seed=seed + 137 * k)
            for k in range(n_flow)
        ])

    def log_prob(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # u: (B, D) or (D,), c: (B, ctx) or (ctx,)
        squeeze = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        if c.dim() == 1:
            c = c.unsqueeze(0)
        x = u
        logdet = torch.zeros(u.shape[0], dtype=u.dtype, device=u.device)
        for layer in self.layers:
            z, ld = layer.forward(x, c)
            logdet = logdet + ld
            x = z[:, layer.perm]                     # permutation (|det| = 1)
        logpz = -0.5 * (x ** 2).sum(dim=-1) - 0.5 * self.dim * _LOG2PI
        out = logpz + logdet
        return out.squeeze(0) if squeeze else out

    def sample(self, c: torch.Tensor, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if c.dim() == 1:
            c = c.unsqueeze(0)
        B = c.shape[0]
        if B == 1 and n > 1:
            c = c.expand(n, -1).contiguous()
        z = torch.randn(c.shape[0], self.dim, dtype=c.dtype, device=c.device,
                        generator=generator)
        x = z
        for layer in reversed(self.layers):
            x = x[:, layer.inv_perm]                 # undo permutation
            x = layer.inverse(x, c)
        return x


# ============================================================================
# Full NPE module
# ============================================================================
class NPE(nn.Module):
    """Conditional density estimator q(u | x).

    Two parameterisations of the u-space support:

    * ``bounded=False`` (original): the MAF models the affinely standardised
      ``u_std = (u - u_mu0) / u_s0`` with an unbounded Gaussian base.  Nothing
      stops mass leaking outside the prior box (~99.95% of draws do).
    * ``bounded=True`` (support-constrained): the MAF models an *unconstrained*
      ``w`` and ``u`` is obtained by a per-dimension sigmoid (logit) bijector
      ``u_k = lo_k + (hi_k - lo_k) * sigmoid(w_k)``.  The density is confined to
      the prior box ``[lo, hi]`` by construction and the log-Jacobian
      ``-sum_k log((hi_k-lo_k) * s_k * (1-s_k))``, ``s = sigmoid(w)``, is added
      to ``log p_flow(w)``.  The map is smooth on the interior, so
      ``log_prob(u, x)`` keeps its exact second derivative w.r.t. ``u``.
    """

    def __init__(self, seq_len: int, sum_hidden: int = 64, embed: int = 48,
                 maf_hidden: int = 96, n_made_layers: int = 4, n_flow: int = 5,
                 seed: int = 0, u_mu0: np.ndarray | None = None,
                 u_s0: np.ndarray | None = None, bounded: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.embed = embed
        self.bounded = bool(bounded)
        self.summary = SummaryNet(seq_len, sum_hidden, embed)
        self.flow = ConditionalMAF(DIM, embed, maf_hidden, n_made_layers, n_flow,
                                   seed=seed)
        # optional affine normalisation of u-space (helps when the prior box is
        # far from the origin, e.g. log i0 ~ -12).  Default = identity.
        mu0 = np.zeros(DIM) if u_mu0 is None else np.asarray(u_mu0, float)
        s0 = np.ones(DIM) if u_s0 is None else np.asarray(u_s0, float)
        self.register_buffer("u_mu0", torch.tensor(mu0, dtype=torch.float32))
        self.register_buffer("u_s0", torch.tensor(s0, dtype=torch.float32))
        # support box of the bounded bijector (u-space = log of the prior box)
        self.register_buffer("box_lo", torch.tensor(np.log(PRIOR_LOW), dtype=torch.float32))
        self.register_buffer("box_hi", torch.tensor(np.log(PRIOR_HIGH), dtype=torch.float32))
        # clamping margin for the inverse (logit) map
        self._eps = 1e-6

    # u -> normalised coordinate fed to the flow
    def _to_std(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mu0) / self.u_s0

    # ---- bounded-support sigmoid bijector ---------------------------------
    def _u_to_w(self, u: torch.Tensor) -> torch.Tensor:
        """logit((u - lo)/(hi - lo)), clamped to [eps, 1-eps]."""
        span = self.box_hi - self.box_lo
        t = (u - self.box_lo) / span
        t = torch.clamp(t, self._eps, 1.0 - self._eps)
        return torch.log(t) - torch.log1p(-t)

    def _w_to_u(self, w: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(w)
        return self.box_lo + (self.box_hi - self.box_lo) * s

    def _log_density_from_c(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """log q(u | x) given the summary embedding c; u in *natural* log-space."""
        if self.bounded:
            w = self._u_to_w(u)
            lp = self.flow.log_prob(w, c)
            s = torch.sigmoid(w)
            span = self.box_hi - self.box_lo
            # |det du/dw| = prod_k span_k s_k (1-s_k)  ->  subtract its log
            return lp - torch.log(span * s * (1.0 - s)).sum(dim=-1)
        lp = self.flow.log_prob(self._to_std(u), c)
        return lp - torch.log(self.u_s0).sum()            # |det du_std/du|

    def log_prob(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """log q(u | x).  ``u``: (..., D); ``x``: (B, W) matching u's batch."""
        c = self.summary(x)
        return self._log_density_from_c(u, c)

    def sample_posterior(self, x: torch.Tensor, n: int,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw ``n`` samples of u ~ q(. | x).  ``x``: (W,) or (B, W)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        c = self.summary(x)                          # (B, embed)
        if x.shape[0] == 1:
            c = c.expand(n, -1).contiguous()
            n_out = n
        else:
            n_out = c.shape[0]
        w = self.flow.sample(c, n_out, generator)
        if self.bounded:
            return self._w_to_u(w)
        return w * self.u_s0 + self.u_mu0


# ============================================================================
# APT / SNPE-C loss
# ============================================================================
def apt_loss(model: NPE, u: torch.Tensor, x: torch.Tensor,
             chunk: int = 4096) -> torch.Tensor:
    """Cross-entropy over the batch with target j = i (SNPE-C).

    ``u`` is in the space the flow is trained on: natural log-space when the
    model is support-constrained (``bounded=True``), otherwise the affinely
    standardised space.
    """
    B = u.shape[0]
    c = model.summary(x)                             # (B, embed)

    # logits[i, j] = log q(u_j | x_i)
    c_rep = c.repeat_interleave(B, dim=0)            # (B*B, embed)
    u_rep = u.repeat(B, 1)                           # (B*B, D)
    logits = torch.empty(B * B, dtype=u.dtype, device=u.device)
    for s in range(0, B * B, chunk):
        e = min(s + chunk, B * B)
        if getattr(model, "bounded", False):
            logits[s:e] = model._log_density_from_c(u_rep[s:e], c_rep[s:e])
        else:
            logits[s:e] = model.flow.log_prob(u_rep[s:e], c_rep[s:e])
    logits = logits.view(B, B)
    target = torch.arange(B, device=u.device)
    return F.cross_entropy(logits, target)


# ============================================================================
# Data generation + training
# ============================================================================
def simulate_dataset(n_sim: int, W: int, seed: int,
                     phi: float | np.ndarray = PHI_DISP
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U, X) with U in u-space (n, D) and X counts (n, W)."""
    rng = np.random.default_rng(seed)
    U = prior_sample(n_sim, rng)
    TH = np.exp(U)
    X = np.empty((n_sim, W))
    if np.isscalar(phi):
        phis = np.full(n_sim, float(phi))
    else:
        phis = np.asarray(phi)
    for i in range(n_sim):
        X[i] = sample_reports(TH[i], n_weeks=W, rng=rng, phi=phis[i])
    return U, X


def train_npe(W: int, n_sim: int = 50_000, epochs: int = 200, batch: int = 256,
              lr: float = 1e-3, seed: int = 0, out_path: Optional[str] = None,
              sum_hidden: int = 64, embed: int = 48, maf_hidden: int = 96,
              n_made_layers: int = 4, n_flow: int = 5,
              phi: float | np.ndarray = PHI_DISP,
              data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              standardize: bool = False,
              weight_decay: float = 0.0,
              bounded: bool = False,
              verbose: bool = True) -> dict:
    """Train an NPE for observation window ``W`` and (optionally) save it."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data is None:
        U, X = simulate_dataset(n_sim, W, seed=seed, phi=phi)
    else:
        U, X = data
        n_sim = U.shape[0]
    t_sim = time.time() - t0

    # affine normalisation of u-space (box centre / uniform sd)
    lo, hi = np.log(PRIOR_LOW), np.log(PRIOR_HIGH)
    u_mu0 = (lo + hi) / 2.0
    u_s0 = (hi - lo) / (2.0 * math.sqrt(3.0))
    if not standardize:
        u_mu0 = np.zeros(DIM)
        u_s0 = np.ones(DIM)
    if bounded:
        # the sigmoid box bijector handles the scale itself; train on natural u
        U_fit = U
    elif standardize:
        U_fit = (U - u_mu0) / u_s0
    else:
        U_fit = U

    Ut = torch.tensor(U_fit, dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32)

    model = NPE(W, sum_hidden, embed, maf_hidden, n_made_layers, n_flow,
                seed=seed, u_mu0=u_mu0, u_s0=u_s0, bounded=bounded)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_batches = max(1, n_sim // batch)
    losses = []
    t_train0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n_sim)
        ep_loss = 0.0
        for it in range(n_batches):
            idx = perm[it * batch:(it + 1) * batch]
            u = Ut[idx]
            x = Xt[idx]
            loss = apt_loss(model, u, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
        sched.step()
        ep_loss /= n_batches
        losses.append(ep_loss)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            el = time.time() - t_train0
            print(f"  [W={W}] epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}  "
                  f"elapsed={el:.1f}s", flush=True)

    t_train = time.time() - t_train0
    meta = {
        "W": W,
        "n_sim": int(n_sim),
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "seed": seed,
        "sum_hidden": sum_hidden,
        "embed": embed,
        "maf_hidden": maf_hidden,
        "n_made_layers": n_made_layers,
        "n_flow": n_flow,
        "standardize": bool(standardize),
        "bounded": bool(bounded),
        "u_mu0": list(map(float, u_mu0)),
        "u_s0": list(map(float, u_s0)),
        "phi_train": (float(phi) if np.isscalar(phi) else list(map(float, np.asarray(phi)))),
        "final_loss": float(losses[-1]),
        "first_loss": float(losses[0]),
        "loss_curve_every_epoch": losses,
        "sim_seconds": t_sim,
        "train_seconds": t_train,
        "total_seconds": time.time() - t0,
    }

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out_path)
        if verbose:
            print(f"  [W={W}] saved -> {out_path}", flush=True)
    return {"model": model, "meta": meta}


def load_npe(path: str, double: bool = False) -> NPE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = NPE(meta["W"], meta.get("sum_hidden", 64), meta.get("embed", 48),
                meta.get("maf_hidden", 96), meta.get("n_made_layers", 4),
                meta.get("n_flow", 5), seed=meta.get("seed", 0),
                u_mu0=meta.get("u_mu0"), u_s0=meta.get("u_s0"),
                bounded=meta.get("bounded", False))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    if double:
        model = model.double()
    return model


def sample_posterior(model: NPE, x, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Sample the posterior in u-space.  Returns numpy (n, D) (float64)."""
    g = torch.Generator().manual_seed(seed)
    xt = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32) \
        if next(model.parameters()).dtype == torch.float32 \
        else torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
    with torch.no_grad():
        u = model.sample_posterior(xt, n, g)
    return u.detach().numpy().astype(np.float64)


# ============================================================================
# Self test / smoke run
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--n_sim", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    res = train_npe(args.W, n_sim=args.n_sim, epochs=args.epochs,
                    batch=args.batch, out_path=None)
    print(json.dumps(res["meta"], indent=2, default=str)[:1200])
