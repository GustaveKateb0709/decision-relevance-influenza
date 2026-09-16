"""Step-1 diagnostic: train W6 with the bounded-support (sigmoid) bijector.

Same data / epochs / architecture as the current official W6 checkpoint, so the
only change is the support constraint.  Reports SBC z_std against the 70-epoch
unbounded model.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

ROOT = "<PROJECT_ROOT>"
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import (N_POP, PARAM_NAMES, PHI_DISP, PRIOR_HIGH, PRIOR_LOW,
                   prior_sample, sample_reports)          # noqa: E402
from npe import load_npe, sample_posterior, train_npe     # noqa: E402

OUT = "/tmp/npe_W6_bounded.pt"
torch.set_num_threads(int(os.environ.get("NPE_THREADS", "4")))


def build_data(n_sim, W, seed):
    rng = np.random.default_rng(seed)
    U = prior_sample(n_sim, rng)
    TH = np.exp(U)
    rngW = np.random.default_rng(seed + 100 + W)
    X = np.empty((n_sim, W))
    for i in range(n_sim):
        X[i] = sample_reports(TH[i], n_weeks=W, rng=rngW, phi=PHI_DISP)
    return U, X


def sbc(model, W, n_seasons=200, n_samples=1000, seed=2024):
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n_seasons):
        u = prior_sample(1, rng)[0]
        u[4] = np.log(float(np.exp(rng.uniform(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4])))))
        theta = np.exp(u)
        x = sample_reports(theta, n_weeks=W, rng=rng, phi=PHI_DISP)
        Us = sample_posterior(model, x, n=n_samples, seed=int(rng.integers(1e9)))
        um, us = Us.mean(0), Us.std(0)
        zs.append((u - um) / np.where(us > 1e-12, us, np.nan))
    Z = np.array(zs)
    return {
        "z_mean": {nm: float(Z[:, j].mean()) for j, nm in enumerate(PARAM_NAMES)},
        "z_std": {nm: float(Z[:, j].std()) for j, nm in enumerate(PARAM_NAMES)},
        "cov90_per_dim": {nm: float(np.mean(np.abs(Z[:, j]) < 1.6448536269514722))
                          for j, nm in enumerate(PARAM_NAMES)},
        "mean_cov90": float(np.mean(np.abs(Z) < 1.6448536269514722)),
        "frac_in_box": float(np.mean(np.all(
            (Z * 0 + 1) > 0, axis=1))),   # reserved-slot, replaced below
    }


if __name__ == "__main__":
    n_sim, W, epochs = 16000, 6, 70
    t0 = time.time()
    U, X = build_data(n_sim, W, seed=0)
    print(f"[bounded] data ready ({time.time()-t0:.1f}s)", flush=True)
    res = train_npe(W, n_sim=n_sim, epochs=epochs, batch=128, lr=1e-3, seed=0,
                    out_path=OUT, data=(U, X), phi=PHI_DISP, verbose=True,
                    embed=48, maf_hidden=96, bounded=True)
    print(f"[bounded] train_seconds={res['meta']['train_seconds']:.1f} "
          f"final_loss={res['meta']['final_loss']:.4f}", flush=True)

    m = load_npe(OUT, double=True)
    out = sbc(m, W)
    # fraction of posterior draws inside the prior box (support check)
    rng = np.random.default_rng(7)
    fracs = []
    for _ in range(30):
        u = prior_sample(1, rng)[0]
        u[4] = np.log(float(np.exp(rng.uniform(np.log(PRIOR_LOW[4]), np.log(PRIOR_HIGH[4])))))
        x = sample_reports(np.exp(u), n_weeks=W, rng=rng, phi=PHI_DISP)
        Us = sample_posterior(m, x, n=1000, seed=int(rng.integers(1e9)))
        fracs.append(float(np.mean(np.all((Us >= np.log(PRIOR_LOW)) & (Us <= np.log(PRIOR_HIGH)), axis=1))))
    out["frac_post_in_box_mean"] = float(np.mean(fracs))
    out["train_seconds"] = res["meta"]["train_seconds"]
    out["final_loss"] = res["meta"]["final_loss"]
    out["epochs"] = epochs
    out["n_sim"] = n_sim
    out["bounded"] = True
    print("[bounded] z_std =", {k: round(v, 3) for k, v in out["z_std"].items()}, flush=True)
    print("[bounded] z_mean =", {k: round(v, 3) for k, v in out["z_mean"].items()}, flush=True)
    print("[bounded] cov90 =", {k: round(v, 3) for k, v in out["cov90_per_dim"].items()}, flush=True)
    print("[bounded] mean_cov90 =", round(out["mean_cov90"], 3),
          " frac_post_in_box =", round(out["frac_post_in_box_mean"], 4), flush=True)
    json.dump(out, open("/tmp/bounded_sbc.json", "w"), indent=2)
    print("[bounded] done in %.1fs" % (time.time() - t0), flush=True)
