import argparse
import csv
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError as e:
    raise SystemExit(
        "This experiment requires PyTorch. Install with: pip install torch"
    ) from e


@dataclass
class Config:
    seed: int = 0
    n_examples: int = 256
    n_tokens: int = 64
    d_model: int = 96
    d_k: int = 32
    d_v: int = 48
    kernel: str = "softmax"
    sigma: float = 0.5
    gamma: float = 0.5
    temperature_sweep_points: int = 21
    temperature_sweep_log10_min: float = -1.0
    temperature_sweep_log10_max: float = 1.0
    null_tol: float = 1e-10
    rank_eff_eps: float = 1e-6
    out_dir: str = "runs"


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = logits.max(dim=dim, keepdim=True).values
    ex = torch.exp(logits - m)
    return ex / ex.sum(dim=dim, keepdim=True)


def attention_from_qkv(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_k = Q.shape[-1]
    logits = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    P = stable_softmax(logits, dim=-1)
    Y = P @ V
    return logits, P, Y


def attention_from_u_tau_kv_softmax(u: torch.Tensor, tau: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.einsum("bqd,bkd->bqk", u, K)
    w = torch.exp(tau.unsqueeze(-1) * s)
    P = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-12)
    Y = torch.einsum("bqk,bkd->bqd", P, V)
    return P, Y


def attention_from_u_tau_kv_gauss_proj(
    u: torch.Tensor, tau: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sigma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.einsum("bqd,bkd->bqk", u, K)
    sig = float(sigma)
    w = torch.exp(-((tau.unsqueeze(-1) - s) ** 2) / (2.0 * (sig * sig)))
    P = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-12)
    Y = torch.einsum("bqk,bkd->bqd", P, V)
    return P, Y


def attention_from_u_tau_kv_cauchy_proj(
    u: torch.Tensor, tau: torch.Tensor, K: torch.Tensor, V: torch.Tensor, gamma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.einsum("bqd,bkd->bqk", u, K)
    g = float(gamma)
    w = 1.0 / (1.0 + ((tau.unsqueeze(-1) - s) / g) ** 2)
    P = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-12)
    Y = torch.einsum("bqk,bkd->bqd", P, V)
    return P, Y


def row_entropy(P: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P_clamped = torch.clamp(P, eps, 1.0)
    return -(P_clamped * torch.log(P_clamped)).sum(dim=-1)


def effective_rank_from_singular_values(s: np.ndarray, eps: float = 1e-12) -> float:
    # Effective rank via entropy of normalized singular values.
    s = np.maximum(s, 0.0)
    if s.sum() <= 0:
        return 0.0
    p = s / (s.sum() + eps)
    H = -(p * np.log(p + eps)).sum()
    return float(np.exp(H))


def participation_ratio(eigs: np.ndarray, eps: float = 1e-12) -> float:
    eigs = np.maximum(eigs, 0.0)
    num = (eigs.sum() ** 2)
    den = (np.square(eigs).sum() + eps)
    return float(num / den)


def svd_nullspace(A: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (V_null, s) where columns of V_null span right-nullspace.
    # A shape: (m,n), nullspace in R^n.
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return Vt.T, s
    mask = s <= tol
    if not mask.any():
        return np.zeros((A.shape[1], 0), dtype=A.dtype), s
    # Vt has shape (n,n) when full_matrices=True and m<=n; otherwise (m,m). We force full_matrices.
    V = Vt.T
    # Identify indices of singular values corresponding to nullspace.
    # For full SVD, s has length min(m,n); nullspace basis are last n-min(m,n) cols plus any with s<=tol.
    m, n = A.shape
    k = min(m, n)
    null_indices = []
    for i in range(k):
        if s[i] <= tol:
            null_indices.append(i)
    for i in range(k, n):
        null_indices.append(i)
    V_null = V[:, null_indices] if null_indices else np.zeros((n, 0), dtype=A.dtype)
    return V_null, s


def mkdir_run_dir(base: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_and_save(path: Path, title: str) -> None:
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def compute_u_tau(Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Q: (B, n_tokens, d_k)
    d_k = Q.shape[-1]
    qnorm = torch.norm(Q, dim=-1)  # (B,n)
    tau = qnorm / math.sqrt(d_k)
    u = Q / torch.clamp(qnorm.unsqueeze(-1), min=1e-12)
    return u, tau


def laplace_radon_reconstruct_y(
    K: torch.Tensor,
    V: torch.Tensor,
    u: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    # Implements y_j = (sum_i e^{tau*s_i} v_i) / (sum_i e^{tau*s_i})
    # Shapes:
    # K: (B, n_k, d_k), V: (B, n_k, d_v), u: (B, n_q, d_k), tau: (B, n_q)
    # returns: (B, n_q, d_v)
    s = torch.einsum("bqd,bkd->bqk", u, K)  # (B,nq,nk)
    w = torch.exp(tau.unsqueeze(-1) * s)  # (B,nq,nk)
    Z = w.sum(dim=-1, keepdim=True)  # (B,nq,1)
    N = torch.einsum("bqk,bkd->bqd", w, V)  # (B,nq,dv)
    return N / torch.clamp(Z, min=1e-12)


def gauge_transform(WQ: torch.Tensor, WK: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # WQ: (d_model,d_k), WK: (d_model,d_k), A: (d_k,d_k)
    # Implements WQ -> WQ A, WK -> WK A^{-T}
    AinvT = torch.inverse(A).transpose(0, 1)
    return WQ @ A, WK @ AinvT


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention tomography experiment: tests, logging, plotting")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--n_examples", type=int, default=Config.n_examples)
    parser.add_argument("--n_tokens", type=int, default=Config.n_tokens)
    parser.add_argument("--d_model", type=int, default=Config.d_model)
    parser.add_argument("--d_k", type=int, default=Config.d_k)
    parser.add_argument("--d_v", type=int, default=Config.d_v)
    parser.add_argument("--out_dir", type=str, default=Config.out_dir)
    parser.add_argument("--null_tol", type=float, default=Config.null_tol)
    parser.add_argument("--rank_eff_eps", type=float, default=Config.rank_eff_eps)
    parser.add_argument("--temperature_sweep_points", type=int, default=Config.temperature_sweep_points)
    parser.add_argument("--temperature_sweep_log10_min", type=float, default=Config.temperature_sweep_log10_min)
    parser.add_argument("--temperature_sweep_log10_max", type=float, default=Config.temperature_sweep_log10_max)
    parser.add_argument("--kernel", type=str, default="softmax", choices=["softmax", "gauss_proj", "cauchy_proj"])
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    # Use parse_known_args so running under notebooks doesn't error.
    args, _unknown = parser.parse_known_args()

    cfg = Config(**vars(args))
    set_seeds(cfg.seed)

    out = mkdir_run_dir(cfg.out_dir)

    device = torch.device("cpu")
    dtype = torch.float64

    # Synthetic token batch: X ~ N(0,1)
    X = torch.randn(cfg.n_examples, cfg.n_tokens, cfg.d_model, device=device, dtype=dtype)

    # Random linear maps for a single head
    WQ = torch.randn(cfg.d_model, cfg.d_k, device=device, dtype=dtype) / math.sqrt(cfg.d_model)
    WK = torch.randn(cfg.d_model, cfg.d_k, device=device, dtype=dtype) / math.sqrt(cfg.d_model)
    WV = torch.randn(cfg.d_model, cfg.d_v, device=device, dtype=dtype) / math.sqrt(cfg.d_model)

    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    u, tau = compute_u_tau(Q)

    logits = None
    Y_hat = None
    if cfg.kernel == "softmax":
        logits, P, Y = attention_from_qkv(Q, K, V)
        Y_hat = laplace_radon_reconstruct_y(K, V, u, tau)
    elif cfg.kernel == "gauss_proj":
        P, Y = attention_from_u_tau_kv_gauss_proj(u, tau, K, V, sigma=cfg.sigma)
    else:
        P, Y = attention_from_u_tau_kv_cauchy_proj(u, tau, K, V, gamma=cfg.gamma)

    # --- Test group A: Laplace–Radon identity ---
    identity_err = None
    if cfg.kernel == "softmax":
        identity_err = torch.norm(Y - Y_hat) / torch.clamp(torch.norm(Y), min=1e-12)

    # --- Test group B: "1D dependence" (logits vs s=u^T k) ---
    logits_err = None
    if cfg.kernel == "softmax":
        s = torch.einsum("bqd,bkd->bqk", u, K)
        logits_from_s = tau.unsqueeze(-1) * s
        logits_err = torch.norm(logits - logits_from_s) / torch.clamp(torch.norm(logits), min=1e-12)

    # --- Test group C: Gauge equivalence ---
    A = torch.randn(cfg.d_k, cfg.d_k, device=device, dtype=dtype)
    A = A + 0.1 * torch.eye(cfg.d_k, device=device, dtype=dtype)  # improve invertibility
    WQ2, WK2 = gauge_transform(WQ, WK, A)
    Q2 = X @ WQ2
    K2 = X @ WK2
    logits2 = None
    if cfg.kernel == "softmax":
        logits2, P2, Y2 = attention_from_qkv(Q2, K2, V)
    elif cfg.kernel == "gauss_proj":
        u2, tau2 = compute_u_tau(Q2)
        P2, Y2 = attention_from_u_tau_kv_gauss_proj(u2, tau2, K2, V, sigma=cfg.sigma)
    else:
        u2, tau2 = compute_u_tau(Q2)
        P2, Y2 = attention_from_u_tau_kv_cauchy_proj(u2, tau2, K2, V, gamma=cfg.gamma)

    B1 = WQ @ WK.transpose(0, 1)
    B2 = WQ2 @ WK2.transpose(0, 1)
    B_err = torch.norm(B1 - B2) / torch.clamp(torch.norm(B1), min=1e-12)
    logits_gauge_err = None
    if cfg.kernel == "softmax":
        logits_gauge_err = torch.norm(logits - logits2) / torch.clamp(torch.norm(logits), min=1e-12)
    Y_gauge_err = torch.norm(Y - Y2) / torch.clamp(torch.norm(Y), min=1e-12)

    B_s = np.linalg.svd(B1.detach().cpu().numpy(), compute_uv=False)

    # --- Null space diagnostics ---
    # Parameter-level nulls: ker(WK^T), ker(WQ^T)
    WKt = WK.transpose(0, 1).cpu().numpy()  # (d_k,d_model)
    WQt = WQ.transpose(0, 1).cpu().numpy()

    # Right-nullspace of WK^T lives in R^{d_model}
    WKt_null, WKt_s = svd_nullspace(WKt, tol=cfg.null_tol)
    WQt_null, WQt_s = svd_nullspace(WQt, tol=cfg.null_tol)

    # Verify invariance to perturbations in ker(WK^T): x -> x + alpha z
    # test a single random null vector if available.
    ker_tests = []
    if WKt_null.shape[1] > 0:
        z = torch.from_numpy(WKt_null[:, 0]).to(device=device, dtype=dtype)  # (d_model,)
        alpha = 0.5
        Xp = X + alpha * z
        Qp = Xp @ WQ
        Kp = Xp @ WK
        Vp = Xp @ WV
        logits_p, P_p, Y_p = attention_from_qkv(Qp, Kp, Vp)
        # If z in ker(WK^T), Kp should equal K; logits should match (since logits depend on K).
        K_pert_err = torch.norm(K - Kp) / torch.clamp(torch.norm(K), min=1e-12)
        logits_pert_err = torch.norm(logits - logits_p) / torch.clamp(torch.norm(logits), min=1e-12)
        ker_tests.append({
            "ker": "ker(WK^T)",
            "basis_dim": int(WKt_null.shape[1]),
            "K_rel_err": float(K_pert_err.item()),
            "logits_rel_err": float(logits_pert_err.item()),
        })

    if WQt_null.shape[1] > 0:
        z = torch.from_numpy(WQt_null[:, 0]).to(device=device, dtype=dtype)  # (d_model,)
        alpha = 0.5
        Xp = X + alpha * z
        Qp = Xp @ WQ
        # z in ker(WQ^T) => Q unchanged
        Q_pert_err = torch.norm(Q - Qp) / torch.clamp(torch.norm(Q), min=1e-12)
        ker_tests.append({
            "ker": "ker(WQ^T)",
            "basis_dim": int(WQt_null.shape[1]),
            "Q_rel_err": float(Q_pert_err.item()),
        })

    # Dataset limited-angle null via span(U): compute eig spectrum of Cov(u)
    u_flat = u.reshape(-1, cfg.d_k).cpu().numpy()
    u_mean = u_flat.mean(axis=0, keepdims=True)
    u_center = u_flat - u_mean
    Cu = (u_center.T @ u_center) / max(1, u_center.shape[0] - 1)
    eig_u = np.linalg.eigvalsh(Cu)
    eig_u = np.sort(eig_u)[::-1]
    dim_eff_u = participation_ratio(eig_u)

    # Token-space normals N = WK u
    # Using definition: n_j = WK u_j (WK is d_model x d_k)
    n = torch.einsum("md,bqd->bqm", WK, u)  # (B,n,d_model)
    n_flat = n.reshape(-1, cfg.d_model).cpu().numpy()
    n_mean = n_flat.mean(axis=0, keepdims=True)
    n_center = n_flat - n_mean
    Cn = (n_center.T @ n_center) / max(1, n_center.shape[0] - 1)
    eig_n = np.linalg.eigvalsh(Cn)
    eig_n = np.sort(eig_n)[::-1]
    dim_eff_n = participation_ratio(eig_n)

    # Forward-pass value null ker(P): pick a single example for SVD
    P0 = P[0].cpu().numpy()  # (n_tokens,n_tokens)
    U_p, s_p, Vt_p = np.linalg.svd(P0, full_matrices=True)
    rank_eff_p = effective_rank_from_singular_values(s_p, eps=cfg.rank_eff_eps)

    P_spectra = []
    for ex in range(min(8, cfg.n_examples)):
        Px = P[ex].detach().cpu().numpy()
        sx = np.linalg.svd(Px, compute_uv=False)
        P_spectra.append(sx.astype(np.float64))
    P_spectra = np.stack(P_spectra, axis=0) if P_spectra else np.zeros((0, cfg.n_tokens), dtype=np.float64)
    # approximate null basis from singular values below tol
    null_mask = s_p <= cfg.null_tol
    # plus any full-matrices extra columns beyond min(m,n) already in Vt when full_matrices=True
    k = min(P0.shape)
    null_indices = [i for i in range(k) if s_p[i] <= cfg.null_tol] + list(range(k, P0.shape[1]))
    V_null_p = Vt_p.T[:, null_indices] if null_indices else np.zeros((P0.shape[1], 0))

    value_null_test = None
    if V_null_p.shape[1] > 0:
        dv = cfg.d_v
        # Build a DeltaV in nullspace: DeltaV = r * a^T where r in ker(P) (n_tokens,) and a in R^{d_v}
        r = V_null_p[:, 0]
        a = np.random.randn(dv)
        DeltaV = np.outer(r, a)
        PV = P0 @ DeltaV
        value_null_test = {
            "kerP_dim": int(V_null_p.shape[1]),
            "PV_fro": float(np.linalg.norm(PV)),
            "DeltaV_fro": float(np.linalg.norm(DeltaV)),
            "PV_rel": float(np.linalg.norm(PV) / (np.linalg.norm(DeltaV) + 1e-12)),
        }

    # --- Radial/temperature sweep ---
    alphas = np.logspace(cfg.temperature_sweep_log10_min, cfg.temperature_sweep_log10_max, cfg.temperature_sweep_points)
    sweep_rows: List[Dict] = []

    logits0 = logits[0] if cfg.kernel == "softmax" else None

    for alpha in alphas:
        if cfg.kernel == "softmax":
            logits_a = alpha * logits0
            P_a = stable_softmax(logits_a, dim=-1)
        elif cfg.kernel == "gauss_proj":
            P_a, _Ya = attention_from_u_tau_kv_gauss_proj(u[0:1], (alpha * tau[0:1]), K[0:1], V[0:1], sigma=cfg.sigma)
            P_a = P_a[0]
        else:
            P_a, _Ya = attention_from_u_tau_kv_cauchy_proj(u[0:1], (alpha * tau[0:1]), K[0:1], V[0:1], gamma=cfg.gamma)
            P_a = P_a[0]
        ent = row_entropy(P_a).mean().item()

        # SVD spectrum for effective rank
        P_a_np = P_a.detach().cpu().numpy()
        s_a = np.linalg.svd(P_a_np, compute_uv=False)
        rank_eff_a = effective_rank_from_singular_values(s_a, eps=cfg.rank_eff_eps)

        sweep_rows.append({
            "alpha": float(alpha),
            "mean_row_entropy": float(ent),
            "rank_eff_P": float(rank_eff_a),
            "max_p_mean": float(P_a.max(dim=-1).values.mean().item()),
        })

    # --- Logging ---
    summary = {
        "config": dataclasses.asdict(cfg),
        "metrics": {
            "identity_rel_err_Y": None if identity_err is None else float(identity_err.item()),
            "logits_from_projection_rel_err": None if logits_err is None else float(logits_err.item()),
            "gauge_rel_err_B": float(B_err.item()),
            "gauge_rel_err_logits": float(logits_gauge_err.item()),
            "gauge_rel_err_Y": float(Y_gauge_err.item()),
            "dim_eff_span_U": float(dim_eff_u),
            "dim_eff_span_N": float(dim_eff_n),
            "rank_eff_P_example0": float(rank_eff_p),
            "ker_WKt_dim": int(WKt_null.shape[1]),
            "ker_WQt_dim": int(WQt_null.shape[1]),
            "B_singular_values": B_s[: min(32, B_s.size)].tolist(),
        },
        "ker_tests": ker_tests,
        "value_null_test": value_null_test,
    }

    save_json(out / "logs" / "summary.json", summary)
    save_csv(out / "logs" / "temperature_sweep.csv", sweep_rows)

    # Save minimal artifacts for a separate theory-vs-empirical comparison script.
    tau_np = tau.detach().cpu().numpy().ravel().astype(np.float64)
    logits_sample = np.zeros((0,), dtype=np.float64)
    if logits is not None:
        logits_np = logits.detach().cpu().numpy().ravel().astype(np.float64)
        sample_n = min(200_000, logits_np.size)
        sample_idx = np.random.choice(logits_np.size, size=sample_n, replace=False)
        logits_sample = logits_np[sample_idx]
    q_var_per_dim = float(Q.detach().cpu().numpy().var())
    k_var_per_dim = float(K.detach().cpu().numpy().var())
    np.savez_compressed(
        out / "logs" / "artifacts.npz",
        tau=tau_np,
        logits_sample=logits_sample,
        q_var_per_dim=q_var_per_dim,
        k_var_per_dim=k_var_per_dim,
        P_spectra=P_spectra,
        B_singular_values=np.array(summary["metrics"]["B_singular_values"], dtype=np.float64),
        kernel=np.array([cfg.kernel]),
        sigma=np.array([getattr(cfg, "sigma", 0.0)], dtype=np.float64),
        gamma=np.array([getattr(cfg, "gamma", 0.0)], dtype=np.float64),
        d_k=np.array([cfg.d_k], dtype=np.int64),
        n_tokens=np.array([cfg.n_tokens], dtype=np.int64),
        seed=np.array([cfg.seed], dtype=np.int64),
    )

    # --- Plotting ---
    # 1) Tau distribution
    plt.figure(figsize=(7, 4))
    tau_np = tau.detach().cpu().numpy().ravel()
    plt.hist(tau_np, bins=50)
    plt.xlabel("tau = ||q||/sqrt(d_k)")
    plt.ylabel("count")
    plot_and_save(out / "plots" / "tau_hist.png", "Radial parameter distribution")

    # 2) Attention entropy distribution
    plt.figure(figsize=(7, 4))
    ent_np = row_entropy(P).detach().cpu().numpy().ravel()
    plt.hist(ent_np, bins=50)
    plt.xlabel("row entropy")
    plt.ylabel("count")
    plot_and_save(out / "plots" / "attention_entropy_hist.png", "Attention row entropy")

    # 3) Tilt covariance spectrum
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.maximum(eig_u, 1e-18))
    plt.xlabel("eigen index")
    plt.ylabel("eigenvalue (log)")
    plot_and_save(out / "plots" / "tilt_cov_spectrum.png", f"Cov(u) spectrum, dim_eff={dim_eff_u:.2f}")

    # 4) Token-normal covariance spectrum
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.maximum(eig_n, 1e-18))
    plt.xlabel("eigen index")
    plt.ylabel("eigenvalue (log)")
    plot_and_save(out / "plots" / "token_normal_cov_spectrum.png", f"Cov(n=WK u) spectrum, dim_eff={dim_eff_n:.2f}")

    # 5) Singular values of P example
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.maximum(s_p, 1e-18), marker="o", linestyle="-")
    plt.xlabel("singular index")
    plt.ylabel("singular value (log)")
    plot_and_save(out / "plots" / "P_singular_values.png", f"Singular values of P[0], rank_eff={rank_eff_p:.2f}")

    # 6) Temperature sweep plots
    al = np.array([r["alpha"] for r in sweep_rows])
    ent = np.array([r["mean_row_entropy"] for r in sweep_rows])
    rk = np.array([r["rank_eff_P"] for r in sweep_rows])
    mx = np.array([r["max_p_mean"] for r in sweep_rows])

    plt.figure(figsize=(7, 4))
    plt.semilogx(al, ent, marker="o")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean row entropy")
    plot_and_save(out / "plots" / "sweep_entropy_vs_alpha.png", "Entropy vs logit scaling")

    plt.figure(figsize=(7, 4))
    plt.semilogx(al, rk, marker="o")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("effective rank(P)")
    plot_and_save(out / "plots" / "sweep_rankeff_vs_alpha.png", "Effective rank(P) vs logit scaling")

    plt.figure(figsize=(7, 4))
    plt.semilogx(al, mx, marker="o")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean max row probability")
    plot_and_save(out / "plots" / "sweep_maxp_vs_alpha.png", "Peakedness vs logit scaling")

    # 7) Sanity scatter: Y vs Y_hat (sample)
    plt.figure(figsize=(6, 6))
    Y_s = Y.detach().cpu().numpy().ravel()
    Yh_s = Y_hat.detach().cpu().numpy().ravel()
    idx = np.random.choice(Y_s.size, size=min(5000, Y_s.size), replace=False)
    plt.scatter(Y_s[idx], Yh_s[idx], s=3, alpha=0.4)
    lo = float(min(Y_s[idx].min(), Yh_s[idx].min()))
    hi = float(max(Y_s[idx].max(), Yh_s[idx].max()))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("Y")
    plt.ylabel("Y_hat")
    plot_and_save(out / "plots" / "Y_vs_Yhat_scatter.png", "Laplace–Radon reconstruction check")

    print(f"Wrote results to: {out}")
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
