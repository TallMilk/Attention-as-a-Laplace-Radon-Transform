import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows: List[Dict[str, float]] = []
        for row in r:
            if not row:
                continue
            rows.append({k: float(v) for k, v in row.items()})
        return rows


def savefig(path: Path, title: str) -> None:
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def chi_scale_pdf(x: np.ndarray, k: int, scale: float) -> np.ndarray:
    # If Z ~ N(0, I_k), then ||Z|| has Chi distribution with k degrees of freedom.
    # For Q ~ N(0, scale^2 I_k), ||Q|| = scale * ||Z||.
    # This returns pdf of ||Q||.
    x = np.asarray(x, dtype=np.float64)
    scale = max(float(scale), 1e-12)
    y = x / scale

    # Chi(k) pdf: f(y) = 2^{1-k/2} / Gamma(k/2) * y^{k-1} * exp(-y^2/2)
    # Then f_x(x) = f_y(x/scale) * 1/scale
    k2 = 0.5 * k
    coeff = (2.0 ** (1.0 - k2)) / math.gamma(k2)
    fy = coeff * np.power(np.maximum(y, 0.0), k - 1) * np.exp(-0.5 * y * y)
    return fy / scale


def predict_tau_pdf(tau_x: np.ndarray, d_k: int, q_var_per_dim: float) -> np.ndarray:
    # tau = ||q||/sqrt(d_k), assume q ~ N(0, q_var_per_dim I)
    # ||q|| has Chi distribution scaled by sqrt(q_var_per_dim)
    # Transform: r = sqrt(d_k) * tau
    scale = math.sqrt(max(q_var_per_dim, 1e-12))
    r = math.sqrt(d_k) * tau_x
    fr = chi_scale_pdf(r, k=d_k, scale=scale)
    # change of variables dr/dtau = sqrt(d_k)
    return fr * math.sqrt(d_k)


def predict_logit_pdf(logit_x: np.ndarray, d_k: int, q_var_per_dim: float, k_var_per_dim: float) -> np.ndarray:
    # Under rough mean-field: q ~ N(0, q_var I), k ~ N(0, k_var I), independent.
    # Then q^T k has variance d_k * q_var * k_var.
    # logits = (q^T k)/sqrt(d_k) -> variance = q_var * k_var * d_k / d_k? actually:
    # Var(q^T k) = d_k * q_var * k_var => Var(logit) = (d_k*q_var*k_var)/d_k = q_var*k_var.
    sigma = math.sqrt(max(q_var_per_dim * k_var_per_dim, 1e-12))
    return normal_pdf(logit_x, mu=0.0, sigma=sigma)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def row_entropy_np(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.clip(P, eps, 1.0)
    return -np.sum(P * np.log(P), axis=-1)


def effective_rank_from_singular_values(s: np.ndarray, eps: float = 1e-12) -> float:
    s = np.maximum(s, 0.0)
    if s.sum() <= 0:
        return 0.0
    p = s / (s.sum() + eps)
    H = -(p * np.log(p + eps)).sum()
    return float(np.exp(H))


def mc_mean_field_predictions(
    alpha: np.ndarray,
    n_tokens: int,
    logit_sigma: float,
    n_rows: int,
    n_mats: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    # Mean-field model: logits rows are i.i.d. N(0, (alpha*sigma)^2)
    # We estimate:
    # - mean row entropy
    # - mean max prob
    # - effective rank(P) for full n_tokens x n_tokens matrices
    rng = np.random.default_rng(seed)
    alpha = np.asarray(alpha, dtype=np.float64)
    ent = np.zeros_like(alpha)
    maxp = np.zeros_like(alpha)
    rankeff = np.zeros_like(alpha)

    for i, a in enumerate(alpha):
        s = float(a) * float(logit_sigma)

        # Row-level stats via many independent rows
        logits_rows = rng.normal(loc=0.0, scale=s, size=(n_rows, n_tokens)).astype(np.float64)
        P_rows = softmax_np(logits_rows, axis=-1)
        ent[i] = float(row_entropy_np(P_rows).mean())
        maxp[i] = float(np.max(P_rows, axis=-1).mean())

        # Matrix-level stat (effective rank) via a few full matrices
        re_list = []
        for _ in range(n_mats):
            logits_mat = rng.normal(loc=0.0, scale=s, size=(n_tokens, n_tokens)).astype(np.float64)
            P = softmax_np(logits_mat, axis=-1)
            sv = np.linalg.svd(P, compute_uv=False)
            re_list.append(effective_rank_from_singular_values(sv))
        rankeff[i] = float(np.mean(re_list))

    return {
        "mean_row_entropy": ent,
        "max_p_mean": maxp,
        "rank_eff_P": rankeff,
    }


def mc_bilinear_predictions(
    alpha: np.ndarray,
    n_tokens: int,
    d_k: int,
    q_var_per_dim: float,
    k_var_per_dim: float,
    n_mats: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    # Structured mean-field model closer to attention:
    # Sample Q,K ~ Gaussian matrices, form logits L = Q K^T / sqrt(d_k)
    # Then P = softmax(alpha * L). Compute matrix-level and row-level stats.
    rng = np.random.default_rng(seed)
    alpha = np.asarray(alpha, dtype=np.float64)
    ent = np.zeros_like(alpha)
    maxp = np.zeros_like(alpha)
    rankeff = np.zeros_like(alpha)

    q_std = math.sqrt(max(q_var_per_dim, 1e-12))
    k_std = math.sqrt(max(k_var_per_dim, 1e-12))
    inv_sqrt_dk = 1.0 / math.sqrt(d_k)

    for i, a in enumerate(alpha):
        ent_list = []
        maxp_list = []
        re_list = []
        for _ in range(n_mats):
            Q = rng.normal(loc=0.0, scale=q_std, size=(n_tokens, d_k)).astype(np.float64)
            K = rng.normal(loc=0.0, scale=k_std, size=(n_tokens, d_k)).astype(np.float64)
            L = (Q @ K.T) * inv_sqrt_dk
            P = softmax_np(float(a) * L, axis=-1)
            ent_list.append(float(row_entropy_np(P).mean()))
            maxp_list.append(float(np.max(P, axis=-1).mean()))
            sv = np.linalg.svd(P, compute_uv=False)
            re_list.append(effective_rank_from_singular_values(sv))
        ent[i] = float(np.mean(ent_list))
        maxp[i] = float(np.mean(maxp_list))
        rankeff[i] = float(np.mean(re_list))

    return {
        "mean_row_entropy": ent,
        "max_p_mean": maxp,
        "rank_eff_P": rankeff,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare theory predictions vs empirical run outputs (overlap plots)")
    p.add_argument("--run_dir", type=str, required=True, help="Path like runs/run_YYYYMMDD_HHMMSS")
    p.add_argument("--mf_rows", type=int, default=20000, help="Rows sampled for mean-field entropy/maxp predictions")
    p.add_argument("--mf_mats", type=int, default=30, help="Full matrices sampled for mean-field rank_eff prediction")
    p.add_argument("--mf_seed", type=int, default=123, help="RNG seed for mean-field simulation")
    p.add_argument(
        "--bilinear_mats",
        type=int,
        default=30,
        help="Full matrices sampled for structured bilinear (QK^T) mean-field predictions",
    )
    p.add_argument("--bilinear_seed", type=int, default=456, help="RNG seed for structured bilinear mean-field")
    args, _unknown = p.parse_known_args()

    run_dir = Path(args.run_dir)
    logs = run_dir / "logs"
    plots = run_dir / "plots"
    out_plots = run_dir / "theory_vs_empirical"
    out_plots.mkdir(parents=True, exist_ok=True)

    summary_path = logs / "summary.json"
    sweep_path = logs / "temperature_sweep.csv"
    artifacts_path = logs / "artifacts.npz"

    if not summary_path.exists():
        raise SystemExit(f"Missing: {summary_path}")
    if not sweep_path.exists():
        raise SystemExit(f"Missing: {sweep_path}")
    if not artifacts_path.exists():
        raise SystemExit(
            f"Missing: {artifacts_path}. Re-run the experiment script after updating it to emit artifacts.npz."
        )

    summary = json.loads(summary_path.read_text())
    cfg = summary.get("config", {})

    sweep = load_csv(sweep_path)
    alpha = np.array([r["alpha"] for r in sweep], dtype=np.float64)
    ent_emp = np.array([r["mean_row_entropy"] for r in sweep], dtype=np.float64)
    rank_emp = np.array([r["rank_eff_P"] for r in sweep], dtype=np.float64)
    maxp_emp = np.array([r["max_p_mean"] for r in sweep], dtype=np.float64)

    art = np.load(artifacts_path)
    tau = np.array(art["tau"], dtype=np.float64)
    logits_sample = np.array(art["logits_sample"], dtype=np.float64)
    logits_sample = logits_sample[np.isfinite(logits_sample)]
    q_var_per_dim = float(art["q_var_per_dim"])
    k_var_per_dim = float(art["k_var_per_dim"])
    d_k = int(np.array(art["d_k"]).reshape(-1)[0])
    n_tokens = int(np.array(art["n_tokens"]).reshape(-1)[0])
    kernel = str(np.array(art["kernel"]).reshape(-1)[0]) if "kernel" in art else "softmax"

    logit_sigma = math.sqrt(max(q_var_per_dim * k_var_per_dim, 1e-12))

    # --- Overlap 1: tau distribution vs chi prediction ---
    plt.figure(figsize=(7, 4))
    bins = 60
    hist, edges = np.histogram(tau, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.plot(centers, hist, label="empirical", linewidth=2)

    tau_grid = np.linspace(max(1e-6, tau.min()), tau.max(), 400)
    tau_pred = predict_tau_pdf(tau_grid, d_k=d_k, q_var_per_dim=q_var_per_dim)
    plt.plot(tau_grid, tau_pred, label="theory (chi)", linewidth=2, linestyle="--")

    plt.xlabel("tau = ||q||/sqrt(d_k)")
    plt.ylabel("density")
    plt.legend()
    savefig(out_plots / "tau_overlap.png", "Tau: empirical vs chi-based theory")

    # --- Overlap 2: logits distribution vs gaussian prediction ---
    logits_overlap_path = out_plots / "logits_overlap.png"
    if logits_sample.size > 0:
        plt.figure(figsize=(7, 4))
        hist, edges = np.histogram(logits_sample, bins=80, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, label="empirical", linewidth=2)

        xg = np.linspace(centers.min(), centers.max(), 600)
        pred = predict_logit_pdf(xg, d_k=d_k, q_var_per_dim=q_var_per_dim, k_var_per_dim=k_var_per_dim)
        plt.plot(xg, pred, label="theory (gaussian)", linewidth=2, linestyle="--")

        plt.xlabel("logits")
        plt.ylabel("density")
        plt.legend()
        savefig(logits_overlap_path, "Logits: empirical vs mean-field gaussian theory")
    else:
        # If all logits are masked (e.g., -inf) or missing, skip this plot.
        logits_overlap_path = None

    # --- Overlap 3: entropy vs alpha (shape-prediction) ---
    plt.figure(figsize=(7, 4))
    plt.semilogx(alpha, ent_emp, marker="o", label="empirical")
    mf_iid = mc_mean_field_predictions(
        alpha,
        n_tokens=n_tokens,
        logit_sigma=logit_sigma,
        n_rows=int(args.mf_rows),
        n_mats=int(args.mf_mats),
        seed=int(args.mf_seed),
    )
    mf_bilin = mc_bilinear_predictions(
        alpha,
        n_tokens=n_tokens,
        d_k=d_k,
        q_var_per_dim=q_var_per_dim,
        k_var_per_dim=k_var_per_dim,
        n_mats=int(args.bilinear_mats),
        seed=int(args.bilinear_seed),
    )
    plt.semilogx(alpha, mf_iid["mean_row_entropy"], linestyle="--", linewidth=2, label="theory (i.i.d. logits MC)")
    plt.semilogx(alpha, mf_bilin["mean_row_entropy"], linestyle="--", linewidth=2, label="theory (bilinear QK^T MC)")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean row entropy")
    plt.legend()
    savefig(out_plots / "entropy_vs_alpha_overlap.png", "Entropy vs alpha: empirical vs theory baselines")

    # --- Overlap 4: rank_eff(P) vs alpha (bump-shape prediction) ---
    plt.figure(figsize=(7, 4))
    plt.semilogx(alpha, rank_emp, marker="o", label="empirical")
    plt.semilogx(alpha, mf_iid["rank_eff_P"], linestyle="--", linewidth=2, label="theory (i.i.d. logits MC)")
    plt.semilogx(alpha, mf_bilin["rank_eff_P"], linestyle="--", linewidth=2, label="theory (bilinear QK^T MC)")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("effective rank(P)")
    plt.legend()
    savefig(out_plots / "rankeff_vs_alpha_overlap.png", "Effective rank vs alpha: empirical vs theory baselines")

    # --- Overlap 5: peakedness vs alpha (monotonic proxy prediction) ---
    plt.figure(figsize=(7, 4))
    plt.semilogx(alpha, maxp_emp, marker="o", label="empirical")
    plt.semilogx(alpha, mf_iid["max_p_mean"], linestyle="--", linewidth=2, label="theory (i.i.d. logits MC)")
    plt.semilogx(alpha, mf_bilin["max_p_mean"], linestyle="--", linewidth=2, label="theory (bilinear QK^T MC)")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean max row probability")
    plt.legend()
    savefig(out_plots / "peakedness_vs_alpha_overlap.png", "Peakedness vs alpha: empirical vs theory baselines")

    report = {
        "run_dir": str(run_dir),
        "d_k": d_k,
        "n_tokens": n_tokens,
        "q_var_per_dim": q_var_per_dim,
        "k_var_per_dim": k_var_per_dim,
        "kernel": kernel,
        "logit_sigma": logit_sigma,
        "mean_field": {
            "mf_rows": int(args.mf_rows),
            "mf_mats": int(args.mf_mats),
            "mf_seed": int(args.mf_seed),
            "bilinear_mats": int(args.bilinear_mats),
            "bilinear_seed": int(args.bilinear_seed),
        },
        "outputs": {
            "tau_overlap": str(out_plots / "tau_overlap.png"),
            "logits_overlap": None if logits_overlap_path is None else str(logits_overlap_path),
            "entropy_overlap": str(out_plots / "entropy_vs_alpha_overlap.png"),
            "rankeff_overlap": str(out_plots / "rankeff_vs_alpha_overlap.png"),
            "peakedness_overlap": str(out_plots / "peakedness_vs_alpha_overlap.png"),
        },
        "notes": {
            "tau_theory": "Assumes Q components are Gaussian with per-dim variance q_var_per_dim; then tau distribution is chi-based.",
            "logit_theory": "Assumes q and k are independent Gaussians; then logits are approx N(0, q_var_per_dim*k_var_per_dim).",
            "entropy_rank_theory": "Computes two MC theory baselines: (1) i.i.d. Gaussian logits with sigma^2=q_var_per_dim*k_var_per_dim; (2) structured bilinear logits L=QK^T/sqrt(d_k) with Q,K Gaussian using observed per-dim variances. Both passed through softmax to estimate entropy/maxp/rank_eff.",
        },
    }
    (out_plots / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"Wrote overlay plots to: {out_plots}")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
