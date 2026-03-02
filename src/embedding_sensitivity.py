"""
Embedding sensitivity analysis: test diversity metrics under different
embedding kernels (RBF, cosine, IMQ) and bandwidths.

Addresses reviewer critique that EPD and other embedding-based metrics
may be sensitive to the choice of kernel/embedding, making the taxonomy
fragile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


@dataclass
class KernelSensitivityResult:
    """Results of kernel sensitivity analysis for a single metric."""
    metric_name: str
    kernel_results: Dict[str, float]  # kernel_name → metric value
    max_relative_change: float
    cv: float  # coefficient of variation across kernels
    is_stable: bool  # CV < 0.15


@dataclass
class EmbeddingSensitivityAnalysis:
    """Full embedding sensitivity analysis results."""
    n_texts: int
    n_kernels: int
    per_metric: List[KernelSensitivityResult]
    tau_stability: Dict[str, float]  # metric_pair → mean |Δτ| across kernels
    summary: Dict[str, Any]


def rbf_kernel(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF kernel matrix."""
    sq_dists = squareform(pdist(X, 'sqeuclidean'))
    return np.exp(-sq_dists / (2 * bandwidth ** 2 + 1e-12))


def imq_kernel(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """Inverse multiquadric kernel matrix."""
    sq_dists = squareform(pdist(X, 'sqeuclidean'))
    return np.power(1.0 + sq_dists / (bandwidth ** 2 + 1e-12), -0.5)


def cosine_kernel(X: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Cosine similarity kernel matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_norm = X / norms
    return X_norm @ X_norm.T


def laplacian_kernel(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """Laplacian kernel matrix: k(x,y) = exp(-||x-y||₁ / h)."""
    l1_dists = squareform(pdist(X, 'cityblock'))
    return np.exp(-l1_dists / (bandwidth + 1e-12))


def polynomial_kernel(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """Polynomial kernel: k(x,y) = (x·y + 1)^d, d = max(2, round(bandwidth))."""
    degree = max(2, round(bandwidth))
    return (X @ X.T + 1.0) ** degree


KERNELS = {
    'rbf': rbf_kernel,
    'imq': imq_kernel,
    'cosine': cosine_kernel,
    'laplacian': laplacian_kernel,
    'polynomial': polynomial_kernel,
}


def epd_with_kernel(
    embeddings: np.ndarray,
    kernel_name: str = 'rbf',
    bandwidth: float = 1.0,
) -> float:
    """Compute Embedding Pairwise Distance using a specific kernel.

    EPD = 1 - mean(off-diagonal kernel values), so higher = more diverse.
    """
    kernel_fn = KERNELS.get(kernel_name, rbf_kernel)
    K = kernel_fn(embeddings, bandwidth)
    n = K.shape[0]
    if n < 2:
        return 0.0
    # Mean off-diagonal
    mask = ~np.eye(n, dtype=bool)
    mean_sim = float(np.mean(K[mask]))
    return 1.0 - mean_sim


def vendi_score_with_kernel(
    embeddings: np.ndarray,
    kernel_name: str = 'rbf',
    bandwidth: float = 1.0,
) -> float:
    """Vendi Score = exp(H(eigenvalues of normalised kernel matrix))."""
    kernel_fn = KERNELS.get(kernel_name, rbf_kernel)
    K = kernel_fn(embeddings, bandwidth)
    n = K.shape[0]
    if n < 2:
        return 1.0
    # Normalise
    trace = np.trace(K)
    if trace < 1e-12:
        return 1.0
    K_norm = K / trace
    eigvals = np.linalg.eigvalsh(K_norm)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 1.0
    entropy = -float(np.sum(eigvals * np.log(eigvals)))
    return float(np.exp(entropy))


def median_bandwidth(X: np.ndarray) -> float:
    """Median heuristic bandwidth."""
    dists = pdist(X, 'sqeuclidean')
    if len(dists) == 0:
        return 1.0
    med = float(np.median(dists))
    return max(math.sqrt(med / (2 * math.log(X.shape[0] + 1) + 1e-12)), 1e-6)


def run_sensitivity_analysis(
    embeddings_list: List[np.ndarray],
    labels: List[str] | None = None,
    bandwidths: List[float] | None = None,
    seed: int = 42,
) -> EmbeddingSensitivityAnalysis:
    """Run embedding sensitivity analysis across kernels and bandwidths.

    Args:
        embeddings_list: List of embedding matrices (one per text group).
        labels: Optional labels for each group.
        bandwidths: Bandwidths to test. If None, uses [0.5h, h, 2h, 5h]
            where h is the median heuristic.
        seed: Random seed.

    Returns:
        EmbeddingSensitivityAnalysis with per-metric stability results.
    """
    rng = np.random.RandomState(seed)

    if not embeddings_list:
        # Generate synthetic data
        n_groups = 20
        embeddings_list = [rng.randn(10, 50) for _ in range(n_groups)]

    n_groups = len(embeddings_list)

    # Determine bandwidths
    all_emb = np.vstack(embeddings_list)
    h_med = median_bandwidth(all_emb)
    if bandwidths is None:
        bandwidths = [0.5 * h_med, h_med, 2.0 * h_med, 5.0 * h_med]

    kernel_names = list(KERNELS.keys())

    # Compute EPD and Vendi for each (kernel, bandwidth, group)
    epd_results: Dict[str, List[float]] = {}
    vendi_results: Dict[str, List[float]] = {}

    for kname in kernel_names:
        for bw in bandwidths:
            key = f"{kname}_bw{bw:.3f}"
            epd_vals = []
            vendi_vals = []
            for emb in embeddings_list:
                epd_vals.append(epd_with_kernel(emb, kname, bw))
                vendi_vals.append(vendi_score_with_kernel(emb, kname, bw))
            epd_results[key] = epd_vals
            vendi_results[key] = vendi_vals

    # Compute ranking stability across kernel configurations
    # For each pair of configurations, compute Kendall τ of the group rankings
    config_keys = list(epd_results.keys())
    n_configs = len(config_keys)

    epd_taus = []
    vendi_taus = []
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            tau_epd, _ = stats.kendalltau(
                epd_results[config_keys[i]], epd_results[config_keys[j]]
            )
            tau_vendi, _ = stats.kendalltau(
                vendi_results[config_keys[i]], vendi_results[config_keys[j]]
            )
            if not np.isnan(tau_epd):
                epd_taus.append(tau_epd)
            if not np.isnan(tau_vendi):
                vendi_taus.append(tau_vendi)

    mean_epd_tau = float(np.mean(epd_taus)) if epd_taus else 0.0
    mean_vendi_tau = float(np.mean(vendi_taus)) if vendi_taus else 0.0

    # Per-metric summary
    per_metric = []
    for metric_name, results_dict in [("EPD", epd_results), ("Vendi", vendi_results)]:
        all_means = {}
        for key, vals in results_dict.items():
            all_means[key] = float(np.mean(vals))

        values = list(all_means.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = float(std_val / (abs(mean_val) + 1e-12))
        max_change = float(
            (max(values) - min(values)) / (abs(mean_val) + 1e-12)
        ) if values else 0.0

        per_metric.append(KernelSensitivityResult(
            metric_name=metric_name,
            kernel_results=all_means,
            max_relative_change=round(max_change, 4),
            cv=round(cv, 4),
            is_stable=cv < 0.15,
        ))

    # Bandwidth sensitivity: fix kernel, vary bandwidth
    bw_sensitivity = {}
    for kname in kernel_names:
        bw_keys = [k for k in config_keys if k.startswith(kname)]
        if len(bw_keys) < 2:
            continue
        bw_taus = []
        for i in range(len(bw_keys)):
            for j in range(i + 1, len(bw_keys)):
                tau, _ = stats.kendalltau(
                    epd_results[bw_keys[i]], epd_results[bw_keys[j]]
                )
                if not np.isnan(tau):
                    bw_taus.append(tau)
        bw_sensitivity[kname] = float(np.mean(bw_taus)) if bw_taus else 0.0

    summary = {
        "n_groups": n_groups,
        "n_kernels": len(kernel_names),
        "n_bandwidths": len(bandwidths),
        "n_configs_total": n_configs,
        "epd_ranking_stability": {
            "mean_cross_kernel_tau": round(mean_epd_tau, 4),
            "interpretation": (
                "stable" if mean_epd_tau > 0.7 else
                "moderate" if mean_epd_tau > 0.4 else
                "unstable"
            ),
        },
        "vendi_ranking_stability": {
            "mean_cross_kernel_tau": round(mean_vendi_tau, 4),
            "interpretation": (
                "stable" if mean_vendi_tau > 0.7 else
                "moderate" if mean_vendi_tau > 0.4 else
                "unstable"
            ),
        },
        "bandwidth_sensitivity": {
            k: round(v, 4) for k, v in bw_sensitivity.items()
        },
        "per_metric_cv": {
            m.metric_name: m.cv for m in per_metric
        },
        "recommendation": (
            "EPD rankings are robust to kernel choice"
            if mean_epd_tau > 0.7 else
            "EPD rankings show moderate kernel sensitivity — "
            "report kernel choice alongside EPD values"
        ),
    }

    return EmbeddingSensitivityAnalysis(
        n_texts=n_groups,
        n_kernels=len(kernel_names),
        per_metric=per_metric,
        tau_stability={
            "EPD_cross_kernel": round(mean_epd_tau, 4),
            "Vendi_cross_kernel": round(mean_vendi_tau, 4),
        },
        summary=summary,
    )


if __name__ == "__main__":
    import json
    result = run_sensitivity_analysis([], seed=42)
    print(json.dumps(result.summary, indent=2, default=str))
