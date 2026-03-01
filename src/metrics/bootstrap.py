"""
Bootstrap confidence intervals for diversity metrics.

Provides generic bootstrap CI computation for any scalar metric,
specialised wrappers for Kendall τ, fair-selection diversity retention,
and selector accuracy metrics.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for *statistic* on *data*.

    Args:
        data: 1-D array of observations.
        statistic: Function mapping a 1-D array to a scalar.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: RNG seed.

    Returns:
        Dict with keys: point, ci_lower, ci_upper, std, n_bootstrap.
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)
    point = float(statistic(data))

    boot_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_vals[i] = statistic(data[idx])

    alpha = 1 - confidence
    return {
        "point": point,
        "ci_lower": float(np.percentile(boot_vals, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_vals, 100 * (1 - alpha / 2))),
        "std": float(np.std(boot_vals)),
        "n_bootstrap": n_bootstrap,
    }


# ------------------------------------------------------------------
# Kendall τ with bootstrap CI
# ------------------------------------------------------------------

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall τ between two arrays (O(n²) simple implementation)."""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def bootstrap_kendall_tau(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Kendall τ with bootstrap CI.

    Returns:
        Dict with point, ci_lower, ci_upper, std, n_bootstrap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(x)
    point = _kendall_tau(x, y)

    boot_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_vals[i] = _kendall_tau(x[idx], y[idx])

    alpha = 1 - confidence
    return {
        "point": point,
        "ci_lower": float(np.percentile(boot_vals, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_vals, 100 * (1 - alpha / 2))),
        "std": float(np.std(boot_vals)),
        "n_bootstrap": n_bootstrap,
    }


# ------------------------------------------------------------------
# Fair-selection diversity retention with bootstrap CI
# ------------------------------------------------------------------

def bootstrap_fair_retention(
    unconstrained_scores: np.ndarray,
    fair_scores: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for diversity retention ratio (fair / unconstrained).

    Both arrays should be paired trial-level observations of total
    pairwise distance (or any diversity measure).
    """
    unc = np.asarray(unconstrained_scores, dtype=float)
    fair = np.asarray(fair_scores, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(unc)

    def retention(idx: np.ndarray) -> float:
        u = np.mean(unc[idx])
        f = np.mean(fair[idx])
        return f / u if u > 0 else 0.0

    point = retention(np.arange(n))

    boot_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_vals[i] = retention(idx)

    alpha = 1 - confidence
    return {
        "point": point,
        "ci_lower": float(np.percentile(boot_vals, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_vals, 100 * (1 - alpha / 2))),
        "std": float(np.std(boot_vals)),
        "n_bootstrap": n_bootstrap,
    }
