"""
Information-theoretic diversity baselines.

Implements mutual information (MI), KL divergence, and entropy rate
estimators for comparing text generation diversity across methods.
All estimators operate on n-gram frequency distributions extracted
from text corpora.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokeniser."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _ngram_distribution(texts: List[str], n: int = 2) -> Dict[tuple, float]:
    """Return normalised n-gram probability distribution over *texts*."""
    counts: Counter = Counter()
    for t in texts:
        tokens = _tokenize(t)
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i : i + n])] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


# ------------------------------------------------------------------
# Shannon entropy
# ------------------------------------------------------------------

def shannon_entropy(texts: List[str], n: int = 2) -> float:
    r"""Shannon entropy of the n-gram distribution (bits).

    .. math::
        H(P) = -\sum_g p(g) \log_2 p(g)
    """
    dist = _ngram_distribution(texts, n)
    if not dist:
        return 0.0
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


# ------------------------------------------------------------------
# KL divergence
# ------------------------------------------------------------------

def kl_divergence(
    texts_p: List[str],
    texts_q: List[str],
    n: int = 2,
    smoothing: float = 1e-10,
) -> float:
    r"""KL divergence D_KL(P || Q) between n-gram distributions (bits).

    .. math::
        D_{KL}(P \| Q) = \sum_g p(g) \log_2 \frac{p(g)}{q(g)}

    Uses additive smoothing to avoid division by zero.
    """
    p = _ngram_distribution(texts_p, n)
    q = _ngram_distribution(texts_q, n)
    if not p:
        return 0.0
    vocab = set(p.keys()) | set(q.keys())
    total = 0.0
    for g in vocab:
        pg = p.get(g, 0.0) + smoothing
        qg = q.get(g, 0.0) + smoothing
        total += pg * math.log2(pg / qg)
    return total


def symmetric_kl(
    texts_a: List[str],
    texts_b: List[str],
    n: int = 2,
    smoothing: float = 1e-10,
) -> float:
    """Jensen–Shannon-style symmetric KL: 0.5*(KL(P||Q) + KL(Q||P))."""
    return 0.5 * (
        kl_divergence(texts_a, texts_b, n, smoothing)
        + kl_divergence(texts_b, texts_a, n, smoothing)
    )


# ------------------------------------------------------------------
# Mutual information between two text sets
# ------------------------------------------------------------------

def mutual_information(
    texts_a: List[str],
    texts_b: List[str],
    n: int = 2,
) -> float:
    r"""Estimated mutual information between two text corpora.

    Computes MI via:
    .. math::
        I(A; B) = H(A) + H(B) - H(A, B)

    where H(A,B) is estimated from the combined corpus.
    """
    ha = shannon_entropy(texts_a, n)
    hb = shannon_entropy(texts_b, n)
    hab = shannon_entropy(texts_a + texts_b, n)
    # MI = H(A) + H(B) - H(A,B); clamp to >= 0 for finite-sample noise
    return max(ha + hb - hab, 0.0)


# ------------------------------------------------------------------
# Entropy rate estimator
# ------------------------------------------------------------------

def entropy_rate(texts: List[str], max_order: int = 5) -> Tuple[float, List[float]]:
    r"""Estimate entropy rate via conditional entropy convergence.

    The entropy rate is :math:`h = \lim_{n\to\infty} H(X_n | X_{n-1}, \ldots, X_1)`.
    We approximate by computing :math:`h_n = H_n - H_{n-1}` for increasing
    n-gram orders and returning the last stable value.

    Returns:
        (estimated_rate, list of h_n for n=1..max_order)
    """
    entropies = []
    for order in range(1, max_order + 1):
        entropies.append(shannon_entropy(texts, n=order))
    # Conditional entropy: h_n = H(n) - H(n-1)
    conditionals = [entropies[0]]
    for i in range(1, len(entropies)):
        conditionals.append(max(entropies[i] - entropies[i - 1], 0.0))
    return conditionals[-1], conditionals


# ------------------------------------------------------------------
# Bootstrap CI wrapper
# ------------------------------------------------------------------

def bootstrap_entropy_ci(
    texts: List[str],
    n: int = 2,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for Shannon entropy."""
    rng = np.random.RandomState(seed)
    point = shannon_entropy(texts, n)
    m = len(texts)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(m, size=m, replace=True)
        sample = [texts[i] for i in idx]
        boot_vals.append(shannon_entropy(sample, n))
    boot_vals = np.array(boot_vals)
    alpha = 1 - confidence
    return {
        "point": point,
        "ci_lower": float(np.percentile(boot_vals, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_vals, 100 * (1 - alpha / 2))),
        "std": float(np.std(boot_vals)),
    }
