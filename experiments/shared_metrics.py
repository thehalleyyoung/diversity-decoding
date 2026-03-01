"""
Shared metric computation for experiments.

ALL experiments should use this module instead of inline metric
reimplementations. This ensures consistency between paper claims
and actual metric implementations.

Usage:
    from experiments.shared_metrics import compute_all_diversity_metrics
"""

import math
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.diversity import (
    SelfBLEU,
    DistinctN,
    NGramEntropy,
    EmbeddingPairwiseDistance,
    VendiScore,
    ParseTreeDiversity,
    BehavioralDiversity,
    tokenize_simple,
)
from src.metrics.neural_diversity import (
    MAUVE,
    BERTScoreDiversity,
    STSDiversity,
    CompressionRatioDiversity,
)


def compute_all_diversity_metrics(
    texts: List[str],
    include_neural: bool = True,
    include_mauve: bool = False,
) -> Dict[str, float]:
    """Compute all diversity metrics using the canonical src implementations.

    Args:
        texts: List of generated texts (at least 2).
        include_neural: Include BERTScore/STS diversity (TF-IDF based).
        include_mauve: Include MAUVE (slower, requires more texts).

    Returns:
        Dict mapping metric names to float values.
    """
    if len(texts) < 2:
        return {}

    metrics = {}

    # Core diversity metrics
    core_instances = [
        ("distinct_2", DistinctN(n=2)),
        ("self_bleu", SelfBLEU(max_order=4)),
        ("ngram_entropy", NGramEntropy(n=2)),
        ("epd", EmbeddingPairwiseDistance()),
        ("vendi_score", VendiScore()),
        ("parse_tree_diversity", ParseTreeDiversity()),
        ("crd", CompressionRatioDiversity()),
    ]

    for name, metric in core_instances:
        try:
            metrics[name] = metric.compute(texts)
        except Exception:
            metrics[name] = float("nan")

    # Neural diversity metrics
    if include_neural:
        neural_instances = [
            ("bertscore_diversity", BERTScoreDiversity(backend="tfidf")),
            ("sts_diversity", STSDiversity(backend="tfidf")),
        ]
        for name, metric in neural_instances:
            try:
                metrics[name] = metric.compute(texts)
            except Exception:
                metrics[name] = float("nan")

    if include_mauve and len(texts) >= 4:
        try:
            mauve = MAUVE(
                n_clusters=min(50, len(texts) // 2), backend="tfidf"
            )
            metrics["mauve"] = mauve.compute(texts)
        except Exception:
            metrics["mauve"] = float("nan")

    # Inline metrics (simple enough to not need a class)
    try:
        all_tokens = []
        for t in texts:
            all_tokens.extend(tokenize_simple(t))
        metrics["ttr"] = len(set(all_tokens)) / max(len(all_tokens), 1)
    except Exception:
        metrics["ttr"] = float("nan")

    try:
        token_sets = [set(tokenize_simple(t)) for t in texts]
        dists = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                union = len(token_sets[i] | token_sets[j])
                inter = len(token_sets[i] & token_sets[j])
                dists.append(1.0 - inter / max(union, 1))
        metrics["jaccard"] = sum(dists) / len(dists) if dists else 0.0
    except Exception:
        metrics["jaccard"] = float("nan")

    metrics["n_unique"] = len(set(texts))

    return metrics


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Kendall τ rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 2:
        return 0.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = x[i] - x[j], y[i] - y[j]
            if xi * yi > 0:
                c += 1
            elif xi * yi < 0:
                d += 1
    return (c - d) / (c + d) if (c + d) > 0 else 0.0
