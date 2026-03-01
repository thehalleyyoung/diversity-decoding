"""
Neural diversity metrics: MAUVE, BERTScore diversity, STS-based semantic diversity.

These metrics use neural embeddings to capture distributional and semantic
diversity properties that purely lexical metrics miss.

References:
    - Pillutla et al. (2021) "MAUVE: Measuring the Gap Between Neural Text
      and Human Text using Divergence Frontiers"
    - Zhang et al. (2020) "BERTScore: Evaluating Text Generation with BERT"
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import rel_entr

from src.metrics.diversity import (
    DiversityMetric,
    bootstrap_confidence_interval,
    tfidf_embeddings,
    tokenize_simple,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding backends — lightweight fallback + optional transformer backends
# ---------------------------------------------------------------------------

class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return (n_texts, dim) matrix of embeddings."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class TFIDFBackend(EmbeddingBackend):
    """TF-IDF fallback (no GPU required)."""

    def __init__(self, max_features: int = 5000):
        self._max_features = max_features

    def encode(self, texts: List[str]) -> np.ndarray:
        return tfidf_embeddings(texts, max_features=self._max_features)

    @property
    def dim(self) -> int:
        return self._max_features


class TransformerBackend(EmbeddingBackend):
    """Sentence-transformer backend using HuggingFace."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._device = device
        self._dim_cache: Optional[int] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.eval()
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._dim_cache = self._model.config.hidden_size
        except ImportError:
            raise ImportError(
                "TransformerBackend requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        import torch

        self._load()
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            # Mean pooling over token embeddings
            attention_mask = encoded["attention_mask"]
            token_embs = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
            sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = (sum_embs / sum_mask).cpu().numpy()
            # L2 normalise
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pooled = pooled / norms
            all_embeddings.append(pooled)
        return np.vstack(all_embeddings)

    @property
    def dim(self) -> int:
        if self._dim_cache is not None:
            return self._dim_cache
        self._load()
        return self._dim_cache


def get_backend(
    backend: str = "tfidf",
    model_name: Optional[str] = None,
    **kwargs,
) -> EmbeddingBackend:
    """Factory for embedding backends."""
    if backend == "tfidf":
        return TFIDFBackend(**kwargs)
    elif backend == "transformer":
        mn = model_name or "all-MiniLM-L6-v2"
        return TransformerBackend(model_name=mn, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# MAUVE metric  (Pillutla et al. 2021)
# ---------------------------------------------------------------------------

def _kmeans_cluster(X: np.ndarray, k: int, max_iter: int = 100,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means returning (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = rng.choice(n, size=min(k, n), replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = cdist(X, centroids, metric="euclidean")
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)
    return labels, centroids


def _compute_divergence_curve(
    p_hist: np.ndarray,
    q_hist: np.ndarray,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the MAUVE divergence frontier curve.

    For each mixture coefficient λ ∈ [0, 1], computes
        KL(p || λp + (1-λ)q)  and  KL(q || λp + (1-λ)q)
    Returns arrays of (KL_p, KL_q) at each λ.
    """
    eps = 1e-10
    p = p_hist + eps
    q = q_hist + eps
    p = p / p.sum()
    q = q / q.sum()

    lambdas = np.linspace(0.0, 1.0, num_points)
    kl_p_vals = np.zeros(num_points)
    kl_q_vals = np.zeros(num_points)

    for i, lam in enumerate(lambdas):
        mixture = lam * p + (1 - lam) * q
        kl_p_vals[i] = float(np.sum(rel_entr(p, mixture)))
        kl_q_vals[i] = float(np.sum(rel_entr(q, mixture)))

    return kl_p_vals, kl_q_vals


def _compute_mauve_area(kl_p: np.ndarray, kl_q: np.ndarray) -> float:
    """Compute area under the divergence frontier curve.

    MAUVE = exp(-c * area) where c is a scaling constant.
    Following Pillutla et al., we use the area under the
    (exp(-KL_q), exp(-KL_p)) curve as the MAUVE score.
    """
    x = np.exp(-kl_q)
    y = np.exp(-kl_p)

    # Sort by x for proper AUC
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Trapezoidal AUC
    area = float(np.trapz(y_sorted, x_sorted))
    return area


class MAUVE(DiversityMetric):
    """MAUVE score measuring distributional divergence between text sets.

    Implements the divergence frontier approach from Pillutla et al. (2021).
    Given a set of generated texts, splits them into two halves and measures
    distributional divergence via quantized embeddings.

    For diversity measurement: we compare the generated set against a
    "reference" distribution. When no reference is given, we use a
    self-diversity variant that measures internal distributional spread.

    Args:
        n_clusters: Number of clusters for quantisation (default 500).
        backend: Embedding backend ("tfidf" or "transformer").
        model_name: Model for transformer backend.
        num_curve_points: Points on the divergence frontier curve.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 500,
        backend: str = "tfidf",
        model_name: Optional[str] = None,
        num_curve_points: int = 100,
        seed: int = 42,
    ):
        self._n_clusters = n_clusters
        self._backend = get_backend(backend, model_name=model_name)
        self._num_curve_points = num_curve_points
        self._seed = seed

    def compute(self, generation_set: List[str]) -> float:
        """Compute self-MAUVE diversity (higher = more diverse)."""
        self.validate_input(generation_set)
        return self.compute_between(generation_set, generation_set)

    def compute_between(
        self,
        p_texts: List[str],
        q_texts: List[str],
    ) -> float:
        """Compute MAUVE between two text distributions.

        When p_texts == q_texts, uses a split-half approach for
        self-diversity measurement.
        """
        same_set = (p_texts is q_texts) or (p_texts == q_texts)

        if same_set:
            # Split-half self-diversity
            rng = np.random.default_rng(self._seed)
            indices = rng.permutation(len(p_texts))
            mid = len(indices) // 2
            p_idx, q_idx = indices[:mid], indices[mid:]
            if len(p_idx) < 2 or len(q_idx) < 2:
                return 1.0  # Too few texts to measure
            # Encode all together to get consistent embedding dimensions
            all_texts = [p_texts[i] for i in p_idx] + [p_texts[i] for i in q_idx]
            all_embs = self._backend.encode(all_texts)
            n_p = len(p_idx)
            p_embs = all_embs[:n_p]
            q_embs = all_embs[n_p:]
        else:
            # Encode all together for consistent dimensions
            all_texts = list(p_texts) + list(q_texts)
            all_embs_full = self._backend.encode(all_texts)
            n_p = len(p_texts)
            p_embs = all_embs_full[:n_p]
            q_embs = all_embs_full[n_p:]

        # Combine and cluster
        all_embs = np.vstack([p_embs, q_embs])
        k = min(self._n_clusters, all_embs.shape[0] // 2)
        k = max(k, 2)  # At least 2 clusters

        labels, _ = _kmeans_cluster(all_embs, k, seed=self._seed)

        n_p = p_embs.shape[0]
        p_labels = labels[:n_p]
        q_labels = labels[n_p:]

        # Build cluster histograms
        p_hist = np.zeros(k, dtype=np.float64)
        q_hist = np.zeros(k, dtype=np.float64)
        for lbl in p_labels:
            p_hist[lbl] += 1
        for lbl in q_labels:
            q_hist[lbl] += 1

        # Normalise
        p_hist = p_hist / (p_hist.sum() + 1e-12)
        q_hist = q_hist / (q_hist.sum() + 1e-12)

        # Compute divergence frontier
        kl_p, kl_q = _compute_divergence_curve(
            p_hist, q_hist, self._num_curve_points
        )

        # MAUVE = area under divergence frontier curve
        mauve_score = _compute_mauve_area(kl_p, kl_q)
        return float(np.clip(mauve_score, 0.0, 1.0))

    def compute_detailed(
        self, generation_set: List[str]
    ) -> Dict[str, Any]:
        """Return MAUVE score with diagnostic information."""
        self.validate_input(generation_set)
        score = self.compute(generation_set)
        return {
            "mauve_score": score,
            "n_texts": len(generation_set),
            "n_clusters": self._n_clusters,
            "backend": type(self._backend).__name__,
        }

    @property
    def name(self) -> str:
        return "MAUVE"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "MAUVE: distributional divergence via quantised embedding "
            "histograms and divergence frontiers (Pillutla et al. 2021)."
        )


# ---------------------------------------------------------------------------
# BERTScore Diversity
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between row-vectors of A and B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


class BERTScoreDiversity(DiversityMetric):
    """Diversity metric based on BERTScore-style pairwise comparisons.

    Instead of measuring similarity to a reference, we compute pairwise
    BERTScore between all generated texts and derive diversity as
    1 - mean(pairwise_bertscore_F1). Lower pairwise similarity means
    higher diversity.

    Uses token-level embeddings (or sentence-level as fallback) to
    compute precision, recall, and F1 between text pairs.

    Args:
        backend: Embedding backend for text representation.
        model_name: Model for transformer backend.
    """

    def __init__(
        self,
        backend: str = "tfidf",
        model_name: Optional[str] = None,
    ):
        self._backend = get_backend(backend, model_name=model_name)

    def compute(self, generation_set: List[str]) -> float:
        """Compute BERTScore diversity: 1 - mean pairwise F1."""
        self.validate_input(generation_set)
        n = len(generation_set)

        embs = self._backend.encode(generation_set)
        sim_matrix = _cosine_similarity_matrix(embs, embs)

        # Extract upper-triangle (exclude diagonal)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]

        if len(pairwise_sims) == 0:
            return 0.0

        mean_sim = float(np.mean(pairwise_sims))
        # Diversity = 1 - mean similarity (clipped to [0, 1])
        diversity = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
        return diversity

    def compute_detailed(
        self, generation_set: List[str]
    ) -> Dict[str, Any]:
        """Return diversity with pairwise similarity statistics."""
        self.validate_input(generation_set)
        n = len(generation_set)
        embs = self._backend.encode(generation_set)
        sim_matrix = _cosine_similarity_matrix(embs, embs)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]

        return {
            "diversity": float(np.clip(1.0 - np.mean(pairwise_sims), 0.0, 1.0)),
            "mean_pairwise_similarity": float(np.mean(pairwise_sims)),
            "std_pairwise_similarity": float(np.std(pairwise_sims)),
            "min_pairwise_similarity": float(np.min(pairwise_sims)),
            "max_pairwise_similarity": float(np.max(pairwise_sims)),
            "n_pairs": len(pairwise_sims),
        }

    @property
    def name(self) -> str:
        return "BERTScore-Diversity"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "BERTScore diversity: 1 - mean pairwise cosine similarity "
            "of sentence embeddings. Higher = more diverse."
        )


# ---------------------------------------------------------------------------
# STS-Based Semantic Diversity
# ---------------------------------------------------------------------------

class STSDiversity(DiversityMetric):
    """Semantic Textual Similarity (STS) based diversity.

    Uses the distribution of pairwise semantic similarities to compute
    diversity. Provides multiple aggregations: mean distance, entropy
    of similarity distribution, and coverage spread.

    Args:
        backend: Embedding backend.
        model_name: Model for transformer backend.
        n_bins: Bins for similarity histogram entropy.
    """

    def __init__(
        self,
        backend: str = "tfidf",
        model_name: Optional[str] = None,
        n_bins: int = 20,
    ):
        self._backend = get_backend(backend, model_name=model_name)
        self._n_bins = n_bins

    def compute(self, generation_set: List[str]) -> float:
        """Compute STS diversity (higher = more diverse)."""
        self.validate_input(generation_set)
        result = self._compute_all(generation_set)
        return result["sts_diversity"]

    def _compute_all(self, texts: List[str]) -> Dict[str, float]:
        """Compute all STS diversity sub-metrics."""
        n = len(texts)
        embs = self._backend.encode(texts)
        sim_matrix = _cosine_similarity_matrix(embs, embs)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]

        if len(pairwise_sims) == 0:
            return {
                "sts_diversity": 0.0,
                "mean_distance": 0.0,
                "sim_entropy": 0.0,
                "coverage_spread": 0.0,
            }

        # 1. Mean pairwise distance
        mean_dist = float(1.0 - np.mean(pairwise_sims))

        # 2. Entropy of similarity distribution
        hist, _ = np.histogram(pairwise_sims, bins=self._n_bins, range=(0, 1))
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum > 0:
            probs = hist / hist_sum
            probs = probs[probs > 0]
            sim_entropy = float(-np.sum(probs * np.log2(probs)))
            max_entropy = np.log2(self._n_bins) if self._n_bins > 1 else 1.0
            sim_entropy_norm = sim_entropy / max_entropy
        else:
            sim_entropy_norm = 0.0

        # 3. Coverage spread (std of distances in embedding space)
        dists_eucl = pdist(embs, metric="euclidean")
        coverage_spread = float(np.std(dists_eucl)) if len(dists_eucl) > 0 else 0.0

        # Combined diversity score
        sts_diversity = float(np.clip(
            0.5 * mean_dist + 0.3 * sim_entropy_norm + 0.2 * min(coverage_spread, 1.0),
            0.0, 1.0
        ))

        return {
            "sts_diversity": sts_diversity,
            "mean_distance": mean_dist,
            "sim_entropy": sim_entropy_norm,
            "coverage_spread": coverage_spread,
        }

    def compute_detailed(self, generation_set: List[str]) -> Dict[str, Any]:
        self.validate_input(generation_set)
        result = self._compute_all(generation_set)
        result["n_texts"] = len(generation_set)
        return result

    @property
    def name(self) -> str:
        return "STS-Diversity"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Semantic Textual Similarity diversity: combines mean pairwise "
            "distance, similarity distribution entropy, and coverage spread."
        )


# ---------------------------------------------------------------------------
# Compression Ratio Diversity (CRD) — FIXED, consistent implementation
# ---------------------------------------------------------------------------

import zlib


class CompressionRatioDiversity(DiversityMetric):
    r"""Compression Ratio Diversity (CRD).

    Canonical definition:

    .. math::
        \text{CRD}(\mathcal{G}) = \frac{|\text{compress}(\text{concat}(\mathcal{G}))|}
                                       {\sum_{i} |\text{compress}(y_i)|}

    Measures cross-text redundancy via compression. When texts are repetitive,
    the concatenation compresses far below the sum of individual compressions
    (CRD ≈ 1/n → low). When texts are diverse, each adds unique information
    and CRD ≈ 1 (high).

    This is the SINGLE canonical implementation. All experiments should
    use this class rather than inline reimplementations.
    """

    def __init__(self, compression_level: int = 9):
        self._level = compression_level

    def compute(self, generation_set: List[str]) -> float:
        """Compute CRD: compress(concat) / Σ compress(yi)."""
        self.validate_input(generation_set)
        sum_individual = sum(
            len(zlib.compress(t.encode("utf-8"), self._level))
            for t in generation_set
        )
        if sum_individual == 0:
            return 0.0
        concatenated = "\n".join(generation_set).encode("utf-8")
        concat_compressed = len(zlib.compress(concatenated, self._level))
        ratio = concat_compressed / sum_individual
        return float(np.clip(ratio, 0.0, 1.0))

    @property
    def name(self) -> str:
        return "CRD"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Compression Ratio Diversity: |compress(concat(G))| / Σ|compress(yi)|. "
            "Higher means more diverse (less cross-text redundancy)."
        )


# ---------------------------------------------------------------------------
# Convenience: NeuralDiversitySuite
# ---------------------------------------------------------------------------

class NeuralDiversitySuite:
    """Bundles all neural diversity metrics for batch computation.

    Args:
        backend: Embedding backend ("tfidf" or "transformer").
        model_name: Model name for transformer backend.
    """

    def __init__(
        self,
        backend: str = "tfidf",
        model_name: Optional[str] = None,
    ):
        self.mauve = MAUVE(backend=backend, model_name=model_name)
        self.bertscore_div = BERTScoreDiversity(backend=backend, model_name=model_name)
        self.sts_div = STSDiversity(backend=backend, model_name=model_name)
        self.crd = CompressionRatioDiversity()

    def compute_all(self, generation_set: List[str]) -> Dict[str, float]:
        """Compute all neural diversity metrics."""
        results = {}
        for metric in [self.mauve, self.bertscore_div, self.sts_div, self.crd]:
            try:
                results[metric.name] = metric.compute(generation_set)
            except Exception as e:
                logger.warning("Failed to compute %s: %s", metric.name, e)
                results[metric.name] = float("nan")
        return results

    def compute_all_detailed(
        self, generation_set: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute all metrics with detailed diagnostics."""
        results = {}
        for metric in [self.mauve, self.bertscore_div, self.sts_div, self.crd]:
            try:
                if hasattr(metric, "compute_detailed"):
                    results[metric.name] = metric.compute_detailed(generation_set)
                else:
                    results[metric.name] = {"value": metric.compute(generation_set)}
            except Exception as e:
                logger.warning("Failed to compute %s: %s", metric.name, e)
                results[metric.name] = {"error": str(e)}
        return results
