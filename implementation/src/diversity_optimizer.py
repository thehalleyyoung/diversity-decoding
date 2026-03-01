"""
Diversity Optimizer — General-purpose diversity optimization for AI pipelines.

Provides tools to optimize diversity across datasets, prompts, model ensembles,
and generation outputs. Supports Pareto-optimal diversity-quality trade-offs,
constrained generation, and submodular maximization.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers (local, lightweight)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "was", "are", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "if", "then", "than", "that", "this", "these", "those", "as", "up",
    "out", "about", "into", "over", "after", "before", "between", "under",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [t for t in _WS_RE.split(text) if t and t not in _STOPWORDS and len(t) > 1]


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tfidf_matrix(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build a simple TF-IDF matrix from a list of texts."""
    docs = [_tokenize(t) for t in texts]
    vocab: Dict[str, int] = {}
    for doc in docs:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    vocab_list = sorted(vocab, key=vocab.get)  # type: ignore[arg-type]
    n_docs = len(docs)
    n_vocab = len(vocab)
    if n_vocab == 0:
        return np.zeros((n_docs, 1)), [""]
    tf = np.zeros((n_docs, n_vocab))
    for i, doc in enumerate(docs):
        for tok in doc:
            tf[i, vocab[tok]] += 1
    # normalize TF
    row_sums = tf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tf /= row_sums
    # IDF
    df = (tf > 0).sum(axis=0).astype(float)
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    return tf * idf, vocab_list


def _pairwise_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = vectors / norms
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiverseSubset:
    """Result of dataset diversity optimization."""
    indices: List[int]
    items: List[Any]
    diversity_score: float
    coverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator:
        return iter(self.items)


@dataclass
class EnsembleWeights:
    """Optimized ensemble weights for diverse model combination."""
    weights: np.ndarray
    model_names: List[str]
    diversity_score: float
    quality_score: float
    pareto_optimal: bool = False

    def weight_dict(self) -> Dict[str, float]:
        return dict(zip(self.model_names, self.weights.tolist()))


@dataclass
class ParetoPoint:
    """A single point on the Pareto front."""
    item: Any
    diversity: float
    quality: float
    index: int


@dataclass
class ParetoFront:
    """Pareto-optimal diversity-quality trade-off frontier."""
    points: List[ParetoPoint]
    dominated: List[ParetoPoint]

    @property
    def diversities(self) -> np.ndarray:
        return np.array([p.diversity for p in self.points])

    @property
    def qualities(self) -> np.ndarray:
        return np.array([p.quality for p in self.points])

    def best_tradeoff(self, alpha: float = 0.5) -> ParetoPoint:
        """Return the point maximizing alpha*diversity + (1-alpha)*quality."""
        scores = [alpha * p.diversity + (1 - alpha) * p.quality for p in self.points]
        return self.points[int(np.argmax(scores))]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[ParetoPoint]:
        return iter(self.points)


@dataclass
class OptimizationConstraint:
    """A constraint for diversity optimization."""
    name: str
    fn: Callable[[List[Any]], float]
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def satisfied(self, items: List[Any]) -> bool:
        val = self.fn(items)
        if self.min_value is not None and val < self.min_value:
            return False
        if self.max_value is not None and val > self.max_value:
            return False
        return True


# ---------------------------------------------------------------------------
# Core: DiversityOptimizer
# ---------------------------------------------------------------------------


class DiversityOptimizer:
    """
    General-purpose diversity optimizer for AI pipelines.

    Wraps a diversity objective and optional constraints, then provides
    greedy submodular maximization (with lazy evaluation) to select
    maximally diverse subsets under those constraints.

    Parameters
    ----------
    objective : callable(List[Any]) -> float
        Function that scores diversity of a set of items.  Higher = more diverse.
    constraints : list of OptimizationConstraint, optional
        Hard constraints that must be satisfied by the selected subset.
    """

    def __init__(
        self,
        objective: Callable[[List[Any]], float],
        constraints: Optional[List[OptimizationConstraint]] = None,
    ) -> None:
        self.objective = objective
        self.constraints = constraints or []
        self._rng = np.random.default_rng(42)

    # -- greedy submodular maximization with lazy evaluation -----------------

    def select(
        self,
        candidates: List[Any],
        k: int,
        *,
        lazy: bool = True,
    ) -> DiverseSubset:
        """
        Greedy submodular maximization to pick *k* items maximizing
        the diversity objective while satisfying constraints.
        """
        if k >= len(candidates):
            score = self.objective(candidates)
            return DiverseSubset(
                indices=list(range(len(candidates))),
                items=list(candidates),
                diversity_score=score,
                coverage=1.0,
            )

        selected_idx: List[int] = []
        remaining = set(range(len(candidates)))

        # lazy evaluation: maintain upper bounds on marginal gain
        upper = {i: float("inf") for i in remaining}

        for _ in range(k):
            best_idx = -1
            best_gain = -float("inf")
            current_items = [candidates[i] for i in selected_idx]
            current_score = self.objective(current_items) if current_items else 0.0

            # sort remaining by upper bound (descending) for lazy eval
            order = sorted(remaining, key=lambda i: -upper[i]) if lazy else list(remaining)

            for idx in order:
                if lazy and upper[idx] < best_gain:
                    break  # no remaining candidate can beat best_gain
                trial = current_items + [candidates[idx]]
                # check constraints
                if not all(c.satisfied(trial) for c in self.constraints):
                    upper[idx] = -float("inf")
                    continue
                gain = self.objective(trial) - current_score
                upper[idx] = gain
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx < 0:
                break
            selected_idx.append(best_idx)
            remaining.discard(best_idx)

        items = [candidates[i] for i in selected_idx]
        score = self.objective(items) if items else 0.0
        coverage = len(items) / k if k > 0 else 0.0
        return DiverseSubset(
            indices=selected_idx,
            items=items,
            diversity_score=score,
            coverage=coverage,
        )

    # -- random restarts for non-submodular objectives ----------------------

    def select_stochastic(
        self,
        candidates: List[Any],
        k: int,
        *,
        restarts: int = 10,
        sample_frac: float = 0.6,
    ) -> DiverseSubset:
        """
        Stochastic greedy with random restarts — useful when the objective
        is not perfectly submodular.
        """
        best: Optional[DiverseSubset] = None
        n = len(candidates)
        sample_size = max(k, int(n * sample_frac))

        for _ in range(restarts):
            idxs = self._rng.choice(n, size=min(sample_size, n), replace=False).tolist()
            sub = [candidates[i] for i in idxs]
            result = self.select(sub, k, lazy=False)
            # remap indices
            result.indices = [idxs[i] for i in result.indices]
            result.items = [candidates[i] for i in result.indices]
            result.diversity_score = self.objective(result.items) if result.items else 0.0
            if best is None or result.diversity_score > best.diversity_score:
                best = result

        assert best is not None
        return best


# ---------------------------------------------------------------------------
# Dataset diversity optimization
# ---------------------------------------------------------------------------


def _text_diversity_objective(items: List[str]) -> float:
    """Average pairwise cosine distance of TF-IDF vectors."""
    if len(items) < 2:
        return 0.0
    mat, _ = _tfidf_matrix(items)
    dist = _pairwise_distance_matrix(mat)
    n = len(items)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += dist[i, j]
            count += 1
    return total / count if count else 0.0


def _vector_diversity_objective(vectors: np.ndarray) -> Callable[[List[int]], float]:
    """Return objective fn that computes avg pairwise distance from precomputed vecs."""
    dist = _pairwise_distance_matrix(vectors)

    def _obj(indices: List[int]) -> float:
        if len(indices) < 2:
            return 0.0
        total = sum(dist[i, j] for i, j in combinations(indices, 2))
        return total / (len(indices) * (len(indices) - 1) / 2)

    return _obj


def optimize_dataset_diversity(
    dataset: Union[List[str], List[Dict[str, Any]]],
    target_size: int,
    *,
    text_key: Optional[str] = None,
    constraints: Optional[List[OptimizationConstraint]] = None,
    restarts: int = 5,
) -> DiverseSubset:
    """
    Select a maximally diverse subset of *target_size* from *dataset*.

    Parameters
    ----------
    dataset : list of str or list of dict
        If dicts, *text_key* specifies which field contains text.
    target_size : int
        Number of items to select.
    text_key : str, optional
        Key for text field when dataset contains dicts.
    constraints : list of OptimizationConstraint, optional
    restarts : int
        Stochastic restarts for robustness.
    """
    if isinstance(dataset[0], dict):
        key = text_key or "text"
        texts = [d[key] for d in dataset]  # type: ignore[index]
    else:
        texts = list(dataset)  # type: ignore[arg-type]

    mat, _ = _tfidf_matrix(texts)
    dist = _pairwise_distance_matrix(mat)

    # objective works on indices into dataset
    def obj(items: List[Any]) -> float:
        idxs = [texts.index(it) if isinstance(it, str) else it for it in items]
        if len(idxs) < 2:
            return 0.0
        total = sum(dist[i, j] for i, j in combinations(idxs, 2))
        return total / (len(idxs) * (len(idxs) - 1) / 2)

    # wrap items as text for the optimizer
    optimizer = DiversityOptimizer(
        objective=lambda items: _text_diversity_objective(items),
        constraints=constraints,
    )
    return optimizer.select_stochastic(texts, target_size, restarts=restarts)


# ---------------------------------------------------------------------------
# Prompt diversity optimization
# ---------------------------------------------------------------------------


def optimize_prompt_diversity(
    prompts: List[str],
    k: int,
    *,
    penalize_length_homogeneity: bool = True,
) -> List[str]:
    """
    Select *k* prompts from *prompts* that are maximally diverse in content
    and, optionally, in length distribution.

    Returns the selected prompts in order of selection.
    """

    def obj(items: List[str]) -> float:
        if len(items) < 2:
            return 0.0
        base = _text_diversity_objective(items)
        if penalize_length_homogeneity:
            lengths = np.array([len(p) for p in items], dtype=float)
            if lengths.std() < 1e-6:
                length_bonus = 0.0
            else:
                # normalized coefficient of variation as bonus
                length_bonus = 0.1 * min(lengths.std() / (lengths.mean() + 1e-9), 1.0)
            base += length_bonus
        return base

    optimizer = DiversityOptimizer(objective=obj)
    result = optimizer.select(prompts, k)
    return result.items


# ---------------------------------------------------------------------------
# Model ensemble diversity
# ---------------------------------------------------------------------------


def optimize_model_ensemble(
    models: List[str],
    dataset: List[Dict[str, Any]],
    *,
    prediction_key: str = "predictions",
    quality_fn: Optional[Callable[[List[str]], float]] = None,
    n_iter: int = 200,
    seed: int = 42,
) -> EnsembleWeights:
    """
    Find ensemble weights that maximize diversity of combined predictions.

    Each entry in *dataset* should be a dict with a *prediction_key* mapping
    to a list of predictions (one per model, same order as *models*).

    Parameters
    ----------
    models : list of model name strings
    dataset : list of dicts with predictions per model
    prediction_key : key in dicts for the predictions list
    quality_fn : optional callable scoring quality of a list of predictions
    n_iter : number of Dirichlet-sample iterations
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    n_models = len(models)
    if n_models == 0:
        raise ValueError("Need at least one model")

    # gather predictions matrix: (n_samples, n_models)
    pred_matrix: List[List[str]] = []
    for entry in dataset:
        preds = entry[prediction_key]
        if len(preds) != n_models:
            raise ValueError(
                f"Expected {n_models} predictions, got {len(preds)}"
            )
        pred_matrix.append(list(preds))

    best_weights = np.ones(n_models) / n_models
    best_div = -1.0
    best_qual = 0.0

    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n_models))
        # select predictions by weighted sampling
        chosen: List[str] = []
        for preds in pred_matrix:
            idx = rng.choice(n_models, p=w)
            chosen.append(preds[idx])
        div = _text_diversity_objective(chosen)
        qual = quality_fn(chosen) if quality_fn else 0.0
        score = div + 0.3 * qual
        if score > best_div + 0.3 * best_qual:
            best_div = div
            best_qual = qual
            best_weights = w.copy()

    return EnsembleWeights(
        weights=best_weights,
        model_names=list(models),
        diversity_score=best_div,
        quality_score=best_qual,
        pareto_optimal=False,
    )


# ---------------------------------------------------------------------------
# Diversity-constrained generation
# ---------------------------------------------------------------------------


def diversity_constrained_generation(
    gen_fn: Callable[[], str],
    n: int = 10,
    min_diversity: float = 0.5,
    max_attempts: int = 200,
) -> List[str]:
    """
    Generate *n* outputs from *gen_fn* such that the set has at least
    *min_diversity* (average pairwise cosine distance of TF-IDF vectors).

    Uses rejection sampling: generates candidates and greedily adds them
    only if they increase the set diversity above the running threshold.
    """
    results: List[str] = []
    attempts = 0

    while len(results) < n and attempts < max_attempts:
        candidate = gen_fn()
        attempts += 1

        if not results:
            results.append(candidate)
            continue

        trial = results + [candidate]
        div = _text_diversity_objective(trial)

        if len(results) == 1:
            # accept second item if it has any distance
            if div > 0.01:
                results.append(candidate)
            continue

        current_div = _text_diversity_objective(results)

        # accept if diversity stays above threshold or improves
        if div >= min(min_diversity, current_div * 0.95):
            results.append(candidate)

    logger.info(
        "diversity_constrained_generation: %d items in %d attempts, diversity=%.3f",
        len(results),
        attempts,
        _text_diversity_objective(results) if len(results) >= 2 else 0.0,
    )
    return results


# ---------------------------------------------------------------------------
# Pareto diversity-quality frontier
# ---------------------------------------------------------------------------


def pareto_diversity_quality(
    responses: List[Any],
    quality_fn: Callable[[Any], float],
    *,
    diversity_fn: Optional[Callable[[Any, List[Any]], float]] = None,
) -> ParetoFront:
    """
    Compute the Pareto front of diversity vs. quality for *responses*.

    Parameters
    ----------
    responses : list
        Items to evaluate (typically strings).
    quality_fn : callable(item) -> float
        Scores quality of a single item.
    diversity_fn : callable(item, others) -> float, optional
        Scores how diverse an item is relative to others.
        Defaults to average TF-IDF cosine distance to all other items.
    """
    n = len(responses)
    if n == 0:
        return ParetoFront(points=[], dominated=[])

    # compute quality scores
    qualities = [quality_fn(r) for r in responses]

    # compute diversity scores
    if diversity_fn is not None:
        diversities = [diversity_fn(responses[i], responses[:i] + responses[i + 1 :]) for i in range(n)]
    else:
        # default: avg cosine distance to all others
        texts = [str(r) for r in responses]
        if n < 2:
            diversities = [0.0] * n
        else:
            mat, _ = _tfidf_matrix(texts)
            dist = _pairwise_distance_matrix(mat)
            diversities = [float(np.mean(dist[i])) for i in range(n)]

    # normalize to [0, 1]
    def _normalize(vals: List[float]) -> List[float]:
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-12:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    norm_d = _normalize(diversities)
    norm_q = _normalize(qualities)

    points = [
        ParetoPoint(item=responses[i], diversity=norm_d[i], quality=norm_q[i], index=i)
        for i in range(n)
    ]

    # identify Pareto-optimal points (non-dominated)
    pareto: List[ParetoPoint] = []
    dominated: List[ParetoPoint] = []

    for i, p in enumerate(points):
        is_dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if q.diversity >= p.diversity and q.quality >= p.quality and (
                q.diversity > p.diversity or q.quality > p.quality
            ):
                is_dominated = True
                break
        if is_dominated:
            dominated.append(p)
        else:
            pareto.append(p)

    # sort Pareto front by diversity ascending
    pareto.sort(key=lambda p: p.diversity)

    return ParetoFront(points=pareto, dominated=dominated)


# ---------------------------------------------------------------------------
# Facility-location diversity (submodular)
# ---------------------------------------------------------------------------


def facility_location_diversity(
    items: List[str],
    k: int,
    *,
    penalty: float = 0.0,
) -> DiverseSubset:
    """
    Select *k* items via the facility-location objective (a classical
    submodular function). Each unselected item is "covered" by its nearest
    selected item; the objective is the sum of max-similarities.

    Higher penalty encourages more spread-out selections.
    """
    mat, _ = _tfidf_matrix(items)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = mat / norms
    sim = normed @ normed.T

    n = len(items)
    selected: List[int] = []
    remaining = set(range(n))
    # track max similarity of each item to any selected item
    max_sim = np.full(n, -np.inf)

    for _ in range(min(k, n)):
        best_idx = -1
        best_gain = -np.inf
        for idx in remaining:
            # marginal gain: sum of improvements in max_sim
            gain = 0.0
            for j in range(n):
                new_sim = sim[j, idx]
                if new_sim > max_sim[j]:
                    gain += new_sim - max(max_sim[j], 0.0)
            gain -= penalty
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
        for j in range(n):
            max_sim[j] = max(max_sim[j], sim[j, best_idx])

    sel_items = [items[i] for i in selected]
    div = _text_diversity_objective(sel_items) if len(sel_items) >= 2 else 0.0
    coverage = float(np.mean(max_sim[max_sim > -np.inf])) if np.any(max_sim > -np.inf) else 0.0

    return DiverseSubset(
        indices=selected,
        items=sel_items,
        diversity_score=div,
        coverage=coverage,
        metadata={"method": "facility_location", "penalty": penalty},
    )


# ---------------------------------------------------------------------------
# DPP-based diversity selection
# ---------------------------------------------------------------------------


def dpp_diversity_select(
    items: List[str],
    k: int,
    *,
    quality_scores: Optional[List[float]] = None,
) -> DiverseSubset:
    """
    Greedy k-DPP selection. Uses TF-IDF similarity as the kernel and
    optionally incorporates per-item quality scores.

    A Determinantal Point Process naturally balances diversity and quality.
    """
    n = len(items)
    mat, _ = _tfidf_matrix(items)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = mat / norms

    # quality-weighted features
    if quality_scores is not None:
        qs = np.array(quality_scores)
        qs = qs / (qs.max() + 1e-12)
        normed = normed * qs[:, None]

    # L-ensemble kernel
    L = normed @ normed.T + 1e-6 * np.eye(n)

    selected: List[int] = []
    remaining = list(range(n))

    for _ in range(min(k, n)):
        best_idx = -1
        best_log_det = -np.inf
        for idx in remaining:
            trial = selected + [idx]
            sub_L = L[np.ix_(trial, trial)]
            sign, log_det = np.linalg.slogdet(sub_L)
            if sign > 0 and log_det > best_log_det:
                best_log_det = log_det
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    sel_items = [items[i] for i in selected]
    div = _text_diversity_objective(sel_items) if len(sel_items) >= 2 else 0.0

    return DiverseSubset(
        indices=selected,
        items=sel_items,
        diversity_score=div,
        coverage=len(selected) / k if k > 0 else 0.0,
        metadata={"method": "greedy_dpp"},
    )


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) selection
# ---------------------------------------------------------------------------


def mmr_select(
    items: List[str],
    query: str,
    k: int,
    *,
    lambda_: float = 0.5,
) -> DiverseSubset:
    """
    Maximal Marginal Relevance selection: balances relevance to *query*
    with diversity among selected items.

    lambda_ = 1 → pure relevance, lambda_ = 0 → pure diversity.
    """
    all_texts = [query] + items
    mat, _ = _tfidf_matrix(all_texts)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = mat / norms
    sim = normed @ normed.T

    query_sim = sim[0, 1:]  # similarity of each item to query
    item_sim = sim[1:, 1:]  # inter-item similarity

    n = len(items)
    selected: List[int] = []
    remaining = set(range(n))

    for _ in range(min(k, n)):
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            rel = query_sim[idx]
            if selected:
                max_sim_sel = max(item_sim[idx, s] for s in selected)
            else:
                max_sim_sel = 0.0
            score = lambda_ * rel - (1 - lambda_) * max_sim_sel
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    sel_items = [items[i] for i in selected]
    div = _text_diversity_objective(sel_items) if len(sel_items) >= 2 else 0.0

    return DiverseSubset(
        indices=selected,
        items=sel_items,
        diversity_score=div,
        coverage=len(selected) / k if k > 0 else 0.0,
        metadata={"method": "mmr", "lambda": lambda_},
    )
