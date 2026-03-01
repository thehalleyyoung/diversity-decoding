"""
Real-time streaming diversity selection.

Maintains a diverse subset as responses arrive one at a time, supporting
both synchronous and async generator interfaces. Uses reservoir sampling
with diversity constraints and multiple distance backends.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import logging
import math
import random
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
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
# Text utilities (self-contained, no external NLP deps)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \\1 ", text)
    return [t for t in _WS_RE.split(text) if t]


def _ngrams(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> np.ndarray:
    tf = Counter(tokens)
    vocab = sorted(idf)
    vec = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        vec[i] = tf.get(w, 0) * idf.get(w, 0.0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Distance methods
# ---------------------------------------------------------------------------


class DistanceMethod:
    """Pluggable distance computation between two texts."""

    def distance(self, a: str, b: str) -> float:
        raise NotImplementedError


class NgramJaccardDistance(DistanceMethod):
    """1 - Jaccard similarity on character or word n-grams."""

    def __init__(self, n: int = 3, char_level: bool = False):
        self.n = n
        self.char_level = char_level

    def distance(self, a: str, b: str) -> float:
        if self.char_level:
            sa = set(a[i : i + self.n] for i in range(len(a) - self.n + 1))
            sb = set(b[i : i + self.n] for i in range(len(b) - self.n + 1))
        else:
            sa = _ngrams(_tokenize(a), self.n)
            sb = _ngrams(_tokenize(b), self.n)
        return 1.0 - _jaccard(sa, sb)


class EditDistance(DistanceMethod):
    """Normalised Levenshtein edit distance on tokens."""

    def distance(self, a: str, b: str) -> float:
        ta, tb = _tokenize(a), _tokenize(b)
        if not ta and not tb:
            return 0.0
        m, n = len(ta), len(tb)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n + 1):
                cost = 0 if ta[i - 1] == tb[j - 1] else 1
                dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
        return dp[n] / max(m, n) if max(m, n) > 0 else 0.0


class FlowDistance(DistanceMethod):
    """Combined structural and lexical distance (default)."""

    def __init__(self, ngram_weight: float = 0.6, length_weight: float = 0.15,
                 vocab_weight: float = 0.25):
        self.ngram_weight = ngram_weight
        self.length_weight = length_weight
        self.vocab_weight = vocab_weight
        self._ngram = NgramJaccardDistance(n=2)

    def distance(self, a: str, b: str) -> float:
        d_ng = self._ngram.distance(a, b)
        la, lb = len(_tokenize(a)), len(_tokenize(b))
        d_len = abs(la - lb) / max(la, lb, 1)
        va, vb = set(_tokenize(a)), set(_tokenize(b))
        d_vocab = 1.0 - _jaccard(va, vb)
        return (self.ngram_weight * d_ng +
                self.length_weight * d_len +
                self.vocab_weight * d_vocab)


_METHODS: Dict[str, Callable[[], DistanceMethod]] = {
    "flow": FlowDistance,
    "ngram": NgramJaccardDistance,
    "edit": EditDistance,
}


# ---------------------------------------------------------------------------
# DiversityStream — core streaming selector
# ---------------------------------------------------------------------------


@dataclass
class StreamStats:
    """Running statistics for the stream."""
    total_seen: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    min_diversity_at_accept: float = float("inf")
    max_diversity_at_accept: float = 0.0
    avg_diversity_at_accept: float = 0.0
    _diversity_sum: float = 0.0


class DiversityStream:
    """Maintain a diverse set of up to *k* items as responses stream in.

    Parameters
    ----------
    k : int
        Maximum number of items to keep.
    method : str
        Distance method name (``"flow"``, ``"ngram"``, ``"edit"``).
    min_distance : float
        Minimum distance from all existing items to accept a new item.
    replacement_strategy : str
        ``"none"`` — never replace once full.
        ``"weakest"`` — replace the item whose removal increases diversity most.
        ``"oldest"`` — replace the oldest item.
    seed : int or None
        Random seed for reservoir tie-breaking.
    """

    def __init__(
        self,
        k: int = 5,
        method: str = "flow",
        min_distance: float = 0.3,
        replacement_strategy: str = "weakest",
        seed: Optional[int] = None,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.min_distance = min_distance
        self.replacement_strategy = replacement_strategy
        self._dist_fn: DistanceMethod = _METHODS.get(method, FlowDistance)()
        self._selected: List[str] = []
        self._timestamps: List[float] = []
        self._distance_cache: Dict[Tuple[int, int], float] = {}
        self._rng = random.Random(seed)
        self.stats = StreamStats()

    # -- public API ---------------------------------------------------------

    def add(self, response: str) -> bool:
        """Add *response* if it is diverse enough. Returns ``True`` if accepted."""
        self.stats.total_seen += 1

        if not response or not response.strip():
            self.stats.total_rejected += 1
            return False

        # First item always accepted
        if not self._selected:
            self._accept(response)
            return True

        # Compute minimum distance to existing items
        min_dist = self._min_distance_to_set(response)

        if min_dist < self.min_distance and len(self._selected) >= self.k:
            self.stats.total_rejected += 1
            return False

        if len(self._selected) < self.k:
            if min_dist >= self.min_distance:
                self._accept(response)
                return True
            # Below threshold but set not full — accept with reduced priority
            self._accept(response)
            return True

        # Set is full — try replacement
        if self.replacement_strategy == "none":
            self.stats.total_rejected += 1
            return False

        idx = self._find_replacement_index(response)
        if idx is not None:
            self._replace(idx, response)
            return True

        self.stats.total_rejected += 1
        return False

    def current_diversity(self) -> float:
        """Return average pairwise distance among selected items (0–1)."""
        n = len(self._selected)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += self._pairwise(i, j)
                count += 1
        return total / count

    def min_pairwise_diversity(self) -> float:
        """Return minimum pairwise distance (bottleneck diversity)."""
        n = len(self._selected)
        if n < 2:
            return 0.0
        best = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                best = min(best, self._pairwise(i, j))
        return best

    def get_selected(self) -> List[str]:
        """Return a copy of the currently selected diverse set."""
        return list(self._selected)

    def get_stats(self) -> StreamStats:
        return self.stats

    def reset(self) -> None:
        """Clear all state."""
        self._selected.clear()
        self._timestamps.clear()
        self._distance_cache.clear()
        self.stats = StreamStats()

    # -- async interface ----------------------------------------------------

    async def add_async(self, response: str) -> bool:
        """Async wrapper around :meth:`add` (non-blocking friendly)."""
        return self.add(response)

    async def consume(self, aiter: AsyncIterator[str]) -> List[str]:
        """Consume an async iterator of responses and return the diverse set.

        Example::

            async def generate():
                for chunk in llm.stream(prompt):
                    yield chunk

            selected = await stream.consume(generate())
        """
        async for response in aiter:
            self.add(response)
        return self.get_selected()

    # -- synchronous generator interface ------------------------------------

    def consume_sync(self, iterable) -> List[str]:
        """Consume a synchronous iterable of responses."""
        for response in iterable:
            self.add(response)
        return self.get_selected()

    def filter_iter(self, iterable) -> Iterator[str]:
        """Yield only accepted items from *iterable*."""
        for response in iterable:
            if self.add(response):
                yield response

    # -- private helpers ----------------------------------------------------

    def _accept(self, response: str) -> None:
        idx = len(self._selected)
        self._selected.append(response)
        self._timestamps.append(time.monotonic())
        div = self.current_diversity()
        self.stats.total_accepted += 1
        self.stats.min_diversity_at_accept = min(self.stats.min_diversity_at_accept, div)
        self.stats.max_diversity_at_accept = max(self.stats.max_diversity_at_accept, div)
        self.stats._diversity_sum += div
        self.stats.avg_diversity_at_accept = (
            self.stats._diversity_sum / self.stats.total_accepted
        )

    def _replace(self, idx: int, response: str) -> None:
        # Invalidate cache entries involving idx
        to_remove = [k for k in self._distance_cache if idx in k]
        for k in to_remove:
            del self._distance_cache[k]
        self._selected[idx] = response
        self._timestamps[idx] = time.monotonic()
        self.stats.total_accepted += 1

    def _pairwise(self, i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in self._distance_cache:
            self._distance_cache[key] = self._dist_fn.distance(
                self._selected[i], self._selected[j]
            )
        return self._distance_cache[key]

    def _min_distance_to_set(self, response: str) -> float:
        return min(
            self._dist_fn.distance(response, s)
            for s in self._selected
        )

    def _find_replacement_index(self, response: str) -> Optional[int]:
        """Find the index to replace, or None if replacement wouldn't help."""
        if self.replacement_strategy == "oldest":
            return int(np.argmin(self._timestamps))

        # "weakest" — replace the element whose removal + new item maximises
        # minimum pairwise distance
        current_min = self.min_pairwise_diversity()
        best_idx = None
        best_min = current_min

        for candidate_idx in range(len(self._selected)):
            # Simulate replacement
            trial = list(self._selected)
            trial[candidate_idx] = response
            trial_min = float("inf")
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    d = self._dist_fn.distance(trial[i], trial[j])
                    trial_min = min(trial_min, d)
            if trial_min > best_min:
                best_min = trial_min
                best_idx = candidate_idx

        return best_idx


# ---------------------------------------------------------------------------
# DiversityReservoir — reservoir sampling with diversity
# ---------------------------------------------------------------------------


class DiversityReservoir:
    """Reservoir sampling that biases toward diverse items.

    Standard reservoir sampling gives each item equal probability of being
    in the final sample. This variant weights acceptance probability by the
    distance of the new item from the current reservoir contents.

    Parameters
    ----------
    k : int
        Reservoir size.
    diversity_bias : float
        0.0 = standard reservoir sampling, 1.0 = fully diversity-biased.
    method : str
        Distance method name.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        k: int = 10,
        diversity_bias: float = 0.7,
        method: str = "flow",
        seed: Optional[int] = None,
    ):
        self.k = k
        self.diversity_bias = max(0.0, min(1.0, diversity_bias))
        self._dist_fn: DistanceMethod = _METHODS.get(method, FlowDistance)()
        self._reservoir: List[str] = []
        self._count = 0
        self._rng = random.Random(seed)

    def add(self, item: str) -> bool:
        """Process one item. Returns True if it entered the reservoir."""
        self._count += 1
        if len(self._reservoir) < self.k:
            self._reservoir.append(item)
            return True

        # Standard reservoir probability
        p_reservoir = self.k / self._count

        # Diversity bonus
        if self._reservoir:
            min_dist = min(
                self._dist_fn.distance(item, r) for r in self._reservoir
            )
        else:
            min_dist = 1.0

        p_accept = (
            (1.0 - self.diversity_bias) * p_reservoir
            + self.diversity_bias * min_dist
        )
        p_accept = min(1.0, p_accept)

        if self._rng.random() < p_accept:
            replace_idx = self._rng.randint(0, self.k - 1)
            self._reservoir[replace_idx] = item
            return True
        return False

    def get_reservoir(self) -> List[str]:
        return list(self._reservoir)

    def current_diversity(self) -> float:
        n = len(self._reservoir)
        if n < 2:
            return 0.0
        total = sum(
            self._dist_fn.distance(self._reservoir[i], self._reservoir[j])
            for i in range(n) for j in range(i + 1, n)
        )
        return total / (n * (n - 1) / 2)

    @property
    def size(self) -> int:
        return len(self._reservoir)

    def reset(self) -> None:
        self._reservoir.clear()
        self._count = 0


# ---------------------------------------------------------------------------
# Convenience: async streaming helper
# ---------------------------------------------------------------------------


async def stream_diverse_responses(
    async_gen: AsyncIterator[str],
    k: int = 5,
    method: str = "flow",
    min_distance: float = 0.3,
) -> List[str]:
    """One-shot helper: consume an async generator, return *k* diverse items.

    Example::

        results = await stream_diverse_responses(
            my_llm_stream(prompt), k=5
        )
    """
    stream = DiversityStream(k=k, method=method, min_distance=min_distance)
    return await stream.consume(async_gen)


def stream_diverse_sync(
    iterable,
    k: int = 5,
    method: str = "flow",
    min_distance: float = 0.3,
) -> List[str]:
    """Synchronous version of :func:`stream_diverse_responses`."""
    stream = DiversityStream(k=k, method=method, min_distance=min_distance)
    return stream.consume_sync(iterable)
