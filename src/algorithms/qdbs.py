"""
Quality-Diversity Beam Search (QD-BS)
=====================================

A novel decoding algorithm that integrates MAP-Elites-style quality-diversity
archive management into beam search.  At every expansion step candidates are
scored for *quality* (e.g. log-probability) and mapped into a discretised
*behaviour space* via a set of behaviour descriptors computed over partial
sequences.  The archive retains the single highest-quality candidate per
behaviour cell, ensuring that the final output set is both high-quality *and*
behaviourally diverse.

Archive update rule (per cell *c* in tessellation T):

    A[c] = argmax_{y in {A[c]} ∪ {y' : β(y') ∈ c}} score(y)

where β(·) is the composite behaviour descriptor and score(·) is the quality
metric.

Key components
--------------
* ``QDBSConfig``         – algorithm hyper-parameters
* ``QualityDiversityBeamSearch`` – main algorithm class
* ``Archive``            – MAP-Elites archive with tessellation back-end
* ``Tessellation``       – abstract cell layout (grid / Voronoi / CVT)
* ``BehaviorDescriptor`` – per-sequence feature extractors
* ``QDBSState``          – extended decoding state
* ``Candidate``          – single beam candidate
* ``QDBSAnalyzer``       – post-hoc analysis utilities
"""

from __future__ import annotations

import abc
import copy
import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence as SequenceT,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
from scipy.spatial import KDTree

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    _log_softmax,
    _stable_softmax,
    _top_k_filter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
_NEG_INF = float("-inf")
_DEFAULT_BEHAVIOR_NAMES: List[str] = [
    "length",
    "pos_diversity",
    "lexical_diversity",
]

# ---------------------------------------------------------------------------
# Heuristic POS-tag approximation tables
# ---------------------------------------------------------------------------
# We approximate POS categories via simple suffix/prefix rules so that the
# algorithm is self-contained (no external NLP library required).

_SUFFIX_TO_POS: List[Tuple[str, str]] = [
    ("ing", "VBG"),
    ("tion", "NN"),
    ("sion", "NN"),
    ("ment", "NN"),
    ("ness", "NN"),
    ("ity", "NN"),
    ("ous", "JJ"),
    ("ive", "JJ"),
    ("ful", "JJ"),
    ("less", "JJ"),
    ("able", "JJ"),
    ("ible", "JJ"),
    ("ly", "RB"),
    ("ed", "VBD"),
    ("er", "NN"),
    ("est", "JJS"),
    ("al", "JJ"),
    ("ent", "JJ"),
    ("ant", "JJ"),
    ("ise", "VB"),
    ("ize", "VB"),
    ("ate", "VB"),
    ("fy", "VB"),
    ("en", "VB"),
    ("'s", "POS"),
    ("s", "NNS"),
]

_FUNCTION_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must", "need", "dare", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what", "where", "when", "how", "why",
    "not", "no", "nor", "neither", "either",
    "if", "then", "else", "than", "so", "as", "because", "since", "while",
    "although", "though", "unless", "until", "before", "after",
}


def _heuristic_pos(word: str) -> str:
    """Return an approximate POS tag for *word* using suffix heuristics."""
    lower = word.lower().strip()
    if not lower:
        return "X"
    if lower in _FUNCTION_WORDS:
        return "FW"
    if lower.isdigit() or lower.replace(".", "", 1).isdigit():
        return "CD"
    for suffix, tag in _SUFFIX_TO_POS:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            return tag
    return "NN"  # default noun


# =========================================================================
# Candidate
# =========================================================================


@dataclass
class Candidate:
    """A single beam candidate during QD-BS expansion."""

    sequence: List[int]
    score: float
    behavior: Optional[np.ndarray] = None
    cell_index: Optional[int] = None
    parent_index: int = -1
    token_added: int = -1

    # -- convenience -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequence)

    def clone(self) -> "Candidate":
        return Candidate(
            sequence=list(self.sequence),
            score=self.score,
            behavior=self.behavior.copy() if self.behavior is not None else None,
            cell_index=self.cell_index,
            parent_index=self.parent_index,
            token_added=self.token_added,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "score": self.score,
            "behavior": self.behavior.tolist() if self.behavior is not None else None,
            "cell_index": self.cell_index,
            "parent_index": self.parent_index,
            "token_added": self.token_added,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candidate":
        beh = np.asarray(d["behavior"]) if d.get("behavior") is not None else None
        return cls(
            sequence=d["sequence"],
            score=d["score"],
            behavior=beh,
            cell_index=d.get("cell_index"),
            parent_index=d.get("parent_index", -1),
            token_added=d.get("token_added", -1),
        )


# =========================================================================
# ArchiveEntry
# =========================================================================


@dataclass
class ArchiveEntry:
    """A single entry stored inside the MAP-Elites archive."""

    sequence: List[int]
    quality: float
    behavior: np.ndarray
    cell_index: int
    step_added: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "quality": self.quality,
            "behavior": self.behavior.tolist(),
            "cell_index": self.cell_index,
            "step_added": self.step_added,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchiveEntry":
        return cls(
            sequence=d["sequence"],
            quality=d["quality"],
            behavior=np.asarray(d["behavior"], dtype=np.float64),
            cell_index=d["cell_index"],
            step_added=d.get("step_added", 0),
            metadata=d.get("metadata", {}),
        )


# =========================================================================
# Tessellation – abstract + concrete implementations
# =========================================================================


class Tessellation(abc.ABC):
    """Abstract tessellation of the behaviour space into discrete cells."""

    @abc.abstractmethod
    def map_to_cell(self, descriptor: np.ndarray) -> int:
        """Map a behaviour descriptor vector to its cell index."""
        ...

    @abc.abstractmethod
    def cell_center(self, cell_index: int) -> np.ndarray:
        """Return the centroid / representative point of *cell_index*."""
        ...

    @property
    @abc.abstractmethod
    def num_cells(self) -> int:
        """Total number of cells in the tessellation."""
        ...

    @abc.abstractmethod
    def cell_bounds(self, cell_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds of the cell.

        For Voronoi / CVT tessellations this returns the bounding box of the
        cell's Voronoi region (approximated).
        """
        ...

    def nearest_cell(self, point: np.ndarray) -> int:
        """Return the index of the cell whose centre is closest to *point*.

        Default implementation delegates to :meth:`map_to_cell`.
        """
        return self.map_to_cell(point)

    # -- serialisation helpers ---------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise tessellation parameters (override in subclasses)."""
        return {"type": self.__class__.__name__, "num_cells": self.num_cells}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Tessellation":
        """Reconstruct a tessellation from a dictionary."""
        ttype = d.get("type", "GridTessellation")
        if ttype == "GridTessellation":
            bounds = np.asarray(d["bounds"], dtype=np.float64)
            return GridTessellation(bounds, d["resolution"])
        elif ttype == "VoronoiTessellation":
            centroids = np.asarray(d["centroids"], dtype=np.float64)
            return VoronoiTessellation(centroids)
        elif ttype == "CVTTessellation":
            bounds = np.asarray(d["bounds"], dtype=np.float64)
            return CVTTessellation(bounds, d["num_cells_requested"], d.get("n_iterations", 50))
        raise ValueError(f"Unknown tessellation type: {ttype}")


# -------------------------------------------------------------------------
# GridTessellation
# -------------------------------------------------------------------------


class GridTessellation(Tessellation):
    """Uniform grid tessellation of a hyper-rectangular behaviour space.

    Parameters
    ----------
    bounds : np.ndarray
        Shape ``(n_dims, 2)`` — per-dimension ``[low, high]``.
    resolution : int
        Number of bins per dimension.
    """

    def __init__(self, bounds: np.ndarray, resolution: int) -> None:
        self._bounds = np.asarray(bounds, dtype=np.float64)
        if self._bounds.ndim != 2 or self._bounds.shape[1] != 2:
            raise ValueError("bounds must be shape (n_dims, 2)")
        self._resolution = max(1, resolution)
        self._n_dims = self._bounds.shape[0]
        self._cell_widths = (self._bounds[:, 1] - self._bounds[:, 0]) / self._resolution
        # Avoid zero-width dimensions
        self._cell_widths = np.where(self._cell_widths < _EPS, 1.0, self._cell_widths)
        self._total_cells = self._resolution ** self._n_dims

        # Pre-compute strides for fast flat-index conversion
        self._strides = np.ones(self._n_dims, dtype=np.int64)
        for d in range(self._n_dims - 2, -1, -1):
            self._strides[d] = self._strides[d + 1] * self._resolution

    # -- Tessellation interface --------------------------------------------

    def map_to_cell(self, descriptor: np.ndarray) -> int:
        descriptor = np.asarray(descriptor, dtype=np.float64).ravel()
        indices = ((descriptor - self._bounds[:, 0]) / self._cell_widths).astype(np.int64)
        indices = np.clip(indices, 0, self._resolution - 1)
        return int(self._multi_index_to_flat(indices))

    def cell_center(self, cell_index: int) -> np.ndarray:
        mi = self._flat_to_multi_index(cell_index)
        return self._bounds[:, 0] + (mi + 0.5) * self._cell_widths

    @property
    def num_cells(self) -> int:
        return self._total_cells

    def cell_bounds(self, cell_index: int) -> Tuple[np.ndarray, np.ndarray]:
        mi = self._flat_to_multi_index(cell_index)
        lower = self._bounds[:, 0] + mi * self._cell_widths
        upper = lower + self._cell_widths
        return lower, upper

    # -- internal helpers --------------------------------------------------

    def _multi_index_to_flat(self, indices: np.ndarray) -> int:
        return int(np.sum(indices * self._strides))

    def _flat_to_multi_index(self, flat: int) -> np.ndarray:
        result = np.zeros(self._n_dims, dtype=np.int64)
        remainder = flat
        for d in range(self._n_dims):
            result[d] = remainder // self._strides[d]
            remainder %= self._strides[d]
        return result

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "GridTessellation",
            "bounds": self._bounds.tolist(),
            "resolution": self._resolution,
            "num_cells": self._total_cells,
        }


# -------------------------------------------------------------------------
# VoronoiTessellation
# -------------------------------------------------------------------------


class VoronoiTessellation(Tessellation):
    """Voronoi tessellation backed by a KD-tree for fast nearest-centroid lookup.

    Parameters
    ----------
    centroids : np.ndarray
        Shape ``(n_cells, n_dims)`` — centroid coordinates.
    """

    def __init__(self, centroids: np.ndarray) -> None:
        self._centroids = np.asarray(centroids, dtype=np.float64)
        if self._centroids.ndim != 2:
            raise ValueError("centroids must be 2-D (n_cells, n_dims)")
        self._n_cells, self._n_dims = self._centroids.shape
        self._tree = KDTree(self._centroids)

    # -- Tessellation interface --------------------------------------------

    def map_to_cell(self, descriptor: np.ndarray) -> int:
        descriptor = np.asarray(descriptor, dtype=np.float64).ravel()
        _, idx = self._tree.query(descriptor)
        return int(idx)

    def cell_center(self, cell_index: int) -> np.ndarray:
        return self._centroids[cell_index].copy()

    @property
    def num_cells(self) -> int:
        return self._n_cells

    def cell_bounds(self, cell_index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Approximate: return the mid-point between this centroid and its
        # nearest neighbours as a bounding box.
        centre = self._centroids[cell_index]
        dists, idxs = self._tree.query(centre, k=min(self._n_cells, 3))
        if np.isscalar(dists):
            half = np.ones(self._n_dims) * 0.5
        else:
            if len(dists) > 1:
                half = np.full(self._n_dims, dists[1] / 2.0)
            else:
                half = np.ones(self._n_dims) * 0.5
        return centre - half, centre + half

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "VoronoiTessellation",
            "centroids": self._centroids.tolist(),
            "num_cells": self._n_cells,
        }


# -------------------------------------------------------------------------
# CVTTessellation (Centroidal Voronoi Tessellation)
# -------------------------------------------------------------------------


class CVTTessellation(Tessellation):
    """Centroidal Voronoi Tessellation computed via Lloyd's algorithm.

    Produces a near-uniform partitioning of the behaviour space by
    iteratively adjusting Voronoi centroids to be the centroids of their
    own regions (approximated by Monte-Carlo sampling).

    Parameters
    ----------
    bounds : np.ndarray
        Shape ``(n_dims, 2)`` — per-dimension ``[low, high]``.
    num_cells : int
        Desired number of cells.
    n_iterations : int
        Number of Lloyd iterations.
    n_samples_per_iter : int
        Monte-Carlo samples per iteration for centroid estimation.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        num_cells: int,
        n_iterations: int = 50,
        n_samples_per_iter: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        self._bounds = np.asarray(bounds, dtype=np.float64)
        if self._bounds.ndim != 2 or self._bounds.shape[1] != 2:
            raise ValueError("bounds must be shape (n_dims, 2)")
        self._n_dims = self._bounds.shape[0]
        self._num_cells_requested = num_cells
        self._n_iterations = n_iterations

        rng = np.random.default_rng(seed)

        # Initialise centroids uniformly
        lows = self._bounds[:, 0]
        highs = self._bounds[:, 1]
        centroids = rng.uniform(lows, highs, size=(num_cells, self._n_dims))

        for _it in range(n_iterations):
            centroids = self._lloyd_iteration(
                centroids, lows, highs, n_samples_per_iter, rng
            )

        self._centroids = centroids
        self._n_cells = centroids.shape[0]
        self._tree = KDTree(self._centroids)

    # -- Lloyd iteration ---------------------------------------------------

    @staticmethod
    def _lloyd_iteration(
        centroids: np.ndarray,
        lows: np.ndarray,
        highs: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """One step of Lloyd's algorithm: sample → assign → recentre."""
        n_dims = centroids.shape[1]
        n_cells = centroids.shape[0]
        samples = rng.uniform(lows, highs, size=(n_samples, n_dims))
        tree = KDTree(centroids)
        _, assignments = tree.query(samples)

        new_centroids = centroids.copy()
        for c in range(n_cells):
            members = samples[assignments == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
        return new_centroids

    # -- Tessellation interface --------------------------------------------

    def map_to_cell(self, descriptor: np.ndarray) -> int:
        descriptor = np.asarray(descriptor, dtype=np.float64).ravel()
        _, idx = self._tree.query(descriptor)
        return int(idx)

    def cell_center(self, cell_index: int) -> np.ndarray:
        return self._centroids[cell_index].copy()

    @property
    def num_cells(self) -> int:
        return self._n_cells

    def cell_bounds(self, cell_index: int) -> Tuple[np.ndarray, np.ndarray]:
        centre = self._centroids[cell_index]
        dists, _ = self._tree.query(centre, k=min(self._n_cells, 3))
        if np.isscalar(dists):
            half = np.ones(self._n_dims) * 0.5
        else:
            half = np.full(self._n_dims, (dists[1] / 2.0) if len(dists) > 1 else 0.5)
        return centre - half, centre + half

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "CVTTessellation",
            "centroids": self._centroids.tolist(),
            "bounds": self._bounds.tolist(),
            "num_cells_requested": self._num_cells_requested,
            "num_cells": self._n_cells,
            "n_iterations": self._n_iterations,
        }


# =========================================================================
# BehaviorDescriptor – abstract + concrete implementations
# =========================================================================


class BehaviorDescriptor(abc.ABC):
    """Abstract base for a single scalar behaviour descriptor."""

    @abc.abstractmethod
    def compute(self, token_ids: List[int]) -> float:
        """Compute the descriptor value for a token-id sequence."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abc.abstractmethod
    def bounds(self) -> Tuple[float, float]:
        """Expected (min, max) range of this descriptor."""
        ...

    def __repr__(self) -> str:
        lo, hi = self.bounds
        return f"{self.__class__.__name__}(name={self.name!r}, bounds=[{lo:.2f}, {hi:.2f}])"


# -------------------------------------------------------------------------
# LengthDescriptor
# -------------------------------------------------------------------------


class LengthDescriptor(BehaviorDescriptor):
    """Normalised sequence length ∈ [0, 1]."""

    def __init__(self, max_length: int = 512) -> None:
        self._max_length = max(1, max_length)

    def compute(self, token_ids: List[int]) -> float:
        return min(len(token_ids) / self._max_length, 1.0)

    @property
    def name(self) -> str:
        return "length"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# -------------------------------------------------------------------------
# POSDiversityDescriptor
# -------------------------------------------------------------------------


class POSDiversityDescriptor(BehaviorDescriptor):
    """Entropy of the (heuristic) POS-tag distribution, normalised to [0, 1].

    Uses suffix-based heuristics so that no external NLP model is needed.
    The entropy is normalised by ``log(num_tag_types)`` to lie in [0, 1].
    """

    _ALL_TAGS: List[str] = sorted(
        {t for _, t in _SUFFIX_TO_POS} | {"FW", "CD", "X", "NN"}
    )

    def __init__(self, simple_tokenizer: Optional[Any] = None) -> None:
        self._tokenizer = simple_tokenizer  # reserved for future use
        self._max_entropy = math.log(len(self._ALL_TAGS) + 1)

    def compute(self, token_ids: List[int]) -> float:
        # Convert token ids to pseudo-words (just use str repr for heuristic)
        words = self._ids_to_words(token_ids)
        if not words:
            return 0.0
        tags = [_heuristic_pos(w) for w in words]
        return self._tag_entropy(tags)

    def _ids_to_words(self, token_ids: List[int]) -> List[str]:
        """Crude id→word mapping: treat each id as a 'word'.

        In practice a real tokenizer would decode ids; here we cluster ids
        into pseudo-words by modular hashing so that the POS heuristic
        still produces varied tags.
        """
        # Generate pseudo-words based on token id ranges
        words: List[str] = []
        for tid in token_ids:
            # Map token id to a deterministic pseudo-word
            bucket = tid % 97  # prime modulus for spread
            if bucket < 10:
                words.append(str(bucket))  # digit-like
            elif bucket < 20:
                words.append("the")  # function word
            elif bucket < 35:
                words.append("running")  # VBG
            elif bucket < 45:
                words.append("beautiful")  # JJ via suffix
            elif bucket < 55:
                words.append("quickly")  # RB
            elif bucket < 65:
                words.append("nation")  # NN via suffix
            elif bucket < 75:
                words.append("realize")  # VB via suffix
            elif bucket < 85:
                words.append("darkness")  # NN via suffix
            else:
                words.append("table")  # default NN
        return words

    def _tag_entropy(self, tags: List[str]) -> float:
        if not tags:
            return 0.0
        counts: Dict[str, int] = defaultdict(int)
        for t in tags:
            counts[t] += 1
        total = len(tags)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log(p)
        return min(entropy / max(self._max_entropy, _EPS), 1.0)

    @property
    def name(self) -> str:
        return "pos_diversity"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# -------------------------------------------------------------------------
# LexicalDiversityDescriptor
# -------------------------------------------------------------------------


class LexicalDiversityDescriptor(BehaviorDescriptor):
    """Type-token ratio (TTR) of the token-id sequence, in [0, 1]."""

    def compute(self, token_ids: List[int]) -> float:
        if not token_ids:
            return 0.0
        return len(set(token_ids)) / len(token_ids)

    @property
    def name(self) -> str:
        return "lexical_diversity"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# -------------------------------------------------------------------------
# NGramDiversityDescriptor
# -------------------------------------------------------------------------


class NGramDiversityDescriptor(BehaviorDescriptor):
    """Ratio of unique n-grams to total n-grams, in [0, 1].

    Parameters
    ----------
    n : int
        N-gram size (default 2 for bigrams).
    """

    def __init__(self, n: int = 2) -> None:
        self._n = max(1, n)

    def compute(self, token_ids: List[int]) -> float:
        if len(token_ids) < self._n:
            return 0.0
        ngrams = [
            tuple(token_ids[i : i + self._n])
            for i in range(len(token_ids) - self._n + 1)
        ]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    @property
    def name(self) -> str:
        return f"ngram_{self._n}_diversity"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# -------------------------------------------------------------------------
# VocabRichnessDescriptor
# -------------------------------------------------------------------------


class VocabRichnessDescriptor(BehaviorDescriptor):
    """Vocabulary richness via Yule's K measure, normalised to [0, 1].

    Yule's K is defined as:
        K = 10000 * (M2 - N) / N^2
    where N is the total number of tokens and M2 = sum_i (i^2 * V(i,N))
    with V(i,N) = number of types occurring exactly i times.

    We normalise by a heuristic upper bound so the value lies in [0, 1].
    """

    def __init__(self, max_k: float = 200.0) -> None:
        self._max_k = max_k

    def compute(self, token_ids: List[int]) -> float:
        if len(token_ids) < 2:
            return 0.0
        freq: Dict[int, int] = defaultdict(int)
        for t in token_ids:
            freq[t] += 1
        n = len(token_ids)
        # V(i, N): number of types appearing exactly i times
        spectrum: Dict[int, int] = defaultdict(int)
        for count in freq.values():
            spectrum[count] += 1
        m2 = sum(i * i * vi for i, vi in spectrum.items())
        k = 10000.0 * (m2 - n) / max(n * n, 1)
        return min(max(k, 0.0) / self._max_k, 1.0)

    @property
    def name(self) -> str:
        return "vocab_richness"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# -------------------------------------------------------------------------
# CompositeBehaviorDescriptor
# -------------------------------------------------------------------------


class CompositeBehaviorDescriptor:
    """Aggregates multiple :class:`BehaviorDescriptor` instances and
    returns a single behaviour vector.

    Parameters
    ----------
    descriptors : list of BehaviorDescriptor
    """

    def __init__(self, descriptors: List[BehaviorDescriptor]) -> None:
        if not descriptors:
            raise ValueError("At least one descriptor is required")
        self.descriptors = list(descriptors)

    @property
    def n_dims(self) -> int:
        return len(self.descriptors)

    @property
    def bounds(self) -> np.ndarray:
        """Shape ``(n_dims, 2)`` array of per-descriptor bounds."""
        return np.array([d.bounds for d in self.descriptors], dtype=np.float64)

    def compute_all(self, token_ids: List[int]) -> np.ndarray:
        """Compute all descriptor values and return as a vector."""
        return np.array(
            [d.compute(token_ids) for d in self.descriptors], dtype=np.float64
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalise raw descriptor values to [0, 1] per dimension."""
        values = np.asarray(values, dtype=np.float64)
        bds = self.bounds
        ranges = bds[:, 1] - bds[:, 0]
        ranges = np.where(ranges < _EPS, 1.0, ranges)
        return (values - bds[:, 0]) / ranges

    def descriptor_names(self) -> List[str]:
        return [d.name for d in self.descriptors]

    def __repr__(self) -> str:
        names = ", ".join(d.name for d in self.descriptors)
        return f"CompositeBehaviorDescriptor([{names}])"


def _build_descriptor(name: str, max_length: int = 512) -> BehaviorDescriptor:
    """Factory function: descriptor name → instance."""
    name_lower = name.lower().strip()
    if name_lower == "length":
        return LengthDescriptor(max_length=max_length)
    elif name_lower == "pos_diversity":
        return POSDiversityDescriptor()
    elif name_lower == "lexical_diversity":
        return LexicalDiversityDescriptor()
    elif name_lower.startswith("ngram"):
        # e.g. "ngram_3_diversity"
        parts = name_lower.split("_")
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2
        return NGramDiversityDescriptor(n=n)
    elif name_lower in ("vocab_richness", "yule"):
        return VocabRichnessDescriptor()
    else:
        raise ValueError(f"Unknown behavior descriptor: {name!r}")


# =========================================================================
# Archive – MAP-Elites archive
# =========================================================================


class Archive:
    """MAP-Elites-style archive mapping behaviour cells to elite entries.

    Parameters
    ----------
    tessellation : Tessellation
        Defines how behaviour vectors are discretised.
    max_cells : int
        Soft cap on archive size (only enforced by tessellation capacity).
    """

    def __init__(self, tessellation: Tessellation, max_cells: int = 0) -> None:
        self.tessellation = tessellation
        self._max_cells = max_cells if max_cells > 0 else tessellation.num_cells
        self.cells: Dict[int, ArchiveEntry] = {}
        self._insert_count: int = 0
        self._improvement_count: int = 0

    # -- core operations ---------------------------------------------------

    def add(
        self,
        candidate: Candidate,
        quality: float,
        behavior: np.ndarray,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attempt to insert a candidate into the archive.

        Returns ``True`` if the candidate was inserted (either into an empty
        cell or by displacing a lower-quality occupant).
        """
        cell_idx = self.tessellation.map_to_cell(behavior)
        entry = ArchiveEntry(
            sequence=list(candidate.sequence),
            quality=quality,
            behavior=behavior.copy(),
            cell_index=cell_idx,
            step_added=step,
            metadata=metadata or {},
        )
        existing = self.cells.get(cell_idx)
        if existing is None or quality > existing.quality:
            self.cells[cell_idx] = entry
            self._insert_count += 1
            if existing is not None:
                self._improvement_count += 1
            return True
        return False

    def get(self, cell_index: int) -> Optional[ArchiveEntry]:
        """Return the entry at *cell_index*, or ``None``."""
        return self.cells.get(cell_index)

    def best_per_cell(self) -> List[ArchiveEntry]:
        """Return all entries (one per occupied cell), sorted by quality."""
        entries = list(self.cells.values())
        entries.sort(key=lambda e: e.quality, reverse=True)
        return entries

    # -- coverage & quality metrics ----------------------------------------

    def coverage(self) -> float:
        """Fraction of cells that are occupied."""
        total = self.tessellation.num_cells
        return len(self.cells) / max(total, 1)

    def total_quality(self) -> float:
        """Sum of quality values across all occupied cells."""
        return sum(e.quality for e in self.cells.values())

    def mean_quality(self) -> float:
        """Mean quality over occupied cells."""
        if not self.cells:
            return 0.0
        return self.total_quality() / len(self.cells)

    def best_quality(self) -> float:
        """Maximum quality in the archive."""
        if not self.cells:
            return _NEG_INF
        return max(e.quality for e in self.cells.values())

    def as_sequences(self) -> List[TokenSequence]:
        """Extract token sequences from all entries, sorted by quality."""
        entries = self.best_per_cell()
        return [e.sequence for e in entries]

    def empty_cells(self) -> List[int]:
        """Indices of unoccupied cells."""
        all_cells = set(range(self.tessellation.num_cells))
        return sorted(all_cells - set(self.cells.keys()))

    def occupied_cells(self) -> List[int]:
        """Indices of occupied cells."""
        return sorted(self.cells.keys())

    def diversity_score(self) -> float:
        """Behavioural diversity: mean pairwise distance between entries.

        Returns 0.0 if fewer than two entries exist.
        """
        if len(self.cells) < 2:
            return 0.0
        behaviors = np.array([e.behavior for e in self.cells.values()])
        n = behaviors.shape[0]
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += float(np.linalg.norm(behaviors[i] - behaviors[j]))
                count += 1
        return total_dist / max(count, 1)

    def qd_score(self) -> float:
        """Quality-Diversity score: total_quality * coverage."""
        return self.total_quality() * self.coverage()

    def clear(self) -> None:
        """Remove all entries from the archive."""
        self.cells.clear()
        self._insert_count = 0
        self._improvement_count = 0

    def merge(self, other: "Archive") -> "Archive":
        """Merge *other* into a new archive, keeping the best per cell.

        Uses this archive's tessellation.
        """
        merged = Archive(self.tessellation, self._max_cells)
        # Insert from self first
        for entry in self.cells.values():
            cand = Candidate(sequence=entry.sequence, score=entry.quality)
            merged.add(cand, entry.quality, entry.behavior, entry.step_added, entry.metadata)
        # Then from other — only wins if quality is higher
        for entry in other.cells.values():
            cand = Candidate(sequence=entry.sequence, score=entry.quality)
            merged.add(cand, entry.quality, entry.behavior, entry.step_added, entry.metadata)
        return merged

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tessellation": self.tessellation.to_dict(),
            "max_cells": self._max_cells,
            "cells": {str(k): v.to_dict() for k, v in self.cells.items()},
            "insert_count": self._insert_count,
            "improvement_count": self._improvement_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Archive":
        tess = Tessellation.from_dict(data["tessellation"])
        archive = cls(tess, data.get("max_cells", 0))
        for k_str, v_dict in data.get("cells", {}).items():
            entry = ArchiveEntry.from_dict(v_dict)
            archive.cells[int(k_str)] = entry
        archive._insert_count = data.get("insert_count", 0)
        archive._improvement_count = data.get("improvement_count", 0)
        return archive

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Archive(occupied={len(self.cells)}/{self.tessellation.num_cells}, "
            f"coverage={self.coverage():.2%}, qd_score={self.qd_score():.4f})"
        )

    def __len__(self) -> int:
        return len(self.cells)

    def __contains__(self, cell_index: int) -> bool:
        return cell_index in self.cells


# =========================================================================
# QDBSConfig
# =========================================================================


@dataclass
class QDBSConfig(DecodingConfig):
    """Configuration for Quality-Diversity Beam Search.

    Extends :class:`DecodingConfig` with QD-BS-specific hyper-parameters.
    """

    algorithm_name: str = "QualityDiversityBeamSearch"

    # -- beam parameters ---------------------------------------------------
    beam_width: int = 50
    archive_size: int = 100

    # -- behaviour space ---------------------------------------------------
    num_behavior_dims: int = 3
    behavior_descriptors: List[str] = field(
        default_factory=lambda: list(_DEFAULT_BEHAVIOR_NAMES)
    )
    tessellation_type: str = "grid"       # grid | voronoi | cvt
    grid_resolution: int = 10

    # -- quality & exploration ---------------------------------------------
    quality_metric: str = "log_prob"      # log_prob | perplexity | length_normalized
    exploration_bonus: float = 0.1
    elite_fraction: float = 0.2
    temperature: float = 1.0
    diversity_pressure: float = 0.5

    # -- scheduling --------------------------------------------------------
    archive_update_freq: int = 1
    behavior_update_freq: int = 5
    length_penalty: float = 1.0
    min_archive_coverage: float = 0.0

    # -- validation --------------------------------------------------------

    def validate(self) -> List[str]:
        errors = super().validate()
        if self.beam_width < 1:
            errors.append("beam_width must be >= 1")
        if self.archive_size < 1:
            errors.append("archive_size must be >= 1")
        if self.num_behavior_dims < 1:
            errors.append("num_behavior_dims must be >= 1")
        if not self.behavior_descriptors:
            errors.append("behavior_descriptors must not be empty")
        if self.tessellation_type not in ("grid", "voronoi", "cvt"):
            errors.append(f"Unknown tessellation_type: {self.tessellation_type}")
        if self.grid_resolution < 1:
            errors.append("grid_resolution must be >= 1")
        if self.quality_metric not in ("log_prob", "perplexity", "length_normalized"):
            errors.append(f"Unknown quality_metric: {self.quality_metric}")
        if not (0.0 <= self.exploration_bonus <= 10.0):
            errors.append("exploration_bonus should be in [0, 10]")
        if not (0.0 <= self.elite_fraction <= 1.0):
            errors.append("elite_fraction must be in [0, 1]")
        if not (0.0 <= self.diversity_pressure <= 1.0):
            errors.append("diversity_pressure must be in [0, 1]")
        if self.archive_update_freq < 1:
            errors.append("archive_update_freq must be >= 1")
        if self.behavior_update_freq < 1:
            errors.append("behavior_update_freq must be >= 1")
        if self.length_penalty <= 0:
            errors.append("length_penalty must be > 0")
        if not (0.0 <= self.min_archive_coverage <= 1.0):
            errors.append("min_archive_coverage must be in [0, 1]")
        return errors


# =========================================================================
# QDBSState – extended decoding state
# =========================================================================


@dataclass
class QDBSState(DecodingState):
    """Extended decoding state for QD-BS, carrying the archive and beam."""

    archive: Optional[Archive] = None
    active_beams: List[Candidate] = field(default_factory=list)
    beam_scores: List[float] = field(default_factory=list)
    behavior_cache: Dict[tuple, np.ndarray] = field(default_factory=dict)
    coverage_history: List[float] = field(default_factory=list)
    qd_score_history: List[float] = field(default_factory=list)

    def clone(self) -> "QDBSState":
        return copy.deepcopy(self)


# =========================================================================
# QualityDiversityBeamSearch – main algorithm
# =========================================================================


class QualityDiversityBeamSearch(DecodingAlgorithm):
    """Quality-Diversity Beam Search (QD-BS).

    Integrates MAP-Elites-style archive management into beam search.
    At each expansion step candidates are scored for quality and mapped
    to behaviour cells.  The archive retains the best candidate per cell,
    guaranteeing diverse, high-quality final outputs.

    Parameters
    ----------
    config : QDBSConfig
        Full algorithm configuration.
    """

    def __init__(self, config: QDBSConfig) -> None:
        self._cfg: QDBSConfig = config
        super().__init__(config)

        # Build composite behaviour descriptor
        self._behavior = self._build_behavior_descriptor()

        # Build tessellation
        self._tessellation = self._build_tessellation()

        # Top-k expansion width (tokens to consider per beam)
        self._expand_k = max(20, config.beam_width)

        logger.info(
            "QD-BS initialised: beam_width=%d, archive_cells=%d, behavior=%s",
            config.beam_width,
            self._tessellation.num_cells,
            self._behavior.descriptor_names(),
        )

    # -- properties --------------------------------------------------------

    @property
    def description(self) -> str:
        return (
            "Quality-Diversity Beam Search — MAP-Elites archive "
            "integrated with beam search for diverse, high-quality decoding."
        )

    # -- builders ----------------------------------------------------------

    def _build_behavior_descriptor(self) -> CompositeBehaviorDescriptor:
        """Construct the composite behaviour descriptor from config."""
        descriptors = [
            _build_descriptor(name, max_length=self._cfg.max_new_tokens)
            for name in self._cfg.behavior_descriptors
        ]
        # Adjust num_behavior_dims if mismatch
        if len(descriptors) != self._cfg.num_behavior_dims:
            logger.warning(
                "num_behavior_dims (%d) != len(behavior_descriptors) (%d); using %d",
                self._cfg.num_behavior_dims,
                len(descriptors),
                len(descriptors),
            )
            self._cfg.num_behavior_dims = len(descriptors)
        return CompositeBehaviorDescriptor(descriptors)

    def _build_tessellation(self) -> Tessellation:
        """Construct the tessellation from config."""
        bounds = self._behavior.bounds  # (n_dims, 2)
        ttype = self._cfg.tessellation_type.lower()

        if ttype == "grid":
            return GridTessellation(bounds, self._cfg.grid_resolution)
        elif ttype == "voronoi":
            # Generate random centroids within bounds
            rng = self._rng or np.random.default_rng(42)
            lows = bounds[:, 0]
            highs = bounds[:, 1]
            centroids = rng.uniform(lows, highs, size=(self._cfg.archive_size, bounds.shape[0]))
            return VoronoiTessellation(centroids)
        elif ttype == "cvt":
            return CVTTessellation(
                bounds,
                self._cfg.archive_size,
                n_iterations=50,
                seed=self._cfg.seed,
            )
        else:
            raise ValueError(f"Unknown tessellation type: {ttype}")

    # -- generate (public entry point) -------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate diverse sequences from the QD-BS archive.

        Overrides the base class to ensure the archive is used for final
        output extraction.
        """
        if self._cfg.seed is not None:
            np.random.seed(self._cfg.seed)
            self._rng = np.random.default_rng(self._cfg.seed)

        state = self._init_state(prompt_ids)
        state = self._generation_loop(state, logit_source)
        return self._finalize(state)

    # -- lifecycle hooks ---------------------------------------------------

    def _init_state(self, prompt_ids: List[int]) -> QDBSState:
        """Initialise QD-BS state with archive and initial beams."""
        prompt = list(prompt_ids)
        prompt_len = len(prompt)

        archive = Archive(self._tessellation, self._cfg.archive_size)

        # Create initial beam candidates (all starting from prompt)
        initial_beams: List[Candidate] = []
        initial_scores: List[float] = []
        for i in range(self._cfg.beam_width):
            cand = Candidate(
                sequence=list(prompt),
                score=0.0,
                behavior=None,
                cell_index=None,
                parent_index=-1,
                token_added=-1,
            )
            initial_beams.append(cand)
            initial_scores.append(0.0)

        state = QDBSState(
            sequences=[list(prompt) for _ in range(self._cfg.beam_width)],
            scores=[0.0] * self._cfg.beam_width,
            is_finished=[False] * self._cfg.beam_width,
            step=0,
            metadata={"prompt_length": prompt_len},
            embeddings=None,
            logit_history=[],
            archive=archive,
            active_beams=initial_beams,
            beam_scores=initial_scores,
            behavior_cache={},
            coverage_history=[],
            qd_score_history=[],
        )
        return state

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute a single QD-BS decoding step.

        1. Expand all active beams by top-k tokens.
        2. Compute quality scores for each candidate.
        3. Compute behaviour descriptors for each candidate.
        4. Map candidates to archive cells.
        5. Update archive: keep best per cell.
        6. Select beam for next step: mix of archive elites + best candidates.
        """
        assert isinstance(state, QDBSState)
        cfg = self._cfg
        step = state.step

        # 1. Expand beams
        candidates = self._expand_beams(state, logit_source)

        if not candidates:
            # All beams finished — mark state
            for i in range(len(state.is_finished)):
                state.is_finished[i] = True
            return state

        # 2. Compute quality scores
        qualities = self._compute_quality(candidates)

        # 3. Compute behaviour descriptors (with caching)
        should_recompute_behavior = (
            step % cfg.behavior_update_freq == 0 or step == 0
        )
        behaviors = self._compute_behavior_descriptors(
            candidates, state.behavior_cache, force=should_recompute_behavior
        )

        # 4. Map to archive cells
        cell_indices = self._map_to_cells(behaviors)

        # Annotate candidates
        for i, cand in enumerate(candidates):
            cand.score = qualities[i]
            cand.behavior = behaviors[i]
            cand.cell_index = cell_indices[i]

        # 5. Update archive
        should_update_archive = (step % cfg.archive_update_freq == 0) or step == 0
        if should_update_archive and state.archive is not None:
            state.archive = self._update_archive(
                state.archive, candidates, qualities, cell_indices, step
            )

        # 6. Select beam for next step
        selected = self._select_beam(state.archive, candidates, cfg.beam_width)

        # Update state
        state.active_beams = selected
        state.beam_scores = [c.score for c in selected]

        # Sync base-class sequences/scores/is_finished
        state.sequences = [c.sequence for c in selected]
        state.scores = [c.score for c in selected]
        state.is_finished = [False] * len(selected)

        # Check EOS
        if cfg.eos_token_id is not None:
            for i, cand in enumerate(selected):
                if cand.sequence and cand.sequence[-1] == cfg.eos_token_id:
                    state.is_finished[i] = True

        # Track archive metrics
        if state.archive is not None:
            state.coverage_history.append(state.archive.coverage())
            state.qd_score_history.append(state.archive.qd_score())

        return state

    def _should_stop(self, state: DecodingState) -> bool:
        """Stop if all beams are finished or archive coverage target is met."""
        if state.all_finished():
            return True
        if isinstance(state, QDBSState) and state.archive is not None:
            cfg = self._cfg
            if (
                cfg.min_archive_coverage > 0
                and state.archive.coverage() >= cfg.min_archive_coverage
                and state.step >= cfg.min_new_tokens
            ):
                logger.info(
                    "Archive coverage target %.2f reached at step %d",
                    cfg.min_archive_coverage,
                    state.step,
                )
                return True
        return False

    # -- expansion ---------------------------------------------------------

    def _expand_beams(
        self, state: QDBSState, logit_source: LogitSource
    ) -> List[Candidate]:
        """Expand each active beam by top-k tokens.

        Returns a flat list of candidate continuations.
        """
        cfg = self._cfg
        active_beams = state.active_beams
        if not active_beams:
            return []

        # Collect sequences for batch logit computation
        input_seqs = [c.sequence for c in active_beams]

        # Get logits (batch call)
        try:
            logits_batch = logit_source(input_seqs)  # (n_beams, vocab_size)
        except Exception as e:
            logger.error("Logit source error: %s", e)
            return []

        logits_batch = np.asarray(logits_batch, dtype=np.float64)

        # Apply temperature
        if cfg.temperature > 0 and cfg.temperature != 1.0:
            logits_batch = logits_batch / cfg.temperature

        candidates: List[Candidate] = []
        k = min(self._expand_k, logits_batch.shape[-1])

        for beam_idx, (beam, logits) in enumerate(zip(active_beams, logits_batch)):
            # Apply constraints
            logits_constrained = self._apply_constraints_single(logits, beam.sequence, state)

            # Log-softmax for scoring
            log_probs = _log_softmax(logits_constrained)

            # Top-k token selection
            top_k_indices = np.argpartition(logits_constrained, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(-logits_constrained[top_k_indices])]

            for token_idx in top_k_indices:
                token_id = int(token_idx)
                token_log_prob = float(log_probs[token_id])
                new_score = beam.score + token_log_prob

                new_seq = list(beam.sequence) + [token_id]
                cand = Candidate(
                    sequence=new_seq,
                    score=new_score,
                    behavior=None,
                    cell_index=None,
                    parent_index=beam_idx,
                    token_added=token_id,
                )
                candidates.append(cand)

        return candidates

    def _apply_constraints_single(
        self,
        logits: np.ndarray,
        sequence: List[int],
        state: QDBSState,
    ) -> np.ndarray:
        """Apply repetition penalty and n-gram blocking to a single beam."""
        cfg = self._cfg
        logits = logits.copy()

        # Repetition penalty
        if cfg.repetition_penalty > 1.0:
            logits = self._apply_repetition_penalty(
                logits, sequence, cfg.repetition_penalty
            )

        # No-repeat n-gram
        if cfg.no_repeat_ngram_size > 0:
            logits = self._apply_no_repeat_ngram(
                logits, sequence, cfg.no_repeat_ngram_size
            )

        # Min length enforcement
        prompt_len = state.metadata.get("prompt_length", 0)
        gen_len = max(len(sequence) - prompt_len, 0)
        logits = self._enforce_min_length(
            logits, gen_len, cfg.min_new_tokens, cfg.eos_token_id
        )

        return logits

    # -- quality computation -----------------------------------------------

    def _compute_quality(self, candidates: List[Candidate]) -> np.ndarray:
        """Compute quality scores for all candidates.

        Supports three quality metrics:
        - ``log_prob``: raw accumulated log-probability.
        - ``length_normalized``: log-prob / length^length_penalty.
        - ``perplexity``: exp(-mean_log_prob); lower is better, so we negate.
        """
        cfg = self._cfg
        n = len(candidates)
        qualities = np.zeros(n, dtype=np.float64)

        for i, cand in enumerate(candidates):
            seq_len = max(len(cand.sequence), 1)
            raw_score = cand.score

            if cfg.quality_metric == "log_prob":
                qualities[i] = raw_score
            elif cfg.quality_metric == "length_normalized":
                penalty = seq_len ** cfg.length_penalty
                qualities[i] = raw_score / max(penalty, _EPS)
            elif cfg.quality_metric == "perplexity":
                # Perplexity = exp(-mean_log_prob); we want higher quality →
                # use negative perplexity (so higher is better).
                mean_lp = raw_score / seq_len
                qualities[i] = -math.exp(-mean_lp) if mean_lp != 0 else 0.0
            else:
                qualities[i] = raw_score

        return qualities

    # -- behaviour computation ---------------------------------------------

    def _compute_behavior_descriptors(
        self,
        candidates: List[Candidate],
        cache: Dict[tuple, np.ndarray],
        force: bool = False,
    ) -> np.ndarray:
        """Compute behaviour descriptors for all candidates.

        Uses a cache keyed by ``tuple(sequence)`` to avoid redundant work.
        When *force* is False and a cached value exists, it is reused.
        """
        n = len(candidates)
        n_dims = self._behavior.n_dims
        behaviors = np.zeros((n, n_dims), dtype=np.float64)

        for i, cand in enumerate(candidates):
            key = tuple(cand.sequence)
            if not force and key in cache:
                behaviors[i] = cache[key]
            else:
                bvec = self._behavior.compute_all(cand.sequence)
                bvec = self._behavior.normalize(bvec)
                cache[key] = bvec
                behaviors[i] = bvec

        return behaviors

    # -- cell mapping ------------------------------------------------------

    def _map_to_cells(self, descriptors: np.ndarray) -> np.ndarray:
        """Map each behaviour descriptor vector to its archive cell index."""
        n = descriptors.shape[0]
        cells = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cells[i] = self._tessellation.map_to_cell(descriptors[i])
        return cells

    # -- archive update ----------------------------------------------------

    def _update_archive(
        self,
        archive: Archive,
        candidates: List[Candidate],
        qualities: np.ndarray,
        cells: np.ndarray,
        step: int,
    ) -> Archive:
        """Update the archive with new candidates.

        For each cell c:
            A[c] = argmax_{y ∈ {A[c]} ∪ {y' : β(y') ∈ c}} score(y)

        Also applies an exploration bonus for candidates mapping to currently
        empty cells.
        """
        cfg = self._cfg
        empty_set = set(archive.empty_cells()) if cfg.exploration_bonus > 0 else set()

        for i, cand in enumerate(candidates):
            quality = float(qualities[i])
            cell_idx = int(cells[i])
            behavior = cand.behavior if cand.behavior is not None else np.zeros(self._behavior.n_dims)

            # Exploration bonus: reward candidates that land in empty cells
            adjusted_quality = quality
            if cfg.exploration_bonus > 0 and cell_idx in empty_set:
                adjusted_quality += cfg.exploration_bonus

            archive.add(
                cand,
                adjusted_quality,
                behavior,
                step=step,
                metadata={"original_quality": quality, "cell_was_empty": cell_idx in empty_set},
            )

        return archive

    # -- beam selection ----------------------------------------------------

    def _select_beam(
        self,
        archive: Optional[Archive],
        candidates: List[Candidate],
        beam_width: int,
    ) -> List[Candidate]:
        """Select the next beam from archive elites and best candidates.

        The beam is composed of:
        - ``elite_fraction`` of beams sampled from the archive's best entries
          (one per cell), providing diversity pressure.
        - Remaining slots filled by the highest-quality candidates from the
          current expansion, providing exploitation.

        Diversity pressure additionally biases candidate selection toward
        candidates that map to under-represented or empty archive cells.
        """
        cfg = self._cfg

        # Number of elite slots
        n_elite = max(1, int(beam_width * cfg.elite_fraction))
        n_exploit = beam_width - n_elite

        selected: List[Candidate] = []

        # --- Elite selection from archive --------------------------------
        if archive is not None and len(archive) > 0:
            elites = archive.best_per_cell()
            # Sample up to n_elite from archive entries
            n_from_archive = min(n_elite, len(elites))
            for entry in elites[:n_from_archive]:
                selected.append(Candidate(
                    sequence=list(entry.sequence),
                    score=entry.quality,
                    behavior=entry.behavior.copy(),
                    cell_index=entry.cell_index,
                    parent_index=-1,
                    token_added=-1,
                ))
            # If fewer elites than n_elite, increase exploitation slots
            n_exploit = beam_width - len(selected)

        # --- Exploitation: best candidates by diversity-adjusted score ---
        if candidates and n_exploit > 0:
            scored_candidates = self._diversity_adjusted_ranking(
                candidates, archive, cfg.diversity_pressure
            )
            # Take top n_exploit candidates not already selected
            selected_seqs = {tuple(c.sequence) for c in selected}
            added = 0
            for cand, adj_score in scored_candidates:
                if added >= n_exploit:
                    break
                key = tuple(cand.sequence)
                if key not in selected_seqs:
                    selected.append(cand)
                    selected_seqs.add(key)
                    added += 1

        # If still short, pad with best raw candidates
        if len(selected) < beam_width and candidates:
            sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
            selected_seqs = {tuple(c.sequence) for c in selected}
            for cand in sorted_cands:
                if len(selected) >= beam_width:
                    break
                key = tuple(cand.sequence)
                if key not in selected_seqs:
                    selected.append(cand)
                    selected_seqs.add(key)

        return selected[:beam_width]

    def _diversity_adjusted_ranking(
        self,
        candidates: List[Candidate],
        archive: Optional[Archive],
        diversity_pressure: float,
    ) -> List[Tuple[Candidate, float]]:
        """Rank candidates by quality with a diversity bonus.

        Candidates that map to empty or under-represented archive cells
        receive a bonus proportional to *diversity_pressure*.

        Returns a list of (candidate, adjusted_score) sorted descending.
        """
        if not candidates:
            return []

        # Collect per-cell occupancy counts
        cell_counts: Dict[int, int] = {}
        if archive is not None:
            for cell_idx in archive.occupied_cells():
                cell_counts[cell_idx] = 1  # binary: occupied or not

        max_score = max(c.score for c in candidates) if candidates else 1.0
        min_score = min(c.score for c in candidates) if candidates else 0.0
        score_range = max(max_score - min_score, _EPS)

        results: List[Tuple[Candidate, float]] = []
        for cand in candidates:
            # Normalised quality
            norm_q = (cand.score - min_score) / score_range

            # Diversity bonus: higher if cell is empty
            cell_idx = cand.cell_index if cand.cell_index is not None else 0
            is_occupied = cell_idx in cell_counts
            diversity_bonus = 0.0 if is_occupied else 1.0

            adjusted = (1.0 - diversity_pressure) * norm_q + diversity_pressure * diversity_bonus
            results.append((cand, adjusted))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # -- finalisation ------------------------------------------------------

    def _finalize(self, state: DecodingState) -> List[TokenSequence]:
        """Extract final sequences from the archive.

        Returns up to ``num_sequences`` sequences sorted by quality.
        """
        if not isinstance(state, QDBSState) or state.archive is None:
            return self._postprocess(state)

        archive = state.archive
        prompt_len = state.metadata.get("prompt_length", 0)

        # Get all archive entries sorted by quality
        entries = archive.best_per_cell()

        # Extract generated portion (strip prompt)
        results: List[Tuple[float, TokenSequence]] = []
        for entry in entries:
            generated = entry.sequence[prompt_len:]
            if generated:
                results.append((entry.quality, generated))

        # Sort by quality descending
        results.sort(key=lambda x: x[0], reverse=True)

        # Limit to num_sequences
        n = self._cfg.num_sequences
        sequences = [seq for _, seq in results[:n]]

        # If archive didn't produce enough, supplement from beams
        if len(sequences) < n and isinstance(state, QDBSState):
            beam_seqs = set(tuple(s) for s in sequences)
            for cand in sorted(state.active_beams, key=lambda c: c.score, reverse=True):
                if len(sequences) >= n:
                    break
                gen = cand.sequence[prompt_len:]
                if gen and tuple(gen) not in beam_seqs:
                    sequences.append(gen)
                    beam_seqs.add(tuple(gen))

        logger.info(
            "QD-BS finalized: %d sequences, archive coverage=%.2f%%, qd_score=%.4f",
            len(sequences),
            archive.coverage() * 100,
            archive.qd_score(),
        )
        return sequences

    # -- hyperparameter space ----------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        base = super().hyperparameter_space()
        base.update({
            "beam_width": {"type": "int", "low": 10, "high": 200},
            "archive_size": {"type": "int", "low": 20, "high": 500},
            "exploration_bonus": {"type": "float", "low": 0.0, "high": 1.0},
            "elite_fraction": {"type": "float", "low": 0.0, "high": 0.5},
            "diversity_pressure": {"type": "float", "low": 0.0, "high": 1.0},
            "grid_resolution": {"type": "int", "low": 3, "high": 20},
            "length_penalty": {"type": "float", "low": 0.5, "high": 2.0},
            "tessellation_type": {
                "type": "categorical",
                "choices": ["grid", "voronoi", "cvt"],
            },
            "quality_metric": {
                "type": "categorical",
                "choices": ["log_prob", "perplexity", "length_normalized"],
            },
        })
        return base

    def validate_config(self) -> List[str]:
        return self._cfg.validate()


# =========================================================================
# QDBSAnalyzer – post-hoc analysis
# =========================================================================


class QDBSAnalyzer:
    """Post-hoc analysis utilities for QD-BS runs.

    Provides methods to assess archive convergence, behaviour-space coverage,
    quality distributions, and exploration-exploitation balance.
    """

    # -- archive convergence -----------------------------------------------

    @staticmethod
    def archive_convergence(
        coverage_history: List[float],
        qd_history: List[float],
        window: int = 10,
    ) -> Dict[str, Any]:
        """Analyse archive convergence from step-wise metrics.

        Parameters
        ----------
        coverage_history : list of float
            Coverage at each step.
        qd_history : list of float
            QD-score at each step.
        window : int
            Smoothing window for derivative estimation.

        Returns
        -------
        dict with keys:
            - converged (bool): whether the QD-score has plateaued
            - final_coverage (float)
            - final_qd_score (float)
            - coverage_velocity (float): mean coverage increase per step
            - qd_velocity (float): mean QD-score increase per step
            - steps_to_50pct_coverage (int or None)
            - plateau_step (int or None): step at which QD-score stopped improving
        """
        result: Dict[str, Any] = {
            "converged": False,
            "final_coverage": 0.0,
            "final_qd_score": 0.0,
            "coverage_velocity": 0.0,
            "qd_velocity": 0.0,
            "steps_to_50pct_coverage": None,
            "plateau_step": None,
        }

        if not coverage_history:
            return result

        result["final_coverage"] = coverage_history[-1]
        result["final_qd_score"] = qd_history[-1] if qd_history else 0.0

        # Coverage velocity
        if len(coverage_history) > 1:
            diffs = [
                coverage_history[i] - coverage_history[i - 1]
                for i in range(1, len(coverage_history))
            ]
            result["coverage_velocity"] = sum(diffs) / len(diffs)

        # QD velocity
        if len(qd_history) > 1:
            diffs = [
                qd_history[i] - qd_history[i - 1]
                for i in range(1, len(qd_history))
            ]
            result["qd_velocity"] = sum(diffs) / len(diffs)

        # Steps to 50% coverage
        for step, cov in enumerate(coverage_history):
            if cov >= 0.5:
                result["steps_to_50pct_coverage"] = step
                break

        # Plateau detection: find step after which QD-score doesn't improve
        # by more than 1% of current value within the next `window` steps
        if len(qd_history) > window:
            for step in range(len(qd_history) - window):
                current = qd_history[step]
                future_max = max(qd_history[step + 1 : step + window + 1])
                if current > 0 and (future_max - current) / abs(current) < 0.01:
                    result["plateau_step"] = step
                    result["converged"] = True
                    break

        return result

    # -- behaviour space coverage ------------------------------------------

    @staticmethod
    def behavior_space_coverage(archive: Archive) -> Dict[str, Any]:
        """Analyse how well the archive covers the behaviour space.

        Returns
        -------
        dict with keys:
            - coverage (float): fraction of cells occupied
            - n_occupied (int)
            - n_total (int)
            - n_empty (int)
            - behavior_centroid (list): mean behaviour vector of entries
            - behavior_spread (list): std dev per behaviour dimension
            - min_behavior (list): per-dimension minimum
            - max_behavior (list): per-dimension maximum
            - uniformity (float): 1 - coefficient of variation of pairwise distances
        """
        result: Dict[str, Any] = {
            "coverage": archive.coverage(),
            "n_occupied": len(archive.occupied_cells()),
            "n_total": archive.tessellation.num_cells,
            "n_empty": len(archive.empty_cells()),
            "behavior_centroid": [],
            "behavior_spread": [],
            "min_behavior": [],
            "max_behavior": [],
            "uniformity": 0.0,
        }

        entries = list(archive.cells.values())
        if not entries:
            return result

        behaviors = np.array([e.behavior for e in entries])
        result["behavior_centroid"] = behaviors.mean(axis=0).tolist()
        result["behavior_spread"] = behaviors.std(axis=0).tolist()
        result["min_behavior"] = behaviors.min(axis=0).tolist()
        result["max_behavior"] = behaviors.max(axis=0).tolist()

        # Uniformity: how evenly distributed are the entries?
        if len(entries) >= 2:
            dists = []
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    dists.append(float(np.linalg.norm(behaviors[i] - behaviors[j])))
            if dists:
                mean_d = np.mean(dists)
                std_d = np.std(dists)
                cv = std_d / max(mean_d, _EPS)
                result["uniformity"] = max(0.0, 1.0 - cv)

        return result

    # -- quality distribution ----------------------------------------------

    @staticmethod
    def quality_distribution(archive: Archive) -> Dict[str, Any]:
        """Analyse the distribution of quality values in the archive.

        Returns
        -------
        dict with keys:
            - mean (float)
            - median (float)
            - std (float)
            - min (float)
            - max (float)
            - p25, p75 (float): 25th and 75th percentiles
            - iqr (float): inter-quartile range
            - n_entries (int)
            - histogram_bins (list): bin edges
            - histogram_counts (list): counts per bin
        """
        entries = list(archive.cells.values())
        result: Dict[str, Any] = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "iqr": 0.0,
            "n_entries": len(entries),
            "histogram_bins": [],
            "histogram_counts": [],
        }

        if not entries:
            return result

        qualities = np.array([e.quality for e in entries])
        result["mean"] = float(np.mean(qualities))
        result["median"] = float(np.median(qualities))
        result["std"] = float(np.std(qualities))
        result["min"] = float(np.min(qualities))
        result["max"] = float(np.max(qualities))
        result["p25"] = float(np.percentile(qualities, 25))
        result["p75"] = float(np.percentile(qualities, 75))
        result["iqr"] = result["p75"] - result["p25"]

        # Histogram
        n_bins = min(20, max(5, len(entries) // 5))
        counts, bin_edges = np.histogram(qualities, bins=n_bins)
        result["histogram_bins"] = bin_edges.tolist()
        result["histogram_counts"] = counts.tolist()

        return result

    # -- exploration vs exploitation analysis ------------------------------

    @staticmethod
    def analyze_exploration_exploitation(
        state_history: List[QDBSState],
    ) -> Dict[str, Any]:
        """Analyse the exploration-exploitation trade-off over a run.

        Examines how archive coverage (exploration) and quality (exploitation)
        evolve together over time.

        Parameters
        ----------
        state_history : list of QDBSState
            Snapshots of QD-BS state at each step.

        Returns
        -------
        dict with keys:
            - exploration_ratio (list): per-step fraction of new cells discovered
            - exploitation_ratio (list): per-step fraction of quality improvements
            - phase_transitions (list): steps where the dominant mode switches
            - overall_balance (float): ratio exploration/exploitation (≈1 = balanced)
            - cumulative_exploration (list)
            - cumulative_exploitation (list)
        """
        result: Dict[str, Any] = {
            "exploration_ratio": [],
            "exploitation_ratio": [],
            "phase_transitions": [],
            "overall_balance": 0.0,
            "cumulative_exploration": [],
            "cumulative_exploitation": [],
        }

        if not state_history:
            return result

        prev_occupied: Set[int] = set()
        prev_qualities: Dict[int, float] = {}
        cum_explore = 0.0
        cum_exploit = 0.0

        for state in state_history:
            if state.archive is None:
                result["exploration_ratio"].append(0.0)
                result["exploitation_ratio"].append(0.0)
                result["cumulative_exploration"].append(cum_explore)
                result["cumulative_exploitation"].append(cum_exploit)
                continue

            current_occupied = set(state.archive.occupied_cells())
            new_cells = current_occupied - prev_occupied
            total_cells = max(state.archive.tessellation.num_cells, 1)

            explore_ratio = len(new_cells) / total_cells
            result["exploration_ratio"].append(explore_ratio)
            cum_explore += explore_ratio
            result["cumulative_exploration"].append(cum_explore)

            # Exploitation: count quality improvements
            n_improved = 0
            for cell_idx, entry in state.archive.cells.items():
                old_q = prev_qualities.get(cell_idx)
                if old_q is not None and entry.quality > old_q:
                    n_improved += 1
            exploit_ratio = n_improved / max(len(current_occupied), 1)
            result["exploitation_ratio"].append(exploit_ratio)
            cum_exploit += exploit_ratio
            result["cumulative_exploitation"].append(cum_exploit)

            # Update tracking
            prev_occupied = current_occupied
            prev_qualities = {
                cell_idx: entry.quality
                for cell_idx, entry in state.archive.cells.items()
            }

        # Phase transitions: where exploration and exploitation cross
        for i in range(1, len(result["exploration_ratio"])):
            prev_dom = (
                "explore"
                if result["exploration_ratio"][i - 1] >= result["exploitation_ratio"][i - 1]
                else "exploit"
            )
            curr_dom = (
                "explore"
                if result["exploration_ratio"][i] >= result["exploitation_ratio"][i]
                else "exploit"
            )
            if prev_dom != curr_dom:
                result["phase_transitions"].append(i)

        # Overall balance
        total_explore = sum(result["exploration_ratio"])
        total_exploit = sum(result["exploitation_ratio"])
        result["overall_balance"] = total_explore / max(total_exploit, _EPS)

        return result

    # -- convenience: full report ------------------------------------------

    @classmethod
    def full_report(cls, state: QDBSState) -> Dict[str, Any]:
        """Generate a comprehensive analysis report for a completed QD-BS run.

        Parameters
        ----------
        state : QDBSState
            Final state after generation.

        Returns
        -------
        dict with sub-reports for convergence, coverage, quality.
        """
        report: Dict[str, Any] = {"algorithm": "QualityDiversityBeamSearch"}

        if state.archive is not None:
            report["convergence"] = cls.archive_convergence(
                state.coverage_history, state.qd_score_history
            )
            report["coverage"] = cls.behavior_space_coverage(state.archive)
            report["quality"] = cls.quality_distribution(state.archive)
            report["archive_size"] = len(state.archive)
            report["archive_qd_score"] = state.archive.qd_score()
            report["archive_diversity"] = state.archive.diversity_score()
        else:
            report["convergence"] = {}
            report["coverage"] = {}
            report["quality"] = {}

        report["total_steps"] = state.step
        report["final_beam_size"] = len(state.active_beams)

        return report


# =========================================================================
# Convenience factory
# =========================================================================


def create_qdbs(
    beam_width: int = 50,
    archive_size: int = 100,
    behavior_descriptors: Optional[List[str]] = None,
    tessellation_type: str = "grid",
    grid_resolution: int = 10,
    quality_metric: str = "log_prob",
    exploration_bonus: float = 0.1,
    elite_fraction: float = 0.2,
    temperature: float = 1.0,
    diversity_pressure: float = 0.5,
    max_new_tokens: int = 100,
    num_sequences: int = 20,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> QualityDiversityBeamSearch:
    """Convenience factory for creating a QD-BS instance.

    Parameters
    ----------
    beam_width : int
        Total beam width.
    archive_size : int
        Maximum archive cells.
    behavior_descriptors : list of str, optional
        Names of behaviour descriptors.
    tessellation_type : str
        ``"grid"``, ``"voronoi"``, or ``"cvt"``.
    grid_resolution : int
        Per-dimension resolution for grid tessellation.
    quality_metric : str
        ``"log_prob"``, ``"perplexity"``, or ``"length_normalized"``.
    exploration_bonus : float
        Bonus for candidates landing in empty cells.
    elite_fraction : float
        Fraction of beam filled by archive elites.
    temperature : float
        Sampling temperature.
    diversity_pressure : float
        Pressure toward empty cells in beam selection.
    max_new_tokens : int
        Maximum generated tokens.
    num_sequences : int
        Number of output sequences.
    seed : int, optional
        Random seed.

    Returns
    -------
    QualityDiversityBeamSearch
    """
    if behavior_descriptors is None:
        behavior_descriptors = list(_DEFAULT_BEHAVIOR_NAMES)

    config = QDBSConfig(
        beam_width=beam_width,
        archive_size=archive_size,
        num_behavior_dims=len(behavior_descriptors),
        behavior_descriptors=behavior_descriptors,
        tessellation_type=tessellation_type,
        grid_resolution=grid_resolution,
        quality_metric=quality_metric,
        exploration_bonus=exploration_bonus,
        elite_fraction=elite_fraction,
        temperature=temperature,
        diversity_pressure=diversity_pressure,
        max_new_tokens=max_new_tokens,
        num_sequences=num_sequences,
        seed=seed,
        **{k: v for k, v in kwargs.items()},
    )

    return QualityDiversityBeamSearch(config)


# =========================================================================
# Module-level exports
# =========================================================================

# =========================================================================
# MAPElitesArchive – Extended archive with multi-resolution & curiosity
# =========================================================================


class MAPElitesArchive:
    """Full MAP-Elites style archive with multi-resolution grid support,
    improvement tracking, curiosity-driven exploration bonuses, and
    snapshot/restore for rollback.

    Parameters
    ----------
    tessellation : Tessellation
        The tessellation back-end for cell assignment.
    curiosity_weight : float
        Weight for the curiosity bonus applied to rarely-visited cells.
    history_length : int
        Maximum number of archive snapshots to retain.
    """

    def __init__(
        self,
        tessellation: Tessellation,
        curiosity_weight: float = 0.1,
        history_length: int = 50,
    ) -> None:
        self.tessellation = tessellation
        self.curiosity_weight = curiosity_weight
        self._history_length = history_length

        self.cells: Dict[int, ArchiveEntry] = {}
        self._visit_counts: Dict[int, int] = defaultdict(int)
        self._insert_count: int = 0
        self._improvement_count: int = 0
        self._generation: int = 0
        self._history: List[Dict[int, ArchiveEntry]] = []

    # -- core operations ---------------------------------------------------

    def add_to_archive(
        self,
        candidate: Candidate,
        quality: float,
        behavior: np.ndarray,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert a candidate into the archive with curiosity bonus.

        The effective quality is augmented by a curiosity bonus for
        rarely-visited cells before comparing with existing occupants.

        Returns ``True`` if the candidate was inserted or improved the cell.
        """
        cell_idx = self.tessellation.map_to_cell(behavior)
        self._visit_counts[cell_idx] += 1

        curiosity = self.get_curiosity_bonus(cell_idx)
        effective_quality = quality + curiosity

        entry = ArchiveEntry(
            sequence=list(candidate.sequence),
            quality=quality,
            behavior=behavior.copy(),
            cell_index=cell_idx,
            step_added=step,
            metadata=metadata or {},
        )

        existing = self.cells.get(cell_idx)
        existing_effective = (
            existing.quality + self.get_curiosity_bonus(cell_idx)
            if existing is not None
            else _NEG_INF
        )

        if existing is None or effective_quality > existing_effective:
            self.cells[cell_idx] = entry
            self._insert_count += 1
            if existing is not None:
                self._improvement_count += 1
            logger.debug(
                "MAPElitesArchive: inserted into cell %d (quality=%.4f, "
                "curiosity=%.4f)",
                cell_idx,
                quality,
                curiosity,
            )
            return True
        return False

    def get_curiosity_bonus(self, cell_index: int) -> float:
        """Compute a curiosity bonus inversely proportional to visit count.

        Cells visited fewer times receive a larger bonus, encouraging
        exploration of under-represented regions.
        """
        visits = self._visit_counts.get(cell_index, 0)
        return self.curiosity_weight / math.sqrt(visits + 1)

    def _compute_cell_visit_counts(self) -> Dict[int, int]:
        """Return a copy of the cell visit counts."""
        return dict(self._visit_counts)

    # -- snapshot / restore ------------------------------------------------

    def snapshot(self) -> int:
        """Take a snapshot of the current archive state.

        Returns the snapshot index for later :meth:`restore`.
        """
        snap = {k: copy.deepcopy(v) for k, v in self.cells.items()}
        self._history.append(snap)
        if len(self._history) > self._history_length:
            self._history.pop(0)
        self._generation += 1
        logger.info(
            "MAPElitesArchive: snapshot %d taken (%d cells)",
            self._generation,
            len(self.cells),
        )
        return len(self._history) - 1

    def restore(self, snapshot_index: int = -1) -> None:
        """Restore the archive to a previous snapshot.

        Parameters
        ----------
        snapshot_index : int
            Index into the snapshot history.  ``-1`` restores the latest.
        """
        if not self._history:
            raise ValueError("No snapshots available for restore")
        snap = self._history[snapshot_index]
        self.cells = {k: copy.deepcopy(v) for k, v in snap.items()}
        logger.info(
            "MAPElitesArchive: restored to snapshot %d (%d cells)",
            snapshot_index,
            len(self.cells),
        )

    # -- metrics -----------------------------------------------------------

    def improvement_rate(self) -> float:
        """Fraction of insertions that improved an existing cell."""
        if self._insert_count == 0:
            return 0.0
        return self._improvement_count / self._insert_count

    def coverage(self) -> float:
        """Fraction of cells that are occupied."""
        total = self.tessellation.num_cells
        return len(self.cells) / max(total, 1)

    def qd_score(self) -> float:
        """Quality-Diversity score: sum of quality across all occupied cells."""
        return sum(e.quality for e in self.cells.values())

    def best_per_cell(self) -> List[ArchiveEntry]:
        """Return all entries (one per occupied cell), sorted by quality."""
        entries = list(self.cells.values())
        entries.sort(key=lambda e: e.quality, reverse=True)
        return entries

    def sample_from_archive(
        self,
        n: int = 1,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[ArchiveEntry]:
        """Sample *n* entries from the archive uniformly at random.

        Parameters
        ----------
        n : int
            Number of entries to sample.
        rng : np.random.RandomState, optional
            Random state for reproducibility.
        """
        if not self.cells:
            return []
        rng = rng or np.random.RandomState()
        keys = list(self.cells.keys())
        chosen = rng.choice(keys, size=min(n, len(keys)), replace=False)
        return [self.cells[k] for k in chosen]

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
        return (
            f"MAPElitesArchive(occupied={len(self.cells)}/"
            f"{self.tessellation.num_cells}, "
            f"coverage={self.coverage():.2%}, "
            f"qd_score={self.qd_score():.4f}, "
            f"generation={self._generation})"
        )


# =========================================================================
# SentimentDescriptor – Lexicon-based sentiment behaviour descriptor
# =========================================================================


class SentimentDescriptor(BehaviorDescriptor):
    """Behaviour characterisation based on sentiment / valence.

    Uses a simple lexicon-based approach with positive and negative word
    lists.  Each token id is hashed into a pseudo-word and looked up in the
    lexicon.  The descriptor outputs a sentiment score in [0, 1] where 0 is
    fully negative and 1 is fully positive.
    """

    def __init__(self) -> None:
        self._positive, self._negative = self._build_lexicon()

    @staticmethod
    def _build_lexicon() -> Tuple[Set[str], Set[str]]:
        """Build simple positive / negative word sets."""
        positive: Set[str] = {
            "good", "great", "happy", "love", "wonderful", "excellent",
            "amazing", "best", "beautiful", "perfect", "joy", "brilliant",
            "fantastic", "superb", "awesome", "pleasant", "nice", "fine",
            "cheerful", "delightful", "positive", "fortunate", "glad",
            "bright", "kind", "warm", "gentle", "proud", "calm", "sweet",
            "hope", "success", "win", "benefit", "gain", "improve",
            "praise", "admire", "treasure", "celebrate",
        }
        negative: Set[str] = {
            "bad", "terrible", "sad", "hate", "awful", "horrible",
            "worst", "ugly", "poor", "pain", "misery", "dreadful",
            "nasty", "grim", "tragic", "cruel", "harsh", "bitter",
            "angry", "fear", "fail", "loss", "damage", "harm",
            "destroy", "suffer", "regret", "blame", "reject", "hostile",
            "dark", "gloom", "despair", "grief", "sorrow", "woe",
            "anxiety", "stress", "burden", "threat",
        }
        return positive, negative

    def compute(self, token_ids: List[int]) -> float:
        """Compute sentiment score in [0, 1] for the token sequence."""
        if not token_ids:
            return 0.5
        pos_count = 0
        neg_count = 0
        for tid in token_ids:
            # Hash token id to a pseudo-word from the combined lexicon
            all_words = sorted(self._positive | self._negative)
            if not all_words:
                continue
            word = all_words[tid % len(all_words)]
            if word in self._positive:
                pos_count += 1
            elif word in self._negative:
                neg_count += 1
        total = pos_count + neg_count
        if total == 0:
            return 0.5
        return pos_count / total

    def describe(self) -> str:
        """Human-readable description of the descriptor."""
        return (
            f"SentimentDescriptor(lexicon_size="
            f"{len(self._positive) + len(self._negative)})"
        )

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# =========================================================================
# TopicDescriptor – TF-IDF-like topic behaviour descriptor
# =========================================================================


class TopicDescriptor(BehaviorDescriptor):
    """Topic-based behaviour characterisation via term frequency analysis.

    Assigns sequences to topic clusters based on TF-IDF-like token
    frequency profiles.  The descriptor outputs a normalised topic index
    in [0, 1].

    Parameters
    ----------
    num_topics : int
        Number of topic clusters to simulate (default 8).
    """

    def __init__(self, num_topics: int = 8) -> None:
        self._num_topics = max(2, num_topics)

    def _compute_tf(self, token_ids: List[int]) -> Dict[int, float]:
        """Compute normalised term frequencies for *token_ids*."""
        if not token_ids:
            return {}
        freq: Dict[int, int] = defaultdict(int)
        for tid in token_ids:
            freq[tid] += 1
        total = len(token_ids)
        return {k: v / total for k, v in freq.items()}

    def _assign_topic(self, tf: Dict[int, float]) -> int:
        """Assign a topic index based on the token frequency profile.

        Uses a simple hash of the dominant tokens to determine the topic
        cluster.
        """
        if not tf:
            return 0
        # Use the top-3 most frequent token ids as a topic signature
        sorted_tokens = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_tokens[: min(3, len(sorted_tokens))]
        signature = sum(tid * (i + 1) for i, (tid, _) in enumerate(top_k))
        return signature % self._num_topics

    def compute(self, token_ids: List[int]) -> float:
        """Compute normalised topic index in [0, 1]."""
        if not token_ids:
            return 0.0
        tf = self._compute_tf(token_ids)
        topic = self._assign_topic(tf)
        return topic / max(self._num_topics - 1, 1)

    def describe(self) -> str:
        """Human-readable description of the descriptor."""
        return f"TopicDescriptor(num_topics={self._num_topics})"

    @property
    def name(self) -> str:
        return "topic"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# =========================================================================
# StyleDescriptor – Formality / verbosity / sophistication descriptor
# =========================================================================


class StyleDescriptor(BehaviorDescriptor):
    """Style characterisation measuring formality, verbosity, and lexical
    sophistication.

    Combines three sub-metrics into a single [0, 1] score:
    * average word length (proxy for sophistication)
    * average sentence length (proxy for verbosity)
    * type-token ratio (proxy for lexical variety)

    Because token ids are not real words, we simulate word properties via
    deterministic hashing of token ids.
    """

    def _avg_word_length(self, token_ids: List[int]) -> float:
        """Simulated average word length from token ids."""
        if not token_ids:
            return 0.0
        # Map each token id to a pseudo-word length in [1, 15]
        lengths = [(tid % 15) + 1 for tid in token_ids]
        return sum(lengths) / len(lengths)

    def _avg_sentence_length(self, token_ids: List[int]) -> float:
        """Simulated average sentence length.

        Treats certain token id modular classes as sentence boundaries.
        """
        if not token_ids:
            return 0.0
        # Token ids divisible by 29 are treated as sentence enders
        sentence_count = max(sum(1 for tid in token_ids if tid % 29 == 0), 1)
        return len(token_ids) / sentence_count

    def _type_token_ratio(self, token_ids: List[int]) -> float:
        """Type-token ratio of the token sequence."""
        if not token_ids:
            return 0.0
        return len(set(token_ids)) / len(token_ids)

    def compute(self, token_ids: List[int]) -> float:
        """Compute a combined style score in [0, 1].

        Equal-weight average of normalised sub-metrics.
        """
        if not token_ids:
            return 0.0
        # Normalise average word length to [0, 1] (max simulated is 15)
        awl = min(self._avg_word_length(token_ids) / 15.0, 1.0)
        # Normalise average sentence length to [0, 1] (cap at 50)
        asl = min(self._avg_sentence_length(token_ids) / 50.0, 1.0)
        # TTR is already in [0, 1]
        ttr = self._type_token_ratio(token_ids)
        return (awl + asl + ttr) / 3.0

    def describe(self) -> str:
        """Human-readable description of the descriptor."""
        return "StyleDescriptor(metrics=[avg_word_length, avg_sentence_length, ttr])"

    @property
    def name(self) -> str:
        return "style"

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, 1.0)


# =========================================================================
# NoveltySearchBeam – Novelty search in beam space
# =========================================================================


class NoveltySearchBeam:
    """Novelty search in beam space with k-nearest novelty scoring.

    Instead of optimising quality, this algorithm optimises for *novelty*
    — the distance to the k-nearest neighbours in behaviour space.  An
    archive of previously observed behaviours is maintained and used for
    novelty computation.

    Parameters
    ----------
    behavior_descriptor : CompositeBehaviorDescriptor
        Descriptor used to compute behaviour vectors.
    k : int
        Number of nearest neighbours for novelty scoring.
    archive_capacity : int
        Maximum number of entries in the novelty archive.
    beam_width : int
        Number of candidates to maintain per step.
    """

    def __init__(
        self,
        behavior_descriptor: CompositeBehaviorDescriptor,
        k: int = 10,
        archive_capacity: int = 1000,
        beam_width: int = 32,
    ) -> None:
        self._descriptor = behavior_descriptor
        self._k = max(1, k)
        self._archive_capacity = archive_capacity
        self._beam_width = beam_width
        self._novelty_archive: List[np.ndarray] = []

    def _compute_novelty(self, behavior: np.ndarray) -> float:
        """Compute the novelty of *behavior* against the archive."""
        if not self._novelty_archive:
            return float("inf")
        return self._knn_distance(behavior, self._novelty_archive, self._k)

    @staticmethod
    def _knn_distance(
        point: np.ndarray,
        archive: List[np.ndarray],
        k: int,
    ) -> float:
        """Mean Euclidean distance to the *k* nearest neighbours."""
        if not archive:
            return float("inf")
        archive_arr = np.array(archive)
        dists = np.linalg.norm(archive_arr - point, axis=1)
        k_actual = min(k, len(dists))
        nearest = np.partition(dists, k_actual - 1)[:k_actual]
        return float(np.mean(nearest))

    def _update_novelty_archive(self, behavior: np.ndarray) -> None:
        """Add a behaviour vector to the novelty archive.

        If the archive exceeds capacity, the oldest entry is removed.
        """
        self._novelty_archive.append(behavior.copy())
        if len(self._novelty_archive) > self._archive_capacity:
            self._novelty_archive.pop(0)

    def _select_most_novel(
        self,
        candidates: List[Candidate],
        n: int,
    ) -> List[Candidate]:
        """Select the *n* most novel candidates from the list."""
        scored: List[Tuple[float, int, Candidate]] = []
        for idx, cand in enumerate(candidates):
            beh = self._descriptor.compute_all(cand.sequence)
            novelty = self._compute_novelty(beh)
            scored.append((novelty, idx, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:n]
        # Update archive with selected behaviours
        for novelty, _, cand in selected:
            beh = self._descriptor.compute_all(cand.sequence)
            self._update_novelty_archive(beh)
        return [c for _, _, c in selected]

    def generate(
        self,
        initial_candidates: List[Candidate],
        expand_fn: Any,
        max_steps: int = 50,
    ) -> List[Candidate]:
        """Run novelty search beam for up to *max_steps* expansions.

        Parameters
        ----------
        initial_candidates : list of Candidate
            Starting beam candidates.
        expand_fn : callable
            ``expand_fn(candidate) -> list[Candidate]`` that expands a
            single candidate by one step.
        max_steps : int
            Maximum number of expansion steps.

        Returns
        -------
        list of Candidate
            The final beam, sorted by novelty (most novel first).
        """
        beam = list(initial_candidates)[: self._beam_width]
        for step in range(max_steps):
            all_children: List[Candidate] = []
            for cand in beam:
                children = expand_fn(cand)
                all_children.extend(children)
            if not all_children:
                logger.info(
                    "NoveltySearchBeam: no children at step %d, stopping", step
                )
                break
            beam = self._select_most_novel(all_children, self._beam_width)
            logger.debug(
                "NoveltySearchBeam: step %d, beam_size=%d, "
                "archive_size=%d",
                step,
                len(beam),
                len(self._novelty_archive),
            )
        return beam

    def get_novelty_archive(self) -> List[np.ndarray]:
        """Return a copy of the current novelty archive."""
        return [b.copy() for b in self._novelty_archive]

    def __repr__(self) -> str:
        return (
            f"NoveltySearchBeam(k={self._k}, "
            f"archive_size={len(self._novelty_archive)}/{self._archive_capacity}, "
            f"beam_width={self._beam_width})"
        )


# =========================================================================
# AdaptiveTessellationArchive – Adaptive quadtree-like tessellation
# =========================================================================


class AdaptiveTessellationArchive:
    """Quality-diversity archive with adaptive tessellation.

    Starts with a coarse grid and recursively subdivides cells that
    accumulate many candidates — similar to a quadtree.  This provides
    finer resolution in densely-populated areas of behaviour space while
    keeping sparsely-populated areas coarse.

    Parameters
    ----------
    n_dims : int
        Number of behaviour-space dimensions.
    initial_resolution : int
        Initial grid resolution per dimension.
    max_depth : int
        Maximum number of subdivision levels.
    split_threshold : int
        Number of insertions into a cell before it is subdivided.
    """

    def __init__(
        self,
        n_dims: int = 2,
        initial_resolution: int = 4,
        max_depth: int = 4,
        split_threshold: int = 5,
    ) -> None:
        self._n_dims = n_dims
        self._initial_resolution = initial_resolution
        self._max_depth = max_depth
        self._split_threshold = split_threshold

        # Cells are keyed by a tuple of (cell_index, depth)
        self._cells: Dict[Tuple[int, ...], ArchiveEntry] = {}
        self._cell_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        self._cell_depths: Dict[Tuple[int, ...], int] = {}
        self._total_inserts: int = 0

    def _should_subdivide(self, cell_key: Tuple[int, ...]) -> bool:
        """Determine whether a cell should be subdivided."""
        depth = self._cell_depths.get(cell_key, 0)
        if depth >= self._max_depth:
            return False
        count = self._cell_counts.get(cell_key, 0)
        return count >= self._split_threshold

    def _subdivide_cell(
        self,
        cell_key: Tuple[int, ...],
    ) -> List[Tuple[int, ...]]:
        """Subdivide a cell into 2^n_dims sub-cells.

        Returns the keys of the new sub-cells.
        """
        depth = self._cell_depths.get(cell_key, 0)
        new_depth = depth + 1
        sub_cells: List[Tuple[int, ...]] = []
        # Each dimension is split in two, so we get 2^n_dims sub-cells
        for offset in range(2 ** self._n_dims):
            bits = tuple(
                (offset >> d) & 1 for d in range(self._n_dims)
            )
            sub_key = cell_key + bits
            self._cell_depths[sub_key] = new_depth
            sub_cells.append(sub_key)
        logger.debug(
            "AdaptiveTessellationArchive: subdivided %s into %d sub-cells "
            "(depth %d)",
            cell_key,
            len(sub_cells),
            new_depth,
        )
        return sub_cells

    def _adaptive_cell_assignment(
        self,
        behavior: np.ndarray,
    ) -> Tuple[int, ...]:
        """Assign a behaviour vector to an adaptive cell.

        Starts at the coarsest level and descends into subdivided cells
        as appropriate.
        """
        # Clamp to [0, 1]
        beh = np.clip(behavior, 0.0, 1.0)
        # Coarse cell at initial resolution
        base = tuple(
            min(int(beh[d] * self._initial_resolution), self._initial_resolution - 1)
            for d in range(min(len(beh), self._n_dims))
        )
        # Pad if fewer dims than expected
        while len(base) < self._n_dims:
            base = base + (0,)

        cell_key = base
        depth = self._cell_depths.get(cell_key, 0)
        self._cell_depths.setdefault(cell_key, 0)

        # Descend into subdivided cells
        while depth < self._max_depth and self._should_subdivide(cell_key):
            sub_cells = self._subdivide_cell(cell_key)
            # Find the sub-cell that best matches the behaviour
            resolution = self._initial_resolution * (2 ** (depth + 1))
            sub_idx = tuple(
                min(int(beh[d] * resolution) % 2, 1)
                for d in range(self._n_dims)
            )
            cell_key = cell_key + sub_idx
            self._cell_depths.setdefault(cell_key, depth + 1)
            depth = self._cell_depths.get(cell_key, depth + 1)

        return cell_key

    def add(
        self,
        candidate: Candidate,
        quality: float,
        behavior: np.ndarray,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a candidate to the adaptive archive.

        Returns ``True`` if the candidate was inserted or improved a cell.
        """
        cell_key = self._adaptive_cell_assignment(behavior)
        self._cell_counts[cell_key] += 1
        self._total_inserts += 1

        entry = ArchiveEntry(
            sequence=list(candidate.sequence),
            quality=quality,
            behavior=behavior.copy(),
            cell_index=hash(cell_key) % (10 ** 9),
            step_added=step,
            metadata=metadata or {},
        )

        existing = self._cells.get(cell_key)
        if existing is None or quality > existing.quality:
            self._cells[cell_key] = entry
            return True
        return False

    def get_cell_depths(self) -> Dict[Tuple[int, ...], int]:
        """Return the depth of each cell in the adaptive tessellation."""
        return dict(self._cell_depths)

    def total_cells(self) -> int:
        """Return the total number of cells (occupied and unoccupied)."""
        return len(self._cell_depths)

    def coverage(self) -> float:
        """Fraction of existing cells that are occupied."""
        total = max(len(self._cell_depths), 1)
        return len(self._cells) / total

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return (
            f"AdaptiveTessellationArchive(occupied={len(self._cells)}, "
            f"total_cells={self.total_cells()}, "
            f"coverage={self.coverage():.2%})"
        )


# =========================================================================
# IlluminationSearch – Full illumination algorithm integration
# =========================================================================


class IlluminationSearch:
    """Full illumination algorithm combining MAP-Elites with beam search.

    Maintains a QD archive and uses it to guide beam search toward
    unexplored regions of behaviour space, producing a diverse and
    high-quality set of text sequences.

    Parameters
    ----------
    archive : MAPElitesArchive
        The MAP-Elites archive to populate.
    behavior_descriptor : CompositeBehaviorDescriptor
        Descriptor for computing behaviour vectors.
    mutation_rate : float
        Probability of mutating each token during sequence mutation.
    num_parents : int
        Number of parents to select from the archive per illumination step.
    max_generations : int
        Maximum number of illumination generations.
    """

    def __init__(
        self,
        archive: MAPElitesArchive,
        behavior_descriptor: CompositeBehaviorDescriptor,
        mutation_rate: float = 0.1,
        num_parents: int = 8,
        max_generations: int = 100,
    ) -> None:
        self._archive = archive
        self._descriptor = behavior_descriptor
        self._mutation_rate = mutation_rate
        self._num_parents = num_parents
        self._max_generations = max_generations
        self._rng = np.random.RandomState(42)
        self._generation_log: List[Dict[str, float]] = []

    def _illumination_step(self) -> int:
        """Execute a single illumination step.

        Select parents from the archive, mutate them, evaluate, and
        archive the results.

        Returns the number of successful insertions.
        """
        parents = self._select_parent_from_archive(self._num_parents)
        if not parents:
            return 0

        insertions = 0
        for parent in parents:
            mutated_seq = self._mutate_sequence(parent.sequence)
            insertions += self._evaluate_and_archive(mutated_seq, parent)

        return insertions

    def _select_parent_from_archive(
        self,
        n: int,
    ) -> List[ArchiveEntry]:
        """Select *n* parent entries from the archive for mutation."""
        return self._archive.sample_from_archive(n, self._rng)

    def _mutate_sequence(self, sequence: List[int]) -> List[int]:
        """Apply random mutations to a token sequence.

        Each token has a ``mutation_rate`` probability of being replaced
        with a random token id drawn uniformly from [0, max(sequence) + 100].
        """
        if not sequence:
            return []
        max_token = max(sequence) + 100
        mutated = list(sequence)
        for i in range(len(mutated)):
            if self._rng.random() < self._mutation_rate:
                mutated[i] = int(self._rng.randint(0, max_token))
        return mutated

    def _evaluate_and_archive(
        self,
        sequence: List[int],
        parent: ArchiveEntry,
    ) -> int:
        """Evaluate a mutated sequence and attempt to add to the archive.

        Quality is estimated as the negative distance from the parent's
        quality (with a small random perturbation).

        Returns 1 if inserted, 0 otherwise.
        """
        behavior = self._descriptor.compute_all(sequence)
        # Quality: inherit from parent with perturbation
        quality = parent.quality + self._rng.normal(0.0, 0.01)
        candidate = Candidate(sequence=sequence, score=quality)
        if self._archive.add_to_archive(candidate, quality, behavior):
            return 1
        return 0

    def generate(
        self,
        initial_sequences: Optional[List[List[int]]] = None,
    ) -> List[ArchiveEntry]:
        """Run the illumination algorithm.

        Parameters
        ----------
        initial_sequences : list of list of int, optional
            Seed sequences to populate the archive.  If ``None``, the
            algorithm assumes the archive is already seeded.

        Returns
        -------
        list of ArchiveEntry
            All entries in the archive after the illumination process,
            sorted by quality.
        """
        # Seed archive with initial sequences if provided
        if initial_sequences:
            for seq in initial_sequences:
                behavior = self._descriptor.compute_all(seq)
                cand = Candidate(sequence=seq, score=0.0)
                self._archive.add_to_archive(cand, 0.0, behavior)

        for gen in range(self._max_generations):
            insertions = self._illumination_step()
            cov = self._archive.coverage()
            qd = self._archive.qd_score()
            self._generation_log.append({
                "generation": gen,
                "insertions": insertions,
                "coverage": cov,
                "qd_score": qd,
                "archive_size": len(self._archive),
            })
            if gen % 10 == 0:
                logger.info(
                    "IlluminationSearch: gen=%d, insertions=%d, "
                    "coverage=%.2%%, qd_score=%.4f, archive_size=%d",
                    gen,
                    insertions,
                    cov * 100,
                    qd,
                    len(self._archive),
                )
            # Take periodic snapshots
            if gen % 25 == 0 and gen > 0:
                self._archive.snapshot()

        return self._archive.best_per_cell()

    def illumination_progress(self) -> List[Dict[str, float]]:
        """Return the per-generation progress log."""
        return list(self._generation_log)

    def __repr__(self) -> str:
        return (
            f"IlluminationSearch(generations={self._max_generations}, "
            f"mutation_rate={self._mutation_rate}, "
            f"num_parents={self._num_parents}, "
            f"archive={self._archive!r})"
        )


# =========================================================================
# Helper functions
# =========================================================================


def compute_qd_score(archive: Union[Archive, MAPElitesArchive]) -> float:
    """Compute the QD-score: sum of quality across all filled cells.

    Parameters
    ----------
    archive : Archive or MAPElitesArchive
        The archive to compute the score for.

    Returns
    -------
    float
        Sum of quality values in the archive.
    """
    if isinstance(archive, (Archive, MAPElitesArchive)):
        return sum(e.quality for e in archive.cells.values())
    return 0.0


def archive_coverage(
    archive: Union[Archive, MAPElitesArchive],
    total_cells: int,
) -> float:
    """Compute the fraction of cells that are filled.

    Parameters
    ----------
    archive : Archive or MAPElitesArchive
        The archive to measure.
    total_cells : int
        Total number of cells in the tessellation.

    Returns
    -------
    float
        Fraction of cells occupied, in [0, 1].
    """
    if total_cells <= 0:
        return 0.0
    return len(archive.cells) / total_cells


def novelty_score(
    behavior: np.ndarray,
    archive_behaviors: List[np.ndarray],
    k: int = 10,
) -> float:
    """Compute the novelty of a behaviour vector.

    Novelty is defined as the mean Euclidean distance to the *k* nearest
    neighbours in the archive.

    Parameters
    ----------
    behavior : np.ndarray
        The behaviour vector to evaluate.
    archive_behaviors : list of np.ndarray
        Previously observed behaviour vectors.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        Mean distance to *k* nearest neighbours.
    """
    if not archive_behaviors:
        return float("inf")
    archive_arr = np.array(archive_behaviors)
    dists = np.linalg.norm(archive_arr - behavior, axis=1)
    k_actual = min(k, len(dists))
    nearest = np.partition(dists, k_actual - 1)[:k_actual]
    return float(np.mean(nearest))


def sentiment_lexicon_score(
    token_ids: List[int],
    vocab_size: int = 50000,
) -> float:
    """Compute a simple lexicon-based sentiment score for token ids.

    Maps token ids into positive / negative buckets based on modular
    arithmetic and returns a score in [0, 1].

    Parameters
    ----------
    token_ids : list of int
        Sequence of token ids.
    vocab_size : int
        Size of the vocabulary (used for normalisation).

    Returns
    -------
    float
        Sentiment score in [0, 1].
    """
    if not token_ids:
        return 0.5
    pos_count = 0
    neg_count = 0
    for tid in token_ids:
        # Simple heuristic: lower-id tokens are more positive
        normalised = (tid % max(vocab_size, 1)) / max(vocab_size, 1)
        if normalised < 0.4:
            pos_count += 1
        elif normalised > 0.6:
            neg_count += 1
    total = pos_count + neg_count
    if total == 0:
        return 0.5
    return pos_count / total


def adaptive_grid_resolution(
    archive: Union[Archive, MAPElitesArchive],
    min_res: int = 4,
    max_res: int = 64,
) -> int:
    """Compute an adaptive grid resolution based on archive occupancy.

    Increases resolution when the archive is well-filled so that denser
    regions of behaviour space can be explored at finer granularity.

    Parameters
    ----------
    archive : Archive or MAPElitesArchive
        Current archive.
    min_res : int
        Minimum grid resolution.
    max_res : int
        Maximum grid resolution.

    Returns
    -------
    int
        Recommended grid resolution.
    """
    cov = archive.coverage()
    # Scale resolution linearly with coverage
    res = int(min_res + (max_res - min_res) * cov)
    return max(min_res, min(res, max_res))


# =========================================================================
# Module-level exports
# =========================================================================

__all__ = [
    # Config
    "QDBSConfig",
    # Main algorithm
    "QualityDiversityBeamSearch",
    # Archive
    "Archive",
    "ArchiveEntry",
    "MAPElitesArchive",
    "AdaptiveTessellationArchive",
    # Tessellation
    "Tessellation",
    "GridTessellation",
    "VoronoiTessellation",
    "CVTTessellation",
    # Behavior descriptors
    "BehaviorDescriptor",
    "LengthDescriptor",
    "POSDiversityDescriptor",
    "LexicalDiversityDescriptor",
    "NGramDiversityDescriptor",
    "VocabRichnessDescriptor",
    "CompositeBehaviorDescriptor",
    "SentimentDescriptor",
    "TopicDescriptor",
    "StyleDescriptor",
    # State
    "QDBSState",
    "Candidate",
    # Analyzer
    "QDBSAnalyzer",
    # Search algorithms
    "NoveltySearchBeam",
    "IlluminationSearch",
    # Helper functions
    "compute_qd_score",
    "archive_coverage",
    "novelty_score",
    "sentiment_lexicon_score",
    "adaptive_grid_resolution",
    # Factory
    "create_qdbs",
]
