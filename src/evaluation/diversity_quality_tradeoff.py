"""
Diversity-quality tradeoff analysis for the Diversity Decoding Arena.

Provides comprehensive tools for analysing, comparing, and optimising the
tradeoff between text-generation diversity and output quality.  Includes
Pareto-frontier computation, operating-point selection, statistical frontier
comparison, constrained optimisation helpers, and visualisation-ready data
preparation.

Depends on:
    numpy, scipy
    dataclasses (stdlib)
    logging (stdlib)
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate as sp_interp
from scipy import optimize as sp_opt
from scipy import spatial as sp_spatial
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
_DEFAULT_SEED = 42
_MAX_EXACT_HV_POINTS = 500
_DEFAULT_MC_SAMPLES = 100_000
_DEFAULT_BOOTSTRAP_ITERS = 1000
_DEFAULT_GRID_RESOLUTION = 50


# ===================================================================
#  1.  Core data classes
# ===================================================================


@dataclass
class TradeoffPoint:
    """Single measurement on the diversity-quality plane.

    Attributes
    ----------
    quality_score : float
        Scalar quality metric (higher is better).
    diversity_score : float
        Scalar diversity metric (higher is better).
    algorithm : str
        Name/identifier of the decoding algorithm.
    config : dict
        Hyper-parameter configuration that produced this point.
    metadata : dict
        Arbitrary extra information (e.g. run id, seed, wall time).
    """

    quality_score: float = 0.0
    diversity_score: float = 0.0
    algorithm: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        quality_score: float = 0.0,
        diversity_score: float = 0.0,
        algorithm: str = "",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        quality: Optional[float] = None,
        diversity: Optional[float] = None,
    ) -> None:
        self.quality_score = quality if quality is not None else quality_score
        self.diversity_score = diversity if diversity is not None else diversity_score
        self.algorithm = algorithm
        self.config = config if config is not None else {}
        self.metadata = metadata if metadata is not None else {}

    @property
    def quality(self) -> float:
        """Alias for quality_score."""
        return self.quality_score

    @quality.setter
    def quality(self, value: float) -> None:
        self.quality_score = value

    @property
    def diversity(self) -> float:
        """Alias for diversity_score."""
        return self.diversity_score

    @diversity.setter
    def diversity(self, value: float) -> None:
        self.diversity_score = value

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "quality_score": float(self.quality_score),
            "diversity_score": float(self.diversity_score),
            "algorithm": self.algorithm,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeoffPoint":
        """Deserialise from a plain dictionary."""
        return cls(
            quality_score=float(d.get("quality_score", 0.0)),
            diversity_score=float(d.get("diversity_score", 0.0)),
            algorithm=str(d.get("algorithm", "")),
            config=dict(d.get("config", {})),
            metadata=dict(d.get("metadata", {})),
        )

    # -- convenience ----------------------------------------------------------

    def as_array(self) -> NDArray[np.float64]:
        """Return ``[quality_score, diversity_score]`` as a numpy array."""
        return np.array(
            [self.quality_score, self.diversity_score], dtype=np.float64
        )

    def copy(self) -> "TradeoffPoint":
        """Deep copy."""
        return TradeoffPoint(
            quality_score=self.quality_score,
            diversity_score=self.diversity_score,
            algorithm=self.algorithm,
            config=copy.deepcopy(self.config),
            metadata=copy.deepcopy(self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TradeoffPoint):
            return NotImplemented
        return (
            math.isclose(self.quality_score, other.quality_score, abs_tol=_EPS)
            and math.isclose(
                self.diversity_score, other.diversity_score, abs_tol=_EPS
            )
            and self.algorithm == other.algorithm
        )

    def __hash__(self) -> int:
        return hash(
            (
                round(self.quality_score, 10),
                round(self.diversity_score, 10),
                self.algorithm,
            )
        )

    def __repr__(self) -> str:
        return (
            f"TradeoffPoint(q={self.quality_score:.4f}, "
            f"d={self.diversity_score:.4f}, alg={self.algorithm!r})"
        )


@dataclass
class TradeoffConfig:
    """Configuration governing tradeoff analyses.

    Attributes
    ----------
    quality_range : tuple[float, float]
        Admissible quality score interval ``(lo, hi)``.
    diversity_range : tuple[float, float]
        Admissible diversity score interval ``(lo, hi)``.
    reference_point : tuple[float, float] | None
        Reference point for hypervolume computation.  If *None* the worst
        observed point is used.
    num_bootstrap : int
        Number of bootstrap resamples for statistical comparisons.
    confidence_level : float
        Confidence level for intervals (e.g. 0.95).
    grid_resolution : int
        Number of grid cells per axis for heatmaps / grid searches.
    interpolation_method : str
        Default interpolation along the frontier (``"linear"``,
        ``"cubic"``, ``"pchip"``).
    epsilon : float
        Tolerance for epsilon-dominance filtering.
    mc_samples : int
        Number of Monte-Carlo samples for approximate hypervolume.
    seed : int
        Random seed for reproducibility.
    quality_weight : float
        Default weight assigned to quality in scalarisation.
    diversity_weight : float
        Default weight assigned to diversity in scalarisation.
    """

    quality_range: Tuple[float, float] = (0.0, 1.0)
    diversity_range: Tuple[float, float] = (0.0, 1.0)
    reference_point: Optional[Tuple[float, float]] = None
    num_bootstrap: int = _DEFAULT_BOOTSTRAP_ITERS
    confidence_level: float = 0.95
    grid_resolution: int = _DEFAULT_GRID_RESOLUTION
    interpolation_method: str = "linear"
    epsilon: float = 0.01
    mc_samples: int = _DEFAULT_MC_SAMPLES
    seed: int = _DEFAULT_SEED
    quality_weight: float = 0.5
    diversity_weight: float = 0.5
    min_quality: float = 0.0
    min_diversity: float = 0.0
    n_interpolation_points: int = 100

    def __post_init__(self) -> None:
        if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.grid_resolution < 1:
            raise ValueError("grid_resolution must be >= 1")
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if self.quality_weight < 0 or self.quality_weight > 1.0:
            raise ValueError("quality_weight must be in [0, 1]")
        if self.diversity_weight < 0 or self.diversity_weight > 1.0:
            raise ValueError("diversity_weight must be in [0, 1]")
        if self.min_quality < 0:
            raise ValueError("min_quality must be >= 0")
        if self.min_diversity < 0:
            raise ValueError("min_diversity must be >= 0")
        if self.n_interpolation_points < 2:
            raise ValueError("n_interpolation_points must be >= 2")

    def validate(self) -> bool:
        """Return True if the configuration is valid, False otherwise."""
        try:
            if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
                return False
            if self.grid_resolution < 1:
                return False
            if self.epsilon < 0:
                return False
            if self.quality_weight < 0 or self.quality_weight > 1.0:
                return False
            if self.diversity_weight < 0 or self.diversity_weight > 1.0:
                return False
            if self.min_quality < 0:
                return False
            if self.min_diversity < 0:
                return False
            if self.n_interpolation_points < 2:
                return False
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_range": list(self.quality_range),
            "diversity_range": list(self.diversity_range),
            "reference_point": (
                list(self.reference_point) if self.reference_point else None
            ),
            "num_bootstrap": self.num_bootstrap,
            "confidence_level": self.confidence_level,
            "grid_resolution": self.grid_resolution,
            "interpolation_method": self.interpolation_method,
            "epsilon": self.epsilon,
            "mc_samples": self.mc_samples,
            "seed": self.seed,
            "quality_weight": self.quality_weight,
            "diversity_weight": self.diversity_weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeoffConfig":
        qr = d.get("quality_range", [0.0, 1.0])
        dr = d.get("diversity_range", [0.0, 1.0])
        rp = d.get("reference_point", None)
        return cls(
            quality_range=tuple(qr),  # type: ignore[arg-type]
            diversity_range=tuple(dr),  # type: ignore[arg-type]
            reference_point=(
                tuple(rp) if rp is not None else None  # type: ignore[arg-type]
            ),
            num_bootstrap=int(d.get("num_bootstrap", _DEFAULT_BOOTSTRAP_ITERS)),
            confidence_level=float(d.get("confidence_level", 0.95)),
            grid_resolution=int(
                d.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
            ),
            interpolation_method=str(d.get("interpolation_method", "linear")),
            epsilon=float(d.get("epsilon", 0.01)),
            mc_samples=int(d.get("mc_samples", _DEFAULT_MC_SAMPLES)),
            seed=int(d.get("seed", _DEFAULT_SEED)),
            quality_weight=float(d.get("quality_weight", 0.5)),
            diversity_weight=float(d.get("diversity_weight", 0.5)),
        )


# ===================================================================
#  2.  Small standalone helpers (used throughout)
# ===================================================================


def _to_array(points: Sequence[TradeoffPoint]) -> NDArray[np.float64]:
    """Stack tradeoff points into an (N, 2) array [quality, diversity]."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(
        [[p.quality_score, p.diversity_score] for p in points],
        dtype=np.float64,
    )


def _normalise_objectives(
    arr: NDArray[np.float64],
    ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> NDArray[np.float64]:
    """Min-max normalise each column of *arr* to [0, 1]."""
    out = arr.copy()
    for col in range(arr.shape[1]):
        if ranges is not None:
            lo, hi = ranges[col]
        else:
            lo, hi = float(arr[:, col].min()), float(arr[:, col].max())
        span = hi - lo
        if span < _EPS:
            out[:, col] = 0.5
        else:
            out[:, col] = (arr[:, col] - lo) / span
    return out


def _dominance_cmp(
    a: NDArray[np.float64], b: NDArray[np.float64], maximise: bool
) -> int:
    """Return 1 if *a* dominates *b*, -1 if *b* dominates *a*, else 0."""
    if maximise:
        a_geq_b = bool(np.all(a >= b - _EPS))
        a_gt_b = bool(np.any(a > b + _EPS))
        b_geq_a = bool(np.all(b >= a - _EPS))
        b_gt_a = bool(np.any(b > a + _EPS))
    else:
        a_geq_b = bool(np.all(a <= b + _EPS))
        a_gt_b = bool(np.any(a < b - _EPS))
        b_geq_a = bool(np.all(b <= a + _EPS))
        b_gt_a = bool(np.any(b < a - _EPS))
    if a_geq_b and a_gt_b:
        return 1
    if b_geq_a and b_gt_a:
        return -1
    return 0


def _non_dominated_mask(
    points: NDArray[np.float64], maximise: bool = True
) -> NDArray[np.bool_]:
    """Boolean mask selecting non-dominated rows."""
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(i + 1, n):
            if not mask[j]:
                continue
            rel = _dominance_cmp(points[i], points[j], maximise)
            if rel == 1:
                mask[j] = False
            elif rel == -1:
                mask[i] = False
                break
    return mask


# ===================================================================
#  3.  Pure-function helpers (module-level)
# ===================================================================


def is_pareto_dominated(
    point: NDArray[np.float64],
    others: NDArray[np.float64],
    maximise: bool = True,
    *,
    maximize: Optional[bool] = None,
) -> bool:
    """Return *True* if *point* is dominated by any row in *others*.

    Parameters
    ----------
    point : 1-D array of shape ``(M,)``
    others : 2-D array of shape ``(N, M)``
    maximise : bool
        If *True* higher values are better on every objective.
    maximize : bool, optional
        Alias for *maximise*.
    """
    if maximize is not None:
        maximise = maximize
    if others.shape[0] == 0:
        return False
    point = np.asarray(point, dtype=np.float64)
    others = np.asarray(others, dtype=np.float64)
    if maximise:
        geq = others >= point - _EPS
        gt = others > point + _EPS
    else:
        geq = others <= point + _EPS
        gt = others < point - _EPS
    dominated_by = np.all(geq, axis=1) & np.any(gt, axis=1)
    return bool(np.any(dominated_by))


def pareto_rank_assignment(
    points: NDArray[np.float64],
    maximise: bool = True,
    *,
    maximize: Optional[bool] = None,
) -> NDArray[np.int64]:
    """Assign non-dominated ranks (0 = first front) via fast NDS.

    Parameters
    ----------
    points : (N, M) array of objective values.
    maximise : bool
    maximize : bool, optional
        Alias for *maximise*.

    Returns
    -------
    ranks : (N,) int array
    """
    if maximize is not None:
        maximise = maximize
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    ranks = np.full(n, -1, dtype=np.int64)
    domination_count = np.zeros(n, dtype=np.int64)
    dominated_set: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            rel = _dominance_cmp(points[i], points[j], maximise)
            if rel == 1:
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif rel == -1:
                dominated_set[j].append(i)
                domination_count[i] += 1

    current_front: List[int] = []
    for i in range(n):
        if domination_count[i] == 0:
            ranks[i] = 0
            current_front.append(i)

    rank = 0
    while current_front:
        next_front: List[int] = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = rank + 1
                    next_front.append(j)
        rank += 1
        current_front = next_front

    return ranks


def compute_dominated_hypervolume(
    points: NDArray[np.float64],
    reference: NDArray[np.float64],
    maximise: bool = True,
    *,
    maximize: Optional[bool] = None,
) -> float:
    """Exact dominated hypervolume indicator.

    Uses O(N log N) sweep in 2-D and recursive slicing for higher
    dimensions.

    Parameters
    ----------
    points : (N, M) array -- non-dominated points.
    reference : (M,) array -- reference / anti-ideal point.
    maximise : bool
    maximize : bool, optional
        Alias for *maximise*.
    """
    if maximize is not None:
        maximise = maximize
    points = np.asarray(points, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    n, m = points.shape
    if n == 0:
        return 0.0

    if not maximise:
        points = -points
        reference = -reference

    # Keep only points that dominate the reference on every axis
    valid = np.all(points > reference + _EPS, axis=1)
    points = points[valid]
    n = points.shape[0]
    if n == 0:
        return 0.0

    if m == 1:
        return float(np.max(points[:, 0]) - reference[0])
    if m == 2:
        return _hypervolume_2d(points, reference)
    return _hypervolume_nd(points, reference)


def _hypervolume_2d(
    points: NDArray[np.float64], ref: NDArray[np.float64]
) -> float:
    """Sweep-line 2-D hypervolume in O(N log N).

    Sorts points by first objective (quality) descending; scans while
    tracking the running maximum of the second objective (diversity).
    Each point contributes a rectangle from its quality to the reference
    quality, with height equal to the gain in diversity.
    """
    order = np.argsort(-points[:, 0])
    pts_s = points[order]
    hv = 0.0
    tallest = ref[1]
    for i in range(pts_s.shape[0]):
        if pts_s[i, 1] > tallest:
            width = pts_s[i, 0] - ref[0]
            height = pts_s[i, 1] - tallest
            hv += height * width
            tallest = pts_s[i, 1]
    return float(hv)


def _hypervolume_nd(
    points: NDArray[np.float64], ref: NDArray[np.float64]
) -> float:
    """Recursive slicing hypervolume for dimensions > 2."""
    n, m = points.shape
    if n == 0:
        return 0.0
    if m == 1:
        return float(np.max(points[:, 0]) - ref[0])
    if m == 2:
        return _hypervolume_2d(points, ref)

    last = m - 1
    order = np.argsort(-points[:, last])
    sorted_pts = points[order]

    hv = 0.0
    prev_val = ref[last]
    accumulated: List[NDArray[np.float64]] = []

    for i in range(n):
        curr_val = sorted_pts[i, last]
        height = curr_val - prev_val
        if height > _EPS and accumulated:
            slice_pts = np.vstack(accumulated)
            nd_mask = _non_dominated_mask(slice_pts, maximise=True)
            hv += _hypervolume_nd(slice_pts[nd_mask], ref[:last]) * height
        accumulated.append(sorted_pts[i, :last].reshape(1, -1))
        prev_val = curr_val

    # Final slice contribution
    if accumulated:
        slice_pts = np.vstack(accumulated)
        nd_mask = _non_dominated_mask(slice_pts, maximise=True)
        top_val = float(sorted_pts[0, last])
        remaining = top_val - prev_val
        if remaining > _EPS:
            hv += _hypervolume_nd(slice_pts[nd_mask], ref[:last]) * remaining

    return float(hv)


def compute_spread_metric(
    frontier: NDArray[np.float64],
    ideal: Optional[NDArray[np.float64]] = None,
) -> float:
    r"""Spread metric measuring the extent of a Pareto front.

    Returns the diagonal distance (Euclidean norm of max - min) across
    all objectives.  Returns 0.0 for single-point or empty fronts.
    """
    frontier = np.asarray(frontier, dtype=np.float64)
    if frontier.ndim == 1:
        frontier = frontier.reshape(1, -1)
    if frontier.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(frontier.max(axis=0) - frontier.min(axis=0)))


def compute_generational_distance(
    obtained: NDArray[np.float64],
    reference_front: NDArray[np.float64],
    p: float = 2.0,
) -> float:
    """Generational Distance -- avg distance from obtained to reference.

    Lower is better.
    """
    if obtained.shape[0] == 0 or reference_front.shape[0] == 0:
        return float("inf")
    tree = sp_spatial.cKDTree(reference_front)
    dists, _ = tree.query(obtained, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return float((np.sum(dists**p) / len(dists)) ** (1.0 / p))


def compute_inverted_generational_distance(
    obtained: NDArray[np.float64],
    reference_front: NDArray[np.float64],
    p: float = 2.0,
) -> float:
    """IGD -- avg distance from each reference-front point to nearest
    obtained point.  Measures both convergence and spread.
    """
    if reference_front.shape[0] == 0:
        return 0.0
    if obtained.shape[0] == 0:
        return float("inf")
    tree = sp_spatial.cKDTree(obtained)
    dists, _ = tree.query(reference_front, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return float((np.sum(dists**p) / len(dists)) ** (1.0 / p))


# ===================================================================
#  4.  ParetoFrontier class
# ===================================================================


class ParetoFrontier:
    """Compute and interrogate a Pareto frontier over TradeoffPoints.

    Both objectives (quality, diversity) are *maximised*.
    """

    def __init__(self, points: Optional[Sequence[TradeoffPoint]] = None, config: Optional[TradeoffConfig] = None) -> None:
        # Support old-style ParetoFrontier(config=...) calls where first arg is config
        if points is not None and isinstance(points, TradeoffConfig):
            config = points
            points = None
        self.config = config or TradeoffConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._points: List[TradeoffPoint] = list(points) if points is not None else []
        self._frontier: List[TradeoffPoint] = []

    @property
    def frontier(self) -> List[TradeoffPoint]:
        """Return the computed frontier."""
        return list(self._frontier)

    # ------------------------------------------------------------------
    # Core: non-dominated set
    # ------------------------------------------------------------------

    def compute_frontier(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
    ) -> List[TradeoffPoint]:
        """Return the non-dominated subset (first Pareto front)."""
        if points is None:
            points = self._points
        if len(points) == 0:
            self._frontier = []
            return []
        result = self._compute_2d_frontier_impl(points)
        self._frontier = result
        return result

    @staticmethod
    def compute_2d_frontier(points_or_arr, **kwargs):
        """Accept either TradeoffPoints or a numpy array.
        
        When called as static method with ndarray, return indices.
        When called as instance method with TradeoffPoints, return points.
        """
        if isinstance(points_or_arr, np.ndarray):
            arr = points_or_arr
            n = arr.shape[0]
            if n == 0:
                return []
            mask = _non_dominated_mask(arr, maximise=True)
            return [i for i in range(n) if mask[i]]
        # Instance method path - delegate
        points = points_or_arr
        if len(points) == 0:
            return []
        arr = _to_array(points)
        order = np.lexsort((-arr[:, 1], -arr[:, 0]))
        frontier_indices: List[int] = []
        max_div = -np.inf
        for idx in order:
            if arr[idx, 1] > max_div - _EPS:
                frontier_indices.append(int(idx))
                if arr[idx, 1] > max_div:
                    max_div = arr[idx, 1]
        cand = arr[frontier_indices]
        mask = _non_dominated_mask(cand, maximise=True)
        return [
            points[frontier_indices[i]]
            for i in range(len(frontier_indices))
            if mask[i]
        ]

    def _compute_2d_frontier_impl(
        self,
        points: Sequence[TradeoffPoint],
    ) -> List[TradeoffPoint]:
        """Efficient O(N log N) 2-D non-dominated sort."""
        if len(points) == 0:
            return []
        arr = _to_array(points)
        # Sort by quality descending; for ties sort diversity descending
        order = np.lexsort((-arr[:, 1], -arr[:, 0]))
        frontier_indices: List[int] = []
        max_div = -np.inf
        for idx in order:
            if arr[idx, 1] > max_div - _EPS:
                frontier_indices.append(int(idx))
                if arr[idx, 1] > max_div:
                    max_div = arr[idx, 1]
        # Clean-up pass for exact non-dominance
        cand = arr[frontier_indices]
        mask = _non_dominated_mask(cand, maximise=True)
        return [
            points[frontier_indices[i]]
            for i in range(len(frontier_indices))
            if mask[i]
        ]

    @staticmethod
    def compute_nd_frontier(points_or_arr, extra_objectives=None):
        """N-dimensional Pareto frontier.

        When called with ndarray, returns indices of non-dominated points.
        When called with TradeoffPoints, returns non-dominated TradeoffPoints.
        """
        if isinstance(points_or_arr, np.ndarray):
            arr = points_or_arr
            n = arr.shape[0]
            if n == 0:
                return []
            mask = _non_dominated_mask(arr, maximise=True)
            return [i for i in range(n) if mask[i]]
        points = points_or_arr
        n = len(points)
        if n == 0:
            return []
        arr = _to_array(points)
        if extra_objectives:
            extras = np.column_stack(
                [np.asarray(v, dtype=np.float64) for v in extra_objectives.values()]
            )
            arr = np.hstack([arr, extras])
        mask = _non_dominated_mask(arr, maximise=True)
        return [points[i] for i in range(n) if mask[i]]

    # ------------------------------------------------------------------
    # Hypervolume
    # ------------------------------------------------------------------

    def hypervolume(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
        reference: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Exact dominated hypervolume."""
        if points is None:
            points = self._frontier if self._frontier else self._points
        arr = _to_array(points)
        if arr.shape[0] == 0:
            return 0.0
        ref = self._resolve_reference(arr, reference)
        return compute_dominated_hypervolume(arr, ref, maximise=True)

    def hypervolume_monte_carlo(
        self,
        points: Sequence[TradeoffPoint],
        reference: Optional[Tuple[float, float]] = None,
        n_samples: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Monte-Carlo hypervolume estimate.  Returns (estimate, stderr)."""
        arr = _to_array(points)
        if arr.shape[0] == 0:
            return 0.0, 0.0
        ref = self._resolve_reference(arr, reference)
        ns = n_samples or self.config.mc_samples

        upper = np.max(arr, axis=0)
        lower = ref.copy()
        box_volume = float(np.prod(upper - lower))
        if box_volume < _EPS:
            return 0.0, 0.0

        samples = self._rng.uniform(lower, upper, size=(ns, arr.shape[1]))
        # Vectorised dominance check in batches
        dominated_count = 0
        batch_size = min(5000, ns)
        for start in range(0, ns, batch_size):
            end = min(start + batch_size, ns)
            batch = samples[start:end]
            geq = arr[np.newaxis, :, :] >= batch[:, np.newaxis, :]
            any_dom = np.any(np.all(geq, axis=2), axis=1)
            dominated_count += int(np.sum(any_dom))

        p_hat = dominated_count / ns
        hv_est = p_hat * box_volume
        se = math.sqrt(p_hat * (1 - p_hat) / ns) * box_volume
        return hv_est, se

    # ------------------------------------------------------------------
    # Crowding distance
    # ------------------------------------------------------------------

    def crowding_distance(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
    ) -> NDArray[np.float64]:
        """Crowding distance for each point.  Extremes get ``inf``."""
        if points is None:
            points = self._frontier if self._frontier else self._points
        arr = _to_array(points)
        n = arr.shape[0]
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n, dtype=np.float64)
        for obj in range(arr.shape[1]):
            order = np.argsort(arr[:, obj])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            obj_range = float(arr[order[-1], obj] - arr[order[0], obj])
            if obj_range < _EPS:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (
                    arr[order[i + 1], obj] - arr[order[i - 1], obj]
                ) / obj_range
        return cd

    # ------------------------------------------------------------------
    # Epsilon dominance
    # ------------------------------------------------------------------

    def epsilon_dominance_filter(
        self,
        points: Sequence[TradeoffPoint],
        epsilon: Optional[float] = None,
    ) -> List[TradeoffPoint]:
        """Epsilon-dominance archive: keep points not eps-dominated."""
        eps = epsilon if epsilon is not None else self.config.epsilon
        arr = _to_array(points)
        n = arr.shape[0]
        if n == 0:
            return []
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                shifted = arr[j] - eps
                if bool(np.all(shifted >= arr[i] - _EPS)) and bool(
                    np.any(shifted > arr[i] + _EPS)
                ):
                    keep[i] = False
                    break
        return [points[i] for i in range(n) if keep[i]]

    def epsilon_dominates(
        self,
        a: TradeoffPoint,
        b: TradeoffPoint,
        eps: Optional[float] = None,
    ) -> bool:
        """Return True if *a* epsilon-dominates *b*."""
        epsilon = eps if eps is not None else self.config.epsilon
        a_arr = a.as_array()
        b_arr = b.as_array()
        return bool(
            np.all(a_arr >= b_arr - epsilon - _EPS)
            and np.any(a_arr > b_arr - epsilon + _EPS)
        )

    # ------------------------------------------------------------------
    # Interpolation along frontier
    # ------------------------------------------------------------------

    def interpolate_frontier(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
        n_interp: int = 200,
        method: Optional[str] = None,
    ) -> NDArray[np.float64]:
        """Interpolate along the frontier returning (n_interp, 2) array.

        Parameters
        ----------
        method : ``"linear"``, ``"cubic"``, or ``"pchip"``.
        """
        method = method or self.config.interpolation_method
        frontier = self.compute_frontier(points)
        if len(frontier) < 2:
            if len(frontier) == 1:
                return np.tile(frontier[0].as_array(), (n_interp, 1))
            return np.empty((0, 2), dtype=np.float64)

        arr = _to_array(frontier)
        order = np.argsort(arr[:, 0])
        xs = arr[order, 0]
        ys = arr[order, 1]

        # Remove duplicate x values by averaging corresponding y values
        ux, inv = np.unique(xs, return_inverse=True)
        uy = np.zeros_like(ux)
        counts = np.zeros_like(ux)
        np.add.at(uy, inv, ys)
        np.add.at(counts, inv, 1)
        uy /= counts
        xs, ys = ux, uy

        if len(xs) < 2:
            pt = np.array([xs[0], ys[0]])
            return np.tile(pt, (n_interp, 1))

        x_interp = np.linspace(float(xs[0]), float(xs[-1]), n_interp)

        if method == "pchip" and len(xs) >= 2:
            fn = sp_interp.PchipInterpolator(xs, ys, extrapolate=False)
            y_interp = fn(x_interp)
        elif method == "cubic" and len(xs) >= 4:
            fn = sp_interp.interp1d(
                xs, ys, kind="cubic", fill_value="extrapolate"
            )
            y_interp = fn(x_interp)
        else:
            fn = sp_interp.interp1d(
                xs, ys, kind="linear", fill_value="extrapolate"
            )
            y_interp = fn(x_interp)

        y_interp = np.nan_to_num(
            np.asarray(y_interp, dtype=np.float64), nan=0.0
        )
        return np.column_stack([x_interp, y_interp])

    def frontier_interpolation(
        self,
        n_points: Optional[int] = None,
        method: Optional[str] = None,
    ) -> NDArray[np.float64]:
        """Interpolate along the stored frontier."""
        n = n_points if n_points is not None else self.config.n_interpolation_points
        return self.interpolate_frontier(n_interp=n, method=method)

    # ------------------------------------------------------------------
    # Frontier shape metrics
    # ------------------------------------------------------------------

    def area_under_frontier(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
        n_interp: int = 500,
    ) -> float:
        """Area under the frontier curve (trapezoidal rule)."""
        curve = self.interpolate_frontier(
            points, n_interp=n_interp, method="linear"
        )
        if curve.shape[0] < 2:
            return 0.0
        return float(np.trapz(curve[:, 1], curve[:, 0]))

    def frontier_spread(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
    ) -> float:
        """Diagonal spread of the frontier."""
        if points is None:
            points = self._frontier if self._frontier else self._points
        frontier = self.compute_frontier(points)
        arr = _to_array(frontier)
        if arr.shape[0] == 0:
            return 0.0
        return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))

    def frontier_uniformity(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
    ) -> float:
        """CV of consecutive distances along the sorted frontier.

        Lower means more uniform.  Returns 0.0 if fewer than 3 points.
        """
        frontier = self.compute_frontier(points)
        arr = _to_array(frontier)
        if arr.shape[0] < 3:
            return 0.0
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        dists = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        mean_d = float(np.mean(dists))
        if mean_d < _EPS:
            return 0.0
        return float(np.std(dists) / mean_d)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve_reference(
        self,
        arr: NDArray[np.float64],
        override: Optional[Tuple[float, float]] = None,
    ) -> NDArray[np.float64]:
        if override is not None:
            return np.array(override, dtype=np.float64)
        if self.config.reference_point is not None:
            return np.array(self.config.reference_point, dtype=np.float64)
        margin = 0.01 * (arr.max(axis=0) - arr.min(axis=0) + _EPS)
        return arr.min(axis=0) - margin


# ===================================================================
#  5.  OperatingPointFinder
# ===================================================================


class OperatingPointFinder:
    """Select specific operating points on the Pareto frontier."""

    def __init__(self, pf_or_config=None) -> None:
        if isinstance(pf_or_config, ParetoFrontier):
            self._pf = pf_or_config
            self.config = pf_or_config.config
        else:
            self.config = pf_or_config or TradeoffConfig()
            self._pf = ParetoFrontier(config=self.config)

    # ------------------------------------------------------------------
    # Knee-point detection
    # ------------------------------------------------------------------

    def find_knee_point(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
        method: str = "curvature",
    ) -> Optional[TradeoffPoint]:
        """Find the knee of the Pareto front.

        Parameters
        ----------
        method : ``"curvature"`` or ``"l_method"``.
        """
        if points is not None:
            frontier = self._pf.compute_frontier(points)
        else:
            frontier = self._pf.frontier if self._pf.frontier else self._pf.compute_frontier()
        if len(frontier) == 0:
            return None
        if len(frontier) <= 2:
            return frontier[0]

        arr = _to_array(frontier)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        frontier_sorted = [frontier[int(i)] for i in order]

        if method == "l_method":
            return self._knee_l_method(arr, frontier_sorted)
        return self._knee_curvature(arr, frontier_sorted)

    def _knee_curvature(
        self,
        arr: NDArray[np.float64],
        frontier: List[TradeoffPoint],
    ) -> TradeoffPoint:
        """Maximum perpendicular distance to the line joining extremes."""
        p0 = arr[0]
        p1 = arr[-1]
        line_vec = p1 - p0
        line_len = float(np.linalg.norm(line_vec))
        if line_len < _EPS:
            return frontier[len(frontier) // 2]
        line_unit = line_vec / line_len
        best_idx = 0
        best_dist = -1.0
        for i in range(arr.shape[0]):
            v = arr[i] - p0
            proj = np.dot(v, line_unit)
            perp = v - proj * line_unit
            d = float(np.linalg.norm(perp))
            if d > best_dist:
                best_dist = d
                best_idx = i
        return frontier[best_idx]

    def _knee_l_method(
        self,
        arr: NDArray[np.float64],
        frontier: List[TradeoffPoint],
    ) -> TradeoffPoint:
        """L-method: partition into two linear segments; minimise total
        weighted residual.
        """
        n = arr.shape[0]
        if n <= 3:
            return frontier[n // 2]
        best_idx = 1
        best_cost = np.inf
        for k in range(2, n - 1):
            left = arr[:k]
            right = arr[k:]
            cl = self._segment_residual(left)
            cr = self._segment_residual(right)
            cost = (cl * k + cr * (n - k)) / n
            if cost < best_cost:
                best_cost = cost
                best_idx = k
        return frontier[best_idx]

    @staticmethod
    def _segment_residual(seg: NDArray[np.float64]) -> float:
        if seg.shape[0] < 2:
            return 0.0
        x, y = seg[:, 0], seg[:, 1]
        if float(np.std(x)) < _EPS:
            return float(np.sum((y - np.mean(y)) ** 2))
        coeffs = np.polyfit(x, y, 1)
        fitted = np.polyval(coeffs, x)
        return float(np.sum((y - fitted) ** 2))

    # ------------------------------------------------------------------
    # Constrained optima
    # ------------------------------------------------------------------

    def find_quality_constrained_optimum(
        self,
        points_or_min=None,
        min_quality: float = 0.0,
    ) -> Optional[TradeoffPoint]:
        """Max diversity subject to quality >= min_quality."""
        if isinstance(points_or_min, (int, float)):
            min_quality = float(points_or_min)
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        elif points_or_min is None:
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        else:
            points = points_or_min
        cands = [p for p in points if p.quality_score >= min_quality - _EPS]
        if not cands:
            return None
        return max(cands, key=lambda p: p.diversity_score)

    def find_diversity_constrained_optimum(
        self,
        points_or_min=None,
        min_diversity: float = 0.0,
    ) -> Optional[TradeoffPoint]:
        """Max quality subject to diversity >= min_diversity."""
        if isinstance(points_or_min, (int, float)):
            min_diversity = float(points_or_min)
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        elif points_or_min is None:
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        else:
            points = points_or_min
        cands = [
            p for p in points if p.diversity_score >= min_diversity - _EPS
        ]
        if not cands:
            return None
        return max(cands, key=lambda p: p.quality_score)

    # ------------------------------------------------------------------
    # Balanced / ratio
    # ------------------------------------------------------------------

    def find_balanced_point(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
    ) -> Optional[TradeoffPoint]:
        """Closest to the ideal (1, 1) in normalised space."""
        if points is None:
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        if not points:
            return None
        arr = _to_array(points)
        norm = _normalise_objectives(arr)
        ideal = np.array([1.0, 1.0])
        dists = np.linalg.norm(norm - ideal, axis=1)
        return points[int(np.argmin(dists))]

    def find_target_ratio_point(
        self,
        points_or_ratio=None,
        quality_to_diversity_ratio: float = 1.0,
        *,
        target_ratio: Optional[float] = None,
    ) -> Optional[TradeoffPoint]:
        """Point whose normalised Q/D ratio is closest to target."""
        if target_ratio is not None:
            quality_to_diversity_ratio = target_ratio
        if isinstance(points_or_ratio, (int, float)):
            quality_to_diversity_ratio = float(points_or_ratio)
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        elif points_or_ratio is None:
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        else:
            points = points_or_ratio
        if not points:
            return None
        arr = _to_array(points)
        norm = _normalise_objectives(arr)
        div_safe = np.maximum(norm[:, 1], _EPS)
        ratios = norm[:, 0] / div_safe
        diffs = np.abs(ratios - quality_to_diversity_ratio)
        return points[int(np.argmin(diffs))]

    # ------------------------------------------------------------------
    # Multi-objective scalarisation
    # ------------------------------------------------------------------

    def multi_objective_scalarization(
        self,
        points: Optional[Sequence[TradeoffPoint]] = None,
        method: str = "weighted_sum",
        weights: Optional[Tuple[float, float]] = None,
        ideal: Optional[NDArray[np.float64]] = None,
        rho: float = 0.05,
    ) -> Optional[TradeoffPoint]:
        """Scalarise and return the best point.

        Parameters
        ----------
        method : ``"weighted_sum"``, ``"tchebycheff"``, ``"augmented"``.
        weights : ``(w_quality, w_diversity)``.
        ideal : ideal point for Tchebycheff variants.
        rho : augmentation coefficient.
        """
        if points is None:
            points = self._pf.frontier if self._pf.frontier else self._pf._points
        if not points:
            return None
        arr = _to_array(points)
        norm = _normalise_objectives(arr)
        w = np.array(
            weights
            or (self.config.quality_weight, self.config.diversity_weight),
            dtype=np.float64,
        )
        w = w / (w.sum() + _EPS)
        z_star = (
            np.asarray(ideal, dtype=np.float64)
            if ideal is not None
            else np.ones(2)
        )

        if method == "weighted_sum":
            scores = norm @ w
            return points[int(np.argmax(scores))]

        if method == "tchebycheff":
            diffs = w * np.abs(z_star - norm)
            worst = np.max(diffs, axis=1)
            return points[int(np.argmin(worst))]

        if method == "augmented":
            diffs = w * np.abs(z_star - norm)
            worst = np.max(diffs, axis=1)
            aug = worst + rho * np.sum(diffs, axis=1)
            return points[int(np.argmin(aug))]

        raise ValueError(f"Unknown scalarisation method: {method!r}")


# ===================================================================
#  6.  TradeoffAnalyzer
# ===================================================================


class TradeoffAnalyzer:
    """High-level analyses comparing algorithms on the DQ tradeoff."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()
        self._pf = ParetoFrontier(self.config)
        self._opf = OperatingPointFinder(self.config)
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # Per-algorithm summary
    # ------------------------------------------------------------------

    def analyze_algorithm_tradeoffs(
        self,
        points,
    ) -> Dict[str, Any]:
        """Per-algorithm summary: frontier, hypervolume, knee, spread, etc.
        
        Accepts either a flat list of TradeoffPoints or a dict mapping
        algorithm names to lists of TradeoffPoints.
        """
        if isinstance(points, dict):
            by_algo = {k: list(v) for k, v in points.items()}
        else:
            by_algo: Dict[str, List[TradeoffPoint]] = {}
            for p in points:
                by_algo.setdefault(p.algorithm, []).append(p)

        results: Dict[str, Any] = {}
        for algo, pts in by_algo.items():
            frontier = self._pf.compute_frontier(pts)
            hv = self._pf.hypervolume(frontier)
            knee = self._opf.find_knee_point(frontier)
            spread = self._pf.frontier_spread(frontier)
            uniformity = self._pf.frontier_uniformity(frontier)
            auc = self._pf.area_under_frontier(frontier)
            cd = self._pf.crowding_distance(frontier)

            finite_cd = cd[np.isfinite(cd)]
            results[algo] = {
                "n_points": len(pts),
                "num_points": len(pts),
                "n_frontier": len(frontier),
                "frontier_size": len(frontier),
                "hypervolume": hv,
                "knee_point": knee.to_dict() if knee else None,
                "spread": spread,
                "uniformity": uniformity,
                "area_under_frontier": auc,
                "mean_crowding_distance": (
                    float(np.mean(finite_cd)) if finite_cd.size > 0 else 0.0
                ),
                "frontier_points": [fp.to_dict() for fp in frontier],
            }
        return results

    # ------------------------------------------------------------------
    # Tradeoff curve construction
    # ------------------------------------------------------------------

    def compute_tradeoff_curves(
        self,
        data_or_fn=None,
        param_name: Optional[str] = None,
        param_values: Optional[Sequence[float]] = None,
        base_config: Optional[Dict[str, Any]] = None,
        *,
        n_points: int = 200,
    ):
        """Compute tradeoff curves.
        
        Accepts either:
        - A dict mapping algorithm names to lists of TradeoffPoints
        - A callable + param_name + param_values for parameter sweep
        """
        if isinstance(data_or_fn, dict):
            result: Dict[str, NDArray[np.float64]] = {}
            for algo, pts in data_or_fn.items():
                if len(pts) == 0:
                    result[algo] = np.empty((0, 2), dtype=np.float64)
                    continue
                pf = ParetoFrontier(config=self.config)
                frontier = pf.compute_frontier(pts)
                if len(frontier) < 2:
                    if len(frontier) == 1:
                        result[algo] = np.array([[frontier[0].quality_score, frontier[0].diversity_score]])
                    else:
                        result[algo] = np.empty((0, 2), dtype=np.float64)
                    continue
                interp = pf.interpolate_frontier(frontier, n_interp=n_points)
                result[algo] = interp
            return result
        # Original callable-based interface
        evaluate_fn = data_or_fn
        base = dict(base_config or {})
        curve: List[TradeoffPoint] = []
        for val in param_values:
            cfg = dict(base)
            cfg[param_name] = val
            try:
                tp = evaluate_fn(cfg)
                curve.append(tp)
            except Exception:
                logger.warning("Evaluation failed for %s=%s", param_name, val)
        return curve

    # ------------------------------------------------------------------
    # Frontier comparison
    # ------------------------------------------------------------------

    def compare_frontiers(
        self,
        frontier_a: Sequence[TradeoffPoint],
        frontier_b: Sequence[TradeoffPoint],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict[str, Any]:
        """Compare two frontiers: HV, GD, IGD, coverage, eps-indicator."""
        arr_a = _to_array(frontier_a)
        arr_b = _to_array(frontier_b)

        if arr_a.shape[0] > 0 and arr_b.shape[0] > 0:
            combined = np.vstack([arr_a, arr_b])
        elif arr_a.shape[0] > 0:
            combined = arr_a
        else:
            combined = arr_b

        ref = combined.min(axis=0) - 0.01 * (
            combined.max(axis=0) - combined.min(axis=0) + _EPS
        )

        hv_a = (
            compute_dominated_hypervolume(arr_a, ref)
            if arr_a.shape[0] > 0
            else 0.0
        )
        hv_b = (
            compute_dominated_hypervolume(arr_b, ref)
            if arr_b.shape[0] > 0
            else 0.0
        )

        have_both = arr_a.shape[0] > 0 and arr_b.shape[0] > 0
        gd_ab = (
            compute_generational_distance(arr_a, arr_b)
            if have_both
            else float("inf")
        )
        gd_ba = (
            compute_generational_distance(arr_b, arr_a)
            if have_both
            else float("inf")
        )
        igd_a = (
            compute_inverted_generational_distance(arr_a, arr_b)
            if have_both
            else float("inf")
        )
        igd_b = (
            compute_inverted_generational_distance(arr_b, arr_a)
            if have_both
            else float("inf")
        )

        cov_a = self._coverage(arr_a, arr_b)
        cov_b = self._coverage(arr_b, arr_a)
        eps_ind = self._epsilon_indicator(arr_a, arr_b)

        return {
            f"hypervolume_{label_a.lower()}": hv_a,
            f"hypervolume_{label_b.lower()}": hv_b,
            "hypervolume_diff": hv_a - hv_b,
            "hypervolume_ratio": hv_a / max(hv_b, _EPS),
            f"GD_{label_a}_to_{label_b}": gd_ab,
            f"GD_{label_b}_to_{label_a}": gd_ba,
            f"IGD_{label_a}": igd_a,
            f"IGD_{label_b}": igd_b,
            f"coverage_{label_a}_over_{label_b}": cov_a,
            f"coverage_{label_b}_over_{label_a}": cov_b,
            "epsilon_indicator": eps_ind,
            "reference_point": ref.tolist(),
            f"spread_{label_a.lower()}": float(self._pf.frontier_spread(frontier_a)),
            f"spread_{label_b.lower()}": float(self._pf.frontier_spread(frontier_b)),
            f"uniformity_{label_a.lower()}": float(self._pf.frontier_uniformity(frontier_a)),
            f"uniformity_{label_b.lower()}": float(self._pf.frontier_uniformity(frontier_b)),
        }

    @staticmethod
    def _coverage(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """C(A,B): fraction of B dominated by at least one point in A."""
        if b.shape[0] == 0 or a.shape[0] == 0:
            return 0.0
        count = sum(
            1 for j in range(b.shape[0])
            if is_pareto_dominated(b[j], a, maximise=True)
        )
        return count / b.shape[0]

    @staticmethod
    def _epsilon_indicator(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> float:
        """Unary additive epsilon-indicator."""
        if b.shape[0] == 0:
            return 0.0
        if a.shape[0] == 0:
            return float("inf")
        eps_vals: List[float] = []
        for j in range(b.shape[0]):
            min_eps = float("inf")
            for i in range(a.shape[0]):
                needed = float(np.max(b[j] - a[i]))
                min_eps = min(min_eps, needed)
            eps_vals.append(min_eps)
        return float(max(eps_vals))

    # ------------------------------------------------------------------
    # Statistical frontier comparison
    # ------------------------------------------------------------------

    def statistical_frontier_comparison(
        self,
        points_a: Sequence[TradeoffPoint],
        points_b: Sequence[TradeoffPoint],
        label_a: str = "A",
        label_b: str = "B",
        n_bootstrap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bootstrap comparison with confidence intervals on HV diff."""
        n_boot = n_bootstrap or self.config.num_bootstrap
        arr_a = _to_array(points_a)
        arr_b = _to_array(points_b)

        if arr_a.shape[0] == 0 or arr_b.shape[0] == 0:
            return {
                "mean_hv_diff": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        combined = np.vstack([arr_a, arr_b])
        ref = combined.min(axis=0) - 0.01 * (
            combined.max(axis=0) - combined.min(axis=0) + _EPS
        )

        hv_diffs = np.empty(n_boot, dtype=np.float64)
        na, nb = arr_a.shape[0], arr_b.shape[0]

        for b_idx in range(n_boot):
            idx_a = self._rng.integers(0, na, size=na)
            idx_b = self._rng.integers(0, nb, size=nb)
            sa = arr_a[idx_a]
            sb = arr_b[idx_b]
            ma = _non_dominated_mask(sa)
            mb = _non_dominated_mask(sb)
            ha = compute_dominated_hypervolume(sa[ma], ref)
            hb = compute_dominated_hypervolume(sb[mb], ref)
            hv_diffs[b_idx] = ha - hb

        alpha = 1.0 - self.config.confidence_level
        ci_lo = float(np.percentile(hv_diffs, 100 * alpha / 2))
        ci_hi = float(np.percentile(hv_diffs, 100 * (1 - alpha / 2)))
        mean_diff = float(np.mean(hv_diffs))

        if mean_diff >= 0:
            p_val = float(np.mean(hv_diffs < 0)) * 2
        else:
            p_val = float(np.mean(hv_diffs > 0)) * 2
        p_val = min(p_val, 1.0)

        sd = (
            float(np.std(hv_diffs, ddof=1)) if n_boot > 1 else _EPS
        )
        cohens_d = mean_diff / max(sd, _EPS)

        bands = self._bootstrap_frontier_bands(
            arr_a, arr_b, ref, n_boot, alpha
        )

        return {
            "mean_hv_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
            "significant": (ci_lo > 0) or (ci_hi < 0),
            "cohens_d": cohens_d,
            "n_bootstrap": n_boot,
            "frontier_bands": bands,
            "label_a": label_a,
            "label_b": label_b,
        }

    def _bootstrap_frontier_bands(
        self,
        arr_a: NDArray[np.float64],
        arr_b: NDArray[np.float64],
        ref: NDArray[np.float64],
        n_boot: int,
        alpha: float,
    ) -> Dict[str, Any]:
        """Point-wise confidence bands on the interpolated frontier."""
        n_grid = 50
        q_lo = min(float(arr_a[:, 0].min()), float(arr_b[:, 0].min()))
        q_hi = max(float(arr_a[:, 0].max()), float(arr_b[:, 0].max()))
        q_grid = np.linspace(q_lo, q_hi, n_grid)

        cap = min(n_boot, 200)
        bands_a = np.full((cap, n_grid), np.nan)
        bands_b = np.full((cap, n_grid), np.nan)
        na, nb = arr_a.shape[0], arr_b.shape[0]

        for b in range(cap):
            ia = self._rng.integers(0, na, size=na)
            ib = self._rng.integers(0, nb, size=nb)
            sa, sb = arr_a[ia], arr_b[ib]
            ma, mb = _non_dominated_mask(sa), _non_dominated_mask(sb)
            bands_a[b] = self._interp_on_grid(sa[ma], q_grid)
            bands_b[b] = self._interp_on_grid(sb[mb], q_grid)

        def _band(mat: NDArray) -> Dict[str, List[float]]:
            lo = np.nanpercentile(mat, 100 * alpha / 2, axis=0)
            hi = np.nanpercentile(mat, 100 * (1 - alpha / 2), axis=0)
            med = np.nanmedian(mat, axis=0)
            return {
                "lower": np.nan_to_num(lo).tolist(),
                "upper": np.nan_to_num(hi).tolist(),
                "median": np.nan_to_num(med).tolist(),
            }

        return {
            "quality_grid": q_grid.tolist(),
            "band_a": _band(bands_a),
            "band_b": _band(bands_b),
        }

    @staticmethod
    def _interp_on_grid(
        front: NDArray[np.float64], q_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if front.shape[0] == 0:
            return np.full(q_grid.shape, np.nan)
        if front.shape[0] == 1:
            out = np.full(q_grid.shape, np.nan)
            out[np.argmin(np.abs(q_grid - front[0, 0]))] = front[0, 1]
            return out
        order = np.argsort(front[:, 0])
        xs, ys = front[order, 0], front[order, 1]
        ux, inv = np.unique(xs, return_inverse=True)
        uy = np.zeros_like(ux)
        np.add.at(uy, inv, ys)
        cnts = np.zeros_like(ux)
        np.add.at(cnts, inv, 1)
        uy /= cnts
        if len(ux) < 2:
            return np.full(q_grid.shape, uy[0])
        fn = sp_interp.interp1d(
            ux, uy, kind="linear", bounds_error=False, fill_value=np.nan
        )
        return fn(q_grid)

    # ------------------------------------------------------------------
    # Marginal rates
    # ------------------------------------------------------------------

    def compute_marginal_rates(
        self,
        points: Sequence[TradeoffPoint],
        n_interp: int = 200,
    ) -> List[float]:
        """Compute marginal rates dDiversity/dQuality between consecutive frontier points."""
        frontier = self._pf.compute_frontier(points)
        if len(frontier) < 2:
            return []
        arr = _to_array(frontier)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        rates = []
        for i in range(len(arr) - 1):
            dq = arr[i + 1, 0] - arr[i, 0]
            dd = arr[i + 1, 1] - arr[i, 1]
            if abs(dq) < _EPS:
                rates.append(0.0)
            else:
                rates.append(float(dd / dq))
        return rates

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        points_or_fn=None,
        base_config=None,
        param_ranges=None,
        n_samples: int = 100,
        *,
        quality_weights: Optional[List[float]] = None,
    ):
        """Sensitivity analysis.
        
        Can be called as:
        - sensitivity_analysis(points, quality_weights=[...])
        - sensitivity_analysis(evaluate_fn, base_config, param_ranges)
        """
        # New interface: points + quality_weights
        if isinstance(points_or_fn, (list, tuple)) or (
            points_or_fn is not None and not callable(points_or_fn) and base_config is None
        ):
            points = points_or_fn if points_or_fn is not None else []
            if quality_weights is None:
                quality_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
            results = []
            for w in quality_weights:
                if len(points) == 0:
                    results.append({
                        "quality_weight": w,
                        "selected_point": None,
                        "quality": None,
                        "diversity": None,
                    })
                    continue
                arr = _to_array(points)
                norm = _normalise_objectives(arr)
                scores = w * norm[:, 0] + (1.0 - w) * norm[:, 1]
                best_idx = int(np.argmax(scores))
                pt = points[best_idx]
                results.append({
                    "quality_weight": w,
                    "selected_point": pt,
                    "quality": pt.quality_score,
                    "diversity": pt.diversity_score,
                })
            return results
        
        # Original interface: evaluate_fn, base_config, param_ranges
        evaluate_fn = points_or_fn
        results_dict: Dict[str, Any] = {}
        base_point = evaluate_fn(dict(base_config))
        baseline_q = base_point.quality_score
        baseline_d = base_point.diversity_score

        for pname, (lo, hi) in param_ranges.items():
            values = np.linspace(lo, hi, n_samples)
            qualities = np.empty(n_samples)
            diversities = np.empty(n_samples)

            for i, val in enumerate(values):
                cfg = dict(base_config)
                cfg[pname] = float(val)
                try:
                    tp = evaluate_fn(cfg)
                    qualities[i] = tp.quality_score
                    diversities[i] = tp.diversity_score
                except Exception:
                    qualities[i] = np.nan
                    diversities[i] = np.nan

            valid = np.isfinite(qualities) & np.isfinite(diversities)
            if valid.sum() < 3:
                results_dict[pname] = {
                    "values": values.tolist(),
                    "qualities": qualities.tolist(),
                    "diversities": diversities.tolist(),
                    "quality_sensitivity": 0.0,
                    "diversity_sensitivity": 0.0,
                    "quality_elasticity": 0.0,
                    "diversity_elasticity": 0.0,
                }
                continue

            q_var = float(np.nanvar(qualities))
            d_var = float(np.nanvar(diversities))
            p_var = float(np.var(values[valid]))
            q_sens = q_var / max(p_var, _EPS)
            d_sens = d_var / max(p_var, _EPS)

            mid = n_samples // 2
            dp = float(
                values[min(mid + 1, n_samples - 1)]
                - values[max(mid - 1, 0)]
            )
            if abs(dp) > _EPS and valid[mid]:
                dq = float(
                    qualities[min(mid + 1, n_samples - 1)]
                    - qualities[max(mid - 1, 0)]
                )
                dd = float(
                    diversities[min(mid + 1, n_samples - 1)]
                    - diversities[max(mid - 1, 0)]
                )
                mv = float(values[mid])
                q_el = (dq / max(abs(baseline_q), _EPS)) / (
                    dp / max(abs(mv), _EPS)
                )
                d_el = (dd / max(abs(baseline_d), _EPS)) / (
                    dp / max(abs(mv), _EPS)
                )
            else:
                q_el, d_el = 0.0, 0.0

            q_corr = float(
                sp_stats.spearmanr(values[valid], qualities[valid]).statistic
            )
            d_corr = float(
                sp_stats.spearmanr(values[valid], diversities[valid]).statistic
            )

            results_dict[pname] = {
                "values": values.tolist(),
                "qualities": qualities.tolist(),
                "diversities": diversities.tolist(),
                "quality_sensitivity": q_sens,
                "diversity_sensitivity": d_sens,
                "quality_elasticity": q_el,
                "diversity_elasticity": d_el,
                "quality_correlation": q_corr,
                "diversity_correlation": d_corr,
            }

        return results_dict


# ===================================================================
#  7.  QualityConstrainedDiversityOptimizer
# ===================================================================


class QualityConstrainedDiversityOptimizer:
    """Optimise diversity subject to a minimum-quality constraint."""

    def __init__(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], TradeoffPoint],
        min_quality: float,
        config: Optional[TradeoffConfig] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.evaluate_fn = evaluate_fn
        self.min_quality = min_quality
        self.config = config or TradeoffConfig()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(self.config.seed)
        self._history: List[TradeoffPoint] = []

    @property
    def history(self) -> List[TradeoffPoint]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def grid_search_with_constraint(
        self,
        param_grids: Dict[str, Sequence[float]],
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradeoffPoint]:
        """Exhaustive grid search; return best feasible point."""
        base = dict(base_config or {})
        names = list(param_grids.keys())
        vlists = [param_grids[k] for k in names]

        best: Optional[TradeoffPoint] = None
        for combo in itertools.product(*vlists):
            cfg = dict(base)
            for nm, vl in zip(names, combo):
                cfg[nm] = vl
            try:
                tp = self.evaluate_fn(cfg)
            except Exception:
                continue
            self._history.append(tp)
            if tp.quality_score < self.min_quality - _EPS:
                continue
            if best is None or tp.diversity_score > best.diversity_score:
                best = tp
        return best

    # ------------------------------------------------------------------
    # Bayesian optimisation step
    # ------------------------------------------------------------------

    def bayesian_optimization_step(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_initial: int = 10,
        n_iter: int = 50,
        base_config: Optional[Dict[str, Any]] = None,
        *,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        n_candidates: Optional[int] = None,
    ) -> Optional[TradeoffPoint]:
        """Simple BO loop with RBF surrogate + feasibility penalty."""
        if param_ranges is not None:
            param_bounds = param_ranges
        if n_candidates is not None:
            n_initial = n_candidates
            n_iter = 0
        base = dict(base_config or {})
        pnames = sorted(param_bounds.keys())
        dim = len(pnames)
        lo = np.array([param_bounds[k][0] for k in pnames])
        hi = np.array([param_bounds[k][1] for k in pnames])

        # Latin-hypercube initial design
        X_init = self._rng.uniform(0, 1, size=(n_initial, dim))
        X_init = lo + X_init * (hi - lo)

        X_all = np.empty((0, dim), dtype=np.float64)
        Y_div = np.empty(0, dtype=np.float64)
        Y_qual = np.empty(0, dtype=np.float64)

        def _eval(x: NDArray) -> Tuple[float, float]:
            cfg = dict(base)
            for i, nm in enumerate(pnames):
                cfg[nm] = float(x[i])
            tp = self.evaluate_fn(cfg)
            self._history.append(tp)
            return tp.diversity_score, tp.quality_score

        for x in X_init:
            try:
                d, q = _eval(x)
            except Exception:
                d, q = 0.0, 0.0
            X_all = np.vstack([X_all, x.reshape(1, -1)])
            Y_div = np.append(Y_div, d)
            Y_qual = np.append(Y_qual, q)

        for _ in range(n_iter):
            x_next = self._bo_next_point(
                X_all, Y_div, Y_qual, lo, hi, pnames
            )
            try:
                d, q = _eval(x_next)
            except Exception:
                d, q = 0.0, 0.0
            X_all = np.vstack([X_all, x_next.reshape(1, -1)])
            Y_div = np.append(Y_div, d)
            Y_qual = np.append(Y_qual, q)

        feasible = Y_qual >= self.min_quality - _EPS
        if not np.any(feasible):
            logger.warning("No feasible point found in BO.")
            closest = int(np.argmin(np.abs(Y_qual - self.min_quality)))
            return self._history[closest] if self._history else None

        best_idx = int(np.argmax(np.where(feasible, Y_div, -np.inf)))
        return self._history[best_idx]

    def _bo_next_point(
        self,
        X: NDArray, Y_div: NDArray, Y_qual: NDArray,
        lo: NDArray, hi: NDArray, pnames: List[str],
    ) -> NDArray[np.float64]:
        """Select next query via random-candidate acquisition."""
        dim = X.shape[1]
        span = hi - lo + _EPS
        X_norm = (X - lo) / span

        try:
            from scipy.interpolate import RBFInterpolator
            rbf_d = RBFInterpolator(
                X_norm, Y_div, kernel="thin_plate_spline", smoothing=1e-3
            )
            rbf_q = RBFInterpolator(
                X_norm, Y_qual, kernel="thin_plate_spline", smoothing=1e-3
            )
        except Exception:
            return self._rng.uniform(lo, hi)

        n_cand = max(200, 20 * dim)
        cands = self._rng.uniform(0, 1, size=(n_cand, dim))
        pd = rbf_d(cands)
        pq = rbf_q(cands)

        bonus = np.where(
            pq >= self.min_quality, 0.0,
            -10.0 * (self.min_quality - pq),
        )
        acq = pd + bonus
        best_c = int(np.argmax(acq))
        return lo + cands[best_c] * span

    # ------------------------------------------------------------------
    # Successive halving
    # ------------------------------------------------------------------

    def successive_halving(
        self,
        param_bounds_or_configs=None,
        n_initial: int = 64,
        halving_rounds: int = 4,
        base_config: Optional[Dict[str, Any]] = None,
        *,
        n_rounds: Optional[int] = None,
    ) -> Optional[TradeoffPoint]:
        """Successive-halving (Hyperband-style) search."""
        if n_rounds is not None:
            halving_rounds = n_rounds
        
        # New interface: list of config dicts
        if isinstance(param_bounds_or_configs, list):
            configs_list = param_bounds_or_configs
            if len(configs_list) == 0:
                return None
            alive: List[Tuple[Dict, TradeoffPoint]] = []
            for cfg in configs_list:
                try:
                    tp = self.evaluate_fn(dict(cfg))
                    self._history.append(tp)
                    alive.append((cfg, tp))
                except Exception:
                    continue
            for rnd in range(halving_rounds):
                if len(alive) <= 1:
                    break
                scores = np.array([
                    tp.diversity_score
                    if tp.quality_score >= self.min_quality - _EPS
                    else tp.diversity_score - 100.0 * (self.min_quality - tp.quality_score)
                    for _, tp in alive
                ])
                n_keep = max(1, len(alive) // 2)
                top = np.argsort(-scores)[:n_keep]
                alive = [alive[i] for i in top]
            best: Optional[TradeoffPoint] = None
            for _, tp in alive:
                if tp.quality_score >= self.min_quality - _EPS:
                    if best is None or tp.diversity_score > best.diversity_score:
                        best = tp
            if best is None and alive:
                best = min(alive, key=lambda pair: abs(pair[1].quality_score - self.min_quality))[1]
            return best

        # Original interface: param_bounds dict
        param_bounds = param_bounds_or_configs
        base = dict(base_config or {})
        pnames = sorted(param_bounds.keys())
        dim = len(pnames)
        lo = np.array([param_bounds[k][0] for k in pnames])
        hi = np.array([param_bounds[k][1] for k in pnames])

        configs = self._rng.uniform(lo, hi, size=(n_initial, dim))
        alive: List[Tuple[NDArray, TradeoffPoint]] = []
        for x in configs:
            cfg = dict(base)
            for i, nm in enumerate(pnames):
                cfg[nm] = float(x[i])
            try:
                tp = self.evaluate_fn(cfg)
                self._history.append(tp)
                alive.append((x, tp))
            except Exception:
                continue

        for rnd in range(halving_rounds):
            if len(alive) <= 1:
                break
            scores = np.array([
                tp.diversity_score
                if tp.quality_score >= self.min_quality - _EPS
                else tp.diversity_score - 100.0 * (self.min_quality - tp.quality_score)
                for _, tp in alive
            ])
            n_keep = max(1, len(alive) // 2)
            top = np.argsort(-scores)[:n_keep]
            alive = [alive[i] for i in top]

            scale = 0.5 ** (rnd + 1)
            new_alive: List[Tuple[NDArray, TradeoffPoint]] = []
            for x_old, tp_old in alive:
                new_alive.append((x_old, tp_old))
                noise = self._rng.normal(0, scale, size=dim) * (hi - lo)
                x_new = np.clip(x_old + noise, lo, hi)
                cfg = dict(base)
                for i, nm in enumerate(pnames):
                    cfg[nm] = float(x_new[i])
                try:
                    tp_new = self.evaluate_fn(cfg)
                    self._history.append(tp_new)
                    new_alive.append((x_new, tp_new))
                except Exception:
                    pass
            alive = new_alive

        best: Optional[TradeoffPoint] = None
        for _, tp in alive:
            if tp.quality_score >= self.min_quality - _EPS:
                if best is None or tp.diversity_score > best.diversity_score:
                    best = tp
        if best is None and alive:
            best = min(
                alive,
                key=lambda pair: abs(pair[1].quality_score - self.min_quality),
            )[1]
        return best

    # ------------------------------------------------------------------
    # Random search with pruning
    # ------------------------------------------------------------------

    def random_search_with_pruning(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_samples: int = 200,
        prune_fraction: float = 0.5,
        base_config: Optional[Dict[str, Any]] = None,
        *,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        n_iterations: Optional[int] = None,
        prune_threshold: Optional[float] = None,
    ) -> Optional[TradeoffPoint]:
        """Random search that progressively shrinks the search region."""
        if param_ranges is not None:
            param_bounds = param_ranges
        if n_iterations is not None:
            n_samples = n_iterations
        if prune_threshold is not None:
            prune_fraction = prune_threshold
        base = dict(base_config or {})
        pnames = sorted(param_bounds.keys())
        dim = len(pnames)
        lo = np.array([param_bounds[k][0] for k in pnames], dtype=np.float64)
        hi = np.array([param_bounds[k][1] for k in pnames], dtype=np.float64)

        best: Optional[TradeoffPoint] = None
        cur_lo, cur_hi = lo.copy(), hi.copy()

        n_rounds = max(1, int(math.log2(max(n_samples, 2))))
        per_round = max(1, n_samples // n_rounds)

        for rnd in range(n_rounds):
            xs = self._rng.uniform(cur_lo, cur_hi, size=(per_round, dim))
            rr: List[Tuple[NDArray, TradeoffPoint]] = []

            for x in xs:
                cfg = dict(base)
                for i, nm in enumerate(pnames):
                    cfg[nm] = float(x[i])
                try:
                    tp = self.evaluate_fn(cfg)
                    self._history.append(tp)
                    rr.append((x, tp))
                    if tp.quality_score >= self.min_quality - _EPS:
                        if best is None or tp.diversity_score > best.diversity_score:
                            best = tp
                except Exception:
                    continue

            if len(rr) < 3:
                continue

            scores = np.array([
                tp.diversity_score
                if tp.quality_score >= self.min_quality - _EPS
                else tp.diversity_score - 100.0 * (self.min_quality - tp.quality_score)
                for _, tp in rr
            ])
            nk = max(2, int(len(rr) * prune_fraction))
            top_idx = np.argsort(-scores)[:nk]
            top_xs = np.array([rr[i][0] for i in top_idx])

            margin = 0.1 * (cur_hi - cur_lo)
            cur_lo = np.maximum(lo, top_xs.min(axis=0) - margin)
            cur_hi = np.minimum(hi, top_xs.max(axis=0) + margin)
            degen = cur_lo >= cur_hi - _EPS
            cur_lo[degen] = lo[degen]
            cur_hi[degen] = hi[degen]

        return best


# ===================================================================
#  8.  TradeoffVisualizer (data preparation, not rendering)
# ===================================================================


class TradeoffVisualizer:
    """Prepare visualisation-ready data structures (no matplotlib)."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()
        self._pf = ParetoFrontier(self.config)

    # ------------------------------------------------------------------
    # Frontier scatter + curve
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_frontier_plot_data(
        points_or_self=None,
        frontier_or_points=None,
        n_interp: int = 200,
        show_dominated: bool = True,
    ) -> Dict[str, Any]:
        """Data for a 2-D scatter with frontier curve overlaid."""
        # Handle both static and instance call patterns
        if isinstance(points_or_self, TradeoffVisualizer):
            # Instance method call: self, points, ...
            self_obj = points_or_self
            points = frontier_or_points
            frontier = self_obj._pf.compute_frontier(points)
        elif frontier_or_points is not None:
            # Static call: TradeoffVisualizer.prepare_frontier_plot_data(pts, frontier)
            points = points_or_self
            frontier = frontier_or_points
        else:
            points = points_or_self or []
            frontier = []

        all_q = [p.quality_score for p in points]
        all_d = [p.diversity_score for p in points]
        
        # Sort frontier by quality
        f_sorted = sorted(frontier, key=lambda p: p.quality_score)
        fq = [p.quality_score for p in f_sorted]
        fd = [p.diversity_score for p in f_sorted]

        return {
            "all_quality": all_q,
            "all_diversity": all_d,
            "frontier_quality": fq,
            "frontier_diversity": fd,
            "n_total": len(points),
            "n_frontier": len(frontier),
        }

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_heatmap_data(
        points_or_self=None,
        points_arg=None,
        value_fn=None,
        resolution=None,
        aggregation: str = "mean",
        *,
        n_bins: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bin points onto a 2-D grid and aggregate."""
        if isinstance(points_or_self, TradeoffVisualizer):
            points = points_arg
            config = points_or_self.config
        else:
            points = points_or_self or []
            config = TradeoffConfig()
        
        if n_bins is not None:
            res = n_bins
        elif resolution is not None:
            res = resolution
        else:
            res = 10

        if value_fn is None:
            value_fn = lambda p: p.quality_score + p.diversity_score

        if len(points) == 0:
            q_edges = np.linspace(0, 1, res + 1)
            d_edges = np.linspace(0, 1, res + 1)
            return {
                "quality_edges": q_edges.tolist(),
                "diversity_edges": d_edges.tolist(),
                "counts": [[0] * res for _ in range(res)],
                "aggregation": aggregation,
            }

        arr = _to_array(points)
        q_min, q_max = float(arr[:, 0].min()), float(arr[:, 0].max())
        d_min, d_max = float(arr[:, 1].min()), float(arr[:, 1].max())
        # Add small margin to ensure all points are within bins
        q_margin = max((q_max - q_min) * 0.001, 1e-10)
        d_margin = max((d_max - d_min) * 0.001, 1e-10)
        q_edges = np.linspace(q_min - q_margin, q_max + q_margin, res + 1)
        d_edges = np.linspace(d_min - d_margin, d_max + d_margin, res + 1)

        counts = [[0] * res for _ in range(res)]

        for idx in range(len(points)):
            qi = int(np.clip(np.searchsorted(q_edges, arr[idx, 0], side="right") - 1, 0, res - 1))
            di = int(np.clip(np.searchsorted(d_edges, arr[idx, 1], side="right") - 1, 0, res - 1))
            counts[qi][di] += 1

        return {
            "quality_edges": q_edges.tolist(),
            "diversity_edges": d_edges.tolist(),
            "counts": counts,
            "aggregation": aggregation,
        }

    # ------------------------------------------------------------------
    # Radar chart
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_radar_chart_data(
        points_or_self=None,
        metrics_arg=None,
    ) -> Dict[str, Any]:
        """Per-algorithm radar chart data."""
        if isinstance(points_or_self, TradeoffVisualizer):
            # Instance call: self, points, metrics
            metrics_dict = metrics_arg
        else:
            # Static call: TradeoffVisualizer.prepare_radar_chart_data(metrics)
            metrics_dict = points_or_self
        
        if not metrics_dict:
            return {"algorithms": [], "axes": [], "values": {}}
        
        algorithms = sorted(metrics_dict.keys())
        # Get axes from first algorithm's metrics
        axes = list(next(iter(metrics_dict.values())).keys())
        values = {}
        for algo in algorithms:
            values[algo] = [metrics_dict[algo].get(a, 0.0) for a in axes]
        
        return {"algorithms": algorithms, "axes": axes, "values": values}

    # ------------------------------------------------------------------
    # Parallel coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_parallel_coordinates_data(
        points_or_self=None,
        points_arg=None,
        param_names=None,
        include_objectives: bool = True,
        *,
        extra_axes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Parallel-coordinates data: one polyline per point."""
        if isinstance(points_or_self, TradeoffVisualizer):
            points = points_arg or []
        else:
            points = points_or_self or []

        axes = ["quality", "diversity"]
        if extra_axes:
            axes.extend(extra_axes)

        data = []
        for p in points:
            entry = {
                "quality": p.quality_score,
                "diversity": p.diversity_score,
                "algorithm": p.algorithm,
            }
            if extra_axes:
                for ax in extra_axes:
                    entry[ax] = p.metadata.get(ax, 0.0)
            data.append(entry)

        return {"axes": axes, "data": data, "num_points": len(points)}


# ===================================================================
#  9.  Extended analysis utilities
# ===================================================================


class FrontierDominanceAnalyzer:
    """Detailed dominance-relationship analysis."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()

    def dominance_matrix(
        self, points: Sequence[TradeoffPoint]
    ) -> NDArray[np.int8]:
        """(N,N) matrix: M[i,j]=1 if i dominates j, -1 if j dom i, 0 otherwise."""
        arr = _to_array(points)
        n = arr.shape[0]
        mat = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                rel = _dominance_cmp(arr[i], arr[j], maximise=True)
                mat[i, j] = rel
                mat[j, i] = -rel
        return mat

    def dominance_counts(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, Any]:
        """Per-point dominance statistics."""
        mat = self.dominance_matrix(points)
        dom_count = np.sum(mat == 1, axis=1)
        domby_count = np.sum(mat == -1, axis=1)
        return {
            "dominates": dom_count.tolist(),
            "dominated_by": domby_count.tolist(),
            "non_dominated_fraction": float(np.mean(domby_count == 0)),
        }

    def layer_decomposition(
        self, points: Sequence[TradeoffPoint]
    ) -> List[List[int]]:
        """Successive non-dominated layers."""
        arr = _to_array(points)
        n = arr.shape[0]
        if n == 0:
            return []
        ranks = pareto_rank_assignment(arr, maximise=True)
        layers: List[List[int]] = []
        for r in range(int(ranks.max()) + 1):
            layer = [int(i) for i in range(n) if ranks[i] == r]
            if layer:
                layers.append(layer)
        return layers

    def attainment_surface(
        self,
        runs: Sequence[Sequence[TradeoffPoint]],
        quantile: float = 0.5,
        n_grid: int = 100,
    ) -> NDArray[np.float64]:
        """Empirical attainment surface at a given quantile."""
        all_q: List[float] = []
        for run in runs:
            for p in run:
                all_q.append(p.quality_score)
        if not all_q:
            return np.empty((0, 2), dtype=np.float64)

        q_grid = np.linspace(min(all_q), max(all_q), n_grid)
        pf = ParetoFrontier(self.config)
        div_at_q: List[List[float]] = [[] for _ in range(n_grid)]

        for run in runs:
            front = pf.compute_frontier(run)
            if not front:
                for gi in range(n_grid):
                    div_at_q[gi].append(0.0)
                continue
            curve = pf.interpolate_frontier(front, n_interp=n_grid)
            fn = sp_interp.interp1d(
                curve[:, 0], curve[:, 1],
                kind="linear", bounds_error=False, fill_value=0.0,
            )
            dv = fn(q_grid)
            for gi in range(n_grid):
                div_at_q[gi].append(float(dv[gi]))

        result = np.empty((n_grid, 2), dtype=np.float64)
        for gi in range(n_grid):
            result[gi, 0] = q_grid[gi]
            result[gi, 1] = float(np.quantile(div_at_q[gi], quantile))
        return result


class TradeoffMetrics:
    """Comprehensive scalar metrics summarising a DQ tradeoff."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()
        self._pf = ParetoFrontier(self.config)

    def compute_all_metrics(
        self,
        points: Sequence[TradeoffPoint],
        reference_frontier: Optional[Sequence[TradeoffPoint]] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of scalar metrics."""
        frontier = self._pf.compute_frontier(points)
        arr_all = _to_array(points)
        arr_front = _to_array(frontier)

        hv = self._pf.hypervolume(frontier)
        spread = self._pf.frontier_spread(frontier)
        uniformity = self._pf.frontier_uniformity(frontier)
        auc = self._pf.area_under_frontier(frontier)

        m: Dict[str, float] = {
            "num_points": float(len(points)),
            "frontier_size": float(len(frontier)),
            "frontier_fraction": len(frontier) / max(len(points), 1),
            "hypervolume": hv,
            "quality_spread": spread["quality_spread"],
            "diversity_spread": spread["diversity_spread"],
            "diagonal_spread": spread["diagonal_spread"],
            "uniformity_cv": uniformity,
            "area_under_frontier": auc,
        }

        if arr_all.shape[0] > 0:
            m["mean_quality"] = float(np.mean(arr_all[:, 0]))
            m["mean_diversity"] = float(np.mean(arr_all[:, 1]))
            m["max_quality"] = float(np.max(arr_all[:, 0]))
            m["max_diversity"] = float(np.max(arr_all[:, 1]))
            m["std_quality"] = float(np.std(arr_all[:, 0]))
            m["std_diversity"] = float(np.std(arr_all[:, 1]))
            mq, md = m["max_quality"], m["max_diversity"]
            m["harmonic_mean_max"] = 2 * mq * md / max(mq + md, _EPS)
            if arr_all.shape[0] > 2:
                corr, pval = sp_stats.pearsonr(arr_all[:, 0], arr_all[:, 1])
                m["quality_diversity_correlation"] = float(corr)
                m["quality_diversity_corr_pvalue"] = float(pval)
            else:
                m["quality_diversity_correlation"] = 0.0
                m["quality_diversity_corr_pvalue"] = 1.0

        if reference_frontier is not None:
            ref_arr = _to_array(reference_frontier)
            if arr_front.shape[0] > 0 and ref_arr.shape[0] > 0:
                m["generational_distance"] = compute_generational_distance(
                    arr_front, ref_arr
                )
                m["inverted_generational_distance"] = (
                    compute_inverted_generational_distance(arr_front, ref_arr)
                )
                m["spread_delta"] = compute_spread_metric(arr_front)

        return m

    def compute_tradeoff_efficiency(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, float]:
        """HV ratio relative to bounding box."""
        frontier = self._pf.compute_frontier(points)
        arr = _to_array(frontier)
        if arr.shape[0] == 0:
            return {"efficiency": 0.0, "hypervolume": 0.0, "bounding_box_area": 0.0}
        ref = arr.min(axis=0) - 0.01 * (arr.max(axis=0) - arr.min(axis=0) + _EPS)
        upper = arr.max(axis=0)
        bbox = float(np.prod(upper - ref))
        hv = compute_dominated_hypervolume(arr, ref)
        return {
            "efficiency": hv / max(bbox, _EPS),
            "hypervolume": hv,
            "bounding_box_area": bbox,
        }

    def compute_algorithm_rankings(
        self,
        points: Sequence[TradeoffPoint],
        metrics_to_rank: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Rank algorithms across multiple metrics."""
        analyzer = TradeoffAnalyzer(self.config)
        ar = analyzer.analyze_algorithm_tradeoffs(points)

        if metrics_to_rank is None:
            metrics_to_rank = [
                "hypervolume",
                "area_under_frontier",
                "uniformity",
                "frontier_size",
            ]

        algos = sorted(ar.keys())
        if not algos:
            return {
                "algorithms": [],
                "rankings": {},
                "aggregate_ranking": [],
            }

        per_metric: Dict[str, Dict[str, int]] = {}
        for mn in metrics_to_rank:
            vals = []
            for algo in algos:
                v = ar[algo].get(mn, 0.0)
                if isinstance(v, dict):
                    v = sum(v.values())
                vals.append(float(v))
            if mn in ("uniformity",):
                order = np.argsort(vals)
            else:
                order = np.argsort(vals)[::-1]
            ranks = np.empty(len(algos), dtype=int)
            for rank, idx in enumerate(order):
                ranks[idx] = rank + 1
            per_metric[mn] = {algos[i]: int(ranks[i]) for i in range(len(algos))}

        avg_ranks = {
            a: float(np.mean([per_metric[m_][a] for m_ in metrics_to_rank]))
            for a in algos
        }
        agg = sorted(algos, key=lambda a: avg_ranks[a])

        return {
            "algorithms": algos,
            "per_metric_ranks": per_metric,
            "average_rank": avg_ranks,
            "aggregate_ranking": agg,
        }


class ParameterSpaceExplorer:
    """Understand how parameters affect the DQ tradeoff."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def sobol_sensitivity(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], TradeoffPoint],
        param_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 256,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sobol first-order sensitivity indices (Saltelli method)."""
        base = dict(base_config or {})
        pnames = sorted(param_bounds.keys())
        d = len(pnames)
        lo = np.array([param_bounds[k][0] for k in pnames])
        hi = np.array([param_bounds[k][1] for k in pnames])

        A_u = self._rng.uniform(0, 1, size=(n_samples, d))
        B_u = self._rng.uniform(0, 1, size=(n_samples, d))
        A = lo + A_u * (hi - lo)
        B = lo + B_u * (hi - lo)

        def _eval_mat(X: NDArray) -> Tuple[NDArray, NDArray]:
            qo = np.empty(X.shape[0])
            do = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                cfg = dict(base)
                for j, nm in enumerate(pnames):
                    cfg[nm] = float(X[i, j])
                try:
                    tp = evaluate_fn(cfg)
                    qo[i] = tp.quality_score
                    do[i] = tp.diversity_score
                except Exception:
                    qo[i] = np.nan
                    do[i] = np.nan
            return qo, do

        qA, dA = _eval_mat(A)
        qB, dB = _eval_mat(B)
        vqA = float(np.nanvar(qA))
        vdA = float(np.nanvar(dA))

        s1q: Dict[str, float] = {}
        s1d: Dict[str, float] = {}
        for j, pn in enumerate(pnames):
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            qABj, dABj = _eval_mat(AB_j)

            ok = np.isfinite(qA) & np.isfinite(qABj) & np.isfinite(qB)
            if ok.sum() > 1 and vqA > _EPS:
                vi = float(np.mean(qB[ok] * (qABj[ok] - qA[ok])))
                s1q[pn] = max(0.0, vi / max(vqA, _EPS))
            else:
                s1q[pn] = 0.0

            ok_d = np.isfinite(dA) & np.isfinite(dABj) & np.isfinite(dB)
            if ok_d.sum() > 1 and vdA > _EPS:
                vi = float(np.mean(dB[ok_d] * (dABj[ok_d] - dA[ok_d])))
                s1d[pn] = max(0.0, vi / max(vdA, _EPS))
            else:
                s1d[pn] = 0.0

        return {
            "parameters": pnames,
            "first_order_quality": s1q,
            "first_order_diversity": s1d,
            "total_variance_quality": vqA,
            "total_variance_diversity": vdA,
        }

    def interaction_effects(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], TradeoffPoint],
        param_a: str,
        param_b: str,
        values_a: Sequence[float],
        values_b: Sequence[float],
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Two-way interaction on a grid of (param_a, param_b)."""
        base = dict(base_config or {})
        na, nb = len(values_a), len(values_b)
        qg = np.full((na, nb), np.nan)
        dg = np.full((na, nb), np.nan)

        for i, va in enumerate(values_a):
            for j, vb in enumerate(values_b):
                cfg = dict(base)
                cfg[param_a] = float(va)
                cfg[param_b] = float(vb)
                try:
                    tp = evaluate_fn(cfg)
                    qg[i, j] = tp.quality_score
                    dg[i, j] = tp.diversity_score
                except Exception:
                    pass

        def _interaction(grid: NDArray) -> float:
            valid = np.isfinite(grid)
            if valid.sum() < 4:
                return 0.0
            gm = float(np.nanmean(grid))
            rm = np.nanmean(grid, axis=1)
            cm = np.nanmean(grid, axis=0)
            ss_a = len(values_b) * float(np.nansum((rm - gm) ** 2))
            ss_b = len(values_a) * float(np.nansum((cm - gm) ** 2))
            ss_t = float(np.nansum((grid[valid] - gm) ** 2))
            ss_i = max(0.0, ss_t - ss_a - ss_b)
            return ss_i / max(ss_t, _EPS)

        return {
            "param_a": param_a,
            "param_b": param_b,
            "values_a": list(values_a),
            "values_b": list(values_b),
            "quality_grid": np.nan_to_num(qg).tolist(),
            "diversity_grid": np.nan_to_num(dg).tolist(),
            "interaction_strength_quality": _interaction(qg),
            "interaction_strength_diversity": _interaction(dg),
        }

    def morris_screening(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], TradeoffPoint],
        param_bounds: Dict[str, Tuple[float, float]],
        n_trajectories: int = 20,
        n_levels: int = 10,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Morris elementary-effects screening."""
        base = dict(base_config or {})
        pnames = sorted(param_bounds.keys())
        d = len(pnames)
        lo = np.array([param_bounds[k][0] for k in pnames])
        hi = np.array([param_bounds[k][1] for k in pnames])
        delta = 1.0 / (n_levels - 1) if n_levels > 1 else 1.0

        ee_q: Dict[str, List[float]] = {p: [] for p in pnames}
        ee_d: Dict[str, List[float]] = {p: [] for p in pnames}

        for _ in range(n_trajectories):
            x0g = self._rng.integers(0, n_levels, size=d)
            x_u = x0g / max(n_levels - 1, 1)
            x = lo + x_u * (hi - lo)

            cfg0 = dict(base)
            for j, nm in enumerate(pnames):
                cfg0[nm] = float(x[j])
            try:
                tp0 = evaluate_fn(cfg0)
            except Exception:
                continue

            perm = self._rng.permutation(d)
            cur_q, cur_d = tp0.quality_score, tp0.diversity_score
            cur_x = x.copy()

            for ji in perm:
                direction = 1.0 if self._rng.random() < 0.5 else -1.0
                step = direction * delta * (hi[ji] - lo[ji])
                new_val = float(np.clip(cur_x[ji] + step, lo[ji], hi[ji]))
                actual = new_val - cur_x[ji]
                if abs(actual) < _EPS:
                    continue
                x_new = cur_x.copy()
                x_new[ji] = new_val
                cfg = dict(base)
                for k, nm in enumerate(pnames):
                    cfg[nm] = float(x_new[k])
                try:
                    tp = evaluate_fn(cfg)
                except Exception:
                    continue
                ee_q[pnames[ji]].append(
                    (tp.quality_score - cur_q) / actual
                )
                ee_d[pnames[ji]].append(
                    (tp.diversity_score - cur_d) / actual
                )
                cur_q, cur_d = tp.quality_score, tp.diversity_score
                cur_x = x_new

        result: Dict[str, Any] = {"parameters": pnames}
        for pn in pnames:
            eqs = np.array(ee_q[pn]) if ee_q[pn] else np.array([0.0])
            eds = np.array(ee_d[pn]) if ee_d[pn] else np.array([0.0])
            result[pn] = {
                "mu_star_quality": float(np.mean(np.abs(eqs))),
                "sigma_quality": float(np.std(eqs)),
                "mu_star_diversity": float(np.mean(np.abs(eds))),
                "sigma_diversity": float(np.std(eds)),
                "n_effects": len(ee_q[pn]),
            }
        return result


class MultiRunAggregator:
    """Aggregate results from multiple independent runs."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()
        self._pf = ParetoFrontier(self.config)
        self._rng = np.random.default_rng(self.config.seed)

    def aggregate_runs(
        self, runs: Sequence[Sequence[TradeoffPoint]]
    ) -> Dict[str, Any]:
        """Compute summary statistics across multiple runs."""
        hvs: List[float] = []
        all_pts: List[TradeoffPoint] = []
        fsizes: List[int] = []

        for run in runs:
            front = self._pf.compute_frontier(run)
            hvs.append(self._pf.hypervolume(front))
            fsizes.append(len(front))
            all_pts.extend(run)

        merged = self._pf.compute_frontier(all_pts)
        hv_arr = np.array(hvs)
        return {
            "num_runs": len(runs),
            "per_run_hypervolume": hvs,
            "mean_hypervolume": float(np.mean(hv_arr)),
            "std_hypervolume": (
                float(np.std(hv_arr, ddof=1)) if len(hvs) > 1 else 0.0
            ),
            "median_hypervolume": float(np.median(hv_arr)),
            "merged_hypervolume": self._pf.hypervolume(merged),
            "merged_frontier_size": len(merged),
            "per_run_frontier_size": fsizes,
            "total_points": len(all_pts),
        }

    def bootstrap_hypervolume_ci(
        self,
        points: Sequence[TradeoffPoint],
        n_bootstrap: Optional[int] = None,
    ) -> Dict[str, float]:
        """Bootstrap confidence interval for hypervolume."""
        n_boot = n_bootstrap or self.config.num_bootstrap
        arr = _to_array(points)
        n = arr.shape[0]
        if n == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

        ref = arr.min(axis=0) - 0.01 * (arr.max(axis=0) - arr.min(axis=0) + _EPS)
        hvs = np.empty(n_boot)
        for b in range(n_boot):
            idx = self._rng.integers(0, n, size=n)
            s = arr[idx]
            m = _non_dominated_mask(s)
            hvs[b] = compute_dominated_hypervolume(s[m], ref)

        alpha = 1.0 - self.config.confidence_level
        return {
            "mean": float(np.mean(hvs)),
            "ci_lower": float(np.percentile(hvs, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(hvs, 100 * (1 - alpha / 2))),
            "std": float(np.std(hvs, ddof=1)),
        }

    def convergence_curve(
        self,
        points: Sequence[TradeoffPoint],
        step: int = 1,
    ) -> Dict[str, Any]:
        """Hypervolume as function of number of evaluated points."""
        n = len(points)
        if n == 0:
            return {"num_evaluations": [], "hypervolume": []}
        arr = _to_array(points)
        ref = arr.min(axis=0) - 0.01 * (arr.max(axis=0) - arr.min(axis=0) + _EPS)
        evals: List[int] = []
        hvs: List[float] = []
        for i in range(step, n + 1, step):
            sub = arr[:i]
            m = _non_dominated_mask(sub)
            hvs.append(compute_dominated_hypervolume(sub[m], ref))
            evals.append(i)
        return {"num_evaluations": evals, "hypervolume": hvs}


class TradeoffRegression:
    """Fit parametric models to the DQ tradeoff curve."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()

    def fit_power_law(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, Any]:
        """Fit diversity = a * quality^b + c."""
        arr = _to_array(points)
        if arr.shape[0] < 3:
            return {"a": 0.0, "b": 0.0, "c": 0.0, "r_squared": 0.0, "success": False}
        q = np.maximum(arr[:, 0], _EPS)
        d = arr[:, 1]
        try:
            popt, _ = sp_opt.curve_fit(
                lambda x, a, b, c: a * np.power(x, b) + c,
                q, d, p0=[1.0, -1.0, 0.0], maxfev=5000,
            )
            dp = popt[0] * np.power(q, popt[1]) + popt[2]
            ssr = float(np.sum((d - dp) ** 2))
            sst = float(np.sum((d - np.mean(d)) ** 2))
            r2 = 1.0 - ssr / max(sst, _EPS)
            return {"a": float(popt[0]), "b": float(popt[1]),
                    "c": float(popt[2]), "r_squared": r2, "success": True}
        except Exception as e:
            logger.warning("Power-law fit failed: %s", e)
            return {"a": 0.0, "b": 0.0, "c": 0.0, "r_squared": 0.0, "success": False}

    def fit_exponential(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, Any]:
        """Fit diversity = a * exp(-b * quality) + c."""
        arr = _to_array(points)
        if arr.shape[0] < 3:
            return {"a": 0.0, "b": 0.0, "c": 0.0, "r_squared": 0.0, "success": False}
        q, d = arr[:, 0], arr[:, 1]
        try:
            popt, _ = sp_opt.curve_fit(
                lambda x, a, b, c: a * np.exp(-b * x) + c,
                q, d, p0=[1.0, 1.0, 0.0], maxfev=5000,
            )
            dp = popt[0] * np.exp(-popt[1] * q) + popt[2]
            ssr = float(np.sum((d - dp) ** 2))
            sst = float(np.sum((d - np.mean(d)) ** 2))
            r2 = 1.0 - ssr / max(sst, _EPS)
            return {"a": float(popt[0]), "b": float(popt[1]),
                    "c": float(popt[2]), "r_squared": r2, "success": True}
        except Exception as e:
            logger.warning("Exponential fit failed: %s", e)
            return {"a": 0.0, "b": 0.0, "c": 0.0, "r_squared": 0.0, "success": False}

    def fit_linear(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, Any]:
        """Fit diversity = a * quality + b."""
        arr = _to_array(points)
        if arr.shape[0] < 2:
            return {"a": 0.0, "b": 0.0, "r_squared": 0.0, "success": False}
        slope, intercept, r, p, se = sp_stats.linregress(arr[:, 0], arr[:, 1])
        return {
            "a": float(slope), "b": float(intercept),
            "r_squared": float(r ** 2), "p_value": float(p),
            "std_err": float(se), "success": True,
        }

    def fit_logistic(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, Any]:
        """Fit diversity = L / (1 + exp(-k*(quality - q0)))."""
        arr = _to_array(points)
        if arr.shape[0] < 4:
            return {"L": 0.0, "k": 0.0, "q0": 0.0, "r_squared": 0.0, "success": False}
        q, d = arr[:, 0], arr[:, 1]
        try:
            p0 = [float(np.max(d)), 1.0, float(np.median(q))]
            popt, _ = sp_opt.curve_fit(
                lambda x, L, k, q0: L / (1.0 + np.exp(-k * (x - q0))),
                q, d, p0=p0, maxfev=5000,
            )
            dp = popt[0] / (1.0 + np.exp(-popt[1] * (q - popt[2])))
            ssr = float(np.sum((d - dp) ** 2))
            sst = float(np.sum((d - np.mean(d)) ** 2))
            r2 = 1.0 - ssr / max(sst, _EPS)
            return {"L": float(popt[0]), "k": float(popt[1]),
                    "q0": float(popt[2]), "r_squared": r2, "success": True}
        except Exception as e:
            logger.warning("Logistic fit failed: %s", e)
            return {"L": 0.0, "k": 0.0, "q0": 0.0, "r_squared": 0.0, "success": False}

    def best_fit(
        self, points: Sequence[TradeoffPoint]
    ) -> Dict[str, Any]:
        """Try all models, return best by R-squared."""
        results = {
            "linear": self.fit_linear(points),
            "power_law": self.fit_power_law(points),
            "exponential": self.fit_exponential(points),
            "logistic": self.fit_logistic(points),
        }
        best = max(
            results,
            key=lambda k: results[k].get("r_squared", -1.0)
            if results[k].get("success", False)
            else -1.0,
        )
        return {
            "best_model": best,
            "best_r_squared": results[best].get("r_squared", 0.0),
            "all_fits": results,
        }


class FrontierEvolution:
    """Track frontier evolution over successive evaluations."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()
        self._pf = ParetoFrontier(self.config)
        self._points: List[TradeoffPoint] = []
        self._snapshots: List[Dict[str, Any]] = []

    def add_point(self, point: TradeoffPoint) -> Dict[str, Any]:
        """Add a new point and record a frontier snapshot."""
        self._points.append(point)
        frontier = self._pf.compute_frontier(self._points)
        hv = self._pf.hypervolume(frontier)
        snap = {
            "num_evaluated": len(self._points),
            "frontier_size": len(frontier),
            "hypervolume": hv,
            "latest_quality": point.quality_score,
            "latest_diversity": point.diversity_score,
            "is_on_frontier": any(
                abs(f.quality_score - point.quality_score) < _EPS
                and abs(f.diversity_score - point.diversity_score) < _EPS
                for f in frontier
            ),
        }
        self._snapshots.append(snap)
        return snap

    def add_points(
        self, points: Sequence[TradeoffPoint]
    ) -> List[Dict[str, Any]]:
        """Add multiple points, returning all snapshots."""
        return [self.add_point(p) for p in points]

    def get_evolution_data(self) -> Dict[str, Any]:
        """Return the full evolution history."""
        if not self._snapshots:
            return {
                "num_evaluations": [],
                "hypervolume": [],
                "frontier_size": [],
                "improvement_events": [],
            }
        evals = [s["num_evaluated"] for s in self._snapshots]
        hvs = [s["hypervolume"] for s in self._snapshots]
        fs = [s["frontier_size"] for s in self._snapshots]
        improvements = [
            {
                "evaluation": evals[i],
                "hv_before": hvs[i - 1],
                "hv_after": hvs[i],
                "improvement": hvs[i] - hvs[i - 1],
            }
            for i in range(1, len(hvs))
            if hvs[i] > hvs[i - 1] + _EPS
        ]
        return {
            "num_evaluations": evals,
            "hypervolume": hvs,
            "frontier_size": fs,
            "improvement_events": improvements,
            "total_improvements": len(improvements),
            "final_hypervolume": hvs[-1] if hvs else 0.0,
        }

    @property
    def current_frontier(self) -> List[TradeoffPoint]:
        return self._pf.compute_frontier(self._points)

    @property
    def all_points(self) -> List[TradeoffPoint]:
        return list(self._points)

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._points.clear()
        self._snapshots.clear()


class NormalisedTradeoffSpace:
    """Work in a normalised [0,1] squared objective space."""

    def __init__(
        self,
        quality_bounds: Tuple[float, float] = (0.0, 1.0),
        diversity_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.q_lo, self.q_hi = quality_bounds
        self.d_lo, self.d_hi = diversity_bounds

    def normalise(self, point: TradeoffPoint) -> TradeoffPoint:
        """Map a point into [0,1] squared space."""
        qs = self.q_hi - self.q_lo
        ds = self.d_hi - self.d_lo
        nq = (point.quality_score - self.q_lo) / max(qs, _EPS)
        nd = (point.diversity_score - self.d_lo) / max(ds, _EPS)
        return TradeoffPoint(
            quality_score=float(np.clip(nq, 0, 1)),
            diversity_score=float(np.clip(nd, 0, 1)),
            algorithm=point.algorithm,
            config=point.config,
            metadata=point.metadata,
        )

    def denormalise(self, point: TradeoffPoint) -> TradeoffPoint:
        """Map from [0,1] squared back to original scale."""
        q = self.q_lo + point.quality_score * (self.q_hi - self.q_lo)
        d = self.d_lo + point.diversity_score * (self.d_hi - self.d_lo)
        return TradeoffPoint(
            quality_score=q,
            diversity_score=d,
            algorithm=point.algorithm,
            config=point.config,
            metadata=point.metadata,
        )

    def normalise_batch(
        self, points: Sequence[TradeoffPoint]
    ) -> List[TradeoffPoint]:
        """Normalise a batch of points."""
        return [self.normalise(p) for p in points]

    def denormalise_batch(
        self, points: Sequence[TradeoffPoint]
    ) -> List[TradeoffPoint]:
        """Denormalise a batch of points."""
        return [self.denormalise(p) for p in points]

    def distance_in_normalised_space(
        self, a: TradeoffPoint, b: TradeoffPoint
    ) -> float:
        """Euclidean distance between two points in normalised space."""
        na = self.normalise(a)
        nb = self.normalise(b)
        return float(np.linalg.norm(na.as_array() - nb.as_array()))


class WeightedTradeoffScorer:
    """Score tradeoff points with configurable weights."""

    def __init__(
        self,
        initial_quality_weight: float = 0.5,
        initial_diversity_weight: float = 0.5,
    ) -> None:
        self.w_q = initial_quality_weight
        self.w_d = initial_diversity_weight

    def score(self, point: TradeoffPoint) -> float:
        """Weighted linear combination score."""
        return self.w_q * point.quality_score + self.w_d * point.diversity_score

    def score_batch(
        self, points: Sequence[TradeoffPoint]
    ) -> NDArray[np.float64]:
        """Score all points."""
        return np.array([self.score(p) for p in points], dtype=np.float64)

    def rank(self, points: Sequence[TradeoffPoint]) -> List[int]:
        """Rank indices from best (0) to worst."""
        scores = self.score_batch(points)
        return list(np.argsort(-scores))

    def best(self, points: Sequence[TradeoffPoint]) -> Optional[TradeoffPoint]:
        """Return the highest-scoring point."""
        if not points:
            return None
        return points[int(np.argmax(self.score_batch(points)))]

    def update_weights(self, qw: float, dw: float) -> None:
        """Update and normalise weights."""
        total = qw + dw
        if total < _EPS:
            self.w_q, self.w_d = 0.5, 0.5
        else:
            self.w_q, self.w_d = qw / total, dw / total

    def linear_weight_schedule(
        self,
        step: int,
        total_steps: int,
        start_quality_weight: float = 0.8,
        end_quality_weight: float = 0.2,
    ) -> None:
        """Linearly interpolate quality weight from start to end."""
        frac = min(step / max(total_steps - 1, 1), 1.0)
        wq = start_quality_weight + frac * (end_quality_weight - start_quality_weight)
        self.update_weights(wq, 1.0 - wq)

    def cosine_weight_schedule(
        self,
        step: int,
        total_steps: int,
        min_quality_weight: float = 0.2,
        max_quality_weight: float = 0.8,
    ) -> None:
        """Cosine-annealed quality weight."""
        frac = min(step / max(total_steps - 1, 1), 1.0)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * frac))
        wq = min_quality_weight + cos_val * (max_quality_weight - min_quality_weight)
        self.update_weights(wq, 1.0 - wq)


class TradeoffReport:
    """Generate structured report data from tradeoff analysis."""

    def __init__(self, config: Optional[TradeoffConfig] = None) -> None:
        self.config = config or TradeoffConfig()

    def generate_report(
        self,
        points: Sequence[TradeoffPoint],
        reference_frontier: Optional[Sequence[TradeoffPoint]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive tradeoff analysis report."""
        mc = TradeoffMetrics(self.config)
        an = TradeoffAnalyzer(self.config)
        viz = TradeoffVisualizer(self.config)
        reg = TradeoffRegression(self.config)

        metrics = mc.compute_all_metrics(points, reference_frontier)
        efficiency = mc.compute_tradeoff_efficiency(points)
        rankings = mc.compute_algorithm_rankings(points)
        algo_an = an.analyze_algorithm_tradeoffs(points)
        mrs = an.compute_marginal_rates(points)
        fp = viz.prepare_frontier_plot_data(points)
        hm = viz.prepare_heatmap_data(points)
        radar = viz.prepare_radar_chart_data(points)
        bf = reg.best_fit(points)

        algos = sorted(set(p.algorithm for p in points))
        pairwise: Dict[str, Any] = {}
        if len(algos) >= 2:
            by_a: Dict[str, List[TradeoffPoint]] = {}
            for p in points:
                by_a.setdefault(p.algorithm, []).append(p)
            for i, a1 in enumerate(algos):
                for a2 in algos[i + 1:]:
                    pairwise[f"{a1}_vs_{a2}"] = an.compare_frontiers(
                        by_a[a1], by_a[a2], a1, a2
                    )

        return {
            "summary_metrics": metrics,
            "efficiency": efficiency,
            "algorithm_rankings": rankings,
            "per_algorithm": algo_an,
            "marginal_rates": mrs,
            "frontier_plot": fp,
            "heatmap": hm,
            "radar_chart": radar,
            "curve_fit": bf,
            "pairwise_comparisons": pairwise,
            "config": self.config.to_dict(),
        }

    def generate_comparison_report(
        self,
        points_a: Sequence[TradeoffPoint],
        points_b: Sequence[TradeoffPoint],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict[str, Any]:
        """Generate a focused two-algorithm comparison report."""
        an = TradeoffAnalyzer(self.config)
        comp = an.compare_frontiers(points_a, points_b, label_a, label_b)
        stat = an.statistical_frontier_comparison(
            points_a, points_b, label_a, label_b
        )
        ma = TradeoffMetrics(self.config).compute_all_metrics(points_a)
        mb = TradeoffMetrics(self.config).compute_all_metrics(points_b)
        return {
            "comparison": comp,
            "statistical": stat,
            f"metrics_{label_a}": ma,
            f"metrics_{label_b}": mb,
            "winner": label_a if comp["hypervolume_diff"] > 0 else label_b,
            "significant": stat["significant"],
        }


# ===================================================================
#  Module-level convenience functions
# ===================================================================


def quick_tradeoff_analysis(
    points: Sequence[TradeoffPoint],
    config: Optional[TradeoffConfig] = None,
) -> Dict[str, Any]:
    """One-call tradeoff analysis returning key metrics."""
    return TradeoffReport(config).generate_report(points)


def quick_compare(
    points_a: Sequence[TradeoffPoint],
    points_b: Sequence[TradeoffPoint],
    label_a: str = "A",
    label_b: str = "B",
    config: Optional[TradeoffConfig] = None,
) -> Dict[str, Any]:
    """One-call comparison of two algorithm point sets."""
    return TradeoffReport(config).generate_comparison_report(
        points_a, points_b, label_a, label_b
    )


def find_best_operating_point(
    points: Sequence[TradeoffPoint],
    method: str = "balanced",
    min_quality: Optional[float] = None,
    min_diversity: Optional[float] = None,
    config: Optional[TradeoffConfig] = None,
) -> Optional[TradeoffPoint]:
    """Find the best operating point using the specified strategy.

    Parameters
    ----------
    method : ``"balanced"``, ``"knee"``, ``"quality_constrained"``,
        ``"diversity_constrained"``, ``"weighted_sum"``,
        ``"tchebycheff"``, ``"augmented"``.
    """
    opf = OperatingPointFinder(config)
    if method == "balanced":
        return opf.find_balanced_point(points)
    if method == "knee":
        return opf.find_knee_point(points)
    if method == "quality_constrained":
        return opf.find_quality_constrained_optimum(
            points, min_quality or 0.5
        )
    if method == "diversity_constrained":
        return opf.find_diversity_constrained_optimum(
            points, min_diversity or 0.5
        )
    if method in ("weighted_sum", "tchebycheff", "augmented"):
        return opf.multi_objective_scalarization(points, method=method)
    raise ValueError(f"Unknown method: {method!r}")


def compute_frontier_from_scores(
    quality_scores: Sequence[float],
    diversity_scores: Sequence[float],
    algorithms: Optional[Sequence[str]] = None,
) -> List[TradeoffPoint]:
    """Convenience: compute frontier directly from parallel score lists."""
    n = len(quality_scores)
    if algorithms is None:
        algorithms = ["unknown"] * n
    points = [
        TradeoffPoint(
            quality_score=float(quality_scores[i]),
            diversity_score=float(diversity_scores[i]),
            algorithm=algorithms[i] if i < len(algorithms) else "unknown",
        )
        for i in range(n)
    ]
    return ParetoFrontier().compute_frontier(points)
