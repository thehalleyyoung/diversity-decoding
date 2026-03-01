"""
Pareto frontier analysis for multi-objective diversity-quality tradeoffs.

Implements non-dominated sorting, hypervolume computation, frontier comparison
metrics, and visualization-ready interpolation for the Diversity Decoding Arena.
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
_DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DominanceRelation(Enum):
    """Pairwise dominance relationship between two objective vectors."""

    DOMINATES = auto()
    DOMINATED = auto()
    NON_DOMINATED = auto()
    EQUAL = auto()


class ObjectiveDirection(Enum):
    """Whether an objective should be maximized or minimized."""

    MAXIMIZE = auto()
    MINIMIZE = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """A single point in objective space.

    Attributes:
        objectives: Mapping from objective name to scalar value.
        algorithm: Name of the algorithm that produced this point.
        config: Algorithm hyper-parameters or configuration dictionary.
        metadata: Arbitrary additional information.
        rank: Non-dominated rank (0 = first Pareto front). Set during sorting.
        crowding: Crowding distance. Set during crowding-distance assignment.
    """

    objectives: Dict[str, float]
    algorithm: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = -1
    crowding: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def objective_vector(self, keys: Sequence[str]) -> NDArray[np.float64]:
        """Return objective values as a numpy array in the given key order."""
        return np.array([self.objectives[k] for k in keys], dtype=np.float64)

    def copy(self) -> "ParetoPoint":
        """Return a deep copy."""
        return ParetoPoint(
            objectives=dict(self.objectives),
            algorithm=self.algorithm,
            config=dict(self.config),
            metadata=dict(self.metadata),
            rank=self.rank,
            crowding=self.crowding,
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParetoPoint):
            return NotImplemented
        return self is other

    def __repr__(self) -> str:
        obj_str = ", ".join(f"{k}={v:.4f}" for k, v in self.objectives.items())
        return f"ParetoPoint({obj_str}, alg={self.algorithm!r})"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _euclidean_distance(a: NDArray, b: NDArray) -> float:
    """Euclidean distance between two vectors."""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def _manhattan_distance(a: NDArray, b: NDArray) -> float:
    """Manhattan (L1) distance between two vectors."""
    return float(np.sum(np.abs(a - b)))


def _chebyshev_distance(a: NDArray, b: NDArray) -> float:
    """Chebyshev (L∞) distance between two vectors."""
    return float(np.max(np.abs(a - b)))


def _point_to_line_distance(
    point: NDArray, line_start: NDArray, line_end: NDArray
) -> float:
    """Perpendicular distance from *point* to the line through *line_start* and
    *line_end*. Works in arbitrary dimensions."""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < _EPS:
        return _euclidean_distance(point, line_start)
    unit_line = line_vec / line_len
    proj_length = np.dot(point - line_start, unit_line)
    proj_point = line_start + proj_length * unit_line
    return _euclidean_distance(point, proj_point)


def _convex_hull_volume_nd(points: NDArray) -> float:
    """Approximate the convex-hull volume of *points* via the Delaunay
    triangulation when scipy is available; otherwise fall back to a bounding-box
    estimate."""
    if points.shape[0] <= points.shape[1]:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        ranges = np.ptp(points, axis=0)
        return float(np.prod(ranges))


def _project_onto_simplex(v: NDArray, s: float = 1.0) -> NDArray:
    """Project *v* onto the probability simplex of size *s*."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    rho = int(np.max(np.where(u * np.arange(1, n + 1) > cssv)))
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _angle_between(a: NDArray, b: NDArray) -> float:
    """Angle in radians between two vectors."""
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _pairwise_distances(matrix: NDArray) -> NDArray:
    """Return a symmetric pairwise Euclidean distance matrix."""
    n = matrix.shape[0]
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean_distance(matrix[i], matrix[j])
            dists[i, j] = d
            dists[j, i] = d
    return dists


def _normalize_matrix(
    matrix: NDArray,
    ideal: Optional[NDArray] = None,
    nadir: Optional[NDArray] = None,
) -> NDArray:
    """Normalize each column of *matrix* to [0, 1] using ideal / nadir."""
    if ideal is None:
        ideal = matrix.min(axis=0)
    if nadir is None:
        nadir = matrix.max(axis=0)
    ranges = nadir - ideal
    ranges[ranges < _EPS] = 1.0
    return (matrix - ideal) / ranges


def _dominated_hypervolume_2d(
    points: NDArray, ref: NDArray
) -> float:
    """Exact dominated hypervolume for two objectives (maximization assumed).

    *points* is (N, 2) and *ref* is the reference point (worst-case corner).
    Points are assumed to already be filtered so that every point dominates *ref*.
    """
    if points.shape[0] == 0:
        return 0.0
    sorted_idx = np.argsort(-points[:, 0])
    pts = points[sorted_idx]
    volume = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y > prev_y:
            volume += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(volume)


def _dominated_hypervolume_3d(
    points: NDArray, ref: NDArray
) -> float:
    """Hypervolume for three objectives via the sweeping-slab algorithm.

    Falls back to the inclusion-exclusion estimator for safety.
    """
    if points.shape[0] == 0:
        return 0.0
    sorted_idx = np.argsort(-points[:, 2])
    pts = points[sorted_idx]
    volume = 0.0
    prev_z = ref[2]
    for i, (x, y, z) in enumerate(pts):
        slab_height = z - prev_z
        if slab_height <= 0:
            continue
        slice_pts = pts[: i + 1, :2]
        area = _dominated_hypervolume_2d(slice_pts, ref[:2])
        volume += area * slab_height
        prev_z = z
    return float(volume)


def _dominated_hypervolume_nd(
    points: NDArray, ref: NDArray
) -> float:
    """General hypervolume via inclusion-exclusion (exact but exponential).

    For more than ~15 points this becomes prohibitive; a Monte-Carlo fallback
    is used when the set is large.
    """
    n, d = points.shape
    if n == 0:
        return 0.0
    if d == 2:
        return _dominated_hypervolume_2d(points, ref)
    if d == 3:
        return _dominated_hypervolume_3d(points, ref)
    if n > 15:
        return _hypervolume_monte_carlo(points, ref, n_samples=100_000)
    volume = 0.0
    for size in range(1, n + 1):
        sign = (-1) ** (size + 1)
        for combo in itertools.combinations(range(n), size):
            subset = points[list(combo)]
            upper = np.min(subset, axis=0)
            box = np.prod(np.maximum(upper - ref, 0.0))
            volume += sign * box
    return float(volume)


def _hypervolume_monte_carlo(
    points: NDArray,
    ref: NDArray,
    n_samples: int = 100_000,
    seed: int = _DEFAULT_SEED,
) -> float:
    """Monte-Carlo hypervolume estimate.

    Uniformly samples the bounding box defined by *ref* and the ideal point,
    and counts the fraction of samples dominated by at least one point.
    """
    rng = np.random.default_rng(seed)
    ideal = np.max(points, axis=0)
    lo = ref
    hi = ideal
    ranges = hi - lo
    if np.any(ranges <= 0):
        return 0.0
    box_volume = float(np.prod(ranges))
    samples = rng.uniform(lo, hi, size=(n_samples, points.shape[1]))
    dominated_count = 0
    for s in samples:
        for p in points:
            if np.all(p >= s):
                dominated_count += 1
                break
    return box_volume * dominated_count / n_samples


def _sort_frontier_points(
    points: List[ParetoPoint],
    keys: Sequence[str],
    primary_index: int = 0,
) -> List[ParetoPoint]:
    """Sort frontier points by the *primary_index*-th objective (ascending)."""
    return sorted(points, key=lambda p: p.objectives.get(keys[primary_index], 0.0))


# ---------------------------------------------------------------------------
# ParetoFrontier
# ---------------------------------------------------------------------------


class ParetoFrontier:
    """Represents a Pareto frontier computed from a set of points.

    Parameters:
        points: Collection of :class:`ParetoPoint` instances.
        objectives_to_maximize: Mapping from objective name to ``True`` if the
            objective should be maximized, ``False`` if minimized.  By default
            every objective is maximized.
    """

    def __init__(
        self,
        points: Iterable[ParetoPoint],
        objectives_to_maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._all_points: List[ParetoPoint] = list(points)
        if not self._all_points:
            self._objective_keys: List[str] = []
        else:
            self._objective_keys = sorted(self._all_points[0].objectives.keys())

        if objectives_to_maximize is None:
            self._maximize: Dict[str, bool] = {k: True for k in self._objective_keys}
        else:
            self._maximize = dict(objectives_to_maximize)

        self._frontier: Optional[List[ParetoPoint]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def objective_keys(self) -> List[str]:
        """Sorted list of objective names."""
        return list(self._objective_keys)

    @property
    def all_points(self) -> List[ParetoPoint]:
        """All points (dominated and non-dominated)."""
        return list(self._all_points)

    @property
    def frontier_points(self) -> List[ParetoPoint]:
        """Non-dominated (Pareto-optimal) points.  Computed lazily."""
        if self._frontier is None:
            self._frontier = self.compute_frontier()
        return list(self._frontier)

    @property
    def n_objectives(self) -> int:
        return len(self._objective_keys)

    @property
    def size(self) -> int:
        return len(self._all_points)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def compute_frontier(self) -> List[ParetoPoint]:
        """Compute the Pareto-optimal (non-dominated) subset of all points.

        Uses pairwise dominance checks with early termination.

        Returns:
            List of non-dominated :class:`ParetoPoint` instances.
        """
        if not self._all_points:
            return []

        keys = self._objective_keys
        maximize = self._maximize

        def _transform(p: ParetoPoint) -> NDArray:
            vec = p.objective_vector(keys)
            signs = np.array(
                [1.0 if maximize.get(k, True) else -1.0 for k in keys],
                dtype=np.float64,
            )
            return vec * signs

        transformed = [_transform(p) for p in self._all_points]
        n = len(transformed)
        is_dominated = [False] * n

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if _vec_dominates(transformed[j], transformed[i]):
                    is_dominated[i] = True
                    break

        frontier = [p for p, dom in zip(self._all_points, is_dominated) if not dom]
        for p in frontier:
            p.rank = 0
        self._frontier = frontier
        return frontier

    def add_point(self, point: ParetoPoint) -> None:
        """Add a point and invalidate the cached frontier."""
        self._all_points.append(point)
        if not self._objective_keys:
            self._objective_keys = sorted(point.objectives.keys())
        self._frontier = None

    def remove_point(self, point: ParetoPoint) -> bool:
        """Remove a point (by identity) and invalidate the frontier cache.

        Returns:
            ``True`` if the point was found and removed.
        """
        try:
            self._all_points.remove(point)
            self._frontier = None
            return True
        except ValueError:
            return False

    def dominates(self, a: ParetoPoint, b: ParetoPoint) -> DominanceRelation:
        """Determine the dominance relationship between *a* and *b*.

        Returns:
            A :class:`DominanceRelation` value.
        """
        keys = self._objective_keys
        maximize = self._maximize
        signs = np.array(
            [1.0 if maximize.get(k, True) else -1.0 for k in keys],
            dtype=np.float64,
        )
        va = a.objective_vector(keys) * signs
        vb = b.objective_vector(keys) * signs
        return _dominance_relation(va, vb)

    def distance_to_frontier(self, point: ParetoPoint) -> float:
        """Minimum Euclidean distance from *point* to the Pareto frontier.

        The distance is computed in the (sign-adjusted, normalized) objective
        space so that maximization/minimization is accounted for.

        Returns:
            Non-negative distance.  Zero if *point* is on the frontier.
        """
        frontier = self.frontier_points
        if not frontier:
            return float("inf")

        keys = self._objective_keys
        signs = self._sign_vector()
        pv = point.objective_vector(keys) * signs

        matrix = self._frontier_matrix(frontier) * signs
        ideal = matrix.min(axis=0)
        nadir = matrix.max(axis=0)
        ranges = nadir - ideal
        ranges[ranges < _EPS] = 1.0

        pv_norm = (pv - ideal) / ranges
        matrix_norm = (matrix - ideal) / ranges

        min_dist = float("inf")
        for row in matrix_norm:
            d = _euclidean_distance(pv_norm, row)
            if d < min_dist:
                min_dist = d
        return min_dist

    def crowding_distance(self, point: ParetoPoint) -> float:
        """Crowding distance of *point* relative to the current frontier.

        Boundary points receive ``float('inf')``.

        Returns:
            Crowding distance value.
        """
        frontier = self.frontier_points
        if point not in frontier:
            return 0.0

        keys = self._objective_keys
        n = len(frontier)
        if n <= 2:
            return float("inf")

        distances: Dict[int, float] = {i: 0.0 for i in range(n)}
        idx_map = {id(p): i for i, p in enumerate(frontier)}

        for m, key in enumerate(keys):
            sorted_front = sorted(frontier, key=lambda p: p.objectives[key])
            obj_min = sorted_front[0].objectives[key]
            obj_max = sorted_front[-1].objectives[key]
            obj_range = obj_max - obj_min
            if obj_range < _EPS:
                continue

            first_idx = idx_map[id(sorted_front[0])]
            last_idx = idx_map[id(sorted_front[-1])]
            distances[first_idx] = float("inf")
            distances[last_idx] = float("inf")

            for k in range(1, n - 1):
                orig_idx = idx_map[id(sorted_front[k])]
                diff = (
                    sorted_front[k + 1].objectives[key]
                    - sorted_front[k - 1].objectives[key]
                )
                distances[orig_idx] += diff / obj_range

        target_idx = idx_map.get(id(point))
        if target_idx is None:
            return 0.0
        return distances[target_idx]

    def spread_metric(self) -> float:
        """Compute the spread (extent) of the Pareto frontier.

        Defined as the Euclidean distance between the extreme points of the
        frontier in normalized objective space.

        Returns:
            Non-negative spread value.
        """
        frontier = self.frontier_points
        if len(frontier) < 2:
            return 0.0

        keys = self._objective_keys
        signs = self._sign_vector()
        matrix = self._frontier_matrix(frontier) * signs
        norm = _normalize_matrix(matrix)
        ideal = norm.min(axis=0)
        nadir = norm.max(axis=0)
        return float(np.linalg.norm(nadir - ideal))

    def spacing_metric(self) -> float:
        """Compute the spacing metric (Schott's spacing metric).

        Measures the uniformity of the distribution of frontier points.
        Lower values indicate a more uniform spread.

        Returns:
            Non-negative spacing value; 0.0 for fewer than 2 points.
        """
        frontier = self.frontier_points
        if len(frontier) < 2:
            return 0.0

        keys = self._objective_keys
        signs = self._sign_vector()
        matrix = self._frontier_matrix(frontier) * signs
        norm = _normalize_matrix(matrix)

        n = len(frontier)
        min_dists = np.full(n, float("inf"))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = _manhattan_distance(norm[i], norm[j])
                if d < min_dists[i]:
                    min_dists[i] = d

        d_mean = float(np.mean(min_dists))
        if d_mean < _EPS:
            return 0.0
        return float(np.sqrt(np.mean((min_dists - d_mean) ** 2)))

    def coverage(self, other: "ParetoFrontier") -> float:
        """Fraction of *other*'s frontier points dominated by this frontier.

        Also known as the C-metric or coverage metric.

        Parameters:
            other: Another :class:`ParetoFrontier` instance.

        Returns:
            Value in [0, 1]. ``1.0`` means this frontier dominates every point
            in *other*.
        """
        my_front = self.frontier_points
        other_front = other.frontier_points
        if not other_front:
            return 1.0
        if not my_front:
            return 0.0

        keys = self._objective_keys
        signs = self._sign_vector()
        my_vecs = [p.objective_vector(keys) * signs for p in my_front]

        dominated_count = 0
        for op in other_front:
            ov = op.objective_vector(keys) * signs
            for mv in my_vecs:
                if _vec_dominates(mv, ov):
                    dominated_count += 1
                    break

        return dominated_count / len(other_front)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sign_vector(self) -> NDArray[np.float64]:
        return np.array(
            [1.0 if self._maximize.get(k, True) else -1.0 for k in self._objective_keys],
            dtype=np.float64,
        )

    def _frontier_matrix(
        self, points: Optional[List[ParetoPoint]] = None
    ) -> NDArray[np.float64]:
        if points is None:
            points = self.frontier_points
        if not points:
            return np.empty((0, self.n_objectives), dtype=np.float64)
        return np.array(
            [p.objective_vector(self._objective_keys) for p in points],
            dtype=np.float64,
        )

    def _all_matrix(self) -> NDArray[np.float64]:
        if not self._all_points:
            return np.empty((0, self.n_objectives), dtype=np.float64)
        return np.array(
            [p.objective_vector(self._objective_keys) for p in self._all_points],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Dominance helpers (module-level for reuse)
# ---------------------------------------------------------------------------


def _vec_dominates(a: NDArray, b: NDArray) -> bool:
    """Return True if *a* dominates *b* (all >=, at least one >)."""
    return bool(np.all(a >= b - _EPS) and np.any(a > b + _EPS))


def _dominance_relation(a: NDArray, b: NDArray) -> DominanceRelation:
    """Pairwise dominance relation between sign-adjusted vectors."""
    if np.allclose(a, b, atol=_EPS):
        return DominanceRelation.EQUAL
    if _vec_dominates(a, b):
        return DominanceRelation.DOMINATES
    if _vec_dominates(b, a):
        return DominanceRelation.DOMINATED
    return DominanceRelation.NON_DOMINATED


# ---------------------------------------------------------------------------
# NonDominatedSorting (NSGA-II)
# ---------------------------------------------------------------------------


class NonDominatedSorting:
    """NSGA-II style non-dominated sorting and crowding-distance assignment.

    Parameters:
        objective_keys: Ordered list of objective names.
        maximize: Mapping from objective name to boolean indicating whether
            the objective should be maximized.
    """

    def __init__(
        self,
        objective_keys: Sequence[str],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._keys = list(objective_keys)
        if maximize is None:
            self._maximize: Dict[str, bool] = {k: True for k in self._keys}
        else:
            self._maximize = dict(maximize)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fast_non_dominated_sort(
        self, points: Sequence[ParetoPoint]
    ) -> List[List[ParetoPoint]]:
        """Partition *points* into successive non-dominated fronts.

        Implements the fast non-dominated sort from Deb et al. (NSGA-II).

        Each point's ``rank`` attribute is set to its front index (0-based).

        Returns:
            List of fronts, where ``fronts[0]`` is the Pareto-optimal set.
        """
        if not points:
            return []

        n = len(points)
        signs = self._sign_vector()
        vecs = [p.objective_vector(self._keys) * signs for p in points]

        domination_count: List[int] = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                rel = _dominance_relation(vecs[i], vecs[j])
                if rel == DominanceRelation.DOMINATES:
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif rel == DominanceRelation.DOMINATED:
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        fronts: List[List[int]] = []
        current_front = [i for i in range(n) if domination_count[i] == 0]

        while current_front:
            fronts.append(current_front)
            next_front: List[int] = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        result: List[List[ParetoPoint]] = []
        for rank, front_indices in enumerate(fronts):
            front_points: List[ParetoPoint] = []
            for idx in front_indices:
                points[idx].rank = rank
                front_points.append(points[idx])
            result.append(front_points)

        return result

    def crowding_distance_assignment(
        self, front: Sequence[ParetoPoint]
    ) -> Dict[ParetoPoint, float]:
        """Compute crowding distance for every point in *front*.

        Boundary points (extreme in any objective) receive ``float('inf')``.

        Returns:
            Mapping from :class:`ParetoPoint` to its crowding distance.
        """
        n = len(front)
        distances: Dict[ParetoPoint, float] = {p: 0.0 for p in front}

        if n <= 2:
            for p in front:
                distances[p] = float("inf")
                p.crowding = float("inf")
            return distances

        for key in self._keys:
            sorted_front = sorted(front, key=lambda p: p.objectives[key])
            obj_min = sorted_front[0].objectives[key]
            obj_max = sorted_front[-1].objectives[key]
            obj_range = obj_max - obj_min

            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            if obj_range < _EPS:
                continue

            for k in range(1, n - 1):
                diff = (
                    sorted_front[k + 1].objectives[key]
                    - sorted_front[k - 1].objectives[key]
                )
                distances[sorted_front[k]] += diff / obj_range

        for p in front:
            p.crowding = distances[p]

        return distances

    def crowded_comparison(self, a: ParetoPoint, b: ParetoPoint) -> int:
        """Compare two points using crowded-comparison operator.

        A point is preferred if it has a lower (better) rank, or, when ranks
        are equal, a larger crowding distance.

        Returns:
            -1 if *a* is preferred, +1 if *b* is preferred, 0 if equal.
        """
        if a.rank < b.rank:
            return -1
        if a.rank > b.rank:
            return 1
        if a.crowding > b.crowding:
            return -1
        if a.crowding < b.crowding:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sign_vector(self) -> NDArray[np.float64]:
        return np.array(
            [1.0 if self._maximize.get(k, True) else -1.0 for k in self._keys],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# ParetoAnalyzer
# ---------------------------------------------------------------------------


class ParetoAnalyzer:
    """High-level analysis utilities built on top of :class:`ParetoFrontier`.

    Parameters:
        results: Sequence of :class:`ParetoPoint` instances representing
            experimental results from one or more algorithms.
    """

    def __init__(self, results: Sequence[ParetoPoint]) -> None:
        self._results: List[ParetoPoint] = list(results)

    # ------------------------------------------------------------------
    # Frontier construction
    # ------------------------------------------------------------------

    def analyze_2d(
        self,
        diversity_metric: str,
        quality_metric: str,
        maximize_diversity: bool = True,
        maximize_quality: bool = True,
    ) -> ParetoFrontier:
        """Build a two-objective Pareto frontier.

        Parameters:
            diversity_metric: Name of the diversity objective.
            quality_metric: Name of the quality objective.
            maximize_diversity: Whether higher diversity is better.
            maximize_quality: Whether higher quality is better.

        Returns:
            :class:`ParetoFrontier` containing only the two selected objectives.
        """
        projected = _project_points(
            self._results, [diversity_metric, quality_metric]
        )
        maximize_flags = {
            diversity_metric: maximize_diversity,
            quality_metric: maximize_quality,
        }
        frontier = ParetoFrontier(projected, maximize_flags)
        frontier.compute_frontier()
        return frontier

    def analyze_nd(
        self,
        metrics: Sequence[str],
        maximize_flags: Optional[Dict[str, bool]] = None,
    ) -> ParetoFrontier:
        """Build an N-dimensional Pareto frontier.

        Parameters:
            metrics: Objective names to include.
            maximize_flags: Per-objective maximization flags.

        Returns:
            :class:`ParetoFrontier`.
        """
        projected = _project_points(self._results, list(metrics))
        if maximize_flags is None:
            maximize_flags = {m: True for m in metrics}
        frontier = ParetoFrontier(projected, maximize_flags)
        frontier.compute_frontier()
        return frontier

    # ------------------------------------------------------------------
    # Hypervolume
    # ------------------------------------------------------------------

    def compute_hypervolume(
        self,
        frontier: ParetoFrontier,
        reference_point: Optional[NDArray] = None,
    ) -> float:
        """Dominated hypervolume indicator.

        Parameters:
            frontier: The :class:`ParetoFrontier` whose hypervolume to compute.
            reference_point: The reference (anti-ideal) point.  If ``None``,
                a default is computed from the worst observed objective values
                shifted by 10 %.

        Returns:
            Hypervolume value (non-negative).
        """
        front = frontier.frontier_points
        if not front:
            return 0.0

        keys = frontier.objective_keys
        signs = frontier._sign_vector()
        matrix = frontier._frontier_matrix(front) * signs

        if reference_point is None:
            all_matrix = frontier._all_matrix() * signs
            ref = all_matrix.min(axis=0) - 0.1 * np.abs(all_matrix.min(axis=0))
        else:
            ref = np.asarray(reference_point, dtype=np.float64) * signs

        valid_mask = np.array(
            [np.all(row > ref + _EPS) for row in matrix]
        )
        valid = matrix[valid_mask]

        return _dominated_hypervolume_nd(valid, ref)

    # ------------------------------------------------------------------
    # Inverted Generational Distance (IGD)
    # ------------------------------------------------------------------

    def compute_igd(
        self,
        frontier: ParetoFrontier,
        reference_front: Sequence[ParetoPoint],
    ) -> float:
        """Inverted Generational Distance (IGD).

        Measures the average minimum distance from each point in the
        *reference_front* to the closest point in *frontier*.

        Parameters:
            frontier: Approximate Pareto front.
            reference_front: True (or best-known) Pareto front.

        Returns:
            Non-negative IGD value.  Lower is better.
        """
        front = frontier.frontier_points
        if not front or not reference_front:
            return float("inf")

        keys = frontier.objective_keys
        signs = frontier._sign_vector()

        f_matrix = np.array(
            [p.objective_vector(keys) * signs for p in front], dtype=np.float64
        )
        r_matrix = np.array(
            [p.objective_vector(keys) * signs for p in reference_front],
            dtype=np.float64,
        )

        f_norm = _normalize_matrix(
            f_matrix,
            ideal=np.minimum(f_matrix.min(0), r_matrix.min(0)),
            nadir=np.maximum(f_matrix.max(0), r_matrix.max(0)),
        )
        r_norm = _normalize_matrix(
            r_matrix,
            ideal=np.minimum(f_matrix.min(0), r_matrix.min(0)),
            nadir=np.maximum(f_matrix.max(0), r_matrix.max(0)),
        )

        total = 0.0
        for rv in r_norm:
            min_d = float("inf")
            for fv in f_norm:
                d = _euclidean_distance(rv, fv)
                if d < min_d:
                    min_d = d
            total += min_d

        return total / len(r_norm)

    # ------------------------------------------------------------------
    # Epsilon indicator
    # ------------------------------------------------------------------

    def compute_epsilon_indicator(
        self,
        frontier_a: ParetoFrontier,
        frontier_b: ParetoFrontier,
    ) -> float:
        """Additive epsilon indicator I_ε+(A, B).

        The minimum epsilon such that every point in *frontier_b* is
        epsilon-dominated by some point in *frontier_a*.

        Parameters:
            frontier_a: First (approximation) front.
            frontier_b: Second (reference) front.

        Returns:
            Epsilon value.  Negative means A is strictly better.
        """
        front_a = frontier_a.frontier_points
        front_b = frontier_b.frontier_points
        if not front_a or not front_b:
            return float("inf")

        keys = frontier_a.objective_keys
        signs = frontier_a._sign_vector()

        mat_a = np.array(
            [p.objective_vector(keys) * signs for p in front_a], dtype=np.float64
        )
        mat_b = np.array(
            [p.objective_vector(keys) * signs for p in front_b], dtype=np.float64
        )

        eps = float("-inf")
        for bv in mat_b:
            min_eps_for_b = float("inf")
            for av in mat_a:
                max_diff = float(np.max(bv - av))
                if max_diff < min_eps_for_b:
                    min_eps_for_b = max_diff
            if min_eps_for_b > eps:
                eps = min_eps_for_b

        return eps

    # ------------------------------------------------------------------
    # Marginal contribution
    # ------------------------------------------------------------------

    def marginal_contribution(
        self,
        algorithm: str,
        frontier: ParetoFrontier,
        reference_point: Optional[NDArray] = None,
    ) -> float:
        """Hypervolume contribution of *algorithm* to *frontier*.

        Defined as the decrease in hypervolume when all points from
        *algorithm* are removed from the frontier.

        Parameters:
            algorithm: Algorithm name.
            frontier: The frontier containing points from multiple algorithms.
            reference_point: Reference point for hypervolume computation.

        Returns:
            Non-negative contribution value.
        """
        hv_full = self.compute_hypervolume(frontier, reference_point)

        remaining = [
            p for p in frontier.all_points if p.algorithm != algorithm
        ]
        if not remaining:
            return hv_full

        reduced = ParetoFrontier(
            remaining,
            {k: frontier._maximize.get(k, True) for k in frontier.objective_keys},
        )
        reduced.compute_frontier()
        hv_reduced = self.compute_hypervolume(reduced, reference_point)

        return max(0.0, hv_full - hv_reduced)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate_frontier(
        self,
        frontier: ParetoFrontier,
        num_points: int = 100,
    ) -> List[ParetoPoint]:
        """Linearly interpolate along the frontier for smooth plotting.

        Only well-defined for 2-objective frontiers (projects to the first
        two objectives otherwise).

        Parameters:
            frontier: Source frontier.
            num_points: Number of interpolated points.

        Returns:
            List of interpolated :class:`ParetoPoint` instances.
        """
        front = frontier.frontier_points
        if len(front) < 2:
            return [p.copy() for p in front]

        keys = frontier.objective_keys[:2]
        sorted_front = sorted(front, key=lambda p: p.objectives[keys[0]])

        xs = np.array([p.objectives[keys[0]] for p in sorted_front])
        ys = np.array([p.objectives[keys[1]] for p in sorted_front])

        x_interp = np.linspace(xs[0], xs[-1], num_points)
        y_interp = np.interp(x_interp, xs, ys)

        result: List[ParetoPoint] = []
        for xi, yi in zip(x_interp, y_interp):
            result.append(
                ParetoPoint(
                    objectives={keys[0]: float(xi), keys[1]: float(yi)},
                    algorithm="interpolated",
                    metadata={"interpolated": True},
                )
            )
        return result

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        frontier: ParetoFrontier,
        perturbation: float = 0.05,
        n_trials: int = 50,
        seed: int = _DEFAULT_SEED,
    ) -> Dict[str, Any]:
        """Assess frontier stability under objective perturbation.

        Each trial adds uniform noise in [-perturbation, perturbation] to each
        objective and recomputes the frontier.

        Parameters:
            frontier: Original frontier.
            perturbation: Relative magnitude of noise.
            n_trials: Number of perturbation trials.
            seed: RNG seed.

        Returns:
            Dictionary with:
              - ``survival_rates``: fraction of trials each point survived on
                the frontier.
              - ``mean_frontier_size``: average frontier size across trials.
              - ``std_frontier_size``: std dev of frontier size.
              - ``hypervolume_mean``: mean hypervolume across trials.
              - ``hypervolume_std``: std dev of hypervolume.
        """
        rng = np.random.default_rng(seed)
        keys = frontier.objective_keys
        all_pts = frontier.all_points
        n = len(all_pts)

        survival_counts = np.zeros(n, dtype=np.int64)
        frontier_sizes: List[int] = []
        hvs: List[float] = []

        base_matrix = np.array(
            [p.objective_vector(keys) for p in all_pts], dtype=np.float64
        )
        ranges = np.ptp(base_matrix, axis=0)
        ranges[ranges < _EPS] = 1.0

        for _ in range(n_trials):
            noise = rng.uniform(
                -perturbation, perturbation, size=base_matrix.shape
            ) * ranges
            perturbed_matrix = base_matrix + noise

            perturbed_pts: List[ParetoPoint] = []
            for i, p in enumerate(all_pts):
                pp = p.copy()
                for j, k in enumerate(keys):
                    pp.objectives[k] = float(perturbed_matrix[i, j])
                perturbed_pts.append(pp)

            pf = ParetoFrontier(
                perturbed_pts,
                {k: frontier._maximize.get(k, True) for k in keys},
            )
            pf_front = pf.compute_frontier()
            frontier_sizes.append(len(pf_front))

            front_ids = {id(pp) for pp in pf_front}
            for i, pp in enumerate(perturbed_pts):
                if id(pp) in front_ids:
                    survival_counts[i] += 1

            hv = self.compute_hypervolume(pf)
            hvs.append(hv)

        survival_rates = survival_counts / n_trials
        return {
            "survival_rates": {
                all_pts[i].algorithm or str(i): float(survival_rates[i])
                for i in range(n)
            },
            "mean_frontier_size": float(np.mean(frontier_sizes)),
            "std_frontier_size": float(np.std(frontier_sizes)),
            "hypervolume_mean": float(np.mean(hvs)),
            "hypervolume_std": float(np.std(hvs)),
        }

    # ------------------------------------------------------------------
    # Bootstrap frontier
    # ------------------------------------------------------------------

    def bootstrap_frontier(
        self,
        data: Sequence[ParetoPoint],
        n_bootstrap: int = 200,
        confidence: float = 0.95,
        seed: int = _DEFAULT_SEED,
    ) -> Dict[str, Any]:
        """Bootstrap confidence band for the Pareto frontier.

        Resamples *data* with replacement, recomputes the frontier each time,
        and collects statistics on hypervolume and frontier composition.

        Parameters:
            data: Points to resample.
            n_bootstrap: Number of bootstrap iterations.
            confidence: Confidence level for intervals (e.g., 0.95).
            seed: RNG seed.

        Returns:
            Dictionary with:
              - ``hypervolume_ci``: (lower, upper) confidence interval.
              - ``mean_hypervolume``: mean hypervolume across bootstraps.
              - ``frontier_inclusion_prob``: per-algorithm probability of
                appearing on the frontier.
              - ``frontier_size_ci``: (lower, upper) for frontier size.
        """
        rng = np.random.default_rng(seed)
        data_list = list(data)
        n = len(data_list)
        if n == 0:
            return {
                "hypervolume_ci": (0.0, 0.0),
                "mean_hypervolume": 0.0,
                "frontier_inclusion_prob": {},
                "frontier_size_ci": (0, 0),
            }

        keys = sorted(data_list[0].objectives.keys())

        hvs: List[float] = []
        f_sizes: List[int] = []
        algo_counts: Dict[str, int] = {}
        total_bootstraps_with_algo: Dict[str, int] = {}

        for _ in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            sample = [data_list[i].copy() for i in indices]

            pf = ParetoFrontier(sample)
            front = pf.compute_frontier()
            f_sizes.append(len(front))

            hv = self.compute_hypervolume(pf)
            hvs.append(hv)

            algos_in_front: Set[str] = set()
            for p in front:
                algos_in_front.add(p.algorithm)
            for alg in algos_in_front:
                algo_counts[alg] = algo_counts.get(alg, 0) + 1

            all_algos: Set[str] = {p.algorithm for p in sample}
            for alg in all_algos:
                total_bootstraps_with_algo[alg] = (
                    total_bootstraps_with_algo.get(alg, 0) + 1
                )

        alpha = 1.0 - confidence
        hv_arr = np.array(hvs)
        hv_lower = float(np.percentile(hv_arr, 100 * alpha / 2))
        hv_upper = float(np.percentile(hv_arr, 100 * (1 - alpha / 2)))

        fs_arr = np.array(f_sizes)
        fs_lower = int(np.percentile(fs_arr, 100 * alpha / 2))
        fs_upper = int(np.percentile(fs_arr, 100 * (1 - alpha / 2)))

        inclusion_prob: Dict[str, float] = {}
        for alg, count in algo_counts.items():
            total = total_bootstraps_with_algo.get(alg, n_bootstrap)
            inclusion_prob[alg] = count / total if total > 0 else 0.0

        return {
            "hypervolume_ci": (hv_lower, hv_upper),
            "mean_hypervolume": float(np.mean(hvs)),
            "frontier_inclusion_prob": inclusion_prob,
            "frontier_size_ci": (fs_lower, fs_upper),
        }

    # ------------------------------------------------------------------
    # Frontier comparison
    # ------------------------------------------------------------------

    def compare_frontiers(
        self,
        front_a: ParetoFrontier,
        front_b: ParetoFrontier,
        reference_point: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        """Compare two Pareto frontiers across multiple quality indicators.

        Parameters:
            front_a: First frontier.
            front_b: Second frontier.
            reference_point: Shared reference point for hypervolume.

        Returns:
            Dictionary of comparison metrics.
        """
        hv_a = self.compute_hypervolume(front_a, reference_point)
        hv_b = self.compute_hypervolume(front_b, reference_point)

        cov_ab = front_a.coverage(front_b)
        cov_ba = front_b.coverage(front_a)

        eps_ab = self.compute_epsilon_indicator(front_a, front_b)
        eps_ba = self.compute_epsilon_indicator(front_b, front_a)

        spread_a = front_a.spread_metric()
        spread_b = front_b.spread_metric()

        spacing_a = front_a.spacing_metric()
        spacing_b = front_b.spacing_metric()

        ref_points = front_a.frontier_points + front_b.frontier_points
        igd_a = self.compute_igd(front_a, front_b.frontier_points) if front_b.frontier_points else float("inf")
        igd_b = self.compute_igd(front_b, front_a.frontier_points) if front_a.frontier_points else float("inf")

        if hv_a > hv_b + _EPS:
            hv_winner = "A"
        elif hv_b > hv_a + _EPS:
            hv_winner = "B"
        else:
            hv_winner = "tie"

        return {
            "hypervolume_a": hv_a,
            "hypervolume_b": hv_b,
            "hypervolume_winner": hv_winner,
            "hypervolume_ratio": hv_a / hv_b if hv_b > _EPS else float("inf"),
            "coverage_a_over_b": cov_ab,
            "coverage_b_over_a": cov_ba,
            "epsilon_a_b": eps_ab,
            "epsilon_b_a": eps_ba,
            "spread_a": spread_a,
            "spread_b": spread_b,
            "spacing_a": spacing_a,
            "spacing_b": spacing_b,
            "igd_a": igd_a,
            "igd_b": igd_b,
            "frontier_size_a": len(front_a.frontier_points),
            "frontier_size_b": len(front_b.frontier_points),
        }

    # ------------------------------------------------------------------
    # Knee point
    # ------------------------------------------------------------------

    def knee_point(self, frontier: ParetoFrontier) -> Optional[ParetoPoint]:
        """Find the knee (elbow) point of the frontier.

        The knee is the point with the maximum perpendicular distance to the
        line connecting the two extreme points of the frontier (in normalized
        objective space).  This heuristic works best for bi-objective problems.

        Parameters:
            frontier: The frontier to search.

        Returns:
            The knee :class:`ParetoPoint`, or ``None`` if the frontier has
            fewer than 3 points.
        """
        front = frontier.frontier_points
        if len(front) < 3:
            return front[0] if front else None

        keys = frontier.objective_keys
        signs = frontier._sign_vector()
        matrix = frontier._frontier_matrix(front) * signs
        norm = _normalize_matrix(matrix)

        line_start = norm[np.argmin(norm[:, 0])]
        line_end = norm[np.argmax(norm[:, 0])]

        best_dist = -1.0
        best_idx = 0
        for i, pt in enumerate(norm):
            d = _point_to_line_distance(pt, line_start, line_end)
            if d > best_dist:
                best_dist = d
                best_idx = i

        return front[best_idx]

    # ------------------------------------------------------------------
    # Ideal / nadir
    # ------------------------------------------------------------------

    def ideal_point(self, frontier: ParetoFrontier) -> NDArray[np.float64]:
        """Ideal (utopia) point: best observed value per objective.

        Parameters:
            frontier: Source frontier.

        Returns:
            1-D array of ideal objective values (in original, un-signed space).
        """
        front = frontier.frontier_points
        if not front:
            return np.array([], dtype=np.float64)

        keys = frontier.objective_keys
        matrix = frontier._frontier_matrix(front)
        ideal = np.empty(len(keys), dtype=np.float64)
        for j, k in enumerate(keys):
            if frontier._maximize.get(k, True):
                ideal[j] = matrix[:, j].max()
            else:
                ideal[j] = matrix[:, j].min()
        return ideal

    def nadir_point(self, frontier: ParetoFrontier) -> NDArray[np.float64]:
        """Nadir point: worst objective value among the frontier points.

        Parameters:
            frontier: Source frontier.

        Returns:
            1-D array of nadir objective values.
        """
        front = frontier.frontier_points
        if not front:
            return np.array([], dtype=np.float64)

        keys = frontier.objective_keys
        matrix = frontier._frontier_matrix(front)
        nadir = np.empty(len(keys), dtype=np.float64)
        for j, k in enumerate(keys):
            if frontier._maximize.get(k, True):
                nadir[j] = matrix[:, j].min()
            else:
                nadir[j] = matrix[:, j].max()
        return nadir

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_objectives(
        self, frontier: ParetoFrontier
    ) -> ParetoFrontier:
        """Return a new frontier whose objectives are normalized to [0, 1].

        Uses the ideal and nadir points as bounds.  The maximize/minimize
        semantics are preserved (maximized objectives map 0 → worst, 1 → best).

        Parameters:
            frontier: Source frontier.

        Returns:
            New :class:`ParetoFrontier` with normalized objective values.
        """
        all_pts = frontier.all_points
        if not all_pts:
            return ParetoFrontier([], frontier._maximize)

        keys = frontier.objective_keys
        matrix = np.array(
            [p.objective_vector(keys) for p in all_pts], dtype=np.float64
        )
        ideal = self.ideal_point(frontier)
        nadir = self.nadir_point(frontier)

        # For the normalization we want 0 = worst, 1 = best per objective.
        new_pts: List[ParetoPoint] = []
        for i, p in enumerate(all_pts):
            new_obj: Dict[str, float] = {}
            for j, k in enumerate(keys):
                r = abs(ideal[j] - nadir[j])
                if r < _EPS:
                    new_obj[k] = 0.5
                else:
                    if frontier._maximize.get(k, True):
                        new_obj[k] = (matrix[i, j] - nadir[j]) / r
                    else:
                        new_obj[k] = (nadir[j] - matrix[i, j]) / r
            pp = p.copy()
            pp.objectives = new_obj
            new_pts.append(pp)

        # After normalization all objectives are effectively "maximize"
        new_maximize = {k: True for k in keys}
        nf = ParetoFrontier(new_pts, new_maximize)
        nf.compute_frontier()
        return nf


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------


def _project_points(
    points: Sequence[ParetoPoint], keys: List[str]
) -> List[ParetoPoint]:
    """Create copies of *points* keeping only the listed objectives."""
    result: List[ParetoPoint] = []
    for p in points:
        new_obj = {k: p.objectives[k] for k in keys if k in p.objectives}
        if len(new_obj) != len(keys):
            continue
        result.append(
            ParetoPoint(
                objectives=new_obj,
                algorithm=p.algorithm,
                config=dict(p.config),
                metadata=dict(p.metadata),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Extended quality indicators
# ---------------------------------------------------------------------------


class QualityIndicators:
    """Collection of Pareto-front quality indicators beyond the basic set
    provided by :class:`ParetoAnalyzer`.

    All methods are static / class methods so no instantiation is needed.
    """

    @staticmethod
    def generational_distance(
        frontier: ParetoFrontier,
        reference_front: Sequence[ParetoPoint],
        p: float = 2.0,
    ) -> float:
        """Generational Distance (GD).

        Average (Lp) distance from each point in *frontier* to the nearest
        point in *reference_front*.

        Parameters:
            frontier: Approximate front.
            reference_front: True front.
            p: Exponent for the Lp norm (default 2).

        Returns:
            Non-negative GD value.
        """
        front = frontier.frontier_points
        if not front or not reference_front:
            return float("inf")

        keys = frontier.objective_keys
        signs = frontier._sign_vector()
        f_mat = np.array(
            [pt.objective_vector(keys) * signs for pt in front], dtype=np.float64
        )
        r_mat = np.array(
            [pt.objective_vector(keys) * signs for pt in reference_front],
            dtype=np.float64,
        )
        combined = np.vstack([f_mat, r_mat])
        ideal = combined.min(axis=0)
        nadir = combined.max(axis=0)
        f_norm = _normalize_matrix(f_mat, ideal, nadir)
        r_norm = _normalize_matrix(r_mat, ideal, nadir)

        total = 0.0
        for fv in f_norm:
            min_d = float("inf")
            for rv in r_norm:
                d = _euclidean_distance(fv, rv)
                if d < min_d:
                    min_d = d
            total += min_d ** p

        return float((total / len(f_norm)) ** (1.0 / p))

    @staticmethod
    def igd_plus(
        frontier: ParetoFrontier,
        reference_front: Sequence[ParetoPoint],
    ) -> float:
        """IGD+ (modified IGD using Pareto-compliant distance).

        Uses the maximum of component-wise differences clipped at zero to
        ensure Pareto compliance.

        Parameters:
            frontier: Approximate front.
            reference_front: True front.

        Returns:
            Non-negative IGD+ value.
        """
        front = frontier.frontier_points
        if not front or not reference_front:
            return float("inf")

        keys = frontier.objective_keys
        signs = frontier._sign_vector()
        f_mat = np.array(
            [pt.objective_vector(keys) * signs for pt in front], dtype=np.float64
        )
        r_mat = np.array(
            [pt.objective_vector(keys) * signs for pt in reference_front],
            dtype=np.float64,
        )
        combined = np.vstack([f_mat, r_mat])
        ideal = combined.min(axis=0)
        nadir = combined.max(axis=0)
        f_norm = _normalize_matrix(f_mat, ideal, nadir)
        r_norm = _normalize_matrix(r_mat, ideal, nadir)

        total = 0.0
        for rv in r_norm:
            min_d = float("inf")
            for fv in f_norm:
                diff = np.maximum(rv - fv, 0.0)
                d = float(np.linalg.norm(diff))
                if d < min_d:
                    min_d = d
            total += min_d
        return total / len(r_norm)

    @staticmethod
    def maximum_spread(frontier: ParetoFrontier) -> float:
        """Maximum Spread (MS) indicator.

        Measures the range covered by the frontier in each objective dimension
        and returns the root of the summed squared ranges.

        Returns:
            Non-negative MS value.
        """
        front = frontier.frontier_points
        if len(front) < 2:
            return 0.0
        keys = frontier.objective_keys
        matrix = frontier._frontier_matrix(front)
        ranges = matrix.max(axis=0) - matrix.min(axis=0)
        return float(np.sqrt(np.sum(ranges ** 2)))

    @staticmethod
    def overall_pareto_spread(frontier: ParetoFrontier) -> float:
        """Overall Pareto Spread (OS).

        Product of per-objective ranges normalized by the objective range of
        all points (not just the front).  Values near 1 indicate good spread.

        Returns:
            Value in [0, 1].
        """
        front = frontier.frontier_points
        if len(front) < 2:
            return 0.0
        keys = frontier.objective_keys
        f_mat = frontier._frontier_matrix(front)
        a_mat = frontier._all_matrix()
        f_ranges = np.ptp(f_mat, axis=0)
        a_ranges = np.ptp(a_mat, axis=0)
        a_ranges[a_ranges < _EPS] = 1.0
        ratios = f_ranges / a_ranges
        return float(np.prod(np.clip(ratios, 0.0, 1.0)))

    @staticmethod
    def uniformity(frontier: ParetoFrontier) -> float:
        """Uniformity metric.

        Standard deviation of consecutive point distances along the frontier
        sorted by the first objective.  Lower values indicate more even spacing.

        Returns:
            Non-negative uniformity value.
        """
        front = frontier.frontier_points
        if len(front) < 3:
            return 0.0
        keys = frontier.objective_keys
        sorted_front = sorted(front, key=lambda p: p.objectives[keys[0]])
        dists: List[float] = []
        for i in range(len(sorted_front) - 1):
            a = sorted_front[i].objective_vector(keys)
            b = sorted_front[i + 1].objective_vector(keys)
            dists.append(_euclidean_distance(a, b))
        return float(np.std(dists))

    @staticmethod
    def r2_indicator(
        frontier: ParetoFrontier,
        weight_vectors: NDArray,
        utopia: Optional[NDArray] = None,
    ) -> float:
        """R2 indicator using a set of uniformly distributed weight vectors.

        For each weight vector, the minimum weighted Tchebycheff distance to
        the utopia point is computed, and the average across all weight vectors
        is returned.

        Parameters:
            frontier: The front to evaluate.
            weight_vectors: (W, M) array of weight vectors.
            utopia: Ideal point (defaults to column-wise max of the front).

        Returns:
            Non-negative R2 value.  Lower is better.
        """
        front = frontier.frontier_points
        if not front:
            return float("inf")
        keys = frontier.objective_keys
        signs = frontier._sign_vector()
        matrix = frontier._frontier_matrix(front) * signs
        if utopia is None:
            utopia = matrix.max(axis=0)

        total = 0.0
        for w in weight_vectors:
            best = float("inf")
            for row in matrix:
                val = float(np.max(w * np.abs(utopia - row)))
                if val < best:
                    best = val
            total += best
        return total / len(weight_vectors)


# ---------------------------------------------------------------------------
# Weight-vector generation utilities
# ---------------------------------------------------------------------------


def generate_uniform_weights(n_objectives: int, divisions: int) -> NDArray:
    """Generate uniformly distributed weight vectors on the unit simplex.

    Uses Das-Dennis systematic sampling (simplex-lattice design).

    Parameters:
        n_objectives: Number of objectives.
        divisions: Number of divisions per axis (``H`` in the MOEA/D literature).

    Returns:
        (W, n_objectives) array of weight vectors summing to 1.
    """
    if n_objectives == 1:
        return np.array([[1.0]])
    if divisions == 0:
        return np.full((1, n_objectives), 1.0 / n_objectives)

    def _recursive(m: int, h: int) -> List[List[float]]:
        if m == 1:
            return [[h / divisions]]
        result: List[List[float]] = []
        for i in range(h + 1):
            rest = _recursive(m - 1, h - i)
            for tail in rest:
                result.append([i / divisions] + tail)
        return result

    raw = _recursive(n_objectives, divisions)
    return np.array(raw, dtype=np.float64)


def generate_random_weights(
    n_objectives: int,
    n_vectors: int,
    seed: int = _DEFAULT_SEED,
) -> NDArray:
    """Generate random weight vectors on the unit simplex.

    Parameters:
        n_objectives: Number of objectives.
        n_vectors: Number of weight vectors.
        seed: RNG seed.

    Returns:
        (n_vectors, n_objectives) array.
    """
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, size=(n_vectors, n_objectives))
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Algorithm-specific analysis
# ---------------------------------------------------------------------------


class AlgorithmAnalysis:
    """Per-algorithm performance analysis across experiments.

    Parameters:
        results: All experimental points.
        frontier: The combined Pareto frontier.
    """

    def __init__(
        self,
        results: Sequence[ParetoPoint],
        frontier: ParetoFrontier,
    ) -> None:
        self._results = list(results)
        self._frontier = frontier
        self._algorithms = sorted({p.algorithm for p in results})

    @property
    def algorithms(self) -> List[str]:
        return list(self._algorithms)

    def points_by_algorithm(self) -> Dict[str, List[ParetoPoint]]:
        """Group all points by algorithm name."""
        groups: Dict[str, List[ParetoPoint]] = {a: [] for a in self._algorithms}
        for p in self._results:
            groups.setdefault(p.algorithm, []).append(p)
        return groups

    def frontier_fraction(self) -> Dict[str, float]:
        """Fraction of frontier points contributed by each algorithm."""
        front = self._frontier.frontier_points
        if not front:
            return {a: 0.0 for a in self._algorithms}
        counts: Dict[str, int] = {a: 0 for a in self._algorithms}
        for p in front:
            counts[p.algorithm] = counts.get(p.algorithm, 0) + 1
        total = len(front)
        return {a: c / total for a, c in counts.items()}

    def dominance_matrix(self) -> Dict[str, Dict[str, float]]:
        """Pairwise dominance fractions between algorithms.

        ``result[A][B]`` is the fraction of B's points dominated by at least
        one point from A.
        """
        groups = self.points_by_algorithm()
        keys = self._frontier.objective_keys
        signs = self._frontier._sign_vector()
        vecs: Dict[str, List[NDArray]] = {}
        for alg, pts in groups.items():
            vecs[alg] = [p.objective_vector(keys) * signs for p in pts]

        result: Dict[str, Dict[str, float]] = {}
        for a in self._algorithms:
            result[a] = {}
            for b in self._algorithms:
                if a == b:
                    result[a][b] = 0.0
                    continue
                dom_count = 0
                for bv in vecs[b]:
                    for av in vecs[a]:
                        if _vec_dominates(av, bv):
                            dom_count += 1
                            break
                nb = len(vecs[b])
                result[a][b] = dom_count / nb if nb > 0 else 0.0
        return result

    def rank_distribution(self) -> Dict[str, Dict[int, int]]:
        """Distribution of non-dominated ranks for each algorithm.

        Requires that ``fast_non_dominated_sort`` has been called beforehand.
        """
        dist: Dict[str, Dict[int, int]] = {a: {} for a in self._algorithms}
        for p in self._results:
            r = p.rank
            if r < 0:
                continue
            alg = p.algorithm
            dist[alg][r] = dist[alg].get(r, 0) + 1
        return dist

    def attainment_surface(
        self,
        algorithm: str,
        n_runs: int = 1,
    ) -> ParetoFrontier:
        """Compute the best attainment surface for a single algorithm.

        Simply returns the Pareto frontier of the algorithm's own points.

        Parameters:
            algorithm: Algorithm name.
            n_runs: Not used (reserved for multi-run attainment).

        Returns:
            :class:`ParetoFrontier` of the algorithm's points.
        """
        pts = [p for p in self._results if p.algorithm == algorithm]
        pf = ParetoFrontier(
            pts,
            {k: self._frontier._maximize.get(k, True) for k in self._frontier.objective_keys},
        )
        pf.compute_frontier()
        return pf

    def summary_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Per-algorithm summary statistics for each objective.

        Returns:
            Nested dict: ``result[algorithm][objective]`` → dict of mean, std,
            min, max, median.
        """
        groups = self.points_by_algorithm()
        keys = self._frontier.objective_keys
        stats: Dict[str, Dict[str, Any]] = {}
        for alg, pts in groups.items():
            stats[alg] = {}
            if not pts:
                continue
            matrix = np.array(
                [p.objective_vector(keys) for p in pts], dtype=np.float64
            )
            for j, k in enumerate(keys):
                col = matrix[:, j]
                stats[alg][k] = {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                    "median": float(np.median(col)),
                    "count": len(col),
                }
        return stats


# ---------------------------------------------------------------------------
# Frontier evolution tracker
# ---------------------------------------------------------------------------


class FrontierTracker:
    """Track how the Pareto frontier evolves as points are added over time.

    Useful for analysing convergence of optimization algorithms.
    """

    def __init__(
        self,
        objective_keys: Sequence[str],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._keys = list(objective_keys)
        self._maximize = maximize or {k: True for k in self._keys}
        self._snapshots: List[ParetoFrontier] = []
        self._timestamps: List[int] = []
        self._current_points: List[ParetoPoint] = []

    def add_point(self, point: ParetoPoint, timestamp: Optional[int] = None) -> None:
        """Add a point and take a snapshot of the frontier."""
        self._current_points.append(point)
        ts = timestamp if timestamp is not None else len(self._timestamps)
        self._timestamps.append(ts)

        pf = ParetoFrontier(
            [p.copy() for p in self._current_points],
            dict(self._maximize),
        )
        pf.compute_frontier()
        self._snapshots.append(pf)

    def add_batch(
        self,
        points: Sequence[ParetoPoint],
        timestamp: Optional[int] = None,
    ) -> None:
        """Add multiple points and take a single snapshot."""
        self._current_points.extend(points)
        ts = timestamp if timestamp is not None else len(self._timestamps)
        self._timestamps.append(ts)

        pf = ParetoFrontier(
            [p.copy() for p in self._current_points],
            dict(self._maximize),
        )
        pf.compute_frontier()
        self._snapshots.append(pf)

    @property
    def snapshots(self) -> List[ParetoFrontier]:
        return list(self._snapshots)

    @property
    def timestamps(self) -> List[int]:
        return list(self._timestamps)

    def hypervolume_over_time(
        self, reference_point: Optional[NDArray] = None
    ) -> List[Tuple[int, float]]:
        """Return (timestamp, hypervolume) pairs for each snapshot."""
        analyzer = ParetoAnalyzer([])
        result: List[Tuple[int, float]] = []
        for ts, snap in zip(self._timestamps, self._snapshots):
            hv = analyzer.compute_hypervolume(snap, reference_point)
            result.append((ts, hv))
        return result

    def frontier_size_over_time(self) -> List[Tuple[int, int]]:
        """Return (timestamp, frontier_size) pairs."""
        return [
            (ts, len(snap.frontier_points))
            for ts, snap in zip(self._timestamps, self._snapshots)
        ]

    def convergence_rate(
        self, reference_point: Optional[NDArray] = None, window: int = 5
    ) -> List[Tuple[int, float]]:
        """Compute smoothed rate of hypervolume improvement.

        Parameters:
            reference_point: For hypervolume computation.
            window: Moving-average window size.

        Returns:
            List of (timestamp, rate) tuples where rate is the change in
            hypervolume per step averaged over *window* steps.
        """
        hv_series = self.hypervolume_over_time(reference_point)
        if len(hv_series) < 2:
            return []

        deltas = [
            hv_series[i][1] - hv_series[i - 1][1]
            for i in range(1, len(hv_series))
        ]

        rates: List[Tuple[int, float]] = []
        for i in range(len(deltas)):
            start = max(0, i - window + 1)
            avg = float(np.mean(deltas[start : i + 1]))
            rates.append((hv_series[i + 1][0], avg))
        return rates


# ---------------------------------------------------------------------------
# Diversity-quality tradeoff utilities
# ---------------------------------------------------------------------------


class DiversityQualityTradeoff:
    """Specialised analysis for the diversity-vs-quality tradeoff that is
    central to the Diversity Decoding Arena.

    Parameters:
        results: Experimental result points.
        diversity_key: Name of the diversity objective.
        quality_key: Name of the quality objective.
        maximize_diversity: Whether higher diversity is better.
        maximize_quality: Whether higher quality is better.
    """

    def __init__(
        self,
        results: Sequence[ParetoPoint],
        diversity_key: str = "diversity",
        quality_key: str = "quality",
        maximize_diversity: bool = True,
        maximize_quality: bool = True,
    ) -> None:
        self._results = list(results)
        self._div_key = diversity_key
        self._qual_key = quality_key
        self._maximize = {
            diversity_key: maximize_diversity,
            quality_key: maximize_quality,
        }

        projected = _project_points(
            self._results, [diversity_key, quality_key]
        )
        self._frontier = ParetoFrontier(projected, self._maximize)
        self._frontier.compute_frontier()

    @property
    def frontier(self) -> ParetoFrontier:
        return self._frontier

    def tradeoff_ratio(self, point: ParetoPoint) -> float:
        """Marginal rate of substitution (slope) at *point* on the frontier.

        Approximated by the finite-difference slope between the point's two
        nearest neighbors on the frontier.

        Returns:
            Slope dQuality/dDiversity.  Negative values indicate the expected
            tradeoff (more diversity costs quality).
        """
        front = self._frontier.frontier_points
        if len(front) < 2:
            return 0.0

        sorted_f = sorted(front, key=lambda p: p.objectives[self._div_key])
        idx = None
        for i, p in enumerate(sorted_f):
            if p is point:
                idx = i
                break
        if idx is None:
            idx = int(
                np.argmin(
                    [
                        abs(p.objectives[self._div_key] - point.objectives[self._div_key])
                        for p in sorted_f
                    ]
                )
            )

        lo = max(0, idx - 1)
        hi = min(len(sorted_f) - 1, idx + 1)
        if lo == hi:
            return 0.0

        dx = sorted_f[hi].objectives[self._div_key] - sorted_f[lo].objectives[self._div_key]
        dy = sorted_f[hi].objectives[self._qual_key] - sorted_f[lo].objectives[self._qual_key]
        if abs(dx) < _EPS:
            return 0.0
        return dy / dx

    def optimal_operating_point(
        self,
        diversity_weight: float = 0.5,
    ) -> ParetoPoint:
        """Find the frontier point that best balances diversity and quality.

        Uses a weighted-sum scalarization (in normalized space).

        Parameters:
            diversity_weight: Weight for diversity (quality weight = 1 - w).

        Returns:
            Best-scoring :class:`ParetoPoint`.
        """
        front = self._frontier.frontier_points
        if not front:
            raise ValueError("Empty frontier")

        qw = 1.0 - diversity_weight
        keys = [self._div_key, self._qual_key]
        signs = self._frontier._sign_vector()
        matrix = self._frontier._frontier_matrix(front) * signs
        norm = _normalize_matrix(matrix)

        scores = diversity_weight * norm[:, 0] + qw * norm[:, 1]
        best = int(np.argmax(scores))
        return front[best]

    def area_under_frontier(self) -> float:
        """Area under the Pareto frontier curve in the diversity-quality plane.

        Higher area means better overall tradeoff performance.

        Returns:
            Non-negative area value (in un-normalized units).
        """
        front = self._frontier.frontier_points
        if len(front) < 2:
            return 0.0

        sorted_f = sorted(front, key=lambda p: p.objectives[self._div_key])
        xs = [p.objectives[self._div_key] for p in sorted_f]
        ys = [p.objectives[self._qual_key] for p in sorted_f]
        return float(np.trapz(ys, xs))

    def efficiency_score(self, point: ParetoPoint) -> float:
        """Distance from *point* to the ideal corner of the normalized
        diversity-quality space.

        Lower values indicate a more efficient point.

        Returns:
            Euclidean distance in [0, sqrt(2)].
        """
        keys = [self._div_key, self._qual_key]
        signs = self._frontier._sign_vector()
        all_mat = self._frontier._all_matrix() * signs
        norm = _normalize_matrix(all_mat)

        vec = point.objective_vector(keys) * signs
        vec_norm = (vec - all_mat.min(axis=0)) / (
            np.ptp(all_mat, axis=0) + _EPS
        )
        ideal = np.ones(2)
        return _euclidean_distance(vec_norm, ideal)

    def pareto_rank_table(self) -> List[Dict[str, Any]]:
        """Return a table with rank, crowding, and tradeoff info per point."""
        nds = NonDominatedSorting(
            [self._div_key, self._qual_key], self._maximize
        )
        fronts = nds.fast_non_dominated_sort(self._frontier.all_points)
        for front in fronts:
            nds.crowding_distance_assignment(front)

        rows: List[Dict[str, Any]] = []
        for p in self._frontier.all_points:
            rows.append({
                "algorithm": p.algorithm,
                self._div_key: p.objectives.get(self._div_key, 0.0),
                self._qual_key: p.objectives.get(self._qual_key, 0.0),
                "rank": p.rank,
                "crowding": p.crowding,
            })
        return sorted(rows, key=lambda r: (r["rank"], -r["crowding"]))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class ParetoReport:
    """Generate human-readable reports from Pareto analysis results.

    Parameters:
        analyzer: A configured :class:`ParetoAnalyzer`.
        frontier: The frontier to report on.
    """

    def __init__(
        self,
        analyzer: ParetoAnalyzer,
        frontier: ParetoFrontier,
    ) -> None:
        self._analyzer = analyzer
        self._frontier = frontier

    def summary(self) -> str:
        """One-paragraph summary of the frontier."""
        front = self._frontier.frontier_points
        keys = self._frontier.objective_keys
        n_total = self._frontier.size
        n_front = len(front)

        lines = [
            f"Pareto frontier: {n_front} non-dominated points out of {n_total} total.",
            f"Objectives ({len(keys)}): {', '.join(keys)}.",
            f"Spread: {self._frontier.spread_metric():.4f}",
            f"Spacing: {self._frontier.spacing_metric():.4f}",
        ]
        ideal = self._analyzer.ideal_point(self._frontier)
        nadir = self._analyzer.nadir_point(self._frontier)
        if len(ideal) > 0:
            ideal_str = ", ".join(
                f"{k}={v:.4f}" for k, v in zip(keys, ideal)
            )
            nadir_str = ", ".join(
                f"{k}={v:.4f}" for k, v in zip(keys, nadir)
            )
            lines.append(f"Ideal: ({ideal_str})")
            lines.append(f"Nadir: ({nadir_str})")

        knee = self._analyzer.knee_point(self._frontier)
        if knee is not None:
            knee_str = ", ".join(
                f"{k}={v:.4f}" for k, v in knee.objectives.items()
            )
            lines.append(f"Knee point: ({knee_str}) [{knee.algorithm}]")

        return "\n".join(lines)

    def algorithm_ranking(self) -> str:
        """Rank algorithms by frontier contribution."""
        aa = AlgorithmAnalysis(self._analyzer._results, self._frontier)
        fractions = aa.frontier_fraction()
        sorted_algs = sorted(fractions.items(), key=lambda x: -x[1])
        lines = ["Algorithm ranking by frontier contribution:"]
        for rank, (alg, frac) in enumerate(sorted_algs, 1):
            lines.append(f"  {rank}. {alg}: {frac:.1%}")
        return "\n".join(lines)

    def full_report(self) -> str:
        """Complete analysis report."""
        sections = [
            "=" * 60,
            "PARETO FRONTIER ANALYSIS REPORT",
            "=" * 60,
            "",
            self.summary(),
            "",
            self.algorithm_ranking(),
            "",
            "=" * 60,
        ]
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_frontier_from_dicts(
    records: Sequence[Dict[str, Any]],
    objective_keys: Sequence[str],
    algorithm_key: str = "algorithm",
    maximize: Optional[Dict[str, bool]] = None,
) -> ParetoFrontier:
    """Create a :class:`ParetoFrontier` from plain dictionaries.

    Parameters:
        records: List of dicts, each containing objective values and
            an optional algorithm identifier.
        objective_keys: Which keys to treat as objectives.
        algorithm_key: Key for the algorithm name (default ``"algorithm"``).
        maximize: Per-objective maximization flags.

    Returns:
        Fully computed :class:`ParetoFrontier`.
    """
    points: List[ParetoPoint] = []
    for rec in records:
        obj = {k: float(rec[k]) for k in objective_keys if k in rec}
        if len(obj) != len(objective_keys):
            continue
        alg = str(rec.get(algorithm_key, ""))
        extra = {k: v for k, v in rec.items() if k not in obj and k != algorithm_key}
        points.append(
            ParetoPoint(objectives=obj, algorithm=alg, metadata=extra)
        )

    if maximize is None:
        maximize = {k: True for k in objective_keys}
    pf = ParetoFrontier(points, maximize)
    pf.compute_frontier()
    return pf


def build_frontier_from_arrays(
    matrix: NDArray,
    objective_names: Sequence[str],
    algorithms: Optional[Sequence[str]] = None,
    maximize: Optional[Dict[str, bool]] = None,
) -> ParetoFrontier:
    """Create a :class:`ParetoFrontier` from a numpy array.

    Parameters:
        matrix: (N, M) array of objective values.
        objective_names: Names for the M objectives.
        algorithms: Optional per-row algorithm labels.
        maximize: Per-objective maximization flags.

    Returns:
        Fully computed :class:`ParetoFrontier`.
    """
    n, m = matrix.shape
    assert m == len(objective_names), "Column count must match objective names"
    points: List[ParetoPoint] = []
    for i in range(n):
        obj = {objective_names[j]: float(matrix[i, j]) for j in range(m)}
        alg = algorithms[i] if algorithms is not None else ""
        points.append(ParetoPoint(objectives=obj, algorithm=alg))

    if maximize is None:
        maximize = {k: True for k in objective_names}
    pf = ParetoFrontier(points, maximize)
    pf.compute_frontier()
    return pf


# ---------------------------------------------------------------------------
# Statistical tests for frontier comparison
# ---------------------------------------------------------------------------


class FrontierStatisticalTests:
    """Non-parametric statistical tests for comparing Pareto frontiers."""

    @staticmethod
    def permutation_test_hypervolume(
        points_a: Sequence[ParetoPoint],
        points_b: Sequence[ParetoPoint],
        n_permutations: int = 1000,
        seed: int = _DEFAULT_SEED,
        reference_point: Optional[NDArray] = None,
    ) -> Dict[str, float]:
        """Two-sample permutation test on hypervolume difference.

        Tests whether the hypervolume difference between the frontiers of
        *points_a* and *points_b* is statistically significant.

        Parameters:
            points_a: Points from algorithm A.
            points_b: Points from algorithm B.
            n_permutations: Number of permutations.
            seed: RNG seed.
            reference_point: For hypervolume computation.

        Returns:
            Dict with ``observed_diff``, ``p_value``, and ``effect_size``.
        """
        rng = np.random.default_rng(seed)
        analyzer = ParetoAnalyzer([])

        all_points = list(points_a) + list(points_b)
        na = len(points_a)

        def _hv(pts: Sequence[ParetoPoint]) -> float:
            if not pts:
                return 0.0
            pf = ParetoFrontier(list(pts))
            pf.compute_frontier()
            return analyzer.compute_hypervolume(pf, reference_point)

        hv_a = _hv(points_a)
        hv_b = _hv(points_b)
        observed = hv_a - hv_b

        count_extreme = 0
        perm_diffs: List[float] = []
        for _ in range(n_permutations):
            perm = rng.permutation(len(all_points))
            group_a = [all_points[i] for i in perm[:na]]
            group_b = [all_points[i] for i in perm[na:]]
            diff = _hv(group_a) - _hv(group_b)
            perm_diffs.append(diff)
            if abs(diff) >= abs(observed):
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_permutations + 1)
        std_perm = float(np.std(perm_diffs)) if perm_diffs else 1.0
        effect_size = observed / std_perm if std_perm > _EPS else 0.0

        return {
            "observed_diff": observed,
            "p_value": p_value,
            "effect_size": effect_size,
        }

    @staticmethod
    def mann_whitney_per_objective(
        points_a: Sequence[ParetoPoint],
        points_b: Sequence[ParetoPoint],
        keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Mann-Whitney U test per objective between two groups.

        Parameters:
            points_a: Group A points.
            points_b: Group B points.
            keys: Objective names (inferred if ``None``).

        Returns:
            ``result[objective]`` → dict with ``U_statistic``, ``p_value``,
            ``rank_biserial``.
        """
        if keys is None:
            keys = sorted(points_a[0].objectives.keys()) if points_a else []

        results: Dict[str, Dict[str, float]] = {}
        for k in keys:
            a_vals = np.array([p.objectives[k] for p in points_a])
            b_vals = np.array([p.objectives[k] for p in points_b])

            na, nb = len(a_vals), len(b_vals)
            if na == 0 or nb == 0:
                results[k] = {"U_statistic": 0.0, "p_value": 1.0, "rank_biserial": 0.0}
                continue

            # Manual Mann-Whitney U
            u_stat = 0.0
            for ai in a_vals:
                for bi in b_vals:
                    if ai > bi:
                        u_stat += 1.0
                    elif ai == bi:
                        u_stat += 0.5

            mu = na * nb / 2.0
            sigma = math.sqrt(na * nb * (na + nb + 1) / 12.0)
            if sigma < _EPS:
                z = 0.0
            else:
                z = (u_stat - mu) / sigma

            # Approximate two-sided p-value from z
            p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
            rank_biserial = (2 * u_stat - na * nb) / (na * nb) if na * nb > 0 else 0.0

            results[k] = {
                "U_statistic": u_stat,
                "p_value": p_value,
                "rank_biserial": rank_biserial,
            }
        return results


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Multi-run analysis
# ---------------------------------------------------------------------------


class MultiRunAnalysis:
    """Analyse results from multiple independent runs of multiple algorithms.

    Parameters:
        runs: ``runs[algorithm][run_index]`` → list of :class:`ParetoPoint`.
    """

    def __init__(
        self,
        runs: Dict[str, List[List[ParetoPoint]]],
    ) -> None:
        self._runs = runs
        self._algorithms = sorted(runs.keys())

    def per_run_hypervolumes(
        self,
        reference_point: Optional[NDArray] = None,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, List[float]]:
        """Compute hypervolume for each run of each algorithm.

        Returns:
            ``result[algorithm]`` → list of hypervolume values.
        """
        analyzer = ParetoAnalyzer([])
        result: Dict[str, List[float]] = {}
        for alg in self._algorithms:
            hvs: List[float] = []
            for run_pts in self._runs[alg]:
                pf = ParetoFrontier(run_pts, maximize)
                pf.compute_frontier()
                hvs.append(analyzer.compute_hypervolume(pf, reference_point))
            result[alg] = hvs
        return result

    def mean_hypervolume(
        self,
        reference_point: Optional[NDArray] = None,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, float]:
        """Mean hypervolume per algorithm across runs."""
        per_run = self.per_run_hypervolumes(reference_point, maximize)
        return {alg: float(np.mean(hvs)) for alg, hvs in per_run.items()}

    def hypervolume_confidence_interval(
        self,
        confidence: float = 0.95,
        reference_point: Optional[NDArray] = None,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Confidence interval for mean hypervolume per algorithm."""
        per_run = self.per_run_hypervolumes(reference_point, maximize)
        alpha = 1.0 - confidence
        result: Dict[str, Tuple[float, float]] = {}
        for alg, hvs in per_run.items():
            arr = np.array(hvs)
            lo = float(np.percentile(arr, 100 * alpha / 2))
            hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
            result[alg] = (lo, hi)
        return result

    def best_run(
        self,
        reference_point: Optional[NDArray] = None,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, int]:
        """Index of the best (highest hypervolume) run per algorithm."""
        per_run = self.per_run_hypervolumes(reference_point, maximize)
        return {
            alg: int(np.argmax(hvs)) for alg, hvs in per_run.items()
        }

    def aggregate_frontier(
        self,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> ParetoFrontier:
        """Merge all runs from all algorithms into a single frontier."""
        all_pts: List[ParetoPoint] = []
        for alg in self._algorithms:
            for run_pts in self._runs[alg]:
                all_pts.extend(run_pts)
        pf = ParetoFrontier(all_pts, maximize)
        pf.compute_frontier()
        return pf


# ---------------------------------------------------------------------------
# Scalarization methods
# ---------------------------------------------------------------------------


class Scalarization:
    """Scalarization functions for converting multi-objective problems into
    single-objective ones."""

    @staticmethod
    def weighted_sum(
        point: ParetoPoint,
        weights: Dict[str, float],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> float:
        """Weighted-sum scalarization.

        Parameters:
            point: The point to scalarize.
            weights: Per-objective weights.
            maximize: Direction flags. Minimized objectives are negated.

        Returns:
            Scalar value (higher is better).
        """
        total = 0.0
        for k, w in weights.items():
            v = point.objectives.get(k, 0.0)
            sign = 1.0 if (maximize is None or maximize.get(k, True)) else -1.0
            total += w * sign * v
        return total

    @staticmethod
    def tchebycheff(
        point: ParetoPoint,
        weights: Dict[str, float],
        ideal: Dict[str, float],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> float:
        """Weighted Tchebycheff scalarization.

        Parameters:
            point: The point to scalarize.
            weights: Per-objective weights.
            ideal: Ideal (utopia) values per objective.
            maximize: Direction flags.

        Returns:
            Scalar value (lower is better).
        """
        max_val = float("-inf")
        for k, w in weights.items():
            v = point.objectives.get(k, 0.0)
            z = ideal.get(k, 0.0)
            sign = 1.0 if (maximize is None or maximize.get(k, True)) else -1.0
            diff = abs(sign * z - sign * v)
            val = w * diff
            if val > max_val:
                max_val = val
        return max_val

    @staticmethod
    def augmented_tchebycheff(
        point: ParetoPoint,
        weights: Dict[str, float],
        ideal: Dict[str, float],
        rho: float = 0.05,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> float:
        """Augmented weighted Tchebycheff (adds a small weighted-sum term).

        Parameters:
            point: The point to scalarize.
            weights: Per-objective weights.
            ideal: Ideal values.
            rho: Augmentation coefficient.
            maximize: Direction flags.

        Returns:
            Scalar value (lower is better).
        """
        tcheb = Scalarization.tchebycheff(point, weights, ideal, maximize)
        ws = 0.0
        for k, w in weights.items():
            v = point.objectives.get(k, 0.0)
            z = ideal.get(k, 0.0)
            sign = 1.0 if (maximize is None or maximize.get(k, True)) else -1.0
            ws += w * abs(sign * z - sign * v)
        return tcheb + rho * ws

    @staticmethod
    def achievement_scalarizing(
        point: ParetoPoint,
        reference: Dict[str, float],
        weights: Dict[str, float],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> float:
        """Achievement scalarizing function.

        Minimizes the maximum weighted deviation from *reference*.

        Parameters:
            point: The point to evaluate.
            reference: Aspiration levels per objective.
            weights: Importance weights.
            maximize: Direction flags.

        Returns:
            Scalar value (lower is better).
        """
        max_dev = float("-inf")
        for k, w in weights.items():
            v = point.objectives.get(k, 0.0)
            r = reference.get(k, 0.0)
            sign = 1.0 if (maximize is None or maximize.get(k, True)) else -1.0
            dev = w * (sign * r - sign * v)
            if dev > max_dev:
                max_dev = dev
        return max_dev

    @staticmethod
    def pbi(
        point: ParetoPoint,
        weights: Dict[str, float],
        ideal: Dict[str, float],
        theta: float = 5.0,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> float:
        """Penalty-based boundary intersection (PBI) scalarization.

        Parameters:
            point: The point to evaluate.
            weights: Weight vector (defines the search direction).
            ideal: Ideal point.
            theta: Penalty parameter.
            maximize: Direction flags.

        Returns:
            Scalar value (lower is better).
        """
        keys = sorted(weights.keys())
        w = np.array([weights[k] for k in keys], dtype=np.float64)
        w_norm = w / (np.linalg.norm(w) + _EPS)

        signs = np.array(
            [1.0 if (maximize is None or maximize.get(k, True)) else -1.0 for k in keys],
            dtype=np.float64,
        )
        z = np.array([ideal.get(k, 0.0) for k in keys]) * signs
        f = np.array([point.objectives.get(k, 0.0) for k in keys]) * signs

        diff = z - f
        d1 = float(np.abs(np.dot(diff, w_norm)))
        d2 = float(np.linalg.norm(diff - np.dot(diff, w_norm) * w_norm))

        return d1 + theta * d2


# ---------------------------------------------------------------------------
# Reference-set generation
# ---------------------------------------------------------------------------


class ReferenceSetGenerator:
    """Generate reference Pareto fronts for indicator computation."""

    @staticmethod
    def from_all_algorithms(
        results: Sequence[ParetoPoint],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> List[ParetoPoint]:
        """Build a reference front from the union of all algorithm results.

        Parameters:
            results: All experimental points.
            maximize: Direction flags.

        Returns:
            Non-dominated subset.
        """
        pf = ParetoFrontier(list(results), maximize)
        return pf.compute_frontier()

    @staticmethod
    def uniform_on_hyperplane(
        n_points: int,
        n_objectives: int,
        seed: int = _DEFAULT_SEED,
    ) -> NDArray:
        """Generate uniformly distributed points on the first-quadrant
        unit hyperplane (sum of coordinates = 1).

        Useful for generating synthetic reference fronts.

        Parameters:
            n_points: Number of points.
            n_objectives: Number of objectives.
            seed: RNG seed.

        Returns:
            (n_points, n_objectives) array.
        """
        rng = np.random.default_rng(seed)
        raw = rng.exponential(1.0, size=(n_points, n_objectives))
        return raw / raw.sum(axis=1, keepdims=True)

    @staticmethod
    def concave_front(
        n_points: int,
        n_objectives: int = 2,
        p: float = 2.0,
        seed: int = _DEFAULT_SEED,
    ) -> NDArray:
        """Generate a concave reference front (Lp sphere in first quadrant).

        Parameters:
            n_points: Number of points.
            n_objectives: Number of objectives.
            p: Shape parameter (2 = circular, <2 = concave, >2 = convex).
            seed: RNG seed.

        Returns:
            (n_points, n_objectives) array.
        """
        if n_objectives == 2:
            theta = np.linspace(0, np.pi / 2, n_points)
            x = np.cos(theta) ** (2.0 / p)
            y = np.sin(theta) ** (2.0 / p)
            return np.column_stack([x, y])

        rng = np.random.default_rng(seed)
        raw = rng.uniform(0, 1, size=(n_points * 10, n_objectives))
        norms = np.sum(np.abs(raw) ** p, axis=1) ** (1.0 / p)
        raw = raw / norms[:, None]
        raw = raw[np.all(raw >= 0, axis=1)]
        if len(raw) > n_points:
            indices = rng.choice(len(raw), n_points, replace=False)
            raw = raw[indices]
        return raw

    @staticmethod
    def convex_front(
        n_points: int,
        n_objectives: int = 2,
        seed: int = _DEFAULT_SEED,
    ) -> NDArray:
        """Generate a convex reference front.

        Parameters:
            n_points: Number of points.
            n_objectives: Number of objectives.
            seed: RNG seed.

        Returns:
            (n_points, n_objectives) array.
        """
        return ReferenceSetGenerator.concave_front(
            n_points, n_objectives, p=0.5, seed=seed
        )


# ---------------------------------------------------------------------------
# Configuration-space exploration
# ---------------------------------------------------------------------------


class ConfigSpaceExplorer:
    """Explore the configuration space of algorithms to discover Pareto-optimal
    configurations.

    Parameters:
        evaluate_fn: Callable that takes a config dict and returns a dict of
            objective values.
        objective_keys: Names of the objectives.
        maximize: Direction flags.
    """

    def __init__(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        objective_keys: Sequence[str],
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._evaluate = evaluate_fn
        self._keys = list(objective_keys)
        self._maximize = maximize or {k: True for k in self._keys}
        self._evaluated: List[ParetoPoint] = []

    def evaluate_config(
        self, config: Dict[str, Any], algorithm: str = ""
    ) -> ParetoPoint:
        """Evaluate a single configuration and record the result."""
        obj = self._evaluate(config)
        point = ParetoPoint(
            objectives={k: obj[k] for k in self._keys},
            algorithm=algorithm,
            config=dict(config),
        )
        self._evaluated.append(point)
        return point

    def evaluate_grid(
        self,
        param_grid: Dict[str, Sequence[Any]],
        algorithm: str = "",
    ) -> List[ParetoPoint]:
        """Evaluate all combinations in a parameter grid.

        Parameters:
            param_grid: ``{param_name: [values]}``.
            algorithm: Algorithm label.

        Returns:
            List of evaluated :class:`ParetoPoint` instances.
        """
        param_names = sorted(param_grid.keys())
        value_lists = [param_grid[k] for k in param_names]
        results: List[ParetoPoint] = []
        for combo in itertools.product(*value_lists):
            config = dict(zip(param_names, combo))
            pt = self.evaluate_config(config, algorithm)
            results.append(pt)
        return results

    def current_frontier(self) -> ParetoFrontier:
        """Return the Pareto frontier of all evaluated points so far."""
        pf = ParetoFrontier(list(self._evaluated), dict(self._maximize))
        pf.compute_frontier()
        return pf

    def suggest_next(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_candidates: int = 100,
        seed: int = _DEFAULT_SEED,
    ) -> Dict[str, float]:
        """Suggest the next configuration to evaluate using a simple
        expected-hypervolume-improvement heuristic.

        Generates random candidates and picks the one whose addition to the
        current frontier would maximize hypervolume.

        Parameters:
            param_ranges: ``{param: (lo, hi)}``.
            n_candidates: Number of random candidates.
            seed: RNG seed.

        Returns:
            Suggested config dict.
        """
        rng = np.random.default_rng(seed)
        current_pf = self.current_frontier()
        analyzer = ParetoAnalyzer(self._evaluated)
        current_hv = analyzer.compute_hypervolume(current_pf)

        best_config: Optional[Dict[str, float]] = None
        best_improvement = float("-inf")

        keys_sorted = sorted(param_ranges.keys())
        for _ in range(n_candidates):
            config = {
                k: float(rng.uniform(param_ranges[k][0], param_ranges[k][1]))
                for k in keys_sorted
            }
            try:
                obj = self._evaluate(config)
            except Exception:
                continue
            candidate = ParetoPoint(
                objectives={k: obj[k] for k in self._keys},
                config=config,
            )
            trial_pts = list(self._evaluated) + [candidate]
            trial_pf = ParetoFrontier(trial_pts, dict(self._maximize))
            trial_pf.compute_frontier()
            trial_hv = analyzer.compute_hypervolume(trial_pf)
            improvement = trial_hv - current_hv
            if improvement > best_improvement:
                best_improvement = improvement
                best_config = config

        return best_config or {k: float(np.mean(v)) for k, v in param_ranges.items()}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class ParetoSerializer:
    """Serialize / deserialize Pareto analysis artifacts."""

    @staticmethod
    def frontier_to_dict(frontier: ParetoFrontier) -> Dict[str, Any]:
        """Convert a :class:`ParetoFrontier` to a JSON-serializable dict."""
        return {
            "objective_keys": frontier.objective_keys,
            "maximize": dict(frontier._maximize),
            "all_points": [
                {
                    "objectives": dict(p.objectives),
                    "algorithm": p.algorithm,
                    "config": dict(p.config),
                    "metadata": {
                        k: v
                        for k, v in p.metadata.items()
                        if isinstance(v, (str, int, float, bool, type(None)))
                    },
                    "rank": p.rank,
                    "crowding": p.crowding,
                }
                for p in frontier.all_points
            ],
            "frontier_indices": [
                i
                for i, p in enumerate(frontier.all_points)
                if p in frontier.frontier_points
            ],
        }

    @staticmethod
    def dict_to_frontier(data: Dict[str, Any]) -> ParetoFrontier:
        """Reconstruct a :class:`ParetoFrontier` from a serialized dict."""
        points: List[ParetoPoint] = []
        for pd_dict in data["all_points"]:
            points.append(
                ParetoPoint(
                    objectives=pd_dict["objectives"],
                    algorithm=pd_dict.get("algorithm", ""),
                    config=pd_dict.get("config", {}),
                    metadata=pd_dict.get("metadata", {}),
                    rank=pd_dict.get("rank", -1),
                    crowding=pd_dict.get("crowding", 0.0),
                )
            )
        maximize = data.get("maximize", {k: True for k in data["objective_keys"]})
        pf = ParetoFrontier(points, maximize)
        pf.compute_frontier()
        return pf

    @staticmethod
    def frontier_to_csv_rows(frontier: ParetoFrontier) -> List[Dict[str, Any]]:
        """Convert frontier points to flat dicts suitable for CSV export."""
        rows: List[Dict[str, Any]] = []
        for p in frontier.frontier_points:
            row: Dict[str, Any] = {"algorithm": p.algorithm, "rank": p.rank}
            row.update(p.objectives)
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Visualization data preparation
# ---------------------------------------------------------------------------


class VisualizationData:
    """Prepare data structures for plotting (does not depend on matplotlib)."""

    @staticmethod
    def frontier_2d(
        frontier: ParetoFrontier,
        x_key: Optional[str] = None,
        y_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare 2-D scatter + frontier-line data.

        Returns:
            Dict with ``all_x``, ``all_y``, ``front_x``, ``front_y``,
            ``algorithms`` arrays.
        """
        keys = frontier.objective_keys
        xk = x_key or keys[0]
        yk = y_key or (keys[1] if len(keys) > 1 else keys[0])

        all_pts = frontier.all_points
        front_pts = frontier.frontier_points

        sorted_front = sorted(front_pts, key=lambda p: p.objectives[xk])

        return {
            "x_key": xk,
            "y_key": yk,
            "all_x": [p.objectives[xk] for p in all_pts],
            "all_y": [p.objectives[yk] for p in all_pts],
            "all_algorithms": [p.algorithm for p in all_pts],
            "front_x": [p.objectives[xk] for p in sorted_front],
            "front_y": [p.objectives[yk] for p in sorted_front],
            "front_algorithms": [p.algorithm for p in sorted_front],
        }

    @staticmethod
    def frontier_3d(
        frontier: ParetoFrontier,
        x_key: Optional[str] = None,
        y_key: Optional[str] = None,
        z_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare 3-D scatter data."""
        keys = frontier.objective_keys
        xk = x_key or keys[0]
        yk = y_key or (keys[1] if len(keys) > 1 else keys[0])
        zk = z_key or (keys[2] if len(keys) > 2 else keys[0])

        all_pts = frontier.all_points
        front_pts = frontier.frontier_points

        return {
            "x_key": xk,
            "y_key": yk,
            "z_key": zk,
            "all_x": [p.objectives[xk] for p in all_pts],
            "all_y": [p.objectives[yk] for p in all_pts],
            "all_z": [p.objectives[zk] for p in all_pts],
            "all_algorithms": [p.algorithm for p in all_pts],
            "front_x": [p.objectives[xk] for p in front_pts],
            "front_y": [p.objectives[yk] for p in front_pts],
            "front_z": [p.objectives[zk] for p in front_pts],
        }

    @staticmethod
    def hypervolume_convergence(
        tracker: FrontierTracker,
        reference_point: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        """Prepare hypervolume-over-time plot data."""
        series = tracker.hypervolume_over_time(reference_point)
        return {
            "timestamps": [t for t, _ in series],
            "hypervolumes": [h for _, h in series],
        }

    @staticmethod
    def parallel_coordinates(
        frontier: ParetoFrontier,
    ) -> Dict[str, Any]:
        """Prepare data for a parallel-coordinates plot of the frontier."""
        front = frontier.frontier_points
        keys = frontier.objective_keys
        data: Dict[str, List[float]] = {k: [] for k in keys}
        algorithms: List[str] = []
        for p in front:
            for k in keys:
                data[k].append(p.objectives[k])
            algorithms.append(p.algorithm)
        return {"objectives": data, "algorithms": algorithms, "keys": keys}

    @staticmethod
    def radar_chart(
        point: ParetoPoint,
        frontier: ParetoFrontier,
    ) -> Dict[str, Any]:
        """Prepare data for a radar / spider chart of one point vs. the
        frontier's ideal and nadir."""
        keys = frontier.objective_keys
        analyzer = ParetoAnalyzer([])
        ideal = analyzer.ideal_point(frontier)
        nadir = analyzer.nadir_point(frontier)

        normalized: Dict[str, float] = {}
        for j, k in enumerate(keys):
            r = abs(ideal[j] - nadir[j])
            if r < _EPS:
                normalized[k] = 0.5
            elif frontier._maximize.get(k, True):
                normalized[k] = (point.objectives[k] - nadir[j]) / r
            else:
                normalized[k] = (nadir[j] - point.objectives[k]) / r

        return {
            "keys": keys,
            "values": [normalized[k] for k in keys],
            "algorithm": point.algorithm,
        }


# ---------------------------------------------------------------------------
# Extended result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HypervolumeResult:
    """Result of a hypervolume computation."""

    value: float
    reference_point: NDArray
    method: str  # "exact" or "monte_carlo"
    n_samples: Optional[int] = None
    std_error: Optional[float] = None


@dataclass
class KneePointResult:
    """Result of knee-point detection on a Pareto front."""

    point: ParetoPoint
    index: int
    angle: float
    marginal_rates: Dict[str, float]


@dataclass
class EpsilonDominanceResult:
    """Result of epsilon-dominance based front approximation."""

    epsilon_front: List[ParetoPoint]
    epsilon: float
    original_size: int
    reduced_size: int


@dataclass
class StabilityResult:
    """Result of bootstrap frontier stability analysis."""

    mean_hv: float
    std_hv: float
    ci_lower: float
    ci_upper: float
    jaccard_similarities: List[float]
    point_inclusion_freq: Dict[int, float]


@dataclass
class MultiObjectiveMetrics:
    """Collection of multi-objective performance metrics."""

    igd: float
    gd: float
    spread: float
    spacing: float
    coverage: float


# ---------------------------------------------------------------------------
# HypervolumeComputer
# ---------------------------------------------------------------------------


class HypervolumeComputer:
    """Hypervolume (S-metric / Lebesgue measure) computation.

    Supports exact computation for 2-D fronts and Monte-Carlo estimation
    for arbitrary dimensionality.
    """

    def __init__(
        self,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._maximize = maximize or {}

    # -- public API ---------------------------------------------------------

    def exact(
        self,
        points: List[ParetoPoint],
        reference_point: NDArray,
    ) -> HypervolumeResult:
        """Compute exact hypervolume of *points* w.r.t. *reference_point*.

        Uses an efficient sweep-line algorithm for 2-D and a recursive
        slicing approach for higher dimensionality.
        """
        if not points:
            return HypervolumeResult(
                value=0.0, reference_point=reference_point, method="exact"
            )
        keys = sorted(points[0].objectives.keys())
        matrix = self._to_matrix(points, keys)
        ref = np.asarray(reference_point, dtype=np.float64)

        if len(keys) == 2:
            vol = self._exact_2d(matrix, ref)
        else:
            vol = self._exact_nd(matrix, ref)

        return HypervolumeResult(
            value=float(vol), reference_point=ref, method="exact"
        )

    def monte_carlo(
        self,
        points: List[ParetoPoint],
        reference_point: NDArray,
        n_samples: int = 100_000,
        seed: int = _DEFAULT_SEED,
    ) -> HypervolumeResult:
        """Estimate hypervolume via Monte-Carlo sampling.

        Uniformly samples the bounding box defined by the ideal point and
        *reference_point*, then counts the fraction of samples dominated
        by at least one point on the front.
        """
        if not points:
            return HypervolumeResult(
                value=0.0,
                reference_point=np.asarray(reference_point),
                method="monte_carlo",
                n_samples=n_samples,
                std_error=0.0,
            )
        rng = np.random.default_rng(seed)
        keys = sorted(points[0].objectives.keys())
        matrix = self._to_matrix(points, keys)
        ref = np.asarray(reference_point, dtype=np.float64)

        ideal = matrix.min(axis=0)
        box_volume = float(np.prod(np.abs(ref - ideal)))
        if box_volume < _EPS:
            return HypervolumeResult(
                value=0.0,
                reference_point=ref,
                method="monte_carlo",
                n_samples=n_samples,
                std_error=0.0,
            )

        lo = np.minimum(ideal, ref)
        hi = np.maximum(ideal, ref)
        samples = rng.uniform(lo, hi, size=(n_samples, len(keys)))

        dominated = np.zeros(n_samples, dtype=bool)
        for row in matrix:
            dominated |= np.all(samples >= row, axis=1) & np.all(
                samples <= ref, axis=1
            )

        frac = dominated.mean()
        std_err = float(np.sqrt(frac * (1.0 - frac) / n_samples))

        return HypervolumeResult(
            value=float(frac * box_volume),
            reference_point=ref,
            method="monte_carlo",
            n_samples=n_samples,
            std_error=std_err * box_volume,
        )

    def compare(
        self,
        points_a: List[ParetoPoint],
        points_b: List[ParetoPoint],
        reference_point: NDArray,
    ) -> Dict[str, float]:
        """Compare hypervolumes of two algorithm point-sets.

        Returns a dict with ``hv_a``, ``hv_b``, ``ratio`` (a/b), and
        ``difference`` (a − b).
        """
        hv_a = self.exact(points_a, reference_point).value
        hv_b = self.exact(points_b, reference_point).value
        return {
            "hv_a": hv_a,
            "hv_b": hv_b,
            "ratio": hv_a / hv_b if abs(hv_b) > _EPS else float("inf"),
            "difference": hv_a - hv_b,
        }

    # -- helpers ------------------------------------------------------------

    def _to_matrix(
        self, points: List[ParetoPoint], keys: List[str]
    ) -> NDArray:
        """Convert points to a numpy matrix, flipping maximised objectives."""
        rows = []
        for p in points:
            row = [p.objectives[k] for k in keys]
            rows.append(row)
        matrix = np.array(rows, dtype=np.float64)
        for j, k in enumerate(keys):
            if self._maximize.get(k, True):
                matrix[:, j] = -matrix[:, j]
        return matrix

    def _exact_2d(self, matrix: NDArray, ref: NDArray) -> float:
        """Sweep-line hypervolume for two objectives (minimisation form)."""
        pts = matrix[np.all(matrix < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        order = np.argsort(pts[:, 0])
        pts = pts[order]

        vol = 0.0
        prev_y = ref[1]
        for i in range(len(pts)):
            x_width = (
                pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
            ) - pts[i, 0]
            y_height = prev_y - pts[i, 1]
            if y_height > 0 and x_width > 0:
                vol += x_width * y_height
            prev_y = min(prev_y, pts[i, 1])

        # Correct 2-D sweep via sorted contributions
        pts_sorted = pts[np.argsort(pts[:, 0])]
        vol = 0.0
        y_bound = ref[1]
        for i in range(len(pts_sorted) - 1, -1, -1):
            if pts_sorted[i, 1] < y_bound:
                x_right = (
                    pts_sorted[i + 1, 0]
                    if i + 1 < len(pts_sorted)
                    else ref[0]
                )
                vol += (x_right - pts_sorted[i, 0]) * (
                    y_bound - pts_sorted[i, 1]
                )
                y_bound = pts_sorted[i, 1]
        return float(vol)

    def _exact_nd(self, matrix: NDArray, ref: NDArray) -> float:
        """Recursive slicing hypervolume for n-D (minimisation form)."""
        pts = matrix[np.all(matrix < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        if pts.shape[1] == 1:
            return float(ref[0] - pts[:, 0].min())
        if pts.shape[1] == 2:
            return self._exact_2d(pts, ref)

        dim = pts.shape[1] - 1
        order = np.argsort(pts[:, dim])
        pts = pts[order]

        vol = 0.0
        prev_slice = ref[dim]
        for i in range(len(pts)):
            if pts[i, dim] < prev_slice:
                slice_height = prev_slice - pts[i, dim]
                sub_pts = pts[: i + 1, :dim]
                sub_ref = ref[:dim].copy()
                sub_vol = self._exact_nd(sub_pts, sub_ref)
                vol += slice_height * sub_vol
                prev_slice = pts[i, dim]
        return float(vol)


# ---------------------------------------------------------------------------
# DominatedHypervolumeComparison
# ---------------------------------------------------------------------------


class DominatedHypervolumeComparison:
    """Compare two point-sets via dominated hypervolume analysis."""

    def __init__(
        self,
        reference_point: NDArray,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._ref = np.asarray(reference_point, dtype=np.float64)
        self._hv = HypervolumeComputer(maximize=maximize)

    def compare(
        self,
        front_a: List[ParetoPoint],
        front_b: List[ParetoPoint],
    ) -> Dict[str, float]:
        """Compare dominated hypervolume of two fronts.

        Returns dict with ``hv_a``, ``hv_b``, ``ratio``, and ``difference``.
        """
        return self._hv.compare(front_a, front_b, self._ref)

    def contribution(
        self, points: List[ParetoPoint]
    ) -> List[float]:
        """Hypervolume contribution of each point.

        The contribution of point *i* is the hypervolume that would be lost
        if it were removed from the set.
        """
        total = self._hv.exact(points, self._ref).value
        contributions: List[float] = []
        for i in range(len(points)):
            subset = points[:i] + points[i + 1 :]
            hv_without = self._hv.exact(subset, self._ref).value
            contributions.append(total - hv_without)
        return contributions

    def _exclusive_hypervolume(
        self,
        point: ParetoPoint,
        others: List[ParetoPoint],
    ) -> float:
        """Hypervolume exclusively dominated by *point* and no other."""
        hv_all = self._hv.exact(others + [point], self._ref).value
        hv_others = self._hv.exact(others, self._ref).value
        return hv_all - hv_others


# ---------------------------------------------------------------------------
# NSGAIISorter
# ---------------------------------------------------------------------------


class NSGAIISorter:
    """NSGA-II style non-dominated sorting with crowding distance.

    Provides ranking, crowding-distance assignment, and tournament
    selection for multi-objective optimisation.
    """

    def __init__(self, maximize: Optional[Dict[str, bool]] = None) -> None:
        self._maximize = maximize or {}

    def sort(
        self, points: List[ParetoPoint]
    ) -> List[List[ParetoPoint]]:
        """Full NSGA-II non-dominated sorting with crowding distance.

        Returns a list of fronts (each a list of ``ParetoPoint``).
        Points are annotated with ``rank`` and ``crowding`` attributes.
        """
        fronts_idx = self._fast_non_dominated_sort(points)
        result: List[List[ParetoPoint]] = []
        for rank, indices in enumerate(fronts_idx):
            front_pts = [points[i] for i in indices]
            self._crowding_distance_assignment(front_pts)
            for p in front_pts:
                p.rank = rank
            result.append(front_pts)
        return result

    def _fast_non_dominated_sort(
        self, points: List[ParetoPoint]
    ) -> List[List[int]]:
        """Return fronts as list of lists of indices."""
        n = len(points)
        domination_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        keys = sorted(points[0].objectives.keys()) if points else []

        for p_i in range(n):
            for q_i in range(n):
                if p_i == q_i:
                    continue
                if self._dominates(points[p_i], points[q_i], keys):
                    dominated_set[p_i].append(q_i)
                elif self._dominates(points[q_i], points[p_i], keys):
                    domination_count[p_i] += 1
            if domination_count[p_i] == 0:
                fronts[0].append(p_i)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[int] = []
            for p_i in fronts[i]:
                for q_i in dominated_set[p_i]:
                    domination_count[q_i] -= 1
                    if domination_count[q_i] == 0:
                        next_front.append(q_i)
            i += 1
            if next_front:
                fronts.append(next_front)

        return [f for f in fronts if f]

    def _crowding_distance_assignment(
        self, front: List[ParetoPoint]
    ) -> None:
        """Assign crowding distance to each point in *front*."""
        n = len(front)
        if n <= 2:
            for p in front:
                p.crowding = float("inf")
            return

        keys = sorted(front[0].objectives.keys())
        distances = [0.0] * n

        for k in keys:
            order = sorted(range(n), key=lambda i: front[i].objectives[k])
            obj_min = front[order[0]].objectives[k]
            obj_max = front[order[-1]].objectives[k]
            span = obj_max - obj_min
            distances[order[0]] = float("inf")
            distances[order[-1]] = float("inf")
            if span < _EPS:
                continue
            for i in range(1, n - 1):
                distances[order[i]] += (
                    front[order[i + 1]].objectives[k]
                    - front[order[i - 1]].objectives[k]
                ) / span

        for i, p in enumerate(front):
            p.crowding = distances[i]

    def select(
        self,
        points: List[ParetoPoint],
        n_select: int,
        seed: int = _DEFAULT_SEED,
    ) -> List[ParetoPoint]:
        """Binary tournament selection using rank then crowding distance."""
        rng = np.random.default_rng(seed)
        self.sort(points)
        selected: List[ParetoPoint] = []
        indices = np.arange(len(points))
        for _ in range(n_select):
            i, j = rng.choice(indices, size=2, replace=False)
            winner = self._crowded_comparison(points[i], points[j])
            selected.append(winner)
        return selected

    def _crowded_comparison(
        self, a: ParetoPoint, b: ParetoPoint
    ) -> ParetoPoint:
        """Return the better individual by rank, breaking ties by crowding."""
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        return a if a.crowding >= b.crowding else b

    def _dominates(
        self,
        p: ParetoPoint,
        q: ParetoPoint,
        keys: List[str],
    ) -> bool:
        """Return True if *p* dominates *q*."""
        dominated_all = True
        strictly_better = False
        for k in keys:
            maximize = self._maximize.get(k, True)
            pv, qv = p.objectives[k], q.objectives[k]
            if maximize:
                if pv < qv:
                    dominated_all = False
                    break
                if pv > qv:
                    strictly_better = True
            else:
                if pv > qv:
                    dominated_all = False
                    break
                if pv < qv:
                    strictly_better = True
        return dominated_all and strictly_better


# ---------------------------------------------------------------------------
# QualityDiversityTradeoffCurves
# ---------------------------------------------------------------------------


class QualityDiversityTradeoffCurves:
    """Compute and analyse quality-vs-diversity tradeoff curves.

    The tradeoff is parameterised by a scalar *lambda* in [0, 1]:
        score = lambda * quality + (1 - lambda) * diversity
    """

    def __init__(
        self,
        quality_key: str = "quality",
        diversity_key: str = "diversity",
    ) -> None:
        self._qk = quality_key
        self._dk = diversity_key

    def compute_curve(
        self,
        points: List[ParetoPoint],
        n_lambdas: int = 50,
    ) -> List[Dict[str, float]]:
        """Compute the quality-diversity tradeoff at *n_lambdas* values.

        Returns a list of dicts with ``lambda``, ``quality``, ``diversity``,
        and ``combined`` keys.
        """
        lambdas = np.linspace(0.0, 1.0, n_lambdas)
        curve: List[Dict[str, float]] = []
        for lam in lambdas:
            best = max(
                points,
                key=lambda p, _l=lam: (
                    _l * p.objectives.get(self._qk, 0.0)
                    + (1.0 - _l) * p.objectives.get(self._dk, 0.0)
                ),
            )
            curve.append(
                {
                    "lambda": float(lam),
                    "quality": best.objectives.get(self._qk, 0.0),
                    "diversity": best.objectives.get(self._dk, 0.0),
                    "combined": float(
                        lam * best.objectives.get(self._qk, 0.0)
                        + (1.0 - lam) * best.objectives.get(self._dk, 0.0)
                    ),
                }
            )
        return curve

    def confidence_bands(
        self,
        points: List[ParetoPoint],
        n_lambdas: int = 50,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        seed: int = _DEFAULT_SEED,
    ) -> Dict[str, NDArray]:
        """Bootstrap confidence bands on the tradeoff curve.

        Returns a dict with ``lambdas``, ``mean``, ``lower``, and ``upper``
        arrays.
        """
        rng = np.random.default_rng(seed)
        lambdas = np.linspace(0.0, 1.0, n_lambdas)
        samples = np.zeros((n_bootstrap, n_lambdas))

        for b in range(n_bootstrap):
            idx = rng.integers(0, len(points), size=len(points))
            boot_pts = [points[i] for i in idx]
            curve = self.compute_curve(boot_pts, n_lambdas)
            samples[b] = [c["combined"] for c in curve]

        return {
            "lambdas": lambdas,
            "mean": samples.mean(axis=0),
            "lower": np.percentile(samples, 100 * alpha / 2, axis=0),
            "upper": np.percentile(samples, 100 * (1 - alpha / 2), axis=0),
        }

    def optimal_tradeoff(
        self,
        points: List[ParetoPoint],
        n_lambdas: int = 200,
    ) -> Dict[str, float]:
        """Find the lambda that maximises the combined metric.

        Returns a dict with the optimal ``lambda``, ``quality``,
        ``diversity``, and ``combined`` score.
        """
        curve = self.compute_curve(points, n_lambdas)
        best = max(curve, key=lambda c: c["combined"])
        return best

    def _interpolate_curve(
        self,
        curve: List[Dict[str, float]],
        target_lambda: float,
    ) -> Dict[str, float]:
        """Linearly interpolate the curve at *target_lambda*."""
        if target_lambda <= curve[0]["lambda"]:
            return curve[0]
        if target_lambda >= curve[-1]["lambda"]:
            return curve[-1]
        for i in range(len(curve) - 1):
            l0 = curve[i]["lambda"]
            l1 = curve[i + 1]["lambda"]
            if l0 <= target_lambda <= l1:
                t = (target_lambda - l0) / (l1 - l0) if l1 > l0 else 0.0
                return {
                    "lambda": target_lambda,
                    "quality": curve[i]["quality"]
                    + t * (curve[i + 1]["quality"] - curve[i]["quality"]),
                    "diversity": curve[i]["diversity"]
                    + t
                    * (curve[i + 1]["diversity"] - curve[i]["diversity"]),
                    "combined": curve[i]["combined"]
                    + t
                    * (curve[i + 1]["combined"] - curve[i]["combined"]),
                }
        return curve[-1]


# ---------------------------------------------------------------------------
# KneePointDetector
# ---------------------------------------------------------------------------


class KneePointDetector:
    """Detect knee (or elbow) points on a 2-D Pareto front.

    A knee point is the location of maximum curvature — the point where
    a small improvement in one objective requires a large sacrifice in
    the other.
    """

    def detect(
        self,
        points: List[ParetoPoint],
        obj_x: str,
        obj_y: str,
    ) -> KneePointResult:
        """Find the knee point via max perpendicular distance to the line
        connecting the two extreme points of the front.

        Parameters
        ----------
        points : list of ParetoPoint
            Front points (need not be sorted).
        obj_x, obj_y : str
            Objective keys for the two axes.

        Returns
        -------
        KneePointResult
        """
        sorted_pts = sorted(points, key=lambda p: p.objectives[obj_x])
        xs = np.array([p.objectives[obj_x] for p in sorted_pts])
        ys = np.array([p.objectives[obj_y] for p in sorted_pts])

        line_start = np.array([xs[0], ys[0]])
        line_end = np.array([xs[-1], ys[-1]])

        best_idx = 0
        best_dist = -1.0
        for i in range(len(sorted_pts)):
            pt = np.array([xs[i], ys[i]])
            d = self._point_line_distance(pt, line_start, line_end)
            if d > best_dist:
                best_dist = d
                best_idx = i

        angle = self._angle_at(xs, ys, best_idx)
        rates = self.marginal_rate(sorted_pts, obj_x, obj_y)

        return KneePointResult(
            point=sorted_pts[best_idx],
            index=best_idx,
            angle=angle,
            marginal_rates=rates,
        )

    def angle_based(
        self,
        points: List[ParetoPoint],
        obj_x: str,
        obj_y: str,
    ) -> KneePointResult:
        """Knee detection using the minimum interior angle at each point.

        The point where the angle formed by its two neighbours is smallest
        is the knee.
        """
        sorted_pts = sorted(points, key=lambda p: p.objectives[obj_x])
        xs = np.array([p.objectives[obj_x] for p in sorted_pts])
        ys = np.array([p.objectives[obj_y] for p in sorted_pts])

        best_idx = 0
        best_angle = float("inf")
        for i in range(1, len(sorted_pts) - 1):
            a = self._angle_at(xs, ys, i)
            if a < best_angle:
                best_angle = a
                best_idx = i

        rates = self.marginal_rate(sorted_pts, obj_x, obj_y)
        return KneePointResult(
            point=sorted_pts[best_idx],
            index=best_idx,
            angle=best_angle,
            marginal_rates=rates,
        )

    def marginal_rate(
        self,
        points: List[ParetoPoint],
        obj_x: str,
        obj_y: str,
    ) -> Dict[str, float]:
        """Marginal rate of substitution (MRS) at each point.

        Returns a dict mapping the stringified index to the MRS value.
        At the endpoints the rate is ``inf`` or ``nan``.
        """
        sorted_pts = sorted(points, key=lambda p: p.objectives[obj_x])
        rates: Dict[str, float] = {}
        for i in range(len(sorted_pts)):
            if i == 0 or i == len(sorted_pts) - 1:
                rates[str(i)] = float("inf")
                continue
            dx = (
                sorted_pts[i + 1].objectives[obj_x]
                - sorted_pts[i - 1].objectives[obj_x]
            )
            dy = (
                sorted_pts[i + 1].objectives[obj_y]
                - sorted_pts[i - 1].objectives[obj_y]
            )
            rates[str(i)] = abs(dy / dx) if abs(dx) > _EPS else float("inf")
        return rates

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _point_line_distance(
        pt: NDArray, line_a: NDArray, line_b: NDArray
    ) -> float:
        """Perpendicular distance from *pt* to the line through *a*, *b*."""
        ab = line_b - line_a
        ap = pt - line_a
        norm_ab = float(np.linalg.norm(ab))
        if norm_ab < _EPS:
            return float(np.linalg.norm(ap))
        cross = abs(float(ab[0] * ap[1] - ab[1] * ap[0]))
        return cross / norm_ab

    @staticmethod
    def _angle_at(xs: NDArray, ys: NDArray, i: int) -> float:
        """Interior angle at index *i* formed by neighbours i-1, i, i+1."""
        if i <= 0 or i >= len(xs) - 1:
            return float("inf")
        v1 = np.array([xs[i - 1] - xs[i], ys[i - 1] - ys[i]])
        v2 = np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]])
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + _EPS
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# EpsilonDominance
# ---------------------------------------------------------------------------


class EpsilonDominance:
    """Epsilon-dominance based approximate Pareto front construction.

    An epsilon-dominant archive retains only representative solutions
    separated by at least *epsilon* in each objective.
    """

    def __init__(
        self, maximize: Optional[Dict[str, bool]] = None
    ) -> None:
        self._maximize = maximize or {}

    def compute(
        self,
        points: List[ParetoPoint],
        epsilon: float,
    ) -> EpsilonDominanceResult:
        """Build an epsilon-dominant archive from *points*.

        Parameters
        ----------
        points : list of ParetoPoint
        epsilon : float
            The epsilon tolerance for each objective.

        Returns
        -------
        EpsilonDominanceResult
        """
        archive: List[ParetoPoint] = []
        for p in points:
            dominated = False
            new_archive: List[ParetoPoint] = []
            for a in archive:
                if self._epsilon_dominates(a, p, epsilon):
                    dominated = True
                    new_archive.append(a)
                elif self._epsilon_dominates(p, a, epsilon):
                    continue  # p epsilon-dominates a; drop a
                else:
                    new_archive.append(a)
            if not dominated:
                new_archive.append(p)
            archive = new_archive

        return EpsilonDominanceResult(
            epsilon_front=archive,
            epsilon=epsilon,
            original_size=len(points),
            reduced_size=len(archive),
        )

    def adaptive_epsilon(
        self,
        points: List[ParetoPoint],
        target_size: int = 20,
    ) -> float:
        """Automatically choose epsilon to achieve roughly *target_size*.

        Uses a binary search over epsilon values.
        """
        if not points:
            return 0.0
        keys = sorted(points[0].objectives.keys())
        spreads = []
        for k in keys:
            vals = [p.objectives[k] for p in points]
            spreads.append(max(vals) - min(vals))
        max_eps = max(spreads) if spreads else 1.0

        lo, hi = 0.0, max_eps
        best_eps = hi / 2.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            res = self.compute(points, mid)
            if res.reduced_size > target_size:
                lo = mid
            else:
                hi = mid
            best_eps = mid
        return best_eps

    def _epsilon_dominates(
        self,
        p: ParetoPoint,
        q: ParetoPoint,
        epsilon: float,
    ) -> bool:
        """Return True if *p* epsilon-dominates *q*.

        For a maximised objective: p_k >= q_k - epsilon for all k,
        and strictly better in at least one.
        """
        keys = sorted(p.objectives.keys())
        all_ok = True
        strictly_better = False
        for k in keys:
            maximize = self._maximize.get(k, True)
            pv, qv = p.objectives[k], q.objectives[k]
            if maximize:
                if pv < qv - epsilon:
                    all_ok = False
                    break
                if pv > qv + _EPS:
                    strictly_better = True
            else:
                if pv > qv + epsilon:
                    all_ok = False
                    break
                if pv < qv - _EPS:
                    strictly_better = True
        return all_ok and strictly_better


# ---------------------------------------------------------------------------
# FrontierStabilityAnalysis
# ---------------------------------------------------------------------------


class FrontierStabilityAnalysis:
    """Assess the stability of a Pareto frontier via bootstrap resampling."""

    def __init__(
        self,
        maximize: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._maximize = maximize or {}

    def bootstrap_stability(
        self,
        points: List[ParetoPoint],
        reference_point: NDArray,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        seed: int = _DEFAULT_SEED,
    ) -> StabilityResult:
        """Bootstrap resample *points*, compute the frontier and HV each time.

        Returns a ``StabilityResult`` with HV statistics, Jaccard
        similarities, and point inclusion frequencies.
        """
        rng = np.random.default_rng(seed)
        hv_comp = HypervolumeComputer(maximize=self._maximize)
        sorter = NSGAIISorter(maximize=self._maximize)

        hvs: List[float] = []
        boot_fronts: List[Set[int]] = []
        n = len(points)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_pts = [points[i] for i in idx]
            fronts = sorter.sort(boot_pts)
            front0 = fronts[0] if fronts else []
            hv = hv_comp.exact(front0, reference_point).value
            hvs.append(hv)
            front_indices = set(int(idx[i]) for i, p in enumerate(boot_pts) if p.rank == 0)
            boot_fronts.append(front_indices)

        hvs_arr = np.array(hvs)
        mean_hv = float(hvs_arr.mean())
        std_hv = float(hvs_arr.std(ddof=1))
        ci_lower = float(np.percentile(hvs_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(hvs_arr, 100 * (1.0 - alpha / 2)))

        jaccards = self._pairwise_jaccard(boot_fronts)
        freq = self._inclusion_frequency(boot_fronts, n, n_bootstrap)

        return StabilityResult(
            mean_hv=mean_hv,
            std_hv=std_hv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            jaccard_similarities=jaccards,
            point_inclusion_freq=freq,
        )

    def point_inclusion_frequency(
        self,
        points: List[ParetoPoint],
        n_bootstrap: int = 200,
        seed: int = _DEFAULT_SEED,
    ) -> Dict[int, float]:
        """How often each point index appears on bootstrap frontiers."""
        rng = np.random.default_rng(seed)
        sorter = NSGAIISorter(maximize=self._maximize)
        n = len(points)
        counts = np.zeros(n)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_pts = [points[i] for i in idx]
            fronts = sorter.sort(boot_pts)
            for i_b, p in enumerate(boot_pts):
                if p.rank == 0:
                    counts[int(idx[i_b])] += 1

        return {i: float(counts[i] / n_bootstrap) for i in range(n)}

    def hypervolume_distribution(
        self,
        points: List[ParetoPoint],
        reference_point: NDArray,
        n_bootstrap: int = 200,
        seed: int = _DEFAULT_SEED,
    ) -> NDArray:
        """Return the distribution of HV across bootstrap samples."""
        rng = np.random.default_rng(seed)
        hv_comp = HypervolumeComputer(maximize=self._maximize)
        sorter = NSGAIISorter(maximize=self._maximize)
        n = len(points)
        hvs = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_pts = [points[i] for i in idx]
            fronts = sorter.sort(boot_pts)
            front0 = fronts[0] if fronts else []
            hvs[b] = hv_comp.exact(front0, reference_point).value

        return hvs

    def jaccard_similarity(
        self,
        front_a: Set[int],
        front_b: Set[int],
    ) -> float:
        """Jaccard similarity between two sets of point indices."""
        if not front_a and not front_b:
            return 1.0
        inter = len(front_a & front_b)
        union = len(front_a | front_b)
        return inter / union if union > 0 else 1.0

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _pairwise_jaccard(
        fronts: List[Set[int]],
    ) -> List[float]:
        """Average pairwise Jaccard over consecutive bootstrap fronts."""
        jaccards: List[float] = []
        for i in range(len(fronts) - 1):
            inter = len(fronts[i] & fronts[i + 1])
            union = len(fronts[i] | fronts[i + 1])
            jaccards.append(inter / union if union > 0 else 1.0)
        return jaccards

    @staticmethod
    def _inclusion_frequency(
        fronts: List[Set[int]],
        n_points: int,
        n_bootstrap: int,
    ) -> Dict[int, float]:
        """Fraction of bootstrap runs each point appears on the front."""
        counts = np.zeros(n_points)
        for f in fronts:
            for idx in f:
                if 0 <= idx < n_points:
                    counts[idx] += 1
        return {i: float(counts[i] / n_bootstrap) for i in range(n_points)}


# ---------------------------------------------------------------------------
# MultiObjectivePerformanceMetrics
# ---------------------------------------------------------------------------


class MultiObjectivePerformanceMetrics:
    """Standard multi-objective quality indicators.

    All methods accept arrays of shape ``(n, d)`` where *n* is the number
    of points and *d* is the number of objectives.  Objectives are assumed
    to be in minimisation form unless *maximize* is specified.
    """

    def __init__(
        self, maximize: Optional[Dict[str, bool]] = None
    ) -> None:
        self._maximize = maximize or {}

    def igd(
        self,
        obtained: NDArray,
        reference: NDArray,
    ) -> float:
        """Inverted Generational Distance (IGD).

        Mean distance from each reference point to the nearest obtained
        point.  Lower is better.
        """
        if len(obtained) == 0 or len(reference) == 0:
            return float("inf")
        dists = self._cdist(reference, obtained)
        return float(dists.min(axis=1).mean())

    def gd(
        self,
        obtained: NDArray,
        reference: NDArray,
    ) -> float:
        """Generational Distance (GD).

        Mean distance from each obtained point to the nearest reference
        point.  Lower is better.
        """
        if len(obtained) == 0 or len(reference) == 0:
            return float("inf")
        dists = self._cdist(obtained, reference)
        return float(dists.min(axis=1).mean())

    def spread(
        self,
        obtained: NDArray,
    ) -> float:
        """Spread / Delta indicator measuring extent of the front.

        Computed as the maximum extent across all objectives.
        """
        if len(obtained) < 2:
            return 0.0
        ranges = obtained.max(axis=0) - obtained.min(axis=0)
        return float(ranges.sum())

    def spacing(
        self,
        obtained: NDArray,
    ) -> float:
        """Spacing metric measuring uniformity of point distribution.

        Lower values indicate more uniform spacing.
        """
        n = len(obtained)
        if n < 2:
            return 0.0
        dists = self._cdist(obtained, obtained)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        d_mean = min_dists.mean()
        return float(np.sqrt(((min_dists - d_mean) ** 2).sum() / n))

    def coverage(
        self,
        front_a: NDArray,
        front_b: NDArray,
    ) -> float:
        """C-metric: fraction of *front_b* points dominated by *front_a*.

        A value of 1.0 means every point in *front_b* is dominated by
        some point in *front_a*.
        """
        if len(front_b) == 0:
            return 1.0
        if len(front_a) == 0:
            return 0.0
        count = 0
        for q in front_b:
            for p in front_a:
                if np.all(p <= q + _EPS) and np.any(p < q - _EPS):
                    count += 1
                    break
        return count / len(front_b)

    def igd_plus(
        self,
        obtained: NDArray,
        reference: NDArray,
    ) -> float:
        """IGD+ (Pareto-compliant variant of IGD).

        Uses the Pareto-compliant distance: for each reference point,
        the distance to the nearest obtained point considers only
        dimensions where the obtained point is worse.
        """
        if len(obtained) == 0 or len(reference) == 0:
            return float("inf")
        total = 0.0
        for r in reference:
            min_d = float("inf")
            for o in obtained:
                diff = np.maximum(o - r, 0.0)
                d = float(np.linalg.norm(diff))
                if d < min_d:
                    min_d = d
            total += min_d
        return total / len(reference)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _cdist(a: NDArray, b: NDArray) -> NDArray:
        """Pairwise Euclidean distance matrix without scipy."""
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))


# ---------------------------------------------------------------------------
# AdaptiveReferencePoint
# ---------------------------------------------------------------------------


class AdaptiveReferencePoint:
    """Strategies for choosing or adapting the hypervolume reference point."""

    @staticmethod
    def from_frontier(
        points: List[ParetoPoint],
        margin: float = 0.1,
    ) -> NDArray:
        """Compute a reference point from the frontier with a relative margin.

        For each objective the reference is set to the worst observed value
        plus *margin* × the objective range.
        """
        if not points:
            return np.array([])
        keys = sorted(points[0].objectives.keys())
        worst = np.array([
            max(p.objectives[k] for p in points) for k in keys
        ])
        best = np.array([
            min(p.objectives[k] for p in points) for k in keys
        ])
        span = np.abs(worst - best)
        return worst + margin * np.where(span > _EPS, span, 1.0)

    @staticmethod
    def from_nadir(
        nadir: NDArray,
        factor: float = 1.1,
    ) -> NDArray:
        """Scale the nadir point by a multiplicative *factor*.

        Parameters
        ----------
        nadir : NDArray
            The nadir (anti-ideal) point.
        factor : float
            Multiplicative factor (>1 to push beyond the nadir).

        Returns
        -------
        NDArray
        """
        return np.asarray(nadir, dtype=np.float64) * factor

    @staticmethod
    def adaptive(
        history: List[List[ParetoPoint]],
        margin: float = 0.1,
    ) -> NDArray:
        """Dynamically adjust reference point as the frontier evolves.

        Uses the union of all historical points to compute a stable
        reference.

        Parameters
        ----------
        history : list of list of ParetoPoint
            Snapshots of frontiers over time.
        margin : float
            Relative margin beyond the worst values.

        Returns
        -------
        NDArray
        """
        all_pts: List[ParetoPoint] = []
        for snapshot in history:
            all_pts.extend(snapshot)
        return AdaptiveReferencePoint.from_frontier(all_pts, margin=margin)


# ---------------------------------------------------------------------------
# FrontierInterpolator
# ---------------------------------------------------------------------------


class FrontierInterpolator:
    """Interpolation and resampling of 2-D Pareto fronts."""

    def __init__(self, obj_x: str, obj_y: str) -> None:
        self._ox = obj_x
        self._oy = obj_y

    def linear(
        self,
        points: List[ParetoPoint],
        n_out: int = 100,
    ) -> NDArray:
        """Piecewise linear interpolation of a 2-D Pareto front.

        Returns an ``(n_out, 2)`` array of interpolated coordinates.
        """
        pts = sorted(points, key=lambda p: p.objectives[self._ox])
        xs = np.array([p.objectives[self._ox] for p in pts])
        ys = np.array([p.objectives[self._oy] for p in pts])

        x_interp = np.linspace(xs[0], xs[-1], n_out)
        y_interp = np.interp(x_interp, xs, ys)
        return np.column_stack([x_interp, y_interp])

    def cubic_spline(
        self,
        points: List[ParetoPoint],
        n_out: int = 100,
    ) -> NDArray:
        """Natural cubic spline interpolation of a 2-D front.

        Falls back to linear interpolation if fewer than 4 points.

        Returns an ``(n_out, 2)`` array.
        """
        pts = sorted(points, key=lambda p: p.objectives[self._ox])
        if len(pts) < 4:
            return self.linear(points, n_out)

        xs = np.array([p.objectives[self._ox] for p in pts])
        ys = np.array([p.objectives[self._oy] for p in pts])

        # Thomas algorithm for natural cubic spline
        n = len(xs) - 1
        h = np.diff(xs)
        alpha = np.zeros(n + 1)
        for i in range(1, n):
            alpha[i] = (
                3.0 / h[i] * (ys[i + 1] - ys[i])
                - 3.0 / h[i - 1] * (ys[i] - ys[i - 1])
            )

        l = np.ones(n + 1)
        mu = np.zeros(n + 1)
        z = np.zeros(n + 1)
        for i in range(1, n):
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = np.zeros(n + 1)
        b = np.zeros(n)
        d = np.zeros(n)
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (
                c[j + 1] + 2.0 * c[j]
            ) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        x_interp = np.linspace(xs[0], xs[-1], n_out)
        y_interp = np.empty(n_out)
        for idx, xv in enumerate(x_interp):
            seg = min(int(np.searchsorted(xs[1:], xv)), n - 1)
            dx = xv - xs[seg]
            y_interp[idx] = (
                ys[seg] + b[seg] * dx + c[seg] * dx ** 2 + d[seg] * dx ** 3
            )

        return np.column_stack([x_interp, y_interp])

    def extrapolate(
        self,
        points: List[ParetoPoint],
        x_target: float,
    ) -> float:
        """Extrapolate beyond frontier bounds using the local linear trend.

        Uses the two nearest boundary points to project.
        """
        pts = sorted(points, key=lambda p: p.objectives[self._ox])
        xs = [p.objectives[self._ox] for p in pts]
        ys = [p.objectives[self._oy] for p in pts]

        if x_target <= xs[0]:
            if len(xs) < 2:
                return ys[0]
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0] + _EPS)
            return ys[0] + slope * (x_target - xs[0])
        else:
            if len(xs) < 2:
                return ys[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2] + _EPS)
            return ys[-1] + slope * (x_target - xs[-1])

    def resample(
        self,
        points: List[ParetoPoint],
        n_out: int = 50,
    ) -> List[ParetoPoint]:
        """Resample the frontier at approximately uniform arc-length spacing.

        Returns new ``ParetoPoint`` instances at the resampled locations.
        """
        pts = sorted(points, key=lambda p: p.objectives[self._ox])
        xs = np.array([p.objectives[self._ox] for p in pts])
        ys = np.array([p.objectives[self._oy] for p in pts])

        # Compute cumulative arc length
        dx = np.diff(xs)
        dy = np.diff(ys)
        seg_len = np.sqrt(dx ** 2 + dy ** 2)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = cum_len[-1]

        if total_len < _EPS:
            return list(pts)

        target_s = np.linspace(0.0, total_len, n_out)
        x_resamp = np.interp(target_s, cum_len, xs)
        y_resamp = np.interp(target_s, cum_len, ys)

        resampled: List[ParetoPoint] = []
        for xi, yi in zip(x_resamp, y_resamp):
            resampled.append(
                ParetoPoint(
                    objectives={self._ox: float(xi), self._oy: float(yi)},
                    algorithm="interpolated",
                    config={},
                    metadata={},
                )
            )
        return resampled
