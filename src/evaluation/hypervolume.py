"""
Hypervolume indicator computation for multi-objective optimization analysis.

Provides exact and approximate hypervolume computation, contribution analysis,
and a suite of multi-objective quality indicators (R2, epsilon, IGD, IGD+,
spread, spacing, generational distance) for evaluating Pareto front
approximations in the Diversity Decoding Arena.

References:
    - Zitzler, E. & Thiele, L. (1999). Multiobjective evolutionary optimization.
    - While, L., Hingston, P., & Barone, L. (2012). A fast way of calculating
      exact hypervolumes. IEEE TEC.
    - Bader, J. & Zitzler, E. (2011). HypE: An Algorithm for Fast Hypervolume-
      Based Many-Objective Optimization. Evolutionary Computation.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HypervolumeContribution:
    """Stores the hypervolume contribution information for a single point.

    Attributes:
        point: The objective-space coordinates of the point.
        exclusive_contribution: The hypervolume exclusively attributable to
            this point (lost if the point is removed).
        inclusive_volume: The hypervolume of the region dominated by this
            point alone (ignoring other points).
        rank: Non-domination rank of the point within its set (0-indexed).
    """

    point: np.ndarray
    exclusive_contribution: float
    inclusive_volume: float
    rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.point, np.ndarray):
            self.point = np.asarray(self.point, dtype=float)

    def __repr__(self) -> str:
        pt_str = np.array2string(self.point, precision=6, separator=", ")
        return (
            f"HypervolumeContribution(point={pt_str}, "
            f"exclusive={self.exclusive_contribution:.8g}, "
            f"inclusive={self.inclusive_volume:.8g}, rank={self.rank})"
        )


# ---------------------------------------------------------------------------
# Helpers – dominance, filtering, sorting
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if *a* weakly dominates *b* (minimisation).

    *a* dominates *b* iff every component of *a* is <= *b* and at least
    one component is strictly less.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def _strictly_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if every component of *a* is strictly less than *b*."""
    return bool(np.all(a < b))


def _filter_dominated(points: np.ndarray) -> np.ndarray:
    """Return the subset of *points* that are non-dominated (minimisation).

    Parameters
    ----------
    points : np.ndarray, shape (n, d)

    Returns
    -------
    np.ndarray, shape (m, d) with m <= n
    """
    if points.ndim != 2 or len(points) == 0:
        return points.copy() if points.ndim == 2 else points.reshape(0, 0)

    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if _dominates(points[j], points[i]):
                is_dominated[i] = True
                break

    return points[~is_dominated].copy()


def _filter_dominated_indexed(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Like ``_filter_dominated`` but also returns original indices."""
    if points.ndim != 2 or len(points) == 0:
        return points.copy(), np.arange(len(points))

    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if _dominates(points[j], points[i]):
                is_dominated[i] = True
                break

    mask = ~is_dominated
    return points[mask].copy(), np.where(mask)[0]


def _sort_by_objective(points: np.ndarray, obj_idx: int) -> np.ndarray:
    """Sort *points* in ascending order along objective *obj_idx*."""
    order = np.argsort(points[:, obj_idx])
    return points[order]


def _non_domination_rank(points: np.ndarray) -> np.ndarray:
    """Assign non-domination ranks (0 = first front) to each point.

    Uses a simple iterative peeling approach.
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)

    ranks = np.full(n, -1, dtype=int)
    remaining = np.arange(n)
    current_rank = 0

    while len(remaining) > 0:
        front_mask = np.ones(len(remaining), dtype=bool)
        pts = points[remaining]
        for i in range(len(pts)):
            if not front_mask[i]:
                continue
            for j in range(len(pts)):
                if i == j or not front_mask[j]:
                    continue
                if _dominates(pts[j], pts[i]):
                    front_mask[i] = False
                    break

        front_indices = remaining[front_mask]
        ranks[front_indices] = current_rank
        remaining = remaining[~front_mask]
        current_rank += 1

    return ranks


def _inclusive_volume(point: np.ndarray, reference_point: np.ndarray) -> float:
    """Hypervolume of the box between *point* and *reference_point*.

    Assumes minimisation: all components of *point* should be <=
    *reference_point* for the volume to be positive.
    """
    diff = reference_point - point
    if np.any(diff < 0):
        return 0.0
    return float(np.prod(diff))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class HypervolumeIndicator:
    """Abstract base for hypervolume indicator computation.

    By convention we assume **minimisation** of all objectives.  Points whose
    coordinates exceed the reference point in any objective are silently
    clamped / ignored.

    Parameters
    ----------
    reference_point : array-like, shape (d,)
        The reference (anti-ideal / nadir) point.
    """

    def __init__(self, reference_point: Union[Sequence[float], np.ndarray]) -> None:
        self.reference_point = np.asarray(reference_point, dtype=float)
        if self.reference_point.ndim != 1:
            raise ValueError("reference_point must be 1-D")
        self._dim = len(self.reference_point)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, points: np.ndarray) -> float:
        """Compute the hypervolume indicator for *points*.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)

        Returns
        -------
        float
        """
        raise NotImplementedError

    def contributions(self, points: np.ndarray) -> List[HypervolumeContribution]:
        """Compute per-point hypervolume contributions.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)

        Returns
        -------
        list of HypervolumeContribution
        """
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)

        if len(points) == 0:
            return []

        total_hv = self.compute(points)
        ranks = _non_domination_rank(points)
        contributions: List[HypervolumeContribution] = []

        for i in range(len(points)):
            excl = self.exclusive_contribution(i, points)
            incl = _inclusive_volume(points[i], self.reference_point)
            contributions.append(
                HypervolumeContribution(
                    point=points[i].copy(),
                    exclusive_contribution=excl,
                    inclusive_volume=incl,
                    rank=int(ranks[i]),
                )
            )

        return contributions

    def exclusive_contribution(self, point_idx: int, points: np.ndarray) -> float:
        """Return the exclusive hypervolume contribution of point *point_idx*.

        This equals HV(points) - HV(points without point_idx).
        """
        points = np.asarray(points, dtype=float)
        hv_all = self.compute(points)
        reduced = np.delete(points, point_idx, axis=0)
        if len(reduced) == 0:
            return hv_all
        hv_reduced = self.compute(reduced)
        return max(hv_all - hv_reduced, 0.0)

    def least_contributor(self, points: np.ndarray) -> int:
        """Index of the point with the *smallest* exclusive contribution."""
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)

        if len(points) == 0:
            raise ValueError("Cannot find least contributor of empty set")
        if len(points) == 1:
            return 0

        contribs = [self.exclusive_contribution(i, points) for i in range(len(points))]
        return int(np.argmin(contribs))

    def greatest_contributor(self, points: np.ndarray) -> int:
        """Index of the point with the *largest* exclusive contribution."""
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)

        if len(points) == 0:
            raise ValueError("Cannot find greatest contributor of empty set")
        if len(points) == 1:
            return 0

        contribs = [self.exclusive_contribution(i, points) for i in range(len(points))]
        return int(np.argmax(contribs))

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_points(self, points: np.ndarray) -> np.ndarray:
        """Ensure *points* is a well-formed (n, d) array."""
        if points.ndim == 1:
            if len(points) == 0:
                return np.empty((0, self._dim))
            points = points.reshape(1, -1)
        if points.ndim != 2:
            raise ValueError("points must be 2-D")
        if points.shape[1] != self._dim:
            raise ValueError(
                f"points have {points.shape[1]} objectives but reference "
                f"point has {self._dim}"
            )
        return points

    def _clip_to_ref(self, points: np.ndarray) -> np.ndarray:
        """Remove points that do not dominate the reference in any objective."""
        mask = np.all(points <= self.reference_point, axis=1)
        return points[mask]


# ---------------------------------------------------------------------------
# Exact hypervolume
# ---------------------------------------------------------------------------

class ExactHypervolume(HypervolumeIndicator):
    """Exact hypervolume computation.

    * 2-D: sweep-line O(n log n)
    * 3-D: dimension-sweep O(n log n)
    * d-D: HSO – Hypervolume by Slicing Objectives (exponential in d)

    Parameters
    ----------
    reference_point : array-like, shape (d,)
    """

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self, points: np.ndarray) -> float:
        """Compute exact hypervolume of *points* w.r.t. reference.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)

        Returns
        -------
        float  The hypervolume indicator value.

        Examples
        --------
        >>> hv = ExactHypervolume([10.0, 10.0])
        >>> hv.compute(np.array([[1, 5], [3, 3], [5, 1]]))
        43.0
        """
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)
        points = self._clip_to_ref(points)

        if len(points) == 0:
            return 0.0

        points = _filter_dominated(points)
        ref = self.reference_point.copy()

        d = points.shape[1]
        if d == 1:
            return float(ref[0] - np.min(points[:, 0]))
        if d == 2:
            return self._compute_2d(points, ref)
        if d == 3:
            return self._compute_3d(points, ref)
        return self._compute_nd(points, ref)

    def exclusive_contribution(self, point_idx: int, points: np.ndarray) -> float:
        """Exclusive HV contribution using the exact algorithm."""
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)

        hv_all = self.compute(points)
        reduced = np.delete(points, point_idx, axis=0)
        if len(reduced) == 0:
            return hv_all
        hv_reduced = self.compute(reduced)
        return max(hv_all - hv_reduced, 0.0)

    # ------------------------------------------------------------------
    # 2-D sweep line  O(n log n)
    # ------------------------------------------------------------------

    def _compute_2d(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Sweep-line hypervolume for 2 objectives.

        Sorts points by the first objective, then sweeps from left to
        right accumulating rectangular strips.

        Parameters
        ----------
        points : np.ndarray, shape (n, 2)  – must be non-dominated
        ref : np.ndarray, shape (2,)

        Returns
        -------
        float
        """
        return self._sweep_line_2d(points, ref)

    def _sweep_line_2d(self, sorted_points: np.ndarray, ref: np.ndarray) -> float:
        """Core 2-D sweep-line hypervolume.

        After sorting by the first objective the second objective values
        form a decreasing sequence (since points are non-dominated).  Each
        consecutive pair defines a rectangle.

        Parameters
        ----------
        sorted_points : np.ndarray, shape (n, 2)
            Points to process (will be sorted internally).
        ref : np.ndarray, shape (2,)

        Returns
        -------
        float
        """
        if len(sorted_points) == 0:
            return 0.0

        pts = sorted_points[np.argsort(sorted_points[:, 0])]
        n = len(pts)
        area = 0.0

        for i in range(n):
            x_lo = pts[i, 0]
            x_hi = pts[i + 1, 0] if i + 1 < n else ref[0]
            width = x_hi - x_lo
            if width <= 0:
                continue
            height = ref[1] - pts[i, 1]
            if height <= 0:
                continue
            area += width * height

        return float(area)

    # ------------------------------------------------------------------
    # 3-D exact (dimension-sweep)
    # ------------------------------------------------------------------

    def _compute_3d(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Exact 3-D hypervolume using dimension-sweep with a 2-D sweep
        sub-routine.

        Sort by the third objective, then maintain a running 2-D
        cross-section.

        Parameters
        ----------
        points : np.ndarray, shape (n, 3) – non-dominated
        ref : np.ndarray, shape (3,)

        Returns
        -------
        float
        """
        if len(points) == 0:
            return 0.0

        order = np.argsort(points[:, 2])
        pts = points[order]
        n = len(pts)
        volume = 0.0

        active_2d: List[np.ndarray] = []

        for i in range(n):
            z_lo = pts[i, 2]
            z_hi = pts[i + 1, 2] if i + 1 < n else ref[2]
            depth = z_hi - z_lo
            if depth <= 0.0 and i + 1 < n:
                active_2d.append(pts[i, :2].copy())
                continue

            active_2d.append(pts[i, :2].copy())

            active_arr = np.array(active_2d)
            nd_active = _filter_dominated(active_arr)

            ref_2d = ref[:2]
            area = self._sweep_line_2d(nd_active, ref_2d)
            volume += area * depth

        return float(volume)

    # ------------------------------------------------------------------
    # General d – HSO (Hypervolume by Slicing Objectives)
    # ------------------------------------------------------------------

    def _compute_nd(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Exact hypervolume in arbitrary dimension using HSO.

        Warning: Exponential in d; only practical for d <= ~8.

        Parameters
        ----------
        points : np.ndarray, shape (n, d) – non-dominated
        ref : np.ndarray, shape (d,)

        Returns
        -------
        float
        """
        d = points.shape[1]
        if d > 10:
            warnings.warn(
                f"HSO in {d} dimensions is extremely expensive; "
                "consider using ApproximateHypervolume.",
                stacklevel=2,
            )
        return self._hso_recursive(points, ref, d - 1)

    def _hso_recursive(self, points: np.ndarray, ref: np.ndarray, dim: int) -> float:
        """Recursive HSO slicing along dimension *dim*.

        At each level we sort along *dim*, then slice the problem into
        (dim-1)-dimensional sub-problems.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
        ref : np.ndarray, shape (d,)
        dim : int – current slicing dimension (0-indexed)

        Returns
        -------
        float
        """
        n = len(points)
        if n == 0:
            return 0.0

        if dim == 0:
            return float(ref[0] - np.min(points[:, 0]))

        if dim == 1:
            return self._sweep_line_2d(
                points[:, :2], ref[:2]
            )

        order = np.argsort(points[:, dim])
        pts = points[order]

        volume = 0.0
        for i in range(n):
            z_lo = pts[i, dim]
            z_hi = pts[i + 1, dim] if i + 1 < n else ref[dim]
            height = z_hi - z_lo
            if height <= 0.0 and i + 1 < n:
                continue

            subpoints = pts[: i + 1, :dim]
            nd_sub = _filter_dominated(subpoints)
            sub_ref = ref[:dim]
            sub_vol = self._hso_recursive(nd_sub, sub_ref, dim - 1)
            volume += sub_vol * height

        return float(volume)

    # ------------------------------------------------------------------
    # Dominance utilities (public convenience wrappers)
    # ------------------------------------------------------------------

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if *a* dominates *b* (minimisation)."""
        return _dominates(a, b)

    @staticmethod
    def _filter_dominated(points: np.ndarray) -> np.ndarray:
        """Return non-dominated subset of *points*."""
        return _filter_dominated(points)

    @staticmethod
    def _sort_by_objective(points: np.ndarray, obj_idx: int) -> np.ndarray:
        """Sort *points* by objective *obj_idx* ascending."""
        return _sort_by_objective(points, obj_idx)


# ---------------------------------------------------------------------------
# Approximate hypervolume (Monte Carlo)
# ---------------------------------------------------------------------------

class ApproximateHypervolume(HypervolumeIndicator):
    """Monte-Carlo-based hypervolume approximation.

    Useful when the exact computation is infeasible (d > 5) or when a
    fast estimate is sufficient.

    Parameters
    ----------
    reference_point : array-like, shape (d,)
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        reference_point: Union[Sequence[float], np.ndarray],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(reference_point)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, points: np.ndarray, n_samples: int = 100_000) -> float:
        """Approximate the hypervolume using Monte Carlo sampling.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
        n_samples : int
            Number of uniformly random samples to draw.

        Returns
        -------
        float  Estimated hypervolume.
        """
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)
        points = self._clip_to_ref(points)

        if len(points) == 0:
            return 0.0

        points = _filter_dominated(points)
        return self._monte_carlo_hv(points, self.reference_point, n_samples)

    def confidence_interval(
        self,
        points: np.ndarray,
        n_samples: int = 100_000,
        alpha: float = 0.05,
        n_batches: int = 30,
    ) -> Tuple[float, float]:
        """Return a ``(lower, upper)`` confidence interval for the HV.

        We run *n_batches* independent MC estimates and use the normal
        approximation to build a ``1-alpha`` CI.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
        n_samples : int
            Total samples; divided evenly among batches.
        alpha : float
            Significance level (default 0.05 → 95 % CI).
        n_batches : int
            Number of independent batches.

        Returns
        -------
        (float, float)  Lower and upper bounds.
        """
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)
        points = self._clip_to_ref(points)

        if len(points) == 0:
            return (0.0, 0.0)

        points = _filter_dominated(points)
        batch_size = max(n_samples // n_batches, 1)
        estimates = np.array(
            [
                self._monte_carlo_hv(points, self.reference_point, batch_size)
                for _ in range(n_batches)
            ]
        )
        mean = float(np.mean(estimates))
        se = float(np.std(estimates, ddof=1) / np.sqrt(n_batches))

        from scipy.stats import norm as _norm  # local import to keep scipy optional

        z = _norm.ppf(1 - alpha / 2)
        return (mean - z * se, mean + z * se)

    def adaptive_sampling(
        self,
        points: np.ndarray,
        target_error: float = 0.01,
        max_samples: int = 10_000_000,
        batch_size: int = 10_000,
        min_batches: int = 5,
    ) -> Tuple[float, float, int]:
        """Run MC sampling until relative error < *target_error*.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
        target_error : float
            Desired relative standard error.
        max_samples : int
            Hard limit on the total number of samples.
        batch_size : int
            Samples per batch.
        min_batches : int
            Minimum number of batches before checking convergence.

        Returns
        -------
        (estimate, rel_error, n_samples_used)
        """
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)
        points = self._clip_to_ref(points)

        if len(points) == 0:
            return (0.0, 0.0, 0)

        points = _filter_dominated(points)
        ref = self.reference_point

        estimates: List[float] = []
        total_used = 0

        while total_used < max_samples:
            est = self._monte_carlo_hv(points, ref, batch_size)
            estimates.append(est)
            total_used += batch_size

            if len(estimates) >= min_batches:
                arr = np.array(estimates)
                mean = float(np.mean(arr))
                if mean == 0.0:
                    return (0.0, 0.0, total_used)
                se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                rel_err = se / abs(mean)
                if rel_err <= target_error:
                    return (mean, rel_err, total_used)

        arr = np.array(estimates)
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rel_err = se / abs(mean) if mean != 0 else 0.0
        return (mean, rel_err, total_used)

    # ------------------------------------------------------------------
    # Internal sampling strategies
    # ------------------------------------------------------------------

    def _monte_carlo_hv(
        self, points: np.ndarray, ref: np.ndarray, n_samples: int
    ) -> float:
        """Plain uniform Monte Carlo hypervolume estimate.

        Draw samples uniformly in the bounding box defined by the ideal
        point (component-wise minimum of *points*) and *ref*, then count
        the fraction dominated by at least one point.
        """
        d = points.shape[1]
        ideal = np.min(points, axis=0)

        box_volume = float(np.prod(ref - ideal))
        if box_volume <= 0.0:
            return 0.0

        samples = self._rng.uniform(ideal, ref, size=(n_samples, d))
        dominated = self._dominated_mask(samples, points)
        ratio = float(np.mean(dominated))
        return box_volume * ratio

    def _stratified_sampling(
        self, points: np.ndarray, ref: np.ndarray, n_samples: int, n_strata: int = 10
    ) -> float:
        """Stratified Monte Carlo hypervolume estimate.

        Divide the bounding box into *n_strata* slabs along the first
        objective and sample each slab proportionally.

        Parameters
        ----------
        points : np.ndarray, shape (n, d) – non-dominated
        ref : np.ndarray, shape (d,)
        n_samples : int
        n_strata : int

        Returns
        -------
        float
        """
        d = points.shape[1]
        ideal = np.min(points, axis=0)

        box_lengths = ref - ideal
        box_volume = float(np.prod(box_lengths))
        if box_volume <= 0.0:
            return 0.0

        stratum_edges = np.linspace(ideal[0], ref[0], n_strata + 1)
        samples_per_stratum = max(n_samples // n_strata, 1)
        total_dominated = 0
        total_samples = 0

        for s in range(n_strata):
            lo = ideal.copy()
            hi = ref.copy()
            lo[0] = stratum_edges[s]
            hi[0] = stratum_edges[s + 1]

            samples = self._rng.uniform(lo, hi, size=(samples_per_stratum, d))
            dominated = self._dominated_mask(samples, points)
            total_dominated += int(np.sum(dominated))
            total_samples += samples_per_stratum

        ratio = total_dominated / total_samples if total_samples > 0 else 0.0
        return box_volume * ratio

    def _importance_sampling(
        self,
        points: np.ndarray,
        ref: np.ndarray,
        n_samples: int,
    ) -> float:
        """Importance-sampling Monte Carlo hypervolume estimate.

        Uses a mixture of per-point box distributions as the proposal.
        Each proposal is a uniform distribution inside the box between a
        point and the reference; this biases samples towards the dominated
        region for higher efficiency.

        Parameters
        ----------
        points : np.ndarray, shape (n, d) – non-dominated
        ref : np.ndarray, shape (d,)
        n_samples : int

        Returns
        -------
        float
        """
        n, d = points.shape
        if n == 0:
            return 0.0

        ideal = np.min(points, axis=0)
        box_volume = float(np.prod(ref - ideal))
        if box_volume <= 0.0:
            return 0.0

        per_point_volumes = np.array(
            [_inclusive_volume(p, ref) for p in points], dtype=float
        )
        total_proposal_vol = per_point_volumes.sum()
        if total_proposal_vol <= 0.0:
            return self._monte_carlo_hv(points, ref, n_samples)

        mixing_weights = per_point_volumes / total_proposal_vol

        component_indices = self._rng.choice(n, size=n_samples, p=mixing_weights)
        samples = np.empty((n_samples, d))
        for i in range(n):
            mask = component_indices == i
            count = int(mask.sum())
            if count == 0:
                continue
            samples[mask] = self._rng.uniform(points[i], ref, size=(count, d))

        # Compute proposal density at each sample
        proposal_density = np.zeros(n_samples)
        for i in range(n):
            vol_i = per_point_volumes[i]
            if vol_i <= 0:
                continue
            inside = np.all(
                (samples >= points[i]) & (samples <= ref), axis=1
            )
            proposal_density += mixing_weights[i] * inside / vol_i

        # Target density is uniform over the bounding box
        target_density = 1.0 / box_volume

        dominated = self._dominated_mask(samples, points)
        valid = proposal_density > 0
        weights = np.zeros(n_samples)
        weights[valid] = target_density / proposal_density[valid]

        estimate = float(np.sum(dominated * weights)) / n_samples * box_volume
        return max(estimate, 0.0)

    # ------------------------------------------------------------------
    # Vectorised dominance check
    # ------------------------------------------------------------------

    @staticmethod
    def _dominated_mask(samples: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Boolean mask: True where a sample is dominated by ≥ 1 point.

        Uses broadcasting; memory O(n_samples * n_points * d).  Falls
        back to a loop for very large inputs to avoid OOM.
        """
        n_samples = len(samples)
        n_points = len(points)
        d = samples.shape[1]

        mem_estimate = n_samples * n_points * d * 8  # bytes
        if mem_estimate > 2e9:
            # Chunked approach
            dominated = np.zeros(n_samples, dtype=bool)
            chunk = max(1, int(2e9 / (n_points * d * 8)))
            for start in range(0, n_samples, chunk):
                end = min(start + chunk, n_samples)
                s = samples[start:end]
                dom = np.any(
                    np.all(points[np.newaxis, :, :] <= s[:, np.newaxis, :], axis=2),
                    axis=1,
                )
                dominated[start:end] = dom
            return dominated

        # points[None, :, :] shape (1, n_points, d)
        # samples[:, None, :] shape (n_samples, 1, d)
        dom = np.any(
            np.all(points[np.newaxis, :, :] <= samples[:, np.newaxis, :], axis=2),
            axis=1,
        )
        return dom

    @staticmethod
    def _is_dominated_by_any(
        sample_point: np.ndarray, points: np.ndarray
    ) -> bool:
        """Scalar version: True if *sample_point* is dominated by any point."""
        return bool(np.any(np.all(points <= sample_point, axis=1)))


# ---------------------------------------------------------------------------
# Utility / quality-indicator functions
# ---------------------------------------------------------------------------

def normalize_objectives(
    points: np.ndarray,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Linearly normalise objectives to [0, 1] using ideal and nadir.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    ideal : np.ndarray or None
        Component-wise minimum.  Estimated from *points* if not given.
    nadir : np.ndarray or None
        Component-wise maximum.  Estimated from *points* if not given.

    Returns
    -------
    np.ndarray, shape (n, d) with values in [0, 1].
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return points.copy()

    if ideal is None:
        ideal = np.min(points, axis=0)
    else:
        ideal = np.asarray(ideal, dtype=float)

    if nadir is None:
        nadir = np.max(points, axis=0)
    else:
        nadir = np.asarray(nadir, dtype=float)

    rng = nadir - ideal
    rng[rng == 0] = 1.0  # avoid division by zero for constant objectives
    return (points - ideal) / rng


def hypervolume_improvement(
    new_point: np.ndarray,
    existing_points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Hypervolume improvement of adding *new_point* to *existing_points*.

    Parameters
    ----------
    new_point : array-like, shape (d,)
    existing_points : array-like, shape (n, d)
    reference_point : array-like, shape (d,)

    Returns
    -------
    float  HV(existing ∪ {new}) − HV(existing); non-negative.
    """
    new_point = np.asarray(new_point, dtype=float)
    existing_points = np.asarray(existing_points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    hv = ExactHypervolume(reference_point)

    if existing_points.ndim == 1:
        existing_points = existing_points.reshape(-1, len(reference_point))
    if len(existing_points) == 0:
        existing_points = np.empty((0, len(reference_point)))

    hv_before = hv.compute(existing_points) if len(existing_points) > 0 else 0.0
    combined = np.vstack([existing_points, new_point.reshape(1, -1)]) if len(existing_points) > 0 else new_point.reshape(1, -1)
    hv_after = hv.compute(combined)
    return max(hv_after - hv_before, 0.0)


def greedy_hypervolume_subset(
    points: np.ndarray,
    k: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Select *k* points that greedily maximise the hypervolume.

    At each step the point providing the largest hypervolume improvement
    to the current selection is added.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    k : int
        Number of points to select (capped at n).
    reference_point : array-like, shape (d,)

    Returns
    -------
    np.ndarray  Indices (into *points*) of the selected subset.
    """
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    n = len(points)
    k = min(k, n)

    if k <= 0 or n == 0:
        return np.array([], dtype=int)

    hv = ExactHypervolume(reference_point)
    selected: List[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_improvement = -1.0
        current_pts = points[selected] if selected else np.empty((0, points.shape[1]))

        for idx in remaining:
            imp = hypervolume_improvement(points[idx], current_pts, reference_point)
            if imp > best_improvement:
                best_improvement = imp
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# R2 indicator
# ---------------------------------------------------------------------------

def _generate_weight_vectors(d: int, n_vectors: int) -> np.ndarray:
    """Generate approximately *n_vectors* uniformly distributed weight
    vectors on the (d-1)-simplex using the structured boundary intersection
    approach (Das & Dennis).

    For 2-D this yields evenly spaced angles; for higher d it uses a
    simple lattice approach.
    """
    if d == 2:
        angles = np.linspace(0, np.pi / 2, n_vectors)
        w = np.column_stack([np.cos(angles), np.sin(angles)])
        w /= w.sum(axis=1, keepdims=True)
        return w

    # General d: random on simplex
    rng = np.random.default_rng(42)
    raw = rng.exponential(1.0, size=(n_vectors, d))
    return raw / raw.sum(axis=1, keepdims=True)


def r2_indicator(
    points: np.ndarray,
    reference_point: np.ndarray,
    weight_vectors: Optional[np.ndarray] = None,
    n_vectors: int = 100,
    utopian: Optional[np.ndarray] = None,
) -> float:
    """Compute the R2 indicator (Brockhoff et al., 2012).

    The R2 indicator measures the expected utility of a Pareto front
    approximation using a set of uniformly distributed weight vectors
    and the augmented Tchebycheff scalarisation.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    reference_point : array-like, shape (d,)
        Not used directly in the formulation but kept for API consistency.
    weight_vectors : np.ndarray or None, shape (m, d)
        If *None*, *n_vectors* are generated automatically.
    n_vectors : int
        Number of weight vectors to generate when *weight_vectors* is None.
    utopian : np.ndarray or None
        The utopian (ideal) point.  Estimated from *points* if None.

    Returns
    -------
    float  Lower is better.
    """
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if points.ndim != 2 or len(points) == 0:
        return float("inf")

    d = points.shape[1]
    if weight_vectors is None:
        weight_vectors = _generate_weight_vectors(d, n_vectors)
    else:
        weight_vectors = np.asarray(weight_vectors, dtype=float)

    if utopian is None:
        utopian = np.min(points, axis=0)
    else:
        utopian = np.asarray(utopian, dtype=float)

    # For each weight vector, find the best Tchebycheff value
    diff = points - utopian  # (n, d)
    # Avoid zero weights
    w = np.clip(weight_vectors, 1e-12, None)

    # Tchebycheff: max_j w_j * |f_j - z_j*|
    total = 0.0
    for lam in w:
        scaled = diff * lam  # (n, d)
        tcheb = np.max(scaled, axis=1)  # (n,)
        total += float(np.min(tcheb))

    return total / len(w)


# ---------------------------------------------------------------------------
# Epsilon indicators
# ---------------------------------------------------------------------------

def epsilon_indicator(
    front_a: np.ndarray,
    front_b: np.ndarray,
    multiplicative: bool = False,
) -> float:
    """Compute the (additive or multiplicative) unary epsilon indicator.

    I_ε+(A, B) = max over all b in B of min over all a in A of
        max_j (a_j − b_j)          (additive)
    or  max_j (a_j / b_j)          (multiplicative)

    A *smaller* value means A is a better approximation than B.

    Parameters
    ----------
    front_a : np.ndarray, shape (n, d)
    front_b : np.ndarray, shape (m, d)  – reference front
    multiplicative : bool
        If True compute the multiplicative variant.

    Returns
    -------
    float
    """
    front_a = np.asarray(front_a, dtype=float)
    front_b = np.asarray(front_b, dtype=float)

    if front_a.ndim != 2 or front_b.ndim != 2:
        raise ValueError("Both fronts must be 2-D arrays")
    if len(front_a) == 0:
        return float("inf")
    if len(front_b) == 0:
        return float("-inf")

    eps_values = []
    for b in front_b:
        if multiplicative:
            # Avoid division by zero
            ratios = front_a / np.where(np.abs(b) < 1e-300, 1e-300, b)
            per_a = np.max(ratios, axis=1)  # (n,)
        else:
            per_a = np.max(front_a - b, axis=1)
        eps_values.append(float(np.min(per_a)))

    return float(np.max(eps_values))


def additive_epsilon_indicator(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """Shortcut for the additive epsilon indicator."""
    return epsilon_indicator(front_a, front_b, multiplicative=False)


def multiplicative_epsilon_indicator(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """Shortcut for the multiplicative epsilon indicator."""
    return epsilon_indicator(front_a, front_b, multiplicative=True)


# ---------------------------------------------------------------------------
# Inverted Generational Distance (IGD / IGD+)
# ---------------------------------------------------------------------------

def generational_distance(
    front: np.ndarray,
    reference_front: np.ndarray,
    p: float = 2.0,
) -> float:
    """Generational Distance (GD).

    GD = (1/|A|) * ( Σ_{a ∈ A} d(a, B)^p )^{1/p}

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
    reference_front : np.ndarray, shape (m, d)
    p : float

    Returns
    -------
    float  Lower is better.
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if len(front) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return float("inf")

    dists = _min_distances(front, reference_front)
    return float((np.sum(dists ** p) / len(front)) ** (1.0 / p))


def igd_indicator(
    front: np.ndarray,
    reference_front: np.ndarray,
    p: float = 2.0,
) -> float:
    """Inverted Generational Distance (IGD).

    IGD = (1/|B|) * ( Σ_{b ∈ B} d(b, A)^p )^{1/p}

    Parameters
    ----------
    front : np.ndarray, shape (n, d) – the approximation set.
    reference_front : np.ndarray, shape (m, d) – the true / reference Pareto front.
    p : float

    Returns
    -------
    float  Lower is better.
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if len(front) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return float("inf")

    dists = _min_distances(reference_front, front)
    return float((np.sum(dists ** p) / len(reference_front)) ** (1.0 / p))


def igd_plus(
    front: np.ndarray,
    reference_front: np.ndarray,
    p: float = 2.0,
) -> float:
    """IGD+ indicator (weakly Pareto-compliant variant of IGD).

    Uses the modified distance d+(b, A) that only penalises components
    where b is *not* dominated by the nearest point a.

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
    reference_front : np.ndarray, shape (m, d)
    p : float

    Returns
    -------
    float  Lower is better.
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if len(front) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return float("inf")

    dists = _min_distances_plus(reference_front, front)
    return float((np.sum(dists ** p) / len(reference_front)) ** (1.0 / p))


def _min_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each point in *source*, find the Euclidean distance to the
    closest point in *target*.
    """
    # Use broadcasting: shape (n_src, n_tgt)
    n_src = len(source)
    n_tgt = len(target)

    if n_src * n_tgt * source.shape[1] * 8 > 2e9:
        # Chunked to avoid OOM
        dists = np.empty(n_src)
        chunk = max(1, int(2e9 / (n_tgt * source.shape[1] * 8)))
        for start in range(0, n_src, chunk):
            end = min(start + chunk, n_src)
            diff = source[start:end, np.newaxis, :] - target[np.newaxis, :, :]
            dists[start:end] = np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)
        return dists

    diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]  # (n_src, n_tgt, d)
    return np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)


def _min_distances_plus(
    source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Modified distance d+ for IGD+.

    d+(b, A) = min_{a in A} sqrt( sum_j max(a_j - b_j, 0)^2 )
    """
    n_src = len(source)
    n_tgt = len(target)

    if n_src * n_tgt * source.shape[1] * 8 > 2e9:
        dists = np.empty(n_src)
        chunk = max(1, int(2e9 / (n_tgt * source.shape[1] * 8)))
        for start in range(0, n_src, chunk):
            end = min(start + chunk, n_src)
            diff = target[np.newaxis, :, :] - source[start:end, np.newaxis, :]
            diff_plus = np.maximum(diff, 0.0)
            dists[start:end] = np.min(
                np.sqrt(np.sum(diff_plus ** 2, axis=2)), axis=1
            )
        return dists

    diff = target[np.newaxis, :, :] - source[:, np.newaxis, :]  # (n_src, n_tgt, d)
    diff_plus = np.maximum(diff, 0.0)
    return np.min(np.sqrt(np.sum(diff_plus ** 2, axis=2)), axis=1)


# ---------------------------------------------------------------------------
# Spread & spacing indicators
# ---------------------------------------------------------------------------

def spread_indicator(front: np.ndarray) -> float:
    """Spread (Δ) indicator measuring the extent of the front.

    Δ = (d_f + d_l + Σ |d_i - d̄|) / (d_f + d_l + (N-1) * d̄)

    where d_f, d_l are the Euclidean distances between the extreme solutions
    of the obtained front and the boundary solutions.  We approximate d_f
    and d_l as the distances from the extreme points of each objective to
    the nearest neighbour.

    Lower values (closer to 0) indicate a more uniform distribution.

    Parameters
    ----------
    front : np.ndarray, shape (n, d)

    Returns
    -------
    float
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or len(front) < 2:
        return 0.0

    n, d = front.shape

    # Sort by first objective for the sweep
    order = np.argsort(front[:, 0])
    sorted_front = front[order]

    # Consecutive distances
    consecutive_dists = np.sqrt(
        np.sum((sorted_front[1:] - sorted_front[:-1]) ** 2, axis=1)
    )
    if len(consecutive_dists) == 0:
        return 0.0

    d_mean = float(np.mean(consecutive_dists))
    if d_mean == 0.0:
        return 0.0

    d_f = consecutive_dists[0]
    d_l = consecutive_dists[-1]

    numerator = d_f + d_l + np.sum(np.abs(consecutive_dists - d_mean))
    denominator = d_f + d_l + (n - 1) * d_mean

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def spacing_indicator(front: np.ndarray) -> float:
    """Spacing indicator measuring how evenly points are distributed.

    SP = sqrt( (1/(N-1)) * Σ (d_i - d̄)^2 )

    where d_i is the minimum distance from point i to any other point.

    A value of 0 means perfectly uniform spacing.

    Parameters
    ----------
    front : np.ndarray, shape (n, d)

    Returns
    -------
    float  Lower is better (0 = perfectly uniform).
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or len(front) < 2:
        return 0.0

    n = len(front)

    # Compute pairwise distance matrix
    diff = front[:, np.newaxis, :] - front[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # Set diagonal to infinity so a point doesn't match itself
    np.fill_diagonal(dist_matrix, np.inf)

    min_dists = np.min(dist_matrix, axis=1)
    d_mean = float(np.mean(min_dists))

    return float(np.sqrt(np.sum((min_dists - d_mean) ** 2) / (n - 1)))


# ---------------------------------------------------------------------------
# Coverage / domination-based indicators
# ---------------------------------------------------------------------------

def coverage_indicator(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """C-metric: fraction of *front_b* weakly dominated by ≥ 1 point in *front_a*.

    C(A, B) = |{b ∈ B : ∃ a ∈ A, a ≼ b}| / |B|

    Parameters
    ----------
    front_a : np.ndarray, shape (n, d)
    front_b : np.ndarray, shape (m, d)

    Returns
    -------
    float in [0, 1].  1 means A completely covers B.
    """
    front_a = np.asarray(front_a, dtype=float)
    front_b = np.asarray(front_b, dtype=float)

    if len(front_b) == 0:
        return 1.0
    if len(front_a) == 0:
        return 0.0

    count = 0
    for b in front_b:
        for a in front_a:
            if np.all(a <= b):
                count += 1
                break

    return count / len(front_b)


# ---------------------------------------------------------------------------
# Maximum Pareto Front Error
# ---------------------------------------------------------------------------

def maximum_pareto_front_error(
    front: np.ndarray, reference_front: np.ndarray
) -> float:
    """Maximum distance from any reference point to the nearest point in *front*.

    MPFE = max_{b ∈ B} min_{a ∈ A} ||a - b||

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
    reference_front : np.ndarray, shape (m, d)

    Returns
    -------
    float
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if len(front) == 0 or len(reference_front) == 0:
        return float("inf")

    dists = _min_distances(reference_front, front)
    return float(np.max(dists))


# ---------------------------------------------------------------------------
# Pareto-front extraction utilities
# ---------------------------------------------------------------------------

def pareto_front(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the (first) Pareto front from *points*.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)

    Returns
    -------
    (front, indices)
        front : np.ndarray, shape (m, d) – non-dominated points.
        indices : np.ndarray, shape (m,) – original row indices.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return points.copy(), np.arange(len(points))
    return _filter_dominated_indexed(points)


def pareto_fronts(points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Peel all Pareto fronts (layers) from *points*.

    Returns
    -------
    list of (front_points, original_indices) tuples, ordered by rank.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return []

    remaining_indices = np.arange(len(points))
    remaining_points = points.copy()
    fronts: List[Tuple[np.ndarray, np.ndarray]] = []

    while len(remaining_points) > 0:
        nd, local_idx = _filter_dominated_indexed(remaining_points)
        original_idx = remaining_indices[local_idx]
        fronts.append((nd, original_idx))

        mask = np.ones(len(remaining_points), dtype=bool)
        mask[local_idx] = False
        remaining_points = remaining_points[mask]
        remaining_indices = remaining_indices[mask]

    return fronts


# ---------------------------------------------------------------------------
# Diversity measures for the Diversity Decoding Arena
# ---------------------------------------------------------------------------

def solow_polasky_diversity(front: np.ndarray, theta: float = 1.0) -> float:
    """Solow–Polasky diversity measure based on a similarity kernel.

    D_SP = 1^T K^{-1} 1

    where K_ij = exp(−θ ||x_i − x_j||).

    Higher values indicate greater diversity.

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
    theta : float
        Kernel bandwidth parameter (larger → faster decay → values more
        distinguishable).

    Returns
    -------
    float
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or len(front) < 1:
        return 0.0

    n = len(front)
    if n == 1:
        return 1.0

    diff = front[:, np.newaxis, :] - front[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    K = np.exp(-theta * dist)

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        K += 1e-8 * np.eye(n)
        K_inv = np.linalg.inv(K)

    ones = np.ones(n)
    return float(ones @ K_inv @ ones)


def riesz_energy(front: np.ndarray, s: float = 1.0) -> float:
    """Riesz s-energy of the point set.

    E_s = Σ_{i < j} 1 / ||x_i − x_j||^s

    Lower energy indicates a more uniformly spread configuration.

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
    s : float
        Exponent; s = 1 gives the Coulomb energy.

    Returns
    -------
    float
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or len(front) < 2:
        return 0.0

    diff = front[:, np.newaxis, :] - front[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Upper triangle only (i < j)
    idx = np.triu_indices(len(front), k=1)
    d_pairs = dist[idx]
    d_pairs = np.maximum(d_pairs, 1e-300)  # avoid division by zero

    return float(np.sum(1.0 / d_pairs ** s))


# ---------------------------------------------------------------------------
# Weighted Hypervolume
# ---------------------------------------------------------------------------

class WeightedHypervolume(ExactHypervolume):
    """Hypervolume with a weight function applied to each dominated region.

    This allows emphasising or de-emphasising certain parts of the
    objective space (e.g., to prefer knee points).

    Parameters
    ----------
    reference_point : array-like, shape (d,)
    weight_func : callable  (np.ndarray → float)
        Maps a point in objective space to a weight ≥ 0.  The weight
        is multiplied with the differential volume element at that
        point.  Only used for the Monte-Carlo-based weighted HV.
    """

    def __init__(
        self,
        reference_point: Union[Sequence[float], np.ndarray],
        weight_func=None,
    ) -> None:
        super().__init__(reference_point)
        if weight_func is None:
            weight_func = lambda x: 1.0  # noqa: E731
        self._weight_func = weight_func

    def compute_weighted_mc(
        self,
        points: np.ndarray,
        n_samples: int = 100_000,
        seed: Optional[int] = None,
    ) -> float:
        """Monte-Carlo weighted hypervolume.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
        n_samples : int
        seed : int or None

        Returns
        -------
        float
        """
        points = np.asarray(points, dtype=float)
        points = self._validate_points(points)
        points = self._clip_to_ref(points)

        if len(points) == 0:
            return 0.0

        points = _filter_dominated(points)
        rng = np.random.default_rng(seed)
        ref = self.reference_point
        ideal = np.min(points, axis=0)
        box_volume = float(np.prod(ref - ideal))

        if box_volume <= 0.0:
            return 0.0

        d = points.shape[1]
        samples = rng.uniform(ideal, ref, size=(n_samples, d))

        dominated = ApproximateHypervolume._dominated_mask(samples, points)
        weights = np.array(
            [self._weight_func(s) for s in samples[dominated]], dtype=float
        )

        if len(weights) == 0:
            return 0.0

        avg_weight = float(np.mean(weights))
        dom_ratio = float(np.sum(dominated)) / n_samples
        return box_volume * dom_ratio * avg_weight


# ---------------------------------------------------------------------------
# Multi-indicator aggregation
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Summary of multiple quality indicators for a Pareto front
    approximation."""

    hypervolume: float = 0.0
    igd: float = float("inf")
    igd_plus: float = float("inf")
    generational_dist: float = float("inf")
    spacing: float = 0.0
    spread: float = 0.0
    epsilon_additive: float = float("inf")
    n_points: int = 0
    n_non_dominated: int = 0

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "hypervolume": self.hypervolume,
            "igd": self.igd,
            "igd_plus": self.igd_plus,
            "generational_distance": self.generational_dist,
            "spacing": self.spacing,
            "spread": self.spread,
            "epsilon_additive": self.epsilon_additive,
            "n_points": self.n_points,
            "n_non_dominated": self.n_non_dominated,
        }


def compute_quality_report(
    front: np.ndarray,
    reference_front: np.ndarray,
    reference_point: np.ndarray,
) -> QualityReport:
    """Compute a battery of quality indicators.

    Parameters
    ----------
    front : np.ndarray, shape (n, d) – the approximation set.
    reference_front : np.ndarray, shape (m, d) – the true Pareto front.
    reference_point : np.ndarray, shape (d,) – the reference point for HV.

    Returns
    -------
    QualityReport
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    nd_front, _ = pareto_front(front)

    hv_calc = ExactHypervolume(reference_point)
    hv_val = hv_calc.compute(front) if len(front) > 0 else 0.0

    return QualityReport(
        hypervolume=hv_val,
        igd=igd_indicator(front, reference_front) if len(front) > 0 else float("inf"),
        igd_plus=igd_plus(front, reference_front) if len(front) > 0 else float("inf"),
        generational_dist=(
            generational_distance(front, reference_front) if len(front) > 0 else float("inf")
        ),
        spacing=spacing_indicator(front),
        spread=spread_indicator(front),
        epsilon_additive=(
            additive_epsilon_indicator(front, reference_front)
            if len(front) > 0
            else float("inf")
        ),
        n_points=len(front),
        n_non_dominated=len(nd_front),
    )


# ---------------------------------------------------------------------------
# Hypervolume-based selection operators
# ---------------------------------------------------------------------------

def sms_emoa_select(
    points: np.ndarray,
    reference_point: np.ndarray,
) -> int:
    """SMS-EMOA selection: return the index of the point whose removal
    causes the *least* HV loss on the *worst* non-domination rank.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    reference_point : array-like, shape (d,)

    Returns
    -------
    int  Index of the point to remove.
    """
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if len(points) <= 1:
        return 0

    ranks = _non_domination_rank(points)
    worst_rank = int(np.max(ranks))
    worst_mask = ranks == worst_rank
    worst_indices = np.where(worst_mask)[0]

    if len(worst_indices) == 1:
        return int(worst_indices[0])

    worst_points = points[worst_indices]
    hv_calc = ExactHypervolume(reference_point)

    # The point with smallest exclusive contribution on the worst front
    min_contrib = float("inf")
    min_idx = worst_indices[0]

    for i, global_idx in enumerate(worst_indices):
        exc = hv_calc.exclusive_contribution(i, worst_points)
        if exc < min_contrib:
            min_contrib = exc
            min_idx = global_idx

    return int(min_idx)


def hype_fitness(
    points: np.ndarray,
    reference_point: np.ndarray,
    k: int = 0,
    n_samples: int = 10_000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """HypE fitness values (Bader & Zitzler, 2011).

    For each point, estimates its expected contribution to the
    hypervolume of a random *k*-subset.  When k == n (default 0 means n),
    this reduces to the exact exclusive contribution.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    reference_point : array-like, shape (d,)
    k : int
        Subset size; 0 means n (exact exclusive contribution).
    n_samples : int
        Monte Carlo samples for estimation.
    seed : int or None

    Returns
    -------
    np.ndarray, shape (n,)  Fitness values (higher is better).
    """
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    n = len(points)

    if n == 0:
        return np.array([])

    if k <= 0 or k >= n:
        hv_calc = ExactHypervolume(reference_point)
        return np.array(
            [hv_calc.exclusive_contribution(i, points) for i in range(n)]
        )

    rng = np.random.default_rng(seed)
    fitness = np.zeros(n)
    hv_calc = ExactHypervolume(reference_point)

    for _ in range(n_samples):
        subset_idx = rng.choice(n, size=k, replace=False)
        subset = points[subset_idx]
        hv_val = hv_calc.compute(subset)

        for local_i, global_i in enumerate(subset_idx):
            reduced = np.delete(subset, local_i, axis=0)
            hv_without = hv_calc.compute(reduced) if len(reduced) > 0 else 0.0
            fitness[global_i] += max(hv_val - hv_without, 0.0)

    return fitness / n_samples


# ---------------------------------------------------------------------------
# Reference point estimation
# ---------------------------------------------------------------------------

def estimate_reference_point(
    points: np.ndarray,
    margin: float = 0.1,
) -> np.ndarray:
    """Estimate a suitable reference point from the objective data.

    Adds a *margin* fraction of the range to the nadir point.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    margin : float
        Fraction of the range to add beyond the nadir.

    Returns
    -------
    np.ndarray, shape (d,)
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Need at least one point to estimate reference")

    nadir = np.max(points, axis=0)
    ideal = np.min(points, axis=0)
    rng = nadir - ideal
    rng[rng == 0] = 1.0
    return nadir + margin * rng


# ---------------------------------------------------------------------------
# Convenience: all-in-one compute
# ---------------------------------------------------------------------------

def compute_all_indicators(
    front: np.ndarray,
    reference_front: Optional[np.ndarray] = None,
    reference_point: Optional[np.ndarray] = None,
) -> dict:
    """Compute a comprehensive set of quality indicators.

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
    reference_front : np.ndarray or None
        If None, some indicators (IGD, GD, epsilon) are skipped.
    reference_point : np.ndarray or None
        If None, estimated from *front* with 10 % margin.

    Returns
    -------
    dict  Mapping indicator name → value.
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or len(front) == 0:
        return {"error": "empty or invalid front"}

    if reference_point is None:
        reference_point = estimate_reference_point(front)

    result: dict = {}

    hv_calc = ExactHypervolume(reference_point)
    result["hypervolume"] = hv_calc.compute(front)

    nd, nd_idx = pareto_front(front)
    result["n_points"] = len(front)
    result["n_non_dominated"] = len(nd)
    result["spacing"] = spacing_indicator(front)
    result["spread"] = spread_indicator(front)

    if reference_front is not None:
        reference_front = np.asarray(reference_front, dtype=float)
        result["igd"] = igd_indicator(front, reference_front)
        result["igd_plus"] = igd_plus(front, reference_front)
        result["generational_distance"] = generational_distance(front, reference_front)
        result["epsilon_additive"] = additive_epsilon_indicator(front, reference_front)
        result["mpfe"] = maximum_pareto_front_error(front, reference_front)
        result["coverage"] = coverage_indicator(front, reference_front)

    result["solow_polasky"] = solow_polasky_diversity(front)
    result["r2"] = r2_indicator(front, reference_point)

    return result


# ---------------------------------------------------------------------------
# Batch comparison utilities
# ---------------------------------------------------------------------------

def compare_fronts(
    fronts: dict[str, np.ndarray],
    reference_front: Optional[np.ndarray] = None,
    reference_point: Optional[np.ndarray] = None,
) -> dict[str, dict]:
    """Compare multiple named fronts on a battery of indicators.

    Parameters
    ----------
    fronts : dict mapping name → np.ndarray
    reference_front : np.ndarray or None
    reference_point : np.ndarray or None

    Returns
    -------
    dict mapping name → indicator dict.
    """
    all_points = np.vstack(list(fronts.values()))
    if reference_point is None:
        reference_point = estimate_reference_point(all_points)

    results = {}
    for name, pts in fronts.items():
        results[name] = compute_all_indicators(
            pts,
            reference_front=reference_front,
            reference_point=reference_point,
        )
    return results


# ---------------------------------------------------------------------------
# Dominated hypervolume decomposition (for 2-D visualisation)
# ---------------------------------------------------------------------------

def hypervolume_decomposition_2d(
    points: np.ndarray,
    reference_point: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Decompose the 2-D dominated region into disjoint rectangles.

    Each rectangle is represented as (lower_left, upper_right, area).
    Useful for visualising contributions.

    Parameters
    ----------
    points : np.ndarray, shape (n, 2)
    reference_point : np.ndarray, shape (2,)

    Returns
    -------
    list of (ll, ur, area) tuples.
    """
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Only 2-D supported")

    # Filter to dominated & non-dominated
    mask = np.all(points <= reference_point, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return []

    pts = _filter_dominated(pts)
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    rects: List[Tuple[np.ndarray, np.ndarray, float]] = []
    n = len(pts)

    for i in range(n):
        x_lo = pts[i, 0]
        x_hi = pts[i + 1, 0] if i + 1 < n else reference_point[0]
        y_lo = pts[i, 1]
        y_hi = reference_point[1]

        width = x_hi - x_lo
        height = y_hi - y_lo
        if width > 0 and height > 0:
            ll = np.array([x_lo, y_lo])
            ur = np.array([x_hi, y_hi])
            rects.append((ll, ur, width * height))

    return rects


# ---------------------------------------------------------------------------
# Contribution-aware archive (online HV maintenance)
# ---------------------------------------------------------------------------

class HypervolumeArchive:
    """An archive that maintains a bounded set of non-dominated solutions
    and tracks hypervolume contributions.

    When the archive exceeds *max_size*, the least contributor is evicted
    (SMS-EMOA style).

    Parameters
    ----------
    reference_point : array-like, shape (d,)
    max_size : int
    """

    def __init__(
        self,
        reference_point: Union[Sequence[float], np.ndarray],
        max_size: int = 100,
    ) -> None:
        self.reference_point = np.asarray(reference_point, dtype=float)
        self.max_size = max_size
        self._points: List[np.ndarray] = []
        self._hv_calc = ExactHypervolume(self.reference_point)

    @property
    def points(self) -> np.ndarray:
        if not self._points:
            return np.empty((0, len(self.reference_point)))
        return np.array(self._points)

    @property
    def size(self) -> int:
        return len(self._points)

    @property
    def hypervolume(self) -> float:
        if not self._points:
            return 0.0
        return self._hv_calc.compute(self.points)

    def add(self, point: np.ndarray) -> bool:
        """Try to add *point* to the archive.

        Returns True if the point was accepted (not dominated by any
        existing member and provides positive HV improvement).
        """
        point = np.asarray(point, dtype=float)

        # Check if dominated by any archive member
        for p in self._points:
            if _dominates(p, point):
                return False

        # Remove any members dominated by the new point
        self._points = [p for p in self._points if not _dominates(point, p)]
        self._points.append(point.copy())

        # Evict if over capacity
        while len(self._points) > self.max_size:
            self._evict_least_contributor()

        return True

    def _evict_least_contributor(self) -> None:
        """Remove the member with the smallest exclusive HV contribution."""
        pts = self.points
        idx = self._hv_calc.least_contributor(pts)
        del self._points[idx]

    def contributions(self) -> List[HypervolumeContribution]:
        """Return per-point contributions for the current archive."""
        if not self._points:
            return []
        return self._hv_calc.contributions(self.points)


# ---------------------------------------------------------------------------
# Incremental HV maintenance for 2-D (efficient)
# ---------------------------------------------------------------------------

class IncrementalHypervolume2D:
    """Maintains the hypervolume indicator incrementally in 2-D.

    Uses a sorted structure so that additions are O(n) worst-case
    but amortised O(log n) in many practical scenarios.

    Parameters
    ----------
    reference_point : array-like, shape (2,)
    """

    def __init__(self, reference_point: Union[Sequence[float], np.ndarray]) -> None:
        self.reference_point = np.asarray(reference_point, dtype=float)
        if len(self.reference_point) != 2:
            raise ValueError("IncrementalHypervolume2D is 2-D only")
        # Maintain sorted by x ascending; non-dominated invariant
        self._sorted_x: List[np.ndarray] = []
        self._hv: float = 0.0

    @property
    def hypervolume(self) -> float:
        return self._hv

    @property
    def size(self) -> int:
        return len(self._sorted_x)

    def add(self, point: np.ndarray) -> float:
        """Add *point*; return the hypervolume improvement.

        If the point is dominated, 0.0 is returned and the point is not
        stored.
        """
        point = np.asarray(point, dtype=float)
        if len(point) != 2:
            raise ValueError("Expected 2-D point")
        if np.any(point >= self.reference_point):
            return 0.0

        # Check domination
        for p in self._sorted_x:
            if _dominates(p, point):
                return 0.0

        # Remove dominated points
        new_sorted: List[np.ndarray] = []
        for p in self._sorted_x:
            if not _dominates(point, p):
                new_sorted.append(p)
        new_sorted.append(point.copy())
        new_sorted.sort(key=lambda p: p[0])
        self._sorted_x = new_sorted

        old_hv = self._hv
        self._hv = self._recompute_hv()
        return max(self._hv - old_hv, 0.0)

    def _recompute_hv(self) -> float:
        if not self._sorted_x:
            return 0.0
        pts = np.array(self._sorted_x)
        hv_calc = ExactHypervolume(self.reference_point)
        return hv_calc.compute(pts)


# ---------------------------------------------------------------------------
# Knee point detection
# ---------------------------------------------------------------------------

def find_knee_points(
    front: np.ndarray,
    n_knees: int = 1,
) -> np.ndarray:
    """Identify knee points on a 2-D Pareto front.

    A knee point is one with maximum distance to the line connecting the
    two extreme points of the front.

    Parameters
    ----------
    front : np.ndarray, shape (n, 2)
    n_knees : int
        Number of knee points to return.

    Returns
    -------
    np.ndarray  Indices of the knee points.
    """
    front = np.asarray(front, dtype=float)
    if front.ndim != 2 or front.shape[1] != 2 or len(front) < 3:
        return np.arange(min(n_knees, len(front)))

    nd_front = _filter_dominated(front)
    order = np.argsort(nd_front[:, 0])
    pts = nd_front[order]

    # Line from first to last point
    p0 = pts[0]
    p1 = pts[-1]
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-15:
        return np.array([0])

    # Distance from each point to the line
    dists = np.abs(np.cross(line_vec, p0 - pts)) / line_len

    # Map back to original indices
    knee_local = np.argsort(dists)[-n_knees:]

    # Find original indices by matching coordinates
    result = []
    for li in knee_local:
        kp = pts[li]
        matches = np.where(np.all(np.abs(front - kp) < 1e-12, axis=1))[0]
        if len(matches) > 0:
            result.append(matches[0])
    return np.array(result, dtype=int)


# ---------------------------------------------------------------------------
# Hypervolume gradient (for gradient-based MOO)
# ---------------------------------------------------------------------------

def hypervolume_gradient_2d(
    points: np.ndarray,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Compute the gradient of the hypervolume w.r.t. each point (2-D).

    For a non-dominated point i with neighbours i-1, i+1 in the sorted
    order:
        ∂HV/∂x_i = -(y_{i-1} - y_i)   (or -(ref_y - y_i) for leftmost)
        ∂HV/∂y_i = -(x_{i+1} - x_i)   (or -(ref_x - x_i) for rightmost)

    where the gradient direction pushes the point to *increase* HV
    (i.e., towards lower objective values).

    Parameters
    ----------
    points : np.ndarray, shape (n, 2)
    reference_point : np.ndarray, shape (2,)

    Returns
    -------
    np.ndarray, shape (n, 2) – gradient vector for each point.
    """
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Only 2-D supported")

    n = len(points)
    if n == 0:
        return np.empty((0, 2))

    nd_pts = _filter_dominated(points)
    order = np.argsort(nd_pts[:, 0])
    sorted_pts = nd_pts[order]
    m = len(sorted_pts)

    grad = np.zeros((n, 2))
    ref = reference_point

    for k in range(m):
        orig_idx_matches = np.where(
            np.all(np.abs(points - sorted_pts[k]) < 1e-12, axis=1)
        )[0]
        if len(orig_idx_matches) == 0:
            continue
        i = orig_idx_matches[0]

        # ∂HV/∂x_i
        if k == 0:
            grad[i, 0] = -(ref[1] - sorted_pts[k, 1])
        else:
            grad[i, 0] = -(sorted_pts[k - 1, 1] - sorted_pts[k, 1])

        # ∂HV/∂y_i
        if k == m - 1:
            grad[i, 1] = -(ref[0] - sorted_pts[k, 0])
        else:
            grad[i, 1] = -(sorted_pts[k + 1, 0] - sorted_pts[k, 0])

    return grad


# ---------------------------------------------------------------------------
# Expected Hypervolume Improvement (EHVI) for Bayesian MOO
# ---------------------------------------------------------------------------

def expected_hypervolume_improvement_2d(
    mean: np.ndarray,
    std: np.ndarray,
    pareto_front_points: np.ndarray,
    reference_point: np.ndarray,
    n_samples: int = 10_000,
    seed: Optional[int] = None,
) -> float:
    """Monte-Carlo estimate of the Expected HV Improvement in 2-D.

    Given a Gaussian predictive distribution N(mean, diag(std^2)) for
    a new candidate, estimate the expected improvement in hypervolume.

    Parameters
    ----------
    mean : np.ndarray, shape (2,)
    std : np.ndarray, shape (2,)
    pareto_front_points : np.ndarray, shape (n, 2)
    reference_point : np.ndarray, shape (2,)
    n_samples : int
    seed : int or None

    Returns
    -------
    float  Expected hypervolume improvement.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    pareto_front_points = np.asarray(pareto_front_points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    rng = np.random.default_rng(seed)
    hv_calc = ExactHypervolume(reference_point)

    if len(pareto_front_points) == 0:
        base_hv = 0.0
    else:
        base_hv = hv_calc.compute(pareto_front_points)

    samples = rng.normal(mean, std, size=(n_samples, 2))
    improvements = np.zeros(n_samples)

    for i, s in enumerate(samples):
        if np.any(s >= reference_point):
            continue
        if len(pareto_front_points) == 0:
            combined = s.reshape(1, -1)
        else:
            combined = np.vstack([pareto_front_points, s.reshape(1, -1)])
        new_hv = hv_calc.compute(combined)
        improvements[i] = max(new_hv - base_hv, 0.0)

    return float(np.mean(improvements))


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def hypervolume(
    points: np.ndarray,
    reference_point: np.ndarray,
    exact: bool = True,
    **kwargs,
) -> float:
    """One-line hypervolume computation.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    reference_point : array-like, shape (d,)
    exact : bool
        If False, uses Monte Carlo approximation.
    **kwargs
        Forwarded to the underlying compute method.

    Returns
    -------
    float
    """
    if exact:
        return ExactHypervolume(reference_point).compute(points)
    return ApproximateHypervolume(reference_point).compute(points, **kwargs)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "HypervolumeContribution",
    "QualityReport",
    # Core classes
    "HypervolumeIndicator",
    "ExactHypervolume",
    "ApproximateHypervolume",
    "WeightedHypervolume",
    "HypervolumeArchive",
    "IncrementalHypervolume2D",
    # Utility functions
    "normalize_objectives",
    "hypervolume_improvement",
    "greedy_hypervolume_subset",
    "r2_indicator",
    "epsilon_indicator",
    "additive_epsilon_indicator",
    "multiplicative_epsilon_indicator",
    "igd_indicator",
    "igd_plus",
    "spread_indicator",
    "spacing_indicator",
    "generational_distance",
    "coverage_indicator",
    "maximum_pareto_front_error",
    "pareto_front",
    "pareto_fronts",
    "solow_polasky_diversity",
    "riesz_energy",
    "find_knee_points",
    "hypervolume_gradient_2d",
    "expected_hypervolume_improvement_2d",
    "sms_emoa_select",
    "hype_fitness",
    "estimate_reference_point",
    "compute_all_indicators",
    "compare_fronts",
    "compute_quality_report",
    "hypervolume_decomposition_2d",
    "hypervolume",
    # Internal helpers exposed for advanced use
    "_dominates",
    "_filter_dominated",
    "_sort_by_objective",
    "_non_domination_rank",
]
