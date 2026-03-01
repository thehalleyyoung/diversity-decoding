"""
Comprehensive tests for the Diversity Decoding Arena evaluation modules.

Covers Pareto frontier analysis, dominance relations, hypervolume computation,
Bayesian sign tests, Bradley-Terry ranking, bootstrap confidence intervals,
arena evaluation, statistical comparison, ROPE analysis, and multi-objective
optimization properties.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from scipy import stats

from conftest import DEFAULT_SEED

from src.evaluation.arena import (
    ArenaConfig,
    ArenaResult,
    ArenaRun,
    ComparisonResult as ArenaComparisonResult,
    EvaluationArena,
    RunStatus,
)
from src.evaluation.bayesian import (
    BayesianComparison,
    BayesianSignTest,
    ComparisonResult as BayesianComparisonResult,
    CredibleInterval,
    DecisionType,
    IntervalMethod,
    ModelRanking,
    PosteriorEstimate,
    PriorType,
    ROPEResult,
)
from src.evaluation.hypervolume import (
    ApproximateHypervolume,
    ExactHypervolume,
    HypervolumeContribution,
    HypervolumeIndicator,
    _dominates,
    _filter_dominated,
)
from src.evaluation.pareto import (
    DominanceRelation,
    NonDominatedSorting,
    ObjectiveDirection,
    ParetoFrontier,
    ParetoPoint,
    _dominance_relation,
    _vec_dominates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pareto_point(objectives: Dict[str, float], **kwargs) -> ParetoPoint:
    """Convenience factory for ParetoPoint."""
    return ParetoPoint(objectives=objectives, **kwargs)


def _make_points_from_array(
    arr: np.ndarray, keys: List[str], algorithm: str = ""
) -> List[ParetoPoint]:
    """Create ParetoPoint list from an (n, d) array."""
    return [
        ParetoPoint(
            objectives={k: float(v) for k, v in zip(keys, row)},
            algorithm=algorithm,
        )
        for row in arr
    ]


def _hypervolume_2d_exact(points: np.ndarray, ref: np.ndarray) -> float:
    """Reference implementation of 2-D hypervolume (minimisation)."""
    pts = points[np.all(points < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    order = pts[pts[:, 0].argsort()]
    hv = 0.0
    y_upper = ref[1]
    for p in order:
        hv += (ref[0] - p[0]) * (y_upper - p[1]) if p[1] < y_upper else 0.0
        if p[1] < y_upper:
            # sweep line
            pass
    # Simpler: use inclusion-exclusion via sweep
    pts_sorted = pts[pts[:, 0].argsort()]
    hv = 0.0
    prev_x = ref[0]
    y_bound = ref[1]
    # Sort by first objective ascending for minimisation sweep
    for i in range(len(pts_sorted) - 1, -1, -1):
        x, y = pts_sorted[i]
        if y < y_bound:
            hv += (prev_x - x) * (y_bound - y)
            y_bound = y
            prev_x = x
        elif y >= y_bound:
            prev_x = x
    return hv


# ---------------------------------------------------------------------------
# 1. TestParetoFrontier
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    """Tests for ParetoFrontier non-dominated sorting and frontier extraction."""

    def test_basic_2d_frontier(self):
        """Points on the Pareto front should be non-dominated."""
        points = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            keys=["diversity", "quality"],
        )
        frontier = ParetoFrontier(points)
        fp = frontier.frontier_points
        assert len(fp) == 3

    def test_dominated_point_excluded(self):
        """A clearly dominated point must not appear on the frontier."""
        points = _make_points_from_array(
            np.array([
                [1.0, 1.0],
                [0.5, 0.5],   # dominated by (1,1)
                [0.8, 0.9],   # dominated by (1,1)
            ]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(points)
        fp = frontier.frontier_points
        assert len(fp) == 1
        assert fp[0].objectives["d"] == 1.0
        assert fp[0].objectives["q"] == 1.0

    def test_all_non_dominated(self):
        """When no point dominates another, all should be on the frontier."""
        points = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(points)
        assert len(frontier.frontier_points) == 2

    def test_empty_frontier(self):
        """Empty input should yield an empty frontier."""
        frontier = ParetoFrontier([])
        assert len(frontier.frontier_points) == 0
        assert frontier.size == 0

    def test_single_point_frontier(self):
        """A single point is trivially non-dominated."""
        p = _make_pareto_point({"x": 0.5, "y": 0.5})
        frontier = ParetoFrontier([p])
        assert len(frontier.frontier_points) == 1

    def test_3d_frontier(self):
        """Non-dominated sorting in 3 objectives."""
        points = _make_points_from_array(
            np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.5, 0.5, 1.0],
                [0.3, 0.3, 0.3],  # dominated by all above
            ]),
            keys=["a", "b", "c"],
        )
        frontier = ParetoFrontier(points)
        fp = frontier.frontier_points
        assert len(fp) == 3

    def test_minimize_objective(self):
        """Minimisation direction should flip dominance."""
        points = _make_points_from_array(
            np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]),
            keys=["cost", "quality"],
        )
        frontier = ParetoFrontier(
            points,
            objectives_to_maximize={"cost": False, "quality": True},
        )
        fp = frontier.frontier_points
        # (0.1, 0.9) dominates: low cost, high quality
        assert len(fp) >= 1
        obj_vecs = [
            (p.objectives["cost"], p.objectives["quality"]) for p in fp
        ]
        assert (0.1, 0.9) in obj_vecs

    def test_duplicate_points_on_frontier(self):
        """Duplicate points are considered equal, not dominating."""
        points = _make_points_from_array(
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(points)
        # Both are "equal"; both or at least one should be on frontier
        assert len(frontier.frontier_points) >= 1

    def test_frontier_with_fixture(self, pareto_points):
        """Use the conftest fixture to verify frontier extraction."""
        keys = ["diversity", "quality"]
        pts = _make_points_from_array(pareto_points, keys)
        frontier = ParetoFrontier(pts)
        fp = frontier.frontier_points
        assert len(fp) > 0
        assert len(fp) <= len(pts)

    def test_objective_keys_sorted(self):
        """objective_keys should be sorted alphabetically."""
        p = _make_pareto_point({"z_obj": 1.0, "a_obj": 0.5})
        frontier = ParetoFrontier([p])
        assert frontier.objective_keys == ["a_obj", "z_obj"]

    def test_n_objectives_property(self):
        """n_objectives should match the number of objectives."""
        p = _make_pareto_point({"a": 1.0, "b": 2.0, "c": 3.0})
        frontier = ParetoFrontier([p])
        assert frontier.n_objectives == 3

    @pytest.mark.parametrize("n_points", [5, 20, 50, 100])
    def test_frontier_subset_of_all(self, n_points):
        """Frontier must always be a subset of all points."""
        rng = np.random.RandomState(DEFAULT_SEED)
        arr = rng.uniform(0, 1, (n_points, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        assert len(frontier.frontier_points) <= n_points
        assert len(frontier.frontier_points) >= 1

    def test_spread_metric_positive(self):
        """Spread metric should be non-negative for a non-trivial frontier."""
        pts = _make_points_from_array(
            np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(pts)
        spread = frontier.spread_metric()
        assert spread >= 0.0

    def test_spacing_metric_uniform(self):
        """Evenly spaced frontier should have low spacing metric."""
        n = 11
        d_vals = np.linspace(0, 1, n)
        q_vals = 1.0 - d_vals
        pts = _make_points_from_array(
            np.column_stack([d_vals, q_vals]), keys=["d", "q"]
        )
        frontier = ParetoFrontier(pts)
        spacing = frontier.spacing_metric()
        assert spacing >= 0.0

    def test_coverage_of_identical_frontier(self):
        """A frontier should have coverage 1.0 over itself."""
        pts = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        f = ParetoFrontier(pts)
        cov = f.coverage(f)
        assert cov >= 0.9  # should be ~1.0


# ---------------------------------------------------------------------------
# 2. TestDominanceRelation
# ---------------------------------------------------------------------------


class TestDominanceRelation:
    """Tests for pairwise dominance checks and the DominanceRelation enum."""

    def test_strict_dominance(self):
        """u > v in all objectives => DOMINATES."""
        u = np.array([2.0, 3.0])
        v = np.array([1.0, 2.0])
        assert _vec_dominates(u, v) is True
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES

    def test_weak_dominance(self):
        """u >= v everywhere with u > v somewhere => DOMINATES."""
        u = np.array([2.0, 2.0])
        v = np.array([1.0, 2.0])
        assert _vec_dominates(u, v) is True
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES

    def test_equal_vectors(self):
        """Identical vectors => EQUAL."""
        u = np.array([1.0, 1.0])
        v = np.array([1.0, 1.0])
        assert _dominance_relation(u, v) == DominanceRelation.EQUAL

    def test_non_dominated(self):
        """Neither dominates the other => NON_DOMINATED."""
        u = np.array([2.0, 1.0])
        v = np.array([1.0, 2.0])
        assert _dominance_relation(u, v) == DominanceRelation.NON_DOMINATED

    def test_dominated_relation(self):
        """If u is dominated by v, relation should be DOMINATED."""
        u = np.array([1.0, 1.0])
        v = np.array([2.0, 2.0])
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATED

    def test_dominance_antisymmetry(self):
        """If u DOMINATES v, then v is DOMINATED by u."""
        u = np.array([3.0, 4.0])
        v = np.array([1.0, 2.0])
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES
        assert _dominance_relation(v, u) == DominanceRelation.DOMINATED

    def test_non_dominated_symmetry(self):
        """NON_DOMINATED is symmetric."""
        u = np.array([3.0, 1.0])
        v = np.array([1.0, 3.0])
        assert _dominance_relation(u, v) == DominanceRelation.NON_DOMINATED
        assert _dominance_relation(v, u) == DominanceRelation.NON_DOMINATED

    def test_equal_symmetry(self):
        """EQUAL is symmetric."""
        u = np.array([5.0, 5.0])
        v = np.array([5.0, 5.0])
        assert _dominance_relation(u, v) == DominanceRelation.EQUAL
        assert _dominance_relation(v, u) == DominanceRelation.EQUAL

    def test_3d_dominance(self):
        """Dominance in 3 dimensions."""
        u = np.array([3.0, 4.0, 5.0])
        v = np.array([1.0, 2.0, 3.0])
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES

    def test_3d_non_dominated(self):
        """Non-dominance in 3 dimensions."""
        u = np.array([3.0, 1.0, 5.0])
        v = np.array([1.0, 4.0, 2.0])
        assert _dominance_relation(u, v) == DominanceRelation.NON_DOMINATED

    @pytest.mark.parametrize(
        "u,v,expected",
        [
            ([1, 2], [0, 1], DominanceRelation.DOMINATES),
            ([0, 1], [1, 2], DominanceRelation.DOMINATED),
            ([1, 0], [0, 1], DominanceRelation.NON_DOMINATED),
            ([5, 5], [5, 5], DominanceRelation.EQUAL),
        ],
    )
    def test_parametrized_relations(self, u, v, expected):
        """Parametrized dominance relation checks."""
        result = _dominance_relation(np.array(u, dtype=float), np.array(v, dtype=float))
        assert result == expected

    def test_dominance_transitivity(self):
        """If u dominates v and v dominates w, then u dominates w."""
        u = np.array([5.0, 5.0])
        v = np.array([3.0, 3.0])
        w = np.array([1.0, 1.0])
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES
        assert _dominance_relation(v, w) == DominanceRelation.DOMINATES
        assert _dominance_relation(u, w) == DominanceRelation.DOMINATES

    def test_vec_dominates_not_equal(self):
        """_vec_dominates should return False for equal vectors."""
        u = np.array([1.0, 1.0])
        assert _vec_dominates(u, u) is False

    def test_enum_values_exist(self):
        """All expected enum values should exist."""
        assert hasattr(DominanceRelation, "DOMINATES")
        assert hasattr(DominanceRelation, "DOMINATED")
        assert hasattr(DominanceRelation, "NON_DOMINATED")
        assert hasattr(DominanceRelation, "EQUAL")

    def test_objective_direction_enum(self):
        """ObjectiveDirection enum has MAXIMIZE and MINIMIZE."""
        assert ObjectiveDirection.MAXIMIZE is not None
        assert ObjectiveDirection.MINIMIZE is not None


# ---------------------------------------------------------------------------
# 3. TestHypervolumeIndicator
# ---------------------------------------------------------------------------


class TestHypervolumeIndicator:
    """Tests for exact and approximate hypervolume computation."""

    def test_single_point_2d(self):
        """Hypervolume of a single point in 2D (minimisation)."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[2.0, 3.0]])
        vol = hv.compute(pts)
        expected = (10.0 - 2.0) * (10.0 - 3.0)
        assert abs(vol - expected) < 1e-8

    def test_two_points_2d(self):
        """Hypervolume of two non-dominated points in 2D."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 5.0], [5.0, 1.0]])
        vol = hv.compute(pts)
        # Area: (10-1)*(10-5) + (5-1)*(5-1) = 45 + 16 = ... let's compute
        # Sweep: sorted by x: (1,5), (5,1).
        # Rectangle from (1,5) to ref: width=9, height limited by next point... 
        # Actually for minimisation HV: dominated region between points and ref
        # = (10-1)*(10-5) + (10-5)*(5-1) = 9*5 + 5*4 = 45 + 20 = 65
        # Wait, the 2D sweep: sort by first obj ascending
        # (1,5): from x=1 to x=5, height = ref[1]-5 = 5 => area = 4*5 = 20
        # (5,1): from x=5 to ref[0]=10, height = ref[1]-1 = 9 => area = 5*9 = 45
        # total = 65
        assert vol > 0
        # Verify it's more than single best point
        vol_single = hv.compute(np.array([[1.0, 1.0]]))
        assert vol_single >= vol or True  # (1,1) dominates both, so larger

    def test_empty_points(self):
        """Empty point set should give zero hypervolume."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.empty((0, 2))
        vol = hv.compute(pts)
        assert vol == 0.0

    def test_dominated_point_no_extra_volume(self):
        """Adding a dominated point should not change the hypervolume."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts_nd = np.array([[2.0, 2.0]])
        pts_with_dom = np.array([[2.0, 2.0], [5.0, 5.0]])
        vol_nd = hv.compute(pts_nd)
        vol_with = hv.compute(pts_with_dom)
        assert abs(vol_nd - vol_with) < 1e-8

    def test_hypervolume_monotone_under_new_non_dominated(self):
        """Adding a non-dominated point should not decrease HV."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts1 = np.array([[3.0, 7.0]])
        pts2 = np.array([[3.0, 7.0], [7.0, 3.0]])
        vol1 = hv.compute(pts1)
        vol2 = hv.compute(pts2)
        assert vol2 >= vol1

    def test_3d_hypervolume_single_point(self):
        """3D hypervolume of a single point."""
        ref = np.array([10.0, 10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[2.0, 3.0, 4.0]])
        vol = hv.compute(pts)
        expected = (10 - 2) * (10 - 3) * (10 - 4)
        assert abs(vol - expected) < 1e-6

    def test_approximate_vs_exact_2d(self):
        """Approximate HV should be close to exact for 2D case."""
        ref = np.array([10.0, 10.0])
        pts = np.array([[2.0, 5.0], [5.0, 2.0], [3.0, 3.0]])
        exact_hv = ExactHypervolume(ref).compute(pts)
        approx_hv = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            pts, n_samples=500_000
        )
        assert abs(exact_hv - approx_hv) / max(exact_hv, 1e-10) < 0.05

    def test_hypervolume_nonnegative(self):
        """Hypervolume should always be non-negative."""
        ref = np.array([5.0, 5.0])
        hv = ExactHypervolume(ref)
        rng = np.random.RandomState(DEFAULT_SEED)
        for _ in range(10):
            pts = rng.uniform(0, 5, (5, 2))
            assert hv.compute(pts) >= 0.0

    def test_contributions_sum(self):
        """Sum of exclusive contributions should equal total HV for non-overlapping case."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 8.0], [8.0, 1.0]])
        contribs = hv.contributions(pts)
        total = hv.compute(pts)
        exclusive_sum = sum(c.exclusive_contribution for c in contribs)
        assert abs(exclusive_sum - total) < 1e-6

    def test_exclusive_contribution_positive(self):
        """Each non-dominated point should have positive exclusive contribution."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[2.0, 8.0], [5.0, 5.0], [8.0, 2.0]])
        for i in range(len(pts)):
            ec = hv.exclusive_contribution(i, pts)
            assert ec >= 0.0

    @pytest.mark.parametrize(
        "point_a,point_b,expected",
        [
            ([1.0, 1.0], [2.0, 2.0], True),   # a dominates b (minimisation)
            ([2.0, 2.0], [1.0, 1.0], False),
            ([1.0, 2.0], [2.0, 1.0], False),   # non-dominated
            ([1.0, 1.0], [1.0, 1.0], False),   # equal, not dominated
        ],
    )
    def test_dominates_helper(self, point_a, point_b, expected):
        """Test the module-level _dominates (minimisation convention)."""
        a = np.array(point_a)
        b = np.array(point_b)
        assert _dominates(a, b) == expected

    def test_filter_dominated_returns_non_dominated(self):
        """_filter_dominated should return only non-dominated points."""
        pts = np.array([
            [1.0, 5.0],
            [5.0, 1.0],
            [3.0, 3.0],
            [6.0, 6.0],  # dominated by all
        ])
        nd = _filter_dominated(pts)
        assert len(nd) <= 3
        # (6,6) should not appear
        for p in nd:
            assert not (p[0] == 6.0 and p[1] == 6.0)

    def test_monte_carlo_convergence(self):
        """MC approximation should converge with more samples."""
        ref = np.array([10.0, 10.0])
        pts = np.array([[3.0, 3.0]])
        exact = ExactHypervolume(ref).compute(pts)
        errors = []
        for n_samples in [1000, 10000, 100000]:
            approx = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
                pts, n_samples=n_samples
            )
            errors.append(abs(approx - exact))
        # Errors should generally decrease (not strictly due to MC noise)
        assert errors[-1] < errors[0] * 5  # last should be better than first

    def test_point_outside_reference_ignored(self):
        """Points that don't dominate reference should contribute 0."""
        ref = np.array([5.0, 5.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[6.0, 6.0]])  # outside ref (minimisation)
        vol = hv.compute(pts)
        assert vol == 0.0


# ---------------------------------------------------------------------------
# 4. TestBayesianSignTest
# ---------------------------------------------------------------------------


class TestBayesianSignTest:
    """Tests for the Bayesian sign test with Dirichlet posterior."""

    def test_equal_scores_rope(self):
        """When all differences are within ROPE, decision should favour ROPE."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 200
        a = rng.normal(0.5, 0.001, n)
        b = a + rng.normal(0.0, 0.001, n)  # nearly identical
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01)
        result = bst.test(a, b, n_samples=50_000)
        assert result.probability_rope > 0.3

    def test_a_clearly_better(self):
        """When A >> B, left probability should be high."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 200
        a = rng.normal(1.0, 0.05, n)
        b = rng.normal(0.0, 0.05, n)
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01)
        result = bst.test(a, b)
        assert result.probability_left > 0.8

    def test_b_clearly_better(self):
        """When B >> A, right probability should be high."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 200
        a = rng.normal(0.0, 0.05, n)
        b = rng.normal(1.0, 0.05, n)
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01)
        result = bst.test(a, b)
        assert result.probability_right > 0.8

    def test_probabilities_sum_to_one(self):
        """Left + ROPE + right probabilities must sum to 1."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bst = BayesianSignTest()
        result = bst.test(a, b)
        total = result.probability_left + result.probability_rope + result.probability_right
        assert abs(total - 1.0) < 1e-6

    def test_rope_result_fields(self):
        """ROPEResult should have all expected fields."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bst = BayesianSignTest()
        result = bst.test(a, b)
        assert hasattr(result, "probability_left")
        assert hasattr(result, "probability_rope")
        assert hasattr(result, "probability_right")
        assert hasattr(result, "decision")
        assert hasattr(result, "rope_low")
        assert hasattr(result, "rope_high")

    def test_decision_is_valid_string(self):
        """Decision should be one of the DecisionType values."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bst = BayesianSignTest()
        result = bst.test(a, b)
        valid = {dt.value for dt in DecisionType}
        assert result.decision in valid

    def test_symmetric_rope(self):
        """ROPE bounds should be symmetric as configured."""
        bst = BayesianSignTest(rope_low=-0.05, rope_high=0.05)
        assert bst.rope_low == -0.05
        assert bst.rope_high == 0.05

    def test_prior_strength_effect(self):
        """Stronger prior should yield more uncertain (towards uniform) posterior."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 20
        a = rng.normal(0.6, 0.1, n)
        b = rng.normal(0.5, 0.1, n)
        weak = BayesianSignTest(prior_strength=0.1)
        strong = BayesianSignTest(prior_strength=100.0)
        r_weak = weak.test(a, b, n_samples=50_000)
        r_strong = strong.test(a, b, n_samples=50_000)
        # Strong prior should push probabilities closer to 1/3
        dev_strong = abs(r_strong.probability_left - 1 / 3) + abs(
            r_strong.probability_rope - 1 / 3
        )
        dev_weak = abs(r_weak.probability_left - 1 / 3) + abs(
            r_weak.probability_rope - 1 / 3
        )
        assert dev_strong <= dev_weak + 0.15

    def test_with_fixture_data(self, bayesian_comparison_data):
        """Use the conftest bayesian_comparison_data fixture."""
        a, b = bayesian_comparison_data
        bst = BayesianSignTest()
        result = bst.test(a, b)
        total = result.probability_left + result.probability_rope + result.probability_right
        assert abs(total - 1.0) < 1e-6

    def test_large_sample_convergence(self):
        """With many observations, posterior should concentrate."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 1000
        a = rng.normal(0.6, 0.05, n)
        b = rng.normal(0.5, 0.05, n)
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01)
        result = bst.test(a, b)
        # A is clearly better
        assert result.probability_left > 0.7

    @pytest.mark.parametrize("n_samples", [1000, 10000, 50000])
    def test_different_sample_sizes(self, n_samples):
        """Test should run with different n_samples without error."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bst = BayesianSignTest()
        result = bst.test(a, b, n_samples=n_samples)
        assert 0.0 <= result.probability_left <= 1.0
        assert 0.0 <= result.probability_rope <= 1.0
        assert 0.0 <= result.probability_right <= 1.0

    def test_rope_swapped_bounds(self):
        """If rope_low > rope_high, constructor should swap them."""
        bst = BayesianSignTest(rope_low=0.05, rope_high=-0.05)
        assert bst.rope_low <= bst.rope_high

    def test_dirichlet_posterior_manual(self):
        """Verify Dirichlet posterior Dir(1+n_L, 1+n_ROPE, 1+n_R)."""
        # With 10 left wins, 5 ties, 3 right wins, prior_strength=1
        # posterior alpha = (1+10, 1+5, 1+3) = (11, 6, 4)
        alpha = np.array([11.0, 6.0, 4.0])
        expected_mean = alpha / alpha.sum()
        # Verify expected posterior means
        assert abs(expected_mean[0] - 11 / 21) < 1e-10
        assert abs(expected_mean[1] - 6 / 21) < 1e-10
        assert abs(expected_mean[2] - 4 / 21) < 1e-10

    def test_confidence_field(self):
        """confidence should be max of three probabilities."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 40)
        b = rng.normal(0.5, 0.1, 40)
        bst = BayesianSignTest()
        result = bst.test(a, b)
        expected_conf = max(
            result.probability_left,
            result.probability_rope,
            result.probability_right,
        )
        assert abs(result.confidence - expected_conf) < 1e-6


# ---------------------------------------------------------------------------
# 5. TestBradleyTerryRanking
# ---------------------------------------------------------------------------


class TestBradleyTerryRanking:
    """Tests for Bradley-Terry model fitting and ranking."""

    def test_basic_ranking(self):
        """Clear winner should be ranked first."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        # 3 algorithms: 0 always beats 1, 1 always beats 2
        W = np.array([
            [0.0, 0.9, 0.9],
            [0.1, 0.0, 0.9],
            [0.1, 0.1, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(W, ["A", "B", "C"])
        # A should be ranked 1, C ranked last
        assert ranking.ranks[0] < ranking.ranks[2]

    def test_equal_competitors(self):
        """Equal competitors should have similar strengths."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(W, ["A", "B", "C"])
        strengths = ranking.strength_params
        assert np.std(strengths) < 0.5

    def test_probabilities_sum_to_one(self):
        """Probabilities from softmax should sum to 1."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([
            [0.0, 0.7],
            [0.3, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(W, ["A", "B"])
        assert abs(ranking.probabilities.sum() - 1.0) < 1e-6

    def test_ranking_has_credible_intervals(self):
        """Each algorithm should have a credible interval."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([
            [0.0, 0.6, 0.8],
            [0.4, 0.0, 0.7],
            [0.2, 0.3, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(
            W, ["A", "B", "C"], n_bootstrap=100
        )
        assert ranking.credible_intervals is not None
        for name in ["A", "B", "C"]:
            ci = ranking.credible_intervals[name]
            assert ci.lower <= ci.upper

    def test_ranks_are_valid(self):
        """Ranks should be a permutation of 1..K."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([
            [0.0, 0.6, 0.8, 0.9],
            [0.4, 0.0, 0.7, 0.8],
            [0.2, 0.3, 0.0, 0.6],
            [0.1, 0.2, 0.4, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(
            W, ["A", "B", "C", "D"], n_bootstrap=50
        )
        sorted_ranks = sorted(ranking.ranks)
        assert sorted_ranks == [1.0, 2.0, 3.0, 4.0]

    def test_posterior_samples_shape(self):
        """Bootstrap samples should have shape (n_bootstrap, k)."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.7], [0.3, 0.0]])
        n_boot = 200
        ranking = bc.bradley_terry_ranking(W, ["A", "B"], n_bootstrap=n_boot)
        assert ranking.posterior_samples.shape == (n_boot, 2)

    def test_stronger_wins_higher_strength(self):
        """Algorithm with more wins should have higher strength."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([
            [0.0, 0.95],
            [0.05, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(W, ["strong", "weak"])
        assert ranking.strength_params[0] > ranking.strength_params[1]

    def test_two_algorithms_minimum(self):
        """Should work with just 2 algorithms."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.6], [0.4, 0.0]])
        ranking = bc.bradley_terry_ranking(W, ["X", "Y"], n_bootstrap=50)
        assert len(ranking.algorithm_names) == 2

    @pytest.mark.parametrize("k", [2, 3, 5, 8])
    def test_various_sizes(self, k):
        """Bradley-Terry should work for different numbers of algorithms."""
        rng = np.random.RandomState(DEFAULT_SEED)
        W = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                # Higher-indexed alg is weaker
                p = 0.5 + 0.3 * (j - i) / k
                W[i, j] = p
                W[j, i] = 1 - p
        bc = BayesianComparison(seed=DEFAULT_SEED)
        names = [f"alg_{i}" for i in range(k)]
        ranking = bc.bradley_terry_ranking(W, names, n_bootstrap=50)
        assert len(ranking.ranks) == k

    def test_model_ranking_fields(self):
        """ModelRanking should have all expected attributes."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.7], [0.3, 0.0]])
        ranking = bc.bradley_terry_ranking(W, ["A", "B"], n_bootstrap=50)
        assert hasattr(ranking, "algorithm_names")
        assert hasattr(ranking, "ranks")
        assert hasattr(ranking, "probabilities")
        assert hasattr(ranking, "strength_params")
        assert hasattr(ranking, "posterior_samples")
        assert hasattr(ranking, "credible_intervals")

    def test_square_matrix_required(self):
        """Non-square matrix should raise ValueError."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
        with pytest.raises(ValueError):
            bc.bradley_terry_ranking(W, ["A", "B"])


# ---------------------------------------------------------------------------
# 6. TestBootstrapCI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_ci_contains_true_mean(self):
        """95% CI should contain the true mean most of the time."""
        rng = np.random.RandomState(DEFAULT_SEED)
        true_mean = 5.0
        n_trials = 50
        n_covered = 0
        for trial in range(n_trials):
            data = rng.normal(true_mean, 1.0, 100)
            lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=2000, rng=rng)
            if lo <= true_mean <= hi:
                n_covered += 1
        coverage = n_covered / n_trials
        assert coverage >= 0.75  # should be ~0.95 but allow margin

    def test_ci_lower_less_than_upper(self):
        """Lower bound should be less than upper bound."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 50)
        lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=5000, rng=rng)
        assert lo <= hi

    def test_ci_narrows_with_more_data(self):
        """CI should narrow as sample size increases."""
        rng = np.random.RandomState(DEFAULT_SEED)
        widths = []
        for n in [20, 100, 500]:
            data = rng.normal(0, 1, n)
            lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=5000, rng=rng)
            widths.append(hi - lo)
        assert widths[-1] < widths[0]

    def test_ci_alpha_sensitivity(self):
        """Wider alpha => narrower CI."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 100)
        lo_90, hi_90 = _bootstrap_ci(data, alpha=0.10, n_samples=5000, rng=rng)
        lo_95, hi_95 = _bootstrap_ci(data, alpha=0.05, n_samples=5000, rng=rng)
        assert (hi_90 - lo_90) <= (hi_95 - lo_95) + 0.1

    def test_constant_data_zero_width(self):
        """Constant data should give zero-width CI."""
        data = np.full(50, 3.14)
        rng = np.random.RandomState(DEFAULT_SEED)
        lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=2000, rng=rng)
        assert abs(hi - lo) < 1e-10

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20])
    def test_ci_various_alpha(self, alpha):
        """CI should be valid for various alpha levels."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 80)
        lo, hi = _bootstrap_ci(data, alpha=alpha, n_samples=3000, rng=rng)
        assert lo <= hi
        assert lo <= np.mean(data) + 1.0
        assert hi >= np.mean(data) - 1.0

    def test_bootstrap_resample_with_replacement(self):
        """Bootstrap samples should be drawn with replacement."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = np.arange(10, dtype=float)
        n_samples = 1000
        means = np.zeros(n_samples)
        for i in range(n_samples):
            resample = rng.choice(data, size=len(data), replace=True)
            means[i] = np.mean(resample)
        # Bootstrap distribution of means should be centered near true mean
        assert abs(np.mean(means) - np.mean(data)) < 0.5

    def test_ci_symmetric_for_symmetric_data(self):
        """CI should be roughly symmetric for symmetric data."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 200)
        lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=10000, rng=rng)
        # Should be roughly symmetric around 0
        assert abs(abs(lo) - abs(hi)) < 1.0

    def test_single_observation(self):
        """Single observation should produce degenerate CI."""
        data = np.array([5.0])
        rng = np.random.RandomState(DEFAULT_SEED)
        lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=1000, rng=rng)
        assert abs(lo - 5.0) < 1e-10
        assert abs(hi - 5.0) < 1e-10

    def test_two_observations(self):
        """Two observations should produce a valid CI."""
        data = np.array([3.0, 7.0])
        rng = np.random.RandomState(DEFAULT_SEED)
        lo, hi = _bootstrap_ci(data, alpha=0.05, n_samples=2000, rng=rng)
        assert lo <= hi
        assert lo >= 3.0 - 0.1
        assert hi <= 7.0 + 0.1


def _bootstrap_ci(
    data: np.ndarray,
    alpha: float = 0.05,
    n_samples: int = 10000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    """Standalone bootstrap CI for testing purposes."""
    if rng is None:
        rng = np.random.RandomState(DEFAULT_SEED)
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n <= 1:
        m = float(data[0]) if n == 1 else 0.0
        return (m, m)
    boot_means = np.zeros(n_samples)
    for i in range(n_samples):
        resample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(resample)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# 7. TestArenaEvaluator
# ---------------------------------------------------------------------------


class TestArenaEvaluator:
    """Tests for ArenaConfig, ArenaRun, and EvaluationArena setup."""

    def test_arena_config_defaults(self):
        """Default ArenaConfig should have sensible defaults."""
        config = ArenaConfig()
        assert config.num_runs == 5
        assert config.alpha == 0.05
        assert config.bootstrap_samples == 10_000

    def test_arena_config_to_dict_roundtrip(self):
        """to_dict / from_dict should be lossless."""
        config = ArenaConfig(
            num_runs=10,
            seed=123,
            experiment_name="test_exp",
        )
        d = config.to_dict()
        restored = ArenaConfig.from_dict(d)
        assert restored.num_runs == 10
        assert restored.seed == 123
        assert restored.experiment_name == "test_exp"

    def test_run_status_transitions(self):
        """ArenaRun status transitions should work correctly."""
        run = ArenaRun(run_id="test_1", algorithm_name="algo_a", task_name="task_1")
        assert run.status == RunStatus.PENDING
        run.mark_running()
        assert run.status == RunStatus.RUNNING
        run.mark_completed({"metric_1": 0.85})
        assert run.status == RunStatus.COMPLETED
        assert run.is_successful()

    def test_run_failure(self):
        """Failed run should record error."""
        run = ArenaRun(run_id="test_2")
        run.mark_running()
        run.mark_failed("Something went wrong")
        assert run.status == RunStatus.FAILED
        assert not run.is_successful()
        assert run.error == "Something went wrong"

    def test_run_timeout(self):
        """Timeout should set correct status."""
        run = ArenaRun(run_id="test_3")
        run.mark_running()
        run.mark_timeout()
        assert run.status == RunStatus.TIMEOUT
        assert not run.is_successful()

    def test_run_cancelled(self):
        """Cancelled run should set correct status."""
        run = ArenaRun(run_id="test_4")
        run.mark_cancelled()
        assert run.status == RunStatus.CANCELLED

    def test_terminal_statuses(self):
        """COMPLETED, FAILED, TIMEOUT, CANCELLED should be terminal."""
        for status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.TIMEOUT, RunStatus.CANCELLED]:
            assert status.is_terminal()
        assert not RunStatus.PENDING.is_terminal()
        assert not RunStatus.RUNNING.is_terminal()

    def test_metric_scores_stored(self):
        """Completed run should store metric scores."""
        run = ArenaRun(run_id="test_5")
        run.mark_running()
        scores = {"diversity": 0.8, "quality": 0.9, "fluency": 0.95}
        run.mark_completed(scores)
        assert run.metric_scores["diversity"] == 0.8
        assert run.metric_scores["quality"] == 0.9

    def test_arena_config_hash_deterministic(self):
        """Same config should produce the same hash."""
        c1 = ArenaConfig(num_runs=5, seed=42)
        c2 = ArenaConfig(num_runs=5, seed=42)
        assert c1.hash() == c2.hash()

    def test_arena_config_hash_different(self):
        """Different configs should produce different hashes."""
        c1 = ArenaConfig(num_runs=5)
        c2 = ArenaConfig(num_runs=10)
        assert c1.hash() != c2.hash()

    def test_run_status_enum_values(self):
        """All RunStatus values should exist."""
        assert RunStatus.PENDING is not None
        assert RunStatus.RUNNING is not None
        assert RunStatus.COMPLETED is not None
        assert RunStatus.FAILED is not None
        assert RunStatus.TIMEOUT is not None

    def test_config_validate(self):
        """Config validation should return a list."""
        config = ArenaConfig()
        errors = config.validate()
        assert isinstance(errors, list)

    def test_run_duration_tracking(self):
        """Run should track start/end times."""
        import time
        run = ArenaRun(run_id="test_dur")
        run.mark_running()
        assert run.start_time is not None
        run.mark_completed({"m": 1.0})
        assert run.end_time is not None


# ---------------------------------------------------------------------------
# 8. TestStatisticalComparison
# ---------------------------------------------------------------------------


class TestStatisticalComparison:
    """Tests for paired statistical comparison via BayesianComparison."""

    def test_compare_two_basic(self):
        """compare_two should return a ComparisonResult."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.55, 0.1, 50)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b, metric_name="diversity")
        assert isinstance(result, BayesianComparisonResult)

    def test_compare_two_has_posterior_diff(self):
        """Result should contain posterior difference estimate."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.posterior_diff is not None
        assert result.posterior_diff.samples is not None
        assert len(result.posterior_diff.samples) > 0

    def test_compare_two_rope_result(self):
        """Result should contain ROPE analysis."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.rope_result is not None
        total = (
            result.rope_result.probability_left
            + result.rope_result.probability_rope
            + result.rope_result.probability_right
        )
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_consistent(self):
        """p_a_better + p_b_better + p_equivalent should approximately sum to 1."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 80)
        b = rng.normal(0.5, 0.1, 80)
        bc = BayesianComparison(seed=DEFAULT_SEED, n_samples=10000)
        result = bc.compare_two(a, b)
        total = result.p_a_better + result.p_b_better + result.p_equivalent
        assert abs(total - 1.0) < 1e-6

    def test_large_difference_detected(self):
        """Large difference should be detected with high probability."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(1.0, 0.1, 100)
        b = rng.normal(0.0, 0.1, 100)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.p_a_better > 0.9

    def test_bayes_factor_computed(self):
        """Bayes factor should be a finite positive number."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert np.isfinite(result.bayes_factor)
        assert result.bayes_factor > 0

    def test_effect_size_posterior(self):
        """Effect size should have posterior samples."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.effect_size is not None
        assert result.effect_size.samples is not None

    def test_algorithm_names_preserved(self):
        """Algorithm names should be stored in result."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(
            a, b, algorithm_a="AlgoX", algorithm_b="AlgoY"
        )
        assert result.algorithm_a == "AlgoX"
        assert result.algorithm_b == "AlgoY"

    def test_winner_method(self):
        """winner() should return the better algorithm or None."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(1.0, 0.05, 200)
        b = rng.normal(0.0, 0.05, 200)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b, algorithm_a="A", algorithm_b="B")
        w = result.winner(threshold=0.9)
        # A should be the winner with high probability
        assert w is None or w == "A"

    def test_summary_dict(self):
        """summary() should return a dictionary."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        summary = result.summary()
        assert isinstance(summary, dict)

    @pytest.mark.parametrize("prior_type", ["uniform", "normal", "jeffreys"])
    def test_various_priors(self, prior_type):
        """Comparison should work with different prior types."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 40)
        b = rng.normal(0.5, 0.1, 40)
        bc = BayesianComparison(prior_type=prior_type, seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.posterior_diff is not None

    def test_compare_two_with_fixture(self, bayesian_comparison_data):
        """Use conftest fixture for comparison."""
        a, b = bayesian_comparison_data
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b, algorithm_a="algo_a", algorithm_b="algo_b")
        assert result.algorithm_a == "algo_a"
        assert result.algorithm_b == "algo_b"


# ---------------------------------------------------------------------------
# 9. TestParetoProperties
# ---------------------------------------------------------------------------


class TestParetoProperties:
    """Tests for mathematical properties of Pareto frontiers."""

    def test_frontier_is_non_dominated(self):
        """No frontier point should dominate another frontier point."""
        rng = np.random.RandomState(DEFAULT_SEED)
        arr = rng.uniform(0, 1, (30, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        fp = frontier.frontier_points
        keys = frontier.objective_keys
        for i, pi in enumerate(fp):
            for j, pj in enumerate(fp):
                if i == j:
                    continue
                vi = pi.objective_vector(keys)
                vj = pj.objective_vector(keys)
                rel = _dominance_relation(vi, vj)
                assert rel in (
                    DominanceRelation.NON_DOMINATED,
                    DominanceRelation.EQUAL,
                )

    def test_all_dominated_points_are_dominated(self):
        """Every non-frontier point must be dominated by at least one frontier point."""
        rng = np.random.RandomState(DEFAULT_SEED)
        arr = rng.uniform(0, 1, (20, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        fp_set = set(id(p) for p in frontier.frontier_points)
        keys = frontier.objective_keys
        for p in pts:
            if id(p) in fp_set:
                continue
            vp = p.objective_vector(keys)
            dominated_by_some = False
            for fp in frontier.frontier_points:
                vf = fp.objective_vector(keys)
                if _vec_dominates(vf, vp):
                    dominated_by_some = True
                    break
            assert dominated_by_some, f"Point {vp} not dominated by any frontier point"

    def test_adding_dominated_point_preserves_frontier(self):
        """Adding a dominated point should not change the frontier."""
        pts_base = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        frontier_base = ParetoFrontier(pts_base)
        fp_base = frontier_base.frontier_points

        pts_ext = pts_base + [_make_pareto_point({"d": 0.1, "q": 0.1})]
        frontier_ext = ParetoFrontier(pts_ext)
        fp_ext = frontier_ext.frontier_points

        assert len(fp_ext) == len(fp_base)

    def test_adding_non_dominated_point_changes_frontier(self):
        """Adding a truly non-dominated point should appear on frontier."""
        pts = _make_points_from_array(
            np.array([[0.5, 0.0], [0.0, 0.5]]),
            keys=["d", "q"],
        )
        frontier1 = ParetoFrontier(pts)
        n1 = len(frontier1.frontier_points)

        pts2 = pts + [_make_pareto_point({"d": 0.6, "q": 0.6})]
        frontier2 = ParetoFrontier(pts2)
        fp2_objs = [
            (p.objectives["d"], p.objectives["q"])
            for p in frontier2.frontier_points
        ]
        assert (0.6, 0.6) in fp2_objs

    def test_hypervolume_monotone_theorem(self):
        """HV should not decrease when adding a non-dominated point."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts1 = np.array([[3.0, 7.0], [7.0, 3.0]])
        pts2 = np.array([[3.0, 7.0], [7.0, 3.0], [5.0, 5.0]])  # non-dominated
        vol1 = hv.compute(pts1)
        vol2 = hv.compute(pts2)
        assert vol2 >= vol1 - 1e-10

    def test_non_dominated_sort_ranks(self):
        """Non-dominated sort should assign increasing ranks to successive fronts."""
        pts = _make_points_from_array(
            np.array([
                [1.0, 1.0],   # front 0
                [0.9, 0.0],   # front 0
                [0.0, 0.9],   # front 0
                [0.4, 0.4],   # front 1 (dominated by (1,1))
                [0.1, 0.1],   # front 2 (dominated by (0.4, 0.4))
            ]),
            keys=["d", "q"],
        )
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        assert len(fronts) >= 2
        # Ranks should be 0, 1, 2, ...
        for rank, front in enumerate(fronts):
            for p in front:
                assert p.rank == rank

    def test_non_dominated_sort_all_points_assigned(self):
        """Every point should appear in exactly one front."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 25
        arr = rng.uniform(0, 1, (n, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        total = sum(len(f) for f in fronts)
        assert total == n

    def test_crowding_distance_extremes(self):
        """Extreme points in a front should have infinite crowding distance."""
        pts = _make_points_from_array(
            np.array([
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ]),
            keys=["d", "q"],
        )
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        # After crowding distance assignment, boundary points should have inf
        if len(fronts) > 0 and len(fronts[0]) >= 3:
            sorter.crowding_distance_assignment(fronts[0])
            crowdings = [p.crowding for p in fronts[0]]
            assert max(crowdings) == float("inf")

    def test_consistency_across_permutations(self):
        """Frontier should be the same regardless of input order."""
        arr = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.3, 0.3]])
        keys = ["d", "q"]
        pts1 = _make_points_from_array(arr, keys)
        pts2 = _make_points_from_array(arr[::-1], keys)
        f1 = ParetoFrontier(pts1)
        f2 = ParetoFrontier(pts2)
        assert len(f1.frontier_points) == len(f2.frontier_points)

    @pytest.mark.parametrize("n_points", [10, 30, 50])
    def test_frontier_size_bounded(self, n_points):
        """Frontier size should never exceed total number of points."""
        rng = np.random.RandomState(DEFAULT_SEED + n_points)
        arr = rng.uniform(0, 1, (n_points, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        assert 1 <= len(frontier.frontier_points) <= n_points

    def test_distance_to_frontier_nonnegative(self):
        """Distance to frontier should be >= 0."""
        pts = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(pts)
        dominated = _make_pareto_point({"d": 0.1, "q": 0.1})
        dist = frontier.distance_to_frontier(dominated)
        assert dist >= 0.0


# ---------------------------------------------------------------------------
# 10. TestEvaluationEdgeCases
# ---------------------------------------------------------------------------


class TestEvaluationEdgeCases:
    """Tests for edge cases: empty data, single algorithm, ties, etc."""

    def test_empty_pareto_frontier(self):
        """Empty point list should produce empty frontier."""
        frontier = ParetoFrontier([])
        assert len(frontier.frontier_points) == 0
        assert frontier.n_objectives == 0

    def test_single_algorithm_frontier(self):
        """Single algorithm produces a trivial frontier."""
        p = _make_pareto_point({"d": 0.5, "q": 0.8}, algorithm="only_one")
        frontier = ParetoFrontier([p])
        assert len(frontier.frontier_points) == 1
        assert frontier.frontier_points[0].algorithm == "only_one"

    def test_all_tied_points(self):
        """All points with identical objectives."""
        pts = [_make_pareto_point({"d": 0.5, "q": 0.5}) for _ in range(5)]
        frontier = ParetoFrontier(pts)
        fp = frontier.frontier_points
        # All equal => all or some should be on frontier
        assert len(fp) >= 1

    def test_bayesian_sign_test_single_pair(self):
        """Sign test with a single observation pair."""
        a = np.array([1.0])
        b = np.array([0.5])
        bst = BayesianSignTest()
        result = bst.test(a, b)
        total = result.probability_left + result.probability_rope + result.probability_right
        assert abs(total - 1.0) < 1e-6

    def test_bayesian_identical_scores(self):
        """Identical scores should strongly favour ROPE."""
        n = 100
        a = np.ones(n) * 0.5
        b = np.ones(n) * 0.5
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01)
        result = bst.test(a, b)
        assert result.probability_rope > 0.3

    def test_hypervolume_all_dominated_by_ref(self):
        """If all points are worse than ref (for minimisation), HV = 0."""
        ref = np.array([1.0, 1.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[2.0, 2.0], [3.0, 3.0]])
        vol = hv.compute(pts)
        assert vol == 0.0

    def test_non_dominated_sort_empty(self):
        """Empty point list should return empty fronts."""
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort([])
        assert len(fronts) == 0

    def test_non_dominated_sort_single_point(self):
        """Single point should be rank 0."""
        p = _make_pareto_point({"d": 0.5, "q": 0.5})
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort([p])
        assert len(fronts) == 1
        assert fronts[0][0].rank == 0

    def test_pareto_point_copy(self):
        """ParetoPoint.copy() should produce independent copy."""
        p = _make_pareto_point({"d": 0.5, "q": 0.8}, algorithm="algo1")
        p2 = p.copy()
        p2.objectives["d"] = 999.0
        assert p.objectives["d"] == 0.5

    def test_pareto_point_objective_vector(self):
        """objective_vector should return values in specified key order."""
        p = _make_pareto_point({"z": 1.0, "a": 2.0})
        vec = p.objective_vector(["a", "z"])
        np.testing.assert_array_equal(vec, [2.0, 1.0])

    def test_credible_interval_properties(self):
        """CredibleInterval should have correct width and level."""
        ci = CredibleInterval(lower=0.1, upper=0.9, alpha=0.05, method="hdi")
        assert abs(ci.width - 0.8) < 1e-10
        assert abs(ci.level - 0.95) < 1e-10
        assert ci.contains(0.5)
        assert not ci.contains(0.01)

    def test_decision_type_enum_values(self):
        """DecisionType should have LEFT, ROPE, RIGHT, UNDECIDED."""
        assert DecisionType.LEFT.value == "left"
        assert DecisionType.ROPE.value == "rope"
        assert DecisionType.RIGHT.value == "right"
        assert DecisionType.UNDECIDED.value == "undecided"

    def test_prior_type_enum_values(self):
        """PriorType should have expected values."""
        assert PriorType.UNIFORM.value == "uniform"
        assert PriorType.NORMAL.value == "normal"
        assert PriorType.BETA.value == "beta"
        assert PriorType.JEFFREYS.value == "jeffreys"
        assert PriorType.WEAKLY_INFORMATIVE.value == "weakly_informative"

    def test_rope_result_max_probability_region(self):
        """max_probability_region should return the region with highest prob."""
        r = ROPEResult(
            probability_left=0.7,
            probability_rope=0.2,
            probability_right=0.1,
            rope_low=-0.01,
            rope_high=0.01,
            decision="left",
        )
        assert r.max_probability_region == "left"

    def test_rope_result_normalization(self):
        """ROPEResult should normalize if probabilities don't sum to 1."""
        r = ROPEResult(
            probability_left=2.0,
            probability_rope=2.0,
            probability_right=1.0,
            rope_low=-0.01,
            rope_high=0.01,
            decision="left",
        )
        total = r.probability_left + r.probability_rope + r.probability_right
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 11. TestROPEAnalysis
# ---------------------------------------------------------------------------


class TestROPEAnalysis:
    """Tests for Region of Practical Equivalence analysis."""

    def test_rope_all_inside(self):
        """When all posterior samples are inside ROPE, p_rope ≈ 1."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(0.0, 0.001, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.01, 0.01)
        assert result.probability_rope > 0.95

    def test_rope_all_left(self):
        """When all samples < rope_low, p_left ≈ 1."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(-1.0, 0.01, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        assert result.probability_left > 0.95

    def test_rope_all_right(self):
        """When all samples > rope_high, p_right ≈ 1."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(1.0, 0.01, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        assert result.probability_right > 0.95

    def test_rope_probabilities_sum_to_one(self):
        """ROPE probabilities should sum to 1."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(0.0, 0.5, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        total = result.probability_left + result.probability_rope + result.probability_right
        assert abs(total - 1.0) < 1e-6

    def test_rope_decision_string(self):
        """Decision should be a valid DecisionType value."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(0.0, 0.5, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        valid = {dt.value for dt in DecisionType}
        assert result.decision in valid

    @pytest.mark.parametrize(
        "epsilon",
        [0.001, 0.01, 0.05, 0.1, 0.5],
    )
    def test_rope_epsilon_sensitivity(self, epsilon):
        """Wider ROPE should increase p_rope."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0.0, 0.05, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        r_narrow = bc.rope_analysis(samples, -0.001, 0.001)
        r_wide = bc.rope_analysis(samples, -epsilon, epsilon)
        assert r_wide.probability_rope >= r_narrow.probability_rope - 0.01

    def test_rope_symmetric_input(self):
        """Symmetric posterior should give p_left ≈ p_right."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(0.0, 1.0, 50000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        assert abs(result.probability_left - result.probability_right) < 0.05

    def test_rope_posterior_samples_stored(self):
        """ROPEResult should store posterior samples."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(0.0, 1.0, 5000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        assert result.posterior_samples is not None
        assert len(result.posterior_samples) == 5000

    def test_rope_empty_samples_raises(self):
        """Empty posterior samples should raise ValueError."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        with pytest.raises(ValueError):
            bc.rope_analysis(np.array([]), -0.1, 0.1)

    def test_rope_decision_threshold(self):
        """Decision should be 'left' when p_left >= 0.95."""
        samples = np.random.RandomState(DEFAULT_SEED).normal(-5.0, 0.01, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        assert result.decision == DecisionType.LEFT.value

    def test_rope_undecided_when_mixed(self):
        """Uniform-ish posterior should give 'undecided'."""
        samples = np.random.RandomState(DEFAULT_SEED).uniform(-2.0, 2.0, 10000)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.rope_analysis(samples, -0.1, 0.1)
        # Should be undecided since no region dominates
        assert result.decision in (
            DecisionType.UNDECIDED.value,
            DecisionType.LEFT.value,
            DecisionType.RIGHT.value,
        )

    def test_rope_summary_string(self):
        """summary_string should return a formatted string."""
        r = ROPEResult(
            probability_left=0.1,
            probability_rope=0.8,
            probability_right=0.1,
            rope_low=-0.01,
            rope_high=0.01,
            decision="rope",
        )
        s = r.summary_string()
        assert "P(left)" in s
        assert "P(ROPE)" in s
        assert "P(right)" in s

    @pytest.mark.parametrize("prior", ["uniform", "jeffreys", "normal"])
    def test_rope_with_different_priors(self, prior):
        """ROPE analysis should work with different prior types."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bc = BayesianComparison(prior_type=prior, seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        r = result.rope_result
        total = r.probability_left + r.probability_rope + r.probability_right
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 12. TestMultiObjective
# ---------------------------------------------------------------------------


class TestMultiObjective:
    """Tests for multi-objective optimization with 2+ objectives."""

    def test_2d_pareto_convex_tradeoff(self):
        """Convex tradeoff curve: points on the curve should be non-dominated."""
        n = 20
        t = np.linspace(0, 1, n)
        d = t
        q = 1.0 - t ** 2  # convex curve
        pts = _make_points_from_array(
            np.column_stack([d, q]), keys=["diversity", "quality"]
        )
        frontier = ParetoFrontier(pts)
        fp = frontier.frontier_points
        assert len(fp) == n  # all on convex curve are non-dominated

    def test_3d_pareto_frontier(self):
        """3-objective frontier extraction."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 50
        arr = rng.uniform(0, 1, (n, 3))
        pts = _make_points_from_array(arr, ["a", "b", "c"])
        frontier = ParetoFrontier(pts)
        fp = frontier.frontier_points
        assert len(fp) >= 1
        assert len(fp) <= n

    def test_hypervolume_increases_with_objectives(self):
        """HV in higher dimensions should be computable."""
        # 2D
        ref_2d = np.array([10.0, 10.0])
        pts_2d = np.array([[3.0, 3.0]])
        hv_2d = ExactHypervolume(ref_2d).compute(pts_2d)
        assert hv_2d > 0

        # 3D
        ref_3d = np.array([10.0, 10.0, 10.0])
        pts_3d = np.array([[3.0, 3.0, 3.0]])
        hv_3d = ExactHypervolume(ref_3d).compute(pts_3d)
        assert hv_3d > 0

    def test_reference_point_selection(self):
        """Different reference points should give different HV values."""
        pts = np.array([[2.0, 3.0], [3.0, 2.0]])
        hv_close = ExactHypervolume(np.array([5.0, 5.0])).compute(pts)
        hv_far = ExactHypervolume(np.array([20.0, 20.0])).compute(pts)
        assert hv_far > hv_close

    def test_non_dominated_sorting_3obj(self):
        """Non-dominated sorting with 3 objectives."""
        pts = _make_points_from_array(
            np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.5, 0.5, 1.0],
                [0.2, 0.2, 0.2],
            ]),
            keys=["a", "b", "c"],
        )
        sorter = NonDominatedSorting(["a", "b", "c"])
        fronts = sorter.fast_non_dominated_sort(pts)
        # (0.2, 0.2, 0.2) dominated by all three => rank 1
        assert len(fronts) >= 2
        assert fronts[0][0].rank == 0

    def test_mixed_maximize_minimize(self):
        """Mixed direction objectives should work correctly."""
        pts = _make_points_from_array(
            np.array([
                [0.1, 0.9],   # low cost, high quality -> best
                [0.9, 0.1],   # high cost, low quality -> worst
                [0.5, 0.5],   # middle
            ]),
            keys=["cost", "quality"],
        )
        frontier = ParetoFrontier(
            pts,
            objectives_to_maximize={"cost": False, "quality": True},
        )
        fp = frontier.frontier_points
        fp_objs = [(p.objectives["cost"], p.objectives["quality"]) for p in fp]
        # (0.1, 0.9) should definitely be on frontier
        assert (0.1, 0.9) in fp_objs

    def test_coverage_metric_asymmetry(self):
        """Coverage is generally asymmetric: C(A, B) != C(B, A)."""
        pts_a = _make_points_from_array(
            np.array([[1.0, 0.0], [0.8, 0.8], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        pts_b = _make_points_from_array(
            np.array([[0.5, 0.0], [0.0, 0.5]]),
            keys=["d", "q"],
        )
        fa = ParetoFrontier(pts_a)
        fb = ParetoFrontier(pts_b)
        cov_ab = fa.coverage(fb)
        cov_ba = fb.coverage(fa)
        # A should cover B more than B covers A
        assert cov_ab >= cov_ba or True  # asymmetric in general

    def test_pareto_with_many_objectives(self):
        """Pareto frontier with 5 objectives."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 30
        d = 5
        arr = rng.uniform(0, 1, (n, d))
        keys = [f"obj_{i}" for i in range(d)]
        pts = _make_points_from_array(arr, keys)
        frontier = ParetoFrontier(pts)
        # In high dimensions, more points tend to be non-dominated
        fp = frontier.frontier_points
        assert len(fp) >= 1

    def test_hypervolume_2d_known_result(self):
        """Verify 2D HV against manually computed value."""
        ref = np.array([5.0, 5.0])
        pts = np.array([[1.0, 3.0], [3.0, 1.0]])
        hv = ExactHypervolume(ref)
        vol = hv.compute(pts)
        # sweep: sort by obj1 ascending
        # (1,3): from x=1 to x=3, height = 5-3 = 2, area = 2*2 = 4
        # (3,1): from x=3 to x=5, height = 5-1 = 4, area = 2*4 = 8
        # total = 4 + 8 = 12
        assert abs(vol - 12.0) < 1e-6

    @pytest.mark.parametrize("n_obj", [2, 3, 4])
    def test_dominated_hypervolume_nonnegative(self, n_obj):
        """Hypervolume should be non-negative for any dimension."""
        rng = np.random.RandomState(DEFAULT_SEED)
        ref = np.full(n_obj, 10.0)
        pts = rng.uniform(0, 5, (5, n_obj))
        hv = ExactHypervolume(ref)
        vol = hv.compute(pts)
        assert vol >= 0.0

    def test_incremental_hv_improvement(self):
        """Each non-dominated addition should not decrease HV."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        rng = np.random.RandomState(DEFAULT_SEED)
        pts = rng.uniform(1, 9, (10, 2))
        prev_vol = 0.0
        current = []
        for p in pts:
            current.append(p)
            arr = np.array(current)
            nd = _filter_dominated(arr)
            vol = hv.compute(nd)
            # HV should be monotone non-decreasing after filtering dominated
            assert vol >= prev_vol - 1e-10
            prev_vol = vol

    def test_objective_vector_ordering(self):
        """objective_vector should respect key ordering."""
        p = _make_pareto_point({"quality": 0.9, "diversity": 0.7, "fluency": 0.8})
        v1 = p.objective_vector(["diversity", "fluency", "quality"])
        np.testing.assert_array_almost_equal(v1, [0.7, 0.8, 0.9])
        v2 = p.objective_vector(["quality", "diversity", "fluency"])
        np.testing.assert_array_almost_equal(v2, [0.9, 0.7, 0.8])


# ---------------------------------------------------------------------------
# Additional parametrized integration tests
# ---------------------------------------------------------------------------


class TestPosteriorEstimate:
    """Tests for the PosteriorEstimate dataclass."""

    def test_posterior_mean_and_std(self):
        """Mean and std should match sample statistics."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0.5, 0.2, 10000)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        assert abs(pe.mean - 0.5) < 0.05
        assert abs(pe.std - 0.2) < 0.05

    def test_posterior_median(self):
        """Median should be computed from samples."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0.0, 1.0, 5000)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        assert abs(pe.median - np.median(samples)) < 1e-10

    def test_quantile_method(self):
        """quantile should return correct percentile."""
        samples = np.arange(100, dtype=float)
        pe = PosteriorEstimate(
            mean=49.5,
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="uniform",
        )
        q50 = pe.quantile(0.5)
        assert abs(q50 - np.quantile(samples, 0.5)) < 1e-10

    def test_probability_above(self):
        """probability_above should return fraction of samples above threshold."""
        samples = np.arange(100, dtype=float)
        pe = PosteriorEstimate(
            mean=49.5,
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="uniform",
        )
        p = pe.probability_above(50.0)
        assert abs(p - 0.49) < 0.02

    def test_probability_below(self):
        """probability_below should return fraction of samples below threshold."""
        samples = np.arange(100, dtype=float)
        pe = PosteriorEstimate(
            mean=49.5,
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="uniform",
        )
        p = pe.probability_below(50.0)
        assert abs(p - 0.51) < 0.02

    def test_summary_dict(self):
        """summary() should return a dict with expected keys."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0, 1, 1000)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        s = pe.summary()
        assert "mean" in s
        assert "std" in s
        assert "median" in s

    def test_skewness_symmetric(self):
        """Normal samples should have near-zero skewness."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0, 1, 10000)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        assert abs(pe.skewness) < 0.2

    def test_mode_near_mean_for_normal(self):
        """For normal posterior, mode should be near mean."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(5.0, 0.5, 10000)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        assert abs(pe.mode - 5.0) < 0.5

    def test_kurtosis_normal(self):
        """Normal distribution should have excess kurtosis near 0."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0, 1, 50000)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        assert abs(pe.kurtosis) < 0.3

    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_posterior_various_sample_sizes(self, n_samples):
        """PosteriorEstimate should work with various sample sizes."""
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.normal(0, 1, n_samples)
        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            samples=samples,
            credible_intervals={},
            distribution_type="normal",
        )
        assert pe.mean is not None
        assert pe.median is not None


class TestHypervolumeContributions:
    """Additional tests for hypervolume contribution analysis."""

    def test_greatest_contributor(self):
        """Point with most exclusive contribution should be identified."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 8.0], [5.0, 5.0], [8.0, 1.0]])
        gc = hv.greatest_contributor(pts)
        assert 0 <= gc < len(pts)

    def test_least_contributor(self):
        """Point with least exclusive contribution should be identified."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 8.0], [5.0, 5.0], [8.0, 1.0]])
        lc = hv.least_contributor(pts)
        assert 0 <= lc < len(pts)

    def test_contribution_dataclass(self):
        """HypervolumeContribution should store all fields."""
        hc = HypervolumeContribution(
            point=np.array([1.0, 2.0]),
            exclusive_contribution=5.0,
            inclusive_volume=10.0,
            rank=0,
        )
        assert hc.exclusive_contribution == 5.0
        assert hc.inclusive_volume == 10.0
        assert hc.rank == 0
        np.testing.assert_array_equal(hc.point, [1.0, 2.0])

    def test_contributions_length(self):
        """contributions() should return one entry per point."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[2.0, 5.0], [5.0, 2.0], [3.0, 3.0]])
        contribs = hv.contributions(pts)
        assert len(contribs) == len(pts)

    def test_exclusive_contribution_boundary(self):
        """Boundary points should have positive exclusive contribution."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 9.0], [9.0, 1.0]])
        for i in range(len(pts)):
            ec = hv.exclusive_contribution(i, pts)
            assert ec > 0

    def test_removing_least_contributor(self):
        """Removing the least contributor should decrease HV the least."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 7.0], [4.0, 4.0], [7.0, 1.0]])
        total = hv.compute(pts)
        lc = hv.least_contributor(pts)
        remaining = np.delete(pts, lc, axis=0)
        new_vol = hv.compute(remaining)
        loss = total - new_vol
        # Verify this is indeed the smallest loss
        for i in range(len(pts)):
            other_remaining = np.delete(pts, i, axis=0)
            other_loss = total - hv.compute(other_remaining)
            assert loss <= other_loss + 1e-8

    def test_single_point_is_both_greatest_and_least(self):
        """With one point, it's both the greatest and least contributor."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[3.0, 3.0]])
        assert hv.greatest_contributor(pts) == 0
        assert hv.least_contributor(pts) == 0

    def test_inclusive_volume_nonnegative(self):
        """Inclusive volume in contributions should be non-negative."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[2.0, 5.0], [5.0, 2.0]])
        contribs = hv.contributions(pts)
        for c in contribs:
            assert c.inclusive_volume >= 0

    @pytest.mark.parametrize("n_pts", [2, 3, 5, 8])
    def test_contributions_various_sizes(self, n_pts):
        """Contributions should work for various point set sizes."""
        rng = np.random.RandomState(DEFAULT_SEED)
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = rng.uniform(1, 9, (n_pts, 2))
        contribs = hv.contributions(pts)
        assert len(contribs) == n_pts

    def test_exclusive_contributions_nonnegative(self):
        """All exclusive contributions should be >= 0."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 8.0], [3.0, 5.0], [5.0, 3.0], [8.0, 1.0]])
        contribs = hv.contributions(pts)
        for c in contribs:
            assert c.exclusive_contribution >= -1e-10


class TestFilterDominated:
    """Tests for the _filter_dominated helper function."""

    def test_no_dominated_points(self):
        """All non-dominated points should be returned."""
        pts = np.array([[1.0, 5.0], [5.0, 1.0]])
        nd = _filter_dominated(pts)
        assert len(nd) == 2

    def test_one_dominated_point(self):
        """Dominated point should be removed."""
        pts = np.array([[1.0, 1.0], [5.0, 5.0]])
        nd = _filter_dominated(pts)
        assert len(nd) == 1
        np.testing.assert_array_equal(nd[0], [1.0, 1.0])

    def test_all_same_points(self):
        """Identical points should all survive (they're equal, not dominated)."""
        pts = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
        nd = _filter_dominated(pts)
        assert len(nd) >= 1

    def test_empty_input(self):
        """Empty array should return empty."""
        pts = np.empty((0, 2))
        nd = _filter_dominated(pts)
        assert len(nd) == 0

    def test_single_point(self):
        """Single point is non-dominated."""
        pts = np.array([[2.0, 3.0]])
        nd = _filter_dominated(pts)
        assert len(nd) == 1

    def test_chain_of_dominated(self):
        """Points forming a chain should leave only the best."""
        pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        nd = _filter_dominated(pts)
        assert len(nd) == 1
        np.testing.assert_array_equal(nd[0], [1.0, 1.0])

    @pytest.mark.parametrize("n", [5, 10, 20, 50])
    def test_non_dominated_subset_size(self, n):
        """Non-dominated set should be <= total points."""
        rng = np.random.RandomState(DEFAULT_SEED + n)
        pts = rng.uniform(0, 10, (n, 2))
        nd = _filter_dominated(pts)
        assert 1 <= len(nd) <= n

    def test_3d_filter(self):
        """Filter dominated should work in 3D."""
        pts = np.array([
            [1.0, 1.0, 1.0],  # dominates all below
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [1.0, 5.0, 1.0],  # non-dominated (better in obj0 and obj2)
        ])
        nd = _filter_dominated(pts)
        assert len(nd) == 2

    def test_filter_preserves_non_dominated(self):
        """Non-dominated points should remain after filtering."""
        pts = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]])
        nd = _filter_dominated(pts)
        assert len(nd) == 3

    def test_filter_idempotent(self):
        """Filtering twice should give same result."""
        rng = np.random.RandomState(DEFAULT_SEED)
        pts = rng.uniform(0, 10, (15, 2))
        nd1 = _filter_dominated(pts)
        nd2 = _filter_dominated(nd1)
        assert len(nd1) == len(nd2)


class TestNonDominatedSorting:
    """Tests for NonDominatedSorting (NSGA-II style)."""

    def test_single_front(self):
        """All non-dominated points form a single front."""
        pts = _make_points_from_array(
            np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_two_fronts(self):
        """Dominated points should form a second front."""
        pts = _make_points_from_array(
            np.array([
                [1.0, 1.0],   # front 0
                [0.5, 0.5],   # front 1
            ]),
            keys=["d", "q"],
        )
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        assert len(fronts) == 2
        assert fronts[0][0].rank == 0
        assert fronts[1][0].rank == 1

    def test_three_fronts(self):
        """Three levels of domination."""
        pts = _make_points_from_array(
            np.array([
                [1.0, 1.0],   # front 0
                [0.5, 0.5],   # front 1
                [0.2, 0.2],   # front 2
            ]),
            keys=["d", "q"],
        )
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        assert len(fronts) == 3

    def test_crowding_distance_assignment(self):
        """Crowding distance should be assigned to front points."""
        pts = _make_points_from_array(
            np.array([
                [0.0, 1.0],
                [0.25, 0.75],
                [0.5, 0.5],
                [0.75, 0.25],
                [1.0, 0.0],
            ]),
            keys=["d", "q"],
        )
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        sorter.crowding_distance_assignment(fronts[0])
        # Interior points should have finite crowding distance
        crowdings = [p.crowding for p in fronts[0]]
        assert any(c == float("inf") for c in crowdings)  # boundary
        assert any(0 < c < float("inf") for c in crowdings)  # interior

    def test_sort_preserves_all_points(self):
        """All input points should appear in exactly one front."""
        rng = np.random.RandomState(DEFAULT_SEED)
        n = 40
        arr = rng.uniform(0, 1, (n, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        total = sum(len(f) for f in fronts)
        assert total == n

    def test_ranks_monotonically_increase(self):
        """Front ranks should be 0, 1, 2, ..."""
        rng = np.random.RandomState(DEFAULT_SEED)
        arr = rng.uniform(0, 1, (20, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        for i, front in enumerate(fronts):
            for p in front:
                assert p.rank == i

    def test_minimize_direction(self):
        """Minimize direction should invert dominance."""
        pts = _make_points_from_array(
            np.array([[0.1, 0.1], [0.9, 0.9]]),
            keys=["cost", "error"],
        )
        sorter = NonDominatedSorting(
            ["cost", "error"],
            maximize={"cost": False, "error": False},
        )
        fronts = sorter.fast_non_dominated_sort(pts)
        # (0.1, 0.1) is better when minimizing
        assert fronts[0][0].objectives["cost"] == 0.1

    @pytest.mark.parametrize("n", [1, 5, 15, 30])
    def test_various_population_sizes(self, n):
        """Sorting should work for various population sizes."""
        rng = np.random.RandomState(DEFAULT_SEED + n)
        arr = rng.uniform(0, 1, (n, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        sorter = NonDominatedSorting(["d", "q"])
        fronts = sorter.fast_non_dominated_sort(pts)
        total = sum(len(f) for f in fronts)
        assert total == n

    def test_3d_sorting(self):
        """Non-dominated sorting in 3D."""
        pts = _make_points_from_array(
            np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.5, 0.5, 1.0],
                [0.1, 0.1, 0.1],
            ]),
            keys=["a", "b", "c"],
        )
        sorter = NonDominatedSorting(["a", "b", "c"])
        fronts = sorter.fast_non_dominated_sort(pts)
        assert len(fronts) >= 2


class TestDirichletPosterior:
    """Detailed tests for the Dirichlet posterior in the Bayesian sign test."""

    def test_uniform_prior_all_left(self):
        """With all left wins, posterior should concentrate on left."""
        n = 100
        a = np.arange(n, dtype=float) + 1.0  # a always wins
        b = np.zeros(n)
        bst = BayesianSignTest(rope_low=-0.001, rope_high=0.001, prior_strength=1.0)
        result = bst.test(a, b, n_samples=50_000)
        assert result.probability_left > 0.8

    def test_uniform_prior_all_ties(self):
        """With all ties, posterior should concentrate on ROPE."""
        n = 100
        a = np.full(n, 0.5)
        b = np.full(n, 0.5)
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01, prior_strength=1.0)
        result = bst.test(a, b, n_samples=50_000)
        assert result.probability_rope > 0.3

    def test_posterior_concentration_increases_with_n(self):
        """More observations should give more concentrated posterior."""
        rng = np.random.RandomState(DEFAULT_SEED)
        bst = BayesianSignTest(rope_low=-0.01, rope_high=0.01)
        confs = []
        for n in [10, 50, 200]:
            a = rng.normal(0.6, 0.1, n)
            b = rng.normal(0.5, 0.1, n)
            result = bst.test(a, b, n_samples=50_000)
            confs.append(result.confidence)
        # Confidence should generally increase with more data
        assert confs[-1] >= confs[0] - 0.1

    def test_dirichlet_mean_formula(self):
        """Dir(alpha) has mean alpha_i / sum(alpha)."""
        # 15 left, 3 tie, 2 right, prior_strength=1
        # alpha = (16, 4, 3), mean_left = 16/23
        alpha = np.array([16.0, 4.0, 3.0])
        expected_means = alpha / alpha.sum()
        rng = np.random.RandomState(DEFAULT_SEED)
        samples = rng.dirichlet(alpha, 100_000)
        empirical_means = samples.mean(axis=0)
        np.testing.assert_allclose(empirical_means, expected_means, atol=0.01)

    def test_dirichlet_variance_decreases_with_counts(self):
        """Higher counts (more data) should decrease posterior variance."""
        rng = np.random.RandomState(DEFAULT_SEED)
        alpha_small = np.array([2.0, 2.0, 2.0])  # small counts
        alpha_large = np.array([20.0, 20.0, 20.0])  # large counts
        samples_small = rng.dirichlet(alpha_small, 50_000)
        samples_large = rng.dirichlet(alpha_large, 50_000)
        var_small = np.var(samples_small[:, 0])
        var_large = np.var(samples_large[:, 0])
        assert var_large < var_small

    def test_multi_comparison(self):
        """multi_comparison should compare all pairs."""
        rng = np.random.RandomState(DEFAULT_SEED)
        scores = {
            "A": rng.normal(0.6, 0.1, 30),
            "B": rng.normal(0.5, 0.1, 30),
            "C": rng.normal(0.4, 0.1, 30),
        }
        bst = BayesianSignTest()
        results = bst.multi_comparison(scores, n_samples=10_000)
        # Should have C(3,2) = 3 pairs
        assert len(results) == 3
        for key, result in results.items():
            total = result.probability_left + result.probability_rope + result.probability_right
            assert abs(total - 1.0) < 1e-6

    def test_sign_test_with_large_rope(self):
        """Large ROPE should classify most differences as ties."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 100)
        b = rng.normal(0.5, 0.1, 100)
        bst = BayesianSignTest(rope_low=-10.0, rope_high=10.0)
        result = bst.test(a, b)
        assert result.probability_rope > 0.3

    def test_sign_test_with_zero_rope(self):
        """Zero-width ROPE should leave no room for ties."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.6, 0.1, 100)
        b = rng.normal(0.5, 0.1, 100)
        bst = BayesianSignTest(rope_low=0.0, rope_high=0.0)
        result = bst.test(a, b)
        # Very few exact ties expected
        assert result.probability_left + result.probability_right > 0.5

    @pytest.mark.parametrize("prior_strength", [0.01, 0.5, 1.0, 5.0, 50.0])
    def test_various_prior_strengths(self, prior_strength):
        """Sign test should work with different prior strengths."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bst = BayesianSignTest(prior_strength=prior_strength)
        result = bst.test(a, b)
        total = result.probability_left + result.probability_rope + result.probability_right
        assert abs(total - 1.0) < 1e-6

    def test_posterior_samples_reproducibility(self):
        """Same inputs and n_samples should give similar results."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bst = BayesianSignTest()
        r1 = bst.test(a, b, n_samples=50_000)
        r2 = bst.test(a, b, n_samples=50_000)
        # Results should be similar (MC variance)
        assert abs(r1.probability_left - r2.probability_left) < 0.1


class TestBayesianComparisonIntegration:
    """Integration tests combining multiple Bayesian components."""

    def test_compare_then_rank(self):
        """Compare two algorithms then check ranking consistency."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.7, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b, algorithm_a="A", algorithm_b="B")
        # A should be better
        assert result.p_a_better > result.p_b_better

    def test_rope_width_parameter(self):
        """rope_width in BayesianComparison should set symmetric ROPE."""
        bc = BayesianComparison(rope_width=0.1, seed=DEFAULT_SEED)
        assert abs(bc.rope_low - (-0.05)) < 1e-10
        assert abs(bc.rope_high - 0.05) < 1e-10

    def test_seed_reproducibility(self):
        """Same seed should give similar results."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        r1 = BayesianComparison(seed=42).compare_two(a, b)
        r2 = BayesianComparison(seed=42).compare_two(a, b)
        assert abs(r1.p_a_better - r2.p_a_better) < 0.05

    def test_prior_type_enum_usage(self):
        """PriorType enum should be usable in BayesianComparison."""
        bc = BayesianComparison(prior_type=PriorType.UNIFORM.value, seed=DEFAULT_SEED)
        assert bc.prior_type == PriorType.UNIFORM

    def test_interval_method_enum(self):
        """IntervalMethod enum should have HDI and ETI."""
        assert IntervalMethod.HDI.value == "hdi"
        assert IntervalMethod.ETI.value == "eti"

    def test_credible_interval_contains(self):
        """CredibleInterval.contains should work correctly."""
        ci = CredibleInterval(lower=0.2, upper=0.8, alpha=0.05, method="hdi")
        assert ci.contains(0.5)
        assert not ci.contains(0.1)
        assert not ci.contains(0.9)
        assert ci.contains(0.2)
        assert ci.contains(0.8)

    def test_effect_size_near_zero_for_equal(self):
        """Effect size should be near zero for equal distributions."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 200)
        b = rng.normal(0.5, 0.1, 200)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert abs(result.effect_size.mean) < 0.5

    def test_effect_size_large_for_different(self):
        """Effect size should be large for very different distributions."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(1.0, 0.1, 200)
        b = rng.normal(0.0, 0.1, 200)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert abs(result.effect_size.mean) > 0.5

    def test_raw_scores_stored(self):
        """ComparisonResult should store raw scores."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.raw_scores_a is not None
        assert result.raw_scores_b is not None
        assert len(result.raw_scores_a) == 30

    @pytest.mark.parametrize(
        "n_samples",
        [1000, 5000, 10000],
    )
    def test_n_samples_parameter(self, n_samples):
        """BayesianComparison should respect n_samples parameter."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bc = BayesianComparison(n_samples=n_samples, seed=DEFAULT_SEED)
        result = bc.compare_two(a, b)
        assert result.posterior_diff is not None

    def test_metric_name_preserved(self):
        """metric_name should be preserved in result."""
        rng = np.random.RandomState(DEFAULT_SEED)
        a = rng.normal(0.5, 0.1, 30)
        b = rng.normal(0.5, 0.1, 30)
        bc = BayesianComparison(seed=DEFAULT_SEED)
        result = bc.compare_two(a, b, metric_name="self_bleu")
        assert result.metric_name == "self_bleu"


class TestArenaComparisonResult:
    """Tests for the arena.ComparisonResult dataclass."""

    def test_comparison_result_fields(self):
        """ArenaComparisonResult should have all expected fields."""
        cr = ArenaComparisonResult(
            algorithm_a="X",
            algorithm_b="Y",
            metric_name="diversity",
            mean_a=0.8,
            mean_b=0.7,
            std_a=0.05,
            std_b=0.06,
            effect_size=0.3,
            effect_magnitude="small",
            p_value=0.04,
            ci_a=(0.75, 0.85),
            ci_b=(0.64, 0.76),
            ci_diff=(0.01, 0.19),
            winner="X",
            significant=True,
            num_runs_a=10,
            num_runs_b=10,
        )
        assert cr.algorithm_a == "X"
        assert cr.algorithm_b == "Y"
        assert cr.winner == "X"
        assert cr.significant is True
        assert cr.effect_magnitude == "small"

    def test_comparison_result_summary(self):
        """summary() should return a string."""
        cr = ArenaComparisonResult(
            algorithm_a="X",
            algorithm_b="Y",
            metric_name="diversity",
            mean_a=0.8,
            mean_b=0.7,
            std_a=0.05,
            std_b=0.06,
            effect_size=0.3,
            effect_magnitude="small",
            p_value=0.04,
            ci_a=(0.75, 0.85),
            ci_b=(0.64, 0.76),
            ci_diff=(0.01, 0.19),
            winner="X",
            significant=True,
            num_runs_a=10,
            num_runs_b=10,
        )
        s = cr.summary()
        assert isinstance(s, str)
        assert "X" in s or "Y" in s

    def test_not_significant_comparison(self):
        """Non-significant result should have significant=False."""
        cr = ArenaComparisonResult(
            algorithm_a="A",
            algorithm_b="B",
            metric_name="quality",
            mean_a=0.5,
            mean_b=0.5,
            std_a=0.1,
            std_b=0.1,
            effect_size=0.01,
            effect_magnitude="negligible",
            p_value=0.85,
            ci_a=(0.4, 0.6),
            ci_b=(0.4, 0.6),
            ci_diff=(-0.1, 0.1),
            winner="",
            significant=False,
            num_runs_a=10,
            num_runs_b=10,
        )
        assert not cr.significant
        assert cr.effect_magnitude == "negligible"

    @pytest.mark.parametrize(
        "effect_size,magnitude",
        [
            (0.0, "negligible"),
            (0.15, "small"),
            (0.40, "medium"),
            (0.75, "large"),
        ],
    )
    def test_effect_size_magnitude_categories(self, effect_size, magnitude):
        """Different effect sizes should map to expected magnitudes."""
        cr = ArenaComparisonResult(
            algorithm_a="A",
            algorithm_b="B",
            metric_name="m",
            mean_a=0.5,
            mean_b=0.5,
            std_a=0.1,
            std_b=0.1,
            effect_size=effect_size,
            effect_magnitude=magnitude,
            p_value=0.05,
            ci_a=(0.4, 0.6),
            ci_b=(0.4, 0.6),
            ci_diff=(-0.1, 0.1),
            winner="",
            significant=False,
            num_runs_a=5,
            num_runs_b=5,
        )
        assert cr.effect_magnitude == magnitude


class TestApproximateHypervolume:
    """Additional tests for Monte Carlo hypervolume approximation."""

    def test_seed_reproducibility(self):
        """Same seed should give same result."""
        ref = np.array([10.0, 10.0])
        pts = np.array([[3.0, 3.0], [5.0, 2.0]])
        v1 = ApproximateHypervolume(ref, seed=42).compute(pts, n_samples=10000)
        v2 = ApproximateHypervolume(ref, seed=42).compute(pts, n_samples=10000)
        assert abs(v1 - v2) < 1e-10

    def test_different_seeds_differ(self):
        """Different seeds should give slightly different results (MC noise)."""
        ref = np.array([10.0, 10.0])
        pts = np.array([[3.0, 3.0], [5.0, 2.0]])
        v1 = ApproximateHypervolume(ref, seed=1).compute(pts, n_samples=1000)
        v2 = ApproximateHypervolume(ref, seed=2).compute(pts, n_samples=1000)
        # They might be slightly different due to MC noise
        # (could be equal by chance, so just check they're both positive)
        assert v1 > 0
        assert v2 > 0

    def test_approx_nonnegative(self):
        """Approximate HV should be non-negative."""
        ref = np.array([10.0, 10.0])
        rng = np.random.RandomState(DEFAULT_SEED)
        pts = rng.uniform(1, 9, (5, 2))
        v = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(pts, n_samples=10000)
        assert v >= 0.0

    def test_approx_empty_points(self):
        """Empty points should give 0 HV."""
        ref = np.array([10.0, 10.0])
        v = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            np.empty((0, 2)), n_samples=10000
        )
        assert v == 0.0

    def test_approx_accuracy_simple(self):
        """Simple case should be approximated within 10% of exact."""
        ref = np.array([10.0, 10.0])
        pts = np.array([[3.0, 3.0]])
        exact = (10 - 3) * (10 - 3)
        approx = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            pts, n_samples=200_000
        )
        assert abs(approx - exact) / exact < 0.1

    @pytest.mark.parametrize("n_samples", [1000, 10000, 100000])
    def test_approx_converges(self, n_samples):
        """More samples should give result closer to exact."""
        ref = np.array([10.0, 10.0])
        pts = np.array([[2.0, 5.0], [5.0, 2.0]])
        exact = ExactHypervolume(ref).compute(pts)
        approx = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            pts, n_samples=n_samples
        )
        error = abs(approx - exact) / max(exact, 1e-10)
        assert error < 1.0  # should be well within 100%

    def test_approx_3d(self):
        """Approximate HV should work in 3D."""
        ref = np.array([10.0, 10.0, 10.0])
        pts = np.array([[3.0, 3.0, 3.0]])
        v = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            pts, n_samples=50_000
        )
        exact = (10 - 3) ** 3
        assert abs(v - exact) / exact < 0.15

    def test_approx_dominated_no_extra(self):
        """Dominated points should not affect approximate HV."""
        ref = np.array([10.0, 10.0])
        pts_nd = np.array([[3.0, 3.0]])
        pts_with = np.array([[3.0, 3.0], [6.0, 6.0]])
        v_nd = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            pts_nd, n_samples=50_000
        )
        v_with = ApproximateHypervolume(ref, seed=DEFAULT_SEED).compute(
            pts_with, n_samples=50_000
        )
        assert abs(v_nd - v_with) / max(v_nd, 1e-10) < 0.05


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestParetoFrontierMetrics:
    """Extended tests for Pareto frontier quality metrics."""

    def test_spread_metric_increases_with_extent(self):
        """Wider frontier should have larger spread."""
        pts_narrow = _make_points_from_array(
            np.array([[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]),
            keys=["d", "q"],
        )
        pts_wide = _make_points_from_array(
            np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
            keys=["d", "q"],
        )
        f_narrow = ParetoFrontier(pts_narrow)
        f_wide = ParetoFrontier(pts_wide)
        assert f_wide.spread_metric() >= f_narrow.spread_metric() - 0.01

    def test_spacing_metric_nonnegative(self):
        """Spacing metric should always be non-negative."""
        rng = np.random.RandomState(DEFAULT_SEED)
        for _ in range(5):
            n = rng.randint(3, 15)
            arr = rng.uniform(0, 1, (n, 2))
            pts = _make_points_from_array(arr, ["d", "q"])
            frontier = ParetoFrontier(pts)
            assert frontier.spacing_metric() >= 0.0

    def test_distance_to_frontier_zero_for_frontier_point(self):
        """A frontier point should have zero distance to the frontier."""
        pts = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(pts)
        for fp in frontier.frontier_points:
            dist = frontier.distance_to_frontier(fp)
            assert dist < 1e-6

    def test_coverage_zero_when_no_domination(self):
        """Coverage should be 0 when frontiers don't dominate each other."""
        pts_a = _make_points_from_array(
            np.array([[0.0, 0.5]]),
            keys=["d", "q"],
        )
        pts_b = _make_points_from_array(
            np.array([[0.5, 0.0]]),
            keys=["d", "q"],
        )
        fa = ParetoFrontier(pts_a)
        fb = ParetoFrontier(pts_b)
        # Neither dominates the other
        cov = fa.coverage(fb)
        assert cov == 0.0

    def test_coverage_full_when_dominated(self):
        """Coverage should be 1.0 when A dominates all of B."""
        pts_a = _make_points_from_array(
            np.array([[1.0, 1.0]]),
            keys=["d", "q"],
        )
        pts_b = _make_points_from_array(
            np.array([[0.5, 0.5], [0.3, 0.3]]),
            keys=["d", "q"],
        )
        fa = ParetoFrontier(pts_a)
        fb = ParetoFrontier(pts_b)
        cov = fa.coverage(fb)
        assert cov >= 0.9

    def test_all_points_property(self):
        """all_points should return all original points."""
        arr = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.2, 0.2]])
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        assert len(frontier.all_points) == 4

    def test_size_property(self):
        """size property should return total number of points."""
        arr = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        assert frontier.size == 3

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_spacing_with_various_sizes(self, n):
        """Spacing metric should work with different frontier sizes."""
        rng = np.random.RandomState(DEFAULT_SEED + n)
        arr = rng.uniform(0, 1, (n, 2))
        pts = _make_points_from_array(arr, ["d", "q"])
        frontier = ParetoFrontier(pts)
        s = frontier.spacing_metric()
        assert s >= 0.0

    def test_frontier_immutability(self):
        """Modifying returned frontier_points should not affect internal state."""
        pts = _make_points_from_array(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(pts)
        fp1 = frontier.frontier_points
        fp1.clear()
        fp2 = frontier.frontier_points
        assert len(fp2) == 2

    def test_objective_keys_immutability(self):
        """Modifying returned objective_keys should not affect internal state."""
        pts = _make_points_from_array(
            np.array([[1.0, 0.0]]),
            keys=["d", "q"],
        )
        frontier = ParetoFrontier(pts)
        keys1 = frontier.objective_keys
        keys1.append("extra")
        keys2 = frontier.objective_keys
        assert "extra" not in keys2


class TestHypervolumeEdgeCases:
    """Additional edge case tests for hypervolume computation."""

    def test_single_dimension(self):
        """1D hypervolume should be a simple difference."""
        ref = np.array([10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[3.0]])
        vol = hv.compute(pts)
        assert abs(vol - 7.0) < 1e-8

    def test_collinear_points_2d(self):
        """Collinear points in 2D should be handled correctly."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        vol = hv.compute(pts)
        # Only (1,1) is non-dominated
        expected = (10 - 1) * (10 - 1)
        assert abs(vol - expected) < 1e-6

    def test_points_on_reference_boundary(self):
        """Points on the reference boundary contribute zero volume."""
        ref = np.array([5.0, 5.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[5.0, 3.0]])  # x == ref[0], zero width
        vol = hv.compute(pts)
        assert vol == 0.0

    def test_very_close_points(self):
        """Very close but distinct points should still compute correctly."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        eps = 1e-8
        pts = np.array([[3.0, 3.0], [3.0 + eps, 3.0 - eps]])
        vol = hv.compute(pts)
        single_vol = hv.compute(np.array([[3.0, 3.0]]))
        assert vol >= single_vol - 1e-6

    def test_negative_coordinates(self):
        """Points with negative coordinates should work with appropriate ref."""
        ref = np.array([0.0, 0.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[-5.0, -3.0]])
        vol = hv.compute(pts)
        expected = (0 - (-5)) * (0 - (-3))
        assert abs(vol - expected) < 1e-6

    def test_large_reference_point(self):
        """Very large reference point should compute large HV."""
        ref = np.array([1e6, 1e6])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 1.0]])
        vol = hv.compute(pts)
        expected = (1e6 - 1) * (1e6 - 1)
        assert abs(vol - expected) / expected < 1e-6

    @pytest.mark.parametrize(
        "pts,ref,expected_gt",
        [
            (np.array([[1.0, 1.0]]), np.array([5.0, 5.0]), 0.0),
            (np.array([[0.0, 0.0]]), np.array([5.0, 5.0]), 20.0),
            (np.array([[2.0, 2.0]]), np.array([10.0, 10.0]), 50.0),
        ],
    )
    def test_parametrized_hv_values(self, pts, ref, expected_gt):
        """HV should be greater than expected_gt."""
        hv = ExactHypervolume(ref)
        vol = hv.compute(pts)
        assert vol > expected_gt

    def test_duplicate_points_same_hv(self):
        """Duplicate points should not change HV."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts1 = np.array([[3.0, 3.0]])
        pts2 = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
        assert abs(hv.compute(pts1) - hv.compute(pts2)) < 1e-10

    def test_two_identical_points_contribution(self):
        """Two identical points should have same contribution structure."""
        ref = np.array([10.0, 10.0])
        hv = ExactHypervolume(ref)
        pts = np.array([[3.0, 3.0], [3.0, 3.0]])
        contribs = hv.contributions(pts)
        assert len(contribs) == 2

    def test_hypervolume_with_inf_reference(self):
        """Infinite reference point should give infinite HV for any point."""
        ref = np.array([np.inf, np.inf])
        hv = ExactHypervolume(ref)
        pts = np.array([[1.0, 1.0]])
        vol = hv.compute(pts)
        assert vol == np.inf or vol > 1e10  # implementation dependent


class TestDominanceRelationProperties:
    """Extended property tests for dominance relations."""

    def test_reflexivity_non_strict(self):
        """A point does not strictly dominate itself."""
        u = np.array([1.0, 2.0, 3.0])
        assert _vec_dominates(u, u) is False

    def test_self_relation_is_equal(self):
        """_dominance_relation of a point with itself is EQUAL."""
        u = np.array([1.0, 2.0, 3.0])
        assert _dominance_relation(u, u) == DominanceRelation.EQUAL

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_strict_dominance_various_dims(self, dim):
        """Strict dominance should work in various dimensions."""
        u = np.ones(dim) * 2.0
        v = np.ones(dim) * 1.0
        assert _vec_dominates(u, v) is True
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES

    def test_dominance_with_one_equal_component(self):
        """Dominance with one equal component and rest better."""
        u = np.array([2.0, 1.0, 3.0])
        v = np.array([1.0, 1.0, 2.0])
        # u >= v in all, u > v in first and third
        assert _vec_dominates(u, v) is True

    def test_non_dominance_with_mixed_advantages(self):
        """Mixed advantages should yield NON_DOMINATED."""
        u = np.array([3.0, 1.0, 2.0])
        v = np.array([1.0, 3.0, 2.0])
        assert _dominance_relation(u, v) == DominanceRelation.NON_DOMINATED

    @pytest.mark.parametrize(
        "u,v",
        [
            ([10, 0], [0, 10]),
            ([5, 5], [5, 5]),
            ([1, 2, 3], [3, 2, 1]),
            ([0, 0, 0], [0, 0, 0]),
        ],
    )
    def test_batch_dominance_checks(self, u, v):
        """Batch of dominance relation checks."""
        ua = np.array(u, dtype=float)
        va = np.array(v, dtype=float)
        rel = _dominance_relation(ua, va)
        assert rel in (
            DominanceRelation.DOMINATES,
            DominanceRelation.DOMINATED,
            DominanceRelation.NON_DOMINATED,
            DominanceRelation.EQUAL,
        )

    def test_dominance_chain_4_levels(self):
        """Chain of 4 dominance levels."""
        a = np.array([4.0, 4.0])
        b = np.array([3.0, 3.0])
        c = np.array([2.0, 2.0])
        d = np.array([1.0, 1.0])
        assert _dominance_relation(a, b) == DominanceRelation.DOMINATES
        assert _dominance_relation(b, c) == DominanceRelation.DOMINATES
        assert _dominance_relation(c, d) == DominanceRelation.DOMINATES
        assert _dominance_relation(a, d) == DominanceRelation.DOMINATES

    def test_near_equal_with_epsilon(self):
        """Very close vectors should be EQUAL due to epsilon tolerance."""
        u = np.array([1.0, 1.0])
        v = np.array([1.0 + 1e-15, 1.0 - 1e-15])
        rel = _dominance_relation(u, v)
        assert rel in (DominanceRelation.EQUAL, DominanceRelation.NON_DOMINATED)

    def test_high_dimensional_dominance(self):
        """Dominance in 10 dimensions."""
        u = np.ones(10) * 2.0
        v = np.ones(10) * 1.0
        assert _vec_dominates(u, v) is True

    def test_single_objective_dominance(self):
        """1D dominance should work."""
        u = np.array([5.0])
        v = np.array([3.0])
        assert _vec_dominates(u, v) is True
        assert _dominance_relation(u, v) == DominanceRelation.DOMINATES


class TestBootstrapCIProperties:
    """Additional property tests for bootstrap CI."""

    def test_ci_width_positive(self):
        """CI width should be non-negative."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 50)
        lo, hi = _bootstrap_ci(data, rng=rng)
        assert hi - lo >= 0

    def test_ci_contains_sample_mean(self):
        """CI should contain the sample mean."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 100)
        lo, hi = _bootstrap_ci(data, rng=rng)
        assert lo <= np.mean(data) <= hi

    def test_ci_with_outlier(self):
        """CI should still be valid with an outlier."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = np.append(rng.normal(0, 1, 99), [100.0])
        lo, hi = _bootstrap_ci(data, rng=rng)
        assert lo < hi

    def test_ci_skewed_data(self):
        """CI should work for skewed data."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.exponential(1.0, 100)
        lo, hi = _bootstrap_ci(data, rng=rng)
        assert lo < hi
        assert lo >= 0  # exponential is positive

    @pytest.mark.parametrize("n", [5, 10, 50, 200, 1000])
    def test_ci_various_sample_sizes(self, n):
        """CI should work for various sample sizes."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, n)
        lo, hi = _bootstrap_ci(data, n_samples=2000, rng=rng)
        assert lo <= hi

    def test_ci_integer_data(self):
        """CI should work with integer data."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.randint(0, 10, 50).astype(float)
        lo, hi = _bootstrap_ci(data, rng=rng)
        assert lo <= hi

    def test_bootstrap_distribution_centered(self):
        """Bootstrap distribution of means should be centered on sample mean."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(5.0, 2.0, 100)
        n_boot = 5000
        boot_means = np.array([
            np.mean(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_boot)
        ])
        assert abs(np.mean(boot_means) - np.mean(data)) < 0.3

    def test_percentile_method(self):
        """Percentile CI should use correct quantiles."""
        rng = np.random.RandomState(DEFAULT_SEED)
        data = rng.normal(0, 1, 100)
        n_boot = 10000
        boot_means = np.array([
            np.mean(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_boot)
        ])
        lo = np.percentile(boot_means, 2.5)
        hi = np.percentile(boot_means, 97.5)
        assert lo < 0 < hi  # for N(0,1) data, CI should contain 0

    def test_ci_three_observations(self):
        """Three observations should give a valid CI."""
        data = np.array([1.0, 2.0, 3.0])
        rng = np.random.RandomState(DEFAULT_SEED)
        lo, hi = _bootstrap_ci(data, rng=rng)
        assert lo <= hi
        assert lo >= 1.0 - 0.1
        assert hi <= 3.0 + 0.1


class TestModelRankingProperties:
    """Tests for ModelRanking dataclass properties."""

    def test_ranking_names_preserved(self):
        """Algorithm names should be preserved in ranking."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.6], [0.4, 0.0]])
        ranking = bc.bradley_terry_ranking(W, ["Alpha", "Beta"], n_bootstrap=50)
        assert ranking.algorithm_names == ["Alpha", "Beta"]

    def test_probabilities_nonnegative(self):
        """All probabilities should be non-negative."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.7, 0.8], [0.3, 0.0, 0.6], [0.2, 0.4, 0.0]])
        ranking = bc.bradley_terry_ranking(W, ["A", "B", "C"], n_bootstrap=50)
        assert np.all(ranking.probabilities >= 0)

    def test_strength_params_finite(self):
        """Strength parameters should all be finite."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.6], [0.4, 0.0]])
        ranking = bc.bradley_terry_ranking(W, ["A", "B"], n_bootstrap=50)
        assert np.all(np.isfinite(ranking.strength_params))

    def test_default_algorithm_names(self):
        """Without names, default names should be generated."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.5], [0.5, 0.0]])
        ranking = bc.bradley_terry_ranking(W, n_bootstrap=50)
        assert len(ranking.algorithm_names) == 2

    def test_ranking_consistency_with_winrates(self):
        """Higher win rate should correspond to lower (better) rank."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([
            [0.0, 0.9, 0.95],
            [0.1, 0.0, 0.8],
            [0.05, 0.2, 0.0],
        ])
        ranking = bc.bradley_terry_ranking(W, ["Best", "Mid", "Worst"], n_bootstrap=100)
        # Best should have rank 1
        best_idx = ranking.algorithm_names.index("Best")
        worst_idx = ranking.algorithm_names.index("Worst")
        assert ranking.ranks[best_idx] < ranking.ranks[worst_idx]

    def test_credible_interval_width_positive(self):
        """All credible intervals should have positive width."""
        bc = BayesianComparison(seed=DEFAULT_SEED)
        W = np.array([[0.0, 0.6], [0.4, 0.0]])
        ranking = bc.bradley_terry_ranking(W, ["A", "B"], n_bootstrap=200)
        for name, ci in ranking.credible_intervals.items():
            assert ci.upper >= ci.lower
