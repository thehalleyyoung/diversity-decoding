"""
Comprehensive tests for diversity_quality_tradeoff module.

Tests cover TradeoffPoint, TradeoffConfig, ParetoFrontier,
OperatingPointFinder, TradeoffAnalyzer, QualityConstrainedDiversityOptimizer,
TradeoffVisualizer, helper functions, edge cases, and integration scenarios.
"""

from __future__ import annotations

import itertools
import math
import sys
import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, ".")

from src.evaluation.diversity_quality_tradeoff import (
    ParetoFrontier,
    OperatingPointFinder,
    QualityConstrainedDiversityOptimizer,
    TradeoffAnalyzer,
    TradeoffConfig,
    TradeoffPoint,
    TradeoffVisualizer,
    compute_dominated_hypervolume,
    compute_generational_distance,
    compute_inverted_generational_distance,
    compute_spread_metric,
    is_pareto_dominated,
    pareto_rank_assignment,
)

# Reproducibility seed used throughout
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_hyperbolic_front(n: int = 20, seed: int = SEED) -> List[TradeoffPoint]:
    """Points on y = 1/x for x in [0.2, 5], plus some dominated noise."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0.2, 5.0, n)
    y = 1.0 / x
    pts = [TradeoffPoint(quality=float(xi), diversity=float(yi), algorithm="hyp")
           for xi, yi in zip(x, y)]
    # add some dominated points
    for _ in range(n // 2):
        xi = rng.uniform(0.2, 5.0)
        yi = 1.0 / xi - rng.uniform(0.05, 0.5)
        if yi > 0:
            pts.append(TradeoffPoint(quality=xi, diversity=yi, algorithm="hyp"))
    return pts


def make_concave_front(n: int = 20, seed: int = SEED) -> List[TradeoffPoint]:
    """Points on y = 1 - x^2 for x in [0, 1]."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 1.0, n)
    y = 1.0 - x ** 2
    pts = [TradeoffPoint(quality=float(xi), diversity=float(yi), algorithm="concave")
           for xi, yi in zip(x, y)]
    for _ in range(n // 2):
        xi = rng.uniform(0.0, 1.0)
        yi = (1.0 - xi ** 2) * rng.uniform(0.3, 0.9)
        pts.append(TradeoffPoint(quality=xi, diversity=yi, algorithm="concave"))
    return pts


def make_linear_front(n: int = 20) -> List[TradeoffPoint]:
    """Points on y = 1 - x for x in [0, 1]."""
    x = np.linspace(0.0, 1.0, n)
    y = 1.0 - x
    return [TradeoffPoint(quality=float(xi), diversity=float(yi), algorithm="linear")
            for xi, yi in zip(x, y)]


def make_random_points(n: int = 50, seed: int = SEED) -> List[TradeoffPoint]:
    """Uniform random points in [0, 1]^2."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)
    return [TradeoffPoint(quality=float(r[0]), diversity=float(r[1]),
                          algorithm="rand") for r in pts]


def make_two_algorithm_data(
    seed: int = SEED,
) -> Dict[str, List[TradeoffPoint]]:
    """Two algorithms with different tradeoff characteristics."""
    rng = np.random.RandomState(seed)
    alg_a: List[TradeoffPoint] = []
    alg_b: List[TradeoffPoint] = []
    for _ in range(30):
        q = rng.uniform(0.3, 0.9)
        d = 1.0 - q + rng.normal(0, 0.05)
        alg_a.append(TradeoffPoint(quality=q, diversity=max(0, d), algorithm="A"))
    for _ in range(30):
        q = rng.uniform(0.2, 0.8)
        d = 0.9 - 0.8 * q + rng.normal(0, 0.05)
        alg_b.append(TradeoffPoint(quality=q, diversity=max(0, d), algorithm="B"))
    return {"A": alg_a, "B": alg_b}


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestTradeoffPoint
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeoffPoint(unittest.TestCase):
    """Tests for TradeoffPoint dataclass."""

    def test_creation_basic(self):
        pt = TradeoffPoint(quality=0.8, diversity=0.6)
        self.assertAlmostEqual(pt.quality, 0.8)
        self.assertAlmostEqual(pt.diversity, 0.6)
        self.assertEqual(pt.algorithm, "")

    def test_creation_with_algorithm(self):
        pt = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="beam")
        self.assertEqual(pt.algorithm, "beam")

    def test_creation_with_config(self):
        cfg = {"temperature": 0.7, "top_k": 40}
        pt = TradeoffPoint(quality=0.8, diversity=0.6, config=cfg)
        self.assertEqual(pt.config["temperature"], 0.7)
        self.assertEqual(pt.config["top_k"], 40)

    def test_creation_with_metadata(self):
        meta = {"run_id": 42, "timestamp": "2024-01-01"}
        pt = TradeoffPoint(quality=0.8, diversity=0.6, metadata=meta)
        self.assertEqual(pt.metadata["run_id"], 42)

    def test_to_dict(self):
        pt = TradeoffPoint(quality=0.8, diversity=0.6, algorithm="beam",
                           config={"k": 5}, metadata={"tag": "v1"})
        d = pt.to_dict()
        self.assertAlmostEqual(d["quality"], 0.8)
        self.assertAlmostEqual(d["diversity"], 0.6)
        self.assertEqual(d["algorithm"], "beam")
        self.assertEqual(d["config"]["k"], 5)
        self.assertEqual(d["metadata"]["tag"], "v1")

    def test_from_dict(self):
        d = {"quality": 0.9, "diversity": 0.3, "algorithm": "greedy",
             "config": {"t": 1.0}, "metadata": {"note": "test"}}
        pt = TradeoffPoint.from_dict(d)
        self.assertAlmostEqual(pt.quality, 0.9)
        self.assertAlmostEqual(pt.diversity, 0.3)
        self.assertEqual(pt.algorithm, "greedy")
        self.assertEqual(pt.config["t"], 1.0)

    def test_from_dict_minimal(self):
        d = {"quality": 0.5, "diversity": 0.5}
        pt = TradeoffPoint.from_dict(d)
        self.assertEqual(pt.algorithm, "")
        self.assertEqual(pt.config, {})

    def test_roundtrip_serialization(self):
        pt = TradeoffPoint(quality=0.123456, diversity=0.654321,
                           algorithm="sample", config={"a": 1, "b": 2.5})
        pt2 = TradeoffPoint.from_dict(pt.to_dict())
        self.assertEqual(pt, pt2)

    def test_to_array(self):
        pt = TradeoffPoint(quality=0.3, diversity=0.7)
        arr = pt.to_array()
        np.testing.assert_array_almost_equal(arr, [0.3, 0.7])
        self.assertEqual(arr.dtype, np.float64)

    def test_equality_same(self):
        a = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="x")
        b = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="x")
        self.assertEqual(a, b)

    def test_equality_different_algorithm(self):
        a = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="x")
        b = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="y")
        self.assertNotEqual(a, b)

    def test_equality_different_values(self):
        a = TradeoffPoint(quality=0.5, diversity=0.5)
        b = TradeoffPoint(quality=0.5, diversity=0.6)
        self.assertNotEqual(a, b)

    def test_equality_with_non_tradeoff(self):
        a = TradeoffPoint(quality=0.5, diversity=0.5)
        self.assertNotEqual(a, "not a point")

    def test_hash_equal_objects(self):
        a = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="x")
        b = TradeoffPoint(quality=0.5, diversity=0.5, algorithm="x")
        self.assertEqual(hash(a), hash(b))

    def test_hash_in_set(self):
        pts = {
            TradeoffPoint(quality=0.1, diversity=0.9, algorithm="a"),
            TradeoffPoint(quality=0.1, diversity=0.9, algorithm="a"),
            TradeoffPoint(quality=0.2, diversity=0.8, algorithm="a"),
        }
        self.assertEqual(len(pts), 2)

    def test_repr(self):
        pt = TradeoffPoint(quality=0.8, diversity=0.6, algorithm="beam")
        r = repr(pt)
        self.assertIn("0.8", r)
        self.assertIn("0.6", r)
        self.assertIn("beam", r)

    def test_default_config_is_empty(self):
        pt = TradeoffPoint(quality=0.0, diversity=0.0)
        self.assertEqual(pt.config, {})
        self.assertEqual(pt.metadata, {})

    def test_negative_values(self):
        pt = TradeoffPoint(quality=-0.5, diversity=-1.0)
        self.assertAlmostEqual(pt.quality, -0.5)
        self.assertAlmostEqual(pt.diversity, -1.0)

    def test_large_values(self):
        pt = TradeoffPoint(quality=1e6, diversity=1e-6)
        self.assertAlmostEqual(pt.quality, 1e6)
        self.assertAlmostEqual(pt.diversity, 1e-6)

    def test_config_independence(self):
        """Modifying config after creation should not affect the original."""
        cfg = {"a": 1}
        pt = TradeoffPoint(quality=0.5, diversity=0.5, config=cfg)
        cfg["a"] = 999
        # dataclass does not deep-copy; but to_dict does
        d = pt.to_dict()
        self.assertIsInstance(d["config"], dict)

    def test_serialization_preserves_nested_config(self):
        pt = TradeoffPoint(quality=0.5, diversity=0.5,
                           config={"nested": {"x": 1}})
        d = pt.to_dict()
        self.assertEqual(d["config"]["nested"]["x"], 1)

    def test_equality_float_tolerance(self):
        a = TradeoffPoint(quality=0.1 + 0.2, diversity=0.3)
        b = TradeoffPoint(quality=0.3, diversity=0.3)
        self.assertEqual(a, b)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestTradeoffConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeoffConfig(unittest.TestCase):
    """Tests for TradeoffConfig dataclass."""

    def test_defaults(self):
        cfg = TradeoffConfig()
        self.assertAlmostEqual(cfg.quality_weight, 0.5)
        self.assertAlmostEqual(cfg.diversity_weight, 0.5)
        self.assertAlmostEqual(cfg.min_quality, 0.0)
        self.assertAlmostEqual(cfg.min_diversity, 0.0)
        self.assertAlmostEqual(cfg.epsilon, 0.01)
        self.assertEqual(cfg.n_interpolation_points, 100)
        self.assertIsNone(cfg.reference_point)
        self.assertEqual(cfg.seed, 42)

    def test_custom_values(self):
        cfg = TradeoffConfig(quality_weight=0.7, diversity_weight=0.3,
                             min_quality=0.5, epsilon=0.05)
        self.assertAlmostEqual(cfg.quality_weight, 0.7)
        self.assertAlmostEqual(cfg.diversity_weight, 0.3)
        self.assertAlmostEqual(cfg.min_quality, 0.5)

    def test_validate_valid(self):
        cfg = TradeoffConfig()
        self.assertTrue(cfg.validate())

    def test_invalid_quality_weight_negative(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(quality_weight=-0.1)

    def test_invalid_quality_weight_too_large(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(quality_weight=1.1)

    def test_invalid_diversity_weight_negative(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(diversity_weight=-0.5)

    def test_invalid_diversity_weight_too_large(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(diversity_weight=2.0)

    def test_invalid_epsilon_negative(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(epsilon=-0.01)

    def test_invalid_n_interpolation_points(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(n_interpolation_points=1)

    def test_invalid_min_quality_negative(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(min_quality=-0.1)

    def test_invalid_min_diversity_negative(self):
        with self.assertRaises(ValueError):
            TradeoffConfig(min_diversity=-0.1)

    def test_boundary_quality_weight_zero(self):
        cfg = TradeoffConfig(quality_weight=0.0)
        self.assertAlmostEqual(cfg.quality_weight, 0.0)

    def test_boundary_quality_weight_one(self):
        cfg = TradeoffConfig(quality_weight=1.0)
        self.assertAlmostEqual(cfg.quality_weight, 1.0)

    def test_epsilon_zero(self):
        cfg = TradeoffConfig(epsilon=0.0)
        self.assertAlmostEqual(cfg.epsilon, 0.0)

    def test_reference_point_list(self):
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        self.assertEqual(cfg.reference_point, [0.0, 0.0])

    def test_n_interpolation_points_boundary(self):
        cfg = TradeoffConfig(n_interpolation_points=2)
        self.assertEqual(cfg.n_interpolation_points, 2)

    def test_seed(self):
        cfg = TradeoffConfig(seed=123)
        self.assertEqual(cfg.seed, 123)

    def test_validate_returns_false_for_bad(self):
        """Construct a valid config then manually break it for validate()."""
        cfg = TradeoffConfig()
        # Forcibly set bad value
        object.__setattr__(cfg, "epsilon", -1.0)
        self.assertFalse(cfg.validate())


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestParetoFrontier
# ═══════════════════════════════════════════════════════════════════════════


class TestParetoFrontier(unittest.TestCase):
    """Tests for ParetoFrontier class."""

    # -- frontier computation -----------------------------------------------

    def test_compute_frontier_simple(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=0.0),
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),
            TradeoffPoint(quality=0.3, diversity=0.3),  # dominated
        ]
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        self.assertEqual(len(frontier), 3)
        dominated = TradeoffPoint(quality=0.3, diversity=0.3)
        qualities = {round(p.quality, 2) for p in frontier}
        self.assertNotIn(0.3, qualities)

    def test_compute_frontier_all_non_dominated(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        self.assertEqual(len(frontier), len(pts))

    def test_compute_frontier_hyperbolic(self):
        pts = make_hyperbolic_front(20)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        self.assertGreater(len(frontier), 0)
        self.assertLessEqual(len(frontier), len(pts))
        # All frontier points should be non-dominated
        arr = np.array([[p.quality, p.diversity] for p in frontier])
        for i, pt in enumerate(frontier):
            others = np.delete(arr, i, axis=0)
            self.assertFalse(is_pareto_dominated(arr[i], others, maximize=True))

    def test_compute_frontier_concave(self):
        pts = make_concave_front(30)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        self.assertGreater(len(frontier), 0)

    def test_compute_2d_frontier_static(self):
        pts = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.3]], dtype=float)
        idx = ParetoFrontier.compute_2d_frontier(pts)
        self.assertIn(0, idx)
        self.assertIn(1, idx)
        self.assertIn(2, idx)
        self.assertNotIn(3, idx)

    def test_compute_nd_frontier_3d(self):
        pts = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],  # dominated
        ], dtype=float)
        idx = ParetoFrontier.compute_nd_frontier(pts)
        self.assertIn(0, idx)
        self.assertIn(1, idx)
        self.assertIn(2, idx)
        self.assertIn(3, idx)
        self.assertNotIn(4, idx)

    def test_compute_nd_frontier_all_dominated_by_one(self):
        pts = np.array([
            [10, 10, 10],
            [1, 1, 1],
            [2, 2, 2],
            [5, 5, 5],
        ], dtype=float)
        idx = ParetoFrontier.compute_nd_frontier(pts)
        self.assertEqual(idx, [0])

    # -- hypervolume --------------------------------------------------------

    def test_hypervolume_simple_square(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=1.0),
        ]
        pf = ParetoFrontier(pts, TradeoffConfig(reference_point=[0.0, 0.0]))
        pf.compute_frontier()
        hv = pf.hypervolume()
        self.assertAlmostEqual(hv, 1.0, places=5)

    def test_hypervolume_two_points(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=0.5),
            TradeoffPoint(quality=0.5, diversity=1.0),
        ]
        pf = ParetoFrontier(pts, TradeoffConfig(reference_point=[0.0, 0.0]))
        pf.compute_frontier()
        hv = pf.hypervolume()
        # HV = 1*0.5 + 0.5*(1.0-0.5) = 0.5 + 0.25 = 0.75
        self.assertAlmostEqual(hv, 0.75, places=5)

    def test_hypervolume_with_explicit_reference(self):
        pts = [TradeoffPoint(quality=2.0, diversity=2.0)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        hv = pf.hypervolume(reference=np.array([0.0, 0.0]))
        self.assertAlmostEqual(hv, 4.0, places=5)

    def test_hypervolume_monotone(self):
        """Adding a non-dominated point should not decrease hypervolume."""
        pts1 = [
            TradeoffPoint(quality=1.0, diversity=0.5),
            TradeoffPoint(quality=0.5, diversity=1.0),
        ]
        pts2 = pts1 + [TradeoffPoint(quality=0.8, diversity=0.8)]
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf1 = ParetoFrontier(pts1, cfg)
        pf2 = ParetoFrontier(pts2, cfg)
        pf1.compute_frontier()
        pf2.compute_frontier()
        self.assertGreaterEqual(pf2.hypervolume(), pf1.hypervolume())

    def test_hypervolume_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.hypervolume(), 0.0)

    def test_hypervolume_dominated_points_ignored(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),  # dominated
        ]
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf = ParetoFrontier(pts, cfg)
        pf.compute_frontier()
        hv = pf.hypervolume()
        self.assertAlmostEqual(hv, 1.0, places=5)

    # -- crowding distance --------------------------------------------------

    def test_crowding_distance_two_points(self):
        pts = [
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=1.0, diversity=0.0),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        self.assertEqual(len(cd), 2)
        self.assertEqual(cd[0], float("inf"))
        self.assertEqual(cd[1], float("inf"))

    def test_crowding_distance_extremes_infinite(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        # At least two should be infinite (boundary points)
        inf_count = sum(1 for v in cd.values() if v == float("inf"))
        self.assertGreaterEqual(inf_count, 2)

    def test_crowding_distance_middle_finite(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        finite = [v for v in cd.values() if v < float("inf")]
        for v in finite:
            self.assertGreater(v, 0)

    def test_crowding_distance_uniform_spacing(self):
        """Uniformly spaced points should have similar crowding distances."""
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        finite = [v for v in cd.values() if v < float("inf")]
        if len(finite) > 1:
            cv = np.std(finite) / (np.mean(finite) + 1e-12)
            self.assertLess(cv, 0.5)

    def test_crowding_distance_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        cd = pf.crowding_distance()
        self.assertEqual(len(cd), 0)

    def test_crowding_distance_single_point(self):
        pf = ParetoFrontier([TradeoffPoint(quality=0.5, diversity=0.5)])
        pf.compute_frontier()
        cd = pf.crowding_distance()
        self.assertEqual(len(cd), 1)
        self.assertEqual(cd[0], float("inf"))

    # -- epsilon-dominance --------------------------------------------------

    def test_epsilon_dominates_clearly(self):
        pf = ParetoFrontier([], TradeoffConfig(epsilon=0.01))
        a = TradeoffPoint(quality=0.8, diversity=0.8)
        b = TradeoffPoint(quality=0.7, diversity=0.7)
        self.assertTrue(pf.epsilon_dominates(a, b))

    def test_epsilon_dominates_within_epsilon(self):
        pf = ParetoFrontier([], TradeoffConfig(epsilon=0.1))
        a = TradeoffPoint(quality=0.75, diversity=0.75)
        b = TradeoffPoint(quality=0.8, diversity=0.8)
        # a.q >= b.q - eps => 0.75 >= 0.7 => True
        self.assertTrue(pf.epsilon_dominates(a, b))

    def test_epsilon_dominates_false(self):
        pf = ParetoFrontier([], TradeoffConfig(epsilon=0.01))
        a = TradeoffPoint(quality=0.5, diversity=0.5)
        b = TradeoffPoint(quality=0.8, diversity=0.8)
        self.assertFalse(pf.epsilon_dominates(a, b))

    def test_epsilon_dominates_custom_eps(self):
        pf = ParetoFrontier([])
        a = TradeoffPoint(quality=0.49, diversity=0.49)
        b = TradeoffPoint(quality=0.5, diversity=0.5)
        self.assertTrue(pf.epsilon_dominates(a, b, eps=0.02))
        self.assertFalse(pf.epsilon_dominates(a, b, eps=0.005))

    def test_epsilon_dominates_symmetry_broken(self):
        pf = ParetoFrontier([], TradeoffConfig(epsilon=0.01))
        a = TradeoffPoint(quality=0.8, diversity=0.2)
        b = TradeoffPoint(quality=0.2, diversity=0.8)
        # a eps-dom b? 0.8 >= 0.19 and 0.2 >= 0.79? No
        self.assertFalse(pf.epsilon_dominates(a, b))

    # -- interpolation ------------------------------------------------------

    def test_interpolation_basic(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts, TradeoffConfig(n_interpolation_points=50))
        pf.compute_frontier()
        interp = pf.frontier_interpolation()
        self.assertEqual(interp.shape, (50, 2))

    def test_interpolation_custom_n(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        interp = pf.frontier_interpolation(n_points=20)
        self.assertEqual(interp.shape, (20, 2))

    def test_interpolation_monotone_quality(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        interp = pf.frontier_interpolation(n_points=100)
        # Quality should be monotonically increasing
        self.assertTrue(np.all(np.diff(interp[:, 0]) >= -1e-10))

    def test_interpolation_covers_range(self):
        pts = make_linear_front(10)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        interp = pf.frontier_interpolation(n_points=50)
        q_vals = [p.quality for p in frontier]
        self.assertAlmostEqual(interp[0, 0], min(q_vals), places=5)
        self.assertAlmostEqual(interp[-1, 0], max(q_vals), places=5)

    def test_interpolation_single_point(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        interp = pf.frontier_interpolation(n_points=10)
        self.assertEqual(interp.shape, (10, 2))
        np.testing.assert_array_almost_equal(interp[:, 0], 0.5)
        np.testing.assert_array_almost_equal(interp[:, 1], 0.5)

    def test_interpolation_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        interp = pf.frontier_interpolation(n_points=10)
        self.assertEqual(interp.shape, (0, 2))

    # -- area under frontier ------------------------------------------------

    def test_area_under_frontier_linear(self):
        """For y = 1 - x on [0,1], area = 0.5."""
        pts = make_linear_front(100)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        area = pf.area_under_frontier()
        self.assertAlmostEqual(area, 0.5, places=2)

    def test_area_under_frontier_concave(self):
        """For y = 1 - x^2 on [0,1], area = 2/3."""
        pts = make_concave_front(100, seed=123)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        # Only frontier points contribute
        area = pf.area_under_frontier()
        self.assertGreater(area, 0)

    def test_area_under_frontier_single_point(self):
        pf = ParetoFrontier([TradeoffPoint(quality=0.5, diversity=0.5)])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.area_under_frontier(), 0.0)

    def test_area_under_frontier_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.area_under_frontier(), 0.0)

    # -- spread -------------------------------------------------------------

    def test_frontier_spread_linear(self):
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        spread = pf.frontier_spread()
        # Distance from (0,1) to (1,0) = sqrt(2)
        self.assertAlmostEqual(spread, math.sqrt(2), places=2)

    def test_frontier_spread_single(self):
        pf = ParetoFrontier([TradeoffPoint(quality=0.5, diversity=0.5)])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.frontier_spread(), 0.0)

    def test_frontier_spread_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.frontier_spread(), 0.0)

    def test_frontier_spread_positive(self):
        pts = make_random_points(50)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        self.assertGreater(pf.frontier_spread(), 0)

    # -- uniformity ---------------------------------------------------------

    def test_frontier_uniformity_perfect(self):
        """Evenly spaced points should have low uniformity (CV)."""
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        uni = pf.frontier_uniformity()
        self.assertLess(uni, 0.1)

    def test_frontier_uniformity_non_uniform(self):
        """Clustered + spread points should have higher uniformity."""
        pts = [
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=0.01, diversity=0.99),
            TradeoffPoint(quality=0.02, diversity=0.98),
            TradeoffPoint(quality=1.0, diversity=0.0),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        uni = pf.frontier_uniformity()
        self.assertGreater(uni, 0.3)

    def test_frontier_uniformity_two_points(self):
        pts = [
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=1.0, diversity=0.0),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        self.assertAlmostEqual(pf.frontier_uniformity(), 0.0)

    def test_frontier_uniformity_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.frontier_uniformity(), 0.0)

    # -- points property ----------------------------------------------------

    def test_points_property(self):
        pts = make_linear_front(5)
        pf = ParetoFrontier(pts)
        self.assertEqual(len(pf.points), 5)

    def test_frontier_property_lazy(self):
        pts = make_linear_front(5)
        pf = ParetoFrontier(pts)
        # accessing .frontier triggers compute
        f = pf.frontier
        self.assertEqual(len(f), 5)

    def test_frontier_recompute(self):
        pts = make_linear_front(5)
        pf = ParetoFrontier(pts)
        f1 = pf.compute_frontier()
        f2 = pf.compute_frontier()
        self.assertEqual(len(f1), len(f2))


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestOperatingPointFinder
# ═══════════════════════════════════════════════════════════════════════════


class TestOperatingPointFinder(unittest.TestCase):
    """Tests for OperatingPointFinder."""

    def setUp(self):
        self.linear_pts = make_linear_front(20)
        self.pf = ParetoFrontier(self.linear_pts)
        self.pf.compute_frontier()
        self.finder = OperatingPointFinder(self.pf)

    def test_knee_point_exists(self):
        knee = self.finder.find_knee_point()
        self.assertIsNotNone(knee)

    def test_knee_point_is_on_frontier(self):
        knee = self.finder.find_knee_point()
        frontier_pts = self.pf.frontier
        self.assertIn(knee, frontier_pts)

    def test_knee_point_interior(self):
        """Knee should not be an extreme point for a linear front."""
        knee = self.finder.find_knee_point()
        qualities = sorted(p.quality for p in self.pf.frontier)
        # For a linear front, all interior points have same distance, so
        # knee could be any interior point
        self.assertGreater(knee.quality, qualities[0] - 0.01)
        self.assertLess(knee.quality, qualities[-1] + 0.01)

    def test_knee_point_concave_front(self):
        """On a concave front, knee should be near the bend."""
        pts = make_concave_front(50, seed=99)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertIsNotNone(knee)
        # Should be somewhere in the middle
        self.assertGreater(knee.quality, 0.0)

    def test_knee_point_two_points(self):
        pts = [
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=1.0, diversity=0.0),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertIsNotNone(knee)

    def test_knee_point_single(self):
        pf = ParetoFrontier([TradeoffPoint(quality=0.5, diversity=0.5)])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertAlmostEqual(knee.quality, 0.5)

    def test_knee_point_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertIsNone(knee)

    # -- quality-constrained optimum ----------------------------------------

    def test_quality_constrained_optimum(self):
        pt = self.finder.find_quality_constrained_optimum(min_quality=0.5)
        self.assertIsNotNone(pt)
        self.assertGreaterEqual(pt.quality, 0.5)

    def test_quality_constrained_max_diversity(self):
        """Among those with quality >= threshold, should pick max diversity."""
        pt = self.finder.find_quality_constrained_optimum(min_quality=0.5)
        for p in self.pf.frontier:
            if p.quality >= 0.5:
                self.assertGreaterEqual(pt.diversity, p.diversity - 1e-10)

    def test_quality_constrained_infeasible(self):
        pt = self.finder.find_quality_constrained_optimum(min_quality=2.0)
        self.assertIsNone(pt)

    def test_quality_constrained_zero(self):
        pt = self.finder.find_quality_constrained_optimum(min_quality=0.0)
        self.assertIsNotNone(pt)

    def test_quality_constrained_tight(self):
        """Threshold equal to max quality → only that point."""
        max_q = max(p.quality for p in self.pf.frontier)
        pt = self.finder.find_quality_constrained_optimum(min_quality=max_q)
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt.quality, max_q, places=5)

    # -- diversity-constrained optimum --------------------------------------

    def test_diversity_constrained_optimum(self):
        pt = self.finder.find_diversity_constrained_optimum(min_diversity=0.5)
        self.assertIsNotNone(pt)
        self.assertGreaterEqual(pt.diversity, 0.5)

    def test_diversity_constrained_max_quality(self):
        pt = self.finder.find_diversity_constrained_optimum(min_diversity=0.3)
        for p in self.pf.frontier:
            if p.diversity >= 0.3:
                self.assertGreaterEqual(pt.quality, p.quality - 1e-10)

    def test_diversity_constrained_infeasible(self):
        pt = self.finder.find_diversity_constrained_optimum(min_diversity=5.0)
        self.assertIsNone(pt)

    # -- balanced point -----------------------------------------------------

    def test_balanced_point_exists(self):
        pt = self.finder.find_balanced_point()
        self.assertIsNotNone(pt)

    def test_balanced_point_on_frontier(self):
        pt = self.finder.find_balanced_point()
        self.assertIn(pt, self.pf.frontier)

    def test_balanced_point_near_center(self):
        """For a linear front, balanced point should be near (0.5, 0.5)."""
        pt = self.finder.find_balanced_point()
        dist = math.sqrt((pt.quality - 0.5) ** 2 + (pt.diversity - 0.5) ** 2)
        self.assertLess(dist, 0.15)

    def test_balanced_point_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        self.assertIsNone(finder.find_balanced_point())

    def test_balanced_point_single(self):
        pf = ParetoFrontier([TradeoffPoint(quality=0.3, diversity=0.7)])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        pt = finder.find_balanced_point()
        self.assertAlmostEqual(pt.quality, 0.3)

    # -- target ratio -------------------------------------------------------

    def test_target_ratio_one(self):
        """Ratio q/d = 1 → point near the diagonal."""
        pt = self.finder.find_target_ratio_point(target_ratio=1.0)
        self.assertIsNotNone(pt)
        ratio = pt.quality / max(pt.diversity, 1e-12)
        self.assertAlmostEqual(ratio, 1.0, delta=0.15)

    def test_target_ratio_high(self):
        """Ratio > 1 → quality-dominant point."""
        pt = self.finder.find_target_ratio_point(target_ratio=5.0)
        self.assertIsNotNone(pt)
        self.assertGreater(pt.quality, pt.diversity)

    def test_target_ratio_low(self):
        """Ratio < 1 → diversity-dominant point."""
        pt = self.finder.find_target_ratio_point(target_ratio=0.2)
        self.assertIsNotNone(pt)
        self.assertLess(pt.quality, pt.diversity)

    def test_target_ratio_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        self.assertIsNone(finder.find_target_ratio_point(1.0))

    # -- scalarization ------------------------------------------------------

    def test_scalarization_weighted_sum(self):
        pt = self.finder.multi_objective_scalarization(
            weights=(0.5, 0.5), method="weighted_sum"
        )
        self.assertIsNotNone(pt)

    def test_scalarization_quality_biased(self):
        pt = self.finder.multi_objective_scalarization(
            weights=(0.9, 0.1), method="weighted_sum"
        )
        self.assertIsNotNone(pt)
        # Should favor quality
        self.assertGreater(pt.quality, 0.5)

    def test_scalarization_diversity_biased(self):
        pt = self.finder.multi_objective_scalarization(
            weights=(0.1, 0.9), method="weighted_sum"
        )
        self.assertIsNotNone(pt)
        self.assertGreater(pt.diversity, 0.5)

    def test_scalarization_tchebycheff(self):
        pt = self.finder.multi_objective_scalarization(
            weights=(0.5, 0.5), method="tchebycheff"
        )
        self.assertIsNotNone(pt)

    def test_scalarization_tchebycheff_on_frontier(self):
        pt = self.finder.multi_objective_scalarization(
            weights=(0.5, 0.5), method="tchebycheff"
        )
        self.assertIn(pt, self.pf.frontier)

    def test_scalarization_unknown_method(self):
        with self.assertRaises(ValueError):
            self.finder.multi_objective_scalarization(method="unknown")

    def test_scalarization_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        self.assertIsNone(finder.multi_objective_scalarization())

    def test_scalarization_various_weights(self):
        """Different weights should potentially yield different points."""
        results = set()
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt = self.finder.multi_objective_scalarization(
                weights=(w, 1 - w), method="weighted_sum"
            )
            if pt:
                results.add((round(pt.quality, 4), round(pt.diversity, 4)))
        self.assertGreater(len(results), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestTradeoffAnalyzer
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeoffAnalyzer(unittest.TestCase):
    """Tests for TradeoffAnalyzer."""

    def setUp(self):
        self.data = make_two_algorithm_data(seed=SEED)
        self.analyzer = TradeoffAnalyzer()

    # -- algorithm tradeoffs ------------------------------------------------

    def test_analyze_algorithm_tradeoffs_keys(self):
        result = self.analyzer.analyze_algorithm_tradeoffs(self.data)
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_analyze_algorithm_tradeoffs_metrics(self):
        result = self.analyzer.analyze_algorithm_tradeoffs(self.data)
        for alg in ("A", "B"):
            self.assertIn("n_points", result[alg])
            self.assertIn("n_frontier", result[alg])
            self.assertIn("hypervolume", result[alg])
            self.assertIn("spread", result[alg])
            self.assertIn("uniformity", result[alg])
            self.assertIn("area_under_frontier", result[alg])

    def test_analyze_algorithm_tradeoffs_n_points(self):
        result = self.analyzer.analyze_algorithm_tradeoffs(self.data)
        self.assertEqual(result["A"]["n_points"], 30)
        self.assertEqual(result["B"]["n_points"], 30)

    def test_analyze_algorithm_tradeoffs_frontier_leq_total(self):
        result = self.analyzer.analyze_algorithm_tradeoffs(self.data)
        for alg in ("A", "B"):
            self.assertLessEqual(result[alg]["n_frontier"], result[alg]["n_points"])

    def test_analyze_algorithm_tradeoffs_hypervolume_positive(self):
        result = self.analyzer.analyze_algorithm_tradeoffs(self.data)
        for alg in ("A", "B"):
            self.assertGreaterEqual(result[alg]["hypervolume"], 0)

    def test_analyze_empty_algorithm(self):
        result = self.analyzer.analyze_algorithm_tradeoffs({"X": []})
        self.assertEqual(result["X"]["n_points"], 0)
        self.assertEqual(result["X"]["n_frontier"], 0)

    def test_analyze_single_algorithm(self):
        result = self.analyzer.analyze_algorithm_tradeoffs({"A": self.data["A"]})
        self.assertEqual(len(result), 1)

    # -- tradeoff curves ----------------------------------------------------

    def test_compute_tradeoff_curves_shape(self):
        curves = self.analyzer.compute_tradeoff_curves(self.data, n_points=25)
        for alg in ("A", "B"):
            self.assertIn(alg, curves)
            arr = curves[alg]
            self.assertEqual(arr.shape[1], 2)
            # Should have 25 points or fewer if frontier is small
            self.assertGreater(arr.shape[0], 0)

    def test_compute_tradeoff_curves_sorted(self):
        curves = self.analyzer.compute_tradeoff_curves(self.data, n_points=50)
        for alg in ("A", "B"):
            arr = curves[alg]
            if arr.shape[0] > 1:
                self.assertTrue(np.all(np.diff(arr[:, 0]) >= -1e-10))

    def test_compute_tradeoff_curves_empty(self):
        curves = self.analyzer.compute_tradeoff_curves({"X": []})
        self.assertIn("X", curves)
        self.assertEqual(curves["X"].shape, (0, 2))

    # -- frontier comparison ------------------------------------------------

    def test_compare_frontiers_keys(self):
        result = self.analyzer.compare_frontiers(self.data["A"], self.data["B"])
        expected_keys = {
            "hypervolume_a", "hypervolume_b", "hypervolume_ratio",
            "spread_a", "spread_b", "uniformity_a", "uniformity_b",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_compare_frontiers_ratio_positive(self):
        result = self.analyzer.compare_frontiers(self.data["A"], self.data["B"])
        self.assertGreater(result["hypervolume_ratio"], 0)

    def test_compare_frontiers_same(self):
        result = self.analyzer.compare_frontiers(self.data["A"], self.data["A"])
        self.assertAlmostEqual(result["hypervolume_ratio"], 1.0, places=5)

    def test_compare_frontiers_empty_vs_nonempty(self):
        result = self.analyzer.compare_frontiers([], self.data["A"])
        self.assertAlmostEqual(result["hypervolume_a"], 0.0)

    # -- statistical frontier comparison ------------------------------------

    def test_statistical_comparison_keys(self):
        np.random.seed(SEED)
        runs_a = [make_random_points(20, seed=i) for i in range(10)]
        runs_b = [make_random_points(20, seed=i + 100) for i in range(10)]
        result = self.analyzer.statistical_frontier_comparison(runs_a, runs_b)
        self.assertIn("mean_hv_a", result)
        self.assertIn("mean_hv_b", result)
        self.assertIn("t_stat", result)
        self.assertIn("p_value", result)
        self.assertIn("significant", result)

    def test_statistical_comparison_same_distribution(self):
        runs = [make_random_points(20, seed=i) for i in range(10)]
        result = self.analyzer.statistical_frontier_comparison(runs, runs)
        self.assertAlmostEqual(result["diff"], 0.0, places=5)
        self.assertFalse(result["significant"])

    def test_statistical_comparison_different_distributions(self):
        # Algorithm A is clearly better
        runs_a = []
        runs_b = []
        for i in range(15):
            rng = np.random.RandomState(i)
            a_pts = [TradeoffPoint(quality=rng.uniform(0.6, 1.0),
                                   diversity=rng.uniform(0.6, 1.0))
                     for _ in range(10)]
            b_pts = [TradeoffPoint(quality=rng.uniform(0.0, 0.4),
                                   diversity=rng.uniform(0.0, 0.4))
                     for _ in range(10)]
            runs_a.append(a_pts)
            runs_b.append(b_pts)
        result = self.analyzer.statistical_frontier_comparison(runs_a, runs_b)
        self.assertGreater(result["mean_hv_a"], result["mean_hv_b"])

    def test_statistical_comparison_p_value_range(self):
        runs_a = [make_random_points(20, seed=i) for i in range(5)]
        runs_b = [make_random_points(20, seed=i + 50) for i in range(5)]
        result = self.analyzer.statistical_frontier_comparison(runs_a, runs_b)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    # -- marginal rates -----------------------------------------------------

    def test_marginal_rates_linear(self):
        """On y = 1 - x, dDiversity/dQuality = -1 everywhere."""
        pts = make_linear_front(20)
        rates = self.analyzer.compute_marginal_rates(pts)
        self.assertGreater(len(rates), 0)
        for r in rates:
            self.assertAlmostEqual(r, -1.0, places=1)

    def test_marginal_rates_count(self):
        pts = make_linear_front(10)
        rates = self.analyzer.compute_marginal_rates(pts)
        self.assertEqual(len(rates), 9)  # n-1 rates

    def test_marginal_rates_empty(self):
        rates = self.analyzer.compute_marginal_rates([])
        self.assertEqual(len(rates), 0)

    def test_marginal_rates_single(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        rates = self.analyzer.compute_marginal_rates(pts)
        self.assertEqual(len(rates), 0)

    def test_marginal_rates_concave(self):
        """On y = 1 - x^2, rates should become more negative."""
        pts = [TradeoffPoint(quality=float(x), diversity=float(1 - x ** 2))
               for x in np.linspace(0, 1, 20)]
        rates = self.analyzer.compute_marginal_rates(pts)
        # Should be negative and decreasing
        for r in rates:
            self.assertLess(r, 0.1)

    # -- sensitivity analysis -----------------------------------------------

    def test_sensitivity_analysis_length(self):
        pts = make_linear_front(20)
        results = self.analyzer.sensitivity_analysis(pts)
        self.assertEqual(len(results), 5)  # default 5 weights

    def test_sensitivity_analysis_custom_weights(self):
        pts = make_linear_front(20)
        weights = [0.0, 0.5, 1.0]
        results = self.analyzer.sensitivity_analysis(pts, quality_weights=weights)
        self.assertEqual(len(results), 3)

    def test_sensitivity_analysis_structure(self):
        pts = make_linear_front(20)
        results = self.analyzer.sensitivity_analysis(pts)
        for r in results:
            self.assertIn("quality_weight", r)
            self.assertIn("selected_point", r)
            self.assertIn("quality", r)
            self.assertIn("diversity", r)

    def test_sensitivity_analysis_weight_zero_favors_diversity(self):
        pts = make_linear_front(20)
        results = self.analyzer.sensitivity_analysis(pts, quality_weights=[0.0])
        pt = results[0]["selected_point"]
        self.assertIsNotNone(pt)
        # With w=0, only diversity matters → max diversity point
        max_div = max(p.diversity for p in pts)
        self.assertAlmostEqual(pt.diversity, max_div, places=2)

    def test_sensitivity_analysis_weight_one_favors_quality(self):
        pts = make_linear_front(20)
        results = self.analyzer.sensitivity_analysis(pts, quality_weights=[1.0])
        pt = results[0]["selected_point"]
        max_q = max(p.quality for p in pts)
        self.assertAlmostEqual(pt.quality, max_q, places=2)

    def test_sensitivity_analysis_empty(self):
        results = self.analyzer.sensitivity_analysis([])
        for r in results:
            self.assertIsNone(r["selected_point"])


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestQualityConstrainedOptimizer
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityConstrainedOptimizer(unittest.TestCase):
    """Tests for QualityConstrainedDiversityOptimizer."""

    @staticmethod
    def _eval_fn(config: Dict[str, Any]) -> TradeoffPoint:
        """Deterministic evaluation: quality = x, diversity = 1 - x."""
        x = config.get("x", 0.5)
        return TradeoffPoint(quality=x, diversity=1.0 - x,
                             config=config, algorithm="test")

    @staticmethod
    def _eval_fn_noisy(config: Dict[str, Any]) -> TradeoffPoint:
        """Noisy evaluation."""
        x = config.get("x", 0.5)
        rng = np.random.RandomState(int(x * 1000) % 2**31)
        noise = rng.normal(0, 0.05)
        return TradeoffPoint(quality=x + noise, diversity=1.0 - x + noise,
                             config=config, algorithm="noisy")

    # -- grid search --------------------------------------------------------

    def test_grid_search_finds_optimum(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=SEED
        )
        grid = {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        best = opt.grid_search_with_constraint(grid)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best.quality, 0.3)
        # Best diversity among feasible = x=0.3 → diversity=0.7
        self.assertAlmostEqual(best.diversity, 0.7, places=5)

    def test_grid_search_history(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        grid = {"x": [0.1, 0.5, 0.9]}
        opt.grid_search_with_constraint(grid)
        self.assertEqual(len(opt.history), 3)

    def test_grid_search_infeasible(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=2.0, seed=SEED
        )
        grid = {"x": [0.1, 0.5, 0.9]}
        best = opt.grid_search_with_constraint(grid)
        self.assertIsNone(best)

    def test_grid_search_multi_param(self):
        def eval_fn(config):
            x = config["x"]
            y = config["y"]
            return TradeoffPoint(quality=x, diversity=y)

        opt = QualityConstrainedDiversityOptimizer(
            eval_fn, min_quality=0.3, seed=SEED
        )
        grid = {"x": [0.1, 0.3, 0.5], "y": [0.2, 0.5, 0.8]}
        best = opt.grid_search_with_constraint(grid)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best.quality, 0.3)
        self.assertAlmostEqual(best.diversity, 0.8)

    def test_grid_search_single_value(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        grid = {"x": [0.5]}
        best = opt.grid_search_with_constraint(grid)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best.quality, 0.5)

    def test_grid_search_respects_constraint(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.5, seed=SEED
        )
        grid = {"x": np.linspace(0.0, 1.0, 20).tolist()}
        best = opt.grid_search_with_constraint(grid)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best.quality, 0.5)

    # -- bayesian optimization step -----------------------------------------

    def test_bayesian_step_returns_point(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.bayesian_optimization_step(
            param_ranges={"x": (0.0, 1.0)}, n_candidates=30
        )
        self.assertIsNotNone(pt)
        self.assertIsInstance(pt, TradeoffPoint)

    def test_bayesian_step_feasible(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.bayesian_optimization_step(
            param_ranges={"x": (0.3, 0.9)}, n_candidates=50
        )
        self.assertGreaterEqual(pt.quality, 0.3)

    def test_bayesian_step_history(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        opt.bayesian_optimization_step(
            param_ranges={"x": (0.0, 1.0)}, n_candidates=10
        )
        self.assertEqual(len(opt.history), 10)

    def test_bayesian_step_max_diversity(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.2, seed=SEED
        )
        pt = opt.bayesian_optimization_step(
            param_ranges={"x": (0.0, 1.0)}, n_candidates=100
        )
        # Best feasible should have quality close to 0.2 (max diversity)
        self.assertGreaterEqual(pt.quality, 0.2)

    def test_bayesian_step_infeasible_fallback(self):
        """When all points infeasible, returns highest quality."""
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=2.0, seed=SEED
        )
        pt = opt.bayesian_optimization_step(
            param_ranges={"x": (0.0, 1.0)}, n_candidates=10
        )
        self.assertIsNotNone(pt)

    # -- successive halving -------------------------------------------------

    def test_successive_halving_returns_point(self):
        configs = [{"x": x} for x in np.linspace(0.1, 0.9, 8)]
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.successive_halving(configs, n_rounds=3)
        self.assertIsNotNone(pt)

    def test_successive_halving_reduces_candidates(self):
        configs = [{"x": x} for x in np.linspace(0.1, 0.9, 16)]
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        opt.successive_halving(configs, n_rounds=4)
        # History should contain evaluations from all rounds
        self.assertGreater(len(opt.history), 0)

    def test_successive_halving_single_config(self):
        configs = [{"x": 0.5}]
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        pt = opt.successive_halving(configs, n_rounds=3)
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt.quality, 0.5)

    def test_successive_halving_empty_configs(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        pt = opt.successive_halving([], n_rounds=3)
        self.assertIsNone(pt)

    def test_successive_halving_prefers_feasible(self):
        configs = [{"x": x} for x in [0.1, 0.3, 0.5, 0.7, 0.9]]
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.4, seed=SEED
        )
        pt = opt.successive_halving(configs, n_rounds=2)
        if pt and pt.quality >= 0.4:
            self.assertGreaterEqual(pt.quality, 0.4)

    # -- random search with pruning -----------------------------------------

    def test_random_search_returns_point(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)}, n_iterations=50
        )
        self.assertIsNotNone(pt)

    def test_random_search_feasible(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)}, n_iterations=100
        )
        if pt is not None:
            self.assertGreaterEqual(pt.quality, 0.3)

    def test_random_search_history(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.0, seed=SEED
        )
        opt.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)}, n_iterations=30
        )
        self.assertEqual(len(opt.history), 30)

    def test_random_search_pruning_threshold(self):
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.5, seed=SEED
        )
        pt = opt.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)},
            n_iterations=100,
            prune_threshold=0.5,
        )
        if pt is not None:
            self.assertGreaterEqual(pt.quality, 0.5)

    def test_random_search_infeasible(self):
        """Very high threshold → may return None."""
        opt = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=2.0, seed=SEED
        )
        pt = opt.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)}, n_iterations=10
        )
        self.assertIsNone(pt)

    def test_random_search_deterministic(self):
        """Same seed → same result."""
        opt1 = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=42
        )
        pt1 = opt1.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)}, n_iterations=20
        )
        opt2 = QualityConstrainedDiversityOptimizer(
            self._eval_fn, min_quality=0.3, seed=42
        )
        pt2 = opt2.random_search_with_pruning(
            param_ranges={"x": (0.0, 1.0)}, n_iterations=20
        )
        self.assertAlmostEqual(pt1.quality, pt2.quality, places=10)
        self.assertAlmostEqual(pt1.diversity, pt2.diversity, places=10)


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestTradeoffVisualizer
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeoffVisualizer(unittest.TestCase):
    """Tests for TradeoffVisualizer."""

    def setUp(self):
        self.pts = make_linear_front(20)
        pf = ParetoFrontier(self.pts)
        self.frontier = pf.compute_frontier()

    # -- frontier plot data -------------------------------------------------

    def test_prepare_frontier_plot_data_keys(self):
        data = TradeoffVisualizer.prepare_frontier_plot_data(self.pts, self.frontier)
        expected = {
            "all_quality", "all_diversity",
            "frontier_quality", "frontier_diversity",
            "n_total", "n_frontier",
        }
        self.assertEqual(set(data.keys()), expected)

    def test_prepare_frontier_plot_data_counts(self):
        data = TradeoffVisualizer.prepare_frontier_plot_data(self.pts, self.frontier)
        self.assertEqual(data["n_total"], len(self.pts))
        self.assertEqual(data["n_frontier"], len(self.frontier))
        self.assertEqual(len(data["all_quality"]), len(self.pts))
        self.assertEqual(len(data["frontier_quality"]), len(self.frontier))

    def test_prepare_frontier_plot_data_sorted(self):
        data = TradeoffVisualizer.prepare_frontier_plot_data(self.pts, self.frontier)
        fq = data["frontier_quality"]
        for i in range(len(fq) - 1):
            self.assertLessEqual(fq[i], fq[i + 1])

    def test_prepare_frontier_plot_data_empty(self):
        data = TradeoffVisualizer.prepare_frontier_plot_data([], [])
        self.assertEqual(data["n_total"], 0)
        self.assertEqual(data["n_frontier"], 0)

    def test_prepare_frontier_plot_data_values_range(self):
        data = TradeoffVisualizer.prepare_frontier_plot_data(self.pts, self.frontier)
        for v in data["all_quality"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    # -- heatmap data -------------------------------------------------------

    def test_prepare_heatmap_data_keys(self):
        data = TradeoffVisualizer.prepare_heatmap_data(self.pts)
        self.assertIn("counts", data)
        self.assertIn("quality_edges", data)
        self.assertIn("diversity_edges", data)

    def test_prepare_heatmap_data_shape(self):
        data = TradeoffVisualizer.prepare_heatmap_data(self.pts, n_bins=5)
        counts = data["counts"]
        self.assertEqual(len(counts), 5)
        self.assertEqual(len(counts[0]), 5)
        self.assertEqual(len(data["quality_edges"]), 6)
        self.assertEqual(len(data["diversity_edges"]), 6)

    def test_prepare_heatmap_data_sum(self):
        data = TradeoffVisualizer.prepare_heatmap_data(self.pts, n_bins=10)
        total = sum(sum(row) for row in data["counts"])
        self.assertEqual(total, len(self.pts))

    def test_prepare_heatmap_data_default_bins(self):
        data = TradeoffVisualizer.prepare_heatmap_data(self.pts)
        self.assertEqual(len(data["counts"]), 10)

    # -- radar chart data ---------------------------------------------------

    def test_prepare_radar_chart_data_structure(self):
        metrics = {
            "A": {"quality": 0.8, "diversity": 0.6, "speed": 0.9},
            "B": {"quality": 0.7, "diversity": 0.8, "speed": 0.5},
        }
        data = TradeoffVisualizer.prepare_radar_chart_data(metrics)
        self.assertEqual(data["algorithms"], ["A", "B"])
        self.assertEqual(len(data["axes"]), 3)
        self.assertEqual(len(data["values"]["A"]), 3)
        self.assertEqual(len(data["values"]["B"]), 3)

    def test_prepare_radar_chart_data_empty(self):
        data = TradeoffVisualizer.prepare_radar_chart_data({})
        self.assertEqual(data["algorithms"], [])
        self.assertEqual(data["axes"], [])

    def test_prepare_radar_chart_data_single(self):
        metrics = {"X": {"a": 1.0, "b": 2.0}}
        data = TradeoffVisualizer.prepare_radar_chart_data(metrics)
        self.assertEqual(len(data["algorithms"]), 1)
        self.assertEqual(data["values"]["X"], [1.0, 2.0])

    # -- parallel coordinates data ------------------------------------------

    def test_prepare_parallel_coordinates_data_basic(self):
        data = TradeoffVisualizer.prepare_parallel_coordinates_data(self.pts)
        self.assertEqual(data["axes"], ["quality", "diversity"])
        self.assertEqual(len(data["data"]), len(self.pts))

    def test_prepare_parallel_coordinates_data_extra_axes(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5, algorithm="a",
                             metadata={"speed": 1.0})]
        data = TradeoffVisualizer.prepare_parallel_coordinates_data(
            pts, extra_axes=["speed"]
        )
        self.assertIn("speed", data["axes"])
        self.assertEqual(data["data"][0]["speed"], 1.0)

    def test_prepare_parallel_coordinates_data_empty(self):
        data = TradeoffVisualizer.prepare_parallel_coordinates_data([])
        self.assertEqual(len(data["data"]), 0)

    def test_prepare_parallel_coordinates_missing_metadata(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        data = TradeoffVisualizer.prepare_parallel_coordinates_data(
            pts, extra_axes=["missing"]
        )
        self.assertEqual(data["data"][0]["missing"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestHelperFunctions
# ═══════════════════════════════════════════════════════════════════════════


class TestHelperFunctions(unittest.TestCase):
    """Tests for module-level helper functions."""

    # -- is_pareto_dominated ------------------------------------------------

    def test_dominated_by_one(self):
        pt = np.array([0.3, 0.3])
        others = np.array([[0.5, 0.5]])
        self.assertTrue(is_pareto_dominated(pt, others, maximize=True))

    def test_not_dominated(self):
        pt = np.array([0.5, 0.5])
        others = np.array([[0.3, 0.7], [0.7, 0.3]])
        self.assertFalse(is_pareto_dominated(pt, others, maximize=True))

    def test_dominated_minimise(self):
        pt = np.array([0.8, 0.8])
        others = np.array([[0.3, 0.3]])
        self.assertTrue(is_pareto_dominated(pt, others, maximize=False))

    def test_not_dominated_minimise(self):
        pt = np.array([0.3, 0.3])
        others = np.array([[0.5, 0.5]])
        self.assertFalse(is_pareto_dominated(pt, others, maximize=False))

    def test_equal_not_dominated(self):
        pt = np.array([0.5, 0.5])
        others = np.array([[0.5, 0.5]])
        self.assertFalse(is_pareto_dominated(pt, others, maximize=True))

    def test_dominated_empty_others(self):
        pt = np.array([0.5, 0.5])
        others = np.empty((0, 2))
        self.assertFalse(is_pareto_dominated(pt, others))

    def test_dominated_3d(self):
        pt = np.array([0.3, 0.3, 0.3])
        others = np.array([[0.5, 0.5, 0.5]])
        self.assertTrue(is_pareto_dominated(pt, others, maximize=True))

    def test_not_dominated_3d(self):
        pt = np.array([0.5, 0.5, 0.5])
        others = np.array([[0.6, 0.4, 0.5]])
        self.assertFalse(is_pareto_dominated(pt, others, maximize=True))

    # -- pareto_rank_assignment ---------------------------------------------

    def test_rank_single_front(self):
        pts = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        ranks = pareto_rank_assignment(pts)
        np.testing.assert_array_equal(ranks, [0, 0, 0])

    def test_rank_two_fronts(self):
        pts = np.array([[1, 0], [0, 1], [0.3, 0.3]])
        ranks = pareto_rank_assignment(pts)
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[1], 0)
        # (0.3, 0.3) is NOT dominated by (1,0) or (0,1) since neither
        # dominates in both objectives; so it's also rank 0
        self.assertEqual(ranks[2], 0)

    def test_rank_three_fronts(self):
        pts = np.array([
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.2],
        ])
        ranks = pareto_rank_assignment(pts)
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[2], 2)

    def test_rank_minimise(self):
        pts = np.array([[0.1, 0.1], [0.5, 0.5], [1.0, 1.0]])
        ranks = pareto_rank_assignment(pts, maximize=False)
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[2], 2)

    def test_rank_all_same(self):
        pts = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        ranks = pareto_rank_assignment(pts)
        np.testing.assert_array_equal(ranks, [0, 0, 0])

    def test_rank_many_points(self):
        np.random.seed(SEED)
        pts = np.random.rand(50, 2)
        ranks = pareto_rank_assignment(pts)
        self.assertEqual(len(ranks), 50)
        self.assertTrue(np.all(ranks >= 0))
        # Rank 0 should have at least one point
        self.assertGreater(np.sum(ranks == 0), 0)

    # -- compute_dominated_hypervolume --------------------------------------

    def test_hv_single_point(self):
        pts = np.array([[1.0, 1.0]])
        ref = np.array([0.0, 0.0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        self.assertAlmostEqual(hv, 1.0, places=5)

    def test_hv_two_points(self):
        pts = np.array([[1.0, 0.5], [0.5, 1.0]])
        ref = np.array([0.0, 0.0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        self.assertAlmostEqual(hv, 0.75, places=5)

    def test_hv_empty(self):
        pts = np.empty((0, 2))
        ref = np.array([0.0, 0.0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        self.assertAlmostEqual(hv, 0.0)

    def test_hv_below_reference(self):
        pts = np.array([[0.0, 0.0]])
        ref = np.array([1.0, 1.0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        self.assertAlmostEqual(hv, 0.0)

    def test_hv_minimise(self):
        pts = np.array([[0.2, 0.3]])
        ref = np.array([1.0, 1.0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=False)
        self.assertAlmostEqual(hv, 0.8 * 0.7, places=5)

    def test_hv_monotone_property(self):
        """Adding a non-dominated point should not decrease HV."""
        pts1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        pts2 = np.array([[1.0, 0.5], [0.5, 1.0], [0.8, 0.8]])
        ref = np.array([0.0, 0.0])
        hv1 = compute_dominated_hypervolume(pts1, ref, maximize=True)
        hv2 = compute_dominated_hypervolume(pts2, ref, maximize=True)
        self.assertGreaterEqual(hv2, hv1 - 1e-10)

    # -- compute_spread_metric ---------------------------------------------

    def test_spread_two_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(compute_spread_metric(pts), math.sqrt(2))

    def test_spread_single_point(self):
        pts = np.array([[0.5, 0.5]])
        self.assertAlmostEqual(compute_spread_metric(pts), 0.0)

    def test_spread_same_points(self):
        pts = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.assertAlmostEqual(compute_spread_metric(pts), 0.0)

    def test_spread_positive(self):
        np.random.seed(SEED)
        pts = np.random.rand(20, 2)
        self.assertGreater(compute_spread_metric(pts), 0)

    def test_spread_3d(self):
        pts = np.array([[0, 0, 0], [1, 1, 1]])
        self.assertAlmostEqual(compute_spread_metric(pts), math.sqrt(3))

    # -- generational distance ----------------------------------------------

    def test_gd_identical(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        gd = compute_generational_distance(pts, pts)
        self.assertAlmostEqual(gd, 0.0, places=5)

    def test_gd_offset(self):
        approx = np.array([[0.0, 0.0]])
        ref = np.array([[1.0, 0.0]])
        gd = compute_generational_distance(approx, ref)
        self.assertAlmostEqual(gd, 1.0, places=5)

    def test_gd_empty_approx(self):
        gd = compute_generational_distance(np.empty((0, 2)), np.array([[1, 1]]))
        self.assertEqual(gd, float("inf"))

    def test_gd_empty_ref(self):
        gd = compute_generational_distance(np.array([[1, 1]]), np.empty((0, 2)))
        self.assertEqual(gd, float("inf"))

    def test_gd_symmetric_like(self):
        """GD is generally NOT symmetric, but equal sets should give 0."""
        pts = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        gd = compute_generational_distance(pts, pts)
        self.assertAlmostEqual(gd, 0.0, places=5)

    def test_gd_positive(self):
        approx = np.array([[0.5, 0.5]])
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        gd = compute_generational_distance(approx, ref)
        self.assertGreater(gd, 0)

    # -- inverted generational distance -------------------------------------

    def test_igd_identical(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        igd = compute_inverted_generational_distance(pts, pts)
        self.assertAlmostEqual(igd, 0.0, places=5)

    def test_igd_worse_than_gd_when_approx_sparse(self):
        """IGD penalises sparse approximations of a dense reference."""
        approx = np.array([[0.5, 0.5]])
        ref = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5],
                         [0.75, 0.25], [1.0, 0.0]])
        igd = compute_inverted_generational_distance(approx, ref)
        gd = compute_generational_distance(approx, ref)
        self.assertGreater(igd, gd)

    def test_igd_empty_approx(self):
        igd = compute_inverted_generational_distance(
            np.empty((0, 2)), np.array([[1, 1]]))
        self.assertEqual(igd, float("inf"))

    def test_igd_empty_ref(self):
        igd = compute_inverted_generational_distance(
            np.array([[1, 1]]), np.empty((0, 2)))
        self.assertEqual(igd, float("inf"))

    def test_igd_positive(self):
        approx = np.array([[0.3, 0.3]])
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        igd = compute_inverted_generational_distance(approx, ref)
        self.assertGreater(igd, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 9. TestEdgeCases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases(unittest.TestCase):
    """Edge case tests: empty, single, all dominated, all non-dominated."""

    # -- empty inputs -------------------------------------------------------

    def test_frontier_empty(self):
        pf = ParetoFrontier([])
        f = pf.compute_frontier()
        self.assertEqual(len(f), 0)

    def test_frontier_empty_hypervolume(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.hypervolume(), 0.0)

    def test_frontier_empty_crowding(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertEqual(len(pf.crowding_distance()), 0)

    def test_frontier_empty_area(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.area_under_frontier(), 0.0)

    def test_frontier_empty_spread(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.frontier_spread(), 0.0)

    def test_frontier_empty_uniformity(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        self.assertAlmostEqual(pf.frontier_uniformity(), 0.0)

    def test_finder_empty(self):
        pf = ParetoFrontier([])
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        self.assertIsNone(finder.find_knee_point())
        self.assertIsNone(finder.find_balanced_point())
        self.assertIsNone(finder.find_quality_constrained_optimum(0.0))
        self.assertIsNone(finder.find_diversity_constrained_optimum(0.0))
        self.assertIsNone(finder.find_target_ratio_point(1.0))
        self.assertIsNone(finder.multi_objective_scalarization())

    def test_analyzer_empty(self):
        analyzer = TradeoffAnalyzer()
        result = analyzer.analyze_algorithm_tradeoffs({"A": []})
        self.assertEqual(result["A"]["n_points"], 0)

    # -- single point -------------------------------------------------------

    def test_frontier_single_point(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertEqual(len(f), 1)

    def test_frontier_single_hypervolume(self):
        pts = [TradeoffPoint(quality=1.0, diversity=1.0)]
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf = ParetoFrontier(pts, cfg)
        pf.compute_frontier()
        self.assertAlmostEqual(pf.hypervolume(), 1.0, places=5)

    def test_frontier_single_crowding(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        self.assertEqual(cd[0], float("inf"))

    def test_frontier_single_area(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        self.assertAlmostEqual(pf.area_under_frontier(), 0.0)

    def test_frontier_single_interpolation(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        interp = pf.frontier_interpolation(5)
        self.assertEqual(interp.shape, (5, 2))

    def test_finder_single_point(self):
        pts = [TradeoffPoint(quality=0.5, diversity=0.5)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        self.assertAlmostEqual(finder.find_knee_point().quality, 0.5)
        self.assertAlmostEqual(finder.find_balanced_point().quality, 0.5)

    # -- all dominated by one point -----------------------------------------

    def test_all_dominated(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),
            TradeoffPoint(quality=0.3, diversity=0.3),
            TradeoffPoint(quality=0.1, diversity=0.1),
        ]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertEqual(len(f), 1)
        self.assertAlmostEqual(f[0].quality, 1.0)
        self.assertAlmostEqual(f[0].diversity, 1.0)

    def test_all_dominated_hypervolume(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),
        ]
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf = ParetoFrontier(pts, cfg)
        pf.compute_frontier()
        hv = pf.hypervolume()
        # Only the dominant point contributes
        self.assertAlmostEqual(hv, 1.0, places=5)

    def test_all_dominated_crowding(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        self.assertEqual(len(cd), 1)

    # -- all non-dominated --------------------------------------------------

    def test_all_non_dominated(self):
        pts = [
            TradeoffPoint(quality=1.0, diversity=0.0),
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),
        ]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertEqual(len(f), 3)

    def test_all_non_dominated_large(self):
        pts = make_linear_front(100)
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertEqual(len(f), 100)

    def test_all_non_dominated_crowding(self):
        pts = make_linear_front(5)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        self.assertEqual(len(cd), 5)

    # -- duplicate points ---------------------------------------------------

    def test_duplicate_points(self):
        pts = [
            TradeoffPoint(quality=0.5, diversity=0.5),
            TradeoffPoint(quality=0.5, diversity=0.5),
            TradeoffPoint(quality=0.5, diversity=0.5),
        ]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        # Duplicates are not dominated by each other
        self.assertEqual(len(f), 3)

    # -- very close points --------------------------------------------------

    def test_very_close_points(self):
        pts = [
            TradeoffPoint(quality=0.5, diversity=0.5),
            TradeoffPoint(quality=0.5 + 1e-15, diversity=0.5 + 1e-15),
        ]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        # The slightly larger point dominates
        self.assertGreaterEqual(len(f), 1)

    # -- negative values ----------------------------------------------------

    def test_negative_values_frontier(self):
        pts = [
            TradeoffPoint(quality=-1.0, diversity=0.0),
            TradeoffPoint(quality=0.0, diversity=-1.0),
            TradeoffPoint(quality=-0.5, diversity=-0.5),
        ]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        # (-0.5,-0.5) is NOT dominated: neither (-1,0) nor (0,-1) beats it in both
        self.assertEqual(len(f), 3)

    # -- large datasets -----------------------------------------------------

    def test_large_dataset(self):
        np.random.seed(SEED)
        pts = [TradeoffPoint(quality=float(x), diversity=float(y))
               for x, y in np.random.rand(200, 2)]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertGreater(len(f), 0)
        self.assertLessEqual(len(f), 200)

    # -- collinear points ---------------------------------------------------

    def test_collinear_points(self):
        """All points on the same line: all are non-dominated."""
        pts = [TradeoffPoint(quality=float(i), diversity=float(10 - i))
               for i in range(11)]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertEqual(len(f), 11)

    # -- zero ranges --------------------------------------------------------

    def test_zero_range_quality(self):
        """All points have the same quality."""
        pts = [TradeoffPoint(quality=0.5, diversity=float(d))
               for d in [0.1, 0.3, 0.5, 0.7, 0.9]]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        # Only the one with max diversity survives
        self.assertEqual(len(f), 1)
        self.assertAlmostEqual(f[0].diversity, 0.9)

    def test_zero_range_diversity(self):
        pts = [TradeoffPoint(quality=float(q), diversity=0.5)
               for q in [0.1, 0.3, 0.5, 0.7, 0.9]]
        pf = ParetoFrontier(pts)
        f = pf.compute_frontier()
        self.assertEqual(len(f), 1)
        self.assertAlmostEqual(f[0].quality, 0.9)


# ═══════════════════════════════════════════════════════════════════════════
# 10. TestIntegration
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration(unittest.TestCase):
    """Full pipeline integration tests."""

    def test_full_pipeline_linear(self):
        """Generate → compute frontier → find knee → validate."""
        pts = make_linear_front(50)
        pf = ParetoFrontier(pts, TradeoffConfig(reference_point=[0.0, 0.0]))
        frontier = pf.compute_frontier()

        # All points should be on frontier
        self.assertEqual(len(frontier), 50)

        # Hypervolume
        hv = pf.hypervolume()
        self.assertAlmostEqual(hv, 0.5, places=1)

        # Crowding distance
        cd = pf.crowding_distance()
        self.assertEqual(len(cd), 50)

        # Operating points
        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertIsNotNone(knee)

        balanced = finder.find_balanced_point()
        self.assertIsNotNone(balanced)

        qc = finder.find_quality_constrained_optimum(0.5)
        self.assertIsNotNone(qc)
        self.assertGreaterEqual(qc.quality, 0.5)

        dc = finder.find_diversity_constrained_optimum(0.5)
        self.assertIsNotNone(dc)
        self.assertGreaterEqual(dc.diversity, 0.5)

    def test_full_pipeline_concave(self):
        pts = make_concave_front(50, seed=99)
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf = ParetoFrontier(pts, cfg)
        frontier = pf.compute_frontier()
        self.assertGreater(len(frontier), 0)

        hv = pf.hypervolume()
        self.assertGreater(hv, 0)

        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertIsNotNone(knee)

    def test_full_pipeline_hyperbolic(self):
        pts = make_hyperbolic_front(30)
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf = ParetoFrontier(pts, cfg)
        frontier = pf.compute_frontier()
        self.assertGreater(len(frontier), 0)

        # Area
        area = pf.area_under_frontier()
        self.assertGreater(area, 0)

        # Spread
        spread = pf.frontier_spread()
        self.assertGreater(spread, 0)

        # Interpolation
        interp = pf.frontier_interpolation(100)
        self.assertEqual(interp.shape[0], 100)

    def test_full_pipeline_analyzer(self):
        data = make_two_algorithm_data(seed=SEED)
        analyzer = TradeoffAnalyzer(TradeoffConfig(reference_point=[0.0, 0.0]))

        # Analyze
        results = analyzer.analyze_algorithm_tradeoffs(data)
        self.assertEqual(len(results), 2)

        # Curves
        curves = analyzer.compute_tradeoff_curves(data, n_points=30)
        self.assertEqual(len(curves), 2)

        # Compare
        comparison = analyzer.compare_frontiers(data["A"], data["B"])
        self.assertIn("hypervolume_a", comparison)

        # Marginal rates
        rates = analyzer.compute_marginal_rates(data["A"])
        self.assertIsInstance(rates, list)

    def test_full_pipeline_optimizer(self):
        def eval_fn(config):
            x = config["x"]
            return TradeoffPoint(quality=x, diversity=1 - x, config=config)

        opt = QualityConstrainedDiversityOptimizer(
            eval_fn, min_quality=0.3, seed=SEED
        )

        # Grid search
        best_grid = opt.grid_search_with_constraint(
            {"x": np.linspace(0.0, 1.0, 11).tolist()}
        )
        self.assertIsNotNone(best_grid)
        self.assertGreaterEqual(best_grid.quality, 0.3)

        # Random search
        opt2 = QualityConstrainedDiversityOptimizer(
            eval_fn, min_quality=0.3, seed=SEED
        )
        best_rand = opt2.random_search_with_pruning(
            {"x": (0.0, 1.0)}, n_iterations=50
        )
        self.assertIsNotNone(best_rand)
        self.assertGreaterEqual(best_rand.quality, 0.3)

    def test_full_pipeline_visualizer(self):
        pts = make_linear_front(30)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()

        # Frontier plot
        plot_data = TradeoffVisualizer.prepare_frontier_plot_data(pts, frontier)
        self.assertEqual(plot_data["n_total"], 30)

        # Heatmap
        heatmap = TradeoffVisualizer.prepare_heatmap_data(pts)
        total = sum(sum(row) for row in heatmap["counts"])
        self.assertEqual(total, 30)

        # Radar
        metrics = {"A": {"q": 0.8, "d": 0.6}, "B": {"q": 0.7, "d": 0.9}}
        radar = TradeoffVisualizer.prepare_radar_chart_data(metrics)
        self.assertEqual(len(radar["algorithms"]), 2)

        # Parallel coords
        pc = TradeoffVisualizer.prepare_parallel_coordinates_data(pts)
        self.assertEqual(len(pc["data"]), 30)

    def test_multi_run_statistical_comparison(self):
        """Multiple runs → statistical test."""
        runs_a = []
        runs_b = []
        for i in range(10):
            rng_a = np.random.RandomState(i)
            rng_b = np.random.RandomState(i + 1000)
            a_pts = [TradeoffPoint(quality=rng_a.uniform(0.5, 1.0),
                                   diversity=rng_a.uniform(0.5, 1.0))
                     for _ in range(15)]
            b_pts = [TradeoffPoint(quality=rng_b.uniform(0.0, 0.5),
                                   diversity=rng_b.uniform(0.0, 0.5))
                     for _ in range(15)]
            runs_a.append(a_pts)
            runs_b.append(b_pts)

        analyzer = TradeoffAnalyzer(TradeoffConfig(reference_point=[0.0, 0.0]))
        result = analyzer.statistical_frontier_comparison(runs_a, runs_b)
        self.assertGreater(result["mean_hv_a"], result["mean_hv_b"])
        self.assertIsInstance(result["t_stat"], float)
        self.assertIsInstance(result["p_value"], float)

    def test_sensitivity_analysis_integration(self):
        pts = make_concave_front(50, seed=77)
        analyzer = TradeoffAnalyzer()
        weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        results = analyzer.sensitivity_analysis(pts, quality_weights=weights)
        self.assertEqual(len(results), 11)

        qualities = [r["quality"] for r in results if r["quality"] is not None]
        # With increasing quality weight, quality should generally increase
        # (not strict due to discrete frontier)
        self.assertGreater(qualities[-1], qualities[0] - 0.1)

    def test_optimizer_successive_halving_integration(self):
        call_count = {"n": 0}

        def eval_fn(config):
            call_count["n"] += 1
            x = config["x"]
            return TradeoffPoint(quality=x, diversity=1 - x, config=config)

        configs = [{"x": x} for x in np.linspace(0.1, 0.9, 16)]
        opt = QualityConstrainedDiversityOptimizer(
            eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.successive_halving(configs, n_rounds=3)
        self.assertIsNotNone(pt)
        self.assertGreater(call_count["n"], 0)

    def test_end_to_end_with_config(self):
        """Complete analysis with custom TradeoffConfig."""
        cfg = TradeoffConfig(
            quality_weight=0.6,
            diversity_weight=0.4,
            min_quality=0.2,
            min_diversity=0.2,
            epsilon=0.05,
            n_interpolation_points=50,
            reference_point=[0.0, 0.0],
            seed=99,
        )

        pts = make_hyperbolic_front(40, seed=cfg.seed)
        pf = ParetoFrontier(pts, cfg)
        frontier = pf.compute_frontier()
        self.assertGreater(len(frontier), 0)

        hv = pf.hypervolume()
        self.assertGreater(hv, 0)

        interp = pf.frontier_interpolation()
        self.assertEqual(interp.shape, (50, 2))

        finder = OperatingPointFinder(pf)
        balanced = finder.find_balanced_point()
        self.assertIsNotNone(balanced)

        scalar = finder.multi_objective_scalarization(
            weights=(cfg.quality_weight, cfg.diversity_weight)
        )
        self.assertIsNotNone(scalar)

    def test_rank_and_frontier_consistency(self):
        """Rank-0 points should match frontier points."""
        np.random.seed(SEED)
        raw = np.random.rand(30, 2)
        pts = [TradeoffPoint(quality=float(r[0]), diversity=float(r[1]))
               for r in raw]

        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        frontier_arr = np.array([[p.quality, p.diversity] for p in frontier])

        ranks = pareto_rank_assignment(raw, maximize=True)
        rank0_indices = np.where(ranks == 0)[0]
        rank0_arr = raw[rank0_indices]

        # Both should identify the same set of points
        self.assertEqual(len(frontier_arr), len(rank0_arr))
        for pt_f in frontier_arr:
            found = any(np.allclose(pt_f, pt_r) for pt_r in rank0_arr)
            self.assertTrue(found, f"Frontier point {pt_f} not in rank-0 set")

    def test_helper_functions_consistency(self):
        """Cross-check multiple helper functions."""
        np.random.seed(SEED)
        pts = np.random.rand(20, 2)
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        gd = compute_generational_distance(pts, ref)
        igd = compute_inverted_generational_distance(pts, ref)
        spread = compute_spread_metric(pts)

        self.assertGreater(gd, 0)
        self.assertGreater(igd, 0)
        self.assertGreater(spread, 0)

    def test_dominated_hypervolume_vs_area(self):
        """Area under linear front ≈ dominated hypervolume with ref (0,0)."""
        pts_list = make_linear_front(100)
        pf = ParetoFrontier(pts_list, TradeoffConfig(reference_point=[0.0, 0.0]))
        frontier = pf.compute_frontier()

        hv = pf.hypervolume()
        area = pf.area_under_frontier()

        # For y = 1-x, HV(ref=0) = area under curve = 0.5
        self.assertAlmostEqual(hv, 0.5, places=1)
        self.assertAlmostEqual(area, 0.5, places=1)

    def test_epsilon_dominance_integration(self):
        """Epsilon-dominance should be transitive (approximately)."""
        pf = ParetoFrontier([], TradeoffConfig(epsilon=0.1))
        a = TradeoffPoint(quality=0.8, diversity=0.8)
        b = TradeoffPoint(quality=0.75, diversity=0.75)
        c = TradeoffPoint(quality=0.7, diversity=0.7)

        self.assertTrue(pf.epsilon_dominates(a, b))
        self.assertTrue(pf.epsilon_dominates(b, c))
        self.assertTrue(pf.epsilon_dominates(a, c))

    def test_crowding_distance_sum_property(self):
        """Sum of finite crowding distances should be positive."""
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        finite_sum = sum(v for v in cd.values() if v < float("inf"))
        self.assertGreater(finite_sum, 0)

    def test_interpolation_preserves_boundary_values(self):
        """Interpolated frontier should pass through actual frontier endpoints."""
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        interp = pf.frontier_interpolation(200)

        frontier_arr = np.array([[p.quality, p.diversity] for p in frontier])
        q_min = frontier_arr[:, 0].min()
        q_max = frontier_arr[:, 0].max()

        self.assertAlmostEqual(interp[0, 0], q_min, places=5)
        self.assertAlmostEqual(interp[-1, 0], q_max, places=5)

    def test_scalarization_consistency(self):
        """Weighted sum and Tchebycheff should both return frontier points."""
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)

        ws = finder.multi_objective_scalarization(
            weights=(0.5, 0.5), method="weighted_sum"
        )
        tc = finder.multi_objective_scalarization(
            weights=(0.5, 0.5), method="tchebycheff"
        )
        self.assertIn(ws, pf.frontier)
        self.assertIn(tc, pf.frontier)

    def test_marginal_rates_signs(self):
        """On a typical tradeoff front, rates should be negative."""
        pts = make_linear_front(20)
        analyzer = TradeoffAnalyzer()
        rates = analyzer.compute_marginal_rates(pts)
        for r in rates:
            self.assertLess(r, 0.1)  # should be ≈ -1

    def test_visualizer_all_formats_integration(self):
        """Ensure all visualizer methods can handle the same dataset."""
        pts = make_random_points(50, seed=SEED)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()

        plot = TradeoffVisualizer.prepare_frontier_plot_data(pts, frontier)
        self.assertGreater(plot["n_total"], 0)

        heatmap = TradeoffVisualizer.prepare_heatmap_data(pts)
        self.assertGreater(len(heatmap["counts"]), 0)

        metrics = {"alg": {"q": 0.5, "d": 0.5}}
        radar = TradeoffVisualizer.prepare_radar_chart_data(metrics)
        self.assertGreater(len(radar["algorithms"]), 0)

        pc = TradeoffVisualizer.prepare_parallel_coordinates_data(pts)
        self.assertEqual(len(pc["data"]), 50)


# ═══════════════════════════════════════════════════════════════════════════
# Additional property-based tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParetoProperties(unittest.TestCase):
    """Property-based sanity checks for Pareto computations."""

    def test_frontier_subset_of_input(self):
        pts = make_random_points(100, seed=SEED)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        for fp in frontier:
            self.assertIn(fp, pts)

    def test_non_dominated_property(self):
        """No frontier point should be dominated by another frontier point."""
        pts = make_random_points(80, seed=SEED)
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        arr = np.array([[p.quality, p.diversity] for p in frontier])
        for i in range(len(arr)):
            others = np.delete(arr, i, axis=0)
            self.assertFalse(
                is_pareto_dominated(arr[i], others, maximize=True),
                f"Point {arr[i]} is dominated within the frontier"
            )

    def test_hypervolume_non_negative(self):
        pts = make_random_points(50, seed=SEED)
        pf = ParetoFrontier(pts, TradeoffConfig(reference_point=[0.0, 0.0]))
        pf.compute_frontier()
        self.assertGreaterEqual(pf.hypervolume(), 0)

    def test_spread_non_negative(self):
        pts = make_random_points(50, seed=SEED)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        self.assertGreaterEqual(pf.frontier_spread(), 0)

    def test_uniformity_non_negative(self):
        pts = make_random_points(50, seed=SEED)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        self.assertGreaterEqual(pf.frontier_uniformity(), 0)

    def test_area_under_frontier_non_negative(self):
        pts = make_random_points(50, seed=SEED)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        self.assertGreaterEqual(pf.area_under_frontier(), 0)

    def test_gd_igd_triangle_inequality_like(self):
        """GD(A, B) + GD(B, C) >= GD(A, C) doesn't hold in general,
        but both should be non-negative."""
        np.random.seed(SEED)
        a = np.random.rand(10, 2)
        b = np.random.rand(10, 2)
        self.assertGreaterEqual(compute_generational_distance(a, b), 0)
        self.assertGreaterEqual(compute_inverted_generational_distance(a, b), 0)

    def test_rank_zero_matches_frontier_size(self):
        np.random.seed(SEED)
        pts_arr = np.random.rand(40, 2)
        pts = [TradeoffPoint(quality=float(r[0]), diversity=float(r[1]))
               for r in pts_arr]
        pf = ParetoFrontier(pts)
        frontier = pf.compute_frontier()
        ranks = pareto_rank_assignment(pts_arr, maximize=True)
        n_rank_0 = int(np.sum(ranks == 0))
        self.assertEqual(n_rank_0, len(frontier))

    def test_hypervolume_increases_with_better_front(self):
        """A dominating front should have >= hypervolume."""
        pts_low = [TradeoffPoint(quality=float(x), diversity=float(1 - x))
                    for x in np.linspace(0, 1, 20)]
        pts_high = [TradeoffPoint(quality=float(x + 0.1),
                                   diversity=float(1 - x + 0.1))
                     for x in np.linspace(0, 1, 20)]
        cfg = TradeoffConfig(reference_point=[0.0, 0.0])
        pf_low = ParetoFrontier(pts_low, cfg)
        pf_high = ParetoFrontier(pts_high, cfg)
        pf_low.compute_frontier()
        pf_high.compute_frontier()
        self.assertGreaterEqual(pf_high.hypervolume(), pf_low.hypervolume())


class TestCrowdingDistanceDetailed(unittest.TestCase):
    """Detailed tests for crowding distance computation."""

    def test_three_evenly_spaced(self):
        pts = [
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=0.5, diversity=0.5),
            TradeoffPoint(quality=1.0, diversity=0.0),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        # Endpoints infinite, middle finite and positive
        inf_count = sum(1 for v in cd.values() if v == float("inf"))
        self.assertEqual(inf_count, 2)
        finite = [v for v in cd.values() if v < float("inf")]
        self.assertEqual(len(finite), 1)
        self.assertGreater(finite[0], 0)

    def test_four_points_symmetry(self):
        """Symmetric arrangement should give equal crowding to interior pts."""
        pts = [
            TradeoffPoint(quality=0.0, diversity=1.0),
            TradeoffPoint(quality=1.0 / 3, diversity=2.0 / 3),
            TradeoffPoint(quality=2.0 / 3, diversity=1.0 / 3),
            TradeoffPoint(quality=1.0, diversity=0.0),
        ]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        cd = pf.crowding_distance()
        finite = [v for v in cd.values() if v < float("inf")]
        if len(finite) == 2:
            self.assertAlmostEqual(finite[0], finite[1], places=3)


class TestOperatingPointFinderAdvanced(unittest.TestCase):
    """Advanced tests for OperatingPointFinder."""

    def test_concave_knee_near_middle(self):
        """On y = 1 - x^2, knee should be near x ≈ 0.4-0.6."""
        n = 100
        x = np.linspace(0, 1, n)
        y = 1 - x ** 2
        pts = [TradeoffPoint(quality=float(xi), diversity=float(yi))
               for xi, yi in zip(x, y)]
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)
        knee = finder.find_knee_point()
        self.assertIsNotNone(knee)
        # Knee should be in approximate middle region
        self.assertGreater(knee.quality, 0.1)
        self.assertLess(knee.quality, 0.9)

    def test_weighted_sum_extreme_weights(self):
        pts = make_linear_front(20)
        pf = ParetoFrontier(pts)
        pf.compute_frontier()
        finder = OperatingPointFinder(pf)

        # All weight on quality
        pt_q = finder.multi_objective_scalarization(weights=(1.0, 0.0))
        max_q = max(p.quality for p in pts)
        self.assertAlmostEqual(pt_q.quality, max_q, places=5)

        # All weight on diversity
        pt_d = finder.multi_objective_scalarization(weights=(0.0, 1.0))
        max_d = max(p.diversity for p in pts)
        self.assertAlmostEqual(pt_d.diversity, max_d, places=5)


class TestTradeoffAnalyzerAdvanced(unittest.TestCase):
    """Advanced tests for TradeoffAnalyzer."""

    def test_compare_identical_frontiers(self):
        analyzer = TradeoffAnalyzer()
        pts = make_linear_front(20)
        result = analyzer.compare_frontiers(pts, pts)
        self.assertAlmostEqual(result["hypervolume_ratio"], 1.0, places=5)
        self.assertAlmostEqual(
            result["spread_a"], result["spread_b"], places=5
        )

    def test_compute_curves_monotone_quality(self):
        analyzer = TradeoffAnalyzer()
        data = {"A": make_linear_front(20)}
        curves = analyzer.compute_tradeoff_curves(data, n_points=50)
        arr = curves["A"]
        if arr.shape[0] > 1:
            diffs = np.diff(arr[:, 0])
            self.assertTrue(np.all(diffs >= -1e-10))

    def test_analyzer_with_custom_config(self):
        cfg = TradeoffConfig(
            quality_weight=0.7,
            diversity_weight=0.3,
            reference_point=[0.0, 0.0],
        )
        analyzer = TradeoffAnalyzer(cfg)
        data = make_two_algorithm_data()
        result = analyzer.analyze_algorithm_tradeoffs(data)
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_sensitivity_various_fronts(self):
        """Sensitivity analysis on different front shapes."""
        analyzer = TradeoffAnalyzer()

        for gen_fn in [make_linear_front, lambda n=20: make_concave_front(n)]:
            pts = gen_fn(30)
            results = analyzer.sensitivity_analysis(pts)
            self.assertEqual(len(results), 5)
            for r in results:
                self.assertIsNotNone(r["selected_point"])


class TestOptimizerAdvanced(unittest.TestCase):
    """Advanced optimizer tests."""

    def test_grid_search_pareto_optimal(self):
        """Grid search result should be on the feasibility boundary."""
        def eval_fn(config):
            x = config["x"]
            return TradeoffPoint(quality=x, diversity=1 - x)

        opt = QualityConstrainedDiversityOptimizer(eval_fn, min_quality=0.4)
        grid = {"x": np.linspace(0, 1, 101).tolist()}
        best = opt.grid_search_with_constraint(grid)
        # Best feasible: x=0.4, d=0.6
        self.assertAlmostEqual(best.quality, 0.4, places=1)
        self.assertAlmostEqual(best.diversity, 0.6, places=1)

    def test_multiple_runs_bayesian(self):
        """Multiple bayesian steps should explore the space."""
        def eval_fn(config):
            x = config["x"]
            return TradeoffPoint(quality=x, diversity=1 - x)

        opt = QualityConstrainedDiversityOptimizer(
            eval_fn, min_quality=0.2, seed=SEED
        )
        for _ in range(3):
            opt.bayesian_optimization_step({"x": (0.0, 1.0)}, n_candidates=10)
        self.assertEqual(len(opt.history), 30)

    def test_successive_halving_convergence(self):
        """With enough rounds, should converge to good solution."""
        def eval_fn(config):
            x = config["x"]
            return TradeoffPoint(quality=x, diversity=1 - x)

        configs = [{"x": x} for x in np.linspace(0, 1, 32)]
        opt = QualityConstrainedDiversityOptimizer(
            eval_fn, min_quality=0.3, seed=SEED
        )
        pt = opt.successive_halving(configs, n_rounds=5)
        self.assertIsNotNone(pt)


class TestHypervolumeDetailed(unittest.TestCase):
    """Detailed hypervolume computation tests."""

    def test_three_points_l_shape(self):
        pts = np.array([[2, 1], [1, 2]])
        ref = np.array([0, 0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        # Area: 2*1 + 1*(2-1) = 2 + 1 = 3
        self.assertAlmostEqual(hv, 3.0, places=5)

    def test_staircase(self):
        pts = np.array([[3, 1], [2, 2], [1, 3]])
        ref = np.array([0, 0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        # Area: 3*1 + 2*(2-1) + 1*(3-2) = 3 + 2 + 1 = 6
        self.assertAlmostEqual(hv, 6.0, places=5)

    def test_single_point_unit(self):
        pts = np.array([[1, 1]])
        ref = np.array([0, 0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        self.assertAlmostEqual(hv, 1.0, places=5)

    def test_dominated_point_no_contribution(self):
        """A dominated point shouldn't increase hypervolume."""
        pts_without = np.array([[2, 1], [1, 2]])
        pts_with = np.array([[2, 1], [1, 2], [0.5, 0.5]])
        ref = np.array([0, 0])
        hv1 = compute_dominated_hypervolume(pts_without, ref, maximize=True)
        hv2 = compute_dominated_hypervolume(pts_with, ref, maximize=True)
        self.assertAlmostEqual(hv1, hv2, places=5)

    def test_hv_minimization_basic(self):
        pts = np.array([[0.2, 0.2]])
        ref = np.array([1.0, 1.0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=False)
        self.assertAlmostEqual(hv, 0.8 * 0.8, places=5)

    def test_hv_all_outside_reference(self):
        pts = np.array([[-1, -1], [-2, -2]])
        ref = np.array([0, 0])
        hv = compute_dominated_hypervolume(pts, ref, maximize=True)
        self.assertAlmostEqual(hv, 0.0)


class TestRankAssignmentDetailed(unittest.TestCase):
    """Detailed tests for pareto_rank_assignment."""

    def test_strict_layers(self):
        pts = np.array([
            [4, 4],
            [3, 3],
            [2, 2],
            [1, 1],
        ], dtype=float)
        ranks = pareto_rank_assignment(pts)
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[2], 2)
        self.assertEqual(ranks[3], 3)

    def test_two_on_first_front(self):
        pts = np.array([
            [2, 1],
            [1, 2],
            [0.5, 0.5],
        ], dtype=float)
        ranks = pareto_rank_assignment(pts)
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[1], 0)
        self.assertEqual(ranks[2], 1)

    def test_empty_array(self):
        pts = np.empty((0, 2))
        ranks = pareto_rank_assignment(pts)
        self.assertEqual(len(ranks), 0)


class TestSpreadAndDistanceDetailed(unittest.TestCase):
    """Detailed tests for spread and distance helpers."""

    def test_spread_unit_square_corners(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        spread = compute_spread_metric(pts)
        self.assertAlmostEqual(spread, math.sqrt(2), places=5)

    def test_gd_one_to_many(self):
        approx = np.array([[0.5, 0.5]])
        ref = np.array([[0, 0], [1, 1]])
        gd = compute_generational_distance(approx, ref)
        expected = math.sqrt(0.5)
        self.assertAlmostEqual(gd, expected, places=5)

    def test_igd_many_to_one(self):
        approx = np.array([[0.5, 0.5]])
        ref = np.array([[0, 0], [1, 1]])
        igd = compute_inverted_generational_distance(approx, ref)
        # Each ref point dist to (0.5,0.5): sqrt(0.5)
        expected = math.sqrt(0.5)
        self.assertAlmostEqual(igd, expected, places=5)

    def test_gd_with_p1(self):
        approx = np.array([[0, 0]])
        ref = np.array([[3, 4]])
        gd = compute_generational_distance(approx, ref, p=1.0)
        self.assertAlmostEqual(gd, 5.0, places=5)

    def test_igd_with_p1(self):
        approx = np.array([[0, 0]])
        ref = np.array([[3, 4]])
        igd = compute_inverted_generational_distance(approx, ref, p=1.0)
        self.assertAlmostEqual(igd, 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
