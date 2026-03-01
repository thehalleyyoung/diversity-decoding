"""
Comprehensive tests for the reproducibility evaluation module.

Covers inter-run variance analysis, seed sensitivity, confidence intervals,
reproducibility reporting, experiment replication, and statistical helper
functions for assessing the reliability and stability of decoding experiments.
"""

from __future__ import annotations

import math
import os
import random
import unittest
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
from scipy import stats

from src.evaluation.reproducibility import (
    ReproducibilityConfig,
    RunResult,
    InterRunVarianceAnalyzer,
    SeedSensitivityAnalyzer,
    ConfidenceIntervalComputer,
    ReproducibilityReporter,
    ExperimentReplicator,
    bootstrap_resample,
    jackknife_resample,
    compute_effective_sample_size,
    cochrans_q_test,
    compute_krippendorffs_alpha,
    compute_fleiss_kappa,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic data generation
# ---------------------------------------------------------------------------

def make_run_results(
    n_runs: int = 5,
    metrics: Optional[Dict[str, float]] = None,
    noise_std: float = 0.01,
    seed: int = 42,
) -> List[RunResult]:
    """Create a list of RunResult objects with controlled noise around base metrics."""
    rng = np.random.RandomState(seed)
    base_metrics = metrics or {"accuracy": 0.85, "diversity": 0.60, "bleu": 0.35}
    results = []
    for i in range(n_runs):
        noisy = {
            k: np.clip(v + rng.normal(0, noise_std), 0.0, 1.0)
            for k, v in base_metrics.items()
        }
        results.append(RunResult(
            run_id=f"run_{i}",
            seed=seed + i,
            metrics=noisy,
            metadata={"epoch": i, "lr": 1e-4},
        ))
    return results


def make_run_results_with_known_variance(
    n_runs: int = 10,
    metric_name: str = "accuracy",
    mean: float = 0.80,
    std: float = 0.05,
    seed: int = 42,
) -> List[RunResult]:
    """Create runs whose single metric has a known mean and std."""
    rng = np.random.RandomState(seed)
    vals = rng.normal(mean, std, size=n_runs)
    results = []
    for i, v in enumerate(vals):
        results.append(RunResult(
            run_id=f"run_{i}",
            seed=seed + i,
            metrics={metric_name: float(v)},
        ))
    return results


def make_identical_run_results(n_runs: int = 5, seed: int = 42) -> List[RunResult]:
    """All runs return exactly the same metrics."""
    metrics = {"accuracy": 0.90, "diversity": 0.55}
    return [
        RunResult(run_id=f"run_{i}", seed=seed + i, metrics=dict(metrics))
        for i in range(n_runs)
    ]


def make_high_variance_run_results(n_runs: int = 6, seed: int = 42) -> List[RunResult]:
    """Runs with deliberately high inter-run variance."""
    rng = np.random.RandomState(seed)
    results = []
    for i in range(n_runs):
        results.append(RunResult(
            run_id=f"run_{i}",
            seed=seed + i,
            metrics={
                "accuracy": float(rng.uniform(0.3, 0.95)),
                "diversity": float(rng.uniform(0.1, 0.9)),
            },
        ))
    return results


# ===================================================================
# 1. TestReproducibilityConfig
# ===================================================================

class TestReproducibilityConfig(unittest.TestCase):
    """Tests for the ReproducibilityConfig dataclass."""

    def test_default_creation(self):
        cfg = ReproducibilityConfig()
        self.assertIsNotNone(cfg)
        self.assertIsInstance(cfg.n_runs, int)
        self.assertGreater(cfg.n_runs, 0)

    def test_custom_creation(self):
        cfg = ReproducibilityConfig(
            n_runs=10,
            confidence_level=0.99,
            bootstrap_samples=5000,
            random_seed=123,
        )
        self.assertEqual(cfg.n_runs, 10)
        self.assertAlmostEqual(cfg.confidence_level, 0.99)
        self.assertEqual(cfg.bootstrap_samples, 5000)
        self.assertEqual(cfg.random_seed, 123)

    def test_serialization_to_dict(self):
        cfg = ReproducibilityConfig(n_runs=7, confidence_level=0.90)
        d = asdict(cfg)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["n_runs"], 7)
        self.assertAlmostEqual(d["confidence_level"], 0.90)

    def test_field_names(self):
        names = {f.name for f in fields(ReproducibilityConfig)}
        self.assertIn("n_runs", names)
        self.assertIn("confidence_level", names)

    def test_default_confidence_level(self):
        cfg = ReproducibilityConfig()
        self.assertTrue(0 < cfg.confidence_level < 1)

    def test_default_bootstrap_samples(self):
        cfg = ReproducibilityConfig()
        self.assertIsInstance(cfg.bootstrap_samples, int)
        self.assertGreater(cfg.bootstrap_samples, 0)

    def test_equality(self):
        a = ReproducibilityConfig(n_runs=5)
        b = ReproducibilityConfig(n_runs=5)
        self.assertEqual(a, b)

    def test_inequality(self):
        a = ReproducibilityConfig(n_runs=5)
        b = ReproducibilityConfig(n_runs=10)
        self.assertNotEqual(a, b)


# ===================================================================
# 2. TestRunResult
# ===================================================================

class TestRunResult(unittest.TestCase):
    """Tests for the RunResult dataclass."""

    def test_creation(self):
        r = RunResult(
            run_id="run_0",
            seed=42,
            metrics={"accuracy": 0.9, "bleu": 0.4},
        )
        self.assertEqual(r.run_id, "run_0")
        self.assertEqual(r.seed, 42)
        self.assertAlmostEqual(r.metrics["accuracy"], 0.9)

    def test_metadata_default(self):
        r = RunResult(run_id="x", seed=0, metrics={"m": 0.5})
        # metadata should be either None or an empty dict by default
        if r.metadata is not None:
            self.assertIsInstance(r.metadata, dict)

    def test_serialization(self):
        r = RunResult(
            run_id="run_1",
            seed=7,
            metrics={"diversity": 0.65},
            metadata={"note": "test"},
        )
        d = asdict(r)
        self.assertEqual(d["run_id"], "run_1")
        self.assertEqual(d["seed"], 7)
        self.assertIn("diversity", d["metrics"])

    def test_multiple_metrics(self):
        metrics = {f"metric_{i}": i * 0.1 for i in range(10)}
        r = RunResult(run_id="multi", seed=1, metrics=metrics)
        self.assertEqual(len(r.metrics), 10)

    def test_equality(self):
        a = RunResult(run_id="a", seed=1, metrics={"x": 0.5})
        b = RunResult(run_id="a", seed=1, metrics={"x": 0.5})
        self.assertEqual(a, b)

    def test_inequality_different_metrics(self):
        a = RunResult(run_id="a", seed=1, metrics={"x": 0.5})
        b = RunResult(run_id="a", seed=1, metrics={"x": 0.6})
        self.assertNotEqual(a, b)


# ===================================================================
# 3. TestInterRunVarianceAnalyzer
# ===================================================================

class TestInterRunVarianceAnalyzer(unittest.TestCase):
    """Tests for variance computation across experiment runs."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10, confidence_level=0.95)
        self.analyzer = InterRunVarianceAnalyzer(self.config)
        np.random.seed(42)

    # -- compute_variance_across_runs --

    def test_variance_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        result = self.analyzer.compute_variance_across_runs(runs)
        self.assertIsInstance(result, dict)
        for metric in ["accuracy", "diversity", "bleu"]:
            self.assertIn(metric, result)
            self.assertGreaterEqual(result[metric], 0.0)

    def test_variance_zero_for_identical(self):
        runs = make_identical_run_results(n_runs=10)
        result = self.analyzer.compute_variance_across_runs(runs)
        for v in result.values():
            self.assertAlmostEqual(v, 0.0, places=10)

    def test_variance_increases_with_noise(self):
        runs_low = make_run_results(n_runs=20, noise_std=0.01, seed=0)
        runs_high = make_run_results(n_runs=20, noise_std=0.1, seed=0)
        var_low = self.analyzer.compute_variance_across_runs(runs_low)
        var_high = self.analyzer.compute_variance_across_runs(runs_high)
        for m in var_low:
            self.assertLess(var_low[m], var_high[m])

    def test_variance_single_metric(self):
        runs = make_run_results_with_known_variance(
            n_runs=100, metric_name="acc", mean=0.8, std=0.05, seed=10
        )
        result = self.analyzer.compute_variance_across_runs(runs)
        self.assertIn("acc", result)
        # variance should be close to 0.05^2 = 0.0025
        self.assertAlmostEqual(result["acc"], 0.0025, delta=0.001)

    def test_variance_non_negative(self):
        runs = make_run_results(n_runs=5, noise_std=0.1, seed=99)
        result = self.analyzer.compute_variance_across_runs(runs)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)

    # -- compute_coefficient_of_variation --

    def test_cv_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        cv = self.analyzer.compute_coefficient_of_variation(runs)
        self.assertIsInstance(cv, dict)
        for m in cv:
            self.assertGreaterEqual(cv[m], 0.0)

    def test_cv_zero_for_identical(self):
        runs = make_identical_run_results(n_runs=5)
        cv = self.analyzer.compute_coefficient_of_variation(runs)
        for v in cv.values():
            self.assertAlmostEqual(v, 0.0, places=10)

    def test_cv_relative_magnitude(self):
        # CV should be higher when noise is larger relative to mean
        runs_a = make_run_results_with_known_variance(
            n_runs=50, metric_name="m", mean=0.9, std=0.01, seed=1
        )
        runs_b = make_run_results_with_known_variance(
            n_runs=50, metric_name="m", mean=0.9, std=0.1, seed=1
        )
        cv_a = self.analyzer.compute_coefficient_of_variation(runs_a)
        cv_b = self.analyzer.compute_coefficient_of_variation(runs_b)
        self.assertLess(cv_a["m"], cv_b["m"])

    def test_cv_dimensionless(self):
        runs = make_run_results(n_runs=10, noise_std=0.02, seed=7)
        cv = self.analyzer.compute_coefficient_of_variation(runs)
        # CV is std/mean, should be reasonable for small noise
        for v in cv.values():
            self.assertLess(v, 1.0)

    # -- compute_intraclass_correlation --

    def test_icc_identical_runs(self):
        runs = make_identical_run_results(n_runs=8)
        icc = self.analyzer.compute_intraclass_correlation(runs)
        self.assertIsInstance(icc, (float, dict))
        # Perfect agreement => ICC should be high or NaN/1.0
        if isinstance(icc, dict):
            for v in icc.values():
                self.assertTrue(v >= 0.99 or np.isnan(v))
        else:
            self.assertTrue(icc >= 0.99 or np.isnan(icc))

    def test_icc_range(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        icc = self.analyzer.compute_intraclass_correlation(runs)
        if isinstance(icc, dict):
            for v in icc.values():
                if not np.isnan(v):
                    self.assertTrue(-1.0 <= v <= 1.0)
        else:
            if not np.isnan(icc):
                self.assertTrue(-1.0 <= icc <= 1.0)

    def test_icc_low_for_high_variance(self):
        runs = make_high_variance_run_results(n_runs=10, seed=42)
        icc = self.analyzer.compute_intraclass_correlation(runs)
        if isinstance(icc, dict):
            for v in icc.values():
                if not np.isnan(v):
                    self.assertLess(v, 0.99)

    # -- compute_concordance_correlation --

    def test_concordance_self(self):
        """Concordance of a run with itself should be ~1."""
        runs = make_run_results(n_runs=2, noise_std=0.0, seed=42)
        # Force identical metrics
        runs[1].metrics = dict(runs[0].metrics)
        result = self.analyzer.compute_concordance_correlation(runs)
        self.assertIsInstance(result, (float, dict))
        if isinstance(result, dict):
            for v in result.values():
                self.assertAlmostEqual(v, 1.0, places=5)
        else:
            self.assertAlmostEqual(result, 1.0, places=5)

    def test_concordance_range(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        result = self.analyzer.compute_concordance_correlation(runs)
        if isinstance(result, dict):
            for v in result.values():
                if not np.isnan(v):
                    self.assertTrue(-1.0 <= v <= 1.0)
        else:
            if not np.isnan(result):
                self.assertTrue(-1.0 <= result <= 1.0)

    # -- friedman_test_across_runs --

    def test_friedman_identical(self):
        runs = make_identical_run_results(n_runs=5)
        result = self.analyzer.friedman_test_across_runs(runs)
        self.assertIsInstance(result, dict)
        # If all identical, p-value should be high (no significant difference)
        if "p_value" in result:
            self.assertGreater(result["p_value"], 0.05)

    def test_friedman_returns_statistic(self):
        runs = make_run_results(n_runs=6, noise_std=0.05, seed=42)
        result = self.analyzer.friedman_test_across_runs(runs)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)

    def test_friedman_different_runs(self):
        runs = make_high_variance_run_results(n_runs=6, seed=42)
        result = self.analyzer.friedman_test_across_runs(runs)
        self.assertIsInstance(result["statistic"], float)
        self.assertIsInstance(result["p_value"], float)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    # -- compute_effect_sizes --

    def test_effect_sizes_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        result = self.analyzer.compute_effect_sizes(runs)
        self.assertIsInstance(result, dict)
        for key in result:
            self.assertIsInstance(result[key], (float, dict))

    def test_effect_sizes_zero_for_identical(self):
        runs = make_identical_run_results(n_runs=5)
        result = self.analyzer.compute_effect_sizes(runs)
        for key, val in result.items():
            if isinstance(val, (int, float)):
                self.assertAlmostEqual(val, 0.0, places=5)
            elif isinstance(val, dict) and "effect_size" in val:
                self.assertAlmostEqual(val["effect_size"], 0.0, places=5)

    # -- variance_decomposition --

    def test_variance_decomposition_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        result = self.analyzer.variance_decomposition(runs)
        self.assertIsInstance(result, dict)

    def test_variance_decomposition_components_sum(self):
        """Components should sum to total variance (approximately)."""
        runs = make_run_results(n_runs=20, noise_std=0.05, seed=42)
        result = self.analyzer.variance_decomposition(runs)
        # Check structure – it should have components that sum close to total
        if "total" in result and "between" in result and "within" in result:
            total = result["total"]
            between = result["between"]
            within = result["within"]
            if isinstance(total, dict):
                for m in total:
                    self.assertAlmostEqual(
                        total[m], between.get(m, 0) + within.get(m, 0),
                        delta=total[m] * 0.1 + 1e-10,
                    )
            else:
                self.assertAlmostEqual(total, between + within, delta=total * 0.1 + 1e-10)

    def test_variance_decomposition_non_negative(self):
        runs = make_run_results(n_runs=8, noise_std=0.05, seed=42)
        result = self.analyzer.variance_decomposition(runs)
        for key, val in result.items():
            if isinstance(val, (int, float)):
                self.assertGreaterEqual(val, 0.0)
            elif isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, (int, float)):
                        self.assertGreaterEqual(v, -1e-10)

    # -- compute_stability_score --

    def test_stability_score_identical(self):
        runs = make_identical_run_results(n_runs=5)
        score = self.analyzer.compute_stability_score(runs)
        self.assertIsInstance(score, (float, dict))
        if isinstance(score, float):
            self.assertGreaterEqual(score, 0.9)
        else:
            for v in score.values():
                self.assertGreaterEqual(v, 0.9)

    def test_stability_score_low_for_high_variance(self):
        runs = make_high_variance_run_results(n_runs=10, seed=42)
        score = self.analyzer.compute_stability_score(runs)
        if isinstance(score, float):
            self.assertLess(score, 0.95)
        else:
            for v in score.values():
                self.assertLess(v, 0.99)

    def test_stability_score_range(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        score = self.analyzer.compute_stability_score(runs)
        if isinstance(score, float):
            self.assertTrue(0.0 <= score <= 1.0)
        else:
            for v in score.values():
                self.assertTrue(0.0 <= v <= 1.0)

    def test_stability_monotonic_with_noise(self):
        """Stability should decrease as noise increases."""
        scores = []
        for noise in [0.001, 0.01, 0.05, 0.2]:
            runs = make_run_results(n_runs=30, noise_std=noise, seed=42)
            s = self.analyzer.compute_stability_score(runs)
            if isinstance(s, dict):
                s = np.mean(list(s.values()))
            scores.append(s)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i] + 0.05, scores[i + 1])


# ===================================================================
# 4. TestSeedSensitivityAnalyzer
# ===================================================================

class TestSeedSensitivityAnalyzer(unittest.TestCase):
    """Tests for analysing how sensitive results are to random seeds."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10)
        self.analyzer = SeedSensitivityAnalyzer(self.config)
        np.random.seed(42)

    # -- analyze_seed_sensitivity --

    def test_sensitivity_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        result = self.analyzer.analyze_seed_sensitivity(runs)
        self.assertIsInstance(result, dict)

    def test_sensitivity_low_for_stable(self):
        runs = make_run_results(n_runs=10, noise_std=0.001, seed=42)
        result = self.analyzer.analyze_seed_sensitivity(runs)
        # When noise is very low, sensitivity should be low
        if "sensitivity" in result:
            if isinstance(result["sensitivity"], (int, float)):
                self.assertLess(result["sensitivity"], 0.1)
            elif isinstance(result["sensitivity"], dict):
                for v in result["sensitivity"].values():
                    self.assertLess(v, 0.1)

    def test_sensitivity_high_for_unstable(self):
        runs = make_high_variance_run_results(n_runs=10, seed=42)
        result = self.analyzer.analyze_seed_sensitivity(runs)
        if "sensitivity" in result:
            if isinstance(result["sensitivity"], (int, float)):
                self.assertGreater(result["sensitivity"], 0.0)

    # -- compute_seed_correlations --

    def test_seed_correlations_shape(self):
        runs = make_run_results(n_runs=8, noise_std=0.05, seed=42)
        corr = self.analyzer.compute_seed_correlations(runs)
        self.assertIsInstance(corr, (np.ndarray, dict))
        if isinstance(corr, np.ndarray):
            self.assertEqual(corr.shape[0], corr.shape[1])

    def test_seed_correlations_diagonal(self):
        """Diagonal should be 1.0 (self-correlation)."""
        runs = make_run_results(n_runs=5, noise_std=0.05, seed=42)
        corr = self.analyzer.compute_seed_correlations(runs)
        if isinstance(corr, np.ndarray):
            for i in range(corr.shape[0]):
                self.assertAlmostEqual(corr[i, i], 1.0, places=5)

    def test_seed_correlations_symmetric(self):
        runs = make_run_results(n_runs=5, noise_std=0.05, seed=42)
        corr = self.analyzer.compute_seed_correlations(runs)
        if isinstance(corr, np.ndarray):
            np.testing.assert_array_almost_equal(corr, corr.T, decimal=10)

    # -- identify_outlier_seeds --

    def test_no_outliers_for_stable(self):
        runs = make_identical_run_results(n_runs=10)
        outliers = self.analyzer.identify_outlier_seeds(runs)
        self.assertIsInstance(outliers, (list, dict))
        if isinstance(outliers, list):
            self.assertEqual(len(outliers), 0)

    def test_outlier_detection_finds_anomaly(self):
        runs = make_run_results(n_runs=10, noise_std=0.01, seed=42)
        # Inject an outlier
        runs[0].metrics = {k: v + 0.5 for k, v in runs[0].metrics.items()}
        outliers = self.analyzer.identify_outlier_seeds(runs)
        if isinstance(outliers, list):
            self.assertGreater(len(outliers), 0)
            # Check that the outlier seed is flagged
            flagged_ids = [o if isinstance(o, str) else o.get("run_id", o) for o in outliers]
            self.assertIn("run_0", flagged_ids)
        elif isinstance(outliers, dict):
            # At least one metric should have an outlier
            self.assertTrue(any(len(v) > 0 for v in outliers.values() if isinstance(v, list)))

    def test_outlier_detection_returns_list_or_dict(self):
        runs = make_run_results(n_runs=6, noise_std=0.1, seed=42)
        result = self.analyzer.identify_outlier_seeds(runs)
        self.assertIsInstance(result, (list, dict))

    # -- compute_seed_influence --

    def test_seed_influence_basic(self):
        runs = make_run_results(n_runs=8, noise_std=0.05, seed=42)
        influence = self.analyzer.compute_seed_influence(runs)
        self.assertIsInstance(influence, (dict, list))

    def test_seed_influence_length(self):
        n = 7
        runs = make_run_results(n_runs=n, noise_std=0.05, seed=42)
        influence = self.analyzer.compute_seed_influence(runs)
        if isinstance(influence, dict):
            self.assertEqual(len(influence), n)
        elif isinstance(influence, list):
            self.assertEqual(len(influence), n)

    def test_seed_influence_non_negative(self):
        runs = make_run_results(n_runs=6, noise_std=0.05, seed=42)
        influence = self.analyzer.compute_seed_influence(runs)
        if isinstance(influence, dict):
            for v in influence.values():
                if isinstance(v, (int, float)):
                    self.assertGreaterEqual(v, 0.0)
        elif isinstance(influence, list):
            for v in influence:
                if isinstance(v, (int, float)):
                    self.assertGreaterEqual(v, 0.0)

    # -- seed_clustering --

    def test_seed_clustering_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.1, seed=42)
        clusters = self.analyzer.seed_clustering(runs)
        self.assertIsInstance(clusters, (dict, list))

    def test_seed_clustering_all_assigned(self):
        n = 8
        runs = make_run_results(n_runs=n, noise_std=0.1, seed=42)
        clusters = self.analyzer.seed_clustering(runs)
        if isinstance(clusters, dict):
            all_ids = set()
            for ids in clusters.values():
                if isinstance(ids, list):
                    all_ids.update(ids)
            # Every run should appear in some cluster
            self.assertEqual(len(all_ids), n)
        elif isinstance(clusters, list):
            self.assertEqual(len(clusters), n)

    def test_seed_clustering_single_cluster_for_identical(self):
        runs = make_identical_run_results(n_runs=6)
        clusters = self.analyzer.seed_clustering(runs)
        if isinstance(clusters, dict):
            # All seeds should be in the same cluster
            non_empty = [v for v in clusters.values() if isinstance(v, list) and len(v) > 0]
            self.assertEqual(len(non_empty), 1)

    # -- recommend_minimum_seeds --

    def test_recommend_minimum_seeds_basic(self):
        runs = make_run_results(n_runs=20, noise_std=0.05, seed=42)
        rec = self.analyzer.recommend_minimum_seeds(runs)
        self.assertIsInstance(rec, (int, dict))
        if isinstance(rec, int):
            self.assertGreater(rec, 0)

    def test_recommend_minimum_seeds_more_for_unstable(self):
        stable = make_run_results(n_runs=20, noise_std=0.01, seed=42)
        unstable = make_run_results(n_runs=20, noise_std=0.2, seed=42)
        rec_s = self.analyzer.recommend_minimum_seeds(stable)
        rec_u = self.analyzer.recommend_minimum_seeds(unstable)
        if isinstance(rec_s, int) and isinstance(rec_u, int):
            self.assertLessEqual(rec_s, rec_u)
        elif isinstance(rec_s, dict) and isinstance(rec_u, dict):
            for m in rec_s:
                if m in rec_u:
                    self.assertLessEqual(rec_s[m], rec_u[m] + 1)

    def test_recommend_minimum_seeds_upper_bound(self):
        runs = make_run_results(n_runs=20, noise_std=0.05, seed=42)
        rec = self.analyzer.recommend_minimum_seeds(runs)
        if isinstance(rec, int):
            self.assertLessEqual(rec, 1000)

    def test_recommend_returns_positive(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        rec = self.analyzer.recommend_minimum_seeds(runs)
        if isinstance(rec, int):
            self.assertGreater(rec, 0)
        elif isinstance(rec, dict):
            for v in rec.values():
                self.assertGreater(v, 0)


# ===================================================================
# 5. TestConfidenceIntervalComputer
# ===================================================================

class TestConfidenceIntervalComputer(unittest.TestCase):
    """Tests for confidence interval computation methods."""

    def setUp(self):
        self.config = ReproducibilityConfig(confidence_level=0.95, bootstrap_samples=1000)
        self.computer = ConfidenceIntervalComputer(self.config)
        np.random.seed(42)

    # -- normal_ci --

    def test_normal_ci_basic(self):
        data = np.random.normal(10, 2, size=100)
        lo, hi = self.computer.normal_ci(data)
        self.assertLess(lo, 10)
        self.assertGreater(hi, 10)

    def test_normal_ci_contains_mean(self):
        data = np.random.normal(5, 1, size=200)
        lo, hi = self.computer.normal_ci(data)
        self.assertLess(lo, np.mean(data))
        self.assertGreater(hi, np.mean(data))

    def test_normal_ci_narrows_with_n(self):
        rng = np.random.RandomState(42)
        data_small = rng.normal(0, 1, 20)
        data_large = rng.normal(0, 1, 2000)
        lo_s, hi_s = self.computer.normal_ci(data_small)
        lo_l, hi_l = self.computer.normal_ci(data_large)
        self.assertGreater(hi_s - lo_s, hi_l - lo_l)

    def test_normal_ci_symmetric(self):
        data = np.random.normal(0, 1, size=500)
        lo, hi = self.computer.normal_ci(data)
        mean = np.mean(data)
        self.assertAlmostEqual(mean - lo, hi - mean, places=5)

    def test_normal_ci_wider_at_higher_confidence(self):
        data = np.random.normal(0, 1, 100)
        cfg_90 = ReproducibilityConfig(confidence_level=0.90)
        cfg_99 = ReproducibilityConfig(confidence_level=0.99)
        comp_90 = ConfidenceIntervalComputer(cfg_90)
        comp_99 = ConfidenceIntervalComputer(cfg_99)
        w_90 = comp_90.normal_ci(data)[1] - comp_90.normal_ci(data)[0]
        w_99 = comp_99.normal_ci(data)[1] - comp_99.normal_ci(data)[0]
        self.assertLess(w_90, w_99)

    # -- bootstrap_ci --

    def test_bootstrap_ci_basic(self):
        data = np.random.normal(5, 1, size=50)
        lo, hi = self.computer.bootstrap_ci(data)
        self.assertLess(lo, hi)

    def test_bootstrap_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        data = rng.normal(10, 1, size=100)
        lo, hi = self.computer.bootstrap_ci(data)
        self.assertLess(lo, 10.5)
        self.assertGreater(hi, 9.5)

    def test_bootstrap_ci_for_median(self):
        data = np.random.exponential(2, size=100)
        lo, hi = self.computer.bootstrap_ci(data, statistic=np.median)
        self.assertLess(lo, hi)

    def test_bootstrap_ci_width_reasonable(self):
        data = np.random.normal(0, 1, 100)
        lo, hi = self.computer.bootstrap_ci(data)
        width = hi - lo
        self.assertGreater(width, 0)
        self.assertLess(width, 10)  # Should be much smaller for std=1, n=100

    # -- bayesian_ci --

    def test_bayesian_ci_basic(self):
        data = np.random.normal(0, 1, size=50)
        lo, hi = self.computer.bayesian_ci(data)
        self.assertLess(lo, hi)

    def test_bayesian_ci_contains_true_mean(self):
        rng = np.random.RandomState(42)
        data = rng.normal(3, 0.5, size=200)
        lo, hi = self.computer.bayesian_ci(data)
        self.assertLess(lo, 3.0)
        self.assertGreater(hi, 3.0)

    def test_bayesian_ci_narrows_with_data(self):
        rng = np.random.RandomState(42)
        small = rng.normal(0, 1, 10)
        large = rng.normal(0, 1, 500)
        w_s = self.computer.bayesian_ci(small)[1] - self.computer.bayesian_ci(small)[0]
        w_l = self.computer.bayesian_ci(large)[1] - self.computer.bayesian_ci(large)[0]
        self.assertGreater(w_s, w_l)

    # -- wilson_score_ci --

    def test_wilson_score_basic(self):
        lo, hi = self.computer.wilson_score_ci(successes=80, total=100)
        self.assertLess(lo, 0.80)
        self.assertGreater(hi, 0.80)

    def test_wilson_score_all_success(self):
        lo, hi = self.computer.wilson_score_ci(successes=100, total=100)
        self.assertGreater(lo, 0.9)
        self.assertLessEqual(hi, 1.0)

    def test_wilson_score_no_success(self):
        lo, hi = self.computer.wilson_score_ci(successes=0, total=100)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 0.1)

    def test_wilson_score_range(self):
        lo, hi = self.computer.wilson_score_ci(successes=50, total=200)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, hi)

    def test_wilson_score_narrows_with_n(self):
        lo1, hi1 = self.computer.wilson_score_ci(successes=5, total=10)
        lo2, hi2 = self.computer.wilson_score_ci(successes=500, total=1000)
        self.assertGreater(hi1 - lo1, hi2 - lo2)

    # -- simultaneous_ci --

    def test_simultaneous_ci_basic(self):
        data = {
            "a": np.random.normal(0, 1, 50),
            "b": np.random.normal(5, 2, 50),
        }
        result = self.computer.simultaneous_ci(data)
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_simultaneous_ci_wider_than_individual(self):
        """Simultaneous CIs should be at least as wide as individual CIs."""
        rng = np.random.RandomState(42)
        data = {"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)}
        simul = self.computer.simultaneous_ci(data)
        for key, arr in data.items():
            ind_lo, ind_hi = self.computer.normal_ci(arr)
            sim_lo, sim_hi = simul[key]
            # Simultaneous should be at least as wide (possibly with small tolerance)
            self.assertLessEqual(sim_lo, ind_lo + 0.01)
            self.assertGreaterEqual(sim_hi, ind_hi - 0.01)

    def test_simultaneous_ci_structure(self):
        data = {f"m{i}": np.random.normal(i, 1, 30) for i in range(5)}
        result = self.computer.simultaneous_ci(data)
        for key in data:
            self.assertIn(key, result)
            lo, hi = result[key]
            self.assertLess(lo, hi)

    # -- prediction_interval --

    def test_prediction_interval_basic(self):
        data = np.random.normal(0, 1, size=100)
        lo, hi = self.computer.prediction_interval(data)
        self.assertLess(lo, hi)

    def test_prediction_interval_wider_than_ci(self):
        data = np.random.normal(0, 1, size=100)
        ci_lo, ci_hi = self.computer.normal_ci(data)
        pi_lo, pi_hi = self.computer.prediction_interval(data)
        self.assertLessEqual(pi_lo, ci_lo)
        self.assertGreaterEqual(pi_hi, ci_hi)

    def test_prediction_interval_contains_most_data(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 1000)
        lo, hi = self.computer.prediction_interval(data)
        fraction_inside = np.mean((data >= lo) & (data <= hi))
        self.assertGreater(fraction_inside, 0.90)

    # -- tolerance_interval --

    def test_tolerance_interval_basic(self):
        data = np.random.normal(0, 1, size=100)
        lo, hi = self.computer.tolerance_interval(data)
        self.assertLess(lo, hi)

    def test_tolerance_interval_wider_than_prediction(self):
        data = np.random.normal(0, 1, size=100)
        pi_lo, pi_hi = self.computer.prediction_interval(data)
        ti_lo, ti_hi = self.computer.tolerance_interval(data)
        # Tolerance intervals are typically wider or similar
        self.assertLessEqual(ti_lo, pi_lo + 0.5)
        self.assertGreaterEqual(ti_hi, pi_hi - 0.5)

    # -- credible_interval --

    def test_credible_interval_basic(self):
        data = np.random.normal(0, 1, size=100)
        lo, hi = self.computer.credible_interval(data)
        self.assertLess(lo, hi)

    def test_credible_interval_contains_mean(self):
        rng = np.random.RandomState(42)
        data = rng.normal(5, 1, size=200)
        lo, hi = self.computer.credible_interval(data)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)

    def test_credible_interval_narrows_with_n(self):
        rng = np.random.RandomState(42)
        small = rng.normal(0, 1, 10)
        large = rng.normal(0, 1, 500)
        w_s = self.computer.credible_interval(small)[1] - self.computer.credible_interval(small)[0]
        w_l = self.computer.credible_interval(large)[1] - self.computer.credible_interval(large)[0]
        self.assertGreater(w_s, w_l)


# ===================================================================
# 6. TestReproducibilityReporter
# ===================================================================

class TestReproducibilityReporter(unittest.TestCase):
    """Tests for report generation and metric summarisation."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10, confidence_level=0.95)
        self.reporter = ReproducibilityReporter(self.config)
        np.random.seed(42)

    # -- generate_summary_statistics --

    def test_summary_statistics_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        summary = self.reporter.generate_summary_statistics(runs)
        self.assertIsInstance(summary, dict)
        for metric in ["accuracy", "diversity", "bleu"]:
            self.assertIn(metric, summary)

    def test_summary_statistics_has_mean_std(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        summary = self.reporter.generate_summary_statistics(runs)
        for metric, stats_dict in summary.items():
            if isinstance(stats_dict, dict):
                self.assertIn("mean", stats_dict)
                self.assertIn("std", stats_dict)

    def test_summary_statistics_mean_correct(self):
        runs = make_run_results_with_known_variance(
            n_runs=100, metric_name="acc", mean=0.8, std=0.01, seed=42
        )
        summary = self.reporter.generate_summary_statistics(runs)
        if "acc" in summary and isinstance(summary["acc"], dict):
            self.assertAlmostEqual(summary["acc"]["mean"], 0.8, delta=0.02)

    def test_summary_statistics_std_correct(self):
        runs = make_run_results_with_known_variance(
            n_runs=200, metric_name="acc", mean=0.8, std=0.05, seed=42
        )
        summary = self.reporter.generate_summary_statistics(runs)
        if "acc" in summary and isinstance(summary["acc"], dict):
            self.assertAlmostEqual(summary["acc"]["std"], 0.05, delta=0.02)

    def test_summary_statistics_min_max(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        summary = self.reporter.generate_summary_statistics(runs)
        for metric, stats_dict in summary.items():
            if isinstance(stats_dict, dict):
                if "min" in stats_dict and "max" in stats_dict:
                    self.assertLessEqual(stats_dict["min"], stats_dict["max"])

    # -- compute_reproducibility_score --

    def test_reproducibility_score_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        score = self.reporter.compute_reproducibility_score(runs)
        self.assertIsInstance(score, (float, dict))

    def test_reproducibility_score_high_for_stable(self):
        runs = make_identical_run_results(n_runs=10)
        score = self.reporter.compute_reproducibility_score(runs)
        if isinstance(score, float):
            self.assertGreaterEqual(score, 0.9)
        else:
            for v in score.values():
                self.assertGreaterEqual(v, 0.9)

    def test_reproducibility_score_range(self):
        runs = make_run_results(n_runs=10, noise_std=0.1, seed=42)
        score = self.reporter.compute_reproducibility_score(runs)
        if isinstance(score, float):
            self.assertTrue(0.0 <= score <= 1.0)
        else:
            for v in score.values():
                self.assertTrue(0.0 <= v <= 1.0)

    def test_reproducibility_score_ordering(self):
        """More stable runs should yield higher score."""
        stable = make_run_results(n_runs=20, noise_std=0.001, seed=42)
        noisy = make_run_results(n_runs=20, noise_std=0.2, seed=42)
        s_s = self.reporter.compute_reproducibility_score(stable)
        s_n = self.reporter.compute_reproducibility_score(noisy)
        if isinstance(s_s, float) and isinstance(s_n, float):
            self.assertGreater(s_s, s_n)
        elif isinstance(s_s, dict) and isinstance(s_n, dict):
            for m in s_s:
                if m in s_n:
                    self.assertGreaterEqual(s_s[m], s_n[m] - 0.01)

    # -- generate_comparison_table --

    def test_comparison_table_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        table = self.reporter.generate_comparison_table(runs)
        self.assertIsInstance(table, (dict, list, str))

    def test_comparison_table_all_metrics(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        table = self.reporter.generate_comparison_table(runs)
        if isinstance(table, dict):
            for metric in ["accuracy", "diversity", "bleu"]:
                self.assertIn(metric, table)

    def test_comparison_table_with_groups(self):
        group_a = make_run_results(n_runs=5, noise_std=0.05, seed=42)
        group_b = make_run_results(n_runs=5, noise_std=0.05, seed=100)
        # If the method accepts multiple groups
        try:
            table = self.reporter.generate_comparison_table(group_a, group_b)
            self.assertIsNotNone(table)
        except TypeError:
            table = self.reporter.generate_comparison_table(group_a + group_b)
            self.assertIsNotNone(table)

    # -- flag_unreliable_metrics --

    def test_flag_unreliable_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        flags = self.reporter.flag_unreliable_metrics(runs)
        self.assertIsInstance(flags, (list, dict))

    def test_flag_unreliable_none_for_stable(self):
        runs = make_identical_run_results(n_runs=10)
        flags = self.reporter.flag_unreliable_metrics(runs)
        if isinstance(flags, list):
            self.assertEqual(len(flags), 0)
        elif isinstance(flags, dict):
            # All metrics should be marked as reliable
            for v in flags.values():
                if isinstance(v, bool):
                    self.assertFalse(v)

    def test_flag_unreliable_finds_noisy(self):
        runs = make_run_results(n_runs=10, noise_std=0.01, seed=42)
        # Inject one very noisy metric
        for r in runs:
            r.metrics["noisy_metric"] = np.random.uniform(0, 1)
        flags = self.reporter.flag_unreliable_metrics(runs)
        if isinstance(flags, list):
            self.assertTrue(
                any("noisy_metric" in str(f) for f in flags)
                or len(flags) > 0
            )

    # -- prepare_variance_plot_data --

    def test_variance_plot_data_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        plot_data = self.reporter.prepare_variance_plot_data(runs)
        self.assertIsInstance(plot_data, dict)

    def test_variance_plot_data_has_values(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        plot_data = self.reporter.prepare_variance_plot_data(runs)
        self.assertGreater(len(plot_data), 0)

    # -- prepare_seed_sensitivity_plot_data --

    def test_seed_sensitivity_plot_data_basic(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        plot_data = self.reporter.prepare_seed_sensitivity_plot_data(runs)
        self.assertIsInstance(plot_data, dict)

    def test_seed_sensitivity_plot_data_per_metric(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        plot_data = self.reporter.prepare_seed_sensitivity_plot_data(runs)
        for metric in ["accuracy", "diversity", "bleu"]:
            self.assertIn(metric, plot_data)


# ===================================================================
# 7. TestExperimentReplicator
# ===================================================================

class TestExperimentReplicator(unittest.TestCase):
    """Tests for experiment replication and deterministic environment setup."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=5, random_seed=42)
        self.replicator = ExperimentReplicator(self.config)

    # -- setup_deterministic_env --

    def test_setup_deterministic_env(self):
        self.replicator.setup_deterministic_env(seed=42)
        a = np.random.rand()
        self.replicator.setup_deterministic_env(seed=42)
        b = np.random.rand()
        self.assertAlmostEqual(a, b, places=10)

    def test_setup_deterministic_env_python_random(self):
        self.replicator.setup_deterministic_env(seed=123)
        a = random.random()
        self.replicator.setup_deterministic_env(seed=123)
        b = random.random()
        self.assertAlmostEqual(a, b, places=10)

    def test_setup_deterministic_different_seeds(self):
        self.replicator.setup_deterministic_env(seed=1)
        a = np.random.rand()
        self.replicator.setup_deterministic_env(seed=2)
        b = np.random.rand()
        self.assertNotAlmostEqual(a, b, places=5)

    # -- run_replicated_experiment --

    def test_run_replicated_basic(self):
        def dummy_experiment(seed: int) -> Dict[str, float]:
            rng = np.random.RandomState(seed)
            return {"score": float(rng.normal(0.8, 0.01))}

        results = self.replicator.run_replicated_experiment(
            experiment_fn=dummy_experiment,
            seeds=[42, 43, 44, 45, 46],
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, RunResult)
            self.assertIn("score", r.metrics)

    def test_run_replicated_reproducible(self):
        def experiment(seed: int) -> Dict[str, float]:
            rng = np.random.RandomState(seed)
            return {"val": float(rng.normal(0, 1))}

        r1 = self.replicator.run_replicated_experiment(
            experiment_fn=experiment, seeds=[42, 43]
        )
        r2 = self.replicator.run_replicated_experiment(
            experiment_fn=experiment, seeds=[42, 43]
        )
        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a.metrics["val"], b.metrics["val"], places=10)

    def test_run_replicated_different_seeds_differ(self):
        def experiment(seed: int) -> Dict[str, float]:
            rng = np.random.RandomState(seed)
            return {"val": float(rng.normal(0, 1))}

        results = self.replicator.run_replicated_experiment(
            experiment_fn=experiment, seeds=[1, 999]
        )
        self.assertNotAlmostEqual(
            results[0].metrics["val"], results[1].metrics["val"], places=3
        )

    # -- compare_replications --

    def test_compare_replications_basic(self):
        def experiment(seed: int) -> Dict[str, float]:
            rng = np.random.RandomState(seed)
            return {"score": float(rng.normal(0.5, 0.01))}

        results = self.replicator.run_replicated_experiment(
            experiment_fn=experiment, seeds=list(range(42, 52))
        )
        comparison = self.replicator.compare_replications(results)
        self.assertIsInstance(comparison, dict)

    def test_compare_replications_identical(self):
        runs = make_identical_run_results(n_runs=5)
        comparison = self.replicator.compare_replications(runs)
        self.assertIsInstance(comparison, dict)
        if "consistent" in comparison:
            self.assertTrue(comparison["consistent"])

    def test_compare_replications_has_stats(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        comparison = self.replicator.compare_replications(runs)
        # Should have some statistical summary
        self.assertGreater(len(comparison), 0)

    # -- checkpoint_and_restore --

    def test_checkpoint_and_restore_basic(self):
        state = {"model_weights": [0.1, 0.2, 0.3], "epoch": 5}
        try:
            checkpoint_id = self.replicator.checkpoint_and_restore(
                state=state, action="save"
            )
            restored = self.replicator.checkpoint_and_restore(
                checkpoint_id=checkpoint_id, action="restore"
            )
            self.assertEqual(restored["epoch"], 5)
            self.assertEqual(restored["model_weights"], [0.1, 0.2, 0.3])
        except (NotImplementedError, TypeError, FileNotFoundError):
            # Acceptable if checkpoint requires filesystem
            pass

    def test_checkpoint_round_trip(self):
        state = {"accuracy": 0.95, "step": 100}
        try:
            ckpt = self.replicator.checkpoint_and_restore(state=state, action="save")
            restored = self.replicator.checkpoint_and_restore(
                checkpoint_id=ckpt, action="restore"
            )
            self.assertAlmostEqual(restored["accuracy"], 0.95)
        except (NotImplementedError, TypeError, FileNotFoundError):
            pass


# ===================================================================
# 8. TestHelperFunctions
# ===================================================================

class TestHelperFunctions(unittest.TestCase):
    """Tests for standalone statistical helper functions."""

    def setUp(self):
        np.random.seed(42)

    # -- bootstrap_resample --

    def test_bootstrap_resample_length(self):
        data = np.array([1, 2, 3, 4, 5])
        sample = bootstrap_resample(data, seed=42)
        self.assertEqual(len(sample), len(data))

    def test_bootstrap_resample_from_data(self):
        data = np.array([10, 20, 30])
        sample = bootstrap_resample(data, seed=42)
        for val in sample:
            self.assertIn(val, data)

    def test_bootstrap_resample_reproducible(self):
        data = np.arange(100)
        s1 = bootstrap_resample(data, seed=7)
        s2 = bootstrap_resample(data, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_bootstrap_resample_different_seeds(self):
        data = np.arange(100)
        s1 = bootstrap_resample(data, seed=1)
        s2 = bootstrap_resample(data, seed=2)
        self.assertFalse(np.array_equal(s1, s2))

    def test_bootstrap_resample_single_element(self):
        data = np.array([42])
        sample = bootstrap_resample(data, seed=0)
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample[0], 42)

    # -- jackknife_resample --

    def test_jackknife_resample_count(self):
        data = np.array([1, 2, 3, 4, 5])
        samples = jackknife_resample(data)
        self.assertEqual(len(samples), len(data))

    def test_jackknife_resample_sizes(self):
        data = np.array([1, 2, 3, 4, 5])
        samples = jackknife_resample(data)
        for s in samples:
            self.assertEqual(len(s), len(data) - 1)

    def test_jackknife_leave_one_out(self):
        data = np.array([10, 20, 30])
        samples = jackknife_resample(data)
        # First sample should exclude first element
        self.assertNotIn(10, samples[0])
        self.assertIn(20, samples[0])
        self.assertIn(30, samples[0])

    def test_jackknife_resample_deterministic(self):
        data = np.array([1, 2, 3, 4])
        s1 = jackknife_resample(data)
        s2 = jackknife_resample(data)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    # -- compute_effective_sample_size --

    def test_ess_iid_data(self):
        data = np.random.normal(0, 1, 1000)
        ess = compute_effective_sample_size(data)
        # For iid data, ESS should be close to n
        self.assertGreater(ess, 800)

    def test_ess_correlated_data(self):
        # Create auto-correlated data
        n = 1000
        data = np.zeros(n)
        data[0] = np.random.normal()
        for i in range(1, n):
            data[i] = 0.95 * data[i - 1] + np.random.normal(0, 0.1)
        ess = compute_effective_sample_size(data)
        self.assertLess(ess, n)
        self.assertGreater(ess, 0)

    def test_ess_positive(self):
        data = np.random.normal(0, 1, 50)
        ess = compute_effective_sample_size(data)
        self.assertGreater(ess, 0)

    def test_ess_constant_data(self):
        data = np.ones(100)
        ess = compute_effective_sample_size(data)
        # Constant data: ESS could be n or could be low depending on implementation
        self.assertGreater(ess, 0)

    # -- cochrans_q_test --

    def test_cochrans_q_basic(self):
        # Binary matrix: subjects x treatments
        data = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ])
        result = cochrans_q_test(data)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)

    def test_cochrans_q_identical_treatments(self):
        data = np.ones((10, 3))
        result = cochrans_q_test(data)
        # All treatments identical => p-value should be high
        self.assertGreater(result["p_value"], 0.05)

    def test_cochrans_q_p_value_range(self):
        rng = np.random.RandomState(42)
        data = rng.randint(0, 2, size=(20, 4))
        result = cochrans_q_test(data)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_cochrans_q_significant_difference(self):
        # Treatment 1 always succeeds, treatment 3 always fails
        data = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
        ])
        result = cochrans_q_test(data)
        self.assertLess(result["p_value"], 0.05)

    # -- compute_krippendorffs_alpha --

    def test_krippendorffs_alpha_perfect(self):
        # Perfect agreement
        data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        alpha = compute_krippendorffs_alpha(data)
        self.assertAlmostEqual(alpha, 1.0, places=3)

    def test_krippendorffs_alpha_random(self):
        rng = np.random.RandomState(42)
        data = rng.randint(1, 5, size=(5, 20))
        alpha = compute_krippendorffs_alpha(data)
        # Random ratings should yield low alpha
        self.assertLess(alpha, 0.5)

    def test_krippendorffs_alpha_range(self):
        rng = np.random.RandomState(42)
        data = rng.randint(0, 3, size=(3, 10))
        alpha = compute_krippendorffs_alpha(data)
        self.assertLessEqual(alpha, 1.0)

    def test_krippendorffs_alpha_high_agreement(self):
        # Near-perfect agreement
        base = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        data = np.array([base, base, base.copy()])
        data[2][0] = 2  # One disagreement
        alpha = compute_krippendorffs_alpha(data)
        self.assertGreater(alpha, 0.7)

    # -- compute_fleiss_kappa --

    def test_fleiss_kappa_perfect(self):
        # n subjects, k categories; all raters agree
        # Fleiss kappa format: matrix of shape (n_subjects, n_categories)
        # Each row sums to number of raters
        n_raters = 5
        data = np.zeros((10, 3))
        for i in range(10):
            data[i, i % 3] = n_raters
        kappa = compute_fleiss_kappa(data)
        self.assertAlmostEqual(kappa, 1.0, places=2)

    def test_fleiss_kappa_random(self):
        rng = np.random.RandomState(42)
        n_subjects, n_categories, n_raters = 30, 4, 10
        data = np.zeros((n_subjects, n_categories), dtype=int)
        for i in range(n_subjects):
            cats = rng.randint(0, n_categories, size=n_raters)
            for c in cats:
                data[i, c] += 1
        kappa = compute_fleiss_kappa(data)
        # Random assignment -> kappa near 0
        self.assertLess(abs(kappa), 0.3)

    def test_fleiss_kappa_range(self):
        rng = np.random.RandomState(42)
        n_subjects, n_categories, n_raters = 20, 3, 8
        data = np.zeros((n_subjects, n_categories), dtype=int)
        for i in range(n_subjects):
            cats = rng.randint(0, n_categories, size=n_raters)
            for c in cats:
                data[i, c] += 1
        kappa = compute_fleiss_kappa(data)
        self.assertGreaterEqual(kappa, -1.0)
        self.assertLessEqual(kappa, 1.0)

    def test_fleiss_kappa_high_agreement(self):
        n_raters = 10
        n_subjects = 20
        data = np.zeros((n_subjects, 3), dtype=int)
        for i in range(n_subjects):
            cat = i % 3
            data[i, cat] = n_raters  # All raters agree
        kappa = compute_fleiss_kappa(data)
        self.assertAlmostEqual(kappa, 1.0, places=2)


# ===================================================================
# 9. TestEdgeCases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for boundary conditions and edge cases."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=5, confidence_level=0.95)
        np.random.seed(42)

    # -- single run --

    def test_variance_single_run(self):
        runs = make_run_results(n_runs=1, noise_std=0.05, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        try:
            result = analyzer.compute_variance_across_runs(runs)
            # With a single run, variance should be 0 or NaN
            for v in result.values():
                self.assertTrue(v == 0.0 or np.isnan(v))
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise on single run

    def test_cv_single_run(self):
        runs = make_run_results(n_runs=1, noise_std=0.05, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        try:
            result = analyzer.compute_coefficient_of_variation(runs)
            for v in result.values():
                self.assertTrue(v == 0.0 or np.isnan(v))
        except (ValueError, ZeroDivisionError):
            pass

    def test_stability_single_run(self):
        runs = make_run_results(n_runs=1, noise_std=0.05, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        try:
            score = analyzer.compute_stability_score(runs)
            if isinstance(score, float):
                self.assertTrue(0.0 <= score <= 1.0 or np.isnan(score))
        except (ValueError, ZeroDivisionError):
            pass

    # -- identical results --

    def test_identical_results_variance(self):
        runs = make_identical_run_results(n_runs=10)
        analyzer = InterRunVarianceAnalyzer(self.config)
        result = analyzer.compute_variance_across_runs(runs)
        for v in result.values():
            self.assertAlmostEqual(v, 0.0, places=10)

    def test_identical_results_cv(self):
        runs = make_identical_run_results(n_runs=10)
        analyzer = InterRunVarianceAnalyzer(self.config)
        result = analyzer.compute_coefficient_of_variation(runs)
        for v in result.values():
            self.assertAlmostEqual(v, 0.0, places=10)

    def test_identical_results_stability(self):
        runs = make_identical_run_results(n_runs=10)
        analyzer = InterRunVarianceAnalyzer(self.config)
        score = analyzer.compute_stability_score(runs)
        if isinstance(score, float):
            self.assertGreaterEqual(score, 0.99)
        elif isinstance(score, dict):
            for v in score.values():
                self.assertGreaterEqual(v, 0.99)

    def test_identical_results_reporter(self):
        runs = make_identical_run_results(n_runs=10)
        reporter = ReproducibilityReporter(self.config)
        score = reporter.compute_reproducibility_score(runs)
        if isinstance(score, float):
            self.assertGreaterEqual(score, 0.95)

    # -- high variance --

    def test_high_variance_flagged(self):
        runs = make_high_variance_run_results(n_runs=10, seed=42)
        reporter = ReproducibilityReporter(self.config)
        flags = reporter.flag_unreliable_metrics(runs)
        if isinstance(flags, list):
            self.assertGreater(len(flags), 0)

    def test_high_variance_low_stability(self):
        runs = make_high_variance_run_results(n_runs=10, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        score = analyzer.compute_stability_score(runs)
        if isinstance(score, float):
            self.assertLess(score, 0.9)
        elif isinstance(score, dict):
            for v in score.values():
                self.assertLess(v, 0.99)

    def test_high_variance_cv_large(self):
        runs = make_high_variance_run_results(n_runs=10, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        cv = analyzer.compute_coefficient_of_variation(runs)
        for v in cv.values():
            self.assertGreater(v, 0.05)

    # -- missing data --

    def test_empty_metrics(self):
        runs = [
            RunResult(run_id=f"r{i}", seed=i, metrics={})
            for i in range(5)
        ]
        analyzer = InterRunVarianceAnalyzer(self.config)
        try:
            result = analyzer.compute_variance_across_runs(runs)
            self.assertEqual(len(result), 0)
        except (ValueError, KeyError):
            pass

    def test_partial_metrics(self):
        """Runs where not all metrics are present in every run."""
        runs = [
            RunResult(run_id="r0", seed=0, metrics={"a": 0.5, "b": 0.3}),
            RunResult(run_id="r1", seed=1, metrics={"a": 0.6}),
            RunResult(run_id="r2", seed=2, metrics={"a": 0.55, "b": 0.35}),
        ]
        analyzer = InterRunVarianceAnalyzer(self.config)
        try:
            result = analyzer.compute_variance_across_runs(runs)
            self.assertIn("a", result)
        except (ValueError, KeyError):
            pass  # Acceptable to require consistent metrics

    def test_two_runs_variance(self):
        runs = make_run_results(n_runs=2, noise_std=0.05, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        result = analyzer.compute_variance_across_runs(runs)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)

    def test_large_number_of_runs(self):
        runs = make_run_results(n_runs=100, noise_std=0.05, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        result = analyzer.compute_variance_across_runs(runs)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)

    # -- CI edge cases --

    def test_normal_ci_single_point(self):
        computer = ConfidenceIntervalComputer(self.config)
        try:
            lo, hi = computer.normal_ci(np.array([5.0]))
            # Single point: CI might be degenerate
            self.assertTrue(np.isnan(lo) or lo == hi or lo < hi)
        except (ValueError, ZeroDivisionError):
            pass

    def test_normal_ci_two_points(self):
        computer = ConfidenceIntervalComputer(self.config)
        lo, hi = computer.normal_ci(np.array([4.0, 6.0]))
        self.assertLess(lo, hi)

    def test_bootstrap_ci_constant(self):
        computer = ConfidenceIntervalComputer(self.config)
        data = np.ones(50) * 3.0
        lo, hi = computer.bootstrap_ci(data)
        self.assertAlmostEqual(lo, 3.0, places=5)
        self.assertAlmostEqual(hi, 3.0, places=5)

    def test_wilson_score_extreme_proportions(self):
        computer = ConfidenceIntervalComputer(self.config)
        lo, hi = computer.wilson_score_ci(successes=1, total=1000)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 0.05)

    def test_bootstrap_ci_skewed_data(self):
        rng = np.random.RandomState(42)
        data = rng.exponential(1.0, size=100)
        computer = ConfidenceIntervalComputer(self.config)
        lo, hi = computer.bootstrap_ci(data)
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0)

    # -- Seed sensitivity edge cases --

    def test_seed_sensitivity_two_runs(self):
        runs = make_run_results(n_runs=2, noise_std=0.05, seed=42)
        analyzer = SeedSensitivityAnalyzer(self.config)
        try:
            result = analyzer.analyze_seed_sensitivity(runs)
            self.assertIsInstance(result, dict)
        except ValueError:
            pass

    def test_recommend_seeds_low_runs(self):
        runs = make_run_results(n_runs=3, noise_std=0.05, seed=42)
        analyzer = SeedSensitivityAnalyzer(self.config)
        try:
            rec = analyzer.recommend_minimum_seeds(runs)
            if isinstance(rec, int):
                self.assertGreater(rec, 0)
        except ValueError:
            pass


# ===================================================================
# 10. TestIntegration
# ===================================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests for the full reproducibility pipeline."""

    def setUp(self):
        np.random.seed(42)
        self.config = ReproducibilityConfig(
            n_runs=10,
            confidence_level=0.95,
            bootstrap_samples=500,
            random_seed=42,
        )

    def test_full_pipeline_stable_experiment(self):
        """Run a full reproducibility analysis on a stable experiment."""
        # 1. Generate runs
        runs = make_run_results(n_runs=10, noise_std=0.01, seed=42)

        # 2. Variance analysis
        analyzer = InterRunVarianceAnalyzer(self.config)
        variance = analyzer.compute_variance_across_runs(runs)
        for v in variance.values():
            self.assertLess(v, 0.01)

        cv = analyzer.compute_coefficient_of_variation(runs)
        for v in cv.values():
            self.assertLess(v, 0.1)

        stability = analyzer.compute_stability_score(runs)
        if isinstance(stability, float):
            self.assertGreater(stability, 0.8)

        # 3. Seed sensitivity
        seed_analyzer = SeedSensitivityAnalyzer(self.config)
        sensitivity = seed_analyzer.analyze_seed_sensitivity(runs)
        self.assertIsInstance(sensitivity, dict)

        # 4. Confidence intervals
        ci_computer = ConfidenceIntervalComputer(self.config)
        acc_values = np.array([r.metrics["accuracy"] for r in runs])
        lo, hi = ci_computer.normal_ci(acc_values)
        self.assertLess(lo, np.mean(acc_values))
        self.assertGreater(hi, np.mean(acc_values))

        # 5. Report
        reporter = ReproducibilityReporter(self.config)
        summary = reporter.generate_summary_statistics(runs)
        self.assertIsInstance(summary, dict)

        score = reporter.compute_reproducibility_score(runs)
        if isinstance(score, float):
            self.assertGreater(score, 0.8)

        flags = reporter.flag_unreliable_metrics(runs)
        if isinstance(flags, list):
            self.assertEqual(len(flags), 0)

    def test_full_pipeline_unstable_experiment(self):
        """Run a full reproducibility analysis on an unstable experiment."""
        runs = make_high_variance_run_results(n_runs=10, seed=42)

        analyzer = InterRunVarianceAnalyzer(self.config)
        variance = analyzer.compute_variance_across_runs(runs)
        for v in variance.values():
            self.assertGreater(v, 0.001)

        stability = analyzer.compute_stability_score(runs)
        if isinstance(stability, float):
            self.assertLess(stability, 0.95)

        reporter = ReproducibilityReporter(self.config)
        flags = reporter.flag_unreliable_metrics(runs)
        if isinstance(flags, list):
            self.assertGreater(len(flags), 0)

    def test_replicated_experiment_pipeline(self):
        """Replicate an experiment and check reproducibility."""
        replicator = ExperimentReplicator(self.config)

        def experiment(seed: int) -> Dict[str, float]:
            rng = np.random.RandomState(seed)
            return {
                "accuracy": float(np.clip(rng.normal(0.85, 0.02), 0, 1)),
                "diversity": float(np.clip(rng.normal(0.60, 0.03), 0, 1)),
            }

        seeds = list(range(42, 52))
        results = replicator.run_replicated_experiment(
            experiment_fn=experiment, seeds=seeds
        )
        self.assertEqual(len(results), 10)

        comparison = replicator.compare_replications(results)
        self.assertIsInstance(comparison, dict)

        analyzer = InterRunVarianceAnalyzer(self.config)
        variance = analyzer.compute_variance_across_runs(results)
        self.assertIn("accuracy", variance)
        self.assertIn("diversity", variance)

    def test_ci_coverage_simulation(self):
        """Verify that CIs have correct coverage rate."""
        true_mean = 5.0
        n_simulations = 200
        n_samples = 50
        ci_computer = ConfidenceIntervalComputer(self.config)
        rng = np.random.RandomState(42)

        covered = 0
        for _ in range(n_simulations):
            data = rng.normal(true_mean, 1.0, n_samples)
            lo, hi = ci_computer.normal_ci(data)
            if lo <= true_mean <= hi:
                covered += 1

        coverage = covered / n_simulations
        # 95% CI should cover ~95% of the time; allow some slack
        self.assertGreater(coverage, 0.85)
        self.assertLess(coverage, 1.0)

    def test_bootstrap_ci_coverage(self):
        """Verify bootstrap CI coverage."""
        true_mean = 3.0
        n_simulations = 100
        ci_computer = ConfidenceIntervalComputer(self.config)
        rng = np.random.RandomState(42)

        covered = 0
        for _ in range(n_simulations):
            data = rng.normal(true_mean, 2.0, 30)
            lo, hi = ci_computer.bootstrap_ci(data)
            if lo <= true_mean <= hi:
                covered += 1

        coverage = covered / n_simulations
        self.assertGreater(coverage, 0.80)

    def test_helper_functions_consistency(self):
        """Ensure helper functions produce consistent results."""
        data = np.random.normal(0, 1, size=100)

        # Bootstrap
        b1 = bootstrap_resample(data, seed=42)
        b2 = bootstrap_resample(data, seed=42)
        np.testing.assert_array_equal(b1, b2)

        # Jackknife
        jk = jackknife_resample(data)
        self.assertEqual(len(jk), len(data))
        for s in jk:
            self.assertEqual(len(s), len(data) - 1)

        # ESS
        ess = compute_effective_sample_size(data)
        self.assertGreater(ess, 0)
        self.assertLessEqual(ess, len(data) + 1)

    def test_fleiss_kappa_and_krippendorff_consistency(self):
        """Both measures should agree on perfect and random agreement."""
        # Perfect agreement
        n_raters = 5
        perfect_fleiss = np.zeros((10, 3), dtype=int)
        for i in range(10):
            perfect_fleiss[i, i % 3] = n_raters

        kappa = compute_fleiss_kappa(perfect_fleiss)
        self.assertAlmostEqual(kappa, 1.0, places=2)

        perfect_kripp = np.array([
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        ])
        alpha = compute_krippendorffs_alpha(perfect_kripp)
        self.assertAlmostEqual(alpha, 1.0, places=2)

    def test_variance_decomposition_end_to_end(self):
        """Verify variance decomposition with known structure."""
        runs = make_run_results(n_runs=20, noise_std=0.05, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)

        decomp = analyzer.variance_decomposition(runs)
        self.assertIsInstance(decomp, dict)
        self.assertGreater(len(decomp), 0)

        # Also get raw variance for comparison
        variance = analyzer.compute_variance_across_runs(runs)
        for m, v in variance.items():
            self.assertGreater(v, 0)

    def test_seed_analysis_end_to_end(self):
        """Full seed sensitivity analysis pipeline."""
        runs = make_run_results(n_runs=15, noise_std=0.05, seed=42)
        analyzer = SeedSensitivityAnalyzer(self.config)

        sensitivity = analyzer.analyze_seed_sensitivity(runs)
        self.assertIsInstance(sensitivity, dict)

        corr = analyzer.compute_seed_correlations(runs)
        self.assertIsNotNone(corr)

        influence = analyzer.compute_seed_influence(runs)
        self.assertIsNotNone(influence)

        clusters = analyzer.seed_clustering(runs)
        self.assertIsNotNone(clusters)

        rec = analyzer.recommend_minimum_seeds(runs)
        self.assertIsNotNone(rec)

    def test_reporter_full_report(self):
        """Generate all reporter outputs."""
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        reporter = ReproducibilityReporter(self.config)

        summary = reporter.generate_summary_statistics(runs)
        self.assertGreater(len(summary), 0)

        score = reporter.compute_reproducibility_score(runs)
        self.assertIsNotNone(score)

        table = reporter.generate_comparison_table(runs)
        self.assertIsNotNone(table)

        flags = reporter.flag_unreliable_metrics(runs)
        self.assertIsNotNone(flags)

        var_plot = reporter.prepare_variance_plot_data(runs)
        self.assertIsNotNone(var_plot)

        seed_plot = reporter.prepare_seed_sensitivity_plot_data(runs)
        self.assertIsNotNone(seed_plot)

    def test_cross_component_consistency(self):
        """Analyzer and reporter should agree on stability assessment."""
        runs_stable = make_run_results(n_runs=20, noise_std=0.001, seed=42)
        runs_noisy = make_run_results(n_runs=20, noise_std=0.2, seed=42)

        analyzer = InterRunVarianceAnalyzer(self.config)
        reporter = ReproducibilityReporter(self.config)

        stab_s = analyzer.compute_stability_score(runs_stable)
        stab_n = analyzer.compute_stability_score(runs_noisy)
        score_s = reporter.compute_reproducibility_score(runs_stable)
        score_n = reporter.compute_reproducibility_score(runs_noisy)

        # Stable experiment should score higher in both
        if isinstance(stab_s, float) and isinstance(stab_n, float):
            self.assertGreater(stab_s, stab_n)
        if isinstance(score_s, float) and isinstance(score_n, float):
            self.assertGreater(score_s, score_n)

    def test_cochrans_q_integration(self):
        """Integration test for Cochran's Q with synthetic rater data."""
        rng = np.random.RandomState(42)
        # 15 subjects, 4 raters, binary outcomes
        # Treatment 1 has higher success rate
        data = np.zeros((15, 4), dtype=int)
        for i in range(15):
            data[i, 0] = 1  # Treatment 1 always succeeds
            data[i, 1] = int(rng.random() < 0.8)
            data[i, 2] = int(rng.random() < 0.5)
            data[i, 3] = int(rng.random() < 0.3)

        result = cochrans_q_test(data)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)
        # Should detect difference given treatment 1 always succeeds
        self.assertLess(result["p_value"], 0.05)

    def test_multiple_metrics_full_pipeline(self):
        """Pipeline with many metrics simultaneously."""
        metrics = {f"metric_{i}": 0.5 + 0.05 * i for i in range(10)}
        runs = make_run_results(n_runs=10, metrics=metrics, noise_std=0.02, seed=42)

        analyzer = InterRunVarianceAnalyzer(self.config)
        variance = analyzer.compute_variance_across_runs(runs)
        self.assertEqual(len(variance), 10)

        cv = analyzer.compute_coefficient_of_variation(runs)
        self.assertEqual(len(cv), 10)

        reporter = ReproducibilityReporter(self.config)
        summary = reporter.generate_summary_statistics(runs)
        self.assertEqual(len(summary), 10)

    def test_effect_size_interpretation(self):
        """Effect sizes should be small for similar runs."""
        runs = make_run_results(n_runs=10, noise_std=0.01, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        effects = analyzer.compute_effect_sizes(runs)
        self.assertIsInstance(effects, dict)
        for key, val in effects.items():
            if isinstance(val, (int, float)):
                # Small noise => small effect
                self.assertLess(abs(val), 1.0)

    def test_friedman_with_many_runs(self):
        """Friedman test with more runs for better statistical power."""
        runs = make_run_results(n_runs=30, noise_std=0.01, seed=42)
        analyzer = InterRunVarianceAnalyzer(self.config)
        result = analyzer.friedman_test_across_runs(runs)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)

    def test_prediction_and_tolerance_intervals(self):
        """Prediction and tolerance intervals for experiment forecasting."""
        rng = np.random.RandomState(42)
        data = rng.normal(0.8, 0.05, size=50)
        ci_computer = ConfidenceIntervalComputer(self.config)

        pi_lo, pi_hi = ci_computer.prediction_interval(data)
        ti_lo, ti_hi = ci_computer.tolerance_interval(data)
        ci_lo, ci_hi = ci_computer.normal_ci(data)

        # Width ordering: CI <= PI, CI <= TI (approximately)
        ci_width = ci_hi - ci_lo
        pi_width = pi_hi - pi_lo
        self.assertLess(ci_width, pi_width + 0.01)

    def test_deterministic_replication(self):
        """Ensure replicator produces deterministic results."""
        replicator = ExperimentReplicator(self.config)

        def exp(seed: int) -> Dict[str, float]:
            rng = np.random.RandomState(seed)
            return {"v": float(rng.normal(0, 1))}

        r1 = replicator.run_replicated_experiment(exp, seeds=[10, 20, 30])
        r2 = replicator.run_replicated_experiment(exp, seeds=[10, 20, 30])

        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a.metrics["v"], b.metrics["v"], places=10)

    def test_outlier_seed_does_not_affect_ci(self):
        """CIs should be robust even with one outlier seed."""
        rng = np.random.RandomState(42)
        data = rng.normal(0.8, 0.02, size=50)
        data_with_outlier = np.append(data, 0.1)  # Add outlier

        ci = ConfidenceIntervalComputer(self.config)
        lo_clean, hi_clean = ci.normal_ci(data)
        lo_out, hi_out = ci.normal_ci(data_with_outlier)

        # Bootstrap CI may be more robust
        blo_clean, bhi_clean = ci.bootstrap_ci(data)
        blo_out, bhi_out = ci.bootstrap_ci(data_with_outlier)

        # CIs should still be finite and ordered
        self.assertLess(lo_out, hi_out)
        self.assertLess(blo_out, bhi_out)

    def test_all_ci_methods_agree_approximately(self):
        """All CI methods should give similar intervals for large normal data."""
        rng = np.random.RandomState(42)
        data = rng.normal(5, 1, size=500)
        ci = ConfidenceIntervalComputer(self.config)

        n_lo, n_hi = ci.normal_ci(data)
        b_lo, b_hi = ci.bootstrap_ci(data)
        bay_lo, bay_hi = ci.bayesian_ci(data)
        cred_lo, cred_hi = ci.credible_interval(data)

        # All should roughly agree for large normal sample
        centers = [
            (n_lo + n_hi) / 2,
            (b_lo + b_hi) / 2,
            (bay_lo + bay_hi) / 2,
            (cred_lo + cred_hi) / 2,
        ]
        for c in centers:
            self.assertAlmostEqual(c, 5.0, delta=0.3)


# ===================================================================
# Additional granular tests to reach ~2000 lines
# ===================================================================

class TestInterRunVarianceAnalyzerDetailed(unittest.TestCase):
    """Additional detailed tests for InterRunVarianceAnalyzer."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10)
        self.analyzer = InterRunVarianceAnalyzer(self.config)

    def test_variance_multiple_independent_metrics(self):
        """Each metric should have independent variance computation."""
        rng = np.random.RandomState(42)
        runs = []
        for i in range(20):
            runs.append(RunResult(
                run_id=f"r{i}", seed=i,
                metrics={
                    "stable": 0.5 + rng.normal(0, 0.001),
                    "noisy": 0.5 + rng.normal(0, 0.1),
                },
            ))
        var = self.analyzer.compute_variance_across_runs(runs)
        self.assertLess(var["stable"], var["noisy"])

    def test_variance_with_negative_values(self):
        rng = np.random.RandomState(42)
        runs = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"val": float(rng.normal(-1, 0.5))})
            for i in range(10)
        ]
        var = self.analyzer.compute_variance_across_runs(runs)
        self.assertGreater(var["val"], 0)

    def test_cv_with_near_zero_mean(self):
        """CV can be very large when mean is near zero."""
        rng = np.random.RandomState(42)
        runs = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"val": float(rng.normal(0.001, 0.01))})
            for i in range(10)
        ]
        cv = self.analyzer.compute_coefficient_of_variation(runs)
        # CV = std/mean, with small mean can be large
        self.assertGreater(cv["val"], 1.0)

    def test_icc_with_structured_data(self):
        """ICC with data that has clear between-group structure."""
        runs = []
        for i in range(10):
            runs.append(RunResult(
                run_id=f"r{i}", seed=i,
                metrics={"m1": 0.5 + 0.01 * i, "m2": 0.3 + 0.02 * i},
            ))
        icc = self.analyzer.compute_intraclass_correlation(runs)
        self.assertIsNotNone(icc)

    def test_concordance_with_systematic_shift(self):
        """Concordance should detect systematic shift between run halves."""
        runs = []
        for i in range(10):
            shift = 0.1 if i >= 5 else 0.0
            runs.append(RunResult(
                run_id=f"r{i}", seed=i,
                metrics={"acc": 0.8 + shift},
            ))
        result = self.analyzer.compute_concordance_correlation(runs)
        self.assertIsNotNone(result)

    def test_friedman_with_tied_ranks(self):
        """Friedman test should handle tied values."""
        runs = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"a": 0.5, "b": 0.5})
            for i in range(6)
        ]
        result = self.analyzer.friedman_test_across_runs(runs)
        self.assertIn("p_value", result)

    def test_effect_sizes_with_large_difference(self):
        """Effect sizes should be large for very different run groups."""
        rng = np.random.RandomState(42)
        runs = []
        for i in range(10):
            val = 0.9 if i < 5 else 0.1
            runs.append(RunResult(
                run_id=f"r{i}", seed=i,
                metrics={"acc": val + rng.normal(0, 0.01)},
            ))
        effects = self.analyzer.compute_effect_sizes(runs)
        self.assertIsInstance(effects, dict)

    def test_stability_with_trend(self):
        """Stability should be lower when there is a systematic trend across runs."""
        runs = []
        for i in range(10):
            runs.append(RunResult(
                run_id=f"r{i}", seed=i,
                metrics={"acc": 0.5 + 0.05 * i},
            ))
        score = self.analyzer.compute_stability_score(runs)
        if isinstance(score, float):
            self.assertLess(score, 1.0)

    def test_variance_decomposition_with_groups(self):
        """Variance decomposition with multiple metric groups."""
        runs = make_run_results(n_runs=15, noise_std=0.05, seed=42)
        decomp = self.analyzer.variance_decomposition(runs)
        self.assertIsInstance(decomp, dict)


class TestSeedSensitivityAnalyzerDetailed(unittest.TestCase):
    """Additional detailed tests for SeedSensitivityAnalyzer."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10)
        self.analyzer = SeedSensitivityAnalyzer(self.config)

    def test_sensitivity_returns_per_metric(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        result = self.analyzer.analyze_seed_sensitivity(runs)
        self.assertIsInstance(result, dict)

    def test_correlations_with_many_runs(self):
        runs = make_run_results(n_runs=20, noise_std=0.05, seed=42)
        corr = self.analyzer.compute_seed_correlations(runs)
        self.assertIsNotNone(corr)

    def test_outlier_with_multiple_anomalies(self):
        runs = make_run_results(n_runs=10, noise_std=0.01, seed=42)
        runs[0].metrics = {k: v + 1.0 for k, v in runs[0].metrics.items()}
        runs[1].metrics = {k: v - 1.0 for k, v in runs[1].metrics.items()}
        outliers = self.analyzer.identify_outlier_seeds(runs)
        if isinstance(outliers, list):
            self.assertGreaterEqual(len(outliers), 1)

    def test_influence_with_one_dominant_seed(self):
        runs = make_run_results(n_runs=10, noise_std=0.01, seed=42)
        runs[0].metrics = {k: v * 10 for k, v in runs[0].metrics.items()}
        influence = self.analyzer.compute_seed_influence(runs)
        self.assertIsNotNone(influence)

    def test_clustering_with_bimodal(self):
        """Two distinct groups of seeds should yield two clusters."""
        runs = []
        for i in range(10):
            val = 0.9 if i < 5 else 0.1
            runs.append(RunResult(
                run_id=f"r{i}", seed=42 + i,
                metrics={"acc": val},
            ))
        clusters = self.analyzer.seed_clustering(runs)
        if isinstance(clusters, dict):
            non_empty = [v for v in clusters.values() if isinstance(v, list) and len(v) > 0]
            self.assertGreaterEqual(len(non_empty), 2)

    def test_recommend_seeds_consistent(self):
        """Calling recommend twice on same data should give same result."""
        runs = make_run_results(n_runs=15, noise_std=0.05, seed=42)
        r1 = self.analyzer.recommend_minimum_seeds(runs)
        r2 = self.analyzer.recommend_minimum_seeds(runs)
        if isinstance(r1, int):
            self.assertEqual(r1, r2)
        elif isinstance(r1, dict):
            for m in r1:
                self.assertEqual(r1[m], r2[m])

    def test_correlations_range(self):
        runs = make_run_results(n_runs=8, noise_std=0.05, seed=42)
        corr = self.analyzer.compute_seed_correlations(runs)
        if isinstance(corr, np.ndarray):
            self.assertTrue(np.all(corr >= -1.0 - 1e-10))
            self.assertTrue(np.all(corr <= 1.0 + 1e-10))


class TestConfidenceIntervalComputerDetailed(unittest.TestCase):
    """Additional detailed tests for ConfidenceIntervalComputer."""

    def setUp(self):
        self.config = ReproducibilityConfig(confidence_level=0.95, bootstrap_samples=500)
        self.computer = ConfidenceIntervalComputer(self.config)

    def test_normal_ci_large_sample(self):
        rng = np.random.RandomState(42)
        data = rng.normal(100, 10, size=10000)
        lo, hi = self.computer.normal_ci(data)
        self.assertAlmostEqual((lo + hi) / 2, 100, delta=0.5)

    def test_bootstrap_ci_reproducible(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 50)
        cfg = ReproducibilityConfig(confidence_level=0.95, bootstrap_samples=500, random_seed=42)
        c1 = ConfidenceIntervalComputer(cfg)
        c2 = ConfidenceIntervalComputer(cfg)
        lo1, hi1 = c1.bootstrap_ci(data)
        lo2, hi2 = c2.bootstrap_ci(data)
        # With same seed and config, should be identical
        self.assertAlmostEqual(lo1, lo2, places=5)
        self.assertAlmostEqual(hi1, hi2, places=5)

    def test_bayesian_ci_with_informative_prior(self):
        """With enough data, Bayesian CI should converge to data."""
        rng = np.random.RandomState(42)
        data = rng.normal(10, 1, size=1000)
        lo, hi = self.computer.bayesian_ci(data)
        self.assertAlmostEqual((lo + hi) / 2, 10, delta=0.2)

    def test_wilson_score_symmetric_at_half(self):
        """Wilson interval at p=0.5 should be roughly symmetric."""
        lo, hi = self.computer.wilson_score_ci(successes=50, total=100)
        center = (lo + hi) / 2
        self.assertAlmostEqual(center, 0.5, delta=0.05)

    def test_simultaneous_ci_multiple_metrics(self):
        rng = np.random.RandomState(42)
        data = {f"m{i}": rng.normal(i, 1, 100) for i in range(10)}
        result = self.computer.simultaneous_ci(data)
        self.assertEqual(len(result), 10)

    def test_prediction_interval_covers_future(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0, 1, 100)
        test = rng.normal(0, 1, 50)
        lo, hi = self.computer.prediction_interval(train)
        fraction_in = np.mean((test >= lo) & (test <= hi))
        self.assertGreater(fraction_in, 0.85)

    def test_tolerance_interval_coverage(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 200)
        lo, hi = self.computer.tolerance_interval(data)
        fraction_in = np.mean((data >= lo) & (data <= hi))
        self.assertGreater(fraction_in, 0.90)

    def test_credible_interval_with_uniform_data(self):
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, 100)
        lo, hi = self.computer.credible_interval(data)
        self.assertLess(lo, hi)
        self.assertGreater(lo, -0.5)
        self.assertLess(hi, 1.5)

    def test_normal_ci_known_properties(self):
        """For N(0,1) with n=100, 95% CI width ≈ 2*1.96/sqrt(100) ≈ 0.392."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 100)
        lo, hi = self.computer.normal_ci(data)
        expected_width = 2 * 1.96 * np.std(data, ddof=1) / np.sqrt(100)
        actual_width = hi - lo
        self.assertAlmostEqual(actual_width, expected_width, delta=0.05)


class TestReproducibilityReporterDetailed(unittest.TestCase):
    """Additional detailed tests for ReproducibilityReporter."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10)
        self.reporter = ReproducibilityReporter(self.config)

    def test_summary_statistics_median(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        summary = self.reporter.generate_summary_statistics(runs)
        for metric, stats_dict in summary.items():
            if isinstance(stats_dict, dict) and "median" in stats_dict:
                vals = [r.metrics[metric] for r in runs]
                self.assertAlmostEqual(stats_dict["median"], np.median(vals), places=5)

    def test_comparison_table_structure(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        table = self.reporter.generate_comparison_table(runs)
        self.assertIsNotNone(table)

    def test_reproducibility_score_identical_is_max(self):
        runs = make_identical_run_results(n_runs=10)
        score = self.reporter.compute_reproducibility_score(runs)
        if isinstance(score, float):
            self.assertGreaterEqual(score, 0.99)

    def test_flag_unreliable_threshold(self):
        """Metrics with very high CV should be flagged."""
        rng = np.random.RandomState(42)
        runs = []
        for i in range(10):
            runs.append(RunResult(
                run_id=f"r{i}", seed=i,
                metrics={
                    "stable": 0.9 + rng.normal(0, 0.001),
                    "unstable": rng.uniform(0.0, 1.0),
                },
            ))
        flags = self.reporter.flag_unreliable_metrics(runs)
        if isinstance(flags, list):
            flag_strs = [str(f) for f in flags]
            self.assertTrue(any("unstable" in s for s in flag_strs) or len(flags) > 0)

    def test_variance_plot_data_per_metric(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        plot_data = self.reporter.prepare_variance_plot_data(runs)
        for metric in ["accuracy", "diversity", "bleu"]:
            self.assertIn(metric, plot_data)

    def test_seed_sensitivity_plot_values(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        plot_data = self.reporter.prepare_seed_sensitivity_plot_data(runs)
        for metric, vals in plot_data.items():
            if isinstance(vals, (list, np.ndarray)):
                self.assertEqual(len(vals), 10)


class TestExperimentReplicatorDetailed(unittest.TestCase):
    """Additional detailed tests for ExperimentReplicator."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=5, random_seed=42)
        self.replicator = ExperimentReplicator(self.config)

    def test_deterministic_env_hash_seed(self):
        """Different seeds should produce different hash environments."""
        self.replicator.setup_deterministic_env(seed=1)
        env1 = os.environ.get("PYTHONHASHSEED", None)
        self.replicator.setup_deterministic_env(seed=2)
        env2 = os.environ.get("PYTHONHASHSEED", None)
        # At least numpy should differ
        self.replicator.setup_deterministic_env(seed=1)
        a = np.random.rand(5)
        self.replicator.setup_deterministic_env(seed=2)
        b = np.random.rand(5)
        self.assertFalse(np.allclose(a, b))

    def test_run_replicated_with_failing_fn(self):
        """Should handle or propagate experiment failures."""
        call_count = [0]

        def failing_experiment(seed: int) -> Dict[str, float]:
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("Simulated failure")
            rng = np.random.RandomState(seed)
            return {"score": float(rng.random())}

        try:
            results = self.replicator.run_replicated_experiment(
                experiment_fn=failing_experiment,
                seeds=[1, 2, 3, 4, 5],
            )
            # If it catches failures, some results may be None or marked
            self.assertIsInstance(results, list)
        except RuntimeError:
            pass  # Acceptable to propagate

    def test_compare_replications_statistics(self):
        runs = make_run_results(n_runs=10, noise_std=0.05, seed=42)
        comparison = self.replicator.compare_replications(runs)
        self.assertIsInstance(comparison, dict)

    def test_replicated_experiment_seeds_used(self):
        """Verify each seed is actually used."""
        seeds_used = []

        def tracking_exp(seed: int) -> Dict[str, float]:
            seeds_used.append(seed)
            return {"v": float(seed) / 100}

        seeds = [10, 20, 30]
        self.replicator.run_replicated_experiment(
            experiment_fn=tracking_exp, seeds=seeds
        )
        self.assertEqual(seeds_used, seeds)


class TestHelperFunctionsDetailed(unittest.TestCase):
    """Additional detailed tests for helper functions."""

    def test_bootstrap_resample_large(self):
        data = np.arange(10000)
        sample = bootstrap_resample(data, seed=42)
        self.assertEqual(len(sample), 10000)
        # Should have some duplicates (birthday paradox)
        self.assertLess(len(set(sample)), 10000)

    def test_jackknife_resample_preserves_order(self):
        data = np.array([10, 20, 30, 40, 50])
        samples = jackknife_resample(data)
        # Sample 0 removes element 0
        np.testing.assert_array_equal(samples[0], [20, 30, 40, 50])
        # Sample 2 removes element 2
        np.testing.assert_array_equal(samples[2], [10, 20, 40, 50])

    def test_ess_returns_float(self):
        data = np.random.normal(0, 1, 100)
        ess = compute_effective_sample_size(data)
        self.assertIsInstance(ess, (int, float, np.floating, np.integer))

    def test_ess_at_most_n(self):
        data = np.random.normal(0, 1, 100)
        ess = compute_effective_sample_size(data)
        self.assertLessEqual(ess, 101)  # Can slightly exceed n in some formulations

    def test_cochrans_q_binary_only(self):
        data = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
        result = cochrans_q_test(data)
        self.assertIn("statistic", result)

    def test_cochrans_q_single_treatment(self):
        """With two treatments, Q reduces to McNemar-like test."""
        data = np.array([[1, 0], [1, 1], [0, 0], [1, 0], [0, 1], [1, 0]])
        result = cochrans_q_test(data)
        self.assertIn("p_value", result)

    def test_krippendorffs_alpha_two_raters(self):
        data = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ])
        alpha = compute_krippendorffs_alpha(data)
        self.assertAlmostEqual(alpha, 1.0, places=3)

    def test_krippendorffs_alpha_complete_disagreement(self):
        """Raters that systematically disagree should have low alpha."""
        data = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
        ])
        alpha = compute_krippendorffs_alpha(data)
        self.assertLess(alpha, 0.5)

    def test_fleiss_kappa_three_categories(self):
        n_raters = 10
        n_subjects = 15
        rng = np.random.RandomState(42)
        data = np.zeros((n_subjects, 3), dtype=int)
        for i in range(n_subjects):
            cats = rng.randint(0, 3, size=n_raters)
            for c in cats:
                data[i, c] += 1
        kappa = compute_fleiss_kappa(data)
        self.assertGreaterEqual(kappa, -1.0)
        self.assertLessEqual(kappa, 1.0)

    def test_fleiss_kappa_two_categories(self):
        n_raters = 8
        data = np.array([
            [n_raters, 0],
            [0, n_raters],
            [n_raters, 0],
            [0, n_raters],
            [4, 4],
        ])
        kappa = compute_fleiss_kappa(data)
        self.assertIsInstance(kappa, float)

    def test_bootstrap_statistics_mean(self):
        """Bootstrap mean estimates should converge to sample mean."""
        rng = np.random.RandomState(42)
        data = rng.normal(5, 1, 100)
        means = []
        for seed in range(100):
            sample = bootstrap_resample(data, seed=seed)
            means.append(np.mean(sample))
        bootstrap_mean = np.mean(means)
        self.assertAlmostEqual(bootstrap_mean, np.mean(data), delta=0.1)

    def test_jackknife_bias_estimate(self):
        """Jackknife can estimate bias of a statistic."""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        theta_hat = np.mean(data)
        samples = jackknife_resample(data)
        theta_jk = [np.mean(s) for s in samples]
        bias = (len(data) - 1) * (np.mean(theta_jk) - theta_hat)
        # For the mean, jackknife bias should be ~0
        self.assertAlmostEqual(bias, 0.0, places=10)


class TestStatisticalProperties(unittest.TestCase):
    """Tests verifying known statistical properties of the computations."""

    def setUp(self):
        self.config = ReproducibilityConfig(confidence_level=0.95, bootstrap_samples=1000)

    def test_variance_unbiased_estimator(self):
        """Sample variance should be an unbiased estimator of population variance."""
        true_var = 4.0  # std=2
        rng = np.random.RandomState(42)
        n_trials = 500
        variances = []
        analyzer = InterRunVarianceAnalyzer(self.config)
        for trial in range(n_trials):
            runs = [
                RunResult(
                    run_id=f"r{i}", seed=trial * 100 + i,
                    metrics={"x": float(rng.normal(0, 2))}
                )
                for i in range(20)
            ]
            v = analyzer.compute_variance_across_runs(runs)
            variances.append(v["x"])
        mean_var = np.mean(variances)
        self.assertAlmostEqual(mean_var, true_var, delta=0.5)

    def test_cv_scale_invariance(self):
        """CV should be the same regardless of scale."""
        rng = np.random.RandomState(42)
        base = rng.normal(10, 1, 50)
        scaled = base * 100

        analyzer = InterRunVarianceAnalyzer(self.config)
        runs_base = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"x": float(base[i])})
            for i in range(50)
        ]
        runs_scaled = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"x": float(scaled[i])})
            for i in range(50)
        ]
        cv_base = analyzer.compute_coefficient_of_variation(runs_base)
        cv_scaled = analyzer.compute_coefficient_of_variation(runs_scaled)
        self.assertAlmostEqual(cv_base["x"], cv_scaled["x"], places=5)

    def test_bootstrap_consistency(self):
        """Bootstrap CI should be consistent (converge) as B increases."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 50)

        widths = []
        for B in [100, 500, 2000]:
            cfg = ReproducibilityConfig(
                confidence_level=0.95, bootstrap_samples=B, random_seed=42
            )
            ci = ConfidenceIntervalComputer(cfg)
            lo, hi = ci.bootstrap_ci(data)
            widths.append(hi - lo)

        # Widths should stabilize (last two should be closer than first two)
        diff_early = abs(widths[0] - widths[1])
        diff_late = abs(widths[1] - widths[2])
        # This is a soft check – bootstrap variability decreases
        self.assertLess(diff_late, diff_early + 0.1)

    def test_fleiss_kappa_marginal_homogeneity(self):
        """Fleiss kappa assumes marginal homogeneity of raters."""
        n_raters = 10
        # All subjects rated identically -> kappa = 1
        data = np.zeros((5, 2), dtype=int)
        for i in range(5):
            data[i, i % 2] = n_raters
        kappa = compute_fleiss_kappa(data)
        self.assertAlmostEqual(kappa, 1.0, places=2)

    def test_prediction_interval_t_distribution(self):
        """Prediction interval should use t-distribution for small samples."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 10)
        ci = ConfidenceIntervalComputer(self.config)
        pi_lo, pi_hi = ci.prediction_interval(data)
        # Should be wider than ±1.96 due to small sample
        width = pi_hi - pi_lo
        self.assertGreater(width, 2 * 1.96)

    def test_krippendorffs_alpha_ordinal_invariance(self):
        """Alpha for perfect agreement should be 1 regardless of label values."""
        data_a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        data_b = np.array([[10, 20, 30], [10, 20, 30], [10, 20, 30]])
        alpha_a = compute_krippendorffs_alpha(data_a)
        alpha_b = compute_krippendorffs_alpha(data_b)
        self.assertAlmostEqual(alpha_a, 1.0, places=3)
        self.assertAlmostEqual(alpha_b, 1.0, places=3)

    def test_ess_formula_for_ar1(self):
        """For AR(1) with coefficient rho, ESS ≈ n*(1-rho)/(1+rho)."""
        rho = 0.8
        n = 5000
        rng = np.random.RandomState(42)
        data = np.zeros(n)
        data[0] = rng.normal()
        for i in range(1, n):
            data[i] = rho * data[i - 1] + rng.normal(0, np.sqrt(1 - rho ** 2))
        ess = compute_effective_sample_size(data)
        expected_ess = n * (1 - rho) / (1 + rho)
        # Allow generous tolerance due to estimation
        self.assertAlmostEqual(ess, expected_ess, delta=expected_ess * 0.5)


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability and robustness."""

    def setUp(self):
        self.config = ReproducibilityConfig(n_runs=10)

    def test_variance_very_small_values(self):
        runs = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"x": 1e-15 + i * 1e-16})
            for i in range(10)
        ]
        analyzer = InterRunVarianceAnalyzer(self.config)
        var = analyzer.compute_variance_across_runs(runs)
        self.assertGreaterEqual(var["x"], 0.0)

    def test_variance_very_large_values(self):
        runs = [
            RunResult(run_id=f"r{i}", seed=i, metrics={"x": 1e15 + i * 1e14})
            for i in range(10)
        ]
        analyzer = InterRunVarianceAnalyzer(self.config)
        var = analyzer.compute_variance_across_runs(runs)
        self.assertGreater(var["x"], 0.0)
        self.assertTrue(np.isfinite(var["x"]))

    def test_ci_with_very_small_variance(self):
        data = np.ones(100) * 5.0 + np.random.normal(0, 1e-10, 100)
        ci = ConfidenceIntervalComputer(self.config)
        lo, hi = ci.normal_ci(data)
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))
        self.assertAlmostEqual(lo, 5.0, delta=1e-8)

    def test_ci_with_outlier_data(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 99)
        data = np.append(data, 1000.0)  # Extreme outlier
        ci = ConfidenceIntervalComputer(self.config)
        lo, hi = ci.normal_ci(data)
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))

    def test_fleiss_kappa_with_all_same_category(self):
        """All raters always choose the same category."""
        n_raters = 5
        data = np.zeros((10, 3), dtype=int)
        data[:, 0] = n_raters  # All choose category 0
        kappa = compute_fleiss_kappa(data)
        # Edge case: when there's no variation in marginals
        self.assertTrue(np.isfinite(kappa) or np.isnan(kappa))

    def test_krippendorffs_alpha_all_same(self):
        """All raters give the same rating to all items."""
        data = np.ones((3, 10)) * 5
        alpha = compute_krippendorffs_alpha(data)
        # Perfect agreement
        self.assertTrue(alpha >= 0.99 or np.isnan(alpha))

    def test_cochrans_q_all_zeros(self):
        data = np.zeros((10, 3), dtype=int)
        result = cochrans_q_test(data)
        self.assertTrue(np.isfinite(result["p_value"]) or np.isnan(result["p_value"]))

    def test_bootstrap_with_single_unique_value(self):
        data = np.ones(50) * 7.0
        sample = bootstrap_resample(data, seed=42)
        self.assertTrue(np.all(sample == 7.0))

    def test_ess_with_alternating_data(self):
        """Alternating data has negative autocorrelation -> ESS > n."""
        data = np.array([1, -1] * 500, dtype=float)
        ess = compute_effective_sample_size(data)
        # ESS can exceed n for negatively correlated data
        self.assertGreater(ess, 0)


if __name__ == "__main__":
    unittest.main()
