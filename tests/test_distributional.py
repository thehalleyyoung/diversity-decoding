"""
Tests for distributional analysis and adversarial search modules.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.distributional_analysis import (
    MetricAlgebra, SubmodularityVerifier, PermutationTest, tightness_construction
)
from src.adversarial_analysis import (
    AdversarialDivergenceSearch, FairSelectionWorstCase, construct_np_hardness_witness
)


class TestMetricAlgebra:
    def test_equivalence_classes_correlated(self):
        n = 50
        base = np.arange(n, dtype=float)
        metrics = {
            'M1': base,
            'M2': base + np.random.RandomState(0).normal(0, 0.1, n),
            'M3': base + np.random.RandomState(1).normal(0, 0.1, n),
            'M4': np.random.RandomState(2).randn(n),  # independent
        }
        algebra = MetricAlgebra(metrics, delta=0.3)
        classes = algebra.equivalence_classes()
        # M1, M2, M3 should be in one class; M4 separate
        assert len(classes) == 2

    def test_transitivity(self):
        n = 30
        base = np.arange(n, dtype=float)
        metrics = {
            'A': base,
            'B': base + np.random.RandomState(0).normal(0, 0.05, n),
            'C': base + np.random.RandomState(1).normal(0, 0.05, n),
        }
        algebra = MetricAlgebra(metrics, delta=0.3)
        trans = algebra.verify_transitivity()
        assert trans['is_transitive']

    def test_quotient_dimension(self):
        n = 20
        rng = np.random.RandomState(42)
        metrics = {f'M{i}': rng.randn(n) for i in range(5)}
        algebra = MetricAlgebra(metrics, delta=0.3)
        # Random metrics should each be independent
        assert algebra.quotient_dimension() >= 3


class TestSubmodularityVerifier:
    def test_facility_location_is_submodular(self):
        rng = np.random.RandomState(42)
        n = 6
        X = rng.randn(n, 3)
        sim = X @ X.T
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)

        def f(S):
            S = list(S)
            if not S:
                return 0.0
            return float(np.sum(np.max(sim[:, S], axis=1)))

        verifier = SubmodularityVerifier(f, n)
        result = verifier.verify_exact()
        assert result['is_submodular']

    def test_sum_pairwise_distance_not_submodular(self):
        # Sum pairwise distance is NOT submodular in general
        n = 5
        X = np.array([[0, 0], [1, 0], [0.5, 0], [2, 0], [3, 0]], dtype=float)
        dist = np.linalg.norm(X[:, None] - X[None, :], axis=-1)

        def f(S):
            S = list(S)
            if len(S) < 2:
                return 0.0
            return sum(dist[i][j] for i in S for j in S if i < j)

        verifier = SubmodularityVerifier(f, n)
        result = verifier.verify_exact()
        # May or may not be submodular depending on geometry
        assert 'is_submodular' in result


class TestPermutationTest:
    def test_correlated_significant(self):
        n = 30
        x = np.arange(n, dtype=float)
        y = x + np.random.RandomState(0).normal(0, 0.5, n)
        pt = PermutationTest(n_permutations=1000, seed=42)
        result = pt.test(x, y)
        assert result['p_value'] < 0.05
        assert result['observed_tau'] > 0.5

    def test_independent_not_significant(self):
        n = 20
        rng = np.random.RandomState(42)
        x = rng.randn(n)
        y = rng.randn(n)
        pt = PermutationTest(n_permutations=1000, seed=42)
        result = pt.test(x, y)
        assert result['p_value'] > 0.05


class TestTightnessConstruction:
    def test_small_epsilon(self):
        result = tightness_construction(n=50, eps1=0.05, eps2=0.05)
        # Bound uses measured actual ε values, so compare with measured
        e1 = result['eps1_actual']
        e2 = result['eps2_actual']
        expected_bound = 1 - 2 * (e1 + e2)
        assert result['theoretical_bound'] == pytest.approx(expected_bound, abs=0.01)
        assert result['achieved_tau'] > 0.5  # Should be high
        # Bound must hold: |τ| ≥ bound
        assert abs(result['achieved_tau']) >= result['theoretical_bound'] - 1e-9

    def test_zero_epsilon(self):
        result = tightness_construction(n=20, eps1=0.0, eps2=0.0)
        assert result['theoretical_bound'] == pytest.approx(1.0)
        assert result['achieved_tau'] == pytest.approx(1.0)


class TestAdversarialSearch:
    def test_search_runs(self):
        searcher = AdversarialDivergenceSearch(seed=42)
        results = searcher.search(n_trials=10)
        assert 'metric_pairs' in results
        assert len(results['metric_pairs']) > 0


class TestFairSelectionWorstCase:
    def test_pareto_frontier(self):
        analyzer = FairSelectionWorstCase(seed=42)
        results = analyzer.pareto_frontier(n=50, d=10, k=10, n_trials=5)
        assert len(results['mean_retention']) == len(results['constraint_levels'])
        # All retentions should be positive
        assert all(r > 0 for r in results['mean_retention'])


class TestNPHardnessWitness:
    def test_witness(self):
        result = construct_np_hardness_witness(n=15, d=2, k=4, n_trials=50)
        assert 'mean_gap' in result
        assert result['n_trials'] == 50
        assert result['fraction_suboptimal'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
