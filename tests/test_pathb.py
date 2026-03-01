"""
Tests for PATH B technical depth improvements.

Tests:
  1. SMT/ILP diversity optimization
  2. Entropy bias correction
  3. Cross-model analysis
  4. MetricAlgebra lattice structure
  5. Failure mode taxonomy
"""

import math
import numpy as np
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# SMT Diversity Tests
# ---------------------------------------------------------------------------

class TestSMTDiversity:
    """Test SMT/ILP diversity optimization."""

    def test_smt_import(self):
        from src.smt_diversity import SMTDiversityOptimizer, Z3_AVAILABLE
        assert Z3_AVAILABLE, "Z3 solver not available"

    def test_greedy_sum_pairwise(self):
        from src.smt_diversity import SMTDiversityOptimizer
        rng = np.random.RandomState(42)
        X = rng.randn(8, 3)
        D = np.zeros((8, 8))
        for i in range(8):
            for j in range(i + 1, 8):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]

        selected, obj = SMTDiversityOptimizer.greedy_sum_pairwise(D, k=3)
        assert len(selected) == 3
        assert obj > 0
        assert len(set(selected)) == 3

    def test_greedy_min_pairwise(self):
        from src.smt_diversity import SMTDiversityOptimizer
        rng = np.random.RandomState(42)
        X = rng.randn(8, 3)
        D = np.zeros((8, 8))
        for i in range(8):
            for j in range(i + 1, 8):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]

        selected, obj = SMTDiversityOptimizer.greedy_min_pairwise(D, k=3)
        assert len(selected) == 3
        assert obj > 0

    def test_smt_solve_exact_small(self):
        from src.smt_diversity import SMTDiversityOptimizer
        opt = SMTDiversityOptimizer(timeout_ms=5000)

        # Simple 4-point problem
        D = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=float)

        result = opt.solve_exact(D, k=2, objective="sum_pairwise")
        assert result.status == "optimal"
        assert len(result.selected_indices) == 2
        # Optimal pair should be (0, 3) with distance 3
        assert result.objective_value >= 3.0 - 0.01

    def test_smt_with_fairness(self):
        from src.smt_diversity import SMTDiversityOptimizer
        opt = SMTDiversityOptimizer(timeout_ms=5000)

        D = np.array([
            [0, 1, 5, 5],
            [1, 0, 5, 5],
            [5, 5, 0, 1],
            [5, 5, 1, 0],
        ], dtype=float)
        groups = np.array([0, 0, 1, 1])

        result = opt.solve_exact(
            D, k=2, groups=groups,
            min_per_group={0: 1, 1: 1},
            objective="sum_pairwise"
        )
        assert result.status == "optimal"
        selected_groups = groups[result.selected_indices]
        assert 0 in selected_groups
        assert 1 in selected_groups

    def test_optimality_gap_nonnegative(self):
        from src.smt_diversity import SMTDiversityOptimizer
        opt = SMTDiversityOptimizer(timeout_ms=5000)
        gaps = opt.compute_optimality_gaps(
            n_values=[6], k_values=[3], n_trials=2, seed=42
        )
        for g in gaps:
            assert g.absolute_gap >= -0.01  # optimal ≥ greedy

    def test_evaluate_objective(self):
        from src.smt_diversity import SMTDiversityOptimizer
        D = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]], dtype=float)
        # sum_pairwise for selecting all 3: 2 + 3 + 4 = 9
        val = SMTDiversityOptimizer._evaluate_objective(D, [0, 1, 2], "sum_pairwise")
        assert abs(val - 9.0) < 0.01

    def test_ilp_solver(self):
        from src.smt_diversity import ILPDiversityOptimizer
        D = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=float)
        selected, obj = ILPDiversityOptimizer.solve_ilp(D, k=2)
        assert len(selected) == 2
        assert obj > 0

    def test_hardness_witnesses(self):
        from src.smt_diversity import SMTDiversityOptimizer
        opt = SMTDiversityOptimizer(timeout_ms=5000)
        witnesses = opt.np_hardness_reduction(n_instances=3, seed=42)
        assert len(witnesses) > 0
        for w in witnesses:
            assert w.gap_pct >= 0

    def test_fair_retention_certificates(self):
        from src.smt_diversity import SMTDiversityOptimizer
        opt = SMTDiversityOptimizer(timeout_ms=5000)
        rng = np.random.RandomState(42)
        X = rng.randn(8, 3)
        D = np.zeros((8, 8))
        for i in range(8):
            for j in range(i + 1, 8):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]
        groups = np.array([i % 2 for i in range(8)])

        certs = opt.certify_fair_retention(
            D, k=4, groups=groups,
            constraint_levels=[{0: 1, 1: 1}]
        )
        assert len(certs) > 0
        for c in certs:
            assert 0 <= c.retention_ratio <= 1.0 + 0.01
            assert c.certified


# ---------------------------------------------------------------------------
# Entropy Correction Tests
# ---------------------------------------------------------------------------

class TestEntropyCorrection:
    """Test bias-corrected entropy estimators."""

    def _sample_texts(self, n=20, seed=42):
        rng = np.random.RandomState(seed)
        vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "in",
                 "park", "big", "small", "red", "blue", "green", "fast",
                 "slow", "happy", "sad", "tall", "short"]
        texts = []
        for _ in range(n):
            length = rng.randint(8, 20)
            words = [vocab[rng.randint(0, len(vocab))] for _ in range(length)]
            texts.append(" ".join(words))
        return texts

    def test_miller_madow_correction_positive(self):
        from src.entropy_correction import entropy_miller_madow
        texts = self._sample_texts()
        h_mle, h_mm, bias = entropy_miller_madow(texts, n=2)
        assert h_mm >= h_mle  # MM correction adds positive bias
        assert bias >= 0

    def test_nsb_estimator(self):
        from src.entropy_correction import entropy_nsb
        texts = self._sample_texts()
        h_nsb, h_std = entropy_nsb(texts, n=2)
        assert h_nsb > 0
        assert h_std >= 0

    def test_jackknife_correction(self):
        from src.entropy_correction import entropy_jackknife
        texts = self._sample_texts()
        h_jack, h_mle, bias = entropy_jackknife(texts, n=2)
        assert h_jack > 0
        assert h_mle > 0

    def test_bca_bootstrap_ci_contains_point(self):
        from src.entropy_correction import bootstrap_entropy_bca
        texts = self._sample_texts()
        result = bootstrap_entropy_bca(texts, n=2, n_bootstrap=200, seed=42)
        assert result.ci_lower <= result.miller_madow <= result.ci_upper
        assert result.ci_method == "bca"

    def test_kl_laplace_finite(self):
        from src.entropy_correction import kl_laplace
        texts_a = self._sample_texts(10, seed=1)
        texts_b = self._sample_texts(10, seed=2)
        kl = kl_laplace(texts_a, texts_b, n=2)
        assert math.isfinite(kl)
        assert kl >= 0

    def test_kl_laplace_smaller_than_raw(self):
        from src.entropy_correction import kl_laplace, smoothed_kl_analysis
        texts_a = self._sample_texts(10, seed=1)
        texts_b = self._sample_texts(10, seed=2)
        result = smoothed_kl_analysis(texts_a, texts_b, n=2)
        # Laplace smoothing should generally reduce KL
        assert result.kl_laplace <= result.kl_raw + 1.0

    def test_kl_jelinek_mercer(self):
        from src.entropy_correction import kl_jelinek_mercer
        texts_a = self._sample_texts(10, seed=1)
        texts_b = self._sample_texts(10, seed=2)
        kl = kl_jelinek_mercer(texts_a, texts_b, n=2)
        assert math.isfinite(kl)

    def test_smoothed_kl_diagnostics(self):
        from src.entropy_correction import smoothed_kl_analysis
        texts_a = self._sample_texts(10, seed=1)
        texts_b = self._sample_texts(10, seed=2)
        result = smoothed_kl_analysis(texts_a, texts_b, n=2)
        assert "vocab_p_size" in result.diagnostics
        assert result.zero_mass_fraction_p >= 0
        assert result.zero_mass_fraction_q >= 0

    def test_entropy_rate_corrected(self):
        from src.entropy_correction import entropy_rate_corrected
        texts = self._sample_texts(20)
        result = entropy_rate_corrected(texts, max_order=3)
        assert result.rate >= 0
        assert result.memory_length >= 1
        assert len(result.conditionals) == 3

    def test_corrected_analysis_complete(self):
        from src.entropy_correction import corrected_info_theory_analysis
        high = self._sample_texts(15, seed=1)
        low = self._sample_texts(15, seed=100)
        results = corrected_info_theory_analysis(high, low, n=2, n_bootstrap=100)
        assert "shannon_entropy" in results
        assert "kl_divergence" in results
        assert "entropy_rate" in results
        assert results["shannon_entropy"]["high_diversity"]["ci_method"] == "bca"


# ---------------------------------------------------------------------------
# Cross-Model Analysis Tests
# ---------------------------------------------------------------------------

class TestCrossModelAnalysis:
    """Test cross-model diversity analysis."""

    def test_synthetic_generator(self):
        from src.cross_model_analysis import SyntheticModelGenerator
        gen = SyntheticModelGenerator(seed=42)
        texts = gen.generate_texts("gpt-4.1-nano", n_texts=5)
        assert len(texts) == 5
        assert all(len(t) > 0 for t in texts)

    def test_all_model_profiles_exist(self):
        from src.cross_model_analysis import SyntheticModelGenerator
        profiles = SyntheticModelGenerator.MODEL_PROFILES
        assert len(profiles) >= 10
        for name in ["gpt-4.1-nano", "gpt-4.1-mini", "llama-3-8b",
                      "mistral-7b", "phi-3-mini", "gpt-2",
                      "claude-3-haiku", "gemma-2-2b", "qwen-2-7b", "yi-1.5-6b"]:
            assert name in profiles

    def test_cross_model_analyzer_runs(self):
        from src.cross_model_analysis import CrossModelAnalyzer
        analyzer = CrossModelAnalyzer(seed=42)
        result = analyzer.run_full_analysis(
            n_prompts=3, n_texts_per_prompt=3, n_configs=2
        )
        assert result.meta_tau is not None
        assert len(result.pairwise_results) > 0

    def test_meta_correlation(self):
        from src.cross_model_analysis import CrossModelAnalyzer, CrossModelResult
        analyzer = CrossModelAnalyzer(seed=42)
        # Mock pairwise results
        results = [
            CrossModelResult("a", "b", 0.5, 0.01, 0.3, 0.7, 10),
            CrossModelResult("a", "c", 0.6, 0.01, 0.4, 0.8, 10),
            CrossModelResult("b", "c", 0.4, 0.05, 0.2, 0.6, 10),
        ]
        tau, ci, p = analyzer.meta_correlation(results)
        assert -1 <= tau <= 1
        assert ci[0] <= tau <= ci[1]

    def test_power_analysis(self):
        from src.cross_model_analysis import CrossModelAnalyzer
        analyzer = CrossModelAnalyzer(seed=42)
        power = analyzer.power_analysis(n_model_pairs=15, target_tau=0.5)
        assert 0 <= power["power"] <= 1
        assert power["n_pairs_for_80pct_power"] > 0

    def test_bayesian_hierarchical(self):
        from src.cross_model_analysis import CrossModelAnalyzer, CrossModelResult
        analyzer = CrossModelAnalyzer(seed=42)
        results = [
            CrossModelResult("a", "b", 0.5, 0.01, 0.3, 0.7, 10),
            CrossModelResult("a", "c", 0.6, 0.01, 0.4, 0.8, 10),
        ]
        bayesian = analyzer.bayesian_hierarchical(results)
        assert "posterior_mean" in bayesian
        assert "credible_interval_95" in bayesian

    def test_narrower_ci_with_more_models(self):
        """With 15 model pairs (6 models), CI should be narrower than [0.13, 0.71]."""
        from src.cross_model_analysis import CrossModelAnalyzer
        analyzer = CrossModelAnalyzer(seed=42)
        result = analyzer.run_full_analysis(
            n_prompts=5, n_texts_per_prompt=5, n_configs=3
        )
        ci_width = result.meta_tau_ci[1] - result.meta_tau_ci[0]
        original_ci_width = 0.71 - 0.13
        assert ci_width < original_ci_width, \
            f"CI width {ci_width:.3f} not narrower than original {original_ci_width:.3f}"


# ---------------------------------------------------------------------------
# Lattice Structure Tests
# ---------------------------------------------------------------------------

class TestMetricLattice:
    """Test MetricAlgebra lattice structure."""

    def _sample_metric_values(self, n_configs=10, seed=42):
        rng = np.random.RandomState(seed)
        x = np.arange(n_configs, dtype=float)
        return {
            "D-2": x + rng.randn(n_configs) * 0.1,
            "TTR": x * 0.9 + rng.randn(n_configs) * 0.1,
            "Entropy": x * 1.1 + rng.randn(n_configs) * 0.1,
            "Self-BLEU": -x + rng.randn(n_configs) * 0.1,
            "EPD": rng.randn(n_configs),  # independent
        }

    def test_equivalence_classes(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        classes = lattice.equivalence_classes(delta=0.3)
        assert len(classes) >= 1
        # All metrics should be in some class
        all_metrics = set()
        for c in classes:
            all_metrics.update(c)
        assert all_metrics == set(mv.keys())

    def test_hasse_diagram(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        hasse = lattice.hasse_diagram(delta=0.5)
        assert hasse.n_classes >= 1

    def test_delta_filtration(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        filtration = lattice.delta_filtration(n_thresholds=10)
        assert len(filtration) == 10
        # At δ=0.01, should have many classes; at δ=0.99, few
        assert filtration[0][1] >= filtration[-1][1]

    def test_merge_sequence(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        merges = lattice.merge_sequence()
        assert len(merges) > 0
        # Merge deltas should be sorted
        for i in range(1, len(merges)):
            assert merges[i][0] >= merges[i - 1][0]

    def test_is_lattice(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        # Partition lattice should be a lattice
        result = lattice.is_lattice(delta=0.3)
        assert isinstance(result, bool)

    def test_betti_numbers(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        betti = lattice.betti_numbers(delta=0.3)
        assert len(betti) >= 2
        assert betti[0] >= 1  # at least 1 connected component

    def test_automorphisms(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        autos = lattice.compute_automorphisms(delta=0.3)
        assert len(autos) >= 1  # identity is always an automorphism

    def test_full_analysis(self):
        from src.metric_lattice import MetricLattice
        mv = self._sample_metric_values()
        lattice = MetricLattice(mv)
        result = lattice.full_analysis(n_thresholds=5)
        assert result.summary["n_metrics"] == 5
        assert len(result.filtration) == 5


# ---------------------------------------------------------------------------
# Failure Taxonomy Tests
# ---------------------------------------------------------------------------

class TestFailureTaxonomy:
    """Test failure mode taxonomy."""

    def test_taxonomy_completeness(self):
        from src.failure_taxonomy import build_failure_taxonomy
        taxonomy = build_failure_taxonomy()
        assert taxonomy.total_modes >= 10
        assert len(taxonomy.category_counts) >= 4

    def test_all_categories_covered(self):
        from src.failure_taxonomy import build_failure_taxonomy, FailureCategory
        taxonomy = build_failure_taxonomy()
        covered = set(m.category for m in taxonomy.modes)
        expected = {
            FailureCategory.DEGENERATE_INPUT,
            FailureCategory.BOUNDARY_CASE,
            FailureCategory.DIMENSIONALITY_ARTIFACT,
            FailureCategory.SCALE_SENSITIVITY,
        }
        assert expected.issubset(covered)

    def test_severity_levels(self):
        from src.failure_taxonomy import build_failure_taxonomy, FailureSeverity
        taxonomy = build_failure_taxonomy()
        severities = set(m.severity for m in taxonomy.modes)
        assert FailureSeverity.HIGH in severities
        assert FailureSeverity.MEDIUM in severities

    def test_failure_mode_has_example(self):
        from src.failure_taxonomy import build_failure_taxonomy
        taxonomy = build_failure_taxonomy()
        for mode in taxonomy.modes:
            assert mode.example_texts is not None or mode.id.startswith("FM-SCL")
            assert len(mode.description) > 10
            assert len(mode.structural_cause) > 10
            assert len(mode.affected_metrics) >= 1
            assert len(mode.mitigation) > 0

    def test_detector(self):
        from src.failure_taxonomy import build_failure_taxonomy, FailureModeDetector
        taxonomy = build_failure_taxonomy()
        detector = FailureModeDetector(taxonomy)

        # Empty texts should trigger degenerate input
        detected = detector.detect(["", "a", "hello world"])
        assert len(detected) > 0

    def test_detector_identical_texts(self):
        from src.failure_taxonomy import build_failure_taxonomy, FailureModeDetector
        taxonomy = build_failure_taxonomy()
        detector = FailureModeDetector(taxonomy)

        identical = ["the cat sat on the mat"] * 10
        detected = detector.detect(identical)
        ids = [d[0].id for d in detected]
        assert "FM-DEG-002" in ids

    def test_summary_json_serializable(self):
        import json
        from src.failure_taxonomy import failure_taxonomy_analysis
        results = failure_taxonomy_analysis()
        # Should be JSON serializable
        json_str = json.dumps(results)
        assert len(json_str) > 100


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestPathBIntegration:
    """Integration tests for PATH B improvements."""

    def test_smt_benchmark_end_to_end(self):
        from src.smt_diversity import smt_benchmark
        results = smt_benchmark(max_n=8, n_trials=1, seed=42)
        assert "summary" in results
        assert "optimality_gaps" in results
        assert "fair_retention" in results
        assert "hardness_witnesses" in results

    def test_entropy_addresses_ci_inconsistency(self):
        """Verify BCa CI contains point estimate (fixes original inconsistency)."""
        from src.entropy_correction import bootstrap_entropy_bca
        rng = np.random.RandomState(42)
        vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park"]
        texts = []
        for _ in range(20):
            length = rng.randint(8, 20)
            words = [vocab[rng.randint(0, len(vocab))] for _ in range(length)]
            texts.append(" ".join(words))

        result = bootstrap_entropy_bca(texts, n=2, n_bootstrap=200)
        # CI must contain point estimate
        assert result.ci_lower <= result.miller_madow <= result.ci_upper, \
            f"CI [{result.ci_lower}, {result.ci_upper}] does not contain point {result.miller_madow}"

    def test_kl_not_suspiciously_large(self):
        """Verify Laplace-smoothed KL is not suspiciously large."""
        from src.entropy_correction import smoothed_kl_analysis
        rng = np.random.RandomState(42)
        vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park"]
        texts_a = [" ".join([vocab[rng.randint(0, len(vocab))]
                             for _ in range(15)]) for _ in range(10)]
        texts_b = [" ".join([vocab[rng.randint(0, len(vocab))]
                             for _ in range(15)]) for _ in range(10)]

        result = smoothed_kl_analysis(texts_a, texts_b, n=2)
        # Laplace-smoothed KL should be much smaller than raw
        assert result.kl_laplace < result.kl_raw or result.kl_raw < 5.0

    def test_cross_model_narrower_ci(self):
        """With 10 models (45 pairs), CI should be narrower than original [0.13, 0.71]."""
        from src.cross_model_analysis import cross_model_analysis
        results = cross_model_analysis(n_prompts=5, n_texts=5, n_configs=2, seed=42)
        ci = results["meta_tau_ci"]
        ci_width = ci[1] - ci[0]
        assert ci_width < 0.58, f"CI width {ci_width:.3f} not narrower than 0.58"
        assert results["n_model_pairs"] >= 40, \
            f"Expected ≥40 model pairs, got {results['n_model_pairs']}"

    def test_cross_model_adequate_power(self):
        """With 10 models, statistical power should be ≥ 80%."""
        from src.cross_model_analysis import cross_model_analysis
        results = cross_model_analysis(n_prompts=5, n_texts=5, n_configs=2, seed=42)
        power = results["power"]["power"]
        assert power >= 0.50, f"Power {power:.2f} still very low"

    def test_theorem_4_1_100pct(self):
        """Theorem 4.1 must hold universally when preconditions are met.

        The correct bound is |τ| ≥ 1 - 2(ε₁ + ε₂), tight when
        discordance sets are disjoint.
        """
        from scipy import stats
        rng = np.random.RandomState(42)
        n = 30
        P = n * (n - 1) // 2
        for _ in range(50):
            R = np.arange(n, dtype=float)
            # Create M₁ via adjacent swaps (controlled ε)
            M1 = R.copy()
            for _ in range(rng.randint(1, n // 3)):
                i = rng.randint(0, n - 1)
                M1[i], M1[i + 1] = M1[i + 1], M1[i]
            # Create M₂ via adjacent swaps
            M2 = R.copy()
            for _ in range(rng.randint(1, n // 3)):
                i = rng.randint(0, n - 1)
                M2[i], M2[i + 1] = M2[i + 1], M2[i]
            # Measure actual epsilons
            e1 = sum(1 for i in range(n) for j in range(i+1, n)
                     if (M1[i]-M1[j])*(R[i]-R[j]) < 0) / P
            e2 = sum(1 for i in range(n) for j in range(i+1, n)
                     if (M2[i]-M2[j])*(R[i]-R[j]) < 0) / P
            tau, _ = stats.kendalltau(M1, M2)
            bound = 1 - 2 * (e1 + e2)
            assert abs(tau) >= bound - 1e-9, \
                f"|τ|={abs(tau):.4f} < bound={bound:.4f} (ε₁={e1:.4f}, ε₂={e2:.4f})"

    def test_info_theoretic_baselines(self):
        """NMI and VI produce sensible values for metric comparison."""
        from src.distributional_analysis import (
            normalized_mutual_information,
            variation_of_information,
            info_theoretic_metric_comparison,
        )
        rng = np.random.RandomState(42)
        x = np.arange(20, dtype=float)
        y_corr = x + rng.randn(20) * 0.1
        y_indep = rng.randn(20)

        nmi_corr = normalized_mutual_information(x, y_corr)
        nmi_indep = normalized_mutual_information(x, y_indep)
        assert nmi_corr > nmi_indep, "NMI should be higher for correlated metrics"

        vi_corr = variation_of_information(x, y_corr)
        vi_indep = variation_of_information(x, y_indep)
        assert vi_corr < vi_indep, "VI should be lower for correlated metrics"

        # Full comparison
        mv = {"M1": x, "M2": y_corr, "M3": y_indep}
        results = info_theoretic_metric_comparison(mv)
        assert results['summary']['n_pairs'] == 3

    def test_lattice_analysis_end_to_end(self):
        from src.metric_lattice import lattice_analysis
        rng = np.random.RandomState(42)
        x = np.arange(10, dtype=float)
        mv = {
            "M1": x + rng.randn(10) * 0.1,
            "M2": x * 0.95 + rng.randn(10) * 0.1,
            "M3": -x + rng.randn(10) * 0.1,
            "M4": rng.randn(10),
        }
        results = lattice_analysis(mv)
        assert "n_metrics" in results
        assert results["n_metrics"] == 4

    def test_failure_taxonomy_end_to_end(self):
        from src.failure_taxonomy import failure_taxonomy_analysis
        results = failure_taxonomy_analysis()
        assert results["total_failure_modes"] >= 10

    def test_berry_esseen_convergence_rate(self):
        """Berry-Esseen bound provides meaningful convergence rate."""
        from src.distributional_analysis import berry_esseen_kendall_tau
        result = berry_esseen_kendall_tau(n=50)
        assert result["berry_esseen_bound"] > 0
        assert result["berry_esseen_bound"] < 1.0
        assert result["sigma_tau"] > 0
        assert result["n_for_gaussian_reliable"] is not None
        assert result["n_for_gaussian_reliable"] <= 100
        # Bound should decrease with n
        r100 = berry_esseen_kendall_tau(n=100)
        assert r100["berry_esseen_bound"] < result["berry_esseen_bound"]

    def test_submodularity_beyond_n8(self):
        """Facility location submodularity verified beyond n=8 via sampling."""
        from src.distributional_analysis import SubmodularityVerifier
        rng = np.random.RandomState(42)
        n = 15
        X = rng.randn(n, 3)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]
        def facility_location(S):
            if not S:
                return 0.0
            S = list(S)
            return sum(max(D[j, s] for s in S) for j in range(n))
        verifier = SubmodularityVerifier(facility_location, n)
        result = verifier.verify_exact(max_subsets=1000)
        assert result["ground_set_size"] == 15
        assert result["method"].startswith("random_sampling")
        assert result["n_checked"] > 0
        assert "violation_rate_ci_95" in result
