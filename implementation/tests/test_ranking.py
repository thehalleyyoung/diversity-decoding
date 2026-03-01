"""Tests for Bradley-Terry ranking and Bayesian sign tests."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.ranking import (
    BradleyTerryModel,
    BayesianSignTest,
    MetricDiscriminativePower,
    rank_metrics_by_discriminative_power,
)


class TestBradleyTerry:
    def test_clear_ranking(self):
        """A always beats B, B always beats C → A > B > C."""
        comparisons = [
            ("A", "B", 1.0),
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("B", "C", 1.0),
        ]
        bt = BradleyTerryModel()
        result = bt.fit(comparisons)
        ranking_names = [name for name, _ in result.ranking]
        assert ranking_names[0] == "A"
        assert ranking_names[-1] == "C"

    def test_equal_items(self):
        """Equal comparisons → similar strengths."""
        comparisons = [
            ("A", "B", 0.5),
            ("B", "A", 0.5),
            ("A", "C", 0.5),
            ("C", "A", 0.5),
            ("B", "C", 0.5),
            ("C", "B", 0.5),
        ]
        bt = BradleyTerryModel()
        result = bt.fit(comparisons)
        # Strengths should be approximately equal
        strengths = list(result.probabilities.values())
        assert max(strengths) - min(strengths) < 0.2

    def test_prediction(self):
        """Predictions should be consistent with ranking."""
        comparisons = [
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
        ]
        bt = BradleyTerryModel()
        result = bt.fit(comparisons)
        # P(A beats C) > P(B beats C)
        p_ac = bt.predict_probability(result, "A", "C")
        p_bc = bt.predict_probability(result, "B", "C")
        assert p_ac > p_bc

    def test_single_item(self):
        """Single item should return valid result."""
        bt = BradleyTerryModel()
        result = bt.fit([("A", "B", 1.0)])
        assert "A" in result.strengths
        assert "B" in result.strengths

    def test_convergence(self):
        """Model should converge within reasonable iterations."""
        comparisons = [("A", "B", 0.7), ("B", "C", 0.6), ("A", "C", 0.8)]
        bt = BradleyTerryModel(max_iter=100)
        result = bt.fit(comparisons)
        assert result.convergence_iterations <= 100


class TestBayesianSignTest:
    def test_clear_winner(self):
        """All positive differences → left wins."""
        diffs = [0.5, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.4]
        test = BayesianSignTest(rope_width=0.05)
        result = test.test(diffs)
        assert result.decision == "left"
        assert result.p_left > 0.5

    def test_clear_loser(self):
        """All negative differences → right wins."""
        diffs = [-0.5, -0.3, -0.4, -0.6, -0.2]
        test = BayesianSignTest(rope_width=0.05)
        result = test.test(diffs)
        assert result.decision == "right"
        assert result.p_right > 0.5

    def test_rope(self):
        """All differences within ROPE → practical equivalence."""
        diffs = [0.001, -0.002, 0.003, -0.001, 0.002]
        test = BayesianSignTest(rope_width=0.01)
        result = test.test(diffs)
        assert result.decision == "rope"
        assert result.p_rope > 0.5

    def test_empty_input(self):
        test = BayesianSignTest()
        result = test.test([])
        assert result.n_comparisons == 0
        assert result.decision == "undecided"

    def test_probabilities_sum_to_one(self):
        diffs = [0.1, -0.2, 0.3, 0.0, -0.1]
        test = BayesianSignTest(rope_width=0.05)
        result = test.test(diffs)
        assert abs(result.p_left + result.p_rope + result.p_right - 1.0) < 1e-6


class TestMetricDiscriminativePower:
    def test_discriminating_vs_constant_metric(self):
        """A metric with variation should have higher power than a constant."""
        config_data = {
            "temp_0.3": {
                "good_metric": [0.1, 0.12, 0.11, 0.13],
                "bad_metric": [0.5, 0.5, 0.5, 0.5],
            },
            "temp_0.7": {
                "good_metric": [0.5, 0.52, 0.48, 0.51],
                "bad_metric": [0.5, 0.5, 0.5, 0.5],
            },
            "temp_1.0": {
                "good_metric": [0.8, 0.82, 0.79, 0.81],
                "bad_metric": [0.5, 0.5, 0.5, 0.5],
            },
        }
        analyzer = MetricDiscriminativePower(effect_threshold=0.1)
        results = analyzer.analyse(config_data)
        assert results["good_metric"].discriminative_power > results["bad_metric"].discriminative_power

    def test_empty_input(self):
        analyzer = MetricDiscriminativePower()
        results = analyzer.analyse({"only_one": {"m": [1.0]}})
        assert len(results) == 0


class TestRankMetricsPipeline:
    def test_full_pipeline(self):
        """Integration test for the full ranking pipeline."""
        config_data = {
            "temp_0.3": {
                "distinct_2": [0.2, 0.22, 0.19],
                "self_bleu": [0.8, 0.78, 0.81],
                "entropy": [1.0, 1.1, 0.95],
            },
            "temp_0.7": {
                "distinct_2": [0.5, 0.48, 0.52],
                "self_bleu": [0.5, 0.52, 0.48],
                "entropy": [2.0, 1.9, 2.1],
            },
            "temp_1.0": {
                "distinct_2": [0.8, 0.79, 0.82],
                "self_bleu": [0.2, 0.22, 0.19],
                "entropy": [3.0, 2.9, 3.1],
            },
        }
        result = rank_metrics_by_discriminative_power(config_data)
        assert "discriminative_power" in result
        assert "bradley_terry_ranking" in result
        assert "bayesian_pairwise_tests" in result
        # All metrics should appear
        assert len(result["discriminative_power"]) == 3
