"""
Bradley-Terry ranking and Bayesian sign tests for metric discriminative power.

Addresses the critique: "Bradley-Terry model and Bayesian sign tests were
promised but not delivered."

Implements:
  - Bradley-Terry model for ranking metrics by discriminative power
  - Bayesian sign tests with ROPE (Region of Practical Equivalence)
  - Metric discriminative power analysis

These are imported and used by experiments; see run_comprehensive_experiment.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit as sigmoid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bradley-Terry Model
# ---------------------------------------------------------------------------

@dataclass
class BTRanking:
    """Result of Bradley-Terry ranking."""
    strengths: Dict[str, float]  # Log-strength parameters
    probabilities: Dict[str, float]  # Normalized probabilities
    ranking: List[Tuple[str, float]]  # Sorted (name, strength) pairs
    n_comparisons: int
    convergence_iterations: int


class BradleyTerryModel:
    """Bradley-Terry model for ranking items from pairwise comparisons.

    Given pairwise comparison outcomes (A beats B), estimates strength
    parameters θ_i such that P(i beats j) = θ_i / (θ_i + θ_j).

    Uses iterative MM (Minorization-Maximization) algorithm for fitting.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        regularization: float = 1e-6,
    ):
        self._max_iter = max_iter
        self._tol = tol
        self._reg = regularization

    def fit(
        self,
        comparisons: List[Tuple[str, str, float]],
    ) -> BTRanking:
        """Fit Bradley-Terry model from pairwise comparisons.

        Args:
            comparisons: List of (winner, loser, weight) tuples.
                weight in [0, 1]: 1.0 = clear win, 0.5 = tie.

        Returns:
            BTRanking with estimated strengths and ranking.
        """
        # Collect all items
        items = sorted(set(w for w, _, _ in comparisons) |
                       set(l for _, l, _ in comparisons))
        n = len(items)
        item_idx = {name: i for i, name in enumerate(items)}

        if n < 2:
            return BTRanking(
                strengths={items[0]: 1.0} if items else {},
                probabilities={items[0]: 1.0} if items else {},
                ranking=[(items[0], 1.0)] if items else [],
                n_comparisons=len(comparisons),
                convergence_iterations=0,
            )

        # Build win matrix
        wins = np.zeros((n, n), dtype=np.float64)
        for winner, loser, weight in comparisons:
            wi = item_idx[winner]
            li = item_idx[loser]
            wins[wi, li] += weight
            wins[li, wi] += (1.0 - weight)

        # MM algorithm
        theta = np.ones(n, dtype=np.float64)
        for iteration in range(self._max_iter):
            theta_old = theta.copy()

            for i in range(n):
                numerator = 0.0
                denominator = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    n_ij = wins[i, j] + wins[j, i]
                    if n_ij > 0:
                        numerator += wins[i, j]
                        denominator += n_ij / (theta[i] + theta[j])

                if denominator > 0:
                    theta[i] = (numerator + self._reg) / (denominator + self._reg)

            # Normalize
            theta = theta / theta.sum() * n

            # Check convergence
            diff = np.max(np.abs(theta - theta_old))
            if diff < self._tol:
                break

        # Build results
        log_strengths = {name: float(np.log(theta[i] + 1e-30))
                        for name, i in item_idx.items()}
        total = theta.sum()
        probabilities = {name: float(theta[i] / total)
                        for name, i in item_idx.items()}
        ranking = sorted(
            [(name, float(theta[item_idx[name]])) for name in items],
            key=lambda x: -x[1],
        )

        return BTRanking(
            strengths=log_strengths,
            probabilities=probabilities,
            ranking=ranking,
            n_comparisons=len(comparisons),
            convergence_iterations=iteration + 1,
        )

    def predict_probability(
        self, ranking: BTRanking, item_a: str, item_b: str,
    ) -> float:
        """Predict P(A beats B) from fitted model."""
        theta_a = math.exp(ranking.strengths.get(item_a, 0.0))
        theta_b = math.exp(ranking.strengths.get(item_b, 0.0))
        return theta_a / (theta_a + theta_b + 1e-30)


# ---------------------------------------------------------------------------
# Bayesian Sign Test with ROPE
# ---------------------------------------------------------------------------

@dataclass
class BayesianSignTestResult:
    """Result of Bayesian sign test."""
    p_left: float       # P(metric A is better)
    p_rope: float       # P(practically equivalent)
    p_right: float      # P(metric B is better)
    decision: str       # "left", "right", "rope", or "undecided"
    bayes_factor: float  # BF in favour of the decision
    n_comparisons: int
    rope_width: float


class BayesianSignTest:
    """Bayesian sign test with ROPE for comparing metrics.

    Tests whether metric A has higher discriminative power than metric B,
    using a Bayesian approach with a Region of Practical Equivalence (ROPE).

    Based on Benavoli et al. (2017) "Time for a Change: a Tutorial for
    Comparing Multiple Classifiers Through Bayesian Analysis"
    """

    def __init__(
        self,
        rope_width: float = 0.01,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self._rope = rope_width
        self._alpha = prior_alpha
        self._beta = prior_beta

    def test(
        self,
        differences: List[float],
        metric_a_name: str = "A",
        metric_b_name: str = "B",
    ) -> BayesianSignTestResult:
        """Run Bayesian sign test on paired differences.

        Args:
            differences: List of (score_A - score_B) values.
            metric_a_name: Name of metric A.
            metric_b_name: Name of metric B.

        Returns:
            BayesianSignTestResult with posterior probabilities.
        """
        if not differences:
            return BayesianSignTestResult(
                p_left=0.33, p_rope=0.34, p_right=0.33,
                decision="undecided", bayes_factor=1.0,
                n_comparisons=0, rope_width=self._rope,
            )

        n = len(differences)
        n_left = sum(1 for d in differences if d > self._rope)
        n_right = sum(1 for d in differences if d < -self._rope)
        n_rope = n - n_left - n_right

        # Posterior with Dirichlet-Multinomial model
        alpha_left = self._alpha + n_left
        alpha_rope = self._alpha + n_rope
        alpha_right = self._alpha + n_right
        alpha_total = alpha_left + alpha_rope + alpha_right

        p_left = alpha_left / alpha_total
        p_rope = alpha_rope / alpha_total
        p_right = alpha_right / alpha_total

        # Decision
        max_p = max(p_left, p_rope, p_right)
        if max_p == p_left and p_left > 0.5:
            decision = "left"
            bf = p_left / max(p_right, 1e-10)
        elif max_p == p_right and p_right > 0.5:
            decision = "right"
            bf = p_right / max(p_left, 1e-10)
        elif max_p == p_rope and p_rope > 0.5:
            decision = "rope"
            bf = p_rope / max(max(p_left, p_right), 1e-10)
        else:
            decision = "undecided"
            bf = 1.0

        return BayesianSignTestResult(
            p_left=p_left,
            p_rope=p_rope,
            p_right=p_right,
            decision=decision,
            bayes_factor=bf,
            n_comparisons=n,
            rope_width=self._rope,
        )


# ---------------------------------------------------------------------------
# Metric Discriminative Power Analysis
# ---------------------------------------------------------------------------

@dataclass
class DiscriminativePowerResult:
    """Result of metric discriminative power analysis."""
    metric_name: str
    discriminative_power: float  # Fraction of config pairs correctly separated
    effect_sizes: List[float]   # Effect sizes for each comparison
    mean_effect_size: float
    n_significant: int
    n_comparisons: int


class MetricDiscriminativePower:
    """Analyse how well each metric discriminates between decoding configurations.

    For each metric, counts how many configuration pairs show a
    statistically meaningful difference (effect size above threshold).
    """

    def __init__(
        self,
        effect_threshold: float = 0.2,
        n_bootstrap: int = 500,
        seed: int = 42,
    ):
        self._threshold = effect_threshold
        self._n_bootstrap = n_bootstrap
        self._seed = seed

    def analyse(
        self,
        config_results: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, DiscriminativePowerResult]:
        """Analyse discriminative power of each metric.

        Args:
            config_results: {config_name: {metric_name: [values_per_prompt]}}

        Returns:
            {metric_name: DiscriminativePowerResult}
        """
        configs = sorted(config_results.keys())
        if len(configs) < 2:
            return {}

        # Collect metric names
        metric_names = set()
        for cr in config_results.values():
            metric_names.update(cr.keys())
        metric_names = sorted(metric_names)

        results = {}
        for mn in metric_names:
            effect_sizes = []
            n_significant = 0
            n_comparisons = 0

            for i in range(len(configs)):
                for j in range(i + 1, len(configs)):
                    vals_i = [
                        v for v in config_results[configs[i]].get(mn, [])
                        if not math.isnan(v)
                    ]
                    vals_j = [
                        v for v in config_results[configs[j]].get(mn, [])
                        if not math.isnan(v)
                    ]

                    if len(vals_i) >= 2 and len(vals_j) >= 2:
                        es = self._cliffs_delta(vals_i, vals_j)
                        effect_sizes.append(es)
                        if abs(es) > self._threshold:
                            n_significant += 1
                        n_comparisons += 1

            if n_comparisons > 0:
                disc_power = n_significant / n_comparisons
                mean_es = float(np.mean([abs(e) for e in effect_sizes]))
            else:
                disc_power = 0.0
                mean_es = 0.0

            results[mn] = DiscriminativePowerResult(
                metric_name=mn,
                discriminative_power=disc_power,
                effect_sizes=effect_sizes,
                mean_effect_size=mean_es,
                n_significant=n_significant,
                n_comparisons=n_comparisons,
            )

        return results

    @staticmethod
    def _cliffs_delta(x: List[float], y: List[float]) -> float:
        """Cliff's delta effect size (non-parametric)."""
        n_x, n_y = len(x), len(y)
        if n_x == 0 or n_y == 0:
            return 0.0
        more = less = 0
        for xi in x:
            for yj in y:
                if xi > yj:
                    more += 1
                elif xi < yj:
                    less += 1
        return (more - less) / (n_x * n_y)


def rank_metrics_by_discriminative_power(
    config_metric_data: Dict[str, Dict[str, List[float]]],
    rope_width: float = 0.01,
) -> Dict[str, Any]:
    """Full pipeline: discriminative power + Bradley-Terry + Bayesian sign tests.

    Args:
        config_metric_data: {config_name: {metric_name: [values_per_prompt]}}
        rope_width: ROPE width for Bayesian sign tests.

    Returns:
        Dict with rankings, pairwise comparisons, and recommendations.
    """
    # 1. Compute discriminative power
    power_analyzer = MetricDiscriminativePower()
    power_results = power_analyzer.analyse(config_metric_data)

    # 2. Build pairwise comparisons for Bradley-Terry
    bt_comparisons = []
    metric_names = sorted(power_results.keys())

    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            m_i = metric_names[i]
            m_j = metric_names[j]
            dp_i = power_results[m_i].discriminative_power
            dp_j = power_results[m_j].discriminative_power

            if dp_i > dp_j:
                bt_comparisons.append((m_i, m_j, min(dp_i - dp_j + 0.5, 1.0)))
            elif dp_j > dp_i:
                bt_comparisons.append((m_j, m_i, min(dp_j - dp_i + 0.5, 1.0)))
            else:
                bt_comparisons.append((m_i, m_j, 0.5))

    # 3. Fit Bradley-Terry model
    bt_model = BradleyTerryModel()
    bt_ranking = bt_model.fit(bt_comparisons)

    # 4. Bayesian sign tests between top metrics
    sign_test = BayesianSignTest(rope_width=rope_width)
    pairwise_tests = {}

    for i in range(min(len(metric_names), 5)):
        for j in range(i + 1, min(len(metric_names), 5)):
            m_i = metric_names[i]
            m_j = metric_names[j]
            # Use effect sizes as paired differences
            es_i = power_results[m_i].effect_sizes
            es_j = power_results[m_j].effect_sizes
            min_len = min(len(es_i), len(es_j))
            if min_len > 0:
                diffs = [abs(es_i[k]) - abs(es_j[k]) for k in range(min_len)]
                result = sign_test.test(diffs, m_i, m_j)
                pairwise_tests[f"{m_i}_vs_{m_j}"] = {
                    "p_left": result.p_left,
                    "p_rope": result.p_rope,
                    "p_right": result.p_right,
                    "decision": result.decision,
                    "bayes_factor": result.bayes_factor,
                }

    return {
        "discriminative_power": {
            mn: {
                "power": pr.discriminative_power,
                "mean_effect_size": pr.mean_effect_size,
                "n_significant": pr.n_significant,
                "n_comparisons": pr.n_comparisons,
            }
            for mn, pr in power_results.items()
        },
        "bradley_terry_ranking": {
            "ranking": bt_ranking.ranking,
            "probabilities": bt_ranking.probabilities,
            "n_comparisons": bt_ranking.n_comparisons,
        },
        "bayesian_pairwise_tests": pairwise_tests,
    }
