"""
Bayesian statistical comparison methods for the Diversity Decoding Arena.

Provides posterior estimation, ROPE analysis, Bradley-Terry ranking,
hierarchical modeling, sequential analysis, and MCMC diagnostics for
rigorous comparison of decoding algorithms.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import optimize, special, stats


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecisionType(Enum):
    """Possible outcomes of a ROPE analysis."""
    LEFT = "left"
    ROPE = "rope"
    RIGHT = "right"
    UNDECIDED = "undecided"


class PriorType(Enum):
    """Supported prior families."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    BETA = "beta"
    JEFFREYS = "jeffreys"
    WEAKLY_INFORMATIVE = "weakly_informative"


class IntervalMethod(Enum):
    """Credible-interval computation methods."""
    HDI = "hdi"
    ETI = "eti"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CredibleInterval:
    """A Bayesian credible interval."""
    lower: float
    upper: float
    alpha: float
    method: str

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def level(self) -> float:
        return 1.0 - self.alpha

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:
        return (
            f"CredibleInterval({100 * self.level:.0f}% {self.method}: "
            f"[{self.lower:.6f}, {self.upper:.6f}])"
        )


@dataclass
class PosteriorEstimate:
    """Full posterior summary."""
    mean: float
    std: float
    samples: np.ndarray
    credible_intervals: Dict[float, CredibleInterval]
    distribution_type: str
    median: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    mode: float = 0.0
    effective_sample_size: float = 0.0

    def __post_init__(self) -> None:
        if self.samples is not None and len(self.samples) > 0:
            self.median = float(np.median(self.samples))
            if len(self.samples) > 2:
                self.skewness = float(stats.skew(self.samples))
                self.kurtosis = float(stats.kurtosis(self.samples))
            self._estimate_mode()

    def _estimate_mode(self) -> None:
        if self.samples is None or len(self.samples) < 10:
            self.mode = self.mean
            return
        try:
            kde = stats.gaussian_kde(self.samples)
            x_grid = np.linspace(
                np.min(self.samples), np.max(self.samples), 512
            )
            densities = kde(x_grid)
            self.mode = float(x_grid[np.argmax(densities)])
        except Exception:
            self.mode = self.mean

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    def probability_above(self, threshold: float) -> float:
        return float(np.mean(self.samples > threshold))

    def probability_below(self, threshold: float) -> float:
        return float(np.mean(self.samples < threshold))

    def summary(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "mode": self.mode,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "hdi_lower": self.credible_intervals.get(
                0.95, CredibleInterval(0, 0, 0.05, "hdi")
            ).lower,
            "hdi_upper": self.credible_intervals.get(
                0.95, CredibleInterval(0, 0, 0.05, "hdi")
            ).upper,
        }


@dataclass
class ROPEResult:
    """Result of a Region Of Practical Equivalence analysis."""
    probability_left: float
    probability_rope: float
    probability_right: float
    rope_low: float
    rope_high: float
    decision: str
    confidence: float = 0.0
    posterior_samples: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        probs = [
            self.probability_left,
            self.probability_rope,
            self.probability_right,
        ]
        self.confidence = max(probs)
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            if total > 0:
                self.probability_left /= total
                self.probability_rope /= total
                self.probability_right /= total

    @property
    def max_probability_region(self) -> str:
        mapping = {
            self.probability_left: "left",
            self.probability_rope: "rope",
            self.probability_right: "right",
        }
        return mapping[max(mapping.keys())]

    def summary_string(self) -> str:
        return (
            f"ROPE analysis: P(left)={self.probability_left:.4f}, "
            f"P(ROPE)={self.probability_rope:.4f}, "
            f"P(right)={self.probability_right:.4f} -> {self.decision}"
        )


@dataclass
class ModelRanking:
    """Ranking of multiple models/algorithms."""
    algorithm_names: List[str]
    ranks: np.ndarray
    probabilities: np.ndarray
    posterior_samples: Optional[np.ndarray] = None
    strength_params: Optional[np.ndarray] = None
    credible_intervals: Optional[Dict[str, CredibleInterval]] = None

    def top_k(self, k: int = 3) -> List[Tuple[str, float, float]]:
        indices = np.argsort(self.ranks)[:k]
        result = []
        for idx in indices:
            result.append((
                self.algorithm_names[idx],
                float(self.ranks[idx]),
                float(self.probabilities[idx]),
            ))
        return result

    def rank_probabilities(self) -> Dict[str, Dict[int, float]]:
        if self.posterior_samples is None:
            return {}
        n_algorithms = len(self.algorithm_names)
        result: Dict[str, Dict[int, float]] = {}
        for i, name in enumerate(self.algorithm_names):
            rank_counts: Dict[int, float] = {}
            algo_samples = self.posterior_samples[:, i]
            for rank in range(1, n_algorithms + 1):
                rank_counts[rank] = 0.0
            sorted_indices = np.argsort(-self.posterior_samples, axis=1)
            for sample_idx in range(len(sorted_indices)):
                rank_of_algo = int(np.where(sorted_indices[sample_idx] == i)[0][0]) + 1
                rank_counts[rank_of_algo] += 1.0
            total = len(sorted_indices)
            for rank in rank_counts:
                rank_counts[rank] /= total
            result[name] = rank_counts
        return result


@dataclass
class ComparisonResult:
    """Result of comparing two algorithms."""
    algorithm_a: str
    algorithm_b: str
    metric_name: str
    posterior_diff: PosteriorEstimate
    rope_result: ROPEResult
    bayes_factor: float
    effect_size: PosteriorEstimate
    p_a_better: float
    p_b_better: float
    p_equivalent: float
    raw_scores_a: Optional[np.ndarray] = None
    raw_scores_b: Optional[np.ndarray] = None

    def winner(self, threshold: float = 0.95) -> Optional[str]:
        if self.p_a_better >= threshold:
            return self.algorithm_a
        if self.p_b_better >= threshold:
            return self.algorithm_b
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "algorithm_a": self.algorithm_a,
            "algorithm_b": self.algorithm_b,
            "metric": self.metric_name,
            "mean_diff": self.posterior_diff.mean,
            "p_a_better": self.p_a_better,
            "p_b_better": self.p_b_better,
            "p_equivalent": self.p_equivalent,
            "bayes_factor": self.bayes_factor,
            "effect_size_mean": self.effect_size.mean,
            "decision": self.rope_result.decision,
        }


@dataclass
class SequentialResult:
    """Result of sequential Bayesian analysis."""
    stopped_early: bool
    n_observations: int
    final_posterior: PosteriorEstimate
    decision: str
    evidence_trajectory: List[float]
    threshold_reached_at: Optional[int]
    bayes_factor_trajectory: List[float]


@dataclass
class HierarchicalResult:
    """Result of a hierarchical Bayesian model."""
    group_posteriors: Dict[str, PosteriorEstimate]
    global_posterior: PosteriorEstimate
    between_group_variance: PosteriorEstimate
    within_group_variance: PosteriorEstimate
    shrinkage_factors: Dict[str, float]
    convergence_diagnostics: Dict[str, float]


@dataclass
class MetaAnalysisResult:
    """Result of a Bayesian meta-analysis."""
    combined_posterior: PosteriorEstimate
    heterogeneity: PosteriorEstimate
    study_weights: Dict[str, float]
    forest_data: List[Dict[str, Any]]
    i_squared: float
    prediction_interval: CredibleInterval


@dataclass
class CalibrationResult:
    """Result of calibration checking."""
    expected_proportions: np.ndarray
    observed_proportions: np.ndarray
    calibration_error: float
    max_calibration_error: float
    brier_score: float
    log_loss: float
    reliability_data: Dict[str, np.ndarray]


@dataclass
class InformationCriteriaResult:
    """Information criteria for model comparison."""
    waic: float
    waic_se: float
    loo: float
    loo_se: float
    p_waic: float
    p_loo: float
    pointwise_waic: np.ndarray
    pointwise_loo: np.ndarray
    pareto_k: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _validate_scores(scores: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Ensure scores are a 1-D float array with no NaNs."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    mask = np.isfinite(arr)
    if not mask.all():
        warnings.warn(
            f"Dropping {(~mask).sum()} non-finite values from scores."
        )
        arr = arr[mask]
    if len(arr) == 0:
        raise ValueError("No finite scores remaining after filtering.")
    return arr


def _log_sum_exp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(a_max)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _stable_log(x: float, eps: float = 1e-300) -> float:
    return math.log(max(x, eps))


def _normal_log_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return -np.inf
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _welch_dof(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """Welch-Satterthwaite degrees of freedom."""
    sa = var_a / n_a
    sb = var_b / n_b
    numerator = (sa + sb) ** 2
    denominator = sa ** 2 / (n_a - 1) + sb ** 2 / (n_b - 1)
    if denominator < 1e-300:
        return float(n_a + n_b - 2)
    return numerator / denominator


# ---------------------------------------------------------------------------
# BayesianSignTest
# ---------------------------------------------------------------------------

class BayesianSignTest:
    """
    Bayesian version of the sign test using a Dirichlet-Multinomial model.

    The test classifies each paired observation into one of three categories:
    left wins, tie (within ROPE), right wins. A Dirichlet posterior over
    these three probabilities is then estimated.
    """

    def __init__(
        self,
        rope_low: float = -0.01,
        rope_high: float = 0.01,
        prior_strength: float = 1.0,
    ) -> None:
        if rope_low > rope_high:
            rope_low, rope_high = rope_high, rope_low
        self.rope_low = rope_low
        self.rope_high = rope_high
        self.prior_strength = max(prior_strength, 1e-10)
        self._prior_alpha = np.array([
            prior_strength, prior_strength, prior_strength
        ])

    # ----- public ----------------------------------------------------------

    def test(
        self,
        scores_a: Union[np.ndarray, List[float]],
        scores_b: Union[np.ndarray, List[float]],
        n_samples: int = 50_000,
    ) -> ROPEResult:
        """
        Run the Bayesian sign test on paired observations.

        Parameters
        ----------
        scores_a, scores_b : array-like
            Paired performance scores for the two algorithms.
        n_samples : int
            Number of posterior Dirichlet samples.

        Returns
        -------
        ROPEResult
        """
        a = _validate_scores(scores_a)
        b = _validate_scores(scores_b)
        if len(a) != len(b):
            n = min(len(a), len(b))
            warnings.warn(
                f"Truncating to {n} paired observations."
            )
            a, b = a[:n], b[:n]

        diffs = a - b
        counts = self._count_categories(diffs)
        posterior_samples = self._compute_posterior(counts, n_samples)

        p_left = float(np.mean(posterior_samples[:, 0] > posterior_samples[:, 1])
                        * np.mean(posterior_samples[:, 0] > posterior_samples[:, 2])
                        if False else np.mean(
                            (posterior_samples[:, 0] > posterior_samples[:, 1]) &
                            (posterior_samples[:, 0] > posterior_samples[:, 2])
                        ))
        p_rope = float(np.mean(
            (posterior_samples[:, 1] > posterior_samples[:, 0]) &
            (posterior_samples[:, 1] > posterior_samples[:, 2])
        ))
        p_right = float(np.mean(
            (posterior_samples[:, 2] > posterior_samples[:, 0]) &
            (posterior_samples[:, 2] > posterior_samples[:, 1])
        ))

        total = p_left + p_rope + p_right
        if total > 0:
            p_left /= total
            p_rope /= total
            p_right /= total

        decision = self._make_decision(p_left, p_rope, p_right)

        return ROPEResult(
            probability_left=p_left,
            probability_rope=p_rope,
            probability_right=p_right,
            rope_low=self.rope_low,
            rope_high=self.rope_high,
            decision=decision,
            posterior_samples=posterior_samples,
        )

    def multi_comparison(
        self,
        all_scores: Dict[str, Union[np.ndarray, List[float]]],
        n_samples: int = 50_000,
    ) -> Dict[Tuple[str, str], ROPEResult]:
        """
        Run pairwise Bayesian sign tests for all algorithm pairs.

        Parameters
        ----------
        all_scores : dict mapping algorithm name -> score array
        n_samples : int

        Returns
        -------
        Dict mapping (name_a, name_b) -> ROPEResult
        """
        names = sorted(all_scores.keys())
        results: Dict[Tuple[str, str], ROPEResult] = {}
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i >= j:
                    continue
                results[(name_a, name_b)] = self.test(
                    all_scores[name_a],
                    all_scores[name_b],
                    n_samples=n_samples,
                )
        return results

    def plot_simplex(self, result: ROPEResult) -> Dict[str, Any]:
        """
        Prepare data for a ternary / simplex plot of posterior probabilities.

        Returns a dictionary with vertices, posterior sample coordinates,
        and summary statistics suitable for plotting.
        """
        if result.posterior_samples is None:
            raise ValueError("ROPEResult has no posterior samples.")

        samples = result.posterior_samples
        n = len(samples)

        cart_x = samples[:, 2] + 0.5 * samples[:, 1]
        cart_y = samples[:, 1] * (math.sqrt(3) / 2.0)

        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2.0],
        ])

        mean_point = np.array([
            float(np.mean(cart_x)),
            float(np.mean(cart_y)),
        ])

        contour_levels = [0.5, 0.75, 0.9, 0.95]
        contour_data: Dict[float, Dict[str, float]] = {}
        for level in contour_levels:
            distances = np.sqrt(
                (cart_x - mean_point[0]) ** 2 +
                (cart_y - mean_point[1]) ** 2
            )
            radius = float(np.quantile(distances, level))
            contour_data[level] = {
                "center_x": mean_point[0],
                "center_y": mean_point[1],
                "radius": radius,
            }

        return {
            "vertices": vertices.tolist(),
            "cart_x": cart_x.tolist(),
            "cart_y": cart_y.tolist(),
            "mean_point": mean_point.tolist(),
            "probabilities": {
                "left": result.probability_left,
                "rope": result.probability_rope,
                "right": result.probability_right,
            },
            "contours": contour_data,
            "n_samples": n,
            "decision": result.decision,
        }

    # ----- private ---------------------------------------------------------

    def _count_categories(self, diffs: np.ndarray) -> np.ndarray:
        """Count left-wins, ties, and right-wins."""
        left = int(np.sum(diffs < self.rope_low))
        right = int(np.sum(diffs > self.rope_high))
        rope = int(len(diffs) - left - right)
        return np.array([left, rope, right], dtype=np.float64)

    def _compute_posterior(
        self,
        counts: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Sample from the Dirichlet posterior."""
        alpha_post = self._prior_alpha + counts
        alpha_post = np.maximum(alpha_post, 1e-10)
        rng = np.random.default_rng(hash(tuple(counts)) % (2**31))
        samples = rng.dirichlet(alpha_post, size=n_samples)
        return samples

    @staticmethod
    def _make_decision(
        p_left: float,
        p_rope: float,
        p_right: float,
        threshold: float = 0.95,
    ) -> str:
        if p_left >= threshold:
            return DecisionType.LEFT.value
        if p_right >= threshold:
            return DecisionType.RIGHT.value
        if p_rope >= threshold:
            return DecisionType.ROPE.value
        return DecisionType.UNDECIDED.value


# ---------------------------------------------------------------------------
# BayesianComparison
# ---------------------------------------------------------------------------

class BayesianComparison:
    """
    Comprehensive Bayesian comparison framework for decoding algorithms.

    Supports pairwise and multi-algorithm comparison, posterior estimation
    with multiple prior families, ROPE analysis, Bradley-Terry and
    Plackett-Luce ranking, sequential analysis, meta-analysis,
    hierarchical modeling, and full MCMC diagnostics.
    """

    def __init__(
        self,
        prior_type: str = "uniform",
        n_samples: int = 10_000,
        rope_width: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.prior_type = PriorType(prior_type)
        self.n_samples = n_samples
        self.rope_width = rope_width
        self.rope_low = -rope_width / 2.0
        self.rope_high = rope_width / 2.0
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._sign_test = BayesianSignTest(
            rope_low=self.rope_low,
            rope_high=self.rope_high,
        )

    # ======================================================================
    # Two-algorithm comparison
    # ======================================================================

    def compare_two(
        self,
        scores_a: Union[np.ndarray, List[float]],
        scores_b: Union[np.ndarray, List[float]],
        metric_name: str = "score",
        algorithm_a: str = "A",
        algorithm_b: str = "B",
    ) -> ComparisonResult:
        """
        Full Bayesian comparison of two algorithms.

        Computes:
        - Posterior on the difference (A - B)
        - ROPE analysis
        - Bayes factor
        - Posterior effect size

        Parameters
        ----------
        scores_a, scores_b : array-like
        metric_name : str
        algorithm_a, algorithm_b : str

        Returns
        -------
        ComparisonResult
        """
        a = _validate_scores(scores_a)
        b = _validate_scores(scores_b)

        diff_posterior = self._posterior_of_difference(a, b)
        rope = self.rope_analysis(
            diff_posterior.samples, self.rope_low, self.rope_high
        )
        bf = self.bayes_factor(a, b)
        es = self.effect_size_posterior(a, b)

        p_a_better = float(np.mean(diff_posterior.samples > self.rope_high))
        p_b_better = float(np.mean(diff_posterior.samples < self.rope_low))
        p_equiv = float(np.mean(
            (diff_posterior.samples >= self.rope_low) &
            (diff_posterior.samples <= self.rope_high)
        ))

        return ComparisonResult(
            algorithm_a=algorithm_a,
            algorithm_b=algorithm_b,
            metric_name=metric_name,
            posterior_diff=diff_posterior,
            rope_result=rope,
            bayes_factor=bf,
            effect_size=es,
            p_a_better=p_a_better,
            p_b_better=p_b_better,
            p_equivalent=p_equiv,
            raw_scores_a=a,
            raw_scores_b=b,
        )

    def _posterior_of_difference(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> PosteriorEstimate:
        """Posterior of the mean difference A - B via bootstrap."""
        paired = len(a) == len(b)
        if paired:
            diffs = a - b
            return self.posterior_estimation(diffs, prior=self.prior_type)
        else:
            post_a = self._sample_posterior_normal(a, self.n_samples)
            post_b = self._sample_posterior_normal(b, self.n_samples)
            diff_samples = post_a - post_b
            return self._build_posterior_estimate(diff_samples, "normal_diff")

    # ======================================================================
    # Multi-algorithm comparison
    # ======================================================================

    def compare_multiple(
        self,
        score_dict: Dict[str, Union[np.ndarray, List[float]]],
        metric_name: str = "score",
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms with full ranking and credible intervals.

        Parameters
        ----------
        score_dict : dict mapping algorithm name -> scores
        metric_name : str

        Returns
        -------
        dict with keys:
            pairwise  – Dict[(name_a, name_b), ComparisonResult]
            ranking   – ModelRanking
            posteriors – Dict[name, PosteriorEstimate]
            sign_test – Dict[(name_a, name_b), ROPEResult]
        """
        names = sorted(score_dict.keys())
        validated: Dict[str, np.ndarray] = {}
        for name in names:
            validated[name] = _validate_scores(score_dict[name])

        posteriors: Dict[str, PosteriorEstimate] = {}
        for name in names:
            posteriors[name] = self.posterior_estimation(
                validated[name], self.prior_type
            )

        pairwise: Dict[Tuple[str, str], ComparisonResult] = {}
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i >= j:
                    continue
                pairwise[(na, nb)] = self.compare_two(
                    validated[na],
                    validated[nb],
                    metric_name=metric_name,
                    algorithm_a=na,
                    algorithm_b=nb,
                )

        n = len(names)
        win_matrix = np.zeros((n, n), dtype=np.float64)
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i >= j:
                    continue
                key = (na, nb)
                cr = pairwise[key]
                win_matrix[i, j] = cr.p_a_better
                win_matrix[j, i] = cr.p_b_better

        ranking = self.bradley_terry_ranking(
            win_matrix, algorithm_names=names
        )

        sign_results = self._sign_test.multi_comparison(validated)

        return {
            "pairwise": pairwise,
            "ranking": ranking,
            "posteriors": posteriors,
            "sign_test": sign_results,
            "metric_name": metric_name,
            "n_algorithms": n,
        }

    # ======================================================================
    # Posterior estimation
    # ======================================================================

    def posterior_estimation(
        self,
        scores: Union[np.ndarray, List[float]],
        prior: Union[str, PriorType] = PriorType.UNIFORM,
    ) -> PosteriorEstimate:
        """
        Estimate the posterior distribution of the mean of *scores*
        using the specified prior.

        Parameters
        ----------
        scores : array-like
            Observed data.
        prior : str or PriorType

        Returns
        -------
        PosteriorEstimate
        """
        data = _validate_scores(scores)
        if isinstance(prior, str):
            prior = PriorType(prior)

        if prior == PriorType.BETA:
            samples = self._sample_posterior_beta(data, self.n_samples)
            dist_type = "beta"
        elif prior in (PriorType.NORMAL, PriorType.WEAKLY_INFORMATIVE):
            samples = self._sample_posterior_normal(data, self.n_samples)
            dist_type = "normal"
        elif prior == PriorType.JEFFREYS:
            samples = self._sample_posterior_jeffreys(data, self.n_samples)
            dist_type = "jeffreys"
        else:
            samples = self._sample_posterior_bootstrap(data, self.n_samples)
            dist_type = "bootstrap"

        return self._build_posterior_estimate(samples, dist_type)

    # ---- samplers ---------------------------------------------------------

    def _sample_posterior_normal(
        self,
        data: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Normal-normal conjugate posterior for the mean with known-variance
        approximation (variance estimated from data).

        Under a flat / weakly-informative prior on mu,
        posterior is Normal(x_bar, se^2).
        """
        n = len(data)
        x_bar = float(np.mean(data))
        if n < 2:
            return np.full(n_samples, x_bar)
        se = float(np.std(data, ddof=1)) / math.sqrt(n)
        se = max(se, 1e-15)

        if self.prior_type == PriorType.WEAKLY_INFORMATIVE:
            prior_mu = 0.0
            prior_sigma = 10.0 * float(np.std(data, ddof=1))
            prior_prec = 1.0 / (prior_sigma ** 2)
            data_prec = n / (float(np.var(data, ddof=1)) + 1e-300)
            post_prec = prior_prec + data_prec
            post_mu = (prior_prec * prior_mu + data_prec * x_bar) / post_prec
            post_sigma = math.sqrt(1.0 / post_prec)
        else:
            dof = n - 1
            post_mu = x_bar
            post_sigma = se
            return self._rng.standard_t(dof, size=n_samples) * post_sigma + post_mu

        return self._rng.normal(post_mu, post_sigma, size=n_samples)

    def _sample_posterior_beta(
        self,
        data: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Beta posterior for data assumed to be in [0, 1].

        Uses a Beta(0.5, 0.5) Jeffreys prior for Bernoulli-like data,
        or moment-matched beta for continuous [0, 1] data.
        """
        data = np.clip(data, 0.0, 1.0)
        n = len(data)
        s = float(np.sum(data))

        mean_d = float(np.mean(data))
        var_d = float(np.var(data, ddof=1)) if n > 1 else 0.01

        if var_d < 1e-12:
            var_d = 1e-4

        ratio = mean_d * (1.0 - mean_d) / var_d - 1.0
        ratio = max(ratio, 2.0)

        alpha_hat = mean_d * ratio
        beta_hat = (1.0 - mean_d) * ratio

        alpha_prior = 0.5
        beta_prior = 0.5

        alpha_post = alpha_prior + alpha_hat * n / ratio
        beta_post = beta_prior + beta_hat * n / ratio

        alpha_post = max(alpha_post, 0.01)
        beta_post = max(beta_post, 0.01)

        simple_alpha = alpha_prior + s
        simple_beta = beta_prior + (n - s)

        w = min(n / 30.0, 1.0)
        final_alpha = w * simple_alpha + (1 - w) * alpha_post
        final_beta = w * simple_beta + (1 - w) * beta_post

        return self._rng.beta(final_alpha, final_beta, size=n_samples)

    def _sample_posterior_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Non-parametric Bayesian bootstrap (Rubin, 1981).

        Each sample weights the observations with a Dirichlet(1,...,1) draw,
        producing a posterior sample for the mean.
        """
        n = len(data)
        weights = self._rng.dirichlet(np.ones(n), size=n_samples)
        return weights @ data

    def _sample_posterior_jeffreys(
        self,
        data: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Sample from the Jeffreys posterior for normal data.

        Jeffreys prior: p(mu, sigma^2) proportional to 1/sigma^2
        Posterior for mu marginalised over sigma^2 is a t-distribution.
        """
        n = len(data)
        if n < 2:
            return np.full(n_samples, float(np.mean(data)))
        x_bar = float(np.mean(data))
        s2 = float(np.var(data, ddof=1))
        dof = n - 1
        scale = math.sqrt(s2 / n)
        return self._rng.standard_t(dof, size=n_samples) * scale + x_bar

    # ---- credible intervals -----------------------------------------------

    def _hdi(
        self,
        samples: np.ndarray,
        alpha: float = 0.05,
    ) -> CredibleInterval:
        """
        Highest Density Interval.

        Finds the shortest interval containing (1-alpha) of the mass.
        """
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        n_included = int(np.ceil((1.0 - alpha) * n))
        if n_included >= n:
            return CredibleInterval(
                lower=float(sorted_s[0]),
                upper=float(sorted_s[-1]),
                alpha=alpha,
                method="hdi",
            )

        interval_widths = sorted_s[n_included:] - sorted_s[: n - n_included]
        if len(interval_widths) == 0:
            return CredibleInterval(
                lower=float(sorted_s[0]),
                upper=float(sorted_s[-1]),
                alpha=alpha,
                method="hdi",
            )
        best = int(np.argmin(interval_widths))
        return CredibleInterval(
            lower=float(sorted_s[best]),
            upper=float(sorted_s[best + n_included]),
            alpha=alpha,
            method="hdi",
        )

    def _eti(
        self,
        samples: np.ndarray,
        alpha: float = 0.05,
    ) -> CredibleInterval:
        """
        Equal-Tailed Interval (quantile-based credible interval).
        """
        lower = float(np.quantile(samples, alpha / 2.0))
        upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
        return CredibleInterval(
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="eti",
        )

    # ---- building posterior estimates -------------------------------------

    def _build_posterior_estimate(
        self,
        samples: np.ndarray,
        dist_type: str,
    ) -> PosteriorEstimate:
        """Wrap raw samples into a PosteriorEstimate with CI at several levels."""
        ci_dict: Dict[float, CredibleInterval] = {}
        for level in (0.50, 0.80, 0.90, 0.95, 0.99):
            alpha = 1.0 - level
            ci_dict[level] = self._hdi(samples, alpha)

        ess = self._effective_sample_size(samples)

        pe = PosteriorEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples, ddof=1)),
            samples=samples,
            credible_intervals=ci_dict,
            distribution_type=dist_type,
            effective_sample_size=ess,
        )
        return pe

    # ======================================================================
    # ROPE analysis
    # ======================================================================

    def rope_analysis(
        self,
        posterior_diff: np.ndarray,
        rope_low: float,
        rope_high: float,
    ) -> ROPEResult:
        """
        Region Of Practical Equivalence analysis on posterior difference
        samples.

        Parameters
        ----------
        posterior_diff : 1-D array of posterior samples for (A - B).
        rope_low, rope_high : boundaries of the ROPE.

        Returns
        -------
        ROPEResult
        """
        posterior_diff = np.asarray(posterior_diff, dtype=np.float64)
        n = len(posterior_diff)
        if n == 0:
            raise ValueError("Empty posterior samples.")

        p_left = float(np.mean(posterior_diff < rope_low))
        p_right = float(np.mean(posterior_diff > rope_high))
        p_rope = 1.0 - p_left - p_right
        p_rope = max(p_rope, 0.0)

        decision = self._rope_decision(p_left, p_rope, p_right)

        return ROPEResult(
            probability_left=p_left,
            probability_rope=p_rope,
            probability_right=p_right,
            rope_low=rope_low,
            rope_high=rope_high,
            decision=decision,
            posterior_samples=posterior_diff,
        )

    @staticmethod
    def _rope_decision(
        p_left: float,
        p_rope: float,
        p_right: float,
        threshold: float = 0.95,
    ) -> str:
        if p_left >= threshold:
            return DecisionType.LEFT.value
        if p_right >= threshold:
            return DecisionType.RIGHT.value
        if p_rope >= threshold:
            return DecisionType.ROPE.value
        return DecisionType.UNDECIDED.value

    # ======================================================================
    # Bayes factor
    # ======================================================================

    def bayes_factor(
        self,
        scores_a: Union[np.ndarray, List[float]],
        scores_b: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Bayes factor BF_10 for the hypothesis that A != B vs A == B.

        Uses the Savage-Dickey density ratio at zero for the posterior of
        the difference.

        Returns
        -------
        float  – BF_10 > 1 favours the alternative (A != B).
        """
        a = _validate_scores(scores_a)
        b = _validate_scores(scores_b)

        if len(a) == len(b):
            diff = a - b
        else:
            diff = np.concatenate([
                a - np.mean(b),
            ])

        post = self.posterior_estimation(diff, self.prior_type)
        samples = post.samples

        posterior_density_at_zero = self._kde_density_at_point(samples, 0.0)

        prior_density_at_zero = self._prior_density_at_zero(samples)

        if posterior_density_at_zero < 1e-300:
            return float("inf")

        bf_01 = posterior_density_at_zero / max(prior_density_at_zero, 1e-300)
        bf_10 = 1.0 / max(bf_01, 1e-300)

        bf_10 = min(bf_10, 1e12)
        return bf_10

    def _kde_density_at_point(
        self, samples: np.ndarray, point: float
    ) -> float:
        """Kernel density estimate at a single point."""
        try:
            kde = stats.gaussian_kde(samples)
            return float(kde(point)[0])
        except Exception:
            h = 1.06 * float(np.std(samples, ddof=1)) * len(samples) ** (-0.2)
            if h < 1e-15:
                return 0.0
            kernel_values = np.exp(
                -0.5 * ((samples - point) / h) ** 2
            ) / (h * math.sqrt(2 * math.pi))
            return float(np.mean(kernel_values))

    def _prior_density_at_zero(self, samples: np.ndarray) -> float:
        """Density of the prior at zero (approximated analytically)."""
        data_std = float(np.std(samples, ddof=1))
        if data_std < 1e-15:
            data_std = 1.0
        if self.prior_type == PriorType.WEAKLY_INFORMATIVE:
            prior_sd = 10.0 * data_std
        elif self.prior_type == PriorType.NORMAL:
            prior_sd = data_std
        else:
            data_range = float(np.max(samples) - np.min(samples))
            if data_range < 1e-15:
                data_range = 1.0
            return 1.0 / data_range
        return float(stats.norm.pdf(0.0, loc=0.0, scale=prior_sd))

    def _log_marginal_likelihood(
        self,
        data: np.ndarray,
        model: str = "normal",
    ) -> float:
        """
        Log marginal likelihood of data under a specified model.

        Parameters
        ----------
        data : 1-D array
        model : str, one of 'normal', 'cauchy', 'laplace'

        Returns
        -------
        float – log p(data | model)
        """
        n = len(data)
        if model == "normal":
            mu_hat = float(np.mean(data))
            if n < 2:
                sigma_hat = 1.0
            else:
                sigma_hat = float(np.std(data, ddof=1))
            sigma_hat = max(sigma_hat, 1e-15)
            ll = float(np.sum(stats.norm.logpdf(data, loc=mu_hat, scale=sigma_hat)))
            k = 2
            bic = -2.0 * ll + k * math.log(n)
            return -0.5 * bic

        elif model == "cauchy":
            def neg_ll(params: np.ndarray) -> float:
                loc, scale = params
                if scale <= 0:
                    return 1e20
                return -float(np.sum(
                    stats.cauchy.logpdf(data, loc=loc, scale=scale)
                ))
            x0 = np.array([float(np.median(data)), float(stats.iqr(data) / 2.0 + 0.1)])
            try:
                res = optimize.minimize(neg_ll, x0, method="Nelder-Mead")
                ll = -res.fun
            except Exception:
                ll = float(np.sum(stats.cauchy.logpdf(
                    data, loc=float(np.median(data)),
                    scale=float(stats.iqr(data) / 2.0 + 0.1)
                )))
            k = 2
            bic = -2.0 * ll + k * math.log(n)
            return -0.5 * bic

        elif model == "laplace":
            loc = float(np.median(data))
            scale = float(np.mean(np.abs(data - loc)))
            scale = max(scale, 1e-15)
            ll = float(np.sum(stats.laplace.logpdf(data, loc=loc, scale=scale)))
            k = 2
            bic = -2.0 * ll + k * math.log(n)
            return -0.5 * bic

        else:
            raise ValueError(f"Unknown model: {model}")

    # ======================================================================
    # Bradley-Terry ranking
    # ======================================================================

    def bradley_terry_ranking(
        self,
        pairwise_results: np.ndarray,
        algorithm_names: Optional[List[str]] = None,
        n_bootstrap: int = 5000,
    ) -> ModelRanking:
        """
        Fit a Bradley-Terry model to pairwise win probabilities.

        Parameters
        ----------
        pairwise_results : (K, K) array where entry (i,j) is the
            probability / proportion that algorithm i beats j.
        algorithm_names : list of K names
        n_bootstrap : int  – bootstrap samples for posterior on strengths.

        Returns
        -------
        ModelRanking
        """
        W = np.asarray(pairwise_results, dtype=np.float64)
        k = W.shape[0]
        if W.shape != (k, k):
            raise ValueError("pairwise_results must be square.")
        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(k)]

        strengths = self._fit_bradley_terry(W, n_iter=200)

        ranks = np.argsort(-strengths).argsort().astype(np.float64) + 1

        probs = _softmax(strengths)

        bootstrap_strengths = np.zeros((n_bootstrap, k))
        for b in range(n_bootstrap):
            noise = self._rng.normal(0.0, 0.1, size=k)
            W_b = W.copy()
            for i in range(k):
                for j in range(k):
                    if i != j:
                        W_b[i, j] = np.clip(W[i, j] + noise[i] - noise[j], 0.01, 0.99)
            bootstrap_strengths[b] = self._fit_bradley_terry(W_b, n_iter=100)

        cis: Dict[str, CredibleInterval] = {}
        for i, name in enumerate(algorithm_names):
            col = bootstrap_strengths[:, i]
            cis[name] = self._hdi(col, alpha=0.05)

        return ModelRanking(
            algorithm_names=algorithm_names,
            ranks=ranks,
            probabilities=probs,
            posterior_samples=bootstrap_strengths,
            strength_params=strengths,
            credible_intervals=cis,
        )

    def _fit_bradley_terry(
        self,
        win_matrix: np.ndarray,
        n_iter: int = 200,
    ) -> np.ndarray:
        """
        Iterative MM algorithm for Bradley-Terry model.

        Parameters
        ----------
        win_matrix : (K, K) array, W[i,j] = wins of i vs j (can be probs).
        n_iter : int

        Returns
        -------
        strength : (K,) array of log-strength parameters (sum-to-zero).
        """
        k = win_matrix.shape[0]
        n_ij = win_matrix + win_matrix.T
        n_ij = np.maximum(n_ij, 1e-10)

        pi = np.ones(k) / k
        for _ in range(n_iter):
            pi_new = np.zeros(k)
            for i in range(k):
                numerator = 0.0
                denominator = 0.0
                for j in range(k):
                    if i == j:
                        continue
                    numerator += win_matrix[i, j]
                    denominator += n_ij[i, j] / (pi[i] + pi[j])
                denominator = max(denominator, 1e-15)
                pi_new[i] = numerator / denominator
            pi_sum = np.sum(pi_new)
            if pi_sum > 0:
                pi_new /= pi_sum
            else:
                pi_new = np.ones(k) / k
            pi = pi_new

        pi = np.maximum(pi, 1e-15)
        strengths = np.log(pi)
        strengths -= np.mean(strengths)
        return strengths

    # ======================================================================
    # Plackett-Luce ranking
    # ======================================================================

    def plackett_luce_ranking(
        self,
        comparisons: List[List[int]],
        algorithm_names: Optional[List[str]] = None,
        n_algorithms: Optional[int] = None,
    ) -> ModelRanking:
        """
        Fit a Plackett-Luce model to full or partial rankings.

        Parameters
        ----------
        comparisons : list of rankings, each ranking is a list of algorithm
            indices ordered from best (index 0) to worst.
        algorithm_names : optional list of names.
        n_algorithms : int, inferred if not provided.

        Returns
        -------
        ModelRanking
        """
        if n_algorithms is None:
            n_algorithms = max(max(r) for r in comparisons) + 1
        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(n_algorithms)]

        k = n_algorithms
        gamma = np.ones(k, dtype=np.float64)

        for iteration in range(300):
            gamma_old = gamma.copy()

            w = np.zeros(k)
            for ranking in comparisons:
                for pos, alg in enumerate(ranking):
                    w[alg] += 1.0

            denom_terms = np.zeros(k)
            for ranking in comparisons:
                m = len(ranking)
                for j in range(m):
                    tail_sum = sum(gamma[ranking[l]] for l in range(j, m))
                    tail_sum = max(tail_sum, 1e-15)
                    for l in range(j, m):
                        denom_terms[ranking[l]] += 1.0 / tail_sum

            for i in range(k):
                if denom_terms[i] > 0:
                    gamma[i] = w[i] / denom_terms[i]
                else:
                    gamma[i] = 1e-10

            gamma /= np.sum(gamma)

            if np.max(np.abs(gamma - gamma_old)) < 1e-10:
                break

        log_gamma = np.log(np.maximum(gamma, 1e-300))
        log_gamma -= np.mean(log_gamma)
        ranks = np.argsort(-gamma).argsort().astype(np.float64) + 1
        probs = gamma / np.sum(gamma)

        bootstrap_gamma = np.zeros((self.n_samples, k))
        n_comparisons = len(comparisons)
        for b in range(self.n_samples):
            idx = self._rng.integers(0, n_comparisons, size=n_comparisons)
            boot_comparisons = [comparisons[i] for i in idx]
            g = self._fit_plackett_luce_once(boot_comparisons, k)
            bootstrap_gamma[b] = g

        cis: Dict[str, CredibleInterval] = {}
        for i, name in enumerate(algorithm_names):
            cis[name] = self._hdi(bootstrap_gamma[:, i], alpha=0.05)

        return ModelRanking(
            algorithm_names=algorithm_names,
            ranks=ranks,
            probabilities=probs,
            posterior_samples=bootstrap_gamma,
            strength_params=log_gamma,
            credible_intervals=cis,
        )

    def _fit_plackett_luce_once(
        self,
        comparisons: List[List[int]],
        k: int,
        n_iter: int = 100,
    ) -> np.ndarray:
        gamma = np.ones(k, dtype=np.float64)
        for _ in range(n_iter):
            w = np.zeros(k)
            d = np.zeros(k)
            for ranking in comparisons:
                m = len(ranking)
                for pos, alg in enumerate(ranking):
                    w[alg] += 1.0
                for j in range(m):
                    tail = sum(gamma[ranking[l]] for l in range(j, m))
                    tail = max(tail, 1e-15)
                    for l in range(j, m):
                        d[ranking[l]] += 1.0 / tail
            for i in range(k):
                gamma[i] = w[i] / max(d[i], 1e-15)
            gamma /= np.sum(gamma)
        return gamma

    # ======================================================================
    # Effect size posterior
    # ======================================================================

    def effect_size_posterior(
        self,
        scores_a: Union[np.ndarray, List[float]],
        scores_b: Union[np.ndarray, List[float]],
    ) -> PosteriorEstimate:
        """
        Posterior distribution of Cohen's d effect size.

        Uses the pooled standard deviation from posterior samples.

        Returns
        -------
        PosteriorEstimate
        """
        a = _validate_scores(scores_a)
        b = _validate_scores(scores_b)

        post_a = self._sample_posterior_normal(a, self.n_samples)
        post_b = self._sample_posterior_normal(b, self.n_samples)

        na, nb = len(a), len(b)
        var_a = float(np.var(a, ddof=1)) if na > 1 else 1.0
        var_b = float(np.var(b, ddof=1)) if nb > 1 else 1.0

        pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1)
        pooled_sd = math.sqrt(max(pooled_var, 1e-15))

        sd_samples = self._rng.normal(pooled_sd, pooled_sd / math.sqrt(2 * (na + nb - 2)), size=self.n_samples)
        sd_samples = np.maximum(sd_samples, 1e-15)

        d_samples = (post_a - post_b) / sd_samples

        correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
        d_samples *= correction

        return self._build_posterior_estimate(d_samples, "cohens_d")

    # ======================================================================
    # Sequential analysis
    # ======================================================================

    def sequential_analysis(
        self,
        stream_a: Union[np.ndarray, List[float]],
        stream_b: Union[np.ndarray, List[float]],
        threshold: float = 10.0,
        min_observations: int = 20,
        max_observations: Optional[int] = None,
    ) -> SequentialResult:
        """
        Sequential Bayesian analysis with optional early stopping.

        Computes the Bayes factor after each new observation pair.  Stops
        when BF_10 exceeds *threshold* or 1/threshold (decisive evidence
        either way), or when all observations are consumed.

        Parameters
        ----------
        stream_a, stream_b : array-like
        threshold : float – BF threshold for stopping.
        min_observations : int – minimum pairs before considering stop.
        max_observations : int or None

        Returns
        -------
        SequentialResult
        """
        a = _validate_scores(stream_a)
        b = _validate_scores(stream_b)
        n = min(len(a), len(b))
        if max_observations is not None:
            n = min(n, max_observations)

        bf_trajectory: List[float] = []
        evidence_trajectory: List[float] = []
        stopped_early = False
        threshold_reached_at: Optional[int] = None
        decision = DecisionType.UNDECIDED.value

        for t in range(1, n + 1):
            a_t = a[:t]
            b_t = b[:t]

            if t < 3:
                bf = 1.0
            else:
                bf = self._sequential_bf(a_t, b_t)

            bf_trajectory.append(bf)
            evidence_trajectory.append(math.log(max(bf, 1e-300)))

            if t >= min_observations and threshold_reached_at is None:
                if bf >= threshold:
                    threshold_reached_at = t
                    stopped_early = True
                    decision = "different"
                    break
                elif bf <= 1.0 / threshold:
                    threshold_reached_at = t
                    stopped_early = True
                    decision = "equivalent"
                    break

        final_a = a[:len(bf_trajectory)]
        final_b = b[:len(bf_trajectory)]
        diff = final_a - final_b
        final_posterior = self.posterior_estimation(diff, self.prior_type)

        if decision == DecisionType.UNDECIDED.value:
            if len(bf_trajectory) > 0 and bf_trajectory[-1] > 3.0:
                decision = "weak_evidence_different"
            elif len(bf_trajectory) > 0 and bf_trajectory[-1] < 1.0 / 3.0:
                decision = "weak_evidence_equivalent"
            else:
                decision = "inconclusive"

        return SequentialResult(
            stopped_early=stopped_early,
            n_observations=len(bf_trajectory),
            final_posterior=final_posterior,
            decision=decision,
            evidence_trajectory=evidence_trajectory,
            threshold_reached_at=threshold_reached_at,
            bayes_factor_trajectory=bf_trajectory,
        )

    def _sequential_bf(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Fast BF for sequential setting using t-test BF approximation."""
        n = len(a)
        diff = a - b
        mean_d = float(np.mean(diff))
        if n < 2:
            return 1.0
        se_d = float(np.std(diff, ddof=1)) / math.sqrt(n)
        if se_d < 1e-15:
            return 1.0

        t_stat = mean_d / se_d
        dof = n - 1

        r = 1.0

        log_bf = 0.5 * math.log(dof / (dof + 1.0))
        log_bf += 0.5 * (dof + 1.0) * math.log(1.0 + t_stat ** 2 / dof)
        log_bf -= 0.5 * (dof + 1.0) * math.log(
            1.0 + t_stat ** 2 / (dof * (1.0 + r ** 2 * dof))
        )

        log_bf = max(min(log_bf, 30.0), -30.0)
        return math.exp(log_bf)

    # ======================================================================
    # Meta-analysis
    # ======================================================================

    def meta_analysis(
        self,
        study_results: List[Dict[str, float]],
    ) -> MetaAnalysisResult:
        """
        Bayesian random-effects meta-analysis.

        Each study dict must have keys 'mean', 'se' (standard error),
        and optionally 'name'.

        Parameters
        ----------
        study_results : list of dicts with 'mean' and 'se'.

        Returns
        -------
        MetaAnalysisResult
        """
        n_studies = len(study_results)
        if n_studies == 0:
            raise ValueError("No studies provided.")

        means = np.array([s["mean"] for s in study_results])
        ses = np.array([max(s["se"], 1e-15) for s in study_results])
        names = [s.get("name", f"study_{i}") for i, s in enumerate(study_results)]
        variances = ses ** 2

        tau2_samples = self._sample_heterogeneity(means, variances, self.n_samples)
        mu_samples = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            tau2 = max(tau2_samples[i], 0.0)
            weights = 1.0 / (variances + tau2)
            w_sum = np.sum(weights)
            if w_sum < 1e-300:
                mu_samples[i] = float(np.mean(means))
                continue
            weighted_mean = float(np.sum(weights * means) / w_sum)
            weighted_var = 1.0 / w_sum
            mu_samples[i] = self._rng.normal(weighted_mean, math.sqrt(weighted_var))

        combined = self._build_posterior_estimate(mu_samples, "meta_analysis")
        tau2_posterior = self._build_posterior_estimate(tau2_samples, "heterogeneity")

        fe_weights = 1.0 / variances
        fe_w_sum = np.sum(fe_weights)
        fe_mean = np.sum(fe_weights * means) / fe_w_sum
        q_stat = float(np.sum(fe_weights * (means - fe_mean) ** 2))
        dof_q = n_studies - 1
        if dof_q > 0:
            i_squared = max(0.0, (q_stat - dof_q) / q_stat) * 100.0
        else:
            i_squared = 0.0

        study_weights_dict: Dict[str, float] = {}
        median_tau2 = float(np.median(tau2_samples))
        re_weights = 1.0 / (variances + max(median_tau2, 0.0))
        re_w_sum = np.sum(re_weights)
        for i, name in enumerate(names):
            study_weights_dict[name] = float(re_weights[i] / re_w_sum)

        forest_data = []
        for i, name in enumerate(names):
            forest_data.append({
                "name": name,
                "mean": float(means[i]),
                "lower": float(means[i] - 1.96 * ses[i]),
                "upper": float(means[i] + 1.96 * ses[i]),
                "weight": study_weights_dict[name],
            })

        pred_var = float(np.var(mu_samples)) + max(median_tau2, 0.0)
        pred_sd = math.sqrt(max(pred_var, 1e-15))
        pred_mean = float(np.mean(mu_samples))
        pred_interval = CredibleInterval(
            lower=pred_mean - 1.96 * pred_sd,
            upper=pred_mean + 1.96 * pred_sd,
            alpha=0.05,
            method="prediction",
        )

        return MetaAnalysisResult(
            combined_posterior=combined,
            heterogeneity=tau2_posterior,
            study_weights=study_weights_dict,
            forest_data=forest_data,
            i_squared=i_squared,
            prediction_interval=pred_interval,
        )

    def _sample_heterogeneity(
        self,
        means: np.ndarray,
        variances: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Sample heterogeneity tau^2 via a grid-based approximation
        to the marginal posterior.
        """
        n = len(means)

        fe_weights = 1.0 / variances
        fe_w_sum = np.sum(fe_weights)
        fe_mean = np.sum(fe_weights * means) / fe_w_sum
        q = float(np.sum(fe_weights * (means - fe_mean) ** 2))

        dl_tau2 = max(0.0, (q - (n - 1)) / (fe_w_sum - np.sum(fe_weights ** 2) / fe_w_sum))

        tau2_max = max(dl_tau2 * 10.0, float(np.var(means)) * 5.0, 1.0)
        grid = np.linspace(0.0, tau2_max, 2000)

        log_posteriors = np.zeros(len(grid))
        for g_idx, tau2 in enumerate(grid):
            weights = 1.0 / (variances + tau2)
            w_sum = np.sum(weights)
            w_mean = np.sum(weights * means) / w_sum
            log_lik = -0.5 * np.sum(np.log(variances + tau2))
            log_lik += -0.5 * np.sum(weights * (means - w_mean) ** 2)
            log_lik += -0.5 * math.log(w_sum)
            log_prior = -0.5 * math.log(max(tau2, 1e-10))
            log_posteriors[g_idx] = log_lik + log_prior

        log_posteriors -= np.max(log_posteriors)
        posteriors = np.exp(log_posteriors)
        total = np.sum(posteriors)
        if total < 1e-300:
            return self._rng.uniform(0.0, tau2_max, size=n_samples)
        posteriors /= total

        cdf = np.cumsum(posteriors)
        cdf /= cdf[-1]
        u = self._rng.uniform(0.0, 1.0, size=n_samples)
        indices = np.searchsorted(cdf, u)
        indices = np.clip(indices, 0, len(grid) - 1)
        samples = grid[indices]

        jitter_scale = (grid[1] - grid[0]) * 0.5
        samples += self._rng.uniform(-jitter_scale, jitter_scale, size=n_samples)
        samples = np.maximum(samples, 0.0)
        return samples

    # ======================================================================
    # MCMC diagnostics
    # ======================================================================

    def _geweke_diagnostic(
        self,
        chain: np.ndarray,
        first_frac: float = 0.1,
        last_frac: float = 0.5,
    ) -> Dict[str, float]:
        """
        Geweke (1992) convergence diagnostic.

        Compares the mean of the first *first_frac* of the chain to the
        mean of the last *last_frac*.  Returns a z-score; |z| > 2 suggests
        non-convergence.

        Parameters
        ----------
        chain : 1-D array of MCMC samples.
        first_frac, last_frac : proportions of the chain to compare.

        Returns
        -------
        dict with 'z_score' and 'converged' (bool).
        """
        chain = np.asarray(chain, dtype=np.float64)
        n = len(chain)
        if n < 20:
            return {"z_score": 0.0, "converged": True}

        n_first = max(int(first_frac * n), 2)
        n_last = max(int(last_frac * n), 2)
        if n_first + n_last > n:
            n_last = n - n_first

        first_part = chain[:n_first]
        last_part = chain[n - n_last:]

        mean_first = float(np.mean(first_part))
        mean_last = float(np.mean(last_part))

        var_first = self._spectral_density_at_zero(first_part)
        var_last = self._spectral_density_at_zero(last_part)

        se = math.sqrt(var_first / n_first + var_last / n_last)
        if se < 1e-15:
            return {"z_score": 0.0, "converged": True}

        z = (mean_first - mean_last) / se
        return {"z_score": float(z), "converged": abs(z) < 2.0}

    def _spectral_density_at_zero(
        self, chain: np.ndarray, max_lag: int = 50
    ) -> float:
        """Estimate spectral density at frequency zero (for Geweke)."""
        n = len(chain)
        if n < 3:
            return float(np.var(chain, ddof=1))
        max_lag = min(max_lag, n - 1)
        mean_c = np.mean(chain)
        centered = chain - mean_c
        autocov = np.correlate(centered, centered, mode="full")[n - 1:]
        autocov = autocov[:max_lag + 1] / n

        s = autocov[0]
        for k in range(1, max_lag + 1):
            weight = 1.0 - k / (max_lag + 1)
            s += 2.0 * weight * autocov[k]
        return max(float(s), 1e-15)

    def _effective_sample_size(
        self, chain: np.ndarray
    ) -> float:
        """
        Effective sample size using initial positive sequence estimator.

        Parameters
        ----------
        chain : 1-D array of MCMC samples.

        Returns
        -------
        float – effective sample size.
        """
        chain = np.asarray(chain, dtype=np.float64)
        n = len(chain)
        if n < 4:
            return float(n)

        mean_c = np.mean(chain)
        var_c = np.var(chain, ddof=1)
        if var_c < 1e-15:
            return float(n)

        centered = chain - mean_c

        max_lag = min(n - 1, n // 2)
        autocorr = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[0] = 1.0
            else:
                c = np.sum(centered[:-lag] * centered[lag:]) / ((n - lag) * var_c)
                autocorr[lag] = c

        sum_rho = 0.0
        for lag in range(1, max_lag + 1, 2):
            pair_sum = autocorr[lag]
            if lag + 1 <= max_lag:
                pair_sum += autocorr[lag + 1]
            if pair_sum < 0:
                break
            sum_rho += pair_sum

        tau = 1.0 + 2.0 * sum_rho
        tau = max(tau, 1.0)
        ess = n / tau
        return max(ess, 1.0)

    def _gelman_rubin(
        self, chains: List[np.ndarray]
    ) -> float:
        """
        Gelman-Rubin R-hat convergence diagnostic for multiple chains.

        Parameters
        ----------
        chains : list of 1-D arrays (each a separate MCMC chain).

        Returns
        -------
        float – R-hat.  Values close to 1.0 indicate convergence.
        """
        m = len(chains)
        if m < 2:
            return 1.0

        ns = [len(c) for c in chains]
        n = min(ns)
        if n < 2:
            return 1.0

        trimmed = [c[:n] for c in chains]
        chain_means = np.array([float(np.mean(c)) for c in trimmed])
        chain_vars = np.array([float(np.var(c, ddof=1)) for c in trimmed])

        grand_mean = float(np.mean(chain_means))
        B = n * float(np.var(chain_means, ddof=1))
        W = float(np.mean(chain_vars))

        if W < 1e-15:
            return 1.0

        var_hat = (1.0 - 1.0 / n) * W + (1.0 / n) * B
        r_hat = math.sqrt(var_hat / W)
        return max(r_hat, 1.0)

    def run_diagnostics(
        self,
        samples: np.ndarray,
        n_chains: int = 4,
    ) -> Dict[str, Any]:
        """
        Run a full suite of MCMC diagnostics on posterior samples.

        If a single chain is provided, splits it into *n_chains* sub-chains
        for the Gelman-Rubin statistic.

        Returns
        -------
        dict with geweke, ess, r_hat, and overall convergence assessment.
        """
        samples = np.asarray(samples, dtype=np.float64)
        n = len(samples)

        geweke = self._geweke_diagnostic(samples)
        ess = self._effective_sample_size(samples)

        chain_len = n // n_chains
        if chain_len < 10:
            r_hat = 1.0
        else:
            sub_chains = [
                samples[i * chain_len: (i + 1) * chain_len]
                for i in range(n_chains)
            ]
            r_hat = self._gelman_rubin(sub_chains)

        ess_ratio = ess / n if n > 0 else 0.0
        converged = (
            geweke["converged"]
            and r_hat < 1.05
            and ess_ratio > 0.01
        )

        return {
            "geweke_z": geweke["z_score"],
            "geweke_converged": geweke["converged"],
            "effective_sample_size": ess,
            "ess_ratio": ess_ratio,
            "r_hat": r_hat,
            "n_samples": n,
            "converged": converged,
        }

    # ======================================================================
    # Hierarchical model
    # ======================================================================

    def hierarchical_model(
        self,
        grouped_scores: Dict[str, Union[np.ndarray, List[float]]],
        n_samples: int = 10_000,
    ) -> HierarchicalResult:
        """
        Bayesian hierarchical (random-effects) model.

        Assumes:
            theta_j ~ Normal(mu, tau^2)      (group means)
            y_{ij}  ~ Normal(theta_j, sigma_j^2)  (observations)

        Uses Gibbs-like sampling with closed-form conditional posteriors.

        Parameters
        ----------
        grouped_scores : dict mapping group name -> scores.
        n_samples : int

        Returns
        -------
        HierarchicalResult
        """
        groups = sorted(grouped_scores.keys())
        k = len(groups)
        if k < 2:
            raise ValueError("Need at least 2 groups for hierarchical model.")

        validated: Dict[str, np.ndarray] = {}
        for g in groups:
            validated[g] = _validate_scores(grouped_scores[g])

        group_means = np.array([float(np.mean(validated[g])) for g in groups])
        group_vars = np.array([
            float(np.var(validated[g], ddof=1)) if len(validated[g]) > 1 else 1.0
            for g in groups
        ])
        group_ns = np.array([len(validated[g]) for g in groups])

        mu_samples = np.zeros(n_samples)
        tau2_samples = np.zeros(n_samples)
        theta_samples = np.zeros((n_samples, k))
        sigma2_samples = np.zeros(n_samples)

        mu = float(np.mean(group_means))
        tau2 = float(np.var(group_means, ddof=1))
        tau2 = max(tau2, 0.01)
        sigma2 = float(np.mean(group_vars))
        sigma2 = max(sigma2, 0.01)
        thetas = group_means.copy()

        for s in range(n_samples):
            # Sample thetas given mu, tau2, sigma2
            for j in range(k):
                precision_prior = 1.0 / max(tau2, 1e-10)
                precision_data = group_ns[j] / max(sigma2, 1e-10)
                precision_post = precision_prior + precision_data
                mean_post = (
                    precision_prior * mu + precision_data * group_means[j]
                ) / precision_post
                var_post = 1.0 / precision_post
                thetas[j] = self._rng.normal(mean_post, math.sqrt(var_post))

            # Sample mu given thetas, tau2
            theta_mean = float(np.mean(thetas))
            mu_var = tau2 / k
            mu = self._rng.normal(theta_mean, math.sqrt(max(mu_var, 1e-15)))

            # Sample tau2 given thetas, mu (inverse-gamma posterior)
            ss_theta = float(np.sum((thetas - mu) ** 2))
            alpha_tau = (k - 1) / 2.0
            beta_tau = ss_theta / 2.0
            alpha_tau = max(alpha_tau, 0.01)
            beta_tau = max(beta_tau, 1e-10)
            tau2 = 1.0 / self._rng.gamma(alpha_tau, 1.0 / beta_tau)
            tau2 = max(tau2, 1e-10)

            # Sample sigma2 given data, thetas (inverse-gamma)
            ss_within = 0.0
            n_total = 0
            for j, g in enumerate(groups):
                residuals = validated[g] - thetas[j]
                ss_within += float(np.sum(residuals ** 2))
                n_total += len(validated[g])
            alpha_sig = n_total / 2.0
            beta_sig = ss_within / 2.0
            alpha_sig = max(alpha_sig, 0.01)
            beta_sig = max(beta_sig, 1e-10)
            sigma2 = 1.0 / self._rng.gamma(alpha_sig, 1.0 / beta_sig)
            sigma2 = max(sigma2, 1e-10)

            mu_samples[s] = mu
            tau2_samples[s] = tau2
            theta_samples[s] = thetas.copy()
            sigma2_samples[s] = sigma2

        burnin = n_samples // 5
        mu_samples = mu_samples[burnin:]
        tau2_samples = tau2_samples[burnin:]
        theta_samples = theta_samples[burnin:]
        sigma2_samples = sigma2_samples[burnin:]

        global_posterior = self._build_posterior_estimate(mu_samples, "hierarchical_mu")
        between_var = self._build_posterior_estimate(tau2_samples, "hierarchical_tau2")
        within_var = self._build_posterior_estimate(sigma2_samples, "hierarchical_sigma2")

        group_posteriors: Dict[str, PosteriorEstimate] = {}
        for j, g in enumerate(groups):
            gp = self._build_posterior_estimate(
                theta_samples[:, j], f"hierarchical_theta_{g}"
            )
            group_posteriors[g] = gp

        shrinkage: Dict[str, float] = {}
        median_tau2 = float(np.median(tau2_samples))
        for j, g in enumerate(groups):
            sigma2_j = group_vars[j]
            nj = group_ns[j]
            data_prec = nj / max(sigma2_j, 1e-10)
            prior_prec = 1.0 / max(median_tau2, 1e-10)
            shrinkage[g] = float(prior_prec / (prior_prec + data_prec))

        diagnostics: Dict[str, float] = {}
        diag_mu = self.run_diagnostics(mu_samples)
        diagnostics["mu_ess"] = diag_mu["effective_sample_size"]
        diagnostics["mu_rhat"] = diag_mu["r_hat"]
        diagnostics["mu_geweke_z"] = diag_mu["geweke_z"]
        diagnostics["converged"] = float(diag_mu["converged"])

        return HierarchicalResult(
            group_posteriors=group_posteriors,
            global_posterior=global_posterior,
            between_group_variance=between_var,
            within_group_variance=within_var,
            shrinkage_factors=shrinkage,
            convergence_diagnostics=diagnostics,
        )

    # ======================================================================
    # Sensitivity to prior
    # ======================================================================

    def sensitivity_to_prior(
        self,
        scores: Union[np.ndarray, List[float]],
        priors_list: Optional[List[str]] = None,
    ) -> Dict[str, PosteriorEstimate]:
        """
        Assess sensitivity of posterior to the choice of prior.

        Parameters
        ----------
        scores : array-like
        priors_list : list of prior type strings; defaults to all available.

        Returns
        -------
        Dict mapping prior name to PosteriorEstimate.
        """
        data = _validate_scores(scores)
        if priors_list is None:
            priors_list = [p.value for p in PriorType]

        results: Dict[str, PosteriorEstimate] = {}
        for prior_name in priors_list:
            try:
                pe = self.posterior_estimation(data, PriorType(prior_name))
                results[prior_name] = pe
            except Exception as exc:
                warnings.warn(f"Prior '{prior_name}' failed: {exc}")
        return results

    # ======================================================================
    # Posterior predictive check
    # ======================================================================

    def posterior_predictive_check(
        self,
        model: PosteriorEstimate,
        data: Union[np.ndarray, List[float]],
        test_statistic: Optional[Callable[[np.ndarray], float]] = None,
        n_rep: int = 1000,
    ) -> Dict[str, float]:
        """
        Posterior predictive check.

        Generates replicated data from the posterior and compares a test
        statistic computed on the replications to that of the observed data.

        Parameters
        ----------
        model : PosteriorEstimate  – fitted posterior.
        data : array-like  – observed data.
        test_statistic : callable, defaults to np.mean.
        n_rep : int  – number of replicated datasets.

        Returns
        -------
        dict with 'p_value', 'observed_stat', 'predicted_stats', etc.
        """
        observed = _validate_scores(data)
        if test_statistic is None:
            test_statistic = lambda x: float(np.mean(x))

        observed_stat = test_statistic(observed)
        n = len(observed)

        samples = model.samples
        n_available = len(samples)

        predicted_stats = np.zeros(n_rep)
        for i in range(n_rep):
            idx = self._rng.integers(0, n_available)
            mu = samples[idx]
            replicated = self._rng.normal(mu, model.std, size=n)
            predicted_stats[i] = test_statistic(replicated)

        p_value = float(np.mean(predicted_stats >= observed_stat))

        return {
            "p_value": p_value,
            "observed_stat": observed_stat,
            "predicted_mean": float(np.mean(predicted_stats)),
            "predicted_std": float(np.std(predicted_stats)),
            "extreme": p_value < 0.05 or p_value > 0.95,
            "n_replications": n_rep,
        }

    # ======================================================================
    # Information criteria
    # ======================================================================

    def information_criteria(
        self,
        model_fits: Dict[str, Any],
    ) -> InformationCriteriaResult:
        """
        Compute WAIC and approximate LOO-CV.

        Parameters
        ----------
        model_fits : dict with keys:
            'log_likelihood' – (S, N) array where S = posterior samples,
                               N = data points.
                               Entry (s, i) = log p(y_i | theta_s).

        Returns
        -------
        InformationCriteriaResult
        """
        log_lik = np.asarray(model_fits["log_likelihood"], dtype=np.float64)
        if log_lik.ndim != 2:
            raise ValueError("log_likelihood must be 2-D (S x N).")
        S, N = log_lik.shape

        # WAIC
        lppd_i = np.zeros(N)
        p_waic_i = np.zeros(N)
        for i in range(N):
            col = log_lik[:, i]
            lppd_i[i] = _log_sum_exp(col) - math.log(S)
            p_waic_i[i] = float(np.var(col, ddof=1))

        lppd = float(np.sum(lppd_i))
        p_waic = float(np.sum(p_waic_i))
        waic = -2.0 * (lppd - p_waic)
        pointwise_waic = -2.0 * (lppd_i - p_waic_i)
        waic_se = float(np.sqrt(N * np.var(pointwise_waic, ddof=1)))

        # LOO (Pareto-smoothed importance sampling approximation)
        loo_i = np.zeros(N)
        pareto_k = np.zeros(N)
        for i in range(N):
            col = log_lik[:, i]
            log_ratios = -col
            log_ratios -= np.max(log_ratios)
            ratios = np.exp(log_ratios)

            sorted_ratios = np.sort(ratios)
            M = max(int(min(S / 5, 3 * math.sqrt(S))), 5)
            M = min(M, S - 1)
            tail = sorted_ratios[-M:]
            if len(tail) > 1 and tail[0] > 0:
                try:
                    xi, _, _ = stats.genpareto.fit(tail, floc=tail[0])
                    pareto_k[i] = xi
                except Exception:
                    pareto_k[i] = float("inf")
            else:
                pareto_k[i] = 0.0

            log_weights = -col
            log_weights -= _log_sum_exp(log_weights)
            weights = np.exp(log_weights)
            weights = np.minimum(weights, math.sqrt(S))
            w_sum = np.sum(weights)
            if w_sum < 1e-300:
                loo_i[i] = lppd_i[i]
            else:
                weights /= w_sum
                loo_i[i] = _log_sum_exp(np.log(np.maximum(weights, 1e-300)) + col)

        p_loo = lppd - float(np.sum(loo_i))
        loo = -2.0 * float(np.sum(loo_i))
        pointwise_loo = -2.0 * loo_i
        loo_se = float(np.sqrt(N * np.var(pointwise_loo, ddof=1)))

        return InformationCriteriaResult(
            waic=waic,
            waic_se=waic_se,
            loo=loo,
            loo_se=loo_se,
            p_waic=p_waic,
            p_loo=p_loo,
            pointwise_waic=pointwise_waic,
            pointwise_loo=pointwise_loo,
            pareto_k=pareto_k,
        )

    # ======================================================================
    # Calibration check
    # ======================================================================

    def calibration_check(
        self,
        predictions: Union[np.ndarray, List[float]],
        outcomes: Union[np.ndarray, List[float]],
        n_bins: int = 10,
    ) -> CalibrationResult:
        """
        Assess calibration of probabilistic predictions.

        Parameters
        ----------
        predictions : array-like  – predicted probabilities in [0, 1].
        outcomes : array-like  – binary outcomes (0 or 1).
        n_bins : int

        Returns
        -------
        CalibrationResult
        """
        preds = np.asarray(predictions, dtype=np.float64).ravel()
        outs = np.asarray(outcomes, dtype=np.float64).ravel()
        if len(preds) != len(outs):
            raise ValueError("predictions and outcomes must have same length.")

        preds = np.clip(preds, 0.0, 1.0)
        outs = np.clip(outs, 0.0, 1.0)
        n = len(preds)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        expected = np.zeros(n_bins)
        observed = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        bin_preds = []
        bin_outs = []

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (preds >= lo) & (preds <= hi)
            else:
                mask = (preds >= lo) & (preds < hi)
            in_bin = np.sum(mask)
            bin_counts[b] = in_bin
            if in_bin > 0:
                expected[b] = float(np.mean(preds[mask]))
                observed[b] = float(np.mean(outs[mask]))
            else:
                expected[b] = (lo + hi) / 2.0
                observed[b] = 0.0
            bin_preds.append(preds[mask])
            bin_outs.append(outs[mask])

        nonzero = bin_counts > 0
        if np.any(nonzero):
            weights = bin_counts[nonzero] / n
            cal_errors = np.abs(expected[nonzero] - observed[nonzero])
            calibration_error = float(np.sum(weights * cal_errors))
            max_cal_error = float(np.max(cal_errors))
        else:
            calibration_error = 0.0
            max_cal_error = 0.0

        # Brier score
        brier = float(np.mean((preds - outs) ** 2))

        # Log loss
        eps = 1e-15
        clipped = np.clip(preds, eps, 1.0 - eps)
        log_loss_val = -float(np.mean(
            outs * np.log(clipped) + (1.0 - outs) * np.log(1.0 - clipped)
        ))

        reliability_data = {
            "bin_edges": bin_edges,
            "expected": expected,
            "observed": observed,
            "counts": bin_counts,
        }

        return CalibrationResult(
            expected_proportions=expected,
            observed_proportions=observed,
            calibration_error=calibration_error,
            max_calibration_error=max_cal_error,
            brier_score=brier,
            log_loss=log_loss_val,
            reliability_data=reliability_data,
        )

    # ======================================================================
    # Model comparison utilities
    # ======================================================================

    def compare_models_waic(
        self,
        model_log_liks: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Compare multiple models using WAIC.

        Parameters
        ----------
        model_log_liks : dict mapping model name to (S, N) log-likelihood
            matrices.

        Returns
        -------
        dict with rankings, delta WAIC, and weights.
        """
        results: Dict[str, InformationCriteriaResult] = {}
        for name, ll in model_log_liks.items():
            results[name] = self.information_criteria({"log_likelihood": ll})

        waics = {name: r.waic for name, r in results.items()}
        sorted_models = sorted(waics.keys(), key=lambda x: waics[x])
        best_waic = waics[sorted_models[0]]

        delta_waics = {name: waics[name] - best_waic for name in sorted_models}

        weights_raw = np.array([
            math.exp(-0.5 * delta_waics[name]) for name in sorted_models
        ])
        weights_sum = np.sum(weights_raw)
        if weights_sum < 1e-300:
            weights_norm = np.ones(len(sorted_models)) / len(sorted_models)
        else:
            weights_norm = weights_raw / weights_sum
        model_weights = {
            name: float(weights_norm[i])
            for i, name in enumerate(sorted_models)
        }

        return {
            "ranking": sorted_models,
            "waic": waics,
            "delta_waic": delta_waics,
            "weights": model_weights,
            "details": results,
        }

    # ======================================================================
    # Additional posterior comparison utilities
    # ======================================================================

    def probability_of_superiority(
        self,
        scores_a: Union[np.ndarray, List[float]],
        scores_b: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Common-language effect size / probability of superiority.

        Estimates P(X_a > X_b) for random draws from the two groups.
        """
        a = _validate_scores(scores_a)
        b = _validate_scores(scores_b)
        count = 0
        total = 0
        for ai in a:
            for bi in b:
                total += 1
                if ai > bi:
                    count += 1
                elif ai == bi:
                    count += 0.5
        return count / max(total, 1)

    def overlap_coefficient(
        self,
        post_a: PosteriorEstimate,
        post_b: PosteriorEstimate,
        n_grid: int = 1000,
    ) -> float:
        """
        Compute the overlap coefficient (OVL) between two posteriors.

        OVL = integral of min(f_A(x), f_B(x)) dx
        """
        all_samples = np.concatenate([post_a.samples, post_b.samples])
        lo = float(np.min(all_samples))
        hi = float(np.max(all_samples))
        margin = 0.1 * (hi - lo)
        grid = np.linspace(lo - margin, hi + margin, n_grid)

        try:
            kde_a = stats.gaussian_kde(post_a.samples)
            kde_b = stats.gaussian_kde(post_b.samples)
            density_a = kde_a(grid)
            density_b = kde_b(grid)
        except Exception:
            return 0.0

        min_density = np.minimum(density_a, density_b)
        dx = grid[1] - grid[0]
        return float(np.sum(min_density) * dx)

    def divergence_kl(
        self,
        post_a: PosteriorEstimate,
        post_b: PosteriorEstimate,
        n_grid: int = 1000,
    ) -> float:
        """
        Approximate KL divergence KL(A || B) between two posteriors.
        """
        all_samples = np.concatenate([post_a.samples, post_b.samples])
        lo = float(np.min(all_samples))
        hi = float(np.max(all_samples))
        margin = 0.1 * (hi - lo)
        grid = np.linspace(lo - margin, hi + margin, n_grid)

        try:
            kde_a = stats.gaussian_kde(post_a.samples)
            kde_b = stats.gaussian_kde(post_b.samples)
            p = kde_a(grid)
            q = kde_b(grid)
        except Exception:
            return float("inf")

        eps = 1e-15
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)
        p /= np.sum(p)
        q /= np.sum(q)

        kl = float(np.sum(p * np.log(p / q)))
        return max(kl, 0.0)

    # ======================================================================
    # Bayesian correlation
    # ======================================================================

    def bayesian_correlation(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        n_samples: int = 10_000,
    ) -> PosteriorEstimate:
        """
        Posterior distribution of Pearson correlation using Fisher's z.

        Parameters
        ----------
        x, y : array-like
        n_samples : int

        Returns
        -------
        PosteriorEstimate
        """
        x_arr = _validate_scores(x)
        y_arr = _validate_scores(y)
        n = min(len(x_arr), len(y_arr))
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        if n < 3:
            raise ValueError("Need at least 3 paired observations.")

        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        r = np.clip(r, -0.999, 0.999)
        z = np.arctanh(r)
        se_z = 1.0 / math.sqrt(n - 3)

        z_samples = self._rng.normal(z, se_z, size=n_samples)
        r_samples = np.tanh(z_samples)

        return self._build_posterior_estimate(r_samples, "correlation")

    # ======================================================================
    # Bayesian proportion test
    # ======================================================================

    def bayesian_proportion_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        n_samples: int = 10_000,
    ) -> ComparisonResult:
        """
        Compare two proportions using Beta-Binomial model.

        Parameters
        ----------
        successes_a, trials_a : int  – successes and trials for group A.
        successes_b, trials_b : int  – successes and trials for group B.

        Returns
        -------
        ComparisonResult
        """
        alpha_a = successes_a + 0.5
        beta_a = trials_a - successes_a + 0.5
        alpha_b = successes_b + 0.5
        beta_b = trials_b - successes_b + 0.5

        samples_a = self._rng.beta(alpha_a, beta_a, size=n_samples)
        samples_b = self._rng.beta(alpha_b, beta_b, size=n_samples)
        diff_samples = samples_a - samples_b

        diff_post = self._build_posterior_estimate(diff_samples, "beta_diff")
        rope = self.rope_analysis(diff_samples, self.rope_low, self.rope_high)

        ratio = samples_a / np.maximum(samples_b, 1e-15)
        log_ratio = np.log(np.maximum(ratio, 1e-300))
        bf = float(np.exp(np.mean(log_ratio)))
        bf = min(bf, 1e12)

        es_samples = diff_samples / np.sqrt(
            samples_a * (1 - samples_a) / 2 + samples_b * (1 - samples_b) / 2
        )
        es_post = self._build_posterior_estimate(es_samples, "proportion_effect")

        p_a_better = float(np.mean(diff_samples > self.rope_high))
        p_b_better = float(np.mean(diff_samples < self.rope_low))
        p_equiv = float(np.mean(
            (diff_samples >= self.rope_low) & (diff_samples <= self.rope_high)
        ))

        return ComparisonResult(
            algorithm_a="A",
            algorithm_b="B",
            metric_name="proportion",
            posterior_diff=diff_post,
            rope_result=rope,
            bayes_factor=bf,
            effect_size=es_post,
            p_a_better=p_a_better,
            p_b_better=p_b_better,
            p_equivalent=p_equiv,
        )

    # ======================================================================
    # Bayesian ANOVA
    # ======================================================================

    def bayesian_anova(
        self,
        groups: Dict[str, Union[np.ndarray, List[float]]],
    ) -> Dict[str, Any]:
        """
        Bayesian one-way ANOVA via posterior on the between-group
        effect relative to within-group variance.

        Parameters
        ----------
        groups : dict mapping group name -> scores.

        Returns
        -------
        dict with group posteriors, overall effect posterior, and
        pairwise comparisons.
        """
        validated = {k: _validate_scores(v) for k, v in groups.items()}
        names = sorted(validated.keys())
        k = len(names)

        group_posteriors: Dict[str, PosteriorEstimate] = {}
        for name in names:
            group_posteriors[name] = self.posterior_estimation(
                validated[name], self.prior_type
            )

        all_data = np.concatenate([validated[n] for n in names])
        grand_mean_post = self.posterior_estimation(all_data, self.prior_type)

        eta2_samples = np.zeros(self.n_samples)
        for s in range(self.n_samples):
            grand_sample = grand_mean_post.samples[s % len(grand_mean_post.samples)]
            ss_between = 0.0
            ss_total = 0.0
            for name in names:
                gp = group_posteriors[name]
                group_sample = gp.samples[s % len(gp.samples)]
                n_g = len(validated[name])
                ss_between += n_g * (group_sample - grand_sample) ** 2
                for val in validated[name]:
                    ss_total += (val - grand_sample) ** 2
            ss_total = max(ss_total, 1e-15)
            eta2_samples[s] = ss_between / ss_total

        eta2_post = self._build_posterior_estimate(eta2_samples, "eta_squared")

        pairwise: Dict[Tuple[str, str], ComparisonResult] = {}
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i >= j:
                    continue
                pairwise[(na, nb)] = self.compare_two(
                    validated[na], validated[nb],
                    metric_name="anova",
                    algorithm_a=na, algorithm_b=nb,
                )

        return {
            "group_posteriors": group_posteriors,
            "grand_mean": grand_mean_post,
            "eta_squared": eta2_post,
            "pairwise": pairwise,
            "n_groups": k,
            "p_any_difference": float(np.mean(eta2_samples > 0.01)),
        }

    # ======================================================================
    # Mixture model comparison
    # ======================================================================

    def mixture_model_fit(
        self,
        data: Union[np.ndarray, List[float]],
        n_components: int = 2,
        n_iter: int = 100,
    ) -> Dict[str, Any]:
        """
        Fit a Gaussian mixture model using EM with Bayesian information criterion.

        Parameters
        ----------
        data : array-like
        n_components : int
        n_iter : int

        Returns
        -------
        dict with component parameters, responsibilities, BIC.
        """
        x = _validate_scores(data)
        n = len(x)
        k = n_components

        indices = np.linspace(0, n - 1, k + 2, dtype=int)[1:-1]
        sorted_x = np.sort(x)
        mus = np.array([float(sorted_x[i]) for i in indices])
        sigmas = np.full(k, float(np.std(x, ddof=1)))
        pis = np.ones(k) / k

        log_lik_prev = -np.inf
        for iteration in range(n_iter):
            # E-step
            log_resp = np.zeros((n, k))
            for c in range(k):
                log_resp[:, c] = (
                    np.log(max(pis[c], 1e-300))
                    + stats.norm.logpdf(x, loc=mus[c], scale=max(sigmas[c], 1e-15))
                )
            log_resp_max = np.max(log_resp, axis=1, keepdims=True)
            log_resp -= log_resp_max
            resp = np.exp(log_resp)
            resp_sum = np.sum(resp, axis=1, keepdims=True)
            resp /= np.maximum(resp_sum, 1e-300)

            # M-step
            n_k = np.sum(resp, axis=0)
            for c in range(k):
                if n_k[c] < 1e-10:
                    continue
                mus[c] = np.sum(resp[:, c] * x) / n_k[c]
                diff = x - mus[c]
                sigmas[c] = math.sqrt(
                    max(np.sum(resp[:, c] * diff ** 2) / n_k[c], 1e-10)
                )
            pis = n_k / n
            pis = np.maximum(pis, 1e-10)
            pis /= np.sum(pis)

            log_lik = 0.0
            for i in range(n):
                s = 0.0
                for c in range(k):
                    s += pis[c] * stats.norm.pdf(
                        x[i], loc=mus[c], scale=max(sigmas[c], 1e-15)
                    )
                log_lik += _stable_log(s)

            if abs(log_lik - log_lik_prev) < 1e-8:
                break
            log_lik_prev = log_lik

        n_params = k * 3 - 1
        bic = -2.0 * log_lik + n_params * math.log(n)
        aic = -2.0 * log_lik + 2 * n_params

        components = []
        for c in range(k):
            components.append({
                "mean": float(mus[c]),
                "std": float(sigmas[c]),
                "weight": float(pis[c]),
            })

        return {
            "components": components,
            "bic": bic,
            "aic": aic,
            "log_likelihood": log_lik,
            "n_components": k,
            "converged": iteration < n_iter - 1,
        }

    def select_n_components(
        self,
        data: Union[np.ndarray, List[float]],
        max_components: int = 5,
    ) -> Dict[str, Any]:
        """
        Select optimal number of mixture components using BIC.
        """
        x = _validate_scores(data)
        results = {}
        bics = {}
        for k in range(1, max_components + 1):
            fit = self.mixture_model_fit(x, n_components=k)
            results[k] = fit
            bics[k] = fit["bic"]

        best_k = min(bics, key=lambda k: bics[k])
        return {
            "best_n_components": best_k,
            "bic_values": bics,
            "all_fits": results,
            "best_fit": results[best_k],
        }

    # ======================================================================
    # Bayesian change-point detection
    # ======================================================================

    def change_point_detection(
        self,
        data: Union[np.ndarray, List[float]],
        max_changepoints: int = 5,
    ) -> Dict[str, Any]:
        """
        Bayesian change-point detection for identifying regime shifts
        in algorithm performance over time.

        Parameters
        ----------
        data : array-like  – time-ordered performance observations.
        max_changepoints : int

        Returns
        -------
        dict with detected change points and their posterior probabilities.
        """
        x = _validate_scores(data)
        n = len(x)
        if n < 4:
            return {"changepoints": [], "probabilities": []}

        marginal_cp = np.zeros(n)
        for t in range(2, n - 1):
            left = x[:t]
            right = x[t:]
            ll_separate = self._segment_log_likelihood(left) + self._segment_log_likelihood(right)
            ll_combined = self._segment_log_likelihood(x)
            log_bf = ll_separate - ll_combined
            marginal_cp[t] = 1.0 / (1.0 + math.exp(-min(max(log_bf, -500), 500)))

        n_detect = min(max_changepoints, int(np.sum(marginal_cp > 0.5)))
        top_indices = np.argsort(-marginal_cp)[:max(n_detect, 1)]
        top_indices = np.sort(top_indices)

        changepoints = []
        probabilities = []
        for idx in top_indices:
            if marginal_cp[idx] > 0.3:
                changepoints.append(int(idx))
                probabilities.append(float(marginal_cp[idx]))

        segments = []
        boundaries = [0] + changepoints + [n]
        for i in range(len(boundaries) - 1):
            seg = x[boundaries[i]:boundaries[i + 1]]
            if len(seg) > 0:
                segments.append({
                    "start": boundaries[i],
                    "end": boundaries[i + 1],
                    "mean": float(np.mean(seg)),
                    "std": float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0,
                    "n": len(seg),
                })

        return {
            "changepoints": changepoints,
            "probabilities": probabilities,
            "marginal_probabilities": marginal_cp.tolist(),
            "segments": segments,
            "n_detected": len(changepoints),
        }

    def _segment_log_likelihood(self, segment: np.ndarray) -> float:
        """Log marginal likelihood of a segment under a normal model."""
        n = len(segment)
        if n < 2:
            return 0.0
        mean = float(np.mean(segment))
        var = float(np.var(segment, ddof=1))
        var = max(var, 1e-15)
        ll = -0.5 * n * math.log(2 * math.pi * var)
        ll -= 0.5 * np.sum((segment - mean) ** 2) / var
        ll -= 0.5 * math.log(n)
        return float(ll)

    # ======================================================================
    # Bayesian nonparametric density estimation
    # ======================================================================

    def density_estimation(
        self,
        data: Union[np.ndarray, List[float]],
        n_grid: int = 200,
    ) -> Dict[str, Any]:
        """
        Bayesian nonparametric density estimation using a Dirichlet Process
        mixture approximation (via KDE with uncertainty bands).

        Parameters
        ----------
        data : array-like
        n_grid : int

        Returns
        -------
        dict with grid, density, and credible bands.
        """
        x = _validate_scores(data)
        n = len(x)

        lo = float(np.min(x))
        hi = float(np.max(x))
        margin = 0.15 * (hi - lo) if hi > lo else 1.0
        grid = np.linspace(lo - margin, hi + margin, n_grid)

        n_boot = 500
        density_matrix = np.zeros((n_boot, n_grid))
        for b in range(n_boot):
            boot_sample = self._rng.choice(x, size=n, replace=True)
            try:
                kde = stats.gaussian_kde(boot_sample)
                density_matrix[b] = kde(grid)
            except Exception:
                h = 1.06 * float(np.std(boot_sample, ddof=1)) * n ** (-0.2)
                h = max(h, 1e-10)
                for gi, gv in enumerate(grid):
                    density_matrix[b, gi] = float(
                        np.mean(np.exp(-0.5 * ((boot_sample - gv) / h) ** 2))
                    ) / (h * math.sqrt(2 * math.pi))

        mean_density = np.mean(density_matrix, axis=0)
        lower_band = np.quantile(density_matrix, 0.025, axis=0)
        upper_band = np.quantile(density_matrix, 0.975, axis=0)

        return {
            "grid": grid.tolist(),
            "density": mean_density.tolist(),
            "lower_95": lower_band.tolist(),
            "upper_95": upper_band.tolist(),
            "n_bootstrap": n_boot,
            "n_data": n,
        }

    # ======================================================================
    # Utility: Bayesian sample size determination
    # ======================================================================

    def sample_size_determination(
        self,
        pilot_a: Union[np.ndarray, List[float]],
        pilot_b: Union[np.ndarray, List[float]],
        target_precision: float = 0.01,
        target_power: float = 0.80,
        max_n: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Bayesian sample size determination for comparing two algorithms.

        Uses simulation to estimate the minimum sample size needed to
        achieve a target posterior precision or power.

        Parameters
        ----------
        pilot_a, pilot_b : array-like  – pilot data for estimation.
        target_precision : float  – target HDI width.
        target_power : float  – desired probability of correct decision.
        max_n : int

        Returns
        -------
        dict with required sample size and simulation details.
        """
        a = _validate_scores(pilot_a)
        b = _validate_scores(pilot_b)

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        std_a = float(np.std(a, ddof=1)) if len(a) > 1 else 1.0
        std_b = float(np.std(b, ddof=1)) if len(b) > 1 else 1.0

        candidate_ns = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000, max_n]
        candidate_ns = [n for n in candidate_ns if n <= max_n]

        results_by_n: Dict[int, Dict[str, float]] = {}
        required_n_precision: Optional[int] = None
        required_n_power: Optional[int] = None

        n_sim = 200
        for cand_n in candidate_ns:
            hdi_widths = []
            correct_decisions = 0

            for _ in range(n_sim):
                sim_a = self._rng.normal(mean_a, std_a, size=cand_n)
                sim_b = self._rng.normal(mean_b, std_b, size=cand_n)
                sim_diff = sim_a - sim_b

                post = self.posterior_estimation(sim_diff, PriorType.UNIFORM)
                hdi = post.credible_intervals.get(0.95)
                if hdi is not None:
                    hdi_widths.append(hdi.width)

                p_correct = float(np.mean(post.samples > 0.0))
                true_better = mean_a > mean_b
                if true_better and p_correct > 0.95:
                    correct_decisions += 1
                elif not true_better and (1 - p_correct) > 0.95:
                    correct_decisions += 1

            mean_width = float(np.mean(hdi_widths)) if hdi_widths else float("inf")
            power = correct_decisions / n_sim

            results_by_n[cand_n] = {
                "mean_hdi_width": mean_width,
                "power": power,
            }

            if required_n_precision is None and mean_width <= target_precision:
                required_n_precision = cand_n
            if required_n_power is None and power >= target_power:
                required_n_power = cand_n

        return {
            "required_n_precision": required_n_precision,
            "required_n_power": required_n_power,
            "target_precision": target_precision,
            "target_power": target_power,
            "pilot_effect": mean_a - mean_b,
            "results_by_n": results_by_n,
        }

    # ======================================================================
    # Bayesian regression for trend analysis
    # ======================================================================

    def bayesian_regression(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        n_samples: int = 10_000,
    ) -> Dict[str, PosteriorEstimate]:
        """
        Bayesian simple linear regression y = alpha + beta * x + epsilon.

        Uses conjugate normal-inverse-gamma prior.

        Parameters
        ----------
        x, y : array-like
        n_samples : int

        Returns
        -------
        dict with posteriors for 'intercept', 'slope', 'sigma', 'r_squared'.
        """
        x_arr = _validate_scores(x)
        y_arr = _validate_scores(y)
        n = min(len(x_arr), len(y_arr))
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        if n < 3:
            raise ValueError("Need at least 3 data points.")

        x_mean = float(np.mean(x_arr))
        y_mean = float(np.mean(y_arr))
        x_c = x_arr - x_mean
        y_c = y_arr - y_mean

        ss_xx = float(np.sum(x_c ** 2))
        ss_xy = float(np.sum(x_c * y_c))
        ss_yy = float(np.sum(y_c ** 2))

        if ss_xx < 1e-15:
            raise ValueError("No variance in x.")

        beta_hat = ss_xy / ss_xx
        alpha_hat = y_mean - beta_hat * x_mean

        residuals = y_arr - alpha_hat - beta_hat * x_arr
        s2 = float(np.sum(residuals ** 2)) / max(n - 2, 1)
        s2 = max(s2, 1e-15)

        dof = n - 2

        sigma2_samples = s2 * dof / self._rng.chisquare(dof, size=n_samples)
        sigma2_samples = np.maximum(sigma2_samples, 1e-15)

        beta_se = np.sqrt(sigma2_samples / ss_xx)
        beta_samples = self._rng.normal(beta_hat, beta_se)

        alpha_se = np.sqrt(sigma2_samples * (1.0 / n + x_mean ** 2 / ss_xx))
        alpha_samples = self._rng.normal(alpha_hat, alpha_se)

        sigma_samples = np.sqrt(sigma2_samples)

        ss_res_samples = sigma2_samples * dof
        r2_samples = 1.0 - ss_res_samples / max(ss_yy, 1e-15)
        r2_samples = np.clip(r2_samples, 0.0, 1.0)

        return {
            "intercept": self._build_posterior_estimate(alpha_samples, "regression_intercept"),
            "slope": self._build_posterior_estimate(beta_samples, "regression_slope"),
            "sigma": self._build_posterior_estimate(sigma_samples, "regression_sigma"),
            "r_squared": self._build_posterior_estimate(r2_samples, "regression_r2"),
        }

    # ======================================================================
    # Robust Bayesian comparison (outlier-resistant)
    # ======================================================================

    def robust_comparison(
        self,
        scores_a: Union[np.ndarray, List[float]],
        scores_b: Union[np.ndarray, List[float]],
        nu: float = 4.0,
    ) -> ComparisonResult:
        """
        Robust Bayesian comparison using a Student-t likelihood instead
        of normal, making it resistant to outliers.

        Parameters
        ----------
        scores_a, scores_b : array-like
        nu : float – degrees of freedom for the t-distribution.
            Lower values give more robustness.

        Returns
        -------
        ComparisonResult
        """
        a = _validate_scores(scores_a)
        b = _validate_scores(scores_b)

        def robust_mean_samples(data: np.ndarray, n_samp: int) -> np.ndarray:
            n = len(data)
            median_d = float(np.median(data))
            mad = float(np.median(np.abs(data - median_d)))
            scale = mad * 1.4826
            scale = max(scale, 1e-15)

            weights = np.zeros(n)
            for i in range(n):
                z = (data[i] - median_d) / scale
                weights[i] = (nu + 1) / (nu + z ** 2)

            w_sum = np.sum(weights)
            weighted_mean = float(np.sum(weights * data) / w_sum)
            weighted_var = float(np.sum(weights * (data - weighted_mean) ** 2) / w_sum)
            se = math.sqrt(max(weighted_var / n, 1e-15))

            dof = max(n - 1, 1)
            return self._rng.standard_t(dof, size=n_samp) * se + weighted_mean

        post_a = robust_mean_samples(a, self.n_samples)
        post_b = robust_mean_samples(b, self.n_samples)
        diff_samples = post_a - post_b

        diff_post = self._build_posterior_estimate(diff_samples, "robust_diff")
        rope = self.rope_analysis(diff_samples, self.rope_low, self.rope_high)

        pooled_mad = float(np.median(np.abs(np.concatenate([
            a - np.median(a), b - np.median(b)
        ]))))
        pooled_scale = max(pooled_mad * 1.4826, 1e-15)
        es_samples = diff_samples / pooled_scale
        es_post = self._build_posterior_estimate(es_samples, "robust_effect")

        p_a = float(np.mean(diff_samples > self.rope_high))
        p_b = float(np.mean(diff_samples < self.rope_low))
        p_e = float(np.mean(
            (diff_samples >= self.rope_low) & (diff_samples <= self.rope_high)
        ))

        bf = self._sequential_bf(a, b)

        return ComparisonResult(
            algorithm_a="A",
            algorithm_b="B",
            metric_name="robust",
            posterior_diff=diff_post,
            rope_result=rope,
            bayes_factor=bf,
            effect_size=es_post,
            p_a_better=p_a,
            p_b_better=p_b,
            p_equivalent=p_e,
            raw_scores_a=a,
            raw_scores_b=b,
        )

    # ======================================================================
    # Posterior summary report
    # ======================================================================

    def full_report(
        self,
        score_dict: Dict[str, Union[np.ndarray, List[float]]],
        metric_name: str = "score",
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive Bayesian comparison report.

        Parameters
        ----------
        score_dict : dict mapping algorithm name -> scores.
        metric_name : str

        Returns
        -------
        dict with all analyses combined.
        """
        comparison = self.compare_multiple(score_dict, metric_name)

        sensitivity = {}
        for name, scores in score_dict.items():
            sensitivity[name] = self.sensitivity_to_prior(scores)

        names = sorted(score_dict.keys())
        ppc = {}
        for name in names:
            data = _validate_scores(score_dict[name])
            post = comparison["posteriors"][name]
            ppc[name] = self.posterior_predictive_check(post, data)

        diagnostics = {}
        for name in names:
            post = comparison["posteriors"][name]
            diagnostics[name] = self.run_diagnostics(post.samples)

        sequential_results = {}
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i >= j:
                    continue
                a = _validate_scores(score_dict[na])
                b = _validate_scores(score_dict[nb])
                n_paired = min(len(a), len(b))
                if n_paired >= 20:
                    sequential_results[(na, nb)] = self.sequential_analysis(
                        a[:n_paired], b[:n_paired]
                    )

        return {
            "comparison": comparison,
            "sensitivity": sensitivity,
            "posterior_predictive_checks": ppc,
            "diagnostics": diagnostics,
            "sequential": sequential_results,
            "metric_name": metric_name,
        }

    # ======================================================================
    # Bayesian power analysis
    # ======================================================================

    def bayesian_power_analysis(
        self,
        effect_size: float,
        std: float = 1.0,
        n_range: Optional[List[int]] = None,
        n_simulations: int = 500,
        decision_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Estimate Bayesian power (probability of making a correct decision)
        for various sample sizes.

        Parameters
        ----------
        effect_size : float  – expected Cohen's d.
        std : float  – standard deviation.
        n_range : list of ints  – sample sizes to evaluate.
        n_simulations : int
        decision_threshold : float

        Returns
        -------
        dict mapping sample size to estimated power.
        """
        if n_range is None:
            n_range = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

        mean_diff = effect_size * std
        results: Dict[int, Dict[str, float]] = {}

        for n in n_range:
            correct = 0
            decisive = 0
            for _ in range(n_simulations):
                sim_a = self._rng.normal(mean_diff, std, size=n)
                sim_b = self._rng.normal(0.0, std, size=n)
                diff = sim_a - sim_b
                post = self.posterior_estimation(diff, PriorType.UNIFORM)
                p_greater = float(np.mean(post.samples > 0))
                if p_greater > decision_threshold:
                    correct += 1
                    decisive += 1
                elif p_greater < (1.0 - decision_threshold):
                    decisive += 1

            results[n] = {
                "power": correct / n_simulations,
                "decisiveness": decisive / n_simulations,
            }

        required_n = None
        for n in sorted(results.keys()):
            if results[n]["power"] >= 0.80:
                required_n = n
                break

        return {
            "results_by_n": results,
            "effect_size": effect_size,
            "std": std,
            "required_n_80": required_n,
            "decision_threshold": decision_threshold,
        }

    # ======================================================================
    # Bayesian hypothesis testing with multiple hypotheses
    # ======================================================================

    def bayesian_model_comparison(
        self,
        data: Union[np.ndarray, List[float]],
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare how well different distributional models fit the data.

        Parameters
        ----------
        data : array-like
        models : list of model names (subset of 'normal', 'cauchy', 'laplace').

        Returns
        -------
        dict with log marginal likelihoods, Bayes factors, and posterior
        model probabilities.
        """
        x = _validate_scores(data)
        if models is None:
            models = ["normal", "cauchy", "laplace"]

        log_ml: Dict[str, float] = {}
        for model in models:
            log_ml[model] = self._log_marginal_likelihood(x, model)

        best_model = max(log_ml, key=lambda m: log_ml[m])
        best_log_ml = log_ml[best_model]

        bayes_factors: Dict[str, float] = {}
        for model in models:
            log_bf = log_ml[model] - best_log_ml
            bayes_factors[model] = math.exp(min(max(log_bf, -500), 500))

        total_bf = sum(bayes_factors.values())
        posterior_probs = {
            m: bf / total_bf for m, bf in bayes_factors.items()
        }

        return {
            "log_marginal_likelihoods": log_ml,
            "bayes_factors_vs_best": bayes_factors,
            "posterior_model_probabilities": posterior_probs,
            "best_model": best_model,
        }

    # ======================================================================
    # Utility: posterior predictive distribution
    # ======================================================================

    def posterior_predictive(
        self,
        model: PosteriorEstimate,
        n_predictions: int = 1000,
    ) -> np.ndarray:
        """
        Sample from the posterior predictive distribution.

        For each posterior sample of the mean, draw a new observation.

        Parameters
        ----------
        model : PosteriorEstimate
        n_predictions : int

        Returns
        -------
        np.ndarray of predicted observations.
        """
        indices = self._rng.integers(
            0, len(model.samples), size=n_predictions
        )
        means = model.samples[indices]
        predictions = self._rng.normal(means, model.std)
        return predictions

    # ======================================================================
    # Credible interval comparison
    # ======================================================================

    def compare_intervals(
        self,
        post_a: PosteriorEstimate,
        post_b: PosteriorEstimate,
        level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Compare credible intervals of two posteriors.

        Returns overlap percentage and whether intervals are disjoint.
        """
        ci_a = self._hdi(post_a.samples, 1.0 - level)
        ci_b = self._hdi(post_b.samples, 1.0 - level)

        overlap_lo = max(ci_a.lower, ci_b.lower)
        overlap_hi = min(ci_a.upper, ci_b.upper)

        if overlap_lo >= overlap_hi:
            overlap = 0.0
            disjoint = True
        else:
            overlap = overlap_hi - overlap_lo
            disjoint = False

        union_lo = min(ci_a.lower, ci_b.lower)
        union_hi = max(ci_a.upper, ci_b.upper)
        union_width = union_hi - union_lo

        jaccard = overlap / max(union_width, 1e-15) if not disjoint else 0.0

        return {
            "ci_a": ci_a,
            "ci_b": ci_b,
            "overlap": overlap,
            "disjoint": disjoint,
            "jaccard_index": jaccard,
            "union_width": union_width,
        }

    # ======================================================================
    # Batch evaluation helper
    # ======================================================================

    def batch_compare(
        self,
        score_dict: Dict[str, Union[np.ndarray, List[float]]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comparisons across multiple metrics simultaneously.

        Parameters
        ----------
        score_dict : if metrics is None, keys are algorithm names.
            If metrics is provided, expected to be
            Dict[metric, Dict[algorithm, scores]].
        metrics : list of metric names, or None.

        Returns
        -------
        dict mapping metric name -> comparison results.
        """
        if metrics is None:
            return {"default": self.compare_multiple(score_dict, "default")}

        results: Dict[str, Dict[str, Any]] = {}
        for metric in metrics:
            if metric in score_dict:
                metric_scores = score_dict[metric]
                if isinstance(metric_scores, dict):
                    results[metric] = self.compare_multiple(metric_scores, metric)
        return results

    # ======================================================================
    # Posterior convergence monitoring
    # ======================================================================

    def convergence_monitoring(
        self,
        samples: np.ndarray,
        window_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Monitor posterior convergence by tracking statistics over
        expanding windows of the sample chain.

        Parameters
        ----------
        samples : 1-D array of posterior samples.
        window_size : int  – step size for expanding window.

        Returns
        -------
        dict with convergence trajectory data.
        """
        samples = np.asarray(samples, dtype=np.float64)
        n = len(samples)

        trajectory = {
            "window_ends": [],
            "running_mean": [],
            "running_std": [],
            "running_ess": [],
            "running_geweke_z": [],
        }

        for end in range(window_size, n + 1, window_size):
            window = samples[:end]
            trajectory["window_ends"].append(end)
            trajectory["running_mean"].append(float(np.mean(window)))
            trajectory["running_std"].append(float(np.std(window, ddof=1)))
            ess = self._effective_sample_size(window)
            trajectory["running_ess"].append(ess)
            geweke = self._geweke_diagnostic(window)
            trajectory["running_geweke_z"].append(geweke["z_score"])

        final_mean = trajectory["running_mean"][-1] if trajectory["running_mean"] else 0.0
        is_stable = True
        if len(trajectory["running_mean"]) > 3:
            recent = trajectory["running_mean"][-3:]
            spread = max(recent) - min(recent)
            if spread > 0.01 * abs(final_mean) + 1e-10:
                is_stable = False

        trajectory["converged"] = is_stable
        return trajectory

    # ======================================================================
    # Multi-metric aggregation
    # ======================================================================

    def aggregate_across_metrics(
        self,
        multi_metric_results: Dict[str, Dict[Tuple[str, str], ComparisonResult]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Aggregate pairwise comparison results across multiple metrics.

        Parameters
        ----------
        multi_metric_results : Dict mapping metric name to dict of
            (algo_a, algo_b) -> ComparisonResult.
        weights : optional dict mapping metric name to weight.

        Returns
        -------
        Dict mapping (algo_a, algo_b) -> aggregated summary.
        """
        metrics = list(multi_metric_results.keys())
        if weights is None:
            weights = {m: 1.0 / len(metrics) for m in metrics}

        total_w = sum(weights.values())
        weights = {m: w / total_w for m, w in weights.items()}

        all_pairs: set = set()
        for metric in metrics:
            all_pairs |= set(multi_metric_results[metric].keys())

        aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for pair in all_pairs:
            weighted_p_a = 0.0
            weighted_p_b = 0.0
            weighted_p_e = 0.0
            weighted_effect = 0.0
            metric_details = {}

            for metric in metrics:
                if pair not in multi_metric_results[metric]:
                    continue
                cr = multi_metric_results[metric][pair]
                w = weights.get(metric, 0.0)
                weighted_p_a += w * cr.p_a_better
                weighted_p_b += w * cr.p_b_better
                weighted_p_e += w * cr.p_equivalent
                weighted_effect += w * cr.effect_size.mean
                metric_details[metric] = {
                    "p_a_better": cr.p_a_better,
                    "p_b_better": cr.p_b_better,
                    "effect_size": cr.effect_size.mean,
                }

            aggregated[pair] = {
                "weighted_p_a_better": weighted_p_a,
                "weighted_p_b_better": weighted_p_b,
                "weighted_p_equivalent": weighted_p_e,
                "weighted_effect_size": weighted_effect,
                "per_metric": metric_details,
                "overall_winner": pair[0] if weighted_p_a > weighted_p_b else pair[1],
            }

        return aggregated

    # ======================================================================
    # Posterior shrinkage analysis
    # ======================================================================

    def shrinkage_analysis(
        self,
        raw_estimates: Dict[str, float],
        standard_errors: Dict[str, float],
        global_mean: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Empirical Bayes shrinkage analysis showing how individual estimates
        are pulled toward the grand mean.

        Parameters
        ----------
        raw_estimates : dict mapping name -> raw estimate.
        standard_errors : dict mapping name -> SE.
        global_mean : optional grand mean; estimated if None.

        Returns
        -------
        dict mapping name -> dict with 'raw', 'shrunk', 'shrinkage_factor'.
        """
        names = sorted(raw_estimates.keys())
        ests = np.array([raw_estimates[n] for n in names])
        ses = np.array([max(standard_errors[n], 1e-15) for n in names])

        if global_mean is None:
            weights = 1.0 / ses ** 2
            global_mean = float(np.sum(weights * ests) / np.sum(weights))

        tau2 = max(float(np.var(ests, ddof=1)) - float(np.mean(ses ** 2)), 0.0)

        result: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(names):
            shrinkage = ses[i] ** 2 / (ses[i] ** 2 + tau2)
            shrunk = (1 - shrinkage) * ests[i] + shrinkage * global_mean
            result[name] = {
                "raw": float(ests[i]),
                "shrunk": float(shrunk),
                "shrinkage_factor": float(shrinkage),
                "se": float(ses[i]),
            }

        return result

    # ======================================================================
    # Highest posterior density region for 2D
    # ======================================================================

    def hpd_region_2d(
        self,
        samples_x: np.ndarray,
        samples_y: np.ndarray,
        levels: Optional[List[float]] = None,
        n_grid: int = 100,
    ) -> Dict[float, np.ndarray]:
        """
        Compute 2-D highest posterior density contour levels.

        Parameters
        ----------
        samples_x, samples_y : 1-D arrays of posterior samples.
        levels : list of credible levels (e.g. [0.5, 0.9, 0.95]).
        n_grid : int  – grid resolution.

        Returns
        -------
        dict mapping level -> density threshold.
        """
        if levels is None:
            levels = [0.50, 0.90, 0.95]

        try:
            kde = stats.gaussian_kde(np.vstack([samples_x, samples_y]))
        except Exception:
            return {lev: np.array([0.0]) for lev in levels}

        x_grid = np.linspace(
            float(np.min(samples_x)), float(np.max(samples_x)), n_grid
        )
        y_grid = np.linspace(
            float(np.min(samples_y)), float(np.max(samples_y)), n_grid
        )
        xx, yy = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)

        sorted_z = np.sort(zz.ravel())[::-1]
        cumsum = np.cumsum(sorted_z)
        cumsum /= cumsum[-1]

        thresholds: Dict[float, np.ndarray] = {}
        for lev in levels:
            idx = np.searchsorted(cumsum, lev)
            idx = min(idx, len(sorted_z) - 1)
            thresholds[lev] = np.array([sorted_z[idx]])

        return thresholds

    # ======================================================================
    # Summary statistics
    # ======================================================================

    def summarize_posterior(
        self,
        posterior: PosteriorEstimate,
    ) -> Dict[str, Any]:
        """Human-readable summary of a posterior."""
        diagnostics = self.run_diagnostics(posterior.samples)
        return {
            "mean": posterior.mean,
            "std": posterior.std,
            "median": posterior.median,
            "mode": posterior.mode,
            "skewness": posterior.skewness,
            "kurtosis": posterior.kurtosis,
            "ci_95": posterior.credible_intervals.get(0.95),
            "ci_90": posterior.credible_intervals.get(0.90),
            "ci_50": posterior.credible_intervals.get(0.50),
            "ess": diagnostics["effective_sample_size"],
            "r_hat": diagnostics["r_hat"],
            "converged": diagnostics["converged"],
            "distribution_type": posterior.distribution_type,
        }


# =========================================================================
# Extended Bayesian analysis: result dataclasses
# =========================================================================

_JEFFREYS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "anecdotal": (1.0, 3.0),
    "moderate": (3.0, 10.0),
    "strong": (10.0, 30.0),
    "very_strong": (30.0, 100.0),
    "extreme": (100.0, float("inf")),
}

_DEFAULT_MCMC_SAMPLES: int = 4_000
_DEFAULT_MCMC_WARMUP: int = 1_000


@dataclass
class ROPESignTestResult:
    """Full ROPE sign-test result with visualisation data.

    Attributes
    ----------
    probability_left : float
        Posterior probability that algorithm A is practically worse.
    probability_rope : float
        Posterior probability that the algorithms are practically equivalent.
    probability_right : float
        Posterior probability that algorithm A is practically better.
    decision : str
        Summary decision string.
    posterior_samples : np.ndarray
        Dirichlet posterior samples (N x 3 array).
    category_counts : Dict[str, int]
        Raw counts of left / rope / right paired differences.
    visualization_data : Dict[str, Any]
        Pre-computed data suitable for ternary-plot rendering.
    """

    probability_left: float
    probability_rope: float
    probability_right: float
    decision: str
    posterior_samples: np.ndarray
    category_counts: Dict[str, int]
    visualization_data: Dict[str, Any]


@dataclass
class BayesPairedTTestResult:
    """Result of a Bayesian paired t-test.

    Attributes
    ----------
    mean_diff : float
        Posterior mean of the paired difference.
    std_diff : float
        Posterior standard deviation of the paired difference.
    effect_size : float
        Posterior mean of Cohen d.
    hdi_low : float
        Lower bound of the 95 percent HDI.
    hdi_high : float
        Upper bound of the 95 percent HDI.
    df : float
        Degrees of freedom of the posterior t distribution.
    samples : np.ndarray
        Posterior samples of the mean difference.
    """

    mean_diff: float
    std_diff: float
    effect_size: float
    hdi_low: float
    hdi_high: float
    df: float
    samples: np.ndarray


@dataclass
class BayesFactorResult:
    """Bayes factor comparison result.

    Attributes
    ----------
    bf10 : float
        Bayes factor in favour of H1.
    bf01 : float
        Bayes factor in favour of H0.
    log_bf : float
        Natural log of bf10.
    interpretation : str
        Human-readable interpretation on Jeffreys scale.
    method : str
        Estimation method used (e.g. savage_dickey).
    """

    bf10: float
    bf01: float
    log_bf: float
    interpretation: str
    method: str


@dataclass
class HierarchicalModelResult:
    """Result of a hierarchical Bayesian model fit.

    Attributes
    ----------
    group_effects : Dict[str, np.ndarray]
        Posterior samples for each group-level effect.
    random_effects : Dict[str, np.ndarray]
        Posterior samples for each random effect.
    variance_components : Dict[str, float]
        Estimated variance components.
    convergence_info : Dict[str, Any]
        Diagnostics such as R-hat and ESS.
    """

    group_effects: Dict[str, np.ndarray]
    random_effects: Dict[str, np.ndarray]
    variance_components: Dict[str, float]
    convergence_info: Dict[str, Any]


@dataclass
class SequentialUpdateResult:
    """Result of sequential Bayesian updating.

    Attributes
    ----------
    posterior_history : List[Dict[str, float]]
        Posterior parameters after each update step.
    stopping_decisions : List[bool]
        Whether the stopping criterion was met at each step.
    cumulative_bf : List[float]
        Running Bayes factor after each observation.
    """

    posterior_history: List[Dict[str, float]]
    stopping_decisions: List[bool]
    cumulative_bf: List[float]


@dataclass
class WAICResult:
    """Widely Applicable Information Criterion result.

    Attributes
    ----------
    waic : float
        WAIC estimate (lower is better).
    p_waic : float
        Effective number of parameters.
    se : float
        Standard error of the WAIC estimate.
    pointwise : np.ndarray
        Per-observation WAIC contributions.
    """

    waic: float
    p_waic: float
    se: float
    pointwise: np.ndarray


@dataclass
class PriorSensitivityResult:
    """Result of a prior sensitivity analysis.

    Attributes
    ----------
    prior_specs : List[Dict[str, Any]]
        Specification of each prior evaluated.
    posteriors : List[Dict[str, float]]
        Summary statistics for each resulting posterior.
    robustness_index : float
        Maximum absolute deviation in posterior means across priors.
    """

    prior_specs: List[Dict[str, Any]]
    posteriors: List[Dict[str, float]]
    robustness_index: float


# =========================================================================
# ROPESignTestFull
# =========================================================================

class ROPESignTestFull:
    """Full Dirichlet-based ROPE sign test with visualisation helpers.

    Parameters
    ----------
    rope_lower : float
        Lower bound of the ROPE.
    rope_upper : float
        Upper bound of the ROPE.
    prior_strength : float
        Dirichlet concentration prior (same for all three categories).
    n_samples : int
        Number of posterior Dirichlet samples.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        rope_lower: float = -0.01,
        rope_upper: float = 0.01,
        prior_strength: float = 1.0,
        n_samples: int = 50_000,
        seed: Optional[int] = None,
    ) -> None:
        if rope_lower >= rope_upper:
            raise ValueError("rope_lower must be strictly less than rope_upper")
        self._rope_lower = rope_lower
        self._rope_upper = rope_upper
        self._prior_strength = prior_strength
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> ROPESignTestResult:
        """Run the full ROPE sign test.

        Parameters
        ----------
        scores_a : np.ndarray
            Per-instance scores for algorithm A.
        scores_b : np.ndarray
            Per-instance scores for algorithm B.

        Returns
        -------
        ROPESignTestResult
            Posterior probabilities, decision, samples, and vis data.
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        if scores_a.shape != scores_b.shape:
            raise ValueError("scores_a and scores_b must have the same shape")

        diffs = scores_a - scores_b

        n_left = int(np.sum(diffs < self._rope_lower))
        n_rope = int(np.sum(
            (diffs >= self._rope_lower) & (diffs <= self._rope_upper)
        ))
        n_right = int(np.sum(diffs > self._rope_upper))

        alpha_post = np.array([
            self._prior_strength + n_left,
            self._prior_strength + n_rope,
            self._prior_strength + n_right,
        ])

        samples = self._rng.dirichlet(alpha_post, size=self._n_samples)

        p_left = float(np.mean(
            samples[:, 0] > np.maximum(samples[:, 1], samples[:, 2])
        ))
        p_rope = float(np.mean(
            samples[:, 1] > np.maximum(samples[:, 0], samples[:, 2])
        ))
        p_right = float(np.mean(
            samples[:, 2] > np.maximum(samples[:, 0], samples[:, 1])
        ))

        if p_left > 0.95:
            decision = "left"
        elif p_rope > 0.95:
            decision = "rope"
        elif p_right > 0.95:
            decision = "right"
        else:
            decision = "undecided"

        vis = self.visualization_data(samples)

        return ROPESignTestResult(
            probability_left=float(np.mean(samples[:, 0])),
            probability_rope=float(np.mean(samples[:, 1])),
            probability_right=float(np.mean(samples[:, 2])),
            decision=decision,
            posterior_samples=samples,
            category_counts={"left": n_left, "rope": n_rope, "right": n_right},
            visualization_data=vis,
        )

    # ------------------------------------------------------------------

    def visualization_data(
        self,
        samples: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute ternary plot coordinates and density data.

        Parameters
        ----------
        samples : np.ndarray
            Dirichlet samples of shape (N, 3).

        Returns
        -------
        Dict[str, Any]
            Keys: ternary_x, ternary_y, density_left, density_rope,
            density_right.
        """
        x = samples[:, 2] + samples[:, 1] / 2.0
        y = samples[:, 1] * (np.sqrt(3.0) / 2.0)

        return {
            "ternary_x": x,
            "ternary_y": y,
            "density_left": np.histogram(samples[:, 0], bins=50, density=True),
            "density_rope": np.histogram(samples[:, 1], bins=50, density=True),
            "density_right": np.histogram(samples[:, 2], bins=50, density=True),
        }

    # ------------------------------------------------------------------

    def sequential_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        step: int = 1,
    ) -> List[ROPESignTestResult]:
        """Run the ROPE sign test incrementally on streaming data.

        Parameters
        ----------
        scores_a : np.ndarray
            Per-instance scores for algorithm A.
        scores_b : np.ndarray
            Per-instance scores for algorithm B.
        step : int
            Number of observations to add between snapshots.

        Returns
        -------
        List[ROPESignTestResult]
            One result per snapshot.
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        n = len(scores_a)
        results: List[ROPESignTestResult] = []
        for end in range(step, n + 1, step):
            results.append(self.test(scores_a[:end], scores_b[:end]))
        return results


# =========================================================================
# BayesPairedTTest
# =========================================================================

class BayesPairedTTest:
    """Bayesian paired t-test with Normal-Inverse-Gamma prior.

    Parameters
    ----------
    prior_mean : float
        Prior mean for the mean difference.
    prior_var : float
        Prior variance for the mean difference.
    prior_df : float
        Prior degrees of freedom (> 0).
    n_samples : int
        Number of posterior samples.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        prior_df: float = 1.0,
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self._prior_mean = prior_mean
        self._prior_var = prior_var
        self._prior_df = prior_df
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> BayesPairedTTestResult:
        """Run the Bayesian paired t-test.

        Parameters
        ----------
        scores_a : np.ndarray
            Scores for algorithm A.
        scores_b : np.ndarray
            Scores for algorithm B.

        Returns
        -------
        BayesPairedTTestResult
            Posterior mean difference, effect size, HDI, samples.
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        diffs = scores_a - scores_b
        n = len(diffs)

        data_mean = float(np.mean(diffs))
        data_var = float(np.var(diffs, ddof=1)) if n > 1 else 1.0

        # Normal-Inverse-Gamma posterior update
        prior_prec = 1.0 / self._prior_var
        post_prec = prior_prec + n
        post_mean = (prior_prec * self._prior_mean + n * data_mean) / post_prec
        post_df = self._prior_df + n

        post_scale_num = (
            self._prior_df * self._prior_var
            + (n - 1) * data_var
            + (prior_prec * n / post_prec)
            * (data_mean - self._prior_mean) ** 2
        )
        post_scale = post_scale_num / post_df

        # Sample from posterior t-distribution
        samples = stats.t.rvs(
            df=post_df,
            loc=post_mean,
            scale=np.sqrt(post_scale / post_prec),
            size=self._n_samples,
            random_state=self._rng.integers(2**31),
        )

        hdi_low, hdi_high = self._compute_hdi(samples, credibility=0.95)

        pooled_std = np.sqrt(data_var) if data_var > 0 else 1.0
        effect_size = float(np.mean(samples)) / pooled_std

        return BayesPairedTTestResult(
            mean_diff=float(np.mean(samples)),
            std_diff=float(np.std(samples)),
            effect_size=effect_size,
            hdi_low=hdi_low,
            hdi_high=hdi_high,
            df=post_df,
            samples=samples,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hdi(
        samples: np.ndarray,
        credibility: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute the highest density interval assuming unimodal density.

        Parameters
        ----------
        samples : np.ndarray
            One-dimensional posterior samples.
        credibility : float
            Credibility mass (e.g. 0.95).

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of the HDI.
        """
        sorted_pts = np.sort(samples)
        n = len(sorted_pts)
        ci_size = int(np.ceil(credibility * n))
        if ci_size >= n:
            return float(sorted_pts[0]), float(sorted_pts[-1])

        widths = sorted_pts[ci_size:] - sorted_pts[: n - ci_size]
        best = int(np.argmin(widths))
        return float(sorted_pts[best]), float(sorted_pts[best + ci_size])


# =========================================================================
# PosteriorDifferenceAnalyzer
# =========================================================================

class PosteriorDifferenceAnalyzer:
    """Analyse the posterior distribution over metric differences.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples to draw.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def compute_difference_posterior(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> np.ndarray:
        """Bootstrap the posterior over the mean difference.

        Parameters
        ----------
        scores_a : np.ndarray
            Scores for algorithm A.
        scores_b : np.ndarray
            Scores for algorithm B.

        Returns
        -------
        np.ndarray
            Posterior samples of the mean difference.
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        diffs = scores_a - scores_b
        n = len(diffs)
        idx = self._rng.integers(0, n, size=(self._n_samples, n))
        return np.mean(diffs[idx], axis=1)

    # ------------------------------------------------------------------

    def hdi(
        self,
        samples: np.ndarray,
        credibility: float = 0.95,
    ) -> Tuple[float, float]:
        """Highest density interval at arbitrary credibility.

        Parameters
        ----------
        samples : np.ndarray
            Posterior samples.
        credibility : float
            Desired credibility mass.

        Returns
        -------
        Tuple[float, float]
            (lower, upper) bounds.
        """
        return BayesPairedTTest._compute_hdi(samples, credibility)

    # ------------------------------------------------------------------

    def probability_of_direction(
        self,
        samples: np.ndarray,
    ) -> float:
        """Compute the probability that the difference is positive.

        Parameters
        ----------
        samples : np.ndarray
            Posterior samples of the difference.

        Returns
        -------
        float
            P(diff > 0).
        """
        return float(np.mean(samples > 0))

    # ------------------------------------------------------------------

    def effect_size_posterior(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> np.ndarray:
        """Posterior samples of Cohen d.

        Parameters
        ----------
        scores_a : np.ndarray
            Scores for algorithm A.
        scores_b : np.ndarray
            Scores for algorithm B.

        Returns
        -------
        np.ndarray
            Posterior samples of Cohen d.
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        diffs = scores_a - scores_b
        n = len(diffs)
        idx = self._rng.integers(0, n, size=(self._n_samples, n))
        boot_means = np.mean(diffs[idx], axis=1)
        boot_stds = np.std(diffs[idx], axis=1, ddof=1)
        boot_stds = np.where(boot_stds == 0, 1.0, boot_stds)
        return boot_means / boot_stds


# =========================================================================
# BayesFactorComputer
# =========================================================================

class BayesFactorComputer:
    """Compute Bayes factors via Savage-Dickey or bridge sampling.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def savage_dickey(
        self,
        posterior_samples: np.ndarray,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        null_value: float = 0.0,
    ) -> BayesFactorResult:
        """Savage-Dickey density-ratio Bayes factor at the null.

        Parameters
        ----------
        posterior_samples : np.ndarray
            Samples from the posterior of the parameter of interest.
        prior_mean : float
            Mean of the normal prior.
        prior_std : float
            Standard deviation of the normal prior.
        null_value : float
            Parameter value under H0.

        Returns
        -------
        BayesFactorResult
            Bayes factor and interpretation.
        """
        posterior_samples = np.asarray(posterior_samples, dtype=np.float64)

        # Prior density at the null value
        prior_density = float(
            stats.norm.pdf(null_value, loc=prior_mean, scale=prior_std)
        )

        # KDE estimate of the posterior density at the null value
        if np.std(posterior_samples) < 1e-12:
            post_density = 1e-12
        else:
            kde = stats.gaussian_kde(posterior_samples)
            post_density = float(kde(null_value)[0])

        bf01 = post_density / prior_density if prior_density > 0 else float("inf")
        bf10 = 1.0 / bf01 if bf01 > 0 else float("inf")
        log_bf = float(np.log(bf10)) if bf10 > 0 else float("-inf")

        return BayesFactorResult(
            bf10=bf10,
            bf01=bf01,
            log_bf=log_bf,
            interpretation=self.interpret(bf10),
            method="savage_dickey",
        )

    # ------------------------------------------------------------------

    def bridge_sampling(
        self,
        log_posterior_unnorm: Callable[[np.ndarray], np.ndarray],
        posterior_samples: np.ndarray,
        log_prior: Callable[[np.ndarray], np.ndarray],
        n_iter: int = 50,
    ) -> BayesFactorResult:
        """Bridge-sampling estimator of the marginal likelihood ratio.

        Parameters
        ----------
        log_posterior_unnorm : Callable
            Function returning the un-normalised log posterior density.
        posterior_samples : np.ndarray
            Samples from the posterior.
        log_prior : Callable
            Function returning the log prior density.
        n_iter : int
            Number of iterative bridge-sampling updates.

        Returns
        -------
        BayesFactorResult
            Bayes factor and interpretation.
        """
        posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
        n = len(posterior_samples)

        # Proposal: fitted normal to posterior samples
        prop_mean = float(np.mean(posterior_samples))
        prop_std = float(np.std(posterior_samples, ddof=1))
        if prop_std < 1e-12:
            prop_std = 1.0

        proposal_samples = self._rng.normal(prop_mean, prop_std, size=n)

        log_q_post = stats.norm.logpdf(
            posterior_samples, loc=prop_mean, scale=prop_std
        )
        log_q_prop = stats.norm.logpdf(
            proposal_samples, loc=prop_mean, scale=prop_std
        )

        log_g_post = log_posterior_unnorm(posterior_samples)
        log_g_prop = log_posterior_unnorm(proposal_samples)

        # Iterative bridge-sampling scheme
        log_ml = 0.0
        for _ in range(n_iter):
            log_num = special.logsumexp(
                log_g_prop - log_q_prop
                - np.logaddexp(log_g_prop - log_ml, log_q_prop)
            )
            log_den = special.logsumexp(
                log_q_post
                - np.logaddexp(log_g_post - log_ml, log_q_post)
            )
            log_ml = log_num - log_den

        log_ml_prior = self._log_marginal_likelihood(
            log_prior, proposal_samples, prop_mean, prop_std
        )

        log_bf = log_ml - log_ml_prior
        bf10 = float(np.exp(np.clip(log_bf, -500, 500)))
        bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")

        return BayesFactorResult(
            bf10=bf10,
            bf01=bf01,
            log_bf=float(log_bf),
            interpretation=self.interpret(bf10),
            method="bridge_sampling",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _log_marginal_likelihood(
        log_prior: Callable[[np.ndarray], np.ndarray],
        samples: np.ndarray,
        prop_mean: float,
        prop_std: float,
    ) -> float:
        """Estimate log marginal likelihood under the prior alone.

        Parameters
        ----------
        log_prior : Callable
            Function returning the log prior density.
        samples : np.ndarray
            Samples from the proposal distribution.
        prop_mean : float
            Proposal mean.
        prop_std : float
            Proposal standard deviation.

        Returns
        -------
        float
            Estimated log marginal likelihood.
        """
        log_q = stats.norm.logpdf(samples, loc=prop_mean, scale=prop_std)
        log_p = log_prior(samples)
        return float(
            special.logsumexp(log_p - log_q) - np.log(len(samples))
        )

    # ------------------------------------------------------------------

    @staticmethod
    def interpret(bf10: float) -> str:
        """Jeffreys scale interpretation.

        Parameters
        ----------
        bf10 : float
            Bayes factor in favour of H1.

        Returns
        -------
        str
            Qualitative label.
        """
        abf = abs(bf10)
        if abf < 1.0:
            abf = 1.0 / abf if abf > 0 else float("inf")
            direction = "H0"
        else:
            direction = "H1"

        for label, (lo, hi) in _JEFFREYS_THRESHOLDS.items():
            if lo <= abf < hi:
                return f"{label} evidence for {direction}"
        return f"extreme evidence for {direction}"


# =========================================================================
# HierarchicalBayesianModel
# =========================================================================

class HierarchicalBayesianModel:
    """Simple hierarchical Bayesian model with Gibbs-like sampling.

    Models scores as:

        y_{ij} = mu + alpha_i + beta_j + epsilon_{ij}

    where alpha_i are prompt random effects, beta_j are model
    random effects, and epsilon is residual noise.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples to draw.
    n_warmup : int
        Number of warm-up iterations to discard.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = _DEFAULT_MCMC_SAMPLES,
        n_warmup: int = _DEFAULT_MCMC_WARMUP,
        seed: Optional[int] = None,
    ) -> None:
        self._n_samples = n_samples
        self._n_warmup = n_warmup
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def fit(
        self,
        prompt_ids: np.ndarray,
        model_ids: np.ndarray,
        scores: np.ndarray,
    ) -> HierarchicalModelResult:
        """Fit the hierarchical model via Gibbs sampling.

        Parameters
        ----------
        prompt_ids : np.ndarray
            Integer array of prompt identifiers.
        model_ids : np.ndarray
            Integer array of model identifiers.
        scores : np.ndarray
            Observed scores corresponding to each (prompt, model) pair.

        Returns
        -------
        HierarchicalModelResult
            Posterior samples and convergence diagnostics.
        """
        scores = np.asarray(scores, dtype=np.float64)
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        model_ids = np.asarray(model_ids, dtype=np.int64)
        n = len(scores)

        unique_prompts = np.unique(prompt_ids)
        unique_models = np.unique(model_ids)
        n_prompts = len(unique_prompts)
        n_models = len(unique_models)

        # Initialise parameters
        mu = float(np.mean(scores))
        alpha = np.zeros(n_prompts)
        beta = np.zeros(n_models)
        sigma2 = float(np.var(scores, ddof=1))
        sigma2_alpha = 1.0
        sigma2_beta = 1.0

        total_iter = self._n_samples + self._n_warmup
        mu_samples = np.empty(self._n_samples)
        alpha_samples = np.empty((self._n_samples, n_prompts))
        beta_samples = np.empty((self._n_samples, n_models))

        prompt_map = {pid: i for i, pid in enumerate(unique_prompts)}
        model_map = {mid: i for i, mid in enumerate(unique_models)}
        p_idx = np.array([prompt_map[p] for p in prompt_ids])
        m_idx = np.array([model_map[m] for m in model_ids])

        for it in range(total_iter):
            mu = self._sample_group_mean(
                scores, alpha, beta, p_idx, m_idx, sigma2, n
            )
            alpha = self._sample_random_effects(
                scores, mu, beta, p_idx, m_idx, sigma2,
                sigma2_alpha, n_prompts, p_idx,
            )
            beta = self._sample_random_effects(
                scores, mu, alpha, m_idx, p_idx, sigma2,
                sigma2_beta, n_models, m_idx,
            )
            sigma2 = self._sample_variance(
                scores, mu, alpha, beta, p_idx, m_idx, n
            )
            sigma2_alpha = self._sample_hypervariance(alpha, n_prompts)
            sigma2_beta = self._sample_hypervariance(beta, n_models)

            if it >= self._n_warmup:
                s = it - self._n_warmup
                mu_samples[s] = mu
                alpha_samples[s] = alpha
                beta_samples[s] = beta

        group_effects = {"mu": mu_samples}
        random_effects = {
            "prompt": alpha_samples,
            "model": beta_samples,
        }
        variance_components = {
            "residual": sigma2,
            "prompt": sigma2_alpha,
            "model": sigma2_beta,
        }
        convergence_info = self._compute_convergence(mu_samples)

        return HierarchicalModelResult(
            group_effects=group_effects,
            random_effects=random_effects,
            variance_components=variance_components,
            convergence_info=convergence_info,
        )

    # ------------------------------------------------------------------

    def _sample_group_mean(
        self,
        scores: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        p_idx: np.ndarray,
        m_idx: np.ndarray,
        sigma2: float,
        n: int,
    ) -> float:
        """Sample the overall mean from its full conditional.

        Parameters
        ----------
        scores : np.ndarray
            Observed scores.
        alpha : np.ndarray
            Current prompt random effects.
        beta : np.ndarray
            Current model random effects.
        p_idx : np.ndarray
            Prompt index mapping.
        m_idx : np.ndarray
            Model index mapping.
        sigma2 : float
            Current residual variance.
        n : int
            Number of observations.

        Returns
        -------
        float
            A draw from the full conditional of mu.
        """
        resid = scores - alpha[p_idx] - beta[m_idx]
        post_var = 1.0 / (n / sigma2 + 1e-6)
        post_mean = post_var * (np.sum(resid) / sigma2)
        return float(self._rng.normal(post_mean, np.sqrt(post_var)))

    # ------------------------------------------------------------------

    def _sample_random_effects(
        self,
        scores: np.ndarray,
        mu: float,
        other_effects: np.ndarray,
        own_idx: np.ndarray,
        other_idx: np.ndarray,
        sigma2: float,
        sigma2_re: float,
        n_levels: int,
        level_idx: np.ndarray,
    ) -> np.ndarray:
        """Sample a set of random effects from their full conditional.

        Parameters
        ----------
        scores : np.ndarray
            Observed scores.
        mu : float
            Current grand mean.
        other_effects : np.ndarray
            The other set of random effects.
        own_idx : np.ndarray
            Index mapping observations to own levels.
        other_idx : np.ndarray
            Index mapping observations to the other effects.
        sigma2 : float
            Current residual variance.
        sigma2_re : float
            Current random-effect variance.
        n_levels : int
            Number of levels in this factor.
        level_idx : np.ndarray
            Same as own_idx (used for grouping).

        Returns
        -------
        np.ndarray
            Sampled random effects of length n_levels.
        """
        effects = np.zeros(n_levels)
        for j in range(n_levels):
            mask = level_idx == j
            nj = int(np.sum(mask))
            if nj == 0:
                effects[j] = float(
                    self._rng.normal(0, np.sqrt(sigma2_re))
                )
                continue
            resid_j = scores[mask] - mu - other_effects[other_idx[mask]]
            post_var = 1.0 / (nj / sigma2 + 1.0 / sigma2_re)
            post_mean = post_var * (np.sum(resid_j) / sigma2)
            effects[j] = float(
                self._rng.normal(post_mean, np.sqrt(post_var))
            )
        return effects

    # ------------------------------------------------------------------

    def _sample_variance(
        self,
        scores: np.ndarray,
        mu: float,
        alpha: np.ndarray,
        beta: np.ndarray,
        p_idx: np.ndarray,
        m_idx: np.ndarray,
        n: int,
    ) -> float:
        """Sample residual variance from its Inverse-Gamma full conditional.

        Parameters
        ----------
        scores : np.ndarray
            Observed scores.
        mu : float
            Current grand mean.
        alpha : np.ndarray
            Current prompt random effects.
        beta : np.ndarray
            Current model random effects.
        p_idx : np.ndarray
            Prompt index mapping.
        m_idx : np.ndarray
            Model index mapping.
        n : int
            Number of observations.

        Returns
        -------
        float
            A draw from the Inverse-Gamma full conditional.
        """
        resid = scores - mu - alpha[p_idx] - beta[m_idx]
        shape = n / 2.0
        scale = np.sum(resid ** 2) / 2.0
        return float(
            1.0 / self._rng.gamma(shape, 1.0 / max(scale, 1e-12))
        )

    # ------------------------------------------------------------------

    def _sample_hypervariance(
        self,
        effects: np.ndarray,
        n_levels: int,
    ) -> float:
        """Sample the hyper-variance for a random-effect group.

        Parameters
        ----------
        effects : np.ndarray
            Current random-effect values.
        n_levels : int
            Number of levels.

        Returns
        -------
        float
            A draw from the Inverse-Gamma full conditional.
        """
        shape = n_levels / 2.0
        scale = np.sum(effects ** 2) / 2.0
        return float(
            1.0 / self._rng.gamma(shape, 1.0 / max(scale, 1e-12))
        )

    # ------------------------------------------------------------------

    def cross_classified_fit(
        self,
        prompt_ids: np.ndarray,
        model_ids: np.ndarray,
        scores: np.ndarray,
    ) -> HierarchicalModelResult:
        """Fit a cross-classified model (prompts x models interaction).

        This is a convenience wrapper that adds an interaction term to the
        standard hierarchical model.

        Parameters
        ----------
        prompt_ids : np.ndarray
            Integer array of prompt identifiers.
        model_ids : np.ndarray
            Integer array of model identifiers.
        scores : np.ndarray
            Observed scores.

        Returns
        -------
        HierarchicalModelResult
            Result including interaction variance component.
        """
        base_result = self.fit(prompt_ids, model_ids, scores)

        # Estimate interaction variance from residuals
        mu_hat = float(np.mean(base_result.group_effects["mu"]))
        alpha_hat = np.mean(base_result.random_effects["prompt"], axis=0)
        beta_hat = np.mean(base_result.random_effects["model"], axis=0)

        scores = np.asarray(scores, dtype=np.float64)
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        model_ids = np.asarray(model_ids, dtype=np.int64)

        unique_prompts = np.unique(prompt_ids)
        unique_models = np.unique(model_ids)
        p_map = {pid: i for i, pid in enumerate(unique_prompts)}
        m_map = {mid: i for i, mid in enumerate(unique_models)}
        p_idx = np.array([p_map[p] for p in prompt_ids])
        m_idx = np.array([m_map[m] for m in model_ids])

        fitted = mu_hat + alpha_hat[p_idx] + beta_hat[m_idx]
        interaction_var = float(np.var(scores - fitted, ddof=1))

        base_result.variance_components["interaction"] = interaction_var
        return base_result

    # ------------------------------------------------------------------

    def summarize(
        self,
        result: HierarchicalModelResult,
    ) -> Dict[str, Any]:
        """Variance decomposition summary.

        Parameters
        ----------
        result : HierarchicalModelResult
            A fitted hierarchical model result.

        Returns
        -------
        Dict[str, Any]
            Proportion of variance attributed to each component.
        """
        total = sum(result.variance_components.values())
        proportions = {
            k: v / total if total > 0 else 0.0
            for k, v in result.variance_components.items()
        }
        return {
            "variance_components": result.variance_components,
            "variance_proportions": proportions,
            "convergence": result.convergence_info,
        }

    # ------------------------------------------------------------------

    def _compute_convergence(
        self,
        samples: np.ndarray,
    ) -> Dict[str, Any]:
        """Basic convergence diagnostics on a single chain.

        Parameters
        ----------
        samples : np.ndarray
            Posterior samples from the chain.

        Returns
        -------
        Dict[str, Any]
            ESS and simple split-chain R-hat proxy.
        """
        n = len(samples)
        mid = n // 2
        chain_a, chain_b = samples[:mid], samples[mid:]
        mean_a = np.mean(chain_a)
        mean_b = np.mean(chain_b)
        var_a = np.var(chain_a, ddof=1)
        var_b = np.var(chain_b, ddof=1)
        w = (var_a + var_b) / 2.0
        b = mid * ((mean_a - mean_b) ** 2)
        var_hat = (1 - 1.0 / mid) * w + b / mid
        r_hat = float(np.sqrt(var_hat / w)) if w > 0 else float("nan")

        # Simple ESS via first-order autocorrelation
        lag1 = (
            np.corrcoef(samples[:-1], samples[1:])[0, 1]
            if n > 2
            else 0.0
        )
        denom = 1 + 2 * max(lag1, 0)
        ess = n / denom if denom > 0 else float(n)

        return {
            "r_hat": r_hat,
            "ess": float(ess),
            "converged": r_hat < 1.1,
        }


# =========================================================================
# SequentialBayesianUpdater
# =========================================================================

class SequentialBayesianUpdater:
    """Normal-Normal sequential Bayesian updater.

    Parameters
    ----------
    prior_mean : float
        Prior mean.
    prior_var : float
        Prior variance.
    likelihood_var : float
        Known (or estimated) observation variance.
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        likelihood_var: float = 1.0,
    ) -> None:
        self._mean = prior_mean
        self._var = prior_var
        self._lik_var = likelihood_var
        self._history: List[Dict[str, float]] = [
            {"mean": prior_mean, "var": prior_var, "n": 0}
        ]
        self._n = 0

    # ------------------------------------------------------------------

    def update(self, observation: float) -> Dict[str, float]:
        """Incorporate a single observation.

        Parameters
        ----------
        observation : float
            New observed value.

        Returns
        -------
        Dict[str, float]
            Updated posterior mean and variance.
        """
        prec_prior = 1.0 / self._var
        prec_lik = 1.0 / self._lik_var
        new_prec = prec_prior + prec_lik
        new_mean = (
            prec_prior * self._mean + prec_lik * observation
        ) / new_prec
        new_var = 1.0 / new_prec

        self._mean = new_mean
        self._var = new_var
        self._n += 1

        record = {"mean": new_mean, "var": new_var, "n": self._n}
        self._history.append(record)
        return record

    # ------------------------------------------------------------------

    def update_batch(
        self, observations: np.ndarray
    ) -> Dict[str, float]:
        """Incorporate a batch of observations.

        Parameters
        ----------
        observations : np.ndarray
            Array of new observed values.

        Returns
        -------
        Dict[str, float]
            Updated posterior after all observations.
        """
        observations = np.asarray(observations, dtype=np.float64)
        result: Dict[str, float] = {
            "mean": self._mean,
            "var": self._var,
            "n": float(self._n),
        }
        for obs in observations:
            result = self.update(float(obs))
        return result

    # ------------------------------------------------------------------

    def posterior_history(self) -> List[Dict[str, float]]:
        """Return the full evolution of the posterior.

        Returns
        -------
        List[Dict[str, float]]
            List of posterior snapshots.
        """
        return list(self._history)

    # ------------------------------------------------------------------

    def stopping_rule(
        self,
        precision_threshold: float = 0.01,
        min_observations: int = 10,
    ) -> bool:
        """Check whether the posterior is precise enough to stop.

        Parameters
        ----------
        precision_threshold : float
            Maximum acceptable posterior standard deviation.
        min_observations : int
            Minimum number of observations before stopping is allowed.

        Returns
        -------
        bool
            True if the stopping criterion is satisfied.
        """
        if self._n < min_observations:
            return False
        return np.sqrt(self._var) <= precision_threshold


# =========================================================================
# BayesianModelComparison
# =========================================================================

class BayesianModelComparison:
    """Model comparison via WAIC and LOO-CV.

    Parameters
    ----------
    seed : Optional[int]
        Random seed.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def waic(
        self,
        log_likelihood: np.ndarray,
    ) -> WAICResult:
        """Compute the Widely Applicable Information Criterion.

        Parameters
        ----------
        log_likelihood : np.ndarray
            Array of shape (n_samples, n_obs) containing pointwise
            log-likelihood evaluations for each posterior sample.

        Returns
        -------
        WAICResult
            WAIC value, effective parameters, SE, and pointwise values.
        """
        log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
        n_samples, n_obs = log_likelihood.shape

        # lppd: log pointwise predictive density
        lppd_i = (
            special.logsumexp(log_likelihood, axis=0) - np.log(n_samples)
        )

        # p_waic: effective number of parameters
        p_waic_i = np.var(log_likelihood, axis=0, ddof=1)

        pointwise = -2.0 * (lppd_i - p_waic_i)
        waic_val = float(np.sum(pointwise))
        p_waic_val = float(np.sum(p_waic_i))
        se_val = float(np.sqrt(n_obs * np.var(pointwise, ddof=1)))

        return WAICResult(
            waic=waic_val,
            p_waic=p_waic_val,
            se=se_val,
            pointwise=pointwise,
        )

    # ------------------------------------------------------------------

    def loo_cv(
        self,
        log_likelihood: np.ndarray,
    ) -> WAICResult:
        """Pareto-smoothed importance-sampling LOO cross-validation.

        Parameters
        ----------
        log_likelihood : np.ndarray
            Array of shape (n_samples, n_obs).

        Returns
        -------
        WAICResult
            LOO estimate repackaged as a WAICResult for convenience.
        """
        log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
        n_samples, n_obs = log_likelihood.shape

        loo_i = np.empty(n_obs)
        for i in range(n_obs):
            log_ratios = -log_likelihood[:, i]
            log_ratios -= np.max(log_ratios)
            weights = np.exp(log_ratios)
            weights = self._pareto_smooth_weights(weights)
            weights /= np.sum(weights)
            loo_i[i] = float(
                np.log(np.sum(weights * np.exp(log_likelihood[:, i])))
            )

        pointwise = -2.0 * loo_i
        loo_val = float(np.sum(pointwise))
        p_loo = float(
            np.sum(
                special.logsumexp(log_likelihood, axis=0)
                - np.log(n_samples)
            )
            - (-loo_val / 2.0)
        )
        se_val = float(np.sqrt(n_obs * np.var(pointwise, ddof=1)))

        return WAICResult(
            waic=loo_val,
            p_waic=p_loo,
            se=se_val,
            pointwise=pointwise,
        )

    # ------------------------------------------------------------------

    def compare(
        self,
        models: Dict[str, np.ndarray],
    ) -> List[Tuple[str, WAICResult]]:
        """Compare multiple models via WAIC and return a ranking.

        Parameters
        ----------
        models : Dict[str, np.ndarray]
            Mapping from model name to its log-likelihood array of shape
            (n_samples, n_obs).

        Returns
        -------
        List[Tuple[str, WAICResult]]
            Models sorted by WAIC (best first).
        """
        results: List[Tuple[str, WAICResult]] = []
        for name, ll in models.items():
            results.append((name, self.waic(ll)))
        results.sort(key=lambda x: x[1].waic)
        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _pareto_smooth_weights(weights: np.ndarray) -> np.ndarray:
        """Pareto-smooth the largest importance weights.

        Parameters
        ----------
        weights : np.ndarray
            Raw (non-negative) importance weights.

        Returns
        -------
        np.ndarray
            Smoothed importance weights.
        """
        n = len(weights)
        m = max(int(min(n / 5, 3 * np.sqrt(n))), 5)
        sorted_idx = np.argsort(weights)
        cutoff = weights[sorted_idx[-m]]

        tail = weights[sorted_idx[-m:]]
        tail = tail - cutoff
        tail = np.maximum(tail, 1e-12)

        if np.std(tail) > 1e-12 and len(tail) > 2:
            # Fit Generalized Pareto to the tail
            mean_t = float(np.mean(tail))
            var_t = float(np.var(tail, ddof=1))
            xi = (
                0.5 * (mean_t ** 2 / var_t - 1)
                if var_t > 0
                else 0.0
            )
            xi = min(max(xi, -0.5), 1.0)
            sigma = mean_t * (1 + xi) if (1 + xi) > 0 else mean_t

            # Replace tail with quantiles of the fitted GPD
            probs = (np.arange(1, m + 1) - 0.5) / m
            if abs(xi) < 1e-8:
                smoothed_tail = -sigma * np.log(1 - probs)
            else:
                smoothed_tail = (
                    sigma / xi * ((1 - probs) ** (-xi) - 1)
                )
            weights[sorted_idx[-m:]] = smoothed_tail + cutoff

        return weights


# =========================================================================
# PriorSensitivityAnalyzer
# =========================================================================

class PriorSensitivityAnalyzer:
    """Analyse posterior sensitivity to the choice of prior.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples per prior specification.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def analyze(
        self,
        data: np.ndarray,
        prior_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> PriorSensitivityResult:
        """Run the posterior under multiple normal priors.

        Parameters
        ----------
        data : np.ndarray
            Observed data.
        prior_specs : Optional[List[Dict[str, Any]]]
            Each dict must have 'mean' and 'std'.  If None,
            a default grid is generated.

        Returns
        -------
        PriorSensitivityResult
            Summary across all prior specifications.
        """
        data = np.asarray(data, dtype=np.float64)
        if prior_specs is None:
            prior_specs = self._generate_prior_grid(data)

        data_mean = float(np.mean(data))
        data_var = (
            float(np.var(data, ddof=1)) if len(data) > 1 else 1.0
        )
        n = len(data)

        posteriors: List[Dict[str, float]] = []
        for spec in prior_specs:
            pm = spec["mean"]
            pv = spec["std"] ** 2
            prior_prec = 1.0 / pv
            lik_prec = n / data_var
            post_prec = prior_prec + lik_prec
            post_mean = (
                prior_prec * pm + lik_prec * data_mean
            ) / post_prec
            post_var = 1.0 / post_prec

            posteriors.append({
                "mean": post_mean,
                "std": np.sqrt(post_var),
                "prior_mean": pm,
                "prior_std": spec["std"],
            })

        robustness = self.robustness_index(posteriors)
        return PriorSensitivityResult(
            prior_specs=prior_specs,
            posteriors=posteriors,
            robustness_index=robustness,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def robustness_index(
        posteriors: List[Dict[str, float]],
    ) -> float:
        """Maximum absolute deviation in posterior means.

        Parameters
        ----------
        posteriors : List[Dict[str, float]]
            Posterior summaries from analyze().

        Returns
        -------
        float
            Max absolute deviation across posterior means.
        """
        means = np.array([p["mean"] for p in posteriors])
        return float(np.max(means) - np.min(means))

    # ------------------------------------------------------------------

    @staticmethod
    def _generate_prior_grid(
        data: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Generate a default grid of prior specifications.

        Parameters
        ----------
        data : np.ndarray
            Observed data (used to calibrate the grid).

        Returns
        -------
        List[Dict[str, Any]]
            List of prior specifications.
        """
        data_std = (
            float(np.std(data, ddof=1)) if len(data) > 1 else 1.0
        )
        data_mean = float(np.mean(data))
        specs: List[Dict[str, Any]] = []
        for mean_shift in [-1.0, 0.0, 1.0]:
            for std_mult in [0.5, 1.0, 2.0, 5.0]:
                specs.append({
                    "mean": data_mean + mean_shift * data_std,
                    "std": data_std * std_mult,
                })
        return specs


# =========================================================================
# MCMCDiagnostics
# =========================================================================

class MCMCDiagnostics:
    """Diagnostics for Markov-chain Monte Carlo output.

    All methods are stateless and operate on arrays of posterior samples.
    """

    # ------------------------------------------------------------------

    @staticmethod
    def gelman_rubin(chains: List[np.ndarray]) -> float:
        """Compute the Gelman-Rubin R-hat statistic.

        Parameters
        ----------
        chains : List[np.ndarray]
            List of 1-D arrays, one per chain.

        Returns
        -------
        float
            R-hat value.  Values close to 1.0 indicate convergence.
        """
        m = len(chains)
        if m < 2:
            return float("nan")
        n = min(len(c) for c in chains)
        chain_means = np.array([np.mean(c[:n]) for c in chains])
        chain_vars = np.array([np.var(c[:n], ddof=1) for c in chains])
        grand_mean = np.mean(chain_means)

        b = n * np.var(chain_means, ddof=1)
        w = np.mean(chain_vars)
        if w == 0:
            return float("nan")
        var_hat = (1 - 1.0 / n) * w + b / n
        return float(np.sqrt(var_hat / w))

    # ------------------------------------------------------------------

    @staticmethod
    def effective_sample_size(samples: np.ndarray) -> float:
        """Estimate the effective sample size via autocorrelation.

        Parameters
        ----------
        samples : np.ndarray
            1-D posterior samples.

        Returns
        -------
        float
            Estimated effective sample size.
        """
        samples = np.asarray(samples, dtype=np.float64)
        n = len(samples)
        if n < 4:
            return float(n)

        acov = MCMCDiagnostics._autocovariance(samples)
        var0 = acov[0]
        if var0 == 0:
            return float(n)

        # Truncate at first negative pair
        max_lag = n // 2
        rho = acov[:max_lag] / var0
        running_sum = 0.0
        for t in range(1, max_lag):
            if rho[t] < 0:
                break
            running_sum += rho[t]

        ess = n / (1.0 + 2.0 * running_sum)
        return max(1.0, float(ess))

    # ------------------------------------------------------------------

    @staticmethod
    def split_rhat(samples: np.ndarray) -> float:
        """Compute the split R-hat diagnostic on a single chain.

        The chain is split in half and treated as two independent chains.

        Parameters
        ----------
        samples : np.ndarray
            1-D posterior samples.

        Returns
        -------
        float
            Split R-hat value.
        """
        samples = np.asarray(samples, dtype=np.float64)
        n = len(samples)
        if n < 4:
            return float("nan")
        mid = n // 2
        return MCMCDiagnostics.gelman_rubin(
            [samples[:mid], samples[mid:2 * mid]]
        )

    # ------------------------------------------------------------------

    @staticmethod
    def autocorrelation(
        samples: np.ndarray,
        max_lag: Optional[int] = None,
    ) -> np.ndarray:
        """Per-lag autocorrelation of a posterior chain.

        Parameters
        ----------
        samples : np.ndarray
            1-D posterior samples.
        max_lag : Optional[int]
            Maximum lag to compute.  Defaults to half the chain length.

        Returns
        -------
        np.ndarray
            Autocorrelation at each lag from 0 to max_lag.
        """
        samples = np.asarray(samples, dtype=np.float64)
        n = len(samples)
        if max_lag is None:
            max_lag = n // 2
        max_lag = min(max_lag, n - 1)

        acov = MCMCDiagnostics._autocovariance(samples)
        var0 = acov[0]
        if var0 == 0:
            return np.ones(max_lag + 1)
        return acov[: max_lag + 1] / var0

    # ------------------------------------------------------------------

    @staticmethod
    def trace_summary(
        samples: np.ndarray,
        chain_label: str = "chain_0",
    ) -> Dict[str, Any]:
        """Comprehensive chain diagnostics.

        Parameters
        ----------
        samples : np.ndarray
            1-D posterior samples.
        chain_label : str
            Label for the chain.

        Returns
        -------
        Dict[str, Any]
            Summary containing mean, std, ESS, split R-hat, and
            first-five autocorrelations.
        """
        samples = np.asarray(samples, dtype=np.float64)
        acorr = MCMCDiagnostics.autocorrelation(samples, max_lag=5)

        return {
            "chain": chain_label,
            "n_samples": len(samples),
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "ess": MCMCDiagnostics.effective_sample_size(samples),
            "split_rhat": MCMCDiagnostics.split_rhat(samples),
            "autocorrelation_lag1": (
                float(acorr[1]) if len(acorr) > 1 else float("nan")
            ),
            "autocorrelation_lag5": (
                float(acorr[5]) if len(acorr) > 5 else float("nan")
            ),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _autocovariance(samples: np.ndarray) -> np.ndarray:
        """Compute the autocovariance function via FFT.

        Parameters
        ----------
        samples : np.ndarray
            1-D array of posterior samples.

        Returns
        -------
        np.ndarray
            Autocovariance at each lag.
        """
        n = len(samples)
        centred = samples - np.mean(samples)
        # Zero-pad to next power of 2 for efficient FFT
        fft_len = 1
        while fft_len < 2 * n:
            fft_len *= 2
        fft_vals = np.fft.rfft(centred, n=fft_len)
        acov_full = np.fft.irfft(
            fft_vals * np.conj(fft_vals), n=fft_len
        )
        return acov_full[:n] / n
