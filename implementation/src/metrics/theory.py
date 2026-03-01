"""
Metric theory for the Diversity Decoding Arena.

Proves and verifies theoretical relationships between diversity metrics,
provides information-theoretic bounds, monotonicity verification,
dimension reduction analysis, analytical calculations for known distributions,
metric axiom checking, and convergence rate estimation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special, stats
from scipy.spatial import distance as sp_distance


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricTheoreticalBound:
    """Stores a theoretical bound on a metric value."""

    metric_name: str
    bound_type: str  # "lower" or "upper"
    bound_value: float
    achieved_by: str  # description of the distribution that achieves equality
    assumptions: List[str] = field(default_factory=list)
    proof_sketch: str = ""
    tightness: Optional[float] = None  # gap between bound and best known value

    def is_tight(self, tolerance: float = 1e-9) -> bool:
        if self.tightness is None:
            return False
        return self.tightness <= tolerance


@dataclass
class MonotonicityResult:
    """Result of a monotonicity verification."""

    metric_name: str
    parameter_name: str
    is_monotone: bool
    direction: str  # "increasing", "decreasing", or "non-monotone"
    violations: List[Tuple[float, float, float, float]] = field(default_factory=list)
    parameter_range: Tuple[float, float] = (0.0, 1.0)
    confidence: float = 1.0

    @property
    def violation_fraction(self) -> float:
        if not self.violations:
            return 0.0
        total_pairs = max(1, len(self.violations))
        return len([v for v in self.violations if abs(v[2] - v[3]) > 1e-12]) / total_pairs


@dataclass
class DimensionReductionEffect:
    """Captures how dimension reduction affects a metric."""

    metric_name: str
    original_dim: int
    reduced_dim: int
    method: str  # "pca", "random_projection", "umap"
    distortion_mean: float
    distortion_std: float
    max_distortion: float
    jl_bound: Optional[float] = None  # Johnson-Lindenstrauss bound if applicable
    variance_retained: Optional[float] = None


@dataclass
class AnalyticalExpectation:
    """Expected metric value under a known distribution."""

    metric_name: str
    distribution_name: str
    distribution_params: Dict[str, float]
    expected_value: float
    variance: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    formula: str = ""


@dataclass
class AxiomCheckResult:
    """Result of checking a metric axiom."""

    axiom_name: str
    metric_name: str
    satisfied: bool
    max_violation: float = 0.0
    violation_examples: List[Dict[str, Any]] = field(default_factory=list)
    num_tests: int = 0
    notes: str = ""


@dataclass
class ConvergenceEstimate:
    """Estimated convergence rate for a metric."""

    metric_name: str
    rate_exponent: float  # convergence as O(n^{-rate_exponent})
    rate_constant: float
    sample_sizes: List[int] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    fit_r_squared: float = 0.0
    asymptotic_bias: float = 0.0


@dataclass
class CalibrationPoint:
    """Single point on a calibration curve."""

    diversity_level: float  # 0 = no diversity, 1 = maximum diversity
    expected_metric_value: float
    std_metric_value: float
    sample_size: int
    distribution_type: str


class DistributionType(Enum):
    UNIFORM = auto()
    ZIPF = auto()
    GEOMETRIC = auto()
    DIRICHLET = auto()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_log(x: float, base: float = math.e) -> float:
    """Logarithm that returns 0 for x <= 0."""
    if x <= 0:
        return 0.0
    return math.log(x) / math.log(base)


def _entropy(probs: np.ndarray, base: float = math.e) -> float:
    """Shannon entropy of a discrete probability distribution."""
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs) / np.log(base)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(p || q) with safe handling of zeros."""
    mask = p > 0
    if not np.all(q[mask] > 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _renyi_entropy(probs: np.ndarray, alpha: float) -> float:
    """Rényi entropy of order alpha."""
    probs = probs[probs > 0]
    if abs(alpha - 1.0) < 1e-12:
        return _entropy(probs)
    if alpha == float("inf"):
        return -math.log(float(np.max(probs)))
    return float(np.log(np.sum(probs ** alpha)) / (1.0 - alpha))


def _pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix."""
    return sp_distance.squareform(sp_distance.pdist(X, metric=metric))


def _zipf_pmf(k: int, s: float, N: int) -> float:
    """Zipf probability mass function: P(k) = (1/k^s) / H_{N,s}."""
    harmonic = sum(1.0 / (i ** s) for i in range(1, N + 1))
    return (1.0 / (k ** s)) / harmonic


def _zipf_probabilities(s: float, N: int) -> np.ndarray:
    """Full probability vector for Zipf(s, N)."""
    ranks = np.arange(1, N + 1, dtype=np.float64)
    unnormalized = 1.0 / (ranks ** s)
    return unnormalized / unnormalized.sum()


def _geometric_probabilities(p: float, N: int) -> np.ndarray:
    """Truncated geometric distribution over {1, ..., N}."""
    ranks = np.arange(N, dtype=np.float64)
    unnormalized = (1.0 - p) ** ranks * p
    return unnormalized / unnormalized.sum()


def _dirichlet_entropy_expectation(alpha: np.ndarray) -> float:
    """Expected entropy of a sample from Dirichlet(alpha)."""
    alpha0 = alpha.sum()
    K = len(alpha)
    # E[H] = psi(alpha0 + 1) - sum_k (alpha_k / alpha0) * psi(alpha_k + 1)
    return float(
        special.digamma(alpha0 + 1)
        - np.sum((alpha / alpha0) * special.digamma(alpha + 1))
    )


def _jl_min_dimension(n_samples: int, epsilon: float) -> int:
    """Johnson-Lindenstrauss minimum dimension for (1+/-eps) distortion."""
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be in (0, 1)")
    return int(math.ceil(8 * math.log(n_samples) / (epsilon ** 2)))


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = C * x^alpha via log-log regression. Returns (alpha, C, r_squared)."""
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    if len(lx) < 2:
        return 0.0, 0.0, 0.0
    slope, intercept, r_value, _, _ = stats.linregress(lx, ly)
    return float(slope), float(np.exp(intercept)), float(r_value ** 2)


# ---------------------------------------------------------------------------
# InformationTheoreticAnalyzer
# ---------------------------------------------------------------------------

class InformationTheoreticAnalyzer:
    """Derives information-theoretic bounds on diversity metrics."""

    def __init__(self, vocab_size: int = 50_000, base: float = 2.0):
        self.vocab_size = vocab_size
        self.base = base

    # -- Entropy bounds -------------------------------------------------

    def entropy_lower_bound(self) -> MetricTheoreticalBound:
        """H(X) >= 0 with equality iff X is deterministic."""
        return MetricTheoreticalBound(
            metric_name="shannon_entropy",
            bound_type="lower",
            bound_value=0.0,
            achieved_by="deterministic (point mass) distribution",
            assumptions=["discrete distribution"],
            proof_sketch="H = -sum p log p >= 0 since p log p <= 0 for p in [0,1]",
            tightness=0.0,
        )

    def entropy_upper_bound(self, support_size: Optional[int] = None) -> MetricTheoreticalBound:
        """H(X) <= log(|support|) with equality iff X is uniform."""
        n = support_size or self.vocab_size
        return MetricTheoreticalBound(
            metric_name="shannon_entropy",
            bound_type="upper",
            bound_value=_safe_log(n, self.base),
            achieved_by=f"uniform distribution over {n} symbols",
            assumptions=["discrete distribution", f"support size = {n}"],
            proof_sketch="By Gibbs' inequality, KL(p||u) >= 0 => H(p) <= log(n)",
            tightness=0.0,
        )

    def conditional_entropy_bound(
        self, joint_probs: np.ndarray
    ) -> MetricTheoreticalBound:
        """H(Y|X) <= H(Y) with equality iff X and Y are independent."""
        # joint_probs: shape (|X|, |Y|)
        p_y = joint_probs.sum(axis=0)
        h_y = _entropy(p_y, self.base)
        p_x = joint_probs.sum(axis=1)
        h_y_given_x = 0.0
        for i in range(joint_probs.shape[0]):
            if p_x[i] > 0:
                row = joint_probs[i] / p_x[i]
                h_y_given_x += p_x[i] * _entropy(row, self.base)
        return MetricTheoreticalBound(
            metric_name="conditional_entropy",
            bound_type="upper",
            bound_value=h_y,
            achieved_by="independent X, Y",
            assumptions=["joint distribution provided"],
            proof_sketch="H(Y|X) = H(X,Y) - H(X) <= H(Y) by non-negativity of MI",
            tightness=h_y - h_y_given_x,
        )

    # -- Mutual information bounds --------------------------------------

    def mutual_information_bounds(
        self, joint_probs: np.ndarray
    ) -> Tuple[MetricTheoreticalBound, MetricTheoreticalBound]:
        """0 <= I(X;Y) <= min(H(X), H(Y))."""
        p_x = joint_probs.sum(axis=1)
        p_y = joint_probs.sum(axis=0)
        h_x = _entropy(p_x, self.base)
        h_y = _entropy(p_y, self.base)
        mi = h_x + h_y - _entropy(joint_probs.ravel(), self.base)
        lower = MetricTheoreticalBound(
            metric_name="mutual_information",
            bound_type="lower",
            bound_value=0.0,
            achieved_by="independent X, Y",
            assumptions=["joint distribution"],
            proof_sketch="I(X;Y) = KL(p_{XY} || p_X p_Y) >= 0",
            tightness=mi,
        )
        upper_val = min(h_x, h_y)
        upper = MetricTheoreticalBound(
            metric_name="mutual_information",
            bound_type="upper",
            bound_value=upper_val,
            achieved_by="X determines Y (or vice versa)",
            assumptions=["joint distribution"],
            proof_sketch="I(X;Y) = H(X) - H(X|Y) <= H(X); likewise <= H(Y)",
            tightness=upper_val - mi,
        )
        return lower, upper

    # -- KL divergence bounds -------------------------------------------

    def kl_divergence_pinsker_bound(
        self, p: np.ndarray, q: np.ndarray
    ) -> MetricTheoreticalBound:
        """Pinsker's inequality: TV(p,q)^2 <= (1/2) KL(p||q)."""
        kl = _kl_divergence(p, q)
        tv = 0.5 * float(np.sum(np.abs(p - q)))
        pinsker_lower = 2.0 * tv * tv
        return MetricTheoreticalBound(
            metric_name="kl_divergence",
            bound_type="lower",
            bound_value=pinsker_lower,
            achieved_by="Bernoulli distributions (asymptotically tight)",
            assumptions=["p, q are probability distributions"],
            proof_sketch="Pinsker: TV(p,q) <= sqrt(KL/2), so KL >= 2 TV^2",
            tightness=kl - pinsker_lower if kl != float("inf") else None,
        )

    def kl_divergence_upper_by_chi_squared(
        self, p: np.ndarray, q: np.ndarray
    ) -> MetricTheoreticalBound:
        """KL(p||q) <= chi^2(p||q) = sum (p_i - q_i)^2 / q_i."""
        mask = q > 0
        if not np.all(mask[p > 0]):
            chi2 = float("inf")
        else:
            chi2 = float(np.sum((p[mask] - q[mask]) ** 2 / q[mask]))
        kl = _kl_divergence(p, q)
        return MetricTheoreticalBound(
            metric_name="kl_divergence",
            bound_type="upper",
            bound_value=chi2,
            achieved_by="equality when p = q (both zero)",
            assumptions=["q_i > 0 wherever p_i > 0"],
            proof_sketch="log(x) <= x - 1 implies KL <= chi^2",
            tightness=chi2 - kl if kl != float("inf") else None,
        )

    # -- Rényi entropy bounds -------------------------------------------

    def renyi_monotonicity_bound(
        self, probs: np.ndarray, alpha1: float, alpha2: float
    ) -> MetricTheoreticalBound:
        """For alpha1 < alpha2: H_{alpha1}(X) >= H_{alpha2}(X)."""
        if alpha1 >= alpha2:
            raise ValueError("alpha1 must be less than alpha2")
        h1 = _renyi_entropy(probs, alpha1)
        h2 = _renyi_entropy(probs, alpha2)
        return MetricTheoreticalBound(
            metric_name=f"renyi_entropy(alpha={alpha2})",
            bound_type="upper",
            bound_value=h1,
            achieved_by=f"Rényi entropy at alpha={alpha1}",
            assumptions=[f"alpha1={alpha1} < alpha2={alpha2}"],
            proof_sketch="Rényi entropy is non-increasing in alpha",
            tightness=h1 - h2,
        )


# ---------------------------------------------------------------------------
# MonotonicityVerifier
# ---------------------------------------------------------------------------

class MonotonicityVerifier:
    """Checks monotonicity of metric families under parameter changes."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def verify_parametric_monotonicity(
        self,
        metric_fn: Callable[[np.ndarray, float], float],
        data: np.ndarray,
        param_values: np.ndarray,
        metric_name: str = "metric",
        param_name: str = "parameter",
        expected_direction: str = "increasing",
    ) -> MonotonicityResult:
        """Check if metric_fn(data, param) is monotone in param."""
        values = np.array([metric_fn(data, p) for p in param_values])
        violations: List[Tuple[float, float, float, float]] = []

        for i in range(len(param_values) - 1):
            p1, p2 = param_values[i], param_values[i + 1]
            v1, v2 = values[i], values[i + 1]
            if expected_direction == "increasing" and v2 < v1 - self.tolerance:
                violations.append((float(p1), float(p2), float(v1), float(v2)))
            elif expected_direction == "decreasing" and v2 > v1 + self.tolerance:
                violations.append((float(p1), float(p2), float(v1), float(v2)))

        is_monotone = len(violations) == 0
        direction = expected_direction if is_monotone else "non-monotone"
        return MonotonicityResult(
            metric_name=metric_name,
            parameter_name=param_name,
            is_monotone=is_monotone,
            direction=direction,
            violations=violations,
            parameter_range=(float(param_values[0]), float(param_values[-1])),
            confidence=1.0 - len(violations) / max(1, len(param_values) - 1),
        )

    def verify_renyi_entropy_monotonicity(
        self, probs: np.ndarray, alpha_range: Tuple[float, float] = (0.1, 10.0), n_points: int = 200
    ) -> MonotonicityResult:
        """Rényi entropy H_alpha should be non-increasing in alpha."""
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
        # skip alpha=1 neighbourhood to avoid numerical issues
        alphas = alphas[np.abs(alphas - 1.0) > 0.02]

        def metric_fn(data: np.ndarray, alpha: float) -> float:
            return _renyi_entropy(data, alpha)

        return self.verify_parametric_monotonicity(
            metric_fn, probs, alphas,
            metric_name="renyi_entropy",
            param_name="alpha",
            expected_direction="decreasing",
        )

    def verify_sample_size_monotonicity(
        self,
        metric_fn: Callable[[np.ndarray], float],
        full_data: np.ndarray,
        sizes: Optional[List[int]] = None,
        n_trials: int = 30,
        metric_name: str = "metric",
    ) -> MonotonicityResult:
        """Check if average metric value is non-decreasing with sample size."""
        if sizes is None:
            n = len(full_data)
            sizes = sorted(set(int(s) for s in np.linspace(10, n, min(20, n // 5)) if s >= 2))

        avg_values = []
        for s in sizes:
            trial_vals = []
            for _ in range(n_trials):
                idx = np.random.choice(len(full_data), size=min(s, len(full_data)), replace=False)
                trial_vals.append(metric_fn(full_data[idx]))
            avg_values.append(np.mean(trial_vals))

        violations: List[Tuple[float, float, float, float]] = []
        for i in range(len(sizes) - 1):
            if avg_values[i + 1] < avg_values[i] - self.tolerance:
                violations.append(
                    (float(sizes[i]), float(sizes[i + 1]),
                     float(avg_values[i]), float(avg_values[i + 1]))
                )

        is_monotone = len(violations) == 0
        return MonotonicityResult(
            metric_name=metric_name,
            parameter_name="sample_size",
            is_monotone=is_monotone,
            direction="increasing" if is_monotone else "non-monotone",
            violations=violations,
            parameter_range=(float(sizes[0]), float(sizes[-1])),
            confidence=1.0 - len(violations) / max(1, len(sizes) - 1),
        )

    def verify_transform_monotonicity(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        transform_fn: Callable[[np.ndarray, float], np.ndarray],
        intensities: np.ndarray,
        metric_name: str = "metric",
        expected_direction: str = "decreasing",
    ) -> MonotonicityResult:
        """Check monotonicity under a family of transforms parameterised by intensity."""
        values = []
        for intensity in intensities:
            transformed = transform_fn(data, intensity)
            values.append(metric_fn(transformed))

        violations: List[Tuple[float, float, float, float]] = []
        for i in range(len(intensities) - 1):
            v1, v2 = values[i], values[i + 1]
            if expected_direction == "increasing" and v2 < v1 - self.tolerance:
                violations.append((float(intensities[i]), float(intensities[i + 1]),
                                   float(v1), float(v2)))
            elif expected_direction == "decreasing" and v2 > v1 + self.tolerance:
                violations.append((float(intensities[i]), float(intensities[i + 1]),
                                   float(v1), float(v2)))

        is_monotone = len(violations) == 0
        return MonotonicityResult(
            metric_name=metric_name,
            parameter_name="transform_intensity",
            is_monotone=is_monotone,
            direction=expected_direction if is_monotone else "non-monotone",
            violations=violations,
            parameter_range=(float(intensities[0]), float(intensities[-1])),
            confidence=1.0 - len(violations) / max(1, len(intensities) - 1),
        )


# ---------------------------------------------------------------------------
# DimensionReductionAnalyzer
# ---------------------------------------------------------------------------

class DimensionReductionAnalyzer:
    """Studies how PCA and random projection affect pairwise-distance-based metrics."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _pca_project(self, X: np.ndarray, target_dim: int) -> np.ndarray:
        """Project X onto its top-target_dim principal components."""
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return X_centered @ Vt[:target_dim].T

    def _random_project(self, X: np.ndarray, target_dim: int) -> np.ndarray:
        """Gaussian random projection."""
        d = X.shape[1]
        R = self.rng.standard_normal((d, target_dim)) / np.sqrt(target_dim)
        return X @ R

    def _compute_distortion(
        self, D_orig: np.ndarray, D_proj: np.ndarray
    ) -> Tuple[float, float, float]:
        """Mean, std, max multiplicative distortion between distance matrices."""
        mask = D_orig > 0
        if not np.any(mask):
            return 0.0, 0.0, 0.0
        ratios = D_proj[mask] / D_orig[mask]
        return float(np.mean(ratios)), float(np.std(ratios)), float(np.max(np.abs(ratios - 1.0)))

    def _variance_retained(self, X: np.ndarray, target_dim: int) -> float:
        """Fraction of total variance retained by top-k PCA components."""
        X_centered = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        total_var = np.sum(S ** 2)
        if total_var == 0:
            return 1.0
        return float(np.sum(S[:target_dim] ** 2) / total_var)

    def analyze_pca(
        self, X: np.ndarray, target_dim: int, metric_name: str = "pairwise_l2"
    ) -> DimensionReductionEffect:
        """Analyze distortion under PCA projection."""
        D_orig = _pairwise_distances(X)
        X_proj = self._pca_project(X, target_dim)
        D_proj = _pairwise_distances(X_proj)
        mean_d, std_d, max_d = self._compute_distortion(D_orig, D_proj)
        vr = self._variance_retained(X, target_dim)
        return DimensionReductionEffect(
            metric_name=metric_name,
            original_dim=X.shape[1],
            reduced_dim=target_dim,
            method="pca",
            distortion_mean=mean_d,
            distortion_std=std_d,
            max_distortion=max_d,
            variance_retained=vr,
        )

    def analyze_random_projection(
        self, X: np.ndarray, target_dim: int, n_trials: int = 10,
        metric_name: str = "pairwise_l2",
    ) -> DimensionReductionEffect:
        """Analyze distortion under random Gaussian projection (averaged over trials)."""
        D_orig = _pairwise_distances(X)
        means, stds, maxes = [], [], []
        for _ in range(n_trials):
            X_proj = self._random_project(X, target_dim)
            D_proj = _pairwise_distances(X_proj)
            m, s, mx = self._compute_distortion(D_orig, D_proj)
            means.append(m)
            stds.append(s)
            maxes.append(mx)

        n = X.shape[0]
        eps_jl = np.sqrt(8 * np.log(n) / target_dim) if target_dim > 0 else float("inf")
        return DimensionReductionEffect(
            metric_name=metric_name,
            original_dim=X.shape[1],
            reduced_dim=target_dim,
            method="random_projection",
            distortion_mean=float(np.mean(means)),
            distortion_std=float(np.mean(stds)),
            max_distortion=float(np.mean(maxes)),
            jl_bound=float(eps_jl),
        )

    def sweep_dimensions(
        self,
        X: np.ndarray,
        dims: Optional[List[int]] = None,
        method: str = "pca",
        metric_name: str = "pairwise_l2",
    ) -> List[DimensionReductionEffect]:
        """Analyze distortion across a range of target dimensions."""
        max_dim = X.shape[1]
        if dims is None:
            dims = sorted(set(
                int(d) for d in np.logspace(np.log10(2), np.log10(max_dim), 15)
                if 2 <= d <= max_dim
            ))
        results = []
        for d in dims:
            if method == "pca":
                results.append(self.analyze_pca(X, d, metric_name))
            else:
                results.append(self.analyze_random_projection(X, d, metric_name=metric_name))
        return results

    def jl_minimum_dimension(self, n_samples: int, epsilon: float = 0.1) -> int:
        """Return the JL lemma minimum target dimension."""
        return _jl_min_dimension(n_samples, epsilon)


# ---------------------------------------------------------------------------
# AnalyticalDistributionCalculator
# ---------------------------------------------------------------------------

class AnalyticalDistributionCalculator:
    """Computes expected diversity-metric values for known distributions."""

    def __init__(self, base: float = 2.0):
        self.base = base

    # -- Uniform --------------------------------------------------------

    def uniform_entropy(self, n: int) -> AnalyticalExpectation:
        """Entropy of Uniform(n)."""
        h = _safe_log(n, self.base)
        return AnalyticalExpectation(
            metric_name="shannon_entropy",
            distribution_name="uniform",
            distribution_params={"n": n},
            expected_value=h,
            variance=0.0,
            formula=f"log_{self.base}({n})",
        )

    def uniform_type_token_ratio(self, n: int, sample_size: int) -> AnalyticalExpectation:
        """Expected type-token ratio when sampling from Uniform(n)."""
        # E[types] = n * (1 - ((n-1)/n)^sample_size)
        expected_types = n * (1.0 - ((n - 1) / n) ** sample_size)
        ttr = expected_types / sample_size
        return AnalyticalExpectation(
            metric_name="type_token_ratio",
            distribution_name="uniform",
            distribution_params={"n": n, "sample_size": sample_size},
            expected_value=ttr,
            formula="n*(1-((n-1)/n)^m) / m",
        )

    def uniform_distinct_count(self, n: int, sample_size: int) -> AnalyticalExpectation:
        """Expected number of distinct elements when sampling from Uniform(n)."""
        expected = n * (1.0 - ((n - 1) / n) ** sample_size)
        return AnalyticalExpectation(
            metric_name="distinct_count",
            distribution_name="uniform",
            distribution_params={"n": n, "sample_size": sample_size},
            expected_value=expected,
            formula="n*(1-((n-1)/n)^m)",
        )

    # -- Zipf -----------------------------------------------------------

    def zipf_entropy(self, s: float, N: int) -> AnalyticalExpectation:
        """Entropy of Zipf(s, N)."""
        probs = _zipf_probabilities(s, N)
        h = _entropy(probs, self.base)
        return AnalyticalExpectation(
            metric_name="shannon_entropy",
            distribution_name="zipf",
            distribution_params={"s": s, "N": N},
            expected_value=h,
            formula="H(Zipf(s,N))",
        )

    def zipf_renyi_entropy(self, s: float, N: int, alpha: float) -> AnalyticalExpectation:
        """Rényi entropy of Zipf(s, N)."""
        probs = _zipf_probabilities(s, N)
        h = _renyi_entropy(probs, alpha)
        return AnalyticalExpectation(
            metric_name=f"renyi_entropy_alpha_{alpha}",
            distribution_name="zipf",
            distribution_params={"s": s, "N": N, "alpha": alpha},
            expected_value=h,
            formula=f"H_{alpha}(Zipf(s,N))",
        )

    def zipf_type_token_ratio(
        self, s: float, N: int, sample_size: int, n_simulations: int = 5000
    ) -> AnalyticalExpectation:
        """Monte-Carlo estimate of TTR under Zipf(s, N)."""
        probs = _zipf_probabilities(s, N)
        ttrs = []
        rng = np.random.default_rng(0)
        for _ in range(n_simulations):
            sample = rng.choice(N, size=sample_size, p=probs)
            ttrs.append(len(np.unique(sample)) / sample_size)
        mean_ttr = float(np.mean(ttrs))
        std_ttr = float(np.std(ttrs))
        return AnalyticalExpectation(
            metric_name="type_token_ratio",
            distribution_name="zipf",
            distribution_params={"s": s, "N": N, "sample_size": sample_size},
            expected_value=mean_ttr,
            variance=std_ttr ** 2,
            confidence_interval=(
                float(np.percentile(ttrs, 2.5)),
                float(np.percentile(ttrs, 97.5)),
            ),
            formula="Monte-Carlo estimate",
        )

    # -- Geometric ------------------------------------------------------

    def geometric_entropy(self, p: float, N: int) -> AnalyticalExpectation:
        """Entropy of truncated Geometric(p, N)."""
        probs = _geometric_probabilities(p, N)
        h = _entropy(probs, self.base)
        return AnalyticalExpectation(
            metric_name="shannon_entropy",
            distribution_name="geometric",
            distribution_params={"p": p, "N": N},
            expected_value=h,
            formula="H(TruncGeom(p,N))",
        )

    # -- Dirichlet ------------------------------------------------------

    def dirichlet_expected_entropy(self, alpha: np.ndarray) -> AnalyticalExpectation:
        """Analytically expected entropy of a draw from Dirichlet(alpha)."""
        expected_h = _dirichlet_entropy_expectation(alpha)
        return AnalyticalExpectation(
            metric_name="shannon_entropy",
            distribution_name="dirichlet",
            distribution_params={"alpha": list(alpha)},
            expected_value=expected_h,
            formula="psi(alpha0+1) - sum_k (alpha_k/alpha0)*psi(alpha_k+1)",
        )

    # -- Comparison helper ---------------------------------------------

    def compare_distributions(
        self, metric_name: str, N: int, sample_size: int
    ) -> Dict[str, AnalyticalExpectation]:
        """Compute a metric across multiple reference distributions."""
        results: Dict[str, AnalyticalExpectation] = {}
        if metric_name == "shannon_entropy":
            results["uniform"] = self.uniform_entropy(N)
            results["zipf_1.0"] = self.zipf_entropy(1.0, N)
            results["zipf_1.5"] = self.zipf_entropy(1.5, N)
            results["geometric_0.01"] = self.geometric_entropy(0.01, N)
        elif metric_name == "type_token_ratio":
            results["uniform"] = self.uniform_type_token_ratio(N, sample_size)
            results["zipf_1.0"] = self.zipf_type_token_ratio(1.0, N, sample_size, n_simulations=500)
        return results


# ---------------------------------------------------------------------------
# CalibrationBaseline
# ---------------------------------------------------------------------------

class CalibrationBaseline:
    """Creates calibration curves by computing metrics at known diversity levels."""

    def __init__(self, vocab_size: int = 10_000, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

    def _generate_distribution(
        self, diversity_level: float, vocab_size: int
    ) -> np.ndarray:
        """Map diversity_level in [0, 1] to a probability distribution.

        0 → point mass, 1 → uniform. Intermediate values interpolate
        via a Zipf exponent: s = max(0.01, 4 * (1 - diversity_level)).
        """
        if diversity_level >= 1.0:
            return np.ones(vocab_size) / vocab_size
        if diversity_level <= 0.0:
            p = np.zeros(vocab_size)
            p[0] = 1.0
            return p
        s = max(0.01, 4.0 * (1.0 - diversity_level))
        return _zipf_probabilities(s, vocab_size)

    def build_calibration_curve(
        self,
        metric_fn: Callable[[np.ndarray], float],
        n_levels: int = 50,
        sample_size: int = 1000,
        n_trials: int = 100,
        metric_name: str = "metric",
    ) -> List[CalibrationPoint]:
        """Build a calibration curve: diversity_level -> metric value."""
        levels = np.linspace(0.0, 1.0, n_levels)
        points: List[CalibrationPoint] = []
        for level in levels:
            probs = self._generate_distribution(level, self.vocab_size)
            trial_values = []
            for _ in range(n_trials):
                sample = self.rng.choice(self.vocab_size, size=sample_size, p=probs)
                trial_values.append(metric_fn(sample))
            points.append(CalibrationPoint(
                diversity_level=float(level),
                expected_metric_value=float(np.mean(trial_values)),
                std_metric_value=float(np.std(trial_values)),
                sample_size=sample_size,
                distribution_type="zipf_interpolated",
            ))
        return points

    def calibration_curve_to_arrays(
        self, curve: List[CalibrationPoint]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (levels, means, stds) arrays from a calibration curve."""
        levels = np.array([p.diversity_level for p in curve])
        means = np.array([p.expected_metric_value for p in curve])
        stds = np.array([p.std_metric_value for p in curve])
        return levels, means, stds

    def invert_calibration(
        self, curve: List[CalibrationPoint], observed_value: float
    ) -> float:
        """Given an observed metric value, infer the diversity level via interpolation."""
        levels, means, _ = self.calibration_curve_to_arrays(curve)
        # ensure monotonic for interpolation
        order = np.argsort(means)
        means_sorted = means[order]
        levels_sorted = levels[order]
        observed_clamped = np.clip(observed_value, means_sorted[0], means_sorted[-1])
        return float(np.interp(observed_clamped, means_sorted, levels_sorted))

    def percentile_rank(
        self, curve: List[CalibrationPoint], observed_value: float
    ) -> float:
        """Return the percentile rank of observed_value w.r.t. the calibration curve."""
        means = np.array([p.expected_metric_value for p in curve])
        return float(np.mean(means <= observed_value) * 100.0)


# ---------------------------------------------------------------------------
# MetricAxiomChecker
# ---------------------------------------------------------------------------

class MetricAxiomChecker:
    """Verifies metric axioms: non-negativity, identity of indiscernibles,
    symmetry, and triangle inequality for distance-like metrics."""

    def __init__(self, tolerance: float = 1e-9, n_tests: int = 500, seed: int = 42):
        self.tolerance = tolerance
        self.n_tests = n_tests
        self.rng = np.random.default_rng(seed)

    def _random_vectors(self, n: int, dim: int) -> np.ndarray:
        return self.rng.standard_normal((n, dim))

    # -- Non-negativity --------------------------------------------------

    def check_nonnegativity(
        self, metric_fn: Callable[[np.ndarray, np.ndarray], float],
        dim: int = 10, metric_name: str = "metric",
    ) -> AxiomCheckResult:
        """d(x, y) >= 0 for all x, y."""
        vecs = self._random_vectors(self.n_tests, dim)
        violations: List[Dict[str, Any]] = []
        max_violation = 0.0
        for i in range(0, len(vecs) - 1, 2):
            val = metric_fn(vecs[i], vecs[i + 1])
            if val < -self.tolerance:
                violations.append({"i": i, "j": i + 1, "value": val})
                max_violation = max(max_violation, abs(val))
        return AxiomCheckResult(
            axiom_name="non-negativity",
            metric_name=metric_name,
            satisfied=len(violations) == 0,
            max_violation=max_violation,
            violation_examples=violations[:10],
            num_tests=self.n_tests // 2,
        )

    # -- Identity of indiscernibles -------------------------------------

    def check_identity_of_indiscernibles(
        self, metric_fn: Callable[[np.ndarray, np.ndarray], float],
        dim: int = 10, metric_name: str = "metric",
    ) -> AxiomCheckResult:
        """d(x, x) = 0 for all x, and d(x, y) > 0 for x != y."""
        vecs = self._random_vectors(self.n_tests, dim)
        violations: List[Dict[str, Any]] = []
        max_violation = 0.0
        # d(x, x) == 0
        for i in range(min(self.n_tests, 200)):
            val = metric_fn(vecs[i], vecs[i])
            if abs(val) > self.tolerance:
                violations.append({"type": "self_distance_nonzero", "i": i, "value": val})
                max_violation = max(max_violation, abs(val))
        # d(x, y) > 0 for x != y (spot check)
        for i in range(0, min(self.n_tests - 1, 200), 2):
            val = metric_fn(vecs[i], vecs[i + 1])
            if val < self.tolerance and not np.allclose(vecs[i], vecs[i + 1]):
                violations.append({"type": "distinct_zero", "i": i, "j": i + 1, "value": val})
                max_violation = max(max_violation, abs(val))
        return AxiomCheckResult(
            axiom_name="identity_of_indiscernibles",
            metric_name=metric_name,
            satisfied=len(violations) == 0,
            max_violation=max_violation,
            violation_examples=violations[:10],
            num_tests=min(self.n_tests, 200) + min(self.n_tests - 1, 200) // 2,
        )

    # -- Symmetry -------------------------------------------------------

    def check_symmetry(
        self, metric_fn: Callable[[np.ndarray, np.ndarray], float],
        dim: int = 10, metric_name: str = "metric",
    ) -> AxiomCheckResult:
        """d(x, y) == d(y, x)."""
        vecs = self._random_vectors(self.n_tests, dim)
        violations: List[Dict[str, Any]] = []
        max_violation = 0.0
        for i in range(0, len(vecs) - 1, 2):
            dxy = metric_fn(vecs[i], vecs[i + 1])
            dyx = metric_fn(vecs[i + 1], vecs[i])
            diff = abs(dxy - dyx)
            if diff > self.tolerance:
                violations.append({"i": i, "j": i + 1, "d_xy": dxy, "d_yx": dyx, "diff": diff})
                max_violation = max(max_violation, diff)
        return AxiomCheckResult(
            axiom_name="symmetry",
            metric_name=metric_name,
            satisfied=len(violations) == 0,
            max_violation=max_violation,
            violation_examples=violations[:10],
            num_tests=self.n_tests // 2,
        )

    # -- Triangle inequality --------------------------------------------

    def check_triangle_inequality(
        self, metric_fn: Callable[[np.ndarray, np.ndarray], float],
        dim: int = 10, metric_name: str = "metric",
    ) -> AxiomCheckResult:
        """d(x, z) <= d(x, y) + d(y, z) for all x, y, z."""
        n_triples = min(self.n_tests, 300)
        vecs = self._random_vectors(n_triples * 3, dim)
        violations: List[Dict[str, Any]] = []
        max_violation = 0.0
        for t in range(n_triples):
            x, y, z = vecs[3 * t], vecs[3 * t + 1], vecs[3 * t + 2]
            dxz = metric_fn(x, z)
            dxy = metric_fn(x, y)
            dyz = metric_fn(y, z)
            gap = dxz - (dxy + dyz)
            if gap > self.tolerance:
                violations.append({
                    "triple": t, "d_xz": dxz, "d_xy": dxy, "d_yz": dyz, "gap": gap
                })
                max_violation = max(max_violation, gap)
        return AxiomCheckResult(
            axiom_name="triangle_inequality",
            metric_name=metric_name,
            satisfied=len(violations) == 0,
            max_violation=max_violation,
            violation_examples=violations[:10],
            num_tests=n_triples,
        )

    # -- Full axiom suite -----------------------------------------------

    def check_all_axioms(
        self, metric_fn: Callable[[np.ndarray, np.ndarray], float],
        dim: int = 10, metric_name: str = "metric",
    ) -> List[AxiomCheckResult]:
        """Run all four metric axiom checks."""
        return [
            self.check_nonnegativity(metric_fn, dim, metric_name),
            self.check_identity_of_indiscernibles(metric_fn, dim, metric_name),
            self.check_symmetry(metric_fn, dim, metric_name),
            self.check_triangle_inequality(metric_fn, dim, metric_name),
        ]

    def check_divergence_axioms(
        self, div_fn: Callable[[np.ndarray, np.ndarray], float],
        dim: int = 10, metric_name: str = "divergence",
    ) -> List[AxiomCheckResult]:
        """Check axioms appropriate for a divergence (non-negativity, identity only)."""
        results = [
            self.check_nonnegativity(div_fn, dim, metric_name),
            self.check_identity_of_indiscernibles(div_fn, dim, metric_name),
        ]
        sym_result = self.check_symmetry(div_fn, dim, metric_name)
        sym_result.notes = "Divergences are not required to be symmetric"
        results.append(sym_result)
        return results


# ---------------------------------------------------------------------------
# ConvergenceRateEstimator
# ---------------------------------------------------------------------------

class ConvergenceRateEstimator:
    """Estimates convergence rates of metrics as sample size grows."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def estimate_convergence_rate(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data_generator: Callable[[int], np.ndarray],
        true_value: float,
        sample_sizes: Optional[List[int]] = None,
        n_trials: int = 100,
        metric_name: str = "metric",
    ) -> ConvergenceEstimate:
        """Estimate the rate at which metric_fn converges to true_value."""
        if sample_sizes is None:
            sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

        mean_errors: List[float] = []
        for n in sample_sizes:
            trial_errors = []
            for _ in range(n_trials):
                data = data_generator(n)
                val = metric_fn(data)
                trial_errors.append(abs(val - true_value))
            mean_errors.append(float(np.mean(trial_errors)))

        sizes_arr = np.array(sample_sizes, dtype=np.float64)
        errors_arr = np.array(mean_errors, dtype=np.float64)
        alpha, C, r2 = _fit_power_law(sizes_arr, errors_arr)

        # asymptotic bias: extrapolate error at n -> inf
        if alpha < 0:
            bias = 0.0
        else:
            bias = float(C * (sample_sizes[-1] ** alpha))

        return ConvergenceEstimate(
            metric_name=metric_name,
            rate_exponent=-alpha,  # error ~ n^{alpha}, rate is -alpha
            rate_constant=C,
            sample_sizes=sample_sizes,
            errors=mean_errors,
            fit_r_squared=r2,
            asymptotic_bias=bias,
        )

    def estimate_entropy_convergence(
        self, vocab_size: int = 1000, base: float = 2.0
    ) -> ConvergenceEstimate:
        """Convergence rate of plug-in entropy estimator on Uniform(vocab_size)."""
        probs = np.ones(vocab_size) / vocab_size
        true_h = _entropy(probs, base)

        def gen(n: int) -> np.ndarray:
            return self.rng.choice(vocab_size, size=n, p=probs)

        def plug_in_entropy(sample: np.ndarray) -> float:
            counts = np.bincount(sample.astype(int), minlength=vocab_size)
            freqs = counts / counts.sum()
            return _entropy(freqs, base)

        return self.estimate_convergence_rate(
            plug_in_entropy, gen, true_h,
            metric_name="plug_in_entropy",
        )

    def estimate_ttr_convergence(
        self, vocab_size: int = 1000
    ) -> ConvergenceEstimate:
        """Convergence rate of type-token ratio on Uniform(vocab_size).

        TTR converges to 0 for fixed vocab as sample grows, so we measure
        convergence of the rescaled estimate n*TTR -> vocab_size.
        """
        def gen(n: int) -> np.ndarray:
            return self.rng.choice(vocab_size, size=n)

        def rescaled_ttr(sample: np.ndarray) -> float:
            return len(np.unique(sample))

        # true value for distinct count at large n is vocab_size
        true_val = float(vocab_size)
        return self.estimate_convergence_rate(
            rescaled_ttr, gen, true_val,
            sample_sizes=[50, 100, 200, 500, 1000, 2000, 5000],
            metric_name="distinct_count",
        )

    def bootstrap_confidence_interval(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for a metric."""
        n = len(data)
        boot_values = []
        for _ in range(n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            boot_values.append(metric_fn(data[idx]))
        alpha = (1.0 - confidence) / 2.0
        lower = float(np.percentile(boot_values, 100 * alpha))
        upper = float(np.percentile(boot_values, 100 * (1 - alpha)))
        return lower, upper

    def effective_sample_size(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        n_bootstrap: int = 500,
    ) -> float:
        """Estimate effective sample size via bootstrap variance ratio.

        ESS = n * Var_iid / Var_bootstrap, estimated by comparing
        bootstrap variance to the variance expected under iid.
        """
        n = len(data)
        boot_values = []
        for _ in range(n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            boot_values.append(metric_fn(data[idx]))
        var_boot = float(np.var(boot_values))
        if var_boot == 0:
            return float(n)
        # under iid, Var scales as 1/n for means
        point_est = metric_fn(data)
        # estimate single-observation variance
        half = n // 2
        v1 = metric_fn(data[:half])
        v2 = metric_fn(data[half:])
        var_single = (v1 - v2) ** 2 / 2.0
        if var_single == 0:
            return float(n)
        return float(min(n, var_single / var_boot))


# ---------------------------------------------------------------------------
# Sensitivity analysis helpers
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Sensitivity of a metric to perturbations."""

    metric_name: str
    perturbation_type: str
    epsilon: float
    max_metric_change: float
    mean_metric_change: float
    lipschitz_estimate: float
    n_trials: int


class PerturbationSensitivityAnalyzer:
    """Bounds on how much a metric can change under small perturbations."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def gaussian_perturbation_sensitivity(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        epsilon: float = 0.01,
        n_trials: int = 200,
        metric_name: str = "metric",
    ) -> SensitivityResult:
        """Measure metric sensitivity to additive Gaussian noise."""
        base_value = metric_fn(data)
        changes = []
        for _ in range(n_trials):
            noise = self.rng.standard_normal(data.shape) * epsilon
            perturbed = data + noise
            new_value = metric_fn(perturbed)
            changes.append(abs(new_value - base_value))
        changes_arr = np.array(changes)
        lipschitz = float(np.max(changes_arr) / epsilon) if epsilon > 0 else 0.0
        return SensitivityResult(
            metric_name=metric_name,
            perturbation_type="gaussian",
            epsilon=epsilon,
            max_metric_change=float(np.max(changes_arr)),
            mean_metric_change=float(np.mean(changes_arr)),
            lipschitz_estimate=lipschitz,
            n_trials=n_trials,
        )

    def swap_perturbation_sensitivity(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        n_swaps: int = 1,
        n_trials: int = 200,
        metric_name: str = "metric",
    ) -> SensitivityResult:
        """Measure metric sensitivity to swapping elements in the data."""
        base_value = metric_fn(data)
        n = len(data)
        changes = []
        for _ in range(n_trials):
            perturbed = data.copy()
            for _ in range(n_swaps):
                i, j = self.rng.choice(n, size=2, replace=False)
                perturbed[i], perturbed[j] = perturbed[j].copy(), perturbed[i].copy()
            new_value = metric_fn(perturbed)
            changes.append(abs(new_value - base_value))
        changes_arr = np.array(changes)
        return SensitivityResult(
            metric_name=metric_name,
            perturbation_type=f"swap_{n_swaps}",
            epsilon=float(n_swaps) / n,
            max_metric_change=float(np.max(changes_arr)),
            mean_metric_change=float(np.mean(changes_arr)),
            lipschitz_estimate=float(np.max(changes_arr)) * n / max(n_swaps, 1),
            n_trials=n_trials,
        )

    def dropout_sensitivity(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        drop_fraction: float = 0.05,
        n_trials: int = 200,
        metric_name: str = "metric",
    ) -> SensitivityResult:
        """Measure metric sensitivity to dropping a fraction of samples."""
        base_value = metric_fn(data)
        n = len(data)
        n_drop = max(1, int(n * drop_fraction))
        changes = []
        for _ in range(n_trials):
            keep = self.rng.choice(n, size=n - n_drop, replace=False)
            new_value = metric_fn(data[keep])
            changes.append(abs(new_value - base_value))
        changes_arr = np.array(changes)
        return SensitivityResult(
            metric_name=metric_name,
            perturbation_type=f"dropout_{drop_fraction}",
            epsilon=drop_fraction,
            max_metric_change=float(np.max(changes_arr)),
            mean_metric_change=float(np.mean(changes_arr)),
            lipschitz_estimate=float(np.max(changes_arr)) / max(drop_fraction, 1e-12),
            n_trials=n_trials,
        )

    def full_sensitivity_report(
        self,
        metric_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        metric_name: str = "metric",
    ) -> List[SensitivityResult]:
        """Run all perturbation sensitivity analyses."""
        results = []
        for eps in [0.001, 0.01, 0.1]:
            results.append(
                self.gaussian_perturbation_sensitivity(metric_fn, data, eps, metric_name=metric_name)
            )
        for n_swaps in [1, 5, 10]:
            results.append(
                self.swap_perturbation_sensitivity(metric_fn, data, n_swaps, metric_name=metric_name)
            )
        for frac in [0.01, 0.05, 0.1]:
            results.append(
                self.dropout_sensitivity(metric_fn, data, frac, metric_name=metric_name)
            )
        return results


# ---------------------------------------------------------------------------
# Composite verification pipeline
# ---------------------------------------------------------------------------

@dataclass
class TheoryVerificationReport:
    """Aggregated report from running all theoretical verifications."""

    bounds: List[MetricTheoreticalBound] = field(default_factory=list)
    monotonicity_results: List[MonotonicityResult] = field(default_factory=list)
    dimension_effects: List[DimensionReductionEffect] = field(default_factory=list)
    analytical_expectations: List[AnalyticalExpectation] = field(default_factory=list)
    axiom_checks: List[AxiomCheckResult] = field(default_factory=list)
    convergence_estimates: List[ConvergenceEstimate] = field(default_factory=list)
    sensitivity_results: List[SensitivityResult] = field(default_factory=list)

    @property
    def all_axioms_satisfied(self) -> bool:
        return all(a.satisfied for a in self.axiom_checks)

    @property
    def all_monotone(self) -> bool:
        return all(m.is_monotone for m in self.monotonicity_results)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_bounds": len(self.bounds),
            "n_monotonicity": len(self.monotonicity_results),
            "all_monotone": self.all_monotone,
            "n_dim_effects": len(self.dimension_effects),
            "n_analytical": len(self.analytical_expectations),
            "n_axiom_checks": len(self.axiom_checks),
            "all_axioms_ok": self.all_axioms_satisfied,
            "n_convergence": len(self.convergence_estimates),
            "n_sensitivity": len(self.sensitivity_results),
        }


def run_full_theory_verification(
    metric_fn: Callable[[np.ndarray], float],
    pairwise_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    data: Optional[np.ndarray] = None,
    vocab_size: int = 1000,
    metric_name: str = "metric",
    seed: int = 42,
) -> TheoryVerificationReport:
    """Run a comprehensive theory-verification pipeline for a single metric.

    Parameters
    ----------
    metric_fn : callable
        Metric that operates on a 1-D or 2-D sample array.
    pairwise_metric_fn : callable, optional
        Metric that operates on two vectors (for axiom checking).
    data : ndarray, optional
        Real data for sensitivity and monotonicity tests. If None,
        synthetic data is generated.
    vocab_size : int
        Size of the token vocabulary for analytical calculations.
    metric_name : str
        Human-readable metric name for labelling results.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    report = TheoryVerificationReport()

    # --- Bounds ---
    it_analyzer = InformationTheoreticAnalyzer(vocab_size=vocab_size)
    report.bounds.append(it_analyzer.entropy_lower_bound())
    report.bounds.append(it_analyzer.entropy_upper_bound(vocab_size))

    # --- Analytical expectations ---
    calc = AnalyticalDistributionCalculator()
    report.analytical_expectations.append(calc.uniform_entropy(vocab_size))
    report.analytical_expectations.append(calc.zipf_entropy(1.0, vocab_size))
    report.analytical_expectations.append(calc.geometric_entropy(0.05, vocab_size))
    report.analytical_expectations.append(
        calc.dirichlet_expected_entropy(np.ones(vocab_size))
    )

    # --- Monotonicity ---
    mono = MonotonicityVerifier()
    probs_uniform = np.ones(vocab_size) / vocab_size
    report.monotonicity_results.append(
        mono.verify_renyi_entropy_monotonicity(probs_uniform)
    )

    # --- Dimension reduction ---
    if data is not None and data.ndim == 2 and data.shape[1] > 2:
        dr = DimensionReductionAnalyzer(seed=seed)
        subset = data[:min(500, len(data))]
        for dim in [2, 5, min(10, data.shape[1])]:
            if dim < data.shape[1]:
                report.dimension_effects.append(dr.analyze_pca(subset, dim, metric_name))
                report.dimension_effects.append(
                    dr.analyze_random_projection(subset, dim, metric_name=metric_name)
                )

    # --- Axiom checks ---
    if pairwise_metric_fn is not None:
        checker = MetricAxiomChecker(seed=seed)
        report.axiom_checks.extend(
            checker.check_all_axioms(pairwise_metric_fn, dim=10, metric_name=metric_name)
        )

    # --- Convergence ---
    conv = ConvergenceRateEstimator(seed=seed)
    report.convergence_estimates.append(
        conv.estimate_entropy_convergence(vocab_size=min(vocab_size, 500))
    )

    # --- Sensitivity ---
    if data is not None:
        sens = PerturbationSensitivityAnalyzer(seed=seed)
        report.sensitivity_results.extend(
            sens.full_sensitivity_report(metric_fn, data, metric_name)
        )

    return report
