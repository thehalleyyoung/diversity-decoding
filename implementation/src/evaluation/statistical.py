"""
Comprehensive statistical test suite for the Diversity Decoding Arena.

Provides non-parametric hypothesis tests, effect size calculators,
confidence interval methods, power analysis, and multiple comparison
corrections for rigorous algorithm comparison.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import special, stats


# ---------------------------------------------------------------------------
# Shared enums and result containers
# ---------------------------------------------------------------------------

class Alternative(str, Enum):
    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"


class EffectMagnitude(str, Enum):
    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class TestResult:
    """Container for a single statistical test outcome."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_magnitude: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional: Dict[str, Any] = field(default_factory=dict)
    rejected: Optional[bool] = None
    alpha: float = 0.05

    def __post_init__(self):
        if self.rejected is None:
            self.rejected = self.p_value < self.alpha

    def summary(self) -> str:
        parts = [
            f"Test: {self.test_name}",
            f"  Statistic = {self.statistic:.6f}",
            f"  p-value   = {self.p_value:.6g}",
        ]
        if self.effect_size is not None:
            parts.append(f"  Effect size = {self.effect_size:.4f}")
        if self.effect_magnitude is not None:
            parts.append(f"  Magnitude   = {self.effect_magnitude}")
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            parts.append(f"  CI          = [{lo:.6f}, {hi:.6f}]")
        parts.append(f"  Reject H0 at alpha={self.alpha}? {self.rejected}")
        return "\n".join(parts)


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two algorithms."""
    algorithm_a: str
    algorithm_b: str
    statistic: float
    p_value: float
    adjusted_p_value: Optional[float] = None
    effect_size: Optional[float] = None
    significant: bool = False


@dataclass
class MultipleComparisonResult:
    """Result of a multiple comparison procedure."""
    method: str
    original_p_values: List[float]
    adjusted_p_values: List[float]
    rejected: List[bool]
    alpha: float = 0.05


@dataclass
class PowerResult:
    """Result of a power analysis."""
    test_name: str
    effect_size: float
    alpha: float
    power: float
    sample_size: Optional[int] = None
    additional: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------

class WilcoxonSignedRankTest:
    """
    Wilcoxon signed-rank test for paired samples.

    Uses exact distribution for small *n* (≤ 25 by default) and the normal
    approximation (with continuity correction) for larger samples.
    """

    EXACT_THRESHOLD = 25

    def __init__(
        self,
        alternative: str = "two-sided",
        zero_method: str = "wilcox",
        exact_threshold: int = 25,
    ):
        self.alternative = Alternative(alternative)
        self.zero_method = zero_method
        self.exact_threshold = exact_threshold

    # ----- public API -----

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
    ) -> TestResult:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        d = x - y
        d, n_zero = self._handle_zeros(d)
        if len(d) == 0:
            return TestResult(
                test_name="Wilcoxon signed-rank",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
            )

        ranks = self._rank_abs(d)
        w_plus = np.sum(ranks[d > 0])
        w_minus = np.sum(ranks[d < 0])
        n = len(d)

        if self.alternative == Alternative.TWO_SIDED:
            T = min(w_plus, w_minus)
        elif self.alternative == Alternative.LESS:
            T = w_plus
        else:
            T = w_minus

        if n <= self.exact_threshold:
            p = self._exact_p(T, n)
        else:
            p = self._normal_approx_p(T, n, ranks)

        N = len(x)
        z = stats.norm.isf(p / 2) if p > 0 else np.inf
        effect_r = z / math.sqrt(N) if N > 0 else 0.0
        mag = self._interpret_r(effect_r)

        ci = self._confidence_interval_median_diff(d, alpha)

        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=float(T),
            p_value=float(p),
            effect_size=float(effect_r),
            effect_magnitude=mag,
            confidence_interval=ci,
            alpha=alpha,
            additional={
                "W_plus": float(w_plus),
                "W_minus": float(w_minus),
                "n_effective": int(n),
                "n_zeros": int(n_zero),
                "alternative": self.alternative.value,
            },
        )

    # ----- internal helpers -----

    def _handle_zeros(self, d: np.ndarray) -> Tuple[np.ndarray, int]:
        mask = d != 0
        n_zero = int(np.sum(~mask))
        if self.zero_method == "wilcox":
            return d[mask], n_zero
        elif self.zero_method == "pratt":
            return d, n_zero
        elif self.zero_method == "zsplit":
            d = d.copy()
            zero_idx = np.where(d == 0)[0]
            half = len(zero_idx) // 2
            d[zero_idx[:half]] = -1e-300
            d[zero_idx[half:]] = 1e-300
            return d, 0
        return d[mask], n_zero

    @staticmethod
    def _rank_abs(d: np.ndarray) -> np.ndarray:
        abs_d = np.abs(d)
        order = np.argsort(abs_d)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(d) + 1, dtype=float)
        # average ties
        unique_vals = np.unique(abs_d)
        for v in unique_vals:
            idx = np.where(abs_d == v)[0]
            if len(idx) > 1:
                ranks[idx] = ranks[idx].mean()
        return ranks

    def _exact_p(self, T: float, n: int) -> float:
        """Enumerate the 2^n sign assignments to compute exact p-value."""
        if n > 30:
            return self._normal_approx_p(T, n, np.arange(1, n + 1, dtype=float))
        count = 0
        total = 1 << n
        rank_vals = np.arange(1, n + 1, dtype=float)
        for mask_int in range(total):
            s = 0.0
            for j in range(n):
                if mask_int & (1 << j):
                    s += rank_vals[j]
            if self.alternative == Alternative.TWO_SIDED:
                w_plus = s
                w_minus = n * (n + 1) / 2.0 - s
                t = min(w_plus, w_minus)
                if t <= T:
                    count += 1
            elif self.alternative == Alternative.LESS:
                if s <= T:
                    count += 1
            else:
                w_minus = n * (n + 1) / 2.0 - s
                if w_minus <= T:
                    count += 1
        return count / total

    def _normal_approx_p(
        self, T: float, n: int, ranks: np.ndarray
    ) -> float:
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        # tie correction
        unique, counts = np.unique(np.abs(ranks), return_counts=True)
        for c in counts:
            if c > 1:
                var -= (c ** 3 - c) / 48.0
        if var <= 0:
            return 1.0
        se = math.sqrt(var)
        z = (T - mean) / se
        if self.alternative == Alternative.TWO_SIDED:
            p = 2.0 * stats.norm.cdf(-abs(z))
        elif self.alternative == Alternative.LESS:
            p = stats.norm.cdf(z)
        else:
            p = stats.norm.sf(z)
        return float(min(p, 1.0))

    @staticmethod
    def _confidence_interval_median_diff(
        d: np.ndarray, alpha: float
    ) -> Tuple[float, float]:
        """Hodges-Lehmann estimator CI via Walsh averages."""
        n = len(d)
        if n == 0:
            return (np.nan, np.nan)
        walsh = []
        for i in range(n):
            for j in range(i, n):
                walsh.append((d[i] + d[j]) / 2.0)
        walsh = np.sort(walsh)
        k = int(np.round(stats.norm.ppf(1 - alpha / 2) * math.sqrt(n * (n + 1) * (2 * n + 1) / 6.0) / 2.0))
        k = max(0, min(k, len(walsh) - 1))
        lo_idx = max(k, 0)
        hi_idx = min(len(walsh) - 1 - k, len(walsh) - 1)
        return (float(walsh[lo_idx]), float(walsh[hi_idx]))

    @staticmethod
    def _interpret_r(r: float) -> str:
        r = abs(r)
        if r < 0.1:
            return EffectMagnitude.NEGLIGIBLE.value
        if r < 0.3:
            return EffectMagnitude.SMALL.value
        if r < 0.5:
            return EffectMagnitude.MEDIUM.value
        return EffectMagnitude.LARGE.value


# ---------------------------------------------------------------------------
# 2. Friedman Test
# ---------------------------------------------------------------------------

class FriedmanTest:
    """
    Friedman test for comparing *k* algorithms across *n* tasks.

    Also computes the Iman-Davenport adjusted statistic which follows
    an *F*-distribution and is generally more powerful.
    """

    def test(
        self,
        data: np.ndarray,
        algorithm_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Parameters
        ----------
        data : array of shape (n_tasks, k_algorithms)
        algorithm_names : optional names for each algorithm column
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be 2-D (n_tasks × k_algorithms)")
        n, k = data.shape
        if k < 2:
            raise ValueError("Need at least 2 algorithms to compare")

        ranks = np.zeros_like(data)
        for i in range(n):
            ranks[i] = stats.rankdata(data[i])

        mean_ranks = ranks.mean(axis=0)
        grand_mean = (k + 1) / 2.0

        ss = n * np.sum((mean_ranks - grand_mean) ** 2)
        chi2 = 12.0 * ss / (k * (k + 1))

        df = k - 1
        p_chi2 = float(stats.chi2.sf(chi2, df))

        # Iman-Davenport adjusted
        if chi2 == 0:
            F_stat = 0.0
            p_f = 1.0
        else:
            F_stat = ((n - 1) * chi2) / (n * (k - 1) - chi2)
            df1 = k - 1
            df2 = (k - 1) * (n - 1)
            p_f = float(stats.f.sf(F_stat, df1, df2))

        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(k)]

        ranking_dict = {
            name: float(r) for name, r in zip(algorithm_names, mean_ranks)
        }

        return TestResult(
            test_name="Friedman",
            statistic=float(chi2),
            p_value=p_chi2,
            alpha=alpha,
            additional={
                "iman_davenport_F": float(F_stat),
                "iman_davenport_p": float(p_f),
                "mean_ranks": ranking_dict,
                "n_tasks": int(n),
                "k_algorithms": int(k),
                "df_chi2": int(df),
            },
        )

    def rank_algorithms(
        self, data: np.ndarray, algorithm_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        data = np.asarray(data, dtype=float)
        n, k = data.shape
        ranks = np.zeros_like(data)
        for i in range(n):
            ranks[i] = stats.rankdata(data[i])
        mean_ranks = ranks.mean(axis=0)
        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(k)]
        pairs = list(zip(algorithm_names, mean_ranks.tolist()))
        pairs.sort(key=lambda x: x[1])
        return pairs


# ---------------------------------------------------------------------------
# 3. Nemenyi Post-Hoc Test
# ---------------------------------------------------------------------------

class NemenyiPostHoc:
    """
    Post-hoc Nemenyi test after a significant Friedman result.

    Uses the Studentized Range (q) distribution to compute critical
    differences for all pairwise comparisons.
    """

    # q_alpha values for alpha=0.05, k=2..20 (approximate)
    _Q_TABLE_005: Dict[int, float] = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.314,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
        18: 3.489, 19: 3.517, 20: 3.544,
    }

    _Q_TABLE_010: Dict[int, float] = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
        6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855,
        10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077,
        14: 3.120, 15: 3.159, 16: 3.196, 17: 3.230,
        18: 3.261, 19: 3.291, 20: 3.319,
    }

    def test(
        self,
        data: np.ndarray,
        algorithm_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float)
        n, k = data.shape
        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(k)]

        ranks = np.zeros_like(data)
        for i in range(n):
            ranks[i] = stats.rankdata(data[i])
        mean_ranks = ranks.mean(axis=0)

        cd = self.critical_difference(n, k, alpha)

        pairwise: List[PairwiseResult] = []
        for i, j in combinations(range(k), 2):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            # approximate p-value via normal
            se = math.sqrt(k * (k + 1) / (6.0 * n))
            z = diff / se
            p_raw = 2.0 * stats.norm.sf(z)
            n_comparisons = k * (k - 1) // 2
            p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni
            sig = diff >= cd
            pairwise.append(PairwiseResult(
                algorithm_a=algorithm_names[i],
                algorithm_b=algorithm_names[j],
                statistic=float(diff),
                p_value=float(p_raw),
                adjusted_p_value=float(p_adj),
                significant=sig,
            ))

        cd_diagram = self._cd_diagram_data(mean_ranks, algorithm_names, cd)

        return {
            "critical_difference": float(cd),
            "mean_ranks": {
                name: float(r) for name, r in zip(algorithm_names, mean_ranks)
            },
            "pairwise": pairwise,
            "cd_diagram": cd_diagram,
            "n_tasks": int(n),
            "k_algorithms": int(k),
            "alpha": alpha,
        }

    def critical_difference(self, n: int, k: int, alpha: float = 0.05) -> float:
        q = self._get_q(k, alpha)
        return q * math.sqrt(k * (k + 1) / (6.0 * n))

    def _get_q(self, k: int, alpha: float) -> float:
        table = self._Q_TABLE_005 if alpha <= 0.05 else self._Q_TABLE_010
        if k in table:
            return table[k]
        if k > max(table):
            return table[max(table)] + 0.03 * (k - max(table))
        keys = sorted(table.keys())
        for idx in range(len(keys) - 1):
            if keys[idx] <= k <= keys[idx + 1]:
                frac = (k - keys[idx]) / (keys[idx + 1] - keys[idx])
                return table[keys[idx]] * (1 - frac) + table[keys[idx + 1]] * frac
        return 2.0

    @staticmethod
    def _cd_diagram_data(
        mean_ranks: np.ndarray,
        names: List[str],
        cd: float,
    ) -> Dict[str, Any]:
        order = np.argsort(mean_ranks)
        sorted_names = [names[i] for i in order]
        sorted_ranks = mean_ranks[order].tolist()

        cliques: List[List[str]] = []
        n = len(sorted_names)
        for i in range(n):
            clique = [sorted_names[i]]
            for j in range(i + 1, n):
                if sorted_ranks[j] - sorted_ranks[i] < cd:
                    clique.append(sorted_names[j])
                else:
                    break
            if len(clique) > 1:
                already = False
                for c in cliques:
                    if set(clique).issubset(set(c)):
                        already = True
                        break
                if not already:
                    cliques.append(clique)

        return {
            "sorted_names": sorted_names,
            "sorted_ranks": sorted_ranks,
            "cd": float(cd),
            "cliques": cliques,
        }


# ---------------------------------------------------------------------------
# 4. Bootstrap Test
# ---------------------------------------------------------------------------

class BootstrapTest:
    """
    Bootstrap-based inference: confidence intervals (percentile, BCa, basic),
    paired comparison, hypothesis tests, and stratified bootstrap.
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        random_state: Optional[int] = None,
    ):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    # ---- confidence intervals ----

    def confidence_interval(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float] = np.mean,
        confidence: float = 0.95,
        method: str = "percentile",
    ) -> Tuple[float, float]:
        data = np.asarray(data, dtype=float)
        dispatch = {
            "percentile": self._ci_percentile,
            "bca": self._ci_bca,
            "basic": self._ci_basic,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method {method!r}, choose from {list(dispatch)}")
        return dispatch[method](data, statistic, confidence)

    def _ci_percentile(
        self,
        data: np.ndarray,
        statistic: Callable,
        confidence: float,
    ) -> Tuple[float, float]:
        boot = self._bootstrap_distribution(data, statistic)
        alpha = 1 - confidence
        lo = np.percentile(boot, 100 * alpha / 2)
        hi = np.percentile(boot, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    def _ci_bca(
        self,
        data: np.ndarray,
        statistic: Callable,
        confidence: float,
    ) -> Tuple[float, float]:
        boot = self._bootstrap_distribution(data, statistic)
        theta_hat = statistic(data)
        alpha = 1 - confidence

        # bias correction
        z0 = stats.norm.ppf(np.mean(boot < theta_hat))

        # acceleration (jackknife)
        n = len(data)
        jack = np.empty(n)
        for i in range(n):
            sample = np.concatenate([data[:i], data[i + 1:]])
            jack[i] = statistic(sample)
        jack_mean = jack.mean()
        num = np.sum((jack_mean - jack) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5
        a = num / den if den != 0 else 0.0

        # adjusted percentiles
        z_alpha_lo = stats.norm.ppf(alpha / 2)
        z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

        def _adj(z_a):
            num_ = z0 + z_a
            denom = 1 - a * num_
            if denom == 0:
                return z_a
            return stats.norm.cdf(z0 + num_ / denom)

        p_lo = _adj(z_alpha_lo)
        p_hi = _adj(z_alpha_hi)

        p_lo = np.clip(p_lo, 0.001, 0.999)
        p_hi = np.clip(p_hi, 0.001, 0.999)

        lo = np.percentile(boot, 100 * p_lo)
        hi = np.percentile(boot, 100 * p_hi)
        return (float(lo), float(hi))

    def _ci_basic(
        self,
        data: np.ndarray,
        statistic: Callable,
        confidence: float,
    ) -> Tuple[float, float]:
        boot = self._bootstrap_distribution(data, statistic)
        theta_hat = statistic(data)
        alpha = 1 - confidence
        lo_boot = np.percentile(boot, 100 * (1 - alpha / 2))
        hi_boot = np.percentile(boot, 100 * alpha / 2)
        lo = 2 * theta_hat - lo_boot
        hi = 2 * theta_hat - hi_boot
        return (float(lo), float(hi))

    # ---- paired bootstrap test ----

    def paired_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        statistic: Callable[[np.ndarray], float] = np.mean,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> TestResult:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        observed = statistic(d)
        boot = self._bootstrap_distribution(d, statistic)

        alt = Alternative(alternative)
        if alt == Alternative.TWO_SIDED:
            p = 2.0 * min(np.mean(boot >= observed), np.mean(boot <= observed))
        elif alt == Alternative.GREATER:
            p = float(np.mean(boot <= 0))
        else:
            p = float(np.mean(boot >= 0))
        p = float(np.clip(p, 0, 1))

        ci = self._ci_percentile(d, statistic, 1 - alpha)

        return TestResult(
            test_name="Paired bootstrap",
            statistic=float(observed),
            p_value=p,
            confidence_interval=ci,
            alpha=alpha,
            additional={"n_bootstrap": self.n_bootstrap, "alternative": alternative},
        )

    # ---- hypothesis test ----

    def hypothesis_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        statistic: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: float(
            np.mean(a) - np.mean(b)
        ),
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> TestResult:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        observed = statistic(x, y)
        pooled = np.concatenate([x, y])
        nx = len(x)
        count = 0
        for _ in range(self.n_bootstrap):
            perm = self.rng.permutation(pooled)
            bx, by = perm[:nx], perm[nx:]
            bs = statistic(bx, by)
            if alternative == "two-sided":
                if abs(bs) >= abs(observed):
                    count += 1
            elif alternative == "greater":
                if bs >= observed:
                    count += 1
            else:
                if bs <= observed:
                    count += 1
        p = (count + 1) / (self.n_bootstrap + 1)

        return TestResult(
            test_name="Bootstrap hypothesis",
            statistic=float(observed),
            p_value=float(p),
            alpha=alpha,
            additional={"n_bootstrap": self.n_bootstrap, "alternative": alternative},
        )

    # ---- stratified bootstrap ----

    def stratified_bootstrap(
        self,
        data: np.ndarray,
        groups: np.ndarray,
        statistic: Callable[[np.ndarray], float] = np.mean,
        confidence: float = 0.95,
    ) -> Tuple[float, Tuple[float, float]]:
        data = np.asarray(data, dtype=float)
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)

        boot_stats = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            sample = []
            for g in unique_groups:
                g_data = data[groups == g]
                idx = self.rng.randint(0, len(g_data), size=len(g_data))
                sample.append(g_data[idx])
            combined = np.concatenate(sample)
            boot_stats[b] = statistic(combined)

        alpha = 1 - confidence
        ci = (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
        return float(statistic(data)), ci

    # ---- internals ----

    def _bootstrap_distribution(
        self, data: np.ndarray, statistic: Callable
    ) -> np.ndarray:
        n = len(data)
        boot = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            idx = self.rng.randint(0, n, size=n)
            boot[b] = statistic(data[idx])
        return boot


# ---------------------------------------------------------------------------
# 5. Permutation Test
# ---------------------------------------------------------------------------

class PermutationTest:
    """
    Exact and Monte Carlo permutation tests.

    Supports paired and unpaired designs and multiple test statistics.
    """

    BUILTIN_STATS = {
        "mean_diff": lambda a, b: float(np.mean(a) - np.mean(b)),
        "median_diff": lambda a, b: float(np.median(a) - np.median(b)),
        "ks": lambda a, b: float(stats.ks_2samp(a, b).statistic),
    }

    def __init__(
        self,
        n_permutations: int = 10_000,
        random_state: Optional[int] = None,
    ):
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(random_state)

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        paired: bool = False,
        statistic: Union[str, Callable] = "mean_diff",
        alternative: str = "two-sided",
        exact: bool = False,
        alpha: float = 0.05,
    ) -> TestResult:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        stat_fn = self._resolve_stat(statistic)
        alt = Alternative(alternative)

        if paired:
            return self._paired_test(x, y, stat_fn, alt, exact, alpha)
        else:
            return self._unpaired_test(x, y, stat_fn, alt, exact, alpha)

    # ----- paired -----

    def _paired_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        stat_fn: Callable,
        alt: Alternative,
        exact: bool,
        alpha: float,
    ) -> TestResult:
        if len(x) != len(y):
            raise ValueError("Paired test requires equal-length arrays")
        n = len(x)
        d = x - y
        observed = float(np.mean(d))

        if exact and n <= 20:
            p = self._exact_paired(d, observed, alt)
        else:
            p = self._mc_paired(d, observed, alt)

        return TestResult(
            test_name="Permutation (paired)",
            statistic=observed,
            p_value=float(p),
            alpha=alpha,
            additional={
                "n_permutations": self.n_permutations if not exact else 2 ** n,
                "exact": exact and n <= 20,
                "alternative": alt.value,
            },
        )

    def _exact_paired(
        self, d: np.ndarray, observed: float, alt: Alternative
    ) -> float:
        n = len(d)
        total = 1 << n
        count = 0
        for mask in range(total):
            signs = np.array(
                [1 if (mask >> j) & 1 else -1 for j in range(n)], dtype=float
            )
            val = float(np.mean(signs * d))
            if self._compare(val, observed, alt):
                count += 1
        return count / total

    def _mc_paired(
        self, d: np.ndarray, observed: float, alt: Alternative
    ) -> float:
        n = len(d)
        count = 0
        for _ in range(self.n_permutations):
            signs = self.rng.choice([-1.0, 1.0], size=n)
            val = float(np.mean(signs * d))
            if self._compare(val, observed, alt):
                count += 1
        return (count + 1) / (self.n_permutations + 1)

    # ----- unpaired -----

    def _unpaired_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        stat_fn: Callable,
        alt: Alternative,
        exact: bool,
        alpha: float,
    ) -> TestResult:
        observed = stat_fn(x, y)
        nx = len(x)
        pooled = np.concatenate([x, y])

        if exact and len(pooled) <= 20:
            p = self._exact_unpaired(pooled, nx, stat_fn, observed, alt)
        else:
            p = self._mc_unpaired(pooled, nx, stat_fn, observed, alt)

        return TestResult(
            test_name="Permutation (unpaired)",
            statistic=float(observed),
            p_value=float(p),
            alpha=alpha,
            additional={
                "n_permutations": self.n_permutations,
                "exact": exact and len(pooled) <= 20,
                "alternative": alt.value,
            },
        )

    def _exact_unpaired(
        self,
        pooled: np.ndarray,
        nx: int,
        stat_fn: Callable,
        observed: float,
        alt: Alternative,
    ) -> float:
        from itertools import combinations as _comb
        n = len(pooled)
        count = 0
        total = 0
        for idx in _comb(range(n), nx):
            idx_set = set(idx)
            bx = pooled[list(idx)]
            by = pooled[[i for i in range(n) if i not in idx_set]]
            val = stat_fn(bx, by)
            if self._compare(val, observed, alt):
                count += 1
            total += 1
        return count / total

    def _mc_unpaired(
        self,
        pooled: np.ndarray,
        nx: int,
        stat_fn: Callable,
        observed: float,
        alt: Alternative,
    ) -> float:
        count = 0
        for _ in range(self.n_permutations):
            perm = self.rng.permutation(pooled)
            bx, by = perm[:nx], perm[nx:]
            val = stat_fn(bx, by)
            if self._compare(val, observed, alt):
                count += 1
        return (count + 1) / (self.n_permutations + 1)

    # ----- helpers -----

    @staticmethod
    def _compare(val: float, observed: float, alt: Alternative) -> bool:
        if alt == Alternative.TWO_SIDED:
            return abs(val) >= abs(observed)
        elif alt == Alternative.GREATER:
            return val >= observed
        else:
            return val <= observed

    def _resolve_stat(
        self, statistic: Union[str, Callable]
    ) -> Callable:
        if callable(statistic):
            return statistic
        if statistic in self.BUILTIN_STATS:
            return self.BUILTIN_STATS[statistic]
        raise ValueError(f"Unknown statistic {statistic!r}")


# ---------------------------------------------------------------------------
# 6. Multiple Comparison Corrections
# ---------------------------------------------------------------------------

class BonferroniCorrection:
    """Bonferroni correction: p_adj = min(p * m, 1)."""

    @staticmethod
    def correct(
        p_values: Sequence[float], alpha: float = 0.05
    ) -> MultipleComparisonResult:
        p = np.asarray(p_values, dtype=float)
        m = len(p)
        adjusted = np.minimum(p * m, 1.0)
        rejected = adjusted < alpha
        return MultipleComparisonResult(
            method="Bonferroni",
            original_p_values=p.tolist(),
            adjusted_p_values=adjusted.tolist(),
            rejected=rejected.tolist(),
            alpha=alpha,
        )


class HolmCorrection:
    """
    Holm step-down correction (Holm 1979).

    Sorts p-values, adjusts sequentially, and maintains monotonicity.
    """

    @staticmethod
    def correct(
        p_values: Sequence[float], alpha: float = 0.05
    ) -> MultipleComparisonResult:
        p = np.asarray(p_values, dtype=float)
        m = len(p)
        order = np.argsort(p)
        sorted_p = p[order]

        adjusted_sorted = np.empty(m)
        for i in range(m):
            adjusted_sorted[i] = sorted_p[i] * (m - i)

        # enforce monotonicity (cumulative max)
        for i in range(1, m):
            adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        # unsort
        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted

        rejected = adjusted < alpha
        return MultipleComparisonResult(
            method="Holm",
            original_p_values=p.tolist(),
            adjusted_p_values=adjusted.tolist(),
            rejected=rejected.tolist(),
            alpha=alpha,
        )


class BenjaminiHochberg:
    """
    Benjamini-Hochberg procedure for controlling the False Discovery Rate.
    """

    @staticmethod
    def correct(
        p_values: Sequence[float], alpha: float = 0.05
    ) -> MultipleComparisonResult:
        p = np.asarray(p_values, dtype=float)
        m = len(p)
        order = np.argsort(p)
        sorted_p = p[order]

        adjusted_sorted = np.empty(m)
        for i in range(m):
            rank = i + 1
            adjusted_sorted[i] = sorted_p[i] * m / rank

        # enforce monotonicity (cumulative min from the right)
        for i in range(m - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted

        rejected = adjusted < alpha
        return MultipleComparisonResult(
            method="Benjamini-Hochberg",
            original_p_values=p.tolist(),
            adjusted_p_values=adjusted.tolist(),
            rejected=rejected.tolist(),
            alpha=alpha,
        )


class HolmBonferroni(HolmCorrection):
    """Alias for Holm correction (sometimes called Holm-Bonferroni)."""
    pass


class BenjaminiYekutieli:
    """
    Benjamini-Yekutieli procedure.

    Controls FDR under arbitrary dependence between tests.
    """

    @staticmethod
    def correct(
        p_values: Sequence[float], alpha: float = 0.05
    ) -> MultipleComparisonResult:
        p = np.asarray(p_values, dtype=float)
        m = len(p)
        c_m = sum(1.0 / i for i in range(1, m + 1))
        order = np.argsort(p)
        sorted_p = p[order]

        adjusted_sorted = np.empty(m)
        for i in range(m):
            rank = i + 1
            adjusted_sorted[i] = sorted_p[i] * m * c_m / rank

        for i in range(m - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        adjusted = np.empty(m)
        adjusted[order] = adjusted_sorted

        rejected = adjusted < alpha
        return MultipleComparisonResult(
            method="Benjamini-Yekutieli",
            original_p_values=p.tolist(),
            adjusted_p_values=adjusted.tolist(),
            rejected=rejected.tolist(),
            alpha=alpha,
        )


# ---------------------------------------------------------------------------
# 7. Effect Size Calculators
# ---------------------------------------------------------------------------

class EffectSizeCalculators:
    """
    Collection of effect-size measures with interpretation helpers.
    """

    # ---- Cohen's d ----

    @staticmethod
    def cohens_d(
        x: np.ndarray,
        y: np.ndarray,
        pooled: bool = True,
    ) -> Tuple[float, str]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        mean_diff = float(np.mean(x) - np.mean(y))

        if pooled:
            sp = math.sqrt(
                ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
                / (nx + ny - 2)
            )
        else:
            sp = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0)

        d = mean_diff / sp if sp != 0 else 0.0
        return d, EffectSizeCalculators._interpret_d(d)

    # ---- Hedges' g (bias-corrected Cohen's d) ----

    @staticmethod
    def hedges_g(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        d, _ = EffectSizeCalculators.cohens_d(x, y, pooled=True)
        nx, ny = len(x), len(y)
        df = nx + ny - 2
        # correction factor J
        if df > 0:
            j = 1 - 3.0 / (4.0 * df - 1)
        else:
            j = 1.0
        g = d * j
        return g, EffectSizeCalculators._interpret_d(g)

    # ---- Glass's delta ----

    @staticmethod
    def glass_delta(
        x: np.ndarray,
        y: np.ndarray,
        control: str = "y",
    ) -> Tuple[float, str]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mean_diff = float(np.mean(x) - np.mean(y))
        if control == "y":
            sd = float(np.std(y, ddof=1))
        else:
            sd = float(np.std(x, ddof=1))
        delta = mean_diff / sd if sd != 0 else 0.0
        return delta, EffectSizeCalculators._interpret_d(delta)

    # ---- Cliff's delta (non-parametric) ----

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """
        Cliff's delta = (|{xi > yj}| - |{xi < yj}|) / (nx * ny).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.0, EffectMagnitude.NEGLIGIBLE.value
        more = 0
        less = 0
        for xi in x:
            for yj in y:
                if xi > yj:
                    more += 1
                elif xi < yj:
                    less += 1
        delta = (more - less) / (nx * ny)
        return float(delta), EffectSizeCalculators._interpret_cliff(delta)

    @staticmethod
    def cliffs_delta_fast(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """Vectorised Cliff's delta using broadcasting."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.0, EffectMagnitude.NEGLIGIBLE.value
        diffs = x[:, None] - y[None, :]
        more = int(np.sum(diffs > 0))
        less = int(np.sum(diffs < 0))
        delta = (more - less) / (nx * ny)
        return float(delta), EffectSizeCalculators._interpret_cliff(delta)

    # ---- Common language effect size (CLES) ----

    @staticmethod
    def common_language(x: np.ndarray, y: np.ndarray) -> float:
        """Probability that a random x exceeds a random y."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.5
        count = 0
        ties = 0
        for xi in x:
            for yj in y:
                if xi > yj:
                    count += 1
                elif xi == yj:
                    ties += 1
        return (count + 0.5 * ties) / (nx * ny)

    @staticmethod
    def common_language_fast(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.5
        diffs = x[:, None] - y[None, :]
        count = float(np.sum(diffs > 0))
        ties = float(np.sum(diffs == 0))
        return (count + 0.5 * ties) / (nx * ny)

    # ---- Rank-biserial correlation ----

    @staticmethod
    def rank_biserial(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """Rank-biserial correlation for Mann-Whitney U."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
        r = 1 - 2 * u_stat / (nx * ny)
        return float(r), EffectSizeCalculators._interpret_r(r)

    # ---- Eta-squared for Kruskal-Wallis / Friedman ----

    @staticmethod
    def eta_squared_kw(H: float, n: int, k: int) -> float:
        """Eta-squared from Kruskal-Wallis H statistic."""
        return (H - k + 1) / (n - k)

    @staticmethod
    def kendall_w(chi2: float, n: int, k: int) -> float:
        """Kendall's W from Friedman chi-square."""
        return chi2 / (n * (k - 1))

    # ---- interpretation helpers ----

    @staticmethod
    def _interpret_d(d: float) -> str:
        d = abs(d)
        if d < 0.2:
            return EffectMagnitude.NEGLIGIBLE.value
        if d < 0.5:
            return EffectMagnitude.SMALL.value
        if d < 0.8:
            return EffectMagnitude.MEDIUM.value
        return EffectMagnitude.LARGE.value

    @staticmethod
    def _interpret_cliff(delta: float) -> str:
        d = abs(delta)
        if d < 0.147:
            return EffectMagnitude.NEGLIGIBLE.value
        if d < 0.33:
            return EffectMagnitude.SMALL.value
        if d < 0.474:
            return EffectMagnitude.MEDIUM.value
        return EffectMagnitude.LARGE.value

    @staticmethod
    def _interpret_r(r: float) -> str:
        r = abs(r)
        if r < 0.1:
            return EffectMagnitude.NEGLIGIBLE.value
        if r < 0.3:
            return EffectMagnitude.SMALL.value
        if r < 0.5:
            return EffectMagnitude.MEDIUM.value
        return EffectMagnitude.LARGE.value

    # ---- summary helper ----

    @classmethod
    def compute_all(
        cls,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d_val, d_mag = cls.cohens_d(x, y)
        g_val, g_mag = cls.hedges_g(x, y)
        glass_val, glass_mag = cls.glass_delta(x, y)
        cliff_val, cliff_mag = cls.cliffs_delta_fast(x, y)
        cles = cls.common_language_fast(x, y)
        rb_val, rb_mag = cls.rank_biserial(x, y)
        return {
            "cohens_d": {"value": d_val, "magnitude": d_mag},
            "hedges_g": {"value": g_val, "magnitude": g_mag},
            "glass_delta": {"value": glass_val, "magnitude": glass_mag},
            "cliffs_delta": {"value": cliff_val, "magnitude": cliff_mag},
            "common_language": cles,
            "rank_biserial": {"value": rb_val, "magnitude": rb_mag},
        }


# ---------------------------------------------------------------------------
# 8. Confidence Interval Computer
# ---------------------------------------------------------------------------

class ConfidenceIntervalComputer:
    """
    Compute confidence intervals via several methods.
    """

    # ---- Normal CI ----

    @staticmethod
    def normal_ci(
        data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        data = np.asarray(data, dtype=float)
        n = len(data)
        mean = float(np.mean(data))
        se = float(np.std(data, ddof=1) / math.sqrt(n))
        z = stats.norm.ppf((1 + confidence) / 2)
        return (mean - z * se, mean + z * se)

    # ---- t-distribution CI ----

    @staticmethod
    def t_ci(
        data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        data = np.asarray(data, dtype=float)
        n = len(data)
        mean = float(np.mean(data))
        se = float(np.std(data, ddof=1) / math.sqrt(n))
        t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        return (mean - t_val * se, mean + t_val * se)

    # ---- Bootstrap CI (delegates) ----

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float] = np.mean,
        confidence: float = 0.95,
        method: str = "percentile",
        n_bootstrap: int = 10_000,
        random_state: Optional[int] = None,
    ) -> Tuple[float, float]:
        bt = BootstrapTest(n_bootstrap=n_bootstrap, random_state=random_state)
        return bt.confidence_interval(data, statistic, confidence, method)

    # ---- Wilson score interval (proportions) ----

    @staticmethod
    def wilson_score(
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        if trials == 0:
            return (0.0, 1.0)
        p_hat = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)
        z2 = z * z
        denom = 1 + z2 / trials
        centre = p_hat + z2 / (2 * trials)
        margin = z * math.sqrt(
            (p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials
        )
        lo = (centre - margin) / denom
        hi = (centre + margin) / denom
        return (max(lo, 0.0), min(hi, 1.0))

    # ---- Agresti-Coull interval ----

    @staticmethod
    def agresti_coull(
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        z = stats.norm.ppf((1 + confidence) / 2)
        n_tilde = trials + z * z
        p_tilde = (successes + z * z / 2) / n_tilde
        margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        lo = max(p_tilde - margin, 0.0)
        hi = min(p_tilde + margin, 1.0)
        return (lo, hi)

    # ---- Clopper-Pearson (exact) interval ----

    @staticmethod
    def clopper_pearson(
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        alpha = 1 - confidence
        if successes == 0:
            lo = 0.0
        else:
            lo = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
        if successes == trials:
            hi = 1.0
        else:
            hi = stats.beta.ppf(
                1 - alpha / 2, successes + 1, trials - successes
            )
        return (float(lo), float(hi))

    # ---- Difference of means CI ----

    @staticmethod
    def diff_means_ci(
        x: np.ndarray,
        y: np.ndarray,
        confidence: float = 0.95,
        equal_var: bool = False,
    ) -> Tuple[float, Tuple[float, float]]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        diff = float(np.mean(x) - np.mean(y))
        nx, ny = len(x), len(y)
        var_x = float(np.var(x, ddof=1))
        var_y = float(np.var(y, ddof=1))

        if equal_var:
            sp2 = ((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)
            se = math.sqrt(sp2 * (1.0 / nx + 1.0 / ny))
            df = nx + ny - 2
        else:
            se = math.sqrt(var_x / nx + var_y / ny)
            num = (var_x / nx + var_y / ny) ** 2
            den = (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
            df = num / den if den > 0 else nx + ny - 2

        t_val = stats.t.ppf((1 + confidence) / 2, df)
        ci = (diff - t_val * se, diff + t_val * se)
        return diff, ci

    # ---- Median CI via sign test inversion ----

    @staticmethod
    def median_ci(
        data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        median = float(np.median(data))
        alpha = 1 - confidence
        # use binomial to find order statistics
        j = int(np.floor(n / 2.0 - stats.norm.ppf(1 - alpha / 2) * math.sqrt(n) / 2.0))
        k = int(np.ceil(n / 2.0 + stats.norm.ppf(1 - alpha / 2) * math.sqrt(n) / 2.0))
        j = max(j, 0)
        k = min(k, n - 1)
        return median, (float(data[j]), float(data[k]))


# ---------------------------------------------------------------------------
# 9. Power Analysis
# ---------------------------------------------------------------------------

class PowerAnalysis:
    """
    Sample-size and power computations for common non-parametric tests.
    """

    # ---- Required sample size ----

    @staticmethod
    def required_sample_size(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        test: str = "wilcoxon",
        alternative: str = "two-sided",
    ) -> PowerResult:
        if test == "wilcoxon":
            n = PowerAnalysis._n_wilcoxon(effect_size, alpha, power, alternative)
        elif test in ("t", "t_test", "ttest"):
            n = PowerAnalysis._n_t_test(effect_size, alpha, power, alternative)
        elif test in ("mann_whitney", "mannwhitney"):
            n = PowerAnalysis._n_mann_whitney(effect_size, alpha, power, alternative)
        elif test == "friedman":
            n = PowerAnalysis._n_friedman(effect_size, alpha, power)
        elif test == "chi2":
            n = PowerAnalysis._n_chi2(effect_size, alpha, power)
        else:
            raise ValueError(f"Unknown test {test!r}")

        return PowerResult(
            test_name=test,
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=int(n),
        )

    # ---- Achieved power ----

    @staticmethod
    def achieved_power(
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
        test: str = "wilcoxon",
        alternative: str = "two-sided",
    ) -> PowerResult:
        if test == "wilcoxon":
            pwr = PowerAnalysis._power_wilcoxon(
                effect_size, sample_size, alpha, alternative
            )
        elif test in ("t", "t_test", "ttest"):
            pwr = PowerAnalysis._power_t_test(
                effect_size, sample_size, alpha, alternative
            )
        elif test in ("mann_whitney", "mannwhitney"):
            pwr = PowerAnalysis._power_mann_whitney(
                effect_size, sample_size, alpha, alternative
            )
        elif test == "friedman":
            pwr = PowerAnalysis._power_friedman(
                effect_size, sample_size, alpha
            )
        elif test == "chi2":
            pwr = PowerAnalysis._power_chi2(
                effect_size, sample_size, alpha
            )
        else:
            raise ValueError(f"Unknown test {test!r}")

        return PowerResult(
            test_name=test,
            effect_size=effect_size,
            alpha=alpha,
            power=float(pwr),
            sample_size=sample_size,
        )

    # ---- Power curves ----

    @staticmethod
    def power_curve(
        effect_sizes: Sequence[float],
        sample_sizes: Sequence[int],
        alpha: float = 0.05,
        test: str = "wilcoxon",
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """Return power for each (effect_size, sample_size) combination."""
        results = {}
        for es in effect_sizes:
            powers = []
            for n in sample_sizes:
                pr = PowerAnalysis.achieved_power(
                    effect_size=es,
                    sample_size=n,
                    alpha=alpha,
                    test=test,
                    alternative=alternative,
                )
                powers.append(pr.power)
            results[es] = {
                "sample_sizes": list(sample_sizes),
                "powers": powers,
            }
        return results

    # ---- Minimum detectable effect ----

    @staticmethod
    def minimum_detectable_effect(
        sample_size: int,
        alpha: float = 0.05,
        power: float = 0.80,
        test: str = "t_test",
        alternative: str = "two-sided",
        tol: float = 0.001,
    ) -> float:
        lo, hi = 0.0, 5.0
        while hi - lo > tol:
            mid = (lo + hi) / 2
            pr = PowerAnalysis.achieved_power(
                effect_size=mid,
                sample_size=sample_size,
                alpha=alpha,
                test=test,
                alternative=alternative,
            )
            if pr.power < power:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # ---- internal: Wilcoxon ----

    @staticmethod
    def _n_wilcoxon(
        d: float, alpha: float, power: float, alt: str
    ) -> int:
        # Asymptotic relative efficiency of Wilcoxon vs t is pi/3 ≈ 1.047
        # under normality, so n_w ≈ n_t * pi/3
        n_t = PowerAnalysis._n_t_test(d, alpha, power, alt)
        return int(math.ceil(n_t * math.pi / 3))

    @staticmethod
    def _power_wilcoxon(
        d: float, n: int, alpha: float, alt: str
    ) -> float:
        # approximate via ARE adjustment
        n_eff = n * 3 / math.pi
        return PowerAnalysis._power_t_test(d, int(max(n_eff, 2)), alpha, alt)

    # ---- internal: t-test ----

    @staticmethod
    def _n_t_test(
        d: float, alpha: float, power: float, alt: str
    ) -> int:
        if d == 0:
            return 1_000_000
        if alt == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        n = math.ceil(((z_alpha + z_beta) / d) ** 2)
        return max(n, 2)

    @staticmethod
    def _power_t_test(
        d: float, n: int, alpha: float, alt: str
    ) -> float:
        if n < 2:
            return 0.0
        df = n - 1
        ncp = d * math.sqrt(n)
        if alt == "two-sided":
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(
                -t_crit, df, ncp
            )
        else:
            t_crit = stats.t.ppf(1 - alpha, df)
            power = 1 - stats.nct.cdf(t_crit, df, ncp)
        return float(np.clip(power, 0, 1))

    # ---- internal: Mann-Whitney ----

    @staticmethod
    def _n_mann_whitney(
        d: float, alpha: float, power: float, alt: str
    ) -> int:
        n_t = PowerAnalysis._n_t_test(d, alpha, power, alt)
        return int(math.ceil(n_t * math.pi / 3))

    @staticmethod
    def _power_mann_whitney(
        d: float, n: int, alpha: float, alt: str
    ) -> float:
        n_eff = int(max(n * 3 / math.pi, 2))
        return PowerAnalysis._power_t_test(d, n_eff, alpha, alt)

    # ---- internal: Friedman ----

    @staticmethod
    def _n_friedman(w: float, alpha: float, power: float, k: int = 3) -> int:
        """Approximate sample size for Friedman test given Kendall's W."""
        if w <= 0:
            return 1_000_000
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        n = math.ceil(((z_alpha + z_beta) ** 2) / (k * w ** 2))
        return max(n, k + 1)

    @staticmethod
    def _power_friedman(
        w: float, n: int, alpha: float, k: int = 3
    ) -> float:
        chi2_stat = n * (k - 1) * w
        df = k - 1
        crit = stats.chi2.ppf(1 - alpha, df)
        # non-central chi2
        power = 1 - stats.ncx2.cdf(crit, df, chi2_stat)
        return float(np.clip(power, 0, 1))

    # ---- internal: chi-squared ----

    @staticmethod
    def _n_chi2(w: float, alpha: float, power: float, df: int = 1) -> int:
        if w <= 0:
            return 1_000_000
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        n = math.ceil(((z_alpha + z_beta) / w) ** 2 + df)
        return max(n, df + 1)

    @staticmethod
    def _power_chi2(
        w: float, n: int, alpha: float, df: int = 1
    ) -> float:
        ncp = n * w * w
        crit = stats.chi2.ppf(1 - alpha, df)
        power = 1 - stats.ncx2.cdf(crit, df, ncp)
        return float(np.clip(power, 0, 1))


# ---------------------------------------------------------------------------
# 10. Statistical Test Suite
# ---------------------------------------------------------------------------

class StatisticalTestSuite:
    """
    Orchestrate all statistical tests for algorithm comparison.

    Provides automatic test selection, result aggregation, and multiple-
    comparison management.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 10_000,
        n_permutations: int = 10_000,
        correction_method: str = "holm",
        random_state: Optional[int] = None,
    ):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.correction_method = correction_method
        self.random_state = random_state

        self._wilcoxon = WilcoxonSignedRankTest()
        self._friedman = FriedmanTest()
        self._nemenyi = NemenyiPostHoc()
        self._bootstrap = BootstrapTest(
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        self._permutation = PermutationTest(
            n_permutations=n_permutations, random_state=random_state
        )
        self._effect = EffectSizeCalculators()
        self._ci = ConfidenceIntervalComputer()
        self._power = PowerAnalysis()

    # ---- main entry point ----

    def compare_two(
        self,
        x: np.ndarray,
        y: np.ndarray,
        paired: bool = True,
        names: Tuple[str, str] = ("A", "B"),
    ) -> Dict[str, Any]:
        """Full comparison of two algorithms."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        results: Dict[str, Any] = {"names": names, "n": len(x)}

        # normality check
        is_normal = self._check_normality(x, y, paired)
        results["normality"] = is_normal

        # select and run tests
        tests: List[TestResult] = []
        if paired:
            tests.append(self._wilcoxon.test(x, y, alpha=self.alpha))
            tests.append(
                self._bootstrap.paired_test(x, y, alpha=self.alpha)
            )
            tests.append(
                self._permutation.test(
                    x, y, paired=True, alpha=self.alpha
                )
            )
            if is_normal:
                t_stat, t_p = stats.ttest_rel(x, y)
                tests.append(
                    TestResult(
                        test_name="Paired t-test",
                        statistic=float(t_stat),
                        p_value=float(t_p),
                        alpha=self.alpha,
                    )
                )
        else:
            u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            tests.append(
                TestResult(
                    test_name="Mann-Whitney U",
                    statistic=float(u_stat),
                    p_value=float(u_p),
                    alpha=self.alpha,
                )
            )
            tests.append(
                self._permutation.test(
                    x, y, paired=False, alpha=self.alpha
                )
            )
            tests.append(
                self._bootstrap.hypothesis_test(x, y, alpha=self.alpha)
            )
            if is_normal:
                t_stat, t_p = stats.ttest_ind(x, y)
                tests.append(
                    TestResult(
                        test_name="Independent t-test",
                        statistic=float(t_stat),
                        p_value=float(t_p),
                        alpha=self.alpha,
                    )
                )

        results["tests"] = tests

        # effect sizes
        results["effect_sizes"] = self._effect.compute_all(x, y)

        # confidence intervals
        if paired:
            d = x - y
            results["ci_t"] = self._ci.t_ci(d)
            results["ci_bootstrap"] = self._ci.bootstrap_ci(
                d, random_state=self.random_state
            )
        else:
            diff, ci = self._ci.diff_means_ci(x, y)
            results["mean_diff"] = diff
            results["ci_diff_means"] = ci

        # power
        d_val = results["effect_sizes"]["cohens_d"]["value"]
        results["power"] = self._power.achieved_power(
            effect_size=abs(d_val),
            sample_size=len(x),
            alpha=self.alpha,
        )

        return results

    def compare_multiple(
        self,
        data: np.ndarray,
        algorithm_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare k algorithms across n tasks (repeated measures)."""
        data = np.asarray(data, dtype=float)
        n, k = data.shape
        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(k)]

        results: Dict[str, Any] = {
            "n_tasks": n,
            "k_algorithms": k,
            "algorithm_names": algorithm_names,
        }

        # Friedman test
        friedman_result = self._friedman.test(data, algorithm_names, self.alpha)
        results["friedman"] = friedman_result

        # Post-hoc if significant
        if friedman_result.rejected:
            nemenyi = self._nemenyi.test(data, algorithm_names, self.alpha)
            results["nemenyi"] = nemenyi

            # pairwise Wilcoxon with correction
            pairwise_p: List[float] = []
            pairwise_info: List[Dict[str, Any]] = []
            for i, j in combinations(range(k), 2):
                wr = self._wilcoxon.test(data[:, i], data[:, j], alpha=self.alpha)
                pairwise_p.append(wr.p_value)
                pairwise_info.append({
                    "a": algorithm_names[i],
                    "b": algorithm_names[j],
                    "result": wr,
                })

            correction = self._apply_correction(pairwise_p)
            for idx, info in enumerate(pairwise_info):
                info["adjusted_p"] = correction.adjusted_p_values[idx]
                info["significant"] = correction.rejected[idx]

            results["pairwise_wilcoxon"] = pairwise_info
            results["correction"] = correction

        # effect sizes for each pair
        pairwise_effects: Dict[str, Dict[str, Any]] = {}
        for i, j in combinations(range(k), 2):
            key = f"{algorithm_names[i]} vs {algorithm_names[j]}"
            pairwise_effects[key] = self._effect.compute_all(
                data[:, i], data[:, j]
            )
        results["pairwise_effect_sizes"] = pairwise_effects

        # rankings
        results["rankings"] = self._friedman.rank_algorithms(
            data, algorithm_names
        )

        return results

    def run_all_tests(
        self,
        x: np.ndarray,
        y: np.ndarray,
        paired: bool = True,
    ) -> List[TestResult]:
        """Run every available two-sample test and return all results."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        results: List[TestResult] = []

        if paired:
            results.append(self._wilcoxon.test(x, y, alpha=self.alpha))
            results.append(
                self._bootstrap.paired_test(x, y, alpha=self.alpha)
            )
            results.append(
                self._permutation.test(x, y, paired=True, alpha=self.alpha)
            )
            t_s, t_p = stats.ttest_rel(x, y)
            results.append(
                TestResult("Paired t-test", float(t_s), float(t_p), alpha=self.alpha)
            )
            # sign test
            d = x - y
            n_pos = int(np.sum(d > 0))
            n_neg = int(np.sum(d < 0))
            n_total = n_pos + n_neg
            sign_p = float(stats.binom_test(min(n_pos, n_neg), n_total, 0.5)) if n_total > 0 else 1.0
            results.append(
                TestResult("Sign test", float(min(n_pos, n_neg)), sign_p, alpha=self.alpha)
            )
        else:
            u_s, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            results.append(
                TestResult("Mann-Whitney U", float(u_s), float(u_p), alpha=self.alpha)
            )
            results.append(
                self._permutation.test(x, y, paired=False, alpha=self.alpha)
            )
            results.append(
                self._bootstrap.hypothesis_test(x, y, alpha=self.alpha)
            )
            t_s, t_p = stats.ttest_ind(x, y)
            results.append(
                TestResult("Independent t-test", float(t_s), float(t_p), alpha=self.alpha)
            )
            ks_s, ks_p = stats.ks_2samp(x, y)
            results.append(
                TestResult("KS two-sample", float(ks_s), float(ks_p), alpha=self.alpha)
            )

        return results

    # ---- automatic test selection ----

    def auto_select_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        paired: bool = True,
    ) -> TestResult:
        """Choose the most appropriate test based on data properties."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        is_normal = self._check_normality(x, y, paired)

        if paired:
            if n < 20:
                return self._permutation.test(
                    x, y, paired=True, exact=(n <= 15), alpha=self.alpha
                )
            if is_normal:
                t_s, t_p = stats.ttest_rel(x, y)
                return TestResult(
                    "Paired t-test", float(t_s), float(t_p), alpha=self.alpha
                )
            return self._wilcoxon.test(x, y, alpha=self.alpha)
        else:
            if n < 20:
                return self._permutation.test(
                    x, y, paired=False, alpha=self.alpha
                )
            if is_normal:
                t_s, t_p = stats.ttest_ind(x, y)
                return TestResult(
                    "Independent t-test", float(t_s), float(t_p), alpha=self.alpha
                )
            u_s, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            return TestResult(
                "Mann-Whitney U", float(u_s), float(u_p), alpha=self.alpha
            )

    # ---- report generation ----

    def generate_report(
        self,
        comparison: Dict[str, Any],
    ) -> str:
        """Generate a human-readable text report from a comparison result."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("STATISTICAL COMPARISON REPORT")
        lines.append("=" * 70)

        if "names" in comparison:
            lines.append(
                f"Algorithms: {comparison['names'][0]} vs {comparison['names'][1]}"
            )
            lines.append(f"Sample size: {comparison.get('n', '?')}")

        lines.append("")
        lines.append("--- Hypothesis Tests ---")
        if "tests" in comparison:
            for t in comparison["tests"]:
                lines.append(t.summary())
                lines.append("")

        if "effect_sizes" in comparison:
            lines.append("--- Effect Sizes ---")
            for name, val in comparison["effect_sizes"].items():
                if isinstance(val, dict):
                    lines.append(
                        f"  {name}: {val['value']:.4f} ({val['magnitude']})"
                    )
                else:
                    lines.append(f"  {name}: {val:.4f}")
            lines.append("")

        if "power" in comparison:
            pr = comparison["power"]
            lines.append(f"--- Power: {pr.power:.4f} ---")
            lines.append("")

        if "friedman" in comparison:
            fr = comparison["friedman"]
            lines.append("--- Friedman Test ---")
            lines.append(fr.summary())
            lines.append("")

        if "rankings" in comparison:
            lines.append("--- Rankings ---")
            for name, rank in comparison["rankings"]:
                lines.append(f"  {name}: {rank:.3f}")
            lines.append("")

        if "nemenyi" in comparison:
            nem = comparison["nemenyi"]
            lines.append(f"--- Nemenyi Post-Hoc (CD={nem['critical_difference']:.4f}) ---")
            for pw in nem["pairwise"]:
                sig = "*" if pw.significant else ""
                lines.append(
                    f"  {pw.algorithm_a} vs {pw.algorithm_b}: "
                    f"rank_diff={pw.statistic:.3f}, "
                    f"p_adj={pw.adjusted_p_value:.4f} {sig}"
                )
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ---- helpers ----

    def _check_normality(
        self,
        x: np.ndarray,
        y: np.ndarray,
        paired: bool,
    ) -> bool:
        try:
            if paired:
                d = x - y
                if len(d) < 8:
                    return False
                _, p = stats.shapiro(d)
                return p > self.alpha
            else:
                if len(x) < 8 or len(y) < 8:
                    return False
                _, px = stats.shapiro(x)
                _, py = stats.shapiro(y)
                return px > self.alpha and py > self.alpha
        except Exception:
            return False

    def _apply_correction(
        self, p_values: List[float]
    ) -> MultipleComparisonResult:
        methods = {
            "bonferroni": BonferroniCorrection.correct,
            "holm": HolmCorrection.correct,
            "bh": BenjaminiHochberg.correct,
            "benjamini_hochberg": BenjaminiHochberg.correct,
            "by": BenjaminiYekutieli.correct,
        }
        fn = methods.get(self.correction_method, HolmCorrection.correct)
        return fn(p_values, self.alpha)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def summarize_results(results: List[TestResult]) -> str:
    lines = []
    for r in results:
        lines.append(r.summary())
        lines.append("")
    return "\n".join(lines)


def compare_algorithms_pairwise(
    data: np.ndarray,
    algorithm_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    correction: str = "holm",
) -> Dict[str, Any]:
    """
    Convenience function: run pairwise Wilcoxon tests across all algorithm
    pairs with multiple-comparison correction.
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    if algorithm_names is None:
        algorithm_names = [f"alg_{i}" for i in range(k)]

    raw_p: List[float] = []
    pairs: List[Tuple[str, str]] = []
    stats_list: List[float] = []
    wsr = WilcoxonSignedRankTest()

    for i, j in combinations(range(k), 2):
        result = wsr.test(data[:, i], data[:, j], alpha=alpha)
        raw_p.append(result.p_value)
        pairs.append((algorithm_names[i], algorithm_names[j]))
        stats_list.append(result.statistic)

    correction_map = {
        "bonferroni": BonferroniCorrection.correct,
        "holm": HolmCorrection.correct,
        "bh": BenjaminiHochberg.correct,
        "by": BenjaminiYekutieli.correct,
    }
    fn = correction_map.get(correction, HolmCorrection.correct)
    mc = fn(raw_p, alpha)

    pairwise = []
    for idx, (a, b) in enumerate(pairs):
        pairwise.append(PairwiseResult(
            algorithm_a=a,
            algorithm_b=b,
            statistic=stats_list[idx],
            p_value=raw_p[idx],
            adjusted_p_value=mc.adjusted_p_values[idx],
            significant=mc.rejected[idx],
        ))

    return {"pairwise": pairwise, "correction": mc}


def full_statistical_analysis(
    data: np.ndarray,
    algorithm_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-call entry point that runs Friedman, Nemenyi, pairwise Wilcoxon,
    effect sizes, power analysis, and generates a report.
    """
    suite = StatisticalTestSuite(
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    comparison = suite.compare_multiple(data, algorithm_names)
    comparison["report"] = suite.generate_report(comparison)
    return comparison


# ---------------------------------------------------------------------------
# Additional helpers used by the Arena evaluation pipeline
# ---------------------------------------------------------------------------

class NormalityTests:
    """Battery of normality tests for pre-analysis diagnostics."""

    @staticmethod
    def shapiro_wilk(data: np.ndarray) -> TestResult:
        data = np.asarray(data, dtype=float)
        if len(data) < 3:
            return TestResult("Shapiro-Wilk", 0.0, 1.0)
        stat, p = stats.shapiro(data)
        return TestResult("Shapiro-Wilk", float(stat), float(p))

    @staticmethod
    def dagostino_pearson(data: np.ndarray) -> TestResult:
        data = np.asarray(data, dtype=float)
        if len(data) < 20:
            return TestResult("D'Agostino-Pearson", 0.0, 1.0,
                              additional={"warning": "n < 20, test unreliable"})
        stat, p = stats.normaltest(data)
        return TestResult("D'Agostino-Pearson", float(stat), float(p))

    @staticmethod
    def anderson_darling(data: np.ndarray) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float)
        result = stats.anderson(data, dist="norm")
        return {
            "statistic": float(result.statistic),
            "critical_values": list(result.critical_values),
            "significance_levels": list(result.significance_level),
        }

    @staticmethod
    def jarque_bera(data: np.ndarray) -> TestResult:
        data = np.asarray(data, dtype=float)
        stat, p = stats.jarque_bera(data)
        return TestResult("Jarque-Bera", float(stat), float(p))

    @staticmethod
    def lilliefors(data: np.ndarray) -> TestResult:
        """Lilliefors test (KS test with estimated parameters)."""
        data = np.asarray(data, dtype=float)
        n = len(data)
        mean, std = np.mean(data), np.std(data, ddof=1)
        if std == 0:
            return TestResult("Lilliefors", 0.0, 1.0)
        z = (data - mean) / std
        stat, _ = stats.kstest(z, "norm")
        # approximate p-value via Dallal-Wilkinson (1986)
        if n <= 100:
            p_approx = math.exp(-7.01256 * stat ** 2 * (n + 2.78019)
                                + 2.99587 * stat * math.sqrt(n + 2.78019)
                                - 0.122119
                                + 0.974598 / math.sqrt(n)
                                + 1.67997 / n)
        else:
            p_approx = math.exp(-7.01256 * stat ** 2 * (n + 2.78019))
        p_approx = min(max(p_approx, 0.0), 1.0)
        return TestResult("Lilliefors", float(stat), float(p_approx))

    @classmethod
    def run_all(cls, data: np.ndarray) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float)
        results: Dict[str, Any] = {}
        results["shapiro_wilk"] = cls.shapiro_wilk(data)
        results["dagostino_pearson"] = cls.dagostino_pearson(data)
        results["anderson_darling"] = cls.anderson_darling(data)
        results["jarque_bera"] = cls.jarque_bera(data)
        results["lilliefors"] = cls.lilliefors(data)
        is_normal = results["shapiro_wilk"].p_value > 0.05
        results["is_normal"] = is_normal
        return results


class HomogeneityTests:
    """Tests for homogeneity of variance."""

    @staticmethod
    def levene(*groups: np.ndarray, center: str = "median") -> TestResult:
        stat, p = stats.levene(*groups, center=center)
        return TestResult("Levene", float(stat), float(p))

    @staticmethod
    def bartlett(*groups: np.ndarray) -> TestResult:
        stat, p = stats.bartlett(*groups)
        return TestResult("Bartlett", float(stat), float(p))

    @staticmethod
    def fligner_killeen(*groups: np.ndarray) -> TestResult:
        stat, p = stats.fligner(*groups)
        return TestResult("Fligner-Killeen", float(stat), float(p))

    @classmethod
    def run_all(cls, *groups: np.ndarray) -> Dict[str, TestResult]:
        return {
            "levene": cls.levene(*groups),
            "bartlett": cls.bartlett(*groups),
            "fligner_killeen": cls.fligner_killeen(*groups),
        }


class DescriptiveStats:
    """Compute descriptive statistics for algorithm score arrays."""

    @staticmethod
    def compute(data: np.ndarray) -> Dict[str, float]:
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)) if n > 1 else 0.0,
            "min": float(np.min(data)),
            "q1": float(np.percentile(data, 25)),
            "median": float(np.median(data)),
            "q3": float(np.percentile(data, 75)),
            "max": float(np.max(data)),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
            "skewness": float(stats.skew(data)) if n > 2 else 0.0,
            "kurtosis": float(stats.kurtosis(data)) if n > 3 else 0.0,
            "se_mean": float(np.std(data, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "cv": float(np.std(data, ddof=1) / abs(np.mean(data)))
            if n > 1 and np.mean(data) != 0
            else 0.0,
        }

    @staticmethod
    def compare_descriptives(
        groups: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        return {
            name: DescriptiveStats.compute(arr) for name, arr in groups.items()
        }


class OutlierDetection:
    """Identify outliers using several methods."""

    @staticmethod
    def iqr_method(
        data: np.ndarray, factor: float = 1.5
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float)
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (data < lower) | (data > upper)
        return {
            "outlier_indices": np.where(mask)[0].tolist(),
            "outlier_values": data[mask].tolist(),
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": int(np.sum(mask)),
        }

    @staticmethod
    def z_score_method(
        data: np.ndarray, threshold: float = 3.0
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return {
                "outlier_indices": [],
                "outlier_values": [],
                "n_outliers": 0,
            }
        z = np.abs((data - mean) / std)
        mask = z > threshold
        return {
            "outlier_indices": np.where(mask)[0].tolist(),
            "outlier_values": data[mask].tolist(),
            "z_scores": z.tolist(),
            "n_outliers": int(np.sum(mask)),
        }

    @staticmethod
    def mad_method(
        data: np.ndarray, threshold: float = 3.5
    ) -> Dict[str, Any]:
        """Median Absolute Deviation method."""
        data = np.asarray(data, dtype=float)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return {
                "outlier_indices": [],
                "outlier_values": [],
                "n_outliers": 0,
            }
        modified_z = 0.6745 * (data - median) / mad
        mask = np.abs(modified_z) > threshold
        return {
            "outlier_indices": np.where(mask)[0].tolist(),
            "outlier_values": data[mask].tolist(),
            "modified_z_scores": modified_z.tolist(),
            "n_outliers": int(np.sum(mask)),
        }

    @staticmethod
    def grubbs_test(
        data: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 3:
            return {"outlier": None, "statistic": 0.0, "critical": 0.0}
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return {"outlier": None, "statistic": 0.0, "critical": 0.0}

        abs_dev = np.abs(data - mean)
        max_idx = int(np.argmax(abs_dev))
        G = abs_dev[max_idx] / std

        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(
            t_crit ** 2 / (n - 2 + t_crit ** 2)
        )

        is_outlier = G > G_crit
        return {
            "outlier": float(data[max_idx]) if is_outlier else None,
            "outlier_index": int(max_idx) if is_outlier else None,
            "statistic": float(G),
            "critical": float(G_crit),
            "is_outlier": is_outlier,
        }


class CorrelationAnalysis:
    """Correlation measures between algorithm score vectors."""

    @staticmethod
    def pearson(x: np.ndarray, y: np.ndarray) -> TestResult:
        r, p = stats.pearsonr(x, y)
        return TestResult("Pearson r", float(r), float(p))

    @staticmethod
    def spearman(x: np.ndarray, y: np.ndarray) -> TestResult:
        r, p = stats.spearmanr(x, y)
        return TestResult("Spearman rho", float(r), float(p))

    @staticmethod
    def kendall(x: np.ndarray, y: np.ndarray) -> TestResult:
        tau, p = stats.kendalltau(x, y)
        return TestResult("Kendall tau", float(tau), float(p))

    @staticmethod
    def concordance_correlation(
        x: np.ndarray, y: np.ndarray
    ) -> float:
        """Lin's concordance correlation coefficient."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
        r, _ = stats.pearsonr(x, y)
        return 2 * r * sx * sy / (sx ** 2 + sy ** 2 + (mx - my) ** 2)

    @classmethod
    def run_all(
        cls, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return {
            "pearson": cls.pearson(x, y),
            "spearman": cls.spearman(x, y),
            "kendall": cls.kendall(x, y),
            "concordance": cls.concordance_correlation(x, y),
        }


class RankingAnalysis:
    """Analyse and compare rankings of algorithms."""

    @staticmethod
    def compute_ranks(
        data: np.ndarray,
        higher_is_better: bool = True,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if not higher_is_better:
            data = -data
        n, k = data.shape
        ranks = np.zeros_like(data)
        for i in range(n):
            ranks[i] = stats.rankdata(-data[i])
        return ranks

    @staticmethod
    def mean_reciprocal_rank(ranks: np.ndarray) -> np.ndarray:
        """Mean reciprocal rank for each algorithm column."""
        return np.mean(1.0 / ranks, axis=0)

    @staticmethod
    def rank_stability(ranks: np.ndarray) -> np.ndarray:
        """Standard deviation of ranks for each algorithm (lower = more stable)."""
        return np.std(ranks, axis=0, ddof=1)

    @staticmethod
    def dominance_matrix(data: np.ndarray) -> np.ndarray:
        """D[i, j] = fraction of tasks where algorithm i beats algorithm j."""
        data = np.asarray(data, dtype=float)
        n, k = data.shape
        D = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    D[i, j] = np.mean(data[:, i] > data[:, j])
        return D

    @staticmethod
    def borda_count(ranks: np.ndarray) -> np.ndarray:
        """Borda count: sum of (k - rank) across tasks."""
        k = ranks.shape[1]
        return np.sum(k - ranks, axis=0)

    @staticmethod
    def copeland_score(data: np.ndarray) -> np.ndarray:
        """Copeland score: wins - losses in pairwise majority voting."""
        D = RankingAnalysis.dominance_matrix(data)
        k = D.shape[0]
        scores = np.zeros(k)
        for i in range(k):
            for j in range(k):
                if i != j:
                    if D[i, j] > 0.5:
                        scores[i] += 1
                    elif D[i, j] < 0.5:
                        scores[i] -= 1
        return scores


class ConvergenceAnalysis:
    """Check whether bootstrap or permutation estimates have converged."""

    @staticmethod
    def bootstrap_convergence(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float] = np.mean,
        max_iterations: int = 50_000,
        step: int = 1_000,
        tolerance: float = 0.001,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run bootstrap with increasing B, tracking CI width."""
        rng = np.random.RandomState(random_state)
        data = np.asarray(data, dtype=float)
        n = len(data)

        iterations = []
        ci_widths = []
        means = []
        boot_values: List[float] = []

        for b in range(1, max_iterations + 1):
            idx = rng.randint(0, n, size=n)
            boot_values.append(statistic(data[idx]))

            if b % step == 0:
                arr = np.array(boot_values)
                lo, hi = np.percentile(arr, [2.5, 97.5])
                width = hi - lo
                iterations.append(b)
                ci_widths.append(width)
                means.append(float(np.mean(arr)))

                if len(ci_widths) >= 3:
                    recent = ci_widths[-3:]
                    if max(recent) - min(recent) < tolerance * abs(means[-1] + 1e-12):
                        break

        converged = len(ci_widths) >= 3 and (
            max(ci_widths[-3:]) - min(ci_widths[-3:])
            < tolerance * abs(means[-1] + 1e-12)
        )

        return {
            "converged": converged,
            "iterations": iterations,
            "ci_widths": ci_widths,
            "means": means,
            "final_n_bootstrap": iterations[-1] if iterations else 0,
        }

    @staticmethod
    def permutation_convergence(
        x: np.ndarray,
        y: np.ndarray,
        max_permutations: int = 50_000,
        step: int = 1_000,
        tolerance: float = 0.005,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Track p-value stability as number of permutations increases."""
        rng = np.random.RandomState(random_state)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pooled = np.concatenate([x, y])
        nx = len(x)
        observed = float(np.mean(x) - np.mean(y))

        count = 0
        iterations: List[int] = []
        p_values: List[float] = []

        for i in range(1, max_permutations + 1):
            perm = rng.permutation(pooled)
            diff = float(np.mean(perm[:nx]) - np.mean(perm[nx:]))
            if abs(diff) >= abs(observed):
                count += 1

            if i % step == 0:
                p = (count + 1) / (i + 1)
                iterations.append(i)
                p_values.append(p)

                if len(p_values) >= 3:
                    recent = p_values[-3:]
                    if max(recent) - min(recent) < tolerance:
                        break

        converged = len(p_values) >= 3 and (
            max(p_values[-3:]) - min(p_values[-3:]) < tolerance
        )

        return {
            "converged": converged,
            "iterations": iterations,
            "p_values": p_values,
            "final_p": p_values[-1] if p_values else None,
        }


class BayesianComparison:
    """Simple Bayesian comparison helpers (non-MCMC)."""

    @staticmethod
    def bayesian_sign_test(
        x: np.ndarray,
        y: np.ndarray,
        rope: float = 0.01,
    ) -> Dict[str, float]:
        """
        Bayesian sign test: probability that x > y, x ≈ y, or x < y.

        Parameters
        ----------
        rope : region of practical equivalence half-width
        """
        d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
        n = len(d)
        wins = int(np.sum(d > rope))
        losses = int(np.sum(d < -rope))
        ties = n - wins - losses

        # Dirichlet-multinomial posterior with uniform prior
        alpha_w = wins + 1
        alpha_t = ties + 1
        alpha_l = losses + 1
        total = alpha_w + alpha_t + alpha_l

        return {
            "p_x_better": alpha_w / total,
            "p_equivalent": alpha_t / total,
            "p_y_better": alpha_l / total,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

    @staticmethod
    def bayesian_correlated_t(
        x: np.ndarray,
        y: np.ndarray,
        rope: float = 0.01,
        rho: float = 0.1,
    ) -> Dict[str, float]:
        """
        Bayesian correlated t-test (Corani et al., 2017).
        Approximation using a shifted/scaled t-distribution.
        """
        d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
        n = len(d)
        mean_d = float(np.mean(d))
        var_d = float(np.var(d, ddof=1))
        se = math.sqrt(var_d * (1.0 / n + rho / (1 - rho)))
        df = n - 1

        if se == 0:
            if mean_d > rope:
                return {"p_x_better": 1.0, "p_equivalent": 0.0, "p_y_better": 0.0}
            if mean_d < -rope:
                return {"p_x_better": 0.0, "p_equivalent": 0.0, "p_y_better": 1.0}
            return {"p_x_better": 0.0, "p_equivalent": 1.0, "p_y_better": 0.0}

        p_left = float(stats.t.cdf(-rope, df, loc=mean_d, scale=se))
        p_right = float(stats.t.sf(rope, df, loc=mean_d, scale=se))
        p_rope = 1 - p_left - p_right

        return {
            "p_x_better": p_right,
            "p_equivalent": max(p_rope, 0.0),
            "p_y_better": p_left,
        }


class ReproducibilityChecker:
    """Utilities for checking reproducibility of statistical results."""

    @staticmethod
    def split_half_reliability(
        data: np.ndarray,
        n_splits: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Estimate split-half reliability of algorithm rankings.

        Randomly split tasks into two halves, compute rankings on each, and
        measure their correlation.
        """
        rng = np.random.RandomState(random_state)
        data = np.asarray(data, dtype=float)
        n, k = data.shape

        correlations: List[float] = []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            half = n // 2
            a = data[perm[:half]]
            b = data[perm[half: 2 * half]]

            ranks_a = np.zeros(k)
            ranks_b = np.zeros(k)
            for col in range(k):
                ranks_a[col] = np.mean(a[:, col])
                ranks_b[col] = np.mean(b[:, col])
            rho, _ = stats.spearmanr(ranks_a, ranks_b)
            correlations.append(float(rho))

        corrs = np.array(correlations)
        return {
            "mean_correlation": float(np.mean(corrs)),
            "std_correlation": float(np.std(corrs, ddof=1)),
            "min_correlation": float(np.min(corrs)),
            "max_correlation": float(np.max(corrs)),
            "ci_95": (
                float(np.percentile(corrs, 2.5)),
                float(np.percentile(corrs, 97.5)),
            ),
            "n_splits": n_splits,
        }

    @staticmethod
    def leave_one_out_sensitivity(
        data: np.ndarray,
        algorithm_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Check how sensitive rankings are to removing each task.
        """
        data = np.asarray(data, dtype=float)
        n, k = data.shape
        if algorithm_names is None:
            algorithm_names = [f"alg_{i}" for i in range(k)]

        full_means = data.mean(axis=0)
        full_order = np.argsort(-full_means)

        rank_changes: List[int] = []
        for i in range(n):
            reduced = np.delete(data, i, axis=0)
            red_means = reduced.mean(axis=0)
            red_order = np.argsort(-red_means)
            changes = int(np.sum(full_order != red_order))
            rank_changes.append(changes)

        return {
            "max_rank_changes": int(np.max(rank_changes)),
            "mean_rank_changes": float(np.mean(rank_changes)),
            "sensitive_tasks": [
                int(i) for i, c in enumerate(rank_changes)
                if c > 0
            ],
            "full_ranking": [algorithm_names[i] for i in full_order],
        }


class SampleSizeEstimation:
    """
    Estimate required sample sizes via simulation.
    """

    @staticmethod
    def simulate_power(
        effect_size: float,
        sample_sizes: Sequence[int],
        n_simulations: int = 1_000,
        test: str = "wilcoxon",
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Monte Carlo power simulation: generate data with a given effect,
        run the test, and estimate rejection rate.
        """
        rng = np.random.RandomState(random_state)
        results: Dict[int, float] = {}

        for n in sample_sizes:
            rejections = 0
            for _ in range(n_simulations):
                x = rng.randn(n) + effect_size
                y = rng.randn(n)
                if test == "wilcoxon":
                    try:
                        _, p = stats.wilcoxon(x - y)
                    except ValueError:
                        p = 1.0
                elif test in ("t", "t_test", "ttest"):
                    _, p = stats.ttest_rel(x, y)
                elif test in ("mann_whitney", "mannwhitney"):
                    _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                else:
                    _, p = stats.ttest_rel(x, y)
                if p < alpha:
                    rejections += 1
            results[n] = rejections / n_simulations

        return results

    @staticmethod
    def required_n_by_simulation(
        effect_size: float,
        target_power: float = 0.80,
        test: str = "wilcoxon",
        alpha: float = 0.05,
        n_simulations: int = 500,
        min_n: int = 5,
        max_n: int = 500,
        random_state: Optional[int] = None,
    ) -> int:
        """Binary search for smallest n achieving target power."""
        lo, hi = min_n, max_n
        while lo < hi:
            mid = (lo + hi) // 2
            power_dict = SampleSizeEstimation.simulate_power(
                effect_size=effect_size,
                sample_sizes=[mid],
                n_simulations=n_simulations,
                test=test,
                alpha=alpha,
                random_state=random_state,
            )
            power = power_dict[mid]
            if power >= target_power:
                hi = mid
            else:
                lo = mid + 1
        return lo


class MultipleDatasetAnalysis:
    """
    Analyse algorithm performance across multiple datasets/domains.
    """

    @staticmethod
    def aggregate_rankings(
        datasets: Dict[str, np.ndarray],
        algorithm_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate mean ranks across multiple datasets.
        """
        all_ranks = []
        per_dataset: Dict[str, Dict[str, float]] = {}
        k = None

        for ds_name, data in datasets.items():
            data = np.asarray(data, dtype=float)
            n, k_ = data.shape
            if k is None:
                k = k_
            ranks = np.zeros_like(data)
            for i in range(n):
                ranks[i] = stats.rankdata(data[i])
            mean_r = ranks.mean(axis=0)
            all_ranks.append(mean_r)
            if algorithm_names is None:
                algorithm_names = [f"alg_{j}" for j in range(k_)]
            per_dataset[ds_name] = {
                algorithm_names[j]: float(mean_r[j]) for j in range(k_)
            }

        stacked = np.vstack(all_ranks)
        overall = stacked.mean(axis=0)
        overall_dict = {
            algorithm_names[j]: float(overall[j]) for j in range(k or 0)
        }

        return {
            "per_dataset_ranks": per_dataset,
            "overall_mean_ranks": overall_dict,
            "n_datasets": len(datasets),
        }

    @staticmethod
    def meta_analysis_fixed_effects(
        effect_sizes: Sequence[float],
        standard_errors: Sequence[float],
    ) -> Dict[str, Any]:
        """
        Fixed-effects meta-analysis combining effect sizes.
        """
        es = np.asarray(effect_sizes, dtype=float)
        se = np.asarray(standard_errors, dtype=float)
        w = 1.0 / (se ** 2)
        combined_es = float(np.sum(w * es) / np.sum(w))
        combined_se = float(1.0 / math.sqrt(np.sum(w)))
        z = combined_es / combined_se
        p = 2.0 * stats.norm.sf(abs(z))

        ci = (
            combined_es - 1.96 * combined_se,
            combined_es + 1.96 * combined_se,
        )

        # heterogeneity Q
        Q = float(np.sum(w * (es - combined_es) ** 2))
        k = len(es)
        Q_p = float(stats.chi2.sf(Q, k - 1)) if k > 1 else 1.0
        I2 = max((Q - (k - 1)) / Q * 100, 0.0) if Q > 0 else 0.0

        return {
            "combined_effect": combined_es,
            "combined_se": combined_se,
            "z": float(z),
            "p_value": float(p),
            "ci_95": ci,
            "Q": Q,
            "Q_p_value": Q_p,
            "I_squared": I2,
        }

    @staticmethod
    def meta_analysis_random_effects(
        effect_sizes: Sequence[float],
        standard_errors: Sequence[float],
    ) -> Dict[str, Any]:
        """
        DerSimonian-Laird random-effects meta-analysis.
        """
        es = np.asarray(effect_sizes, dtype=float)
        se = np.asarray(standard_errors, dtype=float)
        k = len(es)
        w = 1.0 / (se ** 2)

        # fixed effects estimate for Q
        mu_fixed = np.sum(w * es) / np.sum(w)
        Q = float(np.sum(w * (es - mu_fixed) ** 2))

        # DL estimate of tau^2
        c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max((Q - (k - 1)) / c, 0.0) if c > 0 else 0.0

        # random effects weights
        w_star = 1.0 / (se ** 2 + tau2)
        mu_re = float(np.sum(w_star * es) / np.sum(w_star))
        se_re = float(1.0 / math.sqrt(np.sum(w_star)))
        z = mu_re / se_re if se_re > 0 else 0.0
        p = 2.0 * stats.norm.sf(abs(z))

        ci = (mu_re - 1.96 * se_re, mu_re + 1.96 * se_re)
        I2 = max((Q - (k - 1)) / Q * 100, 0.0) if Q > 0 else 0.0

        return {
            "combined_effect": mu_re,
            "combined_se": se_re,
            "z": float(z),
            "p_value": float(p),
            "ci_95": ci,
            "tau_squared": float(tau2),
            "Q": Q,
            "I_squared": I2,
        }


# ---------------------------------------------------------------------------
# Convenience aliases for import ergonomics
# ---------------------------------------------------------------------------

wilcoxon_test = WilcoxonSignedRankTest().test
friedman_test = FriedmanTest().test
nemenyi_posthoc = NemenyiPostHoc().test
bootstrap_ci = ConfidenceIntervalComputer.bootstrap_ci
wilson_score = ConfidenceIntervalComputer.wilson_score
cohens_d = EffectSizeCalculators.cohens_d
hedges_g = EffectSizeCalculators.hedges_g
cliffs_delta = EffectSizeCalculators.cliffs_delta_fast
bonferroni = BonferroniCorrection.correct
holm = HolmCorrection.correct
benjamini_hochberg = BenjaminiHochberg.correct


__all__ = [
    # Enums / containers
    "Alternative",
    "EffectMagnitude",
    "TestResult",
    "PairwiseResult",
    "MultipleComparisonResult",
    "PowerResult",
    # Core tests
    "WilcoxonSignedRankTest",
    "FriedmanTest",
    "NemenyiPostHoc",
    "BootstrapTest",
    "PermutationTest",
    # Corrections
    "BonferroniCorrection",
    "HolmCorrection",
    "HolmBonferroni",
    "BenjaminiHochberg",
    "BenjaminiYekutieli",
    # Effect sizes
    "EffectSizeCalculators",
    # CI
    "ConfidenceIntervalComputer",
    # Power
    "PowerAnalysis",
    # Suite
    "StatisticalTestSuite",
    # Diagnostics
    "NormalityTests",
    "HomogeneityTests",
    "DescriptiveStats",
    "OutlierDetection",
    "CorrelationAnalysis",
    "RankingAnalysis",
    "ConvergenceAnalysis",
    "BayesianComparison",
    "ReproducibilityChecker",
    "SampleSizeEstimation",
    "MultipleDatasetAnalysis",
    # Convenience
    "wilcoxon_test",
    "friedman_test",
    "nemenyi_posthoc",
    "bootstrap_ci",
    "wilson_score",
    "cohens_d",
    "hedges_g",
    "cliffs_delta",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "summarize_results",
    "compare_algorithms_pairwise",
    "full_statistical_analysis",
]
