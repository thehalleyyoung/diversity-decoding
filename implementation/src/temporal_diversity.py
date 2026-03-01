"""
Temporal diversity analysis.

Measures how diversity changes over time, detects trends and seasonal
patterns, forecasts future diversity scores, and analyzes novelty decay
across successive outputs.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TemporalScore:
    """Diversity score computed across multiple time periods."""

    overall_score: float
    per_period_scores: Dict[str, float]
    trend_direction: str
    volatility: float
    autocorrelation: float
    periods: List[str]


@dataclass
class TrendReport:
    """Report describing the trend in a diversity time series."""

    direction: str  # "increasing", "decreasing", or "stable"
    slope: float
    confidence: float
    change_points: List[int]
    period_summaries: Dict[str, float]


@dataclass
class Forecast:
    """Forecast of future diversity scores."""

    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    method_used: str
    model_parameters: Dict[str, float]
    residuals: List[float]


@dataclass
class SeasonalReport:
    """Report on seasonal patterns in a diversity time series."""

    has_seasonality: bool
    period: int
    seasonal_component: List[float]
    trend_component: List[float]
    residual_component: List[float]
    strength: float


@dataclass
class DecayCurve:
    """Fitted novelty decay curve."""

    decay_rate: float
    half_life: float
    fitted_params: Dict[str, float]
    novelty_scores: List[float]
    time_points: List[int]
    r_squared: float


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lower-case whitespace tokenisation with basic punctuation removal."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return a list of n-grams from *tokens*."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _distinct_n(texts: List[str], n: int) -> float:
    """Compute distinct-n ratio across a collection of texts."""
    all_ngrams: List[Tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(_ngrams(_tokenize(text), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard_distance(texts: List[str]) -> float:
    """Mean pairwise Jaccard *distance* (1 - similarity) over token sets."""
    if len(texts) < 2:
        return 0.0
    token_sets = [set(_tokenize(t)) for t in texts]
    total = 0.0
    count = 0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            total += 1.0 - _jaccard(token_sets[i], token_sets[j])
            count += 1
    return total / count


def _period_diversity(texts: List[str]) -> float:
    """Aggregate diversity score for a single period."""
    if not texts:
        return 0.0
    d1 = _distinct_n(texts, 1)
    d2 = _distinct_n(texts, 2)
    jac = _mean_pairwise_jaccard_distance(texts)
    return (d1 + d2 + jac) / 3.0


def _linear_regression(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """Ordinary least-squares fit.  Returns (slope, intercept, r_squared)."""
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0, 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return 0.0, float(y_mean), 0.0
    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_sq = 1.0 - float(ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, intercept, r_sq


# ---------------------------------------------------------------------------
# TimeSeriesAnalyzer
# ---------------------------------------------------------------------------


class TimeSeriesAnalyzer:
    """General time-series analysis primitives."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    # -- autocorrelation ----------------------------------------------------

    def autocorrelation(self, max_lag: int | None = None) -> np.ndarray:
        """Compute the autocorrelation function up to *max_lag*.

        Uses the standard unbiased estimator normalised by lag-0 variance.
        """
        n = len(self._values)
        if max_lag is None:
            max_lag = min(n - 1, 40)
        max_lag = min(max_lag, n - 1)
        mean = np.mean(self._values)
        var = np.var(self._values)
        if var == 0:
            return np.zeros(max_lag + 1)
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            cov = np.mean(
                (self._values[: n - lag] - mean)
                * (self._values[lag:] - mean)
            )
            acf[lag] = cov / var
        return acf

    def partial_autocorrelation(
        self, max_lag: int | None = None
    ) -> np.ndarray:
        """Estimate the partial autocorrelation function via Durbin-Levinson."""
        acf = self.autocorrelation(max_lag)
        k = len(acf) - 1
        if k < 1:
            return np.array([1.0])
        pacf = np.zeros(k + 1)
        pacf[0] = 1.0
        pacf[1] = acf[1]
        phi = np.zeros((k + 1, k + 1))
        phi[1, 1] = acf[1]
        for i in range(2, k + 1):
            num = acf[i] - sum(phi[i - 1, j] * acf[i - j] for j in range(1, i))
            denom = 1.0 - sum(phi[i - 1, j] * acf[j] for j in range(1, i))
            if abs(denom) < 1e-12:
                pacf[i] = 0.0
                continue
            phi[i, i] = num / denom
            for j in range(1, i):
                phi[i, j] = phi[i - 1, j] - phi[i, i] * phi[i - 1, i - j]
            pacf[i] = phi[i, i]
        return pacf

    def stationarity_test(self) -> Dict[str, Any]:
        """Approximate augmented Dickey-Fuller test.

        Returns a dict with the test statistic, an approximate p-value, and
        a boolean ``is_stationary`` flag (based on the 5 % critical value for
        a unit-root model with intercept).
        """
        y = self._values
        n = len(y)
        if n < 4:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "is_stationary": False,
            }
        dy = np.diff(y)
        y_lag = y[:-1]
        x = np.column_stack([np.ones(n - 1), y_lag])
        # OLS: dy = a + gamma * y_{t-1}
        xtx_inv = np.linalg.pinv(x.T @ x)
        beta = xtx_inv @ (x.T @ dy)
        residuals = dy - x @ beta
        sigma2 = float(np.sum(residuals ** 2) / (n - 1 - 2))
        se_gamma = float(np.sqrt(max(sigma2 * xtx_inv[1, 1], 1e-30)))
        gamma = float(beta[1])
        t_stat = gamma / se_gamma if se_gamma > 0 else 0.0
        # Approximate p-value using MacKinnon critical values (tau_c, n→∞)
        crit_1 = -3.43
        crit_5 = -2.86
        crit_10 = -2.57
        if t_stat < crit_1:
            p_approx = 0.005
        elif t_stat < crit_5:
            p_approx = 0.03
        elif t_stat < crit_10:
            p_approx = 0.07
        else:
            p_approx = min(1.0, 0.1 + 0.15 * (t_stat - crit_10))
        return {
            "test_statistic": t_stat,
            "p_value": p_approx,
            "is_stationary": t_stat < crit_5,
        }

    def difference(self, order: int = 1) -> np.ndarray:
        """Return the *order*-th difference of the series."""
        result = self._values.copy()
        for _ in range(order):
            result = np.diff(result)
        return result


# ---------------------------------------------------------------------------
# ChangePointDetector
# ---------------------------------------------------------------------------


class ChangePointDetector:
    """Detect change points in a time series."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    # -- CUSUM --------------------------------------------------------------

    def cusum(self, threshold: float = 1.0) -> List[int]:
        """Cumulative sum (CUSUM) change-point detection.

        Computes the standardised CUSUM chart and flags indices where the
        cumulative deviation from the mean exceeds *threshold* standard
        deviations.
        """
        n = len(self._values)
        if n < 3:
            return []
        mean = np.mean(self._values)
        std = np.std(self._values, ddof=1)
        if std < 1e-12:
            return []
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + (self._values[i] - mean) / std

        change_points: List[int] = []
        s_mean = np.mean(s)
        s_std = np.std(s, ddof=1) if n > 2 else 1.0
        if s_std < 1e-12:
            return []
        for i in range(1, n - 1):
            if abs(s[i] - s_mean) > threshold * s_std:
                # keep only if local extremum
                if (s[i] > s[i - 1] and s[i] > s[i + 1]) or (
                    s[i] < s[i - 1] and s[i] < s[i + 1]
                ):
                    change_points.append(i)
        return change_points

    # -- binary segmentation ------------------------------------------------

    def binary_segmentation(
        self, min_segment: int = 3, p_threshold: float = 0.05
    ) -> List[int]:
        """Recursive binary segmentation for change-point detection.

        At each step the series is split at the point that maximises the
        reduction in total residual sum of squares.  Splitting stops when the
        improvement is not significant (F-test) or the segment is too short.
        """
        change_points: List[int] = []
        self._binseg_recurse(
            0, len(self._values), min_segment, p_threshold, change_points
        )
        change_points.sort()
        return change_points

    def _binseg_recurse(
        self,
        start: int,
        end: int,
        min_seg: int,
        p_thresh: float,
        out: List[int],
    ) -> None:
        n = end - start
        if n < 2 * min_seg:
            return
        segment = self._values[start:end]
        overall_mean = np.mean(segment)
        ss_total = float(np.sum((segment - overall_mean) ** 2))
        if ss_total < 1e-12:
            return

        best_idx = -1
        best_ss_split = ss_total
        for k in range(min_seg, n - min_seg + 1):
            left = segment[:k]
            right = segment[k:]
            ss = float(
                np.sum((left - np.mean(left)) ** 2)
                + np.sum((right - np.mean(right)) ** 2)
            )
            if ss < best_ss_split:
                best_ss_split = ss
                best_idx = k

        if best_idx < 0:
            return
        # F-test for significance of the split
        improvement = ss_total - best_ss_split
        df1 = 1
        df2 = n - 2
        if df2 <= 0:
            return
        f_stat = (improvement / df1) / (best_ss_split / df2) if best_ss_split > 0 else 0.0
        # Approximate p-value from F distribution using Snedecor's approx
        p_val = self._f_pvalue_approx(f_stat, df1, df2)
        if p_val < p_thresh:
            cp = start + best_idx
            out.append(cp)
            self._binseg_recurse(start, cp, min_seg, p_thresh, out)
            self._binseg_recurse(cp, end, min_seg, p_thresh, out)

    @staticmethod
    def _f_pvalue_approx(f: float, df1: int, df2: int) -> float:
        """Rough upper-tail p-value for the F distribution."""
        if f <= 0:
            return 1.0
        # Use the beta-incomplete approximation:  x = df2/(df2 + df1*f)
        x = df2 / (df2 + df1 * f)
        # Regularised incomplete beta via simple series (sufficient here)
        a = df2 / 2.0
        b = df1 / 2.0
        p = ChangePointDetector._betainc_approx(a, b, x)
        return max(0.0, min(1.0, p))

    @staticmethod
    def _betainc_approx(a: float, b: float, x: float) -> float:
        """Simple regularised incomplete beta via continued-fraction."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        # Use the power-series for small x
        lnbeta = (
            math.lgamma(a)
            + math.lgamma(b)
            - math.lgamma(a + b)
        )
        front = math.exp(
            a * math.log(x) + b * math.log(1.0 - x) - lnbeta
        ) / a
        # Series expansion
        s = 1.0
        term = 1.0
        for n in range(1, 200):
            term *= (n - b) * x / (a + n)
            s += term
            if abs(term) < 1e-10:
                break
        return max(0.0, min(1.0, front * s))

    # -- sliding window comparison ------------------------------------------

    def sliding_window(self, window: int = 10) -> List[int]:
        """Compare adjacent windows and flag significant shifts.

        Uses a two-sample t-like statistic between consecutive non-overlapping
        windows of size *window*.
        """
        n = len(self._values)
        if n < 2 * window:
            return []
        change_points: List[int] = []
        for i in range(window, n - window + 1):
            left = self._values[i - window : i]
            right = self._values[i : i + window]
            m1, m2 = np.mean(left), np.mean(right)
            s1, s2 = np.std(left, ddof=1), np.std(right, ddof=1)
            pooled_se = math.sqrt(
                (s1 ** 2 + s2 ** 2) / window
            ) if (s1 + s2) > 0 else 1e-12
            t_stat = abs(m1 - m2) / pooled_se if pooled_se > 0 else 0.0
            # Rough threshold: |t| > 2 ≈ p < 0.05 for moderate df
            if t_stat > 2.0:
                # Only keep if this is the strongest in its neighbourhood
                if not change_points or (i - change_points[-1]) >= window:
                    change_points.append(i)
        return change_points


# ---------------------------------------------------------------------------
# ExponentialSmoother
# ---------------------------------------------------------------------------


class ExponentialSmoother:
    """Holt-Winters exponential smoothing (single / double / triple)."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        seasonal_period: int = 1,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_period = max(1, seasonal_period)
        self._level: float = 0.0
        self._trend: float = 0.0
        self._season: np.ndarray = np.zeros(0)
        self._fitted: np.ndarray = np.zeros(0)
        self._residuals: np.ndarray = np.zeros(0)
        self._data: np.ndarray = np.zeros(0)

    # -- fitting ------------------------------------------------------------

    def fit(self, data: Sequence[float]) -> ExponentialSmoother:
        """Fit the smoother to *data*."""
        self._data = np.asarray(data, dtype=np.float64)
        n = len(self._data)
        if n == 0:
            return self
        m = self.seasonal_period
        use_seasonal = m > 1 and n >= 2 * m

        # Initialisation
        if use_seasonal:
            self._level = float(np.mean(self._data[:m]))
            self._trend = float(
                (np.mean(self._data[m : 2 * m]) - np.mean(self._data[:m])) / m
            )
            self._season = np.zeros(m)
            for j in range(m):
                indices = list(range(j, n, m))
                self._season[j] = float(np.mean(self._data[indices])) - self._level
        else:
            self._level = float(self._data[0])
            self._trend = float(self._data[1] - self._data[0]) if n > 1 else 0.0
            self._season = np.zeros(max(m, 1))

        fitted = np.zeros(n)
        for i in range(n):
            if use_seasonal:
                s_idx = i % m
                forecast_val = self._level + self._trend + self._season[s_idx]
                fitted[i] = forecast_val
                new_level = self.alpha * (self._data[i] - self._season[s_idx]) + (
                    1 - self.alpha
                ) * (self._level + self._trend)
                new_trend = self.beta * (new_level - self._level) + (
                    1 - self.beta
                ) * self._trend
                self._season[s_idx] = self.gamma * (
                    self._data[i] - new_level
                ) + (1 - self.gamma) * self._season[s_idx]
                self._level = new_level
                self._trend = new_trend
            else:
                forecast_val = self._level + self._trend
                fitted[i] = forecast_val
                new_level = self.alpha * self._data[i] + (1 - self.alpha) * (
                    self._level + self._trend
                )
                new_trend = self.beta * (new_level - self._level) + (
                    1 - self.beta
                ) * self._trend
                self._level = new_level
                self._trend = new_trend

        self._fitted = fitted
        self._residuals = self._data - fitted
        return self

    # -- forecasting --------------------------------------------------------

    def forecast(self, horizon: int) -> np.ndarray:
        """Forecast *horizon* steps ahead."""
        predictions = np.zeros(horizon)
        m = self.seasonal_period
        use_seasonal = m > 1 and len(self._season) == m
        for h in range(1, horizon + 1):
            val = self._level + h * self._trend
            if use_seasonal:
                s_idx = (len(self._data) + h - 1) % m
                val += self._season[s_idx]
            predictions[h - 1] = val
        return predictions

    # -- accessors ----------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return current model parameters."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "seasonal_period": self.seasonal_period,
            "level": self._level,
            "trend": self._trend,
            "season": self._season.tolist(),
        }

    @property
    def fitted_values(self) -> np.ndarray:
        return self._fitted

    @property
    def residuals(self) -> np.ndarray:
        return self._residuals


# ---------------------------------------------------------------------------
# DiversityTimeSeries
# ---------------------------------------------------------------------------


class DiversityTimeSeries:
    """Container for a diversity time series with convenience methods."""

    def __init__(self) -> None:
        self._times: List[str] = []
        self._scores: List[float] = []

    def add_observation(self, time: str, diversity_score: float) -> None:
        """Append an observation."""
        self._times.append(time)
        self._scores.append(diversity_score)

    def to_array(self) -> np.ndarray:
        """Return scores as a numpy array."""
        return np.asarray(self._scores, dtype=np.float64)

    def rolling_mean(self, window: int) -> np.ndarray:
        """Compute a centred rolling mean."""
        arr = self.to_array()
        n = len(arr)
        if n == 0 or window < 1:
            return np.array([])
        result = np.full(n, np.nan)
        half = window // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result[i] = np.mean(arr[lo:hi])
        return result

    def rolling_std(self, window: int) -> np.ndarray:
        """Compute a centred rolling standard deviation."""
        arr = self.to_array()
        n = len(arr)
        if n == 0 or window < 1:
            return np.array([])
        result = np.full(n, np.nan)
        half = window // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            if hi - lo > 1:
                result[i] = float(np.std(arr[lo:hi], ddof=1))
            else:
                result[i] = 0.0
        return result

    def plot_data(self) -> Dict[str, Any]:
        """Return a dictionary suitable for plotting."""
        arr = self.to_array()
        rm = self.rolling_mean(max(3, len(arr) // 5))
        return {
            "times": list(self._times),
            "scores": list(self._scores),
            "rolling_mean": rm.tolist(),
            "n_observations": len(self._scores),
            "mean": float(np.mean(arr)) if len(arr) else 0.0,
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    def __len__(self) -> int:
        return len(self._scores)

    def __repr__(self) -> str:
        return f"DiversityTimeSeries(n={len(self)})"


# ---------------------------------------------------------------------------
# Public API — main functions
# ---------------------------------------------------------------------------


def temporal_diversity_score(
    outputs_by_time: Dict[str, List[str]],
) -> TemporalScore:
    """Compute diversity scores across time periods.

    For each period the within-period diversity is measured using a
    combination of distinct-n ratios and mean pairwise Jaccard distance.
    The resulting time series is then analysed for trend, volatility, and
    lag-1 autocorrelation.

    Parameters
    ----------
    outputs_by_time:
        Mapping from period label to list of text outputs in that period.

    Returns
    -------
    TemporalScore
    """
    periods = sorted(outputs_by_time.keys())
    per_period: Dict[str, float] = {}
    scores: List[float] = []
    for period in periods:
        texts = outputs_by_time[period]
        score = _period_diversity(texts)
        per_period[period] = score
        scores.append(score)

    arr = np.asarray(scores, dtype=np.float64)
    overall = float(np.mean(arr)) if len(arr) else 0.0

    # Trend direction via simple linear regression
    if len(arr) >= 2:
        x = np.arange(len(arr), dtype=np.float64)
        slope, _, _ = _linear_regression(x, arr)
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"
    else:
        direction = "stable"

    # Volatility (coefficient of variation)
    volatility = float(np.std(arr, ddof=1) / np.mean(arr)) if len(arr) > 1 and np.mean(arr) != 0 else 0.0

    # Lag-1 autocorrelation
    if len(arr) > 2:
        analyzer = TimeSeriesAnalyzer(arr)
        acf = analyzer.autocorrelation(max_lag=1)
        autocorr = float(acf[1])
    else:
        autocorr = 0.0

    return TemporalScore(
        overall_score=overall,
        per_period_scores=per_period,
        trend_direction=direction,
        volatility=volatility,
        autocorrelation=autocorr,
        periods=periods,
    )


def detect_diversity_trends(
    history: List[Tuple[str, List[str]]],
) -> TrendReport:
    """Detect trends in diversity over time.

    Uses linear regression for the overall direction, CUSUM and binary
    segmentation for change-point detection, and a bootstrap procedure to
    estimate confidence in the trend.

    Parameters
    ----------
    history:
        List of ``(timestamp, texts)`` tuples ordered chronologically.

    Returns
    -------
    TrendReport
    """
    scores: List[float] = []
    period_summaries: Dict[str, float] = {}
    for ts, texts in history:
        s = _period_diversity(texts)
        scores.append(s)
        period_summaries[ts] = s

    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)

    # Linear regression
    if n >= 2:
        x = np.arange(n, dtype=np.float64)
        slope, _, r_sq = _linear_regression(x, arr)
    else:
        slope, r_sq = 0.0, 0.0

    # Direction classification
    if slope > 0.01:
        direction = "increasing"
    elif slope < -0.01:
        direction = "decreasing"
    else:
        direction = "stable"

    # Bootstrap confidence for slope sign
    n_bootstrap = 500
    rng = np.random.RandomState(42)
    same_sign_count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n) if n > 0 else np.array([])
        if len(idx) < 2:
            continue
        bx = np.arange(len(idx), dtype=np.float64)
        by = arr[idx]
        bs, _, _ = _linear_regression(bx, by)
        if (slope >= 0 and bs >= 0) or (slope < 0 and bs < 0):
            same_sign_count += 1
    confidence = same_sign_count / n_bootstrap if n_bootstrap > 0 else 0.0

    # Change-point detection
    change_points: List[int] = []
    if n >= 6:
        detector = ChangePointDetector(arr)
        cp_cusum = detector.cusum(threshold=1.0)
        cp_binseg = detector.binary_segmentation(min_segment=3)
        combined = sorted(set(cp_cusum + cp_binseg))
        # De-duplicate nearby points
        filtered: List[int] = []
        for cp in combined:
            if not filtered or cp - filtered[-1] >= 2:
                filtered.append(cp)
        change_points = filtered

    return TrendReport(
        direction=direction,
        slope=slope,
        confidence=confidence,
        change_points=change_points,
        period_summaries=period_summaries,
    )


def diversity_forecasting(
    history: List[Tuple[str, float]],
    horizon: int,
) -> Forecast:
    """Forecast future diversity scores.

    Implements Holt's linear-trend exponential smoothing as the primary
    method with a simple moving-average fallback.  Prediction intervals are
    computed from the residual standard error.

    Parameters
    ----------
    history:
        List of ``(timestamp, score)`` tuples.
    horizon:
        Number of future periods to forecast.

    Returns
    -------
    Forecast
    """
    scores = np.asarray([s for _, s in history], dtype=np.float64)
    n = len(scores)

    if n < 3:
        # Fallback: constant forecast
        mean_val = float(np.mean(scores)) if n > 0 else 0.0
        return Forecast(
            predicted_values=[mean_val] * horizon,
            confidence_intervals=[(mean_val, mean_val)] * horizon,
            method_used="constant",
            model_parameters={"mean": mean_val},
            residuals=[],
        )

    # --- Holt's method (double exponential smoothing) ----------------------
    best_alpha, best_beta = 0.3, 0.1
    best_mse = float("inf")
    for alpha_cand in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for beta_cand in [0.01, 0.05, 0.1, 0.2]:
            smoother = ExponentialSmoother(alpha=alpha_cand, beta=beta_cand)
            smoother.fit(scores)
            mse = float(np.mean(smoother.residuals ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha_cand
                best_beta = beta_cand

    smoother = ExponentialSmoother(alpha=best_alpha, beta=best_beta)
    smoother.fit(scores)
    predictions = smoother.forecast(horizon)
    residuals = smoother.residuals

    # Prediction intervals
    res_std = float(np.std(residuals, ddof=1)) if n > 2 else 0.0
    intervals: List[Tuple[float, float]] = []
    for h in range(1, horizon + 1):
        width = 1.96 * res_std * math.sqrt(h)
        intervals.append(
            (float(predictions[h - 1]) - width, float(predictions[h - 1]) + width)
        )

    # -- SMA fallback comparison (keep if SMA gives lower error) ------------
    window = min(5, n)
    sma_pred = float(np.mean(scores[-window:]))
    sma_residuals = []
    for i in range(window, n):
        sma_residuals.append(float(scores[i] - np.mean(scores[i - window : i])))
    sma_mse = float(np.mean(np.array(sma_residuals) ** 2)) if sma_residuals else float("inf")

    method = "holt_exponential_smoothing"
    pred_list = predictions.tolist()
    res_list = residuals.tolist()
    params: Dict[str, float] = {"alpha": best_alpha, "beta": best_beta}

    if sma_mse < best_mse and sma_residuals:
        method = "simple_moving_average"
        pred_list = [sma_pred] * horizon
        res_list = sma_residuals
        sma_std = float(np.std(sma_residuals, ddof=1)) if len(sma_residuals) > 1 else 0.0
        intervals = [
            (sma_pred - 1.96 * sma_std * math.sqrt(h), sma_pred + 1.96 * sma_std * math.sqrt(h))
            for h in range(1, horizon + 1)
        ]
        params = {"window": float(window)}

    return Forecast(
        predicted_values=pred_list,
        confidence_intervals=intervals,
        method_used=method,
        model_parameters=params,
        residuals=res_list,
    )


def seasonal_diversity_patterns(
    history: List[Tuple[str, float]],
) -> SeasonalReport:
    """Detect seasonal patterns in a diversity time series.

    Performs classical additive seasonal decomposition: the trend is
    estimated via centred moving average, the seasonal component is the
    average deviation from the trend at each seasonal position, and the
    residual is whatever remains.  Seasonality significance is tested with
    an F-test comparing seasonal variance to residual variance.

    Parameters
    ----------
    history:
        List of ``(timestamp, score)`` tuples.

    Returns
    -------
    SeasonalReport
    """
    scores = np.asarray([s for _, s in history], dtype=np.float64)
    n = len(scores)

    if n < 6:
        return SeasonalReport(
            has_seasonality=False,
            period=1,
            seasonal_component=scores.tolist(),
            trend_component=scores.tolist(),
            residual_component=[0.0] * n,
            strength=0.0,
        )

    # Try candidate periods and pick the one with strongest seasonality
    best_period = 1
    best_strength = 0.0
    best_result: Optional[Dict[str, Any]] = None

    max_period = max(2, n // 3)
    candidates = list(range(2, min(max_period + 1, n // 2 + 1)))

    for period in candidates:
        result = _decompose(scores, period)
        if result is not None and result["strength"] > best_strength:
            best_strength = result["strength"]
            best_period = period
            best_result = result

    if best_result is None:
        return SeasonalReport(
            has_seasonality=False,
            period=1,
            seasonal_component=[0.0] * n,
            trend_component=scores.tolist(),
            residual_component=[0.0] * n,
            strength=0.0,
        )

    # F-test: is the seasonal variance significantly larger than residual?
    seasonal = np.asarray(best_result["seasonal"])
    residual = np.asarray(best_result["residual"])
    var_seasonal = float(np.var(seasonal, ddof=1)) if len(seasonal) > 1 else 0.0
    var_residual = float(np.var(residual, ddof=1)) if len(residual) > 1 else 1e-12
    if var_residual < 1e-12:
        var_residual = 1e-12
    f_stat = var_seasonal / var_residual
    df1 = best_period - 1
    df2 = max(1, n - best_period)
    p_val = ChangePointDetector._f_pvalue_approx(f_stat, df1, df2)
    has_seasonality = p_val < 0.05 and best_strength > 0.1

    return SeasonalReport(
        has_seasonality=has_seasonality,
        period=best_period,
        seasonal_component=best_result["seasonal"],
        trend_component=best_result["trend"],
        residual_component=best_result["residual"],
        strength=best_strength,
    )


def _decompose(
    scores: np.ndarray, period: int
) -> Optional[Dict[str, Any]]:
    """Classical additive decomposition for a given *period*."""
    n = len(scores)
    if n < 2 * period:
        return None

    # Trend via centred moving average
    trend = np.full(n, np.nan)
    half = period // 2
    for i in range(half, n - half):
        window = scores[max(0, i - half) : i + half + 1]
        trend[i] = np.mean(window)

    # Fill edges by linear extrapolation
    valid = np.where(~np.isnan(trend))[0]
    if len(valid) < 2:
        return None
    slope_t, int_t, _ = _linear_regression(
        valid.astype(np.float64), trend[valid]
    )
    for i in range(n):
        if np.isnan(trend[i]):
            trend[i] = slope_t * i + int_t

    # Detrended series
    detrended = scores - trend

    # Seasonal component: average detrended value at each seasonal position
    seasonal_avg = np.zeros(period)
    for j in range(period):
        indices = list(range(j, n, period))
        seasonal_avg[j] = float(np.mean(detrended[indices]))
    # Centre the seasonal component
    seasonal_avg -= np.mean(seasonal_avg)

    seasonal = np.array([seasonal_avg[i % period] for i in range(n)])
    residual = scores - trend - seasonal

    # Strength of seasonality: 1 - Var(residual)/Var(scores - trend)
    var_dt = float(np.var(detrended, ddof=1)) if n > 1 else 1e-12
    var_r = float(np.var(residual, ddof=1)) if n > 1 else 0.0
    strength = max(0.0, 1.0 - var_r / var_dt) if var_dt > 1e-12 else 0.0

    return {
        "trend": trend.tolist(),
        "seasonal": seasonal.tolist(),
        "residual": residual.tolist(),
        "strength": strength,
    }


def novelty_decay_analysis(outputs: List[str]) -> DecayCurve:
    """Analyse how novelty decays over successive outputs.

    For each output *i* (starting from the second), novelty is defined as
    the minimum Jaccard distance to any previous output.  An exponential
    decay curve ``a * exp(-b * t) + c`` is then fitted using iterative
    least-squares.

    Parameters
    ----------
    outputs:
        Ordered list of text outputs.

    Returns
    -------
    DecayCurve
    """
    if len(outputs) < 2:
        return DecayCurve(
            decay_rate=0.0,
            half_life=float("inf"),
            fitted_params={"a": 0.0, "b": 0.0, "c": 0.0},
            novelty_scores=[1.0] if outputs else [],
            time_points=list(range(len(outputs))),
            r_squared=0.0,
        )

    token_sets = [set(_tokenize(t)) for t in outputs]
    novelty_scores: List[float] = [1.0]  # first output is fully novel
    for i in range(1, len(outputs)):
        min_dist = min(
            1.0 - _jaccard(token_sets[i], token_sets[j])
            for j in range(i)
        )
        novelty_scores.append(min_dist)

    time_points = list(range(len(novelty_scores)))
    t = np.asarray(time_points, dtype=np.float64)
    y = np.asarray(novelty_scores, dtype=np.float64)

    # Fit  y = a * exp(-b * t) + c  via Gauss-Newton iterations
    a, b, c = float(y[0]), 0.1, float(y[-1]) if len(y) > 1 else 0.0
    for _ in range(200):
        exp_bt = np.exp(-b * t)
        residuals = y - (a * exp_bt + c)
        # Jacobian columns
        j_a = exp_bt
        j_b = -a * t * exp_bt
        j_c = np.ones_like(t)
        J = np.column_stack([j_a, j_b, j_c])
        try:
            JtJ = J.T @ J
            reg = 1e-8 * np.eye(3)
            delta = np.linalg.solve(JtJ + reg, J.T @ residuals)
        except np.linalg.LinAlgError:
            break
        a += float(delta[0])
        b += float(delta[1])
        c += float(delta[2])
        # Clamp b to be positive
        b = max(b, 1e-8)
        if np.max(np.abs(delta)) < 1e-10:
            break

    # R²
    y_pred = a * np.exp(-b * t) + c
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    half_life = math.log(2) / b if b > 1e-12 else float("inf")

    return DecayCurve(
        decay_rate=b,
        half_life=half_life,
        fitted_params={"a": a, "b": b, "c": c},
        novelty_scores=novelty_scores,
        time_points=time_points,
        r_squared=max(0.0, r_squared),
    )
