"""
Reproducibility analysis module for diversity-promoting text generation evaluation.

Provides tools for analyzing variance across runs, seed sensitivity,
confidence intervals, and experiment replication to ensure reproducible results.
"""

import copy
import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import comb

logger = logging.getLogger(__name__)


class CIResult(dict):
    """Dict subclass that supports tuple unpacking as (lower, upper)."""

    def __iter__(self):
        yield self["lower"]
        yield self["upper"]


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility analysis."""

    n_runs: int = 10
    seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))
    confidence_level: float = 0.95
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            "diversity_score",
            "quality_score",
            "perplexity",
            "distinct_1",
            "distinct_2",
            "self_bleu",
            "entropy",
        ]
    )
    bootstrap_n_resamples: int = 10000
    bootstrap_samples: int = field(default=None, repr=False)
    outlier_z_threshold: float = 2.5
    min_effect_size: float = 0.2
    variance_decomposition_method: str = "anova"
    stability_weights: Dict[str, float] = field(default_factory=dict)
    checkpoint_dir: Optional[str] = None
    deterministic_cudnn: bool = True
    deterministic_algorithms: bool = True
    seed_search_range: Tuple[int, int] = (0, 10000)
    power_analysis_alpha: float = 0.05
    power_analysis_target_power: float = 0.80
    tolerance_coverage: float = 0.95
    tolerance_confidence: float = 0.95
    bayesian_prior_strength: float = 1.0
    random_seed: Optional[int] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self):
        if self.bootstrap_samples is not None and self.bootstrap_n_resamples == 10000:
            self.bootstrap_n_resamples = self.bootstrap_samples
        if self.bootstrap_samples is None:
            self.bootstrap_samples = self.bootstrap_n_resamples
        else:
            self.bootstrap_n_resamples = self.bootstrap_samples

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config to a plain dictionary."""
        d = asdict(self)
        d["seeds"] = list(d["seeds"])
        d["seed_search_range"] = list(d["seed_search_range"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReproducibilityConfig":
        """Deserialize a config from a dictionary."""
        d = dict(d)
        if "bootstrap_samples" in d:
            d.setdefault("bootstrap_n_resamples", d.pop("bootstrap_samples"))
        if "seed_search_range" in d and isinstance(d["seed_search_range"], list):
            d["seed_search_range"] = tuple(d["seed_search_range"])
        if "seeds" in d:
            d["seeds"] = list(d["seeds"])
        return cls(**d)


# ---------------------------------------------------------------------------
# 2. RunResult
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Container for a single experimental run."""

    seed: int
    metrics: Dict[str, float] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=lambda: {
        "start_time": 0.0,
        "end_time": 0.0,
        "elapsed_seconds": 0.0,
        "setup_seconds": 0.0,
        "inference_seconds": 0.0,
        "evaluation_seconds": 0.0,
    })
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_index: int = 0
    run_id: Optional[str] = None
    status: str = "completed"
    error_message: Optional[str] = None
    per_sample_metrics: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunResult":
        """Deserialize a result from a dictionary."""
        d = dict(d)
        return cls(**d)

    @property
    def total_time(self) -> float:
        return self.timing.get("elapsed_seconds", 0.0)

    def get_metric(self, name: str, default: float = float("nan")) -> float:
        return self.metrics.get(name, default)


# ---------------------------------------------------------------------------
# 3. InterRunVarianceAnalyzer
# ---------------------------------------------------------------------------


class InterRunVarianceAnalyzer:
    """Analyzes variance and consistency across multiple experimental runs."""

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        self.config = config or ReproducibilityConfig()

    # ---- public API -------------------------------------------------------

    def compute_variance_across_runs(
        self, results: List[RunResult]
    ) -> Dict[str, float]:
        """Compute per-metric variance across runs.

        Returns a dict keyed by metric name with the variance value.
        """
        if not results:
            logger.warning("No results provided for variance computation.")
            return {}

        metric_names = self._collect_metric_names(results)
        out: Dict[str, float] = {}

        for name in metric_names:
            values = np.array(
                [r.metrics[name] for r in results if name in r.metrics],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            out[name] = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
        logger.info("Computed variance across %d runs for %d metrics.", len(results), len(out))
        return out

    def compute_coefficient_of_variation(
        self, results: List[RunResult]
    ) -> Dict[str, float]:
        """Compute CV (std/mean) for each metric; handles zero-mean gracefully."""
        metric_names = self._collect_metric_names(results)
        cv_map: Dict[str, float] = {}
        for name in metric_names:
            values = self._metric_array(results, name)
            if values.size < 2:
                cv_map[name] = 0.0
                continue
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            if np.abs(mean_val) < 1e-12:
                cv_map[name] = float("inf") if std_val > 1e-12 else 0.0
            else:
                cv_map[name] = float(std_val / np.abs(mean_val))
        return cv_map

    def compute_intraclass_correlation(
        self,
        results: List[RunResult],
        metric_name: Optional[str] = None,
        icc_type: str = "ICC(1,1)",
    ) -> Dict[str, float]:
        """Compute intraclass correlation coefficient for consistency/agreement.

        Supports ICC(1,1), ICC(2,1), ICC(3,1) and their average-measure
        counterparts ICC(1,k), ICC(2,k), ICC(3,k).

        When per_sample_metrics are available, rows = samples, columns = runs.
        Otherwise falls back to treating each run as a single observation
        and computes a simplified one-way ICC.
        """
        metrics_to_compute = [metric_name] if metric_name else self._collect_metric_names(results)
        out: Dict[str, float] = {}

        for m in metrics_to_compute:
            matrix = self._build_rating_matrix(results, m)
            if matrix is None or matrix.shape[0] < 2 or matrix.shape[1] < 2:
                out[m] = float("nan")
                continue

            n, k = matrix.shape  # n subjects, k raters
            grand_mean = np.mean(matrix)

            row_means = np.mean(matrix, axis=1)
            col_means = np.mean(matrix, axis=0)

            ss_total = np.sum((matrix - grand_mean) ** 2)
            ss_rows = k * np.sum((row_means - grand_mean) ** 2)
            ss_cols = n * np.sum((col_means - grand_mean) ** 2)
            ss_error = ss_total - ss_rows - ss_cols

            ms_rows = ss_rows / max(n - 1, 1)
            ms_cols = ss_cols / max(k - 1, 1)
            ms_error = ss_error / max((n - 1) * (k - 1), 1)

            ms_within = (ss_total - ss_rows) / max(n * (k - 1), 1)

            if icc_type == "ICC(1,1)":
                icc_val = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)
            elif icc_type == "ICC(2,1)":
                denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
                icc_val = (ms_rows - ms_error) / denom if abs(denom) > 1e-15 else 0.0
            elif icc_type == "ICC(3,1)":
                denom = ms_rows + (k - 1) * ms_error
                icc_val = (ms_rows - ms_error) / denom if abs(denom) > 1e-15 else 0.0
            elif icc_type == "ICC(1,k)":
                icc_val = (ms_rows - ms_within) / ms_rows if abs(ms_rows) > 1e-15 else 0.0
            elif icc_type == "ICC(2,k)":
                denom = ms_rows + (ms_cols - ms_error) / n
                icc_val = (ms_rows - ms_error) / denom if abs(denom) > 1e-15 else 0.0
            elif icc_type == "ICC(3,k)":
                icc_val = (ms_rows - ms_error) / ms_rows if abs(ms_rows) > 1e-15 else 0.0
            else:
                logger.warning("Unknown ICC type %s, falling back to ICC(1,1).", icc_type)
                icc_val = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)

            icc_val = float(np.clip(icc_val, -1.0, 1.0))
            out[m] = icc_val

        return out

    def compute_concordance_correlation(
        self, results: List[RunResult], metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Lin's concordance correlation coefficient for every pair of runs.

        Returns per-metric dict with keys 'mean_ccc', 'min_ccc', 'pairwise'.
        """
        metrics_to_use = [metric_name] if metric_name else self._collect_metric_names(results)
        out: Dict[str, Dict[str, float]] = {}

        for m in metrics_to_use:
            matrix = self._build_rating_matrix(results, m)
            if matrix is None or matrix.shape[1] < 2:
                out[m] = {"mean_ccc": float("nan"), "min_ccc": float("nan"), "pairwise": {}}
                continue

            n_raters = matrix.shape[1]
            pairwise: Dict[str, float] = {}
            ccc_values: List[float] = []

            for i in range(n_raters):
                for j in range(i + 1, n_raters):
                    x = matrix[:, i]
                    y = matrix[:, j]
                    ccc = self._lins_ccc(x, y)
                    label = f"run_{i}_vs_{j}"
                    pairwise[label] = ccc
                    ccc_values.append(ccc)

            ccc_arr = np.array(ccc_values)
            out[m] = {
                "mean_ccc": float(np.mean(ccc_arr)) if ccc_arr.size else float("nan"),
                "min_ccc": float(np.min(ccc_arr)) if ccc_arr.size else float("nan"),
                "max_ccc": float(np.max(ccc_arr)) if ccc_arr.size else float("nan"),
                "std_ccc": float(np.std(ccc_arr, ddof=1)) if ccc_arr.size > 1 else 0.0,
                "pairwise": pairwise,
            }
        return out

    def friedman_test_across_runs(
        self, results: List[RunResult], metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Non-parametric Friedman test for significant differences across runs.

        Null hypothesis: all runs produce the same distribution of metric values.
        Requires per-sample metrics so that each sample acts as a 'block'.
        Falls back to a Kruskal-Wallis test if per-sample metrics are unavailable.
        """
        metrics_to_use = [metric_name] if metric_name else self._collect_metric_names(results)
        out: Dict[str, Dict[str, float]] = {}

        for m in metrics_to_use:
            matrix = self._build_rating_matrix(results, m)
            if matrix is None or matrix.shape[0] < 3 or matrix.shape[1] < 2:
                out[m] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "test_used": "none",
                }
                continue

            try:
                groups = [matrix[:, j] for j in range(matrix.shape[1])]
                stat, p = scipy_stats.friedmanchisquare(*groups)
                sig = bool(p < (1.0 - self.config.confidence_level))
                out[m] = {
                    "statistic": float(stat),
                    "p_value": float(p),
                    "significant": sig,
                    "test_used": "friedman",
                    "n_subjects": int(matrix.shape[0]),
                    "n_groups": int(matrix.shape[1]),
                }
            except Exception as exc:
                logger.debug("Friedman test failed for %s, trying Kruskal-Wallis: %s", m, exc)
                try:
                    stat, p = scipy_stats.kruskal(*[matrix[:, j] for j in range(matrix.shape[1])])
                    sig = bool(p < (1.0 - self.config.confidence_level))
                    out[m] = {
                        "statistic": float(stat),
                        "p_value": float(p),
                        "significant": sig,
                        "test_used": "kruskal",
                    }
                except Exception as exc2:
                    logger.warning("Both Friedman and Kruskal failed for %s: %s", m, exc2)
                    out[m] = {
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                        "significant": False,
                        "test_used": "none",
                    }
        return out

    def compute_effect_sizes(
        self, results: List[RunResult], metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compute Cohen's d and Hedges' g for all pairs of runs.

        Also includes Glass's delta and common-language effect size.
        """
        metrics_to_use = [metric_name] if metric_name else self._collect_metric_names(results)
        out: Dict[str, Dict[str, Any]] = {}

        for m in metrics_to_use:
            matrix = self._build_rating_matrix(results, m)
            if matrix is None or matrix.shape[1] < 2:
                out[m] = {"max_cohens_d": float("nan"), "max_hedges_g": float("nan"), "pairs": {}}
                continue

            n_raters = matrix.shape[1]
            pairs: Dict[str, Dict[str, float]] = {}
            all_d: List[float] = []
            all_g: List[float] = []

            for i in range(n_raters):
                for j in range(i + 1, n_raters):
                    x = matrix[:, i]
                    y = matrix[:, j]
                    d, g, glass_delta, cles = self._effect_size_pair(x, y)
                    label = f"run_{i}_vs_{j}"
                    pairs[label] = {
                        "cohens_d": d,
                        "hedges_g": g,
                        "glass_delta": glass_delta,
                        "cles": cles,
                    }
                    all_d.append(abs(d))
                    all_g.append(abs(g))

            d_arr = np.array(all_d) if all_d else np.array([float("nan")])
            g_arr = np.array(all_g) if all_g else np.array([float("nan")])

            out[m] = {
                "max_cohens_d": float(np.nanmax(d_arr)),
                "mean_cohens_d": float(np.nanmean(d_arr)),
                "max_hedges_g": float(np.nanmax(g_arr)),
                "mean_hedges_g": float(np.nanmean(g_arr)),
                "any_large_effect": bool(np.any(d_arr > 0.8)),
                "any_medium_effect": bool(np.any(d_arr > 0.5)),
                "pairs": pairs,
            }
        return out

    def variance_decomposition(
        self, results: List[RunResult], metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Decompose total variance into between-run and within-run components.

        Uses one-way random-effects ANOVA model:
            X_ij = mu + a_i + e_ij
        where a_i ~ N(0, sigma_a^2) and e_ij ~ N(0, sigma_e^2).
        """
        metrics_to_use = [metric_name] if metric_name else self._collect_metric_names(results)
        out: Dict[str, Dict[str, float]] = {}

        for m in metrics_to_use:
            matrix = self._build_rating_matrix(results, m)
            if matrix is None or matrix.shape[0] < 2 or matrix.shape[1] < 2:
                out[m] = {
                    "between_run_var": float("nan"),
                    "within_run_var": float("nan"),
                    "total_var": float("nan"),
                    "icc": float("nan"),
                    "between_run_pct": float("nan"),
                    "within_run_pct": float("nan"),
                }
                continue

            n, k = matrix.shape
            grand_mean = np.mean(matrix)

            # Between-run (columns = runs)
            col_means = np.mean(matrix, axis=0)
            ss_between = n * np.sum((col_means - grand_mean) ** 2)
            df_between = k - 1

            # Within-run
            ss_within = np.sum((matrix - col_means[np.newaxis, :]) ** 2)
            df_within = k * (n - 1)

            ms_between = ss_between / max(df_between, 1)
            ms_within = ss_within / max(df_within, 1)

            # Estimated variance components
            sigma_e2 = ms_within
            sigma_a2 = max((ms_between - ms_within) / n, 0.0)
            total_var = sigma_a2 + sigma_e2

            if total_var > 1e-15:
                between_pct = sigma_a2 / total_var
                within_pct = sigma_e2 / total_var
                icc_val = sigma_a2 / total_var
            else:
                between_pct = 0.0
                within_pct = 0.0
                icc_val = 0.0

            # F test
            f_stat = ms_between / ms_within if ms_within > 1e-15 else float("inf")
            p_val = 1.0 - scipy_stats.f.cdf(f_stat, df_between, df_within) if np.isfinite(f_stat) else 0.0

            out[m] = {
                "between_run_var": float(sigma_a2),
                "within_run_var": float(sigma_e2),
                "total_var": float(total_var),
                "between_run_pct": float(between_pct),
                "within_run_pct": float(within_pct),
                "icc": float(icc_val),
                "f_statistic": float(f_stat),
                "f_p_value": float(p_val),
                "ms_between": float(ms_between),
                "ms_within": float(ms_within),
                "df_between": int(df_between),
                "df_within": int(df_within),
            }
        return out

    def compute_stability_score(
        self, results: List[RunResult], weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Aggregate stability metric in [0, 1], higher = more stable.

        Components:
            1. Inverse CV score (low CV → high stability)
            2. ICC score
            3. Friedman non-significance score
            4. Effect-size smallness score
        """
        if not results:
            return {"overall": 0.0, "components": {}, "per_metric": {}}

        w = weights or self.config.stability_weights
        default_w = {"cv": 0.30, "icc": 0.30, "friedman": 0.20, "effect_size": 0.20}
        w = {k: w.get(k, default_w.get(k, 0.25)) for k in default_w}
        total_w = sum(w.values())
        w = {k: v / total_w for k, v in w.items()}

        cvs = self.compute_coefficient_of_variation(results)
        iccs = self.compute_intraclass_correlation(results)
        friedman = self.friedman_test_across_runs(results)
        effects = self.compute_effect_sizes(results)

        metric_names = self._collect_metric_names(results)
        per_metric: Dict[str, Dict[str, float]] = {}

        for m in metric_names:
            # CV score: map CV ∈ [0, 1] to stability 1→0
            cv_val = cvs.get(m, 0.0)
            cv_score = max(0.0, 1.0 - min(cv_val, 1.0))

            # ICC score
            icc_val = iccs.get(m, 0.0)
            icc_score = max(0.0, min(float(icc_val), 1.0)) if not math.isnan(icc_val) else 0.5

            # Friedman: high p-value = stable
            f_info = friedman.get(m, {})
            p_val = f_info.get("p_value", 0.5)
            if math.isnan(p_val):
                p_val = 0.5
            friedman_score = min(p_val / (1.0 - self.config.confidence_level), 1.0)

            # Effect size smallness
            e_info = effects.get(m, {})
            max_d = e_info.get("max_cohens_d", 0.0)
            if math.isnan(max_d):
                max_d = 0.0
            effect_score = max(0.0, 1.0 - min(max_d / 0.8, 1.0))

            combined = (
                w["cv"] * cv_score
                + w["icc"] * icc_score
                + w["friedman"] * friedman_score
                + w["effect_size"] * effect_score
            )
            per_metric[m] = {
                "stability_score": float(combined),
                "cv_score": float(cv_score),
                "icc_score": float(icc_score),
                "friedman_score": float(friedman_score),
                "effect_score": float(effect_score),
            }

        scores = [v["stability_score"] for v in per_metric.values()]
        overall = float(np.mean(scores)) if scores else 0.0

        return {
            "overall": overall,
            "per_metric": per_metric,
            "weights": w,
            "n_metrics": len(metric_names),
            "n_runs": len(results),
        }

    # ---- private helpers --------------------------------------------------

    def _collect_metric_names(self, results: List[RunResult]) -> List[str]:
        names: Set[str] = set()
        for r in results:
            names.update(r.metrics.keys())
        tracked = self.config.metrics_to_track
        if tracked:
            filtered = [n for n in tracked if n in names]
            if filtered:
                return filtered
        return sorted(names)

    def _metric_array(self, results: List[RunResult], name: str) -> np.ndarray:
        return np.array(
            [r.metrics[name] for r in results if name in r.metrics], dtype=np.float64
        )

    def _build_rating_matrix(
        self, results: List[RunResult], metric_name: str
    ) -> Optional[np.ndarray]:
        """Build an (n_samples × n_runs) matrix.

        If per_sample_metrics are available, use those; otherwise each run
        contributes a single value and we tile to create a 2-D matrix.
        """
        has_per_sample = all(
            r.per_sample_metrics is not None
            and metric_name in r.per_sample_metrics
            for r in results
        )
        if has_per_sample:
            lengths = [len(r.per_sample_metrics[metric_name]) for r in results]  # type: ignore[union-attr]
            min_len = min(lengths)
            cols = [
                np.array(r.per_sample_metrics[metric_name][:min_len], dtype=np.float64)  # type: ignore[union-attr]
                for r in results
            ]
            return np.column_stack(cols)

        values = self._metric_array(results, metric_name)
        if values.size < 2:
            return None
        return values.reshape(-1, 1) * np.ones((1, max(len(results), 2)))

    @staticmethod
    def _lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
        """Lin's concordance correlation coefficient."""
        if x.size < 2:
            return float("nan")
        mx, my = np.mean(x), np.mean(y)
        sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
        sxy = np.mean((x - mx) * (y - my))
        denom = sx2 + sy2 + (mx - my) ** 2
        if denom < 1e-15:
            return 1.0
        return float(2.0 * sxy / denom)

    @staticmethod
    def _effect_size_pair(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Return (Cohen's d, Hedges' g, Glass's delta, CLES)."""
        n1, n2 = x.size, y.size
        if n1 < 2 or n2 < 2:
            return (float("nan"),) * 4  # type: ignore[return-value]
        m1, m2 = np.mean(x), np.mean(y)
        s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
        diff = m1 - m2

        # Pooled std for Cohen's d
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        d = diff / sp if sp > 1e-15 else 0.0

        # Hedges' g correction factor
        df = n1 + n2 - 2
        correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
        g = d * correction

        # Glass's delta (use control = first group)
        glass = diff / s1 if s1 > 1e-15 else 0.0

        # Common-language effect size
        se_diff = np.sqrt(s1 ** 2 + s2 ** 2)
        cles = float(scipy_stats.norm.cdf(diff / se_diff)) if se_diff > 1e-15 else 0.5

        return float(d), float(g), float(glass), float(cles)


# ---------------------------------------------------------------------------
# 4. SeedSensitivityAnalyzer
# ---------------------------------------------------------------------------


class SeedSensitivityAnalyzer:
    """Analyzes sensitivity of results to random seed choice."""

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        self.config = config or ReproducibilityConfig()

    def analyze_seed_sensitivity(
        self, results: List[RunResult]
    ) -> Dict[str, Dict[str, float]]:
        """Quantify how much each metric varies as a function of seed.

        Metrics returned per tracked metric:
            sensitivity_index  – normalised range / mean
            entropy_ratio      – entropy of seed distribution vs uniform
            max_deviation       – maximum absolute deviation from mean
            relative_range      – (max-min)/mean
        """
        metric_names = self._metric_names(results)
        out: Dict[str, Dict[str, float]] = {}

        for m in metric_names:
            values = np.array(
                [r.metrics[m] for r in results if m in r.metrics], dtype=np.float64
            )
            if values.size < 2:
                out[m] = {
                    "sensitivity_index": 0.0,
                    "entropy_ratio": 1.0,
                    "max_deviation": 0.0,
                    "relative_range": 0.0,
                }
                continue

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            rng = float(np.ptp(values))

            # Sensitivity index: CV-like measure
            sensitivity = std_val / np.abs(mean_val) if np.abs(mean_val) > 1e-12 else (
                float("inf") if std_val > 1e-12 else 0.0
            )

            # Entropy ratio: how uniform is the distribution of values?
            hist, _ = np.histogram(values, bins=max(int(np.sqrt(values.size)), 2))
            hist = hist[hist > 0].astype(np.float64)
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1.0
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0

            max_dev = float(np.max(np.abs(values - mean_val)))
            rel_range = rng / np.abs(mean_val) if np.abs(mean_val) > 1e-12 else 0.0

            out[m] = {
                "sensitivity_index": float(sensitivity),
                "entropy_ratio": float(entropy_ratio),
                "max_deviation": float(max_dev),
                "relative_range": float(rel_range),
                "std": float(std_val),
                "mean": float(mean_val),
                "range": float(rng),
                "n_seeds": int(values.size),
            }
        logger.info("Seed sensitivity computed for %d metrics.", len(out))
        return out

    def compute_seed_correlations(
        self, results: List[RunResult]
    ) -> Dict[str, Any]:
        """Compute correlation matrix of metric vectors across seeds.

        Each seed defines a vector of metric values; we compute the
        Pearson and Spearman correlation matrices.
        """
        metric_names = self._metric_names(results)
        if not metric_names or len(results) < 2:
            return {"pearson": [], "spearman": [], "seeds": []}

        # Build (n_seeds × n_metrics) matrix
        matrix = np.zeros((len(results), len(metric_names)), dtype=np.float64)
        for i, r in enumerate(results):
            for j, m in enumerate(metric_names):
                matrix[i, j] = r.metrics.get(m, float("nan"))

        # Pairwise Pearson between seeds
        n = matrix.shape[0]
        pearson_mat = np.eye(n)
        spearman_mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                valid = ~(np.isnan(matrix[i]) | np.isnan(matrix[j]))
                if valid.sum() >= 2:
                    pr, _ = scipy_stats.pearsonr(matrix[i, valid], matrix[j, valid])
                    sr, _ = scipy_stats.spearmanr(matrix[i, valid], matrix[j, valid])
                else:
                    pr, sr = float("nan"), float("nan")
                pearson_mat[i, j] = pearson_mat[j, i] = pr
                spearman_mat[i, j] = spearman_mat[j, i] = sr

        seeds = [r.seed for r in results]
        return {
            "pearson": pearson_mat.tolist(),
            "spearman": spearman_mat.tolist(),
            "seeds": seeds,
            "metric_names": metric_names,
            "mean_pearson": float(np.nanmean(pearson_mat[np.triu_indices(n, k=1)])),
            "mean_spearman": float(np.nanmean(spearman_mat[np.triu_indices(n, k=1)])),
        }

    def identify_outlier_seeds(
        self, results: List[RunResult], z_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Detect seeds whose results are anomalous using z-score and Grubbs' test.

        Returns per-metric outlier info and an overall outlier set.
        """
        z_thr = z_threshold or self.config.outlier_z_threshold
        metric_names = self._metric_names(results)
        per_metric: Dict[str, Dict[str, Any]] = {}
        overall_outlier_seeds: Set[int] = set()

        for m in metric_names:
            seeds = []
            values = []
            for r in results:
                if m in r.metrics:
                    seeds.append(r.seed)
                    values.append(r.metrics[m])
            if len(values) < 3:
                per_metric[m] = {"outliers": [], "z_scores": {}}
                continue

            arr = np.array(values, dtype=np.float64)
            mean_v = np.mean(arr)
            std_v = np.std(arr, ddof=1)
            if std_v < 1e-15:
                per_metric[m] = {"outliers": [], "z_scores": {s: 0.0 for s in seeds}}
                continue

            z_scores = (arr - mean_v) / std_v
            outlier_mask = np.abs(z_scores) > z_thr

            # Grubbs' test for the most extreme value
            g_stat = float(np.max(np.abs(z_scores)))
            n = len(arr)
            t_crit_sq = scipy_stats.t.ppf(1 - 0.05 / (2 * n), n - 2) ** 2
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit_sq / (n - 2 + t_crit_sq))
            grubbs_outlier = g_stat > g_crit

            outlier_seeds_m = [seeds[k] for k in range(len(seeds)) if outlier_mask[k]]
            z_map = {seeds[k]: float(z_scores[k]) for k in range(len(seeds))}

            per_metric[m] = {
                "outliers": outlier_seeds_m,
                "z_scores": z_map,
                "grubbs_statistic": float(g_stat),
                "grubbs_critical": float(g_crit),
                "grubbs_outlier_detected": bool(grubbs_outlier),
            }
            overall_outlier_seeds.update(outlier_seeds_m)

        outlier_frequency: Dict[int, int] = {}
        for m_info in per_metric.values():
            for s in m_info.get("outliers", []):
                outlier_frequency[s] = outlier_frequency.get(s, 0) + 1

        return {
            "per_metric": per_metric,
            "overall_outlier_seeds": sorted(overall_outlier_seeds),
            "outlier_frequency": outlier_frequency,
            "total_metrics_checked": len(metric_names),
            "z_threshold_used": z_thr,
        }

    def compute_seed_influence(
        self, results: List[RunResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Leave-one-out influence analysis for each seed.

        For each metric, drop one seed at a time, recompute the mean,
        and measure how much the aggregate changes.
        """
        metric_names = self._metric_names(results)
        out: Dict[str, Dict[str, Any]] = {}

        for m in metric_names:
            seeds_vals = [(r.seed, r.metrics[m]) for r in results if m in r.metrics]
            if len(seeds_vals) < 3:
                out[m] = {"influence_scores": {}, "max_influence_seed": None}
                continue

            seeds_list = [sv[0] for sv in seeds_vals]
            vals = np.array([sv[1] for sv in seeds_vals], dtype=np.float64)
            full_mean = np.mean(vals)
            full_std = np.std(vals, ddof=1)

            influence_scores: Dict[int, float] = {}
            loo_means: Dict[int, float] = {}
            loo_stds: Dict[int, float] = {}

            for idx in range(len(vals)):
                loo = np.delete(vals, idx)
                loo_mean = float(np.mean(loo))
                loo_std = float(np.std(loo, ddof=1))
                shift = abs(loo_mean - full_mean)
                normalizer = full_std if full_std > 1e-15 else 1.0
                influence = shift / normalizer
                influence_scores[seeds_list[idx]] = float(influence)
                loo_means[seeds_list[idx]] = loo_mean
                loo_stds[seeds_list[idx]] = loo_std

            # Cook's distance analogue
            hat_values = 1.0 / len(vals)  # equal leverage in balanced design
            residuals = vals - full_mean
            mse = np.mean(residuals ** 2)
            cooks = (residuals ** 2 * hat_values) / (1 * mse * (1 - hat_values) ** 2) if mse > 1e-15 else np.zeros_like(vals)

            max_seed = max(influence_scores, key=influence_scores.get)  # type: ignore[arg-type]

            out[m] = {
                "influence_scores": influence_scores,
                "loo_means": loo_means,
                "loo_stds": loo_stds,
                "max_influence_seed": int(max_seed),
                "max_influence_value": float(influence_scores[max_seed]),
                "full_mean": float(full_mean),
                "full_std": float(full_std),
                "cooks_distance": {seeds_list[i]: float(cooks[i]) for i in range(len(seeds_list))},
            }
        return out

    def seed_clustering(
        self,
        results: List[RunResult],
        n_clusters: Optional[int] = None,
        distance_metric: str = "euclidean",
        linkage_method: str = "ward",
    ) -> Dict[str, Any]:
        """Cluster seeds by similarity of their output metric profiles.

        Uses hierarchical agglomerative clustering.
        """
        metric_names = self._metric_names(results)
        if not metric_names or len(results) < 2:
            return {"clusters": {}, "n_clusters": 0}

        # Build feature matrix: seeds × metrics
        mat = np.zeros((len(results), len(metric_names)), dtype=np.float64)
        for i, r in enumerate(results):
            for j, m in enumerate(metric_names):
                mat[i, j] = r.metrics.get(m, float("nan"))

        # Impute NaN with column mean
        col_means = np.nanmean(mat, axis=0)
        for j in range(mat.shape[1]):
            mask = np.isnan(mat[:, j])
            mat[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

        # Standardize
        stds = np.std(mat, axis=0, ddof=1)
        stds[stds < 1e-15] = 1.0
        mat_z = (mat - np.mean(mat, axis=0)) / stds

        # Hierarchical clustering
        if mat_z.shape[0] < 2:
            return {"clusters": {0: [results[0].seed]}, "n_clusters": 1}

        dist_method = distance_metric if linkage_method != "ward" else "euclidean"
        dist_mat = pdist(mat_z, metric=dist_method)
        link = linkage(dist_mat, method=linkage_method)

        if n_clusters is None:
            # Use the gap/elbow heuristic: max(diff of merge distances)
            merge_dists = link[:, 2]
            if len(merge_dists) > 1:
                diffs = np.diff(merge_dists)
                optimal_k = len(results) - int(np.argmax(diffs)) - 1
                n_clusters = max(2, min(optimal_k, len(results) - 1))
            else:
                n_clusters = 2

        labels = fcluster(link, t=n_clusters, criterion="maxclust")

        clusters: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(results[i].seed)

        # Silhouette-like score
        full_dist = squareform(dist_mat)
        silhouettes = np.zeros(len(results))
        for i in range(len(results)):
            same = [j for j in range(len(results)) if labels[j] == labels[i] and j != i]
            diff_clusters = set(labels) - {labels[i]}
            if not same:
                silhouettes[i] = 0.0
                continue
            a_i = np.mean([full_dist[i, j] for j in same])
            b_vals = []
            for cl in diff_clusters:
                members = [j for j in range(len(results)) if labels[j] == cl]
                if members:
                    b_vals.append(np.mean([full_dist[i, j] for j in members]))
            b_i = min(b_vals) if b_vals else a_i
            denom = max(a_i, b_i)
            silhouettes[i] = (b_i - a_i) / denom if denom > 1e-15 else 0.0

        return {
            "clusters": clusters,
            "n_clusters": int(n_clusters),
            "labels": [int(l) for l in labels],
            "seeds": [r.seed for r in results],
            "silhouette_scores": {results[i].seed: float(silhouettes[i]) for i in range(len(results))},
            "mean_silhouette": float(np.mean(silhouettes)),
            "linkage_method": linkage_method,
            "distance_metric": distance_metric,
        }

    def recommend_minimum_seeds(
        self,
        results: List[RunResult],
        target_ci_width: Optional[float] = None,
        metric_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Power-analysis-based recommendation for minimum number of seeds.

        Estimates the number of seeds needed to achieve a confidence interval
        of width ≤ target_ci_width (default: 10% of mean) at the configured
        confidence level.
        """
        metrics_to_use = [metric_name] if metric_name else self._metric_names(results)
        out: Dict[str, Dict[str, Any]] = {}

        alpha = 1.0 - self.config.confidence_level
        z = scipy_stats.norm.ppf(1.0 - alpha / 2.0)

        for m in metrics_to_use:
            values = np.array(
                [r.metrics[m] for r in results if m in r.metrics], dtype=np.float64
            )
            if values.size < 2:
                out[m] = {"recommended_seeds": None, "reason": "insufficient_data"}
                continue

            mean_v = np.mean(values)
            std_v = np.std(values, ddof=1)

            target = target_ci_width
            if target is None:
                target = 0.10 * np.abs(mean_v) if np.abs(mean_v) > 1e-12 else 0.01

            # n = (2 * z * std / target_width)^2
            if target < 1e-15:
                n_needed = len(values)
            else:
                n_needed = int(np.ceil((2 * z * std_v / target) ** 2))
            n_needed = max(n_needed, 2)

            # Also provide power-based estimate for detecting a minimum effect
            min_effect = self.config.min_effect_size
            effect_in_units = min_effect * std_v
            if effect_in_units > 1e-15:
                z_beta = scipy_stats.norm.ppf(self.config.power_analysis_target_power)
                n_power = int(np.ceil(((z + z_beta) * std_v / effect_in_units) ** 2))
                n_power = max(n_power, 2)
            else:
                n_power = len(values)

            out[m] = {
                "recommended_seeds_ci": int(n_needed),
                "recommended_seeds_power": int(n_power),
                "recommended_seeds": int(max(n_needed, n_power)),
                "current_seeds": int(values.size),
                "current_ci_width": float(2 * z * std_v / np.sqrt(values.size)),
                "target_ci_width": float(target),
                "std": float(std_v),
                "mean": float(mean_v),
                "confidence_level": float(self.config.confidence_level),
                "target_power": float(self.config.power_analysis_target_power),
                "min_effect_size": float(min_effect),
            }
        return out

    # ---- private helpers --------------------------------------------------

    def _metric_names(self, results: List[RunResult]) -> List[str]:
        names: Set[str] = set()
        for r in results:
            names.update(r.metrics.keys())
        tracked = self.config.metrics_to_track
        if tracked:
            return [n for n in tracked if n in names]
        return sorted(names)


# ---------------------------------------------------------------------------
# 5. ConfidenceIntervalComputer
# ---------------------------------------------------------------------------


class ConfidenceIntervalComputer:
    """Computes various types of confidence and credible intervals."""

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        self.config = config or ReproducibilityConfig()

    def normal_ci(
        self,
        data: np.ndarray,
        confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """Standard normal-theory confidence interval for the mean."""
        data = np.asarray(data, dtype=np.float64).ravel()
        data = data[~np.isnan(data)]
        cl = confidence or self.config.confidence_level
        n = data.size
        if n < 2:
            m = float(np.mean(data)) if n == 1 else float("nan")
            return CIResult({"mean": m, "lower": m, "upper": m, "ci_width": 0.0, "se": 0.0, "n": n})

        mean = float(np.mean(data))
        se = float(scipy_stats.sem(data))
        alpha = 1.0 - cl
        t_crit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        margin = t_crit * se
        return CIResult({
            "mean": mean,
            "lower": mean - margin,
            "upper": mean + margin,
            "ci_width": 2 * margin,
            "se": se,
            "t_critical": float(t_crit),
            "df": int(n - 1),
            "n": int(n),
            "confidence_level": cl,
        })

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable = np.mean,
        confidence: Optional[float] = None,
        n_resamples: Optional[int] = None,
        method: str = "bca",
    ) -> Dict[str, float]:
        """Bootstrap confidence interval using percentile, basic, or BCa method."""
        data = np.asarray(data, dtype=np.float64).ravel()
        data = data[~np.isnan(data)]
        cl = confidence or self.config.confidence_level
        n_boot = n_resamples or self.config.bootstrap_n_resamples
        n = data.size

        if n < 2:
            val = float(statistic(data)) if n == 1 else float("nan")
            return CIResult({"estimate": val, "lower": val, "upper": val, "method": method})

        rng = np.random.RandomState(seed=42)
        theta_hat = float(statistic(data))
        boot_stats = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sample = data[rng.randint(0, n, size=n)]
            boot_stats[b] = statistic(sample)

        alpha = 1.0 - cl

        if method == "percentile":
            lower = float(np.percentile(boot_stats, 100 * alpha / 2))
            upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        elif method == "basic":
            lower = 2 * theta_hat - float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
            upper = 2 * theta_hat - float(np.percentile(boot_stats, 100 * alpha / 2))
        elif method == "bca":
            # Bias correction
            z0 = scipy_stats.norm.ppf(np.mean(boot_stats < theta_hat))
            if np.isinf(z0):
                z0 = 0.0

            # Acceleration via jackknife
            jack = np.empty(n, dtype=np.float64)
            for i in range(n):
                jack[i] = statistic(np.delete(data, i))
            jack_mean = np.mean(jack)
            num = np.sum((jack_mean - jack) ** 3)
            den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
            a = num / den if abs(den) > 1e-15 else 0.0

            z_alpha_low = scipy_stats.norm.ppf(alpha / 2)
            z_alpha_high = scipy_stats.norm.ppf(1 - alpha / 2)

            def _bca_quantile(z_alpha: float) -> float:
                num_q = z0 + z_alpha
                denom_q = 1 - a * num_q
                if abs(denom_q) < 1e-15:
                    return z_alpha
                adjusted = z0 + num_q / denom_q
                return float(scipy_stats.norm.cdf(adjusted))

            p_low = _bca_quantile(z_alpha_low)
            p_high = _bca_quantile(z_alpha_high)
            p_low = np.clip(p_low, 0.001, 0.999)
            p_high = np.clip(p_high, 0.001, 0.999)

            lower = float(np.percentile(boot_stats, 100 * p_low))
            upper = float(np.percentile(boot_stats, 100 * p_high))
        else:
            lower = float(np.percentile(boot_stats, 100 * alpha / 2))
            upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return CIResult({
            "estimate": theta_hat,
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower,
            "method": method,
            "n_resamples": n_boot,
            "boot_mean": float(np.mean(boot_stats)),
            "boot_std": float(np.std(boot_stats, ddof=1)),
            "boot_bias": float(np.mean(boot_stats) - theta_hat),
            "n": int(n),
            "confidence_level": cl,
        })

    def bayesian_ci(
        self,
        data: np.ndarray,
        prior_mean: float = 0.0,
        prior_variance: float = 1e6,
        confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """Bayesian credible interval assuming Normal-Normal conjugate model.

        Prior: mu ~ N(prior_mean, prior_variance)
        Likelihood: x_i | mu ~ N(mu, sigma^2)   (sigma estimated from data)
        Posterior: mu | data ~ N(posterior_mean, posterior_var)
        """
        data = np.asarray(data, dtype=np.float64).ravel()
        data = data[~np.isnan(data)]
        cl = confidence or self.config.confidence_level
        n = data.size

        if n < 1:
            return CIResult({"posterior_mean": float("nan"), "lower": float("nan"), "upper": float("nan")})

        data_mean = float(np.mean(data))
        data_var = float(np.var(data, ddof=1)) if n > 1 else 1.0

        prior_precision = 1.0 / prior_variance if prior_variance > 1e-15 else 1e-10
        likelihood_precision = n / data_var if data_var > 1e-15 else n * 1e10

        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1.0 / posterior_precision
        posterior_mean = posterior_var * (prior_precision * prior_mean + likelihood_precision * data_mean)
        posterior_std = np.sqrt(posterior_var)

        alpha = 1.0 - cl
        z_low = scipy_stats.norm.ppf(alpha / 2)
        z_high = scipy_stats.norm.ppf(1 - alpha / 2)

        lower = posterior_mean + z_low * posterior_std
        upper = posterior_mean + z_high * posterior_std

        return CIResult({
            "posterior_mean": float(posterior_mean),
            "posterior_std": float(posterior_std),
            "posterior_variance": float(posterior_var),
            "lower": float(lower),
            "upper": float(upper),
            "ci_width": float(upper - lower),
            "prior_mean": float(prior_mean),
            "prior_variance": float(prior_variance),
            "prior_weight": float(prior_precision / posterior_precision),
            "data_weight": float(likelihood_precision / posterior_precision),
            "n": int(n),
            "confidence_level": cl,
        })

    def wilson_score_ci(
        self,
        successes: int,
        trials: int = None,
        confidence: Optional[float] = None,
        total: int = None,
    ) -> "CIResult":
        """Wilson score interval for a binomial proportion.

        More accurate than the Wald interval, especially for small samples
        or proportions near 0 or 1.
        """
        if total is not None and trials is None:
            trials = total
        cl = confidence or self.config.confidence_level
        if trials <= 0:
            return CIResult({"proportion": float("nan"), "lower": 0.0, "upper": 0.0})

        p_hat = successes / trials
        alpha = 1.0 - cl
        z = scipy_stats.norm.ppf(1 - alpha / 2)
        z2 = z * z

        denom = 1 + z2 / trials
        centre = p_hat + z2 / (2 * trials)
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials)

        lower = max(0.0, (centre - spread) / denom)
        upper = min(1.0, (centre + spread) / denom)

        # Continuity-corrected variant (Newcombe)
        spread_cc = z * np.sqrt(
            (p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials
        ) + 0.5 / trials
        lower_cc = max(0.0, (centre - spread_cc) / denom)
        upper_cc = min(1.0, (centre + spread_cc) / denom)

        return CIResult({
            "proportion": float(p_hat),
            "lower": float(lower),
            "upper": float(upper),
            "ci_width": float(upper - lower),
            "lower_continuity_corrected": float(lower_cc),
            "upper_continuity_corrected": float(upper_cc),
            "z_critical": float(z),
            "successes": int(successes),
            "trials": int(trials),
            "confidence_level": cl,
        })

    def simultaneous_ci(
        self,
        data_dict: Dict[str, np.ndarray],
        confidence: Optional[float] = None,
        method: str = "bonferroni",
    ) -> Dict[str, Dict[str, float]]:
        """Simultaneous confidence intervals for multiple comparisons.

        Supports Bonferroni, Holm (step-down), and Šidák corrections.
        """
        cl = confidence or self.config.confidence_level
        alpha = 1.0 - cl
        k = len(data_dict)
        if k == 0:
            return {}

        if method == "bonferroni":
            adjusted_alpha = alpha / k
        elif method == "sidak":
            adjusted_alpha = 1.0 - (1.0 - alpha) ** (1.0 / k)
        elif method == "holm":
            adjusted_alpha = alpha  # handled per-interval below
        else:
            adjusted_alpha = alpha / k

        result: Dict[str, Dict[str, float]] = {}

        if method == "holm":
            # Sort by p-value proxy (use SE-based ordering)
            items = []
            for name, arr in data_dict.items():
                arr = np.asarray(arr, dtype=np.float64).ravel()
                arr = arr[~np.isnan(arr)]
                if arr.size > 1:
                    se = scipy_stats.sem(arr)
                    items.append((name, arr, se))
                else:
                    items.append((name, arr, float("inf")))
            # Sort by SE descending (wider = need more correction)
            items.sort(key=lambda x: -x[2])

            for rank, (name, arr, _se) in enumerate(items):
                holm_alpha = alpha / (k - rank)
                adj_cl = 1.0 - holm_alpha
                ci = self.normal_ci(arr, confidence=adj_cl)
                ci["adjustment_method"] = "holm"
                ci["adjusted_alpha"] = holm_alpha
                ci["rank"] = rank + 1
                result[name] = ci
        else:
            adj_cl = 1.0 - adjusted_alpha
            for name, arr in data_dict.items():
                arr = np.asarray(arr, dtype=np.float64).ravel()
                ci = self.normal_ci(arr, confidence=adj_cl)
                ci["adjustment_method"] = method
                ci["adjusted_alpha"] = adjusted_alpha
                ci["original_alpha"] = alpha
                ci["n_comparisons"] = k
                result[name] = ci

        return result

    def prediction_interval(
        self,
        data: np.ndarray,
        confidence: Optional[float] = None,
        n_future: int = 1,
    ) -> Dict[str, float]:
        """Prediction interval for future observations.

        Predicts the range in which future observations will fall.
        """
        data = np.asarray(data, dtype=np.float64).ravel()
        data = data[~np.isnan(data)]
        cl = confidence or self.config.confidence_level
        n = data.size

        if n < 2:
            val = float(np.mean(data)) if n == 1 else float("nan")
            return CIResult({"mean": val, "lower": val, "upper": val})

        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        alpha = 1.0 - cl
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Prediction interval factor accounts for both estimation and sampling uncertainty
        margin = t_crit * std * np.sqrt(1 + 1.0 / n)

        # For m future observations (Bonferroni-style)
        if n_future > 1:
            adjusted_alpha = alpha / n_future
            t_crit_adj = scipy_stats.t.ppf(1 - adjusted_alpha / 2, df=n - 1)
            margin_adj = t_crit_adj * std * np.sqrt(1 + 1.0 / n)
        else:
            margin_adj = margin

        return CIResult({
            "mean": mean,
            "lower": mean - margin_adj,
            "upper": mean + margin_adj,
            "pi_width": 2 * margin_adj,
            "std": std,
            "t_critical": float(t_crit),
            "n_future": n_future,
            "n": int(n),
            "confidence_level": cl,
        })

    def tolerance_interval(
        self,
        data: np.ndarray,
        coverage: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """Normal-theory tolerance interval.

        An interval that covers at least `coverage` proportion of the
        population with `confidence` confidence.

        Uses the exact k-factor for a two-sided tolerance interval.
        """
        data = np.asarray(data, dtype=np.float64).ravel()
        data = data[~np.isnan(data)]
        p = coverage or self.config.tolerance_coverage
        cl = confidence or self.config.tolerance_confidence
        n = data.size

        if n < 2:
            val = float(np.mean(data)) if n == 1 else float("nan")
            return CIResult({"mean": val, "lower": val, "upper": val})

        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))

        # Two-sided tolerance factor (Howe's approximation)
        alpha = 1.0 - cl
        z_p = scipy_stats.norm.ppf((1 + p) / 2)
        chi2_val = scipy_stats.chi2.ppf(1 - alpha, df=n - 1)

        k_factor = z_p * np.sqrt((n - 1) * (1 + 1.0 / n) / chi2_val)

        lower = mean - k_factor * std
        upper = mean + k_factor * std

        return CIResult({
            "mean": mean,
            "lower": float(lower),
            "upper": float(upper),
            "ti_width": float(upper - lower),
            "k_factor": float(k_factor),
            "coverage": p,
            "confidence": cl,
            "std": std,
            "n": int(n),
        })

    def credible_interval(
        self,
        data: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        model: str = "normal",
        confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """Bayesian credible interval with conjugate priors.

        Supports 'normal' (Normal-Normal) and 'beta' (Beta-Binomial) models.
        """
        data = np.asarray(data, dtype=np.float64).ravel()
        data = data[~np.isnan(data)]
        cl = confidence or self.config.confidence_level
        n = data.size
        alpha_q = (1.0 - cl) / 2.0

        if model == "normal":
            # Normal-Inverse-Gamma conjugate prior
            if n < 1:
                return CIResult({"posterior_mean": float("nan"), "lower": float("nan"), "upper": float("nan")})

            prior_mu = prior_alpha
            prior_kappa = prior_beta  # pseudo-count for prior mean
            prior_alpha_ig = 1.0  # shape
            prior_beta_ig = 1.0  # rate

            data_mean = float(np.mean(data))
            data_var = float(np.var(data)) if n > 1 else 1.0

            # Posterior parameters
            kappa_n = prior_kappa + n
            mu_n = (prior_kappa * prior_mu + n * data_mean) / kappa_n
            alpha_n = prior_alpha_ig + n / 2.0
            beta_n = prior_beta_ig + 0.5 * n * data_var + 0.5 * (
                prior_kappa * n * (data_mean - prior_mu) ** 2 / kappa_n
            )

            # Marginal posterior for mu is Student-t
            scale = np.sqrt(beta_n / (alpha_n * kappa_n))
            df = 2 * alpha_n

            t_low = scipy_stats.t.ppf(alpha_q, df=df, loc=mu_n, scale=scale)
            t_high = scipy_stats.t.ppf(1 - alpha_q, df=df, loc=mu_n, scale=scale)

            return CIResult({
                "posterior_mean": float(mu_n),
                "posterior_scale": float(scale),
                "posterior_df": float(df),
                "lower": float(t_low),
                "upper": float(t_high),
                "ci_width": float(t_high - t_low),
                "model": "normal_inverse_gamma",
                "prior_mu": float(prior_mu),
                "prior_kappa": float(prior_kappa),
                "n": int(n),
                "confidence_level": cl,
            })

        elif model == "beta":
            # Beta-Binomial: data should be 0/1
            successes = int(np.sum(data > 0.5))
            failures = n - successes
            post_alpha = prior_alpha + successes
            post_beta = prior_beta + failures

            lower = float(scipy_stats.beta.ppf(alpha_q, post_alpha, post_beta))
            upper = float(scipy_stats.beta.ppf(1 - alpha_q, post_alpha, post_beta))
            post_mean = post_alpha / (post_alpha + post_beta)

            return CIResult({
                "posterior_mean": float(post_mean),
                "posterior_alpha": float(post_alpha),
                "posterior_beta": float(post_beta),
                "lower": float(lower),
                "upper": float(upper),
                "ci_width": float(upper - lower),
                "model": "beta_binomial",
                "prior_alpha": float(prior_alpha),
                "prior_beta": float(prior_beta),
                "successes": int(successes),
                "failures": int(failures),
                "n": int(n),
                "confidence_level": cl,
            })
        else:
            logger.warning("Unknown model %s for credible interval, using normal.", model)
            return self.credible_interval(data, prior_alpha, prior_beta, model="normal", confidence=cl)


# ---------------------------------------------------------------------------
# 6. ReproducibilityReporter
# ---------------------------------------------------------------------------


class ReproducibilityReporter:
    """Generates reports and visualisation data for reproducibility analysis."""

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        self.config = config or ReproducibilityConfig()
        self._variance_analyzer = InterRunVarianceAnalyzer(self.config)
        self._seed_analyzer = SeedSensitivityAnalyzer(self.config)
        self._ci_computer = ConfidenceIntervalComputer(self.config)

    def generate_summary_statistics(
        self, results: List[RunResult]
    ) -> Dict[str, Any]:
        """Comprehensive summary statistics for all metrics across runs."""
        if not results:
            return {"error": "no_results"}

        variance_stats = self._variance_analyzer.compute_variance_across_runs(results)
        cv_stats = self._variance_analyzer.compute_coefficient_of_variation(results)

        summary: Dict[str, Any] = {"n_runs": len(results), "metrics": {}}

        for metric_name, var_val in variance_stats.items():
            values = np.array(
                [r.metrics[metric_name] for r in results if metric_name in r.metrics],
                dtype=np.float64,
            )
            ci = self._ci_computer.normal_ci(values)
            boot_ci = self._ci_computer.bootstrap_ci(values, method="percentile")

            summary["metrics"][metric_name] = {
                "variance": var_val,
                "cv": cv_stats.get(metric_name, float("nan")),
                "normal_ci": ci,
                "bootstrap_ci": boot_ci,
            }

        # Timing summary
        elapsed = [r.timing.get("elapsed_seconds", 0) for r in results]
        if elapsed:
            summary["timing"] = {
                "mean_elapsed": float(np.mean(elapsed)),
                "std_elapsed": float(np.std(elapsed, ddof=1)) if len(elapsed) > 1 else 0.0,
                "total_elapsed": float(np.sum(elapsed)),
            }

        # Seeds used
        summary["seeds_used"] = [r.seed for r in results]
        summary["failed_runs"] = sum(1 for r in results if r.status != "completed")
        return summary

    def compute_reproducibility_score(
        self, results: List[RunResult]
    ) -> Dict[str, Any]:
        """Overall 0–1 reproducibility score aggregating multiple facets.

        Components:
            1. Stability score (from InterRunVarianceAnalyzer)
            2. Seed consistency score (low sensitivity)
            3. CI tightness score
            4. Effect-size negligibility score
            5. Agreement score (ICC)
        """
        if not results or len(results) < 2:
            return {"score": 0.0, "components": {}, "grade": "F"}

        stability = self._variance_analyzer.compute_stability_score(results)
        stability_score = stability.get("overall", 0.0)

        # Seed consistency
        sensitivity = self._seed_analyzer.analyze_seed_sensitivity(results)
        sens_vals = [v.get("sensitivity_index", 0.0) for v in sensitivity.values()]
        finite_sens = [s for s in sens_vals if np.isfinite(s)]
        mean_sensitivity = np.mean(finite_sens) if finite_sens else 1.0
        seed_score = max(0.0, 1.0 - min(mean_sensitivity, 1.0))

        # CI tightness: how narrow are the CIs relative to the mean?
        ci_scores_list: List[float] = []
        for m in self._collect_metric_names(results):
            values = np.array(
                [r.metrics[m] for r in results if m in r.metrics], dtype=np.float64
            )
            if values.size < 2:
                continue
            ci = self._ci_computer.normal_ci(values)
            mean_abs = abs(ci["mean"])
            if mean_abs > 1e-12:
                relative_width = ci["ci_width"] / mean_abs
                ci_scores_list.append(max(0.0, 1.0 - min(relative_width, 1.0)))
            else:
                ci_scores_list.append(0.5)
        ci_score = float(np.mean(ci_scores_list)) if ci_scores_list else 0.5

        # Agreement (ICC)
        iccs = self._variance_analyzer.compute_intraclass_correlation(results)
        icc_vals = [v for v in iccs.values() if not math.isnan(v)]
        agreement_score = float(np.mean(icc_vals)) if icc_vals else 0.5
        agreement_score = max(0.0, min(agreement_score, 1.0))

        # Weighted combination
        weights = {"stability": 0.30, "seed": 0.25, "ci": 0.25, "agreement": 0.20}
        total = (
            weights["stability"] * stability_score
            + weights["seed"] * seed_score
            + weights["ci"] * ci_score
            + weights["agreement"] * agreement_score
        )
        total = float(np.clip(total, 0.0, 1.0))

        # Letter grade
        if total >= 0.9:
            grade = "A"
        elif total >= 0.8:
            grade = "B"
        elif total >= 0.7:
            grade = "C"
        elif total >= 0.6:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": total,
            "grade": grade,
            "components": {
                "stability": stability_score,
                "seed_consistency": seed_score,
                "ci_tightness": ci_score,
                "agreement": agreement_score,
            },
            "weights": weights,
            "n_runs": len(results),
        }

    def generate_comparison_table(
        self,
        results_groups,
        results_b: Optional[List[RunResult]] = None,
    ) -> Dict[str, Any]:
        """Compare reproducibility across runs or multiple experimental configurations.

        Accepts either a list of RunResult or a Dict[str, List[RunResult]].
        If two lists are provided, compares them as two groups.
        """
        if results_b is not None:
            results_groups = {"group_a": results_groups, "group_b": results_b}
        if isinstance(results_groups, list):
            # Single list of results: compute per-metric summary
            results = results_groups
            metric_names: Set[str] = set()
            for r in results:
                metric_names.update(r.metrics.keys())
            table: Dict[str, Any] = {}
            for m in sorted(metric_names):
                values = np.array(
                    [r.metrics[m] for r in results if m in r.metrics], dtype=np.float64
                )
                if values.size > 0:
                    table[m] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n": int(values.size),
                    }
            return table
        table_rows: List[Dict[str, Any]] = []
        metric_names: Set[str] = set()
        for group_results in results_groups.values():
            for r in group_results:
                metric_names.update(r.metrics.keys())
        metric_names_list = sorted(metric_names)

        for group_name, group_results in results_groups.items():
            row: Dict[str, Any] = {"group": group_name, "n_runs": len(group_results)}
            repro = self.compute_reproducibility_score(group_results)
            row["repro_score"] = repro["score"]
            row["grade"] = repro["grade"]

            cvs = self._variance_analyzer.compute_coefficient_of_variation(group_results)
            for m in metric_names_list:
                values = np.array(
                    [r.metrics[m] for r in group_results if m in r.metrics],
                    dtype=np.float64,
                )
                if values.size > 0:
                    row[f"{m}_mean"] = float(np.mean(values))
                    row[f"{m}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                    row[f"{m}_cv"] = cvs.get(m, float("nan"))
                else:
                    row[f"{m}_mean"] = float("nan")
                    row[f"{m}_std"] = float("nan")
                    row[f"{m}_cv"] = float("nan")
            table_rows.append(row)

        # Rank groups by reproducibility score
        table_rows.sort(key=lambda x: x.get("repro_score", 0), reverse=True)
        for rank, row in enumerate(table_rows, 1):
            row["rank"] = rank

        return {
            "table": table_rows,
            "metric_names": metric_names_list,
            "n_groups": len(results_groups),
        }

    def flag_unreliable_metrics(
        self,
        results: List[RunResult],
        cv_threshold: float = 0.15,
        icc_threshold: float = 0.60,
    ) -> Dict[str, Any]:
        """Identify metrics that may be unreliable based on multiple criteria.

        A metric is flagged if ANY of:
            - CV > cv_threshold
            - ICC < icc_threshold
            - Friedman test is significant
            - Any pairwise effect size > 0.5 (medium)
        """
        cvs = self._variance_analyzer.compute_coefficient_of_variation(results)
        iccs = self._variance_analyzer.compute_intraclass_correlation(results)
        friedman = self._variance_analyzer.friedman_test_across_runs(results)
        effects = self._variance_analyzer.compute_effect_sizes(results)

        flags: Dict[str, Dict[str, Any]] = {}
        reliable_metrics: List[str] = []
        unreliable_metrics: List[str] = []

        for m in self._collect_metric_names(results):
            reasons: List[str] = []
            severity: float = 0.0

            cv_val = cvs.get(m, 0.0)
            if cv_val > cv_threshold:
                reasons.append(f"high_cv ({cv_val:.4f} > {cv_threshold})")
                severity += min(cv_val / cv_threshold, 3.0)

            icc_val = iccs.get(m, 1.0)
            if not math.isnan(icc_val) and icc_val < icc_threshold:
                reasons.append(f"low_icc ({icc_val:.4f} < {icc_threshold})")
                severity += (icc_threshold - icc_val) / icc_threshold

            f_info = friedman.get(m, {})
            if f_info.get("significant", False):
                p = f_info.get("p_value", 1.0)
                reasons.append(f"friedman_significant (p={p:.4f})")
                severity += 1.0

            e_info = effects.get(m, {})
            if e_info.get("any_medium_effect", False):
                d = e_info.get("max_cohens_d", 0.0)
                reasons.append(f"medium_effect_size (d={d:.4f})")
                severity += d

            if reasons:
                flags[m] = {
                    "reasons": reasons,
                    "severity": float(severity),
                    "reliable": False,
                    "cv": float(cv_val),
                    "icc": float(icc_val) if not math.isnan(icc_val) else None,
                }
                unreliable_metrics.append(m)
            else:
                flags[m] = {
                    "reasons": [],
                    "severity": 0.0,
                    "reliable": True,
                    "cv": float(cv_val),
                    "icc": float(icc_val) if not math.isnan(icc_val) else None,
                }
                reliable_metrics.append(m)

        return {
            "flags": flags,
            "reliable_metrics": reliable_metrics,
            "unreliable_metrics": unreliable_metrics,
            "n_reliable": len(reliable_metrics),
            "n_unreliable": len(unreliable_metrics),
            "cv_threshold": cv_threshold,
            "icc_threshold": icc_threshold,
        }

    def prepare_variance_plot_data(
        self, results: List[RunResult]
    ) -> Dict[str, Any]:
        """Prepare data structures suitable for plotting variance analysis.

        Returns data for box plots, violin plots, and variance decomposition bars.
        """
        metric_names = self._collect_metric_names(results)
        box_data: Dict[str, List[float]] = {}
        decomposition_data: Dict[str, Dict[str, float]] = {}

        var_decomp = self._variance_analyzer.variance_decomposition(results)

        for m in metric_names:
            values = [r.metrics[m] for r in results if m in r.metrics]
            box_data[m] = values

            vd = var_decomp.get(m, {})
            decomposition_data[m] = {
                "between_pct": vd.get("between_run_pct", 0.0),
                "within_pct": vd.get("within_run_pct", 0.0),
            }

        # Data for seed-metric heatmap
        heatmap_seeds: List[int] = [r.seed for r in results]
        heatmap_matrix: Dict[str, List[float]] = {}
        for m in metric_names:
            heatmap_matrix[m] = [r.metrics.get(m, float("nan")) for r in results]

        # CV bar chart data
        cvs = self._variance_analyzer.compute_coefficient_of_variation(results)

        return {
            "box_plot": {
                "metrics": metric_names,
                "data": box_data,
            },
            "variance_decomposition": {
                "metrics": metric_names,
                "data": decomposition_data,
            },
            "heatmap": {
                "seeds": heatmap_seeds,
                "metrics": metric_names,
                "matrix": heatmap_matrix,
            },
            "cv_bar_chart": {
                "metrics": metric_names,
                "cv_values": [cvs.get(m, 0.0) for m in metric_names],
            },
            "n_runs": len(results),
        }

    def prepare_seed_sensitivity_plot_data(
        self, results: List[RunResult]
    ) -> Dict[str, Any]:
        """Prepare data for seed sensitivity visualisations.

        Returns:
            - line chart data: metric value vs seed
            - tornado chart: influence of each seed
            - cluster dendrogram data
            - correlation heatmap data
        """
        metric_names = self._collect_metric_names(results)
        seeds = [r.seed for r in results]

        # Line chart
        line_data: Dict[str, Dict[str, List]] = {}
        for m in metric_names:
            xs = []
            ys = []
            for r in results:
                if m in r.metrics:
                    xs.append(r.seed)
                    ys.append(r.metrics[m])
            line_data[m] = {"seeds": xs, "values": ys}

        # Influence (tornado)
        influence = self._seed_analyzer.compute_seed_influence(results)
        tornado_data: Dict[str, Dict[int, float]] = {}
        for m in metric_names:
            if m in influence:
                tornado_data[m] = influence[m].get("influence_scores", {})
            else:
                tornado_data[m] = {}

        # Correlation heatmap
        corr_info = self._seed_analyzer.compute_seed_correlations(results)

        # Cluster info
        cluster_info = self._seed_analyzer.seed_clustering(results)

        # Outlier markers
        outlier_info = self._seed_analyzer.identify_outlier_seeds(results)

        return {
            "line_chart": {
                "metrics": metric_names,
                "data": line_data,
            },
            "tornado_chart": {
                "metrics": metric_names,
                "data": tornado_data,
            },
            "correlation_heatmap": {
                "seeds": seeds,
                "pearson": corr_info.get("pearson", []),
                "spearman": corr_info.get("spearman", []),
            },
            "clusters": {
                "labels": cluster_info.get("labels", []),
                "seeds": cluster_info.get("seeds", []),
                "n_clusters": cluster_info.get("n_clusters", 0),
                "silhouette": cluster_info.get("mean_silhouette", 0.0),
            },
            "outliers": {
                "overall_outlier_seeds": outlier_info.get("overall_outlier_seeds", []),
                "frequency": outlier_info.get("outlier_frequency", {}),
            },
            "n_runs": len(results),
        }

    # ---- private helpers --------------------------------------------------

    def _collect_metric_names(self, results: List[RunResult]) -> List[str]:
        names: Set[str] = set()
        for r in results:
            names.update(r.metrics.keys())
        tracked = self.config.metrics_to_track
        if tracked:
            filtered = [n for n in tracked if n in names]
            if filtered:
                return filtered
        return sorted(names)


# ---------------------------------------------------------------------------
# 7. ExperimentReplicator
# ---------------------------------------------------------------------------


class ExperimentReplicator:
    """Manages deterministic replication of experiments."""

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        self.config = config or ReproducibilityConfig()
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._variance_analyzer = InterRunVarianceAnalyzer(self.config)

    def setup_deterministic_env(self, seed: int) -> Dict[str, Any]:
        """Set all random seeds and configure deterministic behaviour.

        Sets seeds for: random, numpy, and (if available) torch/tensorflow.
        Returns a dict of the actions taken.
        """
        actions: List[str] = []

        # Python stdlib
        random.seed(seed)
        actions.append("python_random")

        # NumPy
        np.random.seed(seed)
        actions.append("numpy")

        # Hash seed (informational)
        os.environ["PYTHONHASHSEED"] = str(seed)
        actions.append("pythonhashseed")

        # Try torch
        try:
            import torch  # type: ignore[import-untyped]
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                actions.append("torch_cuda")
            if self.config.deterministic_cudnn and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                actions.append("cudnn_deterministic")
            if self.config.deterministic_algorithms:
                try:
                    torch.use_deterministic_algorithms(True)
                    actions.append("torch_deterministic_algorithms")
                except Exception:
                    pass
            actions.append("torch")
        except ImportError:
            pass

        # Try tensorflow
        try:
            import tensorflow as tf  # type: ignore[import-untyped]
            tf.random.set_seed(seed)
            actions.append("tensorflow")
        except ImportError:
            pass

        env_info = {
            "seed": seed,
            "actions": actions,
            "timestamp": time.time(),
            "numpy_version": np.__version__,
        }
        logger.info("Deterministic environment set up with seed %d: %s", seed, actions)
        return env_info

    def run_replicated_experiment(
        self,
        experiment_fn: Callable[[int], Dict[str, Any]],
        seeds: Optional[List[int]] = None,
        n_runs: Optional[int] = None,
    ) -> List[RunResult]:
        """Run an experiment function multiple times with different seeds.

        Args:
            experiment_fn: callable(seed) -> dict with keys 'metrics', 'outputs'
            seeds: explicit list of seeds; if None, use config.seeds
            n_runs: override number of runs

        Returns:
            List of RunResult objects, one per seed.
        """
        seeds_to_use = seeds or self.config.seeds
        if n_runs is not None:
            seeds_to_use = seeds_to_use[:n_runs]

        results: List[RunResult] = []

        for idx, seed in enumerate(seeds_to_use):
            logger.info("Running replicated experiment %d/%d with seed %d", idx + 1, len(seeds_to_use), seed)

            env_info = self.setup_deterministic_env(seed)
            t_start = time.time()

            try:
                setup_end = time.time()
                output = experiment_fn(seed)
                t_end = time.time()

                metrics = output.get("metrics", {})
                outputs = output.get("outputs", [])
                per_sample = output.get("per_sample_metrics", None)
                metadata = output.get("metadata", {})
                metadata["env_info"] = env_info

                result = RunResult(
                    seed=seed,
                    metrics=metrics,
                    outputs=outputs,
                    timing={
                        "start_time": t_start,
                        "end_time": t_end,
                        "elapsed_seconds": t_end - t_start,
                        "setup_seconds": setup_end - t_start,
                        "inference_seconds": t_end - setup_end,
                        "evaluation_seconds": 0.0,
                    },
                    metadata=metadata,
                    run_index=idx,
                    status="completed",
                    per_sample_metrics=per_sample,
                )
            except Exception as exc:
                t_end = time.time()
                logger.error("Experiment run %d (seed=%d) failed: %s", idx, seed, exc)
                result = RunResult(
                    seed=seed,
                    metrics={},
                    outputs=[],
                    timing={
                        "start_time": t_start,
                        "end_time": t_end,
                        "elapsed_seconds": t_end - t_start,
                        "setup_seconds": 0.0,
                        "inference_seconds": 0.0,
                        "evaluation_seconds": 0.0,
                    },
                    metadata={"error": str(exc)},
                    run_index=idx,
                    status="failed",
                    error_message=str(exc),
                )

            results.append(result)

            # Auto-checkpoint if configured
            if self.config.checkpoint_dir:
                self.checkpoint_and_restore(
                    checkpoint_id=f"run_{idx}_seed_{seed}",
                    data={"result": result.to_dict()},
                    action="save",
                )

        logger.info(
            "Completed %d/%d runs successfully.",
            sum(1 for r in results if r.status == "completed"),
            len(results),
        )
        return results

    def compare_replications(
        self,
        results_a: List[RunResult],
        results_b: Optional[List[RunResult]] = None,
    ) -> Dict[str, Any]:
        """Compare replicated results for consistency.

        If only results_a is provided, computes summary statistics across runs.
        If both are provided, compares the two sets for equivalence.
        """
        if results_b is None:
            # Single-set mode: compute summary stats
            metric_names: Set[str] = set()
            for r in results_a:
                metric_names.update(r.metrics.keys())
            comparisons: Dict[str, Any] = {}
            for m in sorted(metric_names):
                vals = np.array(
                    [r.metrics[m] for r in results_a if m in r.metrics], dtype=np.float64
                )
                if vals.size < 2:
                    continue
                comparisons[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "cv": float(np.std(vals, ddof=1) / np.abs(np.mean(vals))) if np.abs(np.mean(vals)) > 1e-12 else float("inf"),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "range": float(np.ptp(vals)),
                    "n": int(vals.size),
                    "consistent": bool(np.std(vals, ddof=1) / (np.abs(np.mean(vals)) + 1e-12) < 0.1),
                }
            comparisons["consistent"] = all(
                v.get("consistent", True) for v in comparisons.values() if isinstance(v, dict)
            )
            return comparisons

        metric_names_set: Set[str] = set()
        for r in results_a + results_b:
            metric_names_set.update(r.metrics.keys())
        metric_names_list = sorted(metric_names_set)

        comparisons: Dict[str, Dict[str, Any]] = {}

        for m in metric_names_list:
            vals_a = np.array(
                [r.metrics[m] for r in results_a if m in r.metrics], dtype=np.float64
            )
            vals_b = np.array(
                [r.metrics[m] for r in results_b if m in r.metrics], dtype=np.float64
            )

            if vals_a.size < 2 or vals_b.size < 2:
                comparisons[m] = {"error": "insufficient_data"}
                continue

            # Descriptive
            mean_a, mean_b = float(np.mean(vals_a)), float(np.mean(vals_b))
            std_a, std_b = float(np.std(vals_a, ddof=1)), float(np.std(vals_b, ddof=1))

            # Welch's t-test
            t_stat, t_p = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)

            # Mann-Whitney
            try:
                u_stat, u_p = scipy_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            except ValueError:
                u_stat, u_p = float("nan"), float("nan")

            # KS test
            ks_stat, ks_p = scipy_stats.ks_2samp(vals_a, vals_b)

            # Levene's test for equal variances
            lev_stat, lev_p = scipy_stats.levene(vals_a, vals_b)

            # Effect size
            n1, n2 = vals_a.size, vals_b.size
            sp = np.sqrt(((n1 - 1) * std_a ** 2 + (n2 - 1) * std_b ** 2) / (n1 + n2 - 2))
            cohens_d = (mean_a - mean_b) / sp if sp > 1e-15 else 0.0

            # Equivalence test (TOST)
            equiv_margin = self.config.min_effect_size * sp if sp > 1e-15 else 0.1
            se_diff = np.sqrt(std_a ** 2 / n1 + std_b ** 2 / n2)
            df_welch = (std_a ** 2 / n1 + std_b ** 2 / n2) ** 2 / (
                (std_a ** 2 / n1) ** 2 / (n1 - 1) + (std_b ** 2 / n2) ** 2 / (n2 - 1)
            ) if se_diff > 1e-15 else n1 + n2 - 2

            diff = mean_a - mean_b
            if se_diff > 1e-15:
                t_lower = (diff - (-equiv_margin)) / se_diff
                t_upper = (diff - equiv_margin) / se_diff
                p_lower = scipy_stats.t.cdf(t_lower, df=df_welch)  # want this to be small? No, want t_lower large
                p_upper = 1.0 - scipy_stats.t.cdf(t_upper, df=df_welch)
                tost_p = max(1.0 - p_lower, p_upper)
            else:
                tost_p = 0.0

            alpha = 1.0 - self.config.confidence_level
            equivalent = tost_p < alpha
            different = t_p < alpha

            comparisons[m] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": std_a,
                "std_b": std_b,
                "mean_diff": float(diff),
                "cohens_d": float(cohens_d),
                "welch_t": float(t_stat),
                "welch_p": float(t_p),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_p": float(u_p),
                "ks_statistic": float(ks_stat),
                "ks_p": float(ks_p),
                "levene_statistic": float(lev_stat),
                "levene_p": float(lev_p),
                "tost_p": float(tost_p),
                "equivalent": bool(equivalent),
                "significantly_different": bool(different),
                "conclusion": "equivalent" if equivalent else ("different" if different else "inconclusive"),
                "n_a": int(n1),
                "n_b": int(n2),
            }

        # Overall verdict
        conclusions = [c.get("conclusion", "inconclusive") for c in comparisons.values()]
        n_equiv = conclusions.count("equivalent")
        n_diff = conclusions.count("different")
        n_inc = conclusions.count("inconclusive")

        if n_diff == 0 and n_equiv > 0:
            overall = "replications_equivalent"
        elif n_diff > n_equiv:
            overall = "replications_differ"
        else:
            overall = "mixed"

        return {
            "comparisons": comparisons,
            "overall_conclusion": overall,
            "n_equivalent": n_equiv,
            "n_different": n_diff,
            "n_inconclusive": n_inc,
            "n_metrics": len(comparisons),
        }

    def checkpoint_and_restore(
        self,
        checkpoint_id: str,
        data: Optional[Dict[str, Any]] = None,
        action: str = "save",
    ) -> Dict[str, Any]:
        """Save or restore experiment state.

        Args:
            checkpoint_id: unique identifier for the checkpoint
            data: data to save (ignored for 'load' action)
            action: 'save' or 'load'
        """
        if action == "save":
            if data is None:
                return {"status": "error", "message": "no data to save"}

            checkpoint_data = {
                "id": checkpoint_id,
                "data": data,
                "timestamp": time.time(),
                "checksum": hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest(),
            }
            self._checkpoints[checkpoint_id] = checkpoint_data

            # Persist to disk if checkpoint_dir is set
            if self.config.checkpoint_dir:
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                path = os.path.join(self.config.checkpoint_dir, f"{checkpoint_id}.json")
                try:
                    with open(path, "w") as f:
                        json.dump(checkpoint_data, f, indent=2, default=str)
                    logger.info("Checkpoint saved to %s", path)
                except Exception as exc:
                    logger.error("Failed to save checkpoint: %s", exc)
                    return {"status": "error", "message": str(exc)}

            return {"status": "saved", "id": checkpoint_id, "checksum": checkpoint_data["checksum"]}

        elif action == "load":
            # Try in-memory first
            if checkpoint_id in self._checkpoints:
                cp = self._checkpoints[checkpoint_id]
                return {"status": "loaded", "data": cp["data"], "checksum": cp["checksum"]}

            # Try disk
            if self.config.checkpoint_dir:
                path = os.path.join(self.config.checkpoint_dir, f"{checkpoint_id}.json")
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            cp = json.load(f)
                        self._checkpoints[checkpoint_id] = cp
                        return {"status": "loaded", "data": cp["data"], "checksum": cp.get("checksum")}
                    except Exception as exc:
                        return {"status": "error", "message": str(exc)}

            return {"status": "not_found", "id": checkpoint_id}

        elif action == "list":
            checkpoints_list = []
            for cp_id, cp_data in self._checkpoints.items():
                checkpoints_list.append({
                    "id": cp_id,
                    "timestamp": cp_data.get("timestamp"),
                    "checksum": cp_data.get("checksum"),
                })
            return {"status": "ok", "checkpoints": checkpoints_list, "count": len(checkpoints_list)}

        elif action == "delete":
            if checkpoint_id in self._checkpoints:
                del self._checkpoints[checkpoint_id]
            if self.config.checkpoint_dir:
                path = os.path.join(self.config.checkpoint_dir, f"{checkpoint_id}.json")
                if os.path.exists(path):
                    os.remove(path)
            return {"status": "deleted", "id": checkpoint_id}

        else:
            return {"status": "error", "message": f"unknown action: {action}"}


# ---------------------------------------------------------------------------
# 8. Helper functions
# ---------------------------------------------------------------------------


def bootstrap_resample(
    data: np.ndarray,
    statistic: Callable = None,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = 42,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Perform bootstrap resampling and return a single resampled array.

    Args:
        data: 1-D array of observations
        seed: seed for reproducibility

    Returns:
        A bootstrap resampled array of the same size as data.
    """
    if seed is not None:
        random_state = seed
    data = np.asarray(data, dtype=np.float64).ravel()
    n = data.size

    if n == 0:
        return np.array([], dtype=np.float64)

    rng = np.random.RandomState(seed=random_state)
    idx = rng.randint(0, n, size=n)
    return data[idx]


def jackknife_resample(
    data: np.ndarray,
    statistic: Callable = np.mean,
) -> List[np.ndarray]:
    """Leave-one-out jackknife resampling.

    Args:
        data: 1-D array of observations

    Returns:
        List of n arrays, each with one element removed.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n = data.size
    return [np.delete(data, i) for i in range(n)]


def compute_effective_sample_size(
    data: np.ndarray,
    max_lag: Optional[int] = None,
) -> float:
    """Compute effective sample size accounting for autocorrelation.

    Args:
        data: 1-D time series of observations
        max_lag: maximum lag to consider; defaults to n//2

    Returns:
        The effective sample size as a float.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[~np.isnan(data)]
    n = data.size

    if n < 4:
        return float(n)

    if max_lag is None:
        max_lag = n // 2

    # Compute autocorrelation function
    mean_val = np.mean(data)
    centered = data - mean_val
    var_val = np.var(data)
    if var_val < 1e-15:
        return float(n)

    acf = np.correlate(centered, centered, mode="full")
    acf = acf[n - 1:]  # take positive lags
    acf = acf / (var_val * n)
    acf = acf[:max_lag + 1]

    # Geyer's initial positive sequence estimator
    tau = acf[0]  # = 1.0
    for lag in range(1, len(acf) - 1, 2):
        pair_sum = acf[lag] + acf[lag + 1] if lag + 1 < len(acf) else acf[lag]
        if pair_sum < 0:
            break
        tau += 2 * pair_sum

    tau = max(tau, 1.0)
    n_eff = n / tau

    return float(n_eff)


def cochrans_q_test(
    binary_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Cochran's Q test for equality of proportions across related groups.

    Input: (n_subjects × k_treatments) binary matrix.
    Tests H0: all treatments have the same proportion of successes.

    Args:
        binary_matrix: 2-D array of shape (n, k) with binary values

    Returns:
        Dict with Q statistic, p-value, df, pairwise McNemar tests
    """
    matrix = np.asarray(binary_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("binary_matrix must be 2-dimensional")

    n, k = matrix.shape
    if n < 2 or k < 2:
        return {"q_statistic": float("nan"), "p_value": float("nan"), "df": 0}

    # Row and column totals
    row_totals = np.sum(matrix, axis=1)  # L_i
    col_totals = np.sum(matrix, axis=0)  # T_j

    grand_total = np.sum(matrix)
    T_bar = grand_total / k

    # Q statistic
    numerator = (k - 1) * (k * np.sum(col_totals ** 2) - grand_total ** 2)
    denominator = k * grand_total - np.sum(row_totals ** 2)

    if abs(denominator) < 1e-15:
        q_stat = 0.0
        p_value = 1.0
    else:
        q_stat = float(numerator / denominator)
        df = k - 1
        p_value = float(1.0 - scipy_stats.chi2.cdf(q_stat, df=df))

    # Pairwise McNemar tests (post-hoc)
    pairwise: Dict[str, Dict[str, float]] = {}
    for i in range(k):
        for j in range(i + 1, k):
            # 2×2 table of discordant pairs
            b = np.sum((matrix[:, i] == 1) & (matrix[:, j] == 0))
            c = np.sum((matrix[:, i] == 0) & (matrix[:, j] == 1))

            if b + c > 0:
                # McNemar with continuity correction
                mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
                mcnemar_p = float(1.0 - scipy_stats.chi2.cdf(mcnemar_stat, df=1))
            else:
                mcnemar_stat = 0.0
                mcnemar_p = 1.0

            pairwise[f"treatment_{i}_vs_{j}"] = {
                "mcnemar_statistic": float(mcnemar_stat),
                "mcnemar_p_value": float(mcnemar_p),
                "discordant_b": int(b),
                "discordant_c": int(c),
            }

    return {
        "statistic": float(q_stat),
        "q_statistic": float(q_stat),
        "p_value": float(p_value),
        "df": int(k - 1),
        "significant": bool(p_value < 0.05),
        "n_subjects": int(n),
        "k_treatments": int(k),
        "treatment_proportions": col_totals.tolist(),
        "pairwise_mcnemar": pairwise,
    }


def compute_krippendorffs_alpha(
    reliability_data: np.ndarray,
    level_of_measurement: str = "interval",
) -> float:
    """Compute Krippendorff's alpha reliability coefficient.

    Args:
        reliability_data: (n_raters × n_units) matrix of ratings.
            NaN indicates missing ratings.
        level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio'

    Returns:
        The alpha coefficient as a float.
    """
    data = np.asarray(reliability_data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("reliability_data must be 2-dimensional")

    n_raters, n_units = data.shape

    # Collect all non-missing values per unit
    units: List[np.ndarray] = []
    for u in range(n_units):
        vals = data[:, u]
        valid = vals[~np.isnan(vals)]
        if valid.size >= 2:
            units.append(valid)

    if not units:
        return float("nan")

    # Collect all unique values
    all_values = np.concatenate(units)
    n_total = all_values.size

    if n_total < 2:
        return float("nan")

    # Distance function
    def _distance(v1: float, v2: float) -> float:
        if level_of_measurement == "nominal":
            return 0.0 if abs(v1 - v2) < 1e-10 else 1.0
        elif level_of_measurement == "ordinal":
            # Ordinal distances require rank information
            return (v1 - v2) ** 2
        elif level_of_measurement == "interval":
            return (v1 - v2) ** 2
        elif level_of_measurement == "ratio":
            s = v1 + v2
            if abs(s) < 1e-15:
                return 0.0
            return ((v1 - v2) / s) ** 2
        else:
            return (v1 - v2) ** 2

    # Observed disagreement
    d_o = 0.0
    n_pairable = 0.0
    for unit_vals in units:
        m_u = unit_vals.size
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(i + 1, m_u):
                d_o += _distance(unit_vals[i], unit_vals[j])
        n_pairable += m_u * (m_u - 1) / 2.0

    if n_pairable < 1:
        return float("nan")

    d_o /= n_pairable

    # Expected disagreement (across all pairs from the marginal distribution)
    d_e = 0.0
    n_total_pairs = n_total * (n_total - 1) / 2.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d_e += _distance(all_values[i], all_values[j])
    d_e /= n_total_pairs

    # Alpha
    if d_e < 1e-15:
        alpha = 1.0  # perfect agreement when no variation
    else:
        alpha = 1.0 - d_o / d_e

    # Bootstrap CI for alpha
    rng = np.random.RandomState(42)
    n_boot = 1000
    boot_alphas = np.empty(n_boot, dtype=np.float64)
    unit_indices = list(range(len(units)))

    for b in range(n_boot):
        boot_idx = rng.choice(unit_indices, size=len(unit_indices), replace=True)
        boot_units = [units[i] for i in boot_idx]
        boot_all = np.concatenate(boot_units)
        n_boot_total = boot_all.size

        b_do = 0.0
        b_np = 0.0
        for bu in boot_units:
            m_u = bu.size
            if m_u < 2:
                continue
            for ii in range(m_u):
                for jj in range(ii + 1, m_u):
                    b_do += _distance(bu[ii], bu[jj])
            b_np += m_u * (m_u - 1) / 2.0

        if b_np < 1:
            boot_alphas[b] = float("nan")
            continue
        b_do /= b_np

        b_de = 0.0
        b_total_pairs = n_boot_total * (n_boot_total - 1) / 2.0
        # Use random subsample for efficiency
        if n_boot_total > 200:
            sub_idx = rng.choice(n_boot_total, size=200, replace=False)
            sub_vals = boot_all[sub_idx]
            b_total_pairs_sub = 200 * 199 / 2.0
            for ii in range(200):
                for jj in range(ii + 1, 200):
                    b_de += _distance(sub_vals[ii], sub_vals[jj])
            b_de /= b_total_pairs_sub
        else:
            for ii in range(n_boot_total):
                for jj in range(ii + 1, n_boot_total):
                    b_de += _distance(boot_all[ii], boot_all[jj])
            b_de /= max(b_total_pairs, 1)

        if b_de < 1e-15:
            boot_alphas[b] = 1.0
        else:
            boot_alphas[b] = 1.0 - b_do / b_de

    valid_boot = boot_alphas[~np.isnan(boot_alphas)]

    return float(alpha)


def compute_fleiss_kappa(
    ratings_matrix: np.ndarray,
    categories: Optional[List[int]] = None,
) -> float:
    """Compute Fleiss' kappa for inter-rater reliability.

    Args:
        ratings_matrix: (n_subjects × n_categories) matrix where entry (i, j)
            is the number of raters who assigned subject i to category j.
        categories: list of category values (auto-detected if None)

    Returns:
        The kappa coefficient as a float.
    """
    matrix = np.asarray(ratings_matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("ratings_matrix must be 2-dimensional")

    # Detect whether this is already a count matrix or raw ratings
    # If all values are non-negative integers and rows sum to the same value,
    # assume it's a count matrix
    row_sums = np.sum(matrix, axis=1)
    is_count_matrix = (
        np.all(matrix >= 0)
        and np.allclose(matrix, np.round(matrix))
        and np.allclose(row_sums, row_sums[0])
        and row_sums[0] > 1
    )

    if not is_count_matrix:
        # Convert raw ratings (n_raters × n_subjects) to count matrix
        if categories is None:
            all_vals = matrix[~np.isnan(matrix)]
            categories = sorted(set(int(v) for v in all_vals))

        n_raters, n_subjects = matrix.shape
        count_matrix = np.zeros((n_subjects, len(categories)), dtype=np.float64)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        for j in range(n_subjects):
            for i in range(n_raters):
                val = matrix[i, j]
                if not np.isnan(val) and int(val) in cat_to_idx:
                    count_matrix[j, cat_to_idx[int(val)]] += 1
        matrix = count_matrix

    n_subjects, n_categories = matrix.shape
    n_raters_per = matrix[0].sum()  # assumed constant

    if n_subjects < 1 or n_categories < 2 or n_raters_per < 2:
        return float("nan")

    N = n_subjects
    n = n_raters_per

    # Proportion of assignments to each category
    p_j = np.sum(matrix, axis=0) / (N * n)

    # Per-subject agreement
    P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1))

    # Overall observed agreement
    P_bar = float(np.mean(P_i))

    # Expected agreement by chance
    P_e_bar = float(np.sum(p_j ** 2))

    # Fleiss' kappa
    if abs(1.0 - P_e_bar) < 1e-15:
        kappa = 1.0 if P_bar >= P_e_bar else 0.0
    else:
        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)

    # Standard error (Fleiss, 1971)
    if abs(1.0 - P_e_bar) < 1e-15:
        se = 0.0
    else:
        numerator = 2.0 / (N * n * (n - 1))
        se_sq = numerator * (P_e_bar - np.sum(p_j ** 3)) / ((1.0 - P_e_bar) ** 2)
        se = np.sqrt(max(se_sq, 0.0))

    z_score = kappa / se if se > 1e-15 else float("inf")
    p_value = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_score))) if np.isfinite(z_score) else 0.0

    # Per-category kappas
    per_category: Dict[str, Dict[str, float]] = {}
    for j in range(n_categories):
        pj = p_j[j]
        if abs(1.0 - pj) < 1e-15 or abs(pj) < 1e-15:
            kj = 1.0
        else:
            P_j_observed = np.mean(matrix[:, j] * (matrix[:, j] - 1) / (n * (n - 1)))
            P_j_expected = pj ** 2
            kj = (P_j_observed - P_j_expected) / (pj - P_j_expected) if abs(pj - P_j_expected) > 1e-15 else 0.0

        cat_label = str(categories[j]) if categories and j < len(categories) else str(j)
        per_category[cat_label] = {
            "kappa": float(kj),
            "proportion": float(pj),
        }

    # Interpretation
    if kappa >= 0.81:
        interp = "almost_perfect"
    elif kappa >= 0.61:
        interp = "substantial"
    elif kappa >= 0.41:
        interp = "moderate"
    elif kappa >= 0.21:
        interp = "fair"
    elif kappa >= 0.0:
        interp = "slight"
    else:
        interp = "poor"

    return float(kappa)


# ---------------------------------------------------------------------------
# Additional statistical utilities
# ---------------------------------------------------------------------------


def _compute_paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-15:
        return 0.0
    return float(mean_diff / std_diff)


def _compute_overlap_coefficient(
    x: np.ndarray, y: np.ndarray, n_bins: int = 50
) -> float:
    """Overlap coefficient (Szymkiewicz-Simpson) between two distributions."""
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    if hi - lo < 1e-15:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(x, bins=bins, density=True)
    h2, _ = np.histogram(y, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = np.sum(np.minimum(h1, h2)) * bin_width
    return float(np.clip(overlap, 0.0, 1.0))


def _compute_bhattacharyya_distance(
    x: np.ndarray, y: np.ndarray, n_bins: int = 50
) -> float:
    """Bhattacharyya distance between two sample distributions."""
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    if hi - lo < 1e-15:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(x, bins=bins, density=True)
    h2, _ = np.histogram(y, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    h1 = h1 * bin_width + 1e-15
    h2 = h2 * bin_width + 1e-15
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    bc = np.sum(np.sqrt(h1 * h2))
    bc = min(bc, 1.0)
    return float(-np.log(max(bc, 1e-15)))


def _compute_wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Earth mover's distance (1-Wasserstein) between two 1-D samples."""
    return float(scipy_stats.wasserstein_distance(x, y))


def _levene_brown_forsythe(
    groups: List[np.ndarray], center: str = "median"
) -> Tuple[float, float]:
    """Brown-Forsythe / Levene test for homogeneity of variance.

    Args:
        groups: list of 1-D arrays, one per group
        center: 'median' (Brown-Forsythe) or 'mean' (Levene)
    """
    k = len(groups)
    if k < 2:
        return 0.0, 1.0

    transformed: List[np.ndarray] = []
    for g in groups:
        if center == "median":
            c = np.median(g)
        else:
            c = np.mean(g)
        transformed.append(np.abs(g - c))

    stat, p = scipy_stats.f_oneway(*transformed)
    return float(stat), float(p)


def _normality_tests(data: np.ndarray) -> Dict[str, Any]:
    """Run multiple normality tests on a 1-D sample."""
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[~np.isnan(data)]
    n = data.size

    result: Dict[str, Any] = {"n": int(n)}

    if n < 3:
        result["shapiro"] = {"statistic": float("nan"), "p_value": float("nan")}
        result["dagostino"] = {"statistic": float("nan"), "p_value": float("nan")}
        result["anderson"] = {"statistic": float("nan")}
        return result

    # Shapiro-Wilk (max 5000 samples)
    subset = data[:5000] if n > 5000 else data
    try:
        sw_stat, sw_p = scipy_stats.shapiro(subset)
        result["shapiro"] = {"statistic": float(sw_stat), "p_value": float(sw_p)}
    except Exception:
        result["shapiro"] = {"statistic": float("nan"), "p_value": float("nan")}

    # D'Agostino and Pearson (n >= 8)
    if n >= 8:
        try:
            da_stat, da_p = scipy_stats.normaltest(data)
            result["dagostino"] = {"statistic": float(da_stat), "p_value": float(da_p)}
        except Exception:
            result["dagostino"] = {"statistic": float("nan"), "p_value": float("nan")}
    else:
        result["dagostino"] = {"statistic": float("nan"), "p_value": float("nan")}

    # Anderson-Darling
    try:
        ad_result = scipy_stats.anderson(data, dist="norm")
        result["anderson"] = {
            "statistic": float(ad_result.statistic),
            "critical_values": list(ad_result.critical_values),
            "significance_levels": list(ad_result.significance_level),
        }
    except Exception:
        result["anderson"] = {"statistic": float("nan")}

    # Skewness and kurtosis
    result["skewness"] = float(scipy_stats.skew(data))
    result["kurtosis"] = float(scipy_stats.kurtosis(data))

    # Jarque-Bera
    if n >= 8:
        try:
            jb_stat, jb_p = scipy_stats.jarque_bera(data)
            result["jarque_bera"] = {"statistic": float(jb_stat), "p_value": float(jb_p)}
        except Exception:
            result["jarque_bera"] = {"statistic": float("nan"), "p_value": float("nan")}

    return result


def _permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable = lambda a, b: np.mean(a) - np.mean(b),
    n_permutations: int = 10000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Two-sample permutation test.

    Args:
        x, y: two independent samples
        statistic: function(x, y) -> scalar test statistic
        n_permutations: number of permutations
        random_state: seed

    Returns:
        Dict with observed statistic, p-value, null distribution summary
    """
    rng = np.random.RandomState(random_state)
    observed = float(statistic(x, y))
    combined = np.concatenate([x, y])
    n_x = len(x)
    n_total = len(combined)

    count_extreme = 0
    perm_stats = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_x = combined[perm[:n_x]]
        perm_y = combined[perm[n_x:]]
        perm_stat = statistic(perm_x, perm_y)
        perm_stats[i] = perm_stat
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "observed_statistic": observed,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(perm_stats)),
        "null_std": float(np.std(perm_stats, ddof=1)),
        "null_ci_lower": float(np.percentile(perm_stats, 2.5)),
        "null_ci_upper": float(np.percentile(perm_stats, 97.5)),
    }


def _multi_run_summary_table(
    results: List[RunResult], metric_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a summary table of all runs × metrics."""
    if not results:
        return {"table": [], "seeds": [], "metrics": []}

    all_metrics: Set[str] = set()
    for r in results:
        all_metrics.update(r.metrics.keys())
    if metric_names:
        all_metrics = all_metrics.intersection(metric_names)
    metric_list = sorted(all_metrics)

    table: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {
            "seed": r.seed,
            "run_index": r.run_index,
            "status": r.status,
            "elapsed_seconds": r.timing.get("elapsed_seconds", 0.0),
        }
        for m in metric_list:
            row[m] = r.metrics.get(m, float("nan"))
        table.append(row)

    # Add summary row
    summary: Dict[str, Any] = {"seed": "MEAN", "run_index": -1, "status": "summary"}
    for m in metric_list:
        vals = [r.metrics[m] for r in results if m in r.metrics]
        summary[m] = float(np.mean(vals)) if vals else float("nan")
    table.append(summary)

    std_row: Dict[str, Any] = {"seed": "STD", "run_index": -1, "status": "summary"}
    for m in metric_list:
        vals = [r.metrics[m] for r in results if m in r.metrics]
        std_row[m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    table.append(std_row)

    return {
        "table": table,
        "seeds": [r.seed for r in results],
        "metrics": metric_list,
        "n_runs": len(results),
    }


def _compute_relative_efficiency(
    estimator_a_values: np.ndarray,
    estimator_b_values: np.ndarray,
) -> Dict[str, float]:
    """Compute relative efficiency of two estimators.

    RE = Var(B) / Var(A).  RE > 1 means A is more efficient.
    """
    a = np.asarray(estimator_a_values, dtype=np.float64)
    b = np.asarray(estimator_b_values, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    var_a = float(np.var(a, ddof=1)) if a.size > 1 else float("nan")
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else float("nan")

    if var_a > 1e-15:
        re = var_b / var_a
    else:
        re = float("nan")

    return {
        "relative_efficiency": float(re),
        "var_a": var_a,
        "var_b": var_b,
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
        "mse_a": float(np.mean((a - np.mean(a)) ** 2)) if a.size else float("nan"),
        "mse_b": float(np.mean((b - np.mean(b)) ** 2)) if b.size else float("nan"),
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


def _compute_agreement_indices(
    rater1: np.ndarray, rater2: np.ndarray
) -> Dict[str, float]:
    """Compute multiple agreement indices between two raters.

    Returns Pearson r, Spearman rho, Kendall tau, Lin's CCC, and
    total deviation index (TDI).
    """
    r1 = np.asarray(rater1, dtype=np.float64).ravel()
    r2 = np.asarray(rater2, dtype=np.float64).ravel()

    valid = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[valid], r2[valid]
    n = r1.size

    if n < 2:
        return {
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
            "lins_ccc": float("nan"),
            "tdi_95": float("nan"),
        }

    pearson_r, pearson_p = scipy_stats.pearsonr(r1, r2)
    spearman_rho, spearman_p = scipy_stats.spearmanr(r1, r2)
    kendall_tau, kendall_p = scipy_stats.kendalltau(r1, r2)

    # Lin's CCC
    mx, my = np.mean(r1), np.mean(r2)
    sx2 = np.var(r1, ddof=1)
    sy2 = np.var(r2, ddof=1)
    sxy = np.mean((r1 - mx) * (r2 - my))
    denom = sx2 + sy2 + (mx - my) ** 2
    ccc = 2.0 * sxy / denom if denom > 1e-15 else 1.0

    # TDI (Total Deviation Index at 95%)
    diff = r1 - r2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    tdi_95 = abs(mean_diff) + 1.96 * std_diff

    # Bland-Altman limits of agreement
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "lins_ccc": float(ccc),
        "tdi_95": float(tdi_95),
        "bland_altman_bias": float(mean_diff),
        "bland_altman_loa_lower": float(loa_lower),
        "bland_altman_loa_upper": float(loa_upper),
        "n": int(n),
    }


def _robust_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute robust summary statistics resistant to outliers."""
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[~np.isnan(data)]
    n = data.size

    if n == 0:
        return {k: float("nan") for k in [
            "median", "mad", "trimmed_mean_10", "trimmed_mean_20",
            "winsorized_mean_10", "huber_m_estimate",
            "iqr", "q1", "q3", "biweight_midvariance",
        ]}

    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    # Trimmed means
    tm_10 = float(scipy_stats.trim_mean(data, 0.10)) if n >= 4 else float(np.mean(data))
    tm_20 = float(scipy_stats.trim_mean(data, 0.20)) if n >= 6 else float(np.mean(data))

    # Winsorized mean (clip at 10th/90th percentile)
    p10, p90 = np.percentile(data, [10, 90])
    winsorized = np.clip(data, p10, p90)
    w_mean = float(np.mean(winsorized))

    # Huber M-estimate (iteratively re-weighted mean)
    huber_c = 1.345  # 95% efficiency at normal
    loc = median
    for _iteration in range(50):
        residuals = data - loc
        scale = 1.4826 * mad if mad > 1e-15 else 1.0
        u = residuals / scale
        weights = np.where(np.abs(u) <= huber_c, 1.0, huber_c / np.abs(u))
        new_loc = np.sum(weights * data) / np.sum(weights)
        if abs(new_loc - loc) < 1e-10:
            break
        loc = new_loc
    huber_est = float(loc)

    # Biweight midvariance
    if mad > 1e-15:
        u_bw = (data - median) / (9.0 * mad)
        valid_bw = np.abs(u_bw) < 1.0
        if valid_bw.sum() >= 2:
            num = np.sum(((data[valid_bw] - median) ** 2) * (1 - u_bw[valid_bw] ** 2) ** 4)
            den = np.sum((1 - u_bw[valid_bw] ** 2) * (1 - 5 * u_bw[valid_bw] ** 2))
            bw_midvar = float(n * num / (den ** 2)) if abs(den) > 1e-15 else float(np.var(data, ddof=1))
        else:
            bw_midvar = float(np.var(data, ddof=1)) if n > 1 else 0.0
    else:
        bw_midvar = 0.0

    return {
        "median": median,
        "mad": float(mad),
        "trimmed_mean_10": tm_10,
        "trimmed_mean_20": tm_20,
        "winsorized_mean_10": w_mean,
        "huber_m_estimate": huber_est,
        "iqr": float(iqr),
        "q1": float(q1),
        "q3": float(q3),
        "biweight_midvariance": bw_midvar,
        "n": int(n),
    }


def _power_analysis_paired(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n_range: Tuple[int, int] = (2, 1000),
) -> Dict[str, Any]:
    """Compute required sample size for a paired t-test.

    Uses the non-central t-distribution for exact power calculation.
    """
    from scipy.stats import nct as noncentralt  # noqa: F811

    target_power = power
    best_n = None

    for n in range(n_range[0], n_range[1] + 1):
        df = n - 1
        ncp = effect_size * np.sqrt(n)  # non-centrality parameter
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=df)
        # Power = P(reject H0) = P(|T| > t_crit | ncp)
        power_val = 1.0 - noncentralt.cdf(t_crit, df, ncp) + noncentralt.cdf(-t_crit, df, ncp)
        if power_val >= target_power:
            best_n = n
            break

    if best_n is None:
        best_n = n_range[1]

    # Compute actual power at best_n
    df = best_n - 1
    ncp = effect_size * np.sqrt(best_n)
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=df)
    actual_power = 1.0 - noncentralt.cdf(t_crit, df, ncp) + noncentralt.cdf(-t_crit, df, ncp)

    return {
        "required_n": int(best_n),
        "actual_power": float(actual_power),
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "target_power": float(target_power),
    }


def _power_analysis_independent(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    n_range: Tuple[int, int] = (2, 2000),
) -> Dict[str, Any]:
    """Compute required sample size per group for an independent t-test.

    Args:
        effect_size: Cohen's d
        alpha: significance level
        power: target power
        ratio: n2/n1 ratio
    """
    from scipy.stats import nct as noncentralt  # noqa: F811

    target_power = power
    best_n1 = None

    for n1 in range(n_range[0], n_range[1] + 1):
        n2 = max(int(n1 * ratio), 2)
        df = n1 + n2 - 2
        ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=df)
        power_val = 1.0 - noncentralt.cdf(t_crit, df, ncp) + noncentralt.cdf(-t_crit, df, ncp)
        if power_val >= target_power:
            best_n1 = n1
            break

    if best_n1 is None:
        best_n1 = n_range[1]

    n2_final = max(int(best_n1 * ratio), 2)
    df = best_n1 + n2_final - 2
    ncp = effect_size * np.sqrt(best_n1 * n2_final / (best_n1 + n2_final))
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=df)
    actual_power = 1.0 - noncentralt.cdf(t_crit, df, ncp) + noncentralt.cdf(-t_crit, df, ncp)

    return {
        "required_n1": int(best_n1),
        "required_n2": int(n2_final),
        "total_n": int(best_n1 + n2_final),
        "actual_power": float(actual_power),
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "target_power": float(target_power),
        "ratio": float(ratio),
    }


def _compute_density_overlap(
    x: np.ndarray, y: np.ndarray, n_points: int = 500
) -> Dict[str, float]:
    """Compute kernel density overlap between two samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    margin = (hi - lo) * 0.1
    grid = np.linspace(lo - margin, hi + margin, n_points)

    try:
        kde_x = scipy_stats.gaussian_kde(x)
        kde_y = scipy_stats.gaussian_kde(y)
        dx = kde_x(grid)
        dy = kde_y(grid)
    except Exception:
        return {"overlap": float("nan"), "kl_divergence": float("nan")}

    step = grid[1] - grid[0]

    # Overlap coefficient
    overlap = float(np.sum(np.minimum(dx, dy)) * step)

    # KL divergence (symmetrized)
    dx_safe = np.maximum(dx, 1e-15)
    dy_safe = np.maximum(dy, 1e-15)
    kl_xy = float(np.sum(dx * np.log(dx_safe / dy_safe)) * step)
    kl_yx = float(np.sum(dy * np.log(dy_safe / dx_safe)) * step)
    kl_sym = (kl_xy + kl_yx) / 2.0

    # Jensen-Shannon divergence
    m = (dx + dy) / 2.0
    m_safe = np.maximum(m, 1e-15)
    js = 0.5 * np.sum(dx * np.log(dx_safe / m_safe)) * step + \
         0.5 * np.sum(dy * np.log(dy_safe / m_safe)) * step

    return {
        "overlap": overlap,
        "kl_divergence_symmetric": float(kl_sym),
        "jensen_shannon_divergence": float(max(js, 0.0)),
        "hellinger_distance": float(np.sqrt(1.0 - np.sum(np.sqrt(dx * dy) * step))),
    }


def _weighted_mean_and_var(
    values: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
    """Compute weighted mean and unbiased weighted variance."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = ~(np.isnan(values) | np.isnan(weights))
    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return float("nan"), float("nan")

    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return float(np.mean(values)), float(np.var(values, ddof=1))

    w_mean = np.sum(weights * values) / w_sum

    # Bessel-corrected weighted variance
    w2_sum = np.sum(weights ** 2)
    if w_sum ** 2 - w2_sum < 1e-15:
        w_var = 0.0
    else:
        w_var = np.sum(weights * (values - w_mean) ** 2) * w_sum / (w_sum ** 2 - w2_sum)

    return float(w_mean), float(w_var)


def _meta_analysis_fixed_effects(
    effects: np.ndarray,
    standard_errors: np.ndarray,
) -> Dict[str, float]:
    """Fixed-effects meta-analysis (inverse-variance method)."""
    effects = np.asarray(effects, dtype=np.float64)
    ses = np.asarray(standard_errors, dtype=np.float64)

    valid = ~(np.isnan(effects) | np.isnan(ses) | (ses < 1e-15))
    effects = effects[valid]
    ses = ses[valid]
    k = effects.size

    if k == 0:
        return {"pooled_effect": float("nan"), "pooled_se": float("nan")}

    weights = 1.0 / (ses ** 2)
    w_sum = np.sum(weights)
    pooled = np.sum(weights * effects) / w_sum
    pooled_se = 1.0 / np.sqrt(w_sum)

    # Q statistic for heterogeneity
    q = np.sum(weights * (effects - pooled) ** 2)
    df = k - 1
    q_p = float(1.0 - scipy_stats.chi2.cdf(q, df=df)) if df > 0 else 1.0

    # I^2 statistic
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0

    z = pooled / pooled_se if pooled_se > 1e-15 else 0.0
    p_val = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))

    return {
        "pooled_effect": float(pooled),
        "pooled_se": float(pooled_se),
        "pooled_ci_lower": float(pooled - 1.96 * pooled_se),
        "pooled_ci_upper": float(pooled + 1.96 * pooled_se),
        "z_score": float(z),
        "p_value": float(p_val),
        "q_statistic": float(q),
        "q_p_value": float(q_p),
        "i_squared": float(i2),
        "k_studies": int(k),
        "heterogeneity_significant": bool(q_p < 0.10),
    }


def _meta_analysis_random_effects(
    effects: np.ndarray,
    standard_errors: np.ndarray,
) -> Dict[str, float]:
    """Random-effects meta-analysis (DerSimonian-Laird method)."""
    effects = np.asarray(effects, dtype=np.float64)
    ses = np.asarray(standard_errors, dtype=np.float64)

    valid = ~(np.isnan(effects) | np.isnan(ses) | (ses < 1e-15))
    effects = effects[valid]
    ses = ses[valid]
    k = effects.size

    if k == 0:
        return {"pooled_effect": float("nan"), "tau_squared": float("nan")}

    # Fixed-effects first
    w_fe = 1.0 / (ses ** 2)
    w_sum_fe = np.sum(w_fe)
    pooled_fe = np.sum(w_fe * effects) / w_sum_fe

    # Q statistic
    q = np.sum(w_fe * (effects - pooled_fe) ** 2)
    df = k - 1

    # DerSimonian-Laird estimate of tau^2
    c = w_sum_fe - np.sum(w_fe ** 2) / w_sum_fe
    tau2 = max(0.0, (q - df) / c) if c > 1e-15 else 0.0

    # Random-effects weights
    w_re = 1.0 / (ses ** 2 + tau2)
    w_sum_re = np.sum(w_re)
    pooled_re = np.sum(w_re * effects) / w_sum_re
    pooled_se = 1.0 / np.sqrt(w_sum_re)

    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    q_p = float(1.0 - scipy_stats.chi2.cdf(q, df=df)) if df > 0 else 1.0

    z = pooled_re / pooled_se if pooled_se > 1e-15 else 0.0
    p_val = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))

    # Prediction interval for a new study
    pi_margin = scipy_stats.t.ppf(0.975, df=max(df, 1)) * np.sqrt(pooled_se ** 2 + tau2)

    return {
        "pooled_effect": float(pooled_re),
        "pooled_se": float(pooled_se),
        "pooled_ci_lower": float(pooled_re - 1.96 * pooled_se),
        "pooled_ci_upper": float(pooled_re + 1.96 * pooled_se),
        "prediction_lower": float(pooled_re - pi_margin),
        "prediction_upper": float(pooled_re + pi_margin),
        "tau_squared": float(tau2),
        "tau": float(np.sqrt(tau2)),
        "z_score": float(z),
        "p_value": float(p_val),
        "q_statistic": float(q),
        "q_p_value": float(q_p),
        "i_squared": float(i2),
        "k_studies": int(k),
    }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_full_reproducibility_analysis(
    results: List[RunResult],
    config: Optional[ReproducibilityConfig] = None,
) -> Dict[str, Any]:
    """Run all reproducibility analyses and return a combined report.

    This is the recommended single entry point for most users.
    """
    cfg = config or ReproducibilityConfig()
    reporter = ReproducibilityReporter(cfg)
    variance_analyzer = InterRunVarianceAnalyzer(cfg)
    seed_analyzer = SeedSensitivityAnalyzer(cfg)

    logger.info("Starting full reproducibility analysis on %d results.", len(results))

    summary = reporter.generate_summary_statistics(results)
    repro_score = reporter.compute_reproducibility_score(results)
    flags = reporter.flag_unreliable_metrics(results)
    variance_decomp = variance_analyzer.variance_decomposition(results)
    stability = variance_analyzer.compute_stability_score(results)
    seed_sensitivity = seed_analyzer.analyze_seed_sensitivity(results)
    outlier_seeds = seed_analyzer.identify_outlier_seeds(results)
    min_seeds = seed_analyzer.recommend_minimum_seeds(results)
    seed_clusters = seed_analyzer.seed_clustering(results)
    seed_correlations = seed_analyzer.compute_seed_correlations(results)

    variance_plot = reporter.prepare_variance_plot_data(results)
    seed_plot = reporter.prepare_seed_sensitivity_plot_data(results)

    report = {
        "summary_statistics": summary,
        "reproducibility_score": repro_score,
        "unreliable_metrics": flags,
        "variance_decomposition": variance_decomp,
        "stability": stability,
        "seed_sensitivity": seed_sensitivity,
        "outlier_seeds": outlier_seeds,
        "minimum_seeds_recommendation": min_seeds,
        "seed_clusters": seed_clusters,
        "seed_correlations": seed_correlations,
        "plot_data": {
            "variance": variance_plot,
            "seed_sensitivity": seed_plot,
        },
        "config": cfg.to_dict(),
        "n_runs": len(results),
        "n_successful": sum(1 for r in results if r.status == "completed"),
    }

    logger.info(
        "Reproducibility analysis complete. Score: %.3f (%s)",
        repro_score["score"],
        repro_score["grade"],
    )
    return report
