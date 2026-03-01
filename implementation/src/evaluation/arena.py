"""
Evaluation Arena for comparing diversity-promoting decoding algorithms.

Provides infrastructure for running controlled experiments across multiple
algorithms, tasks, and hyperparameter configurations. Supports statistical
comparison (bootstrap CI, Cliff's delta, Bradley-Terry ranking), Pareto
frontier analysis, ablation studies, cross-validation, checkpointing,
and structured export of results.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import signal
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

from src.types import (
    CacheStrategy,
    DecodingStrategy,
    GenerationSet,
    GenerationResult,
    LogitVector,
    MetricType,
    TaskDomain,
    TokenID,
    TokenSequence,
    Token,
)
from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BOOTSTRAP_SAMPLES = 10_000
_DEFAULT_ALPHA = 0.05
_DEFAULT_TIMEOUT_SECONDS = 600
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_K_FOLDS = 5
_CHECKPOINT_VERSION = 2
_ARENA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 hex digest for JSON-serialisable objects."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _timestamp() -> float:
    """Current UNIX timestamp."""
    return time.time()


def _iso_now() -> str:
    """Current UTC time as ISO-8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns *default* when *b* is zero."""
    if b == 0.0:
        return default
    return a / b


def _flatten(nested: List[List[Any]]) -> List[Any]:
    """Flatten one level of nesting."""
    return [item for sublist in nested for item in sublist]


# ---------------------------------------------------------------------------
# Timeout helper (UNIX only, graceful fallback)
# ---------------------------------------------------------------------------


class _TimeoutError(Exception):
    """Raised when a generation run exceeds its time limit."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _TimeoutError("Generation timed out")


class _TimeoutContext:
    """Context manager that raises ``_TimeoutError`` after *seconds*.

    Falls back to no-op on platforms without ``signal.SIGALRM``.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self) -> "_TimeoutContext":
        if self._supported and self.seconds > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._supported:
            signal.alarm(0)
        return False


# ---------------------------------------------------------------------------
# Metric function protocol
# ---------------------------------------------------------------------------


class MetricFn(Protocol):
    """Callable that scores a GenerationSet on a single metric."""

    def __call__(self, generation_set: GenerationSet) -> float:
        ...


# ---------------------------------------------------------------------------
# RunStatus enum
# ---------------------------------------------------------------------------


class RunStatus(Enum):
    """Lifecycle status of a single arena run."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()

    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED,
                        RunStatus.TIMEOUT, RunStatus.CANCELLED)

    def __repr__(self) -> str:
        return f"RunStatus.{self.name}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArenaConfig:
    """Configuration governing the entire arena evaluation session.

    Attributes:
        algorithms: Names of algorithms to evaluate (populated via register).
        tasks: List of (name, domain) tuples for each task.
        metrics: Mapping from metric name to MetricType (DIVERSITY/QUALITY).
        num_runs: Number of independent repetitions per (algo, task, config).
        seed: Master random seed for reproducibility.
        output_dir: Directory for checkpoints and exported results.
        parallel_workers: Number of parallel workers (0 = sequential).
        timeout_seconds: Maximum wall-clock time per single run.
        max_retries: Retries on transient failure before marking FAILED.
        bootstrap_samples: Number of bootstrap resamples for CI computation.
        alpha: Significance level for statistical tests.
        cache_strategy: Caching behaviour for metric computation.
        save_generations: Whether to persist raw generations in results.
        checkpoint_interval: Save checkpoint every N completed runs.
        verbose: Emit per-run log messages at INFO level.
        experiment_name: Human-readable experiment tag.
        experiment_id: Unique identifier (auto-generated if omitted).
        metadata: Arbitrary key-value metadata attached to the experiment.
    """

    algorithms: List[str] = field(default_factory=list)
    tasks: List[Tuple[str, TaskDomain]] = field(default_factory=list)
    metrics: Dict[str, MetricType] = field(default_factory=dict)
    num_runs: int = 5
    seed: int = 42
    output_dir: str = "./arena_output"
    parallel_workers: int = 0
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    max_retries: int = _DEFAULT_MAX_RETRIES
    bootstrap_samples: int = _DEFAULT_BOOTSTRAP_SAMPLES
    alpha: float = _DEFAULT_ALPHA
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    save_generations: bool = True
    checkpoint_interval: int = 10
    verbose: bool = True
    experiment_name: str = "arena_experiment"
    experiment_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = uuid.uuid4().hex[:12]

    # -- Validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Return list of validation error strings (empty == valid)."""
        errors: List[str] = []
        if self.num_runs < 1:
            errors.append("num_runs must be >= 1")
        if self.timeout_seconds < 0:
            errors.append("timeout_seconds must be >= 0")
        if self.max_retries < 0:
            errors.append("max_retries must be >= 0")
        if not 0 < self.alpha < 1:
            errors.append("alpha must be in (0, 1)")
        if self.bootstrap_samples < 100:
            errors.append("bootstrap_samples should be >= 100")
        if self.parallel_workers < 0:
            errors.append("parallel_workers must be >= 0")
        if self.checkpoint_interval < 1:
            errors.append("checkpoint_interval must be >= 1")
        return errors

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cache_strategy"] = self.cache_strategy.name
        d["tasks"] = [(n, td.name) for n, td in self.tasks]
        d["metrics"] = {k: v.name for k, v in self.metrics.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArenaConfig":
        d = dict(d)
        if "cache_strategy" in d and isinstance(d["cache_strategy"], str):
            d["cache_strategy"] = CacheStrategy[d["cache_strategy"]]
        if "tasks" in d:
            d["tasks"] = [(n, TaskDomain[td]) for n, td in d["tasks"]]
        if "metrics" in d:
            d["metrics"] = {k: MetricType[v] for k, v in d["metrics"].items()}
        return cls(**d)

    def hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass
class AlgorithmEntry:
    """An algorithm registered in the arena with optional hyper-parameter grid.

    Attributes:
        name: Unique algorithm identifier.
        algorithm: The decoding algorithm instance.
        config: Base decoding configuration.
        hyperparams_grid: Optional mapping from param name to list of values.
        description: Human-readable description.
        tags: Arbitrary tags for filtering.
        enabled: Whether this entry participates in the next run.
    """

    name: str
    algorithm: DecodingAlgorithm
    config: DecodingConfig
    hyperparams_grid: Dict[str, List[Any]] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    # -- grid expansion -----------------------------------------------------

    def expand_grid(self) -> List[DecodingConfig]:
        """Expand *hyperparams_grid* into concrete ``DecodingConfig`` instances.

        Each point in the Cartesian product of all grid axes generates one
        config with the base config's values overridden accordingly.
        """
        if not self.hyperparams_grid:
            return [copy.deepcopy(self.config)]

        keys = sorted(self.hyperparams_grid.keys())
        value_lists = [self.hyperparams_grid[k] for k in keys]
        configs: List[DecodingConfig] = []
        for combo in itertools.product(*value_lists):
            cfg = copy.deepcopy(self.config)
            for k, v in zip(keys, combo):
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    cfg.params[k] = v
            configs.append(cfg)
        return configs

    def num_configs(self) -> int:
        """Number of configurations after grid expansion."""
        if not self.hyperparams_grid:
            return 1
        n = 1
        for vals in self.hyperparams_grid.values():
            n *= len(vals)
        return n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "hyperparams_grid": self.hyperparams_grid,
            "description": self.description,
            "tags": self.tags,
            "enabled": self.enabled,
        }


@dataclass
class ArenaRun:
    """Record of a single (algorithm, task, config) execution.

    Attributes:
        run_id: Unique identifier for this run.
        algorithm_name: Name of the algorithm evaluated.
        task_name: Name of the task / prompt set.
        config: Decoding configuration used.
        status: Current lifecycle status.
        start_time: Wall-clock start (UNIX timestamp).
        end_time: Wall-clock end (UNIX timestamp), or None if not finished.
        generations: The generated outputs (if config.save_generations).
        metric_scores: Mapping metric_name -> score.
        error: Error message if status is FAILED.
        attempt: Attempt number (1-based).
        duration_seconds: Elapsed wall-clock seconds.
        metadata: Arbitrary auxiliary data.
    """

    run_id: str = ""
    algorithm_name: str = ""
    task_name: str = ""
    config: Optional[DecodingConfig] = None
    status: RunStatus = RunStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    generations: Optional[GenerationSet] = None
    metric_scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    attempt: int = 1
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:16]

    def mark_running(self) -> None:
        self.status = RunStatus.RUNNING
        self.start_time = _timestamp()

    def mark_completed(self, scores: Dict[str, float]) -> None:
        self.status = RunStatus.COMPLETED
        self.end_time = _timestamp()
        self.metric_scores = scores
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time

    def mark_failed(self, error_msg: str) -> None:
        self.status = RunStatus.FAILED
        self.end_time = _timestamp()
        self.error = error_msg
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time

    def mark_timeout(self) -> None:
        self.status = RunStatus.TIMEOUT
        self.end_time = _timestamp()
        self.error = "Run exceeded timeout limit"
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time

    def mark_cancelled(self) -> None:
        self.status = RunStatus.CANCELLED
        self.end_time = _timestamp()
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time

    def is_successful(self) -> bool:
        return self.status == RunStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "algorithm_name": self.algorithm_name,
            "task_name": self.task_name,
            "status": self.status.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metric_scores": self.metric_scores,
            "error": self.error,
            "attempt": self.attempt,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }
        if self.config is not None:
            d["config"] = self.config.to_dict()
        if self.generations is not None:
            d["generations"] = self.generations.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArenaRun":
        config = None
        if "config" in d and d["config"] is not None:
            config = DecodingConfig.from_dict(d["config"])
        generations = None
        if "generations" in d and d["generations"] is not None:
            generations = GenerationSet.from_dict(d["generations"])
        return cls(
            run_id=d.get("run_id", ""),
            algorithm_name=d.get("algorithm_name", ""),
            task_name=d.get("task_name", ""),
            config=config,
            status=RunStatus[d.get("status", "PENDING")],
            start_time=d.get("start_time"),
            end_time=d.get("end_time"),
            generations=generations,
            metric_scores=d.get("metric_scores", {}),
            error=d.get("error"),
            attempt=d.get("attempt", 1),
            duration_seconds=d.get("duration_seconds", 0.0),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"ArenaRun(id={self.run_id!r}, algo={self.algorithm_name!r}, "
            f"task={self.task_name!r}, status={self.status.name})"
        )


@dataclass
class ComparisonResult:
    """Statistical comparison between two algorithms on one metric.

    Attributes:
        algorithm_a: Name of the first algorithm.
        algorithm_b: Name of the second algorithm.
        metric_name: The metric compared.
        mean_a: Mean score for algorithm A.
        mean_b: Mean score for algorithm B.
        std_a: Standard deviation for algorithm A.
        std_b: Standard deviation for algorithm B.
        effect_size: Cliff's delta effect size.
        effect_magnitude: Qualitative magnitude (negligible/small/medium/large).
        p_value: Approximate p-value from permutation test.
        ci_a: 95 % bootstrap confidence interval for A.
        ci_b: 95 % bootstrap confidence interval for B.
        ci_diff: 95 % bootstrap CI for (A - B).
        winner: Name of the better algorithm, or "tie".
        significant: Whether the difference is statistically significant.
        num_runs_a: Number of runs for A.
        num_runs_b: Number of runs for B.
    """

    algorithm_a: str = ""
    algorithm_b: str = ""
    metric_name: str = ""
    mean_a: float = 0.0
    mean_b: float = 0.0
    std_a: float = 0.0
    std_b: float = 0.0
    effect_size: float = 0.0
    effect_magnitude: str = "negligible"
    p_value: float = 1.0
    ci_a: Tuple[float, float] = (0.0, 0.0)
    ci_b: Tuple[float, float] = (0.0, 0.0)
    ci_diff: Tuple[float, float] = (0.0, 0.0)
    winner: str = "tie"
    significant: bool = False
    num_runs_a: int = 0
    num_runs_b: int = 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"{self.metric_name}: {self.algorithm_a} ({self.mean_a:.4f}) vs "
            f"{self.algorithm_b} ({self.mean_b:.4f}) — "
            f"delta={self.effect_size:+.3f} ({self.effect_magnitude}), "
            f"p={self.p_value:.4f} ({sig_str}), winner={self.winner}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_a": self.algorithm_a,
            "algorithm_b": self.algorithm_b,
            "metric_name": self.metric_name,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "effect_size": self.effect_size,
            "effect_magnitude": self.effect_magnitude,
            "p_value": self.p_value,
            "ci_a": list(self.ci_a),
            "ci_b": list(self.ci_b),
            "ci_diff": list(self.ci_diff),
            "winner": self.winner,
            "significant": self.significant,
            "num_runs_a": self.num_runs_a,
            "num_runs_b": self.num_runs_b,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComparisonResult":
        d = dict(d)
        for key in ("ci_a", "ci_b", "ci_diff"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)


@dataclass
class ParetoPoint:
    """A single point on a 2-D Pareto frontier.

    Attributes:
        algorithm_name: Algorithm that produced this point.
        config: The configuration used.
        diversity_score: Score on the diversity axis.
        quality_score: Score on the quality axis.
        dominated: Whether another point dominates this one.
        metadata: Extra information (e.g. full metric dict).
    """

    algorithm_name: str = ""
    config: Optional[DecodingConfig] = None
    diversity_score: float = 0.0
    quality_score: float = 0.0
    dominated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: "ParetoPoint") -> bool:
        """True if *self* Pareto-dominates *other* (higher is better)."""
        return (
            self.diversity_score >= other.diversity_score
            and self.quality_score >= other.quality_score
            and (
                self.diversity_score > other.diversity_score
                or self.quality_score > other.quality_score
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "algorithm_name": self.algorithm_name,
            "diversity_score": self.diversity_score,
            "quality_score": self.quality_score,
            "dominated": self.dominated,
            "metadata": self.metadata,
        }
        if self.config is not None:
            d["config"] = self.config.to_dict()
        return d


@dataclass
class RankingEntry:
    """One row in an algorithm ranking table.

    Attributes:
        rank: 1-based rank position.
        algorithm_name: Identifier.
        score: The ranking score (interpretation depends on method).
        wins: Number of pairwise wins.
        losses: Number of pairwise losses.
        ties: Number of pairwise ties.
        mean_metric: Mean value of the ranked metric.
        ci: Bootstrap confidence interval of the mean metric.
    """

    rank: int = 0
    algorithm_name: str = ""
    score: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    mean_metric: float = 0.0
    ci: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "algorithm_name": self.algorithm_name,
            "score": self.score,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "mean_metric": self.mean_metric,
            "ci": list(self.ci),
        }


@dataclass
class CrossValidationResult:
    """Outcome of k-fold cross-validation for one algorithm-task pair.

    Attributes:
        algorithm_name: Algorithm evaluated.
        task_name: Task evaluated.
        k: Number of folds.
        fold_scores: Per-fold metric dictionaries.
        mean_scores: Mean across folds for each metric.
        std_scores: Std-dev across folds for each metric.
        ci_scores: 95 % CI across folds for each metric.
    """

    algorithm_name: str = ""
    task_name: str = ""
    k: int = 0
    fold_scores: List[Dict[str, float]] = field(default_factory=list)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    ci_scores: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "task_name": self.task_name,
            "k": self.k,
            "fold_scores": self.fold_scores,
            "mean_scores": self.mean_scores,
            "std_scores": self.std_scores,
            "ci_scores": {k: list(v) for k, v in self.ci_scores.items()},
        }


@dataclass
class AblationResult:
    """Outcome of an ablation study for one algorithm.

    Attributes:
        algorithm_name: Algorithm studied.
        baseline_scores: Metrics with all features enabled.
        ablation_scores: Mapping param_name -> metric scores with that param
            set to its default/ablated value.
        deltas: param_name -> metric_name -> change from baseline.
        importance_ranking: Params ordered by impact (most impactful first).
    """

    algorithm_name: str = ""
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    ablation_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    importance_ranking: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "baseline_scores": self.baseline_scores,
            "ablation_scores": self.ablation_scores,
            "deltas": self.deltas,
            "importance_ranking": self.importance_ranking,
        }


@dataclass
class ArenaResult:
    """Aggregated results from a complete arena evaluation.

    Attributes:
        experiment_id: Unique experiment identifier.
        experiment_name: Human-readable label.
        config: The ArenaConfig used.
        runs: All individual ArenaRun records.
        summary_stats: algorithm -> metric -> {mean, std, min, max, median}.
        rankings: metric -> list of RankingEntry.
        pareto_points: List of ParetoPoint on the diversity-quality frontier.
        comparisons: List of all pairwise ComparisonResult.
        total_duration_seconds: Sum of all run durations.
        num_completed: Count of completed runs.
        num_failed: Count of failed runs.
        num_timeout: Count of timed-out runs.
        created_at: ISO-8601 timestamp.
        metadata: Extra information.
    """

    experiment_id: str = ""
    experiment_name: str = ""
    config: Optional[ArenaConfig] = None
    runs: List[ArenaRun] = field(default_factory=list)
    summary_stats: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )
    rankings: Dict[str, List[RankingEntry]] = field(default_factory=dict)
    pareto_points: List[ParetoPoint] = field(default_factory=list)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    num_completed: int = 0
    num_failed: int = 0
    num_timeout: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _iso_now()

    # -- accessors ----------------------------------------------------------

    def successful_runs(self) -> List[ArenaRun]:
        return [r for r in self.runs if r.is_successful()]

    def failed_runs(self) -> List[ArenaRun]:
        return [r for r in self.runs if r.status == RunStatus.FAILED]

    def timed_out_runs(self) -> List[ArenaRun]:
        return [r for r in self.runs if r.status == RunStatus.TIMEOUT]

    def runs_for_algorithm(self, name: str) -> List[ArenaRun]:
        return [r for r in self.runs if r.algorithm_name == name and r.is_successful()]

    def runs_for_task(self, name: str) -> List[ArenaRun]:
        return [r for r in self.runs if r.task_name == name and r.is_successful()]

    def algorithm_names(self) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for r in self.runs:
            if r.algorithm_name not in seen:
                seen.add(r.algorithm_name)
                out.append(r.algorithm_name)
        return out

    def task_names(self) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for r in self.runs:
            if r.task_name not in seen:
                seen.add(r.task_name)
                out.append(r.task_name)
        return out

    def metric_names(self) -> List[str]:
        names: Set[str] = set()
        for r in self.successful_runs():
            names.update(r.metric_scores.keys())
        return sorted(names)

    def scores_matrix(self, metric: str) -> Dict[str, List[float]]:
        """Return {algorithm: [scores...]} for the given metric."""
        result: Dict[str, List[float]] = defaultdict(list)
        for r in self.successful_runs():
            if metric in r.metric_scores:
                result[r.algorithm_name].append(r.metric_scores[metric])
        return dict(result)

    def best_algorithm(self, metric: str, higher_is_better: bool = True) -> str:
        """Return the algorithm with the best mean score on *metric*."""
        matrix = self.scores_matrix(metric)
        if not matrix:
            return ""
        if higher_is_better:
            return max(matrix, key=lambda a: float(np.mean(matrix[a])))
        return min(matrix, key=lambda a: float(np.mean(matrix[a])))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "config": self.config.to_dict() if self.config else None,
            "runs": [r.to_dict() for r in self.runs],
            "summary_stats": self.summary_stats,
            "rankings": {
                m: [e.to_dict() for e in entries]
                for m, entries in self.rankings.items()
            },
            "pareto_points": [p.to_dict() for p in self.pareto_points],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "total_duration_seconds": self.total_duration_seconds,
            "num_completed": self.num_completed,
            "num_failed": self.num_failed,
            "num_timeout": self.num_timeout,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArenaResult":
        config = None
        if d.get("config"):
            config = ArenaConfig.from_dict(d["config"])
        runs = [ArenaRun.from_dict(r) for r in d.get("runs", [])]
        rankings: Dict[str, List[RankingEntry]] = {}
        for m, entries in d.get("rankings", {}).items():
            rankings[m] = [
                RankingEntry(**{
                    **e,
                    "ci": tuple(e["ci"]) if isinstance(e.get("ci"), list) else e.get("ci", (0.0, 0.0)),
                })
                for e in entries
            ]
        comparisons = [ComparisonResult.from_dict(c) for c in d.get("comparisons", [])]
        pareto = [
            ParetoPoint(
                algorithm_name=p.get("algorithm_name", ""),
                diversity_score=p.get("diversity_score", 0.0),
                quality_score=p.get("quality_score", 0.0),
                dominated=p.get("dominated", False),
                metadata=p.get("metadata", {}),
            )
            for p in d.get("pareto_points", [])
        ]
        return cls(
            experiment_id=d.get("experiment_id", ""),
            experiment_name=d.get("experiment_name", ""),
            config=config,
            runs=runs,
            summary_stats=d.get("summary_stats", {}),
            rankings=rankings,
            pareto_points=pareto,
            comparisons=comparisons,
            total_duration_seconds=d.get("total_duration_seconds", 0.0),
            num_completed=d.get("num_completed", 0),
            num_failed=d.get("num_failed", 0),
            num_timeout=d.get("num_timeout", 0),
            created_at=d.get("created_at", ""),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------


@dataclass
class TaskDescriptor:
    """A registered evaluation task.

    Attributes:
        name: Unique task identifier.
        domain: TaskDomain enum member.
        prompts: List of prompt strings.
        reference_texts: Optional reference texts for quality evaluation.
        metadata: Extra task-level metadata.
    """

    name: str = ""
    domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION
    prompts: List[str] = field(default_factory=list)
    reference_texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def num_prompts(self) -> int:
        return len(self.prompts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.name,
            "prompts": self.prompts,
            "reference_texts": self.reference_texts,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Metric registry entry
# ---------------------------------------------------------------------------


@dataclass
class MetricEntry:
    """A registered evaluation metric.

    Attributes:
        name: Unique metric identifier.
        metric_type: DIVERSITY or QUALITY.
        fn: Callable that scores a GenerationSet.
        higher_is_better: Whether larger values are better.
        description: Human-readable description.
    """

    name: str = ""
    metric_type: MetricType = MetricType.DIVERSITY
    fn: Optional[Callable[[GenerationSet], float]] = None
    higher_is_better: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------


class _MetricCache:
    """Simple in-memory cache for metric computations keyed by content hash."""

    def __init__(self, strategy: CacheStrategy) -> None:
        self.strategy = strategy
        self._store: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, metric_name: str, gen_set: GenerationSet) -> str:
        texts = gen_set.texts()
        raw = json.dumps({"metric": metric_name, "texts": texts}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, metric_name: str, gen_set: GenerationSet) -> Optional[float]:
        if self.strategy == CacheStrategy.NONE:
            return None
        key = self._key(metric_name, gen_set)
        val = self._store.get(key)
        if val is not None:
            self._hits += 1
        else:
            self._misses += 1
        return val

    def put(self, metric_name: str, gen_set: GenerationSet, value: float) -> None:
        if self.strategy == CacheStrategy.NONE:
            return
        key = self._key(metric_name, gen_set)
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


class _ProgressTracker:
    """Tracks and logs progress of arena evaluation."""

    def __init__(self, total: int, verbose: bool = True) -> None:
        self.total = max(total, 1)
        self.completed = 0
        self.failed = 0
        self.timed_out = 0
        self.verbose = verbose
        self._start_time = _timestamp()
        self._last_log_time = self._start_time

    def update(self, status: RunStatus) -> None:
        if status == RunStatus.COMPLETED:
            self.completed += 1
        elif status == RunStatus.FAILED:
            self.failed += 1
        elif status == RunStatus.TIMEOUT:
            self.timed_out += 1

    @property
    def done(self) -> int:
        return self.completed + self.failed + self.timed_out

    @property
    def fraction(self) -> float:
        return self.done / self.total

    def elapsed(self) -> float:
        return _timestamp() - self._start_time

    def eta_seconds(self) -> float:
        if self.done == 0:
            return 0.0
        per_run = self.elapsed() / self.done
        remaining = self.total - self.done
        return per_run * remaining

    def log(self, run: ArenaRun) -> None:
        if not self.verbose:
            return
        now = _timestamp()
        if now - self._last_log_time < 1.0 and self.done < self.total:
            return
        self._last_log_time = now
        pct = self.fraction * 100
        eta = self.eta_seconds()
        logger.info(
            "[%d/%d] (%.1f%%) algo=%s task=%s status=%s "
            "elapsed=%.1fs ETA=%.1fs | ok=%d fail=%d timeout=%d",
            self.done,
            self.total,
            pct,
            run.algorithm_name,
            run.task_name,
            run.status.name,
            self.elapsed(),
            eta,
            self.completed,
            self.failed,
            self.timed_out,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "elapsed_seconds": self.elapsed(),
        }


# =========================================================================
# EvaluationArena — main class
# =========================================================================


class EvaluationArena:
    """Central orchestrator for controlled evaluation of decoding algorithms.

    Manages algorithm registration, task setup, metric computation,
    execution of runs, statistical comparison, ranking, Pareto analysis,
    ablation studies, cross-validation, checkpointing, and export.

    Parameters
    ----------
    config : ArenaConfig
        Global configuration for the evaluation session.

    Example
    -------
    >>> arena = EvaluationArena(ArenaConfig(num_runs=10, seed=0))
    >>> arena.register_algorithm("dpp", dpp_algo, dpp_cfg)
    >>> arena.register_task("stories", TaskDomain.STORY_GENERATION, prompts)
    >>> arena.register_metric("diversity", MetricType.DIVERSITY, div_fn)
    >>> result = arena.run_all()
    >>> ranking = arena.rank_algorithms("diversity")
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: ArenaConfig) -> None:
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid ArenaConfig: {errors}")

        self.config = config
        self._rng = np.random.default_rng(config.seed)

        self._algorithms: Dict[str, AlgorithmEntry] = {}
        self._tasks: Dict[str, TaskDescriptor] = {}
        self._metrics: Dict[str, MetricEntry] = {}

        self._runs: List[ArenaRun] = []
        self._result: Optional[ArenaResult] = None
        self._cache = _MetricCache(config.cache_strategy)
        self._cancelled = False

        self._logit_sources: Dict[str, LogitSource] = {}
        self._checkpoint_path: Optional[Path] = None
        self._run_counter = 0

        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(
            "EvaluationArena initialised — experiment=%s id=%s seed=%d",
            config.experiment_name,
            config.experiment_id,
            config.seed,
        )

    # ------------------------------------------------------------------
    # Registration — algorithms
    # ------------------------------------------------------------------

    def register_algorithm(
        self,
        name: str,
        algorithm: DecodingAlgorithm,
        config: DecodingConfig,
        hyperparams_grid: Optional[Dict[str, List[Any]]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        logit_source: Optional[LogitSource] = None,
    ) -> None:
        """Register a decoding algorithm for evaluation.

        Parameters
        ----------
        name : str
            Unique identifier.
        algorithm : DecodingAlgorithm
            The algorithm instance.
        config : DecodingConfig
            Base configuration.
        hyperparams_grid : dict, optional
            Mapping param_name -> [values] for grid search.
        description : str
            Human-readable description.
        tags : list of str, optional
            Arbitrary tags for filtering.
        logit_source : LogitSource, optional
            Logit source to use for this algorithm's generations.
        """
        if name in self._algorithms:
            logger.warning("Overwriting existing algorithm entry: %s", name)

        entry = AlgorithmEntry(
            name=name,
            algorithm=algorithm,
            config=config,
            hyperparams_grid=hyperparams_grid or {},
            description=description,
            tags=tags or [],
            enabled=True,
        )
        self._algorithms[name] = entry

        if name not in self.config.algorithms:
            self.config.algorithms.append(name)

        if logit_source is not None:
            self._logit_sources[name] = logit_source

        num_cfgs = entry.num_configs()
        logger.info(
            "Registered algorithm %r (%d config%s)",
            name,
            num_cfgs,
            "s" if num_cfgs != 1 else "",
        )

    def unregister_algorithm(self, name: str) -> None:
        """Remove a previously registered algorithm."""
        self._algorithms.pop(name, None)
        self._logit_sources.pop(name, None)
        if name in self.config.algorithms:
            self.config.algorithms.remove(name)

    def enable_algorithm(self, name: str) -> None:
        if name in self._algorithms:
            self._algorithms[name].enabled = True

    def disable_algorithm(self, name: str) -> None:
        if name in self._algorithms:
            self._algorithms[name].enabled = False

    def list_algorithms(self) -> List[str]:
        return [n for n, e in self._algorithms.items() if e.enabled]

    def get_algorithm_entry(self, name: str) -> Optional[AlgorithmEntry]:
        return self._algorithms.get(name)

    # ------------------------------------------------------------------
    # Registration — tasks
    # ------------------------------------------------------------------

    def register_task(
        self,
        name: str,
        domain: TaskDomain,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an evaluation task.

        Parameters
        ----------
        name : str
            Unique task identifier.
        domain : TaskDomain
            Task family.
        prompts : list of str
            Input prompts for generation.
        reference_texts : list of str, optional
            Gold references for quality metrics.
        metadata : dict, optional
            Auxiliary metadata.
        """
        if name in self._tasks:
            logger.warning("Overwriting existing task: %s", name)

        task = TaskDescriptor(
            name=name,
            domain=domain,
            prompts=prompts,
            reference_texts=reference_texts or [],
            metadata=metadata or {},
        )
        self._tasks[name] = task

        task_tuple = (name, domain)
        if task_tuple not in self.config.tasks:
            self.config.tasks.append(task_tuple)

        logger.info(
            "Registered task %r (domain=%s, %d prompts)",
            name,
            domain.name,
            len(prompts),
        )

    def unregister_task(self, name: str) -> None:
        task = self._tasks.pop(name, None)
        if task is not None:
            self.config.tasks = [
                (n, d) for n, d in self.config.tasks if n != name
            ]

    def list_tasks(self) -> List[str]:
        return list(self._tasks.keys())

    def get_task(self, name: str) -> Optional[TaskDescriptor]:
        return self._tasks.get(name)

    # ------------------------------------------------------------------
    # Registration — metrics
    # ------------------------------------------------------------------

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        fn: Callable[[GenerationSet], float],
        higher_is_better: bool = True,
        description: str = "",
    ) -> None:
        """Register an evaluation metric.

        Parameters
        ----------
        name : str
            Unique metric identifier.
        metric_type : MetricType
            DIVERSITY or QUALITY.
        fn : callable
            Scoring function ``GenerationSet -> float``.
        higher_is_better : bool
            Whether larger values are better.
        description : str
            Human-readable description.
        """
        if name in self._metrics:
            logger.warning("Overwriting existing metric: %s", name)

        entry = MetricEntry(
            name=name,
            metric_type=metric_type,
            fn=fn,
            higher_is_better=higher_is_better,
            description=description,
        )
        self._metrics[name] = entry
        self.config.metrics[name] = metric_type

        logger.info(
            "Registered metric %r (type=%s, higher_is_better=%s)",
            name,
            metric_type.name,
            higher_is_better,
        )

    def unregister_metric(self, name: str) -> None:
        self._metrics.pop(name, None)
        self.config.metrics.pop(name, None)

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    def get_metric_entry(self, name: str) -> Optional[MetricEntry]:
        return self._metrics.get(name)

    def diversity_metrics(self) -> List[str]:
        return [
            n for n, e in self._metrics.items()
            if e.metric_type == MetricType.DIVERSITY
        ]

    def quality_metrics(self) -> List[str]:
        return [
            n for n, e in self._metrics.items()
            if e.metric_type == MetricType.QUALITY
        ]

    # ------------------------------------------------------------------
    # Logit source management
    # ------------------------------------------------------------------

    def register_logit_source(self, name: str, source: LogitSource) -> None:
        """Register a logit source to use for an algorithm by name."""
        self._logit_sources[name] = source

    def get_logit_source(self, algorithm_name: str) -> Optional[LogitSource]:
        return self._logit_sources.get(algorithm_name)

    # ------------------------------------------------------------------
    # Single run execution
    # ------------------------------------------------------------------

    def run_single(
        self,
        algorithm_name: str,
        task_name: str,
        config: Optional[DecodingConfig] = None,
        logit_source: Optional[LogitSource] = None,
    ) -> ArenaRun:
        """Execute a single (algorithm, task, config) evaluation.

        Parameters
        ----------
        algorithm_name : str
            Registered algorithm name.
        task_name : str
            Registered task name.
        config : DecodingConfig, optional
            Override decoding config (uses algorithm's default otherwise).
        logit_source : LogitSource, optional
            Override logit source for this run.

        Returns
        -------
        ArenaRun
            The completed run record.
        """
        entry = self._algorithms.get(algorithm_name)
        if entry is None:
            raise KeyError(f"Algorithm not registered: {algorithm_name!r}")
        task = self._tasks.get(task_name)
        if task is None:
            raise KeyError(f"Task not registered: {task_name!r}")

        effective_config = config or copy.deepcopy(entry.config)
        effective_source = logit_source or self._logit_sources.get(algorithm_name)

        run = ArenaRun(
            algorithm_name=algorithm_name,
            task_name=task_name,
            config=effective_config,
        )

        run.mark_running()
        logger.debug(
            "Starting run %s: algo=%s task=%s",
            run.run_id,
            algorithm_name,
            task_name,
        )

        try:
            with _TimeoutContext(self.config.timeout_seconds):
                gen_set = self._execute_generation(
                    entry.algorithm, task, effective_config, effective_source
                )

            scores = self._evaluate_metrics(gen_set)

            if self.config.save_generations:
                run.generations = gen_set
            run.mark_completed(scores)

        except _TimeoutError:
            self._handle_timeout(run)
        except Exception as exc:
            self._handle_error(run, exc)

        self._runs.append(run)
        self._run_counter += 1
        return run

    # ------------------------------------------------------------------
    # Full evaluation sweep
    # ------------------------------------------------------------------

    def run_all(self) -> ArenaResult:
        """Run all registered algorithm × task × config combinations.

        Each combination is repeated ``config.num_runs`` times.  Progress
        is logged and checkpoints are saved at the configured interval.

        Returns
        -------
        ArenaResult
            Aggregated results with statistics and rankings.
        """
        schedule = self._build_schedule()
        total = len(schedule) * self.config.num_runs
        logger.info(
            "Starting full evaluation: %d combinations × %d runs = %d total",
            len(schedule),
            self.config.num_runs,
            total,
        )

        tracker = _ProgressTracker(total, self.config.verbose)

        for algo_name, task_name, cfg in schedule:
            if self._cancelled:
                logger.warning("Arena cancelled — stopping early")
                break

            for run_idx in range(self.config.num_runs):
                if self._cancelled:
                    break

                run_config = copy.deepcopy(cfg)
                if run_config.seed is not None:
                    run_config.seed = run_config.seed + run_idx
                elif self.config.seed is not None:
                    run_config.seed = self.config.seed + self._run_counter

                run = self._execute_with_retries(
                    algo_name, task_name, run_config
                )

                tracker.update(run.status)
                tracker.log(run)

                if self._run_counter % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

        result = self._aggregate_results(self._runs)
        self._result = result

        self.save_checkpoint()
        logger.info(
            "Evaluation complete: %d runs (ok=%d fail=%d timeout=%d) in %.1fs",
            len(self._runs),
            result.num_completed,
            result.num_failed,
            result.num_timeout,
            result.total_duration_seconds,
        )
        return result

    def _build_schedule(
        self,
    ) -> List[Tuple[str, str, DecodingConfig]]:
        """Build the Cartesian product of algorithms × tasks × configs."""
        schedule: List[Tuple[str, str, DecodingConfig]] = []

        enabled_algos = [
            (n, e) for n, e in self._algorithms.items() if e.enabled
        ]
        if not enabled_algos:
            logger.warning("No enabled algorithms — schedule is empty")
            return schedule

        if not self._tasks:
            logger.warning("No tasks registered — schedule is empty")
            return schedule

        for algo_name, entry in enabled_algos:
            configs = entry.expand_grid()
            for task_name in self._tasks:
                for cfg in configs:
                    schedule.append((algo_name, task_name, cfg))

        self._rng.shuffle(schedule)  # type: ignore[arg-type]
        return schedule

    def _execute_with_retries(
        self,
        algorithm_name: str,
        task_name: str,
        config: DecodingConfig,
    ) -> ArenaRun:
        """Execute a run with retry logic on failure."""
        last_run: Optional[ArenaRun] = None
        for attempt in range(1, self.config.max_retries + 1):
            run = self.run_single(algorithm_name, task_name, config)
            run.attempt = attempt
            last_run = run

            if run.is_successful():
                return run
            if run.status == RunStatus.TIMEOUT:
                return run

            logger.warning(
                "Run %s failed (attempt %d/%d): %s",
                run.run_id,
                attempt,
                self.config.max_retries,
                run.error,
            )

        assert last_run is not None
        return last_run

    # ------------------------------------------------------------------
    # Hyperparameter sweep
    # ------------------------------------------------------------------

    def run_sweep(
        self,
        algorithm_name: str,
        param_grid: Dict[str, List[Any]],
        task_name: Optional[str] = None,
    ) -> List[ArenaRun]:
        """Run a hyperparameter sweep for a single algorithm.

        Parameters
        ----------
        algorithm_name : str
            Algorithm to sweep.
        param_grid : dict
            Mapping param_name -> [values].
        task_name : str, optional
            Task to evaluate on (defaults to all registered tasks).

        Returns
        -------
        list of ArenaRun
            One run per (config-point, task, repetition).
        """
        entry = self._algorithms.get(algorithm_name)
        if entry is None:
            raise KeyError(f"Algorithm not registered: {algorithm_name!r}")

        keys = sorted(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        configs: List[DecodingConfig] = []
        for combo in itertools.product(*value_lists):
            cfg = copy.deepcopy(entry.config)
            for k, v in zip(keys, combo):
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    cfg.params[k] = v
            configs.append(cfg)

        tasks = [task_name] if task_name else list(self._tasks.keys())
        sweep_runs: List[ArenaRun] = []

        total = len(configs) * len(tasks) * self.config.num_runs
        logger.info(
            "Starting sweep for %s: %d configs × %d tasks × %d reps = %d runs",
            algorithm_name,
            len(configs),
            len(tasks),
            self.config.num_runs,
            total,
        )

        tracker = _ProgressTracker(total, self.config.verbose)

        for cfg in configs:
            for t_name in tasks:
                for run_idx in range(self.config.num_runs):
                    run_cfg = copy.deepcopy(cfg)
                    if self.config.seed is not None:
                        run_cfg.seed = self.config.seed + self._run_counter

                    run = self._execute_with_retries(
                        algorithm_name, t_name, run_cfg
                    )
                    sweep_runs.append(run)
                    tracker.update(run.status)
                    tracker.log(run)

        logger.info(
            "Sweep complete: %d runs, %d successful",
            len(sweep_runs),
            sum(1 for r in sweep_runs if r.is_successful()),
        )
        return sweep_runs

    # ------------------------------------------------------------------
    # Generation execution
    # ------------------------------------------------------------------

    def _execute_generation(
        self,
        algorithm: DecodingAlgorithm,
        task: TaskDescriptor,
        config: DecodingConfig,
        logit_source: Optional[LogitSource] = None,
    ) -> GenerationSet:
        """Execute generation for all prompts in a task.

        Parameters
        ----------
        algorithm : DecodingAlgorithm
            Algorithm to generate with.
        task : TaskDescriptor
            Task containing prompts.
        config : DecodingConfig
            Decoding configuration.
        logit_source : LogitSource, optional
            Source of logits (required for actual generation).

        Returns
        -------
        GenerationSet
            Set of generated outputs.
        """
        all_results: List[GenerationResult] = []

        for prompt_idx, prompt in enumerate(task.prompts):
            prompt_ids = self._tokenize_prompt(prompt)

            if logit_source is not None:
                start = _timestamp()
                sequences = algorithm.generate(logit_source, prompt_ids)
                elapsed = _timestamp() - start
            else:
                sequences = self._generate_dummy(config)
                elapsed = 0.0

            for seq_idx, seq in enumerate(sequences):
                result = GenerationResult(
                    sequence=seq,
                    prompt=prompt,
                    algorithm=algorithm.name,
                    config=config.to_dict(),
                    metadata={
                        "prompt_index": prompt_idx,
                        "sequence_index": seq_idx,
                        "elapsed_seconds": elapsed / max(len(sequences), 1),
                        "task": task.name,
                    },
                    score=0.0,
                )
                all_results.append(result)

        gen_set = GenerationSet(
            results=all_results,
            prompt=task.prompts[0] if task.prompts else "",
            algorithm=algorithm.name,
        )
        return gen_set

    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Convert a prompt string to token IDs.

        Uses a simple byte-level fallback when no real tokenizer is available.
        """
        return [int(b) for b in prompt.encode("utf-8")]

    def _generate_dummy(self, config: DecodingConfig) -> List[TokenSequence]:
        """Generate placeholder sequences when no logit source is available.

        Used for metric-only or dry-run evaluation.
        """
        seqs: List[TokenSequence] = []
        for i in range(config.num_sequences):
            length = self._rng.integers(config.min_new_tokens, config.max_new_tokens + 1)
            tokens = [
                Token(
                    token_id=int(self._rng.integers(0, 50_000)),
                    text=f"t{j}",
                    log_prob=float(self._rng.normal(-3.0, 1.0)),
                )
                for j in range(int(length))
            ]
            seqs.append(TokenSequence(tokens=tokens))
        return seqs

    # ------------------------------------------------------------------
    # Metric evaluation
    # ------------------------------------------------------------------

    def _evaluate_metrics(
        self,
        gen_set: GenerationSet,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate all (or specified) metrics on a generation set.

        Parameters
        ----------
        gen_set : GenerationSet
            Generations to evaluate.
        metric_names : list of str, optional
            Subset of metrics to evaluate. Defaults to all registered.

        Returns
        -------
        dict
            Mapping metric_name -> score.
        """
        names = metric_names or list(self._metrics.keys())
        scores: Dict[str, float] = {}

        for name in names:
            entry = self._metrics.get(name)
            if entry is None or entry.fn is None:
                continue

            cached = self._cache.get(name, gen_set)
            if cached is not None:
                scores[name] = cached
                continue

            try:
                value = entry.fn(gen_set)
                scores[name] = float(value)
                self._cache.put(name, gen_set, scores[name])
            except Exception as exc:
                logger.warning("Metric %r failed: %s", name, exc)
                scores[name] = float("nan")

        return scores

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(self, runs: List[ArenaRun]) -> ArenaResult:
        """Aggregate individual runs into a comprehensive ArenaResult.

        Computes summary statistics per algorithm per metric, pairwise
        comparisons, rankings, and Pareto points.

        Parameters
        ----------
        runs : list of ArenaRun
            All runs to aggregate.

        Returns
        -------
        ArenaResult
        """
        successful = [r for r in runs if r.is_successful()]

        summary_stats = self._compute_summary_stats(successful)

        comparisons = self._compute_all_comparisons(successful)

        rankings: Dict[str, List[RankingEntry]] = {}
        for metric_name in self._metrics:
            try:
                ranking = self.rank_algorithms(
                    metric_name, method="mean", runs=successful
                )
                rankings[metric_name] = ranking
            except Exception as exc:
                logger.warning("Failed to rank on %s: %s", metric_name, exc)

        diversity_metrics = self.diversity_metrics()
        quality_metrics = self.quality_metrics()
        pareto_points: List[ParetoPoint] = []
        if diversity_metrics and quality_metrics:
            try:
                pareto_points = self.get_pareto_frontier(
                    diversity_metrics[0],
                    quality_metrics[0],
                    runs=successful,
                )
            except Exception as exc:
                logger.warning("Pareto analysis failed: %s", exc)

        total_dur = sum(r.duration_seconds for r in runs)

        return ArenaResult(
            experiment_id=self.config.experiment_id,
            experiment_name=self.config.experiment_name,
            config=self.config,
            runs=runs,
            summary_stats=summary_stats,
            rankings=rankings,
            pareto_points=pareto_points,
            comparisons=comparisons,
            total_duration_seconds=total_dur,
            num_completed=sum(1 for r in runs if r.status == RunStatus.COMPLETED),
            num_failed=sum(1 for r in runs if r.status == RunStatus.FAILED),
            num_timeout=sum(1 for r in runs if r.status == RunStatus.TIMEOUT),
            metadata={
                "arena_version": _ARENA_VERSION,
                "cache_stats": self._cache.stats(),
            },
        )

    def _compute_summary_stats(
        self, runs: List[ArenaRun]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute summary statistics: algo -> metric -> {mean, std, ...}.

        Parameters
        ----------
        runs : list of ArenaRun
            Successful runs to summarise.

        Returns
        -------
        dict
            Nested mapping algorithm -> metric -> stat -> value.
        """
        algo_runs: Dict[str, List[ArenaRun]] = defaultdict(list)
        for r in runs:
            algo_runs[r.algorithm_name].append(r)

        stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        for algo_name, a_runs in algo_runs.items():
            metric_vals: Dict[str, List[float]] = defaultdict(list)
            for r in a_runs:
                for m, v in r.metric_scores.items():
                    if not math.isnan(v):
                        metric_vals[m].append(v)

            algo_stats: Dict[str, Dict[str, float]] = {}
            for m, vals in metric_vals.items():
                arr = np.array(vals)
                ci = self._compute_bootstrap_ci(arr, self.config.alpha)
                algo_stats[m] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "count": float(len(arr)),
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                }
            stats[algo_name] = algo_stats

        return stats

    def _compute_all_comparisons(
        self, runs: List[ArenaRun]
    ) -> List[ComparisonResult]:
        """Compute pairwise comparisons for every pair of algorithms on every metric.

        Parameters
        ----------
        runs : list of ArenaRun
            Successful runs.

        Returns
        -------
        list of ComparisonResult
        """
        comparisons: List[ComparisonResult] = []
        algo_names = list({r.algorithm_name for r in runs})
        algo_names.sort()

        for i, a in enumerate(algo_names):
            for b in algo_names[i + 1:]:
                for metric_name in self._metrics:
                    try:
                        comp = self.compare_algorithms(
                            a, b, metric_name, runs=runs
                        )
                        comparisons.append(comp)
                    except Exception as exc:
                        logger.debug(
                            "Comparison %s vs %s on %s failed: %s",
                            a, b, metric_name, exc,
                        )

        return comparisons

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    def compare_algorithms(
        self,
        algo_a: str,
        algo_b: str,
        metric: str,
        runs: Optional[List[ArenaRun]] = None,
    ) -> ComparisonResult:
        """Compare two algorithms on a single metric using bootstrap CI
        and Cliff's delta effect size.

        Parameters
        ----------
        algo_a, algo_b : str
            Algorithm names.
        metric : str
            Metric to compare.
        runs : list of ArenaRun, optional
            Runs to use (defaults to all stored runs).

        Returns
        -------
        ComparisonResult
        """
        source_runs = runs if runs is not None else self._runs
        scores_a = np.array([
            r.metric_scores[metric]
            for r in source_runs
            if r.algorithm_name == algo_a
            and r.is_successful()
            and metric in r.metric_scores
            and not math.isnan(r.metric_scores[metric])
        ])
        scores_b = np.array([
            r.metric_scores[metric]
            for r in source_runs
            if r.algorithm_name == algo_b
            and r.is_successful()
            and metric in r.metric_scores
            and not math.isnan(r.metric_scores[metric])
        ])

        if len(scores_a) == 0 or len(scores_b) == 0:
            return ComparisonResult(
                algorithm_a=algo_a,
                algorithm_b=algo_b,
                metric_name=metric,
                num_runs_a=len(scores_a),
                num_runs_b=len(scores_b),
            )

        mean_a = float(np.mean(scores_a))
        mean_b = float(np.mean(scores_b))
        std_a = float(np.std(scores_a, ddof=1)) if len(scores_a) > 1 else 0.0
        std_b = float(np.std(scores_b, ddof=1)) if len(scores_b) > 1 else 0.0

        ci_a = self._compute_bootstrap_ci(scores_a, self.config.alpha)
        ci_b = self._compute_bootstrap_ci(scores_b, self.config.alpha)

        diffs = self._bootstrap_difference(scores_a, scores_b)
        ci_diff = (
            float(np.percentile(diffs, 100 * self.config.alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - self.config.alpha / 2))),
        )

        effect = self._compute_effect_size(scores_a, scores_b)
        magnitude = self._classify_effect_size(effect)

        p_value = self._permutation_test(scores_a, scores_b)

        me = self._metrics.get(metric)
        higher_is_better = me.higher_is_better if me else True

        if p_value < self.config.alpha:
            significant = True
            if higher_is_better:
                winner = algo_a if mean_a > mean_b else algo_b
            else:
                winner = algo_a if mean_a < mean_b else algo_b
        else:
            significant = False
            winner = "tie"

        return ComparisonResult(
            algorithm_a=algo_a,
            algorithm_b=algo_b,
            metric_name=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            effect_size=effect,
            effect_magnitude=magnitude,
            p_value=p_value,
            ci_a=ci_a,
            ci_b=ci_b,
            ci_diff=ci_diff,
            winner=winner,
            significant=significant,
            num_runs_a=len(scores_a),
            num_runs_b=len(scores_b),
        )

    # ------------------------------------------------------------------
    # Bootstrap & effect-size helpers
    # ------------------------------------------------------------------

    def _compute_bootstrap_ci(
        self,
        scores: np.ndarray,
        alpha: float = _DEFAULT_ALPHA,
        n_samples: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for the mean.

        Parameters
        ----------
        scores : ndarray
            1-D array of observed scores.
        alpha : float
            Significance level (e.g. 0.05 for 95 % CI).
        n_samples : int, optional
            Number of bootstrap resamples.

        Returns
        -------
        tuple of float
            (lower, upper) bounds.
        """
        if len(scores) == 0:
            return (0.0, 0.0)
        if len(scores) == 1:
            return (float(scores[0]), float(scores[0]))

        n = n_samples or self.config.bootstrap_samples
        rng = np.random.default_rng(self.config.seed)
        boot_means = np.empty(n)
        for i in range(n):
            sample = rng.choice(scores, size=len(scores), replace=True)
            boot_means[i] = np.mean(sample)

        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return (lower, upper)

    def _bootstrap_difference(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> np.ndarray:
        """Bootstrap distribution of mean(A) - mean(B).

        Parameters
        ----------
        scores_a, scores_b : ndarray
            Score arrays for two algorithms.

        Returns
        -------
        ndarray
            Array of bootstrapped differences.
        """
        n = self.config.bootstrap_samples
        rng = np.random.default_rng(self.config.seed + 1)
        diffs = np.empty(n)
        na, nb = len(scores_a), len(scores_b)
        for i in range(n):
            sa = rng.choice(scores_a, size=na, replace=True)
            sb = rng.choice(scores_b, size=nb, replace=True)
            diffs[i] = np.mean(sa) - np.mean(sb)
        return diffs

    def _compute_effect_size(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Compute Cliff's delta effect size between two score distributions.

        Cliff's delta is a non-parametric measure of how often values in one
        group are larger than values in the other.  Range is [-1, 1].

        Parameters
        ----------
        a, b : ndarray
            Score arrays.

        Returns
        -------
        float
            Cliff's delta in [-1, 1].
        """
        if len(a) == 0 or len(b) == 0:
            return 0.0

        n_a, n_b = len(a), len(b)
        more = 0
        less = 0
        for ai in a:
            for bi in b:
                if ai > bi:
                    more += 1
                elif ai < bi:
                    less += 1

        delta = (more - less) / (n_a * n_b)
        return float(delta)

    @staticmethod
    def _classify_effect_size(delta: float) -> str:
        """Classify magnitude of Cliff's delta.

        Uses thresholds from Romano et al. (2006):
            |d| < 0.147 -> negligible
            |d| < 0.33  -> small
            |d| < 0.474 -> medium
            else         -> large

        Parameters
        ----------
        delta : float
            Cliff's delta value.

        Returns
        -------
        str
            "negligible", "small", "medium", or "large".
        """
        d = abs(delta)
        if d < 0.147:
            return "negligible"
        if d < 0.33:
            return "small"
        if d < 0.474:
            return "medium"
        return "large"

    def _permutation_test(
        self,
        a: np.ndarray,
        b: np.ndarray,
        n_permutations: int = 10_000,
    ) -> float:
        """Two-sided permutation test for difference in means.

        Parameters
        ----------
        a, b : ndarray
            Score arrays.
        n_permutations : int
            Number of permutations.

        Returns
        -------
        float
            Approximate p-value.
        """
        if len(a) == 0 or len(b) == 0:
            return 1.0

        observed = abs(float(np.mean(a) - np.mean(b)))
        combined = np.concatenate([a, b])
        n_a = len(a)
        rng = np.random.default_rng(self.config.seed + 2)

        count = 0
        for _ in range(n_permutations):
            rng.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_diff = abs(float(np.mean(perm_a) - np.mean(perm_b)))
            if perm_diff >= observed:
                count += 1

        return (count + 1) / (n_permutations + 1)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_algorithms(
        self,
        metric: str,
        method: str = "mean",
        runs: Optional[List[ArenaRun]] = None,
    ) -> List[RankingEntry]:
        """Rank all algorithms by a metric.

        Parameters
        ----------
        metric : str
            Metric to rank on.
        method : str
            "mean" — rank by mean score.
            "bradley_terry" — iterative pairwise model.
            "win_rate" — fraction of pairwise wins.

        Returns
        -------
        list of RankingEntry
            Sorted best-first.
        """
        source_runs = runs if runs is not None else self._runs
        successful = [r for r in source_runs if r.is_successful()]

        algo_scores: Dict[str, List[float]] = defaultdict(list)
        for r in successful:
            if metric in r.metric_scores and not math.isnan(r.metric_scores[metric]):
                algo_scores[r.algorithm_name].append(r.metric_scores[metric])

        if not algo_scores:
            return []

        me = self._metrics.get(metric)
        higher_is_better = me.higher_is_better if me else True

        if method == "bradley_terry":
            return self._rank_bradley_terry(algo_scores, metric, higher_is_better)
        elif method == "win_rate":
            return self._rank_win_rate(algo_scores, metric, higher_is_better)
        else:
            return self._rank_by_mean(algo_scores, metric, higher_is_better)

    def _rank_by_mean(
        self,
        algo_scores: Dict[str, List[float]],
        metric: str,
        higher_is_better: bool,
    ) -> List[RankingEntry]:
        """Rank algorithms by mean score.

        Parameters
        ----------
        algo_scores : dict
            algorithm -> list of scores.
        metric : str
            Metric name (for logging).
        higher_is_better : bool
            Sort direction.

        Returns
        -------
        list of RankingEntry
        """
        entries: List[RankingEntry] = []
        algo_names = sorted(algo_scores.keys())

        win_counts: Dict[str, Dict[str, int]] = {
            a: {"wins": 0, "losses": 0, "ties": 0} for a in algo_names
        }
        for i, a in enumerate(algo_names):
            for b in algo_names[i + 1:]:
                mean_a = float(np.mean(algo_scores[a]))
                mean_b = float(np.mean(algo_scores[b]))
                if abs(mean_a - mean_b) < 1e-12:
                    win_counts[a]["ties"] += 1
                    win_counts[b]["ties"] += 1
                elif (mean_a > mean_b) == higher_is_better:
                    win_counts[a]["wins"] += 1
                    win_counts[b]["losses"] += 1
                else:
                    win_counts[b]["wins"] += 1
                    win_counts[a]["losses"] += 1

        for name, scores_list in algo_scores.items():
            arr = np.array(scores_list)
            ci = self._compute_bootstrap_ci(arr, self.config.alpha)
            entries.append(
                RankingEntry(
                    algorithm_name=name,
                    score=float(np.mean(arr)),
                    wins=win_counts[name]["wins"],
                    losses=win_counts[name]["losses"],
                    ties=win_counts[name]["ties"],
                    mean_metric=float(np.mean(arr)),
                    ci=ci,
                )
            )

        entries.sort(key=lambda e: e.score, reverse=higher_is_better)
        for i, e in enumerate(entries):
            e.rank = i + 1

        return entries

    def _rank_bradley_terry(
        self,
        algo_scores: Dict[str, List[float]],
        metric: str,
        higher_is_better: bool,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> List[RankingEntry]:
        """Rank using a Bradley-Terry pairwise preference model.

        The BT model assigns strength parameter pi_i to each algorithm
        such that P(i beats j) = pi_i / (pi_i + pi_j).  We estimate
        strengths via the iterative MM algorithm of Hunter (2004).

        Parameters
        ----------
        algo_scores : dict
            algorithm -> list of scores.
        metric : str
            Metric name.
        higher_is_better : bool
            Direction.
        max_iter : int
            Maximum MM iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        list of RankingEntry
        """
        names = sorted(algo_scores.keys())
        n = len(names)
        if n < 2:
            return self._rank_by_mean(algo_scores, metric, higher_is_better)

        wins = np.zeros((n, n))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                sa = np.array(algo_scores[a])
                sb = np.array(algo_scores[b])
                w = 0
                for va in sa:
                    for vb in sb:
                        if higher_is_better:
                            if va > vb:
                                w += 1
                            elif va == vb:
                                w += 0.5
                        else:
                            if va < vb:
                                w += 1
                            elif va == vb:
                                w += 0.5
                wins[i, j] = w

        pi = np.ones(n) / n

        for iteration in range(max_iter):
            pi_old = pi.copy()
            for i in range(n):
                numerator = 0.0
                denominator = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    total_ij = wins[i, j] + wins[j, i]
                    if total_ij == 0:
                        continue
                    numerator += wins[i, j]
                    denominator += total_ij / (pi[i] + pi[j])

                if denominator > 0:
                    pi[i] = numerator / denominator
                else:
                    pi[i] = pi_old[i]

            pi_sum = pi.sum()
            if pi_sum > 0:
                pi /= pi_sum

            if np.max(np.abs(pi - pi_old)) < tol:
                logger.debug(
                    "Bradley-Terry converged after %d iterations", iteration + 1
                )
                break

        entries: List[RankingEntry] = []
        for idx, name in enumerate(names):
            arr = np.array(algo_scores[name])
            ci = self._compute_bootstrap_ci(arr, self.config.alpha)
            total_wins = int(np.sum(wins[idx, :]))
            total_losses = int(np.sum(wins[:, idx]))
            entries.append(
                RankingEntry(
                    algorithm_name=name,
                    score=float(pi[idx]),
                    wins=total_wins,
                    losses=total_losses,
                    ties=0,
                    mean_metric=float(np.mean(arr)),
                    ci=ci,
                )
            )

        entries.sort(key=lambda e: e.score, reverse=True)
        for i, e in enumerate(entries):
            e.rank = i + 1
        return entries

    def _rank_win_rate(
        self,
        algo_scores: Dict[str, List[float]],
        metric: str,
        higher_is_better: bool,
    ) -> List[RankingEntry]:
        """Rank by pairwise win rate across all matchups.

        For each pair (A, B) we compare every sample of A against every
        sample of B.  The win rate for A is the fraction of comparisons won.

        Parameters
        ----------
        algo_scores : dict
            algorithm -> list of scores.
        metric : str
            Metric name.
        higher_is_better : bool
            Direction.

        Returns
        -------
        list of RankingEntry
        """
        names = sorted(algo_scores.keys())
        n = len(names)

        win_counts: Dict[str, int] = {a: 0 for a in names}
        loss_counts: Dict[str, int] = {a: 0 for a in names}
        tie_counts: Dict[str, int] = {a: 0 for a in names}
        total_matches: Dict[str, int] = {a: 0 for a in names}

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa = np.array(algo_scores[a])
                sb = np.array(algo_scores[b])
                a_wins = 0
                b_wins = 0
                ties = 0
                for va in sa:
                    for vb in sb:
                        if abs(va - vb) < 1e-12:
                            ties += 1
                        elif (va > vb) == higher_is_better:
                            a_wins += 1
                        else:
                            b_wins += 1

                total = a_wins + b_wins + ties
                win_counts[a] += a_wins
                loss_counts[a] += b_wins
                tie_counts[a] += ties
                total_matches[a] += total

                win_counts[b] += b_wins
                loss_counts[b] += a_wins
                tie_counts[b] += ties
                total_matches[b] += total

        entries: List[RankingEntry] = []
        for name in names:
            arr = np.array(algo_scores[name])
            ci = self._compute_bootstrap_ci(arr, self.config.alpha)
            t = total_matches[name]
            rate = win_counts[name] / t if t > 0 else 0.0
            entries.append(
                RankingEntry(
                    algorithm_name=name,
                    score=rate,
                    wins=win_counts[name],
                    losses=loss_counts[name],
                    ties=tie_counts[name],
                    mean_metric=float(np.mean(arr)),
                    ci=ci,
                )
            )

        entries.sort(key=lambda e: e.score, reverse=True)
        for i, e in enumerate(entries):
            e.rank = i + 1
        return entries

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def get_pareto_frontier(
        self,
        diversity_metric: str,
        quality_metric: str,
        runs: Optional[List[ArenaRun]] = None,
    ) -> List[ParetoPoint]:
        """Compute the 2-D Pareto frontier (higher is better on both axes).

        Each algorithm is represented by its mean scores across all runs.
        Non-dominated points form the frontier.

        Parameters
        ----------
        diversity_metric : str
            Metric for the x-axis (diversity).
        quality_metric : str
            Metric for the y-axis (quality).
        runs : list of ArenaRun, optional
            Runs to use (defaults to stored runs).

        Returns
        -------
        list of ParetoPoint
            All points with ``dominated`` flag set correctly.
        """
        source_runs = runs if runs is not None else self._runs
        successful = [r for r in source_runs if r.is_successful()]

        algo_div: Dict[str, List[float]] = defaultdict(list)
        algo_qual: Dict[str, List[float]] = defaultdict(list)

        for r in successful:
            if diversity_metric in r.metric_scores and quality_metric in r.metric_scores:
                dv = r.metric_scores[diversity_metric]
                qv = r.metric_scores[quality_metric]
                if not (math.isnan(dv) or math.isnan(qv)):
                    algo_div[r.algorithm_name].append(dv)
                    algo_qual[r.algorithm_name].append(qv)

        points: List[ParetoPoint] = []
        for name in algo_div:
            point = ParetoPoint(
                algorithm_name=name,
                diversity_score=float(np.mean(algo_div[name])),
                quality_score=float(np.mean(algo_qual[name])),
                metadata={
                    "diversity_std": float(np.std(algo_div[name])),
                    "quality_std": float(np.std(algo_qual[name])),
                    "n_runs": len(algo_div[name]),
                },
            )
            points.append(point)

        for i, p in enumerate(points):
            p.dominated = any(
                q.dominates(p) for j, q in enumerate(points) if j != i
            )

        frontier = [p for p in points if not p.dominated]
        frontier.sort(key=lambda p: p.diversity_score)

        return points

    def get_pareto_frontier_nondominated(
        self,
        diversity_metric: str,
        quality_metric: str,
        runs: Optional[List[ArenaRun]] = None,
    ) -> List[ParetoPoint]:
        """Return only the non-dominated points from the Pareto frontier.

        Parameters
        ----------
        diversity_metric, quality_metric : str
            Metric names.
        runs : list of ArenaRun, optional
            Source runs.

        Returns
        -------
        list of ParetoPoint
            Non-dominated points only.
        """
        all_pts = self.get_pareto_frontier(diversity_metric, quality_metric, runs)
        return [p for p in all_pts if not p.dominated]

    def compute_hypervolume(
        self,
        diversity_metric: str,
        quality_metric: str,
        reference_point: Optional[Tuple[float, float]] = None,
        runs: Optional[List[ArenaRun]] = None,
    ) -> float:
        """Compute the hypervolume indicator for the Pareto frontier.

        Uses the 2-D sweep-line algorithm.

        Parameters
        ----------
        diversity_metric, quality_metric : str
            Metric names.
        reference_point : tuple of float, optional
            (div_ref, qual_ref) reference point.  Defaults to (0, 0).
        runs : list of ArenaRun, optional
            Source runs.

        Returns
        -------
        float
            Hypervolume dominated by the non-dominated set.
        """
        frontier = self.get_pareto_frontier_nondominated(
            diversity_metric, quality_metric, runs
        )
        if not frontier:
            return 0.0

        ref = reference_point or (0.0, 0.0)

        sorted_pts = sorted(frontier, key=lambda p: p.diversity_score)

        hv = 0.0
        prev_quality = ref[1]

        for pt in sorted_pts:
            if pt.diversity_score <= ref[0] or pt.quality_score <= ref[1]:
                continue
            width = pt.diversity_score - ref[0]
            height = pt.quality_score - prev_quality
            if height > 0:
                hv += width * height
            prev_quality = max(prev_quality, pt.quality_score)

        last_div = sorted_pts[-1].diversity_score if sorted_pts else ref[0]
        if last_div > ref[0]:
            remaining_width = last_div - ref[0]
            max_qual = max(p.quality_score for p in sorted_pts)
            hv = 0.0
            pts_above_ref = [
                p for p in sorted_pts
                if p.diversity_score > ref[0] and p.quality_score > ref[1]
            ]
            if not pts_above_ref:
                return 0.0

            pts_above_ref.sort(key=lambda p: p.diversity_score)

            prev_div = ref[0]
            hv = 0.0
            remaining_pts = list(pts_above_ref)

            for pt in pts_above_ref:
                width = pt.diversity_score - prev_div
                if width > 0 and remaining_pts:
                    max_q = max(p.quality_score for p in remaining_pts)
                    hv += width * (max_q - ref[1])
                prev_div = pt.diversity_score
                remaining_pts = [
                    p for p in remaining_pts
                    if p.diversity_score > pt.diversity_score
                ]

        return hv

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        algorithm_name: str,
        task_name: str,
        k_folds: int = _DEFAULT_K_FOLDS,
    ) -> CrossValidationResult:
        """Perform k-fold cross-validation of an algorithm on a task.

        The task's prompts are split into *k* folds.  For each fold,
        generation and evaluation are run on the held-out split.

        Parameters
        ----------
        algorithm_name : str
            Algorithm to evaluate.
        task_name : str
            Task whose prompts are split.
        k_folds : int
            Number of folds.

        Returns
        -------
        CrossValidationResult
        """
        entry = self._algorithms.get(algorithm_name)
        if entry is None:
            raise KeyError(f"Algorithm not registered: {algorithm_name!r}")
        task = self._tasks.get(task_name)
        if task is None:
            raise KeyError(f"Task not registered: {task_name!r}")

        prompts = list(task.prompts)
        n = len(prompts)
        if n < k_folds:
            k_folds = max(n, 1)
            logger.warning(
                "Reducing k_folds to %d (only %d prompts)", k_folds, n
            )

        indices = list(range(n))
        rng = np.random.default_rng(self.config.seed)
        rng.shuffle(indices)

        fold_sizes = [n // k_folds] * k_folds
        for i in range(n % k_folds):
            fold_sizes[i] += 1

        folds: List[List[int]] = []
        start = 0
        for size in fold_sizes:
            folds.append(indices[start: start + size])
            start += size

        fold_scores: List[Dict[str, float]] = []

        for fold_idx in range(k_folds):
            test_indices = set(folds[fold_idx])
            test_prompts = [prompts[i] for i in sorted(test_indices)]

            if not test_prompts:
                continue

            fold_task = TaskDescriptor(
                name=f"{task_name}_fold{fold_idx}",
                domain=task.domain,
                prompts=test_prompts,
                reference_texts=[
                    task.reference_texts[i]
                    for i in sorted(test_indices)
                    if i < len(task.reference_texts)
                ],
            )

            logit_source = self._logit_sources.get(algorithm_name)
            config = copy.deepcopy(entry.config)
            if self.config.seed is not None:
                config.seed = self.config.seed + fold_idx

            try:
                gen_set = self._execute_generation(
                    entry.algorithm, fold_task, config, logit_source
                )
                scores = self._evaluate_metrics(gen_set)
                fold_scores.append(scores)
            except Exception as exc:
                logger.warning("Fold %d failed: %s", fold_idx, exc)
                fold_scores.append({})

        if not fold_scores:
            return CrossValidationResult(
                algorithm_name=algorithm_name,
                task_name=task_name,
                k=k_folds,
            )

        all_metric_names: Set[str] = set()
        for fs in fold_scores:
            all_metric_names.update(fs.keys())

        mean_scores: Dict[str, float] = {}
        std_scores: Dict[str, float] = {}
        ci_scores: Dict[str, Tuple[float, float]] = {}

        for m in sorted(all_metric_names):
            vals = [
                fs[m] for fs in fold_scores
                if m in fs and not math.isnan(fs[m])
            ]
            if vals:
                arr = np.array(vals)
                mean_scores[m] = float(np.mean(arr))
                std_scores[m] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                ci_scores[m] = self._compute_bootstrap_ci(arr, self.config.alpha)
            else:
                mean_scores[m] = float("nan")
                std_scores[m] = 0.0
                ci_scores[m] = (float("nan"), float("nan"))

        return CrossValidationResult(
            algorithm_name=algorithm_name,
            task_name=task_name,
            k=k_folds,
            fold_scores=fold_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            ci_scores=ci_scores,
        )

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def ablation_study(
        self,
        algorithm_name: str,
        params_to_ablate: Dict[str, Any],
        task_name: Optional[str] = None,
    ) -> AblationResult:
        """Perform an ablation study by removing / resetting parameters.

        For each parameter in *params_to_ablate*, the algorithm is run
        with that parameter set to its ablated (default) value while
        keeping all others at baseline.  The impact of each parameter
        is measured as the change in metric scores.

        Parameters
        ----------
        algorithm_name : str
            Algorithm to study.
        params_to_ablate : dict
            param_name -> ablated (default) value.
        task_name : str, optional
            Task to evaluate on (defaults to first registered task).

        Returns
        -------
        AblationResult
        """
        entry = self._algorithms.get(algorithm_name)
        if entry is None:
            raise KeyError(f"Algorithm not registered: {algorithm_name!r}")

        t_name = task_name or next(iter(self._tasks), None)
        if t_name is None:
            raise ValueError("No tasks registered for ablation study")

        baseline_runs: List[ArenaRun] = []
        for _ in range(self.config.num_runs):
            run = self.run_single(algorithm_name, t_name)
            baseline_runs.append(run)

        baseline_scores: Dict[str, float] = {}
        for metric_name in self._metrics:
            vals = [
                r.metric_scores.get(metric_name, float("nan"))
                for r in baseline_runs
                if r.is_successful()
            ]
            clean = [v for v in vals if not math.isnan(v)]
            baseline_scores[metric_name] = float(np.mean(clean)) if clean else float("nan")

        ablation_scores: Dict[str, Dict[str, float]] = {}
        deltas: Dict[str, Dict[str, float]] = {}

        for param_name, ablated_value in params_to_ablate.items():
            ablated_config = copy.deepcopy(entry.config)
            if hasattr(ablated_config, param_name):
                setattr(ablated_config, param_name, ablated_value)
            else:
                ablated_config.params[param_name] = ablated_value

            abl_runs: List[ArenaRun] = []
            for _ in range(self.config.num_runs):
                run = self.run_single(algorithm_name, t_name, config=ablated_config)
                abl_runs.append(run)

            param_scores: Dict[str, float] = {}
            param_deltas: Dict[str, float] = {}
            for metric_name in self._metrics:
                vals = [
                    r.metric_scores.get(metric_name, float("nan"))
                    for r in abl_runs
                    if r.is_successful()
                ]
                clean = [v for v in vals if not math.isnan(v)]
                score = float(np.mean(clean)) if clean else float("nan")
                param_scores[metric_name] = score

                bl = baseline_scores.get(metric_name, float("nan"))
                if not (math.isnan(score) or math.isnan(bl)):
                    param_deltas[metric_name] = score - bl
                else:
                    param_deltas[metric_name] = 0.0

            ablation_scores[param_name] = param_scores
            deltas[param_name] = param_deltas

        importance: List[Tuple[str, float]] = []
        for param_name, param_deltas in deltas.items():
            total_impact = sum(abs(d) for d in param_deltas.values())
            importance.append((param_name, total_impact))
        importance.sort(key=lambda x: x[1], reverse=True)

        return AblationResult(
            algorithm_name=algorithm_name,
            baseline_scores=baseline_scores,
            ablation_scores=ablation_scores,
            deltas=deltas,
            importance_ranking=importance,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report dictionary.

        Returns
        -------
        dict
            Report with experiment info, summary stats, comparisons,
            rankings, Pareto analysis, and configuration details.
        """
        result = self._result
        if result is None:
            result = self._aggregate_results(self._runs)

        report: Dict[str, Any] = {
            "report_version": _ARENA_VERSION,
            "generated_at": _iso_now(),
            "experiment": {
                "id": self.config.experiment_id,
                "name": self.config.experiment_name,
                "seed": self.config.seed,
                "num_runs": self.config.num_runs,
            },
            "overview": {
                "num_algorithms": len(self.list_algorithms()),
                "num_tasks": len(self._tasks),
                "num_metrics": len(self._metrics),
                "total_runs": len(self._runs),
                "completed": result.num_completed,
                "failed": result.num_failed,
                "timed_out": result.num_timeout,
                "total_duration_seconds": result.total_duration_seconds,
            },
            "algorithms": {
                name: {
                    "description": e.description,
                    "num_configs": e.num_configs(),
                    "tags": e.tags,
                }
                for name, e in self._algorithms.items()
                if e.enabled
            },
            "tasks": {
                name: {
                    "domain": t.domain.name,
                    "num_prompts": t.num_prompts(),
                }
                for name, t in self._tasks.items()
            },
            "metrics": {
                name: {
                    "type": e.metric_type.name,
                    "higher_is_better": e.higher_is_better,
                    "description": e.description,
                }
                for name, e in self._metrics.items()
            },
            "summary_stats": result.summary_stats,
            "rankings": {
                m: [e.to_dict() for e in entries]
                for m, entries in result.rankings.items()
            },
            "comparisons": [c.to_dict() for c in result.comparisons],
            "pareto_frontier": [p.to_dict() for p in result.pareto_points],
            "cache_stats": self._cache.stats(),
        }

        best_per_metric: Dict[str, Dict[str, Any]] = {}
        for metric_name, entries in result.rankings.items():
            if entries:
                best = entries[0]
                best_per_metric[metric_name] = {
                    "algorithm": best.algorithm_name,
                    "mean_score": best.mean_metric,
                    "ci": list(best.ci),
                }
        report["best_per_metric"] = best_per_metric

        significant_comparisons = [
            c.summary() for c in result.comparisons if c.significant
        ]
        report["significant_findings"] = significant_comparisons

        return report

    def generate_report_text(self) -> str:
        """Generate a plain-text summary report.

        Returns
        -------
        str
            Multi-line text report.
        """
        report = self.generate_report()
        lines: List[str] = []

        lines.append("=" * 72)
        lines.append(f"  EVALUATION REPORT: {report['experiment']['name']}")
        lines.append(f"  ID: {report['experiment']['id']}")
        lines.append(f"  Generated: {report['generated_at']}")
        lines.append("=" * 72)
        lines.append("")

        ov = report["overview"]
        lines.append("OVERVIEW")
        lines.append(f"  Algorithms: {ov['num_algorithms']}")
        lines.append(f"  Tasks:      {ov['num_tasks']}")
        lines.append(f"  Metrics:    {ov['num_metrics']}")
        lines.append(f"  Total runs: {ov['total_runs']} "
                      f"(ok={ov['completed']}, fail={ov['failed']}, "
                      f"timeout={ov['timed_out']})")
        lines.append(f"  Duration:   {ov['total_duration_seconds']:.1f}s")
        lines.append("")

        lines.append("SUMMARY STATISTICS")
        for algo, metrics in report.get("summary_stats", {}).items():
            lines.append(f"  {algo}:")
            for metric, stats in metrics.items():
                lines.append(
                    f"    {metric}: mean={stats['mean']:.4f} "
                    f"std={stats['std']:.4f} "
                    f"[{stats.get('ci_lower', 0):.4f}, "
                    f"{stats.get('ci_upper', 0):.4f}]"
                )
        lines.append("")

        lines.append("RANKINGS")
        for metric_name, entries in report.get("rankings", {}).items():
            lines.append(f"  {metric_name}:")
            for e in entries:
                lines.append(
                    f"    #{e['rank']} {e['algorithm_name']}: "
                    f"score={e['score']:.4f} "
                    f"(W={e['wins']} L={e['losses']} T={e['ties']})"
                )
        lines.append("")

        findings = report.get("significant_findings", [])
        if findings:
            lines.append("SIGNIFICANT FINDINGS")
            for f in findings:
                lines.append(f"  • {f}")
            lines.append("")

        best = report.get("best_per_metric", {})
        if best:
            lines.append("BEST ALGORITHMS")
            for metric, info in best.items():
                lines.append(
                    f"  {metric}: {info['algorithm']} "
                    f"(mean={info['mean_score']:.4f})"
                )
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, fmt: str = "json") -> str:
        """Export results in the specified format.

        Parameters
        ----------
        fmt : str
            "json", "csv", or "pickle".

        Returns
        -------
        str
            Serialised string (json/csv) or path to pickle file.

        Raises
        ------
        ValueError
            If *fmt* is unsupported.
        """
        if self._result is None:
            self._result = self._aggregate_results(self._runs)

        if fmt == "json":
            return self._export_json()
        elif fmt == "csv":
            return self._export_csv()
        elif fmt == "pickle":
            return self._export_pickle()
        elif fmt == "latex":
            return self._export_latex()
        else:
            raise ValueError(f"Unsupported export format: {fmt!r}")

    def _export_json(self) -> str:
        """Serialise results as a JSON string."""
        assert self._result is not None
        data = self._result.to_dict()
        return json.dumps(data, indent=2, default=str)

    def _export_csv(self) -> str:
        """Export per-run results as CSV rows.

        Columns: run_id, algorithm, task, status, duration, metric1, metric2, ...
        """
        assert self._result is not None
        metric_names = sorted(self._result.metric_names())

        header = ["run_id", "algorithm", "task", "status", "duration"]
        header.extend(metric_names)
        rows: List[str] = [",".join(header)]

        for run in self._result.runs:
            row = [
                run.run_id,
                run.algorithm_name,
                run.task_name,
                run.status.name,
                f"{run.duration_seconds:.4f}",
            ]
            for m in metric_names:
                val = run.metric_scores.get(m, float("nan"))
                row.append(f"{val:.6f}")
            rows.append(",".join(row))

        return "\n".join(rows)

    def _export_pickle(self) -> str:
        """Pickle results to disk and return the file path."""
        assert self._result is not None
        path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_results.pkl",
        )
        with open(path, "wb") as f:
            pickle.dump(self._result, f)
        logger.info("Results pickled to %s", path)
        return path

    def _export_latex(self) -> str:
        """Export summary statistics as a LaTeX table.

        Returns
        -------
        str
            LaTeX tabular environment.
        """
        assert self._result is not None
        metric_names = sorted(self._result.metric_names())
        algo_names = self._result.algorithm_names()

        col_spec = "l" + "r" * len(metric_names)
        lines: List[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Evaluation results for "
            + self.config.experiment_name.replace("_", r"\_")
            + r"}"
        )
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")

        header = "Algorithm"
        for m in metric_names:
            header += " & " + m.replace("_", r"\_")
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        stats = self._result.summary_stats
        for algo in algo_names:
            row = algo.replace("_", r"\_")
            algo_stats = stats.get(algo, {})
            for m in metric_names:
                m_stats = algo_stats.get(m, {})
                mean_val = m_stats.get("mean", float("nan"))
                std_val = m_stats.get("std", 0.0)
                if math.isnan(mean_val):
                    row += " & --"
                else:
                    row += f" & ${mean_val:.3f} \\pm {std_val:.3f}$"
            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> List[Dict[str, Any]]:
        """Convert results to a list of row-dicts (tabular form).

        Each row corresponds to one run, with columns for run_id,
        algorithm, task, status, duration, and each metric score.

        Returns
        -------
        list of dict
            Rows suitable for pandas.DataFrame construction.
        """
        rows: List[Dict[str, Any]] = []
        for run in self._runs:
            row: Dict[str, Any] = {
                "run_id": run.run_id,
                "algorithm": run.algorithm_name,
                "task": run.task_name,
                "status": run.status.name,
                "duration": run.duration_seconds,
                "attempt": run.attempt,
            }
            if run.config is not None:
                row["temperature"] = run.config.temperature
                row["num_sequences"] = run.config.num_sequences
                row["max_new_tokens"] = run.config.max_new_tokens
                for k, v in run.config.params.items():
                    row[f"param_{k}"] = v

            for m, v in run.metric_scores.items():
                row[f"metric_{m}"] = v
            rows.append(row)

        return rows

    def to_dataframe_wide(self) -> List[Dict[str, Any]]:
        """Convert results to wide format: one row per algorithm with mean
        scores for each metric across all tasks.

        Returns
        -------
        list of dict
            Rows suitable for pandas.DataFrame construction.
        """
        result = self._result or self._aggregate_results(self._runs)
        rows: List[Dict[str, Any]] = []

        for algo_name, metric_stats in result.summary_stats.items():
            row: Dict[str, Any] = {"algorithm": algo_name}
            for metric_name, stats in metric_stats.items():
                row[f"{metric_name}_mean"] = stats["mean"]
                row[f"{metric_name}_std"] = stats["std"]
                row[f"{metric_name}_ci_lower"] = stats.get("ci_lower", 0.0)
                row[f"{metric_name}_ci_upper"] = stats.get("ci_upper", 0.0)
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> Optional[str]:
        """Save current arena state to a checkpoint file.

        Returns
        -------
        str or None
            Path to the checkpoint file, or None on failure.
        """
        path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_checkpoint.pkl",
        )
        try:
            state = {
                "version": _CHECKPOINT_VERSION,
                "config": self.config.to_dict(),
                "runs": [r.to_dict() for r in self._runs],
                "run_counter": self._run_counter,
                "timestamp": _iso_now(),
                "algorithm_names": list(self._algorithms.keys()),
                "task_names": list(self._tasks.keys()),
                "metric_names": list(self._metrics.keys()),
            }
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)
            self._checkpoint_path = Path(path)
            logger.debug("Checkpoint saved: %s (%d runs)", path, len(self._runs))
            return path
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
            return None

    def load_checkpoint(self, path: Optional[str] = None) -> int:
        """Load arena state from a checkpoint file.

        Parameters
        ----------
        path : str, optional
            Checkpoint file path.  Defaults to the standard location.

        Returns
        -------
        int
            Number of runs restored.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        ValueError
            If the checkpoint version is incompatible.
        """
        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f"{self.config.experiment_name}_checkpoint.pkl",
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        version = state.get("version", 0)
        if version > _CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {version} is newer than "
                f"supported version {_CHECKPOINT_VERSION}"
            )

        restored_runs = [ArenaRun.from_dict(rd) for rd in state.get("runs", [])]
        self._runs = restored_runs
        self._run_counter = state.get("run_counter", len(restored_runs))
        self._checkpoint_path = Path(path)

        logger.info(
            "Checkpoint loaded: %s (%d runs, counter=%d)",
            path,
            len(restored_runs),
            self._run_counter,
        )
        return len(restored_runs)

    def has_checkpoint(self) -> bool:
        """Check whether a checkpoint file exists for this experiment."""
        path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_checkpoint.pkl",
        )
        return os.path.exists(path)

    def delete_checkpoint(self) -> bool:
        """Delete the checkpoint file if it exists."""
        path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_checkpoint.pkl",
        )
        if os.path.exists(path):
            os.remove(path)
            logger.info("Checkpoint deleted: %s", path)
            return True
        return False

    # ------------------------------------------------------------------
    # Error / timeout / progress handlers
    # ------------------------------------------------------------------

    def _handle_timeout(self, run: ArenaRun) -> None:
        """Handle a timed-out run.

        Parameters
        ----------
        run : ArenaRun
            The run that timed out.
        """
        run.mark_timeout()
        logger.warning(
            "Run %s timed out after %ds: algo=%s task=%s",
            run.run_id,
            self.config.timeout_seconds,
            run.algorithm_name,
            run.task_name,
        )

    def _handle_error(self, run: ArenaRun, exc: Exception) -> None:
        """Handle a failed run.

        Parameters
        ----------
        run : ArenaRun
            The run that failed.
        exc : Exception
            The exception raised.
        """
        error_msg = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        run.mark_failed(error_msg)
        run.metadata["traceback"] = tb
        logger.error(
            "Run %s failed: algo=%s task=%s error=%s",
            run.run_id,
            run.algorithm_name,
            run.task_name,
            error_msg,
        )

    def _log_progress(self, tracker: _ProgressTracker, run: ArenaRun) -> None:
        """Log progress using the tracker.

        Parameters
        ----------
        tracker : _ProgressTracker
            Progress tracking instance.
        run : ArenaRun
            The most recently completed run.
        """
        tracker.log(run)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation of an ongoing ``run_all`` evaluation."""
        self._cancelled = True
        logger.info("Cancellation requested")

    def reset(self) -> None:
        """Reset arena state, clearing all runs and results."""
        self._runs.clear()
        self._result = None
        self._run_counter = 0
        self._cancelled = False
        self._cache.clear()
        logger.info("Arena state reset")

    # ------------------------------------------------------------------
    # Computed properties & queries
    # ------------------------------------------------------------------

    @property
    def num_runs(self) -> int:
        return len(self._runs)

    @property
    def num_completed(self) -> int:
        return sum(1 for r in self._runs if r.status == RunStatus.COMPLETED)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self._runs if r.status == RunStatus.FAILED)

    def get_runs(
        self,
        algorithm: Optional[str] = None,
        task: Optional[str] = None,
        status: Optional[RunStatus] = None,
    ) -> List[ArenaRun]:
        """Filter stored runs by algorithm, task, and/or status.

        Parameters
        ----------
        algorithm : str, optional
            Filter by algorithm name.
        task : str, optional
            Filter by task name.
        status : RunStatus, optional
            Filter by status.

        Returns
        -------
        list of ArenaRun
        """
        runs = self._runs
        if algorithm is not None:
            runs = [r for r in runs if r.algorithm_name == algorithm]
        if task is not None:
            runs = [r for r in runs if r.task_name == task]
        if status is not None:
            runs = [r for r in runs if r.status == status]
        return runs

    def get_best_run(
        self,
        metric: str,
        algorithm: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Optional[ArenaRun]:
        """Return the run with the highest score on a metric.

        Parameters
        ----------
        metric : str
            Metric to optimise.
        algorithm : str, optional
            Restrict to one algorithm.
        task : str, optional
            Restrict to one task.

        Returns
        -------
        ArenaRun or None
        """
        runs = self.get_runs(algorithm=algorithm, task=task, status=RunStatus.COMPLETED)
        valid = [
            r for r in runs
            if metric in r.metric_scores and not math.isnan(r.metric_scores[metric])
        ]
        if not valid:
            return None

        me = self._metrics.get(metric)
        higher_is_better = me.higher_is_better if me else True
        return max(valid, key=lambda r: r.metric_scores[metric] * (1 if higher_is_better else -1))

    def scores_for_metric(
        self,
        metric: str,
        algorithm: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Return score arrays per algorithm for one metric.

        Parameters
        ----------
        metric : str
            Metric name.
        algorithm : str, optional
            Restrict to one algorithm.

        Returns
        -------
        dict
            algorithm -> ndarray of scores.
        """
        result: Dict[str, List[float]] = defaultdict(list)
        for r in self._runs:
            if not r.is_successful():
                continue
            if algorithm is not None and r.algorithm_name != algorithm:
                continue
            if metric in r.metric_scores and not math.isnan(r.metric_scores[metric]):
                result[r.algorithm_name].append(r.metric_scores[metric])
        return {k: np.array(v) for k, v in result.items()}

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def compute_metric_correlations(
        self,
        algorithm: Optional[str] = None,
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise Pearson correlations between metrics.

        Parameters
        ----------
        algorithm : str, optional
            Restrict to one algorithm.

        Returns
        -------
        dict
            (metric_a, metric_b) -> correlation coefficient.
        """
        metric_names = sorted(self._metrics.keys())
        if len(metric_names) < 2:
            return {}

        metric_vals: Dict[str, List[float]] = {m: [] for m in metric_names}
        runs = self.get_runs(algorithm=algorithm, status=RunStatus.COMPLETED)

        for r in runs:
            has_all = all(
                m in r.metric_scores and not math.isnan(r.metric_scores[m])
                for m in metric_names
            )
            if has_all:
                for m in metric_names:
                    metric_vals[m].append(r.metric_scores[m])

        correlations: Dict[Tuple[str, str], float] = {}
        for i, m_a in enumerate(metric_names):
            for m_b in metric_names[i + 1:]:
                arr_a = np.array(metric_vals[m_a])
                arr_b = np.array(metric_vals[m_b])
                if len(arr_a) < 2:
                    correlations[(m_a, m_b)] = float("nan")
                    continue
                std_a = np.std(arr_a)
                std_b = np.std(arr_b)
                if std_a < 1e-15 or std_b < 1e-15:
                    correlations[(m_a, m_b)] = 0.0
                    continue
                corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                correlations[(m_a, m_b)] = corr

        return correlations

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        algorithm_name: str,
        param_name: str,
        param_values: List[Any],
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyse how a single parameter affects metric scores.

        Runs the algorithm at each value of *param_name*, aggregates
        metrics, and returns the relationship.

        Parameters
        ----------
        algorithm_name : str
            Algorithm to analyse.
        param_name : str
            Parameter to vary.
        param_values : list
            Values to test.
        task_name : str, optional
            Task (defaults to first registered).

        Returns
        -------
        dict
            {param_values, metric_means, metric_stds}
        """
        entry = self._algorithms.get(algorithm_name)
        if entry is None:
            raise KeyError(f"Algorithm not registered: {algorithm_name!r}")

        t_name = task_name or next(iter(self._tasks), None)
        if t_name is None:
            raise ValueError("No tasks registered")

        results: Dict[str, Any] = {
            "param_name": param_name,
            "param_values": param_values,
            "metric_means": defaultdict(list),
            "metric_stds": defaultdict(list),
        }

        for val in param_values:
            cfg = copy.deepcopy(entry.config)
            if hasattr(cfg, param_name):
                setattr(cfg, param_name, val)
            else:
                cfg.params[param_name] = val

            val_runs: List[ArenaRun] = []
            for _ in range(self.config.num_runs):
                run = self.run_single(algorithm_name, t_name, config=cfg)
                val_runs.append(run)

            successful = [r for r in val_runs if r.is_successful()]
            for metric_name in self._metrics:
                vals = [
                    r.metric_scores.get(metric_name, float("nan"))
                    for r in successful
                ]
                clean = [v for v in vals if not math.isnan(v)]
                if clean:
                    results["metric_means"][metric_name].append(float(np.mean(clean)))
                    results["metric_stds"][metric_name].append(
                        float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
                    )
                else:
                    results["metric_means"][metric_name].append(float("nan"))
                    results["metric_stds"][metric_name].append(0.0)

        results["metric_means"] = dict(results["metric_means"])
        results["metric_stds"] = dict(results["metric_stds"])
        return results

    # ------------------------------------------------------------------
    # Task-level analysis
    # ------------------------------------------------------------------

    def per_task_analysis(
        self,
        metric: str,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Break down algorithm performance by task.

        Parameters
        ----------
        metric : str
            Metric to analyse.

        Returns
        -------
        dict
            task -> algorithm -> {mean, std, count}.
        """
        analysis: Dict[str, Dict[str, Dict[str, float]]] = {}

        for task_name in self._tasks:
            task_runs = [
                r for r in self._runs
                if r.task_name == task_name and r.is_successful()
                and metric in r.metric_scores
                and not math.isnan(r.metric_scores[metric])
            ]
            algo_scores: Dict[str, List[float]] = defaultdict(list)
            for r in task_runs:
                algo_scores[r.algorithm_name].append(r.metric_scores[metric])

            task_analysis: Dict[str, Dict[str, float]] = {}
            for algo, scores in algo_scores.items():
                arr = np.array(scores)
                task_analysis[algo] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "count": float(len(arr)),
                }
            analysis[task_name] = task_analysis

        return analysis

    # ------------------------------------------------------------------
    # Consistency analysis
    # ------------------------------------------------------------------

    def consistency_analysis(
        self,
        metric: str,
    ) -> Dict[str, Dict[str, float]]:
        """Measure how consistent each algorithm's ranking is across tasks.

        For each algorithm, computes the coefficient of variation (CV) and
        rank stability (standard deviation of per-task ranks).

        Parameters
        ----------
        metric : str
            Metric to analyse.

        Returns
        -------
        dict
            algorithm -> {cv, rank_std, mean_rank, mean_score}.
        """
        per_task = self.per_task_analysis(metric)
        me = self._metrics.get(metric)
        higher_is_better = me.higher_is_better if me else True

        all_algos = set()
        for task_data in per_task.values():
            all_algos.update(task_data.keys())

        algo_ranks: Dict[str, List[float]] = {a: [] for a in all_algos}
        algo_scores: Dict[str, List[float]] = {a: [] for a in all_algos}

        for task_name, task_data in per_task.items():
            task_algos = sorted(task_data.keys())
            task_means = [(a, task_data[a]["mean"]) for a in task_algos]
            task_means.sort(key=lambda x: x[1], reverse=higher_is_better)

            for rank_idx, (algo, mean_val) in enumerate(task_means):
                algo_ranks[algo].append(float(rank_idx + 1))
                algo_scores[algo].append(mean_val)

        result: Dict[str, Dict[str, float]] = {}
        for algo in sorted(all_algos):
            ranks = np.array(algo_ranks[algo])
            scores = np.array(algo_scores[algo])
            mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
            std_score = float(np.std(scores)) if len(scores) > 0 else 0.0
            cv = std_score / abs(mean_score) if abs(mean_score) > 1e-12 else 0.0
            rank_std = float(np.std(ranks)) if len(ranks) > 0 else 0.0
            mean_rank = float(np.mean(ranks)) if len(ranks) > 0 else 0.0

            result[algo] = {
                "cv": cv,
                "rank_std": rank_std,
                "mean_rank": mean_rank,
                "mean_score": mean_score,
                "num_tasks": float(len(ranks)),
            }

        return result

    # ------------------------------------------------------------------
    # Diversity-quality trade-off analysis
    # ------------------------------------------------------------------

    def tradeoff_analysis(
        self,
        diversity_metric: str,
        quality_metric: str,
    ) -> Dict[str, Dict[str, float]]:
        """Analyse the diversity-quality trade-off for each algorithm.

        Computes the trade-off ratio (diversity_mean / quality_mean),
        harmonic mean, and geometric mean of the two metrics.

        Parameters
        ----------
        diversity_metric, quality_metric : str
            Metric names.

        Returns
        -------
        dict
            algorithm -> {div_mean, qual_mean, ratio, harmonic, geometric}.
        """
        algo_div: Dict[str, List[float]] = defaultdict(list)
        algo_qual: Dict[str, List[float]] = defaultdict(list)

        for r in self._runs:
            if not r.is_successful():
                continue
            if diversity_metric in r.metric_scores and quality_metric in r.metric_scores:
                dv = r.metric_scores[diversity_metric]
                qv = r.metric_scores[quality_metric]
                if not (math.isnan(dv) or math.isnan(qv)):
                    algo_div[r.algorithm_name].append(dv)
                    algo_qual[r.algorithm_name].append(qv)

        result: Dict[str, Dict[str, float]] = {}
        for algo in sorted(algo_div.keys()):
            d_mean = float(np.mean(algo_div[algo]))
            q_mean = float(np.mean(algo_qual[algo]))

            ratio = _safe_divide(d_mean, q_mean)

            if d_mean > 0 and q_mean > 0:
                harmonic = 2 * d_mean * q_mean / (d_mean + q_mean)
                geometric = math.sqrt(d_mean * q_mean)
            else:
                harmonic = 0.0
                geometric = 0.0

            result[algo] = {
                "diversity_mean": d_mean,
                "quality_mean": q_mean,
                "ratio": ratio,
                "harmonic_mean": harmonic,
                "geometric_mean": geometric,
                "diversity_std": float(np.std(algo_div[algo])),
                "quality_std": float(np.std(algo_qual[algo])),
                "n_runs": float(len(algo_div[algo])),
            }

        return result

    # ------------------------------------------------------------------
    # Efficiency analysis
    # ------------------------------------------------------------------

    def efficiency_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyse computational efficiency per algorithm.

        Returns
        -------
        dict
            algorithm -> {mean_duration, std_duration, total_duration,
            runs_per_second, ...}.
        """
        algo_durations: Dict[str, List[float]] = defaultdict(list)

        for r in self._runs:
            if r.is_successful():
                algo_durations[r.algorithm_name].append(r.duration_seconds)

        result: Dict[str, Dict[str, float]] = {}
        for algo in sorted(algo_durations.keys()):
            durs = np.array(algo_durations[algo])
            total = float(np.sum(durs))
            mean_dur = float(np.mean(durs))
            result[algo] = {
                "mean_duration": mean_dur,
                "std_duration": float(np.std(durs, ddof=1)) if len(durs) > 1 else 0.0,
                "min_duration": float(np.min(durs)),
                "max_duration": float(np.max(durs)),
                "median_duration": float(np.median(durs)),
                "total_duration": total,
                "num_runs": float(len(durs)),
                "runs_per_second": _safe_divide(float(len(durs)), total),
            }

        return result

    # ------------------------------------------------------------------
    # Score normalisation helpers
    # ------------------------------------------------------------------

    def normalise_scores(
        self,
        metric: str,
        method: str = "min_max",
    ) -> Dict[str, np.ndarray]:
        """Normalise metric scores across algorithms.

        Parameters
        ----------
        metric : str
            Metric to normalise.
        method : str
            "min_max" or "z_score".

        Returns
        -------
        dict
            algorithm -> normalised score array.
        """
        raw = self.scores_for_metric(metric)
        all_vals = np.concatenate(list(raw.values())) if raw else np.array([])

        if len(all_vals) == 0:
            return {}

        if method == "z_score":
            global_mean = float(np.mean(all_vals))
            global_std = float(np.std(all_vals))
            if global_std < 1e-15:
                return {k: np.zeros_like(v) for k, v in raw.items()}
            return {k: (v - global_mean) / global_std for k, v in raw.items()}
        else:
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))
            rng = vmax - vmin
            if rng < 1e-15:
                return {k: np.ones_like(v) * 0.5 for k, v in raw.items()}
            return {k: (v - vmin) / rng for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Multi-objective dominance counting
    # ------------------------------------------------------------------

    def dominance_count(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Count how many algorithms each algorithm dominates across multiple
        metrics simultaneously.

        Parameters
        ----------
        metrics : list of str, optional
            Metrics to consider (defaults to all).

        Returns
        -------
        dict
            algorithm -> {dominates, dominated_by, non_dominated_count}.
        """
        metric_names = metrics or sorted(self._metrics.keys())
        if not metric_names:
            return {}

        result = self._result or self._aggregate_results(self._runs)
        algo_names = result.algorithm_names()

        algo_means: Dict[str, Dict[str, float]] = {}
        for algo in algo_names:
            algo_stats = result.summary_stats.get(algo, {})
            means: Dict[str, float] = {}
            for m in metric_names:
                m_stats = algo_stats.get(m, {})
                means[m] = m_stats.get("mean", float("nan"))
            algo_means[algo] = means

        counts: Dict[str, Dict[str, int]] = {
            a: {"dominates": 0, "dominated_by": 0, "non_dominated_count": 0}
            for a in algo_names
        }

        for i, a in enumerate(algo_names):
            for b in algo_names[i + 1:]:
                a_dominates_b = True
                b_dominates_a = True
                a_strictly_better_on_at_least_one = False
                b_strictly_better_on_at_least_one = False

                for m in metric_names:
                    va = algo_means[a].get(m, float("nan"))
                    vb = algo_means[b].get(m, float("nan"))
                    if math.isnan(va) or math.isnan(vb):
                        a_dominates_b = False
                        b_dominates_a = False
                        break

                    me = self._metrics.get(m)
                    hib = me.higher_is_better if me else True

                    if hib:
                        if va < vb:
                            a_dominates_b = False
                        if vb < va:
                            b_dominates_a = False
                        if va > vb:
                            a_strictly_better_on_at_least_one = True
                        if vb > va:
                            b_strictly_better_on_at_least_one = True
                    else:
                        if va > vb:
                            a_dominates_b = False
                        if vb > va:
                            b_dominates_a = False
                        if va < vb:
                            a_strictly_better_on_at_least_one = True
                        if vb < va:
                            b_strictly_better_on_at_least_one = True

                a_dominates_b = a_dominates_b and a_strictly_better_on_at_least_one
                b_dominates_a = b_dominates_a and b_strictly_better_on_at_least_one

                if a_dominates_b:
                    counts[a]["dominates"] += 1
                    counts[b]["dominated_by"] += 1
                elif b_dominates_a:
                    counts[b]["dominates"] += 1
                    counts[a]["dominated_by"] += 1

        for a in algo_names:
            counts[a]["non_dominated_count"] = (
                len(algo_names) - 1 - counts[a]["dominates"] - counts[a]["dominated_by"]
            )

        return counts

    # ------------------------------------------------------------------
    # Reproducibility & seed analysis
    # ------------------------------------------------------------------

    def seed_stability_analysis(
        self,
        algorithm_name: str,
        metric: str,
        seeds: Optional[List[int]] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test how stable results are across different random seeds.

        Parameters
        ----------
        algorithm_name : str
            Algorithm to test.
        metric : str
            Metric to measure.
        seeds : list of int, optional
            Seeds to test (defaults to 5 evenly spaced seeds).
        task_name : str, optional
            Task (defaults to first registered).

        Returns
        -------
        dict
            {seeds, scores_per_seed, overall_mean, overall_std, cv}.
        """
        if seeds is None:
            seeds = [42 + i * 100 for i in range(5)]

        entry = self._algorithms.get(algorithm_name)
        if entry is None:
            raise KeyError(f"Algorithm not registered: {algorithm_name!r}")

        t_name = task_name or next(iter(self._tasks), None)
        if t_name is None:
            raise ValueError("No tasks registered")

        seed_scores: Dict[int, List[float]] = {}

        for seed in seeds:
            cfg = copy.deepcopy(entry.config)
            cfg.seed = seed
            vals: List[float] = []
            for _ in range(self.config.num_runs):
                run = self.run_single(algorithm_name, t_name, config=cfg)
                if run.is_successful() and metric in run.metric_scores:
                    v = run.metric_scores[metric]
                    if not math.isnan(v):
                        vals.append(v)
            seed_scores[seed] = vals

        all_scores = _flatten(list(seed_scores.values()))
        if all_scores:
            arr = np.array(all_scores)
            overall_mean = float(np.mean(arr))
            overall_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            cv = overall_std / abs(overall_mean) if abs(overall_mean) > 1e-12 else 0.0
        else:
            overall_mean = float("nan")
            overall_std = 0.0
            cv = 0.0

        return {
            "algorithm_name": algorithm_name,
            "metric": metric,
            "seeds": seeds,
            "scores_per_seed": {str(k): v for k, v in seed_scores.items()},
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "cv": cv,
        }

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary of the current arena state.

        Returns
        -------
        dict
            Overview information.
        """
        return {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "num_algorithms": len(self.list_algorithms()),
            "num_tasks": len(self._tasks),
            "num_metrics": len(self._metrics),
            "num_runs": len(self._runs),
            "num_completed": self.num_completed,
            "num_failed": self.num_failed,
            "has_result": self._result is not None,
            "has_checkpoint": self.has_checkpoint(),
            "cache_stats": self._cache.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationArena(name={self.config.experiment_name!r}, "
            f"algos={len(self.list_algorithms())}, "
            f"tasks={len(self._tasks)}, "
            f"metrics={len(self._metrics)}, "
            f"runs={len(self._runs)})"
        )

    # ------------------------------------------------------------------
    # Serialisation of the entire arena
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Persist the full arena state to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        state = {
            "version": _ARENA_VERSION,
            "config": self.config.to_dict(),
            "runs": [r.to_dict() for r in self._runs],
            "run_counter": self._run_counter,
            "algorithm_entries": {
                n: e.to_dict() for n, e in self._algorithms.items()
            },
            "tasks": {n: t.to_dict() for n, t in self._tasks.items()},
            "metrics": {
                n: {
                    "name": e.name,
                    "metric_type": e.metric_type.name,
                    "higher_is_better": e.higher_is_better,
                    "description": e.description,
                }
                for n, e in self._metrics.items()
            },
            "saved_at": _iso_now(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("Arena state saved to %s", path)

    # ------------------------------------------------------------------
    # Statistical utilities
    # ------------------------------------------------------------------

    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d effect size.

        Parameters
        ----------
        a, b : ndarray
            Score arrays.

        Returns
        -------
        float
        """
        if len(a) < 2 or len(b) < 2:
            return 0.0
        na, nb = len(a), len(b)
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
        if pooled_std < 1e-15:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_std)

    def _wilcoxon_rank_sum(
        self, a: np.ndarray, b: np.ndarray
    ) -> Tuple[float, float]:
        """Approximate Wilcoxon rank-sum (Mann-Whitney U) statistic and p-value.

        Uses the normal approximation for large samples.

        Parameters
        ----------
        a, b : ndarray
            Score arrays.

        Returns
        -------
        tuple
            (U_statistic, p_value_approx).
        """
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return (0.0, 1.0)

        combined = np.concatenate([a, b])
        ranks = np.empty_like(combined, dtype=float)
        order = np.argsort(combined)
        ranks[order] = np.arange(1, len(combined) + 1, dtype=float)

        i = 0
        while i < len(combined):
            j = i + 1
            while j < len(combined) and combined[order[j]] == combined[order[i]]:
                j += 1
            if j > i + 1:
                avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=float))
                for k in range(i, j):
                    ranks[order[k]] = avg_rank
            i = j

        r_a = float(np.sum(ranks[:na]))
        u_a = r_a - na * (na + 1) / 2.0

        mu = na * nb / 2.0
        sigma = math.sqrt(na * nb * (na + nb + 1) / 12.0)

        if sigma < 1e-15:
            return (u_a, 1.0)

        z = (u_a - mu) / sigma
        p = 2.0 * (1.0 - self._standard_normal_cdf(abs(z)))
        return (u_a, p)

    @staticmethod
    def _standard_normal_cdf(z: float) -> float:
        """Approximate standard normal CDF using the error function."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # ------------------------------------------------------------------
    # Elo rating system
    # ------------------------------------------------------------------

    def compute_elo_ratings(
        self,
        metric: str,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ) -> Dict[str, float]:
        """Compute Elo ratings from pairwise comparisons.

        Each pair of algorithms is compared on the given metric.  For each
        comparison, the algorithm with the higher mean score is the "winner"
        and ratings are updated accordingly.

        Parameters
        ----------
        metric : str
            Metric to use for comparisons.
        k_factor : float
            Elo K-factor controlling update magnitude.
        initial_rating : float
            Starting rating for all algorithms.

        Returns
        -------
        dict
            algorithm -> Elo rating.
        """
        algo_scores = self.scores_for_metric(metric)
        algo_names = sorted(algo_scores.keys())

        me = self._metrics.get(metric)
        higher_is_better = me.higher_is_better if me else True

        ratings: Dict[str, float] = {a: initial_rating for a in algo_names}

        matchups: List[Tuple[str, str]] = []
        for i, a in enumerate(algo_names):
            for b in algo_names[i + 1:]:
                matchups.append((a, b))

        rng = np.random.default_rng(self.config.seed + 10)
        rng.shuffle(matchups)  # type: ignore[arg-type]

        for n_round in range(3):
            for a, b in matchups:
                sa = algo_scores[a]
                sb = algo_scores[b]
                mean_a = float(np.mean(sa))
                mean_b = float(np.mean(sb))

                expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
                expected_b = 1.0 - expected_a

                if abs(mean_a - mean_b) < 1e-12:
                    actual_a = 0.5
                elif (mean_a > mean_b) == higher_is_better:
                    actual_a = 1.0
                else:
                    actual_a = 0.0
                actual_b = 1.0 - actual_a

                ratings[a] += k_factor * (actual_a - expected_a)
                ratings[b] += k_factor * (actual_b - expected_b)

        return ratings

    # ------------------------------------------------------------------
    # Head-to-head comparison matrix
    # ------------------------------------------------------------------

    def head_to_head_matrix(
        self,
        metric: str,
    ) -> Tuple[List[str], np.ndarray]:
        """Build a head-to-head win-rate matrix.

        Entry (i, j) is the fraction of pairwise comparisons where
        algorithm i beat algorithm j.

        Parameters
        ----------
        metric : str
            Metric for comparison.

        Returns
        -------
        tuple
            (list of algorithm names, 2-D ndarray of win rates).
        """
        algo_scores = self.scores_for_metric(metric)
        names = sorted(algo_scores.keys())
        n = len(names)

        me = self._metrics.get(metric)
        higher_is_better = me.higher_is_better if me else True

        matrix = np.full((n, n), 0.5)

        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                sa = algo_scores[a]
                sb = algo_scores[b]
                wins = 0
                total = 0
                for va in sa:
                    for vb in sb:
                        total += 1
                        if higher_is_better:
                            if va > vb:
                                wins += 1
                            elif va == vb:
                                wins += 0.5
                        else:
                            if va < vb:
                                wins += 1
                            elif va == vb:
                                wins += 0.5
                matrix[i, j] = wins / total if total > 0 else 0.5

        return names, matrix

    # ------------------------------------------------------------------
    # Config diff utility
    # ------------------------------------------------------------------

    def config_diff(
        self,
        config_a: DecodingConfig,
        config_b: DecodingConfig,
    ) -> Dict[str, Tuple[Any, Any]]:
        """Compute the difference between two decoding configs.

        Parameters
        ----------
        config_a, config_b : DecodingConfig
            Configurations to compare.

        Returns
        -------
        dict
            param_name -> (value_a, value_b) for differing parameters.
        """
        dict_a = config_a.to_dict()
        dict_b = config_b.to_dict()

        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        diff: Dict[str, Tuple[Any, Any]] = {}

        for key in sorted(all_keys):
            va = dict_a.get(key)
            vb = dict_b.get(key)
            if va != vb:
                diff[key] = (va, vb)

        return diff

    # ------------------------------------------------------------------
    # Metric contribution analysis
    # ------------------------------------------------------------------

    def metric_contribution(
        self,
        target_ranking_metric: str,
        candidate_metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Estimate how much each candidate metric contributes to the ranking
        produced by the target metric.

        Uses Kendall tau correlation between the ranking induced by each
        candidate and the target ranking.

        Parameters
        ----------
        target_ranking_metric : str
            The reference metric for ranking.
        candidate_metrics : list of str, optional
            Metrics to evaluate (defaults to all except target).

        Returns
        -------
        dict
            candidate_metric -> Kendall tau correlation with target ranking.
        """
        if candidate_metrics is None:
            candidate_metrics = [
                m for m in self._metrics if m != target_ranking_metric
            ]

        target_ranking = self.rank_algorithms(target_ranking_metric, method="mean")
        if not target_ranking:
            return {}

        target_order = [e.algorithm_name for e in target_ranking]

        contributions: Dict[str, float] = {}

        for cand in candidate_metrics:
            cand_ranking = self.rank_algorithms(cand, method="mean")
            if not cand_ranking:
                contributions[cand] = 0.0
                continue

            cand_order = [e.algorithm_name for e in cand_ranking]

            common = [a for a in target_order if a in cand_order]
            if len(common) < 2:
                contributions[cand] = 0.0
                continue

            target_pos = {a: i for i, a in enumerate(target_order) if a in common}
            cand_pos = {a: i for i, a in enumerate(cand_order) if a in common}

            concordant = 0
            discordant = 0
            for i, a in enumerate(common):
                for b in common[i + 1:]:
                    t_diff = target_pos[a] - target_pos[b]
                    c_diff = cand_pos[a] - cand_pos[b]
                    if t_diff * c_diff > 0:
                        concordant += 1
                    elif t_diff * c_diff < 0:
                        discordant += 1

            n_pairs = len(common) * (len(common) - 1) // 2
            if n_pairs > 0:
                tau = (concordant - discordant) / n_pairs
            else:
                tau = 0.0

            contributions[cand] = tau

        return contributions

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def run_batch(
        self,
        algorithm_task_pairs: List[Tuple[str, str]],
        config: Optional[DecodingConfig] = None,
    ) -> List[ArenaRun]:
        """Run a specific batch of (algorithm, task) pairs.

        Parameters
        ----------
        algorithm_task_pairs : list of (str, str)
            Pairs of (algorithm_name, task_name).
        config : DecodingConfig, optional
            Shared config override.

        Returns
        -------
        list of ArenaRun
        """
        runs: List[ArenaRun] = []
        tracker = _ProgressTracker(
            len(algorithm_task_pairs) * self.config.num_runs,
            self.config.verbose,
        )

        for algo_name, task_name in algorithm_task_pairs:
            for _ in range(self.config.num_runs):
                run = self._execute_with_retries(algo_name, task_name, config or DecodingConfig())
                runs.append(run)
                tracker.update(run.status)
                tracker.log(run)

        return runs

    # ------------------------------------------------------------------
    # Quick comparison
    # ------------------------------------------------------------------

    def quick_compare(
        self,
        algo_a: str,
        algo_b: str,
    ) -> Dict[str, ComparisonResult]:
        """Compare two algorithms across all registered metrics.

        Parameters
        ----------
        algo_a, algo_b : str
            Algorithm names.

        Returns
        -------
        dict
            metric -> ComparisonResult.
        """
        results: Dict[str, ComparisonResult] = {}
        for metric_name in self._metrics:
            try:
                comp = self.compare_algorithms(algo_a, algo_b, metric_name)
                results[metric_name] = comp
            except Exception as exc:
                logger.debug("Quick compare %s vs %s on %s: %s", algo_a, algo_b, metric_name, exc)
        return results

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outlier_runs(
        self,
        metric: str,
        z_threshold: float = 3.0,
    ) -> List[ArenaRun]:
        """Detect outlier runs based on z-score of metric values.

        Parameters
        ----------
        metric : str
            Metric to check.
        z_threshold : float
            Absolute z-score above which a run is considered an outlier.

        Returns
        -------
        list of ArenaRun
            Outlier runs.
        """
        all_scores: List[float] = []
        runs_with_scores: List[Tuple[ArenaRun, float]] = []

        for r in self._runs:
            if r.is_successful() and metric in r.metric_scores:
                v = r.metric_scores[metric]
                if not math.isnan(v):
                    all_scores.append(v)
                    runs_with_scores.append((r, v))

        if len(all_scores) < 3:
            return []

        arr = np.array(all_scores)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        if std < 1e-15:
            return []

        outliers: List[ArenaRun] = []
        for run, score in runs_with_scores:
            z = abs(score - mean) / std
            if z > z_threshold:
                run.metadata["outlier_z_score"] = z
                outliers.append(run)

        return outliers

    # ------------------------------------------------------------------
    # Sample size estimation
    # ------------------------------------------------------------------

    def estimate_required_runs(
        self,
        metric: str,
        desired_ci_width: float,
        alpha: float = 0.05,
    ) -> Dict[str, int]:
        """Estimate the number of runs needed to achieve a desired CI width.

        Uses the current variance estimate and the normal approximation.

        Parameters
        ----------
        metric : str
            Metric to estimate for.
        desired_ci_width : float
            Target width of the confidence interval.
        alpha : float
            Significance level.

        Returns
        -------
        dict
            algorithm -> estimated required number of runs.
        """
        z = self._z_score_for_alpha(alpha)
        half_width = desired_ci_width / 2.0

        algo_scores = self.scores_for_metric(metric)
        required: Dict[str, int] = {}

        for algo, scores in algo_scores.items():
            if len(scores) < 2:
                required[algo] = max(30, self.config.num_runs)
                continue
            std = float(np.std(scores, ddof=1))
            if half_width < 1e-15:
                required[algo] = len(scores)
            else:
                n = math.ceil((z * std / half_width) ** 2)
                required[algo] = max(n, 2)

        return required

    @staticmethod
    def _z_score_for_alpha(alpha: float) -> float:
        """Approximate z-score for a given significance level using Beasley-Springer-Moro."""
        p = 1.0 - alpha / 2.0
        if p <= 0 or p >= 1:
            return 0.0
        t = math.sqrt(-2.0 * math.log(1.0 - p)) if p > 0.5 else math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        if p < 0.5:
            z = -z
        return z

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    def power_analysis(
        self,
        algo_a: str,
        algo_b: str,
        metric: str,
        effect_sizes: Optional[List[float]] = None,
        sample_sizes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Estimate statistical power for detecting a difference.

        Uses bootstrap simulation to estimate the probability of
        detecting a given effect size at each sample size.

        Parameters
        ----------
        algo_a, algo_b : str
            Algorithms to compare.
        metric : str
            Metric to analyse.
        effect_sizes : list of float, optional
            Effect sizes (Cohen's d) to test.
        sample_sizes : list of int, optional
            Sample sizes to test.

        Returns
        -------
        dict
            {effect_sizes, sample_sizes, power_matrix}.
        """
        if effect_sizes is None:
            effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.5]
        if sample_sizes is None:
            sample_sizes = [5, 10, 20, 30, 50, 100]

        scores_a = self.scores_for_metric(metric).get(algo_a, np.array([]))
        scores_b = self.scores_for_metric(metric).get(algo_b, np.array([]))

        if len(scores_a) == 0 or len(scores_b) == 0:
            return {
                "effect_sizes": effect_sizes,
                "sample_sizes": sample_sizes,
                "power_matrix": [[0.0] * len(sample_sizes)] * len(effect_sizes),
            }

        pooled_std = math.sqrt(
            (float(np.var(scores_a, ddof=1)) + float(np.var(scores_b, ddof=1))) / 2
        )
        base_mean = float(np.mean(scores_b))

        rng = np.random.default_rng(self.config.seed + 20)
        n_sim = 500
        alpha = self.config.alpha

        power_matrix: List[List[float]] = []

        for es in effect_sizes:
            row: List[float] = []
            for n in sample_sizes:
                significant_count = 0
                for _ in range(n_sim):
                    sim_a = rng.normal(
                        base_mean + es * pooled_std, pooled_std, size=n
                    )
                    sim_b = rng.normal(base_mean, pooled_std, size=n)

                    _, p = self._wilcoxon_rank_sum(sim_a, sim_b)
                    if p < alpha:
                        significant_count += 1

                row.append(significant_count / n_sim)
            power_matrix.append(row)

        return {
            "effect_sizes": effect_sizes,
            "sample_sizes": sample_sizes,
            "power_matrix": power_matrix,
        }

    # ------------------------------------------------------------------
    # Warm-start from prior results
    # ------------------------------------------------------------------

    def import_runs(self, runs: List[ArenaRun]) -> int:
        """Import externally produced runs into the arena.

        Parameters
        ----------
        runs : list of ArenaRun
            Runs to import.

        Returns
        -------
        int
            Number of runs imported.
        """
        imported = 0
        existing_ids = {r.run_id for r in self._runs}
        for r in runs:
            if r.run_id not in existing_ids:
                self._runs.append(r)
                existing_ids.add(r.run_id)
                imported += 1
        self._run_counter += imported
        logger.info("Imported %d runs (skipped %d duplicates)", imported, len(runs) - imported)
        return imported

    def merge_results(self, other_result: ArenaResult) -> int:
        """Merge runs from another ArenaResult into this arena.

        Parameters
        ----------
        other_result : ArenaResult
            Result to merge.

        Returns
        -------
        int
            Number of new runs merged.
        """
        return self.import_runs(other_result.runs)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_runs(
        self,
        predicate: Callable[[ArenaRun], bool],
    ) -> List[ArenaRun]:
        """Return runs matching a custom predicate.

        Parameters
        ----------
        predicate : callable
            Function returning True for runs to keep.

        Returns
        -------
        list of ArenaRun
        """
        return [r for r in self._runs if predicate(r)]

    def remove_runs(
        self,
        predicate: Callable[[ArenaRun], bool],
    ) -> int:
        """Remove runs matching a predicate.

        Parameters
        ----------
        predicate : callable
            Function returning True for runs to remove.

        Returns
        -------
        int
            Number of runs removed.
        """
        before = len(self._runs)
        self._runs = [r for r in self._runs if not predicate(r)]
        removed = before - len(self._runs)
        if removed > 0:
            logger.info("Removed %d runs", removed)
            self._result = None
        return removed

    # ------------------------------------------------------------------
    # Convenience: run and compare two algorithms
    # ------------------------------------------------------------------

    def run_and_compare(
        self,
        algo_a: str,
        algo_b: str,
        task_name: str,
        num_runs: Optional[int] = None,
    ) -> Dict[str, ComparisonResult]:
        """Run two algorithms on a task and return comparisons.

        Parameters
        ----------
        algo_a, algo_b : str
            Algorithm names.
        task_name : str
            Task to evaluate on.
        num_runs : int, optional
            Number of runs per algorithm (defaults to config.num_runs).

        Returns
        -------
        dict
            metric -> ComparisonResult.
        """
        n = num_runs or self.config.num_runs

        for algo in (algo_a, algo_b):
            entry = self._algorithms.get(algo)
            if entry is None:
                raise KeyError(f"Algorithm not registered: {algo!r}")
            for _ in range(n):
                cfg = copy.deepcopy(entry.config)
                if self.config.seed is not None:
                    cfg.seed = self.config.seed + self._run_counter
                self._execute_with_retries(algo, task_name, cfg)

        return self.quick_compare(algo_a, algo_b)

    # ------------------------------------------------------------------
    # Diagnostic: self-consistency check
    # ------------------------------------------------------------------

    def self_check(self) -> Dict[str, Any]:
        """Run diagnostic checks on the arena configuration.

        Returns
        -------
        dict
            Check results: {valid, warnings, errors}.
        """
        errors: List[str] = []
        warnings_list: List[str] = []

        config_errors = self.config.validate()
        errors.extend(config_errors)

        if not self._algorithms:
            warnings_list.append("No algorithms registered")
        if not self._tasks:
            warnings_list.append("No tasks registered")
        if not self._metrics:
            warnings_list.append("No metrics registered")

        for name, entry in self._algorithms.items():
            if entry.algorithm is None:
                errors.append(f"Algorithm {name!r} has no algorithm instance")
            cfg_errors = entry.config.validate()
            if cfg_errors:
                warnings_list.append(
                    f"Algorithm {name!r} config warnings: {cfg_errors}"
                )

        for name, task in self._tasks.items():
            if not task.prompts:
                warnings_list.append(f"Task {name!r} has no prompts")

        for name, metric in self._metrics.items():
            if metric.fn is None:
                errors.append(f"Metric {name!r} has no scoring function")

        missing_sources = [
            n for n in self._algorithms if n not in self._logit_sources
        ]
        if missing_sources:
            warnings_list.append(
                f"Algorithms without logit sources (will use dummy): "
                f"{missing_sources}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list,
            "num_algorithms": len(self._algorithms),
            "num_tasks": len(self._tasks),
            "num_metrics": len(self._metrics),
        }
