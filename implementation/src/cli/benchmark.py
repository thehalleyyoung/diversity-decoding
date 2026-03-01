"""
Benchmark runner for the Diversity Decoding Arena.

Provides systematic benchmarking of diversity-oriented decoding algorithms
across tasks, metrics, and configurations with detailed reporting.
"""

from __future__ import annotations

import csv
import datetime
import enum
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkSuite(enum.Enum):
    """Pre-defined benchmark suite configurations."""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"
    STRESS_TEST = "stress_test"
    ABLATION = "ablation"


class BenchmarkStatus(enum.Enum):
    """Status of a benchmark run or individual task."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Default catalogues
# ---------------------------------------------------------------------------

DEFAULT_ALGORITHMS = [
    "greedy",
    "beam_search",
    "top_k",
    "top_p",
    "temperature_scaled",
    "diverse_beam_search",
    "stochastic_beam",
    "mcts_diversity",
    "mbr_decoding",
    "contrastive_search",
]

DEFAULT_TASKS = [
    "story_generation",
    "open_qa",
    "paraphrase",
    "code_generation",
    "dialogue",
    "summarization",
    "creative_writing",
    "brainstorming",
]

DEFAULT_METRICS = [
    "self_bleu",
    "distinct_n",
    "entropy",
    "coverage",
    "coherence",
    "quality",
    "latency",
    "memory_peak",
]


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Full configuration for a benchmark run."""
    suite: BenchmarkSuite = BenchmarkSuite.STANDARD
    algorithms: List[str] = field(default_factory=lambda: list(DEFAULT_ALGORITHMS[:5]))
    tasks: List[str] = field(default_factory=lambda: list(DEFAULT_TASKS[:4]))
    metrics: List[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    num_samples: int = 50
    max_length: int = 128
    num_runs: int = 3
    seed: int = 42
    output_dir: str = "benchmark_output"
    timeout_per_run: float = 300.0
    parallel_workers: int = 1
    save_generations: bool = False
    compare_to_baseline: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BenchmarkSuite):
                d[k] = v.value
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        d = dict(d)
        if "suite" in d and isinstance(d["suite"], str):
            d["suite"] = BenchmarkSuite(d["suite"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkResult:
    """Aggregated result for an entire benchmark run."""
    benchmark_id: str = ""
    suite: BenchmarkSuite = BenchmarkSuite.STANDARD
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    status: BenchmarkStatus = BenchmarkStatus.NOT_STARTED
    algorithm_results: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    metric_results: Dict[str, Any] = field(default_factory=dict)
    rankings: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (BenchmarkSuite, BenchmarkStatus)):
                d[k] = v.value
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        d = dict(d)
        if "suite" in d and isinstance(d["suite"], str):
            d["suite"] = BenchmarkSuite(d["suite"])
        if "status" in d and isinstance(d["status"], str):
            d["status"] = BenchmarkStatus(d["status"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkTask:
    """Represents a single (algorithm, task) evaluation unit."""
    task_id: str = ""
    algorithm: str = ""
    task_domain: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    status: BenchmarkStatus = BenchmarkStatus.NOT_STARTED
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BenchmarkStatus):
                d[k] = v.value
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# ProgressBar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple text-based progress bar for terminal output."""

    def __init__(self, total: int, prefix: str = "Progress", bar_length: int = 40):
        self.total = max(total, 1)
        self.prefix = prefix
        self.bar_length = bar_length
        self._current = 0
        self._start_time = time.time()
        self._last_render = ""

    def update(self, current: int, suffix: str = "") -> None:
        self._current = min(current, self.total)
        rendered = self._render(suffix)
        if rendered != self._last_render:
            sys.stderr.write(f"\r{rendered}")
            sys.stderr.flush()
            self._last_render = rendered

    def finish(self) -> None:
        self.update(self.total, "Done")
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, suffix: str = "") -> str:
        fraction = self._current / self.total
        filled = int(self.bar_length * fraction)
        bar = "█" * filled + "░" * (self.bar_length - filled)
        pct = fraction * 100.0
        elapsed = time.time() - self._start_time
        if self._current > 0:
            eta = elapsed / self._current * (self.total - self._current)
        else:
            eta = 0.0
        eta_str = _format_duration(eta)
        parts = [
            f"{self.prefix}: |{bar}| {pct:5.1f}%",
            f"[{self._current}/{self.total}]",
            f"ETA {eta_str}",
        ]
        if suffix:
            parts.append(suffix)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# SystemProfiler
# ---------------------------------------------------------------------------

class SystemProfiler:
    """Collects system information and rough performance estimates."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": SystemProfiler.get_cpu_count(),
            "memory_mb": SystemProfiler.get_memory_usage(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return info

    @staticmethod
    def get_memory_usage() -> float:
        """Return current process RSS in MB (best-effort)."""
        try:
            if sys.platform == "linux":
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024.0
            elif sys.platform == "darwin":
                import subprocess
                pid = os.getpid()
                out = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(pid)], text=True
                )
                return float(out.strip()) / 1024.0
        except Exception:
            pass
        # Fallback: rough estimate from sys.getsizeof is useless; return -1
        return -1.0

    @staticmethod
    def get_cpu_count() -> int:
        return os.cpu_count() or 1

    @staticmethod
    def benchmark_numpy_speed() -> float:
        """Estimate GFLOPS by timing a dense matrix multiply."""
        n = 512
        a = np.random.randn(n, n).astype(np.float64)
        b = np.random.randn(n, n).astype(np.float64)
        # warm up
        _ = a @ b
        t0 = time.perf_counter()
        for _ in range(3):
            _ = a @ b
        elapsed = (time.perf_counter() - t0) / 3.0
        flops = 2.0 * n * n * n  # approx flops for matmul
        gflops = flops / elapsed / 1e9
        return round(gflops, 2)

    @staticmethod
    def estimate_runtime(
        n_algorithms: int, n_tasks: int, n_samples: int
    ) -> float:
        """Rough estimate of total benchmark wall-clock seconds."""
        # Heuristic: 0.05s per sample per (algo, task) pair
        per_sample = 0.05
        total = n_algorithms * n_tasks * n_samples * per_sample
        return round(total, 1)


# ---------------------------------------------------------------------------
# Suite presets
# ---------------------------------------------------------------------------

class BenchmarkSuitePresets:
    """Factory for pre-defined suite configurations."""

    PRESETS: Dict[BenchmarkSuite, Dict[str, Any]] = {
        BenchmarkSuite.QUICK: {
            "algorithms": ["greedy", "top_k"],
            "tasks": ["story_generation", "open_qa"],
            "num_samples": 10,
            "max_length": 64,
            "num_runs": 1,
        },
        BenchmarkSuite.STANDARD: {
            "algorithms": DEFAULT_ALGORITHMS[:5],
            "tasks": DEFAULT_TASKS[:4],
            "num_samples": 50,
            "max_length": 128,
            "num_runs": 3,
        },
        BenchmarkSuite.COMPREHENSIVE: {
            "algorithms": list(DEFAULT_ALGORITHMS),
            "tasks": list(DEFAULT_TASKS),
            "num_samples": 100,
            "max_length": 256,
            "num_runs": 5,
        },
        BenchmarkSuite.STRESS_TEST: {
            "algorithms": list(DEFAULT_ALGORITHMS),
            "tasks": list(DEFAULT_TASKS),
            "num_samples": 500,
            "max_length": 512,
            "num_runs": 1,
        },
        BenchmarkSuite.ABLATION: {
            "algorithms": list(DEFAULT_ALGORITHMS),
            "tasks": ["story_generation"],
            "num_samples": 50,
            "max_length": 128,
            "num_runs": 5,
        },
    }

    @classmethod
    def get(cls, suite: BenchmarkSuite) -> Dict[str, Any]:
        if suite == BenchmarkSuite.CUSTOM:
            return {}
        return dict(cls.PRESETS.get(suite, {}))

    @classmethod
    def build_config(cls, suite: BenchmarkSuite, **overrides: Any) -> BenchmarkConfig:
        preset = cls.get(suite)
        preset["suite"] = suite
        preset.update(overrides)
        return BenchmarkConfig(**{
            k: v for k, v in preset.items() if k in BenchmarkConfig.__dataclass_fields__
        })


# ---------------------------------------------------------------------------
# Synthetic data helpers (simulate algorithm outputs for benchmarking)
# ---------------------------------------------------------------------------

def _simulate_algorithm_output(
    algorithm: str,
    task: str,
    num_samples: int,
    max_length: int,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Produce synthetic token-id sequences that mimic different algorithms.

    Each algorithm has characteristic diversity / quality trade-off encoded
    via vocabulary range and repetition patterns so that downstream metrics
    produce meaningful (if simulated) differentiation.
    """
    vocab_size = 30000

    # Algorithm-specific knobs
    algo_params: Dict[str, Dict[str, float]] = {
        "greedy":               {"temp": 0.1, "rep_penalty": 0.9, "vocab_frac": 0.05},
        "beam_search":          {"temp": 0.2, "rep_penalty": 0.85, "vocab_frac": 0.08},
        "top_k":                {"temp": 0.7, "rep_penalty": 0.5, "vocab_frac": 0.25},
        "top_p":                {"temp": 0.8, "rep_penalty": 0.4, "vocab_frac": 0.30},
        "temperature_scaled":   {"temp": 1.0, "rep_penalty": 0.3, "vocab_frac": 0.40},
        "diverse_beam_search":  {"temp": 0.6, "rep_penalty": 0.3, "vocab_frac": 0.35},
        "stochastic_beam":      {"temp": 0.75, "rep_penalty": 0.35, "vocab_frac": 0.30},
        "mcts_diversity":       {"temp": 0.65, "rep_penalty": 0.25, "vocab_frac": 0.45},
        "mbr_decoding":         {"temp": 0.5, "rep_penalty": 0.4, "vocab_frac": 0.20},
        "contrastive_search":   {"temp": 0.55, "rep_penalty": 0.3, "vocab_frac": 0.32},
    }

    params = algo_params.get(algorithm, {"temp": 0.5, "rep_penalty": 0.5, "vocab_frac": 0.2})
    eff_vocab = max(100, int(vocab_size * params["vocab_frac"]))

    # Task difficulty multiplier
    task_mult: Dict[str, float] = {
        "story_generation": 1.0,
        "open_qa": 0.8,
        "paraphrase": 0.6,
        "code_generation": 0.9,
        "dialogue": 0.85,
        "summarization": 0.7,
        "creative_writing": 1.1,
        "brainstorming": 1.05,
    }
    mult = task_mult.get(task, 1.0)

    sequences: List[List[int]] = []
    for _ in range(num_samples):
        length = rng.randint(max(1, max_length // 2), max_length + 1)
        logits = rng.randn(eff_vocab) * params["temp"] * mult
        probs = _softmax(logits)
        seq = rng.choice(eff_vocab, size=length, replace=True, p=probs).tolist()
        # apply repetition penalty: occasionally replace repeated tokens
        for j in range(1, len(seq)):
            if seq[j] == seq[j - 1] and rng.rand() > params["rep_penalty"]:
                seq[j] = rng.randint(0, eff_vocab)
        sequences.append(seq)

    return {
        "sequences": sequences,
        "eff_vocab": eff_vocab,
        "params": params,
        "task_mult": mult,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _compute_self_bleu(sequences: List[List[int]], n: int = 4) -> float:
    """Approximate Self-BLEU by computing pairwise n-gram overlap on a sample."""
    if len(sequences) < 2:
        return 0.0
    sample_size = min(len(sequences), 50)
    indices = list(range(len(sequences)))
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    indices = indices[:sample_size]
    sampled = [sequences[i] for i in indices]

    def _ngrams(seq: List[int], order: int) -> Dict[Tuple[int, ...], int]:
        counts: Dict[Tuple[int, ...], int] = {}
        for i in range(len(seq) - order + 1):
            g = tuple(seq[i:i + order])
            counts[g] = counts.get(g, 0) + 1
        return counts

    total_bleu = 0.0
    pairs = 0
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            ref_ng = _ngrams(sampled[i], n)
            hyp_ng = _ngrams(sampled[j], n)
            overlap = 0
            total = 0
            for g, c in hyp_ng.items():
                overlap += min(c, ref_ng.get(g, 0))
                total += c
            if total > 0:
                total_bleu += overlap / total
            pairs += 1
    return total_bleu / max(pairs, 1)


def _compute_distinct_n(sequences: List[List[int]], n: int = 2) -> float:
    """Distinct-n: fraction of unique n-grams over total n-grams."""
    unique: set = set()
    total = 0
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i:i + n])
            unique.add(g)
            total += 1
    return len(unique) / max(total, 1)


def _compute_entropy(sequences: List[List[int]]) -> float:
    """Token-level unigram entropy in bits."""
    counts: Dict[int, int] = {}
    total = 0
    for seq in sequences:
        for tok in seq:
            counts[tok] = counts.get(tok, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _compute_coverage(sequences: List[List[int]], vocab_size: int) -> float:
    """Fraction of the effective vocabulary actually used."""
    unique_tokens: set = set()
    for seq in sequences:
        unique_tokens.update(seq)
    return len(unique_tokens) / max(vocab_size, 1)


def _compute_coherence(sequences: List[List[int]]) -> float:
    """Proxy coherence: mean autocorrelation of token embeddings (synthetic).

    We hash tokens to pseudo-embeddings and measure smoothness.
    """
    if not sequences:
        return 0.0
    scores: List[float] = []
    for seq in sequences:
        if len(seq) < 2:
            scores.append(1.0)
            continue
        arr = np.array(seq, dtype=np.float64)
        arr = (arr - arr.mean()) / (arr.std() + 1e-12)
        autocorr = np.correlate(arr, arr, mode="full")
        mid = len(autocorr) // 2
        if autocorr[mid] > 0:
            score = float(autocorr[mid + 1] / autocorr[mid])
        else:
            score = 0.0
        scores.append(max(0.0, min(1.0, (score + 1.0) / 2.0)))
    return float(np.mean(scores))


def _compute_quality(sequences: List[List[int]]) -> float:
    """Synthetic quality proxy based on length variance and repetition rate."""
    if not sequences:
        return 0.0
    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    len_cv = float(lengths.std() / (lengths.mean() + 1e-12))
    # repetition ratio
    rep_ratios: List[float] = []
    for seq in sequences:
        if len(seq) < 2:
            rep_ratios.append(0.0)
            continue
        reps = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
        rep_ratios.append(reps / (len(seq) - 1))
    mean_rep = float(np.mean(rep_ratios))
    # quality = 1 - normalised badness
    quality = 1.0 - 0.5 * min(len_cv, 1.0) - 0.5 * mean_rep
    return max(0.0, min(1.0, quality))


METRIC_FUNCTIONS: Dict[str, Callable] = {
    "self_bleu": lambda seqs, _v: _compute_self_bleu(seqs),
    "distinct_n": lambda seqs, _v: _compute_distinct_n(seqs),
    "entropy": lambda seqs, _v: _compute_entropy(seqs),
    "coverage": lambda seqs, _v: _compute_coverage(seqs, _v),
    "coherence": lambda seqs, _v: _compute_coherence(seqs),
    "quality": lambda seqs, _v: _compute_quality(seqs),
}

# For self_bleu lower is more diverse; for everything else higher is better
METRIC_HIGHER_BETTER: Dict[str, bool] = {
    "self_bleu": False,
    "distinct_n": True,
    "entropy": True,
    "coverage": True,
    "coherence": True,
    "quality": True,
    "latency": False,
    "memory_peak": False,
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _format_number(value: float, precision: int = 4) -> str:
    if abs(value) < 1e-8:
        return "0"
    if abs(value) >= 1e6:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def _format_table(
    headers: List[str],
    rows: List[List[str]],
    alignment: Optional[List[str]] = None,
) -> str:
    """Render a text table with column alignment (l/r/c)."""
    if not rows:
        return "(no data)"
    n_cols = len(headers)
    if alignment is None:
        alignment = ["l"] * n_cols

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    def _pad(text: str, width: int, align: str) -> str:
        if align == "r":
            return text.rjust(width)
        if align == "c":
            return text.center(width)
        return text.ljust(width)

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(
        _pad(h, col_widths[i], alignment[i]) for i, h in enumerate(headers)
    ) + " |"
    lines = [sep, header_line, sep]
    for row in rows:
        padded = []
        for i in range(n_cols):
            cell = row[i] if i < len(row) else ""
            padded.append(_pad(cell, col_widths[i], alignment[i]))
        lines.append("| " + " | ".join(padded) + " |")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates benchmark execution, metric computation, and reporting."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._rng = np.random.RandomState(config.seed)
        self._profiler = SystemProfiler()
        self._progress: Optional[ProgressBar] = None
        self._log_lines: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark suite described by *self.config*."""
        errors = self.validate_config()
        if errors:
            return BenchmarkResult(
                benchmark_id=self._make_id(),
                suite=self.config.suite,
                status=BenchmarkStatus.FAILED,
                errors=errors,
            )

        benchmark_id = self._make_id()
        system_info = self._profiler.get_system_info()
        start_time = time.time()

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            suite=self.config.suite,
            start_time=start_time,
            status=BenchmarkStatus.RUNNING,
            system_info=system_info,
        )

        self._log(f"Starting benchmark {benchmark_id} "
                   f"[suite={self.config.suite.value}]")
        self._log(f"Algorithms: {self.config.algorithms}")
        self._log(f"Tasks: {self.config.tasks}")
        self._log(f"Samples: {self.config.num_samples}, "
                   f"Runs: {self.config.num_runs}")

        total_units = len(self.config.algorithms) * len(self.config.tasks)
        self._progress = self._create_progress_bar(total_units)
        completed = 0

        all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        all_errors: List[str] = []

        for algo in self.config.algorithms:
            all_results[algo] = {}
            for task in self.config.tasks:
                task_label = f"{algo}/{task}"
                self._log_progress(completed, total_units, task_label)

                if self._check_timeout(start_time, self.config.timeout_per_run * total_units):
                    msg = f"Global timeout reached at {task_label}"
                    self._log(msg)
                    all_errors.append(msg)
                    break

                try:
                    task_result = self._run_single_benchmark(
                        algo, task, self.config
                    )
                    all_results[algo][task] = task_result
                except Exception as exc:
                    tb = traceback.format_exc()
                    all_errors.append(f"{task_label}: {exc}\n{tb}")
                    all_results[algo][task] = {
                        "status": BenchmarkStatus.FAILED.value,
                        "error": str(exc),
                    }

                completed += 1
                if self._progress is not None:
                    self._progress.update(completed, task_label)

        if self._progress is not None:
            self._progress.finish()

        end_time = time.time()
        result = self._aggregate_results(all_results)
        result.benchmark_id = benchmark_id
        result.suite = self.config.suite
        result.start_time = start_time
        result.end_time = end_time
        result.duration = end_time - start_time
        result.errors = all_errors
        result.system_info = system_info
        result.status = (
            BenchmarkStatus.COMPLETED if not all_errors
            else BenchmarkStatus.COMPLETED
        )

        if self.config.compare_to_baseline:
            try:
                baseline = self.load_results(self.config.compare_to_baseline)
                comparison = self._compare_to_baseline(result, baseline)
                result.metric_results["_baseline_comparison"] = comparison
            except Exception as exc:
                result.errors.append(f"Baseline comparison failed: {exc}")

        self._log(f"Benchmark completed in {_format_duration(result.duration)}")
        return result

    def run_suite(self, suite: BenchmarkSuite) -> BenchmarkResult:
        """Run a named preset suite, overriding the current config."""
        self.config = self._get_suite_config(suite)
        return self.run()

    # ------------------------------------------------------------------
    # Suite helpers
    # ------------------------------------------------------------------

    def _get_suite_config(self, suite: BenchmarkSuite) -> BenchmarkConfig:
        cfg = BenchmarkSuitePresets.build_config(suite, seed=self.config.seed,
                                                  output_dir=self.config.output_dir)
        return cfg

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run_single_benchmark(
        self,
        algorithm: str,
        task: str,
        config: BenchmarkConfig,
    ) -> Dict[str, Any]:
        """Run *num_runs* repetitions of (algorithm, task) and aggregate."""
        run_metrics: Dict[str, List[float]] = {m: [] for m in config.metrics}
        run_latencies: List[float] = []
        run_memory: List[float] = []

        for run_idx in range(config.num_runs):
            seed = config.seed + run_idx * 1000 + hash(algorithm) % 10000
            rng = np.random.RandomState(seed & 0x7FFFFFFF)

            mem_before = self._profiler.get_memory_usage()
            t0 = time.perf_counter()

            output = _simulate_algorithm_output(
                algorithm, task, config.num_samples, config.max_length, rng,
            )

            elapsed = time.perf_counter() - t0
            mem_after = self._profiler.get_memory_usage()

            sequences = output["sequences"]
            eff_vocab = output["eff_vocab"]

            for metric_name in config.metrics:
                if metric_name == "latency":
                    run_metrics["latency"].append(elapsed)
                elif metric_name == "memory_peak":
                    delta = max(0.0, mem_after - mem_before) if mem_before > 0 else 0.0
                    run_metrics["memory_peak"].append(delta)
                elif metric_name in METRIC_FUNCTIONS:
                    val = METRIC_FUNCTIONS[metric_name](sequences, eff_vocab)
                    run_metrics[metric_name].append(val)

            run_latencies.append(elapsed)
            if mem_before > 0 and mem_after > 0:
                run_memory.append(mem_after - mem_before)

        # Aggregate across runs
        aggregated: Dict[str, Dict[str, float]] = {}
        for metric_name, values in run_metrics.items():
            if not values:
                continue
            arr = np.array(values)
            aggregated[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }

        return {
            "status": BenchmarkStatus.COMPLETED.value,
            "algorithm": algorithm,
            "task": task,
            "num_runs": config.num_runs,
            "num_samples": config.num_samples,
            "metrics": aggregated,
            "mean_latency": float(np.mean(run_latencies)),
            "total_time": float(np.sum(run_latencies)),
        }

    def _run_algorithm_suite(self, algorithm: str) -> Dict[str, Dict[str, Any]]:
        """Run all tasks for a single algorithm."""
        results: Dict[str, Dict[str, Any]] = {}
        for task in self.config.tasks:
            results[task] = self._run_single_benchmark(
                algorithm, task, self.config
            )
        return results

    def _run_task_suite(self, task: str) -> Dict[str, Dict[str, Any]]:
        """Run all algorithms for a single task."""
        results: Dict[str, Dict[str, Any]] = {}
        for algo in self.config.algorithms:
            results[algo] = self._run_single_benchmark(
                algo, task, self.config
            )
        return results

    # ------------------------------------------------------------------
    # Aggregation & ranking
    # ------------------------------------------------------------------

    def _aggregate_results(
        self, all_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> BenchmarkResult:
        """Collapse per-(algo, task) results into algorithm-level and
        task-level summaries plus global rankings."""

        algorithm_results: Dict[str, Any] = {}
        task_results: Dict[str, Any] = {}
        metric_results: Dict[str, Any] = {}

        # --- algorithm-level ---
        for algo, tasks_dict in all_results.items():
            algo_metrics: Dict[str, List[float]] = {}
            for _task, tres in tasks_dict.items():
                if "metrics" not in tres:
                    continue
                for m_name, m_stats in tres["metrics"].items():
                    algo_metrics.setdefault(m_name, []).append(m_stats["mean"])
            summary: Dict[str, Dict[str, float]] = {}
            for m_name, vals in algo_metrics.items():
                arr = np.array(vals)
                summary[m_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            algorithm_results[algo] = {
                "metrics": summary,
                "n_tasks": len(tasks_dict),
            }

        # --- task-level ---
        for task in self.config.tasks:
            task_metrics: Dict[str, List[float]] = {}
            for algo in self.config.algorithms:
                tres = all_results.get(algo, {}).get(task, {})
                if "metrics" not in tres:
                    continue
                for m_name, m_stats in tres["metrics"].items():
                    task_metrics.setdefault(m_name, []).append(m_stats["mean"])
            summary_t: Dict[str, Dict[str, float]] = {}
            for m_name, vals in task_metrics.items():
                arr = np.array(vals)
                summary_t[m_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            task_results[task] = {"metrics": summary_t}

        # --- metric-level ---
        for m_name in self.config.metrics:
            per_algo: Dict[str, float] = {}
            for algo in self.config.algorithms:
                ar = algorithm_results.get(algo, {})
                if "metrics" in ar and m_name in ar["metrics"]:
                    per_algo[algo] = ar["metrics"][m_name]["mean"]
            metric_results[m_name] = per_algo

        # --- rankings ---
        rankings = self._compute_rankings(metric_results)

        # --- raw detail ---
        metric_results["_raw"] = all_results

        return BenchmarkResult(
            algorithm_results=algorithm_results,
            task_results=task_results,
            metric_results=metric_results,
            rankings=rankings,
        )

    def _compute_rankings(
        self, metric_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Rank algorithms per metric."""
        rankings: Dict[str, Any] = {}
        for m_name, algo_vals in metric_results.items():
            if m_name.startswith("_"):
                continue
            if not isinstance(algo_vals, dict):
                continue
            higher = METRIC_HIGHER_BETTER.get(m_name, True)
            sorted_pairs = sorted(
                algo_vals.items(), key=lambda kv: kv[1], reverse=higher
            )
            rankings[m_name] = [
                {"rank": i + 1, "algorithm": algo, "value": val}
                for i, (algo, val) in enumerate(sorted_pairs)
            ]

        # overall ranking via mean reciprocal rank
        algo_rr: Dict[str, List[float]] = {}
        for m_name, ranked in rankings.items():
            for entry in ranked:
                algo_rr.setdefault(entry["algorithm"], []).append(
                    1.0 / entry["rank"]
                )
        if algo_rr:
            overall = sorted(
                [(algo, float(np.mean(rrs))) for algo, rrs in algo_rr.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            rankings["_overall"] = [
                {"rank": i + 1, "algorithm": algo, "mrr": mrr}
                for i, (algo, mrr) in enumerate(overall)
            ]
        return rankings

    def _compare_to_baseline(
        self,
        results: BenchmarkResult,
        baseline_results: BenchmarkResult,
    ) -> Dict[str, Any]:
        """Compare current results to a baseline run."""
        comparison: Dict[str, Any] = {
            "current_id": results.benchmark_id,
            "baseline_id": baseline_results.benchmark_id,
            "metrics": {},
        }
        for m_name in self.config.metrics:
            cur_vals = results.metric_results.get(m_name, {})
            base_vals = baseline_results.metric_results.get(m_name, {})
            if not isinstance(cur_vals, dict) or not isinstance(base_vals, dict):
                continue
            diffs: Dict[str, Any] = {}
            for algo in self.config.algorithms:
                c = cur_vals.get(algo)
                b = base_vals.get(algo)
                if c is not None and b is not None:
                    delta = c - b
                    pct = (delta / abs(b) * 100) if abs(b) > 1e-12 else 0.0
                    higher = METRIC_HIGHER_BETTER.get(m_name, True)
                    improved = (delta > 0) == higher
                    diffs[algo] = {
                        "current": c,
                        "baseline": b,
                        "delta": delta,
                        "pct_change": pct,
                        "improved": improved,
                    }
            comparison["metrics"][m_name] = diffs
        return comparison

    # ------------------------------------------------------------------
    # Timeout & progress
    # ------------------------------------------------------------------

    @staticmethod
    def _check_timeout(start_time: float, timeout: float) -> bool:
        return (time.time() - start_time) > timeout

    def _log_progress(
        self, completed: int, total: int, current_task: str
    ) -> None:
        pct = completed / max(total, 1) * 100
        self._log(f"[{completed}/{total} {pct:.0f}%] {current_task}")

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, result: BenchmarkResult, path: str) -> None:
        """Save benchmark results to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = _result_to_serialisable(result)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self._log(f"Results saved to {path}")

    def load_results(self, path: str) -> BenchmarkResult:
        """Load benchmark results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return BenchmarkResult.from_dict(data)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, result: BenchmarkResult) -> str:
        """Produce a human-readable text summary."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  DIVERSITY DECODING ARENA — BENCHMARK REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"  Benchmark ID : {result.benchmark_id}")
        lines.append(f"  Suite        : {result.suite.value}")
        lines.append(f"  Status       : {result.status.value}")
        lines.append(f"  Duration     : {_format_duration(result.duration)}")
        lines.append(f"  Start        : {_ts(result.start_time)}")
        lines.append(f"  End          : {_ts(result.end_time)}")
        lines.append("")

        # System info
        si = result.system_info
        if si:
            lines.append("  System Info:")
            for k, v in si.items():
                lines.append(f"    {k:20s}: {v}")
            lines.append("")

        # Per-metric rankings
        for m_name in self.config.metrics:
            ranked = result.rankings.get(m_name)
            if not ranked:
                continue
            higher = METRIC_HIGHER_BETTER.get(m_name, True)
            direction = "↑ higher is better" if higher else "↓ lower is better"
            lines.append(f"  Metric: {m_name} ({direction})")
            headers = ["Rank", "Algorithm", "Value"]
            rows = [
                [str(e["rank"]), e["algorithm"], _format_number(e["value"])]
                for e in ranked
            ]
            lines.append(_indent(_format_table(headers, rows, ["r", "l", "r"]), 4))
            lines.append("")

        # Overall ranking
        overall = result.rankings.get("_overall")
        if overall:
            lines.append("  Overall Ranking (Mean Reciprocal Rank):")
            headers = ["Rank", "Algorithm", "MRR"]
            rows = [
                [str(e["rank"]), e["algorithm"], _format_number(e["mrr"])]
                for e in overall
            ]
            lines.append(_indent(_format_table(headers, rows, ["r", "l", "r"]), 4))
            lines.append("")

        # Errors
        if result.errors:
            lines.append("  Errors:")
            for err in result.errors:
                lines.append(f"    • {err[:200]}")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    def generate_html_report(self, result: BenchmarkResult) -> str:
        """Produce an HTML document summarising benchmark results."""
        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html lang='en'><head><meta charset='utf-8'>")
        parts.append("<title>Benchmark Report — Diversity Decoding Arena</title>")
        parts.append("<style>")
        parts.append(
            "body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;"
            "color:#222;background:#fafafa;}"
        )
        parts.append("h1{color:#1a1a2e;} h2{color:#16213e;}")
        parts.append(
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}"
        )
        parts.append(
            "th,td{border:1px solid #ccc;padding:6px 12px;text-align:right;}"
        )
        parts.append("th{background:#e2e8f0;} tr:nth-child(even){background:#f1f5f9;}")
        parts.append(".good{color:#16a34a;} .bad{color:#dc2626;}")
        parts.append("</style></head><body>")
        parts.append(f"<h1>Benchmark Report</h1>")
        parts.append(f"<p><strong>ID:</strong> {result.benchmark_id} &mdash; "
                      f"<strong>Suite:</strong> {result.suite.value} &mdash; "
                      f"<strong>Duration:</strong> {_format_duration(result.duration)}</p>")

        # Metric tables
        for m_name in self.config.metrics:
            ranked = result.rankings.get(m_name)
            if not ranked:
                continue
            higher = METRIC_HIGHER_BETTER.get(m_name, True)
            arrow = "↑" if higher else "↓"
            parts.append(f"<h2>{m_name} {arrow}</h2><table>")
            parts.append("<tr><th>Rank</th><th>Algorithm</th><th>Value</th></tr>")
            for e in ranked:
                parts.append(
                    f"<tr><td>{e['rank']}</td><td style='text-align:left'>"
                    f"{e['algorithm']}</td><td>{_format_number(e['value'])}</td></tr>"
                )
            parts.append("</table>")

        # Overall
        overall = result.rankings.get("_overall")
        if overall:
            parts.append("<h2>Overall Ranking (MRR)</h2><table>")
            parts.append("<tr><th>Rank</th><th>Algorithm</th><th>MRR</th></tr>")
            for e in overall:
                parts.append(
                    f"<tr><td>{e['rank']}</td><td style='text-align:left'>"
                    f"{e['algorithm']}</td><td>{_format_number(e['mrr'])}</td></tr>"
                )
            parts.append("</table>")

        # System info
        si = result.system_info
        if si:
            parts.append("<h2>System Info</h2><table>")
            for k, v in si.items():
                parts.append(
                    f"<tr><td style='text-align:left'><strong>{k}</strong></td>"
                    f"<td style='text-align:left'>{v}</td></tr>"
                )
            parts.append("</table>")

        # Errors
        if result.errors:
            parts.append("<h2>Errors</h2><ul>")
            for err in result.errors:
                parts.append(f"<li>{_html_escape(err[:300])}</li>")
            parts.append("</ul>")

        parts.append("</body></html>")
        return "\n".join(parts)

    def generate_latex_table(self, result: BenchmarkResult) -> str:
        """Generate LaTeX tabular environment for algorithm comparison."""
        metrics = [m for m in self.config.metrics if m in result.metric_results]
        if not metrics:
            return "% No metrics available"

        algos = self.config.algorithms
        n_cols = 1 + len(metrics)
        col_spec = "l" + "r" * len(metrics)

        lines: List[str] = []
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{Diversity Decoding Benchmark Results}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        header = "Algorithm & " + " & ".join(
            m.replace("_", "\\_") for m in metrics
        ) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for algo in algos:
            cells = [algo.replace("_", "\\_")]
            for m in metrics:
                vals = result.metric_results.get(m, {})
                if isinstance(vals, dict) and algo in vals:
                    cells.append(_format_number(vals[algo]))
                else:
                    cells.append("--")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def generate_csv(self, result: BenchmarkResult) -> str:
        """Generate CSV of per-algorithm per-metric results."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        metrics = [m for m in self.config.metrics if m in result.metric_results]
        writer.writerow(["algorithm"] + metrics)
        for algo in self.config.algorithms:
            row = [algo]
            for m in metrics:
                vals = result.metric_results.get(m, {})
                if isinstance(vals, dict) and algo in vals:
                    row.append(_format_number(vals[algo]))
                else:
                    row.append("")
            writer.writerow(row)
        return buf.getvalue()

    def compare_benchmark_runs(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
    ) -> str:
        """Produce a text comparison report between two benchmark results."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  BENCHMARK COMPARISON REPORT")
        lines.append("=" * 72)
        lines.append(f"  Run A: {result_a.benchmark_id}")
        lines.append(f"  Run B: {result_b.benchmark_id}")
        lines.append(f"  Duration A: {_format_duration(result_a.duration)}")
        lines.append(f"  Duration B: {_format_duration(result_b.duration)}")
        lines.append("")

        all_metrics = set()
        for m in (list(result_a.metric_results.keys()) +
                  list(result_b.metric_results.keys())):
            if not m.startswith("_"):
                all_metrics.add(m)

        for m_name in sorted(all_metrics):
            higher = METRIC_HIGHER_BETTER.get(m_name, True)
            arrow = "↑" if higher else "↓"
            lines.append(f"  Metric: {m_name} {arrow}")

            vals_a = result_a.metric_results.get(m_name, {})
            vals_b = result_b.metric_results.get(m_name, {})
            if not isinstance(vals_a, dict) or not isinstance(vals_b, dict):
                lines.append("    (data format mismatch)")
                continue

            all_algos = sorted(set(list(vals_a.keys()) + list(vals_b.keys())))
            headers = ["Algorithm", "Run A", "Run B", "Delta", "Change %", "Winner"]
            rows: List[List[str]] = []
            for algo in all_algos:
                a = vals_a.get(algo)
                b = vals_b.get(algo)
                if a is None or b is None:
                    rows.append([algo, _fmt(a), _fmt(b), "--", "--", "--"])
                    continue
                delta = b - a
                pct = (delta / abs(a) * 100) if abs(a) > 1e-12 else 0.0
                better_b = (delta > 0) == higher
                winner = "B" if better_b else ("A" if abs(delta) > 1e-9 else "=")
                rows.append([
                    algo,
                    _format_number(a),
                    _format_number(b),
                    f"{delta:+.4f}",
                    f"{pct:+.1f}%",
                    winner,
                ])
            lines.append(_indent(
                _format_table(headers, rows, ["l", "r", "r", "r", "r", "c"]), 4
            ))
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation & estimation
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        """Return list of config validation errors (empty = valid)."""
        errors: List[str] = []
        if not self.config.algorithms:
            errors.append("No algorithms specified")
        if not self.config.tasks:
            errors.append("No tasks specified")
        if not self.config.metrics:
            errors.append("No metrics specified")
        if self.config.num_samples < 1:
            errors.append("num_samples must be >= 1")
        if self.config.max_length < 1:
            errors.append("max_length must be >= 1")
        if self.config.num_runs < 1:
            errors.append("num_runs must be >= 1")
        if self.config.timeout_per_run <= 0:
            errors.append("timeout_per_run must be > 0")
        if self.config.parallel_workers < 1:
            errors.append("parallel_workers must be >= 1")
        for algo in self.config.algorithms:
            if not isinstance(algo, str) or not algo.strip():
                errors.append(f"Invalid algorithm name: {algo!r}")
        for task in self.config.tasks:
            if not isinstance(task, str) or not task.strip():
                errors.append(f"Invalid task name: {task!r}")
        for metric in self.config.metrics:
            if metric not in METRIC_FUNCTIONS and metric not in ("latency", "memory_peak"):
                errors.append(f"Unknown metric: {metric!r}")
        return errors

    def estimate_runtime(self) -> float:
        """Estimate wall-clock seconds for the current config."""
        return SystemProfiler.estimate_runtime(
            len(self.config.algorithms),
            len(self.config.tasks),
            self.config.num_samples,
        ) * self.config.num_runs

    def dry_run(self) -> str:
        """Preview what will be executed without actually running."""
        lines: List[str] = []
        lines.append("DRY RUN — Benchmark Preview")
        lines.append("=" * 50)
        lines.append(f"Suite       : {self.config.suite.value}")
        lines.append(f"Algorithms  : {', '.join(self.config.algorithms)}")
        lines.append(f"Tasks       : {', '.join(self.config.tasks)}")
        lines.append(f"Metrics     : {', '.join(self.config.metrics)}")
        lines.append(f"Samples     : {self.config.num_samples}")
        lines.append(f"Max length  : {self.config.max_length}")
        lines.append(f"Runs        : {self.config.num_runs}")
        lines.append(f"Seed        : {self.config.seed}")
        lines.append(f"Timeout/run : {self.config.timeout_per_run}s")
        lines.append(f"Workers     : {self.config.parallel_workers}")
        lines.append("")
        total = len(self.config.algorithms) * len(self.config.tasks)
        lines.append(f"Total benchmark units: {total}")
        lines.append(f"Total executions     : {total * self.config.num_runs}")
        est = self.estimate_runtime()
        lines.append(f"Estimated runtime    : {_format_duration(est)}")
        lines.append("")
        lines.append("Execution plan:")
        idx = 1
        for algo in self.config.algorithms:
            for task in self.config.tasks:
                lines.append(f"  {idx:3d}. {algo} × {task}")
                idx += 1

        errors = self.validate_config()
        if errors:
            lines.append("")
            lines.append("⚠ Config validation errors:")
            for e in errors:
                lines.append(f"  • {e}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Progress bar factory
    # ------------------------------------------------------------------

    def _create_progress_bar(self, total: int) -> ProgressBar:
        return ProgressBar(total, prefix="Benchmark", bar_length=30)

    # ------------------------------------------------------------------
    # Regression checking
    # ------------------------------------------------------------------

    def _regression_check(
        self,
        current: BenchmarkResult,
        previous: BenchmarkResult,
        threshold: float = 0.05,
    ) -> List[str]:
        """Identify metrics that regressed beyond *threshold* (fractional)."""
        regressions: List[str] = []
        for m_name in self.config.metrics:
            if m_name.startswith("_"):
                continue
            cur_vals = current.metric_results.get(m_name, {})
            prev_vals = previous.metric_results.get(m_name, {})
            if not isinstance(cur_vals, dict) or not isinstance(prev_vals, dict):
                continue
            higher = METRIC_HIGHER_BETTER.get(m_name, True)
            for algo in self.config.algorithms:
                c = cur_vals.get(algo)
                p = prev_vals.get(algo)
                if c is None or p is None:
                    continue
                if abs(p) < 1e-12:
                    continue
                change = (c - p) / abs(p)
                regressed = (change < -threshold) if higher else (change > threshold)
                if regressed:
                    regressions.append(
                        f"{algo}/{m_name}: {p:.4f} -> {c:.4f} "
                        f"({change:+.1%} {'worse' if regressed else 'ok'})"
                    )
        return regressions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_id(self) -> str:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(
            f"{self.config.suite.value}:{self.config.seed}:{ts}".encode()
        ).hexdigest()[:8]
        return f"bench_{ts}_{h}"


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def _ts(epoch: float) -> str:
    if epoch <= 0:
        return "N/A"
    return datetime.datetime.fromtimestamp(
        epoch, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.split("\n"))


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(v: Optional[float]) -> str:
    return _format_number(v) if v is not None else "--"


def _result_to_serialisable(result: BenchmarkResult) -> Dict[str, Any]:
    """Recursively convert a BenchmarkResult to JSON-safe dict."""
    d = result.to_dict()
    return _make_serialisable(d)


def _make_serialisable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (BenchmarkSuite, BenchmarkStatus)):
        return obj.value
    return obj


# ---------------------------------------------------------------------------
# CLI entry-point helpers
# ---------------------------------------------------------------------------

def parse_cli_args(argv: Optional[List[str]] = None) -> BenchmarkConfig:
    """Minimal argument parser for benchmark CLI (stdlib only)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Diversity Decoding Arena — Benchmark Runner"
    )
    parser.add_argument(
        "--suite",
        choices=[s.value for s in BenchmarkSuite],
        default="standard",
        help="Preset benchmark suite",
    )
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument("--compare-to", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-format", choices=["text", "html", "latex", "csv"],
                        default="text")

    args = parser.parse_args(argv)
    suite = BenchmarkSuite(args.suite)
    config = BenchmarkSuitePresets.build_config(suite, seed=args.seed)

    if args.algorithms is not None:
        config.algorithms = args.algorithms
    if args.tasks is not None:
        config.tasks = args.tasks
    if args.metrics is not None:
        config.metrics = args.metrics
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    if args.max_length is not None:
        config.max_length = args.max_length
    if args.num_runs is not None:
        config.num_runs = args.num_runs
    config.output_dir = args.output_dir
    config.timeout_per_run = args.timeout
    config.parallel_workers = args.workers
    config.save_generations = args.save_generations
    config.compare_to_baseline = args.compare_to

    return config


def run_from_cli(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse args, run benchmark, produce report."""
    import argparse

    # We need to check for --dry-run before fully running
    config = parse_cli_args(argv)
    runner = BenchmarkRunner(config)

    # Re-parse just for --dry-run and --report-format
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-format", default="text")
    known, _ = parser.parse_known_args(argv)

    if known.dry_run:
        print(runner.dry_run())
        return

    result = runner.run()

    # Save raw JSON
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, f"{result.benchmark_id}.json")
    runner.save_results(result, json_path)

    # Generate requested report
    fmt = known.report_format
    if fmt == "html":
        report = runner.generate_html_report(result)
        ext = "html"
    elif fmt == "latex":
        report = runner.generate_latex_table(result)
        ext = "tex"
    elif fmt == "csv":
        report = runner.generate_csv(result)
        ext = "csv"
    else:
        report = runner.generate_report(result)
        ext = "txt"

    report_path = os.path.join(config.output_dir, f"{result.benchmark_id}.{ext}")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nResults saved to {json_path}")
    print(f"Report  saved to {report_path}")


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if rng is None:
        rng = np.random.RandomState(0)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size (Cohen's d) between two samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = math.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Simplified Mann-Whitney U statistic and approximate p-value."""
    combined = np.concatenate([a, b])
    ranks = np.empty_like(combined, dtype=np.float64)
    order = combined.argsort()
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)

    # Handle ties by averaging ranks
    unique_vals = np.unique(combined)
    for v in unique_vals:
        mask = combined == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    na, nb = len(a), len(b)
    u1 = ranks[:na].sum() - na * (na + 1) / 2
    u2 = na * nb - u1

    u = min(u1, u2)
    # Normal approximation for p-value
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    if sigma < 1e-12:
        return float(u), 1.0
    z = (u - mu) / sigma
    # Two-tailed p from standard normal CDF approximation
    p = 2.0 * _norm_cdf(-abs(z))
    return float(u), float(p)


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Extended report utilities
# ---------------------------------------------------------------------------

def generate_detailed_algorithm_report(
    result: BenchmarkResult,
    algorithm: str,
    config: BenchmarkConfig,
) -> str:
    """Deep-dive report for a single algorithm across all tasks."""
    lines: List[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  Detailed Report: {algorithm}")
    lines.append(f"{'=' * 60}")
    lines.append("")

    algo_data = result.algorithm_results.get(algorithm, {})
    if not algo_data:
        lines.append("  (no data available for this algorithm)")
        return "\n".join(lines)

    metrics_summary = algo_data.get("metrics", {})
    lines.append("  Aggregate Metrics (across all tasks):")
    headers = ["Metric", "Mean", "Std", "Min", "Max"]
    rows: List[List[str]] = []
    for m_name, stats in metrics_summary.items():
        rows.append([
            m_name,
            _format_number(stats["mean"]),
            _format_number(stats["std"]),
            _format_number(stats["min"]),
            _format_number(stats["max"]),
        ])
    lines.append(_indent(_format_table(headers, rows, ["l", "r", "r", "r", "r"]), 4))
    lines.append("")

    # Per-task breakdown
    raw = result.metric_results.get("_raw", {})
    algo_raw = raw.get(algorithm, {})
    if algo_raw:
        lines.append("  Per-Task Breakdown:")
        for task, task_data in algo_raw.items():
            lines.append(f"\n    Task: {task}")
            task_metrics = task_data.get("metrics", {})
            headers_t = ["Metric", "Mean", "Std"]
            rows_t: List[List[str]] = []
            for m_name, stats in task_metrics.items():
                rows_t.append([
                    m_name,
                    _format_number(stats["mean"]),
                    _format_number(stats["std"]),
                ])
            lines.append(_indent(
                _format_table(headers_t, rows_t, ["l", "r", "r"]), 6
            ))

    # Rankings
    lines.append("")
    lines.append("  Rankings:")
    for m_name, ranked in result.rankings.items():
        if m_name.startswith("_"):
            continue
        for entry in ranked:
            if entry["algorithm"] == algorithm:
                lines.append(
                    f"    {m_name}: Rank {entry['rank']} "
                    f"(value={_format_number(entry['value'])})"
                )
                break

    overall = result.rankings.get("_overall", [])
    for entry in overall:
        if entry["algorithm"] == algorithm:
            lines.append(
                f"    Overall: Rank {entry['rank']} "
                f"(MRR={_format_number(entry['mrr'])})"
            )
            break

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def generate_task_difficulty_report(
    result: BenchmarkResult,
    config: BenchmarkConfig,
) -> str:
    """Report on how tasks differ in difficulty for algorithms."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  Task Difficulty Analysis")
    lines.append("=" * 60)
    lines.append("")

    raw = result.metric_results.get("_raw", {})
    if not raw:
        lines.append("  (no raw data available)")
        return "\n".join(lines)

    for task in config.tasks:
        lines.append(f"  Task: {task}")
        algo_scores: Dict[str, Dict[str, float]] = {}
        for algo in config.algorithms:
            task_data = raw.get(algo, {}).get(task, {})
            if "metrics" in task_data:
                algo_scores[algo] = {
                    m: s["mean"] for m, s in task_data["metrics"].items()
                }

        if algo_scores:
            # Show variance across algorithms for each metric
            for m_name in config.metrics:
                vals = [
                    s.get(m_name, float("nan"))
                    for s in algo_scores.values()
                    if m_name in s
                ]
                if vals:
                    arr = np.array(vals)
                    lines.append(
                        f"    {m_name}: mean={np.mean(arr):.4f} "
                        f"std={np.std(arr):.4f} "
                        f"range=[{np.min(arr):.4f}, {np.max(arr):.4f}]"
                    )
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_statistical_comparison(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    config: BenchmarkConfig,
) -> str:
    """Produce statistical significance tests between two runs."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  Statistical Comparison")
    lines.append("=" * 60)
    lines.append(f"  Run A: {result_a.benchmark_id}")
    lines.append(f"  Run B: {result_b.benchmark_id}")
    lines.append("")

    raw_a = result_a.metric_results.get("_raw", {})
    raw_b = result_b.metric_results.get("_raw", {})

    for m_name in config.metrics:
        lines.append(f"  Metric: {m_name}")
        headers = ["Algorithm", "Mean A", "Mean B", "Cohen's d", "U stat", "p-value", "Sig?"]
        rows: List[List[str]] = []

        for algo in config.algorithms:
            vals_a: List[float] = []
            vals_b: List[float] = []
            for task in config.tasks:
                td_a = raw_a.get(algo, {}).get(task, {})
                td_b = raw_b.get(algo, {}).get(task, {})
                if "metrics" in td_a and m_name in td_a["metrics"]:
                    vals_a.append(td_a["metrics"][m_name]["mean"])
                if "metrics" in td_b and m_name in td_b["metrics"]:
                    vals_b.append(td_b["metrics"][m_name]["mean"])

            if len(vals_a) >= 2 and len(vals_b) >= 2:
                arr_a = np.array(vals_a)
                arr_b = np.array(vals_b)
                d = _cohens_d(arr_a, arr_b)
                u, p = _mann_whitney_u(arr_a, arr_b)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                rows.append([
                    algo,
                    _format_number(float(arr_a.mean())),
                    _format_number(float(arr_b.mean())),
                    f"{d:.3f}",
                    f"{u:.1f}",
                    f"{p:.4f}",
                    sig,
                ])
            else:
                rows.append([algo, "--", "--", "--", "--", "--", ""])

        lines.append(_indent(
            _format_table(headers, rows, ["l", "r", "r", "r", "r", "r", "c"]), 4
        ))
        lines.append("")

    lines.append("  Significance levels: * p<0.05  ** p<0.01  *** p<0.001")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ablation study helpers
# ---------------------------------------------------------------------------

class AblationRunner:
    """Run parameter ablations for a single algorithm."""

    def __init__(
        self,
        algorithm: str,
        base_config: BenchmarkConfig,
        param_name: str,
        param_values: List[Any],
    ) -> None:
        self.algorithm = algorithm
        self.base_config = base_config
        self.param_name = param_name
        self.param_values = param_values

    def run(self) -> Dict[str, Any]:
        """Execute ablation and return per-value results."""
        results: Dict[str, Any] = {}
        for val in self.param_values:
            cfg = BenchmarkConfig(
                suite=BenchmarkSuite.ABLATION,
                algorithms=[self.algorithm],
                tasks=self.base_config.tasks,
                metrics=self.base_config.metrics,
                num_samples=self.base_config.num_samples,
                max_length=self.base_config.max_length if self.param_name != "max_length" else val,
                num_runs=self.base_config.num_runs,
                seed=self.base_config.seed,
            )
            if self.param_name == "num_samples":
                cfg.num_samples = val
            elif self.param_name == "max_length":
                cfg.max_length = val

            runner = BenchmarkRunner(cfg)
            result = runner.run()
            key = str(val)
            results[key] = {
                "value": val,
                "metrics": result.metric_results,
                "duration": result.duration,
            }
        return {
            "algorithm": self.algorithm,
            "param_name": self.param_name,
            "param_values": self.param_values,
            "results": results,
        }

    def generate_report(self, ablation_results: Dict[str, Any]) -> str:
        """Text report for ablation study."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"  Ablation Study: {self.algorithm}")
        lines.append(f"  Parameter: {self.param_name}")
        lines.append("=" * 60)
        lines.append("")

        per_value = ablation_results.get("results", {})
        for m_name in self.base_config.metrics:
            lines.append(f"  {m_name}:")
            headers = [self.param_name, "Value", "Duration"]
            rows: List[List[str]] = []
            for key in sorted(per_value.keys(), key=lambda k: per_value[k]["value"]):
                entry = per_value[key]
                m_vals = entry.get("metrics", {}).get(m_name, {})
                if isinstance(m_vals, dict) and self.algorithm in m_vals:
                    score = _format_number(m_vals[self.algorithm])
                else:
                    score = "--"
                rows.append([
                    str(entry["value"]),
                    score,
                    _format_duration(entry.get("duration", 0)),
                ])
            lines.append(_indent(
                _format_table(headers, rows, ["r", "r", "r"]), 4
            ))
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

class Leaderboard:
    """Maintain a persistent leaderboard across benchmark runs."""

    def __init__(self, path: str = "leaderboard.json") -> None:
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.entries = []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the leaderboard."""
        overall = result.rankings.get("_overall", [])
        entry = {
            "benchmark_id": result.benchmark_id,
            "suite": result.suite.value,
            "duration": result.duration,
            "timestamp": _ts(result.start_time),
            "top_algorithm": overall[0]["algorithm"] if overall else "N/A",
            "top_mrr": overall[0]["mrr"] if overall else 0.0,
            "n_algorithms": len(result.algorithm_results),
            "status": result.status.value,
        }
        self.entries.append(entry)
        self._save()

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

    def get_best(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return top-N entries by top MRR."""
        sorted_entries = sorted(
            self.entries, key=lambda e: e.get("top_mrr", 0), reverse=True
        )
        return sorted_entries[:n]

    def render(self, n: int = 10) -> str:
        """Render leaderboard as text table."""
        best = self.get_best(n)
        if not best:
            return "(empty leaderboard)"
        headers = ["#", "ID", "Suite", "Top Algo", "MRR", "Duration", "Date"]
        rows: List[List[str]] = []
        for i, e in enumerate(best):
            rows.append([
                str(i + 1),
                e.get("benchmark_id", "?")[:20],
                e.get("suite", "?"),
                e.get("top_algorithm", "?"),
                _format_number(e.get("top_mrr", 0)),
                _format_duration(e.get("duration", 0)),
                str(e.get("timestamp", "?"))[:19],
            ])
        return _format_table(headers, rows, ["r", "l", "l", "l", "r", "r", "l"])


# ---------------------------------------------------------------------------
# Metric correlation analysis
# ---------------------------------------------------------------------------

def compute_metric_correlations(
    result: BenchmarkResult,
    config: BenchmarkConfig,
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Pearson correlation between metrics across algorithms."""
    metrics = [m for m in config.metrics if m in result.metric_results
               and isinstance(result.metric_results[m], dict)]
    n = len(metrics)
    if n < 2:
        return {}

    vectors: Dict[str, np.ndarray] = {}
    algos = config.algorithms
    for m in metrics:
        vals = result.metric_results[m]
        vectors[m] = np.array([vals.get(a, float("nan")) for a in algos])

    corr: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        mi = metrics[i]
        corr[mi] = {}
        for j in range(n):
            mj = metrics[j]
            vi = vectors[mi]
            vj = vectors[mj]
            mask = ~(np.isnan(vi) | np.isnan(vj))
            if mask.sum() < 2:
                corr[mi][mj] = float("nan")
                continue
            vi_clean = vi[mask]
            vj_clean = vj[mask]
            std_i = vi_clean.std()
            std_j = vj_clean.std()
            if std_i < 1e-12 or std_j < 1e-12:
                corr[mi][mj] = 0.0
            else:
                corr[mi][mj] = float(np.corrcoef(vi_clean, vj_clean)[0, 1])
    return corr


def render_correlation_matrix(
    corr: Dict[str, Dict[str, float]],
) -> str:
    """Pretty-print a correlation matrix."""
    metrics = list(corr.keys())
    if not metrics:
        return "(no correlations)"
    headers = [""] + metrics
    rows: List[List[str]] = []
    for mi in metrics:
        row = [mi]
        for mj in metrics:
            v = corr.get(mi, {}).get(mj, float("nan"))
            if math.isnan(v):
                row.append("  --  ")
            else:
                row.append(f"{v:+.3f}")
        rows.append(row)
    return _format_table(headers, rows)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    result: BenchmarkResult,
    metric_x: str,
    metric_y: str,
) -> List[str]:
    """Return algorithms on the Pareto frontier for two metrics (both maximised)."""
    vals_x = result.metric_results.get(metric_x, {})
    vals_y = result.metric_results.get(metric_y, {})
    if not isinstance(vals_x, dict) or not isinstance(vals_y, dict):
        return []

    points: List[Tuple[str, float, float]] = []
    for algo in set(vals_x.keys()) & set(vals_y.keys()):
        x = vals_x[algo]
        y = vals_y[algo]
        # Flip sign if lower is better so we always maximise
        if not METRIC_HIGHER_BETTER.get(metric_x, True):
            x = -x
        if not METRIC_HIGHER_BETTER.get(metric_y, True):
            y = -y
        points.append((algo, x, y))

    # Find non-dominated set
    pareto: List[str] = []
    for algo_i, xi, yi in points:
        dominated = False
        for algo_j, xj, yj in points:
            if algo_i == algo_j:
                continue
            if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                dominated = True
                break
        if not dominated:
            pareto.append(algo_i)
    return sorted(pareto)


def render_pareto_report(
    result: BenchmarkResult,
    config: BenchmarkConfig,
) -> str:
    """Report Pareto frontiers for all pairs of metrics."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  Pareto Frontier Analysis")
    lines.append("=" * 60)
    lines.append("")

    metrics = [m for m in config.metrics
               if m in result.metric_results
               and isinstance(result.metric_results[m], dict)]
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            mx, my = metrics[i], metrics[j]
            frontier = compute_pareto_frontier(result, mx, my)
            lines.append(f"  {mx} vs {my}: {', '.join(frontier) or 'none'}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def compute_config_hash(config: BenchmarkConfig) -> str:
    """Deterministic hash of a config for reproducibility tracking."""
    d = config.to_dict()
    # Remove non-deterministic fields
    d.pop("output_dir", None)
    d.pop("timeout_per_run", None)
    d.pop("parallel_workers", None)
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def verify_reproducibility(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    tolerance: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """Check if two benchmark runs produced identical results within tolerance."""
    diffs: List[str] = []
    for m_name in set(list(result_a.metric_results.keys()) +
                      list(result_b.metric_results.keys())):
        if m_name.startswith("_"):
            continue
        va = result_a.metric_results.get(m_name, {})
        vb = result_b.metric_results.get(m_name, {})
        if not isinstance(va, dict) or not isinstance(vb, dict):
            continue
        for algo in set(list(va.keys()) + list(vb.keys())):
            a = va.get(algo)
            b = vb.get(algo)
            if a is None or b is None:
                diffs.append(f"{m_name}/{algo}: missing in one run")
                continue
            if abs(a - b) > tolerance:
                diffs.append(
                    f"{m_name}/{algo}: {a:.8f} vs {b:.8f} "
                    f"(delta={abs(a - b):.2e})"
                )
    return len(diffs) == 0, diffs


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class BatchBenchmarkRunner:
    """Run multiple benchmark configurations in sequence and produce a
    combined report."""

    def __init__(self, configs: List[BenchmarkConfig]) -> None:
        self.configs = configs
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> List[BenchmarkResult]:
        self.results = []
        for i, cfg in enumerate(self.configs):
            print(f"\n--- Batch {i + 1}/{len(self.configs)}: "
                  f"{cfg.suite.value} ---")
            runner = BenchmarkRunner(cfg)
            result = runner.run()
            self.results.append(result)
        return self.results

    def combined_report(self) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  BATCH BENCHMARK SUMMARY")
        lines.append("=" * 72)
        lines.append("")
        headers = ["#", "Suite", "Status", "Duration", "Top Algo"]
        rows: List[List[str]] = []
        for i, r in enumerate(self.results):
            overall = r.rankings.get("_overall", [])
            top = overall[0]["algorithm"] if overall else "N/A"
            rows.append([
                str(i + 1),
                r.suite.value,
                r.status.value,
                _format_duration(r.duration),
                top,
            ])
        lines.append(_indent(
            _format_table(headers, rows, ["r", "l", "l", "r", "l"]), 2
        ))
        lines.append("")
        total_dur = sum(r.duration for r in self.results)
        lines.append(f"  Total time: {_format_duration(total_dur)}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result filtering & querying
# ---------------------------------------------------------------------------

def filter_results_by_metric(
    result: BenchmarkResult,
    metric: str,
    threshold: float,
    above: bool = True,
) -> List[str]:
    """Return algorithms whose mean metric value passes a threshold."""
    vals = result.metric_results.get(metric, {})
    if not isinstance(vals, dict):
        return []
    matched: List[str] = []
    for algo, val in vals.items():
        if above and val >= threshold:
            matched.append(algo)
        elif not above and val <= threshold:
            matched.append(algo)
    return sorted(matched)


def top_k_algorithms(
    result: BenchmarkResult,
    metric: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top-k algorithms for a given metric."""
    vals = result.metric_results.get(metric, {})
    if not isinstance(vals, dict):
        return []
    higher = METRIC_HIGHER_BETTER.get(metric, True)
    sorted_pairs = sorted(vals.items(), key=lambda kv: kv[1], reverse=higher)
    return sorted_pairs[:k]


def worst_k_algorithms(
    result: BenchmarkResult,
    metric: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Return worst-k algorithms for a given metric."""
    vals = result.metric_results.get(metric, {})
    if not isinstance(vals, dict):
        return []
    higher = METRIC_HIGHER_BETTER.get(metric, True)
    sorted_pairs = sorted(vals.items(), key=lambda kv: kv[1], reverse=not higher)
    return sorted_pairs[:k]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_results_to_numpy(
    result: BenchmarkResult,
    config: BenchmarkConfig,
) -> Dict[str, np.ndarray]:
    """Export metric values as numpy arrays for further analysis."""
    metrics = [m for m in config.metrics
               if m in result.metric_results
               and isinstance(result.metric_results[m], dict)]
    algos = config.algorithms

    matrix = np.full((len(algos), len(metrics)), np.nan)
    for j, m in enumerate(metrics):
        vals = result.metric_results[m]
        for i, a in enumerate(algos):
            if a in vals:
                matrix[i, j] = vals[a]

    return {
        "matrix": matrix,
        "algorithms": np.array(algos),
        "metrics": np.array(metrics),
    }


def normalise_metrics(
    matrix: np.ndarray,
    higher_is_better: List[bool],
) -> np.ndarray:
    """Min-max normalise each column, flipping direction where needed."""
    normed = np.copy(matrix)
    for j in range(normed.shape[1]):
        col = normed[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            continue
        mn = col[mask].min()
        mx = col[mask].max()
        rng = mx - mn
        if rng < 1e-12:
            normed[mask, j] = 0.5
        else:
            normed[mask, j] = (col[mask] - mn) / rng
            if not higher_is_better[j]:
                normed[mask, j] = 1.0 - normed[mask, j]
    return normed


def compute_composite_score(
    result: BenchmarkResult,
    config: BenchmarkConfig,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Weighted composite score across all metrics for each algorithm."""
    exported = export_results_to_numpy(result, config)
    matrix = exported["matrix"]
    metrics_list = list(exported["metrics"])
    algos = list(exported["algorithms"])

    hib = [METRIC_HIGHER_BETTER.get(m, True) for m in metrics_list]
    normed = normalise_metrics(matrix, hib)

    if weights is None:
        w = np.ones(len(metrics_list)) / len(metrics_list)
    else:
        w = np.array([weights.get(m, 1.0) for m in metrics_list])
        w = w / w.sum()

    scores: Dict[str, float] = {}
    for i, algo in enumerate(algos):
        row = normed[i]
        valid = ~np.isnan(row)
        if valid.any():
            scores[algo] = float(np.dot(row[valid], w[valid]) / w[valid].sum())
        else:
            scores[algo] = 0.0
    return scores


# ---------------------------------------------------------------------------
# Warm-up / smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> bool:
    """Quick self-test that verifies the benchmark pipeline works end-to-end."""
    cfg = BenchmarkSuitePresets.build_config(BenchmarkSuite.QUICK, seed=0)
    cfg.num_samples = 5
    cfg.num_runs = 1
    runner = BenchmarkRunner(cfg)
    result = runner.run()
    ok = result.status in (BenchmarkStatus.COMPLETED,)
    if ok:
        # Verify we got some rankings
        ok = ok and bool(result.rankings)
        ok = ok and "_overall" in result.rankings
    return ok


# ---------------------------------------------------------------------------
# __main__ support
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_from_cli()
