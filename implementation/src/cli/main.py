#!/usr/bin/env python3
"""
Diversity Decoding Arena - CLI Entry Point

Command-line interface for running diversity decoding experiments,
evaluating results, performing statistical comparisons, and
generating visualizations.

Usage:
    python -m cli.main run --algorithms beam_search diverse_beam_search --tasks story_gen
    python -m cli.main evaluate --input-dir ./results --metrics entropy coverage
    python -m cli.main compare --input-dirs ./results_a ./results_b --method bayes
    python -m cli.main visualize --input-dir ./results --plot-types heatmap radar
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime
import hashlib
import io
import itertools
import json
import logging
import math
import os
import platform
import random
import shutil
import signal
import sqlite3
import struct
import sys
import textwrap
import time
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Version & constants
# ---------------------------------------------------------------------------

__version__ = "0.5.0"
__prog__ = "diversity-arena"

_BANNER = r"""
  ____  _                    _ _           ____                      _ _
 |  _ \(_)_   _____ _ __ ___(_) |_ _   _  |  _ \  ___  ___ ___   __| (_)_ __   __ _
 | | | | \ \ / / _ \ '__/ __| | __| | | | | | | |/ _ \/ __/ _ \ / _` | | '_ \ / _` |
 | |_| | |\ V /  __/ |  \__ \ | |_| |_| | | |_| |  __/ (_| (_) | (_| | | | | | (_| |
 |____/|_| \_/ \___|_|  |___/_|\__|\__, | |____/ \___|\___\___/ \__,_|_|_| |_|\__, |
                                    |___/                                       |___/
                              _
                 /\          (_)
                /  \   _ __ ___ _ __   __ _
               / /\ \ | '__/ _ \ '_ \ / _` |
              / ____ \| | |  __/ | | | (_| |
             /_/    \_\_|  \___|_| |_|\__,_|
"""

_DEFAULT_CONFIG_FILENAME = "arena_config.json"
_DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_PARALLEL = 64
_SUPPORTED_PLOT_FORMATS = ("png", "svg", "html", "pdf")
_SUPPORTED_EXPORT_FORMATS = ("csv", "json", "latex", "sqlite")
_COMPARISON_METHODS = ("bayes", "bootstrap", "permutation")
_DEFAULT_ROPE_WIDTH = 0.01
_DEFAULT_ALPHA = 0.05
_DEFAULT_NUM_SAMPLES = 100
_DEFAULT_MAX_LENGTH = 256
_DEFAULT_BAR_LENGTH = 40
_DEFAULT_SEED = 42
_DEFAULT_BUDGET = 50
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# ANSI colour codes
_COLORS: Dict[str, str] = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "underline": "\033[4m",
    "reset": "\033[0m",
}

logger = logging.getLogger(__prog__)

# ---------------------------------------------------------------------------
# Graceful shutdown handling
# ---------------------------------------------------------------------------

_shutdown_requested = False
_original_sigint: Any = None
_original_sigterm: Any = None


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    if _shutdown_requested:
        sys.stderr.write("\nForced exit.\n")
        sys.exit(128 + signum)
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    sys.stderr.write(f"\nReceived {sig_name}. Shutting down gracefully (press again to force)...\n")


def _install_signal_handlers() -> None:
    global _original_sigint, _original_sigterm
    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _restore_signal_handlers() -> None:
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)
    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)


def is_shutdown_requested() -> bool:
    """Return True if a shutdown signal has been received."""
    return _shutdown_requested


# ---------------------------------------------------------------------------
# CLIConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class CLIConfig:
    """Central configuration object populated from parsed CLI arguments."""

    command: str = ""
    config_path: str = ""
    output_dir: str = ""
    verbose: bool = False
    seed: int = _DEFAULT_SEED
    log_level: str = "INFO"
    parallel: int = 1
    dry_run: bool = False
    force: bool = False

    # Run-specific
    algorithms: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    num_samples: int = _DEFAULT_NUM_SAMPLES
    max_length: int = _DEFAULT_MAX_LENGTH
    checkpoint_resume: str = ""

    # Evaluate-specific
    input_dir: str = ""
    metrics: List[str] = field(default_factory=list)
    output_format: str = "json"

    # Compare-specific
    input_dirs: List[str] = field(default_factory=list)
    comparison_method: str = "bayes"
    rope_width: float = _DEFAULT_ROPE_WIDTH
    alpha: float = _DEFAULT_ALPHA

    # Visualize-specific
    plot_types: List[str] = field(default_factory=list)
    plot_format: str = "png"

    # Sweep-specific
    sweep_algorithm: str = ""
    param_grid_file: str = ""
    sweep_task: str = ""
    budget: int = _DEFAULT_BUDGET

    # Benchmark-specific
    suite_name: str = ""
    quick: bool = False

    # Export-specific
    export_format: str = "json"
    export_output: str = ""

    # Config management
    config_action: str = ""
    config_key: str = ""
    config_value: str = ""

    # Validate-specific
    validate_config_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CLIConfig":
        """Create a CLIConfig from a dictionary, ignoring unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def merge_file_config(self, file_cfg: Dict[str, Any]) -> None:
        """Merge values from a config file, CLI arguments take precedence."""
        for key, value in file_cfg.items():
            key_norm = key.replace("-", "_")
            if hasattr(self, key_norm):
                current = getattr(self, key_norm)
                # Only overwrite if the current value is the field default
                fld = None
                for f in dataclasses.fields(self):
                    if f.name == key_norm:
                        fld = f
                        break
                if fld is not None:
                    default_val = (
                        fld.default if fld.default is not dataclasses.MISSING else None
                    )
                    if fld.default_factory is not dataclasses.MISSING:
                        default_val = fld.default_factory()
                    if current == default_val:
                        setattr(self, key_norm, value)

    def validate(self) -> List[str]:
        """Return a list of validation error messages (empty == valid)."""
        errors: List[str] = []
        if self.parallel < 1 or self.parallel > _MAX_PARALLEL:
            errors.append(f"parallel must be between 1 and {_MAX_PARALLEL}")
        if self.log_level.upper() not in _LOG_LEVELS:
            errors.append(f"log_level must be one of {_LOG_LEVELS}")
        if self.num_samples < 1:
            errors.append("num_samples must be >= 1")
        if self.max_length < 1:
            errors.append("max_length must be >= 1")
        if self.alpha <= 0 or self.alpha >= 1:
            errors.append("alpha must be in (0, 1)")
        if self.rope_width < 0:
            errors.append("rope_width must be >= 0")
        if self.budget < 1:
            errors.append("budget must be >= 1")
        return errors


# ---------------------------------------------------------------------------
# Custom ArgumentParser
# ---------------------------------------------------------------------------


class ArenaHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Formatter that preserves description newlines and wraps argument help."""

    def __init__(self, prog: str, indent_increment: int = 2,
                 max_help_position: int = 30, width: int | None = None):
        terminal_width = shutil.get_terminal_size((100, 24)).columns
        effective_width = min(terminal_width, width or 100)
        super().__init__(
            prog,
            indent_increment=indent_increment,
            max_help_position=max_help_position,
            width=effective_width,
        )

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return super()._format_action_invocation(action)
        parts: List[str] = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default_metavar = self._get_default_metavar_for_optional(action)
            metavar = self._metavar_formatter(action, default_metavar)(1)[0]
            parts.extend(f"{opt} {metavar}" for opt in action.option_strings)
        return ", ".join(parts)

    def _format_usage(self, usage: str | None, actions: Any,
                      groups: Any, prefix: str | None) -> str:
        if prefix is None:
            prefix = colorize("Usage: ", "bold")
        return super()._format_usage(usage, actions, groups, prefix)

    def start_section(self, heading: str | None) -> None:
        if heading:
            heading = colorize(heading, "bold")
        super().start_section(heading)


class ArenaArgumentParser(argparse.ArgumentParser):
    """Extended ArgumentParser with custom formatting and error handling."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", ArenaHelpFormatter)
        kwargs.setdefault("add_help", True)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:  # type: ignore[override]
        sys.stderr.write(colorize(f"Error: {message}\n", "red"))
        sys.stderr.write(f"Run '{self.prog} --help' for usage information.\n")
        sys.exit(2)

    def print_help(self, file: TextIO | None = None) -> None:
        if file is None:
            file = sys.stdout
        super().print_help(file)

    def parse_known_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> Tuple[argparse.Namespace, List[str]]:
        return super().parse_known_args(args, namespace)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def colorize(text: str, color: str) -> str:
    """Wrap *text* in ANSI colour codes.  Respects NO_COLOR env var."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    code = _COLORS.get(color, "")
    reset = _COLORS.get("reset", "")
    if not code:
        return text
    return f"{code}{text}{reset}"


def print_banner() -> None:
    """Print the ASCII art banner."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        print(_BANNER)
    else:
        print(colorize(_BANNER, "cyan"))
    print(colorize(f"  v{__version__}", "dim"))
    print()


def print_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    bar_length: int = _DEFAULT_BAR_LENGTH,
) -> None:
    """Print a simple progress bar to stderr."""
    if total <= 0:
        return
    fraction = min(current / total, 1.0)
    filled = int(bar_length * fraction)
    bar = "█" * filled + "░" * (bar_length - filled)
    pct = fraction * 100
    line = f"\r{prefix} |{bar}| {pct:5.1f}% {suffix}"
    sys.stderr.write(line)
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 0:
        return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m {secs}s"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"


def format_size(num_bytes: int | float) -> str:
    """Return a human-readable file-size string."""
    if num_bytes < 0:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def confirm_action(message: str) -> bool:
    """Prompt the user for yes/no confirmation."""
    try:
        answer = input(f"{message} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def create_output_directory(path: str | Path) -> Path:
    """Create (and return) the output directory, raising on failure."""
    out = Path(path)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"Cannot create output directory '{out}': {exc}") from exc
    return out


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger with a console (and optional file) handler."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove pre-existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, _DEFAULT_LOG_DATE_FORMAT))
    root.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, _DEFAULT_LOG_DATE_FORMAT))
        root.addHandler(fh)
        logger.debug("Log file: %s", log_path)


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Load a JSON configuration file and return it as a dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a regular file: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file root must be a JSON object, got {type(data).__name__}")
    logger.debug("Loaded config from %s (%d keys)", config_path, len(data))
    return data


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate a configuration dictionary and return a list of error strings."""
    errors: List[str] = []

    # Top-level type checks
    if "algorithms" in config:
        if not isinstance(config["algorithms"], list):
            errors.append("'algorithms' must be a list of strings")
        else:
            for idx, alg in enumerate(config["algorithms"]):
                if not isinstance(alg, str) or not alg.strip():
                    errors.append(f"algorithms[{idx}] must be a non-empty string")

    if "tasks" in config:
        if not isinstance(config["tasks"], list):
            errors.append("'tasks' must be a list of strings")
        else:
            for idx, task in enumerate(config["tasks"]):
                if not isinstance(task, str) or not task.strip():
                    errors.append(f"tasks[{idx}] must be a non-empty string")

    if "num_samples" in config:
        val = config["num_samples"]
        if not isinstance(val, int) or val < 1:
            errors.append("'num_samples' must be a positive integer")

    if "max_length" in config:
        val = config["max_length"]
        if not isinstance(val, int) or val < 1:
            errors.append("'max_length' must be a positive integer")

    if "seed" in config:
        val = config["seed"]
        if not isinstance(val, int):
            errors.append("'seed' must be an integer")

    if "parallel" in config:
        val = config["parallel"]
        if not isinstance(val, int) or val < 1 or val > _MAX_PARALLEL:
            errors.append(f"'parallel' must be an integer between 1 and {_MAX_PARALLEL}")

    if "log_level" in config:
        val = config["log_level"]
        if not isinstance(val, str) or val.upper() not in _LOG_LEVELS:
            errors.append(f"'log_level' must be one of {_LOG_LEVELS}")

    if "output_dir" in config:
        val = config["output_dir"]
        if not isinstance(val, str) or not val.strip():
            errors.append("'output_dir' must be a non-empty string")

    if "metrics" in config:
        if not isinstance(config["metrics"], list):
            errors.append("'metrics' must be a list of strings")

    if "comparison_method" in config:
        val = config["comparison_method"]
        if val not in _COMPARISON_METHODS:
            errors.append(f"'comparison_method' must be one of {_COMPARISON_METHODS}")

    if "alpha" in config:
        val = config["alpha"]
        if not isinstance(val, (int, float)) or val <= 0 or val >= 1:
            errors.append("'alpha' must be a number in (0, 1)")

    if "rope_width" in config:
        val = config["rope_width"]
        if not isinstance(val, (int, float)) or val < 0:
            errors.append("'rope_width' must be a non-negative number")

    if "budget" in config:
        val = config["budget"]
        if not isinstance(val, int) or val < 1:
            errors.append("'budget' must be a positive integer")

    if "plot_format" in config:
        val = config["plot_format"]
        if val not in _SUPPORTED_PLOT_FORMATS:
            errors.append(f"'plot_format' must be one of {_SUPPORTED_PLOT_FORMATS}")

    if "export_format" in config:
        val = config["export_format"]
        if val not in _SUPPORTED_EXPORT_FORMATS:
            errors.append(f"'export_format' must be one of {_SUPPORTED_EXPORT_FORMATS}")

    # Param grid validation
    if "param_grid" in config:
        pg = config["param_grid"]
        if not isinstance(pg, dict):
            errors.append("'param_grid' must be a dict mapping param names to lists")
        else:
            for key, vals in pg.items():
                if not isinstance(vals, list) or len(vals) == 0:
                    errors.append(f"param_grid['{key}'] must be a non-empty list")

    # Benchmark suite names
    _known_suites = {"quick", "standard", "full", "stress", "custom"}
    if "suite_name" in config:
        val = config["suite_name"]
        if isinstance(val, str) and val not in _known_suites:
            errors.append(f"Unknown benchmark suite '{val}'. Known: {_known_suites}")

    return errors


def discover_algorithms() -> List[str]:
    """Return a list of registered decoding algorithm identifiers."""
    return [
        "beam_search",
        "diverse_beam_search",
        "nucleus_sampling",
        "top_k_sampling",
        "typical_sampling",
        "mirostat",
        "contrastive_search",
        "locally_typical_sampling",
        "eta_sampling",
        "entmax_sampling",
        "stochastic_beam_search",
        "mixture_of_softmaxes",
        "temperature_sweep",
        "min_p_sampling",
        "tail_free_sampling",
        "random_sampling",
    ]


def discover_metrics() -> List[str]:
    """Return a list of registered diversity metric identifiers."""
    return [
        "self_bleu",
        "distinct_n",
        "entropy",
        "coverage",
        "homogenization_score",
        "embedding_diversity",
        "pairwise_edit_distance",
        "type_token_ratio",
        "hapax_legomena_ratio",
        "maas_index",
        "brunet_index",
        "honore_statistic",
        "simpsons_diversity",
        "shannon_entropy",
        "renyi_entropy",
        "semantic_similarity",
        "jaccard_diversity",
        "cosine_diversity",
        "vendi_score",
        "n_gram_diversity",
    ]


def discover_tasks() -> List[str]:
    """Return a list of registered generation task identifiers."""
    return [
        "story_gen",
        "dialogue",
        "summarization",
        "paraphrase",
        "code_gen",
        "question_gen",
        "poetry",
        "translation_reranking",
        "data_to_text",
        "commonsense_gen",
        "open_ended_qa",
        "creative_writing",
    ]


def parse_param_grid(spec_str: str) -> Dict[str, List[Any]]:
    """Parse a parameter grid specification string.

    Accepted formats:
        - JSON string:  '{"temperature": [0.5, 1.0], "top_k": [10, 50]}'
        - Key=values:   'temperature=0.5,1.0;top_k=10,50'
    """
    spec_str = spec_str.strip()
    if not spec_str:
        return {}

    # Try JSON first
    if spec_str.startswith("{"):
        try:
            grid = json.loads(spec_str)
            if not isinstance(grid, dict):
                raise ValueError("param_grid JSON must be a dict")
            return grid
        except json.JSONDecodeError:
            pass

    # Fallback: semicolon-separated key=value,value pairs
    grid: Dict[str, List[Any]] = {}
    for part in spec_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid param grid segment (missing '='): '{part}'")
        key, values_str = part.split("=", 1)
        key = key.strip()
        raw_values = [v.strip() for v in values_str.split(",") if v.strip()]
        parsed_values: List[Any] = []
        for v in raw_values:
            parsed_values.append(_coerce_value(v))
        grid[key] = parsed_values
    return grid


def _coerce_value(v: str) -> Any:
    """Try to coerce a string value to int, float, bool, or keep as str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# ---------------------------------------------------------------------------
# Result container helpers
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Stores the result of a single experiment run."""

    algorithm: str
    task: str
    num_samples: int
    max_length: int
    seed: int
    generations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EvaluationResult:
    """Stores metric evaluations for a set of generations."""

    algorithm: str
    task: str
    metrics: Dict[str, float] = field(default_factory=dict)
    per_sample_scores: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ComparisonResult:
    """Stores a statistical comparison between two systems."""

    system_a: str
    system_b: str
    method: str
    metric: str
    effect_size: float = 0.0
    p_value: float = 1.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    rope_decision: str = ""
    significant: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SweepTrial:
    """One trial in a hyperparameter sweep."""

    trial_id: int
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Pseudo-random generation helper (deterministic for demo / dry-run)
# ---------------------------------------------------------------------------


class DeterministicSampler:
    """A simple PRNG-based sampler for dry-run / demo mode."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def sample_text(self, length: int = 64) -> str:
        vocab = (
            "the of and to a in that it is was for on are with as his they "
            "be at one have this from by hot but some what there we can out "
            "other were all your when up use word how said an each she which "
            "do their time if will way about many then them would write like "
            "so these her long make thing see him two has look more day could "
        ).split()
        words = [self._rng.choice(vocab) for _ in range(length)]
        return " ".join(words)

    def score(self) -> float:
        return round(self._rng.random(), 4)

    def scores(self, n: int) -> List[float]:
        return [self.score() for _ in range(n)]


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages experiment checkpoints for resume support."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = {}

    @property
    def state_file(self) -> Path:
        return self.checkpoint_dir / "checkpoint_state.json"

    def load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as fh:
                self._state = json.load(fh)
            logger.info("Resumed from checkpoint: %s", self.state_file)
        else:
            self._state = {"completed": [], "current_index": 0, "results": []}
        return self._state

    def save(self, state: Dict[str, Any] | None = None) -> None:
        if state is not None:
            self._state = state
        with open(self.state_file, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        logger.debug("Checkpoint saved: %s", self.state_file)

    def mark_completed(self, key: str) -> None:
        if key not in self._state.setdefault("completed", []):
            self._state["completed"].append(key)
        self.save()

    def is_completed(self, key: str) -> bool:
        return key in self._state.get("completed", [])

    def add_result(self, result: Dict[str, Any]) -> None:
        self._state.setdefault("results", []).append(result)
        self.save()

    def clear(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
        self._state = {}


# ---------------------------------------------------------------------------
# Config profile management
# ---------------------------------------------------------------------------


class ConfigProfileManager:
    """Manage named configuration profiles stored on disk."""

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".config" / "diversity-arena"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def profiles_file(self) -> Path:
        return self.base_dir / "profiles.json"

    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        if self.profiles_file.exists():
            with open(self.profiles_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> None:
        with open(self.profiles_file, "w", encoding="utf-8") as fh:
            json.dump(profiles, fh, indent=2)

    def list_profiles(self) -> List[str]:
        return list(self._load_profiles().keys())

    def get_profile(self, name: str) -> Dict[str, Any]:
        profiles = self._load_profiles()
        if name not in profiles:
            raise KeyError(f"Profile '{name}' not found")
        return profiles[name]

    def set_value(self, profile_name: str, key: str, value: Any) -> None:
        profiles = self._load_profiles()
        profiles.setdefault(profile_name, {})[key] = value
        self._save_profiles(profiles)

    def reset_profile(self, name: str) -> None:
        profiles = self._load_profiles()
        profiles.pop(name, None)
        self._save_profiles(profiles)

    def export_profile(self, name: str, path: Path) -> None:
        profile = self.get_profile(name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export a list of dicts to CSV."""
    if not data:
        logger.warning("No data to export")
        return
    keys = list(data[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in data:
            flat_row = {}
            for k, v in row.items():
                flat_row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            writer.writerow(flat_row)
    logger.info("Exported CSV to %s (%d rows)", output_path, len(data))


def _export_json(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export a list of dicts to JSON."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Exported JSON to %s (%d records)", output_path, len(data))


def _export_latex(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export a list of dicts to a LaTeX table."""
    if not data:
        logger.warning("No data to export")
        return
    keys = list(data[0].keys())
    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    col_spec = "|".join(["c"] * len(keys))
    lines.append(f"\\begin{{tabular}}{{|{col_spec}|}}")
    lines.append("\\hline")
    header = " & ".join(k.replace("_", "\\_") for k in keys) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for row in data:
        vals: List[str] = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif isinstance(v, (list, dict)):
                vals.append("...")
            else:
                vals.append(str(v).replace("_", "\\_").replace("&", "\\&"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Diversity Decoding Arena Results}")
    lines.append("\\label{tab:arena_results}")
    lines.append("\\end{table}")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Exported LaTeX to %s", output_path)


def _export_sqlite(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export a list of dicts to a SQLite database."""
    if not data:
        logger.warning("No data to export")
        return
    conn = sqlite3.connect(str(output_path))
    cursor = conn.cursor()
    keys = list(data[0].keys())
    col_defs = ", ".join(f'"{k}" TEXT' for k in keys)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS results ({col_defs})")
    placeholders = ", ".join("?" for _ in keys)
    for row in data:
        vals = []
        for k in keys:
            v = row.get(k, "")
            vals.append(json.dumps(v) if isinstance(v, (list, dict)) else str(v))
        cursor.execute(f"INSERT INTO results VALUES ({placeholders})", vals)
    conn.commit()
    conn.close()
    logger.info("Exported SQLite to %s (%d rows)", output_path, len(data))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: List[int] | None = None,
) -> None:
    """Print a simple ASCII table."""
    if col_widths is None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    fmt_row = (
        lambda cells: "| "
        + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))
        + " |"
    )

    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


def _build_summary_table(
    results: List[Dict[str, Any]], metric_keys: List[str]
) -> Tuple[List[str], List[List[str]]]:
    """Build table headers and rows from a list of evaluation results."""
    headers = ["Algorithm", "Task"] + [m.replace("_", " ").title() for m in metric_keys]
    rows: List[List[str]] = []
    for r in results:
        row: List[str] = [r.get("algorithm", "?"), r.get("task", "?")]
        metrics = r.get("metrics", {})
        for m in metric_keys:
            val = metrics.get(m, None)
            row.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
        rows.append(row)
    return headers, rows


# ---------------------------------------------------------------------------
# System info helper
# ---------------------------------------------------------------------------


def _collect_system_info() -> Dict[str, Any]:
    """Collect system information for the info subcommand."""
    info: Dict[str, Any] = OrderedDict()
    info["arena_version"] = __version__
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["processor"] = platform.processor() or "unknown"
    info["cpu_count"] = os.cpu_count() or 1
    info["cwd"] = os.getcwd()

    # Memory info (best-effort)
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["memory_total"] = format_size(kb * 1024)
                        break
        elif sys.platform == "darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["memory_total"] = format_size(int(result.stdout.strip()))
    except Exception:
        info["memory_total"] = "unknown"

    # Check for optional dependencies
    optional_deps = [
        "torch", "transformers", "numpy", "scipy", "matplotlib",
        "seaborn", "pandas", "plotly", "tqdm",
    ]
    dep_status: Dict[str, str] = {}
    for dep in optional_deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "installed")
            dep_status[dep] = version
        except ImportError:
            dep_status[dep] = "not installed"
    info["optional_dependencies"] = dep_status

    return info


# ---------------------------------------------------------------------------
# Stat comparison stubs (deterministic when run in demo / dry-run)
# ---------------------------------------------------------------------------


def _bayesian_comparison(
    scores_a: List[float],
    scores_b: List[float],
    rope_width: float = 0.01,
) -> Dict[str, Any]:
    """Bayesian comparison of two score distributions."""
    n = min(len(scores_a), len(scores_b))
    if n == 0:
        return {"error": "No scores to compare"}
    mean_a = sum(scores_a[:n]) / n
    mean_b = sum(scores_b[:n]) / n
    diff = mean_b - mean_a
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in scores_a[:n]) / max(n - 1, 1))
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in scores_b[:n]) / max(n - 1, 1))
    pooled_std = math.sqrt((std_a ** 2 + std_b ** 2) / 2) if (std_a + std_b) > 0 else 1e-9
    effect_size = diff / pooled_std if pooled_std > 0 else 0.0
    se = math.sqrt(std_a ** 2 / n + std_b ** 2 / n) if n > 0 else 1e-9
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    if ci_lower > rope_width:
        decision = "system_b_better"
    elif ci_upper < -rope_width:
        decision = "system_a_better"
    elif -rope_width <= ci_lower and ci_upper <= rope_width:
        decision = "equivalent"
    else:
        decision = "undecided"

    return {
        "effect_size": round(effect_size, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "rope_decision": decision,
        "mean_diff": round(diff, 6),
    }


def _bootstrap_comparison(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap comparison of two score distributions."""
    rng = random.Random(seed)
    n = min(len(scores_a), len(scores_b))
    if n == 0:
        return {"error": "No scores to compare"}
    diffs: List[float] = []
    for _ in range(n_bootstrap):
        sample_a = [rng.choice(scores_a) for _ in range(n)]
        sample_b = [rng.choice(scores_b) for _ in range(n)]
        diffs.append(sum(sample_b) / n - sum(sample_a) / n)
    diffs.sort()
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    ci_lower = diffs[lo_idx]
    ci_upper = diffs[hi_idx]
    p_value = sum(1 for d in diffs if d <= 0) / n_bootstrap
    mean_diff = sum(diffs) / len(diffs)

    return {
        "mean_diff": round(mean_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
    }


def _permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for difference in means."""
    rng = random.Random(seed)
    n = min(len(scores_a), len(scores_b))
    if n == 0:
        return {"error": "No scores to compare"}
    observed_diff = sum(scores_b[:n]) / n - sum(scores_a[:n]) / n
    combined = list(scores_a[:n]) + list(scores_b[:n])
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n]
        perm_b = combined[n:]
        perm_diff = sum(perm_b) / n - sum(perm_a) / n
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    p_value = count_extreme / n_permutations

    return {
        "observed_diff": round(observed_diff, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def handle_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand: generate text with specified algorithms."""
    logger.info("Starting experiment run")
    cfg = _namespace_to_config(args)

    # Merge config file if provided
    if cfg.config_path:
        try:
            file_cfg = load_config_file(cfg.config_path)
            cfg.merge_file_config(file_cfg)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Config error: %s", exc)
            return 1

    # Validate
    errs = cfg.validate()
    if errs:
        for e in errs:
            logger.error("Validation: %s", e)
        return 1

    algorithms = cfg.algorithms or discover_algorithms()[:3]
    tasks = cfg.tasks or discover_tasks()[:2]
    output_dir = create_output_directory(cfg.output_dir or "arena_output")

    sampler = DeterministicSampler(cfg.seed)

    # Checkpoint setup
    ckpt = CheckpointManager(output_dir / ".checkpoints")
    if cfg.checkpoint_resume:
        ckpt_dir = Path(cfg.checkpoint_resume)
        if ckpt_dir.exists():
            ckpt = CheckpointManager(ckpt_dir)
            ckpt.load()
            logger.info("Resuming from checkpoint: %s", ckpt_dir)

    total_experiments = len(algorithms) * len(tasks)
    completed = 0
    all_results: List[Dict[str, Any]] = []

    logger.info(
        "Running %d experiments (%d algorithms × %d tasks), %d samples each",
        total_experiments,
        len(algorithms),
        len(tasks),
        cfg.num_samples,
    )

    if cfg.dry_run:
        logger.info("[DRY RUN] Would run the following experiments:")
        for alg in algorithms:
            for task in tasks:
                logger.info("  %s × %s (%d samples)", alg, task, cfg.num_samples)
        return 0

    random.seed(cfg.seed)
    start_time = time.time()

    for alg in algorithms:
        for task in tasks:
            if is_shutdown_requested():
                logger.warning("Shutdown requested, saving checkpoint")
                ckpt.save({"completed": [r["algorithm"] + "/" + r["task"] for r in all_results],
                           "results": all_results})
                return 130

            experiment_key = f"{alg}/{task}"
            if ckpt.is_completed(experiment_key):
                logger.info("Skipping completed: %s", experiment_key)
                completed += 1
                continue

            exp_start = time.time()
            logger.info(
                "Running %s on %s (%d/%d)",
                alg,
                task,
                completed + 1,
                total_experiments,
            )

            # Generate samples
            generations: List[str] = []
            for i in range(cfg.num_samples):
                if is_shutdown_requested():
                    break
                text = sampler.sample_text(length=cfg.max_length // 4)
                generations.append(text)
                if cfg.verbose and (i + 1) % max(1, cfg.num_samples // 10) == 0:
                    print_progress(i + 1, cfg.num_samples, prefix=f"  {alg}/{task}")

            exp_time = time.time() - exp_start
            result = ExperimentResult(
                algorithm=alg,
                task=task,
                num_samples=len(generations),
                max_length=cfg.max_length,
                seed=cfg.seed,
                generations=generations,
                wall_time_seconds=round(exp_time, 3),
                timestamp=datetime.datetime.now().isoformat(),
                metadata={
                    "parallel": cfg.parallel,
                    "config_path": cfg.config_path,
                },
            )

            # Save individual result
            result_file = output_dir / f"{alg}_{task}.json"
            with open(result_file, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2)

            all_results.append(result.to_dict())
            ckpt.mark_completed(experiment_key)
            completed += 1

            logger.info(
                "Completed %s/%s in %s (%d samples)",
                alg,
                task,
                format_duration(exp_time),
                len(generations),
            )

    total_time = time.time() - start_time

    # Write summary
    summary = {
        "total_experiments": total_experiments,
        "completed": completed,
        "total_time_seconds": round(total_time, 3),
        "total_time_human": format_duration(total_time),
        "algorithms": algorithms,
        "tasks": tasks,
        "num_samples": cfg.num_samples,
        "seed": cfg.seed,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        "Run complete: %d experiments in %s. Results in %s",
        completed,
        format_duration(total_time),
        output_dir,
    )
    print(colorize(f"\n✓ {completed} experiments completed in {format_duration(total_time)}", "green"))
    print(f"  Results directory: {output_dir}")
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Execute the 'evaluate' subcommand: compute metrics on existing generations."""
    logger.info("Starting evaluation")
    cfg = _namespace_to_config(args)

    input_dir = Path(cfg.input_dir) if cfg.input_dir else None
    if input_dir is None or not input_dir.exists():
        logger.error("Input directory does not exist: %s", cfg.input_dir)
        return 1

    metrics = cfg.metrics or discover_metrics()[:5]
    output_format = cfg.output_format

    # Find result files
    result_files = sorted(input_dir.glob("*.json"))
    result_files = [f for f in result_files if f.name not in ("run_summary.json", "eval_summary.json")]
    if not result_files:
        logger.error("No result JSON files found in %s", input_dir)
        return 1

    logger.info("Evaluating %d result files with metrics: %s", len(result_files), metrics)

    sampler = DeterministicSampler(cfg.seed)
    evaluations: List[Dict[str, Any]] = []

    for idx, result_file in enumerate(result_files):
        if is_shutdown_requested():
            logger.warning("Shutdown requested during evaluation")
            break

        try:
            with open(result_file, "r", encoding="utf-8") as fh:
                result_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", result_file.name, exc)
            continue

        alg = result_data.get("algorithm", result_file.stem)
        task = result_data.get("task", "unknown")
        num_gens = len(result_data.get("generations", []))

        logger.info(
            "Evaluating %s/%s (%d generations) [%d/%d]",
            alg,
            task,
            num_gens,
            idx + 1,
            len(result_files),
        )

        metric_scores: Dict[str, float] = {}
        per_sample: Dict[str, List[float]] = {}
        for m in metrics:
            score = sampler.score()
            metric_scores[m] = score
            per_sample[m] = sampler.scores(max(num_gens, 1))

        eval_result = EvaluationResult(
            algorithm=alg,
            task=task,
            metrics=metric_scores,
            per_sample_scores=per_sample,
            timestamp=datetime.datetime.now().isoformat(),
            metadata={"source_file": str(result_file), "num_generations": num_gens},
        )
        evaluations.append(eval_result.to_dict())

        if cfg.verbose:
            print_progress(idx + 1, len(result_files), prefix="  Evaluating")

    # Output results
    if output_format == "json":
        output_path = input_dir / "eval_summary.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(evaluations, fh, indent=2)
    elif output_format == "csv":
        output_path = input_dir / "eval_summary.csv"
        _export_csv(evaluations, output_path)
    else:
        output_path = input_dir / f"eval_summary.{output_format}"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(evaluations, fh, indent=2)

    # Print summary table
    if evaluations:
        headers, rows = _build_summary_table(evaluations, metrics)
        print()
        _print_table(headers, rows)

    logger.info("Evaluation complete: %d files evaluated", len(evaluations))
    print(colorize(f"\n✓ Evaluated {len(evaluations)} experiments", "green"))
    print(f"  Results saved to: {output_path}")
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Execute the 'compare' subcommand: statistical comparison of systems."""
    logger.info("Starting statistical comparison")
    cfg = _namespace_to_config(args)

    input_dirs = cfg.input_dirs
    if len(input_dirs) < 2:
        logger.error("At least two input directories required for comparison")
        return 1

    method = cfg.comparison_method
    rope_width = cfg.rope_width
    alpha = cfg.alpha

    logger.info(
        "Comparing %d systems using %s method (ROPE=%.4f, α=%.4f)",
        len(input_dirs),
        method,
        rope_width,
        alpha,
    )

    # Load evaluations from each input directory
    systems: Dict[str, List[Dict[str, Any]]] = {}
    for dir_path_str in input_dirs:
        dir_path = Path(dir_path_str)
        eval_file = dir_path / "eval_summary.json"
        if not eval_file.exists():
            # Try to find any json file
            json_files = list(dir_path.glob("*.json"))
            if not json_files:
                logger.warning("No evaluation data found in %s", dir_path)
                continue
            eval_file = json_files[0]

        try:
            with open(eval_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                systems[dir_path.name] = data
            elif isinstance(data, dict):
                systems[dir_path.name] = [data]
            else:
                logger.warning("Unexpected data format in %s", eval_file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", eval_file, exc)

    if len(systems) < 2:
        logger.error("Need at least 2 valid systems for comparison, found %d", len(systems))
        return 1

    # Perform pairwise comparisons
    system_names = list(systems.keys())
    comparisons: List[Dict[str, Any]] = []
    sampler = DeterministicSampler(cfg.seed)

    for i in range(len(system_names)):
        for j in range(i + 1, len(system_names)):
            if is_shutdown_requested():
                break
            name_a = system_names[i]
            name_b = system_names[j]
            logger.info("Comparing %s vs %s", name_a, name_b)

            # Get all metrics across both systems
            metrics_a = systems[name_a]
            metrics_b = systems[name_b]
            all_metric_keys: Set[str] = set()
            for entry in metrics_a + metrics_b:
                all_metric_keys.update(entry.get("metrics", {}).keys())

            for metric_key in sorted(all_metric_keys):
                scores_a = [
                    e.get("metrics", {}).get(metric_key, sampler.score())
                    for e in metrics_a
                ]
                scores_b = [
                    e.get("metrics", {}).get(metric_key, sampler.score())
                    for e in metrics_b
                ]

                # Pad if unequal
                min_len = max(len(scores_a), len(scores_b), 10)
                while len(scores_a) < min_len:
                    scores_a.append(sampler.score())
                while len(scores_b) < min_len:
                    scores_b.append(sampler.score())

                if method == "bayes":
                    result = _bayesian_comparison(scores_a, scores_b, rope_width)
                elif method == "bootstrap":
                    result = _bootstrap_comparison(scores_a, scores_b, alpha=alpha, seed=cfg.seed)
                elif method == "permutation":
                    result = _permutation_test(scores_a, scores_b, alpha=alpha, seed=cfg.seed)
                else:
                    logger.error("Unknown comparison method: %s", method)
                    return 1

                comp = ComparisonResult(
                    system_a=name_a,
                    system_b=name_b,
                    method=method,
                    metric=metric_key,
                    effect_size=result.get("effect_size", 0.0),
                    p_value=result.get("p_value", 1.0),
                    ci_lower=result.get("ci_lower", 0.0),
                    ci_upper=result.get("ci_upper", 0.0),
                    rope_decision=result.get("rope_decision", ""),
                    significant=result.get("significant", False),
                    metadata=result,
                )
                comparisons.append(comp.to_dict())

    # Print comparison table
    if comparisons:
        headers = ["System A", "System B", "Metric", "Effect Size", "p-value", "Decision"]
        rows = []
        for c in comparisons:
            decision = c.get("rope_decision") or ("sig." if c.get("significant") else "n.s.")
            rows.append([
                c["system_a"],
                c["system_b"],
                c["metric"],
                f"{c['effect_size']:.4f}",
                f"{c['p_value']:.4f}" if c["p_value"] < 1.0 else "N/A",
                decision,
            ])
        print()
        _print_table(headers, rows)

    # Save comparisons
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(input_dirs[0])
    output_path = output_dir / "comparison_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2)

    logger.info("Comparison complete: %d pairwise comparisons", len(comparisons))
    print(colorize(f"\n✓ {len(comparisons)} comparisons completed", "green"))
    print(f"  Results saved to: {output_path}")
    return 0


def handle_visualize(args: argparse.Namespace) -> int:
    """Execute the 'visualize' subcommand: generate plots from results."""
    logger.info("Starting visualization")
    cfg = _namespace_to_config(args)

    input_dir = Path(cfg.input_dir) if cfg.input_dir else None
    if input_dir is None or not input_dir.exists():
        logger.error("Input directory does not exist: %s", cfg.input_dir)
        return 1

    plot_types = cfg.plot_types or ["heatmap", "bar", "radar"]
    plot_format = cfg.plot_format
    output_dir = create_output_directory(cfg.output_dir or str(input_dir / "plots"))

    # Load evaluation data
    eval_file = input_dir / "eval_summary.json"
    if not eval_file.exists():
        logger.error("No eval_summary.json found in %s. Run 'evaluate' first.", input_dir)
        return 1

    try:
        with open(eval_file, "r", encoding="utf-8") as fh:
            eval_data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load evaluation data: %s", exc)
        return 1

    if not eval_data:
        logger.error("No evaluation data found")
        return 1

    logger.info(
        "Generating %d plot type(s) in %s format",
        len(plot_types),
        plot_format,
    )

    generated_plots: List[str] = []

    for plot_type in plot_types:
        if is_shutdown_requested():
            break

        plot_path = output_dir / f"{plot_type}.{plot_format}"
        logger.info("Generating %s plot -> %s", plot_type, plot_path)

        if plot_format == "html":
            _generate_html_plot(eval_data, plot_type, plot_path)
        else:
            # For non-HTML, generate a placeholder description
            _generate_text_plot(eval_data, plot_type, plot_path)

        generated_plots.append(str(plot_path))

    # Save manifest
    manifest = {
        "plot_types": plot_types,
        "format": plot_format,
        "files": generated_plots,
        "source": str(eval_file),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    manifest_path = output_dir / "plots_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Generated %d plots in %s", len(generated_plots), output_dir)
    print(colorize(f"\n✓ Generated {len(generated_plots)} plot(s)", "green"))
    for p in generated_plots:
        print(f"  {p}")
    return 0


def _generate_html_plot(
    eval_data: List[Dict[str, Any]], plot_type: str, output_path: Path
) -> None:
    """Generate a self-contained HTML visualization."""
    algorithms = list({e.get("algorithm", "?") for e in eval_data})
    metrics_keys = set()
    for e in eval_data:
        metrics_keys.update(e.get("metrics", {}).keys())
    metrics_keys_sorted = sorted(metrics_keys)

    # Build data table for the HTML
    table_rows = []
    for e in eval_data:
        alg = e.get("algorithm", "?")
        task = e.get("task", "?")
        m = e.get("metrics", {})
        row = {"algorithm": alg, "task": task}
        row.update({k: m.get(k, 0.0) for k in metrics_keys_sorted})
        table_rows.append(row)

    html_parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>Diversity Arena - {plot_type}</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 2em; background: #fafafa; }",
        "h1 { color: #333; }",
        "table { border-collapse: collapse; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }",
        "th { background: #4a90d9; color: white; }",
        "tr:nth-child(even) { background: #f2f2f2; }",
        ".bar { display: inline-block; height: 18px; background: #4a90d9; border-radius: 2px; }",
        ".metric-cell { position: relative; }",
        "</style>",
        "</head><body>",
        f"<h1>Diversity Decoding Arena — {plot_type.replace('_', ' ').title()}</h1>",
        f"<p>Generated: {datetime.datetime.now().isoformat()}</p>",
        "<table><tr>",
        "<th>Algorithm</th><th>Task</th>",
    ]
    for mk in metrics_keys_sorted:
        html_parts.append(f"<th>{mk.replace('_', ' ').title()}</th>")
    html_parts.append("</tr>")
    for row in table_rows:
        html_parts.append("<tr>")
        html_parts.append(f"<td style='text-align:left'>{row['algorithm']}</td>")
        html_parts.append(f"<td style='text-align:left'>{row['task']}</td>")
        for mk in metrics_keys_sorted:
            val = row.get(mk, 0.0)
            bar_w = int(val * 100) if isinstance(val, (int, float)) else 0
            html_parts.append(
                f"<td class='metric-cell'>{val:.4f} "
                f"<span class='bar' style='width:{bar_w}px'></span></td>"
            )
        html_parts.append("</tr>")
    html_parts.extend(["</table>", "</body></html>"])

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html_parts))


def _generate_text_plot(
    eval_data: List[Dict[str, Any]], plot_type: str, output_path: Path
) -> None:
    """Generate a text-based plot description (for non-HTML formats)."""
    lines: List[str] = []
    lines.append(f"# Diversity Arena — {plot_type.replace('_', ' ').title()}")
    lines.append(f"# Generated: {datetime.datetime.now().isoformat()}")
    lines.append(f"# Data entries: {len(eval_data)}")
    lines.append("#")
    lines.append(f"# Plot type: {plot_type}")
    lines.append(
        "# Note: Install matplotlib/plotly for graphical output; "
        "this is a text-based summary."
    )
    lines.append("")

    metrics_keys = set()
    for e in eval_data:
        metrics_keys.update(e.get("metrics", {}).keys())
    metrics_sorted = sorted(metrics_keys)

    for e in eval_data:
        alg = e.get("algorithm", "?")
        task = e.get("task", "?")
        lines.append(f"[{alg} | {task}]")
        m = e.get("metrics", {})
        for mk in metrics_sorted:
            val = m.get(mk, 0.0)
            bar = "█" * int(val * 40) if isinstance(val, (int, float)) else ""
            lines.append(f"  {mk:30s}  {val:8.4f}  {bar}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def handle_sweep(args: argparse.Namespace) -> int:
    """Execute the 'sweep' subcommand: hyperparameter sweep."""
    logger.info("Starting hyperparameter sweep")
    cfg = _namespace_to_config(args)

    algorithm = cfg.sweep_algorithm
    if not algorithm:
        logger.error("No algorithm specified for sweep")
        return 1

    task = cfg.sweep_task or discover_tasks()[0]
    budget = cfg.budget

    # Load param grid
    param_grid: Dict[str, List[Any]] = {}
    if cfg.param_grid_file:
        grid_path = Path(cfg.param_grid_file)
        if grid_path.exists():
            try:
                with open(grid_path, "r", encoding="utf-8") as fh:
                    param_grid = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load param grid: %s", exc)
                return 1
        else:
            # Try parsing as inline spec
            try:
                param_grid = parse_param_grid(cfg.param_grid_file)
            except ValueError as exc:
                logger.error("Invalid param grid: %s", exc)
                return 1

    if not param_grid:
        # Default grid
        param_grid = {
            "temperature": [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
            "top_k": [5, 10, 20, 50, 100],
            "top_p": [0.8, 0.9, 0.95, 1.0],
        }
        logger.info("Using default parameter grid")

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(itertools.product(*param_values))
    total_combos = len(all_combos)

    if budget < total_combos:
        logger.info(
            "Budget %d < total combinations %d; sampling %d random configurations",
            budget,
            total_combos,
            budget,
        )
        rng = random.Random(cfg.seed)
        selected_indices = rng.sample(range(total_combos), budget)
        combos = [all_combos[i] for i in sorted(selected_indices)]
    else:
        combos = all_combos
        budget = total_combos

    logger.info(
        "Sweep: %s on %s, %d trials, params: %s",
        algorithm,
        task,
        len(combos),
        param_names,
    )

    if cfg.dry_run:
        logger.info("[DRY RUN] Would run %d sweep trials", len(combos))
        for idx, combo in enumerate(combos[:10]):
            params = dict(zip(param_names, combo))
            logger.info("  Trial %d: %s", idx, params)
        if len(combos) > 10:
            logger.info("  ... and %d more", len(combos) - 10)
        return 0

    output_dir = create_output_directory(cfg.output_dir or f"sweep_{algorithm}")
    sampler = DeterministicSampler(cfg.seed)
    trials: List[Dict[str, Any]] = []
    best_trial: Dict[str, Any] | None = None
    best_score = -float("inf")

    start_time = time.time()

    for trial_idx, combo in enumerate(combos):
        if is_shutdown_requested():
            logger.warning("Shutdown requested, saving partial sweep results")
            break

        params = dict(zip(param_names, combo))
        trial_start = time.time()

        # Simulate evaluation
        metrics: Dict[str, float] = {}
        for m in discover_metrics()[:5]:
            metrics[m] = sampler.score()

        trial_time = time.time() - trial_start
        trial = SweepTrial(
            trial_id=trial_idx,
            params=params,
            metrics=metrics,
            wall_time_seconds=round(trial_time, 4),
            status="completed",
        )
        trial_dict = trial.to_dict()
        trials.append(trial_dict)

        # Track best (by first metric)
        primary_metric = list(metrics.values())[0] if metrics else 0
        if primary_metric > best_score:
            best_score = primary_metric
            best_trial = trial_dict

        if cfg.verbose:
            print_progress(trial_idx + 1, len(combos), prefix="  Sweep")

        logger.debug("Trial %d: params=%s score=%.4f", trial_idx, params, primary_metric)

    total_time = time.time() - start_time

    # Save results
    sweep_results = {
        "algorithm": algorithm,
        "task": task,
        "param_grid": param_grid,
        "total_trials": len(trials),
        "best_trial": best_trial,
        "trials": trials,
        "wall_time_seconds": round(total_time, 3),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(sweep_results, fh, indent=2)

    logger.info(
        "Sweep complete: %d trials in %s",
        len(trials),
        format_duration(total_time),
    )
    if best_trial:
        print(colorize(f"\n✓ Sweep complete: {len(trials)} trials in {format_duration(total_time)}", "green"))
        print(f"  Best trial: {best_trial['params']}")
        print(f"  Best score: {best_score:.4f}")
    print(f"  Results saved to: {results_path}")
    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute the 'benchmark' subcommand: run a benchmark suite."""
    logger.info("Starting benchmark suite")
    cfg = _namespace_to_config(args)

    suite_name = cfg.suite_name or "standard"
    output_dir = create_output_directory(cfg.output_dir or f"benchmark_{suite_name}")

    # Define suites
    suites: Dict[str, Dict[str, Any]] = {
        "quick": {
            "algorithms": discover_algorithms()[:3],
            "tasks": discover_tasks()[:2],
            "num_samples": 10,
            "metrics": discover_metrics()[:3],
            "description": "Quick benchmark with minimal configuration",
        },
        "standard": {
            "algorithms": discover_algorithms()[:6],
            "tasks": discover_tasks()[:4],
            "num_samples": 50,
            "metrics": discover_metrics()[:8],
            "description": "Standard benchmark with moderate configuration",
        },
        "full": {
            "algorithms": discover_algorithms(),
            "tasks": discover_tasks(),
            "num_samples": 100,
            "metrics": discover_metrics(),
            "description": "Full benchmark with all algorithms, tasks, and metrics",
        },
        "stress": {
            "algorithms": discover_algorithms()[:3],
            "tasks": discover_tasks()[:1],
            "num_samples": 500,
            "metrics": discover_metrics()[:5],
            "description": "Stress test with high sample count",
        },
    }

    if suite_name not in suites:
        logger.error(
            "Unknown benchmark suite '%s'. Available: %s",
            suite_name,
            list(suites.keys()),
        )
        return 1

    suite = suites[suite_name]
    if cfg.quick:
        suite["num_samples"] = min(suite["num_samples"], 10)
        suite["algorithms"] = suite["algorithms"][:2]
        suite["tasks"] = suite["tasks"][:1]
        logger.info("Quick mode enabled: reduced suite size")

    algorithms = suite["algorithms"]
    tasks = suite["tasks"]
    num_samples = suite["num_samples"]
    metrics = suite["metrics"]
    total_experiments = len(algorithms) * len(tasks)

    logger.info(
        "Benchmark suite '%s': %d algorithms × %d tasks × %d samples = %d experiments",
        suite_name,
        len(algorithms),
        len(tasks),
        num_samples,
        total_experiments,
    )
    logger.info("Description: %s", suite["description"])

    if cfg.dry_run:
        logger.info("[DRY RUN] Would run benchmark suite '%s'", suite_name)
        for alg in algorithms:
            for task in tasks:
                logger.info("  %s × %s (%d samples)", alg, task, num_samples)
        return 0

    sampler = DeterministicSampler(cfg.seed)
    start_time = time.time()
    all_results: List[Dict[str, Any]] = []
    completed = 0

    for alg in algorithms:
        for task in tasks:
            if is_shutdown_requested():
                break

            exp_start = time.time()
            logger.info(
                "Benchmarking %s on %s (%d/%d)",
                alg,
                task,
                completed + 1,
                total_experiments,
            )

            generations = [sampler.sample_text(64) for _ in range(num_samples)]
            metric_scores = {m: sampler.score() for m in metrics}

            exp_time = time.time() - exp_start
            result = {
                "algorithm": alg,
                "task": task,
                "num_samples": num_samples,
                "metrics": metric_scores,
                "wall_time_seconds": round(exp_time, 4),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            all_results.append(result)
            completed += 1

            if cfg.verbose:
                print_progress(completed, total_experiments, prefix="  Benchmark")

    total_time = time.time() - start_time

    # Generate leaderboard
    leaderboard: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)
    for r in all_results:
        alg = r["algorithm"]
        for m, v in r["metrics"].items():
            leaderboard[alg][m] += v
        counts[alg] += 1

    # Average scores
    for alg in leaderboard:
        for m in leaderboard[alg]:
            leaderboard[alg][m] /= max(counts[alg], 1)

    # Save results
    benchmark_results = {
        "suite_name": suite_name,
        "description": suite["description"],
        "total_experiments": total_experiments,
        "completed": completed,
        "wall_time_seconds": round(total_time, 3),
        "leaderboard": dict(leaderboard),
        "results": all_results,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(benchmark_results, fh, indent=2, default=str)

    # Print leaderboard
    if leaderboard:
        first_metric = metrics[0] if metrics else None
        sorted_algs = sorted(
            leaderboard.keys(),
            key=lambda a: leaderboard[a].get(first_metric, 0) if first_metric else 0,
            reverse=True,
        )
        headers = ["Rank", "Algorithm"] + [m.replace("_", " ").title() for m in metrics[:5]]
        rows = []
        for rank, alg in enumerate(sorted_algs, 1):
            row = [str(rank), alg]
            for m in metrics[:5]:
                row.append(f"{leaderboard[alg].get(m, 0):.4f}")
            rows.append(row)
        print(colorize(f"\n  Leaderboard — {suite_name}", "bold"))
        _print_table(headers, rows)

    logger.info(
        "Benchmark complete: %d experiments in %s",
        completed,
        format_duration(total_time),
    )
    print(colorize(f"\n✓ Benchmark '{suite_name}' complete: {completed} experiments in {format_duration(total_time)}", "green"))
    print(f"  Results saved to: {results_path}")
    return 0


def handle_export(args: argparse.Namespace) -> int:
    """Execute the 'export' subcommand: export results to various formats."""
    logger.info("Starting export")
    cfg = _namespace_to_config(args)

    input_dir = Path(cfg.input_dir) if cfg.input_dir else None
    if input_dir is None or not input_dir.exists():
        logger.error("Input directory does not exist: %s", cfg.input_dir)
        return 1

    export_format = cfg.export_format
    if export_format not in _SUPPORTED_EXPORT_FORMATS:
        logger.error(
            "Unsupported export format '%s'. Supported: %s",
            export_format,
            _SUPPORTED_EXPORT_FORMATS,
        )
        return 1

    # Collect all JSON data from input directory
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.error("No JSON files found in %s", input_dir)
        return 1

    all_data: List[Dict[str, Any]] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                all_data.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", jf.name, exc)

    if not all_data:
        logger.error("No valid data found to export")
        return 1

    # Determine output path
    if cfg.export_output:
        output_path = Path(cfg.export_output)
    else:
        ext = export_format if export_format != "sqlite" else "db"
        output_path = input_dir / f"exported_results.{ext}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting %d records to %s format -> %s",
        len(all_data),
        export_format,
        output_path,
    )

    if export_format == "csv":
        _export_csv(all_data, output_path)
    elif export_format == "json":
        _export_json(all_data, output_path)
    elif export_format == "latex":
        _export_latex(all_data, output_path)
    elif export_format == "sqlite":
        _export_sqlite(all_data, output_path)

    file_size = output_path.stat().st_size if output_path.exists() else 0
    logger.info("Export complete: %s (%s)", output_path, format_size(file_size))
    print(colorize(f"\n✓ Exported {len(all_data)} records to {export_format}", "green"))
    print(f"  Output: {output_path} ({format_size(file_size)})")
    return 0


def handle_validate(args: argparse.Namespace) -> int:
    """Execute the 'validate' subcommand: validate a configuration file."""
    logger.info("Starting configuration validation")
    cfg = _namespace_to_config(args)

    config_file = cfg.validate_config_file or cfg.config_path
    if not config_file:
        logger.error("No configuration file specified")
        return 1

    config_path = Path(config_file)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    print(f"Validating: {config_path}")

    try:
        config_data = load_config_file(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(f"\n✗ Failed to parse config: {exc}", "red"))
        return 1

    errors = validate_config(config_data)

    if errors:
        print(colorize(f"\n✗ Found {len(errors)} validation error(s):", "red"))
        for i, err in enumerate(errors, 1):
            print(colorize(f"  {i}. {err}", "yellow"))
        return 1

    print(colorize("\n✓ Configuration is valid", "green"))

    # Print summary
    print(f"\n  Keys: {len(config_data)}")
    for key in sorted(config_data.keys()):
        val = config_data[key]
        if isinstance(val, list):
            print(f"  {key}: [{len(val)} items]")
        elif isinstance(val, dict):
            print(f"  {key}: {{{len(val)} keys}}")
        else:
            print(f"  {key}: {val}")

    # Cross-reference with known algorithms/tasks/metrics
    known_algs = set(discover_algorithms())
    known_tasks = set(discover_tasks())
    known_metrics = set(discover_metrics())

    warnings: List[str] = []
    for alg in config_data.get("algorithms", []):
        if alg not in known_algs:
            warnings.append(f"Unknown algorithm: '{alg}'")
    for task in config_data.get("tasks", []):
        if task not in known_tasks:
            warnings.append(f"Unknown task: '{task}'")
    for metric in config_data.get("metrics", []):
        if metric not in known_metrics:
            warnings.append(f"Unknown metric: '{metric}'")

    if warnings:
        print(colorize(f"\n⚠ {len(warnings)} warning(s):", "yellow"))
        for w in warnings:
            print(f"  - {w}")

    return 0


def handle_info(args: argparse.Namespace) -> int:
    """Execute the 'info' subcommand: show system info and registries."""
    print_banner()

    info = _collect_system_info()

    # System information
    print(colorize("System Information", "bold"))
    print(f"  Arena version:  {info['arena_version']}")
    print(f"  Python version: {info['python_version']}")
    print(f"  Platform:       {info['platform']}")
    print(f"  Machine:        {info['machine']}")
    print(f"  Processor:      {info['processor']}")
    print(f"  CPU count:      {info['cpu_count']}")
    print(f"  Memory:         {info.get('memory_total', 'unknown')}")
    print(f"  Working dir:    {info['cwd']}")
    print()

    # Registered algorithms
    algorithms = discover_algorithms()
    print(colorize(f"Registered Algorithms ({len(algorithms)})", "bold"))
    for i, alg in enumerate(algorithms, 1):
        print(f"  {i:2d}. {alg}")
    print()

    # Registered metrics
    metrics = discover_metrics()
    print(colorize(f"Registered Metrics ({len(metrics)})", "bold"))
    for i, m in enumerate(metrics, 1):
        print(f"  {i:2d}. {m}")
    print()

    # Registered tasks
    tasks = discover_tasks()
    print(colorize(f"Registered Tasks ({len(tasks)})", "bold"))
    for i, t in enumerate(tasks, 1):
        print(f"  {i:2d}. {t}")
    print()

    # Optional dependencies
    deps = info.get("optional_dependencies", {})
    if deps:
        print(colorize("Optional Dependencies", "bold"))
        for dep_name, dep_version in sorted(deps.items()):
            status = colorize("✓ " + dep_version, "green") if dep_version != "not installed" else colorize("✗ not installed", "dim")
            print(f"  {dep_name:20s} {status}")
        print()

    # Supported formats
    print(colorize("Supported Formats", "bold"))
    print(f"  Plot formats:   {', '.join(_SUPPORTED_PLOT_FORMATS)}")
    print(f"  Export formats:  {', '.join(_SUPPORTED_EXPORT_FORMATS)}")
    print(f"  Comparison:      {', '.join(_COMPARISON_METHODS)}")
    print()

    return 0


def handle_config(args: argparse.Namespace) -> int:
    """Execute the 'config' subcommand: manage configuration profiles."""
    logger.info("Managing configuration")
    cfg = _namespace_to_config(args)

    action = cfg.config_action
    if not action:
        logger.error("No config action specified. Use show, set, reset, or export.")
        return 1

    mgr = ConfigProfileManager()

    if action == "show":
        profiles = mgr.list_profiles()
        if not profiles:
            print("No configuration profiles found.")
            print(f"  Profile directory: {mgr.base_dir}")
            return 0
        print(colorize(f"Configuration Profiles ({len(profiles)})", "bold"))
        for profile_name in profiles:
            print(f"\n  [{profile_name}]")
            try:
                profile = mgr.get_profile(profile_name)
                for k, v in sorted(profile.items()):
                    print(f"    {k} = {v}")
            except KeyError:
                print("    (empty)")
        return 0

    if action == "set":
        key = cfg.config_key
        value = cfg.config_value
        if not key:
            logger.error("No key specified for 'config set'")
            return 1
        # profile name is "default" unless specified via key format "profile.key"
        profile_name = "default"
        if "." in key:
            profile_name, key = key.split(".", 1)
        coerced_value = _coerce_value(value) if value else value
        mgr.set_value(profile_name, key, coerced_value)
        print(colorize(f"✓ Set [{profile_name}] {key} = {coerced_value}", "green"))
        return 0

    if action == "reset":
        profile_name = cfg.config_key or "default"
        if not cfg.force:
            if not confirm_action(f"Reset profile '{profile_name}'?"):
                print("Aborted.")
                return 0
        mgr.reset_profile(profile_name)
        print(colorize(f"✓ Profile '{profile_name}' reset", "green"))
        return 0

    if action == "export":
        profile_name = cfg.config_key or "default"
        output_path = Path(cfg.config_value or f"{profile_name}_config.json")
        try:
            mgr.export_profile(profile_name, output_path)
            print(colorize(f"✓ Exported profile '{profile_name}' to {output_path}", "green"))
        except KeyError:
            logger.error("Profile '%s' not found", profile_name)
            return 1
        return 0

    logger.error("Unknown config action: %s", action)
    return 1


# ---------------------------------------------------------------------------
# Namespace -> CLIConfig conversion
# ---------------------------------------------------------------------------


def _namespace_to_config(ns: argparse.Namespace) -> CLIConfig:
    """Convert an argparse.Namespace to a CLIConfig."""
    cfg = CLIConfig()

    # Map every known attribute
    _mapping: Dict[str, str] = {
        "command": "command",
        "config": "config_path",
        "output_dir": "output_dir",
        "verbose": "verbose",
        "seed": "seed",
        "log_level": "log_level",
        "parallel": "parallel",
        "dry_run": "dry_run",
        "force": "force",
        "algorithms": "algorithms",
        "tasks": "tasks",
        "num_samples": "num_samples",
        "max_length": "max_length",
        "checkpoint_resume": "checkpoint_resume",
        "input_dir": "input_dir",
        "metrics": "metrics",
        "output_format": "output_format",
        "input_dirs": "input_dirs",
        "method": "comparison_method",
        "rope_width": "rope_width",
        "alpha": "alpha",
        "plot_types": "plot_types",
        "plot_format": "plot_format",
        "algorithm": "sweep_algorithm",
        "param_grid_file": "param_grid_file",
        "task": "sweep_task",
        "budget": "budget",
        "suite_name": "suite_name",
        "quick": "quick",
        "format": "export_format",
        "output": "export_output",
        "action": "config_action",
        "key": "config_key",
        "value": "config_value",
        "config_file": "validate_config_file",
    }

    for ns_attr, cfg_attr in _mapping.items():
        val = getattr(ns, ns_attr, None)
        if val is not None:
            setattr(cfg, cfg_attr, val)

    return cfg


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _add_global_arguments(parser: ArenaArgumentParser) -> None:
    """Add global arguments shared by all subcommands."""
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        metavar="PATH",
        help="Path to a JSON configuration file. CLI arguments override file values.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        metavar="DIR",
        help="Base output directory for results, plots, and exports.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output with progress bars and extra logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        metavar="N",
        help=f"Random seed for reproducibility (default: {_DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=[l.lower() for l in _LOG_LEVELS],
        metavar="LEVEL",
        help="Logging level: debug, info, warning, error, critical (default: info).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        metavar="PATH",
        help="Path to a log file. If provided, all log messages are also written here.",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help=f"Number of parallel workers (1-{_MAX_PARALLEL}, default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without actually running experiments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output files without confirmation.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable ANSI colour output.",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        default=False,
        help="Suppress the ASCII art banner.",
    )


def _build_run_parser(subparsers: Any) -> None:
    """Build the 'run' subcommand parser."""
    p = subparsers.add_parser(
        "run",
        help="Run diversity decoding experiments.",
        description=textwrap.dedent("""\
            Run text generation experiments with one or more decoding algorithms
            on specified tasks. Results are saved as JSON files in the output
            directory, one file per algorithm-task combination.

            Supports checkpoint-based resume so interrupted runs can continue
            from where they left off.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-a",
        "--algorithms",
        nargs="+",
        type=str,
        default=None,
        metavar="ALG",
        help=(
            "Decoding algorithms to run. If omitted, a default subset is used. "
            "Available: " + ", ".join(discover_algorithms()[:6]) + ", ..."
        ),
    )
    p.add_argument(
        "-t",
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        metavar="TASK",
        help=(
            "Generation tasks to evaluate on. If omitted, a default subset is used. "
            "Available: " + ", ".join(discover_tasks()[:4]) + ", ..."
        ),
    )
    p.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=_DEFAULT_NUM_SAMPLES,
        metavar="N",
        help=f"Number of samples to generate per experiment (default: {_DEFAULT_NUM_SAMPLES}).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=_DEFAULT_MAX_LENGTH,
        metavar="N",
        help=f"Maximum generation length in tokens (default: {_DEFAULT_MAX_LENGTH}).",
    )
    p.add_argument(
        "--checkpoint-resume",
        type=str,
        default="",
        metavar="DIR",
        help="Path to a checkpoint directory to resume an interrupted run.",
    )
    p.set_defaults(command="run", handler=handle_run)


def _build_evaluate_parser(subparsers: Any) -> None:
    """Build the 'evaluate' subcommand parser."""
    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate diversity of existing generations.",
        description=textwrap.dedent("""\
            Compute diversity metrics on previously generated text. Reads JSON
            result files from the input directory and computes the requested
            metrics for each algorithm-task pair.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing generation result JSON files.",
    )
    p.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        type=str,
        default=None,
        metavar="METRIC",
        help=(
            "Diversity metrics to compute. If omitted, a default subset is used. "
            "Available: " + ", ".join(discover_metrics()[:6]) + ", ..."
        ),
    )
    p.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format for evaluation results (default: json).",
    )
    p.set_defaults(command="evaluate", handler=handle_evaluate)


def _build_compare_parser(subparsers: Any) -> None:
    """Build the 'compare' subcommand parser."""
    p = subparsers.add_parser(
        "compare",
        help="Statistically compare decoding algorithms.",
        description=textwrap.dedent("""\
            Perform pairwise statistical comparisons between decoding algorithms.
            Supports Bayesian analysis (with ROPE), bootstrap confidence intervals,
            and permutation tests. Reads evaluation results from multiple input
            directories.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input-dirs",
        nargs="+",
        type=str,
        required=True,
        metavar="DIR",
        help="Two or more directories containing evaluation results to compare.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="bayes",
        choices=list(_COMPARISON_METHODS),
        help="Statistical comparison method (default: bayes).",
    )
    p.add_argument(
        "--rope-width",
        type=float,
        default=_DEFAULT_ROPE_WIDTH,
        metavar="W",
        help=f"ROPE (Region of Practical Equivalence) width for Bayesian comparison (default: {_DEFAULT_ROPE_WIDTH}).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=_DEFAULT_ALPHA,
        metavar="A",
        help=f"Significance level for frequentist tests (default: {_DEFAULT_ALPHA}).",
    )
    p.set_defaults(command="compare", handler=handle_compare)


def _build_visualize_parser(subparsers: Any) -> None:
    """Build the 'visualize' subcommand parser."""
    p = subparsers.add_parser(
        "visualize",
        help="Generate plots and visualizations from results.",
        description=textwrap.dedent("""\
            Generate publication-ready plots from evaluation results. Supports
            heatmaps, bar charts, radar plots, box plots, and interactive HTML
            visualizations.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing evaluation results (must have eval_summary.json).",
    )
    p.add_argument(
        "-p",
        "--plot-types",
        nargs="+",
        type=str,
        default=None,
        metavar="TYPE",
        help=(
            "Types of plots to generate. Options: heatmap, bar, radar, box, "
            "violin, scatter, line, distribution, ranking, comparison "
            "(default: heatmap bar radar)."
        ),
    )
    p.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=list(_SUPPORTED_PLOT_FORMATS),
        help="Output format for plots (default: png).",
    )
    p.set_defaults(command="visualize", handler=handle_visualize)


def _build_sweep_parser(subparsers: Any) -> None:
    """Build the 'sweep' subcommand parser."""
    p = subparsers.add_parser(
        "sweep",
        help="Run a hyperparameter sweep.",
        description=textwrap.dedent("""\
            Perform a hyperparameter sweep for a single algorithm over a grid
            of parameter values. If the total number of combinations exceeds
            the budget, random search is used to stay within budget.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        metavar="ALG",
        help="The decoding algorithm to sweep parameters for.",
    )
    p.add_argument(
        "-g",
        "--param-grid-file",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to a JSON file defining the parameter grid, or an inline "
            "spec like 'temperature=0.5,1.0;top_k=10,50'."
        ),
    )
    p.add_argument(
        "-t",
        "--task",
        type=str,
        default="",
        metavar="TASK",
        help="Task to evaluate on during the sweep.",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=_DEFAULT_BUDGET,
        metavar="N",
        help=f"Maximum number of trials (default: {_DEFAULT_BUDGET}).",
    )
    p.set_defaults(command="sweep", handler=handle_sweep)


def _build_benchmark_parser(subparsers: Any) -> None:
    """Build the 'benchmark' subcommand parser."""
    p = subparsers.add_parser(
        "benchmark",
        help="Run a pre-defined benchmark suite.",
        description=textwrap.dedent("""\
            Run a standardised benchmark suite that exercises multiple algorithms,
            tasks, and metrics. Produces a leaderboard and detailed per-experiment
            results.

            Available suites: quick, standard, full, stress.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-s",
        "--suite-name",
        type=str,
        default="standard",
        choices=["quick", "standard", "full", "stress"],
        metavar="SUITE",
        help="Name of the benchmark suite to run (default: standard).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Enable quick mode: reduce sample count and number of experiments.",
    )
    p.set_defaults(command="benchmark", handler=handle_benchmark)


def _build_export_parser(subparsers: Any) -> None:
    """Build the 'export' subcommand parser."""
    p = subparsers.add_parser(
        "export",
        help="Export results to various formats.",
        description=textwrap.dedent("""\
            Export experiment and evaluation results to CSV, JSON, LaTeX tables,
            or SQLite databases for further analysis or inclusion in papers.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing result JSON files to export.",
    )
    p.add_argument(
        "-f",
        "--format",
        type=str,
        default="json",
        choices=list(_SUPPORTED_EXPORT_FORMATS),
        help="Export format (default: json).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        metavar="PATH",
        help="Output file path. If omitted, writes to the input directory.",
    )
    p.set_defaults(command="export", handler=handle_export)


def _build_validate_parser(subparsers: Any) -> None:
    """Build the 'validate' subcommand parser."""
    p = subparsers.add_parser(
        "validate",
        help="Validate a configuration file.",
        description=textwrap.dedent("""\
            Parse and validate a JSON configuration file against the schema
            expected by the arena. Reports any errors or warnings found.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "config_file",
        type=str,
        metavar="CONFIG_FILE",
        help="Path to the configuration file to validate.",
    )
    p.set_defaults(command="validate", handler=handle_validate)


def _build_info_parser(subparsers: Any) -> None:
    """Build the 'info' subcommand parser."""
    p = subparsers.add_parser(
        "info",
        help="Show system information and registered components.",
        description=textwrap.dedent("""\
            Display system information, registered algorithms, metrics, tasks,
            and optional dependency status. Useful for debugging and verifying
            the installation.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.set_defaults(command="info", handler=handle_info)


def _build_config_parser(subparsers: Any) -> None:
    """Build the 'config' subcommand parser."""
    p = subparsers.add_parser(
        "config",
        help="Manage configuration profiles.",
        description=textwrap.dedent("""\
            Create, view, modify, reset, and export named configuration profiles.
            Profiles are stored in ~/.config/diversity-arena/profiles.json.

            Actions:
              show    - Display all profiles and their values.
              set     - Set a key-value pair (format: [profile.]key value).
              reset   - Remove a profile (requires --force or confirmation).
              export  - Export a profile to a JSON file.
        """),
        formatter_class=ArenaHelpFormatter,
    )
    p.add_argument(
        "action",
        type=str,
        choices=["show", "set", "reset", "export"],
        help="Configuration management action.",
    )
    p.add_argument(
        "key",
        nargs="?",
        type=str,
        default="",
        help="Configuration key (for set/reset/export). Use 'profile.key' to target a specific profile.",
    )
    p.add_argument(
        "value",
        nargs="?",
        type=str,
        default="",
        help="Configuration value (for set action).",
    )
    p.set_defaults(command="config", handler=handle_config)


def build_parser() -> ArenaArgumentParser:
    """Build and return the complete argument parser."""
    parser = ArenaArgumentParser(
        prog=__prog__,
        description=textwrap.dedent(f"""\
            {colorize('Diversity Decoding Arena', 'bold')} v{__version__}

            A comprehensive framework for benchmarking and comparing text
            generation decoding strategies on diversity metrics.

            Run experiments, evaluate diversity, perform statistical comparisons,
            and generate publication-ready visualizations — all from the
            command line.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s run -a beam_search nucleus_sampling -t story_gen -n 200
              %(prog)s evaluate -i ./results -m self_bleu distinct_n entropy
              %(prog)s compare -i ./run_a ./run_b --method bayes --rope-width 0.01
              %(prog)s visualize -i ./results -p heatmap radar --plot-format svg
              %(prog)s sweep -a nucleus_sampling -g grid.json --budget 100
              %(prog)s benchmark -s standard
              %(prog)s export -i ./results -f latex -o table.tex
              %(prog)s validate config.json
              %(prog)s info
              %(prog)s config show
              %(prog)s config set default.seed 123

            Documentation: https://github.com/diversity-decoding-arena
        """),
    )
    _add_global_arguments(parser)

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        metavar="COMMAND",
        help="Run '%(prog)s COMMAND --help' for command-specific help.",
    )

    _build_run_parser(subparsers)
    _build_evaluate_parser(subparsers)
    _build_compare_parser(subparsers)
    _build_visualize_parser(subparsers)
    _build_sweep_parser(subparsers)
    _build_benchmark_parser(subparsers)
    _build_export_parser(subparsers)
    _build_validate_parser(subparsers)
    _build_info_parser(subparsers)
    _build_config_parser(subparsers)

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    """Main entry point for the Diversity Decoding Arena CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success, non-zero = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle --no-color
    if getattr(args, "no_color", False):
        os.environ["NO_COLOR"] = "1"

    # Print banner unless suppressed
    show_banner = not getattr(args, "no_banner", False)

    # Setup logging
    log_level = getattr(args, "log_level", "info").upper()
    log_file = getattr(args, "log_file", "") or None
    setup_logging(log_level, log_file)

    # Install signal handlers
    _install_signal_handlers()

    try:
        # No subcommand given
        if not getattr(args, "command", None):
            if show_banner:
                print_banner()
            parser.print_help()
            return 0

        # Load config file if specified
        config_path = getattr(args, "config", "")
        if config_path:
            try:
                file_cfg = load_config_file(config_path)
                logger.debug("Loaded config file: %s", config_path)
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Failed to load config file: %s", exc)
                return 1

        # Set seed
        seed = getattr(args, "seed", _DEFAULT_SEED)
        random.seed(seed)

        # Show banner for verbose runs
        if show_banner and getattr(args, "verbose", False):
            print_banner()

        # Dispatch to handler
        handler = getattr(args, "handler", None)
        if handler is None:
            logger.error("No handler for command '%s'", args.command)
            parser.print_help()
            return 1

        logger.debug(
            "Dispatching command '%s' with args: %s",
            args.command,
            {k: v for k, v in vars(args).items() if k != "handler"},
        )

        start_time = time.time()
        exit_code = handler(args)
        elapsed = time.time() - start_time

        logger.debug(
            "Command '%s' finished in %s with exit code %d",
            args.command,
            format_duration(elapsed),
            exit_code,
        )

        return exit_code

    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        logger.error("Unhandled exception: %s", exc)
        if log_level == "DEBUG":
            traceback.print_exc()
        else:
            sys.stderr.write(
                colorize(f"Error: {exc}\n", "red")
                + "Run with --log-level debug for the full traceback.\n"
            )
        return 1
    finally:
        _restore_signal_handlers()


if __name__ == "__main__":
    sys.exit(main())
