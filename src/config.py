"""
Configuration system for the Diversity Decoding Arena.

Provides dataclass-based configuration for model loading, generation,
caching, metrics, hyperparameter sweeps, and experiment management.
Supports YAML/JSON serialization, environment variable overrides,
config validation, and merging.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_PREFIX = "ARENA_"

SUPPORTED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
]

SUPPORTED_QUANTIZATIONS = ["none", "int8", "int4", "fp16"]

SUPPORTED_DEVICES = ["cpu", "cuda", "mps"]

SUPPORTED_CACHE_STRATEGIES = ["lru", "lfu", "fifo", "none"]

SUPPORTED_EXPORT_FORMATS = ["json", "csv", "parquet", "markdown"]

SUPPORTED_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_TASK_DOMAINS = [
    "open_ended",
    "story_generation",
    "dialogue",
    "summarization",
    "question_answering",
]

SUPPORTED_DECODING_METHODS = [
    "greedy",
    "top_k",
    "top_p",
    "typical_p",
    "temperature",
    "diverse_beam_search",
    "stochastic_beam_search",
    "mbr_decoding",
    "contrastive_search",
    "eta_sampling",
    "mirostat",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QuantizationType(str, Enum):
    """Supported quantization types for ONNX Runtime."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    NONE = "none"


class ExportFormat(str, Enum):
    """Supported output export formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    MARKDOWN = "markdown"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model-specific configuration.

    Attributes:
        model_name: HuggingFace model identifier (default ``gpt2``).
        quantization: Quantization mode for ONNX Runtime.
        max_seq_len: Maximum sequence length the model supports.
        vocab_size: Vocabulary size (50257 for GPT-2).
        device: Compute device.
        onnx_path: Optional path to a pre-exported ONNX model file.
        use_onnx_runtime: Whether to use ONNX Runtime for inference.
        num_attention_heads: Number of attention heads (12 for GPT-2 124M).
        hidden_size: Hidden dimension size (768 for GPT-2 124M).
        num_layers: Number of transformer layers (12 for GPT-2 124M).
        dtype: Data type string for non-quantized weights.
    """

    model_name: str = "gpt2"
    quantization: str = "int8"
    max_seq_len: int = 1024
    vocab_size: int = 50257
    device: str = "cpu"
    onnx_path: Optional[str] = None
    use_onnx_runtime: bool = True
    num_attention_heads: int = 12
    hidden_size: int = 768
    num_layers: int = 12
    dtype: str = "float32"


@dataclass
class CacheConfig:
    """Cache configuration with eviction policies.

    Attributes:
        cache_dir: Directory for cached artefacts (models, tokenisers, logits).
        max_cache_size_gb: Maximum total cache size in gigabytes.
        cache_strategy: Eviction policy when the cache is full.
        cache_logits: Whether to cache per-token logit tensors.
        cache_embeddings: Whether to cache hidden-state embeddings.
        cache_model: Whether to cache the downloaded/converted model.
        ttl_hours: Time-to-live for cache entries in hours (0 = no expiry).
        compress: Whether to gzip cached numpy arrays.
    """

    cache_dir: str = ".arena_cache"
    max_cache_size_gb: float = 2.0
    cache_strategy: str = "lru"
    cache_logits: bool = True
    cache_embeddings: bool = False
    cache_model: bool = True
    ttl_hours: int = 0
    compress: bool = False


@dataclass
class MetricConfig:
    """Which diversity / quality metrics to compute and their parameters.

    Attributes:
        enabled_metrics: List of metric names to compute.
        ngram_orders: Orders of n-gram diversity / entropy metrics.
        embedding_model: Sentence-transformer model for embedding-based metrics.
        self_bleu_sample_size: Number of pairs for Self-BLEU estimation.
        distinct_n_orders: Orders for Distinct-N metric.
        use_idf_weights: Weight n-grams by inverse document frequency.
        reference_corpus_path: Optional path to a reference corpus for
            metrics that compare against reference text.
        semantic_similarity_threshold: Threshold for near-duplicate detection.
        compute_mauve: Whether to compute the MAUVE score.
        compute_coherence: Whether to compute coherence metrics.
        compute_perplexity: Whether to compute perplexity via the model.
        max_metric_workers: Max parallel workers for metric computation.
    """

    enabled_metrics: List[str] = field(
        default_factory=lambda: [
            "distinct_n",
            "self_bleu",
            "entropy",
            "semantic_diversity",
            "coverage",
            "perplexity",
        ]
    )
    ngram_orders: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    embedding_model: str = "all-MiniLM-L6-v2"
    self_bleu_sample_size: int = 500
    distinct_n_orders: List[int] = field(default_factory=lambda: [1, 2, 3])
    use_idf_weights: bool = False
    reference_corpus_path: Optional[str] = None
    semantic_similarity_threshold: float = 0.95
    compute_mauve: bool = False
    compute_coherence: bool = True
    compute_perplexity: bool = True
    max_metric_workers: int = 4


@dataclass
class SweepConfig:
    """Hyperparameter sweep configuration.

    Defines which decoding parameters to sweep over when comparing
    algorithms side by side.

    Attributes:
        methods: Decoding methods to include in the sweep.
        temperature_range: (min, max, step) for temperature sweep.
        top_k_values: Discrete top-k values to evaluate.
        top_p_values: Discrete top-p (nucleus) thresholds.
        typical_p_values: Discrete typical-p thresholds.
        num_beams_values: Beam widths for beam-search variants.
        diversity_penalty_range: (min, max, step) for diverse beam search.
        repetition_penalty_values: Repetition penalty multipliers.
        eta_values: Eta values for eta-sampling.
        mirostat_tau_values: Target surprise for Mirostat.
        mirostat_lr_values: Learning rate for Mirostat.
        alpha_values: Alpha for contrastive search.
        max_combinations: Safety cap on the total number of configurations.
        parallel_runs: How many sweep configurations to run in parallel.
    """

    methods: List[str] = field(
        default_factory=lambda: [
            "top_k",
            "top_p",
            "typical_p",
            "temperature",
            "diverse_beam_search",
            "contrastive_search",
        ]
    )
    temperature_range: Tuple[float, float, float] = (0.3, 1.5, 0.1)
    top_k_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    top_p_values: List[float] = field(
        default_factory=lambda: [0.8, 0.9, 0.92, 0.95, 0.99]
    )
    typical_p_values: List[float] = field(
        default_factory=lambda: [0.2, 0.5, 0.8, 0.95]
    )
    num_beams_values: List[int] = field(default_factory=lambda: [4, 6, 8, 12])
    diversity_penalty_range: Tuple[float, float, float] = (0.5, 5.0, 0.5)
    repetition_penalty_values: List[float] = field(
        default_factory=lambda: [1.0, 1.1, 1.2, 1.5]
    )
    eta_values: List[float] = field(
        default_factory=lambda: [0.0003, 0.001, 0.003, 0.01]
    )
    mirostat_tau_values: List[float] = field(
        default_factory=lambda: [3.0, 5.0, 7.0]
    )
    mirostat_lr_values: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0]
    )
    alpha_values: List[float] = field(
        default_factory=lambda: [0.4, 0.6, 0.8, 1.0]
    )
    max_combinations: int = 500
    parallel_runs: int = 1


@dataclass
class ExperimentConfig:
    """Full experiment specification.

    Groups model, cache, metrics, and sweep configs together with
    experiment-level metadata.

    Attributes:
        name: Human-readable experiment name.
        description: Free-text description of the experiment goal.
        seed: Global random seed for reproducibility.
        num_prompts: Number of prompts to evaluate per task domain.
        task_domains: List of task domains to include.
        prompt_source: Where prompts come from (``builtin``, ``file``, ``hf_dataset``).
        prompt_file: Path to a prompt file when ``prompt_source == "file"``.
        model: Model configuration.
        cache: Cache configuration.
        metrics: Metric configuration.
        sweep: Hyperparameter sweep configuration.
        tags: Arbitrary tags for filtering / grouping experiments.
        notes: Free-text notes.
    """

    name: str = "default_experiment"
    description: str = ""
    seed: int = 42
    num_prompts: int = 50
    task_domains: List[str] = field(default_factory=lambda: list(DEFAULT_TASK_DOMAINS))
    prompt_source: str = "builtin"
    prompt_file: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ArenaConfig:
    """Top-level configuration for the Diversity Decoding Arena.

    Attributes:
        model_name: HuggingFace model identifier.
        quantization: ONNX Runtime quantization mode.
        max_seq_len: Maximum sequence length.
        vocab_size: Model vocabulary size.
        device: Compute device (``cpu``, ``cuda``, ``mps``).
        num_sequences: Number of sequences to generate per prompt.
        max_new_tokens: Maximum new tokens per generation call.
        batch_size: Batch size for generation.
        cache_dir: Root cache directory.
        max_cache_size_gb: Maximum cache size in GB.
        cache_strategy: Cache eviction strategy.
        output_dir: Directory for experiment outputs.
        save_logits: Persist per-token logits to disk.
        save_embeddings: Persist hidden-state embeddings to disk.
        export_formats: Which formats to export results in.
        seed: Global random seed.
        num_prompts: Prompts to evaluate per task domain.
        task_domains: Task domains for prompt selection.
        log_level: Python logging level string.
        log_file: Optional path to a log file.
        enable_progress_bar: Show tqdm-style progress bars.
        experiment: Nested full experiment specification.
    """

    # -- model --
    model_name: str = "gpt2"
    quantization: str = "int8"
    max_seq_len: int = 1024
    vocab_size: int = 50257
    device: str = "cpu"

    # -- generation --
    num_sequences: int = 25
    max_new_tokens: int = 128
    batch_size: int = 8

    # -- cache --
    cache_dir: str = ".arena_cache"
    max_cache_size_gb: float = 2.0
    cache_strategy: str = "lru"

    # -- output --
    output_dir: str = "arena_output"
    save_logits: bool = False
    save_embeddings: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])

    # -- experiment --
    seed: int = 42
    num_prompts: int = 50
    task_domains: List[str] = field(default_factory=lambda: list(DEFAULT_TASK_DOMAINS))

    # -- logging --
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_progress_bar: bool = True

    # -- nested configs --
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# ---------------------------------------------------------------------------
# Helpers: serialisation
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or nested structure) to a plain dict.

    Tuples are converted to lists so they survive JSON round-tripping.
    """
    if hasattr(obj, "__dataclass_fields__"):
        result: Dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _dict_to_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a plain dict.

    Unknown keys are silently ignored so that forward-compatible config
    files do not cause errors.
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    known_fields = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}

    for key, value in data.items():
        if key not in known_fields:
            logger.debug("Ignoring unknown config key: %s", key)
            continue
        fld = known_fields[key]
        ftype = fld.type

        # Resolve nested dataclass fields
        if isinstance(ftype, str):
            # Handle forward references encoded as strings
            ftype = _resolve_type_str(ftype)

        if isinstance(ftype, type) and hasattr(ftype, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(ftype, value)
        elif ftype in (
            "Tuple[float, float, float]",
            Tuple[float, float, float],
        ) or (hasattr(ftype, "__origin__") and ftype.__origin__ is tuple):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _resolve_type_str(type_str: str) -> Any:
    """Best-effort resolution of stringified type annotations."""
    _mapping = {
        "ModelConfig": ModelConfig,
        "CacheConfig": CacheConfig,
        "MetricConfig": MetricConfig,
        "SweepConfig": SweepConfig,
        "ExperimentConfig": ExperimentConfig,
        "ArenaConfig": ArenaConfig,
    }
    return _mapping.get(type_str, type_str)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """Resolve a path, expanding ``~`` and making it absolute.

    If *base* is given and *path* is relative, it is resolved relative
    to *base*; otherwise relative to the current working directory.

    Args:
        path: The path string or Path object.
        base: Optional base directory for relative paths.

    Returns:
        An absolute ``Path``.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        if base is not None:
            p = base / p
        else:
            p = Path.cwd() / p
    return p.resolve()


def resolve_config_paths(config: ArenaConfig) -> ArenaConfig:
    """Return a *new* config with all path-like fields resolved to absolute paths.

    This resolves ``cache_dir``, ``output_dir``, ``log_file``,
    ``experiment.prompt_file``, ``experiment.model.onnx_path``, and
    ``experiment.metrics.reference_corpus_path``.
    """
    cfg = copy.deepcopy(config)
    cfg.cache_dir = str(resolve_path(cfg.cache_dir))
    cfg.output_dir = str(resolve_path(cfg.output_dir))
    if cfg.log_file:
        cfg.log_file = str(resolve_path(cfg.log_file))
    if cfg.experiment.prompt_file:
        cfg.experiment.prompt_file = str(resolve_path(cfg.experiment.prompt_file))
    if cfg.experiment.model.onnx_path:
        cfg.experiment.model.onnx_path = str(
            resolve_path(cfg.experiment.model.onnx_path)
        )
    if cfg.experiment.metrics.reference_corpus_path:
        cfg.experiment.metrics.reference_corpus_path = str(
            resolve_path(cfg.experiment.metrics.reference_corpus_path)
        )
    return cfg


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

# Mapping from env-var suffix to (dotted config path, type converter).
_ENV_OVERRIDES: List[Tuple[str, str, type]] = [
    ("MODEL", "model_name", str),
    ("QUANTIZATION", "quantization", str),
    ("DEVICE", "device", str),
    ("MAX_SEQ_LEN", "max_seq_len", int),
    ("VOCAB_SIZE", "vocab_size", int),
    ("NUM_SEQUENCES", "num_sequences", int),
    ("MAX_NEW_TOKENS", "max_new_tokens", int),
    ("BATCH_SIZE", "batch_size", int),
    ("CACHE_DIR", "cache_dir", str),
    ("MAX_CACHE_SIZE_GB", "max_cache_size_gb", float),
    ("CACHE_STRATEGY", "cache_strategy", str),
    ("OUTPUT_DIR", "output_dir", str),
    ("SAVE_LOGITS", "save_logits", _bool_converter := lambda v: v.lower() in ("1", "true", "yes")),
    ("SAVE_EMBEDDINGS", "save_embeddings", lambda v: v.lower() in ("1", "true", "yes")),
    ("SEED", "seed", int),
    ("NUM_PROMPTS", "num_prompts", int),
    ("LOG_LEVEL", "log_level", str),
    ("LOG_FILE", "log_file", str),
    ("PROGRESS_BAR", "enable_progress_bar", lambda v: v.lower() in ("1", "true", "yes")),
]


def apply_env_overrides(config: ArenaConfig) -> ArenaConfig:
    """Apply environment variable overrides to *config* (mutates in place).

    Environment variables are prefixed with ``ARENA_``.  For example
    ``ARENA_MODEL=distilgpt2`` overrides ``config.model_name``.

    Args:
        config: The configuration to mutate.

    Returns:
        The same *config* object, for chaining convenience.
    """
    for suffix, attr, converter in _ENV_OVERRIDES:
        env_key = f"{_ENV_PREFIX}{suffix}"
        value = os.environ.get(env_key)
        if value is not None:
            try:
                setattr(config, attr, converter(value))
                logger.info("Env override %s=%s applied to config.%s", env_key, value, attr)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to apply env override %s=%s: %s", env_key, value, exc
                )

    # Handle ARENA_TASK_DOMAINS as comma-separated list
    td_val = os.environ.get(f"{_ENV_PREFIX}TASK_DOMAINS")
    if td_val is not None:
        config.task_domains = [d.strip() for d in td_val.split(",") if d.strip()]

    # Handle ARENA_EXPORT_FORMATS as comma-separated list
    ef_val = os.environ.get(f"{_ENV_PREFIX}EXPORT_FORMATS")
    if ef_val is not None:
        config.export_formats = [f.strip() for f in ef_val.split(",") if f.strip()]

    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(config: ArenaConfig) -> List[str]:
    """Validate *config* and return a list of warning / error strings.

    An empty list means the configuration is valid.

    Args:
        config: The configuration to validate.

    Returns:
        A list of human-readable validation messages.  Messages prefixed
        with ``ERROR:`` are fatal; those prefixed with ``WARNING:`` are
        advisory.
    """
    issues: List[str] = []

    # -- model validation --
    if config.model_name not in SUPPORTED_MODELS:
        issues.append(
            f"WARNING: model_name '{config.model_name}' is not in the "
            f"pre-validated list {SUPPORTED_MODELS}. It may still work if "
            "available on HuggingFace Hub."
        )

    if config.quantization not in SUPPORTED_QUANTIZATIONS:
        issues.append(
            f"ERROR: quantization '{config.quantization}' is not supported. "
            f"Choose from {SUPPORTED_QUANTIZATIONS}."
        )

    if config.device not in SUPPORTED_DEVICES:
        issues.append(
            f"ERROR: device '{config.device}' is not supported. "
            f"Choose from {SUPPORTED_DEVICES}."
        )

    if config.device != "cpu" and config.quantization in ("int8", "int4"):
        issues.append(
            f"WARNING: quantization '{config.quantization}' with device "
            f"'{config.device}' — ONNX Runtime INT8/INT4 is optimised for "
            "CPU; GPU quantisation support may be limited."
        )

    if config.max_seq_len < 1 or config.max_seq_len > 2048:
        issues.append(
            f"ERROR: max_seq_len={config.max_seq_len} is out of range [1, 2048]."
        )

    if config.vocab_size < 1:
        issues.append(f"ERROR: vocab_size must be positive, got {config.vocab_size}.")

    # -- generation validation --
    if config.num_sequences < 1:
        issues.append(
            f"ERROR: num_sequences must be >= 1, got {config.num_sequences}."
        )

    if config.max_new_tokens < 1:
        issues.append(
            f"ERROR: max_new_tokens must be >= 1, got {config.max_new_tokens}."
        )

    if config.max_new_tokens > config.max_seq_len:
        issues.append(
            f"WARNING: max_new_tokens ({config.max_new_tokens}) exceeds "
            f"max_seq_len ({config.max_seq_len}). Generations will be "
            "truncated at max_seq_len."
        )

    if config.batch_size < 1:
        issues.append(f"ERROR: batch_size must be >= 1, got {config.batch_size}.")

    if config.batch_size > config.num_sequences:
        issues.append(
            f"WARNING: batch_size ({config.batch_size}) > num_sequences "
            f"({config.num_sequences}). Effective batch size will be "
            "clamped to num_sequences."
        )

    # -- cache validation --
    if config.cache_strategy not in SUPPORTED_CACHE_STRATEGIES:
        issues.append(
            f"ERROR: cache_strategy '{config.cache_strategy}' is not "
            f"supported. Choose from {SUPPORTED_CACHE_STRATEGIES}."
        )

    if config.max_cache_size_gb <= 0:
        issues.append(
            f"ERROR: max_cache_size_gb must be positive, got "
            f"{config.max_cache_size_gb}."
        )

    # -- output validation --
    for fmt in config.export_formats:
        if fmt not in SUPPORTED_EXPORT_FORMATS:
            issues.append(
                f"WARNING: export format '{fmt}' is not in the supported "
                f"list {SUPPORTED_EXPORT_FORMATS}."
            )

    # -- experiment validation --
    if config.seed < 0:
        issues.append(f"WARNING: seed is negative ({config.seed}). This is unusual.")

    if config.num_prompts < 1:
        issues.append(
            f"ERROR: num_prompts must be >= 1, got {config.num_prompts}."
        )

    for domain in config.task_domains:
        if domain not in DEFAULT_TASK_DOMAINS:
            issues.append(
                f"WARNING: task domain '{domain}' is not in the default "
                f"set {DEFAULT_TASK_DOMAINS}. Custom domains require a "
                "matching prompt provider."
            )

    # -- logging validation --
    if config.log_level not in SUPPORTED_LOG_LEVELS:
        issues.append(
            f"ERROR: log_level '{config.log_level}' is not valid. "
            f"Choose from {SUPPORTED_LOG_LEVELS}."
        )

    # -- nested experiment config --
    issues.extend(_validate_experiment(config.experiment))

    return issues


def _validate_experiment(exp: ExperimentConfig) -> List[str]:
    """Validate the nested ExperimentConfig."""
    issues: List[str] = []

    # Model sub-config
    m = exp.model
    if m.num_attention_heads < 1:
        issues.append(
            f"ERROR: experiment.model.num_attention_heads must be >= 1, "
            f"got {m.num_attention_heads}."
        )
    if m.hidden_size % m.num_attention_heads != 0:
        issues.append(
            f"ERROR: experiment.model.hidden_size ({m.hidden_size}) must "
            f"be divisible by num_attention_heads ({m.num_attention_heads})."
        )
    if m.num_layers < 1:
        issues.append(
            f"ERROR: experiment.model.num_layers must be >= 1, "
            f"got {m.num_layers}."
        )

    # Prompt source
    if exp.prompt_source == "file" and not exp.prompt_file:
        issues.append(
            "ERROR: experiment.prompt_source is 'file' but no "
            "prompt_file path was provided."
        )
    if exp.prompt_file and not Path(exp.prompt_file).exists():
        issues.append(
            f"WARNING: experiment.prompt_file '{exp.prompt_file}' does "
            "not exist (may be created later)."
        )

    # Cache sub-config
    c = exp.cache
    if c.ttl_hours < 0:
        issues.append(
            f"ERROR: experiment.cache.ttl_hours must be >= 0, "
            f"got {c.ttl_hours}."
        )

    # Metric sub-config
    mt = exp.metrics
    if mt.self_bleu_sample_size < 1:
        issues.append(
            f"ERROR: experiment.metrics.self_bleu_sample_size must be >= 1, "
            f"got {mt.self_bleu_sample_size}."
        )
    if not (0 < mt.semantic_similarity_threshold <= 1.0):
        issues.append(
            f"ERROR: experiment.metrics.semantic_similarity_threshold "
            f"must be in (0, 1], got {mt.semantic_similarity_threshold}."
        )
    if mt.max_metric_workers < 1:
        issues.append(
            f"ERROR: experiment.metrics.max_metric_workers must be >= 1, "
            f"got {mt.max_metric_workers}."
        )
    for order in mt.ngram_orders:
        if order < 1:
            issues.append(
                f"ERROR: n-gram order must be >= 1, got {order}."
            )

    # Sweep sub-config
    sw = exp.sweep
    for method in sw.methods:
        if method not in SUPPORTED_DECODING_METHODS:
            issues.append(
                f"WARNING: sweep method '{method}' is not in the supported "
                f"list {SUPPORTED_DECODING_METHODS}."
            )
    if sw.max_combinations < 1:
        issues.append(
            f"ERROR: experiment.sweep.max_combinations must be >= 1, "
            f"got {sw.max_combinations}."
        )
    t_min, t_max, t_step = sw.temperature_range
    if t_min <= 0:
        issues.append(
            f"ERROR: temperature_range min must be > 0, got {t_min}."
        )
    if t_max < t_min:
        issues.append(
            f"ERROR: temperature_range max ({t_max}) < min ({t_min})."
        )
    if t_step <= 0:
        issues.append(
            f"ERROR: temperature_range step must be > 0, got {t_step}."
        )

    return issues


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def _detect_format(path: Union[str, Path]) -> str:
    """Detect config file format from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".json":
        return "json"
    raise ValueError(
        f"Unsupported config file extension '{suffix}'. "
        "Use .json, .yaml, or .yml."
    )


def load_config(path: Union[str, Path]) -> ArenaConfig:
    """Load an ``ArenaConfig`` from a JSON or YAML file.

    Environment variable overrides are applied *after* loading the file.

    Args:
        path: Path to the configuration file.

    Returns:
        A fully populated ``ArenaConfig``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    fmt = _detect_format(path)
    raw_text = path.read_text(encoding="utf-8")

    if fmt == "json":
        data = json.loads(raw_text)
    else:
        # Lazy import so PyYAML is optional at module level
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(raw_text)

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a mapping, got {type(data).__name__}."
        )

    config = _dict_to_dataclass(ArenaConfig, data)
    apply_env_overrides(config)
    logger.info("Loaded config from %s", path)
    return config


def save_config(config: ArenaConfig, path: Union[str, Path]) -> Path:
    """Save an ``ArenaConfig`` to a JSON or YAML file.

    Parent directories are created automatically.

    Args:
        config: The configuration to save.
        path: Destination file path.

    Returns:
        The resolved ``Path`` the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _detect_format(path)
    data = _dataclass_to_dict(config)

    if fmt == "json":
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to save YAML config files. "
                "Install it with: pip install pyyaml"
            ) from exc
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    logger.info("Saved config to %s", path)
    return path.resolve()


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def get_default_config() -> ArenaConfig:
    """Return a fresh ``ArenaConfig`` with all default values.

    The returned config represents the recommended baseline for laptop
    CPU evaluation with GPT-2 124M + ONNX Runtime INT8 quantisation.

    Returns:
        A new ``ArenaConfig`` instance.
    """
    config = ArenaConfig()
    # Sync top-level fields with the nested experiment config
    config.experiment.model.model_name = config.model_name
    config.experiment.model.quantization = config.quantization
    config.experiment.model.max_seq_len = config.max_seq_len
    config.experiment.model.vocab_size = config.vocab_size
    config.experiment.model.device = config.device
    config.experiment.seed = config.seed
    config.experiment.num_prompts = config.num_prompts
    config.experiment.task_domains = list(config.task_domains)
    config.experiment.cache.cache_dir = config.cache_dir
    config.experiment.cache.max_cache_size_gb = config.max_cache_size_gb
    config.experiment.cache.cache_strategy = config.cache_strategy
    return config


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Values in *override* take precedence.  Nested dicts are merged
    recursively rather than replaced wholesale.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_configs(
    base: ArenaConfig,
    override: Union[ArenaConfig, Dict[str, Any]],
) -> ArenaConfig:
    """Merge *override* values on top of *base*, returning a new config.

    Only fields explicitly present in *override* replace those in *base*.
    Nested dataclass fields are merged recursively.

    Args:
        base: The base configuration.
        override: Either another ``ArenaConfig`` or a plain dict of
            overrides (may be partial).

    Returns:
        A new ``ArenaConfig`` with merged values.
    """
    base_dict = _dataclass_to_dict(base)
    if isinstance(override, dict):
        override_dict = override
    else:
        override_dict = _dataclass_to_dict(override)

    merged_dict = _deep_merge_dicts(base_dict, override_dict)
    return _dict_to_dataclass(ArenaConfig, merged_dict)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def config_summary(config: ArenaConfig) -> str:
    """Return a compact human-readable summary of the config.

    Useful for logging at the start of an experiment run.
    """
    lines = [
        "=" * 60,
        "  Diversity Decoding Arena — Configuration Summary",
        "=" * 60,
        f"  Model:           {config.model_name}",
        f"  Quantization:    {config.quantization}",
        f"  Device:          {config.device}",
        f"  Max seq len:     {config.max_seq_len}",
        f"  Vocab size:      {config.vocab_size}",
        "-" * 60,
        f"  Num sequences:   {config.num_sequences}",
        f"  Max new tokens:  {config.max_new_tokens}",
        f"  Batch size:      {config.batch_size}",
        "-" * 60,
        f"  Cache dir:       {config.cache_dir}",
        f"  Cache size:      {config.max_cache_size_gb} GB",
        f"  Cache strategy:  {config.cache_strategy}",
        "-" * 60,
        f"  Output dir:      {config.output_dir}",
        f"  Save logits:     {config.save_logits}",
        f"  Save embeddings: {config.save_embeddings}",
        f"  Export formats:  {', '.join(config.export_formats)}",
        "-" * 60,
        f"  Seed:            {config.seed}",
        f"  Num prompts:     {config.num_prompts}",
        f"  Task domains:    {', '.join(config.task_domains)}",
        "-" * 60,
        f"  Log level:       {config.log_level}",
        f"  Log file:        {config.log_file or '(none)'}",
        f"  Progress bar:    {config.enable_progress_bar}",
        "-" * 60,
        f"  Sweep methods:   {', '.join(config.experiment.sweep.methods)}",
        f"  Max combos:      {config.experiment.sweep.max_combinations}",
        f"  Metrics:         {', '.join(config.experiment.metrics.enabled_metrics)}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def parse_override_string(override_str: str) -> Dict[str, Any]:
    """Parse a ``key=value`` override string into a (possibly nested) dict.

    Dot-separated keys create nested dicts:
    ``"experiment.model.device=cuda"`` → ``{"experiment": {"model": {"device": "cuda"}}}``.

    Values are auto-cast to int, float, bool, or left as str.

    Args:
        override_str: A string like ``"key=value"``.

    Returns:
        A (possibly nested) dict suitable for :func:`merge_configs`.
    """
    if "=" not in override_str:
        raise ValueError(
            f"Override string must contain '=': got '{override_str}'"
        )
    key, raw_value = override_str.split("=", 1)
    value = _auto_cast(raw_value)
    parts = key.split(".")
    result: Dict[str, Any] = {}
    current = result
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def _auto_cast(value: str) -> Any:
    """Auto-cast a string to int, float, bool, list, or leave as str."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("none", "null"):
        return None
    # Comma-separated list
    if "," in value:
        return [_auto_cast(v.strip()) for v in value.split(",")]
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(
    config: ArenaConfig,
    overrides: List[str],
) -> ArenaConfig:
    """Apply a list of ``key=value`` overrides to *config*.

    Args:
        config: The base configuration.
        overrides: List of strings like ``["seed=123", "device=cuda"]``.

    Returns:
        A new ``ArenaConfig`` with overrides applied.
    """
    merged = config
    for ovr in overrides:
        patch = parse_override_string(ovr)
        merged = merge_configs(merged, patch)
    return merged


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------


def setup_logging(config: ArenaConfig) -> None:
    """Configure the Python logging subsystem based on *config*.

    Sets the root logger level and optionally adds a file handler.
    """
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if config.log_file:
        log_path = resolve_path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logger.debug("Logging configured: level=%s, file=%s", config.log_level, config.log_file)


# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------


def ensure_directories(config: ArenaConfig) -> None:
    """Create cache and output directories if they do not exist."""
    for dir_path in (config.cache_dir, config.output_dir):
        resolved = resolve_path(dir_path)
        resolved.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", resolved)
