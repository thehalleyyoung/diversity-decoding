"""
Core type definitions for the Diversity Decoding Arena.

A framework for comparing diversity-promoting decoding algorithms. Provides
strongly-typed primitives for tokens, logit distributions, generation results,
metrics, Pareto analysis, quality-diversity archives, and kernel matrices.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TokenID = int
LogitVector = np.ndarray
EmbeddingVector = np.ndarray

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecodingStrategy(Enum):
    """Enumeration of supported decoding strategies."""

    GREEDY = auto()
    TEMPERATURE = auto()
    TOP_K = auto()
    TOP_P = auto()
    TYPICAL = auto()
    MIROSTAT = auto()
    BEAM_SEARCH = auto()
    DIVERSE_BEAM_SEARCH = auto()
    CONTRASTIVE = auto()
    DPP = auto()
    VENDI = auto()
    STOCHASTIC_BEAM = auto()
    EPSILON_SAMPLING = auto()
    ETA_SAMPLING = auto()
    LOCALLY_TYPICAL = auto()

    def __repr__(self) -> str:
        return f"DecodingStrategy.{self.name}"


class MetricType(Enum):
    """Whether a metric measures diversity or quality."""

    DIVERSITY = auto()
    QUALITY = auto()

    def __repr__(self) -> str:
        return f"MetricType.{self.name}"


class TaskDomain(Enum):
    """Domain / task family for an experiment."""

    OPEN_ENDED_GENERATION = auto()
    STORY_GENERATION = auto()
    CODE_GENERATION = auto()
    DIALOGUE = auto()
    SUMMARIZATION = auto()
    TRANSLATION = auto()
    MATH = auto()
    REASONING = auto()
    QUESTION_ANSWERING = auto()

    def __repr__(self) -> str:
        return f"TaskDomain.{self.name}"


class CacheStrategy(Enum):
    """Caching strategy for generation or metric computation."""

    NONE = auto()
    MEMORY = auto()
    DISK = auto()
    REDIS = auto()

    def __repr__(self) -> str:
        return f"CacheStrategy.{self.name}"


class ExportFormat(Enum):
    """Supported serialization / export formats."""

    JSON = auto()
    CSV = auto()
    PARQUET = auto()
    PICKLE = auto()
    LATEX = auto()

    def __repr__(self) -> str:
        return f"ExportFormat.{self.name}"


# ---------------------------------------------------------------------------
# Token & TokenSequence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A single token with its identifier, surface text, and log-probability.

    Attributes:
        token_id: Integer identifier in the vocabulary.
        text: Surface-form string for the token.
        log_prob: Log-probability assigned by the model.
    """

    token_id: int
    text: str
    log_prob: float

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {"token_id": self.token_id, "text": self.text, "log_prob": self.log_prob}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Token":
        """Deserialize from a dictionary."""
        return cls(token_id=int(d["token_id"]), text=str(d["text"]), log_prob=float(d["log_prob"]))

    def __repr__(self) -> str:
        return f"Token(id={self.token_id}, text={self.text!r}, lp={self.log_prob:.4f})"


@dataclass
class TokenSequence:
    """An ordered sequence of :class:`Token` objects.

    Supports len, indexing, slicing, concatenation, and convenient accessors
    for the underlying text, token-id, and log-prob views.

    Attributes:
        tokens: The list of Token objects forming the sequence.
    """

    tokens: List[Token] = field(default_factory=list)

    # -- Accessors -----------------------------------------------------------

    def text(self) -> str:
        """Return the full decoded text (concatenation of token surfaces)."""
        return "".join(t.text for t in self.tokens)

    def token_ids(self) -> List[int]:
        """Return the list of token IDs."""
        return [t.token_id for t in self.tokens]

    def log_probs(self) -> List[float]:
        """Return the per-token log-probabilities."""
        return [t.log_prob for t in self.tokens]

    def total_log_prob(self) -> float:
        """Sum of all token log-probabilities (sequence log-prob)."""
        return sum(t.log_prob for t in self.tokens)

    def mean_log_prob(self) -> float:
        """Mean per-token log-probability."""
        if not self.tokens:
            return 0.0
        return self.total_log_prob() / len(self.tokens)

    def perplexity(self) -> float:
        """Per-token perplexity of the sequence."""
        if not self.tokens:
            return float("inf")
        return float(np.exp(-self.mean_log_prob()))

    # -- Dunder protocols ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, key: Union[int, slice]) -> Union[Token, "TokenSequence"]:
        if isinstance(key, slice):
            return TokenSequence(tokens=self.tokens[key])
        return self.tokens[key]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __add__(self, other: "TokenSequence") -> "TokenSequence":
        if not isinstance(other, TokenSequence):
            return NotImplemented
        return TokenSequence(tokens=self.tokens + other.tokens)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenSequence):
            return NotImplemented
        return self.tokens == other.tokens

    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __repr__(self) -> str:
        preview = self.text()[:60]
        return f"TokenSequence(len={len(self)}, text={preview!r})"

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens": [t.to_dict() for t in self.tokens]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenSequence":
        return cls(tokens=[Token.from_dict(td) for td in d["tokens"]])

    # -- Convenience builders ------------------------------------------------

    @classmethod
    def from_text_and_ids(
        cls,
        texts: List[str],
        token_ids: List[int],
        log_probs: Optional[List[float]] = None,
    ) -> "TokenSequence":
        """Build a TokenSequence from parallel lists.

        Args:
            texts: Per-token surface strings.
            token_ids: Per-token vocabulary IDs.
            log_probs: Optional per-token log-probabilities (default 0.0).
        """
        if log_probs is None:
            log_probs = [0.0] * len(texts)
        if not (len(texts) == len(token_ids) == len(log_probs)):
            raise ValueError("texts, token_ids, and log_probs must have the same length")
        tokens = [Token(tid, txt, lp) for tid, txt, lp in zip(token_ids, texts, log_probs)]
        return cls(tokens=tokens)


# ---------------------------------------------------------------------------
# LogitDistribution
# ---------------------------------------------------------------------------


class LogitDistribution:
    """Wrapper around a raw logit vector providing sampling and filtering utilities.

    The underlying logits are stored as a 1-D numpy array.  All mutation methods
    return *new* ``LogitDistribution`` instances (the object is effectively
    immutable after construction).

    Args:
        logits: 1-D array of raw (unnormalized) logits.
    """

    def __init__(self, logits: np.ndarray) -> None:
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D, got shape {logits.shape}")
        self._logits: np.ndarray = logits.astype(np.float64, copy=True)

    # -- Properties ----------------------------------------------------------

    @property
    def logits(self) -> np.ndarray:
        """Return a copy of the raw logit vector."""
        return self._logits.copy()

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (length of the logit vector)."""
        return len(self._logits)

    # -- Core transformations ------------------------------------------------

    def softmax(self) -> np.ndarray:
        """Compute the softmax probability distribution.

        Uses the numerically-stable log-sum-exp trick.
        """
        shifted = self._logits - self._logits.max()
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum()

    def log_probs(self) -> np.ndarray:
        """Return log-probabilities (log-softmax)."""
        shifted = self._logits - self._logits.max()
        log_sum_exp = np.log(np.exp(shifted).sum())
        return shifted - log_sum_exp

    def entropy(self) -> float:
        """Shannon entropy of the distribution (in nats)."""
        probs = self.softmax()
        # Avoid log(0)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def temperature_scale(self, temperature: float) -> "LogitDistribution":
        """Return a new distribution with logits scaled by *1 / temperature*.

        Args:
            temperature: Positive temperature value.  Values < 1 sharpen;
                values > 1 flatten.

        Raises:
            ValueError: If *temperature* is not positive.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        return LogitDistribution(self._logits / temperature)

    # -- Filtering / masking -------------------------------------------------

    def top_k(self, k: int) -> "LogitDistribution":
        """Keep only the *k* highest-probability tokens; mask the rest to -inf.

        Args:
            k: Number of tokens to retain. Must be >= 1.
        """
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        k = min(k, self.vocab_size)
        threshold = np.sort(self._logits)[-k]
        new_logits = np.where(self._logits >= threshold, self._logits, -np.inf)
        return LogitDistribution(new_logits)

    def top_p(self, p: float) -> "LogitDistribution":
        """Nucleus (top-p) filtering.

        Retain the smallest set of tokens whose cumulative probability >= *p*.

        Args:
            p: Cumulative probability threshold in (0, 1].
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        probs = self.softmax()
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        # Find the cutoff index
        cutoff = int(np.searchsorted(cumulative, p, side="right")) + 1
        cutoff = min(cutoff, self.vocab_size)
        keep = set(sorted_indices[:cutoff].tolist())
        new_logits = np.full_like(self._logits, -np.inf)
        for idx in keep:
            new_logits[idx] = self._logits[idx]
        return LogitDistribution(new_logits)

    def nucleus_mask(self, p: float) -> np.ndarray:
        """Return a boolean mask of tokens inside the nucleus (top-p set).

        Args:
            p: Cumulative probability threshold in (0, 1].
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        probs = self.softmax()
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumulative, p, side="right")) + 1
        cutoff = min(cutoff, self.vocab_size)
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[sorted_indices[:cutoff]] = True
        return mask

    def typical_mask(self, threshold: float) -> np.ndarray:
        """Return a boolean mask for *locally typical* sampling.

        Tokens are typical if their negative log-probability is within
        *threshold* of the distribution's entropy.

        Args:
            threshold: Allowed deviation from the entropy (in nats).
        """
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        lp = self.log_probs()
        ent = self.entropy()
        deviation = np.abs(-lp - ent)
        return deviation <= threshold

    def filter_tokens(self, mask: np.ndarray) -> "LogitDistribution":
        """Zero-out (set to -inf) tokens where *mask* is ``False``.

        Args:
            mask: Boolean array of shape ``(vocab_size,)``.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self._logits.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match logits shape {self._logits.shape}"
            )
        new_logits = np.where(mask, self._logits, -np.inf)
        return LogitDistribution(new_logits)

    def renormalize(self) -> "LogitDistribution":
        """Return a copy with logits shifted so that log-sum-exp == 0.

        This is equivalent to converting to log-probabilities.
        """
        return LogitDistribution(self.log_probs())

    # -- Sampling ------------------------------------------------------------

    def argmax(self) -> int:
        """Return the token ID with the highest logit."""
        return int(np.argmax(self._logits))

    def sample(self, n: int = 1, *, replace: bool = True) -> np.ndarray:
        """Sample *n* token IDs from the distribution.

        Args:
            n: Number of samples.
            replace: Whether to sample with replacement.

        Returns:
            1-D int array of sampled token IDs.
        """
        probs = self.softmax()
        # Ensure valid distribution (handle -inf logits → 0 probability)
        total = probs.sum()
        if total == 0:
            raise ValueError("Cannot sample from a distribution with all-zero probabilities")
        probs = probs / total
        return np.random.choice(self.vocab_size, size=n, replace=replace, p=probs)

    # -- Numpy interop -------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the logit array."""
        return self._logits.copy()

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "LogitDistribution":
        """Construct from a numpy array (1-D)."""
        return cls(np.asarray(arr, dtype=np.float64))

    # -- Dunder protocols ----------------------------------------------------

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        ent = self.entropy()
        top = int(np.argmax(self._logits))
        return f"LogitDistribution(vocab={self.vocab_size}, entropy={ent:.3f}, argmax={top})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogitDistribution):
            return NotImplemented
        return np.array_equal(self._logits, other._logits)

    def __hash__(self) -> int:
        return hash(self._logits.tobytes())


# ---------------------------------------------------------------------------
# GenerationResult & GenerationSet
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """A single generated output along with its provenance and quality score.

    Attributes:
        sequence: The generated token sequence.
        prompt: The input prompt that produced this generation.
        algorithm: Name of the decoding algorithm used.
        config: Configuration dictionary for the algorithm.
        metadata: Auxiliary data (timing, memory, device, etc.).
        score: Scalar quality score (e.g. from a reward model).
    """

    sequence: TokenSequence
    prompt: str
    algorithm: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    # -- Convenience ---------------------------------------------------------

    def text(self) -> str:
        """Generated text."""
        return self.sequence.text()

    def num_tokens(self) -> int:
        """Number of tokens in the generated sequence."""
        return len(self.sequence)

    def tokens_per_second(self) -> Optional[float]:
        """Throughput if timing metadata is available."""
        elapsed = self.metadata.get("elapsed_seconds")
        if elapsed is not None and elapsed > 0:
            return len(self.sequence) / elapsed
        return None

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence.to_dict(),
            "prompt": self.prompt,
            "algorithm": self.algorithm,
            "config": self.config,
            "metadata": self.metadata,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationResult":
        return cls(
            sequence=TokenSequence.from_dict(d["sequence"]),
            prompt=d["prompt"],
            algorithm=d["algorithm"],
            config=d.get("config", {}),
            metadata=d.get("metadata", {}),
            score=float(d.get("score", 0.0)),
        )

    def __repr__(self) -> str:
        preview = self.text()[:40]
        return (
            f"GenerationResult(algo={self.algorithm!r}, "
            f"score={self.score:.3f}, text={preview!r})"
        )


@dataclass
class GenerationSet:
    """A collection of :class:`GenerationResult` objects sharing a common prompt
    and algorithm.

    Provides convenience accessors for diversity analysis and pairwise
    comparisons.

    Attributes:
        results: Individual generation results.
        prompt: The shared input prompt.
        algorithm: The decoding algorithm that produced these results.
    """

    results: List[GenerationResult] = field(default_factory=list)
    prompt: str = ""
    algorithm: str = ""

    # -- Accessors -----------------------------------------------------------

    def texts(self) -> List[str]:
        """Return the generated text for each result."""
        return [r.text() for r in self.results]

    def sequences(self) -> List[TokenSequence]:
        """Return the :class:`TokenSequence` for each result."""
        return [r.sequence for r in self.results]

    def scores(self) -> List[float]:
        """Return per-result quality scores."""
        return [r.score for r in self.results]

    def num_sequences(self) -> int:
        """Number of generated sequences in the set."""
        return len(self.results)

    def unique_texts(self) -> List[str]:
        """Deduplicated list of generated texts (order-preserving)."""
        seen: set[str] = set()
        out: list[str] = []
        for t in self.texts():
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def diversity_ratio(self) -> float:
        """Fraction of unique texts: ``|unique| / |total|``.

        Returns 0.0 when the set is empty.
        """
        n = self.num_sequences()
        if n == 0:
            return 0.0
        return len(self.unique_texts()) / n

    def pairwise_iterator(self) -> Iterator[Tuple[GenerationResult, GenerationResult]]:
        """Iterate over all unordered pairs of results."""
        yield from combinations(self.results, 2)

    def mean_score(self) -> float:
        """Arithmetic mean of quality scores."""
        if not self.results:
            return 0.0
        return float(np.mean(self.scores()))

    def best_result(self) -> Optional[GenerationResult]:
        """Return the result with the highest score, or ``None`` if empty."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.score)

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "prompt": self.prompt,
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationSet":
        return cls(
            results=[GenerationResult.from_dict(rd) for rd in d["results"]],
            prompt=d.get("prompt", ""),
            algorithm=d.get("algorithm", ""),
        )

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return (
            f"GenerationSet(algo={self.algorithm!r}, n={self.num_sequences()}, "
            f"diversity={self.diversity_ratio():.2f})"
        )


# ---------------------------------------------------------------------------
# MetricResult & MetricSuite
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Result of a single metric evaluation.

    Attributes:
        metric_name: Human-readable name (e.g. ``"self-bleu"``).
        value: Scalar summary value.
        confidence_interval: Optional (lower, upper) 95 % CI.
        per_sample_values: Optional per-sample breakdown.
        metadata: Extra information (e.g. parameters used).
    """

    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    per_sample_values: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Helpers -------------------------------------------------------------

    def ci_width(self) -> Optional[float]:
        """Width of the confidence interval, if available."""
        if self.confidence_interval is None:
            return None
        lo, hi = self.confidence_interval
        return hi - lo

    def std(self) -> Optional[float]:
        """Standard deviation of per-sample values, if available."""
        if self.per_sample_values is None:
            return None
        return float(np.std(self.per_sample_values))

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata,
        }
        if self.confidence_interval is not None:
            d["confidence_interval"] = list(self.confidence_interval)
        if self.per_sample_values is not None:
            d["per_sample_values"] = self.per_sample_values
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricResult":
        ci = d.get("confidence_interval")
        if ci is not None:
            ci = (float(ci[0]), float(ci[1]))
        return cls(
            metric_name=d["metric_name"],
            value=float(d["value"]),
            confidence_interval=ci,
            per_sample_values=d.get("per_sample_values"),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        ci_str = ""
        if self.confidence_interval is not None:
            ci_str = f", ci=[{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]"
        return f"MetricResult({self.metric_name!r}, {self.value:.4f}{ci_str})"


@dataclass
class MetricSuite:
    """A named collection of :class:`MetricResult` values.

    Attributes:
        results: Mapping from metric name to its result.
    """

    results: Dict[str, MetricResult] = field(default_factory=dict)

    # -- Accessors -----------------------------------------------------------

    def __getitem__(self, name: str) -> MetricResult:
        return self.results[name]

    def __contains__(self, name: str) -> bool:
        return name in self.results

    def names(self) -> List[str]:
        return list(self.results.keys())

    def values_dict(self) -> Dict[str, float]:
        """Return ``{name: scalar_value}`` for every metric."""
        return {k: v.value for k, v in self.results.items()}

    # -- Comparison ----------------------------------------------------------

    def compare_with(self, other: "MetricSuite") -> Dict[str, Dict[str, float]]:
        """Compare two suites on shared metrics.

        Returns a dict keyed by metric name, each containing ``self``, ``other``,
        and ``delta`` (other − self) values.
        """
        shared = set(self.results) & set(other.results)
        out: Dict[str, Dict[str, float]] = {}
        for name in sorted(shared):
            s = self.results[name].value
            o = other.results[name].value
            out[name] = {"self": s, "other": o, "delta": o - s}
        return out

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {name: mr.to_dict() for name, mr in self.results.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricSuite":
        return cls(results={name: MetricResult.from_dict(v) for name, v in d.items()})

    def to_dataframe(self) -> Any:
        """Convert to a pandas DataFrame (requires pandas).

        Returns:
            A ``pandas.DataFrame`` with columns ``metric``, ``value``,
            ``ci_lower``, ``ci_upper``.
        """
        import pandas as pd  # type: ignore

        rows: list[dict[str, Any]] = []
        for name, mr in self.results.items():
            row: dict[str, Any] = {"metric": name, "value": mr.value}
            if mr.confidence_interval is not None:
                row["ci_lower"] = mr.confidence_interval[0]
                row["ci_upper"] = mr.confidence_interval[1]
            else:
                row["ci_lower"] = None
                row["ci_upper"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        entries = ", ".join(f"{k}={v.value:.4f}" for k, v in self.results.items())
        return f"MetricSuite({entries})"


# ---------------------------------------------------------------------------
# AlgorithmConfig
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmConfig:
    """Fully-describes a decoding algorithm configuration.

    Two configs with the same content hash are considered identical regardless
    of insertion order.

    Attributes:
        name: Name of the decoding algorithm.
        params: Algorithm-specific hyper-parameters.
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    # -- Content-addressed hashing -------------------------------------------

    def content_hash(self) -> str:
        """Deterministic SHA-256 hex digest based on name + sorted params."""
        canonical = json.dumps({"name": self.name, "params": self.params}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def short_hash(self) -> str:
        """First 12 hex characters of :meth:`content_hash`."""
        return self.content_hash()[:12]

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlgorithmConfig":
        return cls(name=d["name"], params=d.get("params", {}))

    # -- Dunder --------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgorithmConfig):
            return NotImplemented
        return self.name == other.name and self.params == other.params

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __repr__(self) -> str:
        return f"AlgorithmConfig({self.name!r}, hash={self.short_hash()})"


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Bundles an algorithm configuration with its generation set and metrics.

    Attributes:
        config: The algorithm configuration used.
        generation_set: All generated outputs.
        metrics: Evaluated metrics.
        task_domain: The task family.
        timestamp: Unix timestamp of the experiment.
    """

    config: AlgorithmConfig
    generation_set: GenerationSet
    metrics: MetricSuite
    task_domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION
    timestamp: float = field(default_factory=time.time)

    # -- Convenience ---------------------------------------------------------

    def quality_score(self) -> Optional[float]:
        """Return the ``quality`` metric value if present."""
        if "quality" in self.metrics:
            return self.metrics["quality"].value
        return self.generation_set.mean_score()

    def diversity_score(self) -> Optional[float]:
        """Return the ``diversity`` metric value if present."""
        if "diversity" in self.metrics:
            return self.metrics["diversity"].value
        return self.generation_set.diversity_ratio()

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "generation_set": self.generation_set.to_dict(),
            "metrics": self.metrics.to_dict(),
            "task_domain": self.task_domain.name,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            config=AlgorithmConfig.from_dict(d["config"]),
            generation_set=GenerationSet.from_dict(d["generation_set"]),
            metrics=MetricSuite.from_dict(d["metrics"]),
            task_domain=TaskDomain[d.get("task_domain", "OPEN_ENDED_GENERATION")],
            timestamp=float(d.get("timestamp", 0.0)),
        )

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(config={self.config!r}, "
            f"n={self.generation_set.num_sequences()}, "
            f"metrics={self.metrics.names()})"
        )


# ---------------------------------------------------------------------------
# ParetoPoint & ParetoFrontier
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """A single point in objective space linked to its algorithm configuration.

    Attributes:
        objectives: Mapping from objective name to value (higher is better).
        config: The algorithm configuration that produced this point.
        is_dominated: Whether the point is dominated by another point on the
            frontier.
    """

    objectives: Dict[str, float]
    config: AlgorithmConfig
    is_dominated: bool = False

    def dominates(self, other: "ParetoPoint") -> bool:
        """Return ``True`` if *self* Pareto-dominates *other*.

        Domination: self is >= other in all objectives and strictly > in at
        least one.
        """
        shared = set(self.objectives) & set(other.objectives)
        if not shared:
            return False
        at_least_as_good = all(
            self.objectives[k] >= other.objectives[k] for k in shared
        )
        strictly_better = any(
            self.objectives[k] > other.objectives[k] for k in shared
        )
        return at_least_as_good and strictly_better

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": self.objectives,
            "config": self.config.to_dict(),
            "is_dominated": self.is_dominated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParetoPoint":
        return cls(
            objectives=d["objectives"],
            config=AlgorithmConfig.from_dict(d["config"]),
            is_dominated=d.get("is_dominated", False),
        )

    def __repr__(self) -> str:
        obj_str = ", ".join(f"{k}={v:.3f}" for k, v in self.objectives.items())
        tag = " [dominated]" if self.is_dominated else ""
        return f"ParetoPoint({obj_str}{tag})"


class ParetoFrontier:
    """A collection of :class:`ParetoPoint` objects with Pareto analysis
    utilities.

    After construction (or after calling :meth:`add`), invoke
    :meth:`compute_frontier` to update domination labels and extract the
    non-dominated set.

    Args:
        points: Initial list of Pareto points.
    """

    def __init__(self, points: Optional[List[ParetoPoint]] = None) -> None:
        self.points: List[ParetoPoint] = list(points or [])
        self._frontier_computed = False

    # -- Mutation ------------------------------------------------------------

    def add(self, point: ParetoPoint) -> None:
        """Add a point and invalidate the cached frontier."""
        self.points.append(point)
        self._frontier_computed = False

    # -- Frontier computation ------------------------------------------------

    def compute_frontier(self) -> List[ParetoPoint]:
        """Compute the Pareto frontier and mark dominated points.

        Returns:
            The list of non-dominated points.
        """
        for p in self.points:
            p.is_dominated = False
        for i, pi in enumerate(self.points):
            for j, pj in enumerate(self.points):
                if i != j and pj.dominates(pi):
                    pi.is_dominated = True
                    break
        self._frontier_computed = True
        return [p for p in self.points if not p.is_dominated]

    def frontier_points(self) -> List[ParetoPoint]:
        """Return non-dominated points (computes frontier if needed)."""
        if not self._frontier_computed:
            self.compute_frontier()
        return [p for p in self.points if not p.is_dominated]

    def dominated_points(self) -> List[ParetoPoint]:
        """Return dominated points (computes frontier if needed)."""
        if not self._frontier_computed:
            self.compute_frontier()
        return [p for p in self.points if p.is_dominated]

    # -- Hypervolume ---------------------------------------------------------

    def hypervolume(self, reference: Dict[str, float]) -> float:
        """Compute the hypervolume indicator for 2-D objective spaces.

        Uses the standard sweep-line algorithm for two objectives.

        Args:
            reference: The reference (anti-ideal) point; all frontier points
                must dominate it for the indicator to be meaningful.

        Returns:
            Hypervolume scalar.

        Raises:
            ValueError: If the objective space is not 2-D.
        """
        frontier = self.frontier_points()
        if not frontier:
            return 0.0

        obj_names = sorted(reference.keys())
        if len(obj_names) != 2:
            raise ValueError("hypervolume is only implemented for 2-D spaces")

        k0, k1 = obj_names
        # Sort frontier by first objective ascending
        pts = sorted(
            [(p.objectives[k0], p.objectives[k1]) for p in frontier],
            key=lambda t: t[0],
        )
        hv = 0.0
        prev_y = reference[k1]
        for x, y in pts:
            if y > prev_y:
                hv += (x - reference[k0]) * (y - prev_y)
                prev_y = y
        # Final rectangle to the last x
        # Use simple rectangle sum
        # Recompute using standard method for correctness
        hv = 0.0
        pts_sorted = sorted(pts, key=lambda t: t[0])
        prev_y = reference[k1]
        for i, (x, y) in enumerate(pts_sorted):
            width = x - reference[k0] if i == 0 else x - pts_sorted[i - 1][0]
            max_y_rest = max(p[1] for p in pts_sorted[i:])
            hv += width * (max_y_rest - reference[k1])

        return float(hv)

    # -- Plotting data -------------------------------------------------------

    def plot_data(self) -> Dict[str, Any]:
        """Return data suitable for plotting the frontier.

        Returns:
            Dictionary with ``frontier`` and ``dominated`` keys, each
            containing a list of ``{objectives, config_hash}`` dicts.
        """
        if not self._frontier_computed:
            self.compute_frontier()
        make_entry = lambda p: {
            "objectives": p.objectives,
            "config_hash": p.config.short_hash(),
        }
        return {
            "frontier": [make_entry(p) for p in self.frontier_points()],
            "dominated": [make_entry(p) for p in self.dominated_points()],
        }

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"points": [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParetoFrontier":
        return cls(points=[ParetoPoint.from_dict(pd) for pd in d["points"]])

    # -- Dunder --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        n_front = len(self.frontier_points())
        return f"ParetoFrontier(total={len(self)}, frontier={n_front})"


# ---------------------------------------------------------------------------
# BehaviorDescriptor & QDArchive
# ---------------------------------------------------------------------------


@dataclass
class BehaviorDescriptor:
    """A descriptor vector locating a solution in behavior space.

    Used in Quality-Diversity (QD) algorithms such as MAP-Elites.

    Attributes:
        values: Numeric descriptor vector.
        labels: Human-readable names for each dimension.
    """

    values: np.ndarray
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.labels and len(self.labels) != len(self.values):
            raise ValueError(
                f"labels length {len(self.labels)} != values length {len(self.values)}"
            )

    @property
    def dim(self) -> int:
        """Dimensionality of the descriptor."""
        return len(self.values)

    def distance_to(self, other: "BehaviorDescriptor", metric: str = "euclidean") -> float:
        """Compute distance to *other* descriptor.

        Args:
            other: Another behavior descriptor of the same dimensionality.
            metric: ``"euclidean"`` (default) or ``"cosine"``.

        Raises:
            ValueError: On dimensionality mismatch or unknown metric.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        if metric == "euclidean":
            return float(np.linalg.norm(self.values - other.values))
        if metric == "cosine":
            dot = np.dot(self.values, other.values)
            norm_a = np.linalg.norm(self.values)
            norm_b = np.linalg.norm(other.values)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return float(1.0 - dot / (norm_a * norm_b))
        raise ValueError(f"Unknown metric: {metric!r}")

    def cell_index(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        num_cells_per_dim: Union[int, Sequence[int]] = 10,
    ) -> int:
        """Map this descriptor to a cell index in a uniform tessellation.

        Args:
            lower_bounds: Per-dimension lower bounds.
            upper_bounds: Per-dimension upper bounds.
            num_cells_per_dim: Number of cells along each dimension (int for
                uniform, or a sequence for per-dimension).

        Returns:
            A flattened integer cell index.
        """
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        if isinstance(num_cells_per_dim, int):
            cells = np.full(self.dim, num_cells_per_dim, dtype=int)
        else:
            cells = np.asarray(num_cells_per_dim, dtype=int)

        # Clamp to bounds
        clamped = np.clip(self.values, lower_bounds, upper_bounds)
        normalized = (clamped - lower_bounds) / np.maximum(upper_bounds - lower_bounds, 1e-12)
        indices = np.minimum((normalized * cells).astype(int), cells - 1)

        # Row-major flattening
        flat = 0
        for i, idx in enumerate(indices):
            flat = flat * int(cells[i]) + int(idx)
        return flat

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"values": self.values.tolist(), "labels": self.labels}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BehaviorDescriptor":
        return cls(values=np.array(d["values"]), labels=d.get("labels", []))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BehaviorDescriptor):
            return NotImplemented
        return np.array_equal(self.values, other.values) and self.labels == other.labels

    def __hash__(self) -> int:
        return hash((self.values.tobytes(), tuple(self.labels)))

    def __repr__(self) -> str:
        return f"BehaviorDescriptor(dim={self.dim}, labels={self.labels})"


@dataclass
class QDArchive:
    """Quality-Diversity archive (MAP-Elites style).

    Stores the best-scoring :class:`GenerationResult` for each cell in a
    tessellated behavior space.

    Attributes:
        cells: Mapping from cell index to the best result occupying that cell.
        descriptors: Parallel mapping from cell index to the behavior descriptor.
        lower_bounds: Per-dimension lower bounds for the tessellation.
        upper_bounds: Per-dimension upper bounds for the tessellation.
        num_cells_per_dim: Number of cells along each dimension.
    """

    cells: Dict[int, GenerationResult] = field(default_factory=dict)
    descriptors: Dict[int, BehaviorDescriptor] = field(default_factory=dict)
    lower_bounds: np.ndarray = field(default_factory=lambda: np.zeros(2))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.ones(2))
    num_cells_per_dim: int = 10

    @property
    def total_cells(self) -> int:
        """Total number of cells in the tessellation."""
        return int(self.num_cells_per_dim ** len(self.lower_bounds))

    # -- Archive operations --------------------------------------------------

    def add(
        self, result: GenerationResult, descriptor: BehaviorDescriptor
    ) -> bool:
        """Attempt to insert *result* into the archive.

        The result is placed in the cell corresponding to *descriptor*.  If the
        cell is empty or the new result has a higher score, the cell is updated.

        Args:
            result: The generation result to insert.
            descriptor: Its behavior descriptor.

        Returns:
            ``True`` if the archive was updated.
        """
        cell = descriptor.cell_index(
            self.lower_bounds, self.upper_bounds, self.num_cells_per_dim
        )
        if cell not in self.cells or result.score > self.cells[cell].score:
            self.cells[cell] = result
            self.descriptors[cell] = descriptor
            return True
        return False

    def coverage(self) -> float:
        """Fraction of cells that are occupied."""
        total = self.total_cells
        if total == 0:
            return 0.0
        return len(self.cells) / total

    def best_per_cell(self) -> Dict[int, GenerationResult]:
        """Return the archive contents (each cell already stores the best)."""
        return dict(self.cells)

    def all_results(self) -> List[GenerationResult]:
        """Flat list of all results in the archive."""
        return list(self.cells.values())

    def qd_score(self) -> float:
        """QD-score: sum of scores across all occupied cells."""
        return sum(r.score for r in self.cells.values())

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": {str(k): v.to_dict() for k, v in self.cells.items()},
            "descriptors": {str(k): v.to_dict() for k, v in self.descriptors.items()},
            "lower_bounds": self.lower_bounds.tolist(),
            "upper_bounds": self.upper_bounds.tolist(),
            "num_cells_per_dim": self.num_cells_per_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QDArchive":
        cells = {int(k): GenerationResult.from_dict(v) for k, v in d["cells"].items()}
        descriptors = {
            int(k): BehaviorDescriptor.from_dict(v) for k, v in d.get("descriptors", {}).items()
        }
        return cls(
            cells=cells,
            descriptors=descriptors,
            lower_bounds=np.array(d["lower_bounds"]),
            upper_bounds=np.array(d["upper_bounds"]),
            num_cells_per_dim=int(d["num_cells_per_dim"]),
        )

    def __repr__(self) -> str:
        return (
            f"QDArchive(occupied={len(self.cells)}/{self.total_cells}, "
            f"coverage={self.coverage():.2%}, qd_score={self.qd_score():.3f})"
        )


# ---------------------------------------------------------------------------
# KernelMatrix
# ---------------------------------------------------------------------------


class KernelMatrix:
    """A symmetric positive-semidefinite similarity matrix with spectral and
    DPP utilities.

    Args:
        matrix: 2-D square numpy array (or will be converted to one).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
        self._matrix = matrix

    @property
    def n(self) -> int:
        """Number of items (rows/columns)."""
        return self._matrix.shape[0]

    @property
    def matrix(self) -> np.ndarray:
        """Return a copy of the underlying matrix."""
        return self._matrix.copy()

    # -- Spectral methods ----------------------------------------------------

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues in descending order."""
        evals = np.linalg.eigvalsh(self._matrix)
        return np.sort(evals)[::-1]

    def vendi_score(self) -> float:
        """Compute the Vendi Score (exponential of matrix entropy).

        The Vendi Score is ``exp(H)`` where
        ``H = -sum(lambda_i * log(lambda_i))`` for normalized eigenvalues
        ``lambda_i``.  It measures effective diversity.
        """
        evals = self.eigenvalues()
        evals = evals[evals > 0]
        total = evals.sum()
        if total == 0:
            return 0.0
        p = evals / total
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))

    # -- DPP sampling --------------------------------------------------------

    def dpp_sample(self, k: int, rng: Optional[np.random.Generator] = None) -> List[int]:
        """Sample a *k*-DPP subset using the eigendecomposition method.

        This implements the elementary DPP sampling algorithm (Kulesza &
        Taskar, 2012).

        Args:
            k: Desired subset size.
            rng: Optional numpy random generator for reproducibility.

        Returns:
            List of selected indices (length *k*).
        """
        if k <= 0 or k > self.n:
            raise ValueError(f"k must be in [1, {self.n}], got {k}")

        rng = rng or np.random.default_rng()

        # Eigendecomposition
        evals, evecs = np.linalg.eigh(self._matrix)
        # Ensure non-negative eigenvalues
        evals = np.maximum(evals, 0.0)

        # Phase 1: select eigenvectors with probability lambda_i / (lambda_i + 1)
        # For k-DPP we use a fixed-size variant: greedily select k eigenvectors
        # weighted by eigenvalue magnitude.
        indices_by_eval = np.argsort(-evals)
        selected_evecs_idx: List[int] = []
        for idx in indices_by_eval:
            if len(selected_evecs_idx) >= k:
                break
            prob = evals[idx] / (evals[idx] + 1.0) if evals[idx] > 0 else 0.0
            if rng.random() < prob:
                selected_evecs_idx.append(int(idx))

        # If we didn't get enough, fill greedily by largest eigenvalue
        for idx in indices_by_eval:
            if len(selected_evecs_idx) >= k:
                break
            if idx not in selected_evecs_idx:
                selected_evecs_idx.append(int(idx))

        V = evecs[:, selected_evecs_idx[:k]].copy()  # (n, k)

        # Phase 2: iteratively sample items
        selected: List[int] = []
        remaining = list(range(self.n))

        for _ in range(k):
            if V.shape[1] == 0:
                break
            # Probability proportional to squared projection length
            probs = np.sum(V[remaining] ** 2, axis=1)
            total = probs.sum()
            if total < 1e-15:
                # Fall back to uniform
                choice_idx = int(rng.integers(len(remaining)))
            else:
                probs /= total
                choice_idx = int(rng.choice(len(remaining), p=probs))

            chosen = remaining[choice_idx]
            selected.append(chosen)
            remaining.pop(choice_idx)

            if not remaining or V.shape[1] <= 1:
                break

            # Project out the chosen item's component
            b = V[chosen].copy()
            norm_b = np.linalg.norm(b)
            if norm_b > 1e-15:
                b /= norm_b
                V = V - np.outer(V @ b, b)
                # Remove a dimension via QR to keep numerics clean
                if V.shape[1] > 1:
                    Q, R = np.linalg.qr(V[remaining])
                    # Keep only non-degenerate columns
                    keep = np.abs(np.diag(R)) > 1e-12
                    V_new = np.zeros((self.n, int(keep.sum())))
                    V_new[remaining] = Q[:, keep]
                    V = V_new

        return sorted(selected)

    # -- Normalization -------------------------------------------------------

    def normalize(self) -> "KernelMatrix":
        """Return a normalized kernel where ``K_ij' = K_ij / sqrt(K_ii K_jj)``."""
        diag = np.diag(self._matrix).copy()
        diag[diag < 1e-15] = 1e-15
        inv_sqrt = 1.0 / np.sqrt(diag)
        normalized = self._matrix * np.outer(inv_sqrt, inv_sqrt)
        return KernelMatrix(normalized)

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        kernel_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> "KernelMatrix":
        """Build a kernel matrix from embedding vectors.

        Args:
            embeddings: 2-D array of shape ``(n, d)``.
            kernel_fn: Optional pairwise kernel function.  Defaults to the
                linear kernel (dot product).
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got shape {embeddings.shape}")
        n = embeddings.shape[0]

        if kernel_fn is None:
            # Linear kernel
            mat = embeddings @ embeddings.T
        else:
            mat = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i, n):
                    val = kernel_fn(embeddings[i], embeddings[j])
                    mat[i, j] = val
                    mat[j, i] = val
        return cls(mat)

    @classmethod
    def rbf(cls, embeddings: np.ndarray, sigma: float = 1.0) -> "KernelMatrix":
        """Build an RBF (Gaussian) kernel matrix.

        Args:
            embeddings: 2-D array of shape ``(n, d)``.
            sigma: Bandwidth parameter.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        diffs = embeddings[:, None, :] - embeddings[None, :, :]
        sq_dists = np.sum(diffs ** 2, axis=-1)
        mat = np.exp(-sq_dists / (2 * sigma ** 2))
        return cls(mat)

    @classmethod
    def cosine(cls, embeddings: np.ndarray) -> "KernelMatrix":
        """Build a cosine-similarity kernel matrix.

        Args:
            embeddings: 2-D array of shape ``(n, d)``.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        normed = embeddings / norms
        mat = normed @ normed.T
        return cls(mat)

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        vs = self.vendi_score()
        return f"KernelMatrix(n={self.n}, vendi={vs:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelMatrix):
            return NotImplemented
        return np.array_equal(self._matrix, other._matrix)

    def __hash__(self) -> int:
        return hash(self._matrix.tobytes())
