"""
LogitSource abstraction for the Diversity Decoding Arena.

Provides a unified interface for obtaining next-token logits from language models,
along with logit post-processing utilities, statistics tracking, and helper functions
for sampling and numerical operations.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_QUANTIZATIONS = {None, "int8", "int4"}
_VALID_DTYPES = {"float16", "float32", "float64", "bfloat16"}


@dataclass
class LogitSourceConfig:
    """Configuration for a :class:`LogitSource` backend.

    Attributes:
        model_name: HuggingFace-style model identifier (e.g. ``"gpt2"``).
        max_seq_len: Maximum context length the source can handle.
        vocab_size: Size of the token vocabulary.
        device: Compute device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
        dtype: Floating-point precision for logit computation.
        batch_size: Default batch size for batched calls.
        use_kv_cache: Whether to use key/value caching for autoregressive decoding.
        quantization: Optional weight quantization mode.
    """

    model_name: str = "gpt2"
    max_seq_len: int = 1024
    vocab_size: int = 50257
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    use_kv_cache: bool = True
    quantization: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantization not in _VALID_QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {_VALID_QUANTIZATIONS}, "
                f"got {self.quantization!r}"
            )
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {_VALID_DTYPES}, got {self.dtype!r}"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    @property
    def numpy_dtype(self) -> np.dtype:
        """Return the corresponding NumPy dtype."""
        mapping = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "bfloat16": np.float32,  # numpy has no bfloat16; use float32 fallback
        }
        return np.dtype(mapping[self.dtype])

    def copy(self, **overrides: Any) -> "LogitSourceConfig":
        """Return a shallow copy, optionally overriding fields."""
        import dataclasses

        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------
# LogitBatch — batched logit output container
# ---------------------------------------------------------------------------


@dataclass
class LogitBatch:
    """Container for a batch of logit outputs.

    Attributes:
        logits: Array of shape ``[batch, vocab]`` containing raw logit values.
        token_ids: Per-sequence list of token IDs that produced the logits.
        attention_mask: Optional boolean/int mask of shape ``[batch, seq_len]``.
        past_key_values: Opaque KV-cache object for incremental decoding.
        metadata: Arbitrary metadata dict (timing, cache statistics, etc.).
    """

    logits: np.ndarray
    token_ids: List[List[int]]
    attention_mask: Optional[np.ndarray] = None
    past_key_values: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ValueError(
                f"logits must be 2-D [batch, vocab], got shape {self.logits.shape}"
            )
        if len(self.token_ids) != self.logits.shape[0]:
            raise ValueError(
                f"token_ids length ({len(self.token_ids)}) must match "
                f"batch dimension ({self.logits.shape[0]})"
            )

    @property
    def batch_size(self) -> int:
        return self.logits.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.logits.shape[1]

    def slice(self, indices: Sequence[int]) -> "LogitBatch":
        """Return a new :class:`LogitBatch` containing only the given batch indices."""
        idx = list(indices)
        return LogitBatch(
            logits=self.logits[idx],
            token_ids=[self.token_ids[i] for i in idx],
            attention_mask=self.attention_mask[idx] if self.attention_mask is not None else None,
            past_key_values=None,  # cache is invalidated on slice
            metadata=dict(self.metadata),
        )

    def to_probabilities(self, temperature: float = 1.0) -> np.ndarray:
        """Convert logits to probability distributions.

        Uses numerically stable softmax with optional temperature scaling.
        """
        return softmax(self.logits / max(temperature, 1e-8))


# ---------------------------------------------------------------------------
# LogitSourceStats — performance tracking
# ---------------------------------------------------------------------------


class LogitSourceStats:
    """Tracks performance statistics for a :class:`LogitSource`.

    Records per-call latency, cache hit/miss counts, throughput, and memory
    usage estimates.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.total_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._tokens_processed: int = 0
        self._start_time: Optional[float] = None
        self._window_size: int = window_size
        self._peak_memory: float = 0.0
        self._current_memory: float = 0.0

    # -- recording ----------------------------------------------------------

    def record_call(
        self,
        latency: float,
        n_tokens: int = 1,
        cache_hit: bool = False,
    ) -> None:
        """Record the result of a single logit-source call.

        Args:
            latency: Wall-clock seconds elapsed.
            n_tokens: Number of tokens processed in this call.
            cache_hit: Whether the call was served from cache.
        """
        if self._start_time is None:
            self._start_time = time.monotonic()
        self.total_calls += 1
        self._latencies.append(latency)
        self._tokens_processed += n_tokens
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_memory(self, current_bytes: float, peak_bytes: Optional[float] = None) -> None:
        """Update memory usage tracking."""
        self._current_memory = current_bytes
        if peak_bytes is not None:
            self._peak_memory = max(self._peak_memory, peak_bytes)
        else:
            self._peak_memory = max(self._peak_memory, current_bytes)

    # -- derived metrics ----------------------------------------------------

    @property
    def avg_latency(self) -> float:
        """Mean latency over the recording window (seconds)."""
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies))

    @property
    def p95_latency(self) -> float:
        """95th-percentile latency (seconds)."""
        if not self._latencies:
            return 0.0
        return float(np.percentile(self._latencies, 95))

    @property
    def p99_latency(self) -> float:
        """99th-percentile latency (seconds)."""
        if not self._latencies:
            return 0.0
        return float(np.percentile(self._latencies, 99))

    @property
    def min_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.min(self._latencies))

    @property
    def max_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.max(self._latencies))

    @property
    def std_latency(self) -> float:
        if len(self._latencies) < 2:
            return 0.0
        return float(np.std(self._latencies, ddof=1))

    @property
    def tokens_per_second(self) -> float:
        """Throughput measured as tokens processed per wall-clock second."""
        if self._start_time is None or self._tokens_processed == 0:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._tokens_processed / elapsed

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def memory_usage(self) -> Dict[str, float]:
        """Return current and peak memory usage in bytes."""
        return {
            "current_bytes": self._current_memory,
            "peak_bytes": self._peak_memory,
            "current_mb": self._current_memory / (1024 * 1024),
            "peak_mb": self._peak_memory / (1024 * 1024),
        }

    # -- reporting ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of all tracked statistics."""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "avg_latency_s": round(self.avg_latency, 6),
            "p95_latency_s": round(self.p95_latency, 6),
            "p99_latency_s": round(self.p99_latency, 6),
            "min_latency_s": round(self.min_latency, 6),
            "max_latency_s": round(self.max_latency, 6),
            "std_latency_s": round(self.std_latency, 6),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tokens_processed": self._tokens_processed,
            "memory": self.memory_usage,
            "window_size": self._window_size,
            "latencies_recorded": len(self._latencies),
        }

    def reset(self) -> None:
        """Reset all counters and latency history."""
        self.total_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._latencies.clear()
        self._tokens_processed = 0
        self._start_time = None
        self._peak_memory = 0.0
        self._current_memory = 0.0

    def __repr__(self) -> str:
        return (
            f"LogitSourceStats(calls={self.total_calls}, "
            f"avg_lat={self.avg_latency:.4f}s, "
            f"tok/s={self.tokens_per_second:.1f}, "
            f"cache_hit={self.cache_hit_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# LogitSource ABC
# ---------------------------------------------------------------------------


class LogitSource(ABC):
    """Abstract base class for logit-producing backends.

    Subclasses must implement the core ``get_logits``, ``get_logits_batch``,
    ``encode``, and ``decode`` methods.  Higher-level conveniences
    (``get_next_token_logits``, ``warmup``, ``benchmark``, …) are provided
    with sensible defaults that delegate to those primitives.
    """

    def __init__(self, config: LogitSourceConfig) -> None:
        self._config = config
        self._stats = LogitSourceStats()
        self._closed = False

    # -- abstract interface -------------------------------------------------

    @abstractmethod
    def get_logits(self, input_ids: List[int]) -> np.ndarray:
        """Return logits for every position in *input_ids*.

        Args:
            input_ids: Token-ID sequence of length *S*.

        Returns:
            Array of shape ``[S, vocab_size]`` with raw (unnormalised) logits.
        """
        ...

    @abstractmethod
    def get_logits_batch(self, input_ids_batch: List[List[int]]) -> LogitBatch:
        """Batched version of :meth:`get_logits`.

        Args:
            input_ids_batch: List of *B* token-ID sequences.

        Returns:
            A :class:`LogitBatch` with logits for the **last** position of
            each sequence (shape ``[B, vocab_size]``).
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Tokenise *text* into a list of integer token IDs."""
        ...

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a string."""
        ...

    # -- convenience methods (overridable) ----------------------------------

    def get_next_token_logits(self, prefix: List[int]) -> np.ndarray:
        """Return logits for the *next* token given *prefix*.

        Default implementation calls :meth:`get_logits` and returns the last
        position.  Subclasses may override for efficiency (e.g. KV-cache).

        Returns:
            1-D array of shape ``[vocab_size]``.
        """
        self._check_closed()
        if not prefix:
            raise ValueError("prefix must be non-empty")
        if len(prefix) > self._config.max_seq_len:
            warnings.warn(
                f"prefix length {len(prefix)} exceeds max_seq_len "
                f"{self._config.max_seq_len}; truncating from the left"
            )
            prefix = prefix[-self._config.max_seq_len:]

        t0 = time.monotonic()
        all_logits = self.get_logits(prefix)
        latency = time.monotonic() - t0
        self._stats.record_call(latency, n_tokens=len(prefix))

        # last position gives next-token distribution
        return all_logits[-1]

    def get_next_token_logits_batch(
        self, prefixes: List[List[int]]
    ) -> np.ndarray:
        """Batched :meth:`get_next_token_logits`.

        Args:
            prefixes: *B* prefix sequences.

        Returns:
            Array of shape ``[B, vocab_size]``.
        """
        self._check_closed()
        if not prefixes:
            raise ValueError("prefixes must be non-empty")

        # Truncate any sequences exceeding max_seq_len
        truncated: List[List[int]] = []
        for p in prefixes:
            if not p:
                raise ValueError("Each prefix must be non-empty")
            if len(p) > self._config.max_seq_len:
                warnings.warn(
                    f"prefix length {len(p)} exceeds max_seq_len "
                    f"{self._config.max_seq_len}; truncating from the left"
                )
                truncated.append(p[-self._config.max_seq_len:])
            else:
                truncated.append(p)

        t0 = time.monotonic()
        batch = self.get_logits_batch(truncated)
        latency = time.monotonic() - t0
        total_tokens = sum(len(p) for p in truncated)
        self._stats.record_call(latency, n_tokens=total_tokens)

        return batch.logits  # already [B, vocab_size] from get_logits_batch

    # -- properties ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the underlying model."""
        return self._config.vocab_size

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length the source supports."""
        return self._config.max_seq_len

    @property
    def device(self) -> str:
        """Device the source is running on."""
        return self._config.device

    @property
    def config(self) -> LogitSourceConfig:
        """The configuration of this source (read-only)."""
        return self._config

    @property
    def stats(self) -> LogitSourceStats:
        """Performance statistics tracker."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        return self._closed

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release any resources held by this source.

        Idempotent — calling multiple times is safe.
        """
        if not self._closed:
            logger.info("Closing LogitSource (%s)", self._config.model_name)
            self._closed = True

    def __enter__(self) -> "LogitSource":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("LogitSource has been closed")

    # -- warmup / benchmark -------------------------------------------------

    def warmup(self, n_steps: int = 3) -> None:
        """Run *n_steps* dummy forward passes to warm up JIT / CUDA caches.

        Uses a short random prefix so the cost is minimal.
        """
        self._check_closed()
        logger.info("Warming up %s for %d steps …", self._config.model_name, n_steps)
        rng = np.random.RandomState(42)
        warm_len = min(16, self._config.max_seq_len)
        for i in range(n_steps):
            dummy_ids = rng.randint(0, self._config.vocab_size, size=warm_len).tolist()
            _ = self.get_next_token_logits(dummy_ids)
        # Reset stats so warmup doesn't pollute measurements
        self._stats.reset()
        logger.info("Warmup complete.")

    def benchmark(self, n_iterations: int = 50, seq_len: int = 64) -> Dict[str, Any]:
        """Measure throughput and latency over *n_iterations* calls.

        Args:
            n_iterations: Number of forward passes.
            seq_len: Length of the random input used in each pass.

        Returns:
            Dict with ``mean_latency``, ``std_latency``, ``p50_latency``,
            ``p95_latency``, ``p99_latency``, ``tokens_per_second``,
            ``total_time``, and ``n_iterations``.
        """
        self._check_closed()
        rng = np.random.RandomState(0)
        seq_len = min(seq_len, self._config.max_seq_len)

        latencies: List[float] = []
        total_tokens = 0

        logger.info(
            "Benchmarking %s: %d iterations, seq_len=%d",
            self._config.model_name,
            n_iterations,
            seq_len,
        )

        wall_start = time.monotonic()
        for _ in range(n_iterations):
            ids = rng.randint(0, self._config.vocab_size, size=seq_len).tolist()
            t0 = time.monotonic()
            _ = self.get_next_token_logits(ids)
            latencies.append(time.monotonic() - t0)
            total_tokens += seq_len
        wall_total = time.monotonic() - wall_start

        lat = np.array(latencies)
        result: Dict[str, Any] = {
            "model_name": self._config.model_name,
            "n_iterations": n_iterations,
            "seq_len": seq_len,
            "total_time_s": round(wall_total, 4),
            "mean_latency_s": round(float(np.mean(lat)), 6),
            "std_latency_s": round(float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0, 6),
            "min_latency_s": round(float(np.min(lat)), 6),
            "max_latency_s": round(float(np.max(lat)), 6),
            "p50_latency_s": round(float(np.percentile(lat, 50)), 6),
            "p95_latency_s": round(float(np.percentile(lat, 95)), 6),
            "p99_latency_s": round(float(np.percentile(lat, 99)), 6),
            "tokens_per_second": round(total_tokens / wall_total, 2) if wall_total > 0 else 0.0,
        }
        logger.info("Benchmark results: %s", result)
        return result

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self._config.model_name!r}, "
            f"vocab={self._config.vocab_size}, "
            f"device={self._config.device!r})"
        )


# ---------------------------------------------------------------------------
# Numerical helper functions
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Accepts 1-D or 2-D arrays.  Returns the same shape.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return (exp / np.sum(exp)).astype(np.float64)
    elif logits.ndim == 2:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float64)
    else:
        # general N-D
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float64)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis.

    Uses the log-sum-exp trick: ``log_softmax(x)_i = x_i - log(sum(exp(x)))``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    max_val = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_val
    lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return (shifted - lse).astype(np.float64)


def entropy_from_logits(logits: np.ndarray) -> float:
    """Shannon entropy (in nats) of the distribution defined by *logits*.

    Accepts a 1-D logit vector and returns a scalar.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)
    # Avoid log(0) — terms with prob ≈ 0 contribute 0
    return float(-np.sum(probs * log_probs))


def entropy_from_logits_batch(logits: np.ndarray) -> np.ndarray:
    """Batch entropy: *logits* has shape ``[B, V]``, returns ``[B]``."""
    logits = np.asarray(logits, dtype=np.float64)
    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)
    return -np.sum(probs * log_probs, axis=-1)


# ---------------------------------------------------------------------------
# Logit manipulation helpers
# ---------------------------------------------------------------------------


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale *logits* by ``1 / temperature``.

    A temperature of 0 is treated as greedy (returns ``-inf`` everywhere
    except the argmax).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if temperature == 0.0:
        # Greedy: set everything to -inf except the max
        result = np.full_like(logits, -np.inf)
        if logits.ndim == 1:
            result[np.argmax(logits)] = 0.0
        else:
            idx = np.argmax(logits, axis=-1)
            rows = np.arange(logits.shape[0])
            result[rows, idx] = 0.0
        return result
    return logits / temperature


def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out (set to ``-inf``) all but the top-*k* logits.

    Works on 1-D or 2-D arrays (last axis is vocab).
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if logits.ndim == 1:
        if k >= logits.shape[0]:
            return logits
        # Indices of logits NOT in top-k
        threshold_idx = np.argpartition(logits, -k)[-k:]
        mask = np.ones(logits.shape[0], dtype=bool)
        mask[threshold_idx] = False
        logits[mask] = -np.inf
    elif logits.ndim == 2:
        vocab = logits.shape[1]
        if k >= vocab:
            return logits
        for i in range(logits.shape[0]):
            row = logits[i]
            threshold_idx = np.argpartition(row, -k)[-k:]
            mask = np.ones(vocab, dtype=bool)
            mask[threshold_idx] = False
            logits[i, mask] = -np.inf
    else:
        raise ValueError(f"logits must be 1-D or 2-D, got {logits.ndim}-D")
    return logits


def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus (top-*p*) filtering.

    Sorts logits in descending order, computes cumulative softmax
    probabilities, and masks out tokens whose cumulative probability
    exceeds *p*.

    Works on 1-D or 2-D arrays.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")

    logits = np.asarray(logits, dtype=np.float64).copy()

    def _filter_row(row: np.ndarray) -> np.ndarray:
        sorted_indices = np.argsort(row)[::-1]
        sorted_logits = row[sorted_indices]
        probs = softmax(sorted_logits)
        cumulative = np.cumsum(probs)
        # Find the cutoff: first index where cumulative > p
        cutoff_mask = cumulative > p
        # Always keep at least the top token
        cutoff_mask[0] = False
        # Set filtered tokens to -inf
        sorted_logits[cutoff_mask] = -np.inf
        # Unsort
        result = np.empty_like(row)
        result[sorted_indices] = sorted_logits
        return result

    if logits.ndim == 1:
        return _filter_row(logits)
    elif logits.ndim == 2:
        for i in range(logits.shape[0]):
            logits[i] = _filter_row(logits[i])
        return logits
    else:
        raise ValueError(f"logits must be 1-D or 2-D, got {logits.ndim}-D")


def apply_typical(logits: np.ndarray, mass: float) -> np.ndarray:
    """Locally typical sampling filter (Meister et al., 2023).

    Keeps tokens whose negative log-probability is closest to the entropy
    of the distribution, accumulating until *mass* probability is covered.

    Works on 1-D or 2-D arrays.
    """
    if not 0.0 < mass <= 1.0:
        raise ValueError(f"mass must be in (0, 1], got {mass}")

    logits = np.asarray(logits, dtype=np.float64).copy()

    def _filter_row(row: np.ndarray) -> np.ndarray:
        log_probs = log_softmax(row)
        probs = np.exp(log_probs)
        ent = -np.sum(probs * log_probs)

        # Distance of each token's neg-log-prob from entropy
        neg_log_probs = -log_probs
        surprisal_deviation = np.abs(neg_log_probs - ent)

        # Sort by how "typical" each token is (smallest deviation first)
        sorted_indices = np.argsort(surprisal_deviation)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        cutoff_mask = cumulative > mass
        cutoff_mask[0] = False  # keep at least one token

        # Mask out non-typical tokens
        remove_indices = sorted_indices[cutoff_mask]
        row[remove_indices] = -np.inf
        return row

    if logits.ndim == 1:
        return _filter_row(logits)
    elif logits.ndim == 2:
        for i in range(logits.shape[0]):
            logits[i] = _filter_row(logits[i])
        return logits
    else:
        raise ValueError(f"logits must be 1-D or 2-D, got {logits.ndim}-D")


def apply_repetition_penalty(
    logits: np.ndarray,
    input_ids: List[int],
    penalty: float,
) -> np.ndarray:
    """Apply multiplicative repetition penalty (Keskar et al., 2019).

    For each token that appeared in *input_ids*, if its logit is positive
    it is divided by *penalty*; if negative it is multiplied by *penalty*.
    A penalty of 1.0 is a no-op.

    Works on a 1-D logit vector.
    """
    if penalty < 0:
        raise ValueError(f"penalty must be non-negative, got {penalty}")
    if penalty == 1.0:
        return logits

    logits = np.asarray(logits, dtype=np.float64).copy()
    unique_ids = set(input_ids)

    if logits.ndim == 1:
        for tid in unique_ids:
            if 0 <= tid < logits.shape[0]:
                if logits[tid] > 0:
                    logits[tid] /= penalty
                else:
                    logits[tid] *= penalty
    elif logits.ndim == 2:
        # Assume input_ids maps to all rows uniformly
        for tid in unique_ids:
            if 0 <= tid < logits.shape[1]:
                pos_mask = logits[:, tid] > 0
                logits[pos_mask, tid] /= penalty
                logits[~pos_mask, tid] *= penalty
    else:
        raise ValueError(f"logits must be 1-D or 2-D, got {logits.ndim}-D")
    return logits


def apply_repetition_penalty_batch(
    logits: np.ndarray,
    input_ids_batch: List[List[int]],
    penalty: float,
) -> np.ndarray:
    """Per-row repetition penalty for a 2-D logit array."""
    if penalty == 1.0:
        return logits
    logits = np.asarray(logits, dtype=np.float64).copy()
    if logits.ndim != 2:
        raise ValueError("logits must be 2-D for batch repetition penalty")
    if len(input_ids_batch) != logits.shape[0]:
        raise ValueError("input_ids_batch length must match batch dimension")
    for i, ids in enumerate(input_ids_batch):
        for tid in set(ids):
            if 0 <= tid < logits.shape[1]:
                if logits[i, tid] > 0:
                    logits[i, tid] /= penalty
                else:
                    logits[i, tid] *= penalty
    return logits


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_from_logits(
    logits: np.ndarray,
    num_samples: int = 1,
    temperature: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Draw *num_samples* token IDs from a logit distribution.

    Applies temperature scaling, converts to probabilities via softmax,
    and samples without replacement (if ``num_samples > 1`` and
    ``num_samples <= vocab_size``).

    Args:
        logits: 1-D array of shape ``[vocab_size]``.
        num_samples: Number of token IDs to return.
        temperature: Temperature for scaling (0 → greedy).
        rng: Optional NumPy random state for reproducibility.

    Returns:
        List of sampled token IDs.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.RandomState()

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    # Greedy
    if temperature == 0.0:
        # Return top-num_samples by logit value
        top_indices = np.argsort(logits)[::-1][:num_samples]
        return top_indices.tolist()

    scaled = logits / max(temperature, 1e-10)
    probs = softmax(scaled)

    # Clamp to avoid numerical issues
    probs = np.clip(probs, 0.0, None)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        # Fallback: uniform
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / prob_sum

    replace = num_samples > len(probs)
    sampled = rng.choice(len(probs), size=num_samples, replace=replace, p=probs)
    return sampled.tolist()


def sample_from_logits_batch(
    logits: np.ndarray,
    num_samples: int = 1,
    temperature: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> List[List[int]]:
    """Batched sampling: *logits* shape ``[B, V]``, returns ``B`` sample lists."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must be 2-D for batch sampling")
    return [
        sample_from_logits(logits[i], num_samples, temperature, rng)
        for i in range(logits.shape[0])
    ]


# ---------------------------------------------------------------------------
# LogitProcessor ABC & implementations
# ---------------------------------------------------------------------------


class LogitProcessor(ABC):
    """Abstract interface for a single logit post-processing step."""

    @abstractmethod
    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        """Transform *logits* given the token history *input_ids*.

        Args:
            logits: 1-D array of shape ``[vocab_size]``.
            input_ids: The sequence of tokens generated so far.

        Returns:
            Transformed logits (same shape).
        """
        ...

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({attrs})"


class TemperatureProcessor(LogitProcessor):
    """Scale logits by ``1 / temperature``."""

    def __init__(self, temperature: float) -> None:
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        self.temperature = temperature

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        return apply_temperature(logits, self.temperature)


class RepetitionPenaltyProcessor(LogitProcessor):
    """Multiplicative repetition penalty (Keskar et al., 2019)."""

    def __init__(self, penalty: float) -> None:
        if penalty < 0:
            raise ValueError(f"penalty must be non-negative, got {penalty}")
        self.penalty = penalty

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        return apply_repetition_penalty(logits, input_ids, self.penalty)


class TopKProcessor(LogitProcessor):
    """Keep only the top-*k* logits; set the rest to ``-inf``."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        return apply_top_k(logits, self.k)


class TopPProcessor(LogitProcessor):
    """Nucleus (top-*p*) filtering."""

    def __init__(self, p: float) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        return apply_top_p(logits, self.p)


class TypicalProcessor(LogitProcessor):
    """Locally typical sampling filter."""

    def __init__(self, mass: float) -> None:
        if not 0.0 < mass <= 1.0:
            raise ValueError(f"mass must be in (0, 1], got {mass}")
        self.mass = mass

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        return apply_typical(logits, self.mass)


class MinLengthProcessor(LogitProcessor):
    """Prevent EOS tokens from being generated before a minimum length.

    Sets the logit of the specified EOS token ID to ``-inf`` until
    ``len(input_ids) >= min_length``.
    """

    def __init__(self, min_length: int, eos_token_id: int) -> None:
        if min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {min_length}")
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).copy()
        if len(input_ids) < self.min_length:
            if 0 <= self.eos_token_id < logits.shape[-1]:
                logits[..., self.eos_token_id] = -np.inf
        return logits


class NoRepeatNGramProcessor(LogitProcessor):
    """Prevent any n-gram from being generated more than once.

    At each step, checks whether generating a particular next token would
    create an n-gram that already exists in *input_ids*.  If so, that
    token's logit is set to ``-inf``.
    """

    def __init__(self, ngram_size: int) -> None:
        if ngram_size <= 0:
            raise ValueError(f"ngram_size must be positive, got {ngram_size}")
        self.ngram_size = ngram_size

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).copy()
        if len(input_ids) < self.ngram_size:
            return logits

        # Build set of existing n-grams
        generated_ngrams: Dict[Tuple[int, ...], List[int]] = {}
        for i in range(len(input_ids) - self.ngram_size + 1):
            ngram = tuple(input_ids[i : i + self.ngram_size])
            prefix = ngram[:-1]
            continuation = ngram[-1]
            generated_ngrams.setdefault(prefix, []).append(continuation)

        # Check which tokens would create a repeated n-gram
        current_prefix = tuple(input_ids[-(self.ngram_size - 1) :])
        banned = generated_ngrams.get(current_prefix, [])
        for token_id in banned:
            if 0 <= token_id < logits.shape[-1]:
                logits[..., token_id] = -np.inf

        return logits


class FrequencyPenaltyProcessor(LogitProcessor):
    """Additive frequency-based penalty.

    Subtracts ``penalty * count(token)`` from the logit of each token that
    appeared in *input_ids*.
    """

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        if self.penalty == 0.0:
            return logits
        logits = np.asarray(logits, dtype=np.float64).copy()
        from collections import Counter
        counts = Counter(input_ids)
        for tid, count in counts.items():
            if 0 <= tid < logits.shape[-1]:
                logits[..., tid] -= self.penalty * count
        return logits


class PresencePenaltyProcessor(LogitProcessor):
    """Additive presence penalty.

    Subtracts *penalty* from the logit of each token that appeared at least
    once in *input_ids*.
    """

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        if self.penalty == 0.0:
            return logits
        logits = np.asarray(logits, dtype=np.float64).copy()
        for tid in set(input_ids):
            if 0 <= tid < logits.shape[-1]:
                logits[..., tid] -= self.penalty
        return logits


# ---------------------------------------------------------------------------
# LogitProcessorList — composable chain
# ---------------------------------------------------------------------------


class LogitProcessorList:
    """An ordered sequence of :class:`LogitProcessor` instances applied
    in order.

    Supports ``append``, ``extend``, iteration, indexing, and ``len``.
    """

    def __init__(
        self, processors: Optional[List[LogitProcessor]] = None
    ) -> None:
        self._processors: List[LogitProcessor] = list(processors or [])

    def append(self, processor: LogitProcessor) -> "LogitProcessorList":
        """Add a processor to the end of the chain.  Returns *self* for chaining."""
        if not isinstance(processor, LogitProcessor):
            raise TypeError(
                f"Expected LogitProcessor, got {type(processor).__name__}"
            )
        self._processors.append(processor)
        return self

    def extend(self, processors: Sequence[LogitProcessor]) -> "LogitProcessorList":
        for p in processors:
            self.append(p)
        return self

    def insert(self, index: int, processor: LogitProcessor) -> "LogitProcessorList":
        if not isinstance(processor, LogitProcessor):
            raise TypeError(
                f"Expected LogitProcessor, got {type(processor).__name__}"
            )
        self._processors.insert(index, processor)
        return self

    def remove(self, processor: LogitProcessor) -> "LogitProcessorList":
        self._processors.remove(processor)
        return self

    def pop(self, index: int = -1) -> LogitProcessor:
        return self._processors.pop(index)

    def clear(self) -> None:
        self._processors.clear()

    def process(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        """Apply all processors sequentially.

        Args:
            logits: Raw logit vector (1-D or 2-D).
            input_ids: Token history.

        Returns:
            Transformed logits.
        """
        for processor in self._processors:
            logits = processor.process(logits, input_ids)
        return logits

    def __call__(self, logits: np.ndarray, input_ids: List[int]) -> np.ndarray:
        """Shorthand for :meth:`process`."""
        return self.process(logits, input_ids)

    def __len__(self) -> int:
        return len(self._processors)

    def __getitem__(self, index: int) -> LogitProcessor:
        return self._processors[index]

    def __iter__(self):
        return iter(self._processors)

    def __bool__(self) -> bool:
        return bool(self._processors)

    def __repr__(self) -> str:
        inner = ", ".join(repr(p) for p in self._processors)
        return f"LogitProcessorList([{inner}])"

    @classmethod
    def from_config(
        cls,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        min_length: int = 0,
        eos_token_id: int = -1,
    ) -> "LogitProcessorList":
        """Build a processor list from common generation parameters.

        Only adds processors whose parameters differ from their no-op defaults.
        """
        processors: List[LogitProcessor] = []

        if temperature != 1.0:
            processors.append(TemperatureProcessor(temperature))
        if repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyProcessor(repetition_penalty))
        if frequency_penalty != 0.0:
            processors.append(FrequencyPenaltyProcessor(frequency_penalty))
        if presence_penalty != 0.0:
            processors.append(PresencePenaltyProcessor(presence_penalty))
        if no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramProcessor(no_repeat_ngram_size))
        if min_length > 0 and eos_token_id >= 0:
            processors.append(MinLengthProcessor(min_length, eos_token_id))
        if top_k > 0:
            processors.append(TopKProcessor(top_k))
        if 0.0 < top_p < 1.0:
            processors.append(TopPProcessor(top_p))
        if 0.0 < typical_p < 1.0:
            processors.append(TypicalProcessor(typical_p))

        return cls(processors)


# ---------------------------------------------------------------------------
# Additional utility functions
# ---------------------------------------------------------------------------


def pad_sequences(
    sequences: List[List[int]],
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    padding_side: str = "right",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad variable-length sequences into a 2-D array with attention mask.

    Args:
        sequences: List of token-ID lists.
        pad_token_id: Token ID used for padding.
        max_length: Maximum sequence length; defaults to the longest sequence.
        padding_side: ``"right"`` or ``"left"``.

    Returns:
        ``(padded_ids, attention_mask)`` — both of shape ``[B, max_length]``.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty")
    lengths = [len(s) for s in sequences]
    ml = max_length or max(lengths)

    padded = np.full((len(sequences), ml), pad_token_id, dtype=np.int64)
    mask = np.zeros((len(sequences), ml), dtype=np.int64)

    for i, seq in enumerate(sequences):
        length = min(len(seq), ml)
        if padding_side == "right":
            padded[i, :length] = seq[:length]
            mask[i, :length] = 1
        elif padding_side == "left":
            offset = ml - length
            padded[i, offset:] = seq[:length]
            mask[i, offset:] = 1
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")

    return padded, mask


def truncate_sequence(
    input_ids: List[int],
    max_length: int,
    truncation_side: str = "left",
) -> List[int]:
    """Truncate a token sequence to *max_length*.

    Args:
        input_ids: Original token sequence.
        max_length: Maximum allowed length.
        truncation_side: ``"left"`` drops oldest tokens, ``"right"`` drops newest.

    Returns:
        Truncated token list.
    """
    if len(input_ids) <= max_length:
        return input_ids
    if truncation_side == "left":
        return input_ids[-max_length:]
    elif truncation_side == "right":
        return input_ids[:max_length]
    else:
        raise ValueError(f"truncation_side must be 'left' or 'right', got {truncation_side!r}")


def top_k_indices(logits: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-*k* logits (descending order)."""
    logits = np.asarray(logits).ravel()
    if k >= len(logits):
        return np.argsort(logits)[::-1]
    idx = np.argpartition(logits, -k)[-k:]
    return idx[np.argsort(logits[idx])[::-1]]


def kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """KL divergence D_KL(P || Q) from logit vectors.

    Both inputs are 1-D logit vectors.  Converts to log-probabilities
    internally.
    """
    log_p = log_softmax(np.asarray(logits_p, dtype=np.float64).ravel())
    log_q = log_softmax(np.asarray(logits_q, dtype=np.float64).ravel())
    p = np.exp(log_p)
    # Only sum over tokens with nonzero probability
    mask = p > 1e-30
    return float(np.sum(p[mask] * (log_p[mask] - log_q[mask])))


def js_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """Jensen–Shannon divergence (symmetric) from logit vectors."""
    log_p = log_softmax(np.asarray(logits_p, dtype=np.float64).ravel())
    log_q = log_softmax(np.asarray(logits_q, dtype=np.float64).ravel())
    p = np.exp(log_p)
    q = np.exp(log_q)
    m = 0.5 * (p + q)
    log_m = np.log(np.clip(m, 1e-30, None))

    mask_p = p > 1e-30
    mask_q = q > 1e-30
    kl_pm = float(np.sum(p[mask_p] * (log_p[mask_p] - log_m[mask_p])))
    kl_qm = float(np.sum(q[mask_q] * (log_q[mask_q] - log_m[mask_q])))
    return 0.5 * (kl_pm + kl_qm)


def cross_entropy(logits: np.ndarray, target_id: int) -> float:
    """Cross-entropy loss for a single target token.

    Returns ``-log P(target_id)`` using numerically stable log-softmax.
    """
    log_probs = log_softmax(np.asarray(logits, dtype=np.float64).ravel())
    return float(-log_probs[target_id])


def perplexity(logits_seq: np.ndarray, target_ids: List[int]) -> float:
    """Perplexity of a sequence given per-position logits.

    Args:
        logits_seq: Shape ``[S, V]`` — logits at each position.
        target_ids: Length-*S* list of ground-truth token IDs.

    Returns:
        Perplexity (exp of mean cross-entropy).
    """
    logits_seq = np.asarray(logits_seq, dtype=np.float64)
    if logits_seq.ndim != 2:
        raise ValueError("logits_seq must be 2-D [seq_len, vocab]")
    if logits_seq.shape[0] != len(target_ids):
        raise ValueError("logits_seq and target_ids must have the same length")

    total_ce = 0.0
    for i, tid in enumerate(target_ids):
        total_ce += cross_entropy(logits_seq[i], tid)

    mean_ce = total_ce / len(target_ids)
    return float(np.exp(mean_ce))


# ---------------------------------------------------------------------------
# Greedy & beam helpers
# ---------------------------------------------------------------------------


def greedy_decode_step(logits: np.ndarray) -> int:
    """Return the token ID with the highest logit (argmax)."""
    return int(np.argmax(logits))


def beam_candidates(
    logits: np.ndarray,
    beam_width: int,
    length_penalty: float = 1.0,
    current_score: float = 0.0,
) -> List[Tuple[int, float]]:
    """Return top-*beam_width* (token_id, cumulative_score) pairs.

    Scores are accumulated log-probabilities with optional length penalty.
    """
    log_probs = log_softmax(np.asarray(logits, dtype=np.float64).ravel())
    top_ids = top_k_indices(logits, beam_width)
    candidates = []
    for tid in top_ids:
        score = current_score + float(log_probs[tid])
        # Length penalty applied at ranking time; store raw score here
        candidates.append((int(tid), score))
    return candidates


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config & data
    "LogitSourceConfig",
    "LogitBatch",
    # Core ABC
    "LogitSource",
    "LogitSourceStats",
    # Processor framework
    "LogitProcessor",
    "LogitProcessorList",
    # Processor implementations
    "TemperatureProcessor",
    "RepetitionPenaltyProcessor",
    "TopKProcessor",
    "TopPProcessor",
    "TypicalProcessor",
    "MinLengthProcessor",
    "NoRepeatNGramProcessor",
    "FrequencyPenaltyProcessor",
    "PresencePenaltyProcessor",
    # Logit manipulation helpers
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "apply_typical",
    "apply_repetition_penalty",
    "apply_repetition_penalty_batch",
    # Sampling
    "sample_from_logits",
    "sample_from_logits_batch",
    # Numerical functions
    "softmax",
    "log_softmax",
    "entropy_from_logits",
    "entropy_from_logits_batch",
    # Utility
    "pad_sequences",
    "truncate_sequence",
    "top_k_indices",
    "kl_divergence",
    "js_divergence",
    "cross_entropy",
    "perplexity",
    "greedy_decode_step",
    "beam_candidates",
]
