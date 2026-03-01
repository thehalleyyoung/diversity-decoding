"""
Speculative Decoding with Diversity Promotion for the Diversity Decoding Arena.
===============================================================================

Implements speculative decoding algorithms that accelerate text generation
while simultaneously promoting output diversity.  The core idea is the
*draft-then-verify* paradigm: a cheap draft model proposes candidate token
sequences which are then verified (accepted / rejected) against a more
expensive target model, with diversity-aware modifications to the standard
rejection-sampling acceptance criterion.

Key components
--------------
* **SpeculativeConfig** -- dataclass extending ``DecodingConfig`` with
  speculative-decoding hyper-parameters (gamma, draft temperature,
  verification strategy, diversity knobs).
* **DraftModel** protocol and concrete implementations:
  - ``NGramDraftModel`` -- n-gram language model built on-the-fly.
  - ``CachedDraftModel`` -- replays cached target-model logits.
  - ``EnsembleDraftModel`` -- combines multiple draft models.
  - ``AdaptiveDraftModel`` -- dynamically adjusts draft length from
    acceptance-rate statistics.
* **VerificationStrategy** classes:
  - ``StandardVerification`` -- classic rejection sampling.
  - ``DiversityAwareVerification`` -- modified acceptance to boost diversity.
  - ``RelaxedVerification`` -- accept with a tolerance threshold.
  - ``TopKVerification`` -- verify only top-k alignment.
* **SpeculativeDecoder** -- main decoder implementing the full speculative
  loop with diversity-aware acceptance/rejection.
* **DiversitySpeculativeDecoder** -- extended decoder supporting multiple
  diverse draft sequences, cross-sequence diversity penalties, batch
  speculative decoding, and tree-based speculation.
* **SpeculativeTreeNode / SpeculativeTree** -- tree structure for
  multi-path speculation with pruning, expansion, and diversity-aware
  scoring.
* **AcceptanceTracker** -- running statistics on acceptance rates, per-
  position acceptance, adaptive gamma selection, and speedup estimation.
* Helper functions: ``compute_acceptance_probability``,
  ``rejection_sampling_step``, ``diversity_modified_acceptance``,
  ``compute_speculative_speedup``, ``optimal_draft_length``.

References
----------
- Leviathan, Y., Kalman, M. & Matias, Y. (2023). *Fast Inference from
  Transformers via Speculative Decoding*. ICML 2023.
- Chen, C. et al. (2023). *Accelerating Large Language Model Decoding with
  Speculative Sampling*. arXiv:2302.01318.
- Miao, X. et al. (2023). *SpecInfer: Accelerating Generative LLM Serving
  with Speculative Inference and Token Tree Verification*. arXiv:2305.09781.
"""

from __future__ import annotations

import abc
import collections
import copy
import heapq
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Counter,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np
from scipy import stats as sp_stats

from src.algorithms.base import DecodingConfig, LogitSource, TokenSequence, LogitArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
_LOG_EPS = -1e10
_DEFAULT_GAMMA = 5
_DEFAULT_DRAFT_TEMPERATURE = 1.0
_DEFAULT_DIVERSITY_WEIGHT = 0.1
_MAX_GAMMA = 20
_MIN_GAMMA = 1
_DEFAULT_VOCAB_SIZE = 32000
_DEFAULT_NGRAM_ORDER = 3
_TREE_MAX_DEPTH = 10
_TREE_MAX_WIDTH = 5

# ---------------------------------------------------------------------------
# Utility: softmax / log-softmax
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    if temperature <= 0:
        temperature = _EPS
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals, axis=-1, keepdims=True) + _EPS)


def _log_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable log-softmax with temperature scaling."""
    if temperature <= 0:
        temperature = _EPS
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + _EPS)
    return shifted - log_sum_exp


def _sample_from_probs(probs: np.ndarray, rng: np.random.RandomState) -> int:
    """Sample a single token index from a probability distribution."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total < _EPS:
        return int(rng.randint(0, len(probs)))
    probs = probs / total
    return int(rng.choice(len(probs), p=probs))


def _top_k_mask(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out all logits outside the top-k."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits.copy()
    result = np.full_like(logits, -np.inf)
    if logits.ndim == 1:
        indices = np.argpartition(logits, -k)[-k:]
        result[indices] = logits[indices]
    else:
        for i in range(logits.shape[0]):
            indices = np.argpartition(logits[i], -k)[-k:]
            result[i, indices] = logits[i, indices]
    return result


def _nucleus_mask(logits: np.ndarray, p: float) -> np.ndarray:
    """Zero out logits outside the nucleus (top-p)."""
    if p >= 1.0:
        return logits.copy()
    result = np.full_like(logits, -np.inf)
    if logits.ndim == 1:
        probs = _softmax(logits)
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, p) + 1
        keep = sorted_idx[:cutoff]
        result[keep] = logits[keep]
    else:
        for i in range(logits.shape[0]):
            probs = _softmax(logits[i])
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, p) + 1
            keep = sorted_idx[:cutoff]
            result[i, keep] = logits[i, keep]
    return result


# =========================================================================
# SpeculativeConfig
# =========================================================================


@dataclass
class SpeculativeConfig(DecodingConfig):
    """Configuration for speculative decoding with diversity promotion."""

    algorithm_name: str = "speculative"

    # -- core speculative parameters ----------------------------------------
    draft_length: int = _DEFAULT_GAMMA
    gamma: float = 1.0
    draft_temperature: float = _DEFAULT_DRAFT_TEMPERATURE
    target_temperature: float = 1.0
    verification_strategy: str = "standard"
    max_draft_length: int = 10
    min_draft_length: int = _MIN_GAMMA

    # -- draft model --------------------------------------------------------
    draft_model_type: str = "ngram"
    ngram_order: int = _DEFAULT_NGRAM_ORDER
    draft_top_k: int = 0
    draft_top_p: float = 1.0
    ensemble_weights: List[float] = field(default_factory=list)

    # -- diversity parameters -----------------------------------------------
    diversity_penalty: float = 0.0
    diversity_penalty_type: str = "ngram"
    diversity_ngram_size: int = 3
    cross_sequence_penalty: float = 0.0
    diversity_temperature_boost: float = 0.0
    min_distinct_ratio: float = 0.0

    # -- verification parameters -------------------------------------------
    relaxation_tolerance: float = 0.1
    top_k_verify: int = 10
    acceptance_threshold: float = 0.0

    # -- adaptive parameters -----------------------------------------------
    adaptive_draft: bool = False
    adaptive_gamma_target_rate: float = 0.7
    adaptive_gamma_ema_alpha: float = 0.1
    adaptive_gamma_increase_step: int = 1
    adaptive_gamma_decrease_step: int = 1

    # -- tree speculative ---------------------------------------------------
    tree_width: int = 1
    tree_depth: int = 1
    tree_pruning_threshold: float = 0.01
    enable_tree_speculation: bool = False

    # -- batch parameters ---------------------------------------------------
    batch_size: int = 1
    parallel_drafts: int = 1

    # -- backward compatibility aliases ------------------------------------

    @property
    def diversity_weight(self) -> float:
        return self.diversity_penalty

    @property
    def adaptive_gamma(self) -> bool:
        return self.adaptive_draft

    @property
    def max_gamma(self) -> int:
        return self.max_draft_length

    @property
    def min_gamma(self) -> int:
        return self.min_draft_length

    @property
    def relaxed_tolerance(self) -> float:
        return self.relaxation_tolerance

    @property
    def topk_verification_k(self) -> int:
        return self.top_k_verify

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeculativeConfig":
        """Reconstruct from a dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in d.items():
            if k in known_fields:
                init_kwargs[k] = v
            else:
                extra[k] = v
        cfg = cls(**init_kwargs)
        cfg.params.update(extra)
        return cfg

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Return validation error strings (empty == valid)."""
        errors = super().validate()
        if self.draft_length < 1:
            errors.append("draft_length must be >= 1")
        if self.gamma < 0:
            errors.append("gamma must be >= 0")
        if self.draft_temperature <= 0:
            errors.append("draft_temperature must be > 0")
        if self.target_temperature <= 0:
            errors.append("target_temperature must be > 0")
        valid_strategies = {
            "standard", "diversity_aware", "relaxed", "topk",
        }
        if self.verification_strategy not in valid_strategies:
            errors.append(
                f"verification_strategy must be one of {valid_strategies}"
            )
        if self.diversity_penalty < 0:
            errors.append("diversity_penalty must be >= 0")
        if self.relaxation_tolerance < 0:
            errors.append("relaxation_tolerance must be >= 0")
        if self.min_draft_length < 1:
            errors.append("min_draft_length must be >= 1")
        if self.max_draft_length < self.min_draft_length:
            errors.append("max_draft_length must be >= min_draft_length")
        if self.top_k_verify < 1:
            errors.append("top_k_verify must be >= 1")
        if self.tree_width < 1:
            errors.append("tree_width must be >= 1")
        if self.tree_depth < 1:
            errors.append("tree_depth must be >= 1")
        if self.adaptive_gamma_target_rate <= 0 or self.adaptive_gamma_target_rate >= 1:
            errors.append("adaptive_gamma_target_rate must be in (0, 1)")
        if self.ngram_order < 1:
            errors.append("ngram_order must be >= 1")
        if not 0.0 < self.adaptive_gamma_ema_alpha <= 1.0:
            errors.append("adaptive_gamma_ema_alpha must be in (0, 1]")
        return errors


# =========================================================================
# DraftModel protocol & implementations
# =========================================================================


@runtime_checkable
class DraftModel(Protocol):
    """Protocol for draft models used in speculative decoding."""

    def draft_logits(
        self, prefix: TokenSequence, num_tokens: int
    ) -> List[LogitArray]:
        """Return a list of logit arrays, one per drafted position."""
        ...

    def draft_tokens(
        self,
        prefix: TokenSequence,
        num_tokens: int,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, List[LogitArray]]:
        """Draft *num_tokens* tokens and return (tokens, logits)."""
        ...

    def update(self, tokens: TokenSequence) -> None:
        """Update internal state with newly accepted tokens."""
        ...


# -------------------------------------------------------------------------
# NGramDraftModel
# -------------------------------------------------------------------------


class NGramDraftModel:
    """N-gram language model used as a lightweight draft model.

    Maintains frequency tables for n-grams observed in the generated text
    and in an optional seed corpus.  Predictions are formed by looking up
    the (n-1)-gram context and converting raw counts to probabilities
    with add-k smoothing.
    """

    def __init__(
        self,
        order: int = _DEFAULT_NGRAM_ORDER,
        n: Optional[int] = None,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        smoothing: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        fallback_uniform: bool = True,
    ) -> None:
        if n is not None:
            order = n
        self.n = max(order, 1)
        self.order = self.n + 1
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.fallback_uniform = fallback_uniform

        # n-gram tables: context tuple -> Counter[next_token]
        self._tables: Dict[int, DefaultDict[Tuple[int, ...], Counter[int]]] = {}
        for n in range(1, self.order + 1):
            self._tables[n] = collections.defaultdict(collections.Counter)

        self._total_tokens_seen: int = 0
        self._unigram_counts: Counter[int] = collections.Counter()

    # -- building / updating ------------------------------------------------

    def _add_sequence(self, tokens: TokenSequence) -> None:
        """Index a token sequence into all n-gram tables."""
        for i, tok in enumerate(tokens):
            self._unigram_counts[tok] += 1
            self._total_tokens_seen += 1
            for n in range(1, self.order + 1):
                start = i - n + 1
                if start < 0:
                    continue
                context = tuple(tokens[start:i])
                self._tables[n][context][tok] += 1

    def update(self, tokens: TokenSequence) -> None:
        """Update n-gram tables with newly observed tokens."""
        self._add_sequence(tokens)

    def seed(self, corpus: List[TokenSequence]) -> None:
        """Pre-populate n-gram tables from a corpus."""
        for seq in corpus:
            self._add_sequence(seq)

    # -- prediction ---------------------------------------------------------

    def _predict_logits(self, context: TokenSequence) -> LogitArray:
        """Produce a logit vector conditioned on *context*."""
        logits = np.zeros(self.vocab_size, dtype=np.float64)

        found = False
        for n in range(self.order, 0, -1):
            if len(context) < n - 1:
                continue
            ctx_tuple = tuple(context[-(n - 1):]) if n > 1 else ()
            counts = self._tables[n].get(ctx_tuple)
            if counts and sum(counts.values()) > 0:
                total = sum(counts.values())
                for tok, cnt in counts.items():
                    if 0 <= tok < self.vocab_size:
                        logits[tok] = np.log(
                            (cnt + self.smoothing)
                            / (total + self.smoothing * self.vocab_size)
                        )
                unseen_log_prob = np.log(
                    self.smoothing
                    / (total + self.smoothing * self.vocab_size)
                )
                logits[logits == 0.0] = unseen_log_prob
                found = True
                break

        if not found:
            if self._total_tokens_seen > 0 and not self.fallback_uniform:
                total = self._total_tokens_seen
                for tok, cnt in self._unigram_counts.items():
                    if 0 <= tok < self.vocab_size:
                        logits[tok] = np.log(
                            (cnt + self.smoothing)
                            / (total + self.smoothing * self.vocab_size)
                        )
                unseen_log_prob = np.log(
                    self.smoothing
                    / (total + self.smoothing * self.vocab_size)
                )
                logits[logits == 0.0] = unseen_log_prob
            else:
                logits[:] = 0.0

        return logits

    def draft_logits(
        self, prefix: TokenSequence, num_tokens: int
    ) -> List[LogitArray]:
        """Return logit arrays for *num_tokens* speculative positions."""
        result: List[LogitArray] = []
        context = list(prefix)
        rng = np.random.RandomState(hash(tuple(prefix)) % (2 ** 31))

        for _ in range(num_tokens):
            raw_logits = self._predict_logits(context)
            if self.top_k > 0:
                raw_logits = _top_k_mask(raw_logits, self.top_k)
            if self.top_p < 1.0:
                raw_logits = _nucleus_mask(raw_logits, self.top_p)
            result.append(raw_logits)
            probs = _softmax(raw_logits, self.temperature)
            next_tok = _sample_from_probs(probs, rng)
            context.append(next_tok)

        return result

    def draft_tokens(
        self,
        prefix: TokenSequence,
        num_tokens: int,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, List[LogitArray]]:
        """Draft *num_tokens* tokens; return (tokens, logits_per_position)."""
        tokens: TokenSequence = []
        logits_list: List[LogitArray] = []
        context = list(prefix)

        for _ in range(num_tokens):
            raw_logits = self._predict_logits(context)
            if self.top_k > 0:
                raw_logits = _top_k_mask(raw_logits, self.top_k)
            if self.top_p < 1.0:
                raw_logits = _nucleus_mask(raw_logits, self.top_p)
            logits_list.append(raw_logits)
            probs = _softmax(raw_logits, self.temperature)
            next_tok = _sample_from_probs(probs, rng)
            tokens.append(next_tok)
            context.append(next_tok)

        return tokens, logits_list


    def predict(self, prefix, num_tokens):
        """Predict tokens and return (tokens, logits_array)."""
        tokens = []
        all_logits = []
        context = list(prefix)
        for _ in range(num_tokens):
            raw_logits = self._predict_logits(context)
            if self.top_k > 0:
                raw_logits = _top_k_mask(raw_logits, self.top_k)
            if self.top_p < 1.0:
                raw_logits = _nucleus_mask(raw_logits, self.top_p)
            all_logits.append(raw_logits)
            next_tok = int(np.argmax(raw_logits))
            tokens.append(next_tok)
            context.append(next_tok)
        return tokens, np.array(all_logits)

    def build_table(self, sequences):
        """Build n-gram tables from a list of sequences."""
        for seq in sequences:
            self._add_sequence(seq)

    def get_counts(self, context_tuple):
        """Get raw counts for a given context tuple."""
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        n = len(context_tuple) + 1
        if n in self._tables:
            table_counts = self._tables[n].get(context_tuple)
            if table_counts:
                for tok, cnt in table_counts.items():
                    if 0 <= tok < self.vocab_size:
                        counts[tok] = cnt
        return counts


# -------------------------------------------------------------------------
# CachedDraftModel
# -------------------------------------------------------------------------


class CachedDraftModel:
    """Draft model that replays previously computed target-model logits."""

    def __init__(
        self,
        source: Optional[Any] = None,
        cached_logits: Optional[Dict[Tuple[int, ...], LogitArray]] = None,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        temperature: float = 1.0,
        fallback_uniform: bool = True,
        max_cache: Optional[int] = None,
    ) -> None:
        self._source = source
        self._cache: Dict[Tuple[int, ...], LogitArray] = cached_logits or {}
        self._max_cache = max_cache
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.fallback_uniform = fallback_uniform
        self._hit_count: int = 0
        self._miss_count: int = 0

    def add_logits(self, prefix: TokenSequence, logits: LogitArray) -> None:
        """Store logits for a given prefix."""
        key = tuple(prefix)
        self._cache[key] = logits.copy()

    def add_batch(
        self, prefixes: List[TokenSequence], logits_batch: List[LogitArray]
    ) -> None:
        """Store logits for a batch of prefixes."""
        for prefix, logits in zip(prefixes, logits_batch):
            self.add_logits(prefix, logits)

    def clear_cache(self) -> None:
        """Clear all cached logits."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def _lookup(self, prefix: TokenSequence) -> LogitArray:
        """Look up cached logits, falling back to uniform."""
        key = tuple(prefix)
        if key in self._cache:
            self._hit_count += 1
            return self._cache[key].copy()
        for trim in range(1, min(len(prefix), 10) + 1):
            suffix_key = tuple(prefix[trim:])
            if suffix_key in self._cache:
                self._hit_count += 1
                return self._cache[suffix_key].copy()
        self._miss_count += 1
        if self._source is not None:
            logits = self._source([prefix])[0].copy()
            self._cache[key] = logits.copy()
            self._enforce_max_cache()
            return logits
        if self.fallback_uniform:
            return np.zeros(self.vocab_size, dtype=np.float64)
        return np.random.randn(self.vocab_size).astype(np.float64) * 0.01

    def draft_logits(
        self, prefix: TokenSequence, num_tokens: int
    ) -> List[LogitArray]:
        """Return logit arrays for *num_tokens* speculative positions."""
        result: List[LogitArray] = []
        context = list(prefix)
        rng = np.random.RandomState(hash(tuple(prefix)) % (2 ** 31))
        for _ in range(num_tokens):
            logits = self._lookup(context)
            result.append(logits)
            probs = _softmax(logits, self.temperature)
            next_tok = _sample_from_probs(probs, rng)
            context.append(next_tok)
        return result

    def draft_tokens(
        self,
        prefix: TokenSequence,
        num_tokens: int,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, List[LogitArray]]:
        """Draft tokens using cached logits."""
        tokens: TokenSequence = []
        logits_list: List[LogitArray] = []
        context = list(prefix)
        for _ in range(num_tokens):
            logits = self._lookup(context)
            logits_list.append(logits)
            probs = _softmax(logits, self.temperature)
            next_tok = _sample_from_probs(probs, rng)
            tokens.append(next_tok)
            context.append(next_tok)
        return tokens, logits_list

    def predict(self, prefix, num_tokens):
        """Predict tokens using source/cache."""
        tokens = []
        all_logits = []
        context = list(prefix)
        for _ in range(num_tokens):
            key = tuple(context)
            if key in self._cache:
                logits = self._cache[key].copy()
            elif self._source is not None:
                logits = self._source([context])[0].copy()
                self._cache[key] = logits.copy()
                self._enforce_max_cache()
            else:
                logits = np.zeros(self.vocab_size, dtype=np.float64)
            all_logits.append(logits)
            probs = _softmax(logits, self.temperature)
            rng = np.random.RandomState(hash(key) % (2 ** 31))
            next_tok = _sample_from_probs(probs, rng)
            tokens.append(next_tok)
            context.append(next_tok)
        return tokens, np.array(all_logits)

    def get_cached_logits(self, key):
        """Get cached logits for a given key, or None."""
        return self._cache.get(key)

    def _enforce_max_cache(self):
        if self._max_cache is not None and len(self._cache) > self._max_cache:
            keys = list(self._cache.keys())
            while len(self._cache) > self._max_cache:
                self._cache.pop(keys.pop(0))

    def update(self, tokens: TokenSequence) -> None:
        """No-op for cached model."""
        pass


# -------------------------------------------------------------------------
# EnsembleDraftModel
# -------------------------------------------------------------------------


class EnsembleDraftModel:
    """Combines predictions from multiple draft models via weighted averaging."""

    def __init__(
        self,
        models: List[Any],
        weights: Optional[List[float]] = None,
        combine_mode: str = "log_linear",
        temperature: float = 1.0,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
    ) -> None:
        if not models:
            raise ValueError("EnsembleDraftModel requires at least one model")
        self.models = models
        n = len(models)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError("weights length must match models length")
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        if combine_mode not in ("log_linear", "linear"):
            raise ValueError("combine_mode must be 'log_linear' or 'linear'")
        self.combine_mode = combine_mode
        self.temperature = temperature
        self.vocab_size = vocab_size

    def _combine_logits(self, logit_sets: List[LogitArray]) -> LogitArray:
        """Combine logits from multiple models."""
        vs = len(logit_sets[0]) if logit_sets else self.vocab_size
        if self.combine_mode == "log_linear":
            combined = np.zeros(vs, dtype=np.float64)
            for w, logits in zip(self.weights, logit_sets):
                combined += w * logits
            return combined
        else:
            combined_probs = np.zeros(vs, dtype=np.float64)
            for w, logits in zip(self.weights, logit_sets):
                probs = _softmax(logits, self.temperature)
                combined_probs += w * probs
            combined_probs = np.maximum(combined_probs, _EPS)
            return np.log(combined_probs)

    def _get_model_logits(self, model, prefix, num_tokens):
        """Get logits from a model, supporting predict or draft_logits."""
        if hasattr(model, "draft_logits"):
            return model.draft_logits(prefix, num_tokens)
        elif hasattr(model, "predict"):
            _, logits = model.predict(prefix, num_tokens)
            if isinstance(logits, np.ndarray) and logits.ndim == 2:
                return [logits[i] for i in range(logits.shape[0])]
            return logits
        return [np.zeros(self.vocab_size) for _ in range(num_tokens)]

    def draft_logits(
        self, prefix: TokenSequence, num_tokens: int
    ) -> List[LogitArray]:
        """Produce ensemble logits for *num_tokens* positions."""
        per_model = [self._get_model_logits(m, prefix, num_tokens) for m in self.models]
        result: List[LogitArray] = []
        for pos in range(num_tokens):
            pos_logits = [per_model[mi][pos] for mi in range(len(self.models))]
            combined = self._combine_logits(pos_logits)
            result.append(combined)
        return result

    def draft_tokens(
        self,
        prefix: TokenSequence,
        num_tokens: int,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, List[LogitArray]]:
        """Draft tokens using ensemble logits."""
        tokens: TokenSequence = []
        logits_list: List[LogitArray] = []
        context = list(prefix)
        for pos in range(num_tokens):
            pos_logits = []
            for model in self.models:
                ml = self._get_model_logits(model, context, 1)
                pos_logits.append(ml[0])
            combined = self._combine_logits(pos_logits)
            logits_list.append(combined)
            probs = _softmax(combined, self.temperature)
            next_tok = _sample_from_probs(probs, rng)
            tokens.append(next_tok)
            context.append(next_tok)
        return tokens, logits_list

    def predict(self, prefix, num_tokens):
        """Predict tokens using ensemble."""
        tokens = []
        all_logits = []
        context = list(prefix)
        for pos in range(num_tokens):
            pos_logits = []
            for model in self.models:
                if hasattr(model, "predict"):
                    _, ml = model.predict(context, 1)
                    pos_logits.append(ml[0])
                elif hasattr(model, "draft_logits"):
                    sl = model.draft_logits(context, 1)
                    pos_logits.append(sl[0])
            combined = self._combine_logits(pos_logits)
            all_logits.append(combined)
            next_tok = int(np.argmax(combined))
            tokens.append(next_tok)
            context.append(next_tok)
        return tokens, np.array(all_logits)

    def update(self, tokens: TokenSequence) -> None:
        """Propagate update to all constituent models."""
        for model in self.models:
            if hasattr(model, "update"):
                model.update(tokens)


# -------------------------------------------------------------------------
# AdaptiveDraftModel
# -------------------------------------------------------------------------


class AdaptiveDraftModel:
    """Wrapper that dynamically adjusts draft length based on acceptance rate."""

    def __init__(
        self,
        base_model: Any,
        initial_gamma: int = _DEFAULT_GAMMA,
        min_gamma: int = _MIN_GAMMA,
        max_gamma: int = _MAX_GAMMA,
        target_acceptance_rate: float = 0.7,
        ema_alpha: float = 0.1,
        increase_step: int = 1,
        decrease_step: int = 1,
        warmup_steps: int = 5,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        target_acceptance: Optional[float] = None,
    ) -> None:
        if min_length is not None:
            min_gamma = min_length
        if max_length is not None:
            max_gamma = max_length
        if target_acceptance is not None:
            target_acceptance_rate = target_acceptance
        if min_length is not None and max_length is not None:
            initial_gamma = (min_gamma + max_gamma) // 2
        self.base_model = base_model
        self.current_gamma = initial_gamma
        self._current_length = initial_gamma
        self.min_length = min_gamma
        self.max_length = max_gamma
        self.target_acceptance = target_acceptance_rate
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.target_acceptance_rate = target_acceptance_rate
        self.ema_alpha = ema_alpha
        self.increase_step = increase_step
        self.decrease_step = decrease_step
        self.warmup_steps = warmup_steps
        self._acceptance_rate_ema: float = target_acceptance_rate
        self._step_count: int = 0
        self._gamma_history: List[int] = [initial_gamma]
        self._rate_history: List[float] = []
        self._acceptance_history: List[float] = []

    def report_acceptance(self, accepted: int, proposed: int) -> None:
        """Report how many of the proposed tokens were accepted."""
        if proposed <= 0:
            return
        rate = accepted / proposed
        self._rate_history.append(rate)
        self._acceptance_rate_ema = (
            self.ema_alpha * rate
            + (1.0 - self.ema_alpha) * self._acceptance_rate_ema
        )
        self._step_count += 1
        if self._step_count < self.warmup_steps:
            return
        if self._acceptance_rate_ema > self.target_acceptance_rate + 0.05:
            self.current_gamma = min(
                self.current_gamma + self.increase_step, self.max_gamma
            )
        elif self._acceptance_rate_ema < self.target_acceptance_rate - 0.05:
            self.current_gamma = max(
                self.current_gamma - self.decrease_step, self.min_gamma
            )
        self._gamma_history.append(self.current_gamma)

    @property
    def effective_gamma(self) -> int:
        return self.current_gamma

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate_ema

    @property
    def gamma_history(self) -> List[int]:
        return list(self._gamma_history)

    @property
    def rate_history(self) -> List[float]:
        return list(self._rate_history)

    def draft_logits(
        self, prefix: TokenSequence, num_tokens: int
    ) -> List[LogitArray]:
        """Use adaptive gamma to decide how many tokens to draft."""
        effective = min(num_tokens, self.current_gamma)
        return self.base_model.draft_logits(prefix, effective)

    def draft_tokens(
        self,
        prefix: TokenSequence,
        num_tokens: int,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, List[LogitArray]]:
        """Draft with adaptively chosen length."""
        effective = min(num_tokens, self.current_gamma)
        if hasattr(self.base_model, "draft_tokens"):
            return self.base_model.draft_tokens(prefix, effective, rng)
        elif hasattr(self.base_model, "predict"):
            tokens, logits = self.base_model.predict(prefix, effective)
            if isinstance(logits, np.ndarray) and logits.ndim == 2:
                logits = [logits[i] for i in range(logits.shape[0])]
            return tokens, logits
        return [], []

    def update(self, tokens: TokenSequence) -> None:
        """Propagate update to base model."""
        if hasattr(self.base_model, "update"):
            self.base_model.update(tokens)

    @property
    def current_length(self):
        return self._current_length

    @current_length.setter
    def current_length(self, value):
        self._current_length = value

    def predict(self, prefix, num_tokens=None):
        """Predict tokens using the base model."""
        if num_tokens is None:
            num_tokens = self._current_length
        if hasattr(self.base_model, "predict"):
            return self.base_model.predict(prefix, num_tokens)
        tokens, logits = self.base_model.draft_tokens(
            prefix, num_tokens, np.random.RandomState(hash(tuple(prefix)) % (2**31)))
        return tokens, np.array(logits)

    def update_acceptance_rate(self, rate):
        """Update current length based on acceptance rate."""
        self._acceptance_history.append(rate)
        if rate > 0.7:
            self._current_length = min(self._current_length + 1, self.max_length)
        elif rate < 0.3:
            self._current_length = max(self._current_length - 1, self.min_length)

    def get_acceptance_history(self):
        """Return the history of acceptance rates."""
        return list(self._acceptance_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Return adaptive draft statistics."""
        return {
            "current_gamma": self.current_gamma,
            "acceptance_rate_ema": self._acceptance_rate_ema,
            "step_count": self._step_count,
            "gamma_history_len": len(self._gamma_history),
            "mean_gamma": float(np.mean(self._gamma_history)),
            "mean_acceptance_rate": (
                float(np.mean(self._rate_history)) if self._rate_history else 0.0
            ),
        }


# =========================================================================
# VerificationStrategy classes
# =========================================================================


class VerificationStrategy(abc.ABC):
    """Abstract base for verification strategies in speculative decoding."""

    def verify(self, draft_tokens, draft_logits=None, target_logits=None,
               draft_temperature=1.0, target_temperature=1.0, rng=None):
        """Verify draft tokens. Returns (accepted_tokens, num_accepted)."""
        result = self._verify_full(draft_tokens, draft_logits, target_logits,
                                    draft_temperature, target_temperature, rng)
        self._last_mask = result[2]
        return result[0], result[1]

    def verify_full(self, draft_tokens, draft_logits=None, target_logits=None,
                    draft_temperature=1.0, target_temperature=1.0, rng=None):
        """Full verify returning (accepted_tokens, num_accepted, mask)."""
        return self._verify_full(draft_tokens, draft_logits, target_logits,
                                  draft_temperature, target_temperature, rng)

    @abc.abstractmethod
    def _verify_full(self, draft_tokens, draft_logits, target_logits,
                     draft_temperature, target_temperature, rng):
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...


class StandardVerification(VerificationStrategy):
    """Classic speculative decoding rejection sampling."""

    def __init__(self, rng=None):
        self._rng = rng

    def name(self) -> str:
        return "standard"

    def _verify_full(self, draft_tokens, draft_logits=None, target_logits=None,
                     draft_temperature=1.0, target_temperature=1.0, rng=None):
        if rng is None:
            rng = self._rng if self._rng is not None else np.random.RandomState()
        if isinstance(draft_logits, np.ndarray) and draft_logits.ndim == 2:
            draft_logits = [draft_logits[i] for i in range(draft_logits.shape[0])]
        elif draft_logits is None:
            draft_logits = []
        if isinstance(target_logits, np.ndarray) and target_logits.ndim == 2:
            target_logits = [target_logits[i] for i in range(target_logits.shape[0])]
        elif target_logits is None:
            target_logits = []
        accepted: TokenSequence = []
        mask: List[bool] = []
        num_accepted = 0
        gamma = len(draft_tokens)

        for t in range(min(gamma, len(draft_logits), len(target_logits))):
            q = _softmax(draft_logits[t], draft_temperature)
            p = _softmax(target_logits[t], target_temperature)
            token = draft_tokens[t]
            q_tok = q[token] + _EPS
            p_tok = p[token]
            acceptance_prob = min(1.0, p_tok / q_tok)
            u = rng.random()
            if u < acceptance_prob:
                accepted.append(token)
                mask.append(True)
                num_accepted += 1
            else:
                mask.append(False)
                break

        return accepted, num_accepted, mask


class DiversityAwareVerification(VerificationStrategy):
    """Verification that modifies acceptance probability to promote diversity.

    The acceptance probability is scaled by a diversity bonus that rewards
    tokens which differ from those already generated across sequences:

        accept_prob = min(1, (p(x) / q(x)) * (1 + diversity_bonus(x)))
    """

    def __init__(
        self,
        diversity_weight: float = _DEFAULT_DIVERSITY_WEIGHT,
        ngram_size: int = 3,
        existing_sequences: Optional[List[TokenSequence]] = None,
        temperature_boost: float = 0.0,
        diversity_penalty: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        if diversity_penalty is not None:
            diversity_weight = diversity_penalty
        self._rng = rng
        self._seen_tokens: Set[int] = set()
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self.existing_sequences = existing_sequences or []
        self.temperature_boost = temperature_boost
        self._existing_ngrams: Optional[Set[Tuple[int, ...]]] = None

    def _build_ngram_set(self) -> Set[Tuple[int, ...]]:
        """Collect all n-grams from existing sequences."""
        if self._existing_ngrams is not None:
            return self._existing_ngrams
        ngrams: Set[Tuple[int, ...]] = set()
        for seq in self.existing_sequences:
            for i in range(len(seq) - self.ngram_size + 1):
                ngrams.add(tuple(seq[i : i + self.ngram_size]))
        self._existing_ngrams = ngrams
        return ngrams

    def _diversity_bonus(self, token: int, context: TokenSequence) -> float:
        """Compute diversity bonus for placing *token* after *context*."""
        if self.diversity_weight <= 0:
            return 0.0
        existing = self._build_ngram_set()
        if not existing:
            return 0.0
        if len(context) >= self.ngram_size - 1:
            candidate = tuple(context[-(self.ngram_size - 1) :]) + (token,)
        else:
            candidate = tuple(context) + (token,)
        if candidate in existing:
            return -self.diversity_weight
        return self.diversity_weight

    def update_existing(self, sequences: List[TokenSequence]) -> None:
        """Update the set of existing sequences."""
        self.existing_sequences = sequences
        self._existing_ngrams = None

    def name(self) -> str:
        return "diversity_aware"

    def reset_seen(self):
        self._seen_tokens.clear()

    def _verify_full(self, draft_tokens, draft_logits=None, target_logits=None,
                     draft_temperature=1.0, target_temperature=1.0, rng=None):
        if rng is None:
            rng = self._rng if self._rng is not None else np.random.RandomState()
        if isinstance(draft_logits, np.ndarray) and draft_logits.ndim == 2:
            draft_logits = [draft_logits[i] for i in range(draft_logits.shape[0])]
        elif draft_logits is None:
            draft_logits = []
        if isinstance(target_logits, np.ndarray) and target_logits.ndim == 2:
            target_logits = [target_logits[i] for i in range(target_logits.shape[0])]
        elif target_logits is None:
            target_logits = []
        accepted: TokenSequence = []
        mask: List[bool] = []
        num_accepted = 0
        gamma = len(draft_tokens)
        running_context: TokenSequence = []
        effective_target_temp = target_temperature + self.temperature_boost

        for t in range(min(gamma, len(draft_logits), len(target_logits))):
            q = _softmax(draft_logits[t], draft_temperature)
            p = _softmax(target_logits[t], effective_target_temp)
            token = draft_tokens[t]
            q_tok = q[token] + _EPS
            p_tok = p[token]
            bonus = self._diversity_bonus(token, running_context)
            acceptance_prob = min(1.0, (p_tok / q_tok) * (1.0 + bonus))
            acceptance_prob = max(0.0, min(1.0, acceptance_prob))
            self._seen_tokens.add(token)
            u = rng.random()
            if u < acceptance_prob:
                accepted.append(token)
                mask.append(True)
                num_accepted += 1
                running_context.append(token)
            else:
                mask.append(False)
                break

        return accepted, num_accepted, mask


class RelaxedVerification(VerificationStrategy):
    """Accept draft tokens when p and q are within a tolerance threshold."""

    def __init__(self, tolerance=0.1, rng=None):
        self.tolerance = max(tolerance, 0.0)
        self._rng = rng

    def name(self) -> str:
        return "relaxed"

    def _verify_full(self, draft_tokens, draft_logits=None, target_logits=None,
                     draft_temperature=1.0, target_temperature=1.0, rng=None):
        if rng is None:
            rng = self._rng if self._rng is not None else np.random.RandomState()
        if isinstance(draft_logits, np.ndarray) and draft_logits.ndim == 2:
            draft_logits = [draft_logits[i] for i in range(draft_logits.shape[0])]
        elif draft_logits is None:
            draft_logits = []
        if isinstance(target_logits, np.ndarray) and target_logits.ndim == 2:
            target_logits = [target_logits[i] for i in range(target_logits.shape[0])]
        elif target_logits is None:
            target_logits = []
        accepted: TokenSequence = []
        mask: List[bool] = []
        num_accepted = 0
        gamma = len(draft_tokens)

        for t in range(min(gamma, len(draft_logits), len(target_logits))):
            q = _softmax(draft_logits[t], draft_temperature)
            p = _softmax(target_logits[t], target_temperature)
            token = draft_tokens[t]
            q_tok = q[token] + _EPS
            p_tok = p[token]
            diff = abs(p_tok - q_tok)
            if diff <= self.tolerance:
                accepted.append(token)
                mask.append(True)
                num_accepted += 1
            else:
                acceptance_prob = min(1.0, p_tok / q_tok)
                u = rng.random()
                if u < acceptance_prob:
                    accepted.append(token)
                    mask.append(True)
                    num_accepted += 1
                else:
                    mask.append(False)
                    break

        return accepted, num_accepted, mask


class TopKVerification(VerificationStrategy):
    """Verify only that the draft token is within the top-k of the target."""

    def __init__(self, k=10, rng=None):
        self.k = max(k, 1)
        self._rng = rng

    def name(self) -> str:
        return "topk"

    def _verify_full(self, draft_tokens, draft_logits=None, target_logits=None,
                     draft_temperature=1.0, target_temperature=1.0, rng=None):
        if rng is None:
            rng = self._rng if self._rng is not None else np.random.RandomState()
        if isinstance(draft_logits, np.ndarray) and draft_logits.ndim == 2:
            draft_logits = [draft_logits[i] for i in range(draft_logits.shape[0])]
        elif draft_logits is None:
            draft_logits = []
        if isinstance(target_logits, np.ndarray) and target_logits.ndim == 2:
            target_logits = [target_logits[i] for i in range(target_logits.shape[0])]
        elif target_logits is None:
            target_logits = []
        accepted: TokenSequence = []
        mask: List[bool] = []
        num_accepted = 0
        gamma = len(draft_tokens)

        for t in range(min(gamma, len(draft_logits), len(target_logits))):
            p = _softmax(target_logits[t], target_temperature)
            q = _softmax(draft_logits[t], draft_temperature)
            token = draft_tokens[t]
            top_k_indices = set(np.argpartition(p, -self.k)[-self.k:].tolist())

            if token in top_k_indices:
                accepted.append(token)
                mask.append(True)
                num_accepted += 1
            else:
                mask.append(False)
                break

        return accepted, num_accepted, mask


# =========================================================================
# Verification strategy factory
# =========================================================================


def _make_verification_strategy(
    config: SpeculativeConfig,
    existing_sequences: Optional[List[TokenSequence]] = None,
) -> VerificationStrategy:
    """Instantiate the verification strategy specified in *config*."""
    name = config.verification_strategy
    if name == "standard":
        return StandardVerification(rng=None)
    elif name == "diversity_aware":
        return DiversityAwareVerification(
            diversity_weight=config.diversity_weight,
            ngram_size=config.diversity_ngram_size,
            existing_sequences=existing_sequences or [],
            temperature_boost=config.diversity_temperature_boost,
        )
    elif name == "relaxed":
        return RelaxedVerification(tolerance=config.relaxed_tolerance)
    elif name == "topk":
        return TopKVerification(k=config.topk_verification_k)
    else:
        logger.warning(
            "Unknown verification strategy '%s', falling back to standard",
            name,
        )
        return StandardVerification()


# =========================================================================
# AcceptanceTracker
# =========================================================================


class AcceptanceTracker:
    """Track acceptance rates, per-position statistics, and speedup estimates."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        max_gamma: int = _MAX_GAMMA,
        target_rate: float = 0.7,
        window_size: int = 1000,
        initial_gamma: float = 1.0,
    ) -> None:
        self._window_size = window_size
        self.gamma = initial_gamma
        self.ema_alpha = ema_alpha
        self.max_gamma = max_gamma
        self.target_rate = target_rate
        self._total_proposed: int = 0
        self._total_accepted: int = 0
        self._ema_rate: float = target_rate
        self._position_proposed: np.ndarray = np.zeros(max_gamma, dtype=np.int64)
        self._position_accepted: np.ndarray = np.zeros(max_gamma, dtype=np.int64)
        self._step_rates: List[float] = []
        self._step_gammas: List[int] = []
        self._step_accepted_counts: List[int] = []
        self._step_proposed_counts: List[int] = []
        self._step_times: List[float] = []
        self._wall_start: float = time.time()
        self._target_calls: int = 0
        self._tokens_generated: int = 0
        self._window_records = collections.deque(maxlen=window_size)

    def record(
        self,
        proposed: int = 0,
        accepted: int = 0,
        acceptance_mask=None,
        gamma_used: int = 0,
        elapsed_time: float = 0.0,
    ) -> None:
        """Record one verification step's results."""
        self._window_records.append((proposed, accepted))
        self._total_proposed += proposed
        self._total_accepted += accepted
        self._target_calls += 1
        if acceptance_mask is not None:
            self._tokens_generated += len(acceptance_mask)
        step_rate = accepted / proposed if proposed > 0 else 0.0
        self._ema_rate = (
            self.ema_alpha * step_rate + (1.0 - self.ema_alpha) * self._ema_rate
        )
        self._step_rates.append(step_rate)
        self._step_gammas.append(gamma_used)
        self._step_accepted_counts.append(accepted)
        self._step_proposed_counts.append(proposed)
        self._step_times.append(elapsed_time)
        for pos, was_accepted in enumerate(acceptance_mask or []):
            if pos < self.max_gamma:
                self._position_proposed[pos] += 1
                if was_accepted:
                    self._position_accepted[pos] += 1

    def acceptance_rate(self):
        """Compute acceptance rate from windowed records."""
        if not self._window_records:
            return 0.0
        total_p = sum(r[0] for r in self._window_records)
        total_a = sum(r[1] for r in self._window_records)
        if total_p == 0:
            return 0.0
        return total_a / total_p

    def num_records(self):
        return len(self._window_records)

    def adaptive_gamma(self, target_rate=0.7):
        rate = self.acceptance_rate()
        if rate > target_rate:
            self.gamma = min(self.gamma * 1.1, 3.0)
        elif rate < target_rate:
            self.gamma = max(self.gamma * 0.9, 0.1)
        return self.gamma

    @property
    def overall_acceptance_rate(self) -> float:
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    @property
    def ema_acceptance_rate(self) -> float:
        return self._ema_rate

    def position_acceptance_rate(self, position: int) -> float:
        """Acceptance rate at a specific draft position."""
        if position < 0 or position >= self.max_gamma:
            return 0.0
        proposed = self._position_proposed[position]
        if proposed == 0:
            return 0.0
        return float(self._position_accepted[position] / proposed)

    def position_rates(self) -> np.ndarray:
        """Acceptance rates for all positions."""
        rates = np.zeros(self.max_gamma, dtype=np.float64)
        nonzero = self._position_proposed > 0
        rates[nonzero] = (
            self._position_accepted[nonzero] / self._position_proposed[nonzero]
        )
        return rates

    @property
    def mean_accepted_per_step(self) -> float:
        if not self._step_accepted_counts:
            return 0.0
        return float(np.mean(self._step_accepted_counts))

    @property
    def total_steps(self) -> int:
        return len(self._step_rates)

    @property
    def tokens_per_target_call(self) -> float:
        if self._target_calls == 0:
            return 0.0
        return self._tokens_generated / self._target_calls

    def compute_speedup(
        self,
        draft_cost: float = 0.05,
        target_cost: float = 1.0,
    ) -> float:
        """Estimate wall-clock speedup from speculative decoding."""
        if not self._step_gammas:
            return 1.0
        avg_gamma = float(np.mean(self._step_gammas))
        alpha = self.overall_acceptance_rate
        if alpha >= 1.0 - _EPS:
            expected_accepted = avg_gamma + 1
        else:
            expected_accepted = (1.0 - alpha ** avg_gamma) / (1.0 - alpha + _EPS)
            expected_accepted += alpha ** avg_gamma
        cost_speculative = avg_gamma * draft_cost + target_cost
        cost_autoregressive = target_cost
        if cost_speculative < _EPS:
            return 1.0
        tokens_per_unit_cost_spec = expected_accepted / cost_speculative
        tokens_per_unit_cost_ar = 1.0 / cost_autoregressive
        return tokens_per_unit_cost_spec / (tokens_per_unit_cost_ar + _EPS)

    def suggest_gamma(self) -> int:
        """Suggest a draft length based on per-position acceptance rates."""
        rates = self.position_rates()
        best_gamma = 1
        for g in range(self.max_gamma):
            r = rates[g]
            if r < _EPS:
                break
            expected = sum(
                float(np.prod(rates[: j + 1])) for j in range(g + 1)
            )
            efficiency = expected / (g + 2)
            if efficiency >= self.target_rate * 0.5:
                best_gamma = g + 1
        return max(1, min(best_gamma, self.max_gamma))

    def get_statistics(self) -> Dict[str, Any]:
        """Return a dictionary of tracking statistics."""
        return {
            "total_proposed": self._total_proposed,
            "total_accepted": self._total_accepted,
            "overall_acceptance_rate": self.overall_acceptance_rate,
            "ema_acceptance_rate": self._ema_rate,
            "total_steps": self.total_steps,
            "mean_accepted_per_step": self.mean_accepted_per_step,
            "tokens_per_target_call": self.tokens_per_target_call,
            "speedup_estimate": self.compute_speedup(),
            "suggested_gamma": self.suggest_gamma(),
            "position_acceptance_rates": self.position_rates().tolist(),
            "gamma_history": list(self._step_gammas),
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._total_proposed = 0
        self._total_accepted = 0
        self._ema_rate = self.target_rate
        self._position_proposed[:] = 0
        self._position_accepted[:] = 0
        self._step_rates.clear()
        self._step_gammas.clear()
        self._step_accepted_counts.clear()
        self._step_proposed_counts.clear()
        self._step_times.clear()
        self._target_calls = 0
        self._tokens_generated = 0
        self._wall_start = time.time()
        self._window_records.clear()
        self.gamma = 1.0


# =========================================================================
# SpeculativeTreeNode
# =========================================================================


class SpeculativeTreeNode:
    """A single node in a speculative decoding tree.

    Each node stores a token, its draft/target log-probabilities, children
    representing alternative continuations, and bookkeeping for pruning
    and scoring.
    """

    __slots__ = (
        "token",
        "depth",
        "draft_log_prob",
        "target_log_prob",
        "logit",
        "probability",
        "cumulative_log_prob",
        "diversity_score",
        "children",
        "parent",
        "is_accepted",
        "accepted",
        "visit_count",
        "acceptance_prob",
        "metadata",
        "_node_id",
    )

    _next_id: int = 0

    def __init__(
        self,
        token: int = -1,
        depth: int = 0,
        draft_log_prob: float = 0.0,
        target_log_prob: float = 0.0,
        parent: Optional["SpeculativeTreeNode"] = None,
        logit: float = 0.0,
        probability: float = 0.0,
    ) -> None:
        self.token = token
        self.depth = depth
        self.draft_log_prob = draft_log_prob
        self.target_log_prob = target_log_prob
        self.logit = logit
        self.probability = probability
        self.accepted = False
        self.cumulative_log_prob = (
            (parent.cumulative_log_prob if parent else 0.0) + target_log_prob
        )
        self.diversity_score: float = 0.0
        self.children: List["SpeculativeTreeNode"] = []
        self.parent = parent
        self.is_accepted: bool = False
        self.visit_count: int = 0
        self.acceptance_prob: float = 0.0
        self.metadata: Dict[str, Any] = {}
        SpeculativeTreeNode._next_id += 1
        self._node_id = SpeculativeTreeNode._next_id

    @property
    def node_id(self) -> int:
        return self._node_id

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    @property
    def num_children(self) -> int:
        return len(self.children)

    def add_child(self, token, draft_log_prob=0.0, target_log_prob=0.0,
                  logit=0.0, probability=0.0):
        """Create and attach a child node."""
        child = SpeculativeTreeNode(
            token=token,
            depth=self.depth + 1,
            draft_log_prob=draft_log_prob,
            target_log_prob=target_log_prob,
            parent=self,
            logit=logit,
            probability=probability,
        )
        self.children.append(child)
        return child

    def get_path(self) -> TokenSequence:
        """Return the token sequence from root to this node (excluding root)."""
        path: List[int] = []
        node: Optional[SpeculativeTreeNode] = self
        while node is not None and not node.is_root():
            path.append(node.token)
            node = node.parent
        path.reverse()
        return path

    def get_path_log_prob(self) -> float:
        """Cumulative target log-probability along the path from root."""
        return self.cumulative_log_prob

    def subtree_size(self) -> int:
        """Number of nodes in the subtree rooted at this node."""
        count = 1
        for child in self.children:
            count += child.subtree_size()
        return count

    def subtree_depth(self) -> int:
        """Maximum depth of the subtree rooted at this node."""
        if not self.children:
            return 0
        return 1 + max(c.subtree_depth() for c in self.children)

    def subtree_leaves(self) -> List["SpeculativeTreeNode"]:
        """Return all leaf nodes in the subtree."""
        if self.is_leaf():
            return [self]
        leaves: List["SpeculativeTreeNode"] = []
        for child in self.children:
            leaves.extend(child.subtree_leaves())
        return leaves

    def path_from_root(self):
        """Return token path from root to this node (inclusive)."""
        path = []
        node = self
        while node is not None:
            path.append(node.token)
            node = node.parent
        path.reverse()
        return path

    def max_depth(self):
        """Maximum depth reachable from this node."""
        if not self.children:
            return self.depth
        return max(c.max_depth() for c in self.children)

    def leaves(self):
        """Return all leaf nodes in the subtree."""
        if self.is_leaf():
            return [self]
        result = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def best_child(self) -> Optional["SpeculativeTreeNode"]:
        """Return the child with highest cumulative log-probability."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.cumulative_log_prob)

    def prune_below_threshold(self, threshold: float) -> int:
        """Remove children whose acceptance probability is below threshold."""
        pruned = 0
        surviving: List["SpeculativeTreeNode"] = []
        for child in self.children:
            check_val = max(child.acceptance_prob, child.probability)
            if check_val < threshold:
                pruned += child.subtree_size()
            else:
                pruned += child.prune_below_threshold(threshold)
                surviving.append(child)
        self.children = surviving
        return pruned

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the subtree to a dictionary."""
        return {
            "token": self.token,
            "depth": self.depth(),
            "draft_log_prob": self.draft_log_prob,
            "target_log_prob": self.target_log_prob,
            "cumulative_log_prob": self.cumulative_log_prob,
            "diversity_score": self.diversity_score,
            "is_accepted": self.is_accepted,
            "acceptance_prob": self.acceptance_prob,
            "children": [c.to_dict() for c in self.children],
        }


# =========================================================================
# SpeculativeTree
# =========================================================================


class SpeculativeTree:
    """Tree structure for multi-path speculative decoding."""

    def __init__(
        self,
        prefix=None,
        max_depth: int = _TREE_MAX_DEPTH,
        max_width: int = _TREE_MAX_WIDTH,
        root_token=None,
        width=None,
        pruning_threshold: float = 0.01,
        diversity_weight: float = 0.0,
    ) -> None:
        if prefix is None:
            prefix = []
        if width is not None:
            max_width = width
        self.prefix = list(prefix)
        self.max_depth = max_depth
        self.max_width = max_width
        self.width = max_width
        self.pruning_threshold = pruning_threshold
        self.diversity_weight = diversity_weight
        root_tok = root_token if root_token is not None else -1
        self.root = SpeculativeTreeNode(token=root_tok, depth=0)
        self._all_nodes: List[SpeculativeTreeNode] = [self.root]
        self._verified: bool = False

    def expand(self, node, context=None, logits=None, **kwargs):
        """Expand a node, compatible with test API."""
        if logits is not None:
            return self.expand_node(node, logits, **kwargs)
        return []

    def expand_node(
        self,
        node: SpeculativeTreeNode,
        draft_logits: LogitArray,
        temperature: float = 1.0,
        top_k: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[SpeculativeTreeNode]:
        """Expand *node* with children sampled from *draft_logits*."""
        if node.depth >= self.max_depth:
            return []
        if rng is None:
            rng = np.random.RandomState()
        logits = draft_logits.copy()
        if top_k > 0:
            logits = _top_k_mask(logits, top_k)
        probs = _softmax(logits, temperature)
        log_probs = _log_softmax(logits, temperature)
        width = min(self.max_width, int((probs > _EPS).sum()))
        if width <= 0:
            width = 1
        top_indices = np.argpartition(probs, -width)[-width:]
        top_indices = top_indices[np.argsort(-probs[top_indices])]
        new_children: List[SpeculativeTreeNode] = []
        for tok_id in top_indices:
            tok_id = int(tok_id)
            child = node.add_child(
                token=tok_id,
                draft_log_prob=float(log_probs[tok_id]),
                target_log_prob=0.0,
                probability=float(probs[tok_id]),
            )
            self._all_nodes.append(child)
            new_children.append(child)
        return new_children

    def expand_level(
        self,
        draft_model: Any,
        temperature: float = 1.0,
        top_k: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> int:
        """Expand all current leaf nodes by one level."""
        if rng is None:
            rng = np.random.RandomState()
        leaves = self.root.subtree_leaves()
        total_new = 0
        for leaf in leaves:
            if leaf.depth >= self.max_depth:
                continue
            path = self.prefix + leaf.get_path()
            logits_list = draft_model.draft_logits(path, 1)
            if logits_list:
                children = self.expand_node(
                    leaf, logits_list[0], temperature, top_k, rng
                )
                total_new += len(children)
        return total_new

    def build_tree(
        self,
        draft_model: Any,
        depth: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        """Build the full speculation tree to the given depth."""
        if depth <= 0:
            depth = self.max_depth
        if rng is None:
            rng = np.random.RandomState()
        for level in range(depth):
            added = self.expand_level(draft_model, temperature, top_k, rng)
            if added == 0:
                break
            logger.debug(
                "Tree level %d: added %d nodes (total %d)",
                level + 1, added, len(self._all_nodes),
            )

    def verify_tree(
        self,
        target_logit_fn: Callable[[List[List[int]]], np.ndarray],
        strategy: VerificationStrategy,
        draft_temperature: float = 1.0,
        target_temperature: float = 1.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[Tuple[TokenSequence, int]]:
        """Verify all paths in the tree against the target model."""
        if rng is None:
            rng = np.random.RandomState()
        leaves = self.root.subtree_leaves()
        results: List[Tuple[TokenSequence, int]] = []
        for leaf in leaves:
            path = leaf.get_path()
            if not path:
                continue
            path_nodes: List[SpeculativeTreeNode] = []
            node = leaf
            while node is not None and not node.is_root():
                path_nodes.append(node)
                node = node.parent
            path_nodes.reverse()
            batch_input = [self.prefix[:]]
            for i in range(len(path)):
                batch_input.append(self.prefix + path[: i + 1])
            target_logits_batch = []
            for inp in batch_input[:-1]:
                target_out = target_logit_fn([inp])
                if target_out.ndim == 2:
                    target_logits_batch.append(target_out[0])
                else:
                    target_logits_batch.append(target_out)
            if len(batch_input) > len(path):
                bonus_out = target_logit_fn([batch_input[-1]])
                if bonus_out.ndim == 2:
                    target_logits_batch.append(bonus_out[0])
                else:
                    target_logits_batch.append(bonus_out)
            draft_logits_path: List[LogitArray] = []
            for pn in path_nodes:
                vocab_size = target_logits_batch[0].shape[-1] if target_logits_batch else _DEFAULT_VOCAB_SIZE
                dl = np.zeros(vocab_size, dtype=np.float64)
                dl[pn.token] = pn.draft_log_prob
                draft_logits_path.append(dl)
            accepted_tokens, num_accepted, mask = strategy.verify_full(
                path, draft_logits_path, target_logits_batch,
                draft_temperature, target_temperature, rng,
            )
            for i, pn in enumerate(path_nodes):
                if i < len(mask):
                    pn.is_accepted = mask[i]
            results.append((accepted_tokens, num_accepted))
        results.sort(key=lambda x: x[1], reverse=True)
        self._verified = True
        return results

    def get_all_paths(self) -> List[TokenSequence]:
        """Return all root-to-leaf paths."""
        leaves = self.root.subtree_leaves()
        paths = [leaf.get_path() for leaf in leaves]
        if not paths or all(not p for p in paths):
            return [[self.root.token]] if len(self._all_nodes) == 1 else []
        return [p for p in paths if p]

    def best_path(self) -> TokenSequence:
        return self.get_best_path()

    def get_best_path(self) -> TokenSequence:
        """Return the path with highest cumulative target log-probability."""
        leaves = self.root.subtree_leaves()
        if not leaves:
            return []
        best = max(leaves, key=lambda l: l.cumulative_log_prob)
        return best.get_path()

    def get_diverse_paths(
        self,
        n: int = 5,
        diversity_weight: float = 0.0,
        ngram_size: int = 3,
    ) -> List[TokenSequence]:
        """Extract *n* diverse paths using a greedy selection procedure."""
        all_paths = self.get_all_paths()
        if not all_paths:
            return []
        if n >= len(all_paths):
            return all_paths
        if diversity_weight <= 0:
            diversity_weight = self.diversity_weight
        leaves = self.root.subtree_leaves()
        scored_paths = [
            (leaf.get_path(), leaf.cumulative_log_prob)
            for leaf in leaves
            if leaf.get_path()
        ]
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        selected: List[TokenSequence] = []
        selected_ngrams: Set[Tuple[int, ...]] = set()
        for path, score in scored_paths:
            if len(selected) >= n:
                break
            path_ngrams: Set[Tuple[int, ...]] = set()
            for i in range(len(path) - ngram_size + 1):
                path_ngrams.add(tuple(path[i : i + ngram_size]))
            if not selected_ngrams:
                div_score = 1.0
            else:
                overlap = len(path_ngrams & selected_ngrams)
                total = max(len(path_ngrams), 1)
                div_score = 1.0 - (overlap / total)
            if div_score > 0.3 or len(selected) < 2:
                selected.append(path)
                selected_ngrams.update(path_ngrams)
        return selected

    def prune(self, threshold=None, min_probability=None) -> int:
        """Prune low-probability branches. Returns number of pruned nodes."""
        if min_probability is not None:
            threshold = min_probability
        if threshold is None:
            threshold = self.pruning_threshold
        before = len(self._all_nodes)
        pruned = self.root.prune_below_threshold(threshold)
        self._all_nodes = [self.root]
        self._collect_all_nodes(self.root)
        after = len(self._all_nodes)
        logger.debug("Pruned %d nodes (%d -> %d)", pruned, before, after)
        return pruned

    def _collect_all_nodes(self, node: SpeculativeTreeNode) -> None:
        """Recursively collect all nodes into _all_nodes."""
        for child in node.children:
            self._all_nodes.append(child)
            self._collect_all_nodes(child)

    def score_paths_with_diversity(
        self,
        diversity_weight: float = 0.1,
        ngram_size: int = 3,
    ) -> List[Tuple[TokenSequence, float]]:
        """Score all paths with a diversity bonus."""
        all_paths = self.get_all_paths()
        if not all_paths:
            return []
        all_ngrams: Counter[Tuple[int, ...]] = collections.Counter()
        for path in all_paths:
            for i in range(len(path) - ngram_size + 1):
                all_ngrams[tuple(path[i : i + ngram_size])] += 1
        scored: List[Tuple[TokenSequence, float]] = []
        leaves = self.root.subtree_leaves()
        for leaf in leaves:
            path = leaf.get_path()
            if not path:
                continue
            quality = leaf.cumulative_log_prob
            path_ngrams = [
                tuple(path[i : i + ngram_size])
                for i in range(len(path) - ngram_size + 1)
            ]
            if path_ngrams:
                avg_freq = np.mean([all_ngrams[ng] for ng in path_ngrams])
                diversity = 1.0 / (avg_freq + _EPS)
            else:
                diversity = 1.0
            combined = quality + diversity_weight * diversity
            scored.append((path, combined))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def size(self) -> int:
        return len(self._all_nodes)

    def depth(self) -> int:
        return self.root.subtree_depth()

    @property
    def num_leaves(self) -> int:
        return len(self.root.subtree_leaves())

    @property
    def num_paths(self) -> int:
        return len(self.get_all_paths())

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "size": self.size(),
            "depth": self.depth(),
            "num_leaves": self.num_leaves,
            "num_paths": self.num_paths,
            "verified": self._verified,
            "prefix_length": len(self.prefix),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prefix": self.prefix,
            "max_depth": self.max_depth,
            "max_width": self.max_width,
            "statistics": self.get_statistics(),
            "tree": self.root.to_dict(),
        }


# =========================================================================
# Helper functions
# =========================================================================


def compute_acceptance_probability(
    draft_prob: float,
    target_prob: float,
    diversity_bonus: float = 0.0,
    gamma: float = 1.0,
) -> float:
    """Compute the speculative decoding acceptance probability.

    Parameters
    ----------
    draft_prob : float
        Probability of the token under the draft model.
    target_prob : float
        Probability of the token under the target model.
    diversity_bonus : float
        Additive bonus to the acceptance ratio for diversity.

    Returns
    -------
    float
        Acceptance probability in [0, 1].
    """
    if draft_prob < _EPS:
        if target_prob < _EPS:
            return 1.0
        return min(1.0, target_prob / _EPS)
    ratio = target_prob / draft_prob
    if gamma != 1.0:
        ratio = ratio ** gamma
    adjusted = ratio * (1.0 + diversity_bonus)
    return max(0.0, min(1.0, adjusted))


def rejection_sampling_step(
    draft_logits_or_token=None,
    target_logits_or_probs=None,
    token_or_probs=None,
    rng=None,
    diversity_bonus: float = 0.0,
    gamma: float = 1.0,
) -> Tuple[bool, float]:
    """Perform one rejection-sampling step.

    Parameters
    ----------
    token : int
        The proposed draft token.
    draft_probs : np.ndarray
        Full draft probability distribution.
    target_probs : np.ndarray
        Full target probability distribution.
    rng : np.random.RandomState
        Random number generator.
    diversity_bonus : float
        Optional diversity bonus.

    Returns
    -------
    accepted : bool
        Whether the token was accepted.
    output_token : int
        The token to use (original if accepted, correction if rejected).
    """
    if isinstance(draft_logits_or_token, np.ndarray):
        draft_probs = _softmax(draft_logits_or_token)
        target_probs = _softmax(target_logits_or_probs)
        token = int(token_or_probs)
    else:
        token = int(draft_logits_or_token)
        draft_probs = target_logits_or_probs
        target_probs = token_or_probs
    if rng is None:
        rng = np.random.RandomState()
    q = float(draft_probs[token]) + _EPS
    p = float(target_probs[token])
    acceptance = compute_acceptance_probability(q, p, diversity_bonus, gamma)
    u = rng.random()
    if u < acceptance:
        return True, acceptance
    return False, acceptance


def diversity_modified_acceptance(
    draft_prob_or_token=None,
    target_prob_or_probs=None,
    token=None,
    seen_tokens=None,
    diversity_bonus=0.0,
    gamma=1.0,
    existing_sequences=None,
    context=None,
    ngram_size=3,
    diversity_weight=0.1,
) -> float:
    """Compute diversity-modified acceptance probability.

    Adjusts the standard acceptance ratio by a bonus/penalty based on
    how novel the proposed token is relative to existing sequences.

    Parameters
    ----------
    token : int
        Proposed token.
    draft_probs, target_probs : np.ndarray
        Draft and target distributions.
    existing_sequences : List[TokenSequence]
        Already generated sequences.
    context : TokenSequence
        Current generation context.
    ngram_size : int
        Size of n-grams for novelty computation.
    diversity_weight : float
        Weight for the diversity bonus.

    Returns
    -------
    float
        Modified acceptance probability.
    """
    if isinstance(draft_prob_or_token, (int, np.integer)):
        # Old-style: (token, draft_probs, target_probs, ...)
        old_token = int(draft_prob_or_token)
        draft_probs = target_prob_or_probs
        target_probs = existing_sequences if existing_sequences is not None else np.zeros(1)
        q = float(draft_probs[old_token]) + _EPS
        p = float(target_probs[old_token])
        base_ratio = p / q
        ctx = context or []
        existing_seqs = seen_tokens if isinstance(seen_tokens, list) else []
        existing_ngrams: Set[Tuple[int, ...]] = set()
        for seq in existing_seqs:
            for i in range(len(seq) - ngram_size + 1):
                existing_ngrams.add(tuple(seq[i : i + ngram_size]))
        if len(ctx) >= ngram_size - 1:
            candidate = tuple(ctx[-(ngram_size - 1) :]) + (old_token,)
        else:
            candidate = tuple(ctx) + (old_token,)
        if candidate in existing_ngrams:
            bonus = -diversity_weight
        else:
            bonus = diversity_weight
        adjusted = base_ratio * (1.0 + bonus)
        return max(0.0, min(1.0, adjusted))
    else:
        dp = float(draft_prob_or_token)
        tp = float(target_prob_or_probs)
        base = compute_acceptance_probability(dp, tp, gamma=gamma)
        if token is not None and seen_tokens is not None and token not in seen_tokens:
            adjusted = min(1.0, base + diversity_bonus)
        else:
            adjusted = base
        return max(0.0, min(1.0, adjusted))


def compute_speculative_speedup(
    acceptance_rate: float,
    draft_cost: float = 0.05,
    target_cost: float = 1.0,
    draft_length: int = 5,
) -> float:
    """Estimate speedup from speculative decoding.

    Parameters
    ----------
    acceptance_rate : float
        Per-token acceptance rate (alpha).
    gamma : int
        Draft length (number of speculative tokens).
    draft_cost_ratio : float
        Cost of one draft forward pass relative to target.

    Returns
    -------
    float
        Estimated speedup factor.
    """
    if target_cost <= 0 or draft_cost < 0:
        return 1.0
    gamma = draft_length
    alpha = max(0.0, min(1.0, acceptance_rate))
    if alpha >= 1.0 - _EPS:
        expected_tokens = gamma + 1.0
    elif gamma == 0:
        expected_tokens = 1.0
    else:
        expected_tokens = 1.0 + (1.0 - alpha ** gamma) / (1.0 - alpha + _EPS)
    spec_cost = gamma * draft_cost + target_cost
    baseline_cost = target_cost * expected_tokens
    if spec_cost < _EPS:
        return 1.0
    return baseline_cost / spec_cost


def optimal_draft_length(
    acceptance_rate: float,
    draft_cost: float = 0.05,
    target_cost: float = 1.0,
    max_length: int = _MAX_GAMMA,
) -> int:
    """Find the draft length that maximises expected speedup.

    Searches over gamma in [1, max_gamma] and returns the value that
    maximises ``compute_speculative_speedup``.

    Parameters
    ----------
    acceptance_rate : float
        Per-token acceptance rate.
    draft_cost_ratio : float
        Cost ratio of draft to target.
    max_gamma : int
        Maximum allowed gamma.

    Returns
    -------
    int
        Optimal gamma value.
    """
    if target_cost <= 0:
        return 1
    best_gamma = 1
    best_speedup = 0.0
    for g in range(1, max_length + 1):
        s = compute_speculative_speedup(acceptance_rate, draft_cost, target_cost, g)
        if s > best_speedup:
            best_speedup = s
            best_gamma = g
    return best_gamma


def _compute_ngram_diversity(
    sequences: List[TokenSequence],
    ngram_size: int = 3,
) -> float:
    """Compute average pairwise n-gram diversity between sequences.

    Returns a value in [0, 1] where 1 means maximum diversity.
    """
    if len(sequences) < 2:
        return 1.0
    ngram_sets: List[Set[Tuple[int, ...]]] = []
    for seq in sequences:
        s: Set[Tuple[int, ...]] = set()
        for i in range(len(seq) - ngram_size + 1):
            s.add(tuple(seq[i : i + ngram_size]))
        ngram_sets.append(s)
    total_diversity = 0.0
    count = 0
    for i in range(len(ngram_sets)):
        for j in range(i + 1, len(ngram_sets)):
            if not ngram_sets[i] and not ngram_sets[j]:
                total_diversity += 1.0
            else:
                union = len(ngram_sets[i] | ngram_sets[j])
                intersection = len(ngram_sets[i] & ngram_sets[j])
                if union > 0:
                    total_diversity += 1.0 - intersection / union
                else:
                    total_diversity += 1.0
            count += 1
    return total_diversity / max(count, 1)


def _compute_sequence_log_prob(
    tokens: TokenSequence,
    logit_fn: Callable[[List[List[int]]], np.ndarray],
    prefix: TokenSequence,
    temperature: float = 1.0,
) -> float:
    """Compute the total log-probability of a token sequence."""
    total_lp = 0.0
    context = list(prefix)
    for tok in tokens:
        out = logit_fn([context])
        if out.ndim == 2:
            logits = out[0]
        else:
            logits = out
        log_probs = _log_softmax(logits, temperature)
        total_lp += float(log_probs[tok])
        context.append(tok)
    return total_lp


def _apply_repetition_penalty(
    logits: np.ndarray,
    generated_tokens: TokenSequence,
    penalty: float = 1.0,
) -> np.ndarray:
    """Apply repetition penalty to logits based on already generated tokens."""
    if penalty <= 1.0:
        return logits
    result = logits.copy()
    for tok in set(generated_tokens):
        if 0 <= tok < len(result):
            if result[tok] > 0:
                result[tok] /= penalty
            else:
                result[tok] *= penalty
    return result


def _apply_no_repeat_ngram(
    logits: np.ndarray,
    generated_tokens: TokenSequence,
    ngram_size: int,
) -> np.ndarray:
    """Block n-grams that already appeared in generated_tokens."""
    if ngram_size <= 0 or len(generated_tokens) < ngram_size - 1:
        return logits
    result = logits.copy()
    context = tuple(generated_tokens[-(ngram_size - 1):]) if ngram_size > 1 else ()
    for i in range(len(generated_tokens) - ngram_size + 1):
        existing_context = tuple(generated_tokens[i : i + ngram_size - 1])
        if existing_context == context:
            blocked_token = generated_tokens[i + ngram_size - 1]
            if 0 <= blocked_token < len(result):
                result[blocked_token] = -np.inf
    return result


# =========================================================================
# SpeculativeDecoder
# =========================================================================


class SpeculativeDecoder:
    """Main speculative decoder implementing draft-then-verify with diversity.

    Runs the standard speculative decoding loop:
    1. Draft *gamma* tokens from the draft model.
    2. Compute target logits for all drafted positions in parallel.
    3. Verify each token via the chosen verification strategy.
    4. Accept a prefix of the drafted tokens and (optionally) a correction.
    5. Repeat until max_new_tokens or EOS.

    Supports diversity-aware verification, adaptive draft length, and
    multi-sequence generation.
    """

    def __init__(self, target_logit_fn_or_config=None, draft_model=None,
                 config=None, vocab_size=_DEFAULT_VOCAB_SIZE,
                 verification=None, rng=None, target_logit_fn=None):
        if isinstance(target_logit_fn_or_config, SpeculativeConfig):
            config = target_logit_fn_or_config
            if target_logit_fn is None and not isinstance(draft_model, SpeculativeConfig):
                target_logit_fn = vocab_size if callable(vocab_size) else None
                vocab_size = _DEFAULT_VOCAB_SIZE
        else:
            if target_logit_fn is None:
                target_logit_fn = target_logit_fn_or_config
            if config is None:
                config = SpeculativeConfig()
        self.config = config
        self.draft_model = draft_model
        self.target_logit_fn = target_logit_fn
        self.vocab_size = vocab_size
        self._verification = verification
        self._rng = rng if rng is not None else np.random.RandomState(config.seed)
        self._tracker = AcceptanceTracker(
            ema_alpha=config.adaptive_gamma_ema_alpha,
            max_gamma=config.max_draft_length,
            target_rate=config.adaptive_gamma_target_rate,
        )
        self._current_gamma = config.draft_length
        self._generated_sequences: List[TokenSequence] = []
        self._generation_stats: Dict[str, Any] = {}

    def decode(self, prefix):
        seq, _ = self.generate_single(prefix)
        return seq

    def generate_single(
        self,
        prefix: TokenSequence,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[TokenSequence, Dict[str, Any]]:
        """Generate a single sequence using speculative decoding.

        Parameters
        ----------
        prefix : TokenSequence
            Input token ids.
        max_new_tokens : int, optional
            Override for config.max_new_tokens.

        Returns
        -------
        sequence : TokenSequence
            Generated token sequence (prefix + new tokens).
        stats : dict
            Generation statistics.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if self._verification is not None:
            strategy = self._verification
        else:
            strategy = _make_verification_strategy(
                self.config, self._generated_sequences
            )
        generated: TokenSequence = list(prefix)
        new_tokens: TokenSequence = []
        total_draft_tokens = 0
        total_accepted = 0
        target_calls = 0
        start_time = time.time()

        while len(new_tokens) < max_new_tokens:
            gamma = self._effective_gamma()
            remaining = max_new_tokens - len(new_tokens)
            gamma = min(gamma, remaining)
            if gamma <= 0:
                break

            # 1. Draft phase
            if hasattr(self.draft_model, "draft_tokens"):
                draft_tokens, draft_logits = self.draft_model.draft_tokens(
                    generated, gamma, self._rng
                )
            elif hasattr(self.draft_model, "predict"):
                draft_tokens, _dl = self.draft_model.predict(generated, gamma)
                draft_logits = [_dl[i] for i in range(_dl.shape[0])] if isinstance(_dl, np.ndarray) and _dl.ndim == 2 else _dl
            else:
                draft_tokens, draft_logits = [], []
            total_draft_tokens += len(draft_tokens)

            if not draft_tokens:
                target_out = self.target_logit_fn([generated])
                if target_out.ndim == 2:
                    t_logits = target_out[0]
                else:
                    t_logits = target_out
                t_logits = self._apply_penalties(t_logits, generated)
                p = _softmax(t_logits, self.config.target_temperature)
                tok = _sample_from_probs(p, self._rng)
                new_tokens.append(tok)
                generated.append(tok)
                target_calls += 1
                if self.config.eos_token_id is not None and tok == self.config.eos_token_id:
                    break
                continue

            # 2. Target logits for all draft positions (+ 1 for bonus)
            target_logits_list: List[LogitArray] = []
            for i in range(len(draft_tokens) + 1):
                context = generated + draft_tokens[:i]
                target_out = self.target_logit_fn([context])
                if target_out.ndim == 2:
                    t_logits = target_out[0]
                else:
                    t_logits = target_out
                t_logits = self._apply_penalties(t_logits, context)
                target_logits_list.append(t_logits)
            target_calls += 1

            # 3. Verification
            step_start = time.time()
            accepted_tokens, num_accepted, mask = strategy.verify_full(
                draft_tokens, draft_logits, target_logits_list,
                self.config.draft_temperature,
                self.config.target_temperature, self._rng,
            )
            step_elapsed = time.time() - step_start

            # 4. Record and update
            self._tracker.record(
                proposed=len(draft_tokens), accepted=num_accepted,
                acceptance_mask=mask, gamma_used=gamma,
                elapsed_time=step_elapsed,
            )
            total_accepted += num_accepted

            if isinstance(self.draft_model, AdaptiveDraftModel):
                self.draft_model.report_acceptance(num_accepted, len(draft_tokens))

            if self.config.adaptive_gamma:
                self._update_gamma()

            for tok in accepted_tokens:
                if len(new_tokens) >= max_new_tokens:
                    break
                new_tokens.append(tok)
                generated.append(tok)
                if self.config.eos_token_id is not None and tok == self.config.eos_token_id:
                    break

            # If nothing was accepted, sample from target to make progress
            if num_accepted == 0 and len(accepted_tokens) == 0:
                target_out = self.target_logit_fn([generated])
                if target_out.ndim == 2:
                    t_logits = target_out[0]
                else:
                    t_logits = target_out
                t_logits = self._apply_penalties(t_logits, generated)
                p = _softmax(t_logits, self.config.target_temperature)
                tok = _sample_from_probs(p, self._rng)
                new_tokens.append(tok)
                generated.append(tok)
                target_calls += 1

            if (
                self.config.eos_token_id is not None
                and new_tokens
                and new_tokens[-1] == self.config.eos_token_id
            ):
                break

            if hasattr(self.draft_model, "update"):
                self.draft_model.update(accepted_tokens)

            if isinstance(strategy, DiversityAwareVerification):
                strategy.update_existing(
                    self._generated_sequences + [generated]
                )

        elapsed = time.time() - start_time
        stats = {
            "num_new_tokens": len(new_tokens),
            "total_draft_tokens": total_draft_tokens,
            "total_accepted": total_accepted,
            "acceptance_rate": total_accepted / max(total_draft_tokens, 1),
            "target_calls": target_calls,
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(new_tokens) / max(elapsed, _EPS),
            "speedup_estimate": self._tracker.compute_speedup(),
            "final_gamma": self._current_gamma,
            "tracker_stats": self._tracker.get_statistics(),
        }
        return generated, stats

    def generate(
        self,
        prefix: TokenSequence,
        num_sequences: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[TokenSequence], Dict[str, Any]]:
        """Generate multiple sequences with diversity constraints."""
        if num_sequences is None:
            num_sequences = self.config.num_sequences
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        sequences: List[TokenSequence] = []
        all_stats: List[Dict[str, Any]] = []
        for seq_idx in range(num_sequences):
            logger.debug("Generating sequence %d / %d", seq_idx + 1, num_sequences)
            self._generated_sequences = sequences.copy()
            seq, stats = self.generate_single(prefix, max_new_tokens)
            sequences.append(seq)
            all_stats.append(stats)
            if seq_idx % 5 == 0 and seq_idx > 0:
                div = _compute_ngram_diversity(
                    sequences, self.config.diversity_ngram_size
                )
                logger.debug("Diversity after %d sequences: %.4f", seq_idx + 1, div)

        diversity = _compute_ngram_diversity(
            sequences, self.config.diversity_ngram_size
        )
        aggregate = {
            "num_sequences": len(sequences),
            "diversity_score": diversity,
            "mean_acceptance_rate": float(
                np.mean([s["acceptance_rate"] for s in all_stats])
            ),
            "mean_tokens_per_second": float(
                np.mean([s["tokens_per_second"] for s in all_stats])
            ),
            "mean_speedup": float(
                np.mean([s["speedup_estimate"] for s in all_stats])
            ),
            "total_elapsed": sum(s["elapsed_seconds"] for s in all_stats),
            "per_sequence_stats": all_stats,
        }
        self._generation_stats = aggregate
        return sequences, aggregate

    def _apply_penalties(
        self, logits: LogitArray, context: TokenSequence
    ) -> LogitArray:
        """Apply repetition penalty and no-repeat-ngram blocking."""
        result = logits.copy()
        if self.config.repetition_penalty > 1.0:
            result = _apply_repetition_penalty(
                result, context, self.config.repetition_penalty
            )
        if self.config.no_repeat_ngram_size > 0:
            result = _apply_no_repeat_ngram(
                result, context, self.config.no_repeat_ngram_size
            )
        return result

    def _effective_gamma(self) -> int:
        """Return the current effective draft length."""
        if isinstance(self.draft_model, AdaptiveDraftModel):
            return self.draft_model.effective_gamma
        return self._current_gamma

    def _update_gamma(self) -> None:
        """Update gamma based on acceptance rate tracking."""
        rate = self._tracker.ema_acceptance_rate
        target = self.config.adaptive_gamma_target_rate
        if rate > target + 0.05:
            self._current_gamma = min(
                self._current_gamma + self.config.adaptive_gamma_increase_step,
                self.config.max_draft_length,
            )
        elif rate < target - 0.05:
            self._current_gamma = max(
                self._current_gamma - self.config.adaptive_gamma_decrease_step,
                self.config.min_draft_length,
            )

    @property
    def tracker(self) -> AcceptanceTracker:
        return self._tracker

    @property
    def generation_stats(self) -> Dict[str, Any]:
        return self._generation_stats

    def reset(self) -> None:
        """Reset decoder state for a new generation."""
        self._tracker.reset()
        self._current_gamma = self.config.draft_length
        self._generated_sequences.clear()
        self._generation_stats.clear()
        if self.config.seed is not None:
            self._rng = np.random.RandomState(self.config.seed)


# =========================================================================
# DiversitySpeculativeDecoder
# =========================================================================


class DiversitySpeculativeDecoder:
    """Extended speculative decoder with diversity-centric features.

    Builds on ``SpeculativeDecoder`` with:
    * Multiple diverse draft sequences per verification step.
    * Cross-sequence diversity penalties during verification.
    * Batch speculative decoding across sequences.
    * Tree-based speculation with diversity-aware scoring.
    """

    def __init__(self, target_logit_fn_or_config=None, draft_model_or_models=None,
                 config=None, vocab_size=_DEFAULT_VOCAB_SIZE,
                 rng=None, target_logit_fn=None, draft_models=None):
        if isinstance(target_logit_fn_or_config, SpeculativeConfig):
            config = target_logit_fn_or_config
            if draft_models is None:
                draft_models = draft_model_or_models if isinstance(draft_model_or_models, list) else [draft_model_or_models] if draft_model_or_models else []
            if target_logit_fn is None:
                target_logit_fn = vocab_size if callable(vocab_size) else None
                vocab_size = _DEFAULT_VOCAB_SIZE
        else:
            if target_logit_fn is None:
                target_logit_fn = target_logit_fn_or_config
            if config is None:
                config = SpeculativeConfig()
            if draft_models is None:
                if isinstance(draft_model_or_models, list):
                    draft_models = draft_model_or_models
                elif draft_model_or_models is not None:
                    draft_models = [draft_model_or_models]
                else:
                    draft_models = []
        self.config = config
        self.draft_models = draft_models if draft_models else []
        self.target_logit_fn = target_logit_fn
        self.vocab_size = vocab_size
        self._rng = rng if rng is not None else np.random.RandomState(config.seed)
        self._tracker = AcceptanceTracker(
            ema_alpha=config.adaptive_gamma_ema_alpha,
            max_gamma=config.max_draft_length,
            target_rate=config.adaptive_gamma_target_rate,
        )
        self._generated_sequences: List[TokenSequence] = []
        self._generation_stats: Dict[str, Any] = {}

    def decode_multiple(self, prefix, num_sequences=1):
        sequences, _ = self.generate(prefix, num_sequences=num_sequences)
        return sequences

    def _generate_diverse_drafts(
        self,
        prefix: TokenSequence,
        gamma: int,
        num_drafts: int,
    ) -> List[Tuple[TokenSequence, List[LogitArray]]]:
        """Generate multiple diverse draft sequences."""
        drafts: List[Tuple[TokenSequence, List[LogitArray]]] = []
        for i in range(num_drafts):
            if i < len(self.draft_models):
                model = self.draft_models[i]
            else:
                model = self.draft_models[i % max(len(self.draft_models), 1)]
            temp_rng = np.random.RandomState(
                self._rng.randint(0, 2 ** 31) + i
            )
            if hasattr(model, "draft_tokens"):
                tokens, logits = model.draft_tokens(prefix, gamma, temp_rng)
            elif hasattr(model, "predict"):
                tokens, _la = model.predict(prefix, gamma)
                logits = [_la[i] for i in range(_la.shape[0])] if isinstance(_la, np.ndarray) and _la.ndim == 2 else _la
            else:
                tokens, logits = [], []
            if self._generated_sequences and self.config.cross_sequence_penalty > 0:
                tokens, logits = self._apply_cross_sequence_penalty(
                    tokens, logits, prefix, temp_rng
                )
            drafts.append((tokens, logits))
        return drafts

    def _apply_cross_sequence_penalty(
        self,
        tokens: TokenSequence,
        logits: List[LogitArray],
        prefix: TokenSequence,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, List[LogitArray]]:
        """Re-sample tokens with cross-sequence diversity penalty."""
        penalty = self.config.cross_sequence_penalty
        if penalty <= 0:
            return tokens, logits
        adjusted_tokens: TokenSequence = []
        adjusted_logits: List[LogitArray] = []
        context = list(prefix)
        for pos in range(len(tokens)):
            log = logits[pos].copy()
            for seq in self._generated_sequences:
                seq_offset = len(prefix) + pos
                if seq_offset < len(seq):
                    existing_tok = seq[seq_offset]
                    if 0 <= existing_tok < len(log):
                        log[existing_tok] -= penalty
            adjusted_logits.append(log)
            probs = _softmax(log, self.config.draft_temperature)
            tok = _sample_from_probs(probs, rng)
            adjusted_tokens.append(tok)
            context.append(tok)
        return adjusted_tokens, adjusted_logits

    def _select_best_draft(
        self,
        drafts: List[Tuple[TokenSequence, List[LogitArray]]],
        prefix: TokenSequence,
        target_logits_cache: Optional[Dict[int, List[LogitArray]]] = None,
    ) -> Tuple[TokenSequence, List[LogitArray], int]:
        """Select the best draft based on target model alignment and diversity."""
        if len(drafts) == 1:
            return drafts[0][0], drafts[0][1], 0
        best_idx = 0
        best_score = -np.inf
        for idx, (tokens, logits) in enumerate(drafts):
            quality = 0.0
            for pos, tok in enumerate(tokens):
                if pos < len(logits):
                    draft_prob = _softmax(logits[pos], self.config.draft_temperature)
                    quality += np.log(draft_prob[tok] + _EPS)
            div = 0.0
            if self._generated_sequences:
                for seq in self._generated_sequences:
                    overlap = 0
                    for pos, tok in enumerate(tokens):
                        seq_pos = len(prefix) + pos
                        if seq_pos < len(seq) and seq[seq_pos] == tok:
                            overlap += 1
                    div += 1.0 - overlap / max(len(tokens), 1)
                div /= len(self._generated_sequences)
            score = quality + self.config.diversity_weight * div
            if score > best_score:
                best_score = score
                best_idx = idx
        return drafts[best_idx][0], drafts[best_idx][1], best_idx

    def _tree_speculate(
        self,
        prefix: TokenSequence,
        gamma: int,
    ) -> Tuple[TokenSequence, Dict[str, Any]]:
        """Perform tree-based speculative decoding."""
        tree = SpeculativeTree(
            prefix=prefix,
            max_depth=min(gamma, self.config.tree_depth),
            max_width=self.config.tree_width,
            pruning_threshold=self.config.tree_pruning_threshold,
            diversity_weight=self.config.diversity_weight,
        )
        draft_model = self.draft_models[0] if self.draft_models else None
        if draft_model is None:
            return [], {"error": "no_draft_model"}
        tree.build_tree(
            draft_model,
            depth=min(gamma, self.config.tree_depth),
            temperature=self.config.draft_temperature,
            top_k=self.config.draft_top_k,
            rng=self._rng,
        )
        tree.prune()
        strategy = _make_verification_strategy(
            self.config, self._generated_sequences
        )
        results = tree.verify_tree(
            self.target_logit_fn, strategy,
            self.config.draft_temperature,
            self.config.target_temperature, self._rng,
        )
        if not results:
            return [], {"tree_stats": tree.get_statistics()}
        if self.config.diversity_weight > 0 and len(results) > 1:
            scored_results: List[Tuple[float, int]] = []
            for i, (tokens, num_acc) in enumerate(results):
                quality = float(num_acc)
                div = _compute_ngram_diversity(
                    self._generated_sequences + [prefix + tokens],
                    self.config.diversity_ngram_size,
                ) if self._generated_sequences else 1.0
                score = quality + self.config.diversity_weight * div * 10.0
                scored_results.append((score, i))
            scored_results.sort(reverse=True)
            best_idx = scored_results[0][1]
        else:
            best_idx = 0
        best_tokens, best_accepted = results[best_idx]
        tree_stats = tree.get_statistics()
        tree_stats["num_paths_verified"] = len(results)
        tree_stats["best_accepted"] = best_accepted
        return best_tokens, tree_stats

    def generate_single(
        self,
        prefix: TokenSequence,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[TokenSequence, Dict[str, Any]]:
        """Generate one sequence with diversity-aware speculative decoding."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        strategy = _make_verification_strategy(
            self.config, self._generated_sequences
        )
        generated: TokenSequence = list(prefix)
        new_tokens: TokenSequence = []
        total_draft = 0
        total_accepted = 0
        target_calls = 0
        tree_stats_list: List[Dict[str, Any]] = []
        start_time = time.time()

        while len(new_tokens) < max_new_tokens:
            remaining = max_new_tokens - len(new_tokens)
            gamma = min(self.config.draft_length, remaining)
            if gamma <= 0:
                break
            if self.config.enable_tree_speculation:
                tree_tokens, t_stats = self._tree_speculate(generated, gamma)
                tree_stats_list.append(t_stats)
                if tree_tokens:
                    for tok in tree_tokens:
                        if len(new_tokens) >= max_new_tokens:
                            break
                        new_tokens.append(tok)
                        generated.append(tok)
                        if (self.config.eos_token_id is not None
                                and tok == self.config.eos_token_id):
                            break
                    target_calls += 1
                    total_accepted += len(tree_tokens)
                    total_draft += gamma
                else:
                    target_out = self.target_logit_fn([generated])
                    if target_out.ndim == 2:
                        t_logits = target_out[0]
                    else:
                        t_logits = target_out
                    p = _softmax(t_logits, self.config.target_temperature)
                    tok = _sample_from_probs(p, self._rng)
                    new_tokens.append(tok)
                    generated.append(tok)
                    target_calls += 1
            else:
                num_drafts = max(self.config.parallel_drafts, 1)
                drafts = self._generate_diverse_drafts(
                    generated, gamma, num_drafts
                )
                draft_tokens, draft_logits, draft_idx = self._select_best_draft(
                    drafts, prefix
                )
                total_draft += len(draft_tokens)
                if not draft_tokens:
                    target_out = self.target_logit_fn([generated])
                    if target_out.ndim == 2:
                        t_logits = target_out[0]
                    else:
                        t_logits = target_out
                    p = _softmax(t_logits, self.config.target_temperature)
                    tok = _sample_from_probs(p, self._rng)
                    new_tokens.append(tok)
                    generated.append(tok)
                    target_calls += 1
                    if self.config.eos_token_id is not None and tok == self.config.eos_token_id:
                        break
                    continue
                target_logits_list: List[LogitArray] = []
                for i in range(len(draft_tokens) + 1):
                    context = generated + draft_tokens[:i]
                    target_out = self.target_logit_fn([context])
                    if target_out.ndim == 2:
                        t_logits = target_out[0]
                    else:
                        t_logits = target_out
                    if self.config.repetition_penalty > 1.0:
                        t_logits = _apply_repetition_penalty(
                            t_logits, context, self.config.repetition_penalty
                        )
                    if self.config.no_repeat_ngram_size > 0:
                        t_logits = _apply_no_repeat_ngram(
                            t_logits, context, self.config.no_repeat_ngram_size
                        )
                    target_logits_list.append(t_logits)
                target_calls += 1
                step_start = time.time()
                accepted_tokens, num_accepted, mask = strategy.verify_full(
                    draft_tokens, draft_logits, target_logits_list,
                    self.config.draft_temperature,
                    self.config.target_temperature, self._rng,
                )
                step_elapsed = time.time() - step_start
                self._tracker.record(
                    proposed=len(draft_tokens), accepted=num_accepted,
                    acceptance_mask=mask, gamma_used=gamma,
                    elapsed_time=step_elapsed,
                )
                total_accepted += num_accepted
                for tok in accepted_tokens:
                    if len(new_tokens) >= max_new_tokens:
                        break
                    new_tokens.append(tok)
                    generated.append(tok)
                    if (self.config.eos_token_id is not None
                            and tok == self.config.eos_token_id):
                        break
                for model in self.draft_models:
                    if hasattr(model, "update"):
                        model.update(accepted_tokens)
                    if isinstance(model, AdaptiveDraftModel):
                        model.report_acceptance(num_accepted, len(draft_tokens))
                if isinstance(strategy, DiversityAwareVerification):
                    strategy.update_existing(
                        self._generated_sequences + [generated]
                    )
            if (self.config.eos_token_id is not None
                    and new_tokens and new_tokens[-1] == self.config.eos_token_id):
                break

        elapsed = time.time() - start_time
        stats: Dict[str, Any] = {
            "num_new_tokens": len(new_tokens),
            "total_draft_tokens": total_draft,
            "total_accepted": total_accepted,
            "acceptance_rate": total_accepted / max(total_draft, 1),
            "target_calls": target_calls,
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(new_tokens) / max(elapsed, _EPS),
            "speedup_estimate": self._tracker.compute_speedup(),
            "tracker_stats": self._tracker.get_statistics(),
        }
        if tree_stats_list:
            stats["tree_stats"] = tree_stats_list
        return generated, stats

    def generate_batch(
        self,
        prefixes: List[TokenSequence],
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[TokenSequence], Dict[str, Any]]:
        """Run speculative decoding on a batch of prefixes simultaneously."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        batch_size = len(prefixes)
        sequences: List[TokenSequence] = [list(p) for p in prefixes]
        new_counts: List[int] = [0] * batch_size
        finished: List[bool] = [False] * batch_size
        total_target_calls = 0
        start_time = time.time()
        strategy = _make_verification_strategy(
            self.config, self._generated_sequences
        )
        while not all(finished):
            for idx in range(batch_size):
                if finished[idx]:
                    continue
                remaining = max_new_tokens - new_counts[idx]
                gamma = min(self.config.draft_length, remaining)
                if gamma <= 0:
                    finished[idx] = True
                    continue
                num_drafts = max(self.config.parallel_drafts, 1)
                self._generated_sequences = [
                    s for i, s in enumerate(sequences)
                    if i != idx and new_counts[i] > 0
                ]
                drafts = self._generate_diverse_drafts(
                    sequences[idx], gamma, num_drafts
                )
                draft_tokens, draft_logits, _ = self._select_best_draft(
                    drafts, prefixes[idx]
                )
                if not draft_tokens:
                    target_out = self.target_logit_fn([sequences[idx]])
                    if target_out.ndim == 2:
                        t_logits = target_out[0]
                    else:
                        t_logits = target_out
                    p = _softmax(t_logits, self.config.target_temperature)
                    tok = _sample_from_probs(p, self._rng)
                    sequences[idx].append(tok)
                    new_counts[idx] += 1
                    total_target_calls += 1
                    if self.config.eos_token_id is not None and tok == self.config.eos_token_id:
                        finished[idx] = True
                    continue
                target_logits_list: List[LogitArray] = []
                for i in range(len(draft_tokens) + 1):
                    context = sequences[idx] + draft_tokens[:i]
                    target_out = self.target_logit_fn([context])
                    if target_out.ndim == 2:
                        t_logits = target_out[0]
                    else:
                        t_logits = target_out
                    if self.config.repetition_penalty > 1.0:
                        t_logits = _apply_repetition_penalty(
                            t_logits, context, self.config.repetition_penalty
                        )
                    target_logits_list.append(t_logits)
                total_target_calls += 1
                accepted_tokens, num_accepted, mask = strategy.verify_full(
                    draft_tokens, draft_logits, target_logits_list,
                    self.config.draft_temperature,
                    self.config.target_temperature, self._rng,
                )
                self._tracker.record(
                    proposed=len(draft_tokens), accepted=num_accepted,
                    acceptance_mask=mask, gamma_used=gamma,
                )
                for tok in accepted_tokens:
                    if new_counts[idx] >= max_new_tokens:
                        finished[idx] = True
                        break
                    sequences[idx].append(tok)
                    new_counts[idx] += 1
                    if (self.config.eos_token_id is not None
                            and tok == self.config.eos_token_id):
                        finished[idx] = True
                        break
                if isinstance(strategy, DiversityAwareVerification):
                    strategy.update_existing(sequences)

        elapsed = time.time() - start_time
        batch_stats: Dict[str, Any] = {
            "batch_size": batch_size,
            "total_target_calls": total_target_calls,
            "elapsed_seconds": elapsed,
            "mean_new_tokens": float(np.mean(new_counts)),
            "diversity_score": _compute_ngram_diversity(
                sequences, self.config.diversity_ngram_size
            ),
            "tracker_stats": self._tracker.get_statistics(),
        }
        return sequences, batch_stats

    def generate(
        self,
        prefix: TokenSequence,
        num_sequences: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[TokenSequence], Dict[str, Any]]:
        """Generate multiple diverse sequences."""
        if num_sequences is None:
            num_sequences = self.config.num_sequences
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        sequences: List[TokenSequence] = []
        all_stats: List[Dict[str, Any]] = []
        for seq_idx in range(num_sequences):
            logger.debug(
                "DiversitySpeculative: sequence %d / %d",
                seq_idx + 1, num_sequences,
            )
            self._generated_sequences = sequences.copy()
            seq, stats = self.generate_single(prefix, max_new_tokens)
            sequences.append(seq)
            all_stats.append(stats)
        diversity = _compute_ngram_diversity(
            sequences, self.config.diversity_ngram_size
        )
        aggregate: Dict[str, Any] = {
            "num_sequences": len(sequences),
            "diversity_score": diversity,
            "mean_acceptance_rate": float(
                np.mean([s["acceptance_rate"] for s in all_stats])
            ),
            "mean_tokens_per_second": float(
                np.mean([s["tokens_per_second"] for s in all_stats])
            ),
            "mean_speedup": float(
                np.mean([s["speedup_estimate"] for s in all_stats])
            ),
            "total_elapsed": sum(s["elapsed_seconds"] for s in all_stats),
            "per_sequence_stats": all_stats,
        }
        self._generation_stats = aggregate
        return sequences, aggregate

    @property
    def tracker(self) -> AcceptanceTracker:
        return self._tracker

    @property
    def generation_stats(self) -> Dict[str, Any]:
        return self._generation_stats

    def reset(self) -> None:
        """Reset all state."""
        self._tracker.reset()
        self._generated_sequences.clear()
        self._generation_stats.clear()
        if self.config.seed is not None:
            self._rng = np.random.RandomState(self.config.seed)


# =========================================================================
# DraftModelFactory
# =========================================================================


class DraftModelFactory:
    """Factory for creating draft model instances from configuration."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, model_class: Type) -> None:
        """Register a draft model class under a name."""
        cls._registry[name] = model_class

    @classmethod
    def create(
        cls,
        config: SpeculativeConfig,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
    ) -> Any:
        """Create a draft model from config."""
        model_type = config.draft_model_type
        if model_type == "ngram":
            model = NGramDraftModel(
                order=config.ngram_order,
                vocab_size=vocab_size,
                temperature=config.draft_temperature,
                top_k=config.draft_top_k,
                top_p=config.draft_top_p,
            )
        elif model_type == "cached":
            model = CachedDraftModel(
                vocab_size=vocab_size,
                temperature=config.draft_temperature,
            )
        elif model_type == "ensemble":
            sub_models = []
            for i in range(max(config.parallel_drafts, 2)):
                sub = NGramDraftModel(
                    order=max(config.ngram_order - i, 1),
                    vocab_size=vocab_size,
                    temperature=config.draft_temperature * (1.0 + 0.1 * i),
                )
                sub_models.append(sub)
            weights = config.ensemble_weights or None
            model = EnsembleDraftModel(
                models=sub_models,
                weights=weights,
                temperature=config.draft_temperature,
                vocab_size=vocab_size,
            )
        elif model_type in cls._registry:
            model = cls._registry[model_type](config=config, vocab_size=vocab_size)
        else:
            logger.warning(
                "Unknown draft_model_type '%s', falling back to ngram",
                model_type,
            )
            model = NGramDraftModel(
                order=config.ngram_order,
                vocab_size=vocab_size,
                temperature=config.draft_temperature,
            )
        if config.adaptive_gamma:
            model = AdaptiveDraftModel(
                base_model=model,
                initial_gamma=config.draft_length,
                min_gamma=config.min_draft_length,
                max_gamma=config.max_draft_length,
                target_acceptance_rate=config.adaptive_gamma_target_rate,
                ema_alpha=config.adaptive_gamma_ema_alpha,
                increase_step=config.adaptive_gamma_increase_step,
                decrease_step=config.adaptive_gamma_decrease_step,
            )
        return model


DraftModelFactory.register("ngram", NGramDraftModel)
DraftModelFactory.register("cached", CachedDraftModel)


# =========================================================================
# SpeculativeDecoderFactory
# =========================================================================


class SpeculativeDecoderFactory:
    """Factory for creating speculative decoders from configuration."""

    @staticmethod
    def create(
        config: SpeculativeConfig,
        target_logit_fn: LogitSource,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        draft_models: Optional[List[Any]] = None,
    ) -> Union[SpeculativeDecoder, DiversitySpeculativeDecoder]:
        """Create the appropriate decoder based on config."""
        if draft_models is None:
            main_draft = DraftModelFactory.create(config, vocab_size)
            draft_models = [main_draft]
        if (
            config.parallel_drafts > 1
            or config.enable_tree_speculation
            or config.cross_sequence_penalty > 0
        ):
            if len(draft_models) < config.parallel_drafts:
                for i in range(config.parallel_drafts - len(draft_models)):
                    extra = DraftModelFactory.create(config, vocab_size)
                    draft_models.append(extra)
            return DiversitySpeculativeDecoder(
                config=config,
                draft_models=draft_models,
                target_logit_fn=target_logit_fn,
                vocab_size=vocab_size,
            )
        return SpeculativeDecoder(
            config=config,
            draft_model=draft_models[0],
            target_logit_fn=target_logit_fn,
            vocab_size=vocab_size,
        )


# =========================================================================
# Analysis and diagnostics
# =========================================================================


class SpeculativeAnalyzer:
    """Post-hoc analysis of speculative decoding runs."""

    def __init__(self, tracker: AcceptanceTracker) -> None:
        self.tracker = tracker

    def acceptance_rate_by_position(self) -> Dict[int, float]:
        """Return acceptance rates indexed by draft position."""
        rates = self.tracker.position_rates()
        return {
            pos: float(rate)
            for pos, rate in enumerate(rates)
            if self.tracker._position_proposed[pos] > 0
        }

    def acceptance_rate_trend(self, window: int = 10) -> List[float]:
        """Return rolling-window acceptance rate over steps."""
        rates = self.tracker._step_rates
        if len(rates) < window:
            return rates
        result: List[float] = []
        for i in range(len(rates) - window + 1):
            result.append(float(np.mean(rates[i : i + window])))
        return result

    def gamma_trend(self) -> List[int]:
        """Return the history of gamma values used."""
        return list(self.tracker._step_gammas)

    def speedup_sensitivity(
        self,
        draft_cost_ratios: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        """Compute speedup for varying draft cost ratios."""
        if draft_cost_ratios is None:
            draft_cost_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        return {
            ratio: self.tracker.compute_speedup(draft_cost=ratio)
            for ratio in draft_cost_ratios
        }

    def optimal_gamma_analysis(
        self, draft_cost_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        """Analyse optimal gamma for the observed acceptance rate."""
        rate = self.tracker.overall_acceptance_rate
        opt_gamma = optimal_draft_length(rate, draft_cost_ratio, 1.0)
        suggested = self.tracker.suggest_gamma()
        gamma_speedups = {
            g: compute_speculative_speedup(rate, draft_cost_ratio, 1.0, g)
            for g in range(1, _MAX_GAMMA + 1)
        }
        return {
            "overall_acceptance_rate": rate,
            "optimal_gamma": opt_gamma,
            "suggested_gamma": suggested,
            "current_speedup": self.tracker.compute_speedup(
                draft_cost=draft_cost_ratio
            ),
            "optimal_speedup": gamma_speedups.get(opt_gamma, 1.0),
            "gamma_speedup_curve": gamma_speedups,
        }

    def draft_quality_metrics(self) -> Dict[str, Any]:
        """Compute metrics characterising draft model quality."""
        rates = self.tracker._step_rates
        if not rates:
            return {"error": "no_data"}
        arr = np.array(rates)
        return {
            "mean_acceptance": float(np.mean(arr)),
            "std_acceptance": float(np.std(arr)),
            "median_acceptance": float(np.median(arr)),
            "min_acceptance": float(np.min(arr)),
            "max_acceptance": float(np.max(arr)),
            "q25_acceptance": float(np.percentile(arr, 25)),
            "q75_acceptance": float(np.percentile(arr, 75)),
            "acceptance_entropy": float(
                sp_stats.entropy(
                    np.histogram(arr, bins=20, range=(0, 1))[0] + 1
                )
            ),
            "total_steps": len(rates),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        return {
            "acceptance_by_position": self.acceptance_rate_by_position(),
            "acceptance_trend": self.acceptance_rate_trend(),
            "gamma_trend": self.gamma_trend(),
            "speedup_sensitivity": self.speedup_sensitivity(),
            "optimal_gamma_analysis": self.optimal_gamma_analysis(),
            "draft_quality": self.draft_quality_metrics(),
            "tracker_stats": self.tracker.get_statistics(),
        }


# =========================================================================
# Diversity metrics for speculative decoding
# =========================================================================


class SpeculativeDiversityMetrics:
    """Compute diversity metrics specific to speculative decoding outputs."""

    def __init__(self, ngram_sizes: Optional[List[int]] = None) -> None:
        self.ngram_sizes = ngram_sizes or [2, 3, 4]

    def compute_all(
        self, sequences: List[TokenSequence]
    ) -> Dict[str, Any]:
        """Compute all diversity metrics for a set of sequences."""
        if not sequences:
            return {"error": "no_sequences"}
        metrics: Dict[str, Any] = {
            "num_sequences": len(sequences),
            "mean_length": float(np.mean([len(s) for s in sequences])),
        }
        for n in self.ngram_sizes:
            metrics[f"ngram_{n}_diversity"] = _compute_ngram_diversity(
                sequences, n
            )
        metrics["token_diversity"] = self._token_level_diversity(sequences)
        metrics["positional_diversity"] = self._positional_diversity(sequences)
        metrics["suffix_diversity"] = self._suffix_diversity(sequences)
        metrics["unique_token_ratio"] = self._unique_token_ratio(sequences)
        return metrics

    def _token_level_diversity(
        self, sequences: List[TokenSequence]
    ) -> float:
        """Fraction of unique tokens across all sequences."""
        if not sequences:
            return 0.0
        all_tokens: Set[int] = set()
        total = 0
        for seq in sequences:
            all_tokens.update(seq)
            total += len(seq)
        return len(all_tokens) / max(total, 1)

    def _positional_diversity(
        self, sequences: List[TokenSequence]
    ) -> float:
        """Average number of distinct tokens per position."""
        if len(sequences) < 2:
            return 1.0
        min_len = min(len(s) for s in sequences)
        if min_len == 0:
            return 0.0
        diversity_sum = 0.0
        for pos in range(min_len):
            tokens_at_pos = set(s[pos] for s in sequences)
            diversity_sum += len(tokens_at_pos) / len(sequences)
        return diversity_sum / min_len

    def _suffix_diversity(
        self, sequences: List[TokenSequence], suffix_len: int = 5
    ) -> float:
        """Diversity of the last *suffix_len* tokens across sequences."""
        suffixes: List[Tuple[int, ...]] = []
        for seq in sequences:
            if len(seq) >= suffix_len:
                suffixes.append(tuple(seq[-suffix_len:]))
            elif seq:
                suffixes.append(tuple(seq))
        if not suffixes:
            return 0.0
        unique = len(set(suffixes))
        return unique / len(suffixes)

    def _unique_token_ratio(
        self, sequences: List[TokenSequence]
    ) -> float:
        """Mean ratio of unique tokens within each sequence."""
        if not sequences:
            return 0.0
        ratios: List[float] = []
        for seq in sequences:
            if seq:
                ratios.append(len(set(seq)) / len(seq))
        return float(np.mean(ratios)) if ratios else 0.0


# =========================================================================
# Token-level diversity during verification
# =========================================================================


class DiversityTokenSelector:
    """Select tokens during rejection sampling to maximise diversity."""

    def __init__(
        self,
        existing_sequences: List[TokenSequence],
        ngram_size: int = 3,
        diversity_weight: float = 0.2,
    ) -> None:
        self.existing_sequences = existing_sequences
        self.ngram_size = ngram_size
        self.diversity_weight = diversity_weight
        self._ngram_counts: Optional[Counter[Tuple[int, ...]]] = None

    def _build_counts(self) -> Counter[Tuple[int, ...]]:
        """Build n-gram frequency counts from existing sequences."""
        if self._ngram_counts is not None:
            return self._ngram_counts
        counts: Counter[Tuple[int, ...]] = collections.Counter()
        for seq in self.existing_sequences:
            for i in range(len(seq) - self.ngram_size + 1):
                counts[tuple(seq[i : i + self.ngram_size])] += 1
        self._ngram_counts = counts
        return counts

    def select_correction(
        self,
        residual_probs: np.ndarray,
        context: TokenSequence,
        rng: np.random.RandomState,
    ) -> int:
        """Sample a correction token with diversity bias."""
        counts = self._build_counts()
        modified = residual_probs.copy()
        vocab_size = len(modified)
        if len(context) >= self.ngram_size - 1:
            ctx_suffix = tuple(context[-(self.ngram_size - 1) :])
        else:
            ctx_suffix = tuple(context)
        for tok in range(vocab_size):
            candidate_ngram = ctx_suffix + (tok,)
            if len(candidate_ngram) >= self.ngram_size:
                candidate_ngram = candidate_ngram[-self.ngram_size :]
                freq = counts.get(candidate_ngram, 0)
                if freq > 0:
                    modified[tok] *= max(1.0 - self.diversity_weight * freq, _EPS)
                else:
                    modified[tok] *= (1.0 + self.diversity_weight)
        total = modified.sum()
        if total > _EPS:
            modified /= total
        else:
            modified = residual_probs.copy()
            total = modified.sum()
            if total > _EPS:
                modified /= total
        return _sample_from_probs(modified, rng)

    def update(self, sequences: List[TokenSequence]) -> None:
        """Update existing sequences and invalidate cache."""
        self.existing_sequences = sequences
        self._ngram_counts = None


# =========================================================================
# Multi-round verification
# =========================================================================


class MultiRoundVerifier:
    """Verify draft tokens in multiple rounds for higher acceptance."""

    def __init__(
        self,
        strategy: VerificationStrategy,
        draft_model: Any,
        max_rounds: int = 3,
    ) -> None:
        self.strategy = strategy
        self.draft_model = draft_model
        self.max_rounds = max(max_rounds, 1)

    def verify_multi_round(
        self,
        prefix: TokenSequence,
        initial_draft_tokens: TokenSequence,
        initial_draft_logits: List[LogitArray],
        target_logits_fn: Callable[[TokenSequence, int], List[LogitArray]],
        draft_temperature: float,
        target_temperature: float,
        rng: np.random.RandomState,
    ) -> Tuple[TokenSequence, int, Dict[str, Any]]:
        """Run multi-round verification.

        Returns (final_tokens, total_accepted, round_stats).
        """
        all_accepted: TokenSequence = []
        total_accepted = 0
        round_details: List[Dict[str, Any]] = []
        current_prefix = list(prefix)
        draft_tokens = initial_draft_tokens
        draft_logits = initial_draft_logits

        for round_idx in range(self.max_rounds):
            target_logits = target_logits_fn(
                current_prefix, len(draft_tokens) + 1
            )
            accepted, num_acc, mask = self.strategy.verify_full(
                draft_tokens, draft_logits, target_logits,
                draft_temperature, target_temperature, rng,
            )
            round_details.append({
                "round": round_idx,
                "proposed": len(draft_tokens),
                "accepted": num_acc,
                "total_output": len(accepted),
            })
            all_accepted.extend(accepted)
            total_accepted += num_acc
            current_prefix.extend(accepted)
            if num_acc == len(draft_tokens):
                break
            remaining = len(initial_draft_tokens) - len(all_accepted)
            if remaining <= 0:
                break
            draft_tokens, draft_logits = self.draft_model.draft_tokens(
                current_prefix, remaining, rng
            )
            if not draft_tokens:
                break

        stats: Dict[str, Any] = {
            "num_rounds": len(round_details),
            "total_accepted": total_accepted,
            "total_tokens": len(all_accepted),
            "rounds": round_details,
        }
        return all_accepted, total_accepted, stats


# =========================================================================
# KV-cache simulation
# =========================================================================


class KVCacheSimulator:
    """Simulate key-value cache management for speculative decoding."""

    def __init__(self, max_cache_entries: int = 1000) -> None:
        self.max_entries = max_cache_entries
        self._cache: Dict[Tuple[int, ...], float] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._access_order: List[Tuple[int, ...]] = []

    def lookup(self, prefix: TokenSequence) -> bool:
        """Check if prefix is in cache. Returns True on hit."""
        key = tuple(prefix)
        if key in self._cache:
            self._hits += 1
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return True
        self._misses += 1
        return False

    def insert(self, prefix: TokenSequence, cost: float = 1.0) -> None:
        """Insert a prefix into the cache."""
        key = tuple(prefix)
        if key in self._cache:
            self._cache[key] = cost
            return
        if len(self._cache) >= self.max_entries:
            self._evict()
        self._cache[key] = cost
        self._access_order.append(key)

    def _evict(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            victim = self._access_order.pop(0)
            self._cache.pop(victim, None)
            self._evictions += 1

    def invalidate_suffix(self, prefix: TokenSequence) -> int:
        """Invalidate all cache entries that extend *prefix*."""
        key = tuple(prefix)
        to_remove = [k for k in self._cache if k[:len(key)] == key and k != key]
        for k in to_remove:
            del self._cache[k]
            if k in self._access_order:
                self._access_order.remove(k)
        return len(to_remove)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "cache_size": self.size,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "evictions": self._evictions,
        }


# =========================================================================
# Cost model for speculative decoding analysis
# =========================================================================


class SpeculativeCostModel:
    """Analytical cost model for speculative decoding."""

    def __init__(
        self,
        draft_cost: float = 0.05,
        target_cost: float = 1.0,
        verify_overhead: float = 0.0,
    ) -> None:
        self.draft_cost = draft_cost
        self.target_cost = target_cost
        self.verify_overhead = verify_overhead

    def expected_tokens(self, alpha: float, gamma: int) -> float:
        """Expected number of tokens produced per step."""
        alpha = max(0.0, min(1.0, alpha))
        if alpha >= 1.0 - _EPS:
            return float(gamma + 1)
        return (1.0 - alpha ** gamma) / (1.0 - alpha + _EPS) + alpha ** gamma

    def cost_per_step(self, gamma: int) -> float:
        """Total cost of one speculative step (draft + verify)."""
        return gamma * self.draft_cost + self.target_cost + self.verify_overhead

    def throughput(self, alpha: float, gamma: int) -> float:
        """Expected tokens per unit cost."""
        expected = self.expected_tokens(alpha, gamma)
        cost = self.cost_per_step(gamma)
        return expected / max(cost, _EPS)

    def speedup(self, alpha: float, gamma: int) -> float:
        """Speedup relative to autoregressive decoding."""
        spec_throughput = self.throughput(alpha, gamma)
        ar_throughput = 1.0 / self.target_cost
        return spec_throughput / max(ar_throughput, _EPS)

    def optimal_gamma(
        self, alpha: float, max_gamma: int = _MAX_GAMMA
    ) -> Tuple[int, float]:
        """Find gamma that maximises speedup. Returns (gamma, speedup)."""
        best_gamma = 1
        best_s = 0.0
        for g in range(1, max_gamma + 1):
            s = self.speedup(alpha, g)
            if s > best_s:
                best_s = s
                best_gamma = g
        return best_gamma, best_s

    def break_even_acceptance_rate(self, gamma: int) -> float:
        """Find the acceptance rate at which speedup equals 1.0."""
        low, high = 0.0, 1.0
        for _ in range(100):
            mid = (low + high) / 2.0
            s = self.speedup(mid, gamma)
            if s < 1.0:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def efficiency_curve(
        self, gamma: int, num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (alpha_values, speedup_values) for plotting."""
        alphas = np.linspace(0.0, 1.0, num_points)
        speedups = np.array([self.speedup(a, gamma) for a in alphas])
        return alphas, speedups

    def full_analysis(
        self, max_gamma: int = _MAX_GAMMA
    ) -> Dict[str, Any]:
        """Run a full cost analysis across gamma and acceptance rates."""
        analysis: Dict[str, Any] = {}
        sample_alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        for alpha in sample_alphas:
            opt_g, opt_s = self.optimal_gamma(alpha, max_gamma)
            be_rate = self.break_even_acceptance_rate(opt_g)
            analysis[f"alpha_{alpha}"] = {
                "optimal_gamma": opt_g,
                "optimal_speedup": opt_s,
                "break_even_rate": be_rate,
            }
        analysis["draft_cost"] = self.draft_cost
        analysis["target_cost"] = self.target_cost
        analysis["verify_overhead"] = self.verify_overhead
        return analysis


# =========================================================================
# Verification strategy comparison
# =========================================================================


class VerificationComparator:
    """Compare different verification strategies on the same draft/target pairs."""

    def __init__(
        self,
        strategies: Optional[List[VerificationStrategy]] = None,
    ) -> None:
        if strategies is None:
            strategies = [
                StandardVerification(),
                DiversityAwareVerification(),
                RelaxedVerification(tolerance=0.1),
                TopKVerification(k=10),
            ]
        self.strategies = strategies

    def compare(
        self,
        draft_tokens: TokenSequence,
        draft_logits: List[LogitArray],
        target_logits: List[LogitArray],
        draft_temperature: float = 1.0,
        target_temperature: float = 1.0,
        num_trials: int = 100,
        seed: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """Run each strategy multiple times and compare acceptance statistics."""
        results: Dict[str, Dict[str, Any]] = {}
        for strategy in self.strategies:
            accepted_counts: List[int] = []
            total_lengths: List[int] = []
            for trial in range(num_trials):
                rng = np.random.RandomState(seed + trial)
                accepted, num_acc, mask = strategy.verify(
                    draft_tokens, draft_logits, target_logits,
                    draft_temperature, target_temperature, rng,
                )
                accepted_counts.append(num_acc)
                total_lengths.append(len(accepted))
            arr_acc = np.array(accepted_counts)
            arr_len = np.array(total_lengths)
            results[strategy.name()] = {
                "mean_accepted": float(np.mean(arr_acc)),
                "std_accepted": float(np.std(arr_acc)),
                "mean_total_length": float(np.mean(arr_len)),
                "acceptance_rate": float(np.mean(arr_acc)) / max(len(draft_tokens), 1),
                "full_acceptance_rate": float(
                    np.mean(arr_acc == len(draft_tokens))
                ),
                "median_accepted": float(np.median(arr_acc)),
            }
        return results


# =========================================================================
# Configuration presets
# =========================================================================


def make_fast_speculative_config(**overrides: Any) -> SpeculativeConfig:
    """Create a config optimised for speed."""
    defaults: Dict[str, Any] = {
        "algorithm_name": "speculative_fast",
        "gamma": 8,
        "draft_temperature": 1.0,
        "target_temperature": 1.0,
        "verification_strategy": "standard",
        "adaptive_gamma": True,
        "adaptive_gamma_target_rate": 0.8,
        "diversity_weight": 0.0,
    }
    defaults.update(overrides)
    return SpeculativeConfig(**defaults)


def make_diverse_speculative_config(**overrides: Any) -> SpeculativeConfig:
    """Create a config optimised for diversity."""
    defaults: Dict[str, Any] = {
        "algorithm_name": "speculative_diverse",
        "gamma": 5,
        "draft_temperature": 1.2,
        "target_temperature": 1.0,
        "verification_strategy": "diversity_aware",
        "diversity_weight": 0.2,
        "diversity_ngram_size": 3,
        "cross_sequence_penalty": 0.5,
        "adaptive_gamma": True,
        "adaptive_gamma_target_rate": 0.6,
        "parallel_drafts": 3,
    }
    defaults.update(overrides)
    return SpeculativeConfig(**defaults)


def make_tree_speculative_config(**overrides: Any) -> SpeculativeConfig:
    """Create a config for tree-based speculative decoding."""
    defaults: Dict[str, Any] = {
        "algorithm_name": "speculative_tree",
        "gamma": 5,
        "draft_temperature": 1.0,
        "target_temperature": 1.0,
        "verification_strategy": "standard",
        "enable_tree_speculation": True,
        "tree_width": 4,
        "tree_depth": 5,
        "tree_pruning_threshold": 0.01,
        "diversity_weight": 0.1,
        "adaptive_gamma": False,
    }
    defaults.update(overrides)
    return SpeculativeConfig(**defaults)


def make_relaxed_speculative_config(**overrides: Any) -> SpeculativeConfig:
    """Create a config with relaxed verification for higher throughput."""
    defaults: Dict[str, Any] = {
        "algorithm_name": "speculative_relaxed",
        "gamma": 10,
        "draft_temperature": 1.0,
        "target_temperature": 1.0,
        "verification_strategy": "relaxed",
        "relaxed_tolerance": 0.15,
        "adaptive_gamma": True,
        "adaptive_gamma_target_rate": 0.85,
        "diversity_weight": 0.05,
    }
    defaults.update(overrides)
    return SpeculativeConfig(**defaults)


# =========================================================================
# Batch utilities
# =========================================================================


def batch_speculative_decode(
    prefixes: List[TokenSequence],
    config: SpeculativeConfig,
    target_logit_fn: LogitSource,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
    draft_models: Optional[List[Any]] = None,
) -> Tuple[List[TokenSequence], Dict[str, Any]]:
    """Convenience function for batch speculative decoding."""
    if draft_models is None:
        main_draft = DraftModelFactory.create(config, vocab_size)
        draft_models = [main_draft]
    decoder = DiversitySpeculativeDecoder(
        config=config,
        draft_models=draft_models,
        target_logit_fn=target_logit_fn,
        vocab_size=vocab_size,
    )
    return decoder.generate_batch(prefixes, config.max_new_tokens)


def speculative_decode_with_analysis(
    prefix: TokenSequence,
    config: SpeculativeConfig,
    target_logit_fn: LogitSource,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
) -> Tuple[List[TokenSequence], Dict[str, Any], Dict[str, Any]]:
    """Generate sequences and produce an analysis report.

    Returns (sequences, generation_stats, analysis_report).
    """
    decoder = SpeculativeDecoderFactory.create(
        config, target_logit_fn, vocab_size
    )
    if isinstance(decoder, DiversitySpeculativeDecoder):
        sequences, stats = decoder.generate(prefix)
        analyzer = SpeculativeAnalyzer(decoder.tracker)
    else:
        sequences, stats = decoder.generate(prefix)
        analyzer = SpeculativeAnalyzer(decoder.tracker)
    report = analyzer.generate_report()
    diversity_metrics = SpeculativeDiversityMetrics().compute_all(sequences)
    report["diversity_metrics"] = diversity_metrics
    return sequences, stats, report


# =========================================================================
# End-to-end pipeline helper
# =========================================================================


def run_speculative_pipeline(
    prefix: TokenSequence,
    target_logit_fn: LogitSource,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
    num_sequences: int = 10,
    max_new_tokens: int = 50,
    gamma: int = 5,
    diversity_weight: float = 0.1,
    verification_strategy: str = "diversity_aware",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the full speculative decoding pipeline end-to-end.

    This is the highest-level API: creates config, draft model, decoder,
    runs generation, and produces an analysis report.

    Returns a dictionary with sequences, stats, analysis, and diversity
    metrics.
    """
    config = SpeculativeConfig(
        gamma=gamma,
        num_sequences=num_sequences,
        max_new_tokens=max_new_tokens,
        diversity_weight=diversity_weight,
        verification_strategy=verification_strategy,
        seed=seed,
        adaptive_gamma=True,
    )
    draft_model = DraftModelFactory.create(config, vocab_size)
    decoder = SpeculativeDecoder(
        config=config,
        draft_model=draft_model,
        target_logit_fn=target_logit_fn,
        vocab_size=vocab_size,
    )
    sequences, stats = decoder.generate(prefix)
    analyzer = SpeculativeAnalyzer(decoder.tracker)
    report = analyzer.generate_report()
    div_metrics = SpeculativeDiversityMetrics().compute_all(sequences)
    cost_model = SpeculativeCostModel()
    cost_analysis = cost_model.full_analysis()
    return {
        "sequences": sequences,
        "generation_stats": stats,
        "analysis_report": report,
        "diversity_metrics": div_metrics,
        "cost_analysis": cost_analysis,
        "config": config.to_dict(),
    }
