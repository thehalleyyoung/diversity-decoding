"""
Shared sampling utilities for the Diversity Decoding Arena.

Provides logit processors, beam management, sampling helpers,
token tracking / statistics, and batch sampling utilities used
across all concrete decoding algorithms.
"""

from __future__ import annotations

import abc
import copy
import logging
import math
import time
from collections import defaultdict
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
from scipy import stats as sp_stats

from src.algorithms.base import (
    DecodingConfig,
    LogitSource,
    TokenSequence,
    LogitArray,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_EPS = 1e-10
_LOGIT_NEG_INF = -float("inf")
_MIN_TEMPERATURE = 1e-7
_MAX_TEMPERATURE = 100.0

# ===================================================================== #
#                      1.  LogitProcessor hierarchy                      #
# ===================================================================== #


class LogitProcessor(abc.ABC):
    """Base class for all logit processors.

    A logit processor transforms a raw logit array (shape ``(vocab,)`` or
    ``(batch, vocab)``) in-place or returns a modified copy.  Processors
    are chained together via :func:`apply_processor_chain`.
    """

    # Optional human-readable name used for logging.
    name: str = "base"

    @abc.abstractmethod
    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        """Apply the processor and return the (possibly modified) logits.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)`` for single-sequence or
            ``(batch, vocab_size)`` for batched processing.
        input_ids : list of token sequences, optional
            Previously generated tokens – needed by processors that
            condition on history (e.g. repetition penalty).
        step : int
            Current decoding step (0-indexed).

        Returns
        -------
        np.ndarray
            Logits of the same shape, after processing.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (called at the start of a new generation)."""

    def _ensure_2d(self, logits: LogitArray) -> Tuple[LogitArray, bool]:
        """Return a 2-D view and a flag indicating whether the input was 1-D."""
        if logits.ndim == 1:
            return logits[np.newaxis, :], True
        return logits, False

    def _restore_shape(self, logits: LogitArray, was_1d: bool) -> LogitArray:
        if was_1d:
            return logits[0]
        return logits


# --------------------------------------------------------------------- #
# TemperatureProcessor
# --------------------------------------------------------------------- #

class TemperatureProcessor(LogitProcessor):
    """Scale logits by ``1 / temperature``.

    Supports *dynamic temperature schedules* controlled by a callback
    ``schedule_fn(step) -> temperature``.  Several built-in schedules
    are provided as class methods.
    """

    name = "temperature"

    def __init__(
        self,
        temperature: float = 1.0,
        schedule_fn: Optional[Callable[[int], float]] = None,
        min_temperature: float = _MIN_TEMPERATURE,
        max_temperature: float = _MAX_TEMPERATURE,
    ) -> None:
        self.base_temperature = temperature
        self.schedule_fn = schedule_fn
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    # -- built-in schedule factories ------------------------------------- #

    @classmethod
    def linear_warmup(
        cls,
        start: float,
        end: float,
        warmup_steps: int,
        **kwargs: Any,
    ) -> "TemperatureProcessor":
        """Linearly interpolate temperature from *start* to *end* over
        *warmup_steps*, then keep *end* constant."""

        def _schedule(step: int) -> float:
            if step >= warmup_steps:
                return end
            alpha = step / max(warmup_steps, 1)
            return start + alpha * (end - start)

        return cls(temperature=start, schedule_fn=_schedule, **kwargs)

    @classmethod
    def cosine_anneal(
        cls,
        high: float,
        low: float,
        period: int,
        **kwargs: Any,
    ) -> "TemperatureProcessor":
        """Cosine annealing between *high* and *low* with given *period*."""

        def _schedule(step: int) -> float:
            cos_val = math.cos(math.pi * (step % period) / max(period, 1))
            return low + 0.5 * (high - low) * (1.0 + cos_val)

        return cls(temperature=high, schedule_fn=_schedule, **kwargs)

    @classmethod
    def exponential_decay(
        cls,
        start: float,
        decay_rate: float,
        min_temp: float = _MIN_TEMPERATURE,
        **kwargs: Any,
    ) -> "TemperatureProcessor":
        """Exponentially decay temperature: ``start * decay_rate^step``."""

        def _schedule(step: int) -> float:
            return max(start * (decay_rate ** step), min_temp)

        return cls(temperature=start, schedule_fn=_schedule, **kwargs)

    @classmethod
    def step_schedule(
        cls,
        temperatures: List[Tuple[int, float]],
        **kwargs: Any,
    ) -> "TemperatureProcessor":
        """Step-wise schedule: a list of ``(step_threshold, temperature)``
        pairs sorted ascending.  The temperature for the largest threshold
        ≤ current step is used."""

        sorted_temps = sorted(temperatures, key=lambda t: t[0])

        def _schedule(step: int) -> float:
            temp = sorted_temps[0][1]
            for threshold, t in sorted_temps:
                if step >= threshold:
                    temp = t
                else:
                    break
            return temp

        return cls(temperature=sorted_temps[0][1], schedule_fn=_schedule, **kwargs)

    # -- call ------------------------------------------------------------- #

    def _get_temperature(self, step: int) -> float:
        if self.schedule_fn is not None:
            t = self.schedule_fn(step)
        else:
            t = self.base_temperature
        return float(np.clip(t, self.min_temperature, self.max_temperature))

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        temperature = self._get_temperature(step)
        if abs(temperature - 1.0) < 1e-9:
            return logits
        return logits / temperature


# --------------------------------------------------------------------- #
# TopKProcessor
# --------------------------------------------------------------------- #

class TopKProcessor(LogitProcessor):
    """Keep only the top-*k* logits, setting others to ``-inf``."""

    name = "top_k"

    def __init__(self, k: int = 50, min_tokens_to_keep: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.min_tokens_to_keep = max(min_tokens_to_keep, 1)

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        logits, was_1d = self._ensure_2d(logits)
        effective_k = max(self.k, self.min_tokens_to_keep)
        effective_k = min(effective_k, logits.shape[-1])

        for i in range(logits.shape[0]):
            row = logits[i]
            if effective_k >= row.shape[0]:
                continue
            threshold_idx = np.argpartition(row, -effective_k)[-effective_k:]
            threshold_val = np.min(row[threshold_idx])
            mask = row < threshold_val
            logits[i, mask] = _LOGIT_NEG_INF

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# TopPProcessor  (nucleus sampling)
# --------------------------------------------------------------------- #

class TopPProcessor(LogitProcessor):
    """Filter tokens so cumulative probability ≥ *p* is kept (nucleus)."""

    name = "top_p"

    def __init__(self, p: float = 0.9, min_tokens_to_keep: int = 1) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p
        self.min_tokens_to_keep = max(min_tokens_to_keep, 1)

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            row = logits[i].copy()
            sorted_indices = np.argsort(row)[::-1]
            sorted_logits = row[sorted_indices]

            # convert to probabilities
            max_logit = sorted_logits[0]
            shifted = sorted_logits - max_logit
            probs = np.exp(shifted)
            probs /= probs.sum() + _LOG_EPS

            cumulative = np.cumsum(probs)

            # find cutoff: keep tokens until cumulative >= p
            cutoff_mask = cumulative - probs > self.p
            # always keep at least min_tokens_to_keep
            cutoff_mask[: self.min_tokens_to_keep] = False

            remove_indices = sorted_indices[cutoff_mask]
            logits[i, remove_indices] = _LOGIT_NEG_INF

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# TypicalProcessor
# --------------------------------------------------------------------- #

class TypicalProcessor(LogitProcessor):
    """Locally typical sampling (Meister et al., 2023).

    Keep tokens whose information content is close to the conditional
    entropy, accumulating mass until the threshold *p* is reached.
    """

    name = "typical"

    def __init__(self, p: float = 0.9, min_tokens_to_keep: int = 1) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p
        self.min_tokens_to_keep = max(min_tokens_to_keep, 1)

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            row = logits[i].copy()
            # softmax
            max_val = np.max(row)
            shifted = row - max_val
            probs = np.exp(shifted)
            probs /= probs.sum() + _LOG_EPS

            # information content per token  -log p(x)
            log_probs = np.log(probs + _LOG_EPS)
            neg_info = -log_probs

            # conditional entropy H = -sum p log p
            entropy = -np.sum(probs * log_probs)

            # deviation from entropy
            deviation = np.abs(neg_info - entropy)

            # sort by deviation ascending (closest to entropy first)
            sorted_indices = np.argsort(deviation)
            sorted_probs = probs[sorted_indices]
            cumulative = np.cumsum(sorted_probs)

            # keep until cumulative >= p
            cutoff_mask = cumulative - sorted_probs > self.p
            cutoff_mask[: self.min_tokens_to_keep] = False

            remove_indices = sorted_indices[cutoff_mask]
            logits[i, remove_indices] = _LOGIT_NEG_INF

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# MinPProcessor
# --------------------------------------------------------------------- #

class MinPProcessor(LogitProcessor):
    """Min-p sampling: remove tokens whose probability is below
    ``min_p * max_probability``."""

    name = "min_p"

    def __init__(self, min_p: float = 0.05, min_tokens_to_keep: int = 1) -> None:
        if not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {min_p}")
        self.min_p = min_p
        self.min_tokens_to_keep = max(min_tokens_to_keep, 1)

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        if self.min_p <= 0.0:
            return logits

        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            row = logits[i].copy()
            max_val = np.max(row)
            shifted = row - max_val
            probs = np.exp(shifted)
            probs /= probs.sum() + _LOG_EPS

            max_prob = np.max(probs)
            threshold = self.min_p * max_prob

            # tokens below threshold
            below = probs < threshold

            # ensure we keep at least min_tokens_to_keep
            if np.sum(~below) < self.min_tokens_to_keep:
                # keep top-min_tokens_to_keep by probability
                top_indices = np.argpartition(probs, -self.min_tokens_to_keep)[
                    -self.min_tokens_to_keep :
                ]
                below[:] = True
                below[top_indices] = False

            logits[i, below] = _LOGIT_NEG_INF

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# RepetitionPenaltyProcessor
# --------------------------------------------------------------------- #

class RepetitionPenaltyProcessor(LogitProcessor):
    """Combined repetition penalty with frequency, presence, and n-gram
    blocking components.

    * *repetition_penalty*: multiplicative penalty on previously seen
      token logits (> 1 penalises, < 1 encourages).
    * *frequency_penalty*: additive penalty proportional to token count.
    * *presence_penalty*: additive penalty applied once per unique token.
    * *ngram_block_sizes*: list of n-gram sizes to hard-block.
    """

    name = "repetition_penalty"

    def __init__(
        self,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ngram_block_sizes: Optional[List[int]] = None,
        max_history: int = 1024,
    ) -> None:
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.ngram_block_sizes = ngram_block_sizes or []
        self.max_history = max_history

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _count_tokens(token_ids: TokenSequence, max_history: int) -> Dict[int, int]:
        """Count occurrences of each token in the (optionally truncated) history."""
        history = token_ids[-max_history:] if max_history > 0 else token_ids
        counts: Dict[int, int] = defaultdict(int)
        for t in history:
            counts[t] += 1
        return dict(counts)

    @staticmethod
    def _blocked_tokens_from_ngrams(
        token_ids: TokenSequence,
        ngram_sizes: List[int],
    ) -> Set[int]:
        """Return the set of token ids that would complete a repeated n-gram."""
        blocked: Set[int] = set()
        for n in ngram_sizes:
            if n < 1 or len(token_ids) < n:
                continue
            # the last (n-1) tokens form a partial n-gram
            prefix = tuple(token_ids[-(n - 1) :]) if n > 1 else ()
            # scan history for occurrences of that prefix
            for start in range(len(token_ids) - n + 1):
                end = start + n - 1
                if end >= len(token_ids):
                    break
                candidate_prefix = tuple(token_ids[start : start + n - 1]) if n > 1 else ()
                if candidate_prefix == prefix:
                    next_token = token_ids[start + n - 1]
                    blocked.add(next_token)
        return blocked

    # -- call ------------------------------------------------------------- #

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        if input_ids is None:
            return logits

        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            if i >= len(input_ids):
                continue
            token_list = input_ids[i]
            if len(token_list) == 0:
                continue

            # multiplicative repetition penalty
            if abs(self.repetition_penalty - 1.0) > 1e-9:
                unique_tokens = list(set(token_list[-self.max_history :]))
                token_arr = np.array(unique_tokens, dtype=np.intp)
                valid_mask = (token_arr >= 0) & (token_arr < vocab_size)
                token_arr = token_arr[valid_mask]
                if token_arr.size > 0:
                    selected = logits[i, token_arr]
                    # positive logits are divided; negative logits are multiplied
                    pos_mask = selected > 0
                    selected[pos_mask] /= self.repetition_penalty
                    selected[~pos_mask] *= self.repetition_penalty
                    logits[i, token_arr] = selected

            # frequency and presence penalties
            if abs(self.frequency_penalty) > 1e-9 or abs(self.presence_penalty) > 1e-9:
                counts = self._count_tokens(token_list, self.max_history)
                for tid, count in counts.items():
                    if 0 <= tid < vocab_size:
                        logits[i, tid] -= self.frequency_penalty * count
                        logits[i, tid] -= self.presence_penalty

            # n-gram blocking
            if self.ngram_block_sizes:
                blocked = self._blocked_tokens_from_ngrams(
                    token_list, self.ngram_block_sizes,
                )
                for tid in blocked:
                    if 0 <= tid < vocab_size:
                        logits[i, tid] = _LOGIT_NEG_INF

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# LengthPenaltyProcessor
# --------------------------------------------------------------------- #

class LengthPenaltyProcessor(LogitProcessor):
    """Apply a length-based penalty to all logits.

    Two modes:
    * **normalise**: divide logits by ``(step + base) ^ alpha``
      (the classic GNMT length penalty).
    * **exponential**: multiply logits by ``exp(-beta * step)``.
    """

    name = "length_penalty"

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        base: float = 5.0,
        mode: str = "normalise",
        eos_token_id: Optional[int] = None,
        eos_bonus: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.base = base
        if mode not in ("normalise", "exponential", "both"):
            raise ValueError(f"Unknown length penalty mode: {mode}")
        self.mode = mode
        self.eos_token_id = eos_token_id
        self.eos_bonus = eos_bonus

    def _compute_factor(self, step: int) -> float:
        if self.mode == "normalise":
            denom = ((self.base + step) / (self.base + 1)) ** self.alpha
            return 1.0 / max(denom, _LOG_EPS)
        elif self.mode == "exponential":
            return math.exp(-self.beta * step)
        else:  # both
            denom = ((self.base + step) / (self.base + 1)) ** self.alpha
            norm_factor = 1.0 / max(denom, _LOG_EPS)
            exp_factor = math.exp(-self.beta * step)
            return norm_factor * exp_factor

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        factor = self._compute_factor(step)
        if abs(factor - 1.0) < 1e-9 and abs(self.eos_bonus) < 1e-9:
            return logits

        logits = logits * factor

        if self.eos_token_id is not None and abs(self.eos_bonus) > 1e-9:
            if logits.ndim == 1:
                if self.eos_token_id < logits.shape[0]:
                    logits[self.eos_token_id] += self.eos_bonus * step
            else:
                if self.eos_token_id < logits.shape[-1]:
                    logits[:, self.eos_token_id] += self.eos_bonus * step

        return logits


# --------------------------------------------------------------------- #
# NoRepeatNgramProcessor
# --------------------------------------------------------------------- #

class NoRepeatNgramProcessor(LogitProcessor):
    """Hard-block any token that would create a repeated n-gram."""

    name = "no_repeat_ngram"

    def __init__(self, ngram_size: int = 3) -> None:
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")
        self.ngram_size = ngram_size

    def _get_banned_tokens(
        self, token_ids: TokenSequence, vocab_size: int,
    ) -> np.ndarray:
        """Return a boolean mask of shape ``(vocab_size,)`` for banned tokens."""
        banned = np.zeros(vocab_size, dtype=bool)
        n = self.ngram_size
        if len(token_ids) < n:
            return banned

        # build n-gram table
        ngrams: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        for idx in range(len(token_ids) - n + 1):
            gram = tuple(token_ids[idx : idx + n])
            prefix = gram[:-1]
            suffix = gram[-1]
            ngrams[prefix].add(suffix)

        # the current prefix is the last (n-1) tokens
        if n > 1:
            current_prefix = tuple(token_ids[-(n - 1) :])
        else:
            current_prefix = ()

        for tid in ngrams.get(current_prefix, set()):
            if 0 <= tid < vocab_size:
                banned[tid] = True

        return banned

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        if input_ids is None:
            return logits

        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            if i >= len(input_ids):
                continue
            banned = self._get_banned_tokens(input_ids[i], vocab_size)
            logits[i, banned] = _LOGIT_NEG_INF

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# EntropyBasedProcessor
# --------------------------------------------------------------------- #

class EntropyBasedProcessor(LogitProcessor):
    """Entropy-aware temperature scaling.

    When the distribution has **high entropy** (model is uncertain),
    lower the temperature to sharpen.  When entropy is **low** (model
    is confident), raise it to encourage diversity.

    ``effective_temp = base_temp * (target_entropy / (current_entropy + eps)) ^ exponent``
    """

    name = "entropy_based"

    def __init__(
        self,
        base_temperature: float = 1.0,
        target_entropy: float = 3.0,
        exponent: float = 0.5,
        min_temperature: float = _MIN_TEMPERATURE,
        max_temperature: float = _MAX_TEMPERATURE,
        smoothing: float = 0.0,
    ) -> None:
        self.base_temperature = base_temperature
        self.target_entropy = target_entropy
        self.exponent = exponent
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.smoothing = smoothing
        self._prev_temperature: Optional[float] = None

    def reset(self) -> None:
        self._prev_temperature = None

    def _entropy_from_logits(self, row: np.ndarray) -> float:
        """Compute Shannon entropy (nats) from a 1-D logit vector."""
        shifted = row - np.max(row)
        probs = np.exp(shifted)
        probs /= probs.sum() + _LOG_EPS
        log_probs = np.log(probs + _LOG_EPS)
        return float(-np.sum(probs * log_probs))

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            entropy = self._entropy_from_logits(logits[i])
            ratio = self.target_entropy / (entropy + _LOG_EPS)
            temp = self.base_temperature * (ratio ** self.exponent)
            temp = float(np.clip(temp, self.min_temperature, self.max_temperature))

            # exponential moving average smoothing
            if self.smoothing > 0.0 and self._prev_temperature is not None:
                temp = self.smoothing * self._prev_temperature + (1.0 - self.smoothing) * temp
            self._prev_temperature = temp

            if abs(temp - 1.0) > 1e-9:
                logits[i] = logits[i] / temp

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# ContrastiveProcessor
# --------------------------------------------------------------------- #

class ContrastiveProcessor(LogitProcessor):
    """Contrastive search penalty (Su et al., 2022).

    Penalises tokens that are *too similar* to the last ``context_window``
    tokens based on a provided similarity matrix or simple logit overlap.

    When no embedding matrix is available the processor falls back to
    penalising tokens that had high probability at recent steps, stored
    in an internal rolling buffer.
    """

    name = "contrastive"

    def __init__(
        self,
        alpha: float = 0.6,
        context_window: int = 5,
        embedding_matrix: Optional[np.ndarray] = None,
        penalty_type: str = "cosine",
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Weight of the contrastive (degeneration) penalty.  0 = no
            penalty, 1 = full penalty.
        context_window : int
            How many recent tokens to consider for similarity.
        embedding_matrix : np.ndarray, optional
            Shape ``(vocab_size, dim)``.  If provided, cosine similarity
            is used.  Otherwise a logit-probability heuristic is used.
        penalty_type : str
            ``"cosine"`` (requires embeddings) or ``"probability"``.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.context_window = context_window
        self.embedding_matrix = embedding_matrix
        self.penalty_type = penalty_type
        self._recent_probs: List[np.ndarray] = []

    def reset(self) -> None:
        self._recent_probs = []

    def _cosine_sim(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + _LOG_EPS
        return float(np.dot(vec_a, vec_b) / denom)

    def _compute_similarity_penalty(
        self,
        logits_row: np.ndarray,
        token_ids: TokenSequence,
        vocab_size: int,
    ) -> np.ndarray:
        """Return an additive penalty array of shape ``(vocab_size,)``."""
        penalty = np.zeros(vocab_size, dtype=np.float64)
        context = token_ids[-self.context_window :]

        if self.embedding_matrix is not None and self.penalty_type == "cosine":
            for prev_tid in context:
                if 0 <= prev_tid < self.embedding_matrix.shape[0]:
                    prev_emb = self.embedding_matrix[prev_tid]
                    # similarity to every candidate token
                    norms = np.linalg.norm(self.embedding_matrix, axis=1) + _LOG_EPS
                    prev_norm = np.linalg.norm(prev_emb) + _LOG_EPS
                    sims = self.embedding_matrix @ prev_emb / (norms * prev_norm)
                    penalty += np.clip(sims, 0.0, 1.0)
            if len(context) > 0:
                penalty /= len(context)
        else:
            # probability-based fallback
            if self._recent_probs:
                for past_probs in self._recent_probs[-self.context_window :]:
                    overlap_size = min(past_probs.shape[0], vocab_size)
                    penalty[:overlap_size] += past_probs[:overlap_size]
                penalty /= max(len(self._recent_probs[-self.context_window :]), 1)

        return penalty

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        if self.alpha <= 0.0:
            return logits

        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            tokens = input_ids[i] if (input_ids is not None and i < len(input_ids)) else []
            penalty = self._compute_similarity_penalty(logits[i], tokens, vocab_size)

            # combine: logits = (1 - alpha)*logits - alpha*penalty_scaled
            max_logit = np.max(logits[i])
            penalty_scaled = penalty * max_logit if max_logit > 0 else penalty
            logits[i] = (1.0 - self.alpha) * logits[i] - self.alpha * penalty_scaled

        # store current probs for future fallback penalty
        shifted = logits[0] - np.max(logits[0])
        probs = np.exp(shifted)
        probs /= probs.sum() + _LOG_EPS
        self._recent_probs.append(probs.copy())
        if len(self._recent_probs) > self.context_window + 5:
            self._recent_probs = self._recent_probs[-(self.context_window + 5) :]

        return self._restore_shape(logits, was_1d)


# --------------------------------------------------------------------- #
# DiversityBoostProcessor
# --------------------------------------------------------------------- #

class DiversityBoostProcessor(LogitProcessor):
    """Penalise tokens that have already been used (across all sequences)
    to encourage lexical diversity.

    Maintains an internal counter updated from ``input_ids`` each call.
    The penalty is proportional to the log-count of each token.
    """

    name = "diversity_boost"

    def __init__(
        self,
        penalty_weight: float = 1.0,
        decay: float = 0.99,
        cross_sequence: bool = True,
        temperature_boost: float = 0.0,
        count_exponent: float = 0.5,
    ) -> None:
        self.penalty_weight = penalty_weight
        self.decay = decay
        self.cross_sequence = cross_sequence
        self.temperature_boost = temperature_boost
        self.count_exponent = count_exponent
        self._global_counts: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._global_counts = None

    def _update_counts(
        self, input_ids: List[TokenSequence], vocab_size: int,
    ) -> np.ndarray:
        """Update and return the global token-count array."""
        if self._global_counts is None or self._global_counts.shape[0] != vocab_size:
            self._global_counts = np.zeros(vocab_size, dtype=np.float64)
        else:
            self._global_counts *= self.decay

        for seq in input_ids:
            for tid in seq:
                if 0 <= tid < vocab_size:
                    self._global_counts[tid] += 1.0

        return self._global_counts

    def __call__(
        self,
        logits: LogitArray,
        input_ids: Optional[List[TokenSequence]] = None,
        step: int = 0,
    ) -> LogitArray:
        if input_ids is None or self.penalty_weight <= 0.0:
            return logits

        logits, was_1d = self._ensure_2d(logits)
        batch_size, vocab_size = logits.shape

        if self.cross_sequence:
            counts = self._update_counts(input_ids, vocab_size)
        else:
            counts = np.zeros(vocab_size, dtype=np.float64)

        for i in range(batch_size):
            if not self.cross_sequence:
                seq_counts = np.zeros(vocab_size, dtype=np.float64)
                if i < len(input_ids):
                    for tid in input_ids[i]:
                        if 0 <= tid < vocab_size:
                            seq_counts[tid] += 1.0
                local_counts = seq_counts
            else:
                local_counts = counts

            penalty = self.penalty_weight * np.power(
                local_counts + 1.0, self.count_exponent,
            ) - self.penalty_weight
            logits[i] -= penalty

            # optional temperature boost for diversity
            if self.temperature_boost > 0.0:
                avg_count = np.mean(local_counts)
                boost = 1.0 + self.temperature_boost * np.log1p(avg_count)
                logits[i] /= boost

        return self._restore_shape(logits, was_1d)


# ===================================================================== #
#                       2.  SamplingConfig                               #
# ===================================================================== #


@dataclass
class SamplingConfig:
    """Central configuration for all sampling parameters.

    Mirrors the union of knobs exposed by the individual processors
    so that a single config object can drive an entire pipeline.
    """

    # temperature
    temperature: float = 1.0
    temperature_schedule: Optional[str] = None  # "linear", "cosine", "exp", "step"
    temperature_schedule_params: Dict[str, Any] = field(default_factory=dict)

    # top-k
    top_k: int = 0
    top_k_min_tokens: int = 1

    # nucleus / top-p
    top_p: float = 1.0
    top_p_min_tokens: int = 1

    # typical
    typical_p: float = 1.0
    typical_min_tokens: int = 1

    # min-p
    min_p: float = 0.0

    # repetition penalty
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0

    # length penalty
    length_penalty_alpha: float = 0.0
    length_penalty_beta: float = 0.0
    length_penalty_mode: str = "normalise"

    # entropy-based
    entropy_based: bool = False
    entropy_target: float = 3.0
    entropy_exponent: float = 0.5
    entropy_smoothing: float = 0.0

    # contrastive
    contrastive_alpha: float = 0.0
    contrastive_window: int = 5

    # diversity boost
    diversity_penalty: float = 0.0
    diversity_decay: float = 0.99
    diversity_cross_sequence: bool = True
    diversity_count_exponent: float = 0.5
    diversity_temperature_boost: float = 0.0

    # beam search
    num_beams: int = 1
    num_beam_groups: int = 1
    beam_diversity_penalty: float = 0.0
    beam_length_penalty: float = 1.0

    # misc
    seed: Optional[int] = None
    do_sample: bool = True
    num_return_sequences: int = 1
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # serialisation ------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    # convenience --------------------------------------------------------- #

    def build_processor_chain(
        self,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> List[LogitProcessor]:
        """Instantiate and return an ordered list of processors matching
        the current configuration."""
        chain: List[LogitProcessor] = []

        # repetition penalty first (needs unmodified logits)
        if (
            abs(self.repetition_penalty - 1.0) > 1e-9
            or abs(self.frequency_penalty) > 1e-9
            or abs(self.presence_penalty) > 1e-9
        ):
            chain.append(
                RepetitionPenaltyProcessor(
                    repetition_penalty=self.repetition_penalty,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                )
            )

        if self.no_repeat_ngram_size > 0:
            chain.append(NoRepeatNgramProcessor(ngram_size=self.no_repeat_ngram_size))

        # contrastive
        if self.contrastive_alpha > 0.0:
            chain.append(
                ContrastiveProcessor(
                    alpha=self.contrastive_alpha,
                    context_window=self.contrastive_window,
                    embedding_matrix=embedding_matrix,
                )
            )

        # diversity boost
        if self.diversity_penalty > 0.0:
            chain.append(
                DiversityBoostProcessor(
                    penalty_weight=self.diversity_penalty,
                    decay=self.diversity_decay,
                    cross_sequence=self.diversity_cross_sequence,
                    temperature_boost=self.diversity_temperature_boost,
                    count_exponent=self.diversity_count_exponent,
                )
            )

        # length penalty
        if abs(self.length_penalty_alpha) > 1e-9 or abs(self.length_penalty_beta) > 1e-9:
            chain.append(
                LengthPenaltyProcessor(
                    alpha=self.length_penalty_alpha,
                    beta=self.length_penalty_beta,
                    mode=self.length_penalty_mode,
                    eos_token_id=self.eos_token_id,
                )
            )

        # entropy-based temperature
        if self.entropy_based:
            chain.append(
                EntropyBasedProcessor(
                    base_temperature=self.temperature,
                    target_entropy=self.entropy_target,
                    exponent=self.entropy_exponent,
                    smoothing=self.entropy_smoothing,
                )
            )
        else:
            # fixed / scheduled temperature
            schedule_fn = self._build_schedule_fn()
            chain.append(
                TemperatureProcessor(
                    temperature=self.temperature,
                    schedule_fn=schedule_fn,
                )
            )

        # filtering processors (after temperature)
        if self.top_k > 0:
            chain.append(TopKProcessor(k=self.top_k, min_tokens_to_keep=self.top_k_min_tokens))

        if self.top_p < 1.0:
            chain.append(TopPProcessor(p=self.top_p, min_tokens_to_keep=self.top_p_min_tokens))

        if self.typical_p < 1.0:
            chain.append(TypicalProcessor(p=self.typical_p, min_tokens_to_keep=self.typical_min_tokens))

        if self.min_p > 0.0:
            chain.append(MinPProcessor(min_p=self.min_p))

        return chain

    def _build_schedule_fn(self) -> Optional[Callable[[int], float]]:
        """Construct a temperature schedule function from config strings."""
        stype = self.temperature_schedule
        if stype is None:
            return None
        params = self.temperature_schedule_params
        if stype == "linear":
            start = params.get("start", self.temperature)
            end = params.get("end", self.temperature)
            steps = params.get("warmup_steps", 50)

            def _linear(step: int) -> float:
                if step >= steps:
                    return end
                alpha = step / max(steps, 1)
                return start + alpha * (end - start)

            return _linear
        elif stype == "cosine":
            high = params.get("high", self.temperature)
            low = params.get("low", 0.1)
            period = params.get("period", 100)

            def _cosine(step: int) -> float:
                cos_val = math.cos(math.pi * (step % period) / max(period, 1))
                return low + 0.5 * (high - low) * (1.0 + cos_val)

            return _cosine
        elif stype == "exp":
            start = params.get("start", self.temperature)
            rate = params.get("decay_rate", 0.99)
            min_t = params.get("min_temp", _MIN_TEMPERATURE)

            def _exp(step: int) -> float:
                return max(start * (rate ** step), min_t)

            return _exp
        elif stype == "step":
            thresholds: List[Tuple[int, float]] = params.get(
                "thresholds", [(0, self.temperature)],
            )
            sorted_t = sorted(thresholds, key=lambda x: x[0])

            def _step(step: int) -> float:
                temp = sorted_t[0][1]
                for thr, t in sorted_t:
                    if step >= thr:
                        temp = t
                    else:
                        break
                return temp

            return _step
        else:
            logger.warning("Unknown temperature schedule '%s', ignoring.", stype)
            return None


# ===================================================================== #
#                      3.  Beam management utilities                     #
# ===================================================================== #


@dataclass
class BeamState:
    """State of a single beam hypothesis."""

    token_ids: TokenSequence = field(default_factory=list)
    log_prob: float = 0.0
    score: float = 0.0
    is_finished: bool = False
    length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # derived scores ------------------------------------------------------- #

    def length_normalised_score(self, alpha: float = 0.6, base: float = 5.0) -> float:
        """GNMT-style length normalisation."""
        lp = ((base + self.length) / (base + 1)) ** alpha
        return self.log_prob / max(lp, _LOG_EPS)

    def coverage_score(self, attention_weights: Optional[np.ndarray] = None) -> float:
        """Optional coverage penalty (set 0 when no attention is available)."""
        if attention_weights is None:
            return 0.0
        # sum of log min(attn, 1) across source positions
        clamped = np.minimum(attention_weights, 1.0)
        return float(np.sum(np.log(clamped + _LOG_EPS)))

    def clone(self) -> "BeamState":
        return BeamState(
            token_ids=list(self.token_ids),
            log_prob=self.log_prob,
            score=self.score,
            is_finished=self.is_finished,
            length=self.length,
            metadata=copy.deepcopy(self.metadata),
        )

    def extend(self, token_id: int, token_log_prob: float) -> "BeamState":
        new = self.clone()
        new.token_ids.append(token_id)
        new.log_prob += token_log_prob
        new.length += 1
        return new


class BeamManager:
    """Manages a pool of beam hypotheses with pruning, scoring, and merging.

    Supports *diverse beam groups*, where beams are partitioned into
    groups and an inter-group diversity penalty is applied.
    """

    def __init__(
        self,
        num_beams: int = 5,
        num_groups: int = 1,
        diversity_penalty: float = 0.0,
        length_penalty_alpha: float = 0.6,
        length_penalty_base: float = 5.0,
        max_capacity: int = 0,
        early_stopping: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {num_beams}")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if num_beams % num_groups != 0:
            raise ValueError(
                f"num_beams ({num_beams}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.num_beams = num_beams
        self.num_groups = num_groups
        self.beams_per_group = num_beams // num_groups
        self.diversity_penalty = diversity_penalty
        self.length_penalty_alpha = length_penalty_alpha
        self.length_penalty_base = length_penalty_base
        self.max_capacity = max_capacity if max_capacity > 0 else num_beams * 4
        self.early_stopping = early_stopping
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # beams[group_idx] -> list of BeamState
        self.beams: List[List[BeamState]] = [
            [] for _ in range(num_groups)
        ]
        self.finished: List[BeamState] = []
        self._step = 0

    # -- initialisation --------------------------------------------------- #

    def init_beams(self, prefix: Optional[TokenSequence] = None) -> None:
        """Initialise beams with an optional shared prefix."""
        for g in range(self.num_groups):
            self.beams[g] = []
            for _ in range(self.beams_per_group):
                state = BeamState(
                    token_ids=list(prefix) if prefix else [],
                    log_prob=0.0,
                    score=0.0,
                    length=0,
                )
                self.beams[g].append(state)
        self.finished = []
        self._step = 0

    # -- scoring ---------------------------------------------------------- #

    def score_beam(self, beam: BeamState) -> float:
        return beam.length_normalised_score(
            alpha=self.length_penalty_alpha,
            base=self.length_penalty_base,
        )

    def _apply_diversity_penalty(
        self,
        group_idx: int,
        logits: np.ndarray,
    ) -> np.ndarray:
        """Subtract a penalty from logits for tokens already selected
        by earlier groups at this step."""
        if self.diversity_penalty <= 0.0 or group_idx == 0:
            return logits

        penalty = np.zeros(logits.shape[-1], dtype=np.float64)
        for prev_g in range(group_idx):
            for beam in self.beams[prev_g]:
                if beam.token_ids:
                    last_token = beam.token_ids[-1]
                    if 0 <= last_token < logits.shape[-1]:
                        penalty[last_token] += self.diversity_penalty

        if logits.ndim == 1:
            return logits - penalty
        else:
            return logits - penalty[np.newaxis, :]

    # -- expansion -------------------------------------------------------- #

    def expand_beams(
        self,
        group_idx: int,
        logits: np.ndarray,
        vocab_size: int,
    ) -> List[BeamState]:
        """Expand beams in a single group by considering top-k extensions.

        Parameters
        ----------
        group_idx : int
            Which beam group to expand.
        logits : np.ndarray
            Shape ``(beams_per_group, vocab_size)`` – log-probabilities
            for each current beam.
        vocab_size : int
            Vocabulary size.

        Returns
        -------
        list of BeamState
            The new set of beams for this group (pruned to
            ``beams_per_group``).
        """
        logits = self._apply_diversity_penalty(group_idx, logits)

        current_beams = self.beams[group_idx]
        if len(current_beams) == 0:
            return []

        # log-softmax
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(
            np.sum(np.exp(log_probs), axis=-1, keepdims=True) + _LOG_EPS,
        )

        candidates: List[Tuple[float, int, int]] = []  # (score, beam_idx, token_id)

        for beam_idx, beam in enumerate(current_beams):
            if beam.is_finished:
                candidates.append((self.score_beam(beam), beam_idx, -1))
                continue

            row = log_probs[beam_idx] if log_probs.ndim == 2 else log_probs
            # top 2*beams_per_group candidates per beam
            k = min(2 * self.beams_per_group, vocab_size)
            top_indices = np.argpartition(row, -k)[-k:]
            for tid in top_indices:
                new_log_prob = beam.log_prob + row[tid]
                new_state = beam.extend(int(tid), float(row[tid]))
                score = self.score_beam(new_state)
                candidates.append((score, beam_idx, int(tid)))

        # sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        new_beams: List[BeamState] = []
        for score, beam_idx, token_id in candidates:
            if len(new_beams) >= self.beams_per_group:
                break
            beam = current_beams[beam_idx]
            if token_id < 0:
                # finished beam carried forward
                new_beams.append(beam.clone())
                continue
            new_state = beam.extend(token_id, 0.0)
            # recompute log_prob properly
            row = log_probs[beam_idx] if log_probs.ndim == 2 else log_probs
            new_state.log_prob = beam.log_prob + float(row[token_id])
            new_state.score = self.score_beam(new_state)

            # check EOS
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                new_state.is_finished = True
                self.finished.append(new_state)
            else:
                new_beams.append(new_state)

        # pad with carried-over beams if we don't have enough
        while len(new_beams) < self.beams_per_group and current_beams:
            new_beams.append(current_beams[len(new_beams) % len(current_beams)].clone())

        return new_beams

    def step(
        self,
        all_logits: np.ndarray,
        vocab_size: int,
    ) -> None:
        """Advance all groups by one step.

        Parameters
        ----------
        all_logits : np.ndarray
            Shape ``(num_beams, vocab_size)`` – logits for every active beam,
            ordered by group then beam index within group.
        """
        offset = 0
        for g in range(self.num_groups):
            n = self.beams_per_group
            group_logits = all_logits[offset : offset + n]
            self.beams[g] = self.expand_beams(g, group_logits, vocab_size)
            offset += n
        self._step += 1

    # -- retrieval / merging ---------------------------------------------- #

    def all_active_beams(self) -> List[BeamState]:
        """Return a flat list of all active (non-finished) beams."""
        result: List[BeamState] = []
        for group in self.beams:
            result.extend(b for b in group if not b.is_finished)
        return result

    def all_beams_flat(self) -> List[BeamState]:
        result: List[BeamState] = []
        for group in self.beams:
            result.extend(group)
        return result

    def best_finished(self, n: int = 1) -> List[BeamState]:
        """Return the top-*n* finished hypotheses by score."""
        scored = [(self.score_beam(b), b) for b in self.finished]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:n]]

    def best_overall(self, n: int = 1) -> List[BeamState]:
        """Return top-*n* from both finished and active beams."""
        all_beams = list(self.finished)
        for group in self.beams:
            all_beams.extend(group)
        scored = [(self.score_beam(b), b) for b in all_beams]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:n]]

    def is_done(self) -> bool:
        """Check whether search should terminate."""
        if self.early_stopping and len(self.finished) >= self.num_beams:
            return True
        # all beams finished
        active = self.all_active_beams()
        return len(active) == 0

    def prune_finished(self, max_finished: int = 0) -> None:
        """Keep only the top *max_finished* finished hypotheses."""
        if max_finished <= 0:
            max_finished = self.max_capacity
        if len(self.finished) > max_finished:
            scored = [(self.score_beam(b), b) for b in self.finished]
            scored.sort(key=lambda x: x[0], reverse=True)
            self.finished = [b for _, b in scored[:max_finished]]

    def merge_groups(self) -> List[BeamState]:
        """Merge all beam groups, rank by score, and return."""
        all_beams: List[BeamState] = []
        for group in self.beams:
            all_beams.extend(group)
        all_beams.extend(self.finished)
        scored = [(self.score_beam(b), b) for b in all_beams]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored]

    def get_input_ids(self) -> List[TokenSequence]:
        """Return token sequences for all active beams, usable as
        ``input_ids`` for the next model forward pass."""
        return [b.token_ids for b in self.all_beams_flat()]


# --------------------------------------------------------------------- #
# DiverseBeamGroups
# --------------------------------------------------------------------- #

class DiverseBeamGroups:
    """High-level manager for diverse beam search across multiple groups.

    Wraps a :class:`BeamManager` and provides step-level orchestration
    with inter-group dissimilarity penalties.
    """

    def __init__(
        self,
        num_beams: int = 12,
        num_groups: int = 3,
        diversity_penalty: float = 1.0,
        hamming_penalty: float = 0.0,
        length_penalty_alpha: float = 0.6,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        self.num_beams = num_beams
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.hamming_penalty = hamming_penalty
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.manager = BeamManager(
            num_beams=num_beams,
            num_groups=num_groups,
            diversity_penalty=diversity_penalty,
            length_penalty_alpha=length_penalty_alpha,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        # per-group token histograms for Hamming-style penalties
        self._group_token_sets: List[Set[int]] = [set() for _ in range(num_groups)]

    def init(self, prefix: Optional[TokenSequence] = None) -> None:
        self.manager.init_beams(prefix)
        self._group_token_sets = [set() for _ in range(self.num_groups)]

    def _hamming_penalty_array(
        self,
        group_idx: int,
        vocab_size: int,
    ) -> np.ndarray:
        """Build an additive penalty based on token overlap with other groups."""
        penalty = np.zeros(vocab_size, dtype=np.float64)
        if self.hamming_penalty <= 0.0:
            return penalty
        for g in range(self.num_groups):
            if g == group_idx:
                continue
            for tid in self._group_token_sets[g]:
                if 0 <= tid < vocab_size:
                    penalty[tid] += self.hamming_penalty
        return penalty

    def step(
        self,
        logit_fn: Callable[[List[TokenSequence]], np.ndarray],
        vocab_size: int,
    ) -> None:
        """Execute one decoding step across all groups sequentially.

        Groups are expanded in order so that earlier groups' choices
        influence the diversity penalty for later groups.
        """
        for g in range(self.num_groups):
            beams = self.manager.beams[g]
            if not beams:
                continue

            input_ids = [b.token_ids for b in beams]
            logits = logit_fn(input_ids)  # (beams_per_group, vocab_size)

            # apply Hamming penalty
            ham_pen = self._hamming_penalty_array(g, vocab_size)
            if logits.ndim == 2:
                logits = logits - ham_pen[np.newaxis, :]
            else:
                logits = logits - ham_pen

            self.manager.beams[g] = self.manager.expand_beams(
                g, logits, vocab_size,
            )

            # update token sets
            for b in self.manager.beams[g]:
                if b.token_ids:
                    self._group_token_sets[g].add(b.token_ids[-1])

    def is_done(self) -> bool:
        return self.manager.is_done()

    def best(self, n: int = 1) -> List[BeamState]:
        return self.manager.best_overall(n)

    def best_per_group(self, n: int = 1) -> List[List[BeamState]]:
        """Return top-*n* beams from each group separately."""
        results: List[List[BeamState]] = []
        for g in range(self.num_groups):
            group_beams = list(self.manager.beams[g])
            # include finished beams that originated from this group
            scored = [(self.manager.score_beam(b), b) for b in group_beams]
            scored.sort(key=lambda x: x[0], reverse=True)
            results.append([b for _, b in scored[:n]])
        return results

    def group_diversity_score(self) -> float:
        """Compute an aggregate diversity score across groups.

        Uses pairwise Jaccard distance of token sets.
        """
        if self.num_groups < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(self.num_groups):
            for j in range(i + 1, self.num_groups):
                set_i = self._group_token_sets[i]
                set_j = self._group_token_sets[j]
                union_size = len(set_i | set_j)
                if union_size == 0:
                    continue
                jaccard_dist = 1.0 - len(set_i & set_j) / union_size
                total += jaccard_dist
                count += 1
        return total / max(count, 1)


# ===================================================================== #
#             4.  Sampling helper functions                              #
# ===================================================================== #


def _stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / (np.sum(exp_shifted, axis=axis, keepdims=True) + _LOG_EPS)


def _stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + _LOG_EPS)
    return shifted - log_sum_exp


# --------------------------------------------------------------------- #
# sample_from_logits
# --------------------------------------------------------------------- #

def sample_from_logits(
    logits: LogitArray,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    num_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multinomial sampling from logits with temperature.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    temperature : float
        Sampling temperature (applied before softmax).
    rng : numpy Generator, optional
        Random number generator.  If ``None`` a default is used.
    num_samples : int
        Number of independent samples to draw per row.

    Returns
    -------
    token_ids : np.ndarray
        Shape ``(num_samples,)`` or ``(batch, num_samples)``.
    log_probs : np.ndarray
        Log-probabilities of the sampled tokens, same shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    # temperature
    temp = max(temperature, _MIN_TEMPERATURE)
    scaled = logits / temp

    # softmax
    probs = _stable_softmax(scaled, axis=-1)
    log_probs_full = _stable_log_softmax(scaled, axis=-1)

    batch_size, vocab_size = probs.shape
    token_ids = np.empty((batch_size, num_samples), dtype=np.int64)
    sampled_log_probs = np.empty((batch_size, num_samples), dtype=np.float64)

    for i in range(batch_size):
        p = probs[i]
        # renormalise to avoid floating-point drift
        p = np.maximum(p, 0.0)
        total = p.sum()
        if total <= 0.0:
            # uniform fallback
            p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        else:
            p /= total

        chosen = rng.choice(vocab_size, size=num_samples, replace=True, p=p)
        token_ids[i] = chosen
        sampled_log_probs[i] = log_probs_full[i, chosen]

    if is_1d:
        token_ids = token_ids[0]
        sampled_log_probs = sampled_log_probs[0]

    return token_ids, sampled_log_probs


# --------------------------------------------------------------------- #
# Standalone filtering functions
# --------------------------------------------------------------------- #

def top_k_filtering(
    logits: LogitArray,
    k: int,
    min_tokens_to_keep: int = 1,
) -> LogitArray:
    """Apply top-k filtering to logits (standalone function)."""
    proc = TopKProcessor(k=k, min_tokens_to_keep=min_tokens_to_keep)
    return proc(logits.copy())


def top_p_filtering(
    logits: LogitArray,
    p: float,
    min_tokens_to_keep: int = 1,
) -> LogitArray:
    """Apply nucleus (top-p) filtering to logits (standalone function)."""
    proc = TopPProcessor(p=p, min_tokens_to_keep=min_tokens_to_keep)
    return proc(logits.copy())


def typical_filtering(
    logits: LogitArray,
    p: float,
    min_tokens_to_keep: int = 1,
) -> LogitArray:
    """Apply typical sampling filtering (standalone function)."""
    proc = TypicalProcessor(p=p, min_tokens_to_keep=min_tokens_to_keep)
    return proc(logits.copy())


# --------------------------------------------------------------------- #
# apply_processor_chain
# --------------------------------------------------------------------- #

def apply_processor_chain(
    logits: LogitArray,
    processors: List[LogitProcessor],
    input_ids: Optional[List[TokenSequence]] = None,
    step: int = 0,
    copy: bool = True,
) -> LogitArray:
    """Apply a sequence of :class:`LogitProcessor` instances to *logits*.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits from the model.
    processors : list of LogitProcessor
        Processors to apply in order.
    input_ids : list of token sequences, optional
        History of generated tokens.
    step : int
        Current decoding step.
    copy : bool
        If ``True`` (default), work on a copy so the original is
        unchanged.

    Returns
    -------
    np.ndarray
        Processed logits.
    """
    if copy:
        logits = logits.copy()
    for proc in processors:
        logits = proc(logits, input_ids=input_ids, step=step)
    return logits


# --------------------------------------------------------------------- #
# Entropy helpers
# --------------------------------------------------------------------- #

def compute_entropy(
    logits: LogitArray,
    base: str = "nats",
) -> Union[float, np.ndarray]:
    """Compute Shannon entropy from logits.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    base : str
        ``"nats"`` (default, natural log) or ``"bits"`` (log2).

    Returns
    -------
    float or np.ndarray
        Scalar if input was 1-D, else shape ``(batch,)``.
    """
    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    probs = _stable_softmax(logits, axis=-1)
    log_probs = np.log(probs + _LOG_EPS)

    entropy = -np.sum(probs * log_probs, axis=-1)

    if base == "bits":
        entropy /= np.log(2.0)

    if is_1d:
        return float(entropy[0])
    return entropy


def compute_varentropy(
    logits: LogitArray,
    base: str = "nats",
) -> Union[float, np.ndarray]:
    """Compute variance of the information content (varentropy).

    ``Var[-log p(x)]`` under the distribution ``p(x)``.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    base : str
        ``"nats"`` or ``"bits"``.

    Returns
    -------
    float or np.ndarray
    """
    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    probs = _stable_softmax(logits, axis=-1)
    log_probs = np.log(probs + _LOG_EPS)
    neg_info = -log_probs

    if base == "bits":
        neg_info /= np.log(2.0)

    # E[-log p]
    mean_info = np.sum(probs * neg_info, axis=-1, keepdims=True)

    # Var[-log p] = E[(-log p - E[-log p])^2]
    deviation_sq = (neg_info - mean_info) ** 2
    varentropy = np.sum(probs * deviation_sq, axis=-1)

    if is_1d:
        return float(varentropy[0])
    return varentropy


# --------------------------------------------------------------------- #
# Gumbel-softmax sampling
# --------------------------------------------------------------------- #

def gumbel_softmax_sampling(
    logits: LogitArray,
    temperature: float = 1.0,
    hard: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the Gumbel-Softmax distribution.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    temperature : float
        Temperature for the Gumbel-Softmax.
    hard : bool
        If ``True``, return a one-hot (straight-through) sample.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray
        Same shape as *logits*.  Soft probabilities (or one-hot if *hard*).
    """
    if rng is None:
        rng = np.random.default_rng()

    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    # Gumbel noise
    u = rng.uniform(low=1e-20, high=1.0, size=logits.shape)
    gumbel_noise = -np.log(-np.log(u))

    temp = max(temperature, _MIN_TEMPERATURE)
    perturbed = (logits + gumbel_noise) / temp
    y_soft = _stable_softmax(perturbed, axis=-1)

    if hard:
        indices = np.argmax(y_soft, axis=-1)
        y_hard = np.zeros_like(y_soft)
        for i in range(y_soft.shape[0]):
            y_hard[i, indices[i]] = 1.0
        result = y_hard
    else:
        result = y_soft

    if is_1d:
        return result[0]
    return result


# --------------------------------------------------------------------- #
# Systematic resampling
# --------------------------------------------------------------------- #

def systematic_resampling(
    weights: np.ndarray,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Systematic resampling (particle filter style).

    Parameters
    ----------
    weights : np.ndarray
        1-D array of non-negative weights (need not sum to 1).
    num_samples : int
        Number of samples to draw.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray
        1-D array of integer indices (length *num_samples*).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(weights)
    if n == 0:
        return np.array([], dtype=np.int64)

    # normalise
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = w / total

    # cumulative sum
    cumulative = np.cumsum(w)

    # systematic sampling positions
    step = 1.0 / num_samples
    u0 = rng.uniform(0.0, step)
    positions = u0 + step * np.arange(num_samples)

    indices = np.empty(num_samples, dtype=np.int64)
    j = 0
    for i in range(num_samples):
        while j < n - 1 and cumulative[j] < positions[i]:
            j += 1
        indices[i] = j

    return indices


# --------------------------------------------------------------------- #
# Stochastic beam search sampling
# --------------------------------------------------------------------- #

def stochastic_beam_search_sampling(
    log_probs: np.ndarray,
    num_samples: int,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample *num_samples* items using Gumbel top-k (stochastic beam
    search, Kool et al. 2019).

    Parameters
    ----------
    log_probs : np.ndarray
        Shape ``(n,)`` – log-probabilities of items.
    num_samples : int
        Number of items to select.
    temperature : float
        Temperature applied to log-probs before adding Gumbel noise.
    rng : numpy Generator, optional

    Returns
    -------
    indices : np.ndarray  – selected item indices (length *num_samples*).
    perturbed_scores : np.ndarray – Gumbel-perturbed scores for selected items.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = log_probs.shape[0]
    k = min(num_samples, n)

    temp = max(temperature, _MIN_TEMPERATURE)
    scaled = log_probs / temp

    # Gumbel noise
    u = rng.uniform(low=1e-20, high=1.0, size=n)
    gumbel = -np.log(-np.log(u))
    perturbed = scaled + gumbel

    # top-k
    if k >= n:
        top_indices = np.argsort(perturbed)[::-1]
    else:
        top_indices = np.argpartition(perturbed, -k)[-k:]
        # sort descending within the top-k
        order = np.argsort(perturbed[top_indices])[::-1]
        top_indices = top_indices[order]

    return top_indices[:k], perturbed[top_indices[:k]]


# --------------------------------------------------------------------- #
# Sample without replacement (Gumbel trick)
# --------------------------------------------------------------------- #

def sample_without_replacement(
    logits: LogitArray,
    num_samples: int,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample without replacement using the Gumbel-max trick.

    Each item receives a Gumbel-perturbed score; the top-k are
    the samples.  This is equivalent to sequentially sampling from
    the categorical and removing chosen items, but faster.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)`` – unnormalised log-probabilities.
    num_samples : int
        How many distinct tokens to draw.
    temperature : float
        Temperature scaling.
    rng : numpy Generator, optional

    Returns
    -------
    token_ids : np.ndarray  – shape ``(num_samples,)``
    scores : np.ndarray  – Gumbel-perturbed scores for selected tokens.
    """
    if rng is None:
        rng = np.random.default_rng()

    log_probs = _stable_log_softmax(logits)
    return stochastic_beam_search_sampling(
        log_probs, num_samples, temperature=temperature, rng=rng,
    )


# ===================================================================== #
#         5.  Token tracking and statistics                              #
# ===================================================================== #


class TokenUsageTracker:
    """Track per-token usage frequencies over a generation session.

    Provides running counts, entropy-over-time curves, and
    summary statistics.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._counts = np.zeros(vocab_size, dtype=np.int64)
        self._step_counts: List[np.ndarray] = []
        self._total_tokens = 0
        self._unique_tokens: Set[int] = set()
        self._entropy_history: List[float] = []
        self._step_tokens: List[List[int]] = []

    def reset(self) -> None:
        self._counts = np.zeros(self.vocab_size, dtype=np.int64)
        self._step_counts = []
        self._total_tokens = 0
        self._unique_tokens = set()
        self._entropy_history = []
        self._step_tokens = []

    # -- recording -------------------------------------------------------- #

    def record_tokens(self, token_ids: List[int]) -> None:
        """Record a batch of generated tokens for one step."""
        step_arr = np.zeros(self.vocab_size, dtype=np.int64)
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self._counts[tid] += 1
                step_arr[tid] += 1
                self._unique_tokens.add(tid)
                self._total_tokens += 1

        self._step_counts.append(step_arr)
        self._step_tokens.append(list(token_ids))

        # compute running entropy
        entropy = self._compute_current_entropy()
        self._entropy_history.append(entropy)

    def record_sequence(self, token_ids: TokenSequence) -> None:
        """Record an entire sequence at once."""
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self._counts[tid] += 1
                self._unique_tokens.add(tid)
                self._total_tokens += 1

    # -- queries ---------------------------------------------------------- #

    def _compute_current_entropy(self) -> float:
        """Shannon entropy of the current frequency distribution."""
        if self._total_tokens == 0:
            return 0.0
        probs = self._counts.astype(np.float64) / self._total_tokens
        mask = probs > 0
        if not np.any(mask):
            return 0.0
        return float(-np.sum(probs[mask] * np.log(probs[mask])))

    @property
    def entropy_history(self) -> List[float]:
        return list(self._entropy_history)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def unique_token_count(self) -> int:
        return len(self._unique_tokens)

    @property
    def type_token_ratio(self) -> float:
        if self._total_tokens == 0:
            return 0.0
        return len(self._unique_tokens) / self._total_tokens

    def top_tokens(self, n: int = 20) -> List[Tuple[int, int]]:
        """Return the *n* most frequent tokens as ``(token_id, count)``."""
        if n >= self.vocab_size:
            indices = np.argsort(self._counts)[::-1]
        else:
            indices = np.argpartition(self._counts, -n)[-n:]
            order = np.argsort(self._counts[indices])[::-1]
            indices = indices[order]
        return [(int(idx), int(self._counts[idx])) for idx in indices[:n]]

    def frequency_spectrum(self) -> Dict[int, int]:
        """Return a frequency-of-frequency spectrum.

        The key is the count, the value is how many distinct tokens
        appeared that many times.
        """
        spectrum: Dict[int, int] = defaultdict(int)
        for c in self._counts:
            if c > 0:
                spectrum[int(c)] += 1
        return dict(spectrum)

    def hapax_legomena_count(self) -> int:
        """Number of tokens that appeared exactly once."""
        return int(np.sum(self._counts == 1))

    def simpson_diversity_index(self) -> float:
        """Simpson's diversity index (probability that two randomly chosen
        tokens are different)."""
        if self._total_tokens < 2:
            return 0.0
        n = self._total_tokens
        numerator = np.sum(self._counts.astype(np.float64) * (self._counts - 1))
        return 1.0 - float(numerator) / (n * (n - 1))

    def effective_vocabulary_size(self) -> float:
        """Exponential of Shannon entropy – the effective number of
        equiprobable tokens that would give the same entropy."""
        h = self._compute_current_entropy()
        return float(np.exp(h))

    def windowed_entropy(self, window_size: int = 50) -> List[float]:
        """Compute entropy over a sliding window of steps."""
        if len(self._step_counts) == 0:
            return []
        results: List[float] = []
        for end in range(1, len(self._step_counts) + 1):
            start = max(0, end - window_size)
            window_counts = np.sum(
                np.stack(self._step_counts[start:end], axis=0), axis=0,
            )
            total = window_counts.sum()
            if total == 0:
                results.append(0.0)
                continue
            probs = window_counts.astype(np.float64) / total
            mask = probs > 0
            results.append(float(-np.sum(probs[mask] * np.log(probs[mask]))))
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self._total_tokens,
            "unique_tokens": self.unique_token_count,
            "type_token_ratio": self.type_token_ratio,
            "current_entropy": self._compute_current_entropy(),
            "simpson_diversity": self.simpson_diversity_index(),
            "effective_vocab_size": self.effective_vocabulary_size(),
            "hapax_count": self.hapax_legomena_count(),
            "top_20_tokens": self.top_tokens(20),
        }


# --------------------------------------------------------------------- #
# SamplingStatistics
# --------------------------------------------------------------------- #

class SamplingStatistics:
    """Collect and report statistics about the sampling process.

    Records per-step data (entropy, temperatures, acceptance rates, etc.)
    and provides summary reports.
    """

    def __init__(self) -> None:
        self._step_entropy: List[float] = []
        self._step_varentropy: List[float] = []
        self._step_temperature: List[float] = []
        self._step_top_prob: List[float] = []
        self._step_num_active: List[int] = []
        self._step_times: List[float] = []
        self._token_counts: Dict[int, int] = defaultdict(int)
        self._rejected_count = 0
        self._accepted_count = 0
        self._total_samples = 0
        self._custom_metrics: Dict[str, List[float]] = defaultdict(list)
        self._start_time: Optional[float] = None

    def reset(self) -> None:
        self._step_entropy.clear()
        self._step_varentropy.clear()
        self._step_temperature.clear()
        self._step_top_prob.clear()
        self._step_num_active.clear()
        self._step_times.clear()
        self._token_counts.clear()
        self._rejected_count = 0
        self._accepted_count = 0
        self._total_samples = 0
        self._custom_metrics.clear()
        self._start_time = None

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()

    def record_step(
        self,
        logits: Optional[LogitArray] = None,
        temperature: float = 1.0,
        num_active: int = 0,
        selected_tokens: Optional[List[int]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record data for a single decoding step."""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
        self._step_times.append(elapsed)
        self._step_temperature.append(temperature)
        self._step_num_active.append(num_active)

        if logits is not None:
            row = logits if logits.ndim == 1 else logits[0]
            ent = compute_entropy(row)
            vent = compute_varentropy(row)
            probs = _stable_softmax(row)
            top_p = float(np.max(probs))
            self._step_entropy.append(float(ent))
            self._step_varentropy.append(float(vent))
            self._step_top_prob.append(top_p)
        else:
            self._step_entropy.append(0.0)
            self._step_varentropy.append(0.0)
            self._step_top_prob.append(0.0)

        if selected_tokens is not None:
            for tid in selected_tokens:
                self._token_counts[tid] = self._token_counts.get(tid, 0) + 1
                self._total_samples += 1

        if extra_metrics:
            for k, v in extra_metrics.items():
                self._custom_metrics[k].append(v)

    def record_acceptance(self, accepted: bool) -> None:
        if accepted:
            self._accepted_count += 1
        else:
            self._rejected_count += 1

    # -- summary ---------------------------------------------------------- #

    @property
    def num_steps(self) -> int:
        return len(self._step_entropy)

    @property
    def acceptance_rate(self) -> float:
        total = self._accepted_count + self._rejected_count
        if total == 0:
            return 1.0
        return self._accepted_count / total

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary of summary statistics."""
        result: Dict[str, Any] = {
            "num_steps": self.num_steps,
            "total_samples": self._total_samples,
            "unique_tokens": len(self._token_counts),
            "acceptance_rate": self.acceptance_rate,
        }

        if self._step_entropy:
            ent = np.array(self._step_entropy)
            result["entropy_mean"] = float(np.mean(ent))
            result["entropy_std"] = float(np.std(ent))
            result["entropy_min"] = float(np.min(ent))
            result["entropy_max"] = float(np.max(ent))

        if self._step_varentropy:
            ve = np.array(self._step_varentropy)
            result["varentropy_mean"] = float(np.mean(ve))
            result["varentropy_std"] = float(np.std(ve))

        if self._step_temperature:
            temps = np.array(self._step_temperature)
            result["temperature_mean"] = float(np.mean(temps))
            result["temperature_std"] = float(np.std(temps))

        if self._step_top_prob:
            tp = np.array(self._step_top_prob)
            result["top_prob_mean"] = float(np.mean(tp))
            result["top_prob_min"] = float(np.min(tp))

        if self._step_times and len(self._step_times) > 1:
            deltas = np.diff(self._step_times)
            if deltas.size > 0:
                result["mean_step_time_s"] = float(np.mean(deltas))
                result["total_time_s"] = float(self._step_times[-1] - self._step_times[0])

        for k, v in self._custom_metrics.items():
            arr = np.array(v)
            result[f"custom_{k}_mean"] = float(np.mean(arr))
            result[f"custom_{k}_std"] = float(np.std(arr))

        return result

    def entropy_trend(self, window: int = 10) -> List[float]:
        """Moving average of entropy."""
        if len(self._step_entropy) < window:
            return list(self._step_entropy)
        arr = np.array(self._step_entropy)
        kernel = np.ones(window) / window
        convolved = np.convolve(arr, kernel, mode="valid")
        return convolved.tolist()

    def temperature_entropy_correlation(self) -> float:
        """Pearson correlation between temperature and entropy."""
        if len(self._step_temperature) < 3 or len(self._step_entropy) < 3:
            return 0.0
        n = min(len(self._step_temperature), len(self._step_entropy))
        temps = np.array(self._step_temperature[:n])
        ents = np.array(self._step_entropy[:n])
        if np.std(temps) < _LOG_EPS or np.std(ents) < _LOG_EPS:
            return 0.0
        corr = np.corrcoef(temps, ents)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    def confidence_intervals(
        self, metric: str = "entropy", confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Return (mean, lower, upper) for the specified metric."""
        mapping = {
            "entropy": self._step_entropy,
            "varentropy": self._step_varentropy,
            "temperature": self._step_temperature,
            "top_prob": self._step_top_prob,
        }
        data = mapping.get(metric, self._step_entropy)
        if len(data) < 2:
            val = data[0] if data else 0.0
            return val, val, val

        arr = np.array(data)
        mean = float(np.mean(arr))
        se = float(sp_stats.sem(arr))
        z = float(sp_stats.norm.ppf((1.0 + confidence) / 2.0))
        return mean, mean - z * se, mean + z * se


# ===================================================================== #
#           6.  Batch sampling utilities                                 #
# ===================================================================== #


@dataclass
class _CandidateSequence:
    """Internal representation of a candidate sequence in the pool."""
    token_ids: TokenSequence
    log_prob: float = 0.0
    score: float = 0.0
    is_finished: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SequencePool:
    """Manage a pool of candidate sequences for diverse generation.

    Supports adding, scoring, pruning, and selecting diverse subsets.
    """

    def __init__(
        self,
        max_size: int = 200,
        diversity_weight: float = 0.0,
        length_penalty_alpha: float = 0.0,
    ) -> None:
        self.max_size = max_size
        self.diversity_weight = diversity_weight
        self.length_penalty_alpha = length_penalty_alpha
        self._pool: List[_CandidateSequence] = []
        self._finished: List[_CandidateSequence] = []

    def reset(self) -> None:
        self._pool.clear()
        self._finished.clear()

    # -- adding ----------------------------------------------------------- #

    def add(
        self,
        token_ids: TokenSequence,
        log_prob: float = 0.0,
        is_finished: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        candidate = _CandidateSequence(
            token_ids=list(token_ids),
            log_prob=log_prob,
            is_finished=is_finished,
            metadata=metadata or {},
        )
        candidate.score = self._score(candidate)
        if is_finished:
            self._finished.append(candidate)
        else:
            self._pool.append(candidate)

        if len(self._pool) > self.max_size:
            self._prune_pool()

    def add_batch(
        self,
        sequences: List[TokenSequence],
        log_probs: Optional[List[float]] = None,
        finished_flags: Optional[List[bool]] = None,
    ) -> None:
        if log_probs is None:
            log_probs = [0.0] * len(sequences)
        if finished_flags is None:
            finished_flags = [False] * len(sequences)
        for seq, lp, fin in zip(sequences, log_probs, finished_flags):
            self.add(seq, log_prob=lp, is_finished=fin)

    # -- scoring ---------------------------------------------------------- #

    def _score(self, candidate: _CandidateSequence) -> float:
        length = max(len(candidate.token_ids), 1)
        if abs(self.length_penalty_alpha) > 1e-9:
            lp = ((5.0 + length) / 6.0) ** self.length_penalty_alpha
            return candidate.log_prob / lp
        return candidate.log_prob

    def _pairwise_jaccard(
        self,
        seq_a: TokenSequence,
        seq_b: TokenSequence,
        ngram: int = 2,
    ) -> float:
        """Jaccard distance of n-gram sets."""
        def _ngrams(s: TokenSequence, n: int) -> Set[Tuple[int, ...]]:
            return {tuple(s[i : i + n]) for i in range(len(s) - n + 1)}

        a_set = _ngrams(seq_a, ngram)
        b_set = _ngrams(seq_b, ngram)
        if not a_set and not b_set:
            return 0.0
        union = len(a_set | b_set)
        if union == 0:
            return 0.0
        return 1.0 - len(a_set & b_set) / union

    # -- pruning ---------------------------------------------------------- #

    def _prune_pool(self) -> None:
        scored = [(c.score, idx, c) for idx, c in enumerate(self._pool)]
        scored.sort(key=lambda x: x[0], reverse=True)
        self._pool = [c for _, _, c in scored[: self.max_size]]

    def prune_finished(self, max_finished: int = 100) -> None:
        if len(self._finished) > max_finished:
            scored = sorted(self._finished, key=lambda c: c.score, reverse=True)
            self._finished = scored[:max_finished]

    # -- selection -------------------------------------------------------- #

    def top_k_sequences(self, k: int = 10) -> List[TokenSequence]:
        all_candidates = list(self._finished) + list(self._pool)
        scored = sorted(all_candidates, key=lambda c: c.score, reverse=True)
        return [c.token_ids for c in scored[:k]]

    def diverse_top_k(
        self,
        k: int = 10,
        ngram: int = 2,
        diversity_weight: Optional[float] = None,
    ) -> List[TokenSequence]:
        """Select *k* diverse sequences using a greedy MMR-like procedure.

        At each step, select the candidate that maximises:
        ``score + diversity_weight * min_distance_to_selected``
        """
        dw = diversity_weight if diversity_weight is not None else self.diversity_weight
        all_candidates = list(self._finished) + list(self._pool)
        if len(all_candidates) == 0:
            return []
        if dw <= 0.0 or k <= 1:
            scored = sorted(all_candidates, key=lambda c: c.score, reverse=True)
            return [c.token_ids for c in scored[:k]]

        # normalise scores to [0, 1]
        scores = np.array([c.score for c in all_candidates])
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > _LOG_EPS:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)

        selected_indices: List[int] = []
        remaining = set(range(len(all_candidates)))

        # pick the best one first
        first = int(np.argmax(norm_scores))
        selected_indices.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_idx = -1
            best_value = -float("inf")
            for idx in remaining:
                # min distance to any already-selected
                min_dist = min(
                    self._pairwise_jaccard(
                        all_candidates[idx].token_ids,
                        all_candidates[s].token_ids,
                        ngram=ngram,
                    )
                    for s in selected_indices
                )
                value = norm_scores[idx] + dw * min_dist
                if value > best_value:
                    best_value = value
                    best_idx = idx
            if best_idx < 0:
                break
            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        return [all_candidates[i].token_ids for i in selected_indices]

    # -- properties ------------------------------------------------------- #

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def finished_count(self) -> int:
        return len(self._finished)

    @property
    def active_sequences(self) -> List[TokenSequence]:
        return [c.token_ids for c in self._pool]

    @property
    def finished_sequences(self) -> List[TokenSequence]:
        return [c.token_ids for c in self._finished]

    def get_log_probs(self) -> List[float]:
        return [c.log_prob for c in self._pool]

    def summary(self) -> Dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "finished_count": self.finished_count,
            "max_size": self.max_size,
            "best_score": max((c.score for c in self._pool), default=0.0),
            "best_finished_score": max(
                (c.score for c in self._finished), default=0.0,
            ),
        }


class BatchSampler:
    """Handle batched generation with processor chains and tracking.

    Orchestrates sampling across a batch of sequences, applying
    processor chains, tracking statistics, and managing the
    sequence pool.
    """

    def __init__(
        self,
        config: SamplingConfig,
        vocab_size: int,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config
        self.vocab_size = vocab_size
        self.processors = config.build_processor_chain(embedding_matrix)
        self.rng = np.random.default_rng(config.seed)
        self.tracker = TokenUsageTracker(vocab_size)
        self.stats = SamplingStatistics()
        self.pool = SequencePool(
            max_size=max(config.num_return_sequences * 10, 200),
            diversity_weight=config.diversity_penalty,
            length_penalty_alpha=config.length_penalty_alpha,
        )

        # active sequences
        self._active_ids: List[TokenSequence] = []
        self._active_log_probs: List[float] = []
        self._step = 0

    def reset(self) -> None:
        for proc in self.processors:
            proc.reset()
        self.tracker.reset()
        self.stats.reset()
        self.pool.reset()
        self._active_ids = []
        self._active_log_probs = []
        self._step = 0

    # -- initialisation --------------------------------------------------- #

    def init_sequences(
        self,
        prefixes: Optional[List[TokenSequence]] = None,
        batch_size: int = 1,
    ) -> None:
        """Initialise the batch with optional prefixes."""
        self.reset()
        if prefixes is not None:
            self._active_ids = [list(p) for p in prefixes]
        else:
            self._active_ids = [[] for _ in range(batch_size)]
        self._active_log_probs = [0.0] * len(self._active_ids)
        self.stats.start_timer()

    # -- single step ------------------------------------------------------ #

    def step(
        self,
        logit_fn: Callable[[List[TokenSequence]], np.ndarray],
    ) -> Tuple[List[int], np.ndarray]:
        """Execute a single decoding step across the batch.

        Parameters
        ----------
        logit_fn : callable
            ``logit_fn(input_ids) -> np.ndarray`` of shape
            ``(batch, vocab_size)``.

        Returns
        -------
        selected_tokens : list of int
            One token per active sequence.
        log_probs : np.ndarray
            Log-probabilities of the selected tokens.
        """
        if not self._active_ids:
            return [], np.array([])

        # get logits
        logits = logit_fn(self._active_ids)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]

        # apply processor chain
        processed = apply_processor_chain(
            logits,
            self.processors,
            input_ids=self._active_ids,
            step=self._step,
            copy=True,
        )

        # sample
        if self.config.do_sample:
            token_ids, log_probs = sample_from_logits(
                processed, temperature=1.0, rng=self.rng, num_samples=1,
            )
            if token_ids.ndim == 2:
                token_ids = token_ids[:, 0]
                log_probs = log_probs[:, 0]
        else:
            # greedy
            token_ids = np.argmax(processed, axis=-1)
            log_probs_full = _stable_log_softmax(processed, axis=-1)
            log_probs = log_probs_full[np.arange(len(token_ids)), token_ids]

        selected_list = token_ids.tolist()

        # record statistics
        self.stats.record_step(
            logits=processed,
            temperature=self.config.temperature,
            num_active=len(self._active_ids),
            selected_tokens=selected_list,
        )
        self.tracker.record_tokens(selected_list)

        # update active sequences
        finished_indices: List[int] = []
        for idx, tid in enumerate(selected_list):
            self._active_ids[idx].append(tid)
            self._active_log_probs[idx] += float(log_probs[idx])

            if self.config.eos_token_id is not None and tid == self.config.eos_token_id:
                finished_indices.append(idx)

        # move finished sequences to pool
        for idx in reversed(finished_indices):
            self.pool.add(
                self._active_ids[idx],
                log_prob=self._active_log_probs[idx],
                is_finished=True,
            )
            self._active_ids.pop(idx)
            self._active_log_probs.pop(idx)

        self._step += 1
        return selected_list, log_probs

    # -- multi-step generation -------------------------------------------- #

    def generate(
        self,
        logit_fn: Callable[[List[TokenSequence]], np.ndarray],
        max_steps: Optional[int] = None,
        early_stop_fn: Optional[Callable[["BatchSampler"], bool]] = None,
    ) -> List[TokenSequence]:
        """Run generation for up to *max_steps* steps.

        Returns
        -------
        list of TokenSequence
            The best sequences from the pool.
        """
        max_steps = max_steps or self.config.max_new_tokens
        min_steps = self.config.min_new_tokens

        for step_idx in range(max_steps):
            if not self._active_ids:
                break
            if step_idx >= min_steps and early_stop_fn and early_stop_fn(self):
                break

            self.step(logit_fn)

            # enforce max length: finish remaining sequences
            if step_idx == max_steps - 1:
                for idx in range(len(self._active_ids)):
                    self.pool.add(
                        self._active_ids[idx],
                        log_prob=self._active_log_probs[idx],
                        is_finished=True,
                    )

        n = self.config.num_return_sequences
        if self.config.diversity_penalty > 0:
            return self.pool.diverse_top_k(k=n)
        return self.pool.top_k_sequences(k=n)

    # -- queries ---------------------------------------------------------- #

    @property
    def num_active(self) -> int:
        return len(self._active_ids)

    @property
    def current_step(self) -> int:
        return self._step

    def get_active_ids(self) -> List[TokenSequence]:
        return [list(s) for s in self._active_ids]

    def report(self) -> Dict[str, Any]:
        return {
            "sampling_stats": self.stats.summary(),
            "token_usage": self.tracker.to_dict(),
            "pool": self.pool.summary(),
            "steps_completed": self._step,
            "active_sequences": self.num_active,
        }


# ===================================================================== #
#   Additional advanced utilities                                        #
# ===================================================================== #

# --------------------------------------------------------------------- #
# Nucleus-aware multi-sample
# --------------------------------------------------------------------- #

def nucleus_multisample(
    logits: LogitArray,
    p: float,
    num_samples: int,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw multiple samples after nucleus filtering.

    Returns
    -------
    token_ids : np.ndarray  – shape ``(num_samples,)``
    log_probs : np.ndarray  – shape ``(num_samples,)``
    """
    filtered = top_p_filtering(logits, p=p)
    return sample_from_logits(
        filtered, temperature=temperature, rng=rng, num_samples=num_samples,
    )


# --------------------------------------------------------------------- #
# Speculative sampling helpers
# --------------------------------------------------------------------- #

def speculative_acceptance(
    draft_log_probs: np.ndarray,
    target_log_probs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """Compute acceptance mask for speculative decoding.

    For each position, accept with probability
    ``min(1, exp(target_lp - draft_lp))``.

    Parameters
    ----------
    draft_log_probs : np.ndarray
        Shape ``(seq_len,)`` – log-probs under the draft model.
    target_log_probs : np.ndarray
        Shape ``(seq_len,)`` – log-probs under the target model.
    rng : numpy Generator, optional

    Returns
    -------
    accepted : np.ndarray of bool  – shape ``(seq_len,)``
    first_rejection : int
        Index of first rejected token (or seq_len if all accepted).
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_len = draft_log_probs.shape[0]
    accepted = np.ones(seq_len, dtype=bool)

    first_rejection = seq_len
    for i in range(seq_len):
        ratio = np.exp(target_log_probs[i] - draft_log_probs[i])
        accept_prob = min(1.0, ratio)
        u = rng.uniform()
        if u > accept_prob:
            accepted[i] = False
            first_rejection = i
            # reject all subsequent tokens
            accepted[i + 1 :] = False
            break

    return accepted, first_rejection


def compute_speculative_residual(
    draft_probs: np.ndarray,
    target_probs: np.ndarray,
) -> np.ndarray:
    """Compute the residual distribution for re-sampling after rejection
    in speculative decoding.

    ``p_residual(x) ∝ max(0, p_target(x) - p_draft(x))``
    """
    residual = np.maximum(target_probs - draft_probs, 0.0)
    total = residual.sum()
    if total <= _LOG_EPS:
        # fallback to target
        return target_probs / (target_probs.sum() + _LOG_EPS)
    return residual / total


# --------------------------------------------------------------------- #
# Entropy-based adaptive top-k
# --------------------------------------------------------------------- #

def adaptive_top_k(
    logits: LogitArray,
    base_k: int = 50,
    entropy_scale: float = 5.0,
    min_k: int = 1,
    max_k: int = 500,
) -> LogitArray:
    """Adaptively choose *k* based on the entropy of the distribution.

    High entropy → larger *k* (more uncertainty, explore more).
    Low entropy → smaller *k* (confident, be precise).

    ``k = clip(base_k * (entropy / entropy_scale), min_k, max_k)``
    """
    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    result = logits.copy()
    for i in range(result.shape[0]):
        ent = float(compute_entropy(result[i]))
        k = int(np.clip(base_k * (ent / max(entropy_scale, _LOG_EPS)), min_k, max_k))
        proc = TopKProcessor(k=k)
        result[i] = proc(result[i])

    if is_1d:
        return result[0]
    return result


# --------------------------------------------------------------------- #
# Mirostat sampling
# --------------------------------------------------------------------- #

class MirostatSampler:
    """Mirostat v2 sampling (Basu et al., 2021).

    Dynamically adjusts a *surprise* threshold (tau) to target a
    desired perplexity level.
    """

    def __init__(
        self,
        target_surprise: float = 5.0,
        learning_rate: float = 0.1,
        initial_mu: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.target_surprise = target_surprise
        self.learning_rate = learning_rate
        self.mu = initial_mu if initial_mu is not None else 2.0 * target_surprise
        self.rng = rng or np.random.default_rng()
        self._surprise_history: List[float] = []

    def reset(self, initial_mu: Optional[float] = None) -> None:
        self.mu = initial_mu if initial_mu is not None else 2.0 * self.target_surprise
        self._surprise_history = []

    def sample(self, logits: np.ndarray) -> Tuple[int, float]:
        """Sample a single token and update the surprise threshold.

        Returns
        -------
        token_id : int
        surprise : float  – information content of the sampled token.
        """
        # sort logits descending
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]

        # convert to probabilities
        probs = _stable_softmax(sorted_logits)
        log_probs = np.log(probs + _LOG_EPS)
        surprises = -log_probs  # information content

        # find the cutoff: keep tokens with surprise <= mu
        keep_mask = surprises <= self.mu
        if not np.any(keep_mask):
            keep_mask[0] = True  # always keep at least the top token

        kept_indices = sorted_indices[keep_mask]
        kept_probs = probs[keep_mask]

        # renormalise
        kept_probs = kept_probs / (kept_probs.sum() + _LOG_EPS)

        # sample
        idx_in_kept = self.rng.choice(len(kept_indices), p=kept_probs)
        token_id = int(kept_indices[idx_in_kept])
        surprise = float(surprises[keep_mask][idx_in_kept])

        # update mu
        error = surprise - self.target_surprise
        self.mu -= self.learning_rate * error

        self._surprise_history.append(surprise)
        return token_id, surprise

    @property
    def surprise_history(self) -> List[float]:
        return list(self._surprise_history)

    @property
    def current_mu(self) -> float:
        return self.mu


# --------------------------------------------------------------------- #
# Eta sampling
# --------------------------------------------------------------------- #

def eta_sampling(
    logits: LogitArray,
    eta: float = 0.0003,
    min_tokens_to_keep: int = 1,
) -> LogitArray:
    """η-sampling (Hewitt et al., 2022).

    Threshold is ``eta * entropy(softmax(logits))``.  Tokens with
    probability below this threshold are removed.
    """
    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    result = logits.copy()
    for i in range(result.shape[0]):
        probs = _stable_softmax(result[i])
        entropy = float(-np.sum(probs * np.log(probs + _LOG_EPS)))
        threshold = eta * entropy

        below = probs < threshold

        # ensure we keep at least min_tokens_to_keep
        if np.sum(~below) < min_tokens_to_keep:
            top_indices = np.argpartition(probs, -min_tokens_to_keep)[
                -min_tokens_to_keep :
            ]
            below[:] = True
            below[top_indices] = False

        result[i, below] = _LOGIT_NEG_INF

    if is_1d:
        return result[0]
    return result


# --------------------------------------------------------------------- #
# Tail-free sampling
# --------------------------------------------------------------------- #

def tail_free_sampling(
    logits: LogitArray,
    z: float = 0.95,
    min_tokens_to_keep: int = 1,
) -> LogitArray:
    """Tail-free sampling (TFS).

    Uses the second derivative of the sorted probability distribution
    to find the ``tail`` and remove it.
    """
    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    result = logits.copy()
    for i in range(result.shape[0]):
        sorted_indices = np.argsort(result[i])[::-1]
        sorted_logits = result[i, sorted_indices]
        probs = _stable_softmax(sorted_logits)

        if len(probs) < 3:
            continue

        # first derivative
        d1 = np.diff(probs)
        # second derivative
        d2 = np.diff(d1)
        # normalise absolute second derivative
        abs_d2 = np.abs(d2)
        total_d2 = abs_d2.sum()
        if total_d2 < _LOG_EPS:
            continue
        norm_d2 = abs_d2 / total_d2

        # cumulative sum
        cum_d2 = np.cumsum(norm_d2)

        # find cutoff where cumulative >= z
        cutoff_idx = np.searchsorted(cum_d2, z)
        # add 2 because of two differencing steps
        cutoff_idx = cutoff_idx + 2

        cutoff_idx = max(cutoff_idx, min_tokens_to_keep)

        if cutoff_idx < len(sorted_indices):
            remove_indices = sorted_indices[cutoff_idx:]
            result[i, remove_indices] = _LOGIT_NEG_INF

    if is_1d:
        return result[0]
    return result


# --------------------------------------------------------------------- #
# Contrastive decoding helper
# --------------------------------------------------------------------- #

def contrastive_decoding_score(
    expert_logits: np.ndarray,
    amateur_logits: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Compute contrastive decoding scores (Li et al., 2023).

    ``score(x) = log p_expert(x) - log p_amateur(x)``

    Only tokens satisfying ``p_expert(x) >= alpha * max(p_expert)``
    are considered.

    Parameters
    ----------
    expert_logits : np.ndarray – shape ``(vocab_size,)``
    amateur_logits : np.ndarray – shape ``(vocab_size,)``
    alpha : float – plausibility threshold

    Returns
    -------
    np.ndarray – shape ``(vocab_size,)`` with masked scores.
    """
    expert_lp = _stable_log_softmax(expert_logits)
    amateur_lp = _stable_log_softmax(amateur_logits)

    expert_probs = _stable_softmax(expert_logits)
    threshold = alpha * np.max(expert_probs)
    plausible = expert_probs >= threshold

    scores = np.full_like(expert_logits, _LOGIT_NEG_INF)
    scores[plausible] = expert_lp[plausible] - amateur_lp[plausible]
    return scores


# --------------------------------------------------------------------- #
# Diverse sampling via DPP kernel
# --------------------------------------------------------------------- #

def dpp_quality_diversity_scores(
    logits: np.ndarray,
    embeddings: np.ndarray,
    quality_weight: float = 1.0,
    diversity_weight: float = 1.0,
) -> np.ndarray:
    """Compute per-token scores combining quality (log-prob) and
    diversity (embedding distance) using a DPP-inspired decomposition.

    This is a *fast approximation*: rather than sampling from a full
    DPP kernel (which is ``O(k^3)``), we score each token independently
    using quality plus average cosine distance to other high-quality
    tokens.

    Parameters
    ----------
    logits : np.ndarray – shape ``(vocab_size,)``
    embeddings : np.ndarray – shape ``(vocab_size, dim)``
    quality_weight, diversity_weight : float

    Returns
    -------
    np.ndarray – shape ``(vocab_size,)``
    """
    log_probs = _stable_log_softmax(logits)
    probs = _stable_softmax(logits)
    vocab_size = logits.shape[0]

    quality_scores = quality_weight * log_probs

    # identify high-quality candidate set (top-100)
    k_candidates = min(100, vocab_size)
    top_indices = np.argpartition(log_probs, -k_candidates)[-k_candidates:]

    # compute pairwise cosine similarity within top set
    top_emb = embeddings[top_indices]  # (k, dim)
    norms = np.linalg.norm(top_emb, axis=1, keepdims=True) + _LOG_EPS
    top_emb_norm = top_emb / norms

    # average cosine distance of each token to the top set
    all_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + _LOG_EPS
    all_emb_norm = embeddings / all_norms
    mean_top = np.mean(top_emb_norm, axis=0)  # (dim,)
    cosine_to_mean = all_emb_norm @ mean_top  # (vocab_size,)
    diversity_scores = diversity_weight * (1.0 - cosine_to_mean)

    return quality_scores + diversity_scores


# --------------------------------------------------------------------- #
# Temperature annealing with entropy feedback
# --------------------------------------------------------------------- #

class EntropyFeedbackController:
    """PID-like controller for temperature based on entropy feedback.

    Continuously adjusts temperature to drive the output entropy
    towards a target setpoint.
    """

    def __init__(
        self,
        target_entropy: float = 3.0,
        kp: float = 0.3,
        ki: float = 0.05,
        kd: float = 0.1,
        min_temperature: float = _MIN_TEMPERATURE,
        max_temperature: float = _MAX_TEMPERATURE,
        initial_temperature: float = 1.0,
    ) -> None:
        self.target_entropy = target_entropy
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature = initial_temperature

        self._integral = 0.0
        self._prev_error = 0.0
        self._history: List[Tuple[float, float]] = []  # (entropy, temperature)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._history = []
        self.temperature = 1.0

    def update(self, current_entropy: float) -> float:
        """Update the controller with the current entropy and return
        the new temperature."""
        error = self.target_entropy - current_entropy

        # proportional
        p_term = self.kp * error

        # integral (with anti-windup clamp)
        self._integral += error
        self._integral = np.clip(
            self._integral,
            -10.0 / max(self.ki, _LOG_EPS),
            10.0 / max(self.ki, _LOG_EPS),
        )
        i_term = self.ki * self._integral

        # derivative
        d_term = self.kd * (error - self._prev_error)
        self._prev_error = error

        # update temperature
        adjustment = p_term + i_term + d_term
        self.temperature += adjustment
        self.temperature = float(
            np.clip(self.temperature, self.min_temperature, self.max_temperature),
        )

        self._history.append((current_entropy, self.temperature))
        return self.temperature

    @property
    def history(self) -> List[Tuple[float, float]]:
        return list(self._history)


# --------------------------------------------------------------------- #
# Sequence scoring utilities
# --------------------------------------------------------------------- #

def score_sequence_log_prob(
    token_log_probs: List[float],
    length_penalty_alpha: float = 0.0,
    base: float = 5.0,
) -> float:
    """Score a sequence by its cumulative log-probability with optional
    length normalisation."""
    total_lp = sum(token_log_probs)
    if abs(length_penalty_alpha) < 1e-9:
        return total_lp
    length = len(token_log_probs)
    lp = ((base + length) / (base + 1)) ** length_penalty_alpha
    return total_lp / max(lp, _LOG_EPS)


def score_sequence_bleu_self(
    candidate: TokenSequence,
    references: List[TokenSequence],
    max_ngram: int = 4,
) -> float:
    """Self-BLEU: average BLEU of *candidate* against each reference.

    Used as a *diversity* metric (lower = more diverse).
    """
    if not references:
        return 0.0

    def _count_ngrams(seq: TokenSequence, n: int) -> Dict[Tuple[int, ...], int]:
        counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        for i in range(len(seq) - n + 1):
            counts[tuple(seq[i : i + n])] += 1
        return counts

    scores: List[float] = []
    for ref in references:
        if ref == candidate:
            continue
        precisions: List[float] = []
        for n in range(1, max_ngram + 1):
            cand_ngrams = _count_ngrams(candidate, n)
            ref_ngrams = _count_ngrams(ref, n)
            clipped = 0
            total = 0
            for gram, count in cand_ngrams.items():
                clipped += min(count, ref_ngrams.get(gram, 0))
                total += count
            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped / total)

        # geometric mean of precisions (smoothed)
        log_avg = 0.0
        valid = 0
        for p in precisions:
            if p > 0:
                log_avg += math.log(p)
                valid += 1
        if valid == 0:
            scores.append(0.0)
        else:
            log_avg /= valid
            # brevity penalty
            bp = min(1.0, math.exp(1.0 - len(ref) / max(len(candidate), 1)))
            scores.append(bp * math.exp(log_avg))

    return float(np.mean(scores)) if scores else 0.0


def distinct_ngrams(
    sequences: List[TokenSequence],
    n: int = 2,
) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams
    across all sequences."""
    total = 0
    unique: Set[Tuple[int, ...]] = set()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i : i + n])
            unique.add(gram)
            total += 1
    if total == 0:
        return 0.0
    return len(unique) / total


# --------------------------------------------------------------------- #
# Weighted reservoir sampling
# --------------------------------------------------------------------- #

def weighted_reservoir_sampling(
    items: Sequence[Any],
    weights: np.ndarray,
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Any]:
    """Weighted reservoir sampling (Efraimidis & Spirakis, 2006).

    Select *k* items from *items* with probabilities proportional to
    *weights*, in a single pass.

    Parameters
    ----------
    items : sequence
        Items to sample from.
    weights : np.ndarray
        Non-negative weights, same length as *items*.
    k : int
        Number of items to select.
    rng : numpy Generator, optional

    Returns
    -------
    list
        Selected items.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(items)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(items)

    # key = u^(1/w) where u ~ Uniform(0,1)
    u = rng.uniform(low=1e-20, high=1.0, size=n)
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, _LOG_EPS)

    keys = np.power(u, 1.0 / w)

    # top-k by key
    top_indices = np.argpartition(keys, -k)[-k:]
    return [items[i] for i in top_indices]


# --------------------------------------------------------------------- #
# Probability calibration
# --------------------------------------------------------------------- #

def temperature_scaling_calibration(
    logits_list: List[np.ndarray],
    labels: List[int],
    initial_temperature: float = 1.0,
    num_iterations: int = 100,
    learning_rate: float = 0.01,
) -> float:
    """Learn an optimal temperature for calibration via gradient descent
    on the negative log-likelihood.

    Parameters
    ----------
    logits_list : list of np.ndarray
        Each element is shape ``(vocab_size,)`` – logits for one example.
    labels : list of int
        Ground-truth token ids.
    initial_temperature : float
    num_iterations : int
    learning_rate : float

    Returns
    -------
    float – optimised temperature.
    """
    temperature = initial_temperature
    n = len(logits_list)
    if n == 0:
        return temperature

    for _ in range(num_iterations):
        total_nll = 0.0
        total_grad = 0.0

        for logits, label in zip(logits_list, labels):
            scaled = logits / max(temperature, _MIN_TEMPERATURE)
            log_probs = _stable_log_softmax(scaled)
            probs = _stable_softmax(scaled)

            # NLL
            if 0 <= label < len(log_probs):
                total_nll -= log_probs[label]

                # gradient of NLL w.r.t. temperature
                # d(NLL)/d(T) = (1/T^2) * (logits[label] - E[logits])
                expected_logit = np.sum(probs * logits)
                grad = (1.0 / (temperature ** 2)) * (expected_logit - logits[label])
                total_grad += grad

        avg_grad = total_grad / n
        temperature -= learning_rate * avg_grad
        temperature = max(temperature, _MIN_TEMPERATURE)

    return temperature


# --------------------------------------------------------------------- #
# Logit lens utilities
# --------------------------------------------------------------------- #

def compute_kl_divergence(
    logits_p: np.ndarray,
    logits_q: np.ndarray,
) -> float:
    """Compute KL(P || Q) from logit arrays.

    Parameters
    ----------
    logits_p, logits_q : np.ndarray
        Shape ``(vocab_size,)`` – logits for distributions P and Q.

    Returns
    -------
    float – KL divergence in nats.
    """
    p = _stable_softmax(logits_p)
    q = _stable_softmax(logits_q)
    # KL(P||Q) = sum p * log(p/q)
    ratio = p / (q + _LOG_EPS)
    kl = np.sum(p * np.log(ratio + _LOG_EPS))
    return max(float(kl), 0.0)


def compute_js_divergence(
    logits_p: np.ndarray,
    logits_q: np.ndarray,
) -> float:
    """Jensen–Shannon divergence (symmetric)."""
    p = _stable_softmax(logits_p)
    q = _stable_softmax(logits_q)
    m = 0.5 * (p + q)
    log_m = np.log(m + _LOG_EPS)
    log_p = np.log(p + _LOG_EPS)
    log_q = np.log(q + _LOG_EPS)
    kl_pm = np.sum(p * (log_p - log_m))
    kl_qm = np.sum(q * (log_q - log_m))
    return max(0.0, 0.5 * float(kl_pm + kl_qm))


def compute_hellinger_distance(
    logits_p: np.ndarray,
    logits_q: np.ndarray,
) -> float:
    """Hellinger distance between two distributions."""
    p = _stable_softmax(logits_p)
    q = _stable_softmax(logits_q)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


# --------------------------------------------------------------------- #
# Multi-armed bandit for processor selection
# --------------------------------------------------------------------- #

class ProcessorBandit:
    """Thompson-sampling bandit for selecting among processor chains.

    Each *arm* is a processor configuration (e.g. different top-p
    values).  The bandit tracks success/failure and selects the
    configuration that maximises expected reward.
    """

    def __init__(
        self,
        arms: List[List[LogitProcessor]],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.arms = arms
        self.n_arms = len(arms)
        self.alpha = np.full(self.n_arms, prior_alpha)
        self.beta = np.full(self.n_arms, prior_beta)
        self._selection_history: List[int] = []
        self._reward_history: List[float] = []

    def select(self, rng: Optional[np.random.Generator] = None) -> int:
        """Select an arm via Thompson sampling."""
        if rng is None:
            rng = np.random.default_rng()
        samples = rng.beta(self.alpha, self.beta)
        arm = int(np.argmax(samples))
        self._selection_history.append(arm)
        return arm

    def update(self, arm: int, reward: float) -> None:
        """Update the posterior for the selected arm.

        *reward* should be in [0, 1].
        """
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
        self._reward_history.append(reward)

    def get_chain(self, arm: int) -> List[LogitProcessor]:
        return self.arms[arm]

    def expected_rewards(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def reset(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.alpha = np.full(self.n_arms, prior_alpha)
        self.beta = np.full(self.n_arms, prior_beta)
        self._selection_history.clear()
        self._reward_history.clear()


# --------------------------------------------------------------------- #
# Logit warping compositions
# --------------------------------------------------------------------- #

def compose_processors(
    *processors: LogitProcessor,
) -> LogitProcessor:
    """Compose multiple processors into a single processor."""

    class _ComposedProcessor(LogitProcessor):
        name = "composed"

        def __init__(self, procs: Tuple[LogitProcessor, ...]) -> None:
            self._procs = list(procs)

        def __call__(
            self,
            logits: LogitArray,
            input_ids: Optional[List[TokenSequence]] = None,
            step: int = 0,
        ) -> LogitArray:
            for p in self._procs:
                logits = p(logits, input_ids=input_ids, step=step)
            return logits

        def reset(self) -> None:
            for p in self._procs:
                p.reset()

    return _ComposedProcessor(processors)


# --------------------------------------------------------------------- #
# Logit caching
# --------------------------------------------------------------------- #

class LogitCache:
    """Simple LRU cache for logit computations keyed by token sequence
    prefixes."""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._cache: Dict[Tuple[int, ...], np.ndarray] = {}
        self._access_order: List[Tuple[int, ...]] = []

    def get(self, key: TokenSequence) -> Optional[np.ndarray]:
        tkey = tuple(key)
        val = self._cache.get(tkey)
        if val is not None:
            # move to end (most recent)
            if tkey in self._access_order:
                self._access_order.remove(tkey)
            self._access_order.append(tkey)
        return val

    def put(self, key: TokenSequence, logits: np.ndarray) -> None:
        tkey = tuple(key)
        if tkey in self._cache:
            self._access_order.remove(tkey)
        elif len(self._cache) >= self.max_size:
            # evict LRU
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[tkey] = logits.copy()
        self._access_order.append(tkey)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        # This is a simplified metric; in production you'd track hits/misses
        return 0.0  # tracking not implemented in lightweight version


# --------------------------------------------------------------------- #
# Batch utilities: padding / masking
# --------------------------------------------------------------------- #

def pad_sequences(
    sequences: List[TokenSequence],
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    padding_side: str = "right",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a list of variable-length token sequences into a 2-D array.

    Returns
    -------
    padded : np.ndarray – shape ``(batch, max_length)``
    attention_mask : np.ndarray – shape ``(batch, max_length)``
    """
    if not sequences:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    lengths = [len(s) for s in sequences]
    if max_length is None:
        max_length = max(lengths)
    else:
        max_length = max(max_length, 1)

    batch_size = len(sequences)
    padded = np.full((batch_size, max_length), pad_token_id, dtype=np.int64)
    mask = np.zeros((batch_size, max_length), dtype=np.int64)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if padding_side == "right":
            padded[i, :length] = seq[:length]
            mask[i, :length] = 1
        else:
            offset = max_length - length
            padded[i, offset:] = seq[:length]
            mask[i, offset:] = 1

    return padded, mask


def unpad_sequences(
    padded: np.ndarray,
    attention_mask: np.ndarray,
) -> List[TokenSequence]:
    """Remove padding from a batch of sequences."""
    result: List[TokenSequence] = []
    for i in range(padded.shape[0]):
        indices = np.where(attention_mask[i] == 1)[0]
        if len(indices) > 0:
            result.append(padded[i, indices].tolist())
        else:
            result.append([])
    return result


# --------------------------------------------------------------------- #
# Stopping criteria helpers
# --------------------------------------------------------------------- #

class StoppingCriterion(abc.ABC):
    """Abstract stopping criterion for generation loops."""

    @abc.abstractmethod
    def should_stop(
        self,
        token_ids: TokenSequence,
        step: int,
        logits: Optional[LogitArray] = None,
    ) -> bool:
        ...


class MaxLengthStopper(StoppingCriterion):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def should_stop(
        self,
        token_ids: TokenSequence,
        step: int,
        logits: Optional[LogitArray] = None,
    ) -> bool:
        return len(token_ids) >= self.max_length


class EosTokenStopper(StoppingCriterion):
    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def should_stop(
        self,
        token_ids: TokenSequence,
        step: int,
        logits: Optional[LogitArray] = None,
    ) -> bool:
        return len(token_ids) > 0 and token_ids[-1] == self.eos_token_id


class EntropyThresholdStopper(StoppingCriterion):
    """Stop when entropy drops below a threshold (model is very confident)."""

    def __init__(
        self,
        threshold: float = 0.1,
        consecutive_steps: int = 3,
    ) -> None:
        self.threshold = threshold
        self.consecutive_steps = consecutive_steps
        self._low_entropy_count = 0

    def should_stop(
        self,
        token_ids: TokenSequence,
        step: int,
        logits: Optional[LogitArray] = None,
    ) -> bool:
        if logits is None:
            return False
        ent = compute_entropy(logits)
        if isinstance(ent, np.ndarray):
            ent = float(ent.mean())
        if ent < self.threshold:
            self._low_entropy_count += 1
        else:
            self._low_entropy_count = 0
        return self._low_entropy_count >= self.consecutive_steps


class RepetitionStopper(StoppingCriterion):
    """Stop when a repeating pattern is detected."""

    def __init__(self, pattern_length: int = 5, max_repeats: int = 3) -> None:
        self.pattern_length = pattern_length
        self.max_repeats = max_repeats

    def should_stop(
        self,
        token_ids: TokenSequence,
        step: int,
        logits: Optional[LogitArray] = None,
    ) -> bool:
        n = self.pattern_length
        needed = n * self.max_repeats
        if len(token_ids) < needed:
            return False
        tail = token_ids[-needed:]
        pattern = tail[:n]
        for rep in range(1, self.max_repeats):
            segment = tail[rep * n : (rep + 1) * n]
            if segment != pattern:
                return False
        return True


class CompositeStopper(StoppingCriterion):
    """Combine multiple stopping criteria with OR logic."""

    def __init__(self, criteria: List[StoppingCriterion]) -> None:
        self.criteria = criteria

    def should_stop(
        self,
        token_ids: TokenSequence,
        step: int,
        logits: Optional[LogitArray] = None,
    ) -> bool:
        return any(c.should_stop(token_ids, step, logits) for c in self.criteria)


# --------------------------------------------------------------------- #
# Sequence deduplication
# --------------------------------------------------------------------- #

def deduplicate_sequences(
    sequences: List[TokenSequence],
    scores: Optional[List[float]] = None,
    ngram_overlap_threshold: float = 0.9,
    ngram_size: int = 3,
) -> Tuple[List[TokenSequence], List[float]]:
    """Remove near-duplicate sequences based on n-gram overlap.

    Parameters
    ----------
    sequences : list of TokenSequence
    scores : optional list of float – used to keep the higher-scoring
        sequence when deduplicating.
    ngram_overlap_threshold : float
        Maximum Jaccard similarity allowed before sequences are considered
        duplicates.
    ngram_size : int

    Returns
    -------
    unique_sequences, unique_scores
    """
    if scores is None:
        scores = [0.0] * len(sequences)

    def _ngram_set(seq: TokenSequence, n: int) -> Set[Tuple[int, ...]]:
        return {tuple(seq[i : i + n]) for i in range(max(1, len(seq) - n + 1))}

    # pair and sort by score descending
    paired = sorted(zip(scores, sequences), key=lambda x: x[0], reverse=True)

    kept_scores: List[float] = []
    kept_seqs: List[TokenSequence] = []
    kept_ngrams: List[Set[Tuple[int, ...]]] = []

    for score, seq in paired:
        s = _ngram_set(seq, ngram_size)
        is_dup = False
        for existing in kept_ngrams:
            union = len(s | existing)
            if union == 0:
                continue
            jaccard = len(s & existing) / union
            if jaccard >= ngram_overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            kept_scores.append(score)
            kept_seqs.append(seq)
            kept_ngrams.append(s)

    return kept_seqs, kept_scores


# --------------------------------------------------------------------- #
# Reranking utilities
# --------------------------------------------------------------------- #

def rerank_by_diversity(
    sequences: List[TokenSequence],
    scores: List[float],
    diversity_weight: float = 0.5,
    ngram: int = 2,
) -> List[Tuple[TokenSequence, float]]:
    """Re-rank sequences using MMR (Maximal Marginal Relevance).

    Balances quality (score) and diversity (n-gram Jaccard distance
    to already-selected sequences).
    """
    if not sequences:
        return []

    # normalise scores to [0, 1]
    s_arr = np.array(scores, dtype=np.float64)
    s_min, s_max = s_arr.min(), s_arr.max()
    if s_max - s_min > _LOG_EPS:
        norm_scores = (s_arr - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(s_arr)

    def _jaccard_dist(a: TokenSequence, b: TokenSequence) -> float:
        def _ngs(s: TokenSequence) -> Set[Tuple[int, ...]]:
            return {tuple(s[i : i + ngram]) for i in range(max(1, len(s) - ngram + 1))}
        sa, sb = _ngs(a), _ngs(b)
        union = len(sa | sb)
        if union == 0:
            return 0.0
        return 1.0 - len(sa & sb) / union

    selected: List[int] = []
    remaining = set(range(len(sequences)))

    # pick best first
    first = int(np.argmax(norm_scores))
    selected.append(first)
    remaining.discard(first)

    while remaining and len(selected) < len(sequences):
        best_idx = -1
        best_val = -float("inf")
        for idx in remaining:
            min_dist = min(
                _jaccard_dist(sequences[idx], sequences[s])
                for s in selected
            )
            val = (1.0 - diversity_weight) * norm_scores[idx] + diversity_weight * min_dist
            if val > best_val:
                best_val = val
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [(sequences[i], scores[i]) for i in selected]


# --------------------------------------------------------------------- #
# Logit adjustment for constrained generation
# --------------------------------------------------------------------- #

def apply_token_mask(
    logits: LogitArray,
    allowed_tokens: Optional[np.ndarray] = None,
    blocked_tokens: Optional[np.ndarray] = None,
) -> LogitArray:
    """Mask logits to allow/block specific tokens.

    Parameters
    ----------
    logits : np.ndarray
    allowed_tokens : np.ndarray of bool, optional
        Shape ``(vocab_size,)`` – ``True`` for tokens that are allowed.
    blocked_tokens : np.ndarray of bool, optional
        Shape ``(vocab_size,)`` – ``True`` for tokens that are blocked.

    Returns
    -------
    np.ndarray
    """
    result = logits.copy()
    if allowed_tokens is not None:
        if result.ndim == 1:
            result[~allowed_tokens] = _LOGIT_NEG_INF
        else:
            result[:, ~allowed_tokens] = _LOGIT_NEG_INF
    if blocked_tokens is not None:
        if result.ndim == 1:
            result[blocked_tokens] = _LOGIT_NEG_INF
        else:
            result[:, blocked_tokens] = _LOGIT_NEG_INF
    return result


def force_token(
    logits: LogitArray,
    token_id: int,
) -> LogitArray:
    """Force a specific token by setting its logit to a large value and
    all others to ``-inf``."""
    result = np.full_like(logits, _LOGIT_NEG_INF)
    if result.ndim == 1:
        if 0 <= token_id < result.shape[0]:
            result[token_id] = 0.0
    else:
        if 0 <= token_id < result.shape[-1]:
            result[:, token_id] = 0.0
    return result


# --------------------------------------------------------------------- #
# Probability smoothing
# --------------------------------------------------------------------- #

def label_smoothing(
    logits: LogitArray,
    smoothing: float = 0.1,
) -> LogitArray:
    """Apply label smoothing to logits.

    Mixes the softmax distribution with a uniform distribution:
    ``p_smooth = (1 - smoothing) * softmax(logits) + smoothing / vocab_size``

    Returns adjusted logits (log of smoothed probabilities).
    """
    if abs(smoothing) < 1e-9:
        return logits

    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits[np.newaxis, :]

    probs = _stable_softmax(logits, axis=-1)
    vocab_size = probs.shape[-1]
    uniform = 1.0 / vocab_size

    smoothed = (1.0 - smoothing) * probs + smoothing * uniform
    result = np.log(smoothed + _LOG_EPS)

    if is_1d:
        return result[0]
    return result


# --------------------------------------------------------------------- #
# Cross-entropy and perplexity
# --------------------------------------------------------------------- #

def cross_entropy_from_logits(
    logits: np.ndarray,
    target_ids: np.ndarray,
) -> float:
    """Compute cross-entropy loss.

    Parameters
    ----------
    logits : np.ndarray – shape ``(seq_len, vocab_size)``
    target_ids : np.ndarray – shape ``(seq_len,)``

    Returns
    -------
    float – mean cross-entropy (nats).
    """
    log_probs = _stable_log_softmax(logits, axis=-1)
    seq_len = target_ids.shape[0]
    ce = 0.0
    count = 0
    for t in range(seq_len):
        tid = target_ids[t]
        if 0 <= tid < log_probs.shape[-1]:
            ce -= log_probs[t, tid]
            count += 1
    return ce / max(count, 1)


def perplexity_from_logits(
    logits: np.ndarray,
    target_ids: np.ndarray,
) -> float:
    """Compute perplexity from logits and target token ids."""
    ce = cross_entropy_from_logits(logits, target_ids)
    return float(np.exp(ce))


# --------------------------------------------------------------------- #
# Sequence repetition analysis
# --------------------------------------------------------------------- #

def detect_repeating_pattern(
    token_ids: TokenSequence,
    min_pattern_length: int = 2,
    max_pattern_length: int = 20,
) -> Optional[Tuple[int, int]]:
    """Detect if the tail of the sequence contains a repeating pattern.

    Returns
    -------
    (pattern_length, num_repeats) or None if no repetition detected.
    """
    n = len(token_ids)
    for plen in range(min_pattern_length, min(max_pattern_length + 1, n // 2 + 1)):
        pattern = token_ids[-plen:]
        repeats = 1
        pos = n - 2 * plen
        while pos >= 0:
            segment = token_ids[pos : pos + plen]
            if segment == pattern:
                repeats += 1
                pos -= plen
            else:
                break
        if repeats >= 2:
            return (plen, repeats)
    return None


def repetition_ratio(
    token_ids: TokenSequence,
    ngram_size: int = 3,
) -> float:
    """Fraction of n-grams that are repeated (appear more than once)."""
    if len(token_ids) < ngram_size:
        return 0.0
    ngrams: Dict[Tuple[int, ...], int] = defaultdict(int)
    for i in range(len(token_ids) - ngram_size + 1):
        gram = tuple(token_ids[i : i + ngram_size])
        ngrams[gram] += 1
    total = sum(ngrams.values())
    repeated = sum(c - 1 for c in ngrams.values() if c > 1)
    return repeated / max(total, 1)


# ===================================================================== #
#  Module-level convenience: processor registry                          #
# ===================================================================== #

_PROCESSOR_REGISTRY: Dict[str, type] = {
    "temperature": TemperatureProcessor,
    "top_k": TopKProcessor,
    "top_p": TopPProcessor,
    "typical": TypicalProcessor,
    "min_p": MinPProcessor,
    "repetition_penalty": RepetitionPenaltyProcessor,
    "length_penalty": LengthPenaltyProcessor,
    "no_repeat_ngram": NoRepeatNgramProcessor,
    "entropy_based": EntropyBasedProcessor,
    "contrastive": ContrastiveProcessor,
    "diversity_boost": DiversityBoostProcessor,
}


def get_processor(name: str, **kwargs: Any) -> LogitProcessor:
    """Instantiate a processor by name from the registry."""
    cls = _PROCESSOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown processor '{name}'. Available: {list(_PROCESSOR_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def list_processors() -> List[str]:
    """Return the names of all registered processors."""
    return list(_PROCESSOR_REGISTRY.keys())
