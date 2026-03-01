"""
Base decoding algorithm interface for the Diversity Decoding Arena.

Provides abstract base classes, shared utilities, stopping criteria,
an algorithm registry, and helper sampling functions used by all
concrete decoding algorithms.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TokenSequence = List[int]
LogitArray = np.ndarray  # shape (vocab_size,) or (batch, vocab_size)


@runtime_checkable
class LogitSource(Protocol):
    """Anything that can produce next-token logits given a token prefix."""

    def __call__(self, input_ids: List[List[int]]) -> np.ndarray:
        """Return logits of shape ``(batch, vocab_size)``."""
        ...


# =========================================================================
# DecodingConfig
# =========================================================================


@dataclass
class DecodingConfig:
    """Configuration for a decoding algorithm run."""

    algorithm_name: str = ""
    num_sequences: int = 20
    max_new_tokens: int = 100
    min_new_tokens: int = 10
    seed: Optional[int] = None
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the config to a plain dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecodingConfig":
        """Reconstruct a config from a dictionary."""
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
        """Return a list of validation error strings (empty == valid)."""
        errors: List[str] = []
        if self.num_sequences < 1:
            errors.append("num_sequences must be >= 1")
        if self.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1")
        if self.min_new_tokens < 0:
            errors.append("min_new_tokens must be >= 0")
        if self.min_new_tokens > self.max_new_tokens:
            errors.append("min_new_tokens must be <= max_new_tokens")
        if self.temperature <= 0:
            errors.append("temperature must be > 0")
        if self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
        if self.no_repeat_ngram_size < 0:
            errors.append("no_repeat_ngram_size must be >= 0")
        return errors

    # -- hashing ------------------------------------------------------------

    def hash(self) -> str:
        """Deterministic content hash for caching / dedup."""
        serialised = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def __hash__(self) -> int:  # type: ignore[override]
        return int(self.hash()[:16], 16)


# =========================================================================
# DecodingState
# =========================================================================


@dataclass
class DecodingState:
    """Mutable state tracked during token generation."""

    sequences: List[List[int]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    is_finished: List[bool] = field(default_factory=list)
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    logit_history: Optional[List[np.ndarray]] = None

    # -- queries ------------------------------------------------------------

    def active_indices(self) -> List[int]:
        """Indices of sequences that are still generating."""
        return [i for i, done in enumerate(self.is_finished) if not done]

    def num_active(self) -> int:
        """Number of sequences still generating."""
        return sum(1 for done in self.is_finished if not done)

    def all_finished(self) -> bool:
        """True when every sequence has finished."""
        return all(self.is_finished)

    def get_sequence(self, i: int) -> List[int]:
        """Return the *i*-th sequence (copy)."""
        return list(self.sequences[i])

    def update_sequence(self, i: int, token: int) -> None:
        """Append *token* to the *i*-th sequence."""
        self.sequences[i].append(token)

    def mark_finished(self, i: int) -> None:
        """Mark the *i*-th sequence as finished."""
        self.is_finished[i] = True

    # -- convenience --------------------------------------------------------

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def clone(self) -> "DecodingState":
        """Deep copy of the state."""
        return copy.deepcopy(self)


# =========================================================================
# Stopping criteria
# =========================================================================


class StoppingCriteria(abc.ABC):
    """Abstract single stopping criterion."""

    @abc.abstractmethod
    def __call__(self, input_ids: List[List[int]], scores: List[float], **kwargs: Any) -> List[bool]:
        """Return a boolean mask — *True* where generation should stop."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g. timers)."""
        ...


class MaxLengthCriteria(StoppingCriteria):
    """Stop when a sequence reaches *max_length* tokens."""

    def __init__(self, max_length: int) -> None:
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self.max_length = max_length

    def __call__(self, input_ids: List[List[int]], scores: List[float], **kwargs: Any) -> List[bool]:
        return [len(seq) >= self.max_length for seq in input_ids]

    def reset(self) -> None:
        pass


class MinLengthCriteria(StoppingCriteria):
    """Prevent stopping *before* a sequence has *min_length* tokens.

    This criterion is unusual: it returns ``True`` (allow stop) only once the
    minimum length is met.  It is typically used to gate ``EosTokenCriteria``.
    """

    def __init__(self, min_length: int) -> None:
        if min_length < 0:
            raise ValueError("min_length must be >= 0")
        self.min_length = min_length

    def __call__(self, input_ids: List[List[int]], scores: List[float], **kwargs: Any) -> List[bool]:
        return [len(seq) >= self.min_length for seq in input_ids]

    def reset(self) -> None:
        pass


class EosTokenCriteria(StoppingCriteria):
    """Stop when the last generated token is the EOS token."""

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: List[List[int]], scores: List[float], **kwargs: Any) -> List[bool]:
        results: List[bool] = []
        for seq in input_ids:
            if len(seq) == 0:
                results.append(False)
            else:
                results.append(seq[-1] == self.eos_token_id)
        return results

    def reset(self) -> None:
        pass


class MaxTimeCriteria(StoppingCriteria):
    """Stop after *max_time_seconds* wall-clock seconds have elapsed."""

    def __init__(self, max_time_seconds: float) -> None:
        if max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be > 0")
        self.max_time_seconds = max_time_seconds
        self._start_time: Optional[float] = None

    def __call__(self, input_ids: List[List[int]], scores: List[float], **kwargs: Any) -> List[bool]:
        if self._start_time is None:
            self._start_time = time.monotonic()
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.max_time_seconds:
            return [True] * len(input_ids)
        return [False] * len(input_ids)

    def reset(self) -> None:
        self._start_time = None


class StoppingCriteriaList:
    """Aggregate multiple :class:`StoppingCriteria` instances.

    A sequence is considered finished when **any** criterion says so.
    """

    def __init__(self, criteria: Optional[List[StoppingCriteria]] = None) -> None:
        self.criteria: List[StoppingCriteria] = list(criteria) if criteria else []

    def add(self, criterion: StoppingCriteria) -> None:
        self.criteria.append(criterion)

    def __call__(self, input_ids: List[List[int]], scores: List[float], **kwargs: Any) -> List[bool]:
        if not self.criteria:
            return [False] * len(input_ids)

        n = len(input_ids)
        combined = [False] * n
        for criterion in self.criteria:
            flags = criterion(input_ids, scores, **kwargs)
            for i in range(n):
                combined[i] = combined[i] or flags[i]
        return combined

    def reset(self) -> None:
        for c in self.criteria:
            c.reset()

    def __len__(self) -> int:
        return len(self.criteria)

    def __iter__(self):
        return iter(self.criteria)


# =========================================================================
# Helper / sampling functions
# =========================================================================


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero-out all logits outside the top-*k* (set to -inf)."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    logits = logits.copy()
    threshold = np.partition(logits, -k, axis=-1)[..., -k]
    if logits.ndim == 1:
        logits[logits < threshold] = -np.inf
    else:
        mask = logits < threshold[..., np.newaxis]
        logits[mask] = -np.inf
    return logits


def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus (top-*p*) filtering — keep smallest set whose cumulative prob >= *p*."""
    if p >= 1.0:
        return logits
    logits = logits.copy()
    if logits.ndim == 1:
        sorted_indices = np.argsort(-logits)
        sorted_logits = logits[sorted_indices]
        probs = _stable_softmax(sorted_logits)
        cumulative = np.cumsum(probs)
        cutoff_mask = cumulative - probs > p
        sorted_logits[cutoff_mask] = -np.inf
        logits[sorted_indices] = sorted_logits
    else:
        for b in range(logits.shape[0]):
            logits[b] = _top_p_filter(logits[b], p)
    return logits


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a single token from 1-D *logits* with temperature, top-k, and top-p.

    Parameters
    ----------
    logits : np.ndarray
        Raw (unnormalised) logits of shape ``(vocab_size,)``.
    temperature : float
        Sampling temperature. Values < 1 sharpen, > 1 flatten.
    top_k : int
        If > 0, only the *top_k* highest-probability tokens are kept.
    top_p : float
        Nucleus sampling threshold in ``(0, 1]``.

    Returns
    -------
    int
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k > 0:
        logits = _top_k_filter(logits, top_k)
    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)

    probs = _stable_softmax(logits)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return int(np.argmax(logits))
    probs /= total

    return categorical_sample(probs)


def gumbel_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample using the Gumbel-max trick.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits of shape ``(vocab_size,)``.
    temperature : float
        Sampling temperature applied before adding Gumbel noise.

    Returns
    -------
    int
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()

    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits / temperature
    # Gumbel(0,1) noise via inverse-CDF:  -log(-log(U))
    u = np.random.uniform(low=1e-10, high=1.0 - 1e-10, size=scaled.shape)
    gumbel_noise = -np.log(-np.log(u))
    perturbed = scaled + gumbel_noise
    return int(np.argmax(perturbed))


def categorical_sample(probs: np.ndarray) -> int:
    """Draw a single sample from a categorical distribution.

    Parameters
    ----------
    probs : np.ndarray
        Probability vector of shape ``(vocab_size,)``.  Must sum to ~1.

    Returns
    -------
    int
        Sampled index.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return 0
    probs /= total

    cumulative = np.cumsum(probs)
    r = np.random.uniform()
    idx = np.searchsorted(cumulative, r)
    return int(np.clip(idx, 0, len(probs) - 1))


def beam_search_step(
    logits: np.ndarray,
    beam_scores: np.ndarray,
    beam_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one step of beam search.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(current_beams, vocab_size)`` — raw logits for each beam.
    beam_scores : np.ndarray
        Shape ``(current_beams,)`` — accumulated log-prob scores.
    beam_size : int
        Number of beams to keep.

    Returns
    -------
    beam_indices : np.ndarray
        Shape ``(beam_size,)`` — which input beam each output beam continues.
    token_indices : np.ndarray
        Shape ``(beam_size,)`` — chosen next token for each output beam.
    new_scores : np.ndarray
        Shape ``(beam_size,)`` — updated accumulated scores.
    """
    logits = np.asarray(logits, dtype=np.float64)
    beam_scores = np.asarray(beam_scores, dtype=np.float64)

    num_beams, vocab_size = logits.shape

    log_probs = _log_softmax(logits)  # (num_beams, vocab)
    # Expand scores: each beam's score + log-prob of each token
    next_scores = beam_scores[:, np.newaxis] + log_probs  # (num_beams, vocab)

    # Flatten and pick top-k
    flat_scores = next_scores.reshape(-1)
    num_candidates = min(beam_size, flat_scores.shape[0])
    top_indices = np.argpartition(flat_scores, -num_candidates)[-num_candidates:]
    top_indices = top_indices[np.argsort(-flat_scores[top_indices])]

    beam_indices = top_indices // vocab_size
    token_indices = top_indices % vocab_size
    new_scores = flat_scores[top_indices]

    return (
        beam_indices.astype(np.intp),
        token_indices.astype(np.intp),
        new_scores,
    )


# =========================================================================
# GenerationMixin
# =========================================================================


class GenerationMixin:
    """Shared generation utilities mixed in to :class:`DecodingAlgorithm`.

    Expects the host class to expose ``self.config`` (:class:`DecodingConfig`)
    and ``self._step()`` / ``self._should_stop()`` from the algorithm ABC.
    """

    config: DecodingConfig  # provided by host

    # -- preparation --------------------------------------------------------

    def _prepare_generation(self, prompt_ids: List[int]) -> DecodingState:
        """Build the initial :class:`DecodingState` from a prompt.

        Creates ``config.num_sequences`` copies of the prompt and
        initialises scores to zero.
        """
        n = self.config.num_sequences
        state = DecodingState(
            sequences=[list(prompt_ids) for _ in range(n)],
            scores=[0.0] * n,
            is_finished=[False] * n,
            step=0,
            metadata={"prompt_length": len(prompt_ids)},
            embeddings=None,
            logit_history=[],
        )
        return state

    # -- main loop ----------------------------------------------------------

    def _generation_loop(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Run the token-generation loop until all sequences finish or limits are hit.

        Delegates per-step logic to ``self._step(state, logit_source)``.
        """
        stopping = self._build_stopping_criteria()
        stopping.reset()

        max_steps = self.config.max_new_tokens
        step_times: List[float] = []

        for step_idx in range(max_steps):
            if state.all_finished():
                logger.debug("All sequences finished at step %d", step_idx)
                break

            t0 = time.monotonic()
            state = self._step(state, logit_source)  # type: ignore[attr-defined]
            state.step = step_idx + 1
            step_times.append(time.monotonic() - t0)

            # Apply stopping criteria
            stop_flags = stopping(state.sequences, state.scores)
            for i, should_stop in enumerate(stop_flags):
                if should_stop and not state.is_finished[i]:
                    state.mark_finished(i)

            # Secondary check via _should_stop
            if self._should_stop(state):  # type: ignore[attr-defined]
                logger.debug("Algorithm-level stop triggered at step %d", step_idx)
                break

        state.metadata["step_times"] = step_times
        state.metadata["total_steps"] = state.step
        state.metadata["total_time"] = sum(step_times)
        return state

    # -- post-processing ----------------------------------------------------

    def _postprocess(self, state: DecodingState) -> List[TokenSequence]:
        """Extract token sequences from final state, sorted by score descending."""
        prompt_len = state.metadata.get("prompt_length", 0)
        pairs: List[Tuple[float, TokenSequence]] = []
        for i in range(state.num_sequences):
            seq = state.get_sequence(i)
            generated = seq[prompt_len:]
            pairs.append((state.scores[i], generated))

        pairs.sort(key=lambda p: p[0], reverse=True)
        return [seq for _, seq in pairs]

    def _compute_sequence_scores(
        self,
        sequences: List[List[int]],
        logit_source: LogitSource,
        prompt_length: int = 0,
    ) -> List[float]:
        """Re-score completed sequences under the model.

        For each sequence the method feeds the prefix through *logit_source*
        one token at a time and accumulates the log-probability of each
        generated token.

        Parameters
        ----------
        sequences :
            Full token sequences (prompt + generated).
        logit_source :
            Callable returning logits of shape ``(batch, vocab)``.
        prompt_length :
            Number of prompt tokens to skip when accumulating scores.

        Returns
        -------
        List of total log-probability scores, one per sequence.
        """
        scores: List[float] = []
        for seq in sequences:
            total_lp = 0.0
            for t in range(max(prompt_length, 1), len(seq)):
                prefix = [seq[:t]]
                logits = logit_source(prefix)  # (1, vocab)
                log_p = _log_softmax(logits[0])
                total_lp += float(log_p[seq[t]])
            scores.append(total_lp)
        return scores

    # -- constraint helpers -------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(
        logits: np.ndarray,
        input_ids: List[int],
        penalty: float,
    ) -> np.ndarray:
        """Multiplicative repetition penalty (Keskar et al., 2019).

        Tokens that already appear in *input_ids* have their logits divided
        (if positive) or multiplied (if negative) by *penalty*.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        input_ids : list of int
            Previously generated token ids.
        penalty : float
            Penalty factor >= 1.0.  1.0 means no penalty.

        Returns
        -------
        np.ndarray
            Modified logits (in-place on a copy).
        """
        if penalty == 1.0 or len(input_ids) == 0:
            return logits

        logits = logits.copy()
        unique_ids = set(input_ids)
        for token_id in unique_ids:
            if 0 <= token_id < logits.shape[-1]:
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
        return logits

    @staticmethod
    def _apply_no_repeat_ngram(
        logits: np.ndarray,
        input_ids: List[int],
        ngram_size: int,
    ) -> np.ndarray:
        """Ban repeated n-grams by setting offending logits to ``-inf``.

        Scans *input_ids* for all n-grams of size *ngram_size*.  If the
        current ``(ngram_size-1)``-suffix matches a previously seen n-gram
        prefix, the completing token is banned.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        input_ids : list of int
            Previously generated token ids.
        ngram_size : int
            N-gram size.  0 or 1 disables the constraint.

        Returns
        -------
        np.ndarray
            Modified logits (copy).
        """
        if ngram_size < 2 or len(input_ids) < ngram_size - 1:
            return logits

        logits = logits.copy()

        # Build set of all (ngram_size-1)-grams and their completing token.
        banned_tokens: set = set()
        suffix = tuple(input_ids[-(ngram_size - 1):])
        for i in range(len(input_ids) - ngram_size + 1):
            ngram_prefix = tuple(input_ids[i: i + ngram_size - 1])
            if ngram_prefix == suffix:
                banned_tokens.add(input_ids[i + ngram_size - 1])

        for token_id in banned_tokens:
            if 0 <= token_id < logits.shape[-1]:
                logits[token_id] = -np.inf

        return logits

    @staticmethod
    def _enforce_min_length(
        logits: np.ndarray,
        cur_len: int,
        min_length: int,
        eos_id: Optional[int],
    ) -> np.ndarray:
        """Prevent EOS from being generated before *min_length*.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        cur_len : int
            Current generated length (excluding prompt).
        min_length : int
            Minimum generated length before EOS is allowed.
        eos_id : int or None
            EOS token id.

        Returns
        -------
        np.ndarray
            Modified logits (copy).
        """
        if eos_id is None or cur_len >= min_length:
            return logits

        logits = logits.copy()
        if 0 <= eos_id < logits.shape[-1]:
            logits[eos_id] = -np.inf
        return logits

    # -- stopping criteria builder ------------------------------------------

    def _build_stopping_criteria(self) -> StoppingCriteriaList:
        """Construct a :class:`StoppingCriteriaList` from config."""
        criteria: List[StoppingCriteria] = []

        prompt_len = 0  # will be adjusted in _generation_loop if needed
        max_total = prompt_len + self.config.max_new_tokens
        criteria.append(MaxLengthCriteria(max_total))

        if self.config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(self.config.eos_token_id))

        return StoppingCriteriaList(criteria)


# =========================================================================
# DecodingAlgorithm ABC
# =========================================================================


class DecodingAlgorithm(GenerationMixin, abc.ABC):
    """Abstract base class for all decoding algorithms.

    Subclasses **must** implement :meth:`_step`.  They *may* override any
    other method for algorithm-specific behaviour.
    """

    def __init__(self, config: DecodingConfig) -> None:
        self.config = config
        self._rng: Optional[np.random.Generator] = None
        if config.seed is not None:
            self._rng = np.random.default_rng(config.seed)

        errors = self.validate_config()
        if errors:
            logger.warning("Config validation warnings for %s: %s", self.name, errors)

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable algorithm name."""
        return self.config.algorithm_name or self.__class__.__name__

    @property
    def description(self) -> str:
        """One-line description.  Subclasses should override."""
        return f"{self.name} decoding algorithm"

    # -- public entry points ------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate ``num_sequences`` continuations of *prompt_ids*.

        This is the main public entry point.

        Parameters
        ----------
        logit_source :
            Callable ``(List[List[int]]) -> np.ndarray`` returning logits.
        prompt_ids :
            Token ids of the prompt.

        Returns
        -------
        list of TokenSequence
            Generated token sequences sorted by score (best first).
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        state = self._init_state(prompt_ids)
        state = self._generation_loop(state, logit_source)
        return self._finalize(state)

    def generate_batch(
        self,
        logit_source: LogitSource,
        prompt_ids_batch: List[List[int]],
    ) -> List[List[TokenSequence]]:
        """Generate continuations for a batch of prompts.

        Default implementation simply loops; subclasses may parallelise.
        """
        results: List[List[TokenSequence]] = []
        for prompt_ids in prompt_ids_batch:
            results.append(self.generate(logit_source, prompt_ids))
        return results

    # -- lifecycle hooks (overridable) --------------------------------------

    def _init_state(self, prompt_ids: List[int]) -> DecodingState:
        """Initialise the decoding state.  Delegates to :meth:`_prepare_generation`."""
        return self._prepare_generation(prompt_ids)

    @abc.abstractmethod
    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute a single decoding step (abstract).

        Must modify *state* in-place (or return a new state) with:
        - new tokens appended to ``state.sequences``
        - scores updated in ``state.scores``
        - finished flags set in ``state.is_finished``

        Parameters
        ----------
        state : DecodingState
        logit_source : LogitSource

        Returns
        -------
        DecodingState
        """
        ...

    def _finalize(self, state: DecodingState) -> List[TokenSequence]:
        """Extract and rank final token sequences from the completed state."""
        return self._postprocess(state)

    def _should_stop(self, state: DecodingState) -> bool:
        """Algorithm-level stopping check (beyond per-sequence criteria).

        Override in subclasses to implement convergence checks, etc.
        """
        return state.all_finished()

    def _apply_constraints(self, logits: np.ndarray, state: DecodingState) -> np.ndarray:
        """Apply all configured constraints to raw logits for a *single* sequence.

        Applies repetition penalty, n-gram blocking, and min-length enforcement
        using the utilities from :class:`GenerationMixin`.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.
        state : DecodingState
            Current generation state (used for context like generated length).

        Returns
        -------
        np.ndarray
            Constrained logits.
        """
        # Repetition penalty — uses the full sequence as context
        if self.config.repetition_penalty > 1.0:
            # Collect all tokens across active sequences as context
            all_tokens: List[int] = []
            for seq in state.sequences:
                all_tokens.extend(seq)
            logits = self._apply_repetition_penalty(
                logits, all_tokens, self.config.repetition_penalty
            )

        # No-repeat n-gram
        if self.config.no_repeat_ngram_size > 0:
            for i in state.active_indices():
                logits = self._apply_no_repeat_ngram(
                    logits, state.sequences[i], self.config.no_repeat_ngram_size
                )

        # Min-length enforcement
        prompt_len = state.metadata.get("prompt_length", 0)
        gen_len = (len(state.sequences[0]) - prompt_len) if state.sequences else 0
        logits = self._enforce_min_length(
            logits, gen_len, self.config.min_new_tokens, self.config.eos_token_id
        )

        return logits

    # -- introspection ------------------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        """Describe the hyper-parameter search space for this algorithm.

        Returns a dictionary mapping parameter names to dicts with keys
        ``type`` (``"float"``, ``"int"``, ``"categorical"``), ``low``,
        ``high``, ``choices``, etc.

        Subclasses should override and call ``super()`` to include base
        parameters.
        """
        return {
            "temperature": {"type": "float", "low": 0.1, "high": 2.0, "log": True},
            "repetition_penalty": {"type": "float", "low": 1.0, "high": 2.0},
            "no_repeat_ngram_size": {"type": "int", "low": 0, "high": 5},
        }

    def validate_config(self) -> List[str]:
        """Validate the current config and return a list of error strings."""
        return self.config.validate()

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, config={self.config!r})"


# =========================================================================
# AlgorithmRegistry
# =========================================================================


class AlgorithmRegistry:
    """Global registry of available decoding algorithms."""

    _registry: Dict[str, Type[DecodingAlgorithm]] = {}

    @classmethod
    def register(cls, name: str, algorithm_cls: Type[DecodingAlgorithm]) -> None:
        """Register an algorithm class under *name*.

        Parameters
        ----------
        name : str
            Unique identifier for the algorithm.
        algorithm_cls : Type[DecodingAlgorithm]
            The class implementing the algorithm.

        Raises
        ------
        ValueError
            If *name* is already registered (and the class differs).
        TypeError
            If *algorithm_cls* is not a subclass of DecodingAlgorithm.
        """
        if not (isinstance(algorithm_cls, type) and issubclass(algorithm_cls, DecodingAlgorithm)):
            raise TypeError(
                f"Expected a DecodingAlgorithm subclass, got {algorithm_cls!r}"
            )
        if name in cls._registry and cls._registry[name] is not algorithm_cls:
            logger.warning(
                "Overwriting existing registration for '%s': %s -> %s",
                name,
                cls._registry[name],
                algorithm_cls,
            )
        cls._registry[name] = algorithm_cls
        logger.debug("Registered algorithm '%s' -> %s", name, algorithm_cls.__name__)

    @classmethod
    def get(cls, name: str) -> Type[DecodingAlgorithm]:
        """Retrieve the algorithm class for *name*.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown algorithm '{name}'. Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(cls._registry.keys())

    @classmethod
    def create(cls, name: str, config: Optional[DecodingConfig] = None) -> DecodingAlgorithm:
        """Instantiate a registered algorithm.

        Parameters
        ----------
        name : str
            Registered algorithm name.
        config : DecodingConfig, optional
            Configuration.  A default is created if not provided.

        Returns
        -------
        DecodingAlgorithm
            A ready-to-use algorithm instance.
        """
        algorithm_cls = cls.get(name)
        if config is None:
            config = DecodingConfig(algorithm_name=name)
        else:
            config = copy.deepcopy(config)
            config.algorithm_name = name
        return algorithm_cls(config)

    @classmethod
    def register_decorator(cls, name: str) -> Callable[[Type[DecodingAlgorithm]], Type[DecodingAlgorithm]]:
        """Class decorator for convenient registration.

        Usage::

            @AlgorithmRegistry.register_decorator("my_algo")
            class MyAlgo(DecodingAlgorithm):
                ...
        """

        def decorator(algorithm_cls: Type[DecodingAlgorithm]) -> Type[DecodingAlgorithm]:
            cls.register(name, algorithm_cls)
            return algorithm_cls

        return decorator

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful for testing)."""
        cls._registry.clear()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether *name* is registered."""
        return name in cls._registry


# =========================================================================
# Convenience aliases
# =========================================================================

register = AlgorithmRegistry.register_decorator


# =========================================================================
# Module-level sanity check
# =========================================================================

def _self_test() -> None:
    """Minimal smoke test executed when the module is run directly."""

    # -- DecodingConfig round-trip --
    cfg = DecodingConfig(
        algorithm_name="test",
        num_sequences=5,
        temperature=0.8,
        params={"custom_key": 42},
    )
    assert cfg.validate() == []
    d = cfg.to_dict()
    cfg2 = DecodingConfig.from_dict(d)
    assert cfg2.algorithm_name == "test"
    assert cfg2.params["custom_key"] == 42
    assert cfg.hash() == cfg2.hash()

    # -- DecodingState --
    state = DecodingState(
        sequences=[[1, 2, 3], [4, 5, 6]],
        scores=[0.0, 0.0],
        is_finished=[False, True],
    )
    assert state.active_indices() == [0]
    assert state.num_active() == 1
    assert not state.all_finished()
    state.update_sequence(0, 7)
    assert state.get_sequence(0) == [1, 2, 3, 7]
    state.mark_finished(0)
    assert state.all_finished()

    # -- Stopping criteria --
    sc = StoppingCriteriaList([
        MaxLengthCriteria(5),
        EosTokenCriteria(99),
    ])
    flags = sc([[1, 2, 3, 4, 5]], [0.0])
    assert flags == [True]
    flags = sc([[1, 2]], [0.0])
    assert flags == [False]

    # -- Sampling helpers --
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    token = sample_token(logits, temperature=0.0001)  # near-greedy
    assert token == 3

    token_g = gumbel_sample(logits, temperature=0.0001)
    assert token_g == 3

    probs = np.array([0.0, 0.0, 1.0, 0.0])
    assert categorical_sample(probs) == 2

    # -- Beam search step --
    logits_2 = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
    scores_2 = np.array([0.0, -1.0])
    bi, ti, ns = beam_search_step(logits_2, scores_2, beam_size=2)
    assert len(bi) == 2
    assert len(ti) == 2

    # -- Repetition penalty --
    logits_r = np.array([1.0, 2.0, 3.0])
    out = GenerationMixin._apply_repetition_penalty(logits_r, [2], 2.0)
    assert out[2] < logits_r[2]  # positive logit divided

    # -- No-repeat n-gram --
    logits_n = np.array([1.0, 1.0, 1.0, 1.0])
    out_n = GenerationMixin._apply_no_repeat_ngram(logits_n, [0, 1, 2, 0, 1], 3)
    assert out_n[2] == -np.inf  # trigram 0,1,2 would repeat

    # -- Registry --
    AlgorithmRegistry.clear()

    @register("dummy")
    class DummyAlgorithm(DecodingAlgorithm):
        def _step(self, state, logit_source):
            for i in state.active_indices():
                state.update_sequence(i, 0)
                state.mark_finished(i)
            return state

    assert AlgorithmRegistry.is_registered("dummy")
    assert "dummy" in AlgorithmRegistry.list_algorithms()

    algo = AlgorithmRegistry.create("dummy", DecodingConfig(num_sequences=2, max_new_tokens=5))
    assert algo.name == "dummy"

    dummy_source: LogitSource = lambda ids: np.zeros((len(ids), 4))
    results = algo.generate(dummy_source, [1, 2, 3])
    assert len(results) == 2

    AlgorithmRegistry.clear()

    print("base.py self-test passed ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
