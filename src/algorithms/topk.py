"""
Top-K Sampling for the Diversity Decoding Arena.

Implements top-k filtering, dynamic k scheduling, combined top-k + temperature
sampling, and analysis utilities for studying how k affects diversity and
coverage of the output distribution.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import softmax as scipy_softmax, log_softmax as scipy_log_softmax

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    AlgorithmRegistry,
    register,
    _stable_softmax,
    _log_softmax,
    categorical_sample,
    sample_token,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Helper functions (module-level)
# =========================================================================


def top_k_filter(
    logits: np.ndarray,
    k: int,
    filter_value: float = -float("inf"),
) -> np.ndarray:
    """Filter a 1-D logit vector, keeping only the top-*k* values.

    All positions outside the top-*k* are replaced with *filter_value*
    (default ``-inf``).

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)``.
    k : int
        Number of top values to keep.
    filter_value : float
        Value used for masked positions.

    Returns
    -------
    np.ndarray
        Filtered logits (copy).
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    vocab_size = logits.shape[-1]

    if k <= 0:
        logits[:] = filter_value
        return logits
    if k >= vocab_size:
        return logits

    # Find the k-th largest value as threshold
    threshold = np.partition(logits, -k)[-k]
    logits[logits < threshold] = filter_value
    return logits


def top_k_indices(logits: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-*k* logits, sorted descending.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)``.
    k : int
        Number of top indices to return.

    Returns
    -------
    np.ndarray
        Indices of the top-*k* logits in descending order.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    k = min(k, logits.shape[0])
    if k <= 0:
        return np.array([], dtype=np.intp)
    indices = np.argpartition(logits, -k)[-k:]
    # Sort descending by logit value
    indices = indices[np.argsort(-logits[indices])]
    return indices


def probability_mass_in_top_k(logits: np.ndarray, k: int) -> float:
    """Compute the fraction of total probability mass captured by the top-*k* tokens.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)``.
    k : int
        Number of top tokens.

    Returns
    -------
    float
        Probability mass in ``[0, 1]``.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    probs = _stable_softmax(logits)
    if k <= 0:
        return 0.0
    if k >= probs.shape[0]:
        return 1.0
    top_indices = np.argpartition(probs, -k)[-k:]
    return float(np.sum(probs[top_indices]))


def effective_vocabulary_size(logits: np.ndarray) -> float:
    """Compute the effective vocabulary size as ``exp(entropy)``.

    This measures the "effective number of choices" the distribution offers.
    A uniform distribution over *V* tokens yields *V*; a peaked distribution
    yields a value close to 1.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)``.

    Returns
    -------
    float
        Effective vocabulary size.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    probs = _stable_softmax(logits)
    probs = np.clip(probs, 1e-30, None)
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def batch_top_k(logits_batch: np.ndarray, k: int) -> np.ndarray:
    """Apply top-*k* filtering to a batch of logit vectors.

    Parameters
    ----------
    logits_batch : np.ndarray
        Shape ``(batch, vocab_size)``.
    k : int
        Number of top values to keep per row.

    Returns
    -------
    np.ndarray
        Filtered logits of the same shape.
    """
    logits_batch = np.asarray(logits_batch, dtype=np.float64).copy()
    if logits_batch.ndim == 1:
        return top_k_filter(logits_batch, k)

    batch_size, vocab_size = logits_batch.shape
    if k <= 0:
        logits_batch[:] = -np.inf
        return logits_batch
    if k >= vocab_size:
        return logits_batch

    # Per-row threshold via partition
    thresholds = np.partition(logits_batch, -k, axis=1)[:, -k]  # (batch,)
    mask = logits_batch < thresholds[:, np.newaxis]
    logits_batch[mask] = -np.inf
    return logits_batch


# =========================================================================
# TopKConfig
# =========================================================================


@dataclass
class TopKConfig(DecodingConfig):
    """Configuration for Top-K sampling.

    Extends :class:`DecodingConfig` with k-specific parameters including
    dynamic k scheduling and repetition penalty.
    """

    algorithm_name: str = "top_k"

    # Core top-k parameters
    k: int = 50
    temperature: float = 1.0

    # Dynamic k settings
    dynamic_k: bool = False
    k_schedule: str = "constant"  # constant, linear, entropy_based
    k_min: int = 5
    k_max: int = 200

    # Sampling behaviour
    with_replacement: bool = True
    filter_value: float = -float("inf")

    # Constraints
    repetition_penalty: float = 1.0

    def validate(self) -> List[str]:
        """Validate top-k-specific parameters on top of the base checks."""
        errors = super().validate()
        if self.k < 1:
            errors.append("k must be >= 1")
        if self.temperature <= 0:
            errors.append("temperature must be > 0")
        if self.k_min < 1:
            errors.append("k_min must be >= 1")
        if self.k_max < self.k_min:
            errors.append("k_max must be >= k_min")
        if self.k_schedule not in ("constant", "linear", "entropy_based"):
            errors.append(
                f"k_schedule must be one of constant, linear, entropy_based; "
                f"got {self.k_schedule!r}"
            )
        if self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
        return errors


# =========================================================================
# TopKSchedule
# =========================================================================


class TopKSchedule:
    """Manage dynamic k values across generation steps.

    Supports constant, linear decay, step-decay, and entropy-adaptive
    schedules.

    Parameters
    ----------
    schedule_type : str
        One of ``"constant"``, ``"linear"``, ``"entropy_based"``, ``"step_decay"``.
    k_start : int
        Initial k value (used by linear and step_decay).
    k_end : int
        Final k value for linear schedule.
    total_steps : int
        Total number of generation steps (needed for linear schedule).
    k_min : int
        Minimum k for entropy-based schedule.
    k_max : int
        Maximum k for entropy-based schedule.
    decay_rate : float
        Multiplicative decay factor for step_decay schedule (applied each step).
    decay_every : int
        Apply step_decay every *decay_every* steps.
    """

    _VALID_TYPES = ("constant", "linear", "entropy_based", "step_decay")

    def __init__(
        self,
        schedule_type: str = "constant",
        k_start: int = 50,
        k_end: int = 10,
        total_steps: int = 100,
        k_min: int = 5,
        k_max: int = 200,
        decay_rate: float = 0.95,
        decay_every: int = 10,
    ) -> None:
        if schedule_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown schedule_type {schedule_type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        self.schedule_type = schedule_type
        self.k_start = k_start
        self.k_end = k_end
        self.total_steps = max(total_steps, 1)
        self.k_min = k_min
        self.k_max = k_max
        self.decay_rate = decay_rate
        self.decay_every = max(decay_every, 1)

        self._dispatch = {
            "constant": self.constant,
            "linear": self.linear,
            "step_decay": self.step_decay,
        }

    # -- public API ---------------------------------------------------------

    def get_k(self, step: int, logits: Optional[np.ndarray] = None) -> int:
        """Return the k value for the given *step*.

        For entropy-based scheduling, *logits* must be provided.

        Parameters
        ----------
        step : int
            Current generation step (0-indexed).
        logits : np.ndarray, optional
            Raw logits for entropy-based scheduling.

        Returns
        -------
        int
            The k value to use at this step.
        """
        if self.schedule_type == "entropy_based":
            if logits is None:
                logger.warning(
                    "entropy_based schedule requires logits; "
                    "falling back to constant k=%d",
                    self.k_start,
                )
                return self.k_start
            return self.entropy_based(logits)

        fn = self._dispatch.get(self.schedule_type)
        if fn is None:
            return self.k_start
        return fn(step)

    # -- schedule implementations -------------------------------------------

    def constant(self, step: int) -> int:
        """Constant k across all steps."""
        return self.k_start

    def linear(self, step: int) -> int:
        """Linearly interpolate k from *k_start* to *k_end* over *total_steps*.

        The value is clamped between ``k_min`` and ``k_max``.
        """
        if self.total_steps <= 1:
            return self.k_start
        frac = min(step / (self.total_steps - 1), 1.0)
        k_float = self.k_start + frac * (self.k_end - self.k_start)
        k = int(round(k_float))
        return max(self.k_min, min(k, self.k_max))

    def entropy_based(self, logits: np.ndarray) -> int:
        """Choose k proportionally to the distribution entropy.

        High-entropy distributions (many plausible tokens) get a larger k;
        low-entropy distributions (confident predictions) get a smaller k.

        The mapping is:

            k = k_min + (k_max - k_min) * (H / H_max)

        where *H* is the entropy of ``softmax(logits)`` and *H_max* =
        ``log(vocab_size)``.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        probs = _stable_softmax(logits)
        probs = np.clip(probs, 1e-30, None)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(float(logits.shape[0]))
        if max_entropy <= 0:
            return self.k_min
        ratio = float(np.clip(entropy / max_entropy, 0.0, 1.0))
        k = self.k_min + ratio * (self.k_max - self.k_min)
        return max(self.k_min, min(int(round(k)), self.k_max))

    def step_decay(self, step: int) -> int:
        """Multiplicative decay applied every *decay_every* steps.

        ``k = k_start * decay_rate^(step // decay_every)``
        """
        num_decays = step // self.decay_every
        k_float = self.k_start * (self.decay_rate ** num_decays)
        k = int(round(k_float))
        return max(self.k_min, min(k, self.k_max))

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TopKSchedule(type={self.schedule_type!r}, "
            f"k_start={self.k_start}, k_end={self.k_end}, "
            f"total_steps={self.total_steps})"
        )


# =========================================================================
# TopKSampling  (main algorithm)
# =========================================================================


@register("top_k")
class TopKSampling(DecodingAlgorithm):
    """Top-K sampling decoding algorithm.

    At each step the logits are filtered to the *k* highest values, re-normalised,
    and a token is sampled from the resulting distribution.  Supports dynamic
    k scheduling, temperature scaling, and repetition penalty.

    Parameters
    ----------
    config : TopKConfig | DecodingConfig
        Configuration.  If a plain :class:`DecodingConfig` is passed, sensible
        top-k defaults are used.
    """

    def __init__(self, config: Optional[Union[TopKConfig, DecodingConfig]] = None) -> None:
        if config is None:
            config = TopKConfig()
        elif not isinstance(config, TopKConfig):
            # Upgrade a generic DecodingConfig
            cfg = TopKConfig()
            for fld in DecodingConfig.__dataclass_fields__:
                setattr(cfg, fld, getattr(config, fld))
            config = cfg
        self._topk_config: TopKConfig = config  # type: ignore[assignment]
        super().__init__(config)
        self._schedule: Optional[TopKSchedule] = None
        if self._topk_config.dynamic_k:
            self._schedule = TopKSchedule(
                schedule_type=self._topk_config.k_schedule,
                k_start=self._topk_config.k,
                k_end=self._topk_config.k_min,
                total_steps=self._topk_config.max_new_tokens,
                k_min=self._topk_config.k_min,
                k_max=self._topk_config.k_max,
            )

    # -- properties ---------------------------------------------------------

    @property
    def description(self) -> str:
        return f"Top-K sampling (k={self._topk_config.k}, T={self._topk_config.temperature})"

    # -- public overrides ---------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate ``num_sequences`` top-k sampled continuations.

        If ``with_replacement`` is ``False`` in the config, the algorithm
        attempts to produce distinct sequences (up to a retry budget).

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        if self._topk_config.with_replacement:
            state = self._init_state(prompt_ids)
            state = self._generation_loop(state, logit_source)
            return self._finalize(state)

        # Without replacement: generate until we have enough unique sequences
        seen: set = set()
        results: List[TokenSequence] = []
        max_tries = self.config.num_sequences * 4
        attempt = 0
        while len(results) < self.config.num_sequences and attempt < max_tries:
            seq = self._generate_single_sequence(logit_source, prompt_ids)
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                results.append(seq)
            attempt += 1

        if len(results) < self.config.num_sequences:
            logger.warning(
                "Could only produce %d unique sequences out of %d requested "
                "(after %d attempts)",
                len(results),
                self.config.num_sequences,
                max_tries,
            )
        return results

    # -- _step (core loop body) ---------------------------------------------

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute one top-k sampling step for all active sequences.

        1. Obtain logits from *logit_source* for each active sequence.
        2. Apply constraints (repetition penalty, n-gram blocking).
        3. Determine effective k (possibly dynamic).
        4. Apply top-k filtering.
        5. Sample a token and append to each active sequence.
        """
        active = state.active_indices()
        if not active:
            return state

        # Build batch input
        batch_input = [state.get_sequence(i) for i in active]
        logits_batch = logit_source(batch_input)  # (len(active), vocab)

        for local_idx, global_idx in enumerate(active):
            logits = logits_batch[local_idx].copy()

            # Apply constraints
            logits = self._apply_constraints(logits, state)

            # Determine k
            k = self._resolve_k(logits, state)

            # Top-k filter + sample
            token = self._sample_from_top_k(
                logits, k, self._topk_config.temperature
            )

            state.update_sequence(global_idx, token)

            # Score update (log-prob of chosen token)
            log_probs = _log_softmax(logits_batch[local_idx])
            state.scores[global_idx] += float(log_probs[token])

            # Check EOS
            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                prompt_len = state.metadata.get("prompt_length", 0)
                gen_len = len(state.sequences[global_idx]) - prompt_len
                if gen_len >= self.config.min_new_tokens:
                    state.mark_finished(global_idx)

            # Store logits for analysis
            if state.logit_history is not None:
                state.logit_history.append(logits)

        return state

    # -- top-k filtering ----------------------------------------------------

    def _top_k_filter(self, logits: np.ndarray, k: int) -> np.ndarray:
        """Apply top-k filtering to a single logit vector.

        Finds the k-th largest logit value, then masks all logits below that
        threshold to ``filter_value`` (default ``-inf``).

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k : int
            Number of top logits to keep.

        Returns
        -------
        np.ndarray
            Filtered logits (copy).
        """
        return top_k_filter(logits, k, self._topk_config.filter_value)

    def _batch_top_k_filter(self, logits_batch: np.ndarray, k: int) -> np.ndarray:
        """Apply top-k filtering to a batch of logit vectors.

        Parameters
        ----------
        logits_batch : np.ndarray
            Shape ``(batch, vocab_size)``.
        k : int
            Number of top logits to keep per row.

        Returns
        -------
        np.ndarray
            Filtered logits batch (copy).
        """
        logits_batch = np.asarray(logits_batch, dtype=np.float64).copy()
        if logits_batch.ndim == 1:
            return self._top_k_filter(logits_batch, k)

        batch_size, vocab_size = logits_batch.shape
        if k <= 0:
            logits_batch[:] = self._topk_config.filter_value
            return logits_batch
        if k >= vocab_size:
            return logits_batch

        thresholds = np.partition(logits_batch, -k, axis=1)[:, -k]
        mask = logits_batch < thresholds[:, np.newaxis]
        logits_batch[mask] = self._topk_config.filter_value
        return logits_batch

    # -- dynamic k ----------------------------------------------------------

    def _resolve_k(self, logits: np.ndarray, state: DecodingState) -> int:
        """Determine the k value for the current step.

        Uses either the static ``config.k`` or a dynamic schedule.
        """
        if not self._topk_config.dynamic_k or self._schedule is None:
            return self._topk_config.k
        return self._dynamic_k(logits, state)

    def _dynamic_k(self, logits: np.ndarray, state: DecodingState) -> int:
        """Compute k dynamically based on the configured schedule.

        Parameters
        ----------
        logits : np.ndarray
            Current-step logits (needed for entropy-based scheduling).
        state : DecodingState
            Current decoding state (provides step index).

        Returns
        -------
        int
            Effective k for this step.
        """
        if self._schedule is None:
            return self._topk_config.k
        return self._schedule.get_k(state.step, logits)

    def _entropy_based_k(
        self,
        logits: np.ndarray,
        k_min: int,
        k_max: int,
    ) -> int:
        """Compute k proportionally to the entropy of the logit distribution.

        Higher entropy → larger k (more choices are plausible).
        Lower entropy → smaller k (the model is confident).

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k_min, k_max : int
            Bounds for the returned k.

        Returns
        -------
        int
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        probs = _stable_softmax(logits)
        probs = np.clip(probs, 1e-30, None)
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = float(np.log(logits.shape[0]))
        if max_entropy <= 0:
            return k_min
        ratio = np.clip(entropy / max_entropy, 0.0, 1.0)
        k = k_min + ratio * (k_max - k_min)
        return max(k_min, min(int(round(k)), k_max))

    # -- sampling -----------------------------------------------------------

    def _sample_from_top_k(
        self,
        logits: np.ndarray,
        k: int,
        temperature: float,
    ) -> int:
        """Filter to top-k, apply temperature, and sample one token.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k : int
            Number of top logits to keep.
        temperature : float
            Sampling temperature.

        Returns
        -------
        int
            Sampled token id.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()

        if temperature <= 0:
            # Greedy from top-k
            filtered = self._top_k_filter(logits, k)
            return int(np.argmax(filtered))

        # Filter then scale
        filtered = self._top_k_filter(logits, k)
        scaled = filtered / temperature

        probs = _stable_softmax(scaled)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            return int(np.argmax(logits))
        probs /= total

        return categorical_sample(probs)

    # -- single-sequence generation -----------------------------------------

    def _generate_single_sequence(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> TokenSequence:
        """Generate a single sequence using top-k sampling.

        This is used for the ``with_replacement=False`` path, where each
        sequence is generated independently.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        TokenSequence
            The generated token ids (excluding prompt).
        """
        sequence = list(prompt_ids)
        prompt_len = len(prompt_ids)

        for step in range(self.config.max_new_tokens):
            logits = logit_source([sequence])[0]  # (vocab,)
            logits = logits.copy()

            # Repetition penalty
            if self._topk_config.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits, sequence, self._topk_config.repetition_penalty
                )

            # Dynamic or static k
            k = self._topk_config.k
            if self._topk_config.dynamic_k and self._schedule is not None:
                k = self._schedule.get_k(step, logits)

            token = self._sample_from_top_k(
                logits, k, self._topk_config.temperature
            )
            sequence.append(token)

            # EOS check
            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                gen_len = len(sequence) - prompt_len
                if gen_len >= self.config.min_new_tokens:
                    break

        return sequence[prompt_len:]

    # -- hyperparameter space -----------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        space = super().hyperparameter_space()
        space.update(
            {
                "k": {"type": "int", "low": 1, "high": 500},
                "temperature": {"type": "float", "low": 0.01, "high": 3.0, "log": True},
                "dynamic_k": {"type": "categorical", "choices": [True, False]},
                "k_schedule": {
                    "type": "categorical",
                    "choices": ["constant", "linear", "entropy_based"],
                },
                "k_min": {"type": "int", "low": 1, "high": 50},
                "k_max": {"type": "int", "low": 10, "high": 1000},
                "repetition_penalty": {"type": "float", "low": 1.0, "high": 2.0},
            }
        )
        return space

    def validate_config(self) -> List[str]:
        return self._topk_config.validate()


# =========================================================================
# TopKAnalyzer
# =========================================================================


class TopKAnalyzer:
    """Analyse the effect of k on sampling behaviour and diversity.

    Provides tools for studying how the choice of *k* affects probability
    coverage, output diversity, entropy, and truncation loss.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    # -- main analysis entry point ------------------------------------------

    def analyze_k_effect(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        k_values: List[int],
        n_samples: int = 50,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Run a comprehensive analysis of how k affects generation.

        For each k in *k_values*, generates *n_samples* sequences and
        computes coverage, diversity, entropy, and truncation metrics.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        k_values : list of int
        n_samples : int
        max_new_tokens : int
        temperature : float

        Returns
        -------
        dict
            Keys: ``"k_values"``, ``"coverage"``, ``"diversity"``,
            ``"entropy"``, ``"truncation"``, ``"samples_per_k"``.
        """
        results: Dict[str, Any] = {
            "k_values": list(k_values),
            "coverage": {},
            "diversity": {},
            "entropy": {},
            "truncation": {},
            "samples_per_k": {},
        }

        # Get a single set of logits for coverage/entropy/truncation analysis
        logits = logit_source([prompt_ids])[0]

        cov = self.coverage_vs_k(logits, k_values)
        results["coverage"] = cov

        ent = self.k_vs_entropy(logits, k_values)
        results["entropy"] = ent

        trunc = self.truncation_analysis(logits, k_values)
        results["truncation"] = trunc

        div = self.diversity_vs_k(
            logit_source, prompt_ids, k_values, n_samples,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        results["diversity"] = div

        logger.info(
            "Top-K analysis complete for %d k values, %d samples each",
            len(k_values),
            n_samples,
        )
        return results

    # -- coverage -----------------------------------------------------------

    def coverage_vs_k(
        self,
        logits: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, Any]:
        """Compute probability mass covered by the top-k for various k.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k_values : list of int

        Returns
        -------
        dict
            ``{"k_values": [...], "coverage": [...]}``
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        probs = _stable_softmax(logits)
        sorted_probs = np.sort(probs)[::-1]
        cumulative = np.cumsum(sorted_probs)

        coverages: List[float] = []
        for k in k_values:
            if k <= 0:
                coverages.append(0.0)
            elif k >= len(probs):
                coverages.append(1.0)
            else:
                coverages.append(float(cumulative[k - 1]))
        return {"k_values": list(k_values), "coverage": coverages}

    # -- diversity ----------------------------------------------------------

    def diversity_vs_k(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        k_values: List[int],
        n_samples: int = 50,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Measure output diversity for different k values.

        Diversity is measured by the number of unique tokens (type-token
        ratio) and the number of distinct sequences.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        k_values : list of int
        n_samples : int
        max_new_tokens : int
        temperature : float

        Returns
        -------
        dict
            ``{"k_values": [...], "unique_token_ratio": [...],
            "distinct_sequences": [...], "mean_pairwise_distance": [...]}``
        """
        unique_ratios: List[float] = []
        distinct_counts: List[int] = []
        mean_distances: List[float] = []

        for k in k_values:
            cfg = TopKConfig(
                k=k,
                temperature=temperature,
                num_sequences=n_samples,
                max_new_tokens=max_new_tokens,
                seed=int(self._rng.integers(0, 2**31)),
            )
            algo = TopKSampling(cfg)
            sequences = algo.generate(logit_source, prompt_ids)

            # Unique token ratio
            all_tokens = [t for seq in sequences for t in seq]
            if len(all_tokens) > 0:
                unique_ratios.append(len(set(all_tokens)) / len(all_tokens))
            else:
                unique_ratios.append(0.0)

            # Distinct sequences
            distinct = len(set(tuple(s) for s in sequences))
            distinct_counts.append(distinct)

            # Mean pairwise edit distance (sampled for efficiency)
            dist = self._mean_pairwise_distance(sequences, max_pairs=200)
            mean_distances.append(dist)

        return {
            "k_values": list(k_values),
            "unique_token_ratio": unique_ratios,
            "distinct_sequences": distinct_counts,
            "mean_pairwise_distance": mean_distances,
        }

    # -- optimal k ----------------------------------------------------------

    def optimal_k(
        self,
        logits: np.ndarray,
        target_coverage: float = 0.95,
    ) -> int:
        """Find the smallest k that captures at least *target_coverage* probability mass.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        target_coverage : float
            Target cumulative probability (e.g. 0.95).

        Returns
        -------
        int
            Smallest k achieving the target coverage.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        probs = _stable_softmax(logits)
        sorted_probs = np.sort(probs)[::-1]
        cumulative = np.cumsum(sorted_probs)

        indices = np.where(cumulative >= target_coverage)[0]
        if len(indices) == 0:
            return int(logits.shape[0])
        return int(indices[0]) + 1

    # -- entropy vs k -------------------------------------------------------

    def k_vs_entropy(
        self,
        logits: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, Any]:
        """Compute the entropy of the top-k truncated distribution for various k.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k_values : list of int

        Returns
        -------
        dict
            ``{"k_values": [...], "entropy": [...], "max_entropy": [...]}``
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        entropies: List[float] = []
        max_entropies: List[float] = []

        for k in k_values:
            filtered = top_k_filter(logits, k)
            probs = _stable_softmax(filtered)
            probs = np.clip(probs, 1e-30, None)
            # Only count tokens that are not masked
            active_mask = probs > 1e-20
            active_probs = probs[active_mask]
            if len(active_probs) == 0:
                entropies.append(0.0)
                max_entropies.append(0.0)
            else:
                active_probs = active_probs / active_probs.sum()
                ent = -float(np.sum(active_probs * np.log(active_probs)))
                entropies.append(ent)
                max_entropies.append(float(np.log(len(active_probs))))

        return {
            "k_values": list(k_values),
            "entropy": entropies,
            "max_entropy": max_entropies,
        }

    # -- truncation analysis ------------------------------------------------

    def truncation_analysis(
        self,
        logits: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, Any]:
        """Analyse how much probability mass is discarded for different k.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k_values : list of int

        Returns
        -------
        dict
            ``{"k_values": [...], "mass_kept": [...], "mass_discarded": [...],
            "num_tokens_kept": [...], "kl_divergence": [...]}``
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        probs = _stable_softmax(logits)
        sorted_probs = np.sort(probs)[::-1]
        cumulative = np.cumsum(sorted_probs)

        mass_kept: List[float] = []
        mass_discarded: List[float] = []
        num_kept: List[int] = []
        kl_divs: List[float] = []

        for k in k_values:
            if k <= 0:
                mass_kept.append(0.0)
                mass_discarded.append(1.0)
                num_kept.append(0)
                kl_divs.append(float("inf"))
                continue
            if k >= len(probs):
                mass_kept.append(1.0)
                mass_discarded.append(0.0)
                num_kept.append(int(len(probs)))
                kl_divs.append(0.0)
                continue

            mk = float(cumulative[k - 1])
            mass_kept.append(mk)
            mass_discarded.append(1.0 - mk)
            num_kept.append(k)

            # KL divergence between original and truncated
            filtered = top_k_filter(logits, k)
            q = _stable_softmax(filtered)
            q = np.clip(q, 1e-30, None)
            p_clipped = np.clip(probs, 1e-30, None)
            # KL(p || q) = sum p * log(p/q) for tokens where p > 0
            active = probs > 1e-30
            kl = float(np.sum(p_clipped[active] * np.log(p_clipped[active] / q[active])))
            kl_divs.append(kl)

        return {
            "k_values": list(k_values),
            "mass_kept": mass_kept,
            "mass_discarded": mass_discarded,
            "num_tokens_kept": num_kept,
            "kl_divergence": kl_divs,
        }

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _mean_pairwise_distance(
        sequences: List[TokenSequence],
        max_pairs: int = 200,
    ) -> float:
        """Compute mean pairwise normalised edit distance over a sample of pairs.

        For efficiency, only *max_pairs* randomly chosen pairs are compared.
        """
        n = len(sequences)
        if n < 2:
            return 0.0

        rng = np.random.default_rng()
        total_dist = 0.0
        count = 0
        pair_limit = min(max_pairs, n * (n - 1) // 2)

        # Sample pairs
        seen_pairs: set = set()
        attempts = 0
        while count < pair_limit and attempts < pair_limit * 4:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                attempts += 1
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                attempts += 1
                continue
            seen_pairs.add(pair_key)

            d = TopKAnalyzer._edit_distance(sequences[i], sequences[j])
            max_len = max(len(sequences[i]), len(sequences[j]), 1)
            total_dist += d / max_len
            count += 1
            attempts += 1

        return total_dist / max(count, 1)

    @staticmethod
    def _edit_distance(a: List[int], b: List[int]) -> int:
        """Levenshtein edit distance between two token sequences."""
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        # Use two-row DP for memory efficiency
        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)
        for i in range(1, la + 1):
            curr[0] = i
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost,  # substitution
                )
            prev, curr = curr, prev
        return prev[lb]


# =========================================================================
# TopKWithTemperature
# =========================================================================


class TopKWithTemperature:
    """Combined top-k + temperature sampling with joint analysis.

    This class provides a convenient interface for generating sequences
    with both top-k filtering and temperature scaling, and for analysing
    their joint effect on diversity and quality.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        k: int = 50,
        temperature: float = 1.0,
        num_sequences: int = 20,
        max_new_tokens: int = 100,
        repetition_penalty: float = 1.0,
    ) -> List[TokenSequence]:
        """Generate sequences using combined top-k + temperature sampling.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        k : int
        temperature : float
        num_sequences : int
        max_new_tokens : int
        repetition_penalty : float

        Returns
        -------
        list of TokenSequence
        """
        cfg = TopKConfig(
            k=k,
            temperature=temperature,
            num_sequences=num_sequences,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=int(self._rng.integers(0, 2**31)),
        )
        algo = TopKSampling(cfg)
        return algo.generate(logit_source, prompt_ids)

    # -- filter + scale -----------------------------------------------------

    def _filter_and_scale(
        self,
        logits: np.ndarray,
        k: int,
        temperature: float,
    ) -> np.ndarray:
        """Apply top-k filtering then temperature scaling.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k : int
        temperature : float

        Returns
        -------
        np.ndarray
            Filtered and scaled logits.
        """
        filtered = top_k_filter(logits, k)
        if temperature > 0:
            filtered = filtered / temperature
        return filtered

    # -- joint analysis -----------------------------------------------------

    def analyze_joint_effect(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        k_values: List[int],
        temperatures: List[float],
        n_samples: int = 30,
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """Analyse the joint effect of k and temperature on diversity.

        Generates sequences for every ``(k, temperature)`` combination and
        computes diversity metrics.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        k_values : list of int
        temperatures : list of float
        n_samples : int
        max_new_tokens : int

        Returns
        -------
        dict
            ``{"k_values": [...], "temperatures": [...], "grid": {(k,T): metrics}}``
        """
        grid: Dict[str, Dict[str, Any]] = {}

        logits_sample = logit_source([prompt_ids])[0]

        for k in k_values:
            for temp in temperatures:
                key = f"k={k},T={temp}"
                logger.debug("Analysing %s", key)

                # Coverage at this (k, T) — temperature doesn't affect coverage
                coverage = probability_mass_in_top_k(logits_sample, k)

                # Entropy of filtered + scaled distribution
                filtered_scaled = self._filter_and_scale(logits_sample, k, temp)
                probs = _stable_softmax(filtered_scaled)
                probs = np.clip(probs, 1e-30, None)
                active = probs > 1e-20
                active_probs = probs[active]
                if len(active_probs) > 0:
                    active_probs = active_probs / active_probs.sum()
                    entropy = -float(np.sum(active_probs * np.log(active_probs)))
                else:
                    entropy = 0.0

                # Generate samples and measure diversity
                cfg = TopKConfig(
                    k=k,
                    temperature=temp,
                    num_sequences=n_samples,
                    max_new_tokens=max_new_tokens,
                    seed=int(self._rng.integers(0, 2**31)),
                )
                algo = TopKSampling(cfg)
                sequences = algo.generate(logit_source, prompt_ids)

                all_tokens = [t for s in sequences for t in s]
                if len(all_tokens) > 0:
                    type_token_ratio = len(set(all_tokens)) / len(all_tokens)
                else:
                    type_token_ratio = 0.0

                distinct = len(set(tuple(s) for s in sequences))

                grid[key] = {
                    "k": k,
                    "temperature": temp,
                    "coverage": coverage,
                    "entropy": entropy,
                    "type_token_ratio": type_token_ratio,
                    "distinct_sequences": distinct,
                    "n_samples": n_samples,
                }

        return {
            "k_values": list(k_values),
            "temperatures": list(temperatures),
            "grid": grid,
        }

    def recommend_params(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        target_diversity: float = 0.5,
        target_coverage: float = 0.9,
    ) -> Dict[str, Any]:
        """Recommend k and temperature based on target diversity and coverage.

        Uses a simple heuristic: find the smallest k that achieves
        *target_coverage*, then pick a temperature that yields approximately
        *target_diversity* type-token ratio.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        target_diversity : float
            Desired type-token ratio (0–1).
        target_coverage : float
            Desired probability mass coverage.

        Returns
        -------
        dict
            ``{"k": int, "temperature": float, "expected_coverage": float}``
        """
        logits = logit_source([prompt_ids])[0]

        analyzer = TopKAnalyzer(seed=self._seed)
        k = analyzer.optimal_k(logits, target_coverage)

        # Heuristic for temperature: entropy-based
        eff_vocab = effective_vocabulary_size(logits)
        if eff_vocab <= 1:
            temperature = 1.0
        else:
            # Scale temperature so that diversity ≈ target_diversity
            # Higher diversity → higher temperature
            temperature = 0.5 + target_diversity * 1.5

        actual_coverage = probability_mass_in_top_k(logits, k)

        return {
            "k": k,
            "temperature": round(temperature, 3),
            "expected_coverage": round(actual_coverage, 4),
            "effective_vocab_size": round(eff_vocab, 1),
        }


# =========================================================================
# Module self-test
# =========================================================================


def _self_test() -> None:
    """Minimal smoke test for the top-k module."""

    # -- helper functions ---------------------------------------------------
    logits = np.array([1.0, 4.0, 2.0, 5.0, 3.0])

    filtered = top_k_filter(logits, 3)
    assert np.sum(filtered > -np.inf) == 3, f"Expected 3 active, got {np.sum(filtered > -np.inf)}"

    indices = top_k_indices(logits, 3)
    assert list(indices) == [3, 1, 4], f"Expected [3,1,4], got {list(indices)}"

    mass = probability_mass_in_top_k(logits, 3)
    assert 0.0 < mass <= 1.0, f"Mass {mass} out of range"

    eff = effective_vocabulary_size(logits)
    assert 1.0 <= eff <= 5.0, f"Effective vocab {eff} out of range"

    batch = np.array([[1.0, 4.0, 2.0, 5.0, 3.0], [5.0, 1.0, 2.0, 3.0, 4.0]])
    batch_filtered = batch_top_k(batch, 2)
    assert batch_filtered.shape == (2, 5)
    for row in batch_filtered:
        assert np.sum(row > -np.inf) == 2

    # -- TopKConfig ---------------------------------------------------------
    cfg = TopKConfig(k=10, temperature=0.8)
    assert cfg.validate() == []
    bad_cfg = TopKConfig(k=0)
    assert len(bad_cfg.validate()) > 0

    # -- TopKSchedule -------------------------------------------------------
    sched = TopKSchedule("linear", k_start=100, k_end=10, total_steps=10)
    assert sched.get_k(0) == 100
    assert sched.get_k(9) == 10
    k_mid = sched.get_k(5)
    assert 10 <= k_mid <= 100

    sched_const = TopKSchedule("constant", k_start=50)
    assert sched_const.get_k(0) == 50
    assert sched_const.get_k(999) == 50

    sched_ent = TopKSchedule("entropy_based", k_min=5, k_max=200)
    uniform_logits = np.ones(1000)
    k_high = sched_ent.get_k(0, uniform_logits)
    peaked_logits = np.zeros(1000)
    peaked_logits[0] = 100.0
    k_low = sched_ent.get_k(0, peaked_logits)
    assert k_high > k_low, f"Uniform k={k_high} should be > peaked k={k_low}"

    sched_decay = TopKSchedule("step_decay", k_start=100, decay_rate=0.5, decay_every=5, k_min=1)
    assert sched_decay.get_k(0) == 100
    assert sched_decay.get_k(5) == 50
    assert sched_decay.get_k(10) == 25

    # -- TopKSampling with dummy logit source --------------------------------
    vocab_size = 100
    dummy_source: LogitSource = lambda ids: np.random.randn(len(ids), vocab_size)

    algo_cfg = TopKConfig(k=10, num_sequences=3, max_new_tokens=5, seed=42)
    algo = TopKSampling(algo_cfg)
    results = algo.generate(dummy_source, [1, 2, 3])
    assert len(results) == 3
    assert all(len(s) > 0 for s in results)

    # Dynamic k
    dyn_cfg = TopKConfig(
        k=50, dynamic_k=True, k_schedule="linear",
        k_min=5, k_max=50, num_sequences=2, max_new_tokens=5, seed=42,
    )
    dyn_algo = TopKSampling(dyn_cfg)
    dyn_results = dyn_algo.generate(dummy_source, [1, 2])
    assert len(dyn_results) == 2

    # Without replacement
    norep_cfg = TopKConfig(
        k=20, with_replacement=False, num_sequences=3,
        max_new_tokens=5, seed=42,
    )
    norep_algo = TopKSampling(norep_cfg)
    norep_results = norep_algo.generate(dummy_source, [1])
    assert len(norep_results) <= 3

    # -- TopKAnalyzer -------------------------------------------------------
    analyzer = TopKAnalyzer(seed=42)

    test_logits = np.random.randn(vocab_size)
    cov = analyzer.coverage_vs_k(test_logits, [1, 10, 50, 100])
    assert len(cov["coverage"]) == 4
    assert cov["coverage"][0] <= cov["coverage"][-1]

    opt_k = analyzer.optimal_k(test_logits, 0.95)
    assert 1 <= opt_k <= vocab_size

    ent = analyzer.k_vs_entropy(test_logits, [1, 10, 50])
    assert len(ent["entropy"]) == 3

    trunc = analyzer.truncation_analysis(test_logits, [1, 10, 50])
    assert len(trunc["mass_kept"]) == 3
    assert trunc["mass_discarded"][0] >= trunc["mass_discarded"][-1]

    # -- TopKWithTemperature ------------------------------------------------
    combo = TopKWithTemperature(seed=42)
    combo_results = combo.generate(
        dummy_source, [1, 2], k=20, temperature=0.8,
        num_sequences=3, max_new_tokens=5,
    )
    assert len(combo_results) == 3

    fs = combo._filter_and_scale(test_logits, 10, 0.5)
    assert fs.shape == test_logits.shape

    rec = combo.recommend_params(dummy_source, [1, 2])
    assert "k" in rec
    assert "temperature" in rec

    # -- Registry check -----------------------------------------------------
    assert AlgorithmRegistry.is_registered("top_k")

    print("topk.py self-test passed ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
