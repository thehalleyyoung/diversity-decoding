"""
Typical Decoding for the Diversity Decoding Arena.
===================================================

Implements the typical sampling algorithm (Meister et al., 2023) along with
adaptive extensions and analytical utilities.  Typical decoding restricts
the token vocabulary at each step to a *typical set* — the set of tokens
whose information content (negative log-probability) is closest to the
expected information content (entropy) of the distribution.  Tokens are
ranked by their absolute deviation from the entropy, and the smallest set
whose cumulative probability mass exceeds the threshold ``typical_p`` is
retained.

This approach is grounded in information theory: the typical set theorem
states that for i.i.d. random variables, the probability mass concentrates
on sequences whose per-symbol information content is close to the entropy.
Typical decoding applies this principle at the token level, preferring
tokens that are neither surprisingly rare nor surprisingly common.

Key components
--------------
* **TypicalConfig** — dataclass holding all hyper-parameters (typical_p,
  temperature, adaptive settings, combination with nucleus, …).
* **TypicalDecoding** — the main ``DecodingAlgorithm`` implementation with
  step-level typical filtering, tracking, and generation.
* **TypicalAnalyzer** — post-hoc analysis utilities (typicality distributions,
  entropy trajectories, typical set sizes, comparisons with nucleus).
* **TypicalSet** — data class representing the typical set at a single
  decoding step with query utilities.
* **Helper functions** — stand-alone utilities for typical filtering, entropy
  computation, information content, batched operations.

Numerical stability
-------------------
All probability computations go through the log-softmax → exp path to avoid
overflow/underflow.  ``scipy.special.log_softmax`` and ``logsumexp`` are used
throughout.

References
----------
- Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2023).
  *Locally Typical Sampling*.  TACL.
- Cover, T. M., & Thomas, J. A. (2006).
  *Elements of Information Theory*.  Wiley.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import log_softmax, logsumexp, softmax

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    AlgorithmRegistry,
    LogitSource,
    TokenSequence,
    register,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEG_INF: float = float("-inf")
_LOG_EPS: float = 1e-10  # small constant to avoid log(0)


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class TypicalConfig(DecodingConfig):
    """Configuration for typical decoding.

    Attributes
    ----------
    typical_p : float
        Cumulative probability mass threshold for the typical set.  Only the
        smallest set of tokens (ranked by typicality) whose cumulative
        probability exceeds ``typical_p`` is kept.  Values in (0, 1].
        A value of 1.0 effectively disables typical filtering.
    temperature : float
        Softmax temperature applied *before* typical filtering.  Values > 1
        increase entropy (more diverse), values < 1 decrease entropy (more
        peaked).
    min_tokens_to_keep : int
        Minimum number of tokens that must remain after filtering, regardless
        of the cumulative probability.  Prevents degenerate empty typical sets.
    entropy_weight : float
        Scaling weight for the entropy term when computing typicality scores.
        A weight of 1.0 corresponds to standard typical decoding.  Values
        < 1.0 bias toward higher-probability tokens; values > 1.0 bias toward
        tokens with information content closer to weighted entropy.
    adaptive : bool
        When ``True``, the typical mass threshold is adjusted dynamically at
        each step based on the local entropy of the distribution.
    local_entropy_window : int
        Number of recent steps to consider when computing local entropy
        statistics for adaptive mass adjustment.
    combine_with_nucleus : bool
        When ``True``, the typical set is intersected with the nucleus (top-p)
        set.  This can provide tighter control over the sampling distribution.
    nucleus_p : float
        Nucleus (top-p) threshold used when ``combine_with_nucleus`` is
        ``True``.
    repetition_penalty : float
        Multiplicative penalty applied to logits of tokens that have already
        been generated.  1.0 means no penalty.
    no_repeat_ngram_size : int
        If > 0, any n-gram of this size that has already occurred in the
        generated sequence is forbidden from occurring again.
    """

    typical_p: float = 0.95
    temperature: float = 1.0
    min_tokens_to_keep: int = 1
    entropy_weight: float = 1.0
    adaptive: bool = False
    local_entropy_window: int = 5
    combine_with_nucleus: bool = False
    nucleus_p: float = 0.9
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty == valid)."""
        errors = super().validate()
        if not 0.0 < self.typical_p <= 1.0:
            errors.append("typical_p must be in (0, 1]")
        if self.min_tokens_to_keep < 1:
            errors.append("min_tokens_to_keep must be >= 1")
        if self.entropy_weight < 0:
            errors.append("entropy_weight must be >= 0")
        if self.local_entropy_window < 1:
            errors.append("local_entropy_window must be >= 1")
        if self.combine_with_nucleus and not 0.0 < self.nucleus_p <= 1.0:
            errors.append("nucleus_p must be in (0, 1] when combine_with_nucleus is True")
        return errors


# =========================================================================
# Decoding state (extended for typical tracking)
# =========================================================================


@dataclass
class TypicalState(DecodingState):
    """Per-sequence decoding state augmented with typical-specific bookkeeping.

    Attributes
    ----------
    generated_ids : List[int]
        Token IDs generated so far.
    log_prob : float
        Cumulative log-probability of the generated sequence.
    entropies : List[float]
        Shannon entropy of the (pre-filtered) distribution at each step.
    typical_masses_used : List[float]
        The effective typical mass threshold used at each step (may differ
        from ``config.typical_p`` when adaptive mode is enabled).
    typical_set_sizes : List[int]
        Number of tokens remaining in the typical set at each step.
    typicality_scores_history : List[np.ndarray]
        Per-token typicality scores at each step (optional, for analysis).
    information_contents : List[float]
        Information content (-log p) of each sampled token.
    finished : bool
        Whether an EOS token has been sampled.
    """

    generated_ids: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    entropies: List[float] = field(default_factory=list)
    typical_masses_used: List[float] = field(default_factory=list)
    typical_set_sizes: List[int] = field(default_factory=list)
    typicality_scores_history: List[np.ndarray] = field(default_factory=list)
    information_contents: List[float] = field(default_factory=list)
    finished: bool = False


# =========================================================================
# TypicalSet — represents the typical set at a single decoding step
# =========================================================================


class TypicalSet:
    """Represents the typical set for a single decoding step.

    The typical set contains the tokens whose information content is closest
    to the entropy of the distribution, ordered by typicality (most typical
    first).

    Parameters
    ----------
    tokens : List[int]
        Token IDs in the typical set, ordered by typicality.
    log_probs : np.ndarray
        Log-probabilities of tokens in the typical set.
    typicality_scores : np.ndarray
        Absolute deviation from entropy for each token (lower = more typical).
    mass : float
        Total probability mass of the typical set.
    entropy : float
        Entropy of the full distribution from which this set was derived.
    """

    def __init__(
        self,
        tokens: List[int],
        log_probs: np.ndarray,
        typicality_scores: np.ndarray,
        mass: float,
        entropy: float,
    ) -> None:
        self.tokens = list(tokens)
        self.log_probs = np.asarray(log_probs, dtype=np.float64)
        self.typicality_scores = np.asarray(typicality_scores, dtype=np.float64)
        self.mass = float(mass)
        self.entropy = float(entropy)

    @property
    def size(self) -> int:
        """Number of tokens in the typical set."""
        return len(self.tokens)

    def most_typical(self, k: int = 1) -> List[int]:
        """Return the *k* most typical tokens (lowest deviation from entropy).

        Parameters
        ----------
        k : int
            Number of tokens to return.

        Returns
        -------
        List[int]
            Token IDs of the *k* most typical tokens.
        """
        k = min(k, self.size)
        if k <= 0:
            return []
        # Tokens are already sorted by typicality (ascending deviation)
        return self.tokens[:k]

    def least_typical(self, k: int = 1) -> List[int]:
        """Return the *k* least typical tokens (highest deviation from entropy).

        Parameters
        ----------
        k : int
            Number of tokens to return.

        Returns
        -------
        List[int]
            Token IDs of the *k* least typical tokens in the set.
        """
        k = min(k, self.size)
        if k <= 0:
            return []
        return self.tokens[-k:]

    def is_member(self, token_id: int) -> bool:
        """Check whether *token_id* is in the typical set.

        Parameters
        ----------
        token_id : int
            Token ID to check.

        Returns
        -------
        bool
            ``True`` if the token is in the typical set.
        """
        return token_id in self.tokens

    def coverage(self) -> float:
        """Return the probability mass coverage of the typical set.

        This is the sum of probabilities of all tokens in the set, i.e.
        the fraction of the total probability distribution that the typical
        set covers.

        Returns
        -------
        float
            Probability mass in [0, 1].
        """
        return self.mass

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the typical set to a dictionary."""
        return {
            "tokens": self.tokens,
            "log_probs": self.log_probs.tolist(),
            "typicality_scores": self.typicality_scores.tolist(),
            "mass": self.mass,
            "entropy": self.entropy,
            "size": self.size,
        }

    def __repr__(self) -> str:
        return (
            f"TypicalSet(size={self.size}, mass={self.mass:.4f}, "
            f"entropy={self.entropy:.4f})"
        )

    def __len__(self) -> int:
        return self.size

    def __contains__(self, token_id: int) -> bool:
        return self.is_member(token_id)


# =========================================================================
# Helper functions (module-level, stateless)
# =========================================================================


def compute_entropy(log_probs: np.ndarray) -> float:
    """Compute Shannon entropy from log-probabilities.

    Parameters
    ----------
    log_probs : np.ndarray
        Log-probability vector, shape ``(V,)``.  Should sum to ~0 in
        log-space (i.e. probabilities sum to ~1).

    Returns
    -------
    float
        Shannon entropy H = -sum(p * log(p)).  Non-negative.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    probs = np.exp(log_probs)
    # Mask out zero-probability entries to avoid nan from 0 * (-inf)
    mask = probs > 0
    if not np.any(mask):
        return 0.0
    entropy = -np.sum(probs[mask] * log_probs[mask])
    return max(0.0, float(entropy))


def information_content(log_probs: np.ndarray) -> np.ndarray:
    """Compute information content (surprisal) for each token.

    The information content of token *x* is ``-log p(x)``.

    Parameters
    ----------
    log_probs : np.ndarray
        Log-probability vector, shape ``(V,)``.

    Returns
    -------
    np.ndarray
        Information content vector, shape ``(V,)``.  Entries corresponding
        to zero-probability tokens are set to ``inf``.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    return -log_probs


def compute_typicality(
    logits: np.ndarray,
    entropy_weight: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Compute typicality scores and entropy from raw logits.

    The typicality score of token *x* is ``|IC(x) - w * H|`` where
    ``IC(x) = -log p(x)`` is the information content and ``H`` is the
    Shannon entropy of the distribution, weighted by ``entropy_weight``.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector, shape ``(V,)``.
    entropy_weight : float
        Weight applied to the entropy term.

    Returns
    -------
    scores : np.ndarray
        Typicality scores (absolute deviation from weighted entropy),
        shape ``(V,)``.  Lower values indicate more typical tokens.
    entropy : float
        Shannon entropy of the distribution.
    """
    logits = np.asarray(logits, dtype=np.float64)
    log_probs = log_softmax(logits)
    entropy = compute_entropy(log_probs)
    ic = information_content(log_probs)
    scores = np.abs(ic - entropy_weight * entropy)
    return scores, entropy


def typical_filter(
    logits: np.ndarray,
    mass: float = 0.95,
    min_keep: int = 1,
    filter_value: float = _NEG_INF,
    entropy_weight: float = 1.0,
) -> np.ndarray:
    """Apply typical filtering to a logit vector.

    Tokens are ranked by their typicality (absolute deviation of information
    content from the entropy).  The most typical tokens are kept until their
    cumulative probability mass exceeds ``mass``.  All other tokens are set
    to ``filter_value``.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector of shape ``(V,)``.
    mass : float
        Cumulative probability mass threshold in (0, 1].
    min_keep : int
        Minimum number of tokens to retain.
    filter_value : float
        Value assigned to masked positions (default ``-inf``).
    entropy_weight : float
        Weight applied to the entropy term in typicality computation.

    Returns
    -------
    np.ndarray
        Filtered logit vector of the same shape.  Tokens outside the typical
        set have value ``filter_value``; tokens inside retain their original
        logit values.
    """
    if mass >= 1.0:
        return logits.copy()

    logits = np.asarray(logits, dtype=np.float64)
    V = logits.shape[0]
    min_keep = max(1, min(min_keep, V))

    # Compute log-probabilities
    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)

    # Compute entropy
    entropy = compute_entropy(log_probs)

    # Compute information content and typicality scores
    ic = information_content(log_probs)
    typicality_scores = np.abs(ic - entropy_weight * entropy)

    # Sort by typicality (ascending — most typical first)
    sorted_indices = np.argsort(typicality_scores)

    # Accumulate probability mass of sorted tokens
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    # Determine cutoff: keep tokens until cumulative mass >= threshold
    # We shift cumulative probs so the token that crosses the threshold is kept
    cumulative_shifted = np.concatenate([[0.0], cumulative_probs[:-1]])
    sorted_mask = cumulative_shifted >= mass

    # Always keep at least min_keep tokens
    sorted_mask[:min_keep] = False

    # Build mask in original vocabulary order
    filtered = logits.copy()
    remove_indices = sorted_indices[sorted_mask]
    filtered[remove_indices] = filter_value

    return filtered


def batch_typical_filter(
    logits_batch: np.ndarray,
    mass: float = 0.95,
    min_keep: int = 1,
    filter_value: float = _NEG_INF,
    entropy_weight: float = 1.0,
) -> np.ndarray:
    """Apply typical filtering to a batch of logit vectors.

    Parameters
    ----------
    logits_batch : np.ndarray
        Batch of raw logit vectors, shape ``(B, V)``.
    mass : float
        Cumulative probability mass threshold.
    min_keep : int
        Minimum tokens to retain per sample.
    filter_value : float
        Value for masked positions.
    entropy_weight : float
        Weight for entropy term.

    Returns
    -------
    np.ndarray
        Filtered logit batch, shape ``(B, V)``.
    """
    logits_batch = np.asarray(logits_batch, dtype=np.float64)
    if logits_batch.ndim == 1:
        return typical_filter(logits_batch, mass, min_keep, filter_value, entropy_weight)

    B = logits_batch.shape[0]
    result = np.empty_like(logits_batch)
    for i in range(B):
        result[i] = typical_filter(
            logits_batch[i], mass, min_keep, filter_value, entropy_weight
        )
    return result


def conditional_entropy_estimate(log_probs_history: List[np.ndarray]) -> float:
    """Estimate conditional entropy from a history of log-probability vectors.

    Computes the average per-step entropy over the generation history.  This
    provides a simple estimate of the conditional entropy H(X_t | X_{<t})
    averaged over time.

    Parameters
    ----------
    log_probs_history : List[np.ndarray]
        List of log-probability vectors, one per generation step.

    Returns
    -------
    float
        Estimated conditional entropy.  Returns 0.0 for empty history.
    """
    if not log_probs_history:
        return 0.0

    entropies = []
    for log_probs in log_probs_history:
        log_probs = np.asarray(log_probs, dtype=np.float64)
        entropies.append(compute_entropy(log_probs))

    return float(np.mean(entropies))


def _nucleus_filter(
    logits: np.ndarray,
    p: float,
    min_keep: int = 1,
    filter_value: float = _NEG_INF,
) -> np.ndarray:
    """Apply nucleus (top-p) filtering to a logit vector.

    Kept here as a private helper for the combined typical + nucleus mode.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector of shape ``(V,)``.
    p : float
        Cumulative probability threshold.
    min_keep : int
        Minimum number of tokens to retain.
    filter_value : float
        Value for masked positions.

    Returns
    -------
    np.ndarray
        Filtered logit vector.
    """
    if p >= 1.0:
        return logits.copy()

    logits = np.asarray(logits, dtype=np.float64)
    V = logits.shape[0]
    min_keep = max(1, min(min_keep, V))

    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    log_probs_sorted = log_softmax(sorted_logits)
    probs_sorted = np.exp(log_probs_sorted)
    cumulative_probs = np.cumsum(probs_sorted)

    # Shift so the token crossing the threshold is kept
    cumulative_shifted = np.concatenate([[0.0], cumulative_probs[:-1]])
    sorted_mask = cumulative_shifted >= p
    sorted_mask[:min_keep] = False

    filtered = logits.copy()
    remove_indices = sorted_indices[sorted_mask]
    filtered[remove_indices] = filter_value

    return filtered


def _sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample a single token from a logit vector with temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Logit vector, shape ``(V,)``.  May contain ``-inf`` for masked tokens.
    temperature : float
        Temperature divisor.  Must be > 0.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    int
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64)

    if temperature <= 1e-8:
        # Greedy: return argmax
        return int(np.argmax(logits))

    scaled = logits / temperature
    log_probs = log_softmax(scaled)
    probs = np.exp(log_probs)

    # Handle numerical issues: ensure non-negative and sums to 1
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0:
        # Fallback: uniform over non-masked tokens
        finite_mask = np.isfinite(logits)
        if not np.any(finite_mask):
            return 0
        probs = finite_mask.astype(np.float64)
        probs /= probs.sum()
    else:
        probs /= total

    if rng is not None:
        return int(rng.choice(len(probs), p=probs))
    else:
        return int(np.random.choice(len(probs), p=probs))


def _build_typical_set(
    logits: np.ndarray,
    mass: float = 0.95,
    min_keep: int = 1,
    entropy_weight: float = 1.0,
) -> TypicalSet:
    """Construct a :class:`TypicalSet` from a logit vector.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector, shape ``(V,)``.
    mass : float
        Cumulative probability mass threshold.
    min_keep : int
        Minimum number of tokens to retain.
    entropy_weight : float
        Weight for the entropy term.

    Returns
    -------
    TypicalSet
        The typical set object.
    """
    logits = np.asarray(logits, dtype=np.float64)
    V = logits.shape[0]
    min_keep = max(1, min(min_keep, V))

    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)
    entropy = compute_entropy(log_probs)

    ic = information_content(log_probs)
    typicality_scores = np.abs(ic - entropy_weight * entropy)

    # Sort by typicality (ascending — most typical first)
    sorted_indices = np.argsort(typicality_scores)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    # Determine how many to keep
    if mass >= 1.0:
        n_keep = V
    else:
        cumulative_shifted = np.concatenate([[0.0], cumulative_probs[:-1]])
        cutoff_mask = cumulative_shifted >= mass
        # Find first position where mask is True
        cutoff_positions = np.where(cutoff_mask)[0]
        if len(cutoff_positions) > 0:
            n_keep = max(min_keep, cutoff_positions[0])
        else:
            n_keep = V

    n_keep = max(min_keep, n_keep)
    kept_indices = sorted_indices[:n_keep]

    total_mass = float(np.sum(probs[kept_indices]))

    return TypicalSet(
        tokens=kept_indices.tolist(),
        log_probs=log_probs[kept_indices],
        typicality_scores=typicality_scores[kept_indices],
        mass=total_mass,
        entropy=entropy,
    )


# =========================================================================
# TypicalDecoding — main algorithm
# =========================================================================


@register("typical")
class TypicalDecoding(DecodingAlgorithm):
    """Typical Decoding algorithm (Meister et al., 2023).

    At each generation step, the distribution over next tokens is filtered
    to the *typical set* — the smallest set of tokens whose information
    content is closest to the distribution's entropy and whose cumulative
    mass exceeds the configured threshold.  This produces text that avoids
    both the overly-generic outputs of high-probability-only methods and the
    incoherent outputs of unrestricted sampling.

    Parameters
    ----------
    config : TypicalConfig
        Algorithm configuration.  If a plain :class:`DecodingConfig` is
        provided, typical-specific fields default to their dataclass defaults.
    """

    def __init__(self, config: DecodingConfig) -> None:
        if not isinstance(config, TypicalConfig):
            # Promote a generic config to TypicalConfig
            cfg_dict = config.to_dict()
            config = TypicalConfig.from_dict(cfg_dict)
        super().__init__(config)
        self._typical_config: TypicalConfig = config  # type: ignore[assignment]

    # -- properties ---------------------------------------------------------

    @property
    def description(self) -> str:
        return (
            f"Typical decoding (p={self._typical_config.typical_p}, "
            f"T={self._typical_config.temperature})"
        )

    # -- state initialization -----------------------------------------------

    def _init_state(self, prompt_ids: List[int]) -> DecodingState:
        """Initialize decoding state with typical-specific bookkeeping."""
        num_seq = self.config.num_sequences
        state = DecodingState(
            sequences=[list(prompt_ids) for _ in range(num_seq)],
            scores=[0.0] * num_seq,
            is_finished=[False] * num_seq,
            step=0,
            metadata={
                "prompt_length": len(prompt_ids),
                "per_sequence": [
                    TypicalState(
                        generated_ids=[],
                        log_prob=0.0,
                        entropies=[],
                        typical_masses_used=[],
                        typical_set_sizes=[],
                        typicality_scores_history=[],
                        information_contents=[],
                        finished=False,
                    )
                    for _ in range(num_seq)
                ],
            },
        )
        return state

    # -- core step ----------------------------------------------------------

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute a single typical decoding step across all active sequences.

        For each active sequence:
        1. Obtain logits from the logit source.
        2. Apply constraints (repetition penalty, n-gram blocking, min length).
        3. Apply typical filtering (and optionally nucleus intersection).
        4. Sample a token from the filtered distribution.
        5. Update state bookkeeping.

        Parameters
        ----------
        state : DecodingState
            Current generation state.
        logit_source : LogitSource
            Callable returning logits.

        Returns
        -------
        DecodingState
            Updated state.
        """
        active = state.active_indices()
        if not active:
            return state

        # Gather input sequences for active indices
        input_ids = [state.sequences[i] for i in active]

        # Get logits from source — shape (batch, vocab_size)
        try:
            all_logits = logit_source(input_ids)
        except Exception as e:
            logger.error("logit_source failed: %s", e)
            for i in active:
                state.mark_finished(i)
            return state

        all_logits = np.asarray(all_logits, dtype=np.float64)
        if all_logits.ndim == 1:
            all_logits = all_logits[np.newaxis, :]

        per_seq: List[TypicalState] = state.metadata["per_sequence"]

        for batch_idx, seq_idx in enumerate(active):
            logits = all_logits[batch_idx].copy()

            # Apply constraints from base class
            logits = self._apply_constraints(logits, state)

            # Determine typical mass (possibly adaptive)
            if self._typical_config.adaptive:
                mass = self._adaptive_mass(logits, per_seq[seq_idx])
            else:
                mass = self._typical_config.typical_p

            # Apply typical filtering
            if self._typical_config.combine_with_nucleus:
                filtered_logits = self._combined_typical_nucleus(
                    logits, mass, self._typical_config.nucleus_p
                )
            else:
                filtered_logits = self._typical_filter(logits, mass)

            # Compute pre-filter entropy for tracking
            log_probs_raw = log_softmax(logits)
            step_entropy = compute_entropy(log_probs_raw)

            # Compute typical set size (number of non-masked tokens)
            finite_mask = np.isfinite(filtered_logits) & (filtered_logits > _NEG_INF + 1)
            typical_set_size = int(np.sum(finite_mask))

            # Sample token
            token = _sample_from_logits(
                filtered_logits,
                temperature=self._typical_config.temperature,
                rng=self._rng,
            )

            # Compute log-probability and information content of sampled token
            log_probs_full = log_softmax(logits)
            token_log_prob = float(log_probs_full[token])
            token_ic = -token_log_prob

            # Update sequence
            state.update_sequence(seq_idx, token)
            state.scores[seq_idx] += token_log_prob

            # Update per-sequence tracking
            ts = per_seq[seq_idx]
            ts.generated_ids.append(token)
            ts.log_prob += token_log_prob
            ts.entropies.append(step_entropy)
            ts.typical_masses_used.append(mass)
            ts.typical_set_sizes.append(typical_set_size)
            ts.information_contents.append(token_ic)

            # Check for EOS
            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)
                ts.finished = True

            # Check max length
            prompt_len = state.metadata.get("prompt_length", 0)
            gen_len = len(state.sequences[seq_idx]) - prompt_len
            if gen_len >= self.config.max_new_tokens:
                state.mark_finished(seq_idx)
                ts.finished = True

        state.step += 1
        return state

    # -- typical filtering core ---------------------------------------------

    def _typical_filter(self, logits: np.ndarray, mass: float) -> np.ndarray:
        """Apply typical filtering to a single logit vector.

        Implements the core typical decoding algorithm:
        1. Compute log-probabilities via log-softmax.
        2. Compute entropy H of the distribution.
        3. Compute |log p(x) - (-H)| = |IC(x) - H| for each token.
        4. Sort tokens by this deviation (ascending — most typical first).
        5. Accumulate probability mass of sorted tokens.
        6. Keep tokens until accumulated mass >= typical_p.
        7. Mask the rest to -inf.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.
        mass : float
            Typical mass threshold.

        Returns
        -------
        np.ndarray
            Filtered logit vector.
        """
        return typical_filter(
            logits,
            mass=mass,
            min_keep=self._typical_config.min_tokens_to_keep,
            entropy_weight=self._typical_config.entropy_weight,
        )

    def _compute_entropy(self, log_probs: np.ndarray) -> float:
        """Compute Shannon entropy from log-probabilities.

        Parameters
        ----------
        log_probs : np.ndarray
            Log-probability vector, shape ``(V,)``.

        Returns
        -------
        float
            Shannon entropy H = -sum(p * log(p)).
        """
        return compute_entropy(log_probs)

    def _compute_information_content(self, log_probs: np.ndarray) -> np.ndarray:
        """Compute information content (surprisal) for each token.

        Parameters
        ----------
        log_probs : np.ndarray
            Log-probability vector, shape ``(V,)``.

        Returns
        -------
        np.ndarray
            Information content vector: -log p(x) for each token.
        """
        return information_content(log_probs)

    def _compute_typicality_scores(
        self, log_probs: np.ndarray, entropy: float
    ) -> np.ndarray:
        """Compute typicality scores for each token.

        The typicality score is |IC(x) - w * H| where IC(x) = -log p(x)
        is the information content, H is the entropy, and w is the
        entropy weight.  Lower scores indicate more typical tokens.

        Parameters
        ----------
        log_probs : np.ndarray
            Log-probability vector, shape ``(V,)``.
        entropy : float
            Shannon entropy of the distribution.

        Returns
        -------
        np.ndarray
            Typicality scores, shape ``(V,)``.
        """
        ic = self._compute_information_content(log_probs)
        weighted_entropy = self._typical_config.entropy_weight * entropy
        return np.abs(ic - weighted_entropy)

    def _adaptive_mass(self, logits: np.ndarray, ts: TypicalState) -> float:
        """Compute an adaptive typical mass threshold based on local entropy.

        When the distribution is highly uncertain (high entropy), we use a
        tighter mass threshold to avoid generating overly diverse / incoherent
        tokens.  When the distribution is confident (low entropy), we relax
        the threshold to allow more diversity.

        The adaptive mass is computed as:

            mass = typical_p * sigmoid(-(H_local - H_baseline) / scale)

        where ``H_local`` is the average entropy over the recent window,
        ``H_baseline`` is the overall average, and ``scale`` controls
        sensitivity.

        Parameters
        ----------
        logits : np.ndarray
            Current step's raw logit vector.
        ts : TypicalState
            Per-sequence tracking state (contains entropy history).

        Returns
        -------
        float
            Adjusted typical mass threshold in (0, 1].
        """
        base_mass = self._typical_config.typical_p

        # Compute current entropy
        log_probs = log_softmax(logits)
        current_entropy = compute_entropy(log_probs)

        # If not enough history, return base mass
        if len(ts.entropies) < 2:
            return base_mass

        # Local entropy (recent window)
        window = self._typical_config.local_entropy_window
        recent_entropies = ts.entropies[-window:]
        local_entropy = float(np.mean(recent_entropies))

        # Baseline entropy (full history)
        baseline_entropy = float(np.mean(ts.entropies))

        # If baseline is near zero, avoid division issues
        if baseline_entropy < 1e-6:
            return base_mass

        # Compute ratio: high local entropy relative to baseline → tighter mass
        ratio = local_entropy / baseline_entropy

        # Sigmoid-based adjustment
        # ratio > 1 → high entropy → reduce mass (tighter)
        # ratio < 1 → low entropy → increase mass (looser)
        scale = 2.0
        adjustment = 1.0 / (1.0 + np.exp(scale * (ratio - 1.0)))

        # Map adjustment to [0.5 * base_mass, base_mass]
        adaptive = base_mass * (0.5 + 0.5 * adjustment)
        adaptive = float(np.clip(adaptive, 0.1, 1.0))

        logger.debug(
            "Adaptive mass: local_H=%.3f, baseline_H=%.3f, ratio=%.3f, "
            "mass=%.3f (base=%.3f)",
            local_entropy, baseline_entropy, ratio, adaptive, base_mass,
        )

        return adaptive

    def _combined_typical_nucleus(
        self,
        logits: np.ndarray,
        typical_p: float,
        nucleus_p: float,
    ) -> np.ndarray:
        """Apply both typical and nucleus filtering, returning their intersection.

        Tokens must pass *both* filters to be retained.  This produces a
        subset that is both probable (nucleus) and information-theoretically
        typical.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.
        typical_p : float
            Typical mass threshold.
        nucleus_p : float
            Nucleus (top-p) threshold.

        Returns
        -------
        np.ndarray
            Filtered logit vector where only tokens in the intersection
            of both sets retain their original values.
        """
        logits = np.asarray(logits, dtype=np.float64)

        # Apply typical filter
        typical_filtered = self._typical_filter(logits, typical_p)

        # Apply nucleus filter
        nucleus_filtered = _nucleus_filter(
            logits, nucleus_p, min_keep=self._typical_config.min_tokens_to_keep
        )

        # Intersection: a token is kept only if it passes both filters
        typical_mask = np.isfinite(typical_filtered) & (typical_filtered > _NEG_INF + 1)
        nucleus_mask = np.isfinite(nucleus_filtered) & (nucleus_filtered > _NEG_INF + 1)
        combined_mask = typical_mask & nucleus_mask

        # Ensure at least min_tokens_to_keep
        if np.sum(combined_mask) < self._typical_config.min_tokens_to_keep:
            # Fall back to typical-only if intersection is too small
            return typical_filtered

        result = np.full_like(logits, _NEG_INF)
        result[combined_mask] = logits[combined_mask]

        return result

    # -- introspection ------------------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        """Describe the hyper-parameter search space."""
        space = DecodingAlgorithm.hyperparameter_space()
        space.update({
            "typical_p": {
                "type": "float",
                "low": 0.1,
                "high": 1.0,
                "description": "Typical mass threshold",
            },
            "entropy_weight": {
                "type": "float",
                "low": 0.1,
                "high": 3.0,
                "description": "Weight for entropy in typicality computation",
            },
            "adaptive": {
                "type": "categorical",
                "choices": [True, False],
                "description": "Whether to use adaptive typical mass",
            },
            "combine_with_nucleus": {
                "type": "categorical",
                "choices": [True, False],
                "description": "Whether to intersect with nucleus filtering",
            },
            "nucleus_p": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
                "description": "Nucleus threshold when combining",
            },
        })
        return space

    def get_generation_metadata(self, state: DecodingState) -> Dict[str, Any]:
        """Extract metadata from the final generation state.

        Parameters
        ----------
        state : DecodingState
            Completed generation state.

        Returns
        -------
        dict
            Dictionary with per-sequence entropy trajectories, typical set
            sizes, information content trajectories, etc.
        """
        per_seq: List[TypicalState] = state.metadata.get("per_sequence", [])
        metadata: Dict[str, Any] = {
            "algorithm": "typical",
            "typical_p": self._typical_config.typical_p,
            "temperature": self._typical_config.temperature,
            "adaptive": self._typical_config.adaptive,
            "combine_with_nucleus": self._typical_config.combine_with_nucleus,
            "sequences": [],
        }

        for i, ts in enumerate(per_seq):
            seq_meta = {
                "index": i,
                "log_prob": ts.log_prob,
                "mean_entropy": float(np.mean(ts.entropies)) if ts.entropies else 0.0,
                "mean_typical_set_size": (
                    float(np.mean(ts.typical_set_sizes))
                    if ts.typical_set_sizes
                    else 0.0
                ),
                "mean_information_content": (
                    float(np.mean(ts.information_contents))
                    if ts.information_contents
                    else 0.0
                ),
                "entropy_trajectory": ts.entropies,
                "typical_set_sizes": ts.typical_set_sizes,
                "information_contents": ts.information_contents,
                "typical_masses_used": ts.typical_masses_used,
                "num_tokens": len(ts.generated_ids),
                "finished": ts.finished,
            }
            metadata["sequences"].append(seq_meta)

        return metadata


# =========================================================================
# TypicalAnalyzer — post-hoc analysis utilities
# =========================================================================


class TypicalAnalyzer:
    """Post-hoc analysis utilities for typical decoding.

    Provides methods to analyze typicality distributions, compare typical
    decoding with nucleus sampling, compute information-theoretic
    trajectories, and estimate typical set properties.

    Parameters
    ----------
    config : TypicalConfig, optional
        Configuration for analysis.  Uses defaults if not provided.
    """

    def __init__(self, config: Optional[TypicalConfig] = None) -> None:
        self.config = config or TypicalConfig()

    def analyze_typicality_distribution(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Analyze the typicality distribution over multiple generation steps.

        Runs generation for ``num_steps`` using greedy decoding and records
        typicality statistics at each step.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of generation steps to analyze.

        Returns
        -------
        dict
            Analysis results including per-step entropy, typicality score
            distributions, typical set sizes, and summary statistics.
        """
        current_ids = list(prompt_ids)
        results: Dict[str, Any] = {
            "entropies": [],
            "mean_typicality_scores": [],
            "median_typicality_scores": [],
            "std_typicality_scores": [],
            "typical_set_sizes": [],
            "greedy_token_typicality": [],
            "greedy_token_ic": [],
            "vocab_size": None,
        }

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception as e:
                logger.warning("logit_source failed at step %d: %s", step, e)
                break

            if results["vocab_size"] is None:
                results["vocab_size"] = logits.shape[0]

            log_probs = log_softmax(logits)
            entropy = compute_entropy(log_probs)
            scores, _ = compute_typicality(logits, self.config.entropy_weight)

            results["entropies"].append(entropy)
            results["mean_typicality_scores"].append(float(np.mean(scores)))
            results["median_typicality_scores"].append(float(np.median(scores)))
            results["std_typicality_scores"].append(float(np.std(scores)))

            # Typical set size at configured mass
            ts = _build_typical_set(
                logits, self.config.typical_p, self.config.min_tokens_to_keep,
                self.config.entropy_weight,
            )
            results["typical_set_sizes"].append(ts.size)

            # Greedy token info
            greedy_token = int(np.argmax(logits))
            results["greedy_token_typicality"].append(float(scores[greedy_token]))
            results["greedy_token_ic"].append(float(-log_probs[greedy_token]))

            current_ids.append(greedy_token)

        # Summary statistics
        if results["entropies"]:
            results["summary"] = {
                "mean_entropy": float(np.mean(results["entropies"])),
                "std_entropy": float(np.std(results["entropies"])),
                "mean_typical_set_size": float(np.mean(results["typical_set_sizes"])),
                "std_typical_set_size": float(np.std(results["typical_set_sizes"])),
                "mean_greedy_typicality": float(
                    np.mean(results["greedy_token_typicality"])
                ),
                "num_steps": len(results["entropies"]),
            }

        return results

    def compare_typical_vs_nucleus(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        typical_p_values: Optional[List[float]] = None,
        nucleus_p_values: Optional[List[float]] = None,
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Compare typical and nucleus sampling across parameter ranges.

        For each combination of typical_p and nucleus_p, computes the
        effective vocabulary size and probability mass coverage.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        typical_p_values : List[float], optional
            List of typical_p values to test.
        nucleus_p_values : List[float], optional
            List of nucleus_p values to test.
        num_steps : int
            Number of steps to average over.

        Returns
        -------
        dict
            Comparison results with set sizes, overlaps, and coverage.
        """
        if typical_p_values is None:
            typical_p_values = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
        if nucleus_p_values is None:
            nucleus_p_values = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

        current_ids = list(prompt_ids)
        comparison: Dict[str, Any] = {
            "typical_p_values": typical_p_values,
            "nucleus_p_values": nucleus_p_values,
            "per_step": [],
            "average": {},
        }

        # Accumulators for averaging
        typical_sizes_acc = {p: [] for p in typical_p_values}
        nucleus_sizes_acc = {p: [] for p in nucleus_p_values}
        overlap_acc: Dict[str, List[float]] = {}

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception as e:
                logger.warning("logit_source failed at step %d: %s", step, e)
                break

            V = logits.shape[0]
            step_data: Dict[str, Any] = {"step": step}

            # Compute typical sets
            typical_sets: Dict[float, set] = {}
            for tp in typical_p_values:
                filtered = typical_filter(logits, mass=tp, min_keep=1)
                kept = set(np.where(np.isfinite(filtered) & (filtered > _NEG_INF + 1))[0])
                typical_sets[tp] = kept
                typical_sizes_acc[tp].append(len(kept))

            # Compute nucleus sets
            nucleus_sets: Dict[float, set] = {}
            for np_ in nucleus_p_values:
                filtered = _nucleus_filter(logits, p=np_, min_keep=1)
                kept = set(np.where(np.isfinite(filtered) & (filtered > _NEG_INF + 1))[0])
                nucleus_sets[np_] = kept
                nucleus_sizes_acc[np_].append(len(kept))

            # Compute overlaps between matched p values
            for tp in typical_p_values:
                for np_val in nucleus_p_values:
                    key = f"typical_{tp}_nucleus_{np_val}"
                    t_set = typical_sets[tp]
                    n_set = nucleus_sets[np_val]
                    if len(t_set) > 0 or len(n_set) > 0:
                        jaccard = (
                            len(t_set & n_set) / len(t_set | n_set)
                            if len(t_set | n_set) > 0
                            else 0.0
                        )
                    else:
                        jaccard = 1.0
                    if key not in overlap_acc:
                        overlap_acc[key] = []
                    overlap_acc[key].append(jaccard)

            # Advance with greedy token
            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        # Compute averages
        comparison["average"] = {
            "typical_set_sizes": {
                p: float(np.mean(typical_sizes_acc[p]))
                for p in typical_p_values
                if typical_sizes_acc[p]
            },
            "nucleus_set_sizes": {
                p: float(np.mean(nucleus_sizes_acc[p]))
                for p in nucleus_p_values
                if nucleus_sizes_acc[p]
            },
            "jaccard_overlaps": {
                k: float(np.mean(v)) for k, v in overlap_acc.items() if v
            },
        }

        return comparison

    def information_content_trajectory(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> List[float]:
        """Compute information content trajectory using greedy decoding.

        At each step, records the information content (surprisal) of the
        greedy (most probable) token.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of steps.

        Returns
        -------
        List[float]
            Information content values at each step.
        """
        current_ids = list(prompt_ids)
        ic_trajectory: List[float] = []

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            log_probs = log_softmax(logits)
            greedy_token = int(np.argmax(logits))
            ic_value = float(-log_probs[greedy_token])
            ic_trajectory.append(ic_value)
            current_ids.append(greedy_token)

        return ic_trajectory

    def entropy_trajectory(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> List[float]:
        """Compute entropy trajectory using greedy decoding.

        At each step, records the Shannon entropy of the full distribution.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of steps.

        Returns
        -------
        List[float]
            Entropy values at each step.
        """
        current_ids = list(prompt_ids)
        entropy_traj: List[float] = []

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            log_probs = log_softmax(logits)
            entropy_traj.append(compute_entropy(log_probs))
            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        return entropy_traj

    def typical_set_size(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        mass: Optional[float] = None,
        num_steps: int = 50,
    ) -> List[int]:
        """Compute typical set size trajectory.

        At each step, computes the number of tokens in the typical set
        at the given mass threshold.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        mass : float, optional
            Typical mass threshold.  Uses config default if not provided.
        num_steps : int
            Number of steps.

        Returns
        -------
        List[int]
            Typical set sizes at each step.
        """
        if mass is None:
            mass = self.config.typical_p
        current_ids = list(prompt_ids)
        sizes: List[int] = []

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            ts = _build_typical_set(
                logits, mass, self.config.min_tokens_to_keep,
                self.config.entropy_weight,
            )
            sizes.append(ts.size)

            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        return sizes

    def plot_typicality_histogram(
        self,
        logits: np.ndarray,
        num_bins: int = 50,
    ) -> Dict[str, Any]:
        """Compute data for a typicality histogram.

        Returns histogram bin data for the distribution of typicality scores
        across the vocabulary, along with reference lines for the entropy
        and the typical set boundary.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.
        num_bins : int
            Number of histogram bins.

        Returns
        -------
        dict
            Histogram data with keys:
            - ``bin_edges``: bin edge values
            - ``bin_counts``: counts per bin
            - ``entropy``: distribution entropy
            - ``typical_set_boundary``: score threshold at typical_p mass
            - ``mean_score``: mean typicality score
            - ``median_score``: median typicality score
            - ``scores``: raw typicality scores (for custom plotting)
            - ``log_probs``: log-probabilities
        """
        logits = np.asarray(logits, dtype=np.float64)
        log_probs = log_softmax(logits)
        entropy = compute_entropy(log_probs)
        scores, _ = compute_typicality(logits, self.config.entropy_weight)

        # Compute histogram
        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) == 0:
            return {
                "bin_edges": [],
                "bin_counts": [],
                "entropy": entropy,
                "typical_set_boundary": 0.0,
                "mean_score": 0.0,
                "median_score": 0.0,
                "scores": scores.tolist(),
                "log_probs": log_probs.tolist(),
            }

        bin_counts, bin_edges = np.histogram(finite_scores, bins=num_bins)

        # Find the typical set boundary score
        ts = _build_typical_set(
            logits, self.config.typical_p, self.config.min_tokens_to_keep,
            self.config.entropy_weight,
        )
        if ts.size > 0 and len(ts.typicality_scores) > 0:
            boundary = float(np.max(ts.typicality_scores))
        else:
            boundary = 0.0

        return {
            "bin_edges": bin_edges.tolist(),
            "bin_counts": bin_counts.tolist(),
            "entropy": entropy,
            "typical_set_boundary": boundary,
            "mean_score": float(np.mean(finite_scores)),
            "median_score": float(np.median(finite_scores)),
            "scores": scores.tolist(),
            "log_probs": log_probs.tolist(),
        }

    def build_typical_set(
        self,
        logits: np.ndarray,
        mass: Optional[float] = None,
    ) -> TypicalSet:
        """Build a :class:`TypicalSet` from a logit vector.

        Convenience wrapper around :func:`_build_typical_set`.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector.
        mass : float, optional
            Mass threshold.  Uses config default if not provided.

        Returns
        -------
        TypicalSet
            The typical set.
        """
        if mass is None:
            mass = self.config.typical_p
        return _build_typical_set(
            logits, mass, self.config.min_tokens_to_keep, self.config.entropy_weight
        )


# =========================================================================
# Module-level self-test
# =========================================================================


def _self_test() -> None:
    """Minimal smoke test executed when the module is run directly."""

    print("Running typical.py self-test...")

    # -- Helper functions --
    logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0])
    log_probs = log_softmax(logits)

    # Entropy
    H = compute_entropy(log_probs)
    assert H >= 0, f"Entropy should be non-negative, got {H}"
    print(f"  Entropy: {H:.4f}")

    # Information content
    ic = information_content(log_probs)
    assert ic.shape == log_probs.shape
    assert np.all(ic >= 0), "IC should be non-negative"

    # Typicality scores
    scores, ent = compute_typicality(logits)
    assert scores.shape == logits.shape
    assert ent >= 0
    assert np.all(scores >= 0), "Typicality scores should be non-negative"

    # -- typical_filter --
    filtered = typical_filter(logits, mass=0.5, min_keep=1)
    assert filtered.shape == logits.shape
    n_kept = np.sum(np.isfinite(filtered) & (filtered > _NEG_INF + 1))
    assert n_kept >= 1, f"Should keep at least 1 token, got {n_kept}"
    print(f"  Typical filter (mass=0.5): kept {n_kept}/{len(logits)} tokens")

    filtered_full = typical_filter(logits, mass=1.0)
    assert np.allclose(filtered_full, logits), "mass=1.0 should keep all tokens"

    # -- batch_typical_filter --
    batch = np.stack([logits, logits * 0.5])
    batch_filtered = batch_typical_filter(batch, mass=0.8)
    assert batch_filtered.shape == batch.shape

    # -- conditional_entropy_estimate --
    history = [log_probs, log_probs * 0.9]
    cond_ent = conditional_entropy_estimate(history)
    assert cond_ent >= 0
    assert conditional_entropy_estimate([]) == 0.0

    # -- TypicalSet --
    ts = _build_typical_set(logits, mass=0.9, min_keep=1)
    assert ts.size >= 1
    assert 0 <= ts.mass <= 1.0 + 1e-6
    assert ts.entropy >= 0
    print(f"  TypicalSet: size={ts.size}, mass={ts.mass:.4f}, entropy={ts.entropy:.4f}")

    most = ts.most_typical(2)
    assert len(most) <= 2
    least = ts.least_typical(2)
    assert len(least) <= 2

    if ts.size > 0:
        assert ts.is_member(ts.tokens[0])
        assert ts.tokens[0] in ts

    assert ts.coverage() == ts.mass
    d = ts.to_dict()
    assert "tokens" in d and "entropy" in d

    # -- TypicalConfig validation --
    cfg = TypicalConfig(typical_p=0.95, temperature=1.0)
    assert cfg.validate() == []

    bad_cfg = TypicalConfig(typical_p=0.0)
    assert len(bad_cfg.validate()) > 0

    bad_cfg2 = TypicalConfig(typical_p=1.5)
    assert len(bad_cfg2.validate()) > 0

    # -- TypicalDecoding with mock logit source --
    vocab_size = 10

    def mock_logit_source(input_ids: List[List[int]]) -> np.ndarray:
        batch_size = len(input_ids)
        rng = np.random.RandomState(42 + len(input_ids[0]))
        return rng.randn(batch_size, vocab_size).astype(np.float64)

    config = TypicalConfig(
        algorithm_name="typical",
        typical_p=0.9,
        temperature=1.0,
        num_sequences=3,
        max_new_tokens=10,
        seed=42,
    )
    algo = TypicalDecoding(config)
    assert algo.name == "typical"

    results = algo.generate(mock_logit_source, [1, 2, 3])
    assert len(results) == 3
    for seq in results:
        assert len(seq) > 3  # prompt + at least some generated tokens
    print(f"  Generated {len(results)} sequences, lengths: {[len(s) for s in results]}")

    # -- Adaptive mode --
    adaptive_cfg = TypicalConfig(
        algorithm_name="typical_adaptive",
        typical_p=0.9,
        adaptive=True,
        local_entropy_window=3,
        num_sequences=2,
        max_new_tokens=8,
        seed=123,
    )
    adaptive_algo = TypicalDecoding(adaptive_cfg)
    results_adaptive = adaptive_algo.generate(mock_logit_source, [1, 2])
    assert len(results_adaptive) == 2
    print(f"  Adaptive: {len(results_adaptive)} sequences")

    # -- Combined typical + nucleus --
    combined_cfg = TypicalConfig(
        algorithm_name="typical_nucleus",
        typical_p=0.9,
        combine_with_nucleus=True,
        nucleus_p=0.85,
        num_sequences=2,
        max_new_tokens=8,
        seed=99,
    )
    combined_algo = TypicalDecoding(combined_cfg)
    results_combined = combined_algo.generate(mock_logit_source, [1, 2, 3])
    assert len(results_combined) == 2
    print(f"  Combined typical+nucleus: {len(results_combined)} sequences")

    # -- TypicalAnalyzer --
    analyzer = TypicalAnalyzer(TypicalConfig(typical_p=0.9))

    # Typicality distribution analysis
    analysis = analyzer.analyze_typicality_distribution(
        mock_logit_source, [1, 2, 3], num_steps=5
    )
    assert "entropies" in analysis
    assert len(analysis["entropies"]) == 5
    assert "summary" in analysis
    print(f"  Analyzer: {analysis['summary']}")

    # Information content trajectory
    ic_traj = analyzer.information_content_trajectory(
        mock_logit_source, [1, 2, 3], num_steps=5
    )
    assert len(ic_traj) == 5
    assert all(v >= 0 for v in ic_traj)

    # Entropy trajectory
    ent_traj = analyzer.entropy_trajectory(
        mock_logit_source, [1, 2, 3], num_steps=5
    )
    assert len(ent_traj) == 5
    assert all(v >= 0 for v in ent_traj)

    # Typical set size trajectory
    ts_sizes = analyzer.typical_set_size(
        mock_logit_source, [1, 2, 3], mass=0.9, num_steps=5
    )
    assert len(ts_sizes) == 5
    assert all(s >= 1 for s in ts_sizes)

    # Compare typical vs nucleus
    comparison = analyzer.compare_typical_vs_nucleus(
        mock_logit_source, [1, 2, 3],
        typical_p_values=[0.5, 0.9],
        nucleus_p_values=[0.5, 0.9],
        num_steps=3,
    )
    assert "average" in comparison
    assert "typical_set_sizes" in comparison["average"]
    assert "nucleus_set_sizes" in comparison["average"]

    # Plot typicality histogram
    test_logits = np.random.randn(50)
    hist_data = analyzer.plot_typicality_histogram(test_logits, num_bins=10)
    assert "bin_edges" in hist_data
    assert "entropy" in hist_data
    assert "typical_set_boundary" in hist_data
    assert "scores" in hist_data

    # Build typical set via analyzer
    ts_via_analyzer = analyzer.build_typical_set(test_logits, mass=0.8)
    assert isinstance(ts_via_analyzer, TypicalSet)

    # -- Hyperparameter space --
    space = TypicalDecoding.hyperparameter_space()
    assert "typical_p" in space
    assert "entropy_weight" in space

    # -- Registry --
    assert AlgorithmRegistry.is_registered("typical")

    print("typical.py self-test passed ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()


# =========================================================================
# EntropyAdaptiveTypical — entropy-adaptive tau
# =========================================================================


class EntropyAdaptiveTypical:
    """Locally typical sampling with entropy-adaptive tau.

    Instead of a fixed ``typical_p``, dynamically adjusts the typicality
    threshold based on the local entropy of the distribution at each step.
    When entropy is high (model uncertain), uses a broader typical set
    (higher tau).  When entropy is low (confident), uses a tighter typical
    set (lower tau).

    Parameters
    ----------
    min_tau : float
        Minimum typicality threshold.  Used when the model is most
        confident (lowest entropy).
    max_tau : float
        Maximum typicality threshold.  Used when the model is most
        uncertain (highest entropy).
    midpoint : float
        Entropy value at which tau is halfway between ``min_tau`` and
        ``max_tau``.  Controls the sigmoid transition centre.
    temperature : float
        Softmax temperature applied before filtering.
    min_tokens_to_keep : int
        Minimum number of tokens retained after filtering.
    entropy_weight : float
        Weight for the entropy term in typicality computation.
    """

    def __init__(
        self,
        min_tau: float = 0.2,
        max_tau: float = 0.99,
        midpoint: float = 3.0,
        temperature: float = 1.0,
        min_tokens_to_keep: int = 1,
        entropy_weight: float = 1.0,
    ) -> None:
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.midpoint = midpoint
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        self.entropy_weight = entropy_weight
        self.tau_history: List[float] = []
        self.entropy_history: List[float] = []

    def _local_entropy(self, logits: np.ndarray) -> float:
        """Compute Shannon entropy of the distribution defined by *logits*.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.

        Returns
        -------
        float
            Shannon entropy (nats).
        """
        log_probs = log_softmax(np.asarray(logits, dtype=np.float64))
        return compute_entropy(log_probs)

    def _adapt_tau(self, entropy: float) -> float:
        """Map current entropy to an adaptive tau value via a sigmoid.

        Higher entropy yields a larger tau (broader typical set); lower
        entropy yields a smaller tau (tighter typical set).

        Parameters
        ----------
        entropy : float
            Current distribution entropy.

        Returns
        -------
        float
            Adapted tau in ``[min_tau, max_tau]``.
        """
        tau = adaptive_tau_from_entropy(
            entropy,
            min_tau=self.min_tau,
            max_tau=self.max_tau,
            midpoint=self.midpoint,
        )
        logger.debug(
            "EntropyAdaptiveTypical._adapt_tau: H=%.4f → tau=%.4f",
            entropy, tau,
        )
        return tau

    def _step(
        self,
        logits: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, float, float]:
        """Execute a single entropy-adaptive typical decoding step.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.
        rng : np.random.Generator, optional
            Random number generator for sampling.

        Returns
        -------
        token : int
            Sampled token index.
        tau : float
            Tau value used for this step.
        entropy : float
            Entropy of the distribution at this step.
        """
        logits = np.asarray(logits, dtype=np.float64)
        entropy = self._local_entropy(logits)
        tau = self._adapt_tau(entropy)

        # Record history
        self.tau_history.append(tau)
        self.entropy_history.append(entropy)

        # Apply typical filtering with adapted tau
        filtered = typical_filter(
            logits,
            mass=tau,
            min_keep=self.min_tokens_to_keep,
            entropy_weight=self.entropy_weight,
        )

        # Sample
        token = _sample_from_logits(filtered, temperature=self.temperature, rng=rng)
        return token, tau, entropy

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Generate a sequence using entropy-adaptive typical sampling.

        Parameters
        ----------
        logit_source : LogitSource
            Callable that accepts a list of token-ID sequences and returns
            logits of shape ``(batch, vocab_size)``.
        prompt_ids : List[int]
            Token IDs of the prompt.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_token_id : int, optional
            End-of-sequence token ID.  Generation stops when sampled.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        dict
            ``generated_ids``, ``tau_history``, ``entropy_history``,
            ``log_prob``, and ``num_tokens`` keys.
        """
        current_ids = list(prompt_ids)
        generated: List[int] = []
        log_prob = 0.0
        self.tau_history = []
        self.entropy_history = []

        for _ in range(max_new_tokens):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception as exc:
                logger.error("logit_source failed: %s", exc)
                break

            token, tau, entropy = self._step(logits, rng=rng)

            # Accumulate log-probability
            lp = log_softmax(logits)
            log_prob += float(lp[token])

            generated.append(token)
            current_ids.append(token)

            if eos_token_id is not None and token == eos_token_id:
                break

        return {
            "generated_ids": generated,
            "tau_history": list(self.tau_history),
            "entropy_history": list(self.entropy_history),
            "log_prob": log_prob,
            "num_tokens": len(generated),
        }


# =========================================================================
# TypicalNucleusHybrid — combines typical + nucleus sampling
# =========================================================================


class TypicalNucleusHybrid:
    """Hybrid sampler that combines typical and nucleus filtering.

    First constructs the typical set, then intersects it with the nucleus
    set, and samples from the intersection.  If the intersection is empty,
    falls back to the larger of the two sets.

    Parameters
    ----------
    typical_p : float
        Cumulative probability mass threshold for the typical set.
    nucleus_p : float
        Cumulative probability mass threshold for the nucleus set.
    temperature : float
        Softmax temperature applied before sampling.
    min_tokens_to_keep : int
        Minimum number of tokens retained after filtering.
    entropy_weight : float
        Weight for the entropy term in typicality computation.
    """

    def __init__(
        self,
        typical_p: float = 0.95,
        nucleus_p: float = 0.9,
        temperature: float = 1.0,
        min_tokens_to_keep: int = 1,
        entropy_weight: float = 1.0,
    ) -> None:
        self.typical_p = typical_p
        self.nucleus_p = nucleus_p
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        self.entropy_weight = entropy_weight
        self.dominant_filter_history: List[str] = []

    def _typical_filter(self, logits: np.ndarray) -> np.ndarray:
        """Apply typical filtering and return a boolean mask of kept tokens.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(V,)`` — ``True`` for tokens in the
            typical set.
        """
        filtered = typical_filter(
            logits,
            mass=self.typical_p,
            min_keep=self.min_tokens_to_keep,
            entropy_weight=self.entropy_weight,
        )
        return np.isfinite(filtered) & (filtered > _NEG_INF + 1)

    def _nucleus_filter(self, logits: np.ndarray) -> np.ndarray:
        """Apply nucleus (top-p) filtering and return a boolean mask.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(V,)`` — ``True`` for tokens in the
            nucleus set.
        """
        filtered = _nucleus_filter(
            logits, p=self.nucleus_p, min_keep=self.min_tokens_to_keep,
        )
        return np.isfinite(filtered) & (filtered > _NEG_INF + 1)

    def _intersect_filters(
        self,
        typical_mask: np.ndarray,
        nucleus_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute the intersection of typical and nucleus masks.

        Parameters
        ----------
        typical_mask : np.ndarray
            Boolean mask for the typical set.
        nucleus_mask : np.ndarray
            Boolean mask for the nucleus set.

        Returns
        -------
        np.ndarray
            Boolean mask for the intersection.
        """
        return typical_mask & nucleus_mask

    def _fallback_strategy(
        self,
        logits: np.ndarray,
        typical_mask: np.ndarray,
        nucleus_mask: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """Determine fallback when intersection is empty.

        Falls back to the larger of the two sets.  Records which filter
        was dominant.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.
        typical_mask : np.ndarray
            Boolean mask for the typical set.
        nucleus_mask : np.ndarray
            Boolean mask for the nucleus set.

        Returns
        -------
        filtered_logits : np.ndarray
            Filtered logit vector using the fallback set.
        dominant : str
            ``"typical"`` or ``"nucleus"`` depending on which set was used.
        """
        typical_size = int(np.sum(typical_mask))
        nucleus_size = int(np.sum(nucleus_mask))

        if typical_size >= nucleus_size:
            mask = typical_mask
            dominant = "typical"
        else:
            mask = nucleus_mask
            dominant = "nucleus"

        result = np.full_like(logits, _NEG_INF)
        result[mask] = logits[mask]
        logger.debug(
            "TypicalNucleusHybrid fallback: using %s set (size=%d)",
            dominant, int(np.sum(mask)),
        )
        return result, dominant

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Generate a sequence using the typical–nucleus hybrid sampler.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        max_new_tokens : int
            Maximum tokens to generate.
        eos_token_id : int, optional
            End-of-sequence token ID.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        dict
            ``generated_ids``, ``dominant_filter_history``, ``log_prob``,
            and ``num_tokens`` keys.
        """
        current_ids = list(prompt_ids)
        generated: List[int] = []
        log_prob = 0.0
        self.dominant_filter_history = []

        for _ in range(max_new_tokens):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception as exc:
                logger.error("logit_source failed: %s", exc)
                break

            typical_mask = self._typical_filter(logits)
            nucleus_mask = self._nucleus_filter(logits)
            intersection = self._intersect_filters(typical_mask, nucleus_mask)

            if np.sum(intersection) >= self.min_tokens_to_keep:
                filtered = np.full_like(logits, _NEG_INF)
                filtered[intersection] = logits[intersection]
                dominant = "intersection"
            else:
                filtered, dominant = self._fallback_strategy(
                    logits, typical_mask, nucleus_mask,
                )

            self.dominant_filter_history.append(dominant)

            token = _sample_from_logits(filtered, temperature=self.temperature, rng=rng)

            lp = log_softmax(logits)
            log_prob += float(lp[token])

            generated.append(token)
            current_ids.append(token)

            if eos_token_id is not None and token == eos_token_id:
                break

        return {
            "generated_ids": generated,
            "dominant_filter_history": list(self.dominant_filter_history),
            "log_prob": log_prob,
            "num_tokens": len(generated),
        }


# =========================================================================
# LookAheadTypical — typical sampling with look-ahead scoring
# =========================================================================


class LookAheadTypical:
    """Typical sampling with look-ahead scoring.

    Before committing to a token, evaluates each candidate by looking
    ahead *L* steps (using greedy or sampled rollouts) and scoring the
    resulting sub-sequence for typicality.  Tokens that lead to more
    typical continuations are preferred.

    Parameters
    ----------
    lookahead_depth : int
        Number of look-ahead steps (*L*).
    typical_p : float
        Typical mass threshold for the initial candidate set.
    temperature : float
        Softmax temperature for sampling.
    rollout_mode : str
        ``"greedy"`` for greedy rollouts or ``"sample"`` for sampled
        rollouts during look-ahead.
    max_candidates : int
        Maximum number of candidate tokens to evaluate with look-ahead.
        Candidates are drawn from the typical set.
    entropy_weight : float
        Weight for the entropy term in typicality computation.
    min_tokens_to_keep : int
        Minimum number of tokens retained after typical filtering.
    """

    def __init__(
        self,
        lookahead_depth: int = 3,
        typical_p: float = 0.95,
        temperature: float = 1.0,
        rollout_mode: str = "greedy",
        max_candidates: int = 10,
        entropy_weight: float = 1.0,
        min_tokens_to_keep: int = 1,
    ) -> None:
        self.lookahead_depth = lookahead_depth
        self.typical_p = typical_p
        self.temperature = temperature
        self.rollout_mode = rollout_mode
        self.max_candidates = max_candidates
        self.entropy_weight = entropy_weight
        self.min_tokens_to_keep = min_tokens_to_keep

    def _rollout(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        depth: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[int], List[float]]:
        """Perform a rollout of *depth* steps from *prefix*.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prefix : List[int]
            Token IDs preceding the rollout.
        depth : int
            Number of rollout steps.
        rng : np.random.Generator, optional
            Random number generator (used when ``rollout_mode="sample"``).

        Returns
        -------
        tokens : List[int]
            Token IDs generated during the rollout.
        typicality_scores : List[float]
            Per-step typicality score of each generated token.
        """
        current = list(prefix)
        tokens: List[int] = []
        scores: List[float] = []

        for _ in range(depth):
            try:
                logits = logit_source([current])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            log_probs = log_softmax(logits)
            entropy = compute_entropy(log_probs)
            ic = information_content(log_probs)
            typ_scores = np.abs(ic - self.entropy_weight * entropy)

            if self.rollout_mode == "greedy":
                token = int(np.argmax(logits))
            else:
                token = _sample_from_logits(
                    logits, temperature=self.temperature, rng=rng,
                )

            tokens.append(token)
            scores.append(float(typ_scores[token]))
            current.append(token)

        return tokens, scores

    def _aggregate_typicality_scores(self, scores: List[float]) -> float:
        """Aggregate per-step typicality scores into a single value.

        Uses the mean of the scores; lower is more typical.

        Parameters
        ----------
        scores : List[float]
            Per-step typicality scores from a rollout.

        Returns
        -------
        float
            Aggregated score.  Lower values indicate more typical
            continuations.
        """
        if not scores:
            return float("inf")
        return float(np.mean(scores))

    def _score_with_lookahead(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        candidate_token: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Score a candidate token by looking ahead from it.

        Appends *candidate_token* to *prefix*, performs a rollout, and
        returns an aggregated typicality score.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prefix : List[int]
            Token IDs before the candidate.
        candidate_token : int
            Token to evaluate.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        float
            Aggregated look-ahead typicality score (lower = more typical).
        """
        extended = list(prefix) + [candidate_token]
        _, scores = self._rollout(
            logit_source, extended, self.lookahead_depth, rng=rng,
        )
        return self._aggregate_typicality_scores(scores)

    def _select_with_lookahead(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        logits: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Select the best token by look-ahead scoring over typical candidates.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prefix : List[int]
            Current token IDs.
        logits : np.ndarray
            Raw logit vector at the current step.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        int
            Selected token index.
        """
        # Build typical set as the candidate pool
        ts = _build_typical_set(
            logits, mass=self.typical_p,
            min_keep=self.min_tokens_to_keep,
            entropy_weight=self.entropy_weight,
        )

        candidates = ts.tokens[: self.max_candidates]

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate via look-ahead
        best_token = candidates[0]
        best_score = float("inf")

        for cand in candidates:
            score = self._score_with_lookahead(
                logit_source, prefix, cand, rng=rng,
            )
            if score < best_score:
                best_score = score
                best_token = cand

        logger.debug(
            "LookAheadTypical: selected token %d with score %.4f "
            "(candidates=%d)",
            best_token, best_score, len(candidates),
        )
        return best_token

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Generate a sequence using look-ahead typical sampling.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        max_new_tokens : int
            Maximum tokens to generate.
        eos_token_id : int, optional
            End-of-sequence token ID.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        dict
            ``generated_ids``, ``log_prob``, and ``num_tokens`` keys.
        """
        current_ids = list(prompt_ids)
        generated: List[int] = []
        log_prob = 0.0

        for _ in range(max_new_tokens):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception as exc:
                logger.error("logit_source failed: %s", exc)
                break

            token = self._select_with_lookahead(
                logit_source, current_ids, logits, rng=rng,
            )

            lp = log_softmax(logits)
            log_prob += float(lp[token])

            generated.append(token)
            current_ids.append(token)

            if eos_token_id is not None and token == eos_token_id:
                break

        return {
            "generated_ids": generated,
            "log_prob": log_prob,
            "num_tokens": len(generated),
        }


# =========================================================================
# InformationTheoreticAnalyzer — comprehensive IT analysis
# =========================================================================


class InformationTheoreticAnalyzer:
    """Comprehensive information-theoretic analysis of typical set membership.

    Computes KL divergence between the typical set distribution and the full
    distribution, mutual information between consecutive typical sets,
    coverage trajectories, entropy rate estimation, and redundancy analysis.

    Parameters
    ----------
    typical_p : float
        Typical mass threshold for building typical sets.
    entropy_weight : float
        Weight for the entropy term in typicality computation.
    min_tokens_to_keep : int
        Minimum tokens retained when building typical sets.
    """

    def __init__(
        self,
        typical_p: float = 0.95,
        entropy_weight: float = 1.0,
        min_tokens_to_keep: int = 1,
    ) -> None:
        self.typical_p = typical_p
        self.entropy_weight = entropy_weight
        self.min_tokens_to_keep = min_tokens_to_keep

    def kl_divergence_typical_vs_full(self, logits: np.ndarray) -> float:
        """Compute KL(typical || full) for a single logit vector.

        The typical-set distribution is obtained by restricting the full
        distribution to the typical set and renormalising.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.

        Returns
        -------
        float
            KL divergence in nats.  Non-negative.
        """
        return typical_set_kl_divergence(logits, self.typical_p)

    def mutual_information_consecutive(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> List[float]:
        """Estimate mutual information between consecutive typical sets.

        At each pair of consecutive steps, estimates ``I(T_t; T_{t+1})``
        where ``T_t`` is the typical set at step *t*.  The mutual
        information is approximated as:

            I = H(T_t) + H(T_{t+1}) - H(T_t, T_{t+1})

        where set entropies are computed from the renormalised distributions
        and the joint entropy is approximated from the Jaccard overlap.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of generation steps.

        Returns
        -------
        List[float]
            Mutual information estimates for each consecutive pair.
        """
        current_ids = list(prompt_ids)
        mi_values: List[float] = []
        prev_ts: Optional[TypicalSet] = None
        prev_log_probs: Optional[np.ndarray] = None

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            ts = _build_typical_set(
                logits, mass=self.typical_p,
                min_keep=self.min_tokens_to_keep,
                entropy_weight=self.entropy_weight,
            )
            log_probs = log_softmax(logits)

            if prev_ts is not None and prev_log_probs is not None:
                # Entropy of previous typical set distribution
                prev_mask = np.zeros(prev_log_probs.shape[0], dtype=bool)
                for tok in prev_ts.tokens:
                    prev_mask[tok] = True
                prev_lp = prev_log_probs.copy()
                prev_lp[~prev_mask] = _NEG_INF
                prev_lp_norm = log_softmax(prev_lp[prev_mask])
                h_prev = compute_entropy(prev_lp_norm) if len(prev_lp_norm) > 0 else 0.0

                # Entropy of current typical set distribution
                cur_mask = np.zeros(log_probs.shape[0], dtype=bool)
                for tok in ts.tokens:
                    cur_mask[tok] = True
                cur_lp = log_probs.copy()
                cur_lp[~cur_mask] = _NEG_INF
                cur_lp_norm = log_softmax(cur_lp[cur_mask])
                h_cur = compute_entropy(cur_lp_norm) if len(cur_lp_norm) > 0 else 0.0

                # Joint entropy approximation via Jaccard overlap
                prev_set = set(prev_ts.tokens)
                cur_set = set(ts.tokens)
                union_size = len(prev_set | cur_set)
                inter_size = len(prev_set & cur_set)
                jaccard = inter_size / union_size if union_size > 0 else 0.0

                # Approximate joint entropy: H_joint ≈ H_prev + H_cur * (1 - jaccard)
                h_joint = h_prev + h_cur * (1.0 - jaccard)
                mi = max(0.0, h_prev + h_cur - h_joint)
                mi_values.append(mi)

            prev_ts = ts
            prev_log_probs = log_probs

            # Advance with greedy token
            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        return mi_values

    def typical_set_coverage_trajectory(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> List[float]:
        """Compute the probability mass coverage of the typical set per step.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of generation steps.

        Returns
        -------
        List[float]
            Typical set coverage (probability mass) at each step.
        """
        current_ids = list(prompt_ids)
        coverages: List[float] = []

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            ts = _build_typical_set(
                logits, mass=self.typical_p,
                min_keep=self.min_tokens_to_keep,
                entropy_weight=self.entropy_weight,
            )
            coverages.append(ts.coverage())

            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        return coverages

    def entropy_rate_estimation(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Estimate the entropy rate of the source via conditional entropies.

        The entropy rate is approximated as the running average of
        per-step conditional entropies ``H(X_t | X_{<t})``.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of steps to average over.

        Returns
        -------
        dict
            ``entropy_rate``, ``per_step_entropies``, and
            ``cumulative_estimates`` keys.
        """
        rate = estimate_entropy_rate(logit_source, prompt_ids, num_steps)

        # Also return per-step detail
        current_ids = list(prompt_ids)
        per_step: List[float] = []
        cumulative: List[float] = []

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            log_probs = log_softmax(logits)
            h = compute_entropy(log_probs)
            per_step.append(h)
            cumulative.append(float(np.mean(per_step)))

            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        return {
            "entropy_rate": rate,
            "per_step_entropies": per_step,
            "cumulative_estimates": cumulative,
        }

    def redundancy_analysis(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Analyse redundancy in the generation process.

        Redundancy is defined as ``R = 1 - H_rate / H_max`` where
        ``H_rate`` is the estimated entropy rate and ``H_max`` is the
        maximum entropy (log of vocabulary size).

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of steps.

        Returns
        -------
        dict
            ``redundancy``, ``entropy_rate``, ``max_entropy``, and
            per-step ``redundancy_trajectory`` keys.
        """
        current_ids = list(prompt_ids)
        per_step_entropies: List[float] = []
        h_max: Optional[float] = None

        for step in range(num_steps):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                break

            if h_max is None:
                h_max = float(np.log(logits.shape[0]))

            log_probs = log_softmax(logits)
            h = compute_entropy(log_probs)
            per_step_entropies.append(h)

            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        if h_max is None or h_max < _LOG_EPS:
            h_max = 1.0

        entropy_rate = float(np.mean(per_step_entropies)) if per_step_entropies else 0.0
        redundancy = 1.0 - entropy_rate / h_max

        redundancy_trajectory: List[float] = []
        running_sum = 0.0
        for i, h in enumerate(per_step_entropies):
            running_sum += h
            running_rate = running_sum / (i + 1)
            redundancy_trajectory.append(1.0 - running_rate / h_max)

        return {
            "redundancy": redundancy,
            "entropy_rate": entropy_rate,
            "max_entropy": h_max,
            "redundancy_trajectory": redundancy_trajectory,
        }

    def full_analysis(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Run all information-theoretic analyses.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        num_steps : int
            Number of generation steps.

        Returns
        -------
        dict
            Combined results from all analysis methods.
        """
        logger.info(
            "InformationTheoreticAnalyzer.full_analysis: running %d steps",
            num_steps,
        )
        return {
            "kl_divergence_trajectory": [
                self._kl_at_step(logit_source, prompt_ids, s)
                for s in range(min(num_steps, 20))
            ],
            "mutual_information": self.mutual_information_consecutive(
                logit_source, prompt_ids, num_steps,
            ),
            "coverage_trajectory": self.typical_set_coverage_trajectory(
                logit_source, prompt_ids, num_steps,
            ),
            "entropy_rate": self.entropy_rate_estimation(
                logit_source, prompt_ids, num_steps,
            ),
            "redundancy": self.redundancy_analysis(
                logit_source, prompt_ids, num_steps,
            ),
        }

    # -- private helpers ----------------------------------------------------

    def _kl_at_step(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        target_step: int,
    ) -> float:
        """Compute KL divergence at a specific generation step.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits.
        prompt_ids : List[int]
            Token IDs of the prompt.
        target_step : int
            Step index at which to compute KL divergence.

        Returns
        -------
        float
            KL(typical || full) at the given step.
        """
        current_ids = list(prompt_ids)
        for s in range(target_step + 1):
            try:
                logits = logit_source([current_ids])
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 2:
                    logits = logits[0]
            except Exception:
                return 0.0

            if s == target_step:
                return self.kl_divergence_typical_vs_full(logits)

            greedy_token = int(np.argmax(logits))
            current_ids.append(greedy_token)

        return 0.0


# =========================================================================
# Helper functions
# =========================================================================


def adaptive_tau_from_entropy(
    entropy: float,
    min_tau: float = 0.2,
    max_tau: float = 0.99,
    midpoint: float = 3.0,
) -> float:
    """Map entropy to an adaptive typicality threshold via a sigmoid.

    Higher entropy produces a larger tau (broader typical set); lower
    entropy produces a smaller tau (tighter typical set).

    Parameters
    ----------
    entropy : float
        Shannon entropy of the current distribution (nats).
    min_tau : float
        Minimum tau value (returned at very low entropy).
    max_tau : float
        Maximum tau value (returned at very high entropy).
    midpoint : float
        Entropy value at which tau equals ``(min_tau + max_tau) / 2``.

    Returns
    -------
    float
        Adaptive tau in ``[min_tau, max_tau]``.
    """
    # Sigmoid centred at midpoint with reasonable steepness
    x = entropy - midpoint
    sigmoid = 1.0 / (1.0 + math.exp(-x))
    tau = min_tau + (max_tau - min_tau) * sigmoid
    return float(np.clip(tau, min_tau, max_tau))


def typical_nucleus_intersection(
    logits: np.ndarray,
    typical_p: float = 0.95,
    nucleus_p: float = 0.9,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute intersection of typical and nucleus sets.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector, shape ``(V,)``.
    typical_p : float
        Typical mass threshold.
    nucleus_p : float
        Nucleus mass threshold.
    temperature : float
        Temperature applied before filtering.

    Returns
    -------
    filtered_logits : np.ndarray
        Logit vector with non-intersection tokens set to ``-inf``.
    typical_mask : np.ndarray
        Boolean mask of the typical set.
    nucleus_mask : np.ndarray
        Boolean mask of the nucleus set.
    intersection_mask : np.ndarray
        Boolean mask of the intersection.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature > 0 and abs(temperature - 1.0) > 1e-8:
        logits = logits / temperature

    typ_filtered = typical_filter(logits, mass=typical_p, min_keep=1)
    nuc_filtered = _nucleus_filter(logits, p=nucleus_p, min_keep=1)

    typical_mask = np.isfinite(typ_filtered) & (typ_filtered > _NEG_INF + 1)
    nucleus_mask = np.isfinite(nuc_filtered) & (nuc_filtered > _NEG_INF + 1)
    intersection_mask = typical_mask & nucleus_mask

    filtered_logits = np.full_like(logits, _NEG_INF)
    if np.any(intersection_mask):
        filtered_logits[intersection_mask] = logits[intersection_mask]
    else:
        # Fall back to the larger set
        if int(np.sum(typical_mask)) >= int(np.sum(nucleus_mask)):
            filtered_logits[typical_mask] = logits[typical_mask]
        else:
            filtered_logits[nucleus_mask] = logits[nucleus_mask]

    return filtered_logits, typical_mask, nucleus_mask, intersection_mask


def lookahead_typicality_score(
    logit_source: LogitSource,
    prefix: List[int],
    candidate_token: int,
    depth: int = 3,
    temperature: float = 1.0,
) -> float:
    """Score a candidate token by greedy look-ahead typicality.

    Appends *candidate_token* to *prefix*, runs *depth* greedy steps,
    and returns the mean typicality score across the look-ahead window.

    Parameters
    ----------
    logit_source : LogitSource
        Callable returning logits.
    prefix : List[int]
        Token IDs before the candidate.
    candidate_token : int
        Token to evaluate.
    depth : int
        Number of look-ahead steps.
    temperature : float
        Softmax temperature (used only for typicality computation).

    Returns
    -------
    float
        Mean typicality score over the look-ahead (lower = more typical).
    """
    current = list(prefix) + [candidate_token]
    scores: List[float] = []

    for _ in range(depth):
        try:
            logits = logit_source([current])
            logits = np.asarray(logits, dtype=np.float64)
            if logits.ndim == 2:
                logits = logits[0]
        except Exception:
            break

        log_probs = log_softmax(logits)
        entropy = compute_entropy(log_probs)
        ic = information_content(log_probs)
        typ = np.abs(ic - entropy)

        greedy = int(np.argmax(logits))
        scores.append(float(typ[greedy]))
        current.append(greedy)

    if not scores:
        return float("inf")
    return float(np.mean(scores))


def estimate_entropy_rate(
    logit_source: LogitSource,
    prefix: List[int],
    num_steps: int = 50,
) -> float:
    """Estimate the entropy rate by averaging per-step conditional entropies.

    Uses greedy decoding to extend the prefix and computes
    ``H(X_t | X_{<t})`` at each step.

    Parameters
    ----------
    logit_source : LogitSource
        Callable returning logits.
    prefix : List[int]
        Initial token IDs.
    num_steps : int
        Number of steps to average over.

    Returns
    -------
    float
        Estimated entropy rate (nats per token).
    """
    current = list(prefix)
    entropies: List[float] = []

    for _ in range(num_steps):
        try:
            logits = logit_source([current])
            logits = np.asarray(logits, dtype=np.float64)
            if logits.ndim == 2:
                logits = logits[0]
        except Exception:
            break

        log_probs = log_softmax(logits)
        h = compute_entropy(log_probs)
        entropies.append(h)

        greedy = int(np.argmax(logits))
        current.append(greedy)

    if not entropies:
        return 0.0
    return float(np.mean(entropies))


def typical_set_kl_divergence(
    logits: np.ndarray,
    typical_p: float = 0.95,
) -> float:
    """Compute KL(Q_typical || P_full) for a single logit vector.

    ``Q_typical`` is the distribution restricted to the typical set and
    renormalised.  ``P_full`` is the original distribution.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector, shape ``(V,)``.
    typical_p : float
        Typical mass threshold.

    Returns
    -------
    float
        KL divergence in nats.  Non-negative.
    """
    logits = np.asarray(logits, dtype=np.float64)
    log_probs_full = log_softmax(logits)

    # Build typical set
    filtered = typical_filter(logits, mass=typical_p, min_keep=1)
    mask = np.isfinite(filtered) & (filtered > _NEG_INF + 1)

    if not np.any(mask):
        return 0.0

    # Renormalise the typical-set distribution in log-space
    log_probs_typical = log_probs_full.copy()
    log_probs_typical[~mask] = _NEG_INF
    log_z = logsumexp(log_probs_typical[mask])
    log_q = log_probs_typical[mask] - log_z

    # KL(Q || P) = sum_x Q(x) * [log Q(x) - log P(x)]
    log_p = log_probs_full[mask]
    q = np.exp(log_q)

    # Avoid log(0) issues
    valid = q > 0
    if not np.any(valid):
        return 0.0

    kl = float(np.sum(q[valid] * (log_q[valid] - log_p[valid])))
    return max(0.0, kl)
