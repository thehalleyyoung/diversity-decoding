"""
Nucleus (Top-p) Sampling for the Diversity Decoding Arena.
==========================================================

Implements the nucleus sampling algorithm (Holtzman et al., 2020) along with
adaptive and analytical extensions.  Nucleus sampling restricts the token
vocabulary at each decoding step to the smallest set whose cumulative
probability mass exceeds a threshold *p*, then re-normalizes and samples from
that set.  This provides a dynamic vocabulary size that naturally adapts to the
shape of the model's predicted distribution — confident predictions yield small
nuclei while uncertain predictions yield large ones.

Key components
--------------
* **NucleusConfig** — dataclass holding all hyper-parameters (top_p,
  temperature, adaptive settings, repetition penalty, …).
* **NucleusSampling** — the main ``DecodingAlgorithm`` implementation with
  step-level filtering, tracking, and generation.
* **NucleusAnalyzer** — post-hoc analysis utilities (p-sensitivity, entropy
  trajectories, optimal-p estimation, diversity curves).
* **AdaptiveNucleus** — wrapper that adjusts *p* on the fly using the local
  entropy of the predicted distribution.
* **Helper functions** — stand-alone utilities for filtering, cumulative
  probability computation, renormalization, and batched operations.

Numerical stability
-------------------
All probability computations go through the log-softmax → exp path to avoid
overflow/underflow.  ``scipy.special.log_softmax`` and ``logsumexp`` are used
throughout.

References
----------
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020).
  *The Curious Case of Neural Text Degeneration*.  ICLR 2020.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import log_softmax, logsumexp, softmax

from src.algorithms.base import DecodingAlgorithm, DecodingConfig, DecodingState

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
class NucleusConfig(DecodingConfig):
    """Configuration for nucleus (top-p) sampling.

    Attributes
    ----------
    top_p : float
        Cumulative probability threshold.  Only the smallest set of tokens
        whose cumulative probability exceeds ``top_p`` is kept.  Values in
        (0, 1].  A value of 1.0 disables nucleus filtering.
    temperature : float
        Softmax temperature applied *before* nucleus filtering.  Values > 1
        increase entropy (more diverse), values < 1 decrease entropy (more
        peaked).
    min_tokens_to_keep : int
        Minimum number of tokens that must remain after filtering, regardless
        of the cumulative probability.  Prevents degenerate empty nuclei.
    filter_value : float
        Value used to mask filtered-out logits.  Defaults to ``-inf`` so that
        their softmax probability is exactly 0.
    adaptive_p : bool
        When ``True``, *p* is adjusted dynamically at each step based on the
        entropy of the logit distribution (see :class:`AdaptiveNucleus`).
    adaptive_p_min : float
        Lower bound for adaptive *p*.
    adaptive_p_max : float
        Upper bound for adaptive *p*.
    adaptive_entropy_threshold : float
        Entropy value that maps to the midpoint of the adaptive *p* range.
        Distributions with entropy above this value receive a *lower* p
        (tighter nucleus) while distributions below receive a *higher* p.
    typical_p : Optional[float]
        If set, the nucleus is *intersected* with the typical-sampling set
        (Meister et al., 2022).  ``None`` disables typical filtering.
    repetition_penalty : float
        Multiplicative penalty applied to logits of tokens that have already
        been generated.  1.0 means no penalty.
    no_repeat_ngram_size : int
        If > 0, any n-gram of this size that has already occurred in the
        generated sequence is forbidden from occurring again.
    """

    top_p: float = 0.9
    temperature: float = 1.0
    min_tokens_to_keep: int = 1
    filter_value: float = _NEG_INF
    adaptive_p: bool = False
    adaptive_p_min: float = 0.5
    adaptive_p_max: float = 0.95
    adaptive_entropy_threshold: float = 3.0
    typical_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


# =========================================================================
# Decoding state (extended for nucleus tracking)
# =========================================================================


@dataclass
class NucleusState(DecodingState):
    """Per-sequence decoding state augmented with nucleus-specific bookkeeping.

    Attributes
    ----------
    generated_ids : List[int]
        Token IDs generated so far.
    log_prob : float
        Cumulative log-probability of the generated sequence.
    entropies : List[float]
        Shannon entropy of the (pre-filtered) distribution at each step.
    p_values_used : List[float]
        The effective *p* used at each step (may differ from ``config.top_p``
        when adaptive mode is enabled).
    effective_vocab_sizes : List[int]
        Number of tokens remaining in the nucleus at each step.
    finished : bool
        Whether an EOS token has been sampled.
    """

    generated_ids: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    entropies: List[float] = field(default_factory=list)
    p_values_used: List[float] = field(default_factory=list)
    effective_vocab_sizes: List[int] = field(default_factory=list)
    finished: bool = False


# =========================================================================
# Helper functions (module-level, stateless)
# =========================================================================


def compute_cumulative_probs(
    logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sorted logits, their cumulative softmax probabilities, and the
    argsort indices.

    This is the fundamental primitive behind nucleus filtering.  We work in
    log-space for numerical stability: first compute log-softmax, then
    exponentiate.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector of shape ``(V,)`` where *V* is the vocabulary size.

    Returns
    -------
    sorted_logits : np.ndarray
        Logits sorted in **descending** order, shape ``(V,)``.
    cumulative_probs : np.ndarray
        Cumulative sum of softmax probabilities corresponding to the sorted
        logits, shape ``(V,)``.
    sorted_indices : np.ndarray
        Indices that sort the original logits in descending order so that
        ``logits[sorted_indices] == sorted_logits``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # Sort descending
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    # Numerically-stable softmax via log_softmax
    log_probs = log_softmax(sorted_logits)
    probs = np.exp(log_probs)
    cumulative_probs = np.cumsum(probs)
    return sorted_logits, cumulative_probs, sorted_indices


def nucleus_filter(
    logits: np.ndarray,
    p: float,
    min_keep: int = 1,
    filter_value: float = _NEG_INF,
) -> np.ndarray:
    """Apply nucleus (top-p) filtering to a logit vector.

    Tokens outside the smallest set whose cumulative probability exceeds *p*
    are masked with ``filter_value``.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector of shape ``(V,)``.
    p : float
        Cumulative probability threshold in (0, 1].
    min_keep : int
        Minimum number of tokens to retain.
    filter_value : float
        Value assigned to masked positions (default ``-inf``).

    Returns
    -------
    np.ndarray
        Filtered logit vector of the same shape.  Tokens outside the nucleus
        have value ``filter_value``; tokens inside retain their original logit
        values.
    """
    if p >= 1.0:
        return logits.copy()
    logits = np.asarray(logits, dtype=np.float64)
    V = logits.shape[0]
    min_keep = max(1, min(min_keep, V))

    sorted_logits, cumulative_probs, sorted_indices = compute_cumulative_probs(logits)

    # Determine which sorted positions to *remove*.  We shift cumulative_probs
    # right by one so that we keep the token that crosses the threshold.
    sorted_mask = np.zeros(V, dtype=bool)
    # Positions where cumulative prob (of *previous* positions) already >= p
    # are candidates for removal.
    cumulative_shifted = np.concatenate([[0.0], cumulative_probs[:-1]])
    sorted_mask = cumulative_shifted >= p

    # Always keep at least ``min_keep`` tokens (the top-scoring ones).
    sorted_mask[:min_keep] = False

    # Build the mask in the original vocabulary order.
    filtered = logits.copy()
    remove_indices = sorted_indices[sorted_mask]
    filtered[remove_indices] = filter_value

    return filtered


def renormalize_logits(
    logits: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Zero out masked positions and renormalize remaining logits.

    Parameters
    ----------
    logits : np.ndarray
        Logit vector, shape ``(V,)``.
    mask : np.ndarray
        Boolean mask where ``True`` means the position should be **kept**.

    Returns
    -------
    np.ndarray
        Logit vector with masked positions set to ``-inf`` and unmasked
        positions adjusted so that their softmax sums to 1.  (Technically
        softmax always sums to 1, but setting masked positions to ``-inf``
        effectively removes them.)
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    logits[~mask] = _NEG_INF
    # No further arithmetic needed — softmax of the remaining finite entries
    # will naturally sum to 1.  We simply return the modified logits.
    return logits


def sample_with_temperature(
    logits: np.ndarray,
    temperature: float = 1.0,
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Sample token indices from a logit vector with temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Logit vector, shape ``(V,)``.  May contain ``-inf`` for masked tokens.
    temperature : float
        Temperature divisor applied to logits before softmax.  Must be > 0.
    n : int
        Number of independent samples to draw.
    rng : np.random.Generator, optional
        Numpy random number generator.  If ``None``, a default is created.

    Returns
    -------
    List[int]
        Sampled token indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits / temperature
    log_probs = log_softmax(scaled)
    probs = np.exp(log_probs)
    # Clip tiny negatives that might arise from floating-point arithmetic.
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        # Fallback: uniform over non-masked tokens.
        finite_mask = np.isfinite(logits)
        probs = np.where(finite_mask, 1.0, 0.0)
        total = probs.sum()
        if total <= 0:
            probs = np.ones_like(logits) / len(logits)
        else:
            probs /= total
    else:
        probs /= total
    indices = rng.choice(len(probs), size=n, p=probs, replace=True)
    return indices.tolist()


def batch_nucleus_filter(
    logits_batch: np.ndarray,
    p: float,
    min_keep: int = 1,
    filter_value: float = _NEG_INF,
) -> np.ndarray:
    """Apply nucleus filtering to a batch of logit vectors.

    Parameters
    ----------
    logits_batch : np.ndarray
        Logit matrix of shape ``(B, V)`` where *B* is the batch size.
    p : float
        Cumulative probability threshold.
    min_keep : int
        Minimum tokens to retain per row.
    filter_value : float
        Value for masked positions.

    Returns
    -------
    np.ndarray
        Filtered logit matrix of the same shape.
    """
    logits_batch = np.asarray(logits_batch, dtype=np.float64)
    if logits_batch.ndim == 1:
        return nucleus_filter(logits_batch, p, min_keep, filter_value)

    B, V = logits_batch.shape
    min_keep = max(1, min(min_keep, V))

    # Sort each row in descending order.
    sorted_indices = np.argsort(logits_batch, axis=1)[:, ::-1]  # (B, V)
    sorted_logits = np.take_along_axis(logits_batch, sorted_indices, axis=1)

    # Softmax per row (numerically stable via log_softmax).
    log_probs = log_softmax(sorted_logits, axis=1)
    probs = np.exp(log_probs)
    cumulative_probs = np.cumsum(probs, axis=1)

    # Shift cumulative probs right so we keep the crossing token.
    cumulative_shifted = np.concatenate(
        [np.zeros((B, 1), dtype=np.float64), cumulative_probs[:, :-1]], axis=1
    )

    sorted_mask = cumulative_shifted >= p  # True → remove
    sorted_mask[:, :min_keep] = False  # always keep top min_keep

    # Scatter mask back to original order.
    original_mask = np.zeros_like(sorted_mask)
    rows = np.arange(B)[:, None]
    original_mask[rows, sorted_indices] = sorted_mask

    result = logits_batch.copy()
    result[original_mask] = filter_value
    return result


# =========================================================================
# Main algorithm: NucleusSampling
# =========================================================================


class NucleusSampling(DecodingAlgorithm):
    """Nucleus (top-p) sampling decoding algorithm.

    At each generation step the algorithm:

    1. Obtains logits from the ``logit_source``.
    2. Applies repetition penalty and n-gram blocking (if configured).
    3. Scales logits by ``1 / temperature``.
    4. Sorts tokens by descending logit value.
    5. Computes cumulative softmax probabilities.
    6. Finds the smallest prefix whose cumulative probability ≥ *p*.
    7. Masks all tokens outside this prefix.
    8. Optionally intersects with the *typical* set.
    9. Re-samples from the filtered distribution.

    The ``generate`` method produces ``num_sequences`` independent samples,
    and ``generate_with_tracking`` additionally records per-step diagnostics.
    """

    # ---- construction ---------------------------------------------------

    def __init__(
        self,
        config: Optional[NucleusConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config: NucleusConfig = config or NucleusConfig()
        self.rng = np.random.default_rng(seed)
        self._adaptive: Optional[AdaptiveNucleus] = None
        if self.config.adaptive_p:
            self._adaptive = AdaptiveNucleus(self.config)
        logger.debug(
            "NucleusSampling initialised with top_p=%.3f, temperature=%.3f",
            self.config.top_p,
            self.config.temperature,
        )

    # ---- DecodingAlgorithm interface ------------------------------------

    def generate(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        num_sequences: int = 1,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[int]]:
        """Generate ``num_sequences`` token sequences via nucleus sampling.

        Parameters
        ----------
        logit_source : LogitSource
            Callable / object that yields the next-token logit vector given a
            list of token IDs.  Must support ``logit_source(token_ids)`` →
            ``np.ndarray`` of shape ``(V,)`` **or** provide a
            ``get_logits(token_ids)`` method.
        prompt_ids : List[int]
            Token IDs of the prompt (context).
        max_new_tokens : int
            Maximum number of tokens to generate per sequence.
        num_sequences : int
            Number of independent sequences to produce.
        eos_token_id : int, optional
            If the sampled token equals this value the sequence terminates
            early.

        Returns
        -------
        List[List[int]]
            Generated token ID sequences (prompt **not** included).
        """
        sequences: List[List[int]] = []
        for seq_idx in range(num_sequences):
            state = NucleusState()
            context = list(prompt_ids)
            for step in range(max_new_tokens):
                logits = self._get_logits(logit_source, context)
                logits = self._apply_repetition_penalty(logits, state.generated_ids)
                logits = self._apply_no_repeat_ngram(
                    logits, state.generated_ids, prompt_ids
                )
                token_id = self._step(logits, state)
                state.generated_ids.append(token_id)
                context.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    state.finished = True
                    break
            sequences.append(state.generated_ids)
            logger.debug(
                "Sequence %d/%d: %d tokens generated (finished=%s)",
                seq_idx + 1,
                num_sequences,
                len(state.generated_ids),
                state.finished,
            )
        return sequences

    # ---- core step ------------------------------------------------------

    def _step(self, logits: np.ndarray, state: NucleusState) -> int:
        """Execute a single nucleus-sampling decoding step.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector from the model, shape ``(V,)``.
        state : NucleusState
            Current decoding state (mutated in-place with tracking info).

        Returns
        -------
        int
            Sampled token ID.
        """
        logits = np.asarray(logits, dtype=np.float64)

        # Record pre-filter entropy.
        ent = entropy_of_step(logits)
        state.entropies.append(ent)

        # Determine effective p.
        if self._adaptive is not None:
            p = self._adaptive_p(logits, state)
        else:
            p = self.config.top_p
        state.p_values_used.append(p)

        # Temperature scaling.
        if self.config.temperature != 1.0 and self.config.temperature > 0:
            logits = logits / self.config.temperature

        # Nucleus filtering.
        filtered = self._nucleus_filter(logits, p)

        # Optional typical-p intersection.
        if self.config.typical_p is not None:
            filtered = self._combined_filter(
                filtered, p, self.config.typical_p
            )

        # Record effective vocab size.
        evs = effective_vocab_size(filtered, p=1.0)
        state.effective_vocab_sizes.append(evs)

        # Sample.
        token_id = self._sample_from_filtered(filtered, n_samples=1)[0]

        # Accumulate log-probability.
        log_probs = log_softmax(filtered)
        if np.isfinite(log_probs[token_id]):
            state.log_prob += float(log_probs[token_id])

        return int(token_id)

    # ---- nucleus filtering core -----------------------------------------

    def _nucleus_filter(self, logits: np.ndarray, p: float) -> np.ndarray:
        """Core top-p filtering.

        Algorithm
        ---------
        1. Sort logits in descending order.
        2. Compute cumulative softmax probabilities over the sorted logits.
        3. Find the cutoff index *k* such that the cumulative probability of
           the first *k* tokens is ≥ *p*.
        4. Mask (set to ``-inf``) all tokens with rank > *k*.
        5. Return the logits in their original vocabulary order with
           renormalized values (via masking; softmax of unmasked entries
           automatically sums to 1).

        Parameters
        ----------
        logits : np.ndarray
            Logit vector, shape ``(V,)``.
        p : float
            Cumulative probability threshold.

        Returns
        -------
        np.ndarray
            Filtered logit vector.
        """
        return nucleus_filter(
            logits,
            p,
            min_keep=self.config.min_tokens_to_keep,
            filter_value=self.config.filter_value,
        )

    # ---- adaptive p -----------------------------------------------------

    def _adaptive_p(self, logits: np.ndarray, state: NucleusState) -> float:
        """Compute an adaptive *p* based on the entropy of the current
        distribution.

        When the model is highly uncertain (high entropy), we *lower* p to
        restrict the nucleus and avoid sampling from a large, noisy tail.
        When the model is confident (low entropy), we *raise* p to allow more
        diversity — the model's tail tokens are meaningful but rare.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector.
        state : NucleusState
            Current state (used for historical smoothing).

        Returns
        -------
        float
            Adaptive *p* value.
        """
        if self._adaptive is None:
            return self.config.top_p
        return self._adaptive.compute_p(logits, state)

    # ---- combined filter (nucleus + typical) ----------------------------

    def _combined_filter(
        self,
        logits: np.ndarray,
        top_p: float,
        typical_p: Optional[float],
    ) -> np.ndarray:
        """Intersect nucleus and typical filtering.

        Typical sampling (Meister et al., 2022) selects tokens whose
        information content (negative log-probability) is close to the
        expected information content (entropy).  Combining it with nucleus
        sampling keeps only tokens that are both *probable enough* (nucleus)
        and *typical enough* (typical), yielding higher quality samples.

        Parameters
        ----------
        logits : np.ndarray
            Already nucleus-filtered logits.
        top_p : float
            Nucleus cumulative probability (not used directly here since
            ``logits`` are already filtered, but kept for API symmetry).
        typical_p : Optional[float]
            Typical-sampling threshold.  If ``None``, no typical filter is
            applied.

        Returns
        -------
        np.ndarray
            Logits with the intersection mask applied.
        """
        if typical_p is None or typical_p >= 1.0:
            return logits

        logits = np.asarray(logits, dtype=np.float64).copy()
        log_probs = log_softmax(logits)
        probs = np.exp(log_probs)

        # Entropy of the (already filtered) distribution.
        ent = -np.sum(probs * log_probs, where=np.isfinite(log_probs))

        # Information content (surprisal) of each token.
        surprisal = -log_probs

        # Absolute deviation from the expected information content.
        deviation = np.abs(surprisal - ent)

        # Sort by deviation (ascending — most "typical" first).
        sorted_idx = np.argsort(deviation)

        # Cumulative probability in deviation order.
        cum_prob = np.cumsum(probs[sorted_idx])

        # Find cutoff.
        cutoff = np.searchsorted(cum_prob, typical_p, side="right")
        cutoff = max(cutoff, self.config.min_tokens_to_keep)

        # Mask tokens beyond the cutoff.
        keep_set = set(sorted_idx[:cutoff].tolist())
        for i in range(len(logits)):
            if i not in keep_set:
                logits[i] = self.config.filter_value

        return logits

    # ---- sampling -------------------------------------------------------

    def _sample_from_filtered(
        self,
        logits: np.ndarray,
        n_samples: int = 1,
    ) -> List[int]:
        """Draw samples from a filtered logit distribution.

        Parameters
        ----------
        logits : np.ndarray
            Filtered logit vector (masked positions are ``-inf``).
        n_samples : int
            Number of samples.

        Returns
        -------
        List[int]
            Sampled token IDs.
        """
        return sample_with_temperature(
            logits, temperature=1.0, n=n_samples, rng=self.rng
        )

    # ---- generate with tracking -----------------------------------------

    def generate_with_tracking(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        num_sequences: int = 1,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        """Generate sequences **and** return per-step diagnostic metadata.

        For every generated sequence the metadata dictionary contains:

        * ``entropies`` — Shannon entropy at each step *before* filtering.
        * ``effective_vocab_sizes`` — number of tokens in the nucleus at each
          step.
        * ``p_values_used`` — the *p* value actually used (may vary under
          adaptive mode).
        * ``log_prob`` — cumulative log-probability of the sequence.
        * ``length`` — number of generated tokens.
        * ``finished`` — whether an EOS was encountered.

        Parameters
        ----------
        logit_source : LogitSource
            Next-token logit provider.
        prompt_ids : List[int]
            Prompt token IDs.
        max_new_tokens : int
            Maximum generation length.
        num_sequences : int
            Number of independent sequences.
        eos_token_id : int, optional
            Early-stop token.

        Returns
        -------
        sequences : List[List[int]]
            Generated token sequences.
        metadata : List[Dict[str, Any]]
            Per-sequence tracking dictionaries.
        """
        all_sequences: List[List[int]] = []
        all_metadata: List[Dict[str, Any]] = []

        for seq_idx in range(num_sequences):
            state = NucleusState()
            context = list(prompt_ids)

            for step in range(max_new_tokens):
                logits = self._get_logits(logit_source, context)
                logits = self._apply_repetition_penalty(logits, state.generated_ids)
                logits = self._apply_no_repeat_ngram(
                    logits, state.generated_ids, prompt_ids
                )
                token_id = self._step(logits, state)
                state.generated_ids.append(token_id)
                context.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    state.finished = True
                    break

            all_sequences.append(state.generated_ids)
            all_metadata.append(
                {
                    "entropies": list(state.entropies),
                    "effective_vocab_sizes": list(state.effective_vocab_sizes),
                    "p_values_used": list(state.p_values_used),
                    "log_prob": state.log_prob,
                    "length": len(state.generated_ids),
                    "finished": state.finished,
                }
            )
            logger.debug(
                "Tracked sequence %d/%d: length=%d, log_prob=%.4f, "
                "mean_entropy=%.4f, mean_evs=%.1f",
                seq_idx + 1,
                num_sequences,
                len(state.generated_ids),
                state.log_prob,
                float(np.mean(state.entropies)) if state.entropies else 0.0,
                float(np.mean(state.effective_vocab_sizes))
                if state.effective_vocab_sizes
                else 0.0,
            )

        return all_sequences, all_metadata

    # ---- repetition penalty & n-gram blocking ---------------------------

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
    ) -> np.ndarray:
        """Apply multiplicative repetition penalty.

        For every token ID that appears in ``generated_ids``, its logit is
        divided by ``repetition_penalty`` if positive, or multiplied if
        negative.  This discourages (penalty > 1) or encourages (penalty < 1)
        repetition.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector, shape ``(V,)``.
        generated_ids : List[int]
            Tokens already generated.

        Returns
        -------
        np.ndarray
            Penalised logit vector.
        """
        penalty = self.config.repetition_penalty
        if penalty == 1.0 or not generated_ids:
            return logits
        logits = logits.copy()
        unique_ids = set(generated_ids)
        for tid in unique_ids:
            if 0 <= tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= penalty
                else:
                    logits[tid] *= penalty
        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        prompt_ids: List[int],
    ) -> np.ndarray:
        """Block n-grams that have already appeared.

        For a configured ``no_repeat_ngram_size`` of *n*, look at the last
        *n - 1* generated tokens and find all continuations that would form a
        previously-seen n-gram.  Set those logits to ``-inf``.

        Parameters
        ----------
        logits : np.ndarray
            Logit vector.
        generated_ids : List[int]
            Tokens generated so far.
        prompt_ids : List[int]
            Prompt tokens (n-grams in the prompt also count).

        Returns
        -------
        np.ndarray
            Logit vector with forbidden continuations masked.
        """
        n = self.config.no_repeat_ngram_size
        if n <= 0:
            return logits

        full_seq = list(prompt_ids) + list(generated_ids)
        if len(full_seq) < n:
            return logits

        logits = logits.copy()
        # Build set of all n-grams seen so far.
        ngram_set: Dict[Tuple[int, ...], set] = {}
        for i in range(len(full_seq) - n + 1):
            prefix = tuple(full_seq[i : i + n - 1])
            continuation = full_seq[i + n - 1]
            ngram_set.setdefault(prefix, set()).add(continuation)

        # Look at the last n-1 tokens to find what continuations to block.
        if len(full_seq) >= n - 1:
            last_prefix = tuple(full_seq[-(n - 1) :])
            banned = ngram_set.get(last_prefix, set())
            for tid in banned:
                if 0 <= tid < len(logits):
                    logits[tid] = _NEG_INF

        return logits

    # ---- logit source adapter -------------------------------------------

    @staticmethod
    def _get_logits(logit_source: Any, token_ids: List[int]) -> np.ndarray:
        """Obtain logits from a generic logit source.

        Supports callables, objects with a ``get_logits`` method, or objects
        with a ``__call__`` method.

        Parameters
        ----------
        logit_source : Any
            Provider of next-token logit distributions.
        token_ids : List[int]
            Full token sequence (prompt + generated so far).

        Returns
        -------
        np.ndarray
            Logit vector, shape ``(V,)``.
        """
        if hasattr(logit_source, "get_logits"):
            result = logit_source.get_logits(token_ids)
        elif callable(logit_source):
            result = logit_source(token_ids)
        else:
            raise TypeError(
                f"logit_source must be callable or have a get_logits method, "
                f"got {type(logit_source)}"
            )
        return np.asarray(result, dtype=np.float64)

    # ---- utility methods on the instance --------------------------------

    @staticmethod
    def entropy_of_step(logits: np.ndarray) -> float:
        """Compute Shannon entropy of a logit distribution.

        Uses the identity ``H = log Z - (1/Z) * sum(x_i * exp(x_i))`` via
        log-softmax for numerical stability.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector.

        Returns
        -------
        float
            Entropy in nats.
        """
        return entropy_of_step(logits)

    @staticmethod
    def effective_vocab_size(logits: np.ndarray, p: float = 0.9) -> int:
        """Number of tokens whose cumulative probability fits within *p*.

        Parameters
        ----------
        logits : np.ndarray
            Logit vector (may already be filtered).
        p : float
            Cumulative probability threshold.

        Returns
        -------
        int
            Count of tokens in the nucleus.
        """
        return effective_vocab_size(logits, p)


# =========================================================================
# Module-level entropy / effective vocab helpers
# =========================================================================


def entropy_of_step(logits: np.ndarray) -> float:
    """Compute Shannon entropy of a logit distribution (in nats).

    Uses log-softmax for numerical stability::

        H(p) = -sum_i p_i log p_i = -sum_i exp(lp_i) * lp_i

    where ``lp = log_softmax(logits)``.

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector, shape ``(V,)``.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    logits = np.asarray(logits, dtype=np.float64)
    lp = log_softmax(logits)
    p = np.exp(lp)
    # Avoid nan from 0 * -inf.
    ent = -np.sum(p * lp, where=np.isfinite(lp))
    return float(ent)


def effective_vocab_size(logits: np.ndarray, p: float = 0.9) -> int:
    """Count the number of tokens whose cumulative probability ≤ *p*.

    Parameters
    ----------
    logits : np.ndarray
        Logit vector, shape ``(V,)``.
    p : float
        Cumulative probability threshold.

    Returns
    -------
    int
        Size of the nucleus.
    """
    logits = np.asarray(logits, dtype=np.float64)
    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        return 0
    lp = log_softmax(logits)
    probs = np.exp(lp)
    sorted_probs = np.sort(probs)[::-1]
    cum = np.cumsum(sorted_probs)
    count = int(np.searchsorted(cum, p, side="right")) + 1
    return min(count, int(np.sum(finite_mask)))


# =========================================================================
# NucleusAnalyzer
# =========================================================================


class NucleusAnalyzer:
    """Analytical utilities for studying nucleus sampling behaviour.

    All methods are stateless; they accept a ``logit_source`` and prompt and
    return summary statistics or trajectories.
    """

    def __init__(
        self,
        config: Optional[NucleusConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or NucleusConfig()
        self.rng = np.random.default_rng(seed)

    # ---- p sensitivity --------------------------------------------------

    def analyze_p_sensitivity(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        p_values: Optional[List[float]] = None,
        max_new_tokens: int = 64,
        num_sequences: int = 5,
    ) -> Dict[str, Any]:
        """Sweep *p* and measure how it affects generation diversity and
        quality.

        For each value of *p* the method generates ``num_sequences`` samples
        and computes:

        * Mean sequence length.
        * Mean log-probability.
        * Mean per-step entropy.
        * Mean effective vocabulary size.
        * Distinct-1 and distinct-2 ratios (type/token at unigram and bigram
          level).

        Parameters
        ----------
        logit_source : Any
            Logit provider.
        prompt_ids : List[int]
            Prompt tokens.
        p_values : List[float], optional
            Values of *p* to evaluate.  Defaults to
            ``[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]``.
        max_new_tokens : int
            Max generation length.
        num_sequences : int
            Samples per *p* value.

        Returns
        -------
        dict
            Keys: ``p_values``, ``mean_lengths``, ``mean_log_probs``,
            ``mean_entropies``, ``mean_evs``, ``distinct1``, ``distinct2``.
        """
        if p_values is None:
            p_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

        results: Dict[str, list] = {
            "p_values": list(p_values),
            "mean_lengths": [],
            "mean_log_probs": [],
            "mean_entropies": [],
            "mean_evs": [],
            "distinct1": [],
            "distinct2": [],
        }

        for p in p_values:
            cfg = NucleusConfig(
                top_p=p,
                temperature=self.config.temperature,
                min_tokens_to_keep=self.config.min_tokens_to_keep,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            )
            sampler = NucleusSampling(cfg, seed=self.rng.integers(0, 2**31))
            seqs, metas = sampler.generate_with_tracking(
                logit_source,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                num_sequences=num_sequences,
            )

            lengths = [m["length"] for m in metas]
            log_probs = [m["log_prob"] for m in metas]
            entropies = [
                float(np.mean(m["entropies"])) if m["entropies"] else 0.0
                for m in metas
            ]
            evs_vals = [
                float(np.mean(m["effective_vocab_sizes"]))
                if m["effective_vocab_sizes"]
                else 0.0
                for m in metas
            ]
            d1, d2 = self._distinct_n(seqs)

            results["mean_lengths"].append(float(np.mean(lengths)))
            results["mean_log_probs"].append(float(np.mean(log_probs)))
            results["mean_entropies"].append(float(np.mean(entropies)))
            results["mean_evs"].append(float(np.mean(evs_vals)))
            results["distinct1"].append(d1)
            results["distinct2"].append(d2)

            logger.info(
                "p=%.2f  len=%.1f  lp=%.2f  H=%.3f  evs=%.1f  d1=%.3f  d2=%.3f",
                p,
                results["mean_lengths"][-1],
                results["mean_log_probs"][-1],
                results["mean_entropies"][-1],
                results["mean_evs"][-1],
                d1,
                d2,
            )

        return results

    # ---- entropy trajectory ---------------------------------------------

    def compute_entropy_trajectory(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
    ) -> List[float]:
        """Generate one sequence and record the entropy at each step.

        Parameters
        ----------
        logit_source : Any
            Logit provider.
        prompt_ids : List[int]
            Prompt tokens.
        max_new_tokens : int
            Generation length.

        Returns
        -------
        List[float]
            Per-step entropy values (nats).
        """
        sampler = NucleusSampling(self.config, seed=self.rng.integers(0, 2**31))
        _, metas = sampler.generate_with_tracking(
            logit_source,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            num_sequences=1,
        )
        return metas[0]["entropies"]

    # ---- optimal p estimation -------------------------------------------

    def estimate_optimal_p(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        target_entropy: float = 2.0,
        max_new_tokens: int = 64,
        search_resolution: int = 20,
    ) -> float:
        """Estimate the value of *p* that yields a target mean per-step
        entropy in the filtered distribution.

        Uses a simple grid search over ``[0.05, 1.0]``.

        Parameters
        ----------
        logit_source : Any
            Logit provider.
        prompt_ids : List[int]
            Prompt.
        target_entropy : float
            Desired mean entropy (nats).
        max_new_tokens : int
            Generation length for each candidate.
        search_resolution : int
            Number of grid points.

        Returns
        -------
        float
            Best *p* value found.
        """
        p_candidates = np.linspace(0.05, 1.0, search_resolution).tolist()
        best_p = self.config.top_p
        best_diff = float("inf")

        for p in p_candidates:
            cfg = NucleusConfig(
                top_p=p,
                temperature=self.config.temperature,
                min_tokens_to_keep=self.config.min_tokens_to_keep,
            )
            sampler = NucleusSampling(cfg, seed=self.rng.integers(0, 2**31))
            _, metas = sampler.generate_with_tracking(
                logit_source,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                num_sequences=1,
            )
            if not metas[0]["entropies"]:
                continue

            # Compute entropy of the *filtered* distribution at each step.
            # The tracked entropies are pre-filter; we approximate with
            # effective-vocab-based estimate:
            #   H_filtered ≈ log(evs)  (maximum entropy over evs tokens)
            # A better estimate uses the actual filtered logits, but this is
            # efficient for a grid search.
            evs_list = metas[0]["effective_vocab_sizes"]
            if not evs_list:
                continue
            mean_filtered_ent = float(
                np.mean([math.log(max(e, 1)) for e in evs_list])
            )
            diff = abs(mean_filtered_ent - target_entropy)
            if diff < best_diff:
                best_diff = diff
                best_p = p

        logger.info(
            "Estimated optimal p=%.3f for target_entropy=%.2f (diff=%.4f)",
            best_p,
            target_entropy,
            best_diff,
        )
        return best_p

    # ---- nucleus size distribution --------------------------------------

    def nucleus_size_distribution(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        p: float = 0.9,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Compute statistics of the nucleus size across generation steps.

        Parameters
        ----------
        logit_source : Any
            Logit provider.
        prompt_ids : List[int]
            Prompt.
        p : float
            Nucleus threshold.
        max_new_tokens : int
            Generation length.

        Returns
        -------
        dict
            ``sizes`` (per-step list), ``mean``, ``median``, ``std``,
            ``min``, ``max``, ``p`` used.
        """
        cfg = NucleusConfig(
            top_p=p,
            temperature=self.config.temperature,
            min_tokens_to_keep=self.config.min_tokens_to_keep,
        )
        sampler = NucleusSampling(cfg, seed=self.rng.integers(0, 2**31))
        _, metas = sampler.generate_with_tracking(
            logit_source,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            num_sequences=1,
        )
        sizes = metas[0]["effective_vocab_sizes"]
        arr = np.array(sizes, dtype=np.float64) if sizes else np.array([0.0])
        return {
            "sizes": sizes,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "p": p,
        }

    # ---- diversity vs p -------------------------------------------------

    def diversity_vs_p(
        self,
        logit_source: Any,
        prompt_ids: List[int],
        p_values: Optional[List[float]] = None,
        n_samples: int = 10,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Measure diversity metrics as a function of *p*.

        For each *p*, generates ``n_samples`` sequences and computes pairwise
        Jaccard distance, distinct-n, and self-BLEU-2 (simplified).

        Parameters
        ----------
        logit_source : Any
            Logit provider.
        prompt_ids : List[int]
            Prompt.
        p_values : List[float], optional
            Values of *p* to sweep.
        n_samples : int
            Sequences per *p*.
        max_new_tokens : int
            Generation length.

        Returns
        -------
        dict
            ``p_values``, ``jaccard``, ``distinct1``, ``distinct2``,
            ``self_bleu2``.
        """
        if p_values is None:
            p_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

        results: Dict[str, list] = {
            "p_values": list(p_values),
            "jaccard": [],
            "distinct1": [],
            "distinct2": [],
            "self_bleu2": [],
        }

        for p in p_values:
            cfg = NucleusConfig(
                top_p=p,
                temperature=self.config.temperature,
                min_tokens_to_keep=self.config.min_tokens_to_keep,
            )
            sampler = NucleusSampling(cfg, seed=self.rng.integers(0, 2**31))
            seqs = sampler.generate(
                logit_source,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                num_sequences=n_samples,
            )

            d1, d2 = self._distinct_n(seqs)
            jac = self._mean_pairwise_jaccard(seqs)
            sb2 = self._self_bleu_2(seqs)

            results["jaccard"].append(jac)
            results["distinct1"].append(d1)
            results["distinct2"].append(d2)
            results["self_bleu2"].append(sb2)

        return results

    # ---- internal diversity helpers -------------------------------------

    @staticmethod
    def _distinct_n(sequences: List[List[int]]) -> Tuple[float, float]:
        """Compute distinct-1 and distinct-2 ratios over a set of sequences.

        Distinct-n = (# unique n-grams) / (# total n-grams) across all
        sequences.

        Parameters
        ----------
        sequences : List[List[int]]
            Generated token sequences.

        Returns
        -------
        Tuple[float, float]
            (distinct-1, distinct-2).
        """
        all_unigrams: List[int] = []
        all_bigrams: List[Tuple[int, int]] = []
        for seq in sequences:
            all_unigrams.extend(seq)
            for i in range(len(seq) - 1):
                all_bigrams.append((seq[i], seq[i + 1]))
        d1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
        d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
        return d1, d2

    @staticmethod
    def _mean_pairwise_jaccard(sequences: List[List[int]]) -> float:
        """Mean pairwise Jaccard *distance* (1 - IoU) over sequences.

        Parameters
        ----------
        sequences : List[List[int]]
            Generated token sequences.

        Returns
        -------
        float
            Mean Jaccard distance in [0, 1].
        """
        if len(sequences) < 2:
            return 0.0
        sets = [set(s) for s in sequences]
        dists: List[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                if union == 0:
                    dists.append(0.0)
                else:
                    dists.append(1.0 - inter / union)
        return float(np.mean(dists))

    @staticmethod
    def _self_bleu_2(sequences: List[List[int]]) -> float:
        """Simplified self-BLEU-2: for each sequence, compute BLEU-2 against
        every other sequence and average.

        Uses a simple bigram precision (no brevity penalty) for speed.

        Parameters
        ----------
        sequences : List[List[int]]
            Generated sequences.

        Returns
        -------
        float
            Mean self-BLEU-2 (lower → more diverse).
        """
        if len(sequences) < 2:
            return 0.0

        def _bigrams(seq: List[int]) -> Dict[Tuple[int, int], int]:
            counts: Dict[Tuple[int, int], int] = {}
            for i in range(len(seq) - 1):
                bg = (seq[i], seq[i + 1])
                counts[bg] = counts.get(bg, 0) + 1
            return counts

        scores: List[float] = []
        for i, hyp in enumerate(sequences):
            if len(hyp) < 2:
                scores.append(0.0)
                continue
            hyp_bg = _bigrams(hyp)
            # Pool reference bigrams from all other sequences.
            ref_bg: Dict[Tuple[int, int], int] = {}
            for j, ref in enumerate(sequences):
                if j == i:
                    continue
                for bg, c in _bigrams(ref).items():
                    ref_bg[bg] = ref_bg.get(bg, 0) + c
            # Clipped counts.
            clipped = sum(min(c, ref_bg.get(bg, 0)) for bg, c in hyp_bg.items())
            total = sum(hyp_bg.values())
            scores.append(clipped / max(total, 1))
        return float(np.mean(scores))


# =========================================================================
# AdaptiveNucleus
# =========================================================================


class AdaptiveNucleus:
    """Adaptive top-p controller that adjusts *p* based on local entropy.

    Strategy
    --------
    * High entropy (uncertain model) → **lower** *p* to avoid sampling from a
      large, noisy tail.  The model doesn't know what to say, so we restrict
      choices to the most probable options.
    * Low entropy (confident model) → **higher** *p* to permit more diversity.
      The model has a clear preference, so the tail tokens are meaningful
      alternatives worth exploring.

    The mapping from entropy to *p* uses a sigmoid-like function centred on
    ``config.adaptive_entropy_threshold``.  An exponential moving average
    (EMA) of recent entropies is used for smoothing.

    Parameters
    ----------
    config : NucleusConfig
        Configuration carrying adaptive bounds and threshold.
    ema_alpha : float
        Smoothing factor for the EMA (0 < α ≤ 1).  Smaller values → more
        smoothing.
    """

    def __init__(
        self,
        config: NucleusConfig,
        ema_alpha: float = 0.3,
    ) -> None:
        self.config = config
        self.ema_alpha = ema_alpha
        self._ema_entropy: Optional[float] = None

    def compute_p(
        self,
        logits: np.ndarray,
        state: NucleusState,
    ) -> float:
        """Compute the adaptive *p* for the current step.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector.
        state : NucleusState
            Current decoding state (used for EMA history).

        Returns
        -------
        float
            Adaptive *p* value in ``[adaptive_p_min, adaptive_p_max]``.
        """
        ent = entropy_of_step(logits)

        # Update EMA.
        if self._ema_entropy is None:
            self._ema_entropy = ent
        else:
            self._ema_entropy = self._exponential_moving_average(
                [self._ema_entropy, ent], self.ema_alpha
            )

        return self._compute_target_p(self._ema_entropy, self.config)

    @staticmethod
    def _compute_target_p(entropy: float, config: NucleusConfig) -> float:
        """Map an entropy value to a target *p*.

        Uses the *inverse* sigmoid (logistic) function so that:

        * ``entropy → -∞`` ⇒ ``p → adaptive_p_max`` (confident → broad
          nucleus).
        * ``entropy → +∞`` ⇒ ``p → adaptive_p_min`` (uncertain → narrow
          nucleus).
        * ``entropy == adaptive_entropy_threshold`` ⇒ ``p`` is at the
          midpoint.

        Parameters
        ----------
        entropy : float
            Current (possibly smoothed) entropy.
        config : NucleusConfig
            Configuration with bounds and threshold.

        Returns
        -------
        float
            Target *p*.
        """
        p_min = config.adaptive_p_min
        p_max = config.adaptive_p_max
        threshold = config.adaptive_entropy_threshold

        # Normalised deviation from threshold (positive → higher entropy).
        x = (entropy - threshold) / max(threshold, 1e-6)
        # Sigmoid → maps to (0, 1).  We invert so high entropy → low output.
        sigmoid = 1.0 / (1.0 + math.exp(min(x * 4.0, 500)))  # clamp arg
        # Scale to [p_min, p_max].
        p = p_min + (p_max - p_min) * sigmoid
        return float(np.clip(p, p_min, p_max))

    @staticmethod
    def _exponential_moving_average(
        values: Sequence[float],
        alpha: float,
    ) -> float:
        """Compute the EMA of a sequence of values.

        Parameters
        ----------
        values : Sequence[float]
            Ordered values (oldest first).
        alpha : float
            Smoothing factor (0 < α ≤ 1).

        Returns
        -------
        float
            EMA value.
        """
        if not values:
            return 0.0
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1.0 - alpha) * ema
        return float(ema)


# =========================================================================
# TemperatureSweepNucleus
# =========================================================================


class TemperatureSweepNucleus:
    """Generate sequences across a sweep of temperatures using nucleus sampling.

    This utility wraps a :class:`NucleusSampling` instance and systematically
    explores a set of temperature values, collecting generation results and
    diversity statistics for each temperature point.  It is useful for
    empirically determining the temperature that maximises diversity while
    retaining coherence.

    Parameters
    ----------
    base_config : NucleusConfig
        Base configuration.  The ``temperature`` field will be overridden for
        each sweep point.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_config: Optional[NucleusConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.base_config = base_config or NucleusConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def sweep(
        self,
        logit_source: Callable[..., np.ndarray],
        prompt_ids: List[int],
        temperatures: List[float],
        num_sequences: int = 5,
        max_length: int = 64,
    ) -> Dict[float, List[List[int]]]:
        """Run nucleus generation at each temperature and collect outputs.

        Parameters
        ----------
        logit_source : Callable
            Function that accepts token IDs and returns logit vectors.
        prompt_ids : List[int]
            Prompt token IDs to condition on.
        temperatures : List[float]
            Temperatures to sweep over.
        num_sequences : int
            Number of sequences to generate per temperature.
        max_length : int
            Maximum generation length per sequence.

        Returns
        -------
        Dict[float, List[List[int]]]
            Mapping from temperature to generated sequences.
        """
        results: Dict[float, List[List[int]]] = {}
        for temp in temperatures:
            cfg = NucleusConfig(
                top_p=self.base_config.top_p,
                temperature=temp,
                min_tokens_to_keep=self.base_config.min_tokens_to_keep,
                filter_value=self.base_config.filter_value,
                adaptive_p=self.base_config.adaptive_p,
                adaptive_p_min=self.base_config.adaptive_p_min,
                adaptive_p_max=self.base_config.adaptive_p_max,
                adaptive_entropy_threshold=self.base_config.adaptive_entropy_threshold,
                typical_p=self.base_config.typical_p,
                repetition_penalty=self.base_config.repetition_penalty,
                no_repeat_ngram_size=self.base_config.no_repeat_ngram_size,
            )
            sampler = NucleusSampling(config=cfg, seed=self.seed)
            seqs: List[List[int]] = []
            for _ in range(num_sequences):
                ids = list(prompt_ids)
                for _step in range(max_length):
                    logits = logit_source(ids)
                    filtered = nucleus_filter(
                        logits / temp, cfg.top_p, cfg.min_tokens_to_keep
                    )
                    token = sample_with_temperature(
                        filtered, temperature=1.0, rng=self._rng
                    )[0]
                    ids.append(token)
                seqs.append(ids[len(prompt_ids):])
            results[temp] = seqs
            logger.debug(
                "TemperatureSweepNucleus: temp=%.3f  generated %d sequences",
                temp,
                len(seqs),
            )
        return results

    def find_optimal_temperature(
        self,
        logit_source: Callable[..., np.ndarray],
        prompt_ids: List[int],
        temperatures: List[float],
        num_sequences: int = 10,
        max_length: int = 64,
    ) -> Tuple[float, Dict[float, float]]:
        """Find the temperature that maximises token-level diversity.

        Diversity is measured as the mean pairwise Jaccard distance between
        generated sequences (higher → more diverse).

        Parameters
        ----------
        logit_source : Callable
            Logit source function.
        prompt_ids : List[int]
            Prompt token IDs.
        temperatures : List[float]
            Candidate temperatures.
        num_sequences : int
            Sequences per temperature.
        max_length : int
            Maximum generation length.

        Returns
        -------
        best_temp : float
            Temperature with the highest diversity score.
        scores : Dict[float, float]
            Diversity score for each temperature.
        """
        sweep_results = self.sweep(
            logit_source, prompt_ids, temperatures, num_sequences, max_length
        )
        scores: Dict[float, float] = {}
        for temp, seqs in sweep_results.items():
            scores[temp] = batch_diversity_score(seqs)

        best_temp = max(scores, key=scores.get)  # type: ignore[arg-type]
        logger.info(
            "TemperatureSweepNucleus: optimal temperature=%.3f  score=%.4f",
            best_temp,
            scores[best_temp],
        )
        return best_temp, scores


# =========================================================================
# AdaptiveThresholdNucleus
# =========================================================================


class AdaptiveThresholdNucleus:
    """Auto-tuning nucleus threshold based on running entropy statistics.

    Maintains a sliding window of recent entropy values and uses them to
    dynamically adjust the *p* threshold.  When the model is uncertain (high
    entropy), the nucleus is narrowed (lower *p*) to avoid sampling from a
    noisy tail.  When the model is confident (low entropy), the nucleus is
    widened (higher *p*) to explore meaningful alternatives.

    Parameters
    ----------
    p_min : float
        Minimum allowed *p* value.
    p_max : float
        Maximum allowed *p* value.
    window_size : int
        Number of recent entropy values to maintain.
    entropy_low : float
        Entropy below which *p* is pushed toward ``p_max``.
    entropy_high : float
        Entropy above which *p* is pushed toward ``p_min``.
    smoothing_alpha : float
        Exponential smoothing factor (0 < α ≤ 1).
    """

    def __init__(
        self,
        p_min: float = 0.5,
        p_max: float = 0.95,
        window_size: int = 20,
        entropy_low: float = 1.5,
        entropy_high: float = 5.0,
        smoothing_alpha: float = 0.3,
    ) -> None:
        self.p_min = p_min
        self.p_max = p_max
        self.window_size = window_size
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.smoothing_alpha = smoothing_alpha
        self._entropy_window: List[float] = []
        self._current_p: float = (p_min + p_max) / 2.0

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def step(self, logits: np.ndarray) -> float:
        """Observe logits, update internal state, and return current *p*.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector of shape ``(V,)``.

        Returns
        -------
        float
            Updated *p* threshold.
        """
        ent = entropy_of_step(logits)
        self._entropy_window.append(ent)
        if len(self._entropy_window) > self.window_size:
            self._entropy_window = self._entropy_window[-self.window_size:]
        self._update_threshold()
        return self._current_p

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _update_threshold(self) -> None:
        """Recompute ``_current_p`` from the entropy window."""
        smoothed = self._smooth_entropy()
        self._current_p = self._entropy_to_p(smoothed)
        logger.debug(
            "AdaptiveThresholdNucleus: smoothed_entropy=%.4f  p=%.4f",
            smoothed,
            self._current_p,
        )

    def _entropy_to_p(self, entropy: float) -> float:
        """Map a (smoothed) entropy value to a *p* threshold.

        Uses linear interpolation between ``entropy_low`` and
        ``entropy_high``.  Values outside this range are clipped.

        Parameters
        ----------
        entropy : float
            Smoothed entropy value.

        Returns
        -------
        float
            Target *p* in ``[p_min, p_max]``.
        """
        span = self.entropy_high - self.entropy_low
        if span <= 0:
            return (self.p_min + self.p_max) / 2.0
        # Normalise entropy to [0, 1] where 0 → low, 1 → high.
        t = (entropy - self.entropy_low) / span
        t = float(np.clip(t, 0.0, 1.0))
        # High entropy → low p, low entropy → high p (inverted mapping).
        p = self.p_max - t * (self.p_max - self.p_min)
        return float(np.clip(p, self.p_min, self.p_max))

    def _smooth_entropy(self) -> float:
        """Return exponentially smoothed entropy from the window.

        Returns
        -------
        float
            Smoothed entropy value.
        """
        if not self._entropy_window:
            return 0.0
        ema = self._entropy_window[0]
        for v in self._entropy_window[1:]:
            ema = self.smoothing_alpha * v + (1.0 - self.smoothing_alpha) * ema
        return float(ema)


# =========================================================================
# NucleusWithPenalties
# =========================================================================


class NucleusWithPenalties:
    """Nucleus sampling augmented with repetition, frequency, and presence penalties.

    This class applies three orthogonal penalty mechanisms to the logit vector
    *before* nucleus filtering, discouraging the model from repeating tokens
    in undesirable ways.

    * **Repetition penalty** — multiplicative: logits of previously generated
      tokens are divided by ``repetition_penalty`` (if positive) or multiplied
      (if negative).
    * **Frequency penalty** — additive, proportional to the number of times
      each token has appeared in the generated sequence.
    * **Presence penalty** — additive, applied once to any token that has
      appeared at least once.

    Parameters
    ----------
    config : NucleusConfig
        Base nucleus configuration.
    repetition_penalty : float
        Multiplicative penalty (> 1 penalises repetition, 1.0 = no effect).
    frequency_penalty : float
        Per-occurrence additive penalty (≥ 0).
    presence_penalty : float
        One-time additive penalty for any token already seen (≥ 0).
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        config: Optional[NucleusConfig] = None,
        repetition_penalty: float = 1.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or NucleusConfig()
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def sample_step(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
    ) -> int:
        """Apply penalties, perform nucleus filtering, and sample a token.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector of shape ``(V,)``.
        generated_ids : List[int]
            Token IDs generated so far.

        Returns
        -------
        int
            Sampled token index.
        """
        logits = self._apply_all_penalties(logits, generated_ids)
        filtered = nucleus_filter(
            logits / self.config.temperature,
            self.config.top_p,
            self.config.min_tokens_to_keep,
        )
        return sample_with_temperature(filtered, temperature=1.0, rng=self._rng)[0]

    # -----------------------------------------------------------------
    # penalty helpers
    # -----------------------------------------------------------------

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
    ) -> np.ndarray:
        """Apply multiplicative repetition penalty.

        Parameters
        ----------
        logits : np.ndarray
            Logit vector (will be copied).
        generated_ids : List[int]
            Previously generated token IDs.

        Returns
        -------
        np.ndarray
            Penalised logit vector.
        """
        return compute_repetition_penalty_logits(
            logits, generated_ids, self.repetition_penalty
        )

    def _apply_frequency_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
    ) -> np.ndarray:
        """Apply additive frequency penalty proportional to token count.

        Parameters
        ----------
        logits : np.ndarray
            Logit vector (will be copied).
        generated_ids : List[int]
            Previously generated token IDs.

        Returns
        -------
        np.ndarray
            Penalised logit vector.
        """
        counts: Dict[int, int] = {}
        for tid in generated_ids:
            counts[tid] = counts.get(tid, 0) + 1
        return compute_frequency_penalty_logits(
            logits, counts, self.frequency_penalty
        )

    def _apply_presence_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
    ) -> np.ndarray:
        """Apply binary presence penalty (once per unique token).

        Parameters
        ----------
        logits : np.ndarray
            Logit vector (will be copied).
        generated_ids : List[int]
            Previously generated token IDs.

        Returns
        -------
        np.ndarray
            Penalised logit vector.
        """
        logits = logits.copy()
        seen = set(generated_ids)
        for tid in seen:
            if 0 <= tid < len(logits):
                logits[tid] -= self.presence_penalty
        return logits

    def _apply_all_penalties(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
    ) -> np.ndarray:
        """Chain all three penalties in order.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector.
        generated_ids : List[int]
            Previously generated token IDs.

        Returns
        -------
        np.ndarray
            Logit vector with all penalties applied.
        """
        logits = self._apply_repetition_penalty(logits, generated_ids)
        logits = self._apply_frequency_penalty(logits, generated_ids)
        logits = self._apply_presence_penalty(logits, generated_ids)
        return logits


# =========================================================================
# DiversityAwareBatchNucleus
# =========================================================================


class DiversityAwareBatchNucleus:
    """Batch nucleus generation with diversity-aware early stopping.

    Generates multiple sequences in parallel and monitors inter-sequence
    diversity during generation.  If diversity plateaus (stops improving),
    generation halts early to save compute.

    Diversity is measured via a combination of n-gram overlap, Jaccard
    distance between token sets, and (optionally) embedding-space distance.

    Parameters
    ----------
    config : NucleusConfig
        Base nucleus configuration.
    batch_size : int
        Number of sequences to generate in parallel.
    diversity_window : int
        Number of recent diversity measurements to consider for plateau
        detection.
    plateau_threshold : float
        If the relative improvement in diversity over the window is below
        this threshold, stop early.
    ngram_order : int
        N-gram order for overlap computation.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        config: Optional[NucleusConfig] = None,
        batch_size: int = 8,
        diversity_window: int = 5,
        plateau_threshold: float = 0.01,
        ngram_order: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or NucleusConfig()
        self.batch_size = batch_size
        self.diversity_window = diversity_window
        self.plateau_threshold = plateau_threshold
        self.ngram_order = ngram_order
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def generate_batch(
        self,
        logit_source: Callable[..., np.ndarray],
        prompt_ids: List[int],
        max_length: int = 64,
    ) -> Tuple[List[List[int]], List[float]]:
        """Generate a batch of sequences with diversity-aware early stopping.

        Parameters
        ----------
        logit_source : Callable
            Function that accepts token IDs and returns a logit vector.
        prompt_ids : List[int]
            Prompt token IDs.
        max_length : int
            Maximum number of tokens to generate per sequence.

        Returns
        -------
        sequences : List[List[int]]
            Generated token sequences (excluding the prompt).
        diversity_history : List[float]
            Diversity score at each generation step.
        """
        sequences: List[List[int]] = [[] for _ in range(self.batch_size)]
        contexts: List[List[int]] = [list(prompt_ids) for _ in range(self.batch_size)]
        diversity_history: List[float] = []

        for step in range(max_length):
            for i in range(self.batch_size):
                logits = logit_source(contexts[i])
                filtered = nucleus_filter(
                    logits / self.config.temperature,
                    self.config.top_p,
                    self.config.min_tokens_to_keep,
                )
                token = sample_with_temperature(
                    filtered, temperature=1.0, rng=self._rng
                )[0]
                sequences[i].append(token)
                contexts[i].append(token)

            div_score = self._compute_batch_diversity(sequences)
            diversity_history.append(div_score)

            if self._diversity_plateau_check(diversity_history):
                logger.info(
                    "DiversityAwareBatchNucleus: early stop at step %d "
                    "(diversity plateau, score=%.4f)",
                    step,
                    div_score,
                )
                break

        return sequences, diversity_history

    # -----------------------------------------------------------------
    # diversity computation
    # -----------------------------------------------------------------

    def _compute_batch_diversity(
        self,
        sequences: List[List[int]],
    ) -> float:
        """Compute a combined diversity score for the current batch.

        Averages Jaccard distance and (1 - n-gram overlap) across all pairs.

        Parameters
        ----------
        sequences : List[List[int]]
            Current (partial) sequences.

        Returns
        -------
        float
            Diversity score in [0, 1] (higher → more diverse).
        """
        if len(sequences) < 2:
            return 0.0

        jaccard = self._pairwise_jaccard(sequences)
        ngram_div = 1.0 - self._pairwise_ngram_overlap(sequences)
        return 0.5 * jaccard + 0.5 * ngram_div

    def _diversity_plateau_check(
        self,
        history: List[float],
    ) -> bool:
        """Check whether diversity has plateaued.

        Parameters
        ----------
        history : List[float]
            Diversity scores over time.

        Returns
        -------
        bool
            ``True`` if diversity improvement is below ``plateau_threshold``.
        """
        if len(history) < self.diversity_window + 1:
            return False
        recent = history[-self.diversity_window:]
        older = history[-(self.diversity_window + 1):-1]
        mean_recent = float(np.mean(recent))
        mean_older = float(np.mean(older))
        if mean_older <= _LOG_EPS:
            return False
        relative_improvement = (mean_recent - mean_older) / (abs(mean_older) + _LOG_EPS)
        return relative_improvement < self.plateau_threshold

    def _pairwise_ngram_overlap(
        self,
        sequences: List[List[int]],
    ) -> float:
        """Mean pairwise n-gram overlap across sequences.

        Parameters
        ----------
        sequences : List[List[int]]
            Token sequences.

        Returns
        -------
        float
            Mean overlap in [0, 1] (lower → more diverse).
        """
        if len(sequences) < 2:
            return 0.0

        def _ngrams(seq: List[int], n: int) -> set:
            return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}

        overlaps: List[float] = []
        for i in range(len(sequences)):
            ng_i = _ngrams(sequences[i], self.ngram_order)
            for j in range(i + 1, len(sequences)):
                ng_j = _ngrams(sequences[j], self.ngram_order)
                union = len(ng_i | ng_j)
                if union == 0:
                    overlaps.append(0.0)
                else:
                    overlaps.append(len(ng_i & ng_j) / union)
        return float(np.mean(overlaps))

    @staticmethod
    def _pairwise_jaccard(sequences: List[List[int]]) -> float:
        """Mean pairwise Jaccard *distance* between token sets.

        Parameters
        ----------
        sequences : List[List[int]]
            Token sequences.

        Returns
        -------
        float
            Mean Jaccard distance in [0, 1].
        """
        if len(sequences) < 2:
            return 0.0
        sets = [set(s) for s in sequences]
        dists: List[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                if union == 0:
                    dists.append(0.0)
                else:
                    dists.append(1.0 - inter / union)
        return float(np.mean(dists))


# =========================================================================
# TokenDiversityBonusNucleus
# =========================================================================


class TokenDiversityBonusNucleus:
    """Nucleus sampling with a cross-sequence diversity bonus at the token level.

    When generating multiple sequences simultaneously, this class adds a
    bonus to each candidate token's logit that is inversely proportional to
    how many times that token has appeared across *all* currently generating
    sequences.  This encourages the sequences to diverge in token usage.

    Parameters
    ----------
    config : NucleusConfig
        Base nucleus configuration.
    bonus_weight : float
        Scaling factor for the diversity bonus (≥ 0).  Higher values push
        harder for diverse token usage.
    num_sequences : int
        Number of sequences being generated in parallel.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        config: Optional[NucleusConfig] = None,
        bonus_weight: float = 1.0,
        num_sequences: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or NucleusConfig()
        self.bonus_weight = bonus_weight
        self.num_sequences = num_sequences
        self._rng = np.random.default_rng(seed)
        self._sequences: List[List[int]] = [[] for _ in range(num_sequences)]

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def sample_step(
        self,
        logits: np.ndarray,
        seq_index: int,
    ) -> int:
        """Sample one token for a specific sequence with diversity bonus.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector of shape ``(V,)``.
        seq_index : int
            Index of the sequence being decoded (0-based).

        Returns
        -------
        int
            Sampled token index.
        """
        cross_counts = self._cross_sequence_token_counts(
            exclude_index=seq_index
        )
        bonus = self._compute_diversity_bonus(logits, cross_counts)
        modified = self._apply_diversity_bonus(logits, bonus)
        filtered = nucleus_filter(
            modified / self.config.temperature,
            self.config.top_p,
            self.config.min_tokens_to_keep,
        )
        token = sample_with_temperature(
            filtered, temperature=1.0, rng=self._rng
        )[0]
        self._sequences[seq_index].append(token)
        return token

    def reset(self) -> None:
        """Clear all recorded sequences."""
        self._sequences = [[] for _ in range(self.num_sequences)]

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _compute_diversity_bonus(
        self,
        logits: np.ndarray,
        cross_seq_counts: Dict[int, int],
    ) -> np.ndarray:
        """Compute a per-token diversity bonus vector.

        The bonus for token *t* is ``bonus_weight / (1 + count(t))`` where
        ``count(t)`` is the number of times *t* has appeared across all other
        sequences.  Tokens never seen get the full bonus; frequent tokens
        get a smaller bonus.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector (used only for shape).
        cross_seq_counts : Dict[int, int]
            Mapping from token ID to cross-sequence count.

        Returns
        -------
        np.ndarray
            Bonus vector of the same shape as ``logits``.
        """
        V = logits.shape[0]
        bonus = np.full(V, self.bonus_weight, dtype=np.float64)
        for tid, count in cross_seq_counts.items():
            if 0 <= tid < V:
                bonus[tid] = self.bonus_weight / (1.0 + count)
        return bonus

    def _apply_diversity_bonus(
        self,
        logits: np.ndarray,
        bonus: np.ndarray,
    ) -> np.ndarray:
        """Add the diversity bonus to the logit vector.

        Parameters
        ----------
        logits : np.ndarray
            Raw logit vector.
        bonus : np.ndarray
            Diversity bonus vector.

        Returns
        -------
        np.ndarray
            Modified logit vector.
        """
        return logits.astype(np.float64) + bonus

    def _cross_sequence_token_counts(
        self,
        exclude_index: int = -1,
    ) -> Dict[int, int]:
        """Count token occurrences across all sequences except one.

        Parameters
        ----------
        exclude_index : int
            Sequence index to exclude (−1 to include all).

        Returns
        -------
        Dict[int, int]
            Mapping from token ID to aggregate count.
        """
        counts: Dict[int, int] = {}
        for i, seq in enumerate(self._sequences):
            if i == exclude_index:
                continue
            for tid in seq:
                counts[tid] = counts.get(tid, 0) + 1
        return counts


# =========================================================================
# New helper functions
# =========================================================================


def temperature_sweep_analysis(
    logit_source: Callable[..., np.ndarray],
    prompt_ids: List[int],
    temps: List[float],
    p: float = 0.9,
    num_seq: int = 5,
    max_length: int = 64,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run nucleus sampling across temperatures and collect per-temp statistics.

    Parameters
    ----------
    logit_source : Callable
        Function that accepts token IDs and returns a logit vector.
    prompt_ids : List[int]
        Prompt token IDs.
    temps : List[float]
        Temperatures to evaluate.
    p : float
        Nucleus probability threshold.
    num_seq : int
        Sequences per temperature.
    max_length : int
        Maximum generation length.
    seed : int, optional
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``'temperatures'``, ``'diversity_scores'``,
        ``'mean_lengths'``, and ``'sequences'``.
    """
    cfg = NucleusConfig(top_p=p)
    sweeper = TemperatureSweepNucleus(base_config=cfg, seed=seed)
    results = sweeper.sweep(logit_source, prompt_ids, temps, num_seq, max_length)

    diversity_scores: Dict[float, float] = {}
    mean_lengths: Dict[float, float] = {}
    for temp, seqs in results.items():
        diversity_scores[temp] = batch_diversity_score(seqs)
        mean_lengths[temp] = float(np.mean([len(s) for s in seqs])) if seqs else 0.0

    return {
        "temperatures": temps,
        "diversity_scores": diversity_scores,
        "mean_lengths": mean_lengths,
        "sequences": results,
    }


def compute_repetition_penalty_logits(
    logits: np.ndarray,
    generated_ids: Union[List[int], np.ndarray],
    penalty: float,
) -> np.ndarray:
    """Apply multiplicative repetition penalty to logits.

    For each token in ``generated_ids``, the corresponding logit is divided
    by ``penalty`` if positive and multiplied by ``penalty`` if negative.
    This follows the scheme from Keskar et al. (2019).

    Parameters
    ----------
    logits : np.ndarray
        Logit vector of shape ``(V,)``.
    generated_ids : List[int] or np.ndarray
        Token IDs that have been generated.
    penalty : float
        Penalty factor (> 1 penalises repetition).

    Returns
    -------
    np.ndarray
        Logit vector with penalty applied.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits
    for tid in set(generated_ids):
        tid = int(tid)
        if 0 <= tid < len(logits):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty
    return logits


def compute_frequency_penalty_logits(
    logits: np.ndarray,
    token_counts: Dict[int, int],
    penalty: float,
) -> np.ndarray:
    """Apply additive frequency penalty proportional to token counts.

    Parameters
    ----------
    logits : np.ndarray
        Logit vector of shape ``(V,)``.
    token_counts : Dict[int, int]
        Mapping from token ID to occurrence count.
    penalty : float
        Per-count penalty (subtracted from logit).

    Returns
    -------
    np.ndarray
        Penalised logit vector.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    if penalty == 0.0 or not token_counts:
        return logits
    for tid, count in token_counts.items():
        if 0 <= tid < len(logits):
            logits[tid] -= penalty * count
    return logits


def batch_diversity_score(sequences: List[List[int]]) -> float:
    """Compute a diversity score for a batch of token sequences.

    The score combines mean pairwise Jaccard distance (token-set level) and
    unique-token ratio.  Values are in [0, 1] where 1 indicates maximum
    diversity.

    Parameters
    ----------
    sequences : List[List[int]]
        Generated token sequences.

    Returns
    -------
    float
        Diversity score in [0, 1].
    """
    if len(sequences) < 2:
        return 0.0

    # Jaccard distance component.
    sets = [set(s) for s in sequences]
    jaccard_dists: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union == 0:
                jaccard_dists.append(0.0)
            else:
                jaccard_dists.append(1.0 - inter / union)
    mean_jaccard = float(np.mean(jaccard_dists))

    # Unique-token ratio component.
    all_tokens = [t for s in sequences for t in s]
    if len(all_tokens) == 0:
        unique_ratio = 0.0
    else:
        unique_ratio = len(set(all_tokens)) / len(all_tokens)

    return 0.6 * mean_jaccard + 0.4 * unique_ratio


def token_diversity_bonus(
    logits: np.ndarray,
    cross_seq_counts: Dict[int, int],
    bonus_weight: float = 1.0,
) -> np.ndarray:
    """Add a diversity bonus to logits based on cross-sequence token counts.

    Tokens that have been used less frequently across other sequences receive
    a larger additive bonus, encouraging token-level diversity.

    Parameters
    ----------
    logits : np.ndarray
        Logit vector of shape ``(V,)``.
    cross_seq_counts : Dict[int, int]
        Mapping from token ID to count across other sequences.
    bonus_weight : float
        Scaling factor for the bonus.

    Returns
    -------
    np.ndarray
        Modified logit vector with diversity bonus added.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    V = logits.shape[0]
    bonus = np.full(V, bonus_weight, dtype=np.float64)
    for tid, count in cross_seq_counts.items():
        if 0 <= tid < V:
            bonus[tid] = bonus_weight / (1.0 + count)
    return logits + bonus


# =========================================================================
# Convenience aliases & module exports
# =========================================================================


__all__ = [
    # Config & state
    "NucleusConfig",
    "NucleusState",
    # Main algorithm
    "NucleusSampling",
    # Analysis
    "NucleusAnalyzer",
    # Adaptive
    "AdaptiveNucleus",
    # Extended algorithms
    "TemperatureSweepNucleus",
    "AdaptiveThresholdNucleus",
    "NucleusWithPenalties",
    "DiversityAwareBatchNucleus",
    "TokenDiversityBonusNucleus",
    # Original helpers
    "nucleus_filter",
    "compute_cumulative_probs",
    "renormalize_logits",
    "sample_with_temperature",
    "batch_nucleus_filter",
    "entropy_of_step",
    "effective_vocab_size",
    # New helpers
    "temperature_sweep_analysis",
    "compute_repetition_penalty_logits",
    "compute_frequency_penalty_logits",
    "batch_diversity_score",
    "token_diversity_bonus",
]
