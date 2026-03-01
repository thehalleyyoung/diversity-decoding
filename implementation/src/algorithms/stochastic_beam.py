"""
Stochastic Beam Search for the Diversity Decoding Arena.
========================================================

Implements Stochastic Beam Search with Gumbel-top-k sampling without
replacement, following Kool et al. (2019) *"Stochastic Beams and Where to
Find Them"* and extensions from Jinnai et al. (2024).

Unlike deterministic beam search, which always expands the top-scoring
hypotheses, stochastic beam search draws beams *without replacement* from the
sequence distribution by leveraging the Gumbel-top-k trick.  This yields an
unbiased, low-variance estimator for partition-function quantities and
naturally promotes diversity among the returned sequences.

Key components
--------------
* **StochasticBeamConfig** — dataclass extending ``DecodingConfig`` with
  beam-width, Gumbel temperature, length/diversity/repetition penalties,
  and sampling-method selection.
* **BeamHypothesis** — lightweight container for a single beam hypothesis,
  carrying token ids, scores, Gumbel-perturbed scores, and status.
* **GumbelTopK** — the core Gumbel-top-k sampler that draws *k* items
  without replacement from a categorical distribution using perturbed
  log-probabilities.
* **StochasticBeamSearch** — the main ``DecodingAlgorithm`` implementation.
  Overrides ``generate()`` to run stochastic beam expansion with optional
  nucleus-beam and ancestral-beam step variants.

References
----------
- Kool, W., Van Hoof, H., & Welling, M. (2019). *Stochastic Beams and
  Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences
  Without Replacement*. ICML 2019.
- Jinnai, Y., Chen, J., Kandasamy, K., & Kaelbling, L. P. (2024).
  *Improved Diversity-Driven Stochastic Beam Search*.
"""

from __future__ import annotations

import collections
import copy
import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence as SequenceType,
    Set,
    Tuple,
    Union,
)

import numpy as np

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    _log_softmax,
    _stable_softmax,
    _top_k_filter,
    sample_token,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEG_INF: float = float("-inf")
_POS_INF: float = float("inf")
_LOG_EPS: float = 1e-10
_EULER_MASCHERONI: float = 0.5772156649015329


# =========================================================================
# StochasticBeamConfig
# =========================================================================


@dataclass
class StochasticBeamConfig(DecodingConfig):
    """Configuration for Stochastic Beam Search.

    Extends :class:`DecodingConfig` with parameters specific to the
    Gumbel-top-k stochastic beam search algorithm.

    Parameters
    ----------
    beam_width : int
        Number of beams (hypotheses) to maintain at each step.
    temperature : float
        Softmax temperature applied to logits before scoring.
    gumbel_temperature : float
        Temperature for Gumbel perturbation noise. Lower values make the
        sampling more deterministic; higher values increase randomness.
    length_penalty : float
        Exponent *α* in the length normalisation factor ``(5+len)^α / 6^α``
        (following Wu et al., 2016). Set to 0.0 to disable.
    diversity_penalty : float
        Penalty coefficient for promoting diversity among beams.  Applied
        by computing pairwise similarity and penalising beams that are too
        close to previously selected beams.
    n_best : int
        Number of final sequences to return (must be ≤ beam_width).
    use_gumbel_top_k : bool
        If True (default), use the Gumbel-top-k trick for sampling without
        replacement.  If False, fall back to categorical sampling.
    stochastic_expansion : bool
        If True, apply stochastic selection during beam expansion as well
        as during beam pruning.
    expansion_ratio : float
        How many candidate expansions to consider per beam, expressed as a
        multiplier of ``beam_width``.  For example, 2.0 means we consider
        ``2 * beam_width`` candidates from each beam.
    min_length : int
        Minimum number of new tokens to generate (EOS is suppressed until
        this length is reached).
    max_length : int
        Maximum number of new tokens to generate.
    repetition_penalty : float
        Multiplicative penalty for repeated tokens (≥ 1.0).  See
        Keskar et al. (2019).
    no_repeat_ngram_size : int
        If > 0, disallow any n-gram of this size from repeating.
    early_stopping : bool
        If True, stop when ``beam_width`` complete hypotheses have been
        found, even if active beams could still improve.
    sampling_method : str
        Which sampling variant to use inside beam steps.
        One of ``'gumbel'``, ``'categorical'``, ``'nucleus_beam'``.
    nucleus_p : float
        Nucleus (top-p) threshold, used only when
        ``sampling_method='nucleus_beam'``.
    """

    algorithm_name: str = "StochasticBeamSearch"

    # Beam search core
    beam_width: int = 10
    temperature: float = 1.0
    gumbel_temperature: float = 1.0
    length_penalty: float = 0.6
    diversity_penalty: float = 0.5
    n_best: int = 5

    # Gumbel-top-k options
    use_gumbel_top_k: bool = True
    stochastic_expansion: bool = True
    expansion_ratio: float = 2.0

    # Length constraints
    min_length: int = 1
    max_length: int = 512

    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Stopping
    early_stopping: bool = False

    # Sampling variant
    sampling_method: str = "gumbel"
    nucleus_p: float = 0.9

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty == valid)."""
        errors = super().validate()
        if self.beam_width < 1:
            errors.append("beam_width must be >= 1")
        if self.n_best < 1:
            errors.append("n_best must be >= 1")
        if self.n_best > self.beam_width:
            errors.append("n_best must be <= beam_width")
        if self.gumbel_temperature <= 0:
            errors.append("gumbel_temperature must be > 0")
        if self.length_penalty < 0:
            errors.append("length_penalty must be >= 0")
        if self.diversity_penalty < 0:
            errors.append("diversity_penalty must be >= 0")
        if self.expansion_ratio < 1.0:
            errors.append("expansion_ratio must be >= 1.0")
        if self.min_length < 0:
            errors.append("min_length must be >= 0")
        if self.max_length < 1:
            errors.append("max_length must be >= 1")
        if self.min_length > self.max_length:
            errors.append("min_length must be <= max_length")
        if self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
        if self.no_repeat_ngram_size < 0:
            errors.append("no_repeat_ngram_size must be >= 0")
        if self.sampling_method not in ("gumbel", "categorical", "nucleus_beam"):
            errors.append(
                f"sampling_method must be one of 'gumbel', 'categorical', "
                f"'nucleus_beam'; got '{self.sampling_method}'"
            )
        if self.nucleus_p <= 0 or self.nucleus_p > 1.0:
            errors.append("nucleus_p must be in (0, 1]")
        return errors


# =========================================================================
# BeamHypothesis
# =========================================================================


@dataclass
class BeamHypothesis:
    """A single beam hypothesis during stochastic beam search.

    Attributes
    ----------
    tokens : list of int
        Token IDs generated so far (including prompt tokens).
    score : float
        Cumulative log-probability (not length-normalised).
    gumbel_score : float
        Cumulative score with Gumbel perturbation, used for stochastic
        selection.
    log_prob : float
        Accumulated log-probability without any penalty adjustments.
    length : int
        Number of *generated* tokens (excludes prompt tokens).
    is_finished : bool
        Whether this hypothesis has emitted an EOS token or reached
        ``max_length``.
    perturbed_scores : list of float
        Per-step Gumbel-perturbed scores, useful for analysis.
    prompt_length : int
        Number of prompt tokens at the start of ``tokens``.
    normalized_score : float
        Length-normalised score, populated during finalisation.
    diversity_bonus : float
        Accumulated diversity bonus for this hypothesis.
    """

    tokens: List[int] = field(default_factory=list)
    score: float = 0.0
    gumbel_score: float = 0.0
    log_prob: float = 0.0
    length: int = 0
    is_finished: bool = False
    perturbed_scores: List[float] = field(default_factory=list)
    prompt_length: int = 0
    normalized_score: float = 0.0
    diversity_bonus: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def generated_tokens(self) -> List[int]:
        """Return only the tokens that were generated (no prompt)."""
        return self.tokens[self.prompt_length:]

    def clone(self) -> "BeamHypothesis":
        """Deep copy of this hypothesis."""
        return BeamHypothesis(
            tokens=list(self.tokens),
            score=self.score,
            gumbel_score=self.gumbel_score,
            log_prob=self.log_prob,
            length=self.length,
            is_finished=self.is_finished,
            perturbed_scores=list(self.perturbed_scores),
            prompt_length=self.prompt_length,
            normalized_score=self.normalized_score,
            diversity_bonus=self.diversity_bonus,
        )

    def extend(
        self,
        token_id: int,
        token_log_prob: float,
        gumbel_perturbed: float = 0.0,
    ) -> "BeamHypothesis":
        """Return a *new* hypothesis extended with one token."""
        new = self.clone()
        new.tokens.append(token_id)
        new.score += token_log_prob
        new.log_prob += token_log_prob
        new.gumbel_score += gumbel_perturbed
        new.length += 1
        new.perturbed_scores.append(gumbel_perturbed)
        return new

    def __lt__(self, other: "BeamHypothesis") -> bool:
        """Compare by gumbel_score (higher is better)."""
        return self.gumbel_score < other.gumbel_score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeamHypothesis):
            return NotImplemented
        return self.tokens == other.tokens

    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __repr__(self) -> str:
        gen = self.generated_tokens
        preview = gen[:8]
        suffix = "..." if len(gen) > 8 else ""
        return (
            f"BeamHypothesis(len={self.length}, score={self.score:.4f}, "
            f"gumbel={self.gumbel_score:.4f}, fin={self.is_finished}, "
            f"tokens={preview}{suffix})"
        )


# =========================================================================
# GumbelTopK — Gumbel-top-k sampling without replacement
# =========================================================================


class GumbelTopK:
    """Gumbel-top-k sampler for drawing *k* items without replacement.

    Implements the Gumbel-top-k trick from Kool et al. (2019).  Given
    a vector of un-normalised log-probabilities, we add i.i.d. Gumbel(0,1)
    noise and take the *k* largest perturbed values.  This is equivalent to
    sampling *k* items without replacement from the categorical distribution
    defined by ``softmax(log_probs)``.

    Parameters
    ----------
    k : int
        Number of items to sample.
    temperature : float
        Temperature applied to the Gumbel noise.  Higher temperatures
        increase randomness; temperature → 0 recovers deterministic top-k.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.k = k
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_without_replacement(
        self, log_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample *k* items without replacement using Gumbel-top-k.

        Parameters
        ----------
        log_probs : np.ndarray
            Log-probabilities of shape ``(vocab_size,)``.

        Returns
        -------
        indices : np.ndarray of int, shape ``(k,)``
            Indices of the selected items, sorted by perturbed score
            (highest first).
        perturbed_scores : np.ndarray of float, shape ``(k,)``
            The Gumbel-perturbed log-probabilities for the selected items.
        """
        log_probs = np.asarray(log_probs, dtype=np.float64).ravel()
        vocab_size = log_probs.shape[0]
        effective_k = min(self.k, vocab_size)

        if effective_k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # Mask out -inf entries
        valid_mask = np.isfinite(log_probs)
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            indices = self._rng.choice(vocab_size, size=effective_k, replace=False)
            return indices, np.full(effective_k, _NEG_INF)

        effective_k = min(effective_k, n_valid)

        # Add Gumbel noise
        gumbel_noise = self._sample_gumbel(log_probs.shape)
        perturbed = log_probs / self.temperature + gumbel_noise

        # Set invalid positions to -inf
        perturbed[~valid_mask] = _NEG_INF

        # Top-k selection
        if effective_k >= vocab_size:
            sorted_idx = np.argsort(-perturbed)
            return sorted_idx, perturbed[sorted_idx]

        # Use argpartition for efficiency, then sort the top-k
        top_k_indices = np.argpartition(-perturbed, effective_k)[:effective_k]
        top_k_scores = perturbed[top_k_indices]
        order = np.argsort(-top_k_scores)
        top_k_indices = top_k_indices[order]
        top_k_scores = top_k_scores[order]

        return top_k_indices, top_k_scores

    # ------------------------------------------------------------------
    # Internal Gumbel utilities
    # ------------------------------------------------------------------

    def _sample_gumbel(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample Gumbel(0,1) noise via the inverse CDF method.

        Gumbel(0, 1) = −log(−log(U)), U ~ Uniform(0, 1).

        Parameters
        ----------
        shape : tuple of int
            Output shape.

        Returns
        -------
        np.ndarray
            Gumbel(0, 1) samples of the given shape.
        """
        u = self._rng.uniform(low=1e-10, high=1.0 - 1e-10, size=shape)
        return -np.log(-np.log(u))

    def _gumbel_softmax(
        self, logits: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Gumbel-softmax: a differentiable approximation to sampling.

        Computes ``softmax((logits + Gumbel_noise) / temperature)``.  This is
        *not* used for discrete sampling; it's a continuous relaxation useful
        for gradient estimation.

        Parameters
        ----------
        logits : np.ndarray
            Un-normalised log-probabilities of shape ``(n,)``.
        temperature : float
            Softmax temperature (lower → more peaked).

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(n,)``.
        """
        logits = np.asarray(logits, dtype=np.float64)
        gumbel_noise = self._sample_gumbel(logits.shape)
        perturbed = (logits + gumbel_noise) / max(temperature, 1e-10)
        return _stable_softmax(perturbed)

    def _top_k_gumbel(
        self, logits: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select the top-k items from Gumbel-perturbed logits.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(n,)``.
        k : int
            Number of items to select.

        Returns
        -------
        values : np.ndarray, shape ``(k,)``
            Perturbed scores of the selected items (sorted descending).
        indices : np.ndarray, shape ``(k,)``
            Indices of the selected items.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        n = logits.shape[0]
        k = min(k, n)
        if k <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

        gumbel_noise = self._sample_gumbel(logits.shape)
        perturbed = logits / self.temperature + gumbel_noise

        if k >= n:
            order = np.argsort(-perturbed)
            return perturbed[order], order

        top_idx = np.argpartition(-perturbed, k)[:k]
        top_vals = perturbed[top_idx]
        order = np.argsort(-top_vals)
        return top_vals[order], top_idx[order]

    def _truncated_gumbel(
        self,
        logits: np.ndarray,
        k: int,
        already_selected: np.ndarray,
    ) -> np.ndarray:
        """Compute truncated Gumbel-perturbed logits.

        After selecting some items via the Gumbel-top-k trick, the conditional
        distribution of the Gumbel noise for *un*-selected items is a truncated
        Gumbel distribution.  This method perturbs the remaining logits using
        that truncated distribution, which is required for correct sequential
        sampling without replacement (see Kool et al. 2019, §3.2).

        Parameters
        ----------
        logits : np.ndarray
            Original (un-perturbed) logits of shape ``(n,)``.
        k : int
            The rank of the truncation point (i.e., we condition on the
            top-``k`` items having been selected).
        already_selected : np.ndarray
            Indices that have already been selected — their Gumbel values are
            known and fixed.

        Returns
        -------
        np.ndarray
            Perturbed logits of shape ``(n,)`` with already-selected positions
            set to ``-inf``.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        n = logits.shape[0]
        result = logits.copy()

        if len(already_selected) == 0:
            gumbel_noise = self._sample_gumbel(logits.shape)
            return logits / self.temperature + gumbel_noise

        # Compute the truncation bound: the k-th largest perturbed value
        # In practice, we use the log-sum-exp of the un-selected logits as
        # an upper bound for the truncated Gumbel.
        selected_set = set(already_selected.tolist())
        mask = np.ones(n, dtype=bool)
        mask[already_selected] = False
        remaining_logits = logits[mask]

        if remaining_logits.shape[0] == 0:
            return np.full(n, _NEG_INF)

        # The maximum perturbed score of the selected items serves as the
        # upper truncation bound for the remaining items' Gumbel noise.
        selected_logits = logits[already_selected] / self.temperature
        gumbel_selected = self._sample_gumbel(selected_logits.shape)
        max_perturbed = np.max(selected_logits + gumbel_selected)

        # For the remaining items, sample from the truncated Gumbel
        # distribution: Gumbel(logit_i) | G_i < max_perturbed
        remaining_scaled = logits[mask] / self.temperature
        v = self._rng.uniform(low=1e-10, high=1.0 - 1e-10, size=remaining_scaled.shape)

        # Truncated Gumbel via inverse CDF:
        #   G_trunc = -log(exp(-max_perturbed) + (1 - v) * exp(-logit_i))
        #   But more numerically stable using log-sum-exp tricks.
        log_v = np.log(v)
        # CDF of Gumbel at max_perturbed with location logit_i:
        #   F(max_perturbed; logit_i) = exp(-exp(-(max_perturbed - logit_i)))
        z = np.exp(-(max_perturbed - remaining_scaled))
        cdf_at_bound = np.exp(-z)
        # Truncated sample: F^{-1}(U * F(bound))
        # log(F^{-1}(u)) = logit_i - log(-log(u))
        u_truncated = v * cdf_at_bound
        u_truncated = np.clip(u_truncated, 1e-30, 1.0 - 1e-10)
        truncated_gumbel = remaining_scaled - np.log(-np.log(u_truncated))
        truncated_gumbel = np.minimum(truncated_gumbel, max_perturbed)

        result[mask] = truncated_gumbel
        result[already_selected] = _NEG_INF

        return result


# =========================================================================
# Diversity computation helpers
# =========================================================================


def _hamming_distance(seq_a: List[int], seq_b: List[int]) -> int:
    """Hamming distance between two token sequences."""
    min_len = min(len(seq_a), len(seq_b))
    dist = sum(1 for i in range(min_len) if seq_a[i] != seq_b[i])
    dist += abs(len(seq_a) - len(seq_b))
    return dist


def _jaccard_similarity(seq_a: List[int], seq_b: List[int]) -> float:
    """Jaccard similarity between two token sequences (as sets)."""
    set_a = set(seq_a)
    set_b = set(seq_b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _ngram_overlap(seq_a: List[int], seq_b: List[int], n: int = 2) -> float:
    """Fraction of n-grams shared between two sequences."""
    if len(seq_a) < n or len(seq_b) < n:
        return 0.0
    ngrams_a = set(tuple(seq_a[i : i + n]) for i in range(len(seq_a) - n + 1))
    ngrams_b = set(tuple(seq_b[i : i + n]) for i in range(len(seq_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)


def _pairwise_diversity_matrix(
    beams: List[BeamHypothesis], metric: str = "hamming"
) -> np.ndarray:
    """Compute a pairwise diversity (distance) matrix for a set of beams.

    Parameters
    ----------
    beams : list of BeamHypothesis
        The hypotheses to compare.
    metric : str
        One of ``'hamming'``, ``'jaccard'``, ``'ngram'``.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape ``(n, n)``.
    """
    n = len(beams)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            tokens_i = beams[i].generated_tokens
            tokens_j = beams[j].generated_tokens
            if metric == "hamming":
                d = float(_hamming_distance(tokens_i, tokens_j))
            elif metric == "jaccard":
                d = 1.0 - _jaccard_similarity(tokens_i, tokens_j)
            elif metric == "ngram":
                d = 1.0 - _ngram_overlap(tokens_i, tokens_j, n=2)
            else:
                d = float(_hamming_distance(tokens_i, tokens_j))
            mat[i, j] = d
            mat[j, i] = d
    return mat


# =========================================================================
# Length penalty helpers
# =========================================================================


def _wu_length_penalty(length: int, alpha: float) -> float:
    """Length penalty from Wu et al. (2016): ``((5 + len) / 6)^alpha``."""
    if alpha == 0.0:
        return 1.0
    return ((5.0 + length) / 6.0) ** alpha


def _linear_length_penalty(length: int, alpha: float) -> float:
    """Simple linear length penalty: ``length^alpha``."""
    if alpha == 0.0:
        return 1.0
    if length <= 0:
        return 1.0
    return length ** alpha


# =========================================================================
# StochasticBeamSearch — main algorithm
# =========================================================================


class StochasticBeamSearch(DecodingAlgorithm):
    """Stochastic Beam Search with Gumbel-top-k sampling.

    This algorithm maintains a set of *beam_width* hypotheses.  At each step,
    it scores all possible one-token extensions, then uses the Gumbel-top-k
    trick (Kool et al. 2019) to *sample without replacement* the next set of
    beams.  The result is an unbiased set of high-probability sequences with
    natural diversity.

    The algorithm supports three sampling methods:

    * **gumbel** — Standard Gumbel-top-k sampling without replacement.
    * **categorical** — Categorical sampling within each beam's expansion
      (less principled but sometimes useful with diversity penalty).
    * **nucleus_beam** — Nucleus (top-p) filtering within each beam step,
      combined with Gumbel-top-k selection across beams.

    Parameters
    ----------
    config : StochasticBeamConfig
        Algorithm configuration.
    """

    def __init__(self, config: StochasticBeamConfig) -> None:
        super().__init__(config)
        self._config: StochasticBeamConfig = config
        self._gumbel_sampler = GumbelTopK(
            k=config.beam_width,
            temperature=config.gumbel_temperature,
            seed=config.seed,
        )
        self._finished_beams: List[BeamHypothesis] = []
        self._step_count: int = 0
        self._start_time: float = 0.0
        self._vocab_size: int = 0
        self._diversity_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        return self.describe()

    # ------------------------------------------------------------------
    # Main entry point: generate
    # ------------------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: Optional[int] = None,
    ) -> List[TokenSequence]:
        """Generate sequences using stochastic beam search.

        Parameters
        ----------
        logit_source : LogitSource
            Callable that takes ``List[List[int]]`` and returns logits of
            shape ``(batch, vocab_size)``.
        prompt_ids : list of int
            Token IDs of the prompt.
        n_sequences : int or None
            Number of sequences to return.  Defaults to ``config.n_best``.

        Returns
        -------
        list of TokenSequence
            Generated token sequences, sorted by normalised score (best first).
        """
        if self._config.seed is not None:
            np.random.seed(self._config.seed)
            self._gumbel_sampler = GumbelTopK(
                k=self._config.beam_width,
                temperature=self._config.gumbel_temperature,
                seed=self._config.seed,
            )

        n_return = n_sequences if n_sequences is not None else self._config.n_best
        n_return = min(n_return, self._config.beam_width)

        self._finished_beams = []
        self._step_count = 0
        self._start_time = time.time()
        self._diversity_cache = {}

        # Detect vocab size from a single forward pass
        probe_logits = logit_source([prompt_ids])
        if probe_logits.ndim == 1:
            self._vocab_size = probe_logits.shape[0]
        else:
            self._vocab_size = probe_logits.shape[-1]

        # Initialize beams
        beams = self._initialize_beams(prompt_ids)

        # Main decoding loop
        for step in range(self._config.max_length):
            self._step_count = step

            # Separate finished and active beams
            active, finished = self._prune_finished(beams)
            self._finished_beams.extend(finished)

            if len(active) == 0:
                logger.debug("All beams finished at step %d", step)
                break

            # Early stopping check
            if self._config.early_stopping and len(self._finished_beams) >= n_return:
                logger.debug(
                    "Early stopping: %d finished beams >= %d requested",
                    len(self._finished_beams),
                    n_return,
                )
                break

            # Expand active beams
            candidates = self._expand_beams(active, logit_source)

            if len(candidates) == 0:
                logger.warning("No valid candidates at step %d", step)
                break

            # Select next set of beams
            beams = self._select_beams(candidates, self._config.beam_width)

            self._log_beam_state(beams, step)

        # Collect any remaining active beams
        for beam in beams:
            if not beam.is_finished:
                beam.is_finished = True
            self._finished_beams.append(beam)

        # Finalize and return
        results = self._finalize(self._finished_beams)
        return results[:n_return]

    # ------------------------------------------------------------------
    # _step override for base class compatibility
    # ------------------------------------------------------------------

    def _step(
        self,
        state: DecodingState,
        logit_source: LogitSource,
    ) -> DecodingState:
        """Single step for base class compatibility.

        Stochastic beam search overrides ``generate()`` entirely, so this
        method provides a simplified step that can be used if the base class
        ``_generation_loop`` is invoked instead.
        """
        active = state.active_indices()
        if not active:
            return state

        sequences = [state.sequences[i] for i in active]
        all_logits = logit_source(sequences)

        for batch_idx, seq_idx in enumerate(active):
            logits = all_logits[batch_idx]
            logits = self._apply_repetition_penalty(logits, state.sequences[seq_idx])
            token = sample_token(logits, temperature=self._config.temperature)
            state.update_sequence(seq_idx, token)

            if self._config.eos_token_id is not None and token == self._config.eos_token_id:
                if len(state.sequences[seq_idx]) >= self._config.min_length:
                    state.mark_finished(seq_idx)

        state.step += 1
        return state

    # ------------------------------------------------------------------
    # Beam initialization
    # ------------------------------------------------------------------

    def _initialize_beams(self, prompt_ids: List[int]) -> List[BeamHypothesis]:
        """Create the initial set of beam hypotheses from the prompt.

        Parameters
        ----------
        prompt_ids : list of int
            Token IDs of the prompt.

        Returns
        -------
        list of BeamHypothesis
            A list containing a single hypothesis with the prompt tokens.
        """
        initial_beam = BeamHypothesis(
            tokens=list(prompt_ids),
            score=0.0,
            gumbel_score=0.0,
            log_prob=0.0,
            length=0,
            is_finished=False,
            perturbed_scores=[],
            prompt_length=len(prompt_ids),
            normalized_score=0.0,
            diversity_bonus=0.0,
        )
        return [initial_beam]

    # ------------------------------------------------------------------
    # Beam expansion
    # ------------------------------------------------------------------

    def _expand_beams(
        self,
        beams: List[BeamHypothesis],
        logit_source: LogitSource,
    ) -> List[BeamHypothesis]:
        """Expand all active beams by one token.

        For each beam, we query the logit source to get next-token logits,
        apply penalties and constraints, score the expansions, and create
        candidate hypotheses.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Active (non-finished) beams.
        logit_source : LogitSource
            The logit source callable.

        Returns
        -------
        list of BeamHypothesis
            All candidate expansions across all beams.
        """
        if not beams:
            return []

        # Batch forward pass
        input_sequences = [beam.tokens for beam in beams]
        try:
            all_logits = logit_source(input_sequences)
        except Exception as e:
            logger.error("Logit source failed: %s", e)
            return []

        if all_logits.ndim == 1:
            all_logits = all_logits.reshape(1, -1)

        candidates: List[BeamHypothesis] = []
        n_expand = max(
            1,
            int(self._config.beam_width * self._config.expansion_ratio),
        )

        for beam_idx, beam in enumerate(beams):
            logits = all_logits[beam_idx].copy().astype(np.float64)

            # Apply repetition penalty
            if self._config.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits, beam.tokens
                )

            # Block n-gram repetition
            if self._config.no_repeat_ngram_size > 0:
                logits = self._block_ngram_repetitions(
                    logits, beam.tokens, self._config.no_repeat_ngram_size
                )

            # Suppress EOS before min_length
            if (
                beam.length < self._config.min_length
                and self._config.eos_token_id is not None
            ):
                logits[self._config.eos_token_id] = _NEG_INF

            # Score expansions
            scored = self._score_expansions(beam, logits)

            # Select top candidates from this beam using the chosen method
            if self._config.sampling_method == "gumbel" and self._config.use_gumbel_top_k:
                indices, perturbed = self._gumbel_sampler.sample_without_replacement(
                    scored
                )
                indices = indices[:n_expand]
                perturbed = perturbed[:n_expand]
            elif self._config.sampling_method == "nucleus_beam":
                indices, perturbed = self._nucleus_beam_step(
                    scored, self._config.nucleus_p
                )
                indices = indices[:n_expand]
                perturbed = perturbed[:n_expand]
            elif self._config.sampling_method == "categorical":
                indices, perturbed = self._ancestral_beam_step(
                    scored, self._config.temperature
                )
                indices = indices[:n_expand]
                perturbed = perturbed[:n_expand]
            else:
                # Fallback: deterministic top-k
                if n_expand >= scored.shape[0]:
                    indices = np.argsort(-scored)
                else:
                    indices = np.argpartition(-scored, n_expand)[:n_expand]
                    order = np.argsort(-scored[indices])
                    indices = indices[order]
                perturbed = scored[indices]

            # Create candidate hypotheses
            log_probs = _log_softmax(logits)
            for i in range(len(indices)):
                token_id = int(indices[i])
                token_log_prob = float(log_probs[token_id])
                gumbel_val = float(perturbed[i])

                new_beam = beam.extend(token_id, token_log_prob, gumbel_val)

                # Check if finished
                if self._config.eos_token_id is not None and token_id == self._config.eos_token_id:
                    if new_beam.length >= self._config.min_length:
                        new_beam.is_finished = True

                if new_beam.length >= self._config.max_length:
                    new_beam.is_finished = True

                candidates.append(new_beam)

        return candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_expansions(
        self, beam: BeamHypothesis, logits: np.ndarray
    ) -> np.ndarray:
        """Score all possible one-token expansions for a single beam.

        Combines the temperature-scaled logits with the beam's accumulated
        score to produce a score for each candidate next token.

        Parameters
        ----------
        beam : BeamHypothesis
            The parent beam.
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.

        Returns
        -------
        np.ndarray
            Scores of shape ``(vocab_size,)`` for each token.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()

        # Apply temperature
        if self._config.temperature > 0 and self._config.temperature != 1.0:
            scaled_logits = logits / self._config.temperature
        else:
            scaled_logits = logits

        # Log-softmax to get log-probabilities
        log_probs = _log_softmax(scaled_logits)

        # Combine with beam's accumulated score
        scores = beam.score + log_probs

        return scores

    # ------------------------------------------------------------------
    # Beam selection
    # ------------------------------------------------------------------

    def _select_beams(
        self,
        candidates: List[BeamHypothesis],
        beam_width: int,
    ) -> List[BeamHypothesis]:
        """Select the next set of beams from candidates.

        If ``use_gumbel_top_k`` is True, applies Gumbel-based stochastic
        selection.  Otherwise, falls back to deterministic top-k by score.

        Parameters
        ----------
        candidates : list of BeamHypothesis
            All candidate expansions.
        beam_width : int
            Number of beams to keep.

        Returns
        -------
        list of BeamHypothesis
            The selected beams for the next step.
        """
        if not candidates:
            return []

        if len(candidates) <= beam_width:
            return candidates

        # Apply diversity penalty
        if self._config.diversity_penalty > 0:
            candidates = self._apply_diversity_penalty(candidates)

        if (
            self._config.use_gumbel_top_k
            and self._config.stochastic_expansion
        ):
            return self._gumbel_select(candidates, beam_width)
        else:
            # Deterministic selection by gumbel_score
            candidates.sort(key=lambda b: b.gumbel_score, reverse=True)
            return candidates[:beam_width]

    def _gumbel_select(
        self,
        candidates: List[BeamHypothesis],
        beam_width: int,
    ) -> List[BeamHypothesis]:
        """Stochastic beam selection using the Gumbel-top-k trick.

        Given a pool of candidate hypotheses, we perturb their scores with
        Gumbel noise and select the top-k, implementing sampling without
        replacement from the (unnormalised) distribution over candidates.

        Parameters
        ----------
        candidates : list of BeamHypothesis
            Pool of candidate hypotheses.
        beam_width : int
            Number of beams to select.

        Returns
        -------
        list of BeamHypothesis
            The selected beams.
        """
        n_candidates = len(candidates)
        effective_k = min(beam_width, n_candidates)

        if effective_k <= 0:
            return []

        # Extract scores
        scores = np.array(
            [c.gumbel_score for c in candidates], dtype=np.float64
        )

        # Handle all-identical scores
        score_range = scores.max() - scores.min()
        if score_range < 1e-12:
            selected_idx = self._gumbel_sampler._rng.choice(
                n_candidates, size=effective_k, replace=False
            )
            return [candidates[i] for i in selected_idx]

        # Gumbel perturbation for selection
        gumbel_noise = self._gumbel_sampler._sample_gumbel((n_candidates,))
        perturbed = scores / self._config.gumbel_temperature + gumbel_noise

        # Top-k selection
        if effective_k >= n_candidates:
            order = np.argsort(-perturbed)
        else:
            top_idx = np.argpartition(-perturbed, effective_k)[:effective_k]
            order_within = np.argsort(-perturbed[top_idx])
            order = top_idx[order_within]

        selected = []
        for idx in order[:effective_k]:
            beam = candidates[int(idx)]
            beam.gumbel_score = float(perturbed[int(idx)])
            selected.append(beam)

        return selected

    # ------------------------------------------------------------------
    # Penalties and constraints
    # ------------------------------------------------------------------

    def _apply_length_penalty(self, score: float, length: int) -> float:
        """Apply length normalisation to a score.

        Uses the Wu et al. (2016) formula:
        ``score / ((5 + length) / 6)^alpha``.

        Parameters
        ----------
        score : float
            Raw cumulative log-probability.
        length : int
            Number of generated tokens.

        Returns
        -------
        float
            Length-normalised score.
        """
        penalty = _wu_length_penalty(length, self._config.length_penalty)
        if penalty == 0:
            return score
        return score / penalty

    def _apply_diversity_penalty(
        self, beams: List[BeamHypothesis]
    ) -> List[BeamHypothesis]:
        """Adjust beam scores by a diversity bonus/penalty.

        Beams that are more different from existing beams get a bonus;
        beams that are too similar get penalised.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Candidate beams.

        Returns
        -------
        list of BeamHypothesis
            Beams with adjusted ``gumbel_score`` values.
        """
        if len(beams) <= 1 or self._config.diversity_penalty <= 0:
            return beams

        diversity_bonuses = self._compute_diversity_bonus(beams)

        for i, beam in enumerate(beams):
            beam.gumbel_score += diversity_bonuses[i]
            beam.diversity_bonus += diversity_bonuses[i]

        return beams

    def _apply_repetition_penalty(
        self, logits: np.ndarray, generated_tokens: List[int]
    ) -> np.ndarray:
        """Apply repetition penalty to logits.

        Implements the repetition penalty from Keskar et al. (2019):
        for tokens that have appeared in the generated sequence, if their
        logit is positive, divide by the penalty; if negative, multiply.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.
        generated_tokens : list of int
            Tokens generated so far.

        Returns
        -------
        np.ndarray
            Adjusted logits.
        """
        if self._config.repetition_penalty <= 1.0:
            return logits

        logits = logits.copy()
        penalty = self._config.repetition_penalty

        # Unique tokens that have appeared
        unique_tokens = set(generated_tokens)
        for token_id in unique_tokens:
            if 0 <= token_id < logits.shape[0]:
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty

        return logits

    def _block_ngram_repetitions(
        self, logits: np.ndarray, tokens: List[int], n: int
    ) -> np.ndarray:
        """Block tokens that would create a repeated n-gram.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.
        tokens : list of int
            Current token sequence.
        n : int
            N-gram size.

        Returns
        -------
        np.ndarray
            Logits with banned positions set to ``-inf``.
        """
        if n <= 0 or len(tokens) < n:
            return logits

        logits = logits.copy()
        generated = tokens

        # Collect all (n-1)-grams seen so far
        ngrams: Dict[Tuple[int, ...], Set[int]] = collections.defaultdict(set)
        for i in range(len(generated) - n + 1):
            prefix = tuple(generated[i : i + n - 1])
            continuation = generated[i + n - 1]
            ngrams[prefix].add(continuation)

        # The current (n-1)-gram suffix
        if len(generated) >= n - 1:
            current_prefix = tuple(generated[-(n - 1):])
            if current_prefix in ngrams:
                for banned_token in ngrams[current_prefix]:
                    if 0 <= banned_token < logits.shape[0]:
                        logits[banned_token] = _NEG_INF

        return logits

    def _check_ngram_repetition(self, tokens: List[int], n: int) -> bool:
        """Check if the last n-gram in tokens is a repetition.

        Parameters
        ----------
        tokens : list of int
            Token sequence to check.
        n : int
            N-gram size.

        Returns
        -------
        bool
            True if the last n-gram has appeared earlier in the sequence.
        """
        if len(tokens) < n:
            return False

        last_ngram = tuple(tokens[-n:])
        for i in range(len(tokens) - n):
            if tuple(tokens[i : i + n]) == last_ngram:
                return True
        return False

    # ------------------------------------------------------------------
    # Finish detection
    # ------------------------------------------------------------------

    def _is_finished(self, beam: BeamHypothesis) -> bool:
        """Check whether a beam hypothesis should be considered finished.

        Parameters
        ----------
        beam : BeamHypothesis
            The hypothesis to check.

        Returns
        -------
        bool
            True if the hypothesis is finished.
        """
        if beam.is_finished:
            return True

        # EOS token check
        if (
            self._config.eos_token_id is not None
            and len(beam.tokens) > beam.prompt_length
            and beam.tokens[-1] == self._config.eos_token_id
        ):
            if beam.length >= self._config.min_length:
                return True

        # Max length check
        if beam.length >= self._config.max_length:
            return True

        return False

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def _finalize(self, beams: List[BeamHypothesis]) -> List[TokenSequence]:
        """Convert finished beam hypotheses to final token sequences.

        Applies length normalisation, deduplicates, and sorts by normalised
        score.

        Parameters
        ----------
        beams : list of BeamHypothesis
            All finished (and possibly some active) beams.

        Returns
        -------
        list of TokenSequence
            Final sequences sorted by normalised score (best first).
        """
        if not beams:
            return []

        # Length normalise scores
        for beam in beams:
            effective_length = max(beam.length, 1)
            beam.normalized_score = self._apply_length_penalty(
                beam.score, effective_length
            )

        # Sort by normalised score (higher is better)
        beams.sort(key=lambda b: b.normalized_score, reverse=True)

        # Deduplicate by generated tokens
        seen: Set[Tuple[int, ...]] = set()
        unique_beams: List[BeamHypothesis] = []
        for beam in beams:
            key = tuple(beam.generated_tokens)
            if key not in seen:
                seen.add(key)
                unique_beams.append(beam)

        # Extract token sequences (generated tokens only, no prompt)
        results: List[TokenSequence] = []
        for beam in unique_beams:
            results.append(beam.generated_tokens)

        return results

    # ------------------------------------------------------------------
    # Pruning and merging
    # ------------------------------------------------------------------

    def _prune_finished(
        self, beams: List[BeamHypothesis]
    ) -> Tuple[List[BeamHypothesis], List[BeamHypothesis]]:
        """Separate beams into active and finished.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Current set of beams.

        Returns
        -------
        active : list of BeamHypothesis
            Beams that are still generating.
        finished : list of BeamHypothesis
            Beams that have finished (EOS or max length).
        """
        active: List[BeamHypothesis] = []
        finished: List[BeamHypothesis] = []

        for beam in beams:
            if self._is_finished(beam):
                beam.is_finished = True
                finished.append(beam)
            else:
                active.append(beam)

        return active, finished

    def _merge_and_select(
        self,
        active: List[BeamHypothesis],
        finished: List[BeamHypothesis],
        beam_width: int,
    ) -> List[BeamHypothesis]:
        """Merge active and finished beams, then select top candidates.

        This is used when we want to compare active beams against already-
        finished beams to decide whether to continue or replace active beams
        with better finished ones.

        Parameters
        ----------
        active : list of BeamHypothesis
            Beams still generating.
        finished : list of BeamHypothesis
            Previously finished beams.
        beam_width : int
            Maximum number of beams to keep active.

        Returns
        -------
        list of BeamHypothesis
            Selected beams (may include both active and finished).
        """
        # Score all beams with length normalisation
        all_beams: List[Tuple[float, int, BeamHypothesis]] = []

        for i, beam in enumerate(active):
            length = max(beam.length, 1)
            norm_score = self._apply_length_penalty(beam.score, length)
            all_beams.append((norm_score, i, beam))

        for i, beam in enumerate(finished):
            length = max(beam.length, 1)
            norm_score = self._apply_length_penalty(beam.score, length)
            all_beams.append((norm_score, len(active) + i, beam))

        # Sort descending by normalised score
        all_beams.sort(key=lambda x: x[0], reverse=True)

        # Keep up to beam_width active beams
        result_active: List[BeamHypothesis] = []
        result_finished: List[BeamHypothesis] = []

        for norm_score, _, beam in all_beams:
            if beam.is_finished:
                result_finished.append(beam)
            else:
                if len(result_active) < beam_width:
                    result_active.append(beam)

        # If we don't have enough active, pad with finished (they're done)
        result = result_active
        remaining_slots = beam_width - len(result)
        if remaining_slots > 0:
            result.extend(result_finished[:remaining_slots])

        return result

    # ------------------------------------------------------------------
    # Diversity bonus computation
    # ------------------------------------------------------------------

    def _compute_diversity_bonus(
        self, beams: List[BeamHypothesis]
    ) -> np.ndarray:
        """Compute per-beam diversity bonuses.

        The diversity bonus for beam *i* is proportional to its average
        distance from all other beams, encouraging the search to maintain
        a diverse set of hypotheses.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Current set of beams.

        Returns
        -------
        np.ndarray
            Diversity bonus for each beam, shape ``(n_beams,)``.
        """
        n = len(beams)
        if n <= 1:
            return np.zeros(n, dtype=np.float64)

        # Compute pairwise distances
        dist_matrix = _pairwise_diversity_matrix(beams, metric="hamming")

        # Normalise distances
        max_dist = dist_matrix.max()
        if max_dist > 0:
            dist_matrix = dist_matrix / max_dist

        # Average distance to other beams
        avg_distances = np.sum(dist_matrix, axis=1) / max(n - 1, 1)

        # Scale by diversity penalty coefficient
        bonuses = avg_distances * self._config.diversity_penalty

        return bonuses

    # ------------------------------------------------------------------
    # Nucleus beam step
    # ------------------------------------------------------------------

    def _nucleus_beam_step(
        self, logits: np.ndarray, p: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a nucleus (top-p) filtered selection for one beam step.

        First filters the logits to the smallest set whose cumulative
        probability mass exceeds ``p``, then samples or ranks within that
        set.

        Parameters
        ----------
        logits : np.ndarray
            Scores of shape ``(vocab_size,)``.
        p : float
            Nucleus threshold in ``(0, 1]``.

        Returns
        -------
        indices : np.ndarray
            Indices of selected tokens, shape ``(n_selected,)``.
        scores : np.ndarray
            Scores of selected tokens, shape ``(n_selected,)``.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        vocab_size = logits.shape[0]

        # Compute probabilities
        probs = _stable_softmax(logits)

        # Sort by probability (descending)
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Find the nucleus (smallest set with cumulative prob >= p)
        cumulative = np.cumsum(sorted_probs)
        # Keep tokens up to and including the one that pushes us past p
        cutoff_idx = np.searchsorted(cumulative, p, side="right")
        cutoff_idx = min(cutoff_idx + 1, vocab_size)

        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_logits = logits[nucleus_indices]

        # Apply Gumbel perturbation within the nucleus
        gumbel_noise = self._gumbel_sampler._sample_gumbel(nucleus_logits.shape)
        perturbed = nucleus_logits / self._config.gumbel_temperature + gumbel_noise

        # Sort by perturbed score
        order = np.argsort(-perturbed)
        nucleus_indices = nucleus_indices[order]
        perturbed = perturbed[order]

        return nucleus_indices, perturbed

    # ------------------------------------------------------------------
    # Ancestral (categorical) beam step
    # ------------------------------------------------------------------

    def _ancestral_beam_step(
        self, logits: np.ndarray, temperature: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform categorical sampling for one beam step.

        Draws multiple samples from the categorical distribution defined by
        ``softmax(logits / temperature)`` and returns unique tokens with their
        log-probabilities.

        Parameters
        ----------
        logits : np.ndarray
            Scores of shape ``(vocab_size,)``.
        temperature : float
            Sampling temperature.

        Returns
        -------
        indices : np.ndarray
            Indices of sampled tokens (unique), shape ``(n_sampled,)``.
        scores : np.ndarray
            Log-probabilities of sampled tokens, shape ``(n_sampled,)``.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        vocab_size = logits.shape[0]

        if temperature <= 0:
            # Greedy
            best = int(np.argmax(logits))
            return np.array([best], dtype=np.int64), np.array(
                [logits[best]], dtype=np.float64
            )

        scaled = logits / temperature
        probs = _stable_softmax(scaled)

        # Ensure valid probability distribution
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            best = int(np.argmax(logits))
            return np.array([best], dtype=np.int64), np.array(
                [logits[best]], dtype=np.float64
            )
        probs /= total

        # Sample multiple times to get diverse tokens
        n_samples = min(
            self._config.beam_width * 3,
            vocab_size,
        )
        try:
            if hasattr(self, "_gumbel_sampler") and self._gumbel_sampler._rng is not None:
                sampled = self._gumbel_sampler._rng.choice(
                    vocab_size, size=n_samples, replace=True, p=probs
                )
            else:
                sampled = np.random.choice(
                    vocab_size, size=n_samples, replace=True, p=probs
                )
        except ValueError:
            # Fallback if probs are invalid
            best = int(np.argmax(logits))
            return np.array([best], dtype=np.int64), np.array(
                [logits[best]], dtype=np.float64
            )

        # Unique tokens
        unique_tokens = list(dict.fromkeys(sampled.tolist()))
        indices = np.array(unique_tokens, dtype=np.int64)
        log_probs = _log_softmax(scaled)
        scores = log_probs[indices]

        # Sort by score (descending)
        order = np.argsort(-scores)
        indices = indices[order]
        scores = scores[order]

        return indices, scores

    # ------------------------------------------------------------------
    # Batch decoding
    # ------------------------------------------------------------------

    def decode_batch(
        self,
        logit_source: LogitSource,
        prompt_batch: List[List[int]],
        n_sequences: Optional[int] = None,
    ) -> List[List[TokenSequence]]:
        """Generate sequences for a batch of prompts.

        Parameters
        ----------
        logit_source : LogitSource
            The logit source callable.
        prompt_batch : list of list of int
            Batch of prompt token ID lists.
        n_sequences : int or None
            Number of sequences per prompt.

        Returns
        -------
        list of list of TokenSequence
            Generated sequences for each prompt.
        """
        results: List[List[TokenSequence]] = []
        for prompt_ids in prompt_batch:
            seqs = self.generate(logit_source, prompt_ids, n_sequences)
            results.append(seqs)
        return results

    # ------------------------------------------------------------------
    # Hyperparameter grid
    # ------------------------------------------------------------------

    @staticmethod
    def get_hyperparameter_grid() -> Dict[str, List[Any]]:
        """Return a grid of hyperparameter values for search.

        Returns
        -------
        dict
            Mapping from parameter name to list of candidate values.
        """
        return {
            "beam_width": [5, 10, 20, 50],
            "temperature": [0.5, 0.7, 1.0, 1.2, 1.5],
            "gumbel_temperature": [0.5, 0.8, 1.0, 1.5, 2.0],
            "length_penalty": [0.0, 0.3, 0.6, 1.0, 1.5],
            "diversity_penalty": [0.0, 0.3, 0.5, 1.0, 2.0],
            "expansion_ratio": [1.5, 2.0, 3.0],
            "sampling_method": ["gumbel", "categorical", "nucleus_beam"],
            "nucleus_p": [0.8, 0.9, 0.95],
            "repetition_penalty": [1.0, 1.1, 1.2, 1.5],
            "no_repeat_ngram_size": [0, 2, 3, 4],
        }

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_memory(
        beam_width: int, max_length: int, vocab_size: int
    ) -> int:
        """Estimate peak memory usage in bytes.

        Provides a rough upper-bound on memory required for the beam search
        state, excluding the model itself.

        Parameters
        ----------
        beam_width : int
            Number of beams.
        max_length : int
            Maximum sequence length.
        vocab_size : int
            Vocabulary size (determines logit array sizes).

        Returns
        -------
        int
            Estimated memory in bytes.
        """
        # Per beam: tokens (int64) + scores + metadata
        tokens_per_beam = max_length * 8  # int64
        scores_per_beam = max_length * 8  # float64 perturbed_scores
        metadata_per_beam = 64  # score, gumbel_score, log_prob, etc.
        per_beam = tokens_per_beam + scores_per_beam + metadata_per_beam

        # Logit arrays during expansion
        logits_per_step = beam_width * vocab_size * 8  # float64

        # Candidate hypotheses (expansion_ratio * beam_width)
        expansion_factor = 2  # default expansion_ratio
        candidates = expansion_factor * beam_width * per_beam

        # Gumbel noise arrays
        gumbel_arrays = vocab_size * 8 * beam_width

        # Diversity matrix
        diversity_matrix = beam_width * beam_width * 8

        total = (
            beam_width * per_beam  # active beams
            + beam_width * per_beam  # finished beams buffer
            + logits_per_step
            + candidates
            + gumbel_arrays
            + diversity_matrix
        )

        return total

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_beam_state(self, beams: List[BeamHypothesis], step: int) -> None:
        """Log the current beam state for debugging.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Current set of beams.
        step : int
            Current decoding step.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        n_active = sum(1 for b in beams if not b.is_finished)
        n_finished = len(beams) - n_active
        scores = [b.score for b in beams]
        gumbel_scores = [b.gumbel_score for b in beams]

        avg_score = np.mean(scores) if scores else 0.0
        avg_gumbel = np.mean(gumbel_scores) if gumbel_scores else 0.0
        max_len = max((b.length for b in beams), default=0)
        min_len = min((b.length for b in beams), default=0)

        elapsed = time.time() - self._start_time

        logger.debug(
            "Step %d | active=%d finished=%d total_finished=%d | "
            "avg_score=%.4f avg_gumbel=%.4f | "
            "len_range=[%d, %d] | elapsed=%.2fs",
            step,
            n_active,
            n_finished,
            len(self._finished_beams),
            avg_score,
            avg_gumbel,
            min_len,
            max_len,
            elapsed,
        )

        # Log top 3 beams
        sorted_beams = sorted(beams, key=lambda b: b.gumbel_score, reverse=True)
        for i, beam in enumerate(sorted_beams[:3]):
            gen = beam.generated_tokens
            preview = gen[:10]
            logger.debug(
                "  beam[%d]: score=%.4f gumbel=%.4f len=%d tokens=%s%s",
                i,
                beam.score,
                beam.gumbel_score,
                beam.length,
                preview,
                "..." if len(gen) > 10 else "",
            )

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of this algorithm instance.

        Returns
        -------
        str
            Description string.
        """
        parts = [
            f"Stochastic Beam Search (Kool et al., 2019)",
            f"  beam_width={self._config.beam_width}",
            f"  temperature={self._config.temperature}",
            f"  gumbel_temperature={self._config.gumbel_temperature}",
            f"  sampling_method={self._config.sampling_method}",
            f"  length_penalty={self._config.length_penalty}",
            f"  diversity_penalty={self._config.diversity_penalty}",
            f"  use_gumbel_top_k={self._config.use_gumbel_top_k}",
            f"  expansion_ratio={self._config.expansion_ratio}",
            f"  n_best={self._config.n_best}",
            f"  max_length={self._config.max_length}",
            f"  min_length={self._config.min_length}",
        ]
        if self._config.repetition_penalty > 1.0:
            parts.append(f"  repetition_penalty={self._config.repetition_penalty}")
        if self._config.no_repeat_ngram_size > 0:
            parts.append(
                f"  no_repeat_ngram_size={self._config.no_repeat_ngram_size}"
            )
        return "\n".join(parts)


# =========================================================================
# StochasticBeamAnalyzer — post-hoc analysis
# =========================================================================


class StochasticBeamAnalyzer:
    """Analysis utilities for stochastic beam search results.

    Provides methods for computing diversity metrics, score distributions,
    and other diagnostics on the output of stochastic beam search.
    """

    @staticmethod
    def compute_self_bleu(
        sequences: List[List[int]], n: int = 4
    ) -> float:
        """Compute self-BLEU as a diversity metric.

        Self-BLEU measures how similar each sequence is to the others.
        Lower self-BLEU means higher diversity.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences to evaluate.
        n : int
            Maximum n-gram order for BLEU computation.

        Returns
        -------
        float
            Average self-BLEU score in ``[0, 1]``.
        """
        if len(sequences) <= 1:
            return 0.0

        def _count_ngrams(seq: List[int], order: int) -> Dict[Tuple[int, ...], int]:
            counts: Dict[Tuple[int, ...], int] = {}
            for i in range(len(seq) - order + 1):
                ngram = tuple(seq[i : i + order])
                counts[ngram] = counts.get(ngram, 0) + 1
            return counts

        def _modified_precision(
            candidate: List[int],
            references: List[List[int]],
            order: int,
        ) -> float:
            cand_counts = _count_ngrams(candidate, order)
            if not cand_counts:
                return 0.0

            # Clipped counts
            max_ref_counts: Dict[Tuple[int, ...], int] = {}
            for ref in references:
                ref_counts = _count_ngrams(ref, order)
                for ngram, count in ref_counts.items():
                    max_ref_counts[ngram] = max(
                        max_ref_counts.get(ngram, 0), count
                    )

            clipped = 0
            total = 0
            for ngram, count in cand_counts.items():
                clipped += min(count, max_ref_counts.get(ngram, 0))
                total += count

            return clipped / total if total > 0 else 0.0

        bleu_scores: List[float] = []
        for i, seq in enumerate(sequences):
            refs = [sequences[j] for j in range(len(sequences)) if j != i]
            if not refs or len(seq) == 0:
                continue

            precisions: List[float] = []
            for order in range(1, n + 1):
                p = _modified_precision(seq, refs, order)
                precisions.append(p)

            if all(p > 0 for p in precisions):
                log_avg = sum(math.log(p) for p in precisions) / len(precisions)
                bleu = math.exp(log_avg)
            else:
                bleu = 0.0

            # Brevity penalty
            ref_lens = [len(r) for r in refs]
            closest_len = min(ref_lens, key=lambda r: abs(r - len(seq)))
            if len(seq) < closest_len:
                bp = math.exp(1 - closest_len / max(len(seq), 1))
            else:
                bp = 1.0

            bleu_scores.append(bleu * bp)

        return float(np.mean(bleu_scores)) if bleu_scores else 0.0

    @staticmethod
    def compute_distinct_ngrams(
        sequences: List[List[int]], n: int = 2
    ) -> float:
        """Compute distinct-n metric (fraction of unique n-grams).

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.
        n : int
            N-gram order.

        Returns
        -------
        float
            Fraction of distinct n-grams across all sequences.
        """
        all_ngrams: List[Tuple[int, ...]] = []
        for seq in sequences:
            for i in range(len(seq) - n + 1):
                all_ngrams.append(tuple(seq[i : i + n]))

        if not all_ngrams:
            return 0.0

        return len(set(all_ngrams)) / len(all_ngrams)

    @staticmethod
    def compute_pairwise_edit_distance(
        sequences: List[List[int]],
    ) -> float:
        """Compute average pairwise edit distance (normalised).

        Uses a simplified Levenshtein distance.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.

        Returns
        -------
        float
            Average normalised edit distance in ``[0, 1]``.
        """
        if len(sequences) <= 1:
            return 0.0

        def _edit_distance(a: List[int], b: List[int]) -> int:
            m, n = len(a), len(b)
            # Use O(n) space DP
            prev = list(range(n + 1))
            curr = [0] * (n + 1)
            for i in range(1, m + 1):
                curr[0] = i
                for j in range(1, n + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    curr[j] = min(
                        prev[j] + 1,      # deletion
                        curr[j - 1] + 1,   # insertion
                        prev[j - 1] + cost  # substitution
                    )
                prev, curr = curr, prev
            return prev[n]

        distances: List[float] = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                max_len = max(len(sequences[i]), len(sequences[j]), 1)
                dist = _edit_distance(sequences[i], sequences[j])
                distances.append(dist / max_len)

        return float(np.mean(distances))

    @staticmethod
    def score_distribution_stats(
        beams: List[BeamHypothesis],
    ) -> Dict[str, float]:
        """Compute statistics on beam score distributions.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Beam hypotheses (typically the final set).

        Returns
        -------
        dict
            Statistics including mean, std, min, max, and entropy-like
            measures for both raw scores and Gumbel scores.
        """
        if not beams:
            return {
                "score_mean": 0.0,
                "score_std": 0.0,
                "score_min": 0.0,
                "score_max": 0.0,
                "gumbel_score_mean": 0.0,
                "gumbel_score_std": 0.0,
                "length_mean": 0.0,
                "length_std": 0.0,
            }

        scores = np.array([b.score for b in beams])
        gumbel_scores = np.array([b.gumbel_score for b in beams])
        lengths = np.array([b.length for b in beams], dtype=np.float64)

        return {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "gumbel_score_mean": float(np.mean(gumbel_scores)),
            "gumbel_score_std": float(np.std(gumbel_scores)),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
        }

    @staticmethod
    def analyze_gumbel_perturbation(
        beams: List[BeamHypothesis],
    ) -> Dict[str, Any]:
        """Analyze the Gumbel perturbation across beams.

        Parameters
        ----------
        beams : list of BeamHypothesis
            Beam hypotheses with ``perturbed_scores`` populated.

        Returns
        -------
        dict
            Analysis results including perturbation statistics,
            rank correlation with unperturbed scores, and per-step noise.
        """
        if not beams:
            return {"n_beams": 0}

        all_perturbations: List[float] = []
        for beam in beams:
            all_perturbations.extend(beam.perturbed_scores)

        perturbations = np.array(all_perturbations) if all_perturbations else np.array([0.0])

        # Rank correlation: do Gumbel scores preserve the ordering of raw scores?
        raw_scores = np.array([b.score for b in beams])
        gumbel_scores = np.array([b.gumbel_score for b in beams])

        if len(beams) >= 2:
            raw_ranks = np.argsort(np.argsort(-raw_scores))
            gumbel_ranks = np.argsort(np.argsort(-gumbel_scores))
            rank_diff = np.abs(raw_ranks - gumbel_ranks).astype(np.float64)
            avg_rank_displacement = float(np.mean(rank_diff))
            max_rank_displacement = float(np.max(rank_diff))
        else:
            avg_rank_displacement = 0.0
            max_rank_displacement = 0.0

        return {
            "n_beams": len(beams),
            "perturbation_mean": float(np.mean(perturbations)),
            "perturbation_std": float(np.std(perturbations)),
            "perturbation_min": float(np.min(perturbations)),
            "perturbation_max": float(np.max(perturbations)),
            "avg_rank_displacement": avg_rank_displacement,
            "max_rank_displacement": max_rank_displacement,
            "score_gumbel_correlation": float(
                np.corrcoef(raw_scores, gumbel_scores)[0, 1]
            )
            if len(beams) >= 2
            else 1.0,
        }


# =========================================================================
# StochasticBeamSearchWithSIS — Sequential Importance Sampling variant
# =========================================================================


class StochasticBeamSearchWithSIS(StochasticBeamSearch):
    """Stochastic Beam Search with Sequential Importance Sampling weights.

    Extends the base stochastic beam search with importance weights that
    correct for the bias introduced by the beam selection procedure.  This
    follows the SIS interpretation from Kool et al. (2019, §4) and the
    improved estimator from Jinnai et al. (2024).

    The importance weights allow unbiased estimation of expectations under
    the true sequence distribution, even though we are using a beam search
    proposal distribution.

    Parameters
    ----------
    config : StochasticBeamConfig
        Algorithm configuration.
    """

    def __init__(self, config: StochasticBeamConfig) -> None:
        super().__init__(config)
        self._importance_weights: List[float] = []
        self._proposal_log_probs: List[float] = []
        self._target_log_probs: List[float] = []

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: Optional[int] = None,
    ) -> List[TokenSequence]:
        """Generate with SIS importance weight tracking.

        In addition to the base generate(), this variant tracks importance
        weights for each beam, enabling unbiased estimation.
        """
        self._importance_weights = []
        self._proposal_log_probs = []
        self._target_log_probs = []

        results = super().generate(logit_source, prompt_ids, n_sequences)

        # Compute importance weights for finished beams
        self._compute_importance_weights()

        return results

    def _compute_importance_weights(self) -> None:
        """Compute normalised importance weights for all finished beams.

        The importance weight for beam *i* is:
        ``w_i = p_target(y_i) / q_proposal(y_i)``

        where ``p_target`` is the model distribution and ``q_proposal`` is
        the beam search proposal distribution.
        """
        if not self._finished_beams:
            return

        log_weights: List[float] = []
        for beam in self._finished_beams:
            # Target: cumulative log-probability under the model
            target_log_p = beam.log_prob

            # Proposal: the beam search selection probability
            # Approximation: use the Gumbel score as a proxy for proposal log-prob
            proposal_log_p = beam.gumbel_score if beam.gumbel_score != 0 else beam.score

            log_w = target_log_p - proposal_log_p
            log_weights.append(log_w)

        if not log_weights:
            return

        # Normalise in log-space
        log_weights_arr = np.array(log_weights, dtype=np.float64)
        max_log_w = np.max(log_weights_arr)
        shifted = log_weights_arr - max_log_w
        exp_shifted = np.exp(shifted)
        total = np.sum(exp_shifted)

        if total > 0 and np.isfinite(total):
            self._importance_weights = (exp_shifted / total).tolist()
        else:
            n = len(log_weights)
            self._importance_weights = [1.0 / n] * n

    def get_importance_weights(self) -> List[float]:
        """Return the normalised importance weights.

        Returns
        -------
        list of float
            Normalised importance weights, one per finished beam.
        """
        return list(self._importance_weights)

    def weighted_score(self, beam_idx: int) -> float:
        """Return the importance-weighted score for a beam.

        Parameters
        ----------
        beam_idx : int
            Index into the finished beams list.

        Returns
        -------
        float
            Weighted score.
        """
        if beam_idx >= len(self._importance_weights):
            return 0.0
        if beam_idx >= len(self._finished_beams):
            return 0.0
        return (
            self._finished_beams[beam_idx].normalized_score
            * self._importance_weights[beam_idx]
        )

    def effective_sample_size(self) -> float:
        """Compute the effective sample size (ESS) of the importance weights.

        ESS measures how many "effective" independent samples the weighted
        beam set corresponds to.  ESS = 1/sum(w_i^2) when weights are
        normalised.

        Returns
        -------
        float
            Effective sample size.
        """
        if not self._importance_weights:
            return 0.0

        weights = np.array(self._importance_weights, dtype=np.float64)
        sum_sq = np.sum(weights ** 2)

        if sum_sq <= 0 or not np.isfinite(sum_sq):
            return 0.0

        return 1.0 / sum_sq


# =========================================================================
# Factory function
# =========================================================================


def create_stochastic_beam_search(
    beam_width: int = 10,
    temperature: float = 1.0,
    gumbel_temperature: float = 1.0,
    length_penalty: float = 0.6,
    diversity_penalty: float = 0.5,
    n_best: int = 5,
    sampling_method: str = "gumbel",
    seed: Optional[int] = None,
    use_sis: bool = False,
    **kwargs: Any,
) -> StochasticBeamSearch:
    """Convenience factory for creating a StochasticBeamSearch instance.

    Parameters
    ----------
    beam_width : int
        Number of beams.
    temperature : float
        Softmax temperature.
    gumbel_temperature : float
        Gumbel noise temperature.
    length_penalty : float
        Length normalisation exponent.
    diversity_penalty : float
        Diversity penalty coefficient.
    n_best : int
        Number of sequences to return.
    sampling_method : str
        One of ``'gumbel'``, ``'categorical'``, ``'nucleus_beam'``.
    seed : int or None
        Random seed.
    use_sis : bool
        If True, return a ``StochasticBeamSearchWithSIS`` instance.
    **kwargs
        Additional configuration parameters.

    Returns
    -------
    StochasticBeamSearch
        Configured algorithm instance.
    """
    config = StochasticBeamConfig(
        beam_width=beam_width,
        temperature=temperature,
        gumbel_temperature=gumbel_temperature,
        length_penalty=length_penalty,
        diversity_penalty=diversity_penalty,
        n_best=n_best,
        sampling_method=sampling_method,
        seed=seed,
        **{k: v for k, v in kwargs.items() if hasattr(StochasticBeamConfig, k)},
    )

    if use_sis:
        return StochasticBeamSearchWithSIS(config)
    return StochasticBeamSearch(config)
