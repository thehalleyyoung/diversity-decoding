"""
Diverse Beam Search for the Diversity Decoding Arena.
=====================================================

Implements the Diverse Beam Search algorithm (Vijayakumar et al., 2018) which
partitions beams into groups and applies a diversity-augmented scoring function
to encourage inter-group diversity.  Within each group, standard beam search
operates; across groups, a diversity penalty discourages selecting tokens that
were already chosen by earlier groups in the same time step.

Key components
--------------
* **DiverseBeamConfig** — dataclass extending ``DecodingConfig`` with beam
  search and diversity hyper-parameters (number of beams/groups, penalty
  type and strength, length normalisation, etc.).
* **DiverseBeamSearch** — the main ``DecodingAlgorithm`` implementation.
  Overrides ``generate()`` entirely to run group-sequential beam search with
  diversity penalties at every decoding step.
* **BeamState / BeamGroup / FinishedBeam** — lightweight dataclasses
  representing the internal search state.
* **BeamScorer** — utilities for scoring, length penalty, and top-k selection.
* **DiversityPenalizer** — encapsulates the three supported penalty types
  (Hamming, n-gram, embedding).
* **DiverseBeamAnalyzer** — post-hoc analysis of group diversity, penalty
  sensitivity, and beam trajectories.

References
----------
- Vijayakumar, A. K., Cogswell, M., Selvaraju, R. R., Sun, Q., Lee, S.,
  Corse, D., & Batra, D. (2018). *Diverse Beam Search: Decoding Diverse
  Solutions from Neural Sequence Models*. AAAI 2018.
"""

from __future__ import annotations

import collections
import copy
import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.special import log_softmax, logsumexp, softmax

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEG_INF: float = float("-inf")
_LOG_EPS: float = 1e-10


# =========================================================================
# Helper functions
# =========================================================================


def _stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    return log_softmax(logits, axis=-1)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    return softmax(logits, axis=-1)


def _extract_ngrams(token_ids: List[int], order: int) -> Dict[Tuple[int, ...], int]:
    """Extract n-gram counts from a token-id sequence.

    Parameters
    ----------
    token_ids : list of int
        Token sequence.
    order : int
        N-gram order (e.g. 2 for bigrams).

    Returns
    -------
    dict mapping n-gram tuples to their occurrence count.
    """
    ngrams: Dict[Tuple[int, ...], int] = collections.defaultdict(int)
    for i in range(len(token_ids) - order + 1):
        ng = tuple(token_ids[i : i + order])
        ngrams[ng] += 1
    return dict(ngrams)


def _jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    """Jaccard similarity between two integer sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _hamming_distance_tokens(seq_a: List[int], seq_b: List[int]) -> int:
    """Positional Hamming distance between two token sequences.

    For sequences of different length, the shorter one is padded conceptually
    with a sentinel value that never matches.
    """
    max_len = max(len(seq_a), len(seq_b))
    dist = 0
    for i in range(max_len):
        t_a = seq_a[i] if i < len(seq_a) else -1
        t_b = seq_b[i] if i < len(seq_b) else -2
        if t_a != t_b:
            dist += 1
    return dist


def _ngram_overlap(
    seq_a: List[int], seq_b: List[int], order: int = 2
) -> float:
    """Fraction of n-grams shared between two sequences (Jaccard on n-gram sets)."""
    if len(seq_a) < order or len(seq_b) < order:
        return 0.0
    set_a = set(_extract_ngrams(seq_a, order).keys())
    set_b = set(_extract_ngrams(seq_b, order).keys())
    return _jaccard_similarity(set_a, set_b)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _LOG_EPS or norm_b < _LOG_EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pairwise_bleu_diversity(sequences: List[List[int]], max_order: int = 4) -> float:
    """Average pairwise (1 - BLEU overlap) across sequences.

    Uses a simplified n-gram precision proxy for BLEU.
    """
    if len(sequences) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            overlap = 0.0
            for order in range(1, max_order + 1):
                overlap += _ngram_overlap(sequences[i], sequences[j], order)
            overlap /= max_order
            total += 1.0 - overlap
            count += 1
    return total / count if count > 0 else 0.0


def _self_bleu(sequences: List[List[int]], max_order: int = 4) -> float:
    """Self-BLEU: average BLEU of each sequence against all others.

    Lower self-BLEU indicates higher diversity.
    """
    if len(sequences) < 2:
        return 0.0
    total = 0.0
    for i in range(len(sequences)):
        refs = [sequences[j] for j in range(len(sequences)) if j != i]
        overlaps = []
        for ref in refs:
            overlap = 0.0
            for order in range(1, max_order + 1):
                overlap += _ngram_overlap(sequences[i], ref, order)
            overlaps.append(overlap / max_order)
        total += max(overlaps) if overlaps else 0.0
    return total / len(sequences)


def _distinct_ngrams(sequences: List[List[int]], order: int = 2) -> float:
    """Distinct-N: fraction of unique n-grams in the union of all sequences."""
    all_ngrams: List[Tuple[int, ...]] = []
    for seq in sequences:
        for i in range(len(seq) - order + 1):
            all_ngrams.append(tuple(seq[i : i + order]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class DiverseBeamConfig(DecodingConfig):
    """Configuration for Diverse Beam Search.

    Attributes
    ----------
    num_beams : int
        Total number of beams across all groups.
    num_beam_groups : int
        Number of beam groups.  Must evenly divide ``num_beams``.
    diversity_penalty : float
        Strength of the diversity penalty applied to later groups.
        Higher values push groups towards more distinct outputs.
    diversity_type : str
        Type of diversity penalty: ``"hamming"``, ``"ngram"``, or
        ``"embedding"``.
    length_penalty : float
        Exponential length penalty (alpha).  Values > 1 favour longer
        sequences; values < 1 favour shorter ones.
    early_stopping : bool
        If ``True``, stop a group as soon as ``group_size`` hypotheses
        have been completed.
    num_return_sequences : int
        Number of sequences to return (must be <= ``num_beams``).
    ngram_diversity_order : int
        N-gram order used when ``diversity_type == "ngram"``.
    """

    algorithm_name: str = "DiverseBeamSearch"
    num_beams: int = 20
    num_beam_groups: int = 4
    diversity_penalty: float = 1.0
    diversity_type: str = "hamming"
    length_penalty: float = 1.0
    early_stopping: bool = False
    num_return_sequences: int = 20
    ngram_diversity_order: int = 2

    @property
    def group_size(self) -> int:
        """Number of beams per group."""
        return self.num_beams // self.num_beam_groups

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty == valid)."""
        errors = super().validate()
        if self.num_beams < 1:
            errors.append("num_beams must be >= 1")
        if self.num_beam_groups < 1:
            errors.append("num_beam_groups must be >= 1")
        if self.num_beams % self.num_beam_groups != 0:
            errors.append(
                f"num_beams ({self.num_beams}) must be divisible by "
                f"num_beam_groups ({self.num_beam_groups})"
            )
        if self.diversity_penalty < 0:
            errors.append("diversity_penalty must be >= 0")
        if self.diversity_type not in ("hamming", "ngram", "embedding"):
            errors.append(
                f"diversity_type must be 'hamming', 'ngram', or 'embedding', "
                f"got '{self.diversity_type}'"
            )
        if self.length_penalty < 0:
            errors.append("length_penalty must be >= 0")
        if self.num_return_sequences < 1:
            errors.append("num_return_sequences must be >= 1")
        if self.num_return_sequences > self.num_beams:
            errors.append(
                f"num_return_sequences ({self.num_return_sequences}) must be "
                f"<= num_beams ({self.num_beams})"
            )
        if self.ngram_diversity_order < 1:
            errors.append("ngram_diversity_order must be >= 1")
        return errors

    # -- serialisation helpers ----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["group_size"] = self.group_size
        return d

    # -- hyperparameter space -----------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        return {
            "num_beams": {"type": "int", "low": 4, "high": 60},
            "num_beam_groups": {"type": "int", "low": 2, "high": 10},
            "diversity_penalty": {"type": "float", "low": 0.0, "high": 5.0},
            "diversity_type": {
                "type": "categorical",
                "choices": ["hamming", "ngram", "embedding"],
            },
            "length_penalty": {"type": "float", "low": 0.5, "high": 2.0},
            "ngram_diversity_order": {"type": "int", "low": 1, "high": 4},
        }


# =========================================================================
# Internal state dataclasses
# =========================================================================


@dataclass
class BeamGroup:
    """State for a single beam group.

    Attributes
    ----------
    beam_ids : list of list of int
        Token sequences for each beam in this group.
    beam_scores : list of float
        Accumulated log-probability scores for each beam.
    beam_indices : list of int
        Original beam indices (used for backtracking).
    selected_tokens : list of set of int
        Per-step sets of tokens selected by this group (used for penalty).
    is_done : list of bool
        Whether each beam in the group has produced a completed hypothesis.
    """

    beam_ids: List[List[int]] = field(default_factory=list)
    beam_scores: List[float] = field(default_factory=list)
    beam_indices: List[int] = field(default_factory=list)
    selected_tokens: List[Set[int]] = field(default_factory=list)
    is_done: List[bool] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of beams in the group."""
        return len(self.beam_ids)

    @property
    def all_done(self) -> bool:
        """True when every beam in the group is done."""
        return all(self.is_done) if self.is_done else False

    @property
    def num_active(self) -> int:
        """Number of beams still generating."""
        return sum(1 for d in self.is_done if not d)

    def get_active_sequences(self) -> List[List[int]]:
        """Return sequences of beams that are still generating."""
        return [
            self.beam_ids[i]
            for i in range(self.size)
            if not self.is_done[i]
        ]

    def clone(self) -> "BeamGroup":
        """Deep copy of this group."""
        return copy.deepcopy(self)


@dataclass
class FinishedBeam:
    """A completed hypothesis.

    Attributes
    ----------
    sequence : list of int
        Full token sequence (prompt + generated).
    score : float
        Raw accumulated log-probability score.
    group_id : int
        Index of the group that produced this hypothesis.
    length_normalized_score : float
        Score after length normalisation.
    """

    sequence: List[int] = field(default_factory=list)
    score: float = 0.0
    group_id: int = 0
    length_normalized_score: float = 0.0

    def __lt__(self, other: "FinishedBeam") -> bool:
        """Comparison for heap operations (min-heap by normalised score)."""
        return self.length_normalized_score < other.length_normalized_score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinishedBeam):
            return NotImplemented
        return self.length_normalized_score == other.length_normalized_score

    def __repr__(self) -> str:
        return (
            f"FinishedBeam(group={self.group_id}, "
            f"score={self.score:.4f}, "
            f"norm_score={self.length_normalized_score:.4f}, "
            f"len={len(self.sequence)})"
        )


@dataclass
class BeamState:
    """Full search state across all beam groups.

    Attributes
    ----------
    groups : list of BeamGroup
        One entry per beam group.
    finished_sequences : list of FinishedBeam
        All completed hypotheses across groups.
    step : int
        Current decoding step (0-indexed).
    prompt_ids : list of int
        The original prompt tokens.
    """

    groups: List[BeamGroup] = field(default_factory=list)
    finished_sequences: List[FinishedBeam] = field(default_factory=list)
    step: int = 0
    prompt_ids: List[int] = field(default_factory=list)

    @property
    def all_groups_done(self) -> bool:
        """True when every group is finished."""
        return all(g.all_done for g in self.groups)

    @property
    def total_beams(self) -> int:
        """Total number of active beams across all groups."""
        return sum(g.size for g in self.groups)

    @property
    def total_active(self) -> int:
        """Total number of beams still generating."""
        return sum(g.num_active for g in self.groups)

    def top_k_finished(self, k: int) -> List[FinishedBeam]:
        """Return the *k* best finished hypotheses by normalised score."""
        return sorted(
            self.finished_sequences,
            key=lambda fb: fb.length_normalized_score,
            reverse=True,
        )[:k]

    def all_sequences_flat(self) -> List[List[int]]:
        """All active beam sequences across groups as a flat list."""
        seqs: List[List[int]] = []
        for g in self.groups:
            seqs.extend(g.beam_ids)
        return seqs

    def clone(self) -> "BeamState":
        """Deep copy of the full state."""
        return copy.deepcopy(self)


# =========================================================================
# BeamScorer
# =========================================================================


class BeamScorer:
    """Utilities for scoring beam hypotheses and selecting top-k candidates.

    This class is stateless and groups related scoring operations.
    """

    def __init__(self, length_penalty_alpha: float = 1.0) -> None:
        """
        Parameters
        ----------
        length_penalty_alpha : float
            Exponent for length penalty.  ``1.0`` means no penalty.
        """
        self.alpha = length_penalty_alpha

    # -- public interface ---------------------------------------------------

    def score_hypotheses(
        self,
        beam_scores: np.ndarray,
        next_logits: np.ndarray,
        beam_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score all next-token candidates and return top-k.

        Parameters
        ----------
        beam_scores : np.ndarray
            Shape ``(beam_size,)`` — accumulated scores for current beams.
        next_logits : np.ndarray
            Shape ``(beam_size, vocab_size)`` — next-token log-probabilities.
        beam_size : int
            Number of beams to keep.

        Returns
        -------
        beam_indices : np.ndarray of int — shape ``(beam_size,)``
            Which beam each new hypothesis originates from.
        token_indices : np.ndarray of int — shape ``(beam_size,)``
            Token id selected for each new hypothesis.
        new_scores : np.ndarray of float — shape ``(beam_size,)``
            Accumulated scores after adding the new token.
        """
        vocab_size = next_logits.shape[-1]

        # Broadcast beam scores to (beam_size, vocab_size) and add
        # log-probabilities to get total scores for all continuations.
        expanded_scores = beam_scores[:, np.newaxis] + next_logits  # (B, V)

        # Flatten and pick top-k
        flat_scores = expanded_scores.reshape(-1)
        beam_indices, token_indices, new_scores = self._top_k_candidates(
            flat_scores, beam_size, vocab_size
        )
        return beam_indices, token_indices, new_scores

    def length_penalty(self, length: int, alpha: Optional[float] = None) -> float:
        """Wu et al. (2016) length penalty: ``((5 + length) / 6) ^ alpha``.

        Parameters
        ----------
        length : int
            Length of the hypothesis (number of generated tokens).
        alpha : float, optional
            Override for the default alpha.

        Returns
        -------
        float
            Penalty divisor (>= 1 for alpha >= 0).
        """
        a = alpha if alpha is not None else self.alpha
        if a == 0.0:
            return 1.0
        return ((5.0 + length) / 6.0) ** a

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _top_k_candidates(
        scores: np.ndarray, k: int, vocab_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select the *k* highest-scoring (beam, token) pairs.

        Parameters
        ----------
        scores : np.ndarray
            Flattened scores of shape ``(beam_size * vocab_size,)``.
        k : int
            Number of candidates to keep.
        vocab_size : int
            Size of the vocabulary (for un-flattening indices).

        Returns
        -------
        beam_indices, token_indices, values — each shape ``(k,)``.
        """
        k = min(k, len(scores))

        # Use argpartition for efficiency when vocab_size is very large
        if len(scores) > 2 * k:
            top_flat = np.argpartition(scores, -k)[-k:]
            # Sort the top-k by score descending
            sorted_order = np.argsort(scores[top_flat])[::-1]
            top_flat = top_flat[sorted_order]
        else:
            top_flat = np.argsort(scores)[::-1][:k]

        beam_indices = top_flat // vocab_size
        token_indices = top_flat % vocab_size
        values = scores[top_flat]

        return (
            beam_indices.astype(np.intp),
            token_indices.astype(np.intp),
            values.astype(np.float64),
        )


# =========================================================================
# DiversityPenalizer
# =========================================================================


class DiversityPenalizer:
    """Computes diversity penalties for Diverse Beam Search.

    Supports three penalty modes:

    * **hamming** — penalises tokens chosen by earlier groups at the current step.
    * **ngram** — penalises tokens that would create n-grams already present
      in earlier groups' hypotheses.
    * **embedding** — penalises tokens whose embeddings are close to the
      average embeddings of tokens selected by earlier groups.

    Parameters
    ----------
    penalty_type : str
        One of ``"hamming"``, ``"ngram"``, ``"embedding"``.
    penalty_strength : float
        Multiplier for the penalty term.
    ngram_order : int
        N-gram order for the ``"ngram"`` penalty (default 2).
    """

    SUPPORTED_TYPES = ("hamming", "ngram", "embedding")

    def __init__(
        self,
        penalty_type: str = "hamming",
        penalty_strength: float = 1.0,
        ngram_order: int = 2,
    ) -> None:
        if penalty_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown penalty type '{penalty_type}'. "
                f"Supported: {self.SUPPORTED_TYPES}"
            )
        self.penalty_type = penalty_type
        self.penalty_strength = penalty_strength
        self.ngram_order = ngram_order

    # -- public interface ---------------------------------------------------

    def compute_penalty(
        self,
        previous_groups_tokens: List[Set[int]],
        current_logits: np.ndarray,
        previous_group_sequences: Optional[List[List[List[int]]]] = None,
        previous_group_embeddings: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute the diversity penalty for a batch of beams.

        Parameters
        ----------
        previous_groups_tokens : list of set of int
            Per-group sets of tokens selected at the current step by all
            groups processed *before* the current one.
        current_logits : np.ndarray
            Shape ``(beam_size, vocab_size)`` — raw logits for the current
            group's beams.
        previous_group_sequences : list of list of list of int, optional
            Full sequences from previous groups (needed for n-gram penalty).
        previous_group_embeddings : list of np.ndarray, optional
            Average embeddings of previous groups' selected tokens (needed
            for embedding penalty).

        Returns
        -------
        np.ndarray
            Penalty array of shape ``(beam_size, vocab_size)`` to be
            *subtracted* from logits.
        """
        if self.penalty_strength == 0.0 or not previous_groups_tokens:
            return np.zeros_like(current_logits)

        if self.penalty_type == "hamming":
            return self.hamming_penalty(previous_groups_tokens, current_logits)
        elif self.penalty_type == "ngram":
            seqs = previous_group_sequences if previous_group_sequences else []
            return self.ngram_penalty(seqs, current_logits, self.ngram_order)
        elif self.penalty_type == "embedding":
            embs = previous_group_embeddings if previous_group_embeddings else []
            return self.embedding_penalty(embs, current_logits)
        else:
            return np.zeros_like(current_logits)

    # -- penalty implementations --------------------------------------------

    def hamming_penalty(
        self, prev_tokens: List[Set[int]], logits: np.ndarray
    ) -> np.ndarray:
        """Hamming diversity penalty.

        For each token that was selected by *any* previous group at this step,
        add ``penalty_strength`` to the penalty vector.  This makes it less
        likely for the current group to select the same token.

        Parameters
        ----------
        prev_tokens : list of set of int
            Token sets selected by each previous group at this step.
        logits : np.ndarray
            Shape ``(beam_size, vocab_size)`` or ``(vocab_size,)``.

        Returns
        -------
        np.ndarray
            Penalty of same shape as *logits*.
        """
        is_1d = logits.ndim == 1
        if is_1d:
            logits = logits[np.newaxis, :]

        penalty = np.zeros_like(logits)
        all_prev = set()
        for token_set in prev_tokens:
            all_prev.update(token_set)

        for token_id in all_prev:
            if 0 <= token_id < penalty.shape[-1]:
                # Count how many groups selected this token
                count = sum(1 for ts in prev_tokens if token_id in ts)
                penalty[:, token_id] += self.penalty_strength * count

        return penalty[0] if is_1d else penalty

    def ngram_penalty(
        self,
        prev_sequences: List[List[List[int]]],
        logits: np.ndarray,
        order: int = 2,
    ) -> np.ndarray:
        """N-gram diversity penalty.

        Penalises tokens that would create an n-gram already seen in any
        previous group's hypotheses.  For each beam in the current group,
        looks at the last ``(order - 1)`` tokens to form a partial n-gram,
        then checks which completing tokens would yield an n-gram found in
        previous groups.

        Parameters
        ----------
        prev_sequences : list of list of list of int
            ``prev_sequences[g][b]`` = token sequence for beam *b* of
            previous group *g*.
        logits : np.ndarray
            Shape ``(beam_size, vocab_size)`` or ``(vocab_size,)``.
        order : int
            N-gram order.

        Returns
        -------
        np.ndarray
            Penalty of same shape as *logits*.
        """
        is_1d = logits.ndim == 1
        if is_1d:
            logits = logits[np.newaxis, :]

        penalty = np.zeros_like(logits)
        vocab_size = logits.shape[-1]

        # Collect all n-grams from previous groups
        prev_ngrams: Dict[Tuple[int, ...], int] = collections.defaultdict(int)
        for group_seqs in prev_sequences:
            for seq in group_seqs:
                for ng, count in _extract_ngrams(seq, order).items():
                    prev_ngrams[ng] += count

        if not prev_ngrams:
            return penalty[0] if is_1d else penalty

        # For the penalty, we assume the "current suffix" is the last (order-1)
        # tokens of a representative beam. Since we don't know each individual
        # beam's suffix here, we penalise globally: any token that could complete
        # a previously seen n-gram based on any (order-1) suffix present in
        # previous groups.
        # Build a mapping: (prefix) -> set of completing tokens seen in prev groups
        prefix_to_completions: Dict[Tuple[int, ...], Set[int]] = collections.defaultdict(set)
        for ng in prev_ngrams:
            prefix = ng[:-1]
            completion = ng[-1]
            prefix_to_completions[prefix].add(completion)

        # Penalise: for each possible prefix of length (order-1), penalise
        # the completing tokens across all beams.
        # Since we don't have per-beam context here, apply a blanket penalty
        # for any completion token that appeared in a previous group's n-gram.
        all_completing_tokens: Set[int] = set()
        for completions in prefix_to_completions.values():
            all_completing_tokens.update(completions)

        for token_id in all_completing_tokens:
            if 0 <= token_id < vocab_size:
                penalty[:, token_id] += self.penalty_strength

        return penalty[0] if is_1d else penalty

    def embedding_penalty(
        self, prev_embeddings: List[np.ndarray], logits: np.ndarray
    ) -> np.ndarray:
        """Embedding-based diversity penalty.

        Uses the logit vector itself as a proxy for token embeddings.  Computes
        cosine similarity between the current logits and the average of previous
        groups' logit patterns, then penalises tokens proportionally.

        Parameters
        ----------
        prev_embeddings : list of np.ndarray
            Average logit/embedding vectors from previous groups, each of
            shape ``(vocab_size,)``.
        logits : np.ndarray
            Shape ``(beam_size, vocab_size)`` or ``(vocab_size,)``.

        Returns
        -------
        np.ndarray
            Penalty of same shape as *logits*.
        """
        is_1d = logits.ndim == 1
        if is_1d:
            logits = logits[np.newaxis, :]

        penalty = np.zeros_like(logits)

        if not prev_embeddings:
            return penalty[0] if is_1d else penalty

        # Average embedding of previous groups
        avg_emb = np.mean(np.stack(prev_embeddings), axis=0)
        avg_norm = np.linalg.norm(avg_emb)
        if avg_norm < _LOG_EPS:
            return penalty[0] if is_1d else penalty

        avg_emb_normed = avg_emb / avg_norm

        for b in range(logits.shape[0]):
            beam_logits = logits[b]
            beam_norm = np.linalg.norm(beam_logits)
            if beam_norm < _LOG_EPS:
                continue
            beam_normed = beam_logits / beam_norm

            # Cosine similarity per token dimension gives a proxy for similarity
            # We use the element-wise product to weight the penalty per token
            sim_scores = avg_emb_normed * beam_normed

            # Positive similarity means the token direction aligns with previous
            # groups — penalise proportionally
            positive_sim = np.maximum(sim_scores, 0.0)
            penalty[b] += self.penalty_strength * positive_sim

        return penalty[0] if is_1d else penalty


# =========================================================================
# DiverseBeamSearch — main algorithm
# =========================================================================


class DiverseBeamSearch(DecodingAlgorithm):
    """Diverse Beam Search decoding algorithm.

    Partitions the beam budget into ``num_beam_groups`` groups.  At each
    step, groups are processed *sequentially*: group 0 runs a standard
    beam-search step, then groups 1 … G−1 run a beam-search step with an
    additional diversity penalty based on tokens selected by all preceding
    groups.

    This class overrides :meth:`generate` entirely because the search state
    is fundamentally different from the per-sequence state used by most
    other algorithms.  The abstract :meth:`_step` is implemented for
    compatibility but delegates to the internal beam logic.
    """

    def __init__(self, config: DiverseBeamConfig) -> None:
        self.beam_config: DiverseBeamConfig = config
        super().__init__(config)
        self.scorer = BeamScorer(length_penalty_alpha=config.length_penalty)
        self.penalizer = DiversityPenalizer(
            penalty_type=config.diversity_type,
            penalty_strength=config.diversity_penalty,
            ngram_order=config.ngram_diversity_order,
        )
        self._beam_state_history: List[BeamState] = []

    # -- properties ---------------------------------------------------------

    @property
    def description(self) -> str:
        return (
            f"Diverse Beam Search ({self.beam_config.num_beams} beams, "
            f"{self.beam_config.num_beam_groups} groups, "
            f"{self.beam_config.diversity_type} penalty "
            f"λ={self.beam_config.diversity_penalty})"
        )

    # -- main entry point ---------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate diverse sequences via group-based beam search.

        Overrides the base class ``generate()`` entirely because Diverse Beam
        Search maintains a :class:`BeamState` rather than the standard
        :class:`DecodingState`.

        Parameters
        ----------
        logit_source : LogitSource
            Callable ``(List[List[int]]) -> np.ndarray`` returning logits of
            shape ``(batch, vocab_size)``.
        prompt_ids : list of int
            Token ids of the prompt.

        Returns
        -------
        list of TokenSequence
            Up to ``num_return_sequences`` generated sequences, sorted by
            length-normalised score (best first).
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        logger.info(
            "Starting Diverse Beam Search: %d beams, %d groups, "
            "penalty_type=%s, λ=%.3f",
            self.beam_config.num_beams,
            self.beam_config.num_beam_groups,
            self.beam_config.diversity_type,
            self.beam_config.diversity_penalty,
        )

        beam_state = self._init_beams(prompt_ids)
        self._beam_state_history = [beam_state.clone()]

        for step_idx in range(self.beam_config.max_new_tokens):
            if beam_state.all_groups_done:
                logger.debug("All groups done at step %d", step_idx)
                break

            beam_state = self._step_beams(beam_state, logit_source)
            beam_state.step = step_idx + 1
            self._beam_state_history.append(beam_state.clone())

            # Early stopping check
            if self.beam_config.early_stopping:
                if len(beam_state.finished_sequences) >= self.beam_config.num_return_sequences:
                    logger.debug(
                        "Early stopping: %d finished sequences at step %d",
                        len(beam_state.finished_sequences),
                        step_idx,
                    )
                    break

        result = self._finalize_beams(beam_state)
        logger.info(
            "Diverse Beam Search completed: %d sequences, %d steps",
            len(result),
            beam_state.step,
        )
        return result

    # -- compatibility _step -----------------------------------------------

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Compatibility bridge for the base class interface.

        This method is required by the ABC but the actual logic runs through
        :meth:`_step_beams`.  If called directly (e.g. via
        ``_generation_loop``), it performs a single diverse beam step using
        a temporary internal :class:`BeamState` stored in ``state.metadata``.
        """
        beam_state: Optional[BeamState] = state.metadata.get("_beam_state")

        if beam_state is None:
            beam_state = self._init_beams(
                state.sequences[0] if state.sequences else []
            )
            state.metadata["_beam_state"] = beam_state

        beam_state = self._step_beams(beam_state, logit_source)
        beam_state.step = state.step + 1
        state.metadata["_beam_state"] = beam_state

        # Sync back to DecodingState
        all_seqs = beam_state.all_sequences_flat()
        state.sequences = all_seqs[: len(state.sequences)]
        for i in range(len(state.sequences)):
            if i < len(all_seqs):
                state.sequences[i] = all_seqs[i]
        state.step = beam_state.step

        if beam_state.all_groups_done:
            for i in range(len(state.is_finished)):
                state.is_finished[i] = True

        return state

    # -- beam initialisation ------------------------------------------------

    def _init_beams(self, prompt_ids: List[int]) -> BeamState:
        """Initialise the beam state with ``num_beam_groups`` groups.

        Each group starts with ``group_size`` identical copies of the prompt.

        Parameters
        ----------
        prompt_ids : list of int
            Prompt token ids.

        Returns
        -------
        BeamState
        """
        config = self.beam_config
        groups: List[BeamGroup] = []

        for g in range(config.num_beam_groups):
            group = BeamGroup(
                beam_ids=[list(prompt_ids) for _ in range(config.group_size)],
                beam_scores=[0.0] * config.group_size,
                beam_indices=list(range(config.group_size)),
                selected_tokens=[],
                is_done=[False] * config.group_size,
            )
            groups.append(group)

        beam_state = BeamState(
            groups=groups,
            finished_sequences=[],
            step=0,
            prompt_ids=list(prompt_ids),
        )

        logger.debug(
            "Initialised %d groups × %d beams, prompt length %d",
            config.num_beam_groups,
            config.group_size,
            len(prompt_ids),
        )
        return beam_state

    # -- core step logic ----------------------------------------------------

    def _step_beams(
        self, beam_state: BeamState, logit_source: LogitSource
    ) -> BeamState:
        """Execute one decoding step across all groups.

        Groups are processed *sequentially*.  For each group:
        1. Get logits for all beams in the group.
        2. Apply diversity penalty from tokens chosen by earlier groups.
        3. Run standard beam search within the group.
        4. Track selected tokens for subsequent groups' penalties.

        Parameters
        ----------
        beam_state : BeamState
            Current search state.
        logit_source : LogitSource
            Model logit provider.

        Returns
        -------
        BeamState
            Updated search state.
        """
        config = self.beam_config

        # Tokens selected at this step, one set per group
        step_selected_tokens: List[Set[int]] = []
        # Sequences from previous groups at this step (for n-gram penalty)
        step_prev_sequences: List[List[List[int]]] = []
        # Average logits from previous groups (for embedding penalty)
        step_prev_embeddings: List[np.ndarray] = []

        for g_idx, group in enumerate(beam_state.groups):
            if group.all_done:
                step_selected_tokens.append(set())
                step_prev_sequences.append(group.beam_ids)
                continue

            # 1. Get logits for all beams in this group
            active_seqs = []
            active_indices = []
            for b_idx in range(group.size):
                if not group.is_done[b_idx]:
                    active_seqs.append(group.beam_ids[b_idx])
                    active_indices.append(b_idx)

            if not active_seqs:
                step_selected_tokens.append(set())
                step_prev_sequences.append(group.beam_ids)
                continue

            # Pad inactive beams with a copy of the first active sequence
            # so that we always query group_size beams
            full_seqs = list(group.beam_ids)
            logits = logit_source(full_seqs)  # (group_size, vocab)

            if logits.ndim == 1:
                logits = logits[np.newaxis, :]

            # Convert to log-probabilities
            log_probs = _stable_log_softmax(logits)

            # 2. Apply diversity penalty from earlier groups
            if g_idx > 0 and config.diversity_penalty > 0:
                penalty = self._compute_diversity_penalty(
                    step_selected_tokens[:g_idx],
                    log_probs,
                    step_prev_sequences[:g_idx],
                    step_prev_embeddings[:g_idx] if step_prev_embeddings else None,
                )
                log_probs = log_probs - penalty

            # Apply constraints (repetition penalty, no-repeat n-gram, etc.)
            for b_idx in range(log_probs.shape[0]):
                if b_idx < group.size and not group.is_done[b_idx]:
                    log_probs[b_idx] = self._apply_beam_constraints(
                        log_probs[b_idx], group.beam_ids[b_idx], beam_state
                    )

            # 3. Standard beam search step within the group
            beam_scores_arr = np.array(group.beam_scores, dtype=np.float64)
            new_beam_ids, new_beam_scores, new_tokens, selected_tokens = (
                self._beam_step(
                    log_probs,
                    beam_scores_arr,
                    config.group_size,
                    group,
                    beam_state,
                    g_idx,
                )
            )

            # 4. Update group state
            group.beam_ids = new_beam_ids
            group.beam_scores = new_beam_scores
            group.selected_tokens.append(selected_tokens)

            # Track for subsequent groups
            step_selected_tokens.append(selected_tokens)
            step_prev_sequences.append(new_beam_ids)

            # Store average logits for embedding penalty
            if log_probs.shape[0] > 0:
                avg_logits = np.mean(log_probs, axis=0)
                step_prev_embeddings.append(avg_logits)

        return beam_state

    def _compute_diversity_penalty(
        self,
        selected_tokens_prev_groups: List[Set[int]],
        current_logits: np.ndarray,
        prev_group_sequences: Optional[List[List[List[int]]]] = None,
        prev_group_embeddings: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute the diversity penalty to apply to the current group.

        Delegates to :class:`DiversityPenalizer`.

        Parameters
        ----------
        selected_tokens_prev_groups : list of set of int
            Tokens selected by each previous group at the current step.
        current_logits : np.ndarray
            Shape ``(beam_size, vocab_size)``.
        prev_group_sequences : list of list of list of int, optional
            Sequences from previous groups.
        prev_group_embeddings : list of np.ndarray, optional
            Average embeddings/logits from previous groups.

        Returns
        -------
        np.ndarray
            Penalty array, shape ``(beam_size, vocab_size)``.
        """
        return self.penalizer.compute_penalty(
            previous_groups_tokens=selected_tokens_prev_groups,
            current_logits=current_logits,
            previous_group_sequences=prev_group_sequences,
            previous_group_embeddings=prev_group_embeddings,
        )

    def _hamming_diversity_penalty(
        self, selected_tokens: List[Set[int]], logits: np.ndarray
    ) -> np.ndarray:
        """Compute Hamming diversity penalty (direct access).

        Parameters
        ----------
        selected_tokens : list of set of int
            Tokens selected by each previous group.
        logits : np.ndarray
            Current logits.

        Returns
        -------
        np.ndarray
            Penalty array.
        """
        return self.penalizer.hamming_penalty(selected_tokens, logits)

    def _ngram_diversity_penalty(
        self,
        prev_group_sequences: List[List[List[int]]],
        logits: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """Compute n-gram diversity penalty (direct access).

        Parameters
        ----------
        prev_group_sequences : list of list of list of int
            Sequences from previous groups.
        logits : np.ndarray
            Current logits.
        order : int
            N-gram order.

        Returns
        -------
        np.ndarray
            Penalty array.
        """
        return self.penalizer.ngram_penalty(prev_group_sequences, logits, order)

    def _embedding_diversity_penalty(
        self,
        prev_group_embeddings: List[np.ndarray],
        logits: np.ndarray,
    ) -> np.ndarray:
        """Compute embedding diversity penalty (direct access).

        Parameters
        ----------
        prev_group_embeddings : list of np.ndarray
            Average embeddings from previous groups.
        logits : np.ndarray
            Current logits.

        Returns
        -------
        np.ndarray
            Penalty array.
        """
        return self.penalizer.embedding_penalty(prev_group_embeddings, logits)

    # -- beam search step ---------------------------------------------------

    def _beam_step(
        self,
        logits: np.ndarray,
        beam_scores: np.ndarray,
        beam_size: int,
        group: BeamGroup,
        beam_state: BeamState,
        group_id: int,
    ) -> Tuple[List[List[int]], List[float], List[int], Set[int]]:
        """Perform one standard beam search step within a group.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(current_beam_count, vocab_size)`` — log-probabilities
            (after diversity penalty).
        beam_scores : np.ndarray
            Shape ``(current_beam_count,)`` — accumulated scores.
        beam_size : int
            Number of beams to keep.
        group : BeamGroup
            The group being processed.
        beam_state : BeamState
            Full search state (for adding finished hypotheses).
        group_id : int
            Index of the current group.

        Returns
        -------
        next_beam_ids : list of list of int
            Updated token sequences for each beam.
        next_beam_scores : list of float
            Updated accumulated scores.
        next_tokens : list of int
            Tokens selected at this step.
        selected_tokens : set of int
            Set of unique tokens selected (for diversity penalty).
        """
        vocab_size = logits.shape[-1]
        config = self.beam_config
        prompt_len = len(beam_state.prompt_ids)

        # Score all candidates
        beam_indices, token_indices, new_scores = self.scorer.score_hypotheses(
            beam_scores, logits, beam_size * 2  # over-generate for filtering
        )

        next_beam_ids: List[List[int]] = []
        next_beam_scores: List[float] = []
        next_tokens: List[int] = []
        selected_tokens: Set[int] = set()

        filled = 0
        for rank in range(len(beam_indices)):
            if filled >= beam_size:
                break

            b_idx = int(beam_indices[rank])
            t_idx = int(token_indices[rank])
            score = float(new_scores[rank])

            if b_idx >= group.size:
                continue

            if group.is_done[b_idx]:
                # Propagate finished beam unchanged
                if filled < beam_size:
                    next_beam_ids.append(list(group.beam_ids[b_idx]))
                    next_beam_scores.append(group.beam_scores[b_idx])
                    next_tokens.append(-1)
                    filled += 1
                continue

            # Build new hypothesis
            new_seq = list(group.beam_ids[b_idx]) + [t_idx]

            # Check if this hypothesis is finished (EOS)
            is_eos = (
                config.eos_token_id is not None and t_idx == config.eos_token_id
            )
            gen_len = len(new_seq) - prompt_len

            if is_eos and gen_len >= config.min_new_tokens:
                # Add to finished set
                lp = self.scorer.length_penalty(gen_len, config.length_penalty)
                norm_score = score / lp if lp > 0 else score
                finished = FinishedBeam(
                    sequence=new_seq,
                    score=score,
                    group_id=group_id,
                    length_normalized_score=norm_score,
                )
                heapq.heappush(beam_state.finished_sequences, finished)

                # If early stopping, mark beam as done
                if config.early_stopping:
                    next_beam_ids.append(new_seq)
                    next_beam_scores.append(score)
                    next_tokens.append(t_idx)
                    selected_tokens.add(t_idx)
                    filled += 1
                    # Mark corresponding beam as done in the next state
                    continue
                else:
                    # Still keep generating from this beam
                    next_beam_ids.append(new_seq)
                    next_beam_scores.append(score)
                    next_tokens.append(t_idx)
                    selected_tokens.add(t_idx)
                    filled += 1
            else:
                next_beam_ids.append(new_seq)
                next_beam_scores.append(score)
                next_tokens.append(t_idx)
                selected_tokens.add(t_idx)
                filled += 1

        # Pad if we didn't fill all beam slots
        while len(next_beam_ids) < beam_size:
            if group.beam_ids:
                next_beam_ids.append(list(group.beam_ids[0]))
                next_beam_scores.append(_NEG_INF)
                next_tokens.append(-1)
            else:
                next_beam_ids.append(list(beam_state.prompt_ids))
                next_beam_scores.append(_NEG_INF)
                next_tokens.append(-1)

        # Update is_done flags
        new_is_done: List[bool] = []
        for b_idx in range(beam_size):
            if b_idx < len(next_tokens):
                tok = next_tokens[b_idx]
                is_eos = (
                    config.eos_token_id is not None
                    and tok == config.eos_token_id
                )
                gen_len = len(next_beam_ids[b_idx]) - prompt_len
                is_max_len = gen_len >= config.max_new_tokens
                new_is_done.append(is_eos or is_max_len or next_beam_scores[b_idx] == _NEG_INF)
            else:
                new_is_done.append(True)
        group.is_done = new_is_done

        return next_beam_ids, next_beam_scores, next_tokens, selected_tokens

    # -- finalisation -------------------------------------------------------

    def _finalize_beams(self, beam_state: BeamState) -> List[TokenSequence]:
        """Extract final sequences from the beam state.

        Combines finished hypotheses with the best still-active beams,
        applies length normalisation, and returns the top
        ``num_return_sequences``.

        Parameters
        ----------
        beam_state : BeamState
            Final search state.

        Returns
        -------
        list of TokenSequence
            Generated token sequences (prompt stripped), sorted by normalised
            score descending.
        """
        config = self.beam_config
        prompt_len = len(beam_state.prompt_ids)

        # Add any remaining active beams as finished hypotheses
        for g_idx, group in enumerate(beam_state.groups):
            for b_idx in range(group.size):
                seq = group.beam_ids[b_idx]
                score = group.beam_scores[b_idx]
                gen_len = max(len(seq) - prompt_len, 1)
                lp = self.scorer.length_penalty(gen_len, config.length_penalty)
                norm_score = score / lp if lp > 0 else score

                finished = FinishedBeam(
                    sequence=seq,
                    score=score,
                    group_id=g_idx,
                    length_normalized_score=norm_score,
                )
                beam_state.finished_sequences.append(finished)

        # De-duplicate by sequence content
        seen: Dict[Tuple[int, ...], FinishedBeam] = {}
        for fb in beam_state.finished_sequences:
            key = tuple(fb.sequence)
            if key not in seen or fb.length_normalized_score > seen[key].length_normalized_score:
                seen[key] = fb

        # Sort by normalised score
        all_finished = sorted(
            seen.values(),
            key=lambda fb: fb.length_normalized_score,
            reverse=True,
        )

        # Take top-k and strip prompt
        result: List[TokenSequence] = []
        for fb in all_finished[: config.num_return_sequences]:
            generated = fb.sequence[prompt_len:]
            if generated:
                result.append(generated)

        # If we have fewer sequences than requested, pad with available beams
        if len(result) < config.num_return_sequences:
            logger.warning(
                "Only %d sequences generated (requested %d)",
                len(result),
                config.num_return_sequences,
            )

        return result

    # -- constraint helpers -------------------------------------------------

    def _apply_beam_constraints(
        self,
        logits: np.ndarray,
        beam_ids: List[int],
        beam_state: BeamState,
    ) -> np.ndarray:
        """Apply repetition penalty and n-gram blocking to a single beam.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)`` — log-probabilities for one beam.
        beam_ids : list of int
            Full token sequence for this beam.
        beam_state : BeamState
            Search state (for prompt length, etc.).

        Returns
        -------
        np.ndarray
            Constrained logits.
        """
        config = self.beam_config
        prompt_len = len(beam_state.prompt_ids)

        # Repetition penalty
        if config.repetition_penalty > 1.0:
            logits = self._apply_repetition_penalty(
                logits, beam_ids, config.repetition_penalty
            )

        # No-repeat n-gram
        if config.no_repeat_ngram_size > 0:
            logits = self._apply_no_repeat_ngram(
                logits, beam_ids, config.no_repeat_ngram_size
            )

        # Min-length enforcement
        gen_len = len(beam_ids) - prompt_len
        logits = self._enforce_min_length(
            logits, gen_len, config.min_new_tokens, config.eos_token_id
        )

        return logits

    # -- length normalisation -----------------------------------------------

    @staticmethod
    def _length_normalize(score: float, length: int, alpha: float = 1.0) -> float:
        """Apply Wu et al. (2016) length normalisation.

        Parameters
        ----------
        score : float
            Raw accumulated log-probability.
        length : int
            Number of generated tokens.
        alpha : float
            Length penalty exponent.

        Returns
        -------
        float
            Length-normalised score.
        """
        if alpha == 0.0:
            return score
        lp = ((5.0 + length) / 6.0) ** alpha
        return score / lp if lp > 0 else score

    # -- hyperparameter introspection ---------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        base = super().hyperparameter_space()
        base.update(DiverseBeamConfig.hyperparameter_space())
        return base

    def validate_config(self) -> List[str]:
        return self.beam_config.validate()

    # -- beam state history access ------------------------------------------

    @property
    def beam_state_history(self) -> List[BeamState]:
        """Return the recorded beam state history (one per step)."""
        return self._beam_state_history


# =========================================================================
# DiverseBeamAnalyzer — post-hoc analysis
# =========================================================================


class DiverseBeamAnalyzer:
    """Analysis utilities for Diverse Beam Search outputs.

    Provides methods to measure inter-group and intra-group diversity,
    analyse penalty sensitivity, and trace beam trajectories through the
    search process.
    """

    # -- group diversity analysis -------------------------------------------

    def analyze_group_diversity(
        self, sequences_by_group: Dict[int, List[List[int]]]
    ) -> Dict[str, Any]:
        """Comprehensive diversity analysis across beam groups.

        Parameters
        ----------
        sequences_by_group : dict mapping group_id -> list of sequences
            Sequences partitioned by group.

        Returns
        -------
        dict with keys:
            - ``"num_groups"`` : int
            - ``"group_sizes"`` : list of int
            - ``"inter_group_diversity"`` : float
            - ``"intra_group_diversity"`` : dict mapping group_id to float
            - ``"overall_diversity"`` : float — combined measure
            - ``"per_group_stats"`` : list of per-group statistics dicts
            - ``"pairwise_group_distances"`` : dict
            - ``"distinct_1"`` : float
            - ``"distinct_2"`` : float
            - ``"self_bleu"`` : float
        """
        all_sequences: List[List[int]] = []
        for seqs in sequences_by_group.values():
            all_sequences.extend(seqs)

        num_groups = len(sequences_by_group)
        group_ids = sorted(sequences_by_group.keys())

        # Per-group statistics
        per_group_stats: List[Dict[str, Any]] = []
        intra_divs: Dict[int, float] = {}
        for gid in group_ids:
            seqs = sequences_by_group[gid]
            intra_div = self.intra_group_diversity(seqs)
            intra_divs[gid] = intra_div

            avg_len = np.mean([len(s) for s in seqs]) if seqs else 0.0
            len_std = np.std([len(s) for s in seqs]) if seqs else 0.0

            unique_tokens = set()
            for s in seqs:
                unique_tokens.update(s)

            stats: Dict[str, Any] = {
                "group_id": gid,
                "num_sequences": len(seqs),
                "avg_length": float(avg_len),
                "length_std": float(len_std),
                "unique_tokens": len(unique_tokens),
                "intra_group_diversity": intra_div,
                "distinct_1": _distinct_ngrams(seqs, 1) if seqs else 0.0,
                "distinct_2": _distinct_ngrams(seqs, 2) if seqs else 0.0,
            }
            per_group_stats.append(stats)

        # Inter-group diversity
        inter_div = self.inter_group_diversity(sequences_by_group)

        # Pairwise group distances
        pairwise: Dict[str, float] = {}
        for i, gid_a in enumerate(group_ids):
            for gid_b in group_ids[i + 1 :]:
                seqs_a = sequences_by_group[gid_a]
                seqs_b = sequences_by_group[gid_b]
                dist = self._group_pair_distance(seqs_a, seqs_b)
                pairwise[f"{gid_a}-{gid_b}"] = dist

        # Overall metrics
        overall_div = (
            0.5 * inter_div + 0.5 * np.mean(list(intra_divs.values()))
            if intra_divs
            else inter_div
        )

        return {
            "num_groups": num_groups,
            "group_sizes": [len(sequences_by_group[g]) for g in group_ids],
            "inter_group_diversity": inter_div,
            "intra_group_diversity": intra_divs,
            "overall_diversity": float(overall_div),
            "per_group_stats": per_group_stats,
            "pairwise_group_distances": pairwise,
            "distinct_1": _distinct_ngrams(all_sequences, 1),
            "distinct_2": _distinct_ngrams(all_sequences, 2),
            "self_bleu": _self_bleu(all_sequences),
        }

    def inter_group_diversity(
        self, groups: Dict[int, List[List[int]]]
    ) -> float:
        """Measure diversity *between* groups.

        Computes the average pairwise distance between groups, where each
        group is represented by its set of n-grams.

        Parameters
        ----------
        groups : dict mapping group_id -> list of sequences

        Returns
        -------
        float
            Inter-group diversity in [0, 1].  Higher = more diverse.
        """
        group_ids = sorted(groups.keys())
        if len(group_ids) < 2:
            return 0.0

        # Represent each group by its set of bigrams
        group_ngram_sets: Dict[int, Set[Tuple[int, ...]]] = {}
        for gid in group_ids:
            all_ngrams: Set[Tuple[int, ...]] = set()
            for seq in groups[gid]:
                all_ngrams.update(_extract_ngrams(seq, 2).keys())
            group_ngram_sets[gid] = all_ngrams

        total_dist = 0.0
        count = 0
        for i, gid_a in enumerate(group_ids):
            for gid_b in group_ids[i + 1 :]:
                sim = _jaccard_similarity(
                    group_ngram_sets[gid_a], group_ngram_sets[gid_b]
                )
                total_dist += 1.0 - sim
                count += 1

        return total_dist / count if count > 0 else 0.0

    def intra_group_diversity(self, group: List[List[int]]) -> float:
        """Measure diversity *within* a single group.

        Computes the average pairwise n-gram distance between all sequences
        in the group.

        Parameters
        ----------
        group : list of list of int
            Sequences in the group.

        Returns
        -------
        float
            Intra-group diversity in [0, 1].  Higher = more diverse.
        """
        if len(group) < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                overlap = _ngram_overlap(group[i], group[j], 2)
                total_dist += 1.0 - overlap
                count += 1

        return total_dist / count if count > 0 else 0.0

    # -- penalty sensitivity analysis ---------------------------------------

    def analyze_penalty_sensitivity(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        penalties: List[float],
        num_beams: int = 12,
        num_groups: int = 3,
        max_tokens: int = 30,
    ) -> Dict[str, Any]:
        """Analyse how different penalty strengths affect diversity.

        Runs Diverse Beam Search with each penalty value and measures the
        resulting diversity.

        Parameters
        ----------
        logit_source : LogitSource
            Model logit provider.
        prompt_ids : list of int
            Prompt tokens.
        penalties : list of float
            Diversity penalty values to test.
        num_beams : int
            Number of beams to use.
        num_groups : int
            Number of beam groups.
        max_tokens : int
            Maximum new tokens per run.

        Returns
        -------
        dict with keys:
            - ``"penalties"`` : list of float
            - ``"diversity_scores"`` : list of float — inter-group diversity
            - ``"quality_scores"`` : list of float — average beam score
            - ``"num_unique_sequences"`` : list of int
            - ``"distinct_2_scores"`` : list of float
            - ``"best_penalty"`` : float — penalty with highest diversity
            - ``"pareto_optimal"`` : list of dict — Pareto-optimal configs
        """
        diversity_scores: List[float] = []
        quality_scores: List[float] = []
        num_unique: List[int] = []
        distinct_2_scores: List[float] = []

        for penalty in penalties:
            config = DiverseBeamConfig(
                num_beams=num_beams,
                num_beam_groups=num_groups,
                diversity_penalty=penalty,
                diversity_type="hamming",
                max_new_tokens=max_tokens,
                num_return_sequences=num_beams,
            )
            searcher = DiverseBeamSearch(config)
            try:
                sequences = searcher.generate(logit_source, prompt_ids)
            except Exception as e:
                logger.warning("Penalty %.3f failed: %s", penalty, e)
                diversity_scores.append(0.0)
                quality_scores.append(_NEG_INF)
                num_unique.append(0)
                distinct_2_scores.append(0.0)
                continue

            # Partition into groups (approximate via round-robin assignment)
            groups_dict: Dict[int, List[List[int]]] = collections.defaultdict(list)
            for idx, seq in enumerate(sequences):
                groups_dict[idx % num_groups].append(seq)

            inter_div = self.inter_group_diversity(groups_dict)
            diversity_scores.append(inter_div)

            # Quality: average distinct-2 as a proxy
            d2 = _distinct_ngrams(sequences, 2) if sequences else 0.0
            distinct_2_scores.append(d2)

            # Unique sequences
            unique_seqs = set(tuple(s) for s in sequences)
            num_unique.append(len(unique_seqs))

            # Quality proxy: negative self-BLEU (lower self-BLEU = better)
            quality = 1.0 - _self_bleu(sequences) if sequences else 0.0
            quality_scores.append(quality)

        # Find best penalty
        best_idx = int(np.argmax(diversity_scores)) if diversity_scores else 0
        best_penalty = penalties[best_idx] if penalties else 0.0

        # Pareto front (diversity vs. quality)
        pareto: List[Dict[str, Any]] = []
        for i in range(len(penalties)):
            is_dominated = False
            for j in range(len(penalties)):
                if i == j:
                    continue
                if (
                    diversity_scores[j] >= diversity_scores[i]
                    and quality_scores[j] >= quality_scores[i]
                    and (
                        diversity_scores[j] > diversity_scores[i]
                        or quality_scores[j] > quality_scores[i]
                    )
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(
                    {
                        "penalty": penalties[i],
                        "diversity": diversity_scores[i],
                        "quality": quality_scores[i],
                    }
                )

        return {
            "penalties": penalties,
            "diversity_scores": diversity_scores,
            "quality_scores": quality_scores,
            "num_unique_sequences": num_unique,
            "distinct_2_scores": distinct_2_scores,
            "best_penalty": best_penalty,
            "pareto_optimal": pareto,
        }

    # -- beam trajectory analysis -------------------------------------------

    def beam_trajectory_analysis(
        self, beam_state_history: List[BeamState]
    ) -> Dict[str, Any]:
        """Analyse the evolution of beams through the search process.

        Parameters
        ----------
        beam_state_history : list of BeamState
            One state snapshot per decoding step.

        Returns
        -------
        dict with keys:
            - ``"num_steps"`` : int
            - ``"active_beams_per_step"`` : list of int
            - ``"finished_per_step"`` : list of int
            - ``"score_trajectories"`` : dict mapping group_id to list of
              (step, [scores]) tuples
            - ``"group_convergence"`` : dict mapping group_id to step at which
              the group's top beam stabilised
            - ``"diversity_trajectory"`` : list of float — inter-group
              diversity at each step
            - ``"token_entropy_per_step"`` : list of float
            - ``"beam_survival_rates"`` : dict mapping group_id to float
        """
        if not beam_state_history:
            return {
                "num_steps": 0,
                "active_beams_per_step": [],
                "finished_per_step": [],
                "score_trajectories": {},
                "group_convergence": {},
                "diversity_trajectory": [],
                "token_entropy_per_step": [],
                "beam_survival_rates": {},
            }

        num_steps = len(beam_state_history)
        num_groups = len(beam_state_history[0].groups) if beam_state_history else 0

        active_per_step: List[int] = []
        finished_per_step: List[int] = []
        score_trajectories: Dict[int, List[Tuple[int, List[float]]]] = {
            g: [] for g in range(num_groups)
        }
        diversity_trajectory: List[float] = []
        token_entropy_per_step: List[float] = []

        for step_idx, state in enumerate(beam_state_history):
            active_per_step.append(state.total_active)
            finished_per_step.append(len(state.finished_sequences))

            # Score trajectories per group
            for g_idx, group in enumerate(state.groups):
                score_trajectories[g_idx].append(
                    (step_idx, list(group.beam_scores))
                )

            # Inter-group diversity at this step
            if num_groups > 1:
                groups_dict: Dict[int, List[List[int]]] = {}
                for g_idx, group in enumerate(state.groups):
                    groups_dict[g_idx] = group.beam_ids
                inter_div = self.inter_group_diversity(groups_dict)
                diversity_trajectory.append(inter_div)
            else:
                diversity_trajectory.append(0.0)

            # Token entropy: distribution of unique tokens at this step
            all_last_tokens: List[int] = []
            for group in state.groups:
                for seq in group.beam_ids:
                    if seq:
                        all_last_tokens.append(seq[-1])

            if all_last_tokens:
                token_counts = collections.Counter(all_last_tokens)
                total = sum(token_counts.values())
                probs = np.array([c / total for c in token_counts.values()])
                entropy = -float(np.sum(probs * np.log2(probs + _LOG_EPS)))
                token_entropy_per_step.append(entropy)
            else:
                token_entropy_per_step.append(0.0)

        # Group convergence: step at which top beam's last token stops changing
        group_convergence: Dict[int, int] = {}
        for g_idx in range(num_groups):
            converged_step = num_steps - 1
            if num_steps >= 3:
                for step in range(num_steps - 2, 0, -1):
                    state_cur = beam_state_history[step]
                    state_prev = beam_state_history[step - 1]
                    if g_idx < len(state_cur.groups) and g_idx < len(state_prev.groups):
                        cur_top = state_cur.groups[g_idx].beam_ids[0] if state_cur.groups[g_idx].beam_ids else []
                        prev_top = state_prev.groups[g_idx].beam_ids[0] if state_prev.groups[g_idx].beam_ids else []
                        if cur_top != prev_top:
                            converged_step = step + 1
                            break
            group_convergence[g_idx] = converged_step

        # Beam survival rate: fraction of initial beams that survive to the end
        beam_survival: Dict[int, float] = {}
        if beam_state_history:
            first_state = beam_state_history[0]
            last_state = beam_state_history[-1]
            for g_idx in range(num_groups):
                if g_idx < len(first_state.groups) and g_idx < len(last_state.groups):
                    initial = first_state.groups[g_idx].size
                    final_active = last_state.groups[g_idx].num_active
                    beam_survival[g_idx] = final_active / initial if initial > 0 else 0.0
                else:
                    beam_survival[g_idx] = 0.0

        return {
            "num_steps": num_steps,
            "active_beams_per_step": active_per_step,
            "finished_per_step": finished_per_step,
            "score_trajectories": score_trajectories,
            "group_convergence": group_convergence,
            "diversity_trajectory": diversity_trajectory,
            "token_entropy_per_step": token_entropy_per_step,
            "beam_survival_rates": beam_survival,
        }

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _group_pair_distance(
        seqs_a: List[List[int]], seqs_b: List[List[int]]
    ) -> float:
        """Average pairwise n-gram distance between two groups.

        Parameters
        ----------
        seqs_a, seqs_b : lists of token sequences

        Returns
        -------
        float
            Average (1 - Jaccard) distance.
        """
        if not seqs_a or not seqs_b:
            return 1.0

        total = 0.0
        count = 0
        for sa in seqs_a:
            for sb in seqs_b:
                overlap = _ngram_overlap(sa, sb, 2)
                total += 1.0 - overlap
                count += 1
        return total / count if count > 0 else 1.0

    @staticmethod
    def sequences_to_groups(
        sequences: List[List[int]],
        finished_beams: Optional[List[FinishedBeam]] = None,
        num_groups: int = 4,
    ) -> Dict[int, List[List[int]]]:
        """Partition sequences into groups.

        If ``finished_beams`` is provided, uses their ``group_id`` attribute.
        Otherwise falls back to round-robin assignment.

        Parameters
        ----------
        sequences : list of list of int
            All generated sequences.
        finished_beams : list of FinishedBeam, optional
            Finished beam objects with group assignments.
        num_groups : int
            Number of groups for round-robin fallback.

        Returns
        -------
        dict mapping group_id -> list of sequences
        """
        groups: Dict[int, List[List[int]]] = collections.defaultdict(list)

        if finished_beams is not None and len(finished_beams) == len(sequences):
            for seq, fb in zip(sequences, finished_beams):
                groups[fb.group_id].append(seq)
        else:
            for idx, seq in enumerate(sequences):
                groups[idx % num_groups].append(seq)

        return dict(groups)

    def compute_diversity_quality_tradeoff(
        self,
        sequences_by_group: Dict[int, List[List[int]]],
        scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Compute diversity-quality trade-off metrics.

        Parameters
        ----------
        sequences_by_group : dict mapping group_id -> list of sequences
        scores : list of float, optional
            Sequence scores (e.g. log-probabilities).

        Returns
        -------
        dict with keys ``"diversity"``, ``"quality"``, ``"tradeoff"``
        """
        all_seqs: List[List[int]] = []
        for seqs in sequences_by_group.values():
            all_seqs.extend(seqs)

        diversity = _pairwise_bleu_diversity(all_seqs)

        if scores:
            quality = float(np.mean(scores))
        else:
            quality = 1.0 - _self_bleu(all_seqs)

        # Harmonic mean of diversity and normalised quality
        if diversity + quality > 0:
            tradeoff = 2.0 * diversity * quality / (diversity + quality)
        else:
            tradeoff = 0.0

        return {
            "diversity": diversity,
            "quality": quality,
            "tradeoff": tradeoff,
        }

    def compare_with_baseline(
        self,
        diverse_sequences: List[List[int]],
        baseline_sequences: List[List[int]],
    ) -> Dict[str, Any]:
        """Compare Diverse Beam Search output against a baseline.

        Parameters
        ----------
        diverse_sequences : list of list of int
            Sequences from Diverse Beam Search.
        baseline_sequences : list of list of int
            Sequences from standard beam search or another baseline.

        Returns
        -------
        dict with comparison metrics
        """
        div_d1 = _distinct_ngrams(diverse_sequences, 1)
        div_d2 = _distinct_ngrams(diverse_sequences, 2)
        div_sbleu = _self_bleu(diverse_sequences)
        div_pairwise = _pairwise_bleu_diversity(diverse_sequences)

        base_d1 = _distinct_ngrams(baseline_sequences, 1)
        base_d2 = _distinct_ngrams(baseline_sequences, 2)
        base_sbleu = _self_bleu(baseline_sequences)
        base_pairwise = _pairwise_bleu_diversity(baseline_sequences)

        return {
            "diverse_beam": {
                "distinct_1": div_d1,
                "distinct_2": div_d2,
                "self_bleu": div_sbleu,
                "pairwise_diversity": div_pairwise,
                "num_unique": len(set(tuple(s) for s in diverse_sequences)),
            },
            "baseline": {
                "distinct_1": base_d1,
                "distinct_2": base_d2,
                "self_bleu": base_sbleu,
                "pairwise_diversity": base_pairwise,
                "num_unique": len(set(tuple(s) for s in baseline_sequences)),
            },
            "improvement": {
                "distinct_1_delta": div_d1 - base_d1,
                "distinct_2_delta": div_d2 - base_d2,
                "self_bleu_delta": div_sbleu - base_sbleu,
                "pairwise_diversity_delta": div_pairwise - base_pairwise,
            },
        }
