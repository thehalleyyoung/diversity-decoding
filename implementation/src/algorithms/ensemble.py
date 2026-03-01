"""
Ensemble decoding module for the Diversity Decoding Arena.

Combines multiple decoding algorithms through various strategies including
logit-level ensembling, probability-level ensembling, mixture of experts,
cascade decoding, stacked decoding, and voting-based approaches.
"""

from __future__ import annotations

import abc
import copy
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
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
    categorical_sample,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AlgorithmFactory = Callable[[DecodingConfig], DecodingAlgorithm]
MemberSpec = Tuple[str, float, DecodingConfig]  # (name, weight, config)


# =========================================================================
# EnsembleConfig
# =========================================================================


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble of decoding algorithms.

    Parameters
    ----------
    members : list of (algorithm_name, weight, config) tuples
        Each member specifies the algorithm to use, its relative weight,
        and the algorithm-specific configuration.
    combination_strategy : str
        How to combine outputs. One of ``"logit"``, ``"probability"``,
        ``"sample_and_merge"``.
    voting_method : str
        For sample_and_merge: ``"majority"``, ``"weighted"``, ``"rank"``.
    temperature : float
        Final sampling temperature applied after combination.
    top_k : int
        Top-k filtering on combined distribution.
    top_p : float
        Nucleus filtering on combined distribution.
    normalize_weights : bool
        Whether to normalize member weights to sum to 1.
    diversity_bonus : float
        Bonus applied to tokens that increase output diversity.
    num_sequences : int
        Number of output sequences to generate.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    min_new_tokens : int
        Minimum number of new tokens before EOS allowed.
    seed : Optional[int]
        Random seed for reproducibility.
    eos_token_id : Optional[int]
        End-of-sequence token id.
    pad_token_id : Optional[int]
        Padding token id.
    merge_top_n : int
        Number of top candidates to keep from each member during merge.
    rerank_after_merge : bool
        Whether to rescore merged candidates under a reference model.
    """

    members: List[MemberSpec] = field(default_factory=list)
    combination_strategy: str = "logit"
    voting_method: str = "majority"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    normalize_weights: bool = True
    diversity_bonus: float = 0.0
    num_sequences: int = 20
    max_new_tokens: int = 100
    min_new_tokens: int = 10
    seed: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    merge_top_n: int = 5
    rerank_after_merge: bool = False

    def get_normalized_weights(self) -> List[float]:
        """Return member weights normalized to sum to 1."""
        weights = [w for _, w, _ in self.members]
        if not weights:
            return []
        total = sum(weights)
        if total <= 0:
            n = len(weights)
            return [1.0 / n] * n
        if self.normalize_weights:
            return [w / total for w in weights]
        return weights

    def validate(self) -> List[str]:
        """Return a list of validation error strings."""
        errors: List[str] = []
        if not self.members:
            errors.append("At least one ensemble member is required.")
        if self.combination_strategy not in ("logit", "probability", "sample_and_merge"):
            errors.append(
                f"Unknown combination_strategy: {self.combination_strategy}"
            )
        if self.voting_method not in ("majority", "weighted", "rank"):
            errors.append(f"Unknown voting_method: {self.voting_method}")
        if self.temperature <= 0:
            errors.append("temperature must be > 0")
        if self.num_sequences < 1:
            errors.append("num_sequences must be >= 1")
        if self.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1")
        for i, (name, weight, cfg) in enumerate(self.members):
            if weight < 0:
                errors.append(f"Member {i} ({name}) has negative weight: {weight}")
        return errors

    def to_decoding_config(self) -> DecodingConfig:
        """Convert to a base DecodingConfig for compatibility."""
        return DecodingConfig(
            algorithm_name="ensemble",
            num_sequences=self.num_sequences,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            seed=self.seed,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            params={
                "combination_strategy": self.combination_strategy,
                "voting_method": self.voting_method,
            },
        )


# =========================================================================
# Algorithm registry helper
# =========================================================================

_ALGORITHM_REGISTRY: Dict[str, AlgorithmFactory] = {}


def register_algorithm(name: str, factory: AlgorithmFactory) -> None:
    """Register an algorithm factory for use in ensembles."""
    _ALGORITHM_REGISTRY[name] = factory


def get_algorithm(name: str, config: DecodingConfig) -> DecodingAlgorithm:
    """Instantiate a registered algorithm by name."""
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Registered: {list(_ALGORITHM_REGISTRY.keys())}"
        )
    return _ALGORITHM_REGISTRY[name](config)


# =========================================================================
# Logit combination utilities
# =========================================================================


def _weighted_logit_average(
    logit_arrays: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Compute weighted average of logit arrays.

    Parameters
    ----------
    logit_arrays : list of np.ndarray
        Each array has shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    weights : list of float
        Weights for each logit array (should sum to 1).

    Returns
    -------
    np.ndarray
        Weighted average logits, same shape as input arrays.
    """
    if len(logit_arrays) == 0:
        raise ValueError("No logit arrays provided")
    if len(logit_arrays) != len(weights):
        raise ValueError("Number of logit arrays must match number of weights")

    result = np.zeros_like(logit_arrays[0], dtype=np.float64)
    for logits, w in zip(logit_arrays, weights):
        result += w * logits.astype(np.float64)
    return result


def _weighted_probability_average(
    logit_arrays: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Compute weighted average in probability space, return as logits.

    Converts each logit array to probabilities, averages, then converts
    back to log-space for sampling.

    Parameters
    ----------
    logit_arrays : list of np.ndarray
        Each array has shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    weights : list of float
        Weights for each logit array.

    Returns
    -------
    np.ndarray
        Combined distribution as log-probabilities.
    """
    if len(logit_arrays) == 0:
        raise ValueError("No logit arrays provided")

    combined_probs = np.zeros_like(logit_arrays[0], dtype=np.float64)
    for logits, w in zip(logit_arrays, weights):
        probs = _stable_softmax(logits.astype(np.float64))
        combined_probs += w * probs

    combined_probs = np.maximum(combined_probs, 1e-30)
    return np.log(combined_probs)


def _geometric_mean_logits(
    logit_arrays: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Compute the weighted geometric mean of probability distributions.

    Equivalent to a weighted average of log-probabilities, which is
    the standard product-of-experts formulation.

    Parameters
    ----------
    logit_arrays : list of np.ndarray
        Each array has shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
    weights : list of float
        Weights for each expert.

    Returns
    -------
    np.ndarray
        Combined log-probabilities (unnormalized).
    """
    result = np.zeros_like(logit_arrays[0], dtype=np.float64)
    for logits, w in zip(logit_arrays, weights):
        log_probs = _log_softmax(logits.astype(np.float64))
        result += w * log_probs
    return result


def _compute_entropy(logits: np.ndarray) -> float:
    """Compute the entropy of the distribution defined by logits.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)``.

    Returns
    -------
    float
        Entropy in nats.
    """
    probs = _stable_softmax(logits.astype(np.float64))
    probs = np.clip(probs, 1e-30, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _compute_confidence(logits: np.ndarray) -> float:
    """Compute the confidence (max probability) of the distribution.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)``.

    Returns
    -------
    float
        Maximum probability in [0, 1].
    """
    probs = _stable_softmax(logits.astype(np.float64))
    return float(np.max(probs))


def _kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """Compute KL(P || Q) from logits.

    Parameters
    ----------
    logits_p, logits_q : np.ndarray
        Shape ``(vocab_size,)``.

    Returns
    -------
    float
        KL divergence in nats.
    """
    p = _stable_softmax(logits_p.astype(np.float64))
    log_p = _log_softmax(logits_p.astype(np.float64))
    log_q = _log_softmax(logits_q.astype(np.float64))
    kl = np.sum(p * (log_p - log_q))
    return float(max(0.0, kl))


def _jensen_shannon_divergence(
    logit_arrays: List[np.ndarray],
    weights: List[float],
) -> float:
    """Compute the Jensen-Shannon divergence among multiple distributions.

    Parameters
    ----------
    logit_arrays : list of np.ndarray
        Each of shape ``(vocab_size,)``.
    weights : list of float
        Weights for each distribution (should sum to 1).

    Returns
    -------
    float
        JS divergence in nats.
    """
    prob_arrays = [_stable_softmax(l.astype(np.float64)) for l in logit_arrays]
    mixture = np.zeros_like(prob_arrays[0])
    for p, w in zip(prob_arrays, weights):
        mixture += w * p

    mixture = np.clip(mixture, 1e-30, 1.0)
    log_mixture = np.log(mixture)

    jsd = 0.0
    for p, w in zip(prob_arrays, weights):
        p_clipped = np.clip(p, 1e-30, 1.0)
        jsd += w * float(np.sum(p_clipped * (np.log(p_clipped) - log_mixture)))

    return float(max(0.0, jsd))


# =========================================================================
# EnsembleDecoding — main ensemble class
# =========================================================================


class EnsembleDecoding(DecodingAlgorithm):
    """Ensemble that combines multiple decoding algorithms.

    Supports three combination strategies:

    - ``"logit"``: Weighted average of logits from each member algorithm.
    - ``"probability"``: Weighted average in probability space.
    - ``"sample_and_merge"``: Run each algorithm independently and merge.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration (num_sequences, max_new_tokens, etc.).
    ensemble_config : EnsembleConfig
        Ensemble-specific configuration (members, weights, strategy).
    algorithm_instances : list of DecodingAlgorithm, optional
        Pre-instantiated algorithm objects. If not provided, algorithms
        are looked up from the registry.
    """

    def __init__(
        self,
        config: DecodingConfig,
        ensemble_config: EnsembleConfig,
        algorithm_instances: Optional[List[DecodingAlgorithm]] = None,
    ) -> None:
        super().__init__(config)
        self.ensemble_config = ensemble_config
        self._weights = ensemble_config.get_normalized_weights()

        if algorithm_instances is not None:
            self._members = list(algorithm_instances)
        else:
            self._members = []
            for name, _weight, member_cfg in ensemble_config.members:
                algo = get_algorithm(name, member_cfg)
                self._members.append(algo)

        self._member_logit_cache: Dict[int, np.ndarray] = {}
        self._step_agreement_log: List[float] = []
        self._step_entropy_log: List[float] = []

    @property
    def description(self) -> str:
        member_names = [m.name for m in self._members]
        return (
            f"Ensemble of {len(self._members)} algorithms "
            f"({', '.join(member_names)}) using "
            f"{self.ensemble_config.combination_strategy} combination"
        )

    # -- Public API ---------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate sequences using the configured ensemble strategy."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        strategy = self.ensemble_config.combination_strategy

        if strategy == "sample_and_merge":
            return self._sample_and_merge(logit_source, prompt_ids)
        else:
            state = self._init_state(prompt_ids)
            state = self._generation_loop(state, logit_source)
            state.metadata["step_agreement"] = list(self._step_agreement_log)
            state.metadata["step_entropy"] = list(self._step_entropy_log)
            return self._finalize(state)

    # -- Step implementation ------------------------------------------------

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        """Execute one decoding step using the ensemble strategy."""
        strategy = self.ensemble_config.combination_strategy
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        base_logits = logit_source(active_seqs)  # (batch, vocab)

        for batch_idx, seq_idx in enumerate(active):
            member_logits_list: List[np.ndarray] = []

            for member in self._members:
                m_logits = self._get_member_logits(
                    member, logit_source, [active_seqs[batch_idx]]
                )
                member_logits_list.append(m_logits[0])

            if strategy == "logit":
                combined = _weighted_logit_average(
                    member_logits_list, self._weights
                )
            elif strategy == "probability":
                combined = _weighted_probability_average(
                    member_logits_list, self._weights
                )
            else:
                combined = _weighted_logit_average(
                    member_logits_list, self._weights
                )

            if self.ensemble_config.diversity_bonus > 0:
                combined = self._apply_diversity_bonus(
                    combined, state.sequences[seq_idx]
                )

            agreement = self._compute_step_agreement(member_logits_list)
            self._step_agreement_log.append(agreement)

            entropy = _compute_entropy(combined)
            self._step_entropy_log.append(entropy)

            combined = self._apply_constraints(combined, state)

            token = sample_token(
                combined,
                temperature=self.ensemble_config.temperature,
                top_k=self.ensemble_config.top_k,
                top_p=self.ensemble_config.top_p,
            )

            state.update_sequence(seq_idx, token)

            log_probs = _log_softmax(combined)
            state.scores[seq_idx] += float(log_probs[token])

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        return state

    # -- Internal helpers ---------------------------------------------------

    def _get_member_logits(
        self,
        member: DecodingAlgorithm,
        logit_source: LogitSource,
        input_seqs: List[List[int]],
    ) -> np.ndarray:
        """Get logits from a member, applying its specific transformations.

        Each member algorithm may apply its own temperature, top-k, etc.
        We approximate this by applying the member's temperature to the
        base logits.
        """
        raw_logits = logit_source(input_seqs)
        member_temp = member.config.temperature
        if member_temp != 1.0 and member_temp > 0:
            raw_logits = raw_logits / member_temp

        if member.config.repetition_penalty > 1.0:
            for b in range(raw_logits.shape[0]):
                raw_logits[b] = DecodingAlgorithm._apply_repetition_penalty(
                    raw_logits[b],
                    input_seqs[b],
                    member.config.repetition_penalty,
                )
        return raw_logits

    def _apply_diversity_bonus(
        self, logits: np.ndarray, sequence: List[int]
    ) -> np.ndarray:
        """Apply a bonus to tokens that haven't appeared recently."""
        logits = logits.copy()
        bonus = self.ensemble_config.diversity_bonus
        if len(sequence) == 0:
            return logits

        recent_tokens = set(sequence[-50:])
        vocab_size = logits.shape[-1]
        for t in range(vocab_size):
            if t not in recent_tokens:
                logits[t] += bonus
        return logits

    def _compute_step_agreement(
        self, member_logits: List[np.ndarray]
    ) -> float:
        """Compute agreement rate: fraction of members agreeing on argmax."""
        if len(member_logits) <= 1:
            return 1.0
        argmaxes = [int(np.argmax(l)) for l in member_logits]
        counter = Counter(argmaxes)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(member_logits)

    def _sample_and_merge(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Run each member independently and merge results.

        Each member generates its own sequences. All sequences are collected,
        optionally rescored, and the top candidates are returned.
        """
        all_sequences: List[Tuple[float, TokenSequence, str]] = []

        for member_idx, member in enumerate(self._members):
            member_name = member.name
            weight = self._weights[member_idx]

            try:
                seqs = member.generate(logit_source, prompt_ids)
            except Exception as exc:
                logger.warning(
                    "Member %s failed during generation: %s", member_name, exc
                )
                continue

            for seq in seqs[: self.ensemble_config.merge_top_n]:
                score = self._score_sequence(
                    seq, logit_source, len(prompt_ids)
                )
                weighted_score = score * weight
                all_sequences.append((weighted_score, seq, member_name))

        if not all_sequences:
            return [list(prompt_ids)]

        if self.ensemble_config.rerank_after_merge:
            all_sequences = self._rerank_sequences(
                all_sequences, logit_source, prompt_ids
            )

        all_sequences.sort(key=lambda x: x[0], reverse=True)

        num_out = self.ensemble_config.num_sequences
        selected = all_sequences[:num_out]
        return [seq for _, seq, _ in selected]

    def _score_sequence(
        self,
        sequence: List[int],
        logit_source: LogitSource,
        prompt_length: int,
    ) -> float:
        """Score a single completed sequence under the model."""
        total_lp = 0.0
        for t in range(max(prompt_length, 1), len(sequence)):
            prefix = [sequence[:t]]
            logits = logit_source(prefix)
            log_p = _log_softmax(logits[0])
            token_id = sequence[t]
            if 0 <= token_id < log_p.shape[0]:
                total_lp += float(log_p[token_id])
        length = max(len(sequence) - prompt_length, 1)
        return total_lp / length

    def _rerank_sequences(
        self,
        candidates: List[Tuple[float, TokenSequence, str]],
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[Tuple[float, TokenSequence, str]]:
        """Rescore all candidate sequences under each ensemble member."""
        reranked: List[Tuple[float, TokenSequence, str]] = []
        prompt_len = len(prompt_ids)

        for _orig_score, seq, source_name in candidates:
            full_seq = list(prompt_ids) + list(seq)
            total_score = 0.0

            for member_idx, member in enumerate(self._members):
                w = self._weights[member_idx]
                member_score = 0.0
                for t in range(max(prompt_len, 1), len(full_seq)):
                    prefix = [full_seq[:t]]
                    logits = logit_source(prefix)
                    if member.config.temperature > 0:
                        logits = logits / member.config.temperature
                    log_p = _log_softmax(logits[0])
                    token_id = full_seq[t]
                    if 0 <= token_id < log_p.shape[0]:
                        member_score += float(log_p[token_id])

                length = max(len(full_seq) - prompt_len, 1)
                total_score += w * (member_score / length)

            reranked.append((total_score, seq, source_name))

        return reranked

    def validate_config(self) -> List[str]:
        """Validate both the base and ensemble configs."""
        errors = self.config.validate()
        errors.extend(self.ensemble_config.validate())
        return errors


# =========================================================================
# Router base and implementations for Mixture of Experts
# =========================================================================


class Router(abc.ABC):
    """Abstract router that decides which expert to use at each step."""

    @abc.abstractmethod
    def route(
        self,
        logits: np.ndarray,
        sequence: List[int],
        step: int,
        num_experts: int,
    ) -> List[float]:
        """Return a weight vector over experts for the current step.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits from the base model, shape ``(vocab_size,)``.
        sequence : list of int
            Tokens generated so far.
        step : int
            Current generation step.
        num_experts : int
            Total number of expert algorithms.

        Returns
        -------
        list of float
            Weight for each expert, should sum to 1.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state between generations."""
        pass


class EntropyBasedRouter(Router):
    """Route based on entropy of the current token distribution.

    When entropy is high (model is uncertain), prefer exploration-oriented
    algorithms. When entropy is low (model is confident), prefer
    exploitation-oriented algorithms.

    Parameters
    ----------
    entropy_thresholds : list of float
        Threshold boundaries dividing experts into entropy regimes.
        Length should be ``num_experts - 1``. Below the first threshold,
        expert 0 gets highest weight; above the last, the final expert does.
    sharpness : float
        Controls how sharply the router transitions between experts.
        Higher values create more abrupt switches.
    base_weight : float
        Minimum weight given to each expert (prevents total exclusion).
    """

    def __init__(
        self,
        entropy_thresholds: Optional[List[float]] = None,
        sharpness: float = 5.0,
        base_weight: float = 0.05,
    ) -> None:
        self.entropy_thresholds = entropy_thresholds or [2.0, 4.0]
        self.sharpness = sharpness
        self.base_weight = base_weight
        self._entropy_history: List[float] = []

    def route(
        self,
        logits: np.ndarray,
        sequence: List[int],
        step: int,
        num_experts: int,
    ) -> List[float]:
        entropy = _compute_entropy(logits)
        self._entropy_history.append(entropy)

        weights = [self.base_weight] * num_experts

        thresholds = self.entropy_thresholds
        if len(thresholds) < num_experts - 1:
            max_thresh = thresholds[-1] if thresholds else 5.0
            extra = num_experts - 1 - len(thresholds)
            step_size = max_thresh / (num_experts - 1) if num_experts > 1 else 1.0
            for i in range(extra):
                thresholds.append(max_thresh + step_size * (i + 1))

        for i in range(num_experts):
            if i == 0:
                upper = thresholds[0] if thresholds else float("inf")
                activation = self._sigmoid(
                    self.sharpness * (upper - entropy)
                )
            elif i == num_experts - 1:
                lower = thresholds[min(i - 1, len(thresholds) - 1)]
                activation = self._sigmoid(
                    self.sharpness * (entropy - lower)
                )
            else:
                lower = thresholds[min(i - 1, len(thresholds) - 1)]
                upper = thresholds[min(i, len(thresholds) - 1)]
                act_low = self._sigmoid(self.sharpness * (entropy - lower))
                act_high = self._sigmoid(self.sharpness * (upper - entropy))
                activation = act_low * act_high
            weights[i] += activation

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        return weights

    def reset(self) -> None:
        self._entropy_history = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))


class ConfidenceBasedRouter(Router):
    """Route based on the confidence (max probability) of the distribution.

    High confidence → use greedy/deterministic experts.
    Low confidence → use stochastic/exploratory experts.

    Parameters
    ----------
    confidence_threshold : float
        Boundary between low and high confidence regimes.
    smoothing : float
        Controls transition smoothness (higher = sharper).
    expert_order : str
        ``"confidence_first"`` means expert 0 handles high-confidence tokens.
        ``"exploration_first"`` means expert 0 handles low-confidence tokens.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        smoothing: float = 10.0,
        expert_order: str = "confidence_first",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.smoothing = smoothing
        self.expert_order = expert_order
        self._confidence_history: List[float] = []

    def route(
        self,
        logits: np.ndarray,
        sequence: List[int],
        step: int,
        num_experts: int,
    ) -> List[float]:
        confidence = _compute_confidence(logits)
        self._confidence_history.append(confidence)

        weights = np.zeros(num_experts, dtype=np.float64)

        raw_high = self._sigmoid(
            self.smoothing * (confidence - self.confidence_threshold)
        )
        raw_low = 1.0 - raw_high

        if num_experts == 1:
            return [1.0]

        if num_experts == 2:
            if self.expert_order == "confidence_first":
                weights[0] = raw_high
                weights[1] = raw_low
            else:
                weights[0] = raw_low
                weights[1] = raw_high
        else:
            n_high = num_experts // 2
            n_low = num_experts - n_high
            for i in range(n_high):
                if self.expert_order == "confidence_first":
                    weights[i] = raw_high / n_high
                else:
                    weights[i] = raw_low / n_low
            for i in range(n_high, num_experts):
                if self.expert_order == "confidence_first":
                    weights[i] = raw_low / n_low
                else:
                    weights[i] = raw_high / n_high

        total = float(np.sum(weights))
        if total > 0:
            weights /= total
        else:
            weights[:] = 1.0 / num_experts

        return weights.tolist()

    def reset(self) -> None:
        self._confidence_history = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))


class PositionBasedRouter(Router):
    """Route based on the position (step number) in the sequence.

    Different stages of generation can be handled by different experts.
    Early tokens might benefit from one strategy, while later tokens
    benefit from another.

    Parameters
    ----------
    position_boundaries : list of int
        Step numbers that define region boundaries.
        E.g. ``[10, 50]`` creates three regions: [0,10), [10,50), [50,∞).
    region_expert_weights : list of list of float
        For each region, the weight vector over experts.
        Length must equal ``len(position_boundaries) + 1``.
    transition_width : int
        Number of steps over which to smoothly transition between regions.
    """

    def __init__(
        self,
        position_boundaries: Optional[List[int]] = None,
        region_expert_weights: Optional[List[List[float]]] = None,
        transition_width: int = 5,
    ) -> None:
        self.position_boundaries = position_boundaries or [20, 60]
        self.region_expert_weights = region_expert_weights
        self.transition_width = max(1, transition_width)

    def route(
        self,
        logits: np.ndarray,
        sequence: List[int],
        step: int,
        num_experts: int,
    ) -> List[float]:
        if self.region_expert_weights is not None:
            return self._route_with_explicit_weights(step, num_experts)
        return self._route_with_linear_interpolation(step, num_experts)

    def _route_with_explicit_weights(
        self, step: int, num_experts: int
    ) -> List[float]:
        """Use explicitly configured weights per region with smooth transitions."""
        boundaries = self.position_boundaries
        region_weights = self.region_expert_weights
        assert region_weights is not None

        num_regions = len(boundaries) + 1
        while len(region_weights) < num_regions:
            uniform = [1.0 / num_experts] * num_experts
            region_weights.append(uniform)

        for rw in region_weights:
            while len(rw) < num_experts:
                rw.append(0.0)

        region_idx = 0
        for b in boundaries:
            if step >= b:
                region_idx += 1
            else:
                break

        blend_factor = 0.0
        if region_idx > 0:
            boundary = boundaries[region_idx - 1]
            dist_from_boundary = step - boundary
            if dist_from_boundary < self.transition_width:
                blend_factor = dist_from_boundary / self.transition_width
            else:
                blend_factor = 1.0
        else:
            if boundaries and step < boundaries[0]:
                dist_to_boundary = boundaries[0] - step
                if dist_to_boundary < self.transition_width:
                    blend_factor = 1.0 - dist_to_boundary / self.transition_width
                else:
                    blend_factor = 0.0

        current_weights = list(region_weights[region_idx])

        if region_idx > 0 and blend_factor < 1.0:
            prev_weights = region_weights[region_idx - 1]
            current_weights = [
                (1 - blend_factor) * pw + blend_factor * cw
                for pw, cw in zip(prev_weights, current_weights)
            ]

        total = sum(current_weights)
        if total > 0:
            current_weights = [w / total for w in current_weights]
        return current_weights

    def _route_with_linear_interpolation(
        self, step: int, num_experts: int
    ) -> List[float]:
        """Linearly shift weight from early experts to late experts."""
        if num_experts == 1:
            return [1.0]

        boundaries = self.position_boundaries
        max_step = boundaries[-1] if boundaries else 100
        progress = min(1.0, step / max(1, max_step))

        weights = [0.0] * num_experts
        for i in range(num_experts):
            center = i / max(1, num_experts - 1)
            distance = abs(progress - center)
            width = 1.0 / max(1, num_experts - 1) if num_experts > 1 else 1.0
            activation = max(0.0, 1.0 - distance / width)
            weights[i] = activation

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / num_experts] * num_experts
        return weights


class AdaptiveRouter(Router):
    """Router that adapts based on running statistics of entropy and confidence.

    Maintains an exponential moving average of entropy and routes based
    on how the current entropy compares to recent history.

    Parameters
    ----------
    ema_alpha : float
        Exponential moving average decay factor.
    adaptation_rate : float
        How quickly the router adapts to changing conditions.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        adaptation_rate: float = 0.5,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.adaptation_rate = adaptation_rate
        self._ema_entropy: Optional[float] = None
        self._ema_confidence: Optional[float] = None
        self._step_count = 0

    def route(
        self,
        logits: np.ndarray,
        sequence: List[int],
        step: int,
        num_experts: int,
    ) -> List[float]:
        entropy = _compute_entropy(logits)
        confidence = _compute_confidence(logits)

        if self._ema_entropy is None:
            self._ema_entropy = entropy
            self._ema_confidence = confidence
        else:
            self._ema_entropy = (
                self.ema_alpha * entropy + (1 - self.ema_alpha) * self._ema_entropy
            )
            self._ema_confidence = (
                self.ema_alpha * confidence
                + (1 - self.ema_alpha) * self._ema_confidence
            )

        entropy_deviation = entropy - self._ema_entropy
        conf_deviation = confidence - (self._ema_confidence or 0.5)

        weights = np.ones(num_experts, dtype=np.float64)

        for i in range(num_experts):
            expert_position = i / max(1, num_experts - 1) if num_experts > 1 else 0.5

            if entropy_deviation > 0:
                if expert_position > 0.5:
                    weights[i] += self.adaptation_rate * entropy_deviation
                else:
                    weights[i] -= self.adaptation_rate * entropy_deviation * 0.5
            else:
                if expert_position < 0.5:
                    weights[i] += self.adaptation_rate * abs(entropy_deviation)
                else:
                    weights[i] -= self.adaptation_rate * abs(entropy_deviation) * 0.5

            weights[i] = max(0.01, weights[i])

        self._step_count += 1
        total = float(np.sum(weights))
        if total > 0:
            weights /= total
        return weights.tolist()

    def reset(self) -> None:
        self._ema_entropy = None
        self._ema_confidence = None
        self._step_count = 0


# =========================================================================
# MixtureOfExperts
# =========================================================================


class MixtureOfExperts(DecodingAlgorithm):
    """Route different tokens/positions to different decoding algorithms.

    Uses a Router to determine expert weights at each step, then combines
    the logits from each expert according to those weights.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    experts : list of DecodingAlgorithm
        The expert algorithms.
    router : Router
        Routing strategy that assigns weights to experts.
    combination_mode : str
        How to combine expert outputs: ``"logit"``, ``"probability"``,
        ``"geometric"``.
    gating_temperature : float
        Temperature applied to router weights before normalization.
    load_balancing_loss_weight : float
        Weight of the auxiliary load-balancing loss (penalizes routing
        imbalance).
    """

    def __init__(
        self,
        config: DecodingConfig,
        experts: List[DecodingAlgorithm],
        router: Router,
        combination_mode: str = "logit",
        gating_temperature: float = 1.0,
        load_balancing_loss_weight: float = 0.01,
    ) -> None:
        super().__init__(config)
        self.experts = experts
        self.router = router
        self.combination_mode = combination_mode
        self.gating_temperature = gating_temperature
        self.load_balancing_loss_weight = load_balancing_loss_weight

        self._expert_usage_counts: List[float] = [0.0] * len(experts)
        self._total_steps = 0
        self._routing_history: List[List[float]] = []

    @property
    def description(self) -> str:
        return (
            f"MixtureOfExperts with {len(self.experts)} experts, "
            f"router={self.router.__class__.__name__}"
        )

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        base_logits = logit_source(active_seqs)

        for batch_idx, seq_idx in enumerate(active):
            token_logits = base_logits[batch_idx]
            sequence = state.sequences[seq_idx]

            expert_weights = self.router.route(
                token_logits, sequence, state.step, len(self.experts)
            )

            if self.gating_temperature != 1.0 and self.gating_temperature > 0:
                log_weights = [
                    math.log(max(w, 1e-10)) / self.gating_temperature
                    for w in expert_weights
                ]
                max_lw = max(log_weights)
                exp_weights = [math.exp(lw - max_lw) for lw in log_weights]
                total = sum(exp_weights)
                expert_weights = [ew / total for ew in exp_weights]

            self._routing_history.append(list(expert_weights))

            expert_logits_list: List[np.ndarray] = []
            for expert in self.experts:
                e_logits = logit_source([sequence])
                if expert.config.temperature > 0 and expert.config.temperature != 1.0:
                    e_logits = e_logits / expert.config.temperature
                expert_logits_list.append(e_logits[0])

            if self.combination_mode == "probability":
                combined = _weighted_probability_average(
                    expert_logits_list, expert_weights
                )
            elif self.combination_mode == "geometric":
                combined = _geometric_mean_logits(
                    expert_logits_list, expert_weights
                )
            else:
                combined = _weighted_logit_average(
                    expert_logits_list, expert_weights
                )

            for i, w in enumerate(expert_weights):
                self._expert_usage_counts[i] += w
            self._total_steps += 1

            combined = self._apply_constraints(combined, state)
            token = sample_token(
                combined,
                temperature=self.config.temperature,
                top_k=int(self.config.params.get("top_k", 0)),
                top_p=float(self.config.params.get("top_p", 1.0)),
            )

            state.update_sequence(seq_idx, token)
            log_probs = _log_softmax(combined)
            state.scores[seq_idx] += float(log_probs[token])

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        return state

    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Return statistics on expert usage balance."""
        if self._total_steps == 0:
            return {"total_steps": 0, "expert_fractions": [], "balance_score": 0.0}

        fractions = [c / self._total_steps for c in self._expert_usage_counts]
        n = len(self.experts)
        ideal = 1.0 / n if n > 0 else 0.0
        imbalance = sum(abs(f - ideal) for f in fractions) / n if n > 0 else 0.0
        balance_score = 1.0 - imbalance

        return {
            "total_steps": self._total_steps,
            "expert_fractions": fractions,
            "balance_score": balance_score,
            "routing_history_length": len(self._routing_history),
        }

    def get_load_balancing_loss(self) -> float:
        """Compute the auxiliary load-balancing loss.

        Encourages uniform expert usage by penalizing deviation from
        the ideal uniform distribution. Based on the Switch Transformer
        load-balancing formulation.
        """
        if self._total_steps == 0 or not self._routing_history:
            return 0.0

        n = len(self.experts)
        fractions = [c / self._total_steps for c in self._expert_usage_counts]

        routing_arr = np.array(self._routing_history, dtype=np.float64)
        avg_routing = np.mean(routing_arr, axis=0)

        loss = float(n * np.sum(np.array(fractions) * avg_routing))
        return loss * self.load_balancing_loss_weight

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.experts:
            errors.append("MixtureOfExperts requires at least one expert.")
        return errors


# =========================================================================
# CascadeDecoding
# =========================================================================


@dataclass
class CascadeStage:
    """One stage of a cascade decoding pipeline.

    Parameters
    ----------
    algorithm : DecodingAlgorithm
        The algorithm used in this stage.
    name : str
        Human-readable name for the stage.
    num_candidates : int
        Number of candidates this stage should produce or keep.
    filter_ratio : float
        Fraction of candidates to pass to the next stage (0, 1].
    score_weight : float
        How much this stage's scores contribute to the final ranking.
    timeout_seconds : float
        Maximum time allowed for this stage.
    """

    algorithm: DecodingAlgorithm
    name: str = ""
    num_candidates: int = 50
    filter_ratio: float = 0.5
    score_weight: float = 1.0
    timeout_seconds: float = 60.0


@dataclass
class CascadeConfig:
    """Configuration for cascade decoding.

    Parameters
    ----------
    stages : list of CascadeStage
        Ordered list of cascade stages.
    final_num_sequences : int
        Number of sequences to return from the final stage.
    rescore_between_stages : bool
        Whether to rescore candidates between stages.
    dedup_between_stages : bool
        Whether to deduplicate candidates between stages.
    length_penalty : float
        Length normalization penalty for scoring.
    diversity_penalty : float
        Penalty for too-similar candidates between stages.
    """

    stages: List[CascadeStage] = field(default_factory=list)
    final_num_sequences: int = 20
    rescore_between_stages: bool = True
    dedup_between_stages: bool = True
    length_penalty: float = 0.0
    diversity_penalty: float = 0.0


class CascadeDecoding(DecodingAlgorithm):
    """Sequential multi-stage decoding pipeline.

    Stage 1 generates many candidates quickly (e.g., using a fast
    stochastic algorithm). Stage 2 refines/reranks using a more
    expensive algorithm. Additional stages can be chained.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    cascade_config : CascadeConfig
        Cascade-specific configuration with stage definitions.
    """

    def __init__(
        self,
        config: DecodingConfig,
        cascade_config: CascadeConfig,
    ) -> None:
        super().__init__(config)
        self.cascade_config = cascade_config
        self._stage_results: List[Dict[str, Any]] = []

    @property
    def description(self) -> str:
        stage_names = [s.name or s.algorithm.name for s in self.cascade_config.stages]
        return f"CascadeDecoding with {len(stage_names)} stages: {' → '.join(stage_names)}"

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Run the full cascade pipeline."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self._stage_results = []
        candidates: List[Tuple[float, TokenSequence]] = []

        for stage_idx, stage in enumerate(self.cascade_config.stages):
            t0 = time.monotonic()
            logger.info(
                "Cascade stage %d (%s): processing %d candidates",
                stage_idx,
                stage.name or stage.algorithm.name,
                len(candidates) if candidates else "initial",
            )

            if stage_idx == 0:
                raw_seqs = self._run_initial_stage(
                    stage, logit_source, prompt_ids
                )
                candidates = [
                    (0.0, seq) for seq in raw_seqs[: stage.num_candidates]
                ]
            else:
                candidates = self._run_refinement_stage(
                    stage, candidates, logit_source, prompt_ids
                )

            if self.cascade_config.rescore_between_stages:
                candidates = self._rescore_candidates(
                    candidates, logit_source, prompt_ids
                )

            if self.cascade_config.dedup_between_stages:
                candidates = self._dedup_candidates(candidates)

            num_keep = max(
                1, int(len(candidates) * stage.filter_ratio)
            )
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:num_keep]

            elapsed = time.monotonic() - t0
            self._stage_results.append({
                "stage_idx": stage_idx,
                "stage_name": stage.name or stage.algorithm.name,
                "num_input": len(candidates),
                "num_output": len(candidates),
                "elapsed_seconds": elapsed,
            })

        candidates.sort(key=lambda x: x[0], reverse=True)
        final_n = self.cascade_config.final_num_sequences
        return [seq for _, seq in candidates[:final_n]]

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        """Not used directly; cascade uses generate() override."""
        return state

    def _run_initial_stage(
        self,
        stage: CascadeStage,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Run the first stage to generate initial candidates."""
        orig_num = stage.algorithm.config.num_sequences
        stage.algorithm.config.num_sequences = stage.num_candidates
        try:
            seqs = stage.algorithm.generate(logit_source, prompt_ids)
        finally:
            stage.algorithm.config.num_sequences = orig_num
        return seqs

    def _run_refinement_stage(
        self,
        stage: CascadeStage,
        candidates: List[Tuple[float, TokenSequence]],
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[Tuple[float, TokenSequence]]:
        """Refine/rerank candidates using the stage's algorithm."""
        refined: List[Tuple[float, TokenSequence]] = []
        prompt_len = len(prompt_ids)

        for orig_score, seq in candidates:
            full_seq = list(prompt_ids) + list(seq)

            new_score = 0.0
            for t in range(max(prompt_len, 1), len(full_seq)):
                prefix = [full_seq[:t]]
                logits = logit_source(prefix)
                if stage.algorithm.config.temperature > 0:
                    logits = logits / stage.algorithm.config.temperature
                log_p = _log_softmax(logits[0])
                token_id = full_seq[t]
                if 0 <= token_id < log_p.shape[0]:
                    new_score += float(log_p[token_id])

            gen_length = max(len(seq), 1)
            if self.cascade_config.length_penalty != 0:
                length_factor = (
                    (5.0 + gen_length) / 6.0
                ) ** self.cascade_config.length_penalty
                new_score /= length_factor
            else:
                new_score /= gen_length

            combined_score = (
                orig_score * (1 - stage.score_weight)
                + new_score * stage.score_weight
            )
            refined.append((combined_score, seq))

        if self.cascade_config.diversity_penalty > 0:
            refined = self._apply_diversity_penalty(refined)

        return refined

    def _rescore_candidates(
        self,
        candidates: List[Tuple[float, TokenSequence]],
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[Tuple[float, TokenSequence]]:
        """Rescore all candidates under the base model."""
        prompt_len = len(prompt_ids)
        rescored: List[Tuple[float, TokenSequence]] = []

        for orig_score, seq in candidates:
            full_seq = list(prompt_ids) + list(seq)
            log_prob = 0.0

            for t in range(max(prompt_len, 1), len(full_seq)):
                prefix = [full_seq[:t]]
                logits = logit_source(prefix)
                lp = _log_softmax(logits[0])
                token_id = full_seq[t]
                if 0 <= token_id < lp.shape[0]:
                    log_prob += float(lp[token_id])

            gen_length = max(len(seq), 1)
            avg_lp = log_prob / gen_length
            combined = 0.5 * orig_score + 0.5 * avg_lp
            rescored.append((combined, seq))

        return rescored

    def _dedup_candidates(
        self,
        candidates: List[Tuple[float, TokenSequence]],
    ) -> List[Tuple[float, TokenSequence]]:
        """Remove duplicate sequences, keeping the highest-scored version."""
        seen: Dict[Tuple[int, ...], float] = {}
        best: Dict[Tuple[int, ...], Tuple[float, TokenSequence]] = {}

        for score, seq in candidates:
            key = tuple(seq)
            if key not in seen or score > seen[key]:
                seen[key] = score
                best[key] = (score, seq)

        return list(best.values())

    def _apply_diversity_penalty(
        self,
        candidates: List[Tuple[float, TokenSequence]],
    ) -> List[Tuple[float, TokenSequence]]:
        """Penalize candidates that are too similar to higher-ranked ones."""
        if len(candidates) <= 1:
            return candidates

        candidates.sort(key=lambda x: x[0], reverse=True)
        penalized: List[Tuple[float, TokenSequence]] = [candidates[0]]

        for i in range(1, len(candidates)):
            score, seq = candidates[i]
            max_overlap = 0.0

            for _, prev_seq in penalized:
                overlap = self._sequence_overlap(seq, prev_seq)
                max_overlap = max(max_overlap, overlap)

            penalty = self.cascade_config.diversity_penalty * max_overlap
            penalized.append((score - penalty, seq))

        return penalized

    @staticmethod
    def _sequence_overlap(seq1: List[int], seq2: List[int]) -> float:
        """Compute normalized n-gram overlap between two sequences."""
        if not seq1 or not seq2:
            return 0.0

        ngram_size = 4
        ngrams1: set = set()
        ngrams2: set = set()

        for i in range(len(seq1) - ngram_size + 1):
            ngrams1.add(tuple(seq1[i : i + ngram_size]))
        for i in range(len(seq2) - ngram_size + 1):
            ngrams2.add(tuple(seq2[i : i + ngram_size]))

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0

    def get_stage_results(self) -> List[Dict[str, Any]]:
        """Return recorded results from each cascade stage."""
        return list(self._stage_results)

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.cascade_config.stages:
            errors.append("CascadeDecoding requires at least one stage.")
        for i, stage in enumerate(self.cascade_config.stages):
            if stage.filter_ratio <= 0 or stage.filter_ratio > 1:
                errors.append(
                    f"Stage {i} filter_ratio must be in (0, 1], got {stage.filter_ratio}"
                )
        return errors


# =========================================================================
# StackedDecoding — meta-learning style combination
# =========================================================================


class CrossValidationWeightLearner:
    """Learn optimal combination weights via cross-validation.

    Given a set of algorithms and validation data (sequences with scores),
    finds the weight vector that maximizes combined scoring performance.

    Parameters
    ----------
    num_folds : int
        Number of cross-validation folds.
    learning_rate : float
        Step size for gradient-based weight optimization.
    num_iterations : int
        Number of optimization iterations per fold.
    regularization : float
        L2 regularization strength on weights.
    """

    def __init__(
        self,
        num_folds: int = 5,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        regularization: float = 0.001,
    ) -> None:
        self.num_folds = num_folds
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self._learned_weights: Optional[np.ndarray] = None
        self._fold_results: List[Dict[str, Any]] = []

    def learn_weights(
        self,
        algorithm_scores: List[np.ndarray],
        reference_scores: np.ndarray,
    ) -> np.ndarray:
        """Learn combination weights from algorithm scores and reference.

        Parameters
        ----------
        algorithm_scores : list of np.ndarray
            Each array has shape ``(num_samples,)`` containing the score
            each algorithm assigned to each sample.
        reference_scores : np.ndarray
            Shape ``(num_samples,)`` with ground-truth quality scores.

        Returns
        -------
        np.ndarray
            Optimal weight vector of shape ``(num_algorithms,)``.
        """
        num_algos = len(algorithm_scores)
        num_samples = len(reference_scores)

        if num_samples == 0 or num_algos == 0:
            return np.ones(num_algos) / max(num_algos, 1)

        score_matrix = np.column_stack(
            [s.astype(np.float64) for s in algorithm_scores]
        )
        ref = reference_scores.astype(np.float64)

        fold_size = max(1, num_samples // self.num_folds)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        all_fold_weights: List[np.ndarray] = []
        self._fold_results = []

        for fold in range(self.num_folds):
            val_start = fold * fold_size
            val_end = min(val_start + fold_size, num_samples)
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            train_scores = score_matrix[train_idx]
            train_ref = ref[train_idx]
            val_scores = score_matrix[val_idx]
            val_ref = ref[val_idx]

            weights = self._optimize_weights(train_scores, train_ref, num_algos)

            train_loss = self._compute_loss(train_scores, train_ref, weights)
            val_loss = self._compute_loss(val_scores, val_ref, weights)

            self._fold_results.append({
                "fold": fold,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "weights": weights.tolist(),
            })
            all_fold_weights.append(weights)

        if all_fold_weights:
            avg_weights = np.mean(all_fold_weights, axis=0)
        else:
            avg_weights = np.ones(num_algos, dtype=np.float64) / num_algos

        avg_weights = np.maximum(avg_weights, 0.0)
        total = np.sum(avg_weights)
        if total > 0:
            avg_weights /= total

        self._learned_weights = avg_weights
        return avg_weights

    def _optimize_weights(
        self,
        score_matrix: np.ndarray,
        reference: np.ndarray,
        num_algos: int,
    ) -> np.ndarray:
        """Optimize weights using projected gradient descent.

        Minimizes MSE between weighted combination and reference scores
        while keeping weights on the probability simplex.
        """
        weights = np.ones(num_algos, dtype=np.float64) / num_algos

        for iteration in range(self.num_iterations):
            combined = score_matrix @ weights
            residual = combined - reference

            gradient = (2.0 / len(reference)) * (score_matrix.T @ residual)
            gradient += 2.0 * self.regularization * weights

            lr = self.learning_rate / (1.0 + 0.01 * iteration)
            weights -= lr * gradient

            weights = np.maximum(weights, 0.0)
            total = np.sum(weights)
            if total > 0:
                weights /= total

        return weights

    def _compute_loss(
        self,
        score_matrix: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute MSE + regularization loss."""
        combined = score_matrix @ weights
        mse = float(np.mean((combined - reference) ** 2))
        reg = self.regularization * float(np.sum(weights ** 2))
        return mse + reg

    def get_fold_results(self) -> List[Dict[str, Any]]:
        """Return per-fold training results."""
        return list(self._fold_results)

    @property
    def learned_weights(self) -> Optional[np.ndarray]:
        return self._learned_weights


class StackedDecoding(DecodingAlgorithm):
    """Meta-learning style combination of decoding algorithms.

    First learns optimal combination weights from validation data,
    then uses those weights during generation.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    base_algorithms : list of DecodingAlgorithm
        Component algorithms to combine.
    weight_learner : CrossValidationWeightLearner
        Learns combination weights from data.
    combination_mode : str
        ``"logit"``, ``"probability"``, or ``"geometric"``.
    use_learned_weights : bool
        If True, use weights learned via cross-validation.
        If False, use uniform weights.
    """

    def __init__(
        self,
        config: DecodingConfig,
        base_algorithms: List[DecodingAlgorithm],
        weight_learner: Optional[CrossValidationWeightLearner] = None,
        combination_mode: str = "probability",
        use_learned_weights: bool = True,
    ) -> None:
        super().__init__(config)
        self.base_algorithms = base_algorithms
        self.weight_learner = weight_learner or CrossValidationWeightLearner()
        self.combination_mode = combination_mode
        self.use_learned_weights = use_learned_weights
        self._weights: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return (
            f"StackedDecoding with {len(self.base_algorithms)} algorithms, "
            f"mode={self.combination_mode}"
        )

    def train_weights(
        self,
        validation_sequences: List[List[int]],
        logit_source: LogitSource,
        reference_scores: Optional[np.ndarray] = None,
        prompt_length: int = 0,
    ) -> np.ndarray:
        """Train combination weights on validation data.

        Parameters
        ----------
        validation_sequences : list of list of int
            Complete sequences (prompt + generation) for training.
        logit_source : LogitSource
            Model for scoring.
        reference_scores : np.ndarray, optional
            Ground truth quality scores. If not provided, uses model
            log-probability as reference.
        prompt_length : int
            Length of the prompt prefix.

        Returns
        -------
        np.ndarray
            Learned weight vector.
        """
        num_seqs = len(validation_sequences)
        num_algos = len(self.base_algorithms)

        algo_scores_list: List[np.ndarray] = []
        for algo in self.base_algorithms:
            scores = np.zeros(num_seqs, dtype=np.float64)
            for seq_idx, seq in enumerate(validation_sequences):
                score = self._score_under_algorithm(
                    algo, seq, logit_source, prompt_length
                )
                scores[seq_idx] = score
            algo_scores_list.append(scores)

        if reference_scores is None:
            reference_scores = np.zeros(num_seqs, dtype=np.float64)
            for seq_idx, seq in enumerate(validation_sequences):
                ref_score = 0.0
                for t in range(max(prompt_length, 1), len(seq)):
                    prefix = [seq[:t]]
                    logits = logit_source(prefix)
                    lp = _log_softmax(logits[0])
                    if 0 <= seq[t] < lp.shape[0]:
                        ref_score += float(lp[seq[t]])
                gen_len = max(len(seq) - prompt_length, 1)
                reference_scores[seq_idx] = ref_score / gen_len

        self._weights = self.weight_learner.learn_weights(
            algo_scores_list, reference_scores
        )
        return self._weights

    def _score_under_algorithm(
        self,
        algo: DecodingAlgorithm,
        sequence: List[int],
        logit_source: LogitSource,
        prompt_length: int,
    ) -> float:
        """Score a sequence using an algorithm's temperature/config."""
        total_lp = 0.0
        temp = algo.config.temperature if algo.config.temperature > 0 else 1.0

        for t in range(max(prompt_length, 1), len(sequence)):
            prefix = [sequence[:t]]
            logits = logit_source(prefix)
            logits = logits[0] / temp
            lp = _log_softmax(logits)
            if 0 <= sequence[t] < lp.shape[0]:
                total_lp += float(lp[sequence[t]])

        gen_len = max(len(sequence) - prompt_length, 1)
        return total_lp / gen_len

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        """One step of stacked decoding with learned weights."""
        active = state.active_indices()
        if not active:
            return state

        if self._weights is not None and self.use_learned_weights:
            weights = self._weights.tolist()
        else:
            n = len(self.base_algorithms)
            weights = [1.0 / n] * n

        active_seqs = [state.sequences[i] for i in active]

        for batch_idx, seq_idx in enumerate(active):
            seq = active_seqs[batch_idx]
            algo_logits: List[np.ndarray] = []

            for algo in self.base_algorithms:
                raw = logit_source([seq])
                temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                algo_logits.append(raw[0] / temp)

            if self.combination_mode == "probability":
                combined = _weighted_probability_average(algo_logits, weights)
            elif self.combination_mode == "geometric":
                combined = _geometric_mean_logits(algo_logits, weights)
            else:
                combined = _weighted_logit_average(algo_logits, weights)

            combined = self._apply_constraints(combined, state)

            token = sample_token(
                combined,
                temperature=self.config.temperature,
            )

            state.update_sequence(seq_idx, token)
            log_probs = _log_softmax(combined)
            state.scores[seq_idx] += float(log_probs[token])

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        return state

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.base_algorithms:
            errors.append("StackedDecoding requires at least one base algorithm.")
        return errors


# =========================================================================
# VotingDecoding — majority / plurality voting
# =========================================================================


class TokenLevelVoting:
    """Aggregate token predictions from multiple algorithms via voting.

    At each step, each algorithm proposes a token. The final token is
    selected by majority vote (or weighted vote).

    Parameters
    ----------
    voting_method : str
        ``"majority"`` for simple majority, ``"weighted"`` for weighted vote,
        ``"rank"`` for Borda-count style ranking.
    tie_breaking : str
        How to break ties: ``"random"``, ``"first"``, ``"highest_prob"``.
    """

    def __init__(
        self,
        voting_method: str = "majority",
        tie_breaking: str = "random",
    ) -> None:
        self.voting_method = voting_method
        self.tie_breaking = tie_breaking
        self._vote_history: List[Dict[str, Any]] = []

    def vote(
        self,
        member_logits: List[np.ndarray],
        weights: List[float],
        temperature: float = 1.0,
    ) -> int:
        """Conduct a vote among members and return the winning token.

        Parameters
        ----------
        member_logits : list of np.ndarray
            Each member's logit distribution, shape ``(vocab_size,)``.
        weights : list of float
            Vote weight for each member.
        temperature : float
            Temperature for sampling individual member votes.

        Returns
        -------
        int
            The winning token id.
        """
        if self.voting_method == "majority":
            return self._majority_vote(member_logits, weights, temperature)
        elif self.voting_method == "weighted":
            return self._weighted_vote(member_logits, weights, temperature)
        elif self.voting_method == "rank":
            return self._rank_vote(member_logits, weights)
        else:
            return self._majority_vote(member_logits, weights, temperature)

    def _majority_vote(
        self,
        member_logits: List[np.ndarray],
        weights: List[float],
        temperature: float,
    ) -> int:
        """Simple majority vote: each member picks its argmax token."""
        votes: Counter = Counter()

        for logits, w in zip(member_logits, weights):
            token = int(np.argmax(logits))
            votes[token] += 1

        self._vote_history.append({
            "method": "majority",
            "votes": dict(votes),
            "num_unique": len(votes),
        })

        max_count = votes.most_common(1)[0][1]
        tied = [t for t, c in votes.items() if c == max_count]

        if len(tied) == 1:
            return tied[0]

        return self._break_tie(tied, member_logits, weights)

    def _weighted_vote(
        self,
        member_logits: List[np.ndarray],
        weights: List[float],
        temperature: float,
    ) -> int:
        """Weighted vote: each member's vote is weighted by its ensemble weight."""
        vote_scores: Dict[int, float] = defaultdict(float)

        for logits, w in zip(member_logits, weights):
            if temperature > 0:
                token = sample_token(logits, temperature=temperature)
            else:
                token = int(np.argmax(logits))
            vote_scores[token] += w

        self._vote_history.append({
            "method": "weighted",
            "vote_scores": dict(vote_scores),
            "num_unique": len(vote_scores),
        })

        winner = max(vote_scores, key=lambda t: vote_scores[t])
        return winner

    def _rank_vote(
        self,
        member_logits: List[np.ndarray],
        weights: List[float],
    ) -> int:
        """Borda-count style ranking vote.

        Each member ranks tokens by logit value. Points are assigned
        inversely to rank (top token gets most points). Weighted by
        member weight.
        """
        top_k_for_voting = min(100, member_logits[0].shape[-1])
        borda_scores: Dict[int, float] = defaultdict(float)

        for logits, w in zip(member_logits, weights):
            ranked_tokens = np.argsort(-logits)[:top_k_for_voting]
            for rank, token_id in enumerate(ranked_tokens):
                points = (top_k_for_voting - rank) * w
                borda_scores[int(token_id)] += points

        self._vote_history.append({
            "method": "rank",
            "top_candidates": sorted(
                borda_scores.items(), key=lambda x: x[1], reverse=True
            )[:10],
        })

        winner = max(borda_scores, key=lambda t: borda_scores[t])
        return winner

    def _break_tie(
        self,
        tied_tokens: List[int],
        member_logits: List[np.ndarray],
        weights: List[float],
    ) -> int:
        """Break a tie between tokens."""
        if self.tie_breaking == "random":
            return int(np.random.choice(tied_tokens))
        elif self.tie_breaking == "first":
            return tied_tokens[0]
        elif self.tie_breaking == "highest_prob":
            combined = _weighted_logit_average(member_logits, weights)
            best_token = tied_tokens[0]
            best_logit = combined[best_token]
            for t in tied_tokens[1:]:
                if combined[t] > best_logit:
                    best_logit = combined[t]
                    best_token = t
            return best_token
        else:
            return tied_tokens[0]

    def get_vote_history(self) -> List[Dict[str, Any]]:
        return list(self._vote_history)

    def reset(self) -> None:
        self._vote_history = []


class SequenceLevelVoting:
    """Aggregate complete sequences from multiple algorithms via voting.

    Each algorithm generates full sequences independently. The best
    sequences are selected by voting across the algorithms.

    Parameters
    ----------
    voting_method : str
        ``"majority"``, ``"weighted"``, ``"rank"``.
    similarity_threshold : float
        Minimum similarity for two sequences to be considered "the same"
        in majority voting.
    ngram_size : int
        N-gram size used for sequence similarity computation.
    """

    def __init__(
        self,
        voting_method: str = "majority",
        similarity_threshold: float = 0.8,
        ngram_size: int = 4,
    ) -> None:
        self.voting_method = voting_method
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    def vote(
        self,
        member_sequences: List[List[TokenSequence]],
        weights: List[float],
        num_outputs: int = 20,
    ) -> List[TokenSequence]:
        """Select the best sequences via voting.

        Parameters
        ----------
        member_sequences : list of list of TokenSequence
            Each outer list is one member's generated sequences.
        weights : list of float
            Vote weight for each member.
        num_outputs : int
            Number of sequences to return.

        Returns
        -------
        list of TokenSequence
            Selected sequences.
        """
        if self.voting_method == "majority":
            return self._majority_vote(member_sequences, weights, num_outputs)
        elif self.voting_method == "weighted":
            return self._weighted_vote(member_sequences, weights, num_outputs)
        elif self.voting_method == "rank":
            return self._rank_vote(member_sequences, weights, num_outputs)
        else:
            return self._majority_vote(member_sequences, weights, num_outputs)

    def _majority_vote(
        self,
        member_sequences: List[List[TokenSequence]],
        weights: List[float],
        num_outputs: int,
    ) -> List[TokenSequence]:
        """Cluster sequences by similarity, select clusters with most support."""
        all_seqs: List[Tuple[TokenSequence, float]] = []
        for member_seqs, w in zip(member_sequences, weights):
            for seq in member_seqs:
                all_seqs.append((seq, w))

        clusters: List[List[Tuple[TokenSequence, float]]] = []
        cluster_reps: List[TokenSequence] = []

        for seq, weight in all_seqs:
            placed = False
            for ci, rep in enumerate(cluster_reps):
                sim = self._sequence_similarity(seq, rep)
                if sim >= self.similarity_threshold:
                    clusters[ci].append((seq, weight))
                    placed = True
                    break
            if not placed:
                clusters.append([(seq, weight)])
                cluster_reps.append(seq)

        cluster_scores: List[Tuple[float, int]] = []
        for ci, cluster in enumerate(clusters):
            total_weight = sum(w for _, w in cluster)
            cluster_scores.append((total_weight, ci))

        cluster_scores.sort(reverse=True)

        results: List[TokenSequence] = []
        for _, ci in cluster_scores:
            if len(results) >= num_outputs:
                break
            best_seq = max(clusters[ci], key=lambda x: x[1])[0]
            results.append(best_seq)

        return results

    def _weighted_vote(
        self,
        member_sequences: List[List[TokenSequence]],
        weights: List[float],
        num_outputs: int,
    ) -> List[TokenSequence]:
        """Score each sequence by weighted sum of member support."""
        scored: List[Tuple[float, TokenSequence]] = []

        for member_idx, (member_seqs, w) in enumerate(
            zip(member_sequences, weights)
        ):
            for rank, seq in enumerate(member_seqs):
                rank_bonus = 1.0 / (1.0 + rank)
                score = w * rank_bonus
                scored.append((score, seq))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[TokenSequence] = []
        seen: set = set()
        for score, seq in scored:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                results.append(seq)
            if len(results) >= num_outputs:
                break

        return results

    def _rank_vote(
        self,
        member_sequences: List[List[TokenSequence]],
        weights: List[float],
        num_outputs: int,
    ) -> List[TokenSequence]:
        """Borda-count style ranking across members.

        Each member ranks its sequences. Points are assigned inversely
        to rank, weighted by member weight.
        """
        all_seqs: List[TokenSequence] = []
        seq_key_to_idx: Dict[Tuple[int, ...], int] = {}

        for member_seqs in member_sequences:
            for seq in member_seqs:
                key = tuple(seq)
                if key not in seq_key_to_idx:
                    seq_key_to_idx[key] = len(all_seqs)
                    all_seqs.append(seq)

        borda_scores = np.zeros(len(all_seqs), dtype=np.float64)

        for member_seqs, w in zip(member_sequences, weights):
            n = len(member_seqs)
            for rank, seq in enumerate(member_seqs):
                key = tuple(seq)
                idx = seq_key_to_idx[key]
                points = (n - rank) * w
                borda_scores[idx] += points

        ranked_indices = np.argsort(-borda_scores)

        results: List[TokenSequence] = []
        for idx in ranked_indices[:num_outputs]:
            results.append(all_seqs[idx])

        return results

    def _sequence_similarity(
        self, seq1: List[int], seq2: List[int]
    ) -> float:
        """Compute n-gram based similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0

        n = self.ngram_size

        ngrams1: set = set()
        for i in range(len(seq1) - n + 1):
            ngrams1.add(tuple(seq1[i : i + n]))

        ngrams2: set = set()
        for i in range(len(seq2) - n + 1):
            ngrams2.add(tuple(seq2[i : i + n]))

        if not ngrams1 and not ngrams2:
            return 1.0 if seq1 == seq2 else 0.0

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0


class VotingDecoding(DecodingAlgorithm):
    """Decoding via voting across multiple algorithms.

    Supports both token-level voting (one step at a time) and
    sequence-level voting (generate all, then vote).

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    members : list of DecodingAlgorithm
        Algorithms to vote among.
    weights : list of float
        Vote weight for each member.
    voting_level : str
        ``"token"`` for per-step voting, ``"sequence"`` for full-generation voting.
    token_voter : TokenLevelVoting, optional
        Token-level voting strategy.
    sequence_voter : SequenceLevelVoting, optional
        Sequence-level voting strategy.
    """

    def __init__(
        self,
        config: DecodingConfig,
        members: List[DecodingAlgorithm],
        weights: Optional[List[float]] = None,
        voting_level: str = "token",
        token_voter: Optional[TokenLevelVoting] = None,
        sequence_voter: Optional[SequenceLevelVoting] = None,
    ) -> None:
        super().__init__(config)
        self.members = members
        self.voting_level = voting_level

        if weights is not None:
            total = sum(weights)
            self._weights = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)
        else:
            n = len(members)
            self._weights = [1.0 / n] * n

        self.token_voter = token_voter or TokenLevelVoting()
        self.sequence_voter = sequence_voter or SequenceLevelVoting()

    @property
    def description(self) -> str:
        return (
            f"VotingDecoding ({self.voting_level}-level) "
            f"with {len(self.members)} members"
        )

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate via the configured voting strategy."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        if self.voting_level == "sequence":
            return self._sequence_level_generate(logit_source, prompt_ids)
        else:
            state = self._init_state(prompt_ids)
            state = self._generation_loop(state, logit_source)
            return self._finalize(state)

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        """One step of token-level voting."""
        active = state.active_indices()
        if not active:
            return state

        for seq_idx in active:
            seq = state.sequences[seq_idx]
            member_logits: List[np.ndarray] = []

            for member in self.members:
                raw = logit_source([seq])
                temp = member.config.temperature if member.config.temperature > 0 else 1.0
                adjusted = raw[0] / temp

                if member.config.repetition_penalty > 1.0:
                    adjusted = DecodingAlgorithm._apply_repetition_penalty(
                        adjusted, seq, member.config.repetition_penalty
                    )
                member_logits.append(adjusted)

            token = self.token_voter.vote(
                member_logits,
                self._weights,
                temperature=self.config.temperature,
            )

            state.update_sequence(seq_idx, token)

            combined = _weighted_logit_average(member_logits, self._weights)
            log_probs = _log_softmax(combined)
            state.scores[seq_idx] += float(log_probs[token])

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        return state

    def _sequence_level_generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate sequences from each member and vote at the sequence level."""
        member_outputs: List[List[TokenSequence]] = []

        for member in self.members:
            try:
                seqs = member.generate(logit_source, prompt_ids)
                member_outputs.append(seqs)
            except Exception as exc:
                logger.warning("Member %s failed: %s", member.name, exc)
                member_outputs.append([])

        return self.sequence_voter.vote(
            member_outputs,
            self._weights,
            num_outputs=self.config.num_sequences,
        )

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.members:
            errors.append("VotingDecoding requires at least one member.")
        if len(self._weights) != len(self.members):
            errors.append("Number of weights must match number of members.")
        return errors


# =========================================================================
# EnsembleAnalyzer — analyze ensemble behavior
# =========================================================================


class EnsembleAnalyzer:
    """Analyze the behavior and contribution of ensemble members.

    Provides metrics for understanding how well ensemble members
    collaborate, the diversity each member contributes, and
    sensitivity to weight changes.

    Parameters
    ----------
    algorithms : list of DecodingAlgorithm
        The ensemble member algorithms.
    weights : list of float
        Current ensemble weights.
    """

    def __init__(
        self,
        algorithms: List[DecodingAlgorithm],
        weights: List[float],
    ) -> None:
        self.algorithms = algorithms
        total = sum(weights)
        self.weights = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)
        self._analysis_cache: Dict[str, Any] = {}

    def compute_agreement_matrix(
        self,
        logit_source: LogitSource,
        test_sequences: List[List[int]],
        prompt_length: int = 0,
    ) -> np.ndarray:
        """Compute pairwise argmax agreement rates between algorithms.

        For each position in each test sequence, checks whether two
        algorithms would predict the same top token.

        Parameters
        ----------
        logit_source : LogitSource
            Model for computing logits.
        test_sequences : list of list of int
            Test sequences to analyze.
        prompt_length : int
            Number of prompt tokens to skip.

        Returns
        -------
        np.ndarray
            Shape ``(num_algorithms, num_algorithms)`` agreement matrix.
            Entry (i, j) is the fraction of positions where algorithms
            i and j agree on the argmax token.
        """
        n = len(self.algorithms)
        agree_count = np.zeros((n, n), dtype=np.float64)
        total_count = 0

        for seq in test_sequences:
            for t in range(max(prompt_length, 1), len(seq)):
                prefix = [seq[:t]]
                base_logits = logit_source(prefix)

                argmaxes: List[int] = []
                for algo in self.algorithms:
                    temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                    adjusted = base_logits[0] / temp
                    argmaxes.append(int(np.argmax(adjusted)))

                for i in range(n):
                    for j in range(n):
                        if argmaxes[i] == argmaxes[j]:
                            agree_count[i, j] += 1
                total_count += 1

        if total_count > 0:
            agree_count /= total_count

        self._analysis_cache["agreement_matrix"] = agree_count
        return agree_count

    def compute_diversity_contribution(
        self,
        logit_source: LogitSource,
        test_sequences: List[List[int]],
        prompt_length: int = 0,
    ) -> List[Dict[str, float]]:
        """Measure how much each algorithm contributes to ensemble diversity.

        Uses leave-one-out Jensen-Shannon divergence: for each member,
        computes how much the JSD changes when that member is removed.

        Parameters
        ----------
        logit_source : LogitSource
            Model for computing logits.
        test_sequences : list of list of int
            Test sequences to analyze.
        prompt_length : int
            Number of prompt tokens to skip.

        Returns
        -------
        list of dict
            One dict per algorithm with keys:
            ``"name"``, ``"weight"``, ``"jsd_contribution"``,
            ``"entropy_contribution"``, ``"unique_top_fraction"``.
        """
        n = len(self.algorithms)
        results: List[Dict[str, float]] = []

        jsd_full_values: List[float] = []
        jsd_leave_one_out: List[List[float]] = [[] for _ in range(n)]
        entropy_per_algo: List[List[float]] = [[] for _ in range(n)]
        unique_top_counts: List[int] = [0] * n
        total_positions = 0

        for seq in test_sequences:
            for t in range(max(prompt_length, 1), len(seq)):
                prefix = [seq[:t]]
                base_logits = logit_source(prefix)

                all_member_logits: List[np.ndarray] = []
                for algo in self.algorithms:
                    temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                    adjusted = base_logits[0] / temp
                    all_member_logits.append(adjusted)
                    entropy_per_algo[len(all_member_logits) - 1].append(
                        _compute_entropy(adjusted)
                    )

                full_jsd = _jensen_shannon_divergence(
                    all_member_logits, self.weights
                )
                jsd_full_values.append(full_jsd)

                argmaxes = [int(np.argmax(l)) for l in all_member_logits]
                argmax_counts = Counter(argmaxes)

                for i in range(n):
                    if argmax_counts[argmaxes[i]] == 1:
                        unique_top_counts[i] += 1

                for leave_out in range(n):
                    remaining_logits = [
                        l for j, l in enumerate(all_member_logits) if j != leave_out
                    ]
                    remaining_weights = [
                        w for j, w in enumerate(self.weights) if j != leave_out
                    ]
                    rw_total = sum(remaining_weights)
                    if rw_total > 0:
                        remaining_weights = [w / rw_total for w in remaining_weights]

                    if remaining_logits:
                        loo_jsd = _jensen_shannon_divergence(
                            remaining_logits, remaining_weights
                        )
                    else:
                        loo_jsd = 0.0
                    jsd_leave_one_out[leave_out].append(loo_jsd)

                total_positions += 1

        avg_full_jsd = float(np.mean(jsd_full_values)) if jsd_full_values else 0.0

        for i in range(n):
            avg_loo_jsd = (
                float(np.mean(jsd_leave_one_out[i]))
                if jsd_leave_one_out[i]
                else 0.0
            )
            jsd_contribution = avg_full_jsd - avg_loo_jsd

            avg_entropy = (
                float(np.mean(entropy_per_algo[i]))
                if entropy_per_algo[i]
                else 0.0
            )

            unique_frac = (
                unique_top_counts[i] / total_positions
                if total_positions > 0
                else 0.0
            )

            results.append({
                "name": self.algorithms[i].name,
                "weight": self.weights[i],
                "jsd_contribution": jsd_contribution,
                "entropy_contribution": avg_entropy,
                "unique_top_fraction": unique_frac,
            })

        self._analysis_cache["diversity_contribution"] = results
        return results

    def weight_sensitivity_analysis(
        self,
        logit_source: LogitSource,
        test_sequences: List[List[int]],
        prompt_length: int = 0,
        perturbation_size: float = 0.1,
        num_perturbations: int = 10,
    ) -> List[Dict[str, Any]]:
        """Analyze how sensitive the ensemble output is to weight changes.

        For each algorithm, perturbs its weight up and down and measures
        the change in the combined distribution (as KL divergence from
        the unperturbed ensemble).

        Parameters
        ----------
        logit_source : LogitSource
            Model for computing logits.
        test_sequences : list of list of int
            Test sequences to analyze.
        prompt_length : int
            Number of prompt tokens to skip.
        perturbation_size : float
            Magnitude of weight perturbation.
        num_perturbations : int
            Number of random perturbation trials per algorithm.

        Returns
        -------
        list of dict
            One dict per algorithm with sensitivity metrics:
            ``"name"``, ``"mean_kl_change"``, ``"max_kl_change"``,
            ``"sensitivity_rank"``.
        """
        n = len(self.algorithms)
        sensitivity_scores: List[float] = []

        positions: List[Tuple[List[int], int]] = []
        for seq in test_sequences:
            for t in range(max(prompt_length, 1), len(seq)):
                positions.append((seq[:t], t))

        sample_size = min(len(positions), 50)
        if sample_size == 0:
            return [
                {"name": algo.name, "mean_kl_change": 0.0, "max_kl_change": 0.0, "sensitivity_rank": i}
                for i, algo in enumerate(self.algorithms)
            ]

        sampled_positions = [
            positions[i]
            for i in np.random.choice(len(positions), sample_size, replace=False)
        ]

        for algo_idx in range(n):
            kl_changes: List[float] = []

            for _trial in range(num_perturbations):
                perturbed_weights = list(self.weights)
                delta = np.random.uniform(-perturbation_size, perturbation_size)
                perturbed_weights[algo_idx] = max(0.0, perturbed_weights[algo_idx] + delta)

                pw_total = sum(perturbed_weights)
                if pw_total > 0:
                    perturbed_weights = [w / pw_total for w in perturbed_weights]

                for prefix, _t in sampled_positions:
                    base_logits = logit_source([prefix])

                    all_member_logits: List[np.ndarray] = []
                    for algo in self.algorithms:
                        temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                        all_member_logits.append(base_logits[0] / temp)

                    orig_combined = _weighted_probability_average(
                        all_member_logits, self.weights
                    )
                    pert_combined = _weighted_probability_average(
                        all_member_logits, perturbed_weights
                    )

                    kl = _kl_divergence(orig_combined, pert_combined)
                    kl_changes.append(kl)

            mean_kl = float(np.mean(kl_changes)) if kl_changes else 0.0
            max_kl = float(np.max(kl_changes)) if kl_changes else 0.0
            sensitivity_scores.append(mean_kl)

        rank_order = np.argsort(-np.array(sensitivity_scores))
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(len(rank_order))

        results: List[Dict[str, Any]] = []
        for i in range(n):
            results.append({
                "name": self.algorithms[i].name,
                "mean_kl_change": sensitivity_scores[i],
                "max_kl_change": float(np.max([
                    abs(s) for s in sensitivity_scores
                ])) if sensitivity_scores else 0.0,
                "sensitivity_rank": int(ranks[i]),
            })

        self._analysis_cache["weight_sensitivity"] = results
        return results

    def compute_oracle_performance(
        self,
        logit_source: LogitSource,
        test_sequences: List[List[int]],
        prompt_length: int = 0,
    ) -> Dict[str, float]:
        """Compute oracle (best-possible) performance by always picking
        the best member at each step.

        This gives an upper bound on how well the ensemble could perform
        if routing were perfect.

        Parameters
        ----------
        logit_source : LogitSource
            Model for computing logits.
        test_sequences : list of list of int
            Test sequences with ground-truth tokens.
        prompt_length : int
            Number of prompt tokens to skip.

        Returns
        -------
        dict
            Oracle statistics including ``"oracle_accuracy"``,
            ``"ensemble_accuracy"``, ``"best_member_accuracy"``,
            ``"oracle_vs_ensemble_gap"``.
        """
        n = len(self.algorithms)

        oracle_correct = 0
        ensemble_correct = 0
        member_correct = [0] * n
        total = 0

        for seq in test_sequences:
            for t in range(max(prompt_length, 1), len(seq) - 1):
                prefix = [seq[:t]]
                target = seq[t]
                base_logits = logit_source(prefix)

                all_member_logits: List[np.ndarray] = []
                member_preds: List[int] = []
                for algo in self.algorithms:
                    temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                    adjusted = base_logits[0] / temp
                    all_member_logits.append(adjusted)
                    member_preds.append(int(np.argmax(adjusted)))

                for i, pred in enumerate(member_preds):
                    if pred == target:
                        member_correct[i] += 1

                any_correct = any(p == target for p in member_preds)
                if any_correct:
                    oracle_correct += 1

                combined = _weighted_probability_average(
                    all_member_logits, self.weights
                )
                ens_pred = int(np.argmax(combined))
                if ens_pred == target:
                    ensemble_correct += 1

                total += 1

        if total == 0:
            return {
                "oracle_accuracy": 0.0,
                "ensemble_accuracy": 0.0,
                "best_member_accuracy": 0.0,
                "oracle_vs_ensemble_gap": 0.0,
                "total_positions": 0,
            }

        oracle_acc = oracle_correct / total
        ens_acc = ensemble_correct / total
        best_member_acc = max(c / total for c in member_correct) if member_correct else 0.0

        return {
            "oracle_accuracy": oracle_acc,
            "ensemble_accuracy": ens_acc,
            "best_member_accuracy": best_member_acc,
            "oracle_vs_ensemble_gap": oracle_acc - ens_acc,
            "total_positions": total,
            "member_accuracies": [c / total for c in member_correct],
        }

    def compute_correlation_matrix(
        self,
        logit_source: LogitSource,
        test_sequences: List[List[int]],
        prompt_length: int = 0,
    ) -> np.ndarray:
        """Compute pairwise rank correlation between algorithms.

        Uses Spearman rank correlation on the top-k logits at each position.

        Parameters
        ----------
        logit_source : LogitSource
        test_sequences : list of list of int
        prompt_length : int

        Returns
        -------
        np.ndarray
            Shape ``(num_algorithms, num_algorithms)`` correlation matrix.
        """
        n = len(self.algorithms)
        top_k = 100
        correlation_sums = np.zeros((n, n), dtype=np.float64)
        count = 0

        for seq in test_sequences:
            for t in range(max(prompt_length, 1), len(seq)):
                prefix = [seq[:t]]
                base_logits = logit_source(prefix)

                member_ranks: List[np.ndarray] = []
                for algo in self.algorithms:
                    temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                    adjusted = base_logits[0] / temp
                    top_indices = np.argsort(-adjusted)[:top_k]
                    ranks = np.zeros(top_k, dtype=np.float64)
                    for r, idx in enumerate(top_indices):
                        ranks[r] = float(adjusted[idx])
                    member_ranks.append(ranks)

                for i in range(n):
                    for j in range(i, n):
                        corr = self._spearman_correlation(
                            member_ranks[i], member_ranks[j]
                        )
                        correlation_sums[i, j] += corr
                        correlation_sums[j, i] += corr

                count += 1

        if count > 0:
            correlation_sums /= count

        self._analysis_cache["correlation_matrix"] = correlation_sums
        return correlation_sums

    @staticmethod
    def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman rank correlation between two arrays."""
        n = len(x)
        if n < 2:
            return 0.0

        rank_x = np.empty(n)
        rank_y = np.empty(n)

        order_x = np.argsort(-x)
        order_y = np.argsort(-y)

        for r, idx in enumerate(order_x):
            rank_x[idx] = float(r)
        for r, idx in enumerate(order_y):
            rank_y[idx] = float(r)

        d = rank_x - rank_y
        d_sq_sum = float(np.sum(d ** 2))
        rho = 1.0 - (6.0 * d_sq_sum) / (n * (n ** 2 - 1))
        return max(-1.0, min(1.0, rho))

    def generate_report(
        self,
        logit_source: LogitSource,
        test_sequences: List[List[int]],
        prompt_length: int = 0,
    ) -> Dict[str, Any]:
        """Generate a comprehensive analysis report.

        Runs all analysis methods and compiles results into a single report.

        Parameters
        ----------
        logit_source : LogitSource
        test_sequences : list of list of int
        prompt_length : int

        Returns
        -------
        dict
            Complete analysis report with all metrics.
        """
        agreement = self.compute_agreement_matrix(
            logit_source, test_sequences, prompt_length
        )
        diversity = self.compute_diversity_contribution(
            logit_source, test_sequences, prompt_length
        )
        sensitivity = self.weight_sensitivity_analysis(
            logit_source, test_sequences, prompt_length
        )
        oracle = self.compute_oracle_performance(
            logit_source, test_sequences, prompt_length
        )
        correlation = self.compute_correlation_matrix(
            logit_source, test_sequences, prompt_length
        )

        return {
            "num_algorithms": len(self.algorithms),
            "algorithm_names": [a.name for a in self.algorithms],
            "weights": list(self.weights),
            "agreement_matrix": agreement.tolist(),
            "diversity_contributions": diversity,
            "weight_sensitivity": sensitivity,
            "oracle_performance": oracle,
            "correlation_matrix": correlation.tolist(),
            "summary": self._generate_summary(
                agreement, diversity, sensitivity, oracle
            ),
        }

    def _generate_summary(
        self,
        agreement: np.ndarray,
        diversity: List[Dict[str, float]],
        sensitivity: List[Dict[str, Any]],
        oracle: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate a human-readable summary of the analysis."""
        n = len(self.algorithms)

        off_diag_agreement = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag_agreement.append(agreement[i, j])

        avg_agreement = float(np.mean(off_diag_agreement)) if off_diag_agreement else 0.0
        min_agreement = float(np.min(off_diag_agreement)) if off_diag_agreement else 0.0
        max_agreement = float(np.max(off_diag_agreement)) if off_diag_agreement else 0.0

        top_contributor = max(diversity, key=lambda d: d["jsd_contribution"]) if diversity else {}
        most_sensitive = max(sensitivity, key=lambda s: s["mean_kl_change"]) if sensitivity else {}

        return {
            "average_pairwise_agreement": avg_agreement,
            "min_pairwise_agreement": min_agreement,
            "max_pairwise_agreement": max_agreement,
            "top_diversity_contributor": top_contributor.get("name", "N/A"),
            "most_weight_sensitive": most_sensitive.get("name", "N/A"),
            "oracle_accuracy": oracle.get("oracle_accuracy", 0.0),
            "ensemble_accuracy": oracle.get("ensemble_accuracy", 0.0),
            "room_for_improvement": oracle.get("oracle_vs_ensemble_gap", 0.0),
        }


# =========================================================================
# DynamicEnsemble — adaptive ensemble that adjusts weights at runtime
# =========================================================================


class DynamicEnsemble(DecodingAlgorithm):
    """Ensemble that dynamically adjusts member weights during generation.

    Uses a multi-armed bandit-inspired approach to learn which algorithms
    perform best as generation progresses.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    members : list of DecodingAlgorithm
        Algorithms to combine.
    initial_weights : list of float, optional
        Starting weights (uniform if not provided).
    adaptation_rate : float
        How quickly weights adapt based on performance.
    exploration_bonus : float
        UCB-style exploration bonus for under-used members.
    window_size : int
        Number of recent steps to consider for adaptation.
    """

    def __init__(
        self,
        config: DecodingConfig,
        members: List[DecodingAlgorithm],
        initial_weights: Optional[List[float]] = None,
        adaptation_rate: float = 0.1,
        exploration_bonus: float = 0.5,
        window_size: int = 20,
    ) -> None:
        super().__init__(config)
        self.members = members
        n = len(members)

        if initial_weights is not None:
            total = sum(initial_weights)
            self._weights = np.array(
                [w / total for w in initial_weights] if total > 0 else [1.0 / n] * n,
                dtype=np.float64,
            )
        else:
            self._weights = np.ones(n, dtype=np.float64) / n

        self.adaptation_rate = adaptation_rate
        self.exploration_bonus = exploration_bonus
        self.window_size = window_size

        self._reward_history: List[List[float]] = [[] for _ in range(n)]
        self._usage_counts = np.zeros(n, dtype=np.float64)
        self._total_steps = 0
        self._weight_history: List[List[float]] = []

    @property
    def description(self) -> str:
        return (
            f"DynamicEnsemble with {len(self.members)} members, "
            f"adaptation_rate={self.adaptation_rate}"
        )

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        self._weight_history.append(self._weights.tolist())

        for seq_idx in active:
            seq = state.sequences[seq_idx]
            member_logits: List[np.ndarray] = []

            for member in self.members:
                raw = logit_source([seq])
                temp = member.config.temperature if member.config.temperature > 0 else 1.0
                member_logits.append(raw[0] / temp)

            current_weights = self._compute_ucb_weights()
            combined = _weighted_probability_average(
                member_logits, current_weights.tolist()
            )

            combined = self._apply_constraints(combined, state)
            token = sample_token(combined, temperature=self.config.temperature)

            state.update_sequence(seq_idx, token)
            log_probs = _log_softmax(combined)
            state.scores[seq_idx] += float(log_probs[token])

            self._update_rewards(member_logits, token)

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        self._total_steps += 1
        return state

    def _compute_ucb_weights(self) -> np.ndarray:
        """Compute weights using upper confidence bound exploration."""
        n = len(self.members)
        weights = self._weights.copy()

        if self._total_steps > 0:
            for i in range(n):
                recent_rewards = self._reward_history[i][-self.window_size:]
                if recent_rewards:
                    avg_reward = np.mean(recent_rewards)
                else:
                    avg_reward = 0.0

                usage_frac = (self._usage_counts[i] + 1) / (self._total_steps + n)
                exploration = self.exploration_bonus * np.sqrt(
                    np.log(self._total_steps + 1) / (self._usage_counts[i] + 1)
                )

                weights[i] = weights[i] * (1 - self.adaptation_rate) + \
                    self.adaptation_rate * (avg_reward + exploration)

        weights = np.maximum(weights, 0.01)
        weights /= np.sum(weights)
        return weights

    def _update_rewards(
        self, member_logits: List[np.ndarray], chosen_token: int
    ) -> None:
        """Update reward history for each member based on the chosen token."""
        for i, logits in enumerate(member_logits):
            probs = _stable_softmax(logits)
            reward = float(probs[chosen_token])
            self._reward_history[i].append(reward)
            self._usage_counts[i] += self._weights[i]

    def get_weight_history(self) -> List[List[float]]:
        """Return the history of weight values across steps."""
        return list(self._weight_history)

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.members:
            errors.append("DynamicEnsemble requires at least one member.")
        return errors


# =========================================================================
# SpeculativeEnsemble — speculative decoding with ensemble verification
# =========================================================================


class SpeculativeEnsemble(DecodingAlgorithm):
    """Speculative decoding where a fast draft model proposes tokens
    and the ensemble verifies/accepts them.

    The draft algorithm generates a sequence of speculative tokens.
    The ensemble then checks each token against its combined distribution
    and accepts or rejects.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    draft_algorithm : DecodingAlgorithm
        Fast algorithm for generating draft tokens.
    verifier_algorithms : list of DecodingAlgorithm
        Algorithms used to verify drafts.
    verifier_weights : list of float
        Weights for each verifier.
    speculation_length : int
        Number of tokens to speculatively generate before verification.
    acceptance_threshold : float
        Minimum combined probability for acceptance.
    """

    def __init__(
        self,
        config: DecodingConfig,
        draft_algorithm: DecodingAlgorithm,
        verifier_algorithms: List[DecodingAlgorithm],
        verifier_weights: Optional[List[float]] = None,
        speculation_length: int = 5,
        acceptance_threshold: float = 0.1,
    ) -> None:
        super().__init__(config)
        self.draft_algorithm = draft_algorithm
        self.verifier_algorithms = verifier_algorithms
        self.speculation_length = speculation_length
        self.acceptance_threshold = acceptance_threshold

        n = len(verifier_algorithms)
        if verifier_weights is not None:
            total = sum(verifier_weights)
            self._verifier_weights = [w / total for w in verifier_weights] if total > 0 else [1.0 / n] * n
        else:
            self._verifier_weights = [1.0 / n] * n

        self._accept_count = 0
        self._reject_count = 0
        self._total_speculated = 0

    @property
    def description(self) -> str:
        return (
            f"SpeculativeEnsemble with draft={self.draft_algorithm.name}, "
            f"{len(self.verifier_algorithms)} verifiers"
        )

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        for seq_idx in active:
            seq = list(state.sequences[seq_idx])

            draft_tokens = self._generate_draft(seq, logit_source)
            accepted_tokens = self._verify_draft(
                seq, draft_tokens, logit_source
            )

            for token in accepted_tokens:
                state.update_sequence(seq_idx, token)
                if (
                    self.config.eos_token_id is not None
                    and token == self.config.eos_token_id
                ):
                    state.mark_finished(seq_idx)
                    break

            if not accepted_tokens:
                fallback = self._generate_fallback(seq, logit_source)
                state.update_sequence(seq_idx, fallback)
                if (
                    self.config.eos_token_id is not None
                    and fallback == self.config.eos_token_id
                ):
                    state.mark_finished(seq_idx)

        return state

    def _generate_draft(
        self, prefix: List[int], logit_source: LogitSource
    ) -> List[int]:
        """Generate speculative draft tokens using the draft algorithm."""
        draft_tokens: List[int] = []
        current = list(prefix)

        for _ in range(self.speculation_length):
            logits = logit_source([current])
            temp = self.draft_algorithm.config.temperature
            if temp > 0 and temp != 1.0:
                logits = logits / temp
            token = sample_token(logits[0], temperature=1.0)
            draft_tokens.append(token)
            current.append(token)

        self._total_speculated += len(draft_tokens)
        return draft_tokens

    def _verify_draft(
        self,
        prefix: List[int],
        draft_tokens: List[int],
        logit_source: LogitSource,
    ) -> List[int]:
        """Verify draft tokens against the ensemble verifiers.

        Uses rejection sampling: accept each token if the ensemble
        probability is sufficiently high relative to the draft probability.
        """
        accepted: List[int] = []
        current = list(prefix)

        for token in draft_tokens:
            verifier_logits: List[np.ndarray] = []
            for verifier in self.verifier_algorithms:
                raw = logit_source([current])
                temp = verifier.config.temperature if verifier.config.temperature > 0 else 1.0
                verifier_logits.append(raw[0] / temp)

            combined = _weighted_probability_average(
                verifier_logits, self._verifier_weights
            )
            combined_probs = _stable_softmax(combined)

            draft_logits = logit_source([current])
            draft_temp = self.draft_algorithm.config.temperature
            if draft_temp > 0 and draft_temp != 1.0:
                draft_logits = draft_logits / draft_temp
            draft_probs = _stable_softmax(draft_logits[0])

            p_target = float(combined_probs[token])
            p_draft = float(draft_probs[token])

            if p_draft > 0:
                acceptance_ratio = min(1.0, p_target / p_draft)
            else:
                acceptance_ratio = 1.0 if p_target > self.acceptance_threshold else 0.0

            if np.random.random() < acceptance_ratio:
                accepted.append(token)
                current.append(token)
                self._accept_count += 1
            else:
                self._reject_count += 1
                break

        return accepted

    def _generate_fallback(
        self, prefix: List[int], logit_source: LogitSource
    ) -> int:
        """Generate a single token using the full ensemble when draft is rejected."""
        verifier_logits: List[np.ndarray] = []
        for verifier in self.verifier_algorithms:
            raw = logit_source([prefix])
            temp = verifier.config.temperature if verifier.config.temperature > 0 else 1.0
            verifier_logits.append(raw[0] / temp)

        combined = _weighted_probability_average(
            verifier_logits, self._verifier_weights
        )
        return sample_token(combined, temperature=self.config.temperature)

    def get_acceptance_stats(self) -> Dict[str, float]:
        """Return statistics on draft token acceptance rates."""
        total = self._accept_count + self._reject_count
        return {
            "total_speculated": self._total_speculated,
            "accepted": self._accept_count,
            "rejected": self._reject_count,
            "acceptance_rate": self._accept_count / max(total, 1),
            "speedup_estimate": (
                self._total_speculated / max(total, 1)
                if self._accept_count > 0
                else 1.0
            ),
        }

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.verifier_algorithms:
            errors.append("SpeculativeEnsemble requires at least one verifier.")
        if self.speculation_length < 1:
            errors.append("speculation_length must be >= 1.")
        return errors


# =========================================================================
# BoostingEnsemble — boosting-style sequential ensemble
# =========================================================================


class BoostingEnsemble(DecodingAlgorithm):
    """Boosting-style ensemble that focuses on tokens where previous
    algorithms disagree or perform poorly.

    Each algorithm is applied sequentially. Later algorithms receive
    higher weight on positions where earlier algorithms were uncertain.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    base_algorithms : list of DecodingAlgorithm
        Algorithms applied in order.
    num_boosting_rounds : int
        Number of rounds of boosting refinement.
    focus_strength : float
        How strongly to focus on uncertain positions.
    """

    def __init__(
        self,
        config: DecodingConfig,
        base_algorithms: List[DecodingAlgorithm],
        num_boosting_rounds: int = 3,
        focus_strength: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.base_algorithms = base_algorithms
        self.num_boosting_rounds = num_boosting_rounds
        self.focus_strength = focus_strength
        self._round_weights: List[float] = []

    @property
    def description(self) -> str:
        return (
            f"BoostingEnsemble with {len(self.base_algorithms)} algorithms, "
            f"{self.num_boosting_rounds} rounds"
        )

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        for seq_idx in active:
            seq = state.sequences[seq_idx]

            cumulative_logits = np.zeros_like(logit_source([seq])[0], dtype=np.float64)
            round_weights: List[float] = []

            for round_idx in range(
                min(self.num_boosting_rounds, len(self.base_algorithms))
            ):
                algo = self.base_algorithms[round_idx % len(self.base_algorithms)]
                raw = logit_source([seq])
                temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                algo_logits = raw[0] / temp

                if round_idx > 0:
                    current_entropy = _compute_entropy(cumulative_logits)
                    weight = 1.0 + self.focus_strength * current_entropy
                else:
                    weight = 1.0

                round_weights.append(weight)
                cumulative_logits += weight * algo_logits

            total_weight = sum(round_weights)
            if total_weight > 0:
                cumulative_logits /= total_weight

            self._round_weights = round_weights

            cumulative_logits = self._apply_constraints(cumulative_logits, state)
            token = sample_token(
                cumulative_logits, temperature=self.config.temperature
            )

            state.update_sequence(seq_idx, token)
            log_probs = _log_softmax(cumulative_logits)
            state.scores[seq_idx] += float(log_probs[token])

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        return state

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.base_algorithms:
            errors.append("BoostingEnsemble requires at least one base algorithm.")
        if self.num_boosting_rounds < 1:
            errors.append("num_boosting_rounds must be >= 1.")
        return errors


# =========================================================================
# ContrastiveEnsemble — contrastive combination of algorithms
# =========================================================================


class ContrastiveEnsemble(DecodingAlgorithm):
    """Contrastive ensemble that amplifies differences between a target
    algorithm and a baseline.

    Similar to contrastive decoding (Li et al., 2022) but generalized
    to ensembles. The final distribution amplifies tokens that the
    target ensemble prefers over the baseline.

    Parameters
    ----------
    config : DecodingConfig
        Base configuration.
    target_algorithms : list of DecodingAlgorithm
        Main algorithms whose preferences are amplified.
    baseline_algorithms : list of DecodingAlgorithm
        Baseline algorithms to contrast against.
    target_weights : list of float
        Weights for target algorithms.
    baseline_weights : list of float
        Weights for baseline algorithms.
    alpha : float
        Contrastive strength. Higher values amplify differences more.
    plausibility_threshold : float
        Minimum probability in the target distribution for a token
        to be considered (adaptive plausibility constraint).
    """

    def __init__(
        self,
        config: DecodingConfig,
        target_algorithms: List[DecodingAlgorithm],
        baseline_algorithms: List[DecodingAlgorithm],
        target_weights: Optional[List[float]] = None,
        baseline_weights: Optional[List[float]] = None,
        alpha: float = 0.5,
        plausibility_threshold: float = 0.1,
    ) -> None:
        super().__init__(config)
        self.target_algorithms = target_algorithms
        self.baseline_algorithms = baseline_algorithms
        self.alpha = alpha
        self.plausibility_threshold = plausibility_threshold

        nt = len(target_algorithms)
        nb = len(baseline_algorithms)

        if target_weights is not None:
            total = sum(target_weights)
            self._target_weights = [w / total for w in target_weights] if total > 0 else [1.0 / nt] * nt
        else:
            self._target_weights = [1.0 / nt] * nt

        if baseline_weights is not None:
            total = sum(baseline_weights)
            self._baseline_weights = [w / total for w in baseline_weights] if total > 0 else [1.0 / nb] * nb
        else:
            self._baseline_weights = [1.0 / nb] * nb

    @property
    def description(self) -> str:
        return (
            f"ContrastiveEnsemble: {len(self.target_algorithms)} targets "
            f"vs {len(self.baseline_algorithms)} baselines, alpha={self.alpha}"
        )

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        for seq_idx in active:
            seq = state.sequences[seq_idx]

            target_logits_list: List[np.ndarray] = []
            for algo in self.target_algorithms:
                raw = logit_source([seq])
                temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                target_logits_list.append(raw[0] / temp)

            baseline_logits_list: List[np.ndarray] = []
            for algo in self.baseline_algorithms:
                raw = logit_source([seq])
                temp = algo.config.temperature if algo.config.temperature > 0 else 1.0
                baseline_logits_list.append(raw[0] / temp)

            target_log_probs = _weighted_probability_average(
                target_logits_list, self._target_weights
            )
            baseline_log_probs = _weighted_probability_average(
                baseline_logits_list, self._baseline_weights
            )

            contrastive_logits = (
                (1.0 + self.alpha) * target_log_probs
                - self.alpha * baseline_log_probs
            )

            target_probs = _stable_softmax(target_log_probs)
            max_target_prob = float(np.max(target_probs))
            threshold = self.plausibility_threshold * max_target_prob

            mask = target_probs < threshold
            contrastive_logits[mask] = -np.inf

            contrastive_logits = self._apply_constraints(contrastive_logits, state)
            token = sample_token(
                contrastive_logits, temperature=self.config.temperature
            )

            state.update_sequence(seq_idx, token)
            log_probs = _log_softmax(
                np.where(np.isfinite(contrastive_logits), contrastive_logits, -1e10)
            )
            state.scores[seq_idx] += float(log_probs[token])

            if (
                self.config.eos_token_id is not None
                and token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        return state

    def validate_config(self) -> List[str]:
        errors = self.config.validate()
        if not self.target_algorithms:
            errors.append("ContrastiveEnsemble requires at least one target algorithm.")
        if not self.baseline_algorithms:
            errors.append("ContrastiveEnsemble requires at least one baseline algorithm.")
        return errors


# =========================================================================
# Utility: EnsembleBuilder — fluent API for constructing ensembles
# =========================================================================


class EnsembleBuilder:
    """Fluent builder for constructing ensemble configurations.

    Example usage::

        builder = EnsembleBuilder()
        ensemble = (
            builder
            .add_member("temperature", 0.5, DecodingConfig(temperature=0.7))
            .add_member("nucleus", 0.3, DecodingConfig(params={"top_p": 0.9}))
            .add_member("topk", 0.2, DecodingConfig(params={"top_k": 50}))
            .set_strategy("probability")
            .set_temperature(1.0)
            .build(base_config)
        )
    """

    def __init__(self) -> None:
        self._members: List[MemberSpec] = []
        self._strategy: str = "logit"
        self._voting_method: str = "majority"
        self._temperature: float = 1.0
        self._top_k: int = 0
        self._top_p: float = 1.0
        self._diversity_bonus: float = 0.0
        self._num_sequences: int = 20
        self._max_new_tokens: int = 100
        self._seed: Optional[int] = None

    def add_member(
        self, name: str, weight: float, config: DecodingConfig
    ) -> "EnsembleBuilder":
        """Add an ensemble member."""
        self._members.append((name, weight, config))
        return self

    def set_strategy(self, strategy: str) -> "EnsembleBuilder":
        """Set the combination strategy."""
        self._strategy = strategy
        return self

    def set_voting_method(self, method: str) -> "EnsembleBuilder":
        """Set the voting method for sample_and_merge strategy."""
        self._voting_method = method
        return self

    def set_temperature(self, temperature: float) -> "EnsembleBuilder":
        """Set the final sampling temperature."""
        self._temperature = temperature
        return self

    def set_top_k(self, top_k: int) -> "EnsembleBuilder":
        """Set top-k filtering."""
        self._top_k = top_k
        return self

    def set_top_p(self, top_p: float) -> "EnsembleBuilder":
        """Set nucleus filtering threshold."""
        self._top_p = top_p
        return self

    def set_diversity_bonus(self, bonus: float) -> "EnsembleBuilder":
        """Set diversity bonus."""
        self._diversity_bonus = bonus
        return self

    def set_num_sequences(self, n: int) -> "EnsembleBuilder":
        """Set number of output sequences."""
        self._num_sequences = n
        return self

    def set_max_new_tokens(self, n: int) -> "EnsembleBuilder":
        """Set max new tokens."""
        self._max_new_tokens = n
        return self

    def set_seed(self, seed: int) -> "EnsembleBuilder":
        """Set random seed."""
        self._seed = seed
        return self

    def build_config(self) -> EnsembleConfig:
        """Build the EnsembleConfig."""
        return EnsembleConfig(
            members=list(self._members),
            combination_strategy=self._strategy,
            voting_method=self._voting_method,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            diversity_bonus=self._diversity_bonus,
            num_sequences=self._num_sequences,
            max_new_tokens=self._max_new_tokens,
            seed=self._seed,
        )

    def build(
        self,
        base_config: DecodingConfig,
        algorithm_instances: Optional[List[DecodingAlgorithm]] = None,
    ) -> EnsembleDecoding:
        """Build the EnsembleDecoding instance."""
        ensemble_config = self.build_config()
        return EnsembleDecoding(
            config=base_config,
            ensemble_config=ensemble_config,
            algorithm_instances=algorithm_instances,
        )
