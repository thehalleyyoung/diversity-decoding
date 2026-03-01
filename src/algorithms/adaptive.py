"""
Adaptive algorithm selection for the Diversity Decoding Arena.

Provides bandit-based algorithm selection, adaptive decoding strategies,
online learning for algorithm weights, scheduling, and analysis utilities.
Uses multi-armed bandit theory to dynamically select the best decoding
algorithm based on observed diversity and quality rewards.
"""

from __future__ import annotations

import abc
import copy
import logging
import math
import time
from dataclasses import dataclass, field
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
    sample_token,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RewardValue = float
FeatureVector = np.ndarray  # shape (d,)


# =========================================================================
# BanditState
# =========================================================================


@dataclass
class BanditState:
    """Tracks per-arm statistics for multi-armed bandit algorithms."""

    arm_names: List[str] = field(default_factory=list)
    pull_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    cumulative_rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ucb_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    total_pulls: int = 0
    reward_history: List[List[float]] = field(default_factory=list)
    selection_history: List[int] = field(default_factory=list)
    timestamp_history: List[float] = field(default_factory=list)
    # Thompson sampling state
    alpha_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    beta_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # EXP3 state
    exp3_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    exp3_gamma: float = 0.1

    @classmethod
    def create(cls, arm_names: List[str], gamma: float = 0.1) -> "BanditState":
        """Factory to create a fresh bandit state for the given arms."""
        n = len(arm_names)
        return cls(
            arm_names=list(arm_names),
            pull_counts=np.zeros(n, dtype=np.float64),
            cumulative_rewards=np.zeros(n, dtype=np.float64),
            ucb_scores=np.full(n, float("inf"), dtype=np.float64),
            total_pulls=0,
            reward_history=[[] for _ in range(n)],
            selection_history=[],
            timestamp_history=[],
            alpha_params=np.ones(n, dtype=np.float64),
            beta_params=np.ones(n, dtype=np.float64),
            exp3_weights=np.ones(n, dtype=np.float64),
            exp3_gamma=gamma,
        )

    @property
    def num_arms(self) -> int:
        return len(self.arm_names)

    def mean_rewards(self) -> np.ndarray:
        """Mean reward per arm, zero for unpulled arms."""
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(
                self.pull_counts > 0,
                self.cumulative_rewards / self.pull_counts,
                0.0,
            )
        return means

    def record_pull(self, arm: int, reward: float) -> None:
        """Record a pull of *arm* with the observed *reward*."""
        self.pull_counts[arm] += 1
        self.cumulative_rewards[arm] += reward
        self.total_pulls += 1
        self.reward_history[arm].append(reward)
        self.selection_history.append(arm)
        self.timestamp_history.append(time.monotonic())

    def clone(self) -> "BanditState":
        return copy.deepcopy(self)


# =========================================================================
# ContextualBanditState
# =========================================================================


@dataclass
class ContextualBanditState:
    """State for contextual bandit algorithms with linear reward models."""

    arm_names: List[str] = field(default_factory=list)
    feature_dim: int = 0
    # Per-arm linear model parameters (ridge regression)
    A_inv: List[np.ndarray] = field(default_factory=list)  # (d, d) inverse design matrices
    b_vectors: List[np.ndarray] = field(default_factory=list)  # (d,) reward-weighted features
    theta_hat: List[np.ndarray] = field(default_factory=list)  # (d,) estimated params
    pull_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    total_pulls: int = 0
    alpha: float = 1.0  # exploration parameter
    reward_history: List[List[float]] = field(default_factory=list)
    context_history: List[Tuple[int, np.ndarray, float]] = field(default_factory=list)
    selection_history: List[int] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        arm_names: List[str],
        feature_dim: int,
        alpha: float = 1.0,
        regularization: float = 1.0,
    ) -> "ContextualBanditState":
        """Create a contextual bandit state with LinUCB-style models."""
        n = len(arm_names)
        A_inv = [np.eye(feature_dim, dtype=np.float64) / regularization for _ in range(n)]
        b_vectors = [np.zeros(feature_dim, dtype=np.float64) for _ in range(n)]
        theta_hat = [np.zeros(feature_dim, dtype=np.float64) for _ in range(n)]
        return cls(
            arm_names=list(arm_names),
            feature_dim=feature_dim,
            A_inv=A_inv,
            b_vectors=b_vectors,
            theta_hat=theta_hat,
            pull_counts=np.zeros(n, dtype=np.float64),
            total_pulls=0,
            alpha=alpha,
            reward_history=[[] for _ in range(n)],
            context_history=[],
            selection_history=[],
        )

    @property
    def num_arms(self) -> int:
        return len(self.arm_names)

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update the model for *arm* after observing *reward* in *context*."""
        x = context.astype(np.float64)
        # Sherman-Morrison rank-1 update of A_inv
        Ax = self.A_inv[arm] @ x
        denominator = 1.0 + x @ Ax
        if abs(denominator) > 1e-12:
            self.A_inv[arm] -= np.outer(Ax, Ax) / denominator
        self.b_vectors[arm] += reward * x
        self.theta_hat[arm] = self.A_inv[arm] @ self.b_vectors[arm]
        self.pull_counts[arm] += 1
        self.total_pulls += 1
        self.reward_history[arm].append(reward)
        self.context_history.append((arm, x.copy(), reward))
        self.selection_history.append(arm)

    def predict(self, arm: int, context: np.ndarray) -> Tuple[float, float]:
        """Return (mean_reward_estimate, confidence_width) for *arm* in *context*."""
        x = context.astype(np.float64)
        mean = float(self.theta_hat[arm] @ x)
        # Confidence width: alpha * sqrt(x^T A_inv x)
        width = self.alpha * math.sqrt(max(0.0, float(x @ (self.A_inv[arm] @ x))))
        return mean, width

    def clone(self) -> "ContextualBanditState":
        return copy.deepcopy(self)


# =========================================================================
# Reward Functions
# =========================================================================


class RewardFunction(abc.ABC):
    """Base class for reward computation."""

    @abc.abstractmethod
    def compute(
        self,
        sequences: List[TokenSequence],
        new_sequence: TokenSequence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward for adding *new_sequence* to the set of *sequences*."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset any internal state."""
        ...


class DiversityReward(RewardFunction):
    """Measures diversity improvement from adding a new sequence.

    Uses pairwise edit-distance diversity: the average minimum edit distance
    of the new sequence to all existing sequences, normalised by length.
    """

    def __init__(
        self,
        ngram_order: int = 3,
        use_jaccard: bool = True,
        use_edit_distance: bool = False,
        length_penalty: float = 0.0,
    ) -> None:
        self.ngram_order = ngram_order
        self.use_jaccard = use_jaccard
        self.use_edit_distance = use_edit_distance
        self.length_penalty = length_penalty
        self._call_count = 0

    def _extract_ngrams(self, seq: TokenSequence, n: int) -> set:
        """Extract character n-grams from a token sequence."""
        if len(seq) < n:
            return {tuple(seq)}
        return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}

    def _jaccard_distance(self, a: set, b: set) -> float:
        """Jaccard distance between two sets."""
        if not a and not b:
            return 0.0
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - len(a & b) / union

    def _edit_distance(self, a: TokenSequence, b: TokenSequence) -> float:
        """Normalised edit distance between two sequences using dynamic programming."""
        la, lb = len(a), len(b)
        if la == 0 and lb == 0:
            return 0.0
        # Use two-row DP for memory efficiency
        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)
        for i in range(1, la + 1):
            curr[0] = i
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev, curr = curr, prev
        return prev[lb] / max(la, lb)

    def compute(
        self,
        sequences: List[TokenSequence],
        new_sequence: TokenSequence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute diversity reward for adding *new_sequence*."""
        self._call_count += 1
        if not sequences:
            return 1.0

        distances: List[float] = []
        new_ngrams = self._extract_ngrams(new_sequence, self.ngram_order)

        for existing in sequences:
            if self.use_jaccard:
                existing_ngrams = self._extract_ngrams(existing, self.ngram_order)
                d = self._jaccard_distance(new_ngrams, existing_ngrams)
                distances.append(d)
            if self.use_edit_distance:
                # Truncate for efficiency
                max_len = 200
                d = self._edit_distance(
                    new_sequence[:max_len], existing[:max_len]
                )
                distances.append(d)

        if not distances:
            return 0.5

        avg_dist = sum(distances) / len(distances)

        # Apply length penalty: prefer sequences of moderate length
        if self.length_penalty > 0.0 and sequences:
            avg_len = sum(len(s) for s in sequences) / len(sequences)
            if avg_len > 0:
                len_ratio = len(new_sequence) / avg_len
                len_factor = math.exp(-self.length_penalty * abs(math.log(max(len_ratio, 1e-6))))
                avg_dist *= len_factor

        return float(np.clip(avg_dist, 0.0, 1.0))

    def reset(self) -> None:
        self._call_count = 0


class QualityDiversityReward(RewardFunction):
    """Composite reward: alpha * quality + (1 - alpha) * diversity.

    Quality is estimated from log-probability scores; diversity from
    pairwise distance to existing sequences.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        diversity_reward: Optional[DiversityReward] = None,
        quality_scale: float = 1.0,
        min_quality: float = -20.0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.diversity_reward = diversity_reward or DiversityReward()
        self.quality_scale = quality_scale
        self.min_quality = min_quality
        self._call_count = 0

    def _estimate_quality(
        self, sequence: TokenSequence, metadata: Optional[Dict[str, Any]]
    ) -> float:
        """Estimate sequence quality from metadata or heuristics."""
        if metadata and "log_prob" in metadata:
            raw = float(metadata["log_prob"])
            clamped = max(raw, self.min_quality)
            # Normalise to [0, 1] using sigmoid-like transform
            return 1.0 / (1.0 + math.exp(-self.quality_scale * (clamped - self.min_quality / 2)))

        if metadata and "score" in metadata:
            score = float(metadata["score"])
            return 1.0 / (1.0 + math.exp(-score))

        # Heuristic: longer sequences tend to be higher quality up to a point
        length = len(sequence)
        return min(1.0, length / 50.0) * 0.5 + 0.25

    def compute(
        self,
        sequences: List[TokenSequence],
        new_sequence: TokenSequence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        self._call_count += 1
        quality = self._estimate_quality(new_sequence, metadata)
        diversity = self.diversity_reward.compute(sequences, new_sequence, metadata)
        composite = self.alpha * quality + (1.0 - self.alpha) * diversity
        return float(np.clip(composite, 0.0, 1.0))

    def reset(self) -> None:
        self._call_count = 0
        self.diversity_reward.reset()


class CurriculumReward(RewardFunction):
    """Time-varying reward that shifts from quality to diversity over time.

    Starts with emphasis on quality (alpha close to 1) and gradually
    increases diversity weight using a cosine or linear schedule.
    """

    def __init__(
        self,
        initial_alpha: float = 0.9,
        final_alpha: float = 0.1,
        total_steps: int = 100,
        schedule: str = "cosine",
        diversity_reward: Optional[DiversityReward] = None,
        quality_scale: float = 1.0,
    ) -> None:
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_steps = max(total_steps, 1)
        self.schedule = schedule
        self.diversity_reward = diversity_reward or DiversityReward()
        self.quality_scale = quality_scale
        self._step = 0
        self._qd_reward = QualityDiversityReward(
            alpha=initial_alpha,
            diversity_reward=self.diversity_reward,
            quality_scale=quality_scale,
        )

    def _get_alpha(self) -> float:
        """Get current alpha based on schedule."""
        progress = min(self._step / self.total_steps, 1.0)
        if self.schedule == "cosine":
            # Cosine annealing from initial to final
            alpha = self.final_alpha + (self.initial_alpha - self.final_alpha) * (
                1.0 + math.cos(math.pi * progress)
            ) / 2.0
        elif self.schedule == "linear":
            alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        elif self.schedule == "exponential":
            decay = math.exp(-5.0 * progress)
            alpha = self.final_alpha + (self.initial_alpha - self.final_alpha) * decay
        elif self.schedule == "step":
            # Step function: switch at midpoint
            if progress < 0.5:
                alpha = self.initial_alpha
            else:
                alpha = self.final_alpha
        else:
            alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        return float(np.clip(alpha, min(self.initial_alpha, self.final_alpha),
                             max(self.initial_alpha, self.final_alpha)))

    def compute(
        self,
        sequences: List[TokenSequence],
        new_sequence: TokenSequence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        alpha = self._get_alpha()
        self._qd_reward.alpha = alpha
        reward = self._qd_reward.compute(sequences, new_sequence, metadata)
        self._step += 1
        return reward

    def reset(self) -> None:
        self._step = 0
        self._qd_reward.reset()
        self.diversity_reward.reset()


# =========================================================================
# Bandit Strategies
# =========================================================================


class BanditStrategy(abc.ABC):
    """Abstract strategy for arm selection in a multi-armed bandit."""

    @abc.abstractmethod
    def select_arm(self, state: BanditState, rng: np.random.Generator) -> int:
        """Select which arm to pull next."""
        ...

    @abc.abstractmethod
    def update(self, state: BanditState, arm: int, reward: float) -> None:
        """Update the strategy state after observing a reward."""
        ...


class UCB1Strategy(BanditStrategy):
    """Upper Confidence Bound (UCB1) algorithm.

    Selects arm = argmax(mean_reward + c * sqrt(ln(T) / n_i)),
    where T is total pulls and n_i is pulls of arm i.
    """

    def __init__(self, exploration_constant: float = 2.0) -> None:
        self.exploration_constant = exploration_constant

    def select_arm(self, state: BanditState, rng: np.random.Generator) -> int:
        n = state.num_arms
        if n == 0:
            raise ValueError("No arms available")

        # Pull each arm at least once
        for i in range(n):
            if state.pull_counts[i] == 0:
                return i

        means = state.mean_rewards()
        log_t = math.log(state.total_pulls)
        exploration = np.sqrt(self.exploration_constant * log_t / state.pull_counts)
        ucb_values = means + exploration
        state.ucb_scores = ucb_values.copy()

        # Break ties randomly
        max_val = np.max(ucb_values)
        candidates = np.where(np.abs(ucb_values - max_val) < 1e-10)[0]
        return int(rng.choice(candidates))

    def update(self, state: BanditState, arm: int, reward: float) -> None:
        state.record_pull(arm, reward)
        # Recompute UCB scores
        if state.total_pulls > 0 and np.all(state.pull_counts > 0):
            means = state.mean_rewards()
            log_t = math.log(state.total_pulls)
            exploration = np.sqrt(self.exploration_constant * log_t / state.pull_counts)
            state.ucb_scores = means + exploration


class ThompsonSamplingStrategy(BanditStrategy):
    """Thompson Sampling with Beta(alpha, beta) posteriors.

    Rewards are assumed to be in [0, 1]. Each arm has a Beta
    posterior updated with Bernoulli-like observations.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_arm(self, state: BanditState, rng: np.random.Generator) -> int:
        n = state.num_arms
        if n == 0:
            raise ValueError("No arms available")

        # Sample from each arm's Beta posterior
        samples = np.array([
            rng.beta(state.alpha_params[i], state.beta_params[i])
            for i in range(n)
        ])

        max_val = np.max(samples)
        candidates = np.where(np.abs(samples - max_val) < 1e-10)[0]
        return int(rng.choice(candidates))

    def update(self, state: BanditState, arm: int, reward: float) -> None:
        # Clip reward to [0, 1] for Beta update
        r = float(np.clip(reward, 0.0, 1.0))
        state.alpha_params[arm] += r
        state.beta_params[arm] += (1.0 - r)
        state.record_pull(arm, reward)


class EXP3Strategy(BanditStrategy):
    """EXP3 (Exponential-weight algorithm for Exploration and Exploitation).

    Designed for adversarial bandits. Maintains probability weights
    over arms and updates using importance-weighted rewards.
    """

    def __init__(self, gamma: float = 0.1) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma
        self._last_probs: Optional[np.ndarray] = None

    def _compute_probs(self, state: BanditState) -> np.ndarray:
        """Compute mixed strategy from EXP3 weights."""
        n = state.num_arms
        weights = state.exp3_weights.copy()
        total_w = weights.sum()
        if total_w <= 0 or not np.isfinite(total_w):
            return np.ones(n) / n

        probs = (1.0 - self.gamma) * (weights / total_w) + self.gamma / n
        probs = np.maximum(probs, 1e-10)
        probs /= probs.sum()
        return probs

    def select_arm(self, state: BanditState, rng: np.random.Generator) -> int:
        n = state.num_arms
        if n == 0:
            raise ValueError("No arms available")

        probs = self._compute_probs(state)
        self._last_probs = probs.copy()
        return int(rng.choice(n, p=probs))

    def update(self, state: BanditState, arm: int, reward: float) -> None:
        n = state.num_arms
        probs = self._last_probs if self._last_probs is not None else self._compute_probs(state)

        # Importance-weighted reward estimate
        r = float(np.clip(reward, 0.0, 1.0))
        estimated_reward = r / max(probs[arm], 1e-10)
        # Clip to prevent weight explosion
        estimated_reward = min(estimated_reward, 10.0)

        # Update weight for pulled arm
        state.exp3_weights[arm] *= math.exp(self.gamma * estimated_reward / n)

        # Normalise weights to prevent overflow
        max_w = np.max(state.exp3_weights)
        if max_w > 1e6:
            state.exp3_weights /= max_w

        state.record_pull(arm, reward)


class ContextualBanditStrategy:
    """LinUCB contextual bandit strategy.

    Uses linear reward models per arm with UCB-style exploration
    based on the confidence ellipsoid.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def select_arm(
        self,
        state: ContextualBanditState,
        context: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        n = state.num_arms
        if n == 0:
            raise ValueError("No arms available")

        ucb_values = np.zeros(n)
        for i in range(n):
            mean, width = state.predict(i, context)
            ucb_values[i] = mean + width

        max_val = np.max(ucb_values)
        candidates = np.where(np.abs(ucb_values - max_val) < 1e-10)[0]
        return int(rng.choice(candidates))

    def update(
        self,
        state: ContextualBanditState,
        arm: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        state.update(arm, context, reward)


# =========================================================================
# Feature Extractors (for contextual bandits)
# =========================================================================


class PromptFeatureExtractor:
    """Extracts features from prompt token ids for contextual bandits."""

    def __init__(self, feature_dim: int = 16, vocab_size: int = 50000) -> None:
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size

    def extract(self, prompt_ids: List[int], step: int = 0, max_steps: int = 100) -> np.ndarray:
        """Extract a feature vector from prompt ids and generation progress."""
        features = np.zeros(self.feature_dim, dtype=np.float64)

        if not prompt_ids:
            features[0] = 1.0  # bias
            return features

        prompt_len = len(prompt_ids)
        features[0] = 1.0  # bias
        features[1] = min(prompt_len / 100.0, 1.0)  # normalised prompt length
        features[2] = step / max(max_steps, 1)  # generation progress

        # Token statistics
        unique_tokens = len(set(prompt_ids))
        features[3] = unique_tokens / max(prompt_len, 1)  # type-token ratio

        # Simple token distribution features using hash buckets
        bucket_size = max(self.feature_dim - 8, 4)
        bucket_counts = np.zeros(bucket_size, dtype=np.float64)
        for tok in prompt_ids:
            bucket = tok % bucket_size
            bucket_counts[bucket] += 1
        if prompt_len > 0:
            bucket_counts /= prompt_len
        # Entropy of bucket distribution
        ent = 0.0
        for p in bucket_counts:
            if p > 0:
                ent -= p * math.log(p + 1e-10)
        features[4] = ent / max(math.log(bucket_size), 1.0)

        # Last-token features
        features[5] = (prompt_ids[-1] % 1000) / 1000.0
        features[6] = (prompt_ids[0] % 1000) / 1000.0

        # Repetition feature
        if prompt_len >= 2:
            bigrams = [(prompt_ids[i], prompt_ids[i + 1]) for i in range(prompt_len - 1)]
            unique_bigrams = len(set(bigrams))
            features[7] = unique_bigrams / max(len(bigrams), 1)

        # Fill remaining with bucket distribution
        remaining = min(bucket_size, self.feature_dim - 8)
        features[8 : 8 + remaining] = bucket_counts[:remaining]

        return features


class SequenceFeatureExtractor:
    """Extracts features from partially generated sequences."""

    def __init__(self, feature_dim: int = 12) -> None:
        self.feature_dim = feature_dim

    def extract(
        self,
        sequences: List[TokenSequence],
        step: int,
        max_steps: int,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract features from current generation state."""
        features = np.zeros(self.feature_dim, dtype=np.float64)
        features[0] = 1.0  # bias
        features[1] = step / max(max_steps, 1)  # progress

        if not sequences:
            return features

        # Average sequence length
        avg_len = sum(len(s) for s in sequences) / len(sequences)
        features[2] = min(avg_len / 100.0, 1.0)

        # Diversity of current set (pairwise token overlap)
        if len(sequences) >= 2:
            overlaps = []
            for i in range(min(len(sequences), 10)):
                for j in range(i + 1, min(len(sequences), 10)):
                    si = set(sequences[i][-20:]) if len(sequences[i]) > 20 else set(sequences[i])
                    sj = set(sequences[j][-20:]) if len(sequences[j]) > 20 else set(sequences[j])
                    union = len(si | sj)
                    if union > 0:
                        overlaps.append(len(si & sj) / union)
            if overlaps:
                features[3] = 1.0 - sum(overlaps) / len(overlaps)  # diversity

        # Number of active sequences
        features[4] = len(sequences) / 100.0

        # Entropy features from logits
        if logits is not None and logits.size > 0:
            if logits.ndim == 1:
                logits_2d = logits.reshape(1, -1)
            else:
                logits_2d = logits
            for idx, row in enumerate(logits_2d[:3]):
                probs = _stable_softmax(row)
                ent = -np.sum(probs * np.log(probs + 1e-10))
                max_ent = math.log(len(row)) if len(row) > 1 else 1.0
                features[5 + idx] = ent / max_ent

        # Token type-token ratio for recent tokens
        if sequences:
            recent = [s[-10:] for s in sequences if len(s) > 0]
            all_recent = [t for s in recent for t in s]
            if all_recent:
                features[8] = len(set(all_recent)) / len(all_recent)

        # Variance in sequence lengths
        if len(sequences) > 1:
            lengths = [len(s) for s in sequences]
            features[9] = min(np.std(lengths) / max(avg_len, 1.0), 1.0)

        # Recent unique token ratio across all sequences
        if sequences:
            last_tokens = [s[-1] for s in sequences if len(s) > 0]
            if last_tokens:
                features[10] = len(set(last_tokens)) / len(last_tokens)

        features[11] = min(len(sequences) / 50.0, 1.0)  # normalised seq count

        return features


# =========================================================================
# BanditAlgorithmSelector
# =========================================================================


class BanditAlgorithmSelector:
    """Multi-armed bandit for selecting among decoding algorithms.

    Each arm corresponds to one decoding algorithm. The selector uses
    a bandit strategy (UCB1, Thompson Sampling, EXP3, or contextual)
    to choose which algorithm to run next, then updates rewards based
    on the diversity / quality of the generated sequences.
    """

    def __init__(
        self,
        algorithm_names: List[str],
        strategy: str = "ucb1",
        reward_function: Optional[RewardFunction] = None,
        exploration_constant: float = 2.0,
        gamma: float = 0.1,
        seed: Optional[int] = None,
        feature_dim: int = 16,
        contextual_alpha: float = 1.0,
    ) -> None:
        self.algorithm_names = list(algorithm_names)
        self.strategy_name = strategy
        self.reward_function = reward_function or DiversityReward()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Initialise bandit state
        self.state = BanditState.create(algorithm_names, gamma=gamma)

        # Initialise strategy
        if strategy == "ucb1":
            self._strategy: BanditStrategy = UCB1Strategy(exploration_constant)
        elif strategy == "thompson":
            self._strategy = ThompsonSamplingStrategy()
        elif strategy == "exp3":
            self._strategy = EXP3Strategy(gamma)
        elif strategy == "contextual":
            self._contextual_strategy = ContextualBanditStrategy(contextual_alpha)
            self._contextual_state = ContextualBanditState.create(
                algorithm_names, feature_dim, alpha=contextual_alpha
            )
            self._feature_extractor = PromptFeatureExtractor(feature_dim)
            self._strategy = UCB1Strategy(exploration_constant)  # fallback
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self._is_contextual = strategy == "contextual"
        self._selection_times: List[float] = []

    def select(
        self,
        context: Optional[np.ndarray] = None,
        prompt_ids: Optional[List[int]] = None,
        step: int = 0,
    ) -> int:
        """Select which algorithm (arm index) to use next."""
        t0 = time.monotonic()

        if self._is_contextual:
            if context is None and prompt_ids is not None:
                context = self._feature_extractor.extract(prompt_ids, step)
            if context is not None:
                arm = self._contextual_strategy.select_arm(
                    self._contextual_state, context, self._rng
                )
            else:
                arm = self._strategy.select_arm(self.state, self._rng)
        else:
            arm = self._strategy.select_arm(self.state, self._rng)

        self._selection_times.append(time.monotonic() - t0)
        return arm

    def update(
        self,
        arm: int,
        reward: float,
        context: Optional[np.ndarray] = None,
    ) -> None:
        """Update the bandit after observing a reward for the selected arm."""
        if self._is_contextual and context is not None:
            self._contextual_strategy.update(
                self._contextual_state, arm, context, reward
            )
        else:
            self._strategy.update(self.state, arm, reward)

    def compute_reward(
        self,
        existing_sequences: List[TokenSequence],
        new_sequence: TokenSequence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute the reward for a newly generated sequence."""
        return self.reward_function.compute(existing_sequences, new_sequence, metadata)

    def select_and_name(
        self,
        context: Optional[np.ndarray] = None,
        prompt_ids: Optional[List[int]] = None,
        step: int = 0,
    ) -> Tuple[int, str]:
        """Select an arm and return (arm_index, algorithm_name)."""
        arm = self.select(context, prompt_ids, step)
        return arm, self.algorithm_names[arm]

    def get_arm_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for each arm."""
        stats: Dict[str, Any] = {}
        for i, name in enumerate(self.algorithm_names):
            pulls = int(self.state.pull_counts[i])
            mean_r = float(self.state.mean_rewards()[i]) if pulls > 0 else 0.0
            recent_rewards = self.state.reward_history[i][-10:]
            stats[name] = {
                "pulls": pulls,
                "mean_reward": mean_r,
                "recent_mean": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0,
                "ucb_score": float(self.state.ucb_scores[i]) if i < len(self.state.ucb_scores) else 0.0,
            }
        stats["total_pulls"] = self.state.total_pulls
        stats["avg_selection_time"] = (
            sum(self._selection_times) / len(self._selection_times)
            if self._selection_times
            else 0.0
        )
        return stats

    def reset(self) -> None:
        """Reset all bandit state."""
        self.state = BanditState.create(self.algorithm_names, gamma=self.state.exp3_gamma)
        if self._is_contextual:
            self._contextual_state = ContextualBanditState.create(
                self.algorithm_names,
                self._contextual_state.feature_dim,
                alpha=self._contextual_state.alpha,
            )
        self.reward_function.reset()
        self._selection_times.clear()
        self._rng = np.random.default_rng(self.seed)

    def best_arm(self) -> Tuple[int, str]:
        """Return the arm with highest empirical mean reward."""
        means = self.state.mean_rewards()
        best = int(np.argmax(means))
        return best, self.algorithm_names[best]

    def regret_bound(self) -> float:
        """Compute theoretical UCB1 regret bound: O(sqrt(K * T * ln(T)))."""
        k = self.state.num_arms
        t = max(self.state.total_pulls, 1)
        return math.sqrt(k * t * math.log(max(t, 2)))


# =========================================================================
# Adaptive Schedulers
# =========================================================================


class AdaptiveScheduler(abc.ABC):
    """Base class for scheduling algorithm switches during generation."""

    @abc.abstractmethod
    def select(self, step: int, total_steps: int, state: Optional[Any] = None) -> int:
        """Return the arm/algorithm index to use at this step."""
        ...

    @abc.abstractmethod
    def update(self, step: int, arm: int, reward: float) -> None:
        """Update scheduler state after observing a reward."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset scheduler state."""
        ...

    @abc.abstractmethod
    def get_weights(self) -> np.ndarray:
        """Return current algorithm weights."""
        ...


class WarmupScheduler(AdaptiveScheduler):
    """Start with one algorithm for warmup, then switch to bandit selection.

    During the warmup phase, uses a single fixed algorithm. After warmup,
    delegates to a bandit selector for adaptive selection.
    """

    def __init__(
        self,
        num_arms: int,
        warmup_arm: int = 0,
        warmup_steps: int = 10,
        post_warmup_strategy: str = "round_robin",
        seed: Optional[int] = None,
    ) -> None:
        self.num_arms = num_arms
        self.warmup_arm = warmup_arm
        self.warmup_steps = warmup_steps
        self.post_warmup_strategy = post_warmup_strategy
        self._rng = np.random.default_rng(seed)
        self._weights = np.ones(num_arms, dtype=np.float64) / num_arms
        self._step_count = 0
        self._reward_sums = np.zeros(num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(num_arms, dtype=np.float64)

    def select(self, step: int, total_steps: int, state: Optional[Any] = None) -> int:
        self._step_count = step
        if step < self.warmup_steps:
            return self.warmup_arm

        if self.post_warmup_strategy == "round_robin":
            return (step - self.warmup_steps) % self.num_arms
        elif self.post_warmup_strategy == "best":
            if np.any(self._pull_counts > 0):
                means = np.where(
                    self._pull_counts > 0,
                    self._reward_sums / self._pull_counts,
                    0.0,
                )
                return int(np.argmax(means))
            return 0
        elif self.post_warmup_strategy == "weighted":
            total = self._weights.sum()
            if total > 0:
                probs = self._weights / total
                return int(self._rng.choice(self.num_arms, p=probs))
            return int(self._rng.integers(self.num_arms))
        else:
            return step % self.num_arms

    def update(self, step: int, arm: int, reward: float) -> None:
        self._reward_sums[arm] += reward
        self._pull_counts[arm] += 1
        # Update weights based on cumulative performance
        if np.any(self._pull_counts > 0):
            means = np.where(
                self._pull_counts > 0,
                self._reward_sums / self._pull_counts,
                0.0,
            )
            # Softmax weights
            max_m = np.max(means)
            exp_m = np.exp(means - max_m)
            self._weights = exp_m / exp_m.sum()

    def reset(self) -> None:
        self._step_count = 0
        self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms
        self._reward_sums = np.zeros(self.num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(self.num_arms, dtype=np.float64)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()


class CyclicScheduler(AdaptiveScheduler):
    """Cycle through algorithms with optional per-arm duration control.

    Each algorithm gets a fixed number of steps per cycle, allowing
    different algorithms to contribute at different phases.
    """

    def __init__(
        self,
        num_arms: int,
        steps_per_arm: Optional[List[int]] = None,
        cycle_order: Optional[List[int]] = None,
    ) -> None:
        self.num_arms = num_arms
        self.steps_per_arm = steps_per_arm or [1] * num_arms
        self.cycle_order = cycle_order or list(range(num_arms))
        self._weights = np.ones(num_arms, dtype=np.float64) / num_arms
        self._reward_sums = np.zeros(num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(num_arms, dtype=np.float64)

        # Build expanded cycle
        self._cycle: List[int] = []
        for arm_idx in self.cycle_order:
            duration = self.steps_per_arm[arm_idx] if arm_idx < len(self.steps_per_arm) else 1
            self._cycle.extend([arm_idx] * duration)
        if not self._cycle:
            self._cycle = list(range(num_arms))

    def select(self, step: int, total_steps: int, state: Optional[Any] = None) -> int:
        cycle_len = len(self._cycle)
        idx = step % cycle_len
        return self._cycle[idx]

    def update(self, step: int, arm: int, reward: float) -> None:
        self._reward_sums[arm] += reward
        self._pull_counts[arm] += 1

    def reset(self) -> None:
        self._reward_sums = np.zeros(self.num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(self.num_arms, dtype=np.float64)
        self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms

    def get_weights(self) -> np.ndarray:
        # Weights proportional to time spent on each arm
        total = sum(self.steps_per_arm)
        if total > 0:
            return np.array([s / total for s in self.steps_per_arm], dtype=np.float64)
        return self._weights.copy()


class DecayScheduler(AdaptiveScheduler):
    """Gradually shift weights from initial distribution to performance-based.

    Uses exponential or polynomial decay to transition from uniform
    exploration to exploitation of the best-performing algorithm.
    """

    def __init__(
        self,
        num_arms: int,
        initial_weights: Optional[np.ndarray] = None,
        decay_rate: float = 0.01,
        decay_type: str = "exponential",
        min_weight: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.num_arms = num_arms
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.min_weight = min_weight
        self._rng = np.random.default_rng(seed)

        if initial_weights is not None:
            self._initial_weights = initial_weights.copy().astype(np.float64)
        else:
            self._initial_weights = np.ones(num_arms, dtype=np.float64) / num_arms

        self._weights = self._initial_weights.copy()
        self._reward_sums = np.zeros(num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(num_arms, dtype=np.float64)

    def _compute_performance_weights(self) -> np.ndarray:
        """Compute weights based on observed performance."""
        if not np.any(self._pull_counts > 0):
            return np.ones(self.num_arms, dtype=np.float64) / self.num_arms

        means = np.where(
            self._pull_counts > 0,
            self._reward_sums / self._pull_counts,
            0.0,
        )
        # Softmax with temperature
        max_m = np.max(means)
        exp_m = np.exp(2.0 * (means - max_m))
        weights = exp_m / exp_m.sum()
        return np.maximum(weights, self.min_weight)

    def select(self, step: int, total_steps: int, state: Optional[Any] = None) -> int:
        progress = step / max(total_steps, 1)

        if self.decay_type == "exponential":
            exploration_weight = math.exp(-self.decay_rate * step)
        elif self.decay_type == "polynomial":
            exploration_weight = 1.0 / (1.0 + self.decay_rate * step)
        elif self.decay_type == "linear":
            exploration_weight = max(0.0, 1.0 - progress)
        elif self.decay_type == "cosine":
            exploration_weight = (1.0 + math.cos(math.pi * progress)) / 2.0
        else:
            exploration_weight = math.exp(-self.decay_rate * step)

        performance_weights = self._compute_performance_weights()
        # Blend initial (exploratory) and performance-based weights
        self._weights = (
            exploration_weight * self._initial_weights
            + (1.0 - exploration_weight) * performance_weights
        )
        self._weights = np.maximum(self._weights, self.min_weight)
        self._weights /= self._weights.sum()

        return int(self._rng.choice(self.num_arms, p=self._weights))

    def update(self, step: int, arm: int, reward: float) -> None:
        self._reward_sums[arm] += reward
        self._pull_counts[arm] += 1

    def reset(self) -> None:
        self._weights = self._initial_weights.copy()
        self._reward_sums = np.zeros(self.num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(self.num_arms, dtype=np.float64)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()


# =========================================================================
# Online Learners
# =========================================================================


class OnlineLearner(abc.ABC):
    """Abstract base for online learning algorithms for algorithm weights."""

    @abc.abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Update weights after observing reward for arm."""
        ...

    @abc.abstractmethod
    def get_weights(self) -> np.ndarray:
        """Return current probability weights over arms."""
        ...

    @abc.abstractmethod
    def select(self, rng: np.random.Generator) -> int:
        """Sample an arm from current weight distribution."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset learner state."""
        ...


class ExponentialWeightUpdate(OnlineLearner):
    """Multiplicative Weights / Hedge algorithm.

    Maintains weights w_i and updates: w_i *= exp(eta * reward_i).
    Selection proportional to normalised weights.
    """

    def __init__(
        self,
        num_arms: int,
        learning_rate: float = 0.1,
        min_weight: float = 1e-6,
    ) -> None:
        self.num_arms = num_arms
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self._weights = np.ones(num_arms, dtype=np.float64) / num_arms
        self._log_weights = np.zeros(num_arms, dtype=np.float64)
        self._step = 0
        self._cumulative_rewards = np.zeros(num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(num_arms, dtype=np.float64)

    def update(self, arm: int, reward: float) -> None:
        self._step += 1
        self._cumulative_rewards[arm] += reward
        self._pull_counts[arm] += 1

        # Importance-weighted update for unbiased estimator
        prob = max(self._weights[arm], self.min_weight)
        estimated_reward = reward / prob

        # Update log weights
        self._log_weights[arm] += self.learning_rate * estimated_reward

        # Recompute weights from log space for numerical stability
        max_lw = np.max(self._log_weights)
        self._weights = np.exp(self._log_weights - max_lw)
        total = self._weights.sum()
        if total > 0 and np.isfinite(total):
            self._weights /= total
        else:
            self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms

        self._weights = np.maximum(self._weights, self.min_weight)
        self._weights /= self._weights.sum()

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def select(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.num_arms, p=self._weights))

    def reset(self) -> None:
        self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms
        self._log_weights = np.zeros(self.num_arms, dtype=np.float64)
        self._step = 0
        self._cumulative_rewards = np.zeros(self.num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(self.num_arms, dtype=np.float64)


class MirrorDescent(OnlineLearner):
    """Online Mirror Descent with entropic regularisation (KL divergence).

    Updates weights using the mirror map (softmax / negative entropy).
    The update rule in dual space: theta_i += eta * reward_hat_i,
    then project back: w_i = exp(theta_i) / sum(exp(theta_j)).
    """

    def __init__(
        self,
        num_arms: int,
        learning_rate: float = 0.05,
        adaptive_lr: bool = True,
    ) -> None:
        self.num_arms = num_arms
        self.initial_learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        self._theta = np.zeros(num_arms, dtype=np.float64)  # dual variables
        self._weights = np.ones(num_arms, dtype=np.float64) / num_arms
        self._step = 0
        self._gradient_sq_sum = np.zeros(num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(num_arms, dtype=np.float64)

    def _learning_rate(self) -> float:
        """Compute current learning rate, optionally adaptive."""
        if self.adaptive_lr and self._step > 0:
            return self.initial_learning_rate / math.sqrt(self._step)
        return self.initial_learning_rate

    def _mirror_map(self) -> None:
        """Apply the mirror map (softmax) to project dual variables to simplex."""
        max_theta = np.max(self._theta)
        self._weights = np.exp(self._theta - max_theta)
        total = self._weights.sum()
        if total > 0 and np.isfinite(total):
            self._weights /= total
        else:
            self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms

    def update(self, arm: int, reward: float) -> None:
        self._step += 1
        self._pull_counts[arm] += 1

        lr = self._learning_rate()

        # Construct gradient estimate (importance-weighted)
        prob = max(self._weights[arm], 1e-10)
        gradient = np.zeros(self.num_arms, dtype=np.float64)
        gradient[arm] = reward / prob

        # Accumulate gradient squares for adaptive learning rate per arm
        self._gradient_sq_sum += gradient ** 2

        # Dual update with per-coordinate adaptive scaling
        if self.adaptive_lr:
            adaptive_lr = lr / (np.sqrt(self._gradient_sq_sum) + 1e-8)
            self._theta += adaptive_lr * gradient
        else:
            self._theta += lr * gradient

        # Project back via mirror map
        self._mirror_map()

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def select(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.num_arms, p=self._weights))

    def reset(self) -> None:
        self._theta = np.zeros(self.num_arms, dtype=np.float64)
        self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms
        self._step = 0
        self._gradient_sq_sum = np.zeros(self.num_arms, dtype=np.float64)
        self._pull_counts = np.zeros(self.num_arms, dtype=np.float64)


class FollowTheRegularizedLeader(OnlineLearner):
    """Follow The Regularized Leader (FTRL) with negative entropy regulariser.

    Maintains cumulative losses and selects the arm minimising
    cumulative loss + regularisation. With entropic regulariser,
    this is equivalent to Hedge/EWA with a particular learning rate.
    """

    def __init__(
        self,
        num_arms: int,
        regularization: float = 1.0,
        loss_type: str = "negative_reward",
    ) -> None:
        self.num_arms = num_arms
        self.regularization = regularization
        self.loss_type = loss_type
        self._cumulative_loss = np.zeros(num_arms, dtype=np.float64)
        self._cumulative_reward = np.zeros(num_arms, dtype=np.float64)
        self._weights = np.ones(num_arms, dtype=np.float64) / num_arms
        self._step = 0
        self._pull_counts = np.zeros(num_arms, dtype=np.float64)

    def _compute_weights(self) -> None:
        """Solve the FTRL optimisation: argmin_w <w, cum_loss> + reg * sum(w_i ln w_i)."""
        # With entropic regulariser, solution is proportional to exp(-cum_loss / reg)
        scaled = -self._cumulative_loss / max(self.regularization, 1e-10)
        max_s = np.max(scaled)
        self._weights = np.exp(scaled - max_s)
        total = self._weights.sum()
        if total > 0 and np.isfinite(total):
            self._weights /= total
        else:
            self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms

    def update(self, arm: int, reward: float) -> None:
        self._step += 1
        self._pull_counts[arm] += 1
        self._cumulative_reward[arm] += reward

        # Convert reward to loss
        prob = max(self._weights[arm], 1e-10)
        if self.loss_type == "negative_reward":
            loss_estimate = np.zeros(self.num_arms, dtype=np.float64)
            loss_estimate[arm] = -reward / prob
        elif self.loss_type == "inverse_reward":
            loss_estimate = np.zeros(self.num_arms, dtype=np.float64)
            loss_estimate[arm] = (1.0 - reward) / prob
        else:
            loss_estimate = np.zeros(self.num_arms, dtype=np.float64)
            loss_estimate[arm] = -reward / prob

        self._cumulative_loss += loss_estimate
        self._compute_weights()

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def select(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.num_arms, p=self._weights))

    def reset(self) -> None:
        self._cumulative_loss = np.zeros(self.num_arms, dtype=np.float64)
        self._cumulative_reward = np.zeros(self.num_arms, dtype=np.float64)
        self._weights = np.ones(self.num_arms, dtype=np.float64) / self.num_arms
        self._step = 0
        self._pull_counts = np.zeros(self.num_arms, dtype=np.float64)


# =========================================================================
# AdaptiveAnalyzer
# =========================================================================


class AdaptiveAnalyzer:
    """Analyze adaptive bandit / scheduling behavior.

    Computes arm selection frequencies, cumulative regret,
    convergence metrics, and performance summaries.
    """

    def __init__(self) -> None:
        self._selection_history: List[int] = []
        self._reward_history: List[float] = []
        self._arm_names: List[str] = []
        self._timestamps: List[float] = []
        self._best_reward_per_step: List[float] = []

    def record(
        self,
        arm: int,
        reward: float,
        best_possible_reward: Optional[float] = None,
        arm_names: Optional[List[str]] = None,
    ) -> None:
        """Record a single selection and its outcome."""
        self._selection_history.append(arm)
        self._reward_history.append(reward)
        self._timestamps.append(time.monotonic())
        if arm_names and not self._arm_names:
            self._arm_names = list(arm_names)
        if best_possible_reward is not None:
            self._best_reward_per_step.append(best_possible_reward)

    def arm_frequencies(self, window: Optional[int] = None) -> Dict[int, float]:
        """Compute selection frequency per arm, optionally in a recent window."""
        history = self._selection_history
        if window is not None and window > 0:
            history = history[-window:]
        if not history:
            return {}

        counts: Dict[int, int] = {}
        for arm in history:
            counts[arm] = counts.get(arm, 0) + 1
        total = len(history)
        return {arm: count / total for arm, count in sorted(counts.items())}

    def named_frequencies(self, window: Optional[int] = None) -> Dict[str, float]:
        """Arm frequencies using algorithm names."""
        freqs = self.arm_frequencies(window)
        if not self._arm_names:
            return {str(k): v for k, v in freqs.items()}
        return {self._arm_names[k] if k < len(self._arm_names) else str(k): v for k, v in freqs.items()}

    def cumulative_regret(self) -> np.ndarray:
        """Compute cumulative regret over time.

        Regret_t = sum_{s=1}^{t} (best_reward_s - actual_reward_s).
        If best_possible_reward was not recorded, estimates it as the
        maximum observed reward so far (optimistic estimate).
        """
        if not self._reward_history:
            return np.array([], dtype=np.float64)

        t = len(self._reward_history)
        regret = np.zeros(t, dtype=np.float64)
        running_max = -float("inf")

        for i in range(t):
            running_max = max(running_max, self._reward_history[i])
            if i < len(self._best_reward_per_step):
                best = self._best_reward_per_step[i]
            else:
                best = running_max
            instant_regret = max(0.0, best - self._reward_history[i])
            regret[i] = (regret[i - 1] if i > 0 else 0.0) + instant_regret

        return regret

    def average_regret(self) -> float:
        """Average per-step regret."""
        regret = self.cumulative_regret()
        if len(regret) == 0:
            return 0.0
        return float(regret[-1] / len(regret))

    def convergence_analysis(self, window: int = 20) -> Dict[str, Any]:
        """Analyze whether the algorithm selection has converged.

        Convergence is measured by the stability of arm selection
        frequencies and the trend in recent rewards.
        """
        result: Dict[str, Any] = {
            "total_steps": len(self._selection_history),
            "converged": False,
            "dominant_arm": -1,
            "dominant_frequency": 0.0,
            "reward_trend": 0.0,
            "reward_variance": 0.0,
            "frequency_stability": 0.0,
        }

        if len(self._selection_history) < 2 * window:
            return result

        # Check if recent selections are dominated by one arm
        recent_freqs = self.arm_frequencies(window)
        if recent_freqs:
            dominant = max(recent_freqs, key=lambda k: recent_freqs[k])
            result["dominant_arm"] = dominant
            result["dominant_frequency"] = recent_freqs[dominant]

        # Reward trend: linear regression on recent rewards
        recent_rewards = self._reward_history[-window:]
        if len(recent_rewards) >= 2:
            x = np.arange(len(recent_rewards), dtype=np.float64)
            y = np.array(recent_rewards, dtype=np.float64)
            x_mean = x.mean()
            y_mean = y.mean()
            ss_xy = np.sum((x - x_mean) * (y - y_mean))
            ss_xx = np.sum((x - x_mean) ** 2)
            if ss_xx > 0:
                slope = ss_xy / ss_xx
                result["reward_trend"] = float(slope)
            result["reward_variance"] = float(np.var(y))

        # Frequency stability: compare first half vs second half of recent window
        half = window // 2
        first_half = self._selection_history[-(2 * half) : -half]
        second_half = self._selection_history[-half:]
        if first_half and second_half:
            freq1 = self._compute_freq_vector(first_half)
            freq2 = self._compute_freq_vector(second_half)
            # L1 distance between frequency vectors
            max_arm = max(max(first_half), max(second_half)) + 1
            v1 = np.zeros(max_arm)
            v2 = np.zeros(max_arm)
            for a, f in freq1.items():
                if a < max_arm:
                    v1[a] = f
            for a, f in freq2.items():
                if a < max_arm:
                    v2[a] = f
            stability = 1.0 - np.sum(np.abs(v1 - v2)) / 2.0
            result["frequency_stability"] = float(stability)

        # Convergence heuristic: high dominant frequency + high stability + low variance
        result["converged"] = (
            result["dominant_frequency"] > 0.6
            and result["frequency_stability"] > 0.7
            and result["reward_variance"] < 0.05
        )

        return result

    def _compute_freq_vector(self, history: List[int]) -> Dict[int, float]:
        counts: Dict[int, int] = {}
        for arm in history:
            counts[arm] = counts.get(arm, 0) + 1
        total = len(history)
        return {arm: count / total for arm, count in counts.items()}

    def reward_summary(self) -> Dict[str, float]:
        """Summary statistics over all rewards."""
        if not self._reward_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
        arr = np.array(self._reward_history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "total": float(np.sum(arr)),
        }

    def per_arm_summary(self) -> Dict[int, Dict[str, float]]:
        """Per-arm reward statistics."""
        arm_rewards: Dict[int, List[float]] = {}
        for arm, reward in zip(self._selection_history, self._reward_history):
            arm_rewards.setdefault(arm, []).append(reward)

        result: Dict[int, Dict[str, float]] = {}
        for arm, rewards in arm_rewards.items():
            arr = np.array(rewards)
            result[arm] = {
                "count": len(rewards),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return result

    def moving_average_rewards(self, window: int = 10) -> np.ndarray:
        """Compute moving average of rewards."""
        if not self._reward_history:
            return np.array([], dtype=np.float64)
        rewards = np.array(self._reward_history, dtype=np.float64)
        if len(rewards) < window:
            return np.array([np.mean(rewards)], dtype=np.float64)
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(rewards, kernel, mode="valid")

    def reset(self) -> None:
        """Clear all recorded data."""
        self._selection_history.clear()
        self._reward_history.clear()
        self._arm_names.clear()
        self._timestamps.clear()
        self._best_reward_per_step.clear()


# =========================================================================
# AdaptiveDecoding
# =========================================================================


class AdaptiveDecoding(DecodingAlgorithm):
    """Adaptive decoding that selects algorithms per-step.

    At each generation step, uses a bandit or heuristic to decide
    which algorithm's sampling strategy to apply for the next token.
    Supports entropy-based, progress-based, and quality-based adaptation.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.adaptation_strategy = params.get("adaptation_strategy", "entropy")
        self.num_candidate_algorithms = params.get("num_candidate_algorithms", 3)
        self.use_bandit = params.get("use_bandit", True)
        self.bandit_strategy = params.get("bandit_strategy", "ucb1")
        self.exploration_constant = params.get("exploration_constant", 2.0)
        self.entropy_threshold_low = params.get("entropy_threshold_low", 1.0)
        self.entropy_threshold_high = params.get("entropy_threshold_high", 3.0)
        self.quality_threshold = params.get("quality_threshold", -5.0)
        self.temperature_conservative = params.get("temperature_conservative", 0.7)
        self.temperature_exploratory = params.get("temperature_exploratory", 1.3)
        self.temperature_moderate = params.get("temperature_moderate", 1.0)
        self.top_k_conservative = params.get("top_k_conservative", 10)
        self.top_k_exploratory = params.get("top_k_exploratory", 50)
        self.top_p_conservative = params.get("top_p_conservative", 0.85)
        self.top_p_exploratory = params.get("top_p_exploratory", 0.95)
        self.warmup_steps = params.get("warmup_steps", 5)
        self.progress_phases = params.get("progress_phases", 3)

        # Algorithm pool: each "algorithm" is a parameterised sampling config
        self._algo_configs = self._build_algorithm_pool()
        algo_names = [f"algo_{i}" for i in range(len(self._algo_configs))]

        if self.use_bandit:
            self._selector = BanditAlgorithmSelector(
                algorithm_names=algo_names,
                strategy=self.bandit_strategy,
                reward_function=DiversityReward(),
                exploration_constant=self.exploration_constant,
                seed=config.seed,
            )
        else:
            self._selector = None

        self._analyzer = AdaptiveAnalyzer()
        self._step_entropies: List[float] = []
        self._step_algo_choices: List[int] = []

    def _build_algorithm_pool(self) -> List[Dict[str, Any]]:
        """Build a pool of parameterised sampling configurations."""
        pool: List[Dict[str, Any]] = []

        # Conservative (greedy-ish)
        pool.append({
            "name": "conservative",
            "temperature": self.temperature_conservative,
            "top_k": self.top_k_conservative,
            "top_p": self.top_p_conservative,
        })

        # Moderate
        pool.append({
            "name": "moderate",
            "temperature": self.temperature_moderate,
            "top_k": 30,
            "top_p": 0.9,
        })

        # Exploratory
        pool.append({
            "name": "exploratory",
            "temperature": self.temperature_exploratory,
            "top_k": self.top_k_exploratory,
            "top_p": self.top_p_exploratory,
        })

        # Add more if requested
        if self.num_candidate_algorithms > 3:
            pool.append({
                "name": "very_conservative",
                "temperature": 0.5,
                "top_k": 5,
                "top_p": 0.7,
            })
        if self.num_candidate_algorithms > 4:
            pool.append({
                "name": "very_exploratory",
                "temperature": 1.5,
                "top_k": 100,
                "top_p": 0.98,
            })
        if self.num_candidate_algorithms > 5:
            pool.append({
                "name": "nucleus_focused",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 0.92,
            })

        return pool[:self.num_candidate_algorithms]

    @property
    def description(self) -> str:
        return f"Adaptive decoding ({self.adaptation_strategy})"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute one adaptive decoding step."""
        active = state.active_indices()
        if not active:
            return state

        # Get logits for all active sequences
        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            # Select algorithm for this token
            algo_idx = self._select_algorithm(
                row_logits, state, seq_idx, batch_idx
            )
            algo_config = self._algo_configs[algo_idx]

            # Apply the selected algorithm's sampling parameters
            token = sample_token(
                row_logits,
                temperature=algo_config["temperature"],
                top_k=algo_config["top_k"],
                top_p=algo_config["top_p"],
            )

            state.update_sequence(seq_idx, token)

            # Check EOS
            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            # Update score with log probability
            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

            self._step_algo_choices.append(algo_idx)

            # Update bandit reward
            if self._selector is not None and state.step > 0:
                reward = self._compute_step_reward(state, seq_idx, token)
                self._selector.update(algo_idx, reward)
                self._analyzer.record(algo_idx, reward)

        # Check max length
        for i in active:
            gen_len = len(state.sequences[i]) - len(state.sequences[i])  # approximate
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _select_algorithm(
        self,
        logits: np.ndarray,
        state: DecodingState,
        seq_idx: int,
        batch_idx: int,
    ) -> int:
        """Select which algorithm config to use based on adaptation strategy."""
        if self.adaptation_strategy == "entropy":
            return self._entropy_adaptive_select(logits, state)
        elif self.adaptation_strategy == "progress":
            return self._progress_adaptive_select(state)
        elif self.adaptation_strategy == "quality":
            return self._quality_adaptive_select(state, seq_idx)
        elif self.adaptation_strategy == "bandit":
            if self._selector is not None:
                return self._selector.select(step=state.step)
            return 0
        elif self.adaptation_strategy == "hybrid":
            return self._hybrid_adaptive_select(logits, state, seq_idx)
        else:
            return 0

    def _entropy_adaptive_select(self, logits: np.ndarray, state: DecodingState) -> int:
        """Select algorithm based on logit entropy.

        Low entropy → model is confident → use conservative sampling.
        High entropy → model is uncertain → use exploratory sampling.
        """
        probs = _stable_softmax(logits)
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        self._step_entropies.append(entropy)

        n = len(self._algo_configs)
        if entropy < self.entropy_threshold_low:
            return 0  # conservative
        elif entropy > self.entropy_threshold_high:
            return min(n - 1, 2)  # exploratory
        else:
            return min(n - 1, 1)  # moderate

    def _progress_adaptive_select(self, state: DecodingState) -> int:
        """Select algorithm based on generation progress.

        Early steps → exploratory (build diversity).
        Middle steps → moderate.
        Late steps → conservative (ensure quality).
        """
        progress = state.step / max(self.config.max_new_tokens, 1)
        n = len(self._algo_configs)
        phase_size = 1.0 / max(self.progress_phases, 1)

        if progress < phase_size:
            # Early: exploratory
            return min(n - 1, 2)
        elif progress < 2 * phase_size:
            # Middle: moderate
            return min(n - 1, 1)
        else:
            # Late: conservative
            return 0

    def _quality_adaptive_select(self, state: DecodingState, seq_idx: int) -> int:
        """Select algorithm based on sequence quality.

        If quality (log prob per token) drops below threshold,
        switch to more conservative sampling.
        """
        n = len(self._algo_configs)
        seq_len = len(state.sequences[seq_idx])
        if seq_len == 0:
            return 1  # moderate as default

        avg_log_prob = state.scores[seq_idx] / max(seq_len, 1)

        if avg_log_prob < self.quality_threshold:
            return 0  # conservative to recover quality
        elif avg_log_prob > self.quality_threshold * 0.5:
            return min(n - 1, 2)  # quality is fine, explore
        else:
            return min(n - 1, 1)  # moderate

    def _hybrid_adaptive_select(
        self, logits: np.ndarray, state: DecodingState, seq_idx: int
    ) -> int:
        """Combine entropy and progress signals for hybrid selection."""
        probs = _stable_softmax(logits)
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        self._step_entropies.append(entropy)

        progress = state.step / max(self.config.max_new_tokens, 1)
        n = len(self._algo_configs)

        # Compute a score for each algo based on entropy and progress
        scores = np.zeros(n, dtype=np.float64)

        # Conservative score: high when entropy is low or progress is late
        scores[0] = (
            (1.0 - min(entropy / self.entropy_threshold_high, 1.0)) * 0.6
            + progress * 0.4
        )

        # Moderate score: high in middle entropy/progress
        if n > 1:
            mid_entropy = (self.entropy_threshold_low + self.entropy_threshold_high) / 2
            entropy_mid_score = 1.0 - abs(entropy - mid_entropy) / max(mid_entropy, 1.0)
            progress_mid_score = 1.0 - abs(progress - 0.5) * 2.0
            scores[1] = entropy_mid_score * 0.5 + max(progress_mid_score, 0.0) * 0.5

        # Exploratory score: high when entropy is high or progress is early
        if n > 2:
            scores[2] = (
                min(entropy / self.entropy_threshold_high, 1.0) * 0.6
                + (1.0 - progress) * 0.4
            )

        # Bandit-informed selection if available
        if self._selector is not None and self._selector.state.total_pulls > self.warmup_steps:
            means = self._selector.state.mean_rewards()
            combined = scores + 0.3 * means[:n]
            return int(np.argmax(combined))

        return int(np.argmax(scores))

    def _compute_step_reward(
        self, state: DecodingState, seq_idx: int, token: int
    ) -> float:
        """Compute reward for this generation step.

        Uses token-level diversity as a quick reward signal.
        """
        # Check how different this token is from other sequences' last tokens
        other_last_tokens = []
        for i, seq in enumerate(state.sequences):
            if i != seq_idx and len(seq) > 0:
                other_last_tokens.append(seq[-1])

        if not other_last_tokens:
            return 0.5

        # Reward is proportion of sequences with different last token
        different = sum(1 for t in other_last_tokens if t != token)
        diversity_reward = different / len(other_last_tokens)

        # Small quality bonus for high-probability tokens
        return float(np.clip(diversity_reward, 0.0, 1.0))

    def validate_config(self) -> List[str]:
        errors = []
        if self.config.num_sequences < 1:
            errors.append("num_sequences must be >= 1")
        if self.config.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1")
        return errors

    def get_analysis(self) -> Dict[str, Any]:
        """Return analysis of the adaptive behavior."""
        result: Dict[str, Any] = {
            "adaptation_strategy": self.adaptation_strategy,
            "num_algos": len(self._algo_configs),
            "total_steps": len(self._step_algo_choices),
        }
        if self._step_algo_choices:
            choices = np.array(self._step_algo_choices)
            for i, cfg in enumerate(self._algo_configs):
                count = int(np.sum(choices == i))
                result[f"algo_{i}_{cfg['name']}_count"] = count
                result[f"algo_{i}_{cfg['name']}_freq"] = count / len(choices)

        if self._step_entropies:
            ent = np.array(self._step_entropies)
            result["mean_entropy"] = float(np.mean(ent))
            result["std_entropy"] = float(np.std(ent))

        if self._selector is not None:
            result["bandit_stats"] = self._selector.get_arm_statistics()

        result["convergence"] = self._analyzer.convergence_analysis()
        result["reward_summary"] = self._analyzer.reward_summary()

        return result


# =========================================================================
# EntropyAdaptiveDecoding
# =========================================================================


class EntropyAdaptiveDecoding(DecodingAlgorithm):
    """Adaptive decoding that uses logit entropy to switch between
    conservative and exploratory sampling.

    When the model distribution has low entropy (confident), uses
    lower temperature / tighter nucleus. When entropy is high,
    increases exploration to promote diversity.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params
        self.low_threshold = params.get("entropy_low", 1.0)
        self.high_threshold = params.get("entropy_high", 3.5)
        self.temp_low = params.get("temp_low", 0.6)
        self.temp_mid = params.get("temp_mid", 1.0)
        self.temp_high = params.get("temp_high", 1.4)
        self.top_p_low = params.get("top_p_low", 0.8)
        self.top_p_mid = params.get("top_p_mid", 0.9)
        self.top_p_high = params.get("top_p_high", 0.95)
        self.smoothing_window = params.get("smoothing_window", 5)
        self._entropy_buffer: List[float] = []
        self._regime_history: List[str] = []

    @property
    def description(self) -> str:
        return "Entropy-adaptive decoding"

    def _compute_entropy(self, logits: np.ndarray) -> float:
        """Compute Shannon entropy from logits."""
        probs = _stable_softmax(logits)
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        return entropy

    def _smoothed_entropy(self, new_entropy: float) -> float:
        """Exponentially smoothed entropy for stability."""
        self._entropy_buffer.append(new_entropy)
        if len(self._entropy_buffer) <= 1:
            return new_entropy
        window = self._entropy_buffer[-self.smoothing_window :]
        weights = np.exp(np.linspace(-1, 0, len(window)))
        weights /= weights.sum()
        return float(np.dot(weights, window))

    def _get_regime(self, entropy: float) -> str:
        """Determine sampling regime from entropy value."""
        if entropy < self.low_threshold:
            return "conservative"
        elif entropy > self.high_threshold:
            return "exploratory"
        else:
            return "moderate"

    def _get_temperature(self, regime: str, entropy: float) -> float:
        """Get temperature for current regime with smooth interpolation."""
        if regime == "conservative":
            # Interpolate between temp_low and temp_mid
            t = entropy / max(self.low_threshold, 1e-6)
            return self.temp_low + (self.temp_mid - self.temp_low) * min(t, 1.0)
        elif regime == "exploratory":
            # Interpolate between temp_mid and temp_high
            range_width = max(self.high_threshold, self.low_threshold + 0.1) - self.low_threshold
            t = min((entropy - self.high_threshold) / max(range_width, 1e-6) + 0.5, 1.0)
            return self.temp_mid + (self.temp_high - self.temp_mid) * t
        else:
            return self.temp_mid

    def _get_top_p(self, regime: str) -> float:
        """Get nucleus threshold for current regime."""
        if regime == "conservative":
            return self.top_p_low
        elif regime == "exploratory":
            return self.top_p_high
        else:
            return self.top_p_mid

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]
            entropy = self._compute_entropy(row_logits)
            smoothed = self._smoothed_entropy(entropy)
            regime = self._get_regime(smoothed)
            self._regime_history.append(regime)

            temperature = self._get_temperature(regime, smoothed)
            top_p = self._get_top_p(regime)

            token = sample_token(row_logits, temperature=temperature, top_p=top_p)
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def validate_config(self) -> List[str]:
        errors = []
        if self.low_threshold >= self.high_threshold:
            errors.append("entropy_low must be < entropy_high")
        return errors

    def get_regime_summary(self) -> Dict[str, float]:
        """Return fraction of steps spent in each regime."""
        if not self._regime_history:
            return {}
        total = len(self._regime_history)
        counts: Dict[str, int] = {}
        for r in self._regime_history:
            counts[r] = counts.get(r, 0) + 1
        return {k: v / total for k, v in counts.items()}


# =========================================================================
# ProgressAdaptiveDecoding
# =========================================================================


class ProgressAdaptiveDecoding(DecodingAlgorithm):
    """Vary sampling algorithm based on position in the generated sequence.

    Divides generation into phases (e.g. beginning, middle, end) and
    applies different sampling strategies in each phase.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.num_phases = params.get("num_phases", 3)
        # Phase configs: list of (temperature, top_k, top_p) tuples
        self.phase_configs = params.get("phase_configs", None)
        if self.phase_configs is None:
            self.phase_configs = [
                {"temperature": 1.3, "top_k": 50, "top_p": 0.95},  # early: explore
                {"temperature": 1.0, "top_k": 30, "top_p": 0.9},   # middle: balanced
                {"temperature": 0.7, "top_k": 10, "top_p": 0.85},  # late: conservative
            ]

        self.transition_type = params.get("transition_type", "smooth")
        self._phase_history: List[int] = []

    @property
    def description(self) -> str:
        return "Progress-adaptive decoding"

    def _get_phase(self, step: int, max_steps: int) -> int:
        """Determine current phase index."""
        if max_steps <= 0:
            return 0
        progress = step / max_steps
        phase = int(progress * self.num_phases)
        return min(phase, self.num_phases - 1)

    def _interpolate_params(
        self, step: int, max_steps: int
    ) -> Dict[str, float]:
        """Get smoothly interpolated parameters between phases."""
        if max_steps <= 0 or len(self.phase_configs) < 2:
            return self.phase_configs[0] if self.phase_configs else {"temperature": 1.0, "top_k": 0, "top_p": 1.0}

        progress = step / max_steps
        # Map progress to position in phase array
        pos = progress * (len(self.phase_configs) - 1)
        lower = int(pos)
        upper = min(lower + 1, len(self.phase_configs) - 1)
        frac = pos - lower

        if self.transition_type == "smooth":
            # Cosine interpolation
            t = (1.0 - math.cos(math.pi * frac)) / 2.0
        elif self.transition_type == "linear":
            t = frac
        else:
            t = 0.0 if frac < 0.5 else 1.0  # step function

        low_cfg = self.phase_configs[lower]
        up_cfg = self.phase_configs[upper]

        return {
            "temperature": low_cfg["temperature"] * (1 - t) + up_cfg["temperature"] * t,
            "top_k": int(low_cfg["top_k"] * (1 - t) + up_cfg["top_k"] * t),
            "top_p": low_cfg["top_p"] * (1 - t) + up_cfg["top_p"] * t,
        }

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        max_steps = self.config.max_new_tokens
        phase = self._get_phase(state.step, max_steps)
        self._phase_history.append(phase)

        if self.transition_type in ("smooth", "linear"):
            params = self._interpolate_params(state.step, max_steps)
        else:
            params = self.phase_configs[min(phase, len(self.phase_configs) - 1)]

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            token = sample_token(
                row_logits,
                temperature=params["temperature"],
                top_k=int(params["top_k"]),
                top_p=params["top_p"],
            )
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        for i in active:
            if state.step >= max_steps:
                state.mark_finished(i)

        state.step += 1
        return state

    def validate_config(self) -> List[str]:
        errors = []
        if self.num_phases < 1:
            errors.append("num_phases must be >= 1")
        if not self.phase_configs:
            errors.append("phase_configs must not be empty")
        return errors


# =========================================================================
# QualityAdaptiveDecoding
# =========================================================================


class QualityAdaptiveDecoding(DecodingAlgorithm):
    """Switch strategy when sequence quality drops below a threshold.

    Monitors the running average log probability per token. When it
    drops below a threshold, switches to conservative sampling to
    recover quality. When it is above, uses exploratory sampling.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.quality_threshold = params.get("quality_threshold", -5.0)
        self.recovery_threshold = params.get("recovery_threshold", -3.0)
        self.conservative_temp = params.get("conservative_temp", 0.6)
        self.exploratory_temp = params.get("exploratory_temp", 1.2)
        self.conservative_top_p = params.get("conservative_top_p", 0.8)
        self.exploratory_top_p = params.get("exploratory_top_p", 0.95)
        self.ema_alpha = params.get("ema_alpha", 0.1)
        self._quality_ema: Dict[int, float] = {}
        self._mode: Dict[int, str] = {}
        self._mode_history: List[str] = []

    @property
    def description(self) -> str:
        return "Quality-adaptive decoding"

    def _update_quality(self, seq_idx: int, log_prob: float) -> float:
        """Update exponential moving average of quality for sequence."""
        if seq_idx not in self._quality_ema:
            self._quality_ema[seq_idx] = log_prob
        else:
            self._quality_ema[seq_idx] = (
                self.ema_alpha * log_prob
                + (1.0 - self.ema_alpha) * self._quality_ema[seq_idx]
            )
        return self._quality_ema[seq_idx]

    def _get_mode(self, seq_idx: int, quality: float) -> str:
        """Determine current mode with hysteresis."""
        current = self._mode.get(seq_idx, "exploratory")

        if current == "exploratory" and quality < self.quality_threshold:
            self._mode[seq_idx] = "conservative"
            return "conservative"
        elif current == "conservative" and quality > self.recovery_threshold:
            self._mode[seq_idx] = "exploratory"
            return "exploratory"

        return current

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            # Determine mode based on quality
            seq_len = len(state.sequences[seq_idx])
            avg_quality = state.scores[seq_idx] / max(seq_len, 1)
            quality = self._update_quality(seq_idx, avg_quality)
            mode = self._get_mode(seq_idx, quality)
            self._mode_history.append(mode)

            if mode == "conservative":
                temperature = self.conservative_temp
                top_p = self.conservative_top_p
            else:
                temperature = self.exploratory_temp
                top_p = self.exploratory_top_p

            token = sample_token(row_logits, temperature=temperature, top_p=top_p)
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def validate_config(self) -> List[str]:
        errors = []
        if self.quality_threshold > self.recovery_threshold:
            errors.append("quality_threshold should be <= recovery_threshold")
        return errors

    def get_mode_summary(self) -> Dict[str, float]:
        """Return fraction of steps in each mode."""
        if not self._mode_history:
            return {}
        total = len(self._mode_history)
        counts: Dict[str, int] = {}
        for m in self._mode_history:
            counts[m] = counts.get(m, 0) + 1
        return {k: v / total for k, v in counts.items()}


# =========================================================================
# BanditGuidedDecoding
# =========================================================================


class BanditGuidedDecoding(DecodingAlgorithm):
    """Full bandit-guided decoding using BanditAlgorithmSelector.

    Maintains a pool of decoding configurations and uses a bandit
    to select among them at each step, with proper reward feedback.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.bandit_strategy = params.get("bandit_strategy", "thompson")
        self.reward_type = params.get("reward_type", "diversity")
        self.exploration = params.get("exploration_constant", 2.0)
        self.gamma = params.get("gamma", 0.1)
        self.update_frequency = params.get("update_frequency", 1)

        # Build algorithm pool
        self._algo_pool = params.get("algo_pool", None)
        if self._algo_pool is None:
            self._algo_pool = [
                {"name": "greedy", "temperature": 0.3, "top_k": 5, "top_p": 0.7},
                {"name": "standard", "temperature": 1.0, "top_k": 0, "top_p": 0.9},
                {"name": "creative", "temperature": 1.2, "top_k": 40, "top_p": 0.95},
                {"name": "wild", "temperature": 1.5, "top_k": 80, "top_p": 0.98},
            ]

        algo_names = [a["name"] for a in self._algo_pool]

        # Set up reward function
        if self.reward_type == "diversity":
            reward_fn: RewardFunction = DiversityReward()
        elif self.reward_type == "quality_diversity":
            reward_fn = QualityDiversityReward(alpha=params.get("qd_alpha", 0.5))
        elif self.reward_type == "curriculum":
            reward_fn = CurriculumReward(
                total_steps=config.max_new_tokens,
                schedule=params.get("curriculum_schedule", "cosine"),
            )
        else:
            reward_fn = DiversityReward()

        self._selector = BanditAlgorithmSelector(
            algorithm_names=algo_names,
            strategy=self.bandit_strategy,
            reward_function=reward_fn,
            exploration_constant=self.exploration,
            gamma=self.gamma,
            seed=config.seed,
        )

        self._analyzer = AdaptiveAnalyzer()
        self._step_selections: List[int] = []
        self._pending_tokens: Dict[int, List[int]] = {}

    @property
    def description(self) -> str:
        return f"Bandit-guided decoding ({self.bandit_strategy})"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            # Select algorithm
            arm = self._selector.select(step=state.step)
            algo_cfg = self._algo_pool[arm]
            self._step_selections.append(arm)

            token = sample_token(
                row_logits,
                temperature=algo_cfg["temperature"],
                top_k=algo_cfg.get("top_k", 0),
                top_p=algo_cfg.get("top_p", 1.0),
            )
            state.update_sequence(seq_idx, token)

            # Track tokens for diversity reward
            if seq_idx not in self._pending_tokens:
                self._pending_tokens[seq_idx] = []
            self._pending_tokens[seq_idx].append(token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

            # Periodic reward update
            if state.step % self.update_frequency == 0 and state.step > 0:
                reward = self._compute_reward(state, seq_idx)
                self._selector.update(arm, reward)
                self._analyzer.record(arm, reward, arm_names=[a["name"] for a in self._algo_pool])

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _compute_reward(self, state: DecodingState, seq_idx: int) -> float:
        """Compute reward based on diversity of recent tokens."""
        other_seqs = [
            state.sequences[i]
            for i in range(len(state.sequences))
            if i != seq_idx and len(state.sequences[i]) > 0
        ]
        current_seq = state.sequences[seq_idx]
        if not other_seqs or not current_seq:
            return 0.5

        return self._selector.compute_reward(other_seqs, current_seq)

    def validate_config(self) -> List[str]:
        errors = []
        if not self._algo_pool:
            errors.append("algo_pool must not be empty")
        return errors

    def get_analysis(self) -> Dict[str, Any]:
        result = self._selector.get_arm_statistics()
        result["convergence"] = self._analyzer.convergence_analysis()
        result["reward_summary"] = self._analyzer.reward_summary()
        result["arm_frequencies"] = self._analyzer.named_frequencies()
        return result


# =========================================================================
# MetaBanditDecoding
# =========================================================================


class MetaBanditDecoding(DecodingAlgorithm):
    """Two-level bandit: outer level selects strategy, inner level selects parameters.

    The outer bandit chooses which type of adaptation to use
    (entropy, progress, quality), and the inner bandit fine-tunes
    the parameters within that adaptation type.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.outer_strategy = params.get("outer_strategy", "ucb1")
        self.inner_strategy = params.get("inner_strategy", "thompson")
        self.outer_exploration = params.get("outer_exploration", 2.0)
        self.inner_exploration = params.get("inner_exploration", 1.5)
        self.meta_update_interval = params.get("meta_update_interval", 5)

        # Outer bandit: selects among adaptation strategies
        outer_arms = ["entropy", "progress", "quality", "uniform"]
        self._outer_selector = BanditAlgorithmSelector(
            algorithm_names=outer_arms,
            strategy=self.outer_strategy,
            reward_function=DiversityReward(),
            exploration_constant=self.outer_exploration,
            seed=config.seed,
        )

        # Inner bandits: one per strategy, selects among parameter configs
        self._inner_configs = {
            "entropy": [
                {"temperature": 0.5, "top_p": 0.8},
                {"temperature": 0.8, "top_p": 0.85},
                {"temperature": 1.0, "top_p": 0.9},
                {"temperature": 1.3, "top_p": 0.95},
            ],
            "progress": [
                {"temperature": 0.7, "top_p": 0.85},
                {"temperature": 1.0, "top_p": 0.9},
                {"temperature": 1.2, "top_p": 0.92},
            ],
            "quality": [
                {"temperature": 0.6, "top_p": 0.8},
                {"temperature": 0.9, "top_p": 0.88},
                {"temperature": 1.1, "top_p": 0.92},
            ],
            "uniform": [
                {"temperature": 1.0, "top_p": 0.9},
            ],
        }

        self._inner_selectors: Dict[str, BanditAlgorithmSelector] = {}
        for strat, configs in self._inner_configs.items():
            names = [f"{strat}_{i}" for i in range(len(configs))]
            seed_offset = hash(strat) % 10000
            self._inner_selectors[strat] = BanditAlgorithmSelector(
                algorithm_names=names,
                strategy=self.inner_strategy,
                reward_function=DiversityReward(),
                exploration_constant=self.inner_exploration,
                seed=(config.seed + seed_offset) if config.seed is not None else None,
            )

        self._outer_choices: List[str] = []
        self._inner_choices: List[int] = []
        self._analyzer = AdaptiveAnalyzer()

    @property
    def description(self) -> str:
        return "Meta-bandit decoding"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        # Outer selection
        outer_arm = self._outer_selector.select(step=state.step)
        outer_name = self._outer_selector.algorithm_names[outer_arm]
        self._outer_choices.append(outer_name)

        # Inner selection
        inner_selector = self._inner_selectors[outer_name]
        inner_arm = inner_selector.select(step=state.step)
        inner_configs = self._inner_configs[outer_name]
        chosen_config = inner_configs[inner_arm]
        self._inner_choices.append(inner_arm)

        # Strategy-specific parameter adjustment
        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            temperature = chosen_config["temperature"]
            top_p = chosen_config["top_p"]

            # Strategy-specific modulation
            if outer_name == "entropy":
                probs = _stable_softmax(row_logits)
                entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                # Scale temperature by entropy
                entropy_scale = min(entropy / 3.0, 2.0)
                temperature *= (0.5 + 0.5 * entropy_scale)

            elif outer_name == "progress":
                progress = state.step / max(self.config.max_new_tokens, 1)
                # Early = more exploratory, late = more conservative
                temperature *= (1.3 - 0.6 * progress)

            elif outer_name == "quality":
                seq_len = len(state.sequences[seq_idx])
                avg_lp = state.scores[seq_idx] / max(seq_len, 1)
                if avg_lp < -5.0:
                    temperature *= 0.7  # more conservative

            token = sample_token(row_logits, temperature=max(temperature, 0.1), top_p=top_p)
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        # Periodic reward update
        if state.step % self.meta_update_interval == 0 and state.step > 0:
            reward = self._compute_aggregate_reward(state)
            self._outer_selector.update(outer_arm, reward)
            inner_selector.update(inner_arm, reward)
            self._analyzer.record(outer_arm, reward)

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _compute_aggregate_reward(self, state: DecodingState) -> float:
        """Compute aggregate diversity reward across all sequences."""
        seqs = [s for s in state.sequences if len(s) > 0]
        if len(seqs) < 2:
            return 0.5

        # Pairwise diversity using recent tokens
        distances = []
        for i in range(min(len(seqs), 10)):
            for j in range(i + 1, min(len(seqs), 10)):
                tail_i = set(seqs[i][-15:])
                tail_j = set(seqs[j][-15:])
                union = len(tail_i | tail_j)
                if union > 0:
                    distances.append(1.0 - len(tail_i & tail_j) / union)
        if distances:
            return sum(distances) / len(distances)
        return 0.5

    def validate_config(self) -> List[str]:
        return []


# =========================================================================
# EnsembleAdaptiveDecoding
# =========================================================================


class EnsembleAdaptiveDecoding(DecodingAlgorithm):
    """Ensemble adaptive decoding: uses an online learner to maintain
    a weighted mixture over multiple sampling strategies.

    At each step, samples a strategy according to current weights,
    generates a token, and updates weights using the observed reward.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.learner_type = params.get("learner_type", "exponential")
        self.learning_rate = params.get("learning_rate", 0.1)

        self._algo_pool = params.get("algo_pool", [
            {"name": "conservative", "temperature": 0.6, "top_k": 10, "top_p": 0.85},
            {"name": "moderate", "temperature": 1.0, "top_k": 30, "top_p": 0.9},
            {"name": "exploratory", "temperature": 1.3, "top_k": 50, "top_p": 0.95},
            {"name": "diverse", "temperature": 1.1, "top_k": 0, "top_p": 0.92},
        ])

        n = len(self._algo_pool)

        if self.learner_type == "exponential":
            self._learner: OnlineLearner = ExponentialWeightUpdate(
                n, learning_rate=self.learning_rate
            )
        elif self.learner_type == "mirror_descent":
            self._learner = MirrorDescent(n, learning_rate=self.learning_rate)
        elif self.learner_type == "ftrl":
            self._learner = FollowTheRegularizedLeader(
                n, regularization=params.get("regularization", 1.0)
            )
        else:
            self._learner = ExponentialWeightUpdate(n, learning_rate=self.learning_rate)

        self._rng_local = np.random.default_rng(config.seed)
        self._selection_history: List[int] = []
        self._reward_history: List[float] = []
        self._weight_history: List[np.ndarray] = []

    @property
    def description(self) -> str:
        return f"Ensemble adaptive ({self.learner_type})"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        # Record current weights
        self._weight_history.append(self._learner.get_weights())

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            # Select algorithm using online learner
            arm = self._learner.select(self._rng_local)
            algo_cfg = self._algo_pool[arm]
            self._selection_history.append(arm)

            token = sample_token(
                row_logits,
                temperature=algo_cfg["temperature"],
                top_k=algo_cfg.get("top_k", 0),
                top_p=algo_cfg.get("top_p", 1.0),
            )
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            token_log_prob = float(log_probs[token])
            state.scores[seq_idx] += token_log_prob

            # Compute reward: token-level diversity + quality
            reward = self._token_reward(state, seq_idx, token, token_log_prob)
            self._learner.update(arm, reward)
            self._reward_history.append(reward)

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _token_reward(
        self, state: DecodingState, seq_idx: int, token: int, log_prob: float
    ) -> float:
        """Token-level reward combining diversity and quality."""
        # Diversity: how different is this token from others' last tokens
        other_last = [
            s[-1] for i, s in enumerate(state.sequences) if i != seq_idx and len(s) > 0
        ]
        if other_last:
            diff_count = sum(1 for t in other_last if t != token)
            diversity = diff_count / len(other_last)
        else:
            diversity = 0.5

        # Quality: normalised log prob
        quality = 1.0 / (1.0 + math.exp(-log_prob - 2.0))

        return float(np.clip(0.4 * quality + 0.6 * diversity, 0.0, 1.0))

    def validate_config(self) -> List[str]:
        return []

    def get_weight_trajectory(self) -> List[np.ndarray]:
        """Return the history of weight vectors."""
        return list(self._weight_history)


# =========================================================================
# AdaptiveDecodingWithScheduler
# =========================================================================


class AdaptiveDecodingWithScheduler(DecodingAlgorithm):
    """Adaptive decoding driven by an explicit scheduler.

    Combines a scheduler (warmup, cyclic, decay) with an algorithm
    pool to manage algorithm selection throughout generation.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.scheduler_type = params.get("scheduler_type", "warmup")
        self._algo_pool = params.get("algo_pool", [
            {"name": "conservative", "temperature": 0.7, "top_k": 10, "top_p": 0.85},
            {"name": "moderate", "temperature": 1.0, "top_k": 0, "top_p": 0.9},
            {"name": "exploratory", "temperature": 1.3, "top_k": 50, "top_p": 0.95},
        ])

        n = len(self._algo_pool)

        if self.scheduler_type == "warmup":
            self._scheduler: AdaptiveScheduler = WarmupScheduler(
                num_arms=n,
                warmup_arm=params.get("warmup_arm", 0),
                warmup_steps=params.get("warmup_steps", 10),
                post_warmup_strategy=params.get("post_warmup_strategy", "weighted"),
                seed=config.seed,
            )
        elif self.scheduler_type == "cyclic":
            self._scheduler = CyclicScheduler(
                num_arms=n,
                steps_per_arm=params.get("steps_per_arm", None),
                cycle_order=params.get("cycle_order", None),
            )
        elif self.scheduler_type == "decay":
            self._scheduler = DecayScheduler(
                num_arms=n,
                decay_rate=params.get("decay_rate", 0.01),
                decay_type=params.get("decay_type", "exponential"),
                min_weight=params.get("min_weight", 0.01),
                seed=config.seed,
            )
        else:
            self._scheduler = WarmupScheduler(n, seed=config.seed)

        self._analyzer = AdaptiveAnalyzer()
        self._diversity_reward = DiversityReward()

    @property
    def description(self) -> str:
        return f"Scheduled adaptive decoding ({self.scheduler_type})"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        # Get algorithm from scheduler
        arm = self._scheduler.select(state.step, self.config.max_new_tokens)
        algo_cfg = self._algo_pool[min(arm, len(self._algo_pool) - 1)]

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            token = sample_token(
                row_logits,
                temperature=algo_cfg["temperature"],
                top_k=algo_cfg.get("top_k", 0),
                top_p=algo_cfg.get("top_p", 1.0),
            )
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        # Compute reward and update scheduler
        if state.step > 0 and len(state.sequences) > 1:
            seqs = [s for s in state.sequences if len(s) > 0]
            if len(seqs) >= 2:
                reward = self._diversity_reward.compute(
                    seqs[:-1], seqs[-1]
                )
            else:
                reward = 0.5
        else:
            reward = 0.5

        self._scheduler.update(state.step, arm, reward)
        self._analyzer.record(arm, reward)

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def validate_config(self) -> List[str]:
        return []


# =========================================================================
# ContextualAdaptiveDecoding
# =========================================================================


class ContextualAdaptiveDecoding(DecodingAlgorithm):
    """Contextual bandit-guided decoding that uses sequence features.

    Uses LinUCB to select algorithms based on features extracted
    from the current generation state (entropy, progress, diversity, etc).
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.feature_dim = params.get("feature_dim", 12)
        self.alpha = params.get("linucb_alpha", 1.0)

        self._algo_pool = params.get("algo_pool", [
            {"name": "greedy", "temperature": 0.4, "top_k": 5, "top_p": 0.75},
            {"name": "standard", "temperature": 1.0, "top_k": 0, "top_p": 0.9},
            {"name": "diverse", "temperature": 1.2, "top_k": 40, "top_p": 0.95},
            {"name": "wild", "temperature": 1.5, "top_k": 80, "top_p": 0.98},
        ])

        algo_names = [a["name"] for a in self._algo_pool]
        self._contextual_state = ContextualBanditState.create(
            algo_names, self.feature_dim, alpha=self.alpha
        )
        self._ctx_strategy = ContextualBanditStrategy(alpha=self.alpha)
        self._feature_extractor = SequenceFeatureExtractor(self.feature_dim)
        self._rng_local = np.random.default_rng(config.seed)
        self._analyzer = AdaptiveAnalyzer()

    @property
    def description(self) -> str:
        return "Contextual adaptive decoding (LinUCB)"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        # Extract features from current state
        context = self._feature_extractor.extract(
            state.sequences, state.step, self.config.max_new_tokens, logits
        )

        # Select algorithm using contextual bandit
        arm = self._ctx_strategy.select_arm(
            self._contextual_state, context, self._rng_local
        )
        algo_cfg = self._algo_pool[arm]

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            token = sample_token(
                row_logits,
                temperature=algo_cfg["temperature"],
                top_k=algo_cfg.get("top_k", 0),
                top_p=algo_cfg.get("top_p", 1.0),
            )
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        # Compute reward and update contextual model
        reward = self._compute_step_diversity(state)
        self._ctx_strategy.update(self._contextual_state, arm, context, reward)
        self._analyzer.record(arm, reward)

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _compute_step_diversity(self, state: DecodingState) -> float:
        """Compute diversity of last tokens across active sequences."""
        active = state.active_indices()
        if len(active) < 2:
            return 0.5

        last_tokens = []
        for i in active:
            if len(state.sequences[i]) > 0:
                last_tokens.append(state.sequences[i][-1])
        if not last_tokens:
            return 0.5

        unique = len(set(last_tokens))
        return unique / len(last_tokens)

    def validate_config(self) -> List[str]:
        return []


# =========================================================================
# AdaptiveMixtureDecoding
# =========================================================================


class AdaptiveMixtureDecoding(DecodingAlgorithm):
    """Mixture-of-experts style decoding: combines logits from multiple
    temperature/sampling configurations using learned weights.

    Instead of selecting a single algorithm, mixes the probability
    distributions from multiple configurations.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.temperatures = params.get("temperatures", [0.5, 0.8, 1.0, 1.3])
        self.initial_weights = params.get(
            "initial_weights",
            [1.0 / len(self.temperatures)] * len(self.temperatures),
        )
        self.adaptation_rate = params.get("adaptation_rate", 0.05)
        self.diversity_target = params.get("diversity_target", 0.7)

        n = len(self.temperatures)
        self._weights = np.array(self.initial_weights, dtype=np.float64)
        self._weights /= self._weights.sum()
        self._learner = ExponentialWeightUpdate(n, learning_rate=self.adaptation_rate)
        # Sync initial weights
        self._learner._weights = self._weights.copy()
        self._rng_local = np.random.default_rng(config.seed)
        self._weight_history: List[np.ndarray] = []
        self._diversity_history: List[float] = []

    @property
    def description(self) -> str:
        return "Adaptive mixture decoding"

    def _mix_distributions(
        self, logits: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Create a mixture distribution from multiple temperature settings."""
        probs_list = []
        for temp in self.temperatures:
            scaled = logits / max(temp, 1e-6)
            probs_list.append(_stable_softmax(scaled))

        # Weighted mixture of probability distributions
        mixed = np.zeros_like(probs_list[0])
        for w, p in zip(weights, probs_list):
            mixed += w * p

        # Ensure valid distribution
        mixed = np.maximum(mixed, 0.0)
        total = mixed.sum()
        if total > 0 and np.isfinite(total):
            mixed /= total
        else:
            mixed = np.ones_like(mixed) / len(mixed)

        return mixed

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        weights = self._learner.get_weights()
        self._weight_history.append(weights.copy())

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            # Create mixture distribution
            mixed_probs = self._mix_distributions(row_logits, weights)

            # Sample from mixture
            total = mixed_probs.sum()
            if total > 0 and np.isfinite(total):
                mixed_probs /= total
                token = int(self._rng_local.choice(len(mixed_probs), p=mixed_probs))
            else:
                token = int(np.argmax(row_logits))

            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        # Update mixture weights based on diversity feedback
        diversity = self._compute_current_diversity(state)
        self._diversity_history.append(diversity)

        # Reward temperatures that would increase diversity when we're below target,
        # or maintain quality when we're above target
        for i, temp in enumerate(self.temperatures):
            if diversity < self.diversity_target:
                # Prefer higher temperatures
                reward = min(temp / max(self.temperatures), 1.0)
            else:
                # Prefer lower temperatures for quality
                reward = 1.0 - min(temp / max(self.temperatures), 1.0)
            self._learner.update(i, reward * diversity)

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _compute_current_diversity(self, state: DecodingState) -> float:
        """Compute current diversity across all sequences."""
        seqs = [s for s in state.sequences if len(s) > 0]
        if len(seqs) < 2:
            return 0.5

        tail_len = min(10, min(len(s) for s in seqs))
        distances = []
        for i in range(min(len(seqs), 10)):
            for j in range(i + 1, min(len(seqs), 10)):
                tail_i = set(seqs[i][-tail_len:])
                tail_j = set(seqs[j][-tail_len:])
                union = len(tail_i | tail_j)
                if union > 0:
                    distances.append(1.0 - len(tail_i & tail_j) / union)
        return sum(distances) / len(distances) if distances else 0.5

    def validate_config(self) -> List[str]:
        errors = []
        if not self.temperatures:
            errors.append("temperatures must not be empty")
        return errors


# =========================================================================
# RegretMinimizingDecoding
# =========================================================================


class RegretMinimizingDecoding(DecodingAlgorithm):
    """Decoding that explicitly minimises cumulative regret.

    Uses FTRL or other regret minimisation algorithms to select
    among decoding configurations, with the goal of minimising
    the difference between achieved diversity and the best
    fixed algorithm in hindsight.
    """

    def __init__(self, config: DecodingConfig) -> None:
        super().__init__(config)
        params = config.params

        self.regret_algorithm = params.get("regret_algorithm", "ftrl")
        self.regularization = params.get("regularization", 1.0)

        self._algo_pool = params.get("algo_pool", [
            {"name": "temp_0.5", "temperature": 0.5, "top_p": 0.8},
            {"name": "temp_0.8", "temperature": 0.8, "top_p": 0.88},
            {"name": "temp_1.0", "temperature": 1.0, "top_p": 0.9},
            {"name": "temp_1.2", "temperature": 1.2, "top_p": 0.93},
            {"name": "temp_1.5", "temperature": 1.5, "top_p": 0.96},
        ])

        n = len(self._algo_pool)

        if self.regret_algorithm == "ftrl":
            self._learner: OnlineLearner = FollowTheRegularizedLeader(
                n, regularization=self.regularization
            )
        elif self.regret_algorithm == "mirror_descent":
            self._learner = MirrorDescent(n, learning_rate=params.get("learning_rate", 0.05))
        elif self.regret_algorithm == "exponential":
            self._learner = ExponentialWeightUpdate(n, learning_rate=params.get("learning_rate", 0.1))
        else:
            self._learner = FollowTheRegularizedLeader(n, regularization=self.regularization)

        self._rng_local = np.random.default_rng(config.seed)
        self._all_rewards: List[List[float]] = [[] for _ in range(n)]
        self._cumulative_regret: List[float] = []
        self._best_fixed_reward = 0.0

    @property
    def description(self) -> str:
        return f"Regret-minimizing decoding ({self.regret_algorithm})"

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        active = state.active_indices()
        if not active:
            return state

        active_seqs = [state.sequences[i] for i in active]
        logits = logit_source(active_seqs)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        arm = self._learner.select(self._rng_local)
        algo_cfg = self._algo_pool[arm]

        for batch_idx, seq_idx in enumerate(active):
            row_logits = logits[batch_idx]

            token = sample_token(
                row_logits,
                temperature=algo_cfg["temperature"],
                top_p=algo_cfg.get("top_p", 1.0),
            )
            state.update_sequence(seq_idx, token)

            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                if state.step >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            log_probs = _log_softmax(row_logits)
            state.scores[seq_idx] += float(log_probs[token])

        # Estimate reward for chosen arm
        reward = self._estimate_reward(state)
        self._all_rewards[arm].append(reward)
        self._learner.update(arm, reward)

        # Track cumulative regret
        # Compute best fixed arm reward so far
        arm_means = []
        for r_list in self._all_rewards:
            if r_list:
                arm_means.append(sum(r_list) / len(r_list))
            else:
                arm_means.append(0.0)
        self._best_fixed_reward = max(arm_means) if arm_means else 0.0

        total_actual = sum(sum(r) for r in self._all_rewards)
        total_steps = sum(len(r) for r in self._all_rewards)
        best_total = self._best_fixed_reward * total_steps
        regret = max(0.0, best_total - total_actual)
        self._cumulative_regret.append(regret)

        for i in active:
            if state.step >= self.config.max_new_tokens:
                state.mark_finished(i)

        state.step += 1
        return state

    def _estimate_reward(self, state: DecodingState) -> float:
        """Estimate diversity reward from current state."""
        seqs = [s for s in state.sequences if len(s) > 0]
        if len(seqs) < 2:
            return 0.5

        distances = []
        for i in range(min(len(seqs), 8)):
            for j in range(i + 1, min(len(seqs), 8)):
                tail_len = min(10, len(seqs[i]), len(seqs[j]))
                si = set(seqs[i][-tail_len:])
                sj = set(seqs[j][-tail_len:])
                union = len(si | sj)
                if union > 0:
                    distances.append(1.0 - len(si & sj) / union)
        return sum(distances) / len(distances) if distances else 0.5

    def validate_config(self) -> List[str]:
        return []

    def get_cumulative_regret(self) -> List[float]:
        return list(self._cumulative_regret)


# =========================================================================
# Utility functions
# =========================================================================


def create_adaptive_decoder(
    strategy: str = "entropy",
    config: Optional[DecodingConfig] = None,
    **kwargs: Any,
) -> DecodingAlgorithm:
    """Factory function to create adaptive decoders by strategy name.

    Parameters
    ----------
    strategy : str
        One of: "entropy", "progress", "quality", "bandit", "meta",
        "ensemble", "scheduled", "contextual", "mixture", "regret".
    config : DecodingConfig, optional
        Base configuration. If None, creates a default.
    **kwargs :
        Additional parameters merged into config.params.

    Returns
    -------
    DecodingAlgorithm
        The configured adaptive decoder.
    """
    if config is None:
        config = DecodingConfig(algorithm_name=f"adaptive_{strategy}")

    config.params.update(kwargs)

    strategy_map: Dict[str, Type[DecodingAlgorithm]] = {
        "entropy": EntropyAdaptiveDecoding,
        "progress": ProgressAdaptiveDecoding,
        "quality": QualityAdaptiveDecoding,
        "bandit": BanditGuidedDecoding,
        "meta": MetaBanditDecoding,
        "ensemble": EnsembleAdaptiveDecoding,
        "scheduled": AdaptiveDecodingWithScheduler,
        "contextual": ContextualAdaptiveDecoding,
        "mixture": AdaptiveMixtureDecoding,
        "regret": RegretMinimizingDecoding,
        "adaptive": AdaptiveDecoding,
    }

    decoder_cls = strategy_map.get(strategy)
    if decoder_cls is None:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {sorted(strategy_map.keys())}"
        )

    return decoder_cls(config)


def run_bandit_experiment(
    algorithm_names: List[str],
    reward_fn: Callable[[int], float],
    num_rounds: int = 1000,
    strategy: str = "ucb1",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a standalone bandit experiment (useful for testing / ablation).

    Parameters
    ----------
    algorithm_names : list of str
        Names of arms.
    reward_fn : callable
        Function mapping arm index to a stochastic reward.
    num_rounds : int
        Number of rounds to simulate.
    strategy : str
        Bandit strategy to use.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Experiment results including per-arm stats and regret.
    """
    selector = BanditAlgorithmSelector(
        algorithm_names=algorithm_names,
        strategy=strategy,
        seed=seed,
    )
    analyzer = AdaptiveAnalyzer()

    for t in range(num_rounds):
        arm = selector.select(step=t)
        reward = reward_fn(arm)
        selector.update(arm, reward)
        analyzer.record(arm, reward, arm_names=algorithm_names)

    return {
        "arm_statistics": selector.get_arm_statistics(),
        "best_arm": selector.best_arm(),
        "cumulative_regret": analyzer.cumulative_regret().tolist(),
        "arm_frequencies": analyzer.named_frequencies(),
        "convergence": analyzer.convergence_analysis(),
        "reward_summary": analyzer.reward_summary(),
        "per_arm_summary": analyzer.per_arm_summary(),
    }


def compare_strategies(
    algorithm_names: List[str],
    reward_fn: Callable[[int], float],
    num_rounds: int = 1000,
    strategies: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple bandit strategies on the same reward function.

    Returns per-strategy experiment results for comparison.
    """
    if strategies is None:
        strategies = ["ucb1", "thompson", "exp3"]

    results: Dict[str, Dict[str, Any]] = {}
    for strat in strategies:
        results[strat] = run_bandit_experiment(
            algorithm_names=algorithm_names,
            reward_fn=reward_fn,
            num_rounds=num_rounds,
            strategy=strat,
            seed=seed,
        )
    return results
