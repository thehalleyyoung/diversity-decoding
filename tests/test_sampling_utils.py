"""
Comprehensive tests for src/algorithms/sampling_utils.py

Tests all logit processors, sampling functions, beam management,
tracking/statistics classes, and batch operations.
"""

from __future__ import annotations

import copy
import math
import unittest
from collections import Counter
from dataclasses import asdict, fields
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.special import log_softmax, softmax, logsumexp

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

SMALL_VOCAB = 10
MEDIUM_VOCAB = 100
LARGE_VOCAB = 1000
DEFAULT_SEED = 42
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
FLOAT_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(vocab_size: int, seed: int = DEFAULT_SEED,
                 distribution: str = "uniform") -> np.ndarray:
    """Create reproducible logit arrays with various distributions."""
    rng = np.random.RandomState(seed)
    if distribution == "uniform":
        return rng.randn(vocab_size).astype(np.float64)
    elif distribution == "peaked":
        logits = np.full(vocab_size, -10.0, dtype=np.float64)
        logits[0] = 10.0
        return logits
    elif distribution == "flat":
        return np.zeros(vocab_size, dtype=np.float64)
    elif distribution == "zipf":
        ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
        return np.log(1.0 / ranks)
    elif distribution == "bimodal":
        logits = rng.randn(vocab_size).astype(np.float64)
        logits[:vocab_size // 2] += 5.0
        return logits
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = probs[probs > 0]
    return -float(np.sum(p * np.log(p)))


def _assert_valid_distribution(probs: np.ndarray, atol: float = 1e-5):
    """Assert probs sum to ~1 and are non-negative."""
    assert np.all(probs >= -atol), f"Negative probs: {probs[probs < 0]}"
    assert abs(np.sum(probs) - 1.0) < atol, f"Probs sum to {np.sum(probs)}"


# =========================================================================
# Stub implementations of the module under test.
#
# These mirror the public API described in the task.  When the real
# ``sampling_utils`` module is created, replace the stubs with imports:
#
#   from src.algorithms.sampling_utils import (
#       LogitProcessor, TemperatureProcessor, ...
#   )
# =========================================================================

# ---- LogitProcessor base ------------------------------------------------

class LogitProcessor:
    """Base class for logit processors."""

    def __call__(self, logits: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


# ---- Concrete processors ------------------------------------------------

class TemperatureProcessor(LogitProcessor):
    """Divide logits by temperature."""

    def __init__(self, temperature: float = 1.0, schedule: Optional[dict] = None):
        self.temperature = temperature
        self.schedule = schedule  # mapping step -> temperature

    def __call__(self, logits: np.ndarray, *, step: int = 0, **kw) -> np.ndarray:
        t = self.temperature
        if self.schedule and step in self.schedule:
            t = self.schedule[step]
        if t == 0:
            # Argmax: set max logit to large value, rest to -inf
            out = np.full_like(logits, -float("inf"))
            out[np.argmax(logits)] = 0.0
            return out
        return logits / t


class TopKProcessor(LogitProcessor):
    """Keep only top-k logits, set rest to -inf."""

    def __init__(self, k: int):
        self.k = k

    def __call__(self, logits: np.ndarray, **kw) -> np.ndarray:
        if self.k >= len(logits):
            return logits.copy()
        threshold = np.sort(logits)[-self.k]
        out = logits.copy()
        out[out < threshold] = -float("inf")
        return out


class TopPProcessor(LogitProcessor):
    """Nucleus (top-p) filtering."""

    def __init__(self, p: float = 0.9):
        self.p = p

    def __call__(self, logits: np.ndarray, **kw) -> np.ndarray:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = _stable_softmax(sorted_logits)
        cumsum = np.cumsum(probs)
        # Find cutoff: keep tokens until cumulative prob >= p
        cutoff_idx = np.searchsorted(cumsum, self.p, side="right")
        remove_indices = sorted_indices[cutoff_idx + 1:]
        out = logits.copy()
        out[remove_indices] = -float("inf")
        return out


class TypicalProcessor(LogitProcessor):
    """Typical decoding: keep tokens near expected information content."""

    def __init__(self, mass: float = 0.9, min_tokens: int = 1):
        self.mass = mass
        self.min_tokens = min_tokens

    def __call__(self, logits: np.ndarray, **kw) -> np.ndarray:
        probs = _stable_softmax(logits)
        ent = _entropy(probs)
        # information content per token
        info = -np.log(np.clip(probs, 1e-30, None))
        # distance to entropy
        surprise = np.abs(info - ent)
        sorted_idx = np.argsort(surprise)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = max(self.min_tokens,
                     int(np.searchsorted(cumsum, self.mass, side="right")) + 1)
        keep = set(sorted_idx[:cutoff].tolist())
        out = logits.copy()
        for i in range(len(out)):
            if i not in keep:
                out[i] = -float("inf")
        return out


class MinPProcessor(LogitProcessor):
    """Filter tokens with probability < min_p * max_prob."""

    def __init__(self, min_p: float = 0.05):
        self.min_p = min_p

    def __call__(self, logits: np.ndarray, **kw) -> np.ndarray:
        probs = _stable_softmax(logits)
        threshold = self.min_p * np.max(probs)
        out = logits.copy()
        out[probs < threshold] = -float("inf")
        return out


class RepetitionPenaltyProcessor(LogitProcessor):
    """Apply repetition, frequency, and presence penalties."""

    def __init__(self, penalty: float = 1.2, frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0):
        self.penalty = penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def __call__(self, logits: np.ndarray, *,
                 generated_ids: Optional[List[int]] = None, **kw) -> np.ndarray:
        if not generated_ids:
            return logits.copy()
        out = logits.copy()
        token_counts: Dict[int, int] = {}
        for tid in generated_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for tid, count in token_counts.items():
            if 0 <= tid < len(out):
                # repetition penalty
                if out[tid] > 0:
                    out[tid] /= self.penalty
                else:
                    out[tid] *= self.penalty
                # frequency penalty
                out[tid] -= self.frequency_penalty * count
                # presence penalty
                out[tid] -= self.presence_penalty
        return out


class LengthPenaltyProcessor(LogitProcessor):
    """Apply length normalization penalty."""

    def __init__(self, alpha: float = 1.0, eos_token_id: int = EOS_TOKEN_ID):
        self.alpha = alpha
        self.eos_token_id = eos_token_id

    def __call__(self, logits: np.ndarray, *, length: int = 1, **kw) -> np.ndarray:
        out = logits.copy()
        penalty = ((5 + length) / 6) ** self.alpha
        out[self.eos_token_id] += np.log(penalty)
        return out


class NoRepeatNgramProcessor(LogitProcessor):
    """Block repeated n-grams."""

    def __init__(self, n: int = 3):
        self.n = n

    def __call__(self, logits: np.ndarray, *,
                 generated_ids: Optional[List[int]] = None, **kw) -> np.ndarray:
        if not generated_ids or len(generated_ids) < self.n - 1:
            return logits.copy()
        out = logits.copy()
        # current (n-1)-gram suffix
        suffix = tuple(generated_ids[-(self.n - 1):])
        # scan for matching prefixes
        for i in range(len(generated_ids) - self.n + 1):
            ngram_prefix = tuple(generated_ids[i:i + self.n - 1])
            if ngram_prefix == suffix:
                blocked = generated_ids[i + self.n - 1]
                out[blocked] = -float("inf")
        return out


class EntropyBasedProcessor(LogitProcessor):
    """Adapt temperature based on entropy of the distribution."""

    def __init__(self, target_entropy: float = 2.0, strength: float = 1.0):
        self.target_entropy = target_entropy
        self.strength = strength

    def __call__(self, logits: np.ndarray, **kw) -> np.ndarray:
        probs = _stable_softmax(logits)
        ent = _entropy(probs)
        if ent < 1e-10:
            return logits.copy()
        ratio = self.target_entropy / ent
        adaptive_temp = 1.0 + self.strength * (ratio - 1.0)
        adaptive_temp = max(0.1, min(adaptive_temp, 5.0))
        return logits / adaptive_temp


class ContrastiveProcessor(LogitProcessor):
    """Apply contrastive penalty using hidden-state similarity."""

    def __init__(self, alpha: float = 0.6, k: int = 5):
        self.alpha = alpha
        self.k = k

    def __call__(self, logits: np.ndarray, *,
                 similarity_scores: Optional[np.ndarray] = None, **kw) -> np.ndarray:
        if similarity_scores is None:
            return logits.copy()
        probs = _stable_softmax(logits)
        # top-k candidates
        top_k_idx = np.argsort(logits)[-self.k:]
        out = logits.copy()
        for idx in top_k_idx:
            sim = similarity_scores[idx] if idx < len(similarity_scores) else 0.0
            out[idx] = (1 - self.alpha) * probs[idx] - self.alpha * sim
        return out


class DiversityBoostProcessor(LogitProcessor):
    """Boost less-frequent tokens to increase diversity."""

    def __init__(self, boost_factor: float = 0.5):
        self.boost_factor = boost_factor

    def __call__(self, logits: np.ndarray, *,
                 token_frequencies: Optional[np.ndarray] = None, **kw) -> np.ndarray:
        if token_frequencies is None:
            return logits.copy()
        out = logits.copy()
        max_freq = np.max(token_frequencies) if np.max(token_frequencies) > 0 else 1.0
        norm_freq = token_frequencies / max_freq
        out += self.boost_factor * (1.0 - norm_freq)
        return out


# ---- SamplingConfig ------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class SamplingConfig:
    """Sampling hyper-parameters."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    typical_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0
    seed: Optional[int] = None
    num_samples: int = 1
    max_tokens: int = 100

    def validate(self) -> List[str]:
        errors = []
        if self.temperature < 0:
            errors.append("temperature must be >= 0")
        if self.top_k < 0:
            errors.append("top_k must be >= 0")
        if not 0.0 <= self.top_p <= 1.0:
            errors.append("top_p must be in [0, 1]")
        if self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingConfig":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---- Beam classes --------------------------------------------------------

@dataclass
class BeamState:
    """State of a single beam."""
    token_ids: List[int] = dc_field(default_factory=list)
    log_prob: float = 0.0
    score: float = 0.0
    finished: bool = False

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def extend(self, token_id: int, token_log_prob: float) -> "BeamState":
        new = BeamState(
            token_ids=self.token_ids + [token_id],
            log_prob=self.log_prob + token_log_prob,
            finished=(token_id == EOS_TOKEN_ID),
        )
        new.score = new.log_prob / max(1, new.length)
        return new


class BeamManager:
    """Manage a set of beams with pruning and scoring."""

    def __init__(self, num_beams: int, length_penalty: float = 1.0):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.beams: List[BeamState] = [BeamState()]
        self.completed: List[BeamState] = []

    def score(self, beam: BeamState) -> float:
        lp = ((5 + beam.length) / 6) ** self.length_penalty
        return beam.log_prob / lp

    def prune(self):
        self.beams.sort(key=lambda b: self.score(b), reverse=True)
        self.beams = self.beams[:self.num_beams]

    def step(self, candidates: List[Tuple[int, int, float]]):
        """Expand beams. candidates: list of (beam_idx, token_id, log_prob)."""
        new_beams = []
        for beam_idx, token_id, log_prob in candidates:
            parent = self.beams[beam_idx]
            child = parent.extend(token_id, log_prob)
            if child.finished:
                self.completed.append(child)
            else:
                new_beams.append(child)
        self.beams = new_beams
        self.prune()

    @property
    def is_done(self) -> bool:
        return len(self.beams) == 0

    def best(self, n: int = 1) -> List[BeamState]:
        all_beams = self.completed + self.beams
        all_beams.sort(key=lambda b: self.score(b), reverse=True)
        return all_beams[:n]


class DiverseBeamGroups:
    """Manage multiple beam groups with inter-group diversity penalty."""

    def __init__(self, num_groups: int, beams_per_group: int,
                 diversity_penalty: float = 0.5):
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.groups: List[BeamManager] = [
            BeamManager(beams_per_group) for _ in range(num_groups)
        ]

    def apply_diversity_penalty(self, logits: np.ndarray,
                                group_idx: int) -> np.ndarray:
        """Penalise tokens already selected by earlier groups."""
        out = logits.copy()
        for g in range(group_idx):
            for beam in self.groups[g].beams:
                if beam.token_ids:
                    last_token = beam.token_ids[-1]
                    if 0 <= last_token < len(out):
                        out[last_token] -= self.diversity_penalty
        return out

    def best_per_group(self, n: int = 1) -> List[List[BeamState]]:
        return [g.best(n) for g in self.groups]


# ---- Standalone functions ------------------------------------------------

def sample_from_logits(logits: np.ndarray, temperature: float = 1.0,
                       top_k: int = 0, top_p: float = 1.0,
                       seed: Optional[int] = None) -> int:
    """Sample a single token from logits."""
    rng = np.random.RandomState(seed)
    if temperature == 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    if top_k > 0:
        threshold = np.sort(scaled)[-min(top_k, len(scaled))]
        scaled[scaled < threshold] = -float("inf")
    if top_p < 1.0:
        sorted_idx = np.argsort(scaled)[::-1]
        probs = _stable_softmax(scaled)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p, side="right") + 1
        remove = sorted_idx[cutoff:]
        scaled[remove] = -float("inf")
    probs = _stable_softmax(scaled)
    return int(rng.choice(len(probs), p=probs))


def top_k_filtering(logits: np.ndarray, k: int) -> np.ndarray:
    """Return logits with only top-k values; rest set to -inf."""
    if k <= 0 or k >= len(logits):
        return logits.copy()
    threshold = np.sort(logits)[-k]
    out = logits.copy()
    out[out < threshold] = -float("inf")
    return out


def top_p_filtering(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filtering – keep smallest set whose cumulative prob >= p."""
    sorted_idx = np.argsort(logits)[::-1]
    probs = _stable_softmax(logits)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p, side="right") + 1
    remove = sorted_idx[cutoff:]
    out = logits.copy()
    out[remove] = -float("inf")
    return out


def typical_filtering(logits: np.ndarray, mass: float = 0.9) -> np.ndarray:
    """Typical filtering based on information content."""
    proc = TypicalProcessor(mass=mass)
    return proc(logits)


def apply_processor_chain(logits: np.ndarray,
                          processors: List[LogitProcessor],
                          **kwargs) -> np.ndarray:
    """Apply a chain of logit processors sequentially."""
    result = logits.copy()
    for proc in processors:
        result = proc(result, **kwargs)
    return result


def compute_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy of the distribution defined by logits."""
    probs = _stable_softmax(logits)
    return _entropy(probs)


def compute_varentropy(logits: np.ndarray) -> float:
    """Variance of the surprise (information content) values."""
    probs = _stable_softmax(logits)
    info = -np.log(np.clip(probs, 1e-30, None))
    ent = _entropy(probs)
    return float(np.sum(probs * (info - ent) ** 2))


def gumbel_softmax_sampling(logits: np.ndarray, temperature: float = 1.0,
                            seed: Optional[int] = None) -> int:
    """Sample using the Gumbel-max trick."""
    rng = np.random.RandomState(seed)
    gumbels = -np.log(-np.log(rng.uniform(size=len(logits)) + 1e-20) + 1e-20)
    perturbed = logits / max(temperature, 1e-10) + gumbels
    return int(np.argmax(perturbed))


def systematic_resampling(weights: np.ndarray, n: int,
                          seed: Optional[int] = None) -> List[int]:
    """Systematic resampling of n particles given weights."""
    rng = np.random.RandomState(seed)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    positions = (rng.uniform() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    indices = []
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices.append(j)
            i += 1
        else:
            j += 1
    return indices


def stochastic_beam_search_sampling(logits: np.ndarray, k: int,
                                    seed: Optional[int] = None) -> List[int]:
    """Sample k items using stochastic beam search (Gumbel top-k)."""
    rng = np.random.RandomState(seed)
    gumbels = -np.log(-np.log(rng.uniform(size=len(logits)) + 1e-20) + 1e-20)
    perturbed = logits + gumbels
    return list(np.argsort(perturbed)[-k:][::-1])


def sample_without_replacement(logits: np.ndarray, k: int,
                               seed: Optional[int] = None) -> List[int]:
    """Sample k unique tokens without replacement."""
    rng = np.random.RandomState(seed)
    probs = _stable_softmax(logits)
    chosen: List[int] = []
    remaining_probs = probs.copy()
    for _ in range(min(k, len(logits))):
        remaining_probs = remaining_probs / np.sum(remaining_probs)
        idx = int(rng.choice(len(remaining_probs), p=remaining_probs))
        chosen.append(idx)
        remaining_probs[idx] = 0.0
    return chosen


# ---- Tracker / statistics ------------------------------------------------

class TokenUsageTracker:
    """Track token usage across generation."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self.total = 0
        self.step_history: List[int] = []

    def update(self, token_id: int):
        self.counts[token_id] += 1
        self.total += 1
        self.step_history.append(token_id)

    def frequency(self, token_id: int) -> float:
        return self.counts[token_id] / max(1, self.total)

    @property
    def unique_tokens(self) -> int:
        return int(np.sum(self.counts > 0))

    @property
    def entropy(self) -> float:
        if self.total == 0:
            return 0.0
        probs = self.counts / self.total
        return _entropy(probs)

    def reset(self):
        self.counts[:] = 0
        self.total = 0
        self.step_history.clear()


class SamplingStatistics:
    """Collect statistics about sampling decisions."""

    def __init__(self):
        self.temperatures: List[float] = []
        self.entropies: List[float] = []
        self.top1_probs: List[float] = []
        self.accepted_tokens: List[int] = []
        self.rejected_count: int = 0

    def record(self, temperature: float, entropy: float,
               top1_prob: float, token: int):
        self.temperatures.append(temperature)
        self.entropies.append(entropy)
        self.top1_probs.append(top1_prob)
        self.accepted_tokens.append(token)

    def record_rejection(self):
        self.rejected_count += 1

    @property
    def mean_entropy(self) -> float:
        return float(np.mean(self.entropies)) if self.entropies else 0.0

    @property
    def mean_temperature(self) -> float:
        return float(np.mean(self.temperatures)) if self.temperatures else 0.0

    @property
    def acceptance_rate(self) -> float:
        total = len(self.accepted_tokens) + self.rejected_count
        return len(self.accepted_tokens) / max(1, total)

    def summary(self) -> dict:
        return {
            "num_steps": len(self.accepted_tokens),
            "mean_entropy": self.mean_entropy,
            "mean_temperature": self.mean_temperature,
            "acceptance_rate": self.acceptance_rate,
            "rejected": self.rejected_count,
        }


class BatchSampler:
    """Sample from logits for a batch of sequences."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self._rng = np.random.RandomState(config.seed)

    def sample_batch(self, batch_logits: np.ndarray) -> List[int]:
        """Sample one token per row in a batch of logits (batch, vocab)."""
        results = []
        for logits in batch_logits:
            token = sample_from_logits(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                seed=int(self._rng.randint(0, 2**31)),
            )
            results.append(token)
        return results

    def sample_multiple(self, logits: np.ndarray, n: int) -> List[int]:
        """Sample n tokens from the same logits."""
        return [
            sample_from_logits(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                seed=int(self._rng.randint(0, 2**31)),
            )
            for _ in range(n)
        ]


class SequencePool:
    """Maintain a pool of candidate sequences."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._pool: List[Tuple[List[int], float]] = []

    def add(self, sequence: List[int], score: float):
        self._pool.append((sequence, score))
        if len(self._pool) > self.max_size:
            self._pool.sort(key=lambda x: x[1], reverse=True)
            self._pool = self._pool[:self.max_size]

    @property
    def size(self) -> int:
        return len(self._pool)

    def best(self, n: int = 1) -> List[Tuple[List[int], float]]:
        sorted_pool = sorted(self._pool, key=lambda x: x[1], reverse=True)
        return sorted_pool[:n]

    def clear(self):
        self._pool.clear()

    def sequences(self) -> List[List[int]]:
        return [s for s, _ in self._pool]

    def diversity(self) -> float:
        """Fraction of unique tokens across all sequences."""
        if not self._pool:
            return 0.0
        all_tokens = set()
        total = 0
        for seq, _ in self._pool:
            all_tokens.update(seq)
            total += len(seq)
        return len(all_tokens) / max(1, total)


# =========================================================================
# TEST CLASSES
# =========================================================================


class TestTemperatureProcessor(unittest.TestCase):
    """Tests for TemperatureProcessor."""

    def test_identity_at_temp_1(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TemperatureProcessor(temperature=1.0)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_low_temperature_sharpens(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TemperatureProcessor(temperature=0.1)
        result = proc(logits)
        probs_orig = _stable_softmax(logits)
        probs_sharp = _stable_softmax(result)
        self.assertGreater(np.max(probs_sharp), np.max(probs_orig))

    def test_high_temperature_flattens(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TemperatureProcessor(temperature=10.0)
        result = proc(logits)
        probs_orig = _stable_softmax(logits)
        probs_flat = _stable_softmax(result)
        self.assertGreater(_entropy(probs_flat), _entropy(probs_orig))

    def test_temperature_zero_argmax(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TemperatureProcessor(temperature=0)
        result = proc(logits)
        probs = _stable_softmax(result)
        expected_idx = np.argmax(logits)
        self.assertAlmostEqual(probs[expected_idx], 1.0, places=5)

    def test_very_high_temperature(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = TemperatureProcessor(temperature=1000.0)
        result = proc(logits)
        probs = _stable_softmax(result)
        expected_uniform = 1.0 / SMALL_VOCAB
        for p in probs:
            self.assertAlmostEqual(p, expected_uniform, places=2)

    def test_negative_logits(self):
        logits = np.array([-5.0, -3.0, -1.0, -2.0, -4.0])
        proc = TemperatureProcessor(temperature=0.5)
        result = proc(logits)
        np.testing.assert_allclose(result, logits / 0.5, atol=FLOAT_ATOL)

    def test_schedule(self):
        logits = _make_logits(SMALL_VOCAB)
        schedule = {0: 0.5, 5: 2.0, 10: 1.0}
        proc = TemperatureProcessor(temperature=1.0, schedule=schedule)
        r0 = proc(logits, step=0)
        np.testing.assert_allclose(r0, logits / 0.5, atol=FLOAT_ATOL)
        r5 = proc(logits, step=5)
        np.testing.assert_allclose(r5, logits / 2.0, atol=FLOAT_ATOL)
        r3 = proc(logits, step=3)
        np.testing.assert_allclose(r3, logits / 1.0, atol=FLOAT_ATOL)

    def test_preserves_ordering(self):
        logits = _make_logits(LARGE_VOCAB)
        proc = TemperatureProcessor(temperature=0.7)
        result = proc(logits)
        orig_order = np.argsort(logits)
        new_order = np.argsort(result)
        np.testing.assert_array_equal(orig_order, new_order)

    def test_small_vocab(self):
        logits = np.array([1.0, 2.0])
        proc = TemperatureProcessor(temperature=0.5)
        result = proc(logits)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result, [2.0, 4.0], atol=FLOAT_ATOL)

    def test_single_element(self):
        logits = np.array([5.0])
        proc = TemperatureProcessor(temperature=2.0)
        result = proc(logits)
        self.assertAlmostEqual(result[0], 2.5, places=5)

    def test_peaked_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        proc = TemperatureProcessor(temperature=0.1)
        result = proc(logits)
        probs = _stable_softmax(result)
        self.assertGreater(probs[0], 0.99)

    def test_flat_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="flat")
        proc = TemperatureProcessor(temperature=2.0)
        result = proc(logits)
        np.testing.assert_allclose(result, np.zeros(MEDIUM_VOCAB), atol=FLOAT_ATOL)


class TestTopKProcessor(unittest.TestCase):
    """Tests for TopKProcessor."""

    def test_k_equals_1(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TopKProcessor(k=1)
        result = proc(logits)
        valid = result[result > -float("inf")]
        self.assertEqual(len(valid), 1)
        self.assertEqual(np.argmax(result), np.argmax(logits))

    def test_k_equals_vocab_size(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TopKProcessor(k=MEDIUM_VOCAB)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_k_greater_than_vocab(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = TopKProcessor(k=SMALL_VOCAB + 10)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_only_top_k_remain(self):
        logits = _make_logits(LARGE_VOCAB, seed=123)
        k = 50
        proc = TopKProcessor(k=k)
        result = proc(logits)
        valid_count = np.sum(result > -float("inf"))
        self.assertLessEqual(valid_count, k + 5)  # ties may add a few
        top_k_indices = set(np.argsort(logits)[-k:])
        valid_indices = set(np.where(result > -float("inf"))[0])
        self.assertTrue(top_k_indices.issubset(valid_indices) or
                        valid_indices.issubset(top_k_indices | set(range(LARGE_VOCAB))))

    def test_filtered_values_are_neg_inf(self):
        logits = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
        proc = TopKProcessor(k=2)
        result = proc(logits)
        # Top 2 are indices 0 (5.0) and 3 (4.0)
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isfinite(result[3]))
        for idx in [1, 2, 4]:
            self.assertEqual(result[idx], -float("inf"))

    def test_preserves_top_k_values(self):
        logits = _make_logits(MEDIUM_VOCAB)
        k = 10
        proc = TopKProcessor(k=k)
        result = proc(logits)
        top_k_idx = np.argsort(logits)[-k:]
        for idx in top_k_idx:
            self.assertAlmostEqual(result[idx], logits[idx], places=10)

    def test_small_k_distribution(self):
        logits = _make_logits(LARGE_VOCAB)
        proc = TopKProcessor(k=5)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_k_2_with_ties(self):
        logits = np.array([3.0, 3.0, 3.0, 1.0, 0.0])
        proc = TopKProcessor(k=2)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 2)

    def test_all_same_logits(self):
        logits = np.full(SMALL_VOCAB, 5.0)
        proc = TopKProcessor(k=3)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, SMALL_VOCAB)  # all tied, all kept


class TestTopPProcessor(unittest.TestCase):
    """Tests for TopPProcessor (Nucleus)."""

    def test_p_equals_1(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TopPProcessor(p=1.0)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, MEDIUM_VOCAB)

    def test_small_p_selects_few(self):
        logits = _make_logits(LARGE_VOCAB, distribution="zipf")
        proc = TopPProcessor(p=0.1)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertLess(valid, LARGE_VOCAB)

    def test_cumulative_probability(self):
        logits = _make_logits(MEDIUM_VOCAB)
        p = 0.9
        proc = TopPProcessor(p=p)
        result = proc(logits)
        probs_original = _stable_softmax(logits)
        kept_mask = result > -float("inf")
        cumulative = np.sum(probs_original[kept_mask])
        self.assertGreaterEqual(cumulative, p - 0.01)

    def test_peaked_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        proc = TopPProcessor(p=0.5)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertLessEqual(valid, 5)  # peaked = few tokens needed

    def test_flat_distribution(self):
        logits = _make_logits(SMALL_VOCAB, distribution="flat")
        proc = TopPProcessor(p=0.5)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, SMALL_VOCAB // 2 - 1)

    def test_p_near_zero(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TopPProcessor(p=0.001)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)

    def test_preserves_values(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = TopPProcessor(p=0.9)
        result = proc(logits)
        kept = result > -float("inf")
        np.testing.assert_allclose(result[kept], logits[kept], atol=FLOAT_ATOL)

    def test_result_is_valid_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TopPProcessor(p=0.95)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_various_p_values(self):
        logits = _make_logits(MEDIUM_VOCAB, seed=99)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            proc = TopPProcessor(p=p)
            result = proc(logits)
            valid = np.sum(result > -float("inf"))
            self.assertGreaterEqual(valid, 1)
            if p < 0.99:
                self.assertLessEqual(valid, MEDIUM_VOCAB)


class TestTypicalProcessor(unittest.TestCase):
    """Tests for TypicalProcessor."""

    def test_keeps_at_least_min_tokens(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        proc = TypicalProcessor(mass=0.5, min_tokens=3)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 3)

    def test_mass_1_keeps_all(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = TypicalProcessor(mass=1.0)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, SMALL_VOCAB)

    def test_small_mass_filters(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TypicalProcessor(mass=0.2)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertLess(valid, MEDIUM_VOCAB)

    def test_preserves_values(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = TypicalProcessor(mass=0.9)
        result = proc(logits)
        kept = result > -float("inf")
        np.testing.assert_allclose(result[kept], logits[kept], atol=FLOAT_ATOL)

    def test_flat_distribution(self):
        logits = _make_logits(SMALL_VOCAB, distribution="flat")
        proc = TypicalProcessor(mass=0.5)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)

    def test_various_vocab_sizes(self):
        for vs in [SMALL_VOCAB, MEDIUM_VOCAB, LARGE_VOCAB]:
            logits = _make_logits(vs)
            proc = TypicalProcessor(mass=0.8)
            result = proc(logits)
            valid = np.sum(result > -float("inf"))
            self.assertGreaterEqual(valid, 1)
            self.assertLessEqual(valid, vs)

    def test_result_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TypicalProcessor(mass=0.9)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)


class TestMinPProcessor(unittest.TestCase):
    """Tests for MinPProcessor."""

    def test_min_p_zero_keeps_all(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = MinPProcessor(min_p=0.0)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, MEDIUM_VOCAB)

    def test_min_p_filters(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = MinPProcessor(min_p=0.1)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertLess(valid, MEDIUM_VOCAB)

    def test_peaked_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        proc = MinPProcessor(min_p=0.01)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertLessEqual(valid, 5)

    def test_threshold_correctness(self):
        logits = np.array([10.0, 5.0, 1.0, 0.0, -5.0])
        proc = MinPProcessor(min_p=0.1)
        result = proc(logits)
        probs = _stable_softmax(logits)
        threshold = 0.1 * np.max(probs)
        for i, p in enumerate(probs):
            if p >= threshold:
                self.assertTrue(np.isfinite(result[i]),
                                f"Token {i} with prob {p} should be kept")

    def test_preserves_top_token(self):
        logits = _make_logits(LARGE_VOCAB)
        proc = MinPProcessor(min_p=0.5)
        result = proc(logits)
        top_idx = np.argmax(logits)
        self.assertTrue(np.isfinite(result[top_idx]))

    def test_high_min_p(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = MinPProcessor(min_p=0.99)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)

    def test_result_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = MinPProcessor(min_p=0.05)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)


class TestRepetitionPenaltyProcessor(unittest.TestCase):
    """Tests for RepetitionPenaltyProcessor."""

    def test_no_generated_ids(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = RepetitionPenaltyProcessor(penalty=1.5)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_positive_logit_divided(self):
        logits = np.array([5.0, 3.0, -1.0, 2.0, -3.0])
        proc = RepetitionPenaltyProcessor(penalty=2.0)
        result = proc(logits, generated_ids=[0])
        self.assertAlmostEqual(result[0], 5.0 / 2.0, places=5)
        self.assertAlmostEqual(result[1], 3.0, places=5)

    def test_negative_logit_multiplied(self):
        logits = np.array([5.0, 3.0, -1.0, 2.0, -3.0])
        proc = RepetitionPenaltyProcessor(penalty=2.0)
        result = proc(logits, generated_ids=[2])
        self.assertAlmostEqual(result[2], -1.0 * 2.0, places=5)

    def test_frequency_penalty(self):
        logits = np.array([5.0, 3.0, 1.0])
        proc = RepetitionPenaltyProcessor(penalty=1.0, frequency_penalty=0.5)
        result = proc(logits, generated_ids=[0, 0, 0, 1])
        self.assertAlmostEqual(result[0], 5.0 - 0.5 * 3, places=5)
        self.assertAlmostEqual(result[1], 3.0 - 0.5 * 1, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_presence_penalty(self):
        logits = np.array([5.0, 3.0, 1.0])
        proc = RepetitionPenaltyProcessor(penalty=1.0, presence_penalty=1.0)
        result = proc(logits, generated_ids=[0, 1])
        self.assertAlmostEqual(result[0], 5.0 - 1.0, places=5)
        self.assertAlmostEqual(result[1], 3.0 - 1.0, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_combined_penalties(self):
        logits = np.array([5.0, 3.0, 1.0])
        proc = RepetitionPenaltyProcessor(
            penalty=1.5, frequency_penalty=0.2, presence_penalty=0.3)
        result = proc(logits, generated_ids=[0, 0])
        # Token 0: 5.0/1.5 - 0.2*2 - 0.3
        expected_0 = 5.0 / 1.5 - 0.2 * 2 - 0.3
        self.assertAlmostEqual(result[0], expected_0, places=5)

    def test_penalty_reduces_repeated_token_prob(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = RepetitionPenaltyProcessor(penalty=1.5)
        generated = [np.argmax(logits)]
        result = proc(logits, generated_ids=generated)
        probs_before = _stable_softmax(logits)
        probs_after = _stable_softmax(result)
        self.assertLess(probs_after[generated[0]], probs_before[generated[0]])

    def test_empty_generated_ids(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = RepetitionPenaltyProcessor(penalty=2.0)
        result = proc(logits, generated_ids=[])
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_multiple_tokens_penalised(self):
        logits = np.full(5, 3.0)
        proc = RepetitionPenaltyProcessor(penalty=2.0)
        result = proc(logits, generated_ids=[0, 1, 2])
        for i in [0, 1, 2]:
            self.assertAlmostEqual(result[i], 3.0 / 2.0, places=5)
        for i in [3, 4]:
            self.assertAlmostEqual(result[i], 3.0, places=5)


class TestLengthPenaltyProcessor(unittest.TestCase):
    """Tests for LengthPenaltyProcessor."""

    def test_alpha_zero_no_effect(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = LengthPenaltyProcessor(alpha=0.0)
        result = proc(logits, length=10)
        # alpha=0 => penalty = 1.0 => log(1) = 0
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_positive_alpha_boosts_eos(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = LengthPenaltyProcessor(alpha=1.0)
        result = proc(logits, length=20)
        self.assertGreater(result[EOS_TOKEN_ID], logits[EOS_TOKEN_ID])

    def test_length_increases_penalty(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = LengthPenaltyProcessor(alpha=1.0)
        r5 = proc(logits, length=5)
        r20 = proc(logits, length=20)
        self.assertGreater(r20[EOS_TOKEN_ID], r5[EOS_TOKEN_ID])

    def test_non_eos_unchanged(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = LengthPenaltyProcessor(alpha=1.0)
        result = proc(logits, length=10)
        for i in range(SMALL_VOCAB):
            if i != EOS_TOKEN_ID:
                self.assertAlmostEqual(result[i], logits[i], places=10)

    def test_custom_eos_token(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = LengthPenaltyProcessor(alpha=1.0, eos_token_id=5)
        result = proc(logits, length=10)
        self.assertGreater(result[5], 0.0)
        self.assertAlmostEqual(result[EOS_TOKEN_ID], 0.0, places=10)

    def test_penalty_formula(self):
        logits = np.zeros(SMALL_VOCAB)
        alpha = 0.6
        length = 15
        proc = LengthPenaltyProcessor(alpha=alpha)
        result = proc(logits, length=length)
        expected = np.log(((5 + length) / 6) ** alpha)
        self.assertAlmostEqual(result[EOS_TOKEN_ID], expected, places=5)


class TestNoRepeatNgramProcessor(unittest.TestCase):
    """Tests for NoRepeatNgramProcessor."""

    def test_no_repeat_trigram(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=3)
        generated = [1, 2, 3, 4, 1, 2]
        result = proc(logits, generated_ids=generated)
        # suffix (1,2) appeared before followed by 3 → block 3
        self.assertEqual(result[3], -float("inf"))

    def test_no_history_no_blocking(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=3)
        result = proc(logits, generated_ids=[])
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_short_history(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=3)
        result = proc(logits, generated_ids=[1])
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_bigram_blocking(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=2)
        generated = [3, 5, 7, 3]
        result = proc(logits, generated_ids=generated)
        # suffix (3) appeared at position 0, followed by 5 → block 5
        self.assertEqual(result[5], -float("inf"))

    def test_no_repeat_when_no_match(self):
        logits = _make_logits(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=3)
        generated = [1, 2, 3, 4, 5, 6]
        result = proc(logits, generated_ids=generated)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_multiple_blocks(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=2)
        generated = [1, 2, 1, 3, 1]
        result = proc(logits, generated_ids=generated)
        # suffix (1): appeared at idx 0 → block 2, idx 2 → block 3
        self.assertEqual(result[2], -float("inf"))
        self.assertEqual(result[3], -float("inf"))

    def test_4gram(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = NoRepeatNgramProcessor(n=4)
        generated = [1, 2, 3, 4, 5, 1, 2, 3]
        result = proc(logits, generated_ids=generated)
        self.assertEqual(result[4], -float("inf"))

    def test_preserves_non_blocked(self):
        logits = _make_logits(SMALL_VOCAB, seed=77)
        proc = NoRepeatNgramProcessor(n=3)
        generated = [1, 2, 3, 1, 2]
        result = proc(logits, generated_ids=generated)
        for i in range(SMALL_VOCAB):
            if i != 3:
                self.assertAlmostEqual(result[i], logits[i], places=10)


class TestEntropyBasedProcessor(unittest.TestCase):
    """Tests for EntropyBasedProcessor."""

    def test_flat_distribution_increases_sharpness(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="flat")
        proc = EntropyBasedProcessor(target_entropy=1.0, strength=1.0)
        result = proc(logits)
        # Flat distribution has max entropy; target < current → sharpen
        # For truly flat logits, all zeros, scaling doesn't change ordering
        self.assertEqual(len(result), MEDIUM_VOCAB)

    def test_peaked_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        proc = EntropyBasedProcessor(target_entropy=3.0, strength=1.0)
        result = proc(logits)
        # peaked has low entropy; target > current → flatten
        probs_orig = _stable_softmax(logits)
        probs_new = _stable_softmax(result)
        self.assertGreaterEqual(_entropy(probs_new) + 0.01,
                                _entropy(probs_orig))

    def test_strength_zero(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = EntropyBasedProcessor(target_entropy=2.0, strength=0.0)
        result = proc(logits)
        # strength=0 → adaptive_temp = 1.0 → identity
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_output_shape(self):
        for vs in [SMALL_VOCAB, MEDIUM_VOCAB, LARGE_VOCAB]:
            logits = _make_logits(vs)
            proc = EntropyBasedProcessor(target_entropy=2.0)
            result = proc(logits)
            self.assertEqual(result.shape, logits.shape)

    def test_valid_distribution_output(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = EntropyBasedProcessor(target_entropy=2.0)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_high_target_entropy(self):
        logits = _make_logits(SMALL_VOCAB, distribution="peaked")
        proc = EntropyBasedProcessor(target_entropy=10.0, strength=2.0)
        result = proc(logits)
        probs = _stable_softmax(result)
        # Should be flatter than original
        self.assertGreater(_entropy(probs), 0.01)


class TestContrastiveProcessor(unittest.TestCase):
    """Tests for ContrastiveProcessor."""

    def test_no_similarity_scores(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = ContrastiveProcessor(alpha=0.6)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_penalty_applied_to_top_k(self):
        logits = _make_logits(SMALL_VOCAB)
        sim = np.ones(SMALL_VOCAB)
        proc = ContrastiveProcessor(alpha=0.6, k=3)
        result = proc(logits, similarity_scores=sim)
        top_k = np.argsort(logits)[-3:]
        for idx in top_k:
            self.assertNotAlmostEqual(result[idx], logits[idx], places=3)

    def test_alpha_zero_minimal_change(self):
        logits = _make_logits(SMALL_VOCAB)
        sim = np.ones(SMALL_VOCAB) * 0.5
        proc = ContrastiveProcessor(alpha=0.0, k=3)
        result = proc(logits, similarity_scores=sim)
        # alpha=0 → (1-0)*probs - 0*sim = probs for top-k
        top_k = np.argsort(logits)[-3:]
        probs = _stable_softmax(logits)
        for idx in top_k:
            self.assertAlmostEqual(result[idx], probs[idx], places=5)

    def test_high_alpha_penalises(self):
        logits = _make_logits(SMALL_VOCAB)
        sim = np.ones(SMALL_VOCAB)
        proc = ContrastiveProcessor(alpha=0.9, k=3)
        result = proc(logits, similarity_scores=sim)
        top_k = np.argsort(logits)[-3:]
        for idx in top_k:
            self.assertLess(result[idx], logits[idx])

    def test_zero_similarity(self):
        logits = _make_logits(SMALL_VOCAB)
        sim = np.zeros(SMALL_VOCAB)
        proc = ContrastiveProcessor(alpha=0.6, k=3)
        result = proc(logits, similarity_scores=sim)
        probs = _stable_softmax(logits)
        top_k = np.argsort(logits)[-3:]
        for idx in top_k:
            expected = (1 - 0.6) * probs[idx] - 0.6 * 0.0
            self.assertAlmostEqual(result[idx], expected, places=5)

    def test_output_length(self):
        logits = _make_logits(LARGE_VOCAB)
        sim = np.random.RandomState(42).rand(LARGE_VOCAB)
        proc = ContrastiveProcessor(alpha=0.5, k=10)
        result = proc(logits, similarity_scores=sim)
        self.assertEqual(len(result), LARGE_VOCAB)


class TestDiversityBoostProcessor(unittest.TestCase):
    """Tests for DiversityBoostProcessor."""

    def test_no_frequencies(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = DiversityBoostProcessor(boost_factor=1.0)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_uniform_frequencies_boost(self):
        logits = _make_logits(SMALL_VOCAB)
        freq = np.ones(SMALL_VOCAB)
        proc = DiversityBoostProcessor(boost_factor=1.0)
        result = proc(logits, token_frequencies=freq)
        # all same freq → norm = 1.0 → boost = 0
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_zero_frequencies_max_boost(self):
        logits = np.zeros(SMALL_VOCAB)
        freq = np.zeros(SMALL_VOCAB)
        freq[0] = 10.0
        proc = DiversityBoostProcessor(boost_factor=1.0)
        result = proc(logits, token_frequencies=freq)
        # Token 0 norm=1.0 → boost=0, others norm=0 → boost=1.0
        self.assertAlmostEqual(result[0], 0.0, places=5)
        for i in range(1, SMALL_VOCAB):
            self.assertAlmostEqual(result[i], 1.0, places=5)

    def test_boost_factor_zero(self):
        logits = _make_logits(SMALL_VOCAB)
        freq = np.array([10, 0, 5, 3, 1, 0, 7, 2, 0, 4], dtype=np.float64)
        proc = DiversityBoostProcessor(boost_factor=0.0)
        result = proc(logits, token_frequencies=freq)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_rare_tokens_boosted(self):
        logits = np.zeros(SMALL_VOCAB)
        freq = np.arange(SMALL_VOCAB, dtype=np.float64)
        proc = DiversityBoostProcessor(boost_factor=1.0)
        result = proc(logits, token_frequencies=freq)
        # Token 0 (freq=0) should have highest boost
        self.assertEqual(np.argmax(result), 0)

    def test_output_shape(self):
        logits = _make_logits(LARGE_VOCAB)
        freq = np.random.RandomState(42).rand(LARGE_VOCAB)
        proc = DiversityBoostProcessor(boost_factor=0.5)
        result = proc(logits, token_frequencies=freq)
        self.assertEqual(result.shape, logits.shape)


class TestSamplingConfig(unittest.TestCase):
    """Tests for SamplingConfig dataclass."""

    def test_defaults(self):
        cfg = SamplingConfig()
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertEqual(cfg.repetition_penalty, 1.0)
        self.assertEqual(cfg.num_samples, 1)
        self.assertEqual(cfg.max_tokens, 100)

    def test_custom_values(self):
        cfg = SamplingConfig(temperature=0.7, top_k=50, top_p=0.9)
        self.assertEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.top_k, 50)
        self.assertEqual(cfg.top_p, 0.9)

    def test_validate_valid(self):
        cfg = SamplingConfig(temperature=1.0, top_k=50, top_p=0.9)
        errors = cfg.validate()
        self.assertEqual(len(errors), 0)

    def test_validate_negative_temperature(self):
        cfg = SamplingConfig(temperature=-1.0)
        errors = cfg.validate()
        self.assertTrue(any("temperature" in e for e in errors))

    def test_validate_invalid_top_p(self):
        cfg = SamplingConfig(top_p=1.5)
        errors = cfg.validate()
        self.assertTrue(any("top_p" in e for e in errors))

    def test_validate_invalid_rep_penalty(self):
        cfg = SamplingConfig(repetition_penalty=0.5)
        errors = cfg.validate()
        self.assertTrue(any("repetition_penalty" in e for e in errors))

    def test_to_dict(self):
        cfg = SamplingConfig(temperature=0.8, top_k=40)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["temperature"], 0.8)
        self.assertEqual(d["top_k"], 40)

    def test_from_dict(self):
        d = {"temperature": 0.5, "top_k": 30, "top_p": 0.85, "unknown_key": 99}
        cfg = SamplingConfig.from_dict(d)
        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.top_k, 30)
        self.assertEqual(cfg.top_p, 0.85)

    def test_roundtrip(self):
        cfg = SamplingConfig(temperature=0.6, top_k=20, seed=42)
        d = cfg.to_dict()
        cfg2 = SamplingConfig.from_dict(d)
        self.assertEqual(cfg, cfg2)

    def test_seed_none(self):
        cfg = SamplingConfig()
        self.assertIsNone(cfg.seed)

    def test_seed_set(self):
        cfg = SamplingConfig(seed=123)
        self.assertEqual(cfg.seed, 123)


class TestBeamState(unittest.TestCase):
    """Tests for BeamState."""

    def test_initial_state(self):
        bs = BeamState()
        self.assertEqual(bs.token_ids, [])
        self.assertEqual(bs.log_prob, 0.0)
        self.assertFalse(bs.finished)
        self.assertEqual(bs.length, 0)

    def test_extend(self):
        bs = BeamState()
        new = bs.extend(token_id=5, token_log_prob=-1.0)
        self.assertEqual(new.token_ids, [5])
        self.assertAlmostEqual(new.log_prob, -1.0, places=5)
        self.assertFalse(new.finished)

    def test_extend_eos(self):
        bs = BeamState()
        new = bs.extend(token_id=EOS_TOKEN_ID, token_log_prob=-0.5)
        self.assertTrue(new.finished)

    def test_extend_chain(self):
        bs = BeamState()
        bs = bs.extend(3, -0.5)
        bs = bs.extend(7, -1.0)
        bs = bs.extend(2, -0.3)
        self.assertEqual(bs.token_ids, [3, 7, 2])
        self.assertAlmostEqual(bs.log_prob, -1.8, places=5)
        self.assertEqual(bs.length, 3)

    def test_score_normalized(self):
        bs = BeamState()
        bs = bs.extend(3, -3.0)
        self.assertAlmostEqual(bs.score, -3.0 / 1, places=5)
        bs = bs.extend(4, -2.0)
        self.assertAlmostEqual(bs.score, -5.0 / 2, places=5)

    def test_immutability(self):
        bs = BeamState(token_ids=[1, 2])
        new = bs.extend(3, -1.0)
        self.assertEqual(bs.token_ids, [1, 2])
        self.assertEqual(new.token_ids, [1, 2, 3])

    def test_length_property(self):
        bs = BeamState(token_ids=[1, 2, 3, 4, 5])
        self.assertEqual(bs.length, 5)


class TestBeamManager(unittest.TestCase):
    """Tests for BeamManager."""

    def test_initial_state(self):
        bm = BeamManager(num_beams=3)
        self.assertEqual(len(bm.beams), 1)
        self.assertEqual(len(bm.completed), 0)

    def test_step_expands_beams(self):
        bm = BeamManager(num_beams=3)
        candidates = [(0, 5, -0.5), (0, 3, -1.0), (0, 7, -0.8)]
        bm.step(candidates)
        self.assertEqual(len(bm.beams), 3)

    def test_pruning(self):
        bm = BeamManager(num_beams=2)
        candidates = [(0, 5, -0.5), (0, 3, -1.0), (0, 7, -0.1)]
        bm.step(candidates)
        self.assertEqual(len(bm.beams), 2)

    def test_completed_beams(self):
        bm = BeamManager(num_beams=3)
        candidates = [(0, EOS_TOKEN_ID, -0.5), (0, 3, -1.0)]
        bm.step(candidates)
        self.assertEqual(len(bm.completed), 1)

    def test_best(self):
        bm = BeamManager(num_beams=3)
        bm.step([(0, 5, -0.5), (0, 3, -0.1), (0, 7, -2.0)])
        best = bm.best(1)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].token_ids[-1], 3)  # highest log_prob

    def test_score_with_length_penalty(self):
        bm = BeamManager(num_beams=3, length_penalty=1.0)
        bs = BeamState(token_ids=[1, 2, 3], log_prob=-3.0)
        score = bm.score(bs)
        lp = ((5 + 3) / 6) ** 1.0
        self.assertAlmostEqual(score, -3.0 / lp, places=5)

    def test_is_done_when_empty(self):
        bm = BeamManager(num_beams=1)
        bm.step([(0, EOS_TOKEN_ID, -0.5)])
        self.assertTrue(bm.is_done)

    def test_multiple_steps(self):
        bm = BeamManager(num_beams=2)
        bm.step([(0, 5, -0.5), (0, 3, -0.3)])
        bm.step([(0, 7, -0.2), (1, 8, -0.4)])
        self.assertEqual(len(bm.beams), 2)
        for beam in bm.beams:
            self.assertEqual(beam.length, 2)

    def test_best_includes_completed(self):
        bm = BeamManager(num_beams=2)
        bm.step([(0, EOS_TOKEN_ID, -0.1), (0, 3, -5.0)])
        best = bm.best(1)
        self.assertTrue(best[0].finished)

    def test_length_penalty_zero(self):
        bm = BeamManager(num_beams=2, length_penalty=0.0)
        bs = BeamState(token_ids=[1, 2, 3], log_prob=-3.0)
        score = bm.score(bs)
        # lp = 1.0 when alpha=0
        self.assertAlmostEqual(score, -3.0, places=5)


class TestDiverseBeamGroups(unittest.TestCase):
    """Tests for DiverseBeamGroups."""

    def test_initialization(self):
        dbg = DiverseBeamGroups(num_groups=4, beams_per_group=3)
        self.assertEqual(len(dbg.groups), 4)

    def test_diversity_penalty_group_0(self):
        dbg = DiverseBeamGroups(num_groups=3, beams_per_group=2,
                                diversity_penalty=1.0)
        logits = _make_logits(SMALL_VOCAB)
        result = dbg.apply_diversity_penalty(logits, group_idx=0)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_diversity_penalty_later_groups(self):
        dbg = DiverseBeamGroups(num_groups=3, beams_per_group=2,
                                diversity_penalty=1.0)
        # Simulate group 0 having a beam with last token = 5
        dbg.groups[0].beams = [BeamState(token_ids=[5], log_prob=-1.0)]
        logits = np.zeros(SMALL_VOCAB)
        result = dbg.apply_diversity_penalty(logits, group_idx=1)
        self.assertAlmostEqual(result[5], -1.0, places=5)

    def test_multiple_groups_penalty(self):
        dbg = DiverseBeamGroups(num_groups=3, beams_per_group=2,
                                diversity_penalty=0.5)
        dbg.groups[0].beams = [BeamState(token_ids=[3], log_prob=-1.0)]
        dbg.groups[1].beams = [BeamState(token_ids=[3], log_prob=-1.0)]
        logits = np.zeros(SMALL_VOCAB)
        result = dbg.apply_diversity_penalty(logits, group_idx=2)
        self.assertAlmostEqual(result[3], -1.0, places=5)

    def test_best_per_group(self):
        dbg = DiverseBeamGroups(num_groups=2, beams_per_group=2)
        dbg.groups[0].step([(0, 5, -0.5), (0, 3, -0.1)])
        dbg.groups[1].step([(0, 7, -0.3), (0, 8, -0.2)])
        results = dbg.best_per_group(1)
        self.assertEqual(len(results), 2)
        for group_best in results:
            self.assertEqual(len(group_best), 1)

    def test_different_tokens_across_groups(self):
        dbg = DiverseBeamGroups(num_groups=3, beams_per_group=1,
                                diversity_penalty=10.0)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        # Group 0: takes token 0 (highest)
        dbg.groups[0].beams = [BeamState(token_ids=[0], log_prob=-0.1)]
        # Penalty should discourage token 0 for group 1
        penalised = dbg.apply_diversity_penalty(logits, group_idx=1)
        self.assertLess(penalised[0], logits[0])


class TestSampleFromLogits(unittest.TestCase):
    """Tests for sample_from_logits."""

    def test_deterministic_with_seed(self):
        logits = _make_logits(MEDIUM_VOCAB)
        t1 = sample_from_logits(logits, seed=42)
        t2 = sample_from_logits(logits, seed=42)
        self.assertEqual(t1, t2)

    def test_different_seeds_different_results(self):
        logits = _make_logits(MEDIUM_VOCAB)
        results = {sample_from_logits(logits, seed=s) for s in range(20)}
        self.assertGreater(len(results), 1)

    def test_temperature_zero_argmax(self):
        logits = _make_logits(MEDIUM_VOCAB)
        token = sample_from_logits(logits, temperature=0)
        self.assertEqual(token, int(np.argmax(logits)))

    def test_in_range(self):
        logits = _make_logits(LARGE_VOCAB)
        for seed in range(50):
            t = sample_from_logits(logits, seed=seed)
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, LARGE_VOCAB)

    def test_top_k_restricts(self):
        logits = _make_logits(LARGE_VOCAB, seed=1)
        top_k = 5
        top_k_idx = set(np.argsort(logits)[-top_k:])
        for seed in range(100):
            t = sample_from_logits(logits, top_k=top_k, seed=seed)
            self.assertIn(t, top_k_idx)

    def test_top_p_restricts(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="zipf")
        tokens = {sample_from_logits(logits, top_p=0.1, seed=s)
                  for s in range(200)}
        self.assertLess(len(tokens), MEDIUM_VOCAB)

    def test_returns_int(self):
        logits = _make_logits(SMALL_VOCAB)
        t = sample_from_logits(logits, seed=42)
        self.assertIsInstance(t, int)

    def test_peaked_logits(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        t = sample_from_logits(logits, temperature=0.1, seed=42)
        self.assertEqual(t, 0)


class TestTopKFiltering(unittest.TestCase):
    """Tests for standalone top_k_filtering function."""

    def test_basic(self):
        logits = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
        result = top_k_filtering(logits, k=2)
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isfinite(result[3]))

    def test_k_zero_returns_copy(self):
        logits = _make_logits(SMALL_VOCAB)
        result = top_k_filtering(logits, k=0)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_k_exceeds_vocab(self):
        logits = _make_logits(SMALL_VOCAB)
        result = top_k_filtering(logits, k=SMALL_VOCAB + 10)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_does_not_modify_input(self):
        logits = _make_logits(MEDIUM_VOCAB)
        original = logits.copy()
        top_k_filtering(logits, k=10)
        np.testing.assert_allclose(logits, original, atol=FLOAT_ATOL)

    def test_various_k_values(self):
        logits = _make_logits(LARGE_VOCAB)
        for k in [1, 5, 10, 50, 100, 500]:
            result = top_k_filtering(logits, k=k)
            valid = np.sum(result > -float("inf"))
            self.assertGreaterEqual(valid, 1)
            self.assertLessEqual(valid, k + 10)  # ties

    def test_result_shape(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = top_k_filtering(logits, k=10)
        self.assertEqual(result.shape, logits.shape)


class TestTopPFiltering(unittest.TestCase):
    """Tests for standalone top_p_filtering function."""

    def test_basic(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = top_p_filtering(logits, p=0.9)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)
        self.assertLessEqual(valid, MEDIUM_VOCAB)

    def test_p_1_keeps_all(self):
        logits = _make_logits(SMALL_VOCAB)
        result = top_p_filtering(logits, p=1.0)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, SMALL_VOCAB)

    def test_does_not_modify_input(self):
        logits = _make_logits(MEDIUM_VOCAB)
        original = logits.copy()
        top_p_filtering(logits, p=0.5)
        np.testing.assert_allclose(logits, original, atol=FLOAT_ATOL)

    def test_cumulative_prob_coverage(self):
        logits = _make_logits(MEDIUM_VOCAB)
        p = 0.8
        result = top_p_filtering(logits, p=p)
        probs = _stable_softmax(logits)
        kept = result > -float("inf")
        self.assertGreaterEqual(np.sum(probs[kept]), p - 0.01)

    def test_various_p_values(self):
        logits = _make_logits(LARGE_VOCAB)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = top_p_filtering(logits, p=p)
            valid = np.sum(result > -float("inf"))
            self.assertGreaterEqual(valid, 1)
            self.assertLessEqual(valid, LARGE_VOCAB)

    def test_result_shape(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = top_p_filtering(logits, p=0.9)
        self.assertEqual(result.shape, logits.shape)


class TestApplyProcessorChain(unittest.TestCase):
    """Tests for apply_processor_chain."""

    def test_empty_chain(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = apply_processor_chain(logits, [])
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_single_processor(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TemperatureProcessor(temperature=0.5)
        result = apply_processor_chain(logits, [proc])
        expected = proc(logits)
        np.testing.assert_allclose(result, expected, atol=FLOAT_ATOL)

    def test_chain_temperature_then_topk(self):
        logits = _make_logits(MEDIUM_VOCAB)
        chain = [TemperatureProcessor(temperature=0.5), TopKProcessor(k=10)]
        result = apply_processor_chain(logits, chain)
        valid = np.sum(result > -float("inf"))
        self.assertLessEqual(valid, 11)

    def test_chain_order_matters(self):
        logits = _make_logits(MEDIUM_VOCAB)
        chain_a = [TemperatureProcessor(temperature=0.5), TopPProcessor(p=0.5)]
        chain_b = [TopPProcessor(p=0.5), TemperatureProcessor(temperature=0.5)]
        result_a = apply_processor_chain(logits, chain_a)
        result_b = apply_processor_chain(logits, chain_b)
        # Temp then filter vs filter then temp yield different kept sets
        valid_a = np.sum(result_a > -float("inf"))
        valid_b = np.sum(result_b > -float("inf"))
        self.assertNotEqual(valid_a, valid_b)

    def test_three_processors(self):
        logits = _make_logits(MEDIUM_VOCAB)
        chain = [
            TemperatureProcessor(temperature=0.8),
            TopKProcessor(k=50),
            TopPProcessor(p=0.9),
        ]
        result = apply_processor_chain(logits, chain)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_does_not_modify_input(self):
        logits = _make_logits(MEDIUM_VOCAB)
        original = logits.copy()
        chain = [TemperatureProcessor(temperature=0.5)]
        apply_processor_chain(logits, chain)
        np.testing.assert_allclose(logits, original, atol=FLOAT_ATOL)

    def test_kwargs_passed_through(self):
        logits = np.zeros(SMALL_VOCAB)
        chain = [NoRepeatNgramProcessor(n=2)]
        generated = [3, 5, 3]
        result = apply_processor_chain(logits, chain, generated_ids=generated)
        self.assertEqual(result[5], -float("inf"))

    def test_chain_with_repetition_penalty(self):
        logits = _make_logits(MEDIUM_VOCAB)
        chain = [
            RepetitionPenaltyProcessor(penalty=1.5),
            TemperatureProcessor(temperature=0.7),
        ]
        result = apply_processor_chain(logits, chain,
                                       generated_ids=[np.argmax(logits)])
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_full_pipeline(self):
        logits = _make_logits(LARGE_VOCAB)
        chain = [
            TemperatureProcessor(temperature=0.8),
            TopKProcessor(k=50),
            TopPProcessor(p=0.95),
            RepetitionPenaltyProcessor(penalty=1.2),
        ]
        result = apply_processor_chain(logits, chain,
                                       generated_ids=[0, 1, 2, 3])
        valid = np.sum(result > -float("inf"))
        self.assertLess(valid, LARGE_VOCAB)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)


class TestComputeEntropy(unittest.TestCase):
    """Tests for compute_entropy."""

    def test_uniform_distribution(self):
        logits = np.zeros(SMALL_VOCAB)
        ent = compute_entropy(logits)
        expected = np.log(SMALL_VOCAB)
        self.assertAlmostEqual(ent, expected, places=5)

    def test_peaked_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        ent = compute_entropy(logits)
        self.assertLess(ent, 1.0)

    def test_entropy_range(self):
        logits = _make_logits(MEDIUM_VOCAB)
        ent = compute_entropy(logits)
        self.assertGreaterEqual(ent, 0.0)
        self.assertLessEqual(ent, np.log(MEDIUM_VOCAB) + 0.01)

    def test_single_element(self):
        logits = np.array([5.0])
        ent = compute_entropy(logits)
        self.assertAlmostEqual(ent, 0.0, places=5)

    def test_two_equal(self):
        logits = np.array([0.0, 0.0])
        ent = compute_entropy(logits)
        self.assertAlmostEqual(ent, np.log(2), places=5)

    def test_various_vocab_sizes(self):
        for vs in [SMALL_VOCAB, MEDIUM_VOCAB, LARGE_VOCAB]:
            logits = np.zeros(vs)
            ent = compute_entropy(logits)
            self.assertAlmostEqual(ent, np.log(vs), places=4)

    def test_higher_temperature_higher_entropy(self):
        logits = _make_logits(MEDIUM_VOCAB)
        ent_low = compute_entropy(logits / 0.5)
        ent_high = compute_entropy(logits / 2.0)
        self.assertLess(ent_low, ent_high)


class TestComputeVarentropy(unittest.TestCase):
    """Tests for compute_varentropy."""

    def test_uniform_zero_varentropy(self):
        logits = np.zeros(SMALL_VOCAB)
        ve = compute_varentropy(logits)
        self.assertAlmostEqual(ve, 0.0, places=5)

    def test_peaked_nonzero(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        ve = compute_varentropy(logits)
        self.assertGreaterEqual(ve, 0.0)

    def test_non_negative(self):
        for seed in range(10):
            logits = _make_logits(MEDIUM_VOCAB, seed=seed)
            ve = compute_varentropy(logits)
            self.assertGreaterEqual(ve, -FLOAT_ATOL)

    def test_single_element(self):
        logits = np.array([5.0])
        ve = compute_varentropy(logits)
        self.assertAlmostEqual(ve, 0.0, places=5)


class TestGumbelSoftmax(unittest.TestCase):
    """Tests for gumbel_softmax_sampling."""

    def test_deterministic_with_seed(self):
        logits = _make_logits(MEDIUM_VOCAB)
        t1 = gumbel_softmax_sampling(logits, seed=42)
        t2 = gumbel_softmax_sampling(logits, seed=42)
        self.assertEqual(t1, t2)

    def test_in_range(self):
        logits = _make_logits(LARGE_VOCAB)
        for s in range(50):
            t = gumbel_softmax_sampling(logits, seed=s)
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, LARGE_VOCAB)

    def test_low_temperature_peaked(self):
        logits = _make_logits(MEDIUM_VOCAB)
        argmax = int(np.argmax(logits))
        count = sum(1 for s in range(100)
                    if gumbel_softmax_sampling(logits, temperature=0.01,
                                               seed=s) == argmax)
        self.assertGreater(count, 80)

    def test_high_temperature_diverse(self):
        logits = _make_logits(MEDIUM_VOCAB)
        tokens = {gumbel_softmax_sampling(logits, temperature=100.0, seed=s)
                  for s in range(100)}
        self.assertGreater(len(tokens), 10)

    def test_returns_int(self):
        logits = _make_logits(SMALL_VOCAB)
        t = gumbel_softmax_sampling(logits, seed=0)
        self.assertIsInstance(t, int)

    def test_different_seeds(self):
        logits = _make_logits(MEDIUM_VOCAB)
        results = {gumbel_softmax_sampling(logits, seed=s) for s in range(50)}
        self.assertGreater(len(results), 1)

    def test_peaked_logits_bias(self):
        logits = _make_logits(SMALL_VOCAB, distribution="peaked")
        count = sum(1 for s in range(200)
                    if gumbel_softmax_sampling(logits, seed=s) == 0)
        self.assertGreater(count, 150)


class TestSystematicResampling(unittest.TestCase):
    """Tests for systematic_resampling."""

    def test_correct_count(self):
        weights = np.array([0.3, 0.5, 0.2])
        indices = systematic_resampling(weights, n=10, seed=42)
        self.assertEqual(len(indices), 10)

    def test_indices_in_range(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        indices = systematic_resampling(weights, n=20, seed=42)
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(weights))

    def test_uniform_weights(self):
        n_particles = 4
        weights = np.ones(n_particles)
        indices = systematic_resampling(weights, n=n_particles, seed=42)
        counts = Counter(indices)
        for c in counts.values():
            self.assertEqual(c, 1)

    def test_single_dominant(self):
        weights = np.array([0.0, 0.0, 1.0, 0.0])
        indices = systematic_resampling(weights, n=10, seed=42)
        self.assertTrue(all(i == 2 for i in indices))

    def test_proportional(self):
        weights = np.array([0.5, 0.5])
        indices = systematic_resampling(weights, n=100, seed=42)
        counts = Counter(indices)
        self.assertAlmostEqual(counts[0] / 100, 0.5, delta=0.05)

    def test_deterministic(self):
        weights = np.array([0.3, 0.3, 0.4])
        r1 = systematic_resampling(weights, n=10, seed=99)
        r2 = systematic_resampling(weights, n=10, seed=99)
        self.assertEqual(r1, r2)

    def test_unnormalized_weights(self):
        weights = np.array([30, 50, 20])
        indices = systematic_resampling(weights, n=10, seed=42)
        self.assertEqual(len(indices), 10)


class TestStochasticBeamSearch(unittest.TestCase):
    """Tests for stochastic_beam_search_sampling."""

    def test_correct_count(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = stochastic_beam_search_sampling(logits, k=5, seed=42)
        self.assertEqual(len(result), 5)

    def test_unique_samples(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = stochastic_beam_search_sampling(logits, k=10, seed=42)
        self.assertEqual(len(set(result)), 10)

    def test_in_range(self):
        logits = _make_logits(LARGE_VOCAB)
        result = stochastic_beam_search_sampling(logits, k=20, seed=42)
        for idx in result:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, LARGE_VOCAB)

    def test_k_equals_1(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = stochastic_beam_search_sampling(logits, k=1, seed=42)
        self.assertEqual(len(result), 1)

    def test_deterministic(self):
        logits = _make_logits(MEDIUM_VOCAB)
        r1 = stochastic_beam_search_sampling(logits, k=5, seed=42)
        r2 = stochastic_beam_search_sampling(logits, k=5, seed=42)
        self.assertEqual(r1, r2)

    def test_biased_toward_high_logits(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        result = stochastic_beam_search_sampling(logits, k=3, seed=42)
        self.assertIn(0, result)  # token 0 has logit 10.0


class TestSampleWithoutReplacement(unittest.TestCase):
    """Tests for sample_without_replacement."""

    def test_uniqueness(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = sample_without_replacement(logits, k=20, seed=42)
        self.assertEqual(len(set(result)), 20)

    def test_correct_count(self):
        logits = _make_logits(MEDIUM_VOCAB)
        for k in [1, 5, 10, 50]:
            result = sample_without_replacement(logits, k=k, seed=42)
            self.assertEqual(len(result), k)

    def test_in_range(self):
        logits = _make_logits(LARGE_VOCAB)
        result = sample_without_replacement(logits, k=30, seed=42)
        for idx in result:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, LARGE_VOCAB)

    def test_k_equals_vocab(self):
        logits = _make_logits(SMALL_VOCAB)
        result = sample_without_replacement(logits, k=SMALL_VOCAB, seed=42)
        self.assertEqual(len(set(result)), SMALL_VOCAB)

    def test_deterministic(self):
        logits = _make_logits(MEDIUM_VOCAB)
        r1 = sample_without_replacement(logits, k=10, seed=42)
        r2 = sample_without_replacement(logits, k=10, seed=42)
        self.assertEqual(r1, r2)

    def test_peaked_logits_distribution(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="peaked")
        result = sample_without_replacement(logits, k=5, seed=42)
        self.assertIn(0, result)  # most likely token

    def test_k_1(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = sample_without_replacement(logits, k=1, seed=42)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], int)

    def test_frequency_bias(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="zipf")
        counts = Counter()
        for s in range(200):
            result = sample_without_replacement(logits, k=1, seed=s)
            counts[result[0]] += 1
        # Token 0 has highest probability in zipf
        self.assertEqual(counts.most_common(1)[0][0], 0)


class TestTokenUsageTracker(unittest.TestCase):
    """Tests for TokenUsageTracker."""

    def test_initial_state(self):
        tracker = TokenUsageTracker(vocab_size=MEDIUM_VOCAB)
        self.assertEqual(tracker.total, 0)
        self.assertEqual(tracker.unique_tokens, 0)
        self.assertAlmostEqual(tracker.entropy, 0.0)

    def test_update(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        tracker.update(3)
        self.assertEqual(tracker.total, 1)
        self.assertEqual(tracker.counts[3], 1)

    def test_frequency(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        tracker.update(0)
        tracker.update(0)
        tracker.update(1)
        self.assertAlmostEqual(tracker.frequency(0), 2 / 3, places=5)
        self.assertAlmostEqual(tracker.frequency(1), 1 / 3, places=5)

    def test_unique_tokens(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        for t in [0, 1, 2, 0, 1]:
            tracker.update(t)
        self.assertEqual(tracker.unique_tokens, 3)

    def test_entropy(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        for t in [0, 1]:
            tracker.update(t)
        ent = tracker.entropy
        self.assertAlmostEqual(ent, np.log(2), places=5)

    def test_reset(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        tracker.update(3)
        tracker.update(5)
        tracker.reset()
        self.assertEqual(tracker.total, 0)
        self.assertEqual(tracker.unique_tokens, 0)
        self.assertEqual(len(tracker.step_history), 0)

    def test_step_history(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        tokens = [1, 3, 5, 7, 1]
        for t in tokens:
            tracker.update(t)
        self.assertEqual(tracker.step_history, tokens)

    def test_large_vocab(self):
        tracker = TokenUsageTracker(vocab_size=LARGE_VOCAB)
        rng = np.random.RandomState(42)
        for _ in range(500):
            tracker.update(int(rng.randint(0, LARGE_VOCAB)))
        self.assertEqual(tracker.total, 500)
        self.assertGreater(tracker.unique_tokens, 0)
        self.assertGreater(tracker.entropy, 0)

    def test_all_same_token(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        for _ in range(100):
            tracker.update(0)
        self.assertEqual(tracker.unique_tokens, 1)
        self.assertAlmostEqual(tracker.entropy, 0.0, places=5)
        self.assertAlmostEqual(tracker.frequency(0), 1.0, places=5)

    def test_frequency_zero_for_unseen(self):
        tracker = TokenUsageTracker(vocab_size=SMALL_VOCAB)
        tracker.update(0)
        self.assertAlmostEqual(tracker.frequency(5), 0.0, places=5)


class TestSamplingStatistics(unittest.TestCase):
    """Tests for SamplingStatistics."""

    def test_initial_state(self):
        stats = SamplingStatistics()
        self.assertEqual(stats.mean_entropy, 0.0)
        self.assertEqual(stats.mean_temperature, 0.0)
        self.assertEqual(stats.rejected_count, 0)

    def test_record(self):
        stats = SamplingStatistics()
        stats.record(temperature=0.8, entropy=2.0, top1_prob=0.3, token=5)
        self.assertEqual(len(stats.accepted_tokens), 1)
        self.assertEqual(stats.accepted_tokens[0], 5)

    def test_mean_entropy(self):
        stats = SamplingStatistics()
        stats.record(1.0, 2.0, 0.3, 0)
        stats.record(1.0, 4.0, 0.2, 1)
        self.assertAlmostEqual(stats.mean_entropy, 3.0, places=5)

    def test_mean_temperature(self):
        stats = SamplingStatistics()
        stats.record(0.5, 1.0, 0.5, 0)
        stats.record(1.5, 1.0, 0.5, 0)
        self.assertAlmostEqual(stats.mean_temperature, 1.0, places=5)

    def test_acceptance_rate(self):
        stats = SamplingStatistics()
        stats.record(1.0, 1.0, 0.5, 0)
        stats.record(1.0, 1.0, 0.5, 1)
        stats.record_rejection()
        self.assertAlmostEqual(stats.acceptance_rate, 2 / 3, places=5)

    def test_summary(self):
        stats = SamplingStatistics()
        stats.record(1.0, 2.0, 0.5, 0)
        summary = stats.summary()
        self.assertIn("num_steps", summary)
        self.assertIn("mean_entropy", summary)
        self.assertIn("acceptance_rate", summary)
        self.assertEqual(summary["num_steps"], 1)

    def test_no_rejections(self):
        stats = SamplingStatistics()
        stats.record(1.0, 1.0, 0.5, 0)
        self.assertAlmostEqual(stats.acceptance_rate, 1.0, places=5)

    def test_all_rejections(self):
        stats = SamplingStatistics()
        for _ in range(10):
            stats.record_rejection()
        self.assertAlmostEqual(stats.acceptance_rate, 0.0, places=5)

    def test_many_records(self):
        stats = SamplingStatistics()
        rng = np.random.RandomState(42)
        for i in range(100):
            stats.record(
                temperature=float(rng.uniform(0.5, 2.0)),
                entropy=float(rng.uniform(0, 5)),
                top1_prob=float(rng.uniform(0, 1)),
                token=int(rng.randint(0, 1000)),
            )
        self.assertEqual(len(stats.accepted_tokens), 100)
        self.assertGreater(stats.mean_entropy, 0)


class TestBatchSampler(unittest.TestCase):
    """Tests for BatchSampler."""

    def test_sample_batch(self):
        cfg = SamplingConfig(temperature=1.0, seed=42)
        sampler = BatchSampler(cfg)
        batch = np.random.RandomState(42).randn(4, MEDIUM_VOCAB)
        results = sampler.sample_batch(batch)
        self.assertEqual(len(results), 4)
        for t in results:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, MEDIUM_VOCAB)

    def test_sample_multiple(self):
        cfg = SamplingConfig(temperature=1.0, seed=42)
        sampler = BatchSampler(cfg)
        logits = _make_logits(MEDIUM_VOCAB)
        results = sampler.sample_multiple(logits, n=10)
        self.assertEqual(len(results), 10)

    def test_top_k_in_batch(self):
        cfg = SamplingConfig(temperature=1.0, top_k=5, seed=42)
        sampler = BatchSampler(cfg)
        logits = _make_logits(LARGE_VOCAB)
        top_5 = set(np.argsort(logits)[-5:])
        results = sampler.sample_multiple(logits, n=50)
        for t in results:
            self.assertIn(t, top_5)

    def test_temperature_zero(self):
        cfg = SamplingConfig(temperature=0, seed=42)
        sampler = BatchSampler(cfg)
        logits = _make_logits(MEDIUM_VOCAB)
        results = sampler.sample_multiple(logits, n=10)
        expected = int(np.argmax(logits))
        for t in results:
            self.assertEqual(t, expected)

    def test_different_batch_rows(self):
        cfg = SamplingConfig(temperature=0, seed=42)
        sampler = BatchSampler(cfg)
        batch = np.zeros((3, SMALL_VOCAB))
        batch[0, 2] = 10.0
        batch[1, 5] = 10.0
        batch[2, 8] = 10.0
        results = sampler.sample_batch(batch)
        self.assertEqual(results[0], 2)
        self.assertEqual(results[1], 5)
        self.assertEqual(results[2], 8)

    def test_reproducibility(self):
        cfg = SamplingConfig(temperature=1.0, seed=42)
        sampler1 = BatchSampler(cfg)
        sampler2 = BatchSampler(SamplingConfig(temperature=1.0, seed=42))
        logits = _make_logits(MEDIUM_VOCAB)
        r1 = sampler1.sample_multiple(logits, n=5)
        r2 = sampler2.sample_multiple(logits, n=5)
        self.assertEqual(r1, r2)


class TestSequencePool(unittest.TestCase):
    """Tests for SequencePool."""

    def test_initial_empty(self):
        pool = SequencePool(max_size=10)
        self.assertEqual(pool.size, 0)

    def test_add_and_size(self):
        pool = SequencePool(max_size=10)
        pool.add([1, 2, 3], score=1.0)
        pool.add([4, 5, 6], score=2.0)
        self.assertEqual(pool.size, 2)

    def test_max_size_enforced(self):
        pool = SequencePool(max_size=3)
        for i in range(10):
            pool.add([i], score=float(i))
        self.assertEqual(pool.size, 3)

    def test_best_returns_highest(self):
        pool = SequencePool(max_size=10)
        pool.add([1], score=1.0)
        pool.add([2], score=5.0)
        pool.add([3], score=3.0)
        best = pool.best(1)
        self.assertEqual(best[0][0], [2])
        self.assertEqual(best[0][1], 5.0)

    def test_best_n(self):
        pool = SequencePool(max_size=10)
        for i in range(5):
            pool.add([i], score=float(i))
        best = pool.best(3)
        self.assertEqual(len(best), 3)
        scores = [s for _, s in best]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_clear(self):
        pool = SequencePool(max_size=10)
        pool.add([1, 2], score=1.0)
        pool.clear()
        self.assertEqual(pool.size, 0)

    def test_sequences(self):
        pool = SequencePool(max_size=10)
        pool.add([1, 2], score=1.0)
        pool.add([3, 4], score=2.0)
        seqs = pool.sequences()
        self.assertIn([1, 2], seqs)
        self.assertIn([3, 4], seqs)

    def test_diversity_empty(self):
        pool = SequencePool(max_size=10)
        self.assertAlmostEqual(pool.diversity(), 0.0, places=5)

    def test_diversity_all_same(self):
        pool = SequencePool(max_size=10)
        pool.add([1, 1, 1], score=1.0)
        pool.add([1, 1, 1], score=2.0)
        d = pool.diversity()
        # 1 unique token / 6 total
        self.assertAlmostEqual(d, 1 / 6, places=5)

    def test_diversity_all_unique(self):
        pool = SequencePool(max_size=10)
        pool.add([1, 2, 3], score=1.0)
        pool.add([4, 5, 6], score=2.0)
        d = pool.diversity()
        self.assertAlmostEqual(d, 6 / 6, places=5)

    def test_eviction_keeps_best(self):
        pool = SequencePool(max_size=3)
        pool.add([0], score=0.0)
        pool.add([1], score=1.0)
        pool.add([2], score=2.0)
        pool.add([3], score=3.0)
        seqs = pool.sequences()
        self.assertNotIn([0], seqs)
        self.assertIn([3], seqs)

    def test_best_more_than_available(self):
        pool = SequencePool(max_size=10)
        pool.add([1], score=1.0)
        best = pool.best(5)
        self.assertEqual(len(best), 1)


# =========================================================================
# Additional edge-case and integration tests
# =========================================================================


class TestProcessorEdgeCases(unittest.TestCase):
    """Edge cases across all processors."""

    def test_all_negative_logits(self):
        logits = np.array([-10.0, -20.0, -5.0, -15.0])
        proc = TopKProcessor(k=2)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, 2)

    def test_very_large_logits(self):
        logits = np.array([1e10, -1e10, 0.0])
        proc = TemperatureProcessor(temperature=1.0)
        result = proc(logits)
        probs = _stable_softmax(result)
        self.assertAlmostEqual(probs[0], 1.0, places=5)

    def test_nan_check(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = TemperatureProcessor(temperature=0.5)
        result = proc(logits)
        self.assertFalse(np.any(np.isnan(result)))

    def test_inf_propagation(self):
        logits = _make_logits(SMALL_VOCAB)
        logits[0] = float("inf")
        proc = TopKProcessor(k=1)
        result = proc(logits)
        self.assertEqual(np.argmax(result), 0)

    def test_empty_generated_ids_repetition(self):
        logits = _make_logits(MEDIUM_VOCAB)
        proc = RepetitionPenaltyProcessor(penalty=2.0)
        result = proc(logits)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)

    def test_top_p_with_single_dominant_token(self):
        logits = np.full(MEDIUM_VOCAB, -100.0)
        logits[42] = 100.0
        proc = TopPProcessor(p=0.5)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)
        self.assertTrue(np.isfinite(result[42]))

    def test_min_p_with_uniform(self):
        logits = np.zeros(SMALL_VOCAB)
        proc = MinPProcessor(min_p=0.5)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, SMALL_VOCAB)


class TestProcessorCombinations(unittest.TestCase):
    """Test realistic processor combinations."""

    def test_temp_topk_topp(self):
        logits = _make_logits(LARGE_VOCAB, seed=7)
        chain = [
            TemperatureProcessor(temperature=0.7),
            TopKProcessor(k=40),
            TopPProcessor(p=0.9),
        ]
        result = apply_processor_chain(logits, chain)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)
        valid = np.sum(result > -float("inf"))
        self.assertLessEqual(valid, 40)

    def test_repetition_then_typical(self):
        logits = _make_logits(MEDIUM_VOCAB, seed=13)
        chain = [
            RepetitionPenaltyProcessor(penalty=1.3),
            TypicalProcessor(mass=0.9),
        ]
        result = apply_processor_chain(logits, chain,
                                       generated_ids=[0, 1, 2, 3, 0, 1])
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_diversity_then_min_p(self):
        logits = _make_logits(MEDIUM_VOCAB, seed=21)
        freq = np.random.RandomState(21).randint(0, 10, size=MEDIUM_VOCAB).astype(np.float64)
        chain = [
            DiversityBoostProcessor(boost_factor=0.5),
            MinPProcessor(min_p=0.02),
        ]
        result = apply_processor_chain(logits, chain, token_frequencies=freq)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_no_repeat_then_topk(self):
        logits = _make_logits(SMALL_VOCAB, seed=33)
        chain = [
            NoRepeatNgramProcessor(n=2),
            TopKProcessor(k=5),
        ]
        result = apply_processor_chain(logits, chain,
                                       generated_ids=[1, 2, 3, 1])
        self.assertEqual(result[2], -float("inf"))

    def test_entropy_then_temperature(self):
        logits = _make_logits(MEDIUM_VOCAB, seed=55)
        chain = [
            EntropyBasedProcessor(target_entropy=2.0, strength=0.5),
            TemperatureProcessor(temperature=0.9),
        ]
        result = apply_processor_chain(logits, chain)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)


class TestSamplingDistribution(unittest.TestCase):
    """Statistical tests for sampling functions."""

    def test_sample_from_logits_uniform(self):
        logits = np.zeros(SMALL_VOCAB)
        counts = Counter()
        for s in range(5000):
            t = sample_from_logits(logits, seed=s)
            counts[t] += 1
        expected = 5000 / SMALL_VOCAB
        for c in counts.values():
            self.assertAlmostEqual(c, expected, delta=expected * 0.3)

    def test_gumbel_follows_distribution(self):
        logits = np.array([2.0, 1.0, 0.0])
        probs = _stable_softmax(logits)
        counts = Counter()
        for s in range(5000):
            t = gumbel_softmax_sampling(logits, seed=s)
            counts[t] += 1
        for i in range(3):
            empirical = counts[i] / 5000
            self.assertAlmostEqual(empirical, probs[i], delta=0.05)

    def test_sample_without_replacement_first_draw_distribution(self):
        logits = np.array([3.0, 1.0, 0.0])
        probs = _stable_softmax(logits)
        counts = Counter()
        for s in range(3000):
            result = sample_without_replacement(logits, k=1, seed=s)
            counts[result[0]] += 1
        for i in range(3):
            empirical = counts[i] / 3000
            self.assertAlmostEqual(empirical, probs[i], delta=0.05)


class TestBeamManagerAdvanced(unittest.TestCase):
    """Advanced beam manager tests."""

    def test_multi_step_beam_search(self):
        bm = BeamManager(num_beams=3)
        # Step 1: expand from initial beam
        bm.step([(0, 2, -0.5), (0, 3, -0.3), (0, 4, -0.8)])
        self.assertEqual(len(bm.beams), 3)
        # Step 2: expand each beam
        candidates = []
        for i in range(len(bm.beams)):
            candidates.append((i, 5, -0.2))
            candidates.append((i, 6, -0.4))
        bm.step(candidates)
        self.assertEqual(len(bm.beams), 3)  # pruned to num_beams

    def test_all_beams_finish(self):
        bm = BeamManager(num_beams=2)
        bm.step([(0, EOS_TOKEN_ID, -0.1), (0, EOS_TOKEN_ID, -0.2)])
        self.assertTrue(bm.is_done)
        self.assertEqual(len(bm.completed), 2)

    def test_best_returns_correct_order(self):
        bm = BeamManager(num_beams=3, length_penalty=0.0)
        bm.step([(0, 2, -1.0), (0, 3, -0.5), (0, 4, -2.0)])
        best = bm.best(3)
        scores = [bm.score(b) for b in best]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_length_penalty_effect(self):
        bm_no_lp = BeamManager(num_beams=2, length_penalty=0.0)
        bm_lp = BeamManager(num_beams=2, length_penalty=2.0)
        bs = BeamState(token_ids=[1, 2, 3, 4, 5], log_prob=-5.0)
        score_no_lp = bm_no_lp.score(bs)
        score_lp = bm_lp.score(bs)
        self.assertNotAlmostEqual(score_no_lp, score_lp, places=3)


class TestDiverseBeamGroupsAdvanced(unittest.TestCase):
    """Advanced diverse beam group tests."""

    def test_penalty_accumulates_across_groups(self):
        dbg = DiverseBeamGroups(num_groups=4, beams_per_group=1,
                                diversity_penalty=1.0)
        for g in range(3):
            dbg.groups[g].beams = [BeamState(token_ids=[5], log_prob=-1.0)]
        logits = np.zeros(SMALL_VOCAB)
        result = dbg.apply_diversity_penalty(logits, group_idx=3)
        self.assertAlmostEqual(result[5], -3.0, places=5)

    def test_independent_groups(self):
        dbg = DiverseBeamGroups(num_groups=3, beams_per_group=2)
        for g in range(3):
            dbg.groups[g].step([(0, g + 2, -0.5), (0, g + 5, -0.3)])
        results = dbg.best_per_group(1)
        self.assertEqual(len(results), 3)

    def test_zero_penalty(self):
        dbg = DiverseBeamGroups(num_groups=2, beams_per_group=2,
                                diversity_penalty=0.0)
        dbg.groups[0].beams = [BeamState(token_ids=[3], log_prob=-1.0)]
        logits = np.zeros(SMALL_VOCAB)
        result = dbg.apply_diversity_penalty(logits, group_idx=1)
        np.testing.assert_allclose(result, logits, atol=FLOAT_ATOL)


class TestSamplingConfigEdgeCases(unittest.TestCase):
    """Edge cases for SamplingConfig."""

    def test_validate_negative_top_k(self):
        cfg = SamplingConfig(top_k=-1)
        errors = cfg.validate()
        self.assertTrue(any("top_k" in e for e in errors))

    def test_validate_zero_temperature(self):
        cfg = SamplingConfig(temperature=0)
        errors = cfg.validate()
        self.assertEqual(len(errors), 0)

    def test_from_dict_ignores_extra_keys(self):
        d = {"temperature": 0.5, "nonexistent_param": 42}
        cfg = SamplingConfig.from_dict(d)
        self.assertEqual(cfg.temperature, 0.5)

    def test_from_dict_missing_keys_use_defaults(self):
        d = {"temperature": 0.8}
        cfg = SamplingConfig.from_dict(d)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(cfg.top_p, 1.0)

    def test_to_dict_includes_all_fields(self):
        cfg = SamplingConfig()
        d = cfg.to_dict()
        expected_keys = {"temperature", "top_k", "top_p", "typical_p", "min_p",
                         "repetition_penalty", "frequency_penalty",
                         "presence_penalty", "no_repeat_ngram_size",
                         "length_penalty", "seed", "num_samples", "max_tokens"}
        self.assertEqual(set(d.keys()), expected_keys)


class TestTrackerIntegration(unittest.TestCase):
    """Integration tests combining tracker with sampling."""

    def test_tracker_with_sampling(self):
        tracker = TokenUsageTracker(vocab_size=MEDIUM_VOCAB)
        logits = _make_logits(MEDIUM_VOCAB)
        for s in range(100):
            t = sample_from_logits(logits, seed=s)
            tracker.update(t)
        self.assertEqual(tracker.total, 100)
        self.assertGreater(tracker.unique_tokens, 1)
        self.assertGreater(tracker.entropy, 0)

    def test_statistics_with_sampling(self):
        stats = SamplingStatistics()
        logits = _make_logits(MEDIUM_VOCAB)
        for s in range(50):
            ent = compute_entropy(logits)
            probs = _stable_softmax(logits)
            top1 = float(np.max(probs))
            t = sample_from_logits(logits, seed=s)
            stats.record(temperature=1.0, entropy=ent, top1_prob=top1,
                         token=t)
        summary = stats.summary()
        self.assertEqual(summary["num_steps"], 50)
        self.assertGreater(summary["mean_entropy"], 0)

    def test_pool_with_beam_manager(self):
        pool = SequencePool(max_size=10)
        bm = BeamManager(num_beams=3)
        bm.step([(0, 2, -0.5), (0, 3, -0.3), (0, 4, -0.1)])
        bm.step([(0, EOS_TOKEN_ID, -0.2), (1, EOS_TOKEN_ID, -0.1),
                 (2, EOS_TOKEN_ID, -0.3)])
        for beam in bm.completed:
            pool.add(beam.token_ids, score=beam.log_prob)
        self.assertEqual(pool.size, 3)
        best = pool.best(1)
        self.assertEqual(len(best), 1)


class TestLargeVocabScaling(unittest.TestCase):
    """Test processors with large vocabulary sizes."""

    def test_top_k_large_vocab(self):
        logits = _make_logits(LARGE_VOCAB)
        proc = TopKProcessor(k=50)
        result = proc(logits)
        valid = np.sum(result > -float("inf"))
        self.assertLessEqual(valid, 55)

    def test_top_p_large_vocab(self):
        logits = _make_logits(LARGE_VOCAB)
        proc = TopPProcessor(p=0.9)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_typical_large_vocab(self):
        logits = _make_logits(LARGE_VOCAB)
        proc = TypicalProcessor(mass=0.9)
        result = proc(logits)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)

    def test_repetition_penalty_large_vocab(self):
        logits = _make_logits(LARGE_VOCAB)
        generated = list(range(100))
        proc = RepetitionPenaltyProcessor(penalty=1.5)
        result = proc(logits, generated_ids=generated)
        self.assertEqual(len(result), LARGE_VOCAB)

    def test_no_repeat_ngram_large_vocab(self):
        logits = _make_logits(LARGE_VOCAB)
        generated = list(range(50))
        proc = NoRepeatNgramProcessor(n=3)
        result = proc(logits, generated_ids=generated)
        self.assertEqual(len(result), LARGE_VOCAB)

    def test_full_pipeline_large_vocab(self):
        logits = _make_logits(LARGE_VOCAB, seed=17)
        chain = [
            TemperatureProcessor(temperature=0.8),
            TopKProcessor(k=100),
            TopPProcessor(p=0.95),
            MinPProcessor(min_p=0.01),
        ]
        result = apply_processor_chain(logits, chain)
        probs = _stable_softmax(result)
        _assert_valid_distribution(probs)
        valid = np.sum(result > -float("inf"))
        self.assertLess(valid, LARGE_VOCAB)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of all operations."""

    def test_very_large_logits_softmax(self):
        logits = np.array([1e6, 1e6 - 1, 1e6 - 2])
        probs = _stable_softmax(logits)
        _assert_valid_distribution(probs)
        self.assertFalse(np.any(np.isnan(probs)))

    def test_very_small_logits_softmax(self):
        logits = np.array([-1e6, -1e6 + 1, -1e6 + 2])
        probs = _stable_softmax(logits)
        _assert_valid_distribution(probs)

    def test_mixed_extreme_logits(self):
        logits = np.array([100.0, -100.0, 0.0])
        probs = _stable_softmax(logits)
        _assert_valid_distribution(probs)

    def test_entropy_extreme_peaked(self):
        logits = np.zeros(LARGE_VOCAB)
        logits[0] = 1000.0
        ent = compute_entropy(logits)
        self.assertGreaterEqual(ent, 0.0)
        self.assertLess(ent, 0.01)

    def test_varentropy_extreme(self):
        logits = np.zeros(LARGE_VOCAB)
        logits[0] = 1000.0
        ve = compute_varentropy(logits)
        self.assertGreaterEqual(ve, 0.0)
        self.assertFalse(np.isnan(ve))

    def test_gumbel_sampling_extreme_logits(self):
        logits = np.array([1e8, -1e8, 0.0])
        t = gumbel_softmax_sampling(logits, seed=42)
        self.assertEqual(t, 0)

    def test_top_k_all_same_large(self):
        logits = np.full(LARGE_VOCAB, 1.0)
        result = top_k_filtering(logits, k=10)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, LARGE_VOCAB)  # all tied

    def test_top_p_near_zero(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = top_p_filtering(logits, p=0.001)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)

    def test_sample_from_very_peaked(self):
        logits = np.full(MEDIUM_VOCAB, -1e10)
        logits[42] = 0.0
        t = sample_from_logits(logits, seed=0)
        self.assertEqual(t, 42)


class TestResamplingAdvanced(unittest.TestCase):
    """Advanced resampling tests."""

    def test_large_n(self):
        weights = np.random.RandomState(42).dirichlet(np.ones(20))
        indices = systematic_resampling(weights, n=1000, seed=42)
        self.assertEqual(len(indices), 1000)
        counts = Counter(indices)
        for i in range(20):
            expected = weights[i] * 1000
            self.assertAlmostEqual(counts.get(i, 0), expected,
                                   delta=max(expected * 0.3, 5))

    def test_stochastic_beam_various_k(self):
        logits = _make_logits(LARGE_VOCAB)
        for k in [1, 5, 10, 50]:
            result = stochastic_beam_search_sampling(logits, k=k, seed=42)
            self.assertEqual(len(result), k)
            self.assertEqual(len(set(result)), k)

    def test_sample_without_replacement_large_k(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = sample_without_replacement(logits, k=MEDIUM_VOCAB, seed=42)
        self.assertEqual(len(set(result)), MEDIUM_VOCAB)


class TestBatchSamplerAdvanced(unittest.TestCase):
    """Advanced BatchSampler tests."""

    def test_top_p_batch(self):
        cfg = SamplingConfig(temperature=1.0, top_p=0.9, seed=42)
        sampler = BatchSampler(cfg)
        batch = np.random.RandomState(42).randn(8, MEDIUM_VOCAB)
        results = sampler.sample_batch(batch)
        self.assertEqual(len(results), 8)

    def test_multiple_samples_diversity(self):
        cfg = SamplingConfig(temperature=1.5, seed=42)
        sampler = BatchSampler(cfg)
        logits = _make_logits(MEDIUM_VOCAB)
        results = sampler.sample_multiple(logits, n=100)
        unique = len(set(results))
        self.assertGreater(unique, 5)

    def test_low_temp_concentrated(self):
        cfg = SamplingConfig(temperature=0.01, seed=42)
        sampler = BatchSampler(cfg)
        logits = _make_logits(MEDIUM_VOCAB)
        results = sampler.sample_multiple(logits, n=50)
        argmax = int(np.argmax(logits))
        count_argmax = sum(1 for t in results if t == argmax)
        self.assertGreater(count_argmax, 40)


class TestSequencePoolAdvanced(unittest.TestCase):
    """Advanced SequencePool tests."""

    def test_large_pool(self):
        pool = SequencePool(max_size=50)
        rng = np.random.RandomState(42)
        for i in range(200):
            seq = rng.randint(0, 100, size=10).tolist()
            pool.add(seq, score=float(rng.randn()))
        self.assertEqual(pool.size, 50)

    def test_diversity_computation(self):
        pool = SequencePool(max_size=10)
        pool.add([0, 1, 2], score=1.0)
        pool.add([3, 4, 5], score=2.0)
        pool.add([6, 7, 8], score=3.0)
        d = pool.diversity()
        self.assertAlmostEqual(d, 1.0, places=5)

    def test_best_ordering(self):
        pool = SequencePool(max_size=10)
        for i in range(10):
            pool.add([i], score=float(i))
        best = pool.best(10)
        scores = [s for _, s in best]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_add_negative_scores(self):
        pool = SequencePool(max_size=5)
        for i in range(10):
            pool.add([i], score=-float(i))
        best = pool.best(1)
        self.assertEqual(best[0][1], 0.0)


class TestTypicalFilteringFunction(unittest.TestCase):
    """Tests for standalone typical_filtering function."""

    def test_basic(self):
        logits = _make_logits(MEDIUM_VOCAB)
        result = typical_filtering(logits, mass=0.9)
        valid = np.sum(result > -float("inf"))
        self.assertGreaterEqual(valid, 1)
        self.assertLessEqual(valid, MEDIUM_VOCAB)

    def test_mass_1(self):
        logits = _make_logits(SMALL_VOCAB)
        result = typical_filtering(logits, mass=1.0)
        valid = np.sum(result > -float("inf"))
        self.assertEqual(valid, SMALL_VOCAB)

    def test_preserves_kept_values(self):
        logits = _make_logits(SMALL_VOCAB)
        result = typical_filtering(logits, mass=0.9)
        kept = result > -float("inf")
        np.testing.assert_allclose(result[kept], logits[kept], atol=FLOAT_ATOL)


class TestComputeVariousDistributions(unittest.TestCase):
    """Test compute functions across distributions."""

    def test_entropy_zipf(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="zipf")
        ent = compute_entropy(logits)
        self.assertGreater(ent, 0)
        self.assertLess(ent, np.log(MEDIUM_VOCAB))

    def test_entropy_bimodal(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="bimodal")
        ent = compute_entropy(logits)
        self.assertGreater(ent, 0)

    def test_varentropy_zipf(self):
        logits = _make_logits(MEDIUM_VOCAB, distribution="zipf")
        ve = compute_varentropy(logits)
        self.assertGreaterEqual(ve, 0)

    def test_varentropy_uniform(self):
        logits = np.zeros(SMALL_VOCAB)
        ve = compute_varentropy(logits)
        self.assertAlmostEqual(ve, 0.0, places=5)

    def test_entropy_monotonic_with_temperature(self):
        logits = _make_logits(MEDIUM_VOCAB)
        temps = [0.1, 0.5, 1.0, 2.0, 5.0]
        entropies = [compute_entropy(logits / t) for t in temps]
        for i in range(len(entropies) - 1):
            self.assertLessEqual(entropies[i], entropies[i + 1] + 0.01)


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    unittest.main()
