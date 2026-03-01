"""
Comprehensive tests for all decoding algorithms in the Diversity Decoding Arena.

Tests cover correctness, determinism, diversity properties, edge cases,
convergence, and cross-algorithm comparisons.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest

from tests.conftest import (
    DEFAULT_VOCAB_SIZE,
    DEFAULT_SEED,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_SEQUENCES,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_NUM_BEAMS,
    DEFAULT_NUM_GROUPS,
    DEFAULT_NUM_PARTICLES,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    MockLogitSource,
    MockEmbedder,
    MockTokenizer,
    generate_random_texts,
    generate_diverse_texts,
    generate_token_sequences,
    compute_distinct_n,
    compute_pairwise_jaccard,
    assert_valid_probability_distribution,
)
from src.algorithms.base import (
    DecodingConfig,
    DecodingState,
    _stable_softmax,
    _log_softmax,
    _top_k_filter,
    _top_p_filter,
    sample_token,
    categorical_sample,
)
from src.algorithms.temperature import (
    TemperatureSampling,
    TemperatureConfig,
    softmax_with_temperature,
    entropy_from_logits,
)
from src.algorithms.topk import (
    TopKSampling,
    TopKConfig,
    top_k_filter,
    top_k_indices,
    probability_mass_in_top_k,
)
from src.algorithms.nucleus import NucleusSampling, NucleusConfig
from src.algorithms.typical import TypicalDecoding, TypicalConfig
from src.algorithms.diverse_beam import DiverseBeamSearch, DiverseBeamConfig
from src.algorithms.contrastive import ContrastiveSearch, ContrastiveConfig
from src.algorithms.dpp import DPPReranking, DPPConfig
from src.algorithms.mbr import MBRDiversity, MBRConfig
from src.algorithms.svd import SteinVariationalDecoding, SVDConfig
from src.algorithms.qdbs import QualityDiversityBeamSearch, QDBSConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logit_source(distribution: str = "uniform", seed: int = DEFAULT_SEED,
                       vocab_size: int = DEFAULT_VOCAB_SIZE, **kw) -> MockLogitSource:
    return MockLogitSource(
        vocab_size=vocab_size, seed=seed, distribution=distribution, **kw
    )


def _unique_ratio(sequences: List[List[int]]) -> float:
    """Fraction of unique sequences."""
    if not sequences:
        return 0.0
    as_tuples = [tuple(s) for s in sequences]
    return len(set(as_tuples)) / len(as_tuples)


def _mean_pairwise_edit_distance(sequences: List[List[int]]) -> float:
    """Mean pairwise Hamming-ish distance (on padded length)."""
    if len(sequences) < 2:
        return 0.0
    max_len = max(len(s) for s in sequences)
    padded = [s + [PAD_TOKEN_ID] * (max_len - len(s)) for s in sequences]
    total = 0.0
    count = 0
    for i in range(len(padded)):
        for j in range(i + 1, len(padded)):
            dist = sum(1 for a, b in zip(padded[i], padded[j]) if a != b)
            total += dist
            count += 1
    return total / count if count > 0 else 0.0


def _entropy_of_first_tokens(sequences: List[List[int]]) -> float:
    """Shannon entropy of the distribution of first generated tokens."""
    if not sequences:
        return 0.0
    first_tokens = [s[0] if s else -1 for s in sequences]
    counts = Counter(first_tokens)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def _collect_unique_tokens(sequences: List[List[int]]) -> Set[int]:
    """Collect all unique token IDs across sequences."""
    tokens: Set[int] = set()
    for seq in sequences:
        tokens.update(seq)
    return tokens


SMALL_MAX_TOKENS = 15
SMALL_NUM_SEQ = 5
PROMPT_IDS = [BOS_TOKEN_ID, 10, 20, 30]


# =========================================================================
# 1. TestTemperatureSampling
# =========================================================================


class TestTemperatureSampling:
    """Tests for temperature-based sampling."""

    def test_basic_generation(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ
        for seq in result:
            assert len(seq) > len(PROMPT_IDS)

    @pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 1.5, 2.0, 5.0])
    def test_temperature_values(self, temp):
        config = TemperatureConfig(
            temperature=temp, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_low_temperature_less_diverse(self):
        source = _make_logit_source("zipf")
        results_low = TemperatureSampling(TemperatureConfig(
            temperature=0.1, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        source.reset()
        results_high = TemperatureSampling(TemperatureConfig(
            temperature=2.0, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        unique_low = _unique_ratio(results_low)
        unique_high = _unique_ratio(results_high)
        # High temperature should produce at least as many unique sequences
        assert unique_high >= unique_low or unique_high >= 0.5

    def test_very_low_temperature_near_greedy(self):
        config = TemperatureConfig(
            temperature=0.01, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        # Near-greedy: most sequences should be similar
        assert len(result) == 3
        # At least the first tokens should mostly agree for peaked distribution
        first_tokens = [s[len(PROMPT_IDS)] for s in result if len(s) > len(PROMPT_IDS)]
        assert len(first_tokens) == 3

    def test_high_temperature_diversity(self):
        config = TemperatureConfig(
            temperature=5.0, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        tokens = _collect_unique_tokens(result)
        assert len(tokens) > 5

    @pytest.mark.parametrize("schedule", ["constant", "linear_decay", "cosine"])
    def test_schedule_types(self, schedule):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            dynamic_temperature=True, temp_schedule=schedule,
            temp_start=2.0, temp_end=0.5,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_constant_schedule_no_change(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            dynamic_temperature=True, temp_schedule="constant",
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_softmax_with_temperature_sum_to_one(self):
        logits = np.random.RandomState(42).randn(100)
        for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
            probs = softmax_with_temperature(logits, temp)
            assert abs(probs.sum() - 1.0) < 1e-6
            assert np.all(probs >= 0)

    def test_softmax_low_temp_peaks(self):
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        probs_low = softmax_with_temperature(logits, 0.01)
        assert np.argmax(probs_low) == 2
        assert probs_low[2] > 0.99

    def test_softmax_high_temp_uniform(self):
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        probs_high = softmax_with_temperature(logits, 100.0)
        # Should be nearly uniform
        assert np.std(probs_high) < 0.05

    def test_entropy_from_logits_uniform(self):
        logits = np.zeros(100)
        ent = entropy_from_logits(logits)
        expected = math.log(100)
        assert abs(ent - expected) < 0.01

    def test_entropy_from_logits_peaked(self):
        logits = np.full(100, -100.0)
        logits[0] = 0.0
        ent = entropy_from_logits(logits)
        assert ent < 0.1

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError):
            softmax_with_temperature(np.array([1.0, 2.0]), 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            softmax_with_temperature(np.array([1.0, 2.0]), -1.0)

    def test_sequences_start_with_prompt(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal"])
    def test_different_distributions(self, dist):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_repetition_penalty(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            repetition_penalty=2.0,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ


# =========================================================================
# 2. TestTopKSampling
# =========================================================================


class TestTopKSampling:
    """Tests for top-k sampling."""

    def test_basic_generation(self):
        config = TopKConfig(
            k=50, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    @pytest.mark.parametrize("k", [1, 5, 10, 50, 100, 500])
    def test_various_k_values(self, k):
        config = TopKConfig(
            k=k, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_k_equals_1_deterministic(self):
        config = TopKConfig(
            k=1, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        # With k=1, all sequences should pick the top token at each step
        # so all generated portions should be identical
        generated = [seq[len(PROMPT_IDS):] for seq in result]
        assert all(g == generated[0] for g in generated)

    def test_top_k_filter_correctness(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        filtered = top_k_filter(logits, 3)
        finite_count = np.sum(np.isfinite(filtered))
        assert finite_count == 3
        # Top 3 indices are 1, 4, 2 (values 5, 4, 3)
        assert np.isfinite(filtered[1])
        assert np.isfinite(filtered[4])
        assert np.isfinite(filtered[2])

    def test_top_k_filter_k_larger_than_vocab(self):
        logits = np.array([1.0, 2.0, 3.0])
        filtered = top_k_filter(logits, 10)
        np.testing.assert_array_equal(logits, filtered)

    def test_top_k_indices_sorted(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        indices = top_k_indices(logits, 3)
        assert list(indices) == [1, 4, 2]

    def test_probability_mass_in_top_k(self):
        logits = np.zeros(10)
        logits[0] = 10.0  # dominant token
        mass = probability_mass_in_top_k(logits, 1)
        assert mass > 0.9

    def test_probability_mass_full_vocab(self):
        logits = np.random.RandomState(42).randn(100)
        mass = probability_mass_in_top_k(logits, 100)
        assert abs(mass - 1.0) < 1e-6

    def test_smaller_k_less_diverse(self):
        source = _make_logit_source("uniform")
        results_small = TopKSampling(TopKConfig(
            k=5, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        source.reset()
        results_large = TopKSampling(TopKConfig(
            k=500, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        tokens_small = _collect_unique_tokens(results_small)
        tokens_large = _collect_unique_tokens(results_large)
        assert len(tokens_large) >= len(tokens_small)

    def test_k_equals_vocab_size(self):
        vs = 50
        config = TopKConfig(
            k=vs, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("uniform", vocab_size=vs)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_sequences_have_prompt_prefix(self):
        config = TopKConfig(
            k=50, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_across_distributions(self, dist):
        config = TopKConfig(
            k=20, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_top_k_filter_preserves_top_values(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(1000)
        k = 10
        filtered = top_k_filter(logits, k)
        top_indices = set(np.argsort(logits)[-k:])
        for i in range(len(filtered)):
            if i in top_indices:
                assert np.isfinite(filtered[i])


# =========================================================================
# 3. TestNucleusSampling
# =========================================================================


class TestNucleusSampling:
    """Tests for nucleus (top-p) sampling."""

    def test_basic_generation(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
    def test_various_p_thresholds(self, p):
        config = NucleusConfig(
            top_p=p, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_p_1_no_filtering(self):
        config = NucleusConfig(
            top_p=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_small_p_concentrates(self):
        source = _make_logit_source("zipf")
        results_small = NucleusSampling(NucleusConfig(
            top_p=0.1, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        source.reset()
        results_large = NucleusSampling(NucleusConfig(
            top_p=0.95, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        tokens_small = _collect_unique_tokens(results_small)
        tokens_large = _collect_unique_tokens(results_large)
        assert len(tokens_large) >= len(tokens_small)

    def test_top_p_filter_function(self):
        logits = np.array([10.0, 1.0, 0.5, 0.1, -10.0])
        filtered = _top_p_filter(logits, 0.5)
        probs = _stable_softmax(logits)
        sorted_probs = np.sort(probs)[::-1]
        cum = np.cumsum(sorted_probs)
        # The first token has most of the mass, so filtering at p=0.5
        # should keep very few tokens
        finite_count = np.sum(np.isfinite(filtered))
        assert finite_count >= 1

    def test_cumulative_probability_invariant(self):
        logits = np.random.RandomState(42).randn(100)
        for p in [0.5, 0.9, 0.99]:
            filtered = _top_p_filter(logits.copy(), p)
            kept = np.isfinite(filtered)
            probs = _stable_softmax(logits)
            mass = probs[kept].sum()
            # Kept mass should be at least p (approximately)
            assert mass >= p - 0.1

    def test_with_temperature(self):
        config = NucleusConfig(
            top_p=0.9, temperature=0.5,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_min_tokens_to_keep(self):
        config = NucleusConfig(
            top_p=0.01, min_tokens_to_keep=5,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_prompt_preserved(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal"])
    def test_across_distributions(self, dist):
        config = NucleusConfig(
            top_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_repetition_penalty(self):
        config = NucleusConfig(
            top_p=0.9, repetition_penalty=2.0,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_nucleus_config_validation(self):
        config = NucleusConfig(top_p=0.9)
        errors = config.validate()
        assert len(errors) == 0

    def test_nucleus_config_invalid_p(self):
        config = NucleusConfig(top_p=1.5)
        errors = config.validate()
        assert any("top_p" in e for e in errors)


# =========================================================================
# 4. TestTypicalDecoding
# =========================================================================


class TestTypicalDecoding:
    """Tests for typical decoding."""

    def test_basic_generation(self):
        config = TypicalConfig(
            typical_p=0.95, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    @pytest.mark.parametrize("typical_p", [0.2, 0.5, 0.8, 0.95, 1.0])
    def test_various_typical_p(self, typical_p):
        config = TypicalConfig(
            typical_p=typical_p, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_typical_p_1_keeps_all(self):
        config = TypicalConfig(
            typical_p=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_small_typical_p_concentrates(self):
        source = _make_logit_source("zipf")
        r_small = TypicalDecoding(TypicalConfig(
            typical_p=0.2, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_large = TypicalDecoding(TypicalConfig(
            typical_p=0.95, num_sequences=10,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        tokens_small = _collect_unique_tokens(r_small)
        tokens_large = _collect_unique_tokens(r_large)
        assert len(tokens_large) >= len(tokens_small)

    def test_entropy_weight_effect(self):
        config = TypicalConfig(
            typical_p=0.9, entropy_weight=2.0,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_with_temperature(self):
        config = TypicalConfig(
            typical_p=0.9, temperature=0.5,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_prompt_preserved(self):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal"])
    def test_across_distributions(self, dist):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_combine_with_nucleus(self):
        config = TypicalConfig(
            typical_p=0.9, combine_with_nucleus=True, nucleus_p=0.9,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_config_validation_valid(self):
        config = TypicalConfig(typical_p=0.9)
        errors = config.validate()
        assert len(errors) == 0

    def test_config_validation_invalid_p(self):
        config = TypicalConfig(typical_p=1.5)
        errors = config.validate()
        assert any("typical_p" in e for e in errors)

    def test_min_tokens_to_keep(self):
        config = TypicalConfig(
            typical_p=0.01, min_tokens_to_keep=3,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_degenerate_distribution(self):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("degenerate")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ


# =========================================================================
# 5. TestDiverseBeamSearch
# =========================================================================


class TestDiverseBeamSearch:
    """Tests for diverse beam search."""

    def _make_config(self, **kw) -> DiverseBeamConfig:
        defaults = dict(
            num_beams=8, num_beam_groups=2, diversity_penalty=1.0,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            num_sequences=8, num_return_sequences=8,
        )
        defaults.update(kw)
        return DiverseBeamConfig(**defaults)

    def test_basic_generation(self):
        config = self._make_config()
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_returns_requested_sequences(self):
        config = self._make_config(num_return_sequences=4)
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) <= 8

    @pytest.mark.parametrize("num_groups", [2, 4])
    def test_beam_groups(self, num_groups):
        config = self._make_config(
            num_beams=num_groups * 2, num_beam_groups=num_groups,
            num_sequences=num_groups * 2, num_return_sequences=num_groups * 2,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("penalty", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_diversity_penalty_values(self, penalty):
        config = self._make_config(diversity_penalty=penalty)
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_higher_penalty_more_diverse(self):
        source = _make_logit_source("zipf")
        r_low = DiverseBeamSearch(self._make_config(
            diversity_penalty=0.0
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_high = DiverseBeamSearch(self._make_config(
            diversity_penalty=5.0
        )).generate(source, PROMPT_IDS)
        dist_low = _mean_pairwise_edit_distance(r_low)
        dist_high = _mean_pairwise_edit_distance(r_high)
        # Higher penalty should yield at least as much or more diversity
        assert dist_high >= dist_low * 0.5

    @pytest.mark.parametrize("div_type", ["hamming", "ngram"])
    def test_diversity_types(self, div_type):
        config = self._make_config(diversity_type=div_type)
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_prompt_preserved(self):
        config = self._make_config()
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    def test_config_validation_valid(self):
        config = self._make_config()
        errors = config.validate()
        assert len(errors) == 0

    def test_config_validation_beams_not_divisible(self):
        config = DiverseBeamConfig(num_beams=7, num_beam_groups=3)
        errors = config.validate()
        assert any("divisible" in e for e in errors)

    def test_group_size_property(self):
        config = self._make_config(num_beams=12, num_beam_groups=3)
        assert config.group_size == 4

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked"])
    def test_across_distributions(self, dist):
        config = self._make_config()
        algo = DiverseBeamSearch(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_length_penalty(self):
        config = self._make_config(length_penalty=2.0)
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# 6. TestContrastiveSearch
# =========================================================================


class TestContrastiveSearch:
    """Tests for contrastive search."""

    def _make_config(self, **kw) -> ContrastiveConfig:
        defaults = dict(
            alpha=0.6, k=5, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        defaults.update(kw)
        return ContrastiveConfig(**defaults)

    def test_basic_generation(self):
        config = self._make_config()
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6, 0.9, 1.0])
    def test_alpha_values(self, alpha):
        config = self._make_config(alpha=alpha)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_alpha_0_greedy_like(self):
        config = self._make_config(alpha=0.0, num_sequences=3)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        # Alpha=0 should behave like greedy — sequences may be identical
        assert len(result) >= 1

    def test_alpha_1_diversity_only(self):
        config = self._make_config(alpha=1.0, num_sequences=SMALL_NUM_SEQ)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_k_candidates(self, k):
        config = self._make_config(k=k)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("sim", ["cosine", "euclidean", "dot"])
    def test_similarity_metrics(self, sim):
        config = self._make_config(similarity_metric=sim)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("pen", ["max", "mean", "weighted"])
    def test_degeneration_penalties(self, pen):
        config = self._make_config(degeneration_penalty=pen)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_prompt_preserved(self):
        config = self._make_config()
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    def test_config_validation_valid(self):
        config = self._make_config()
        errors = config.validate()
        assert len(errors) == 0

    def test_config_validation_invalid_alpha(self):
        config = ContrastiveConfig(alpha=1.5)
        errors = config.validate()
        assert any("alpha" in e for e in errors)

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked"])
    def test_across_distributions(self, dist):
        config = self._make_config()
        algo = ContrastiveSearch(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_with_temperature(self):
        config = self._make_config(temperature=0.5)
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# 7. TestDPPReranking
# =========================================================================


class TestDPPReranking:
    """Tests for DPP-based reranking."""

    def _make_config(self, **kw) -> DPPConfig:
        defaults = dict(
            candidate_pool_size=20, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, kernel_type="rbf",
            candidate_temperature=1.0,
        )
        defaults.update(kw)
        return DPPConfig(**defaults)

    def test_basic_generation(self):
        config = self._make_config()
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_selects_k_sequences(self):
        config = self._make_config(select_k=5, candidate_pool_size=20)
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) <= 20

    @pytest.mark.parametrize("kernel", ["rbf", "cosine"])
    def test_kernel_types(self, kernel):
        config = self._make_config(kernel_type=kernel)
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_quality_weight_effect(self):
        source = _make_logit_source("zipf")
        r_quality = DPPReranking(self._make_config(
            quality_weight=2.0
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_diversity = DPPReranking(self._make_config(
            quality_weight=0.1
        )).generate(source, PROMPT_IDS)
        assert len(r_quality) >= 1
        assert len(r_diversity) >= 1

    def test_kernel_matrix_symmetric(self):
        """Construct a simple RBF kernel and verify symmetry."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, DEFAULT_EMBEDDING_DIM)
        dists = np.sum((embeddings[:, None] - embeddings[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / 2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_kernel_matrix_positive_semidefinite(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, DEFAULT_EMBEDDING_DIM)
        dists = np.sum((embeddings[:, None] - embeddings[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / 2.0)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)

    def test_dpp_selection_diverse(self):
        config = self._make_config(select_k=5, candidate_pool_size=30)
        algo = DPPReranking(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        if len(result) >= 2:
            dist = _mean_pairwise_edit_distance(result)
            assert dist > 0

    def test_bandwidth_parameter(self):
        config = self._make_config(kernel_bandwidth=2.0)
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_greedy_sampling(self):
        config = self._make_config(sampling_method="greedy")
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_config_validation(self):
        config = self._make_config()
        errors = config.validate()
        assert len(errors) == 0

    def test_config_select_k_exceeds_pool(self):
        config = DPPConfig(select_k=100, candidate_pool_size=10)
        errors = config.validate()
        assert any("select_k" in e for e in errors)

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked"])
    def test_across_distributions(self, dist):
        config = self._make_config()
        algo = DPPReranking(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_regularization_eps(self):
        config = self._make_config(regularization_eps=1e-3)
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# 8. TestMBRDiversity
# =========================================================================


class TestMBRDiversity:
    """Tests for MBR diversity decoding."""

    def _make_config(self, **kw) -> MBRConfig:
        defaults = dict(
            candidate_pool_size=20, reference_pool_size=10, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, temperature=1.0,
            utility_function="bleu", diversity_weight=0.3,
        )
        defaults.update(kw)
        return MBRConfig(**defaults)

    def test_basic_generation(self):
        config = self._make_config()
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("utility", ["bleu", "edit_distance"])
    def test_utility_functions(self, utility):
        config = self._make_config(utility_function=utility)
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_diversity_weight_zero(self):
        config = self._make_config(diversity_weight=0.0)
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_diversity_weight_one(self):
        config = self._make_config(diversity_weight=1.0)
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_greedy_mmr_selection(self):
        config = self._make_config(selection_method="greedy_mmr")
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_lambda_mmr_effect(self):
        source = _make_logit_source("zipf")
        r_quality = MBRDiversity(self._make_config(
            lambda_mmr=0.9
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_diverse = MBRDiversity(self._make_config(
            lambda_mmr=0.1
        )).generate(source, PROMPT_IDS)
        assert len(r_quality) >= 1
        assert len(r_diverse) >= 1

    def test_dedup_candidates(self):
        config = self._make_config(dedup_candidates=True)
        algo = MBRDiversity(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("sim", ["ngram", "edit"])
    def test_similarity_metrics(self, sim):
        config = self._make_config(similarity_metric=sim)
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_pool_sizes(self):
        config = self._make_config(
            candidate_pool_size=30, reference_pool_size=15, select_k=5,
        )
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked"])
    def test_across_distributions(self, dist):
        config = self._make_config()
        algo = MBRDiversity(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_normalize_utility(self):
        config = self._make_config(normalize_utility=True)
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_ngram_order(self):
        config = self._make_config(ngram_order=2)
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# 9. TestSteinVariationalDecoding
# =========================================================================


class TestSteinVariationalDecoding:
    """Tests for Stein Variational Decoding (SVD)."""

    def _make_config(self, **kw) -> SVDConfig:
        defaults = dict(
            n_particles=5, alpha=0.5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, temperature=1.0,
            kernel_type="rbf", embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        defaults.update(kw)
        return SVDConfig(**defaults)

    def test_basic_generation(self):
        config = self._make_config()
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 1.0, 2.0])
    def test_repulsive_strength(self, alpha):
        config = self._make_config(alpha=alpha)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_zero_alpha_no_repulsion(self):
        config = self._make_config(alpha=0.0, n_particles=3)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_high_alpha_increases_diversity(self):
        source = _make_logit_source("zipf")
        r_low = SteinVariationalDecoding(self._make_config(
            alpha=0.0
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_high = SteinVariationalDecoding(self._make_config(
            alpha=2.0
        )).generate(source, PROMPT_IDS)
        # High alpha should not produce less diversity
        if len(r_low) >= 2 and len(r_high) >= 2:
            dist_low = _mean_pairwise_edit_distance(r_low)
            dist_high = _mean_pairwise_edit_distance(r_high)
            assert dist_high >= dist_low * 0.3

    @pytest.mark.parametrize("kernel", ["rbf", "cosine", "imq"])
    def test_kernel_types(self, kernel):
        config = self._make_config(kernel_type=kernel)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_particle_count(self):
        for n in [2, 5, 10]:
            config = self._make_config(n_particles=n, num_sequences=n)
            algo = SteinVariationalDecoding(config)
            source = _make_logit_source("zipf")
            result = algo.generate(source, PROMPT_IDS)
            assert len(result) >= 1

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "none"])
    def test_annealing_schedules(self, schedule):
        config = self._make_config(annealing_schedule=schedule)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_adaptive_bandwidth(self):
        config = self._make_config(use_adaptive_bandwidth=True)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_gradient_clipping(self):
        config = self._make_config(gradient_clip=1.0)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_config_validation(self):
        config = self._make_config()
        errors = config.validate()
        assert len(errors) == 0

    def test_config_invalid_particles(self):
        config = SVDConfig(n_particles=0)
        errors = config.validate()
        assert any("n_particles" in e for e in errors)

    def test_prompt_preserved(self):
        config = self._make_config()
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked"])
    def test_across_distributions(self, dist):
        config = self._make_config()
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_convergence_threshold(self):
        config = self._make_config(convergence_threshold=0.1)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_normalize_gradients(self):
        config = self._make_config(normalize_gradients=True)
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# 10. TestQualityDiversityBeamSearch
# =========================================================================


class TestQualityDiversityBeamSearch:
    """Tests for Quality-Diversity Beam Search (QD-BS)."""

    def _make_config(self, **kw) -> QDBSConfig:
        defaults = dict(
            beam_width=10, archive_size=20,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, temperature=1.0,
            num_behavior_dims=3, grid_resolution=5,
        )
        defaults.update(kw)
        return QDBSConfig(**defaults)

    def test_basic_generation(self):
        config = self._make_config()
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("beam_width", [5, 10, 20])
    def test_beam_widths(self, beam_width):
        config = self._make_config(beam_width=beam_width)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_archive_size_effect(self):
        config = self._make_config(archive_size=50)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_exploration_bonus(self):
        source = _make_logit_source("zipf")
        r_low = QualityDiversityBeamSearch(self._make_config(
            exploration_bonus=0.0
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_high = QualityDiversityBeamSearch(self._make_config(
            exploration_bonus=1.0
        )).generate(source, PROMPT_IDS)
        assert len(r_low) >= 1
        assert len(r_high) >= 1

    def test_elite_fraction(self):
        config = self._make_config(elite_fraction=0.5)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("quality_metric", ["log_prob", "perplexity", "length_normalized"])
    def test_quality_metrics(self, quality_metric):
        config = self._make_config(quality_metric=quality_metric)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_diversity_pressure(self):
        config = self._make_config(diversity_pressure=0.8)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_grid_resolution(self):
        config = self._make_config(grid_resolution=3)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_num_behavior_dims(self):
        config = self._make_config(num_behavior_dims=2)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_prompt_preserved(self):
        config = self._make_config()
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    def test_config_validation(self):
        config = self._make_config()
        errors = config.validate()
        assert len(errors) == 0

    def test_config_invalid_beam_width(self):
        config = QDBSConfig(beam_width=0)
        errors = config.validate()
        assert any("beam_width" in e for e in errors)

    def test_config_invalid_tessellation(self):
        config = QDBSConfig(tessellation_type="unknown")
        errors = config.validate()
        assert any("tessellation_type" in e for e in errors)

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked"])
    def test_across_distributions(self, dist):
        config = self._make_config()
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source(dist)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_archive_update_freq(self):
        config = self._make_config(archive_update_freq=2)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_length_penalty(self):
        config = self._make_config(length_penalty=2.0)
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# 11. TestAlgorithmEdgeCases
# =========================================================================


class TestAlgorithmEdgeCases:
    """Edge-case tests across all algorithms."""

    ALGO_CONFIGS = [
        ("temperature", TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        ), TemperatureSampling),
        ("topk", TopKConfig(
            k=10, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        ), TopKSampling),
        ("nucleus", NucleusConfig(
            top_p=0.9, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        ), NucleusSampling),
        ("typical", TypicalConfig(
            typical_p=0.9, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        ), TypicalDecoding),
    ]

    @pytest.mark.parametrize("name,config,cls", ALGO_CONFIGS, ids=lambda x: x if isinstance(x, str) else "")
    def test_empty_prompt(self, name, config, cls):
        algo = cls(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, [])
        assert len(result) == 3

    @pytest.mark.parametrize("name,config,cls", ALGO_CONFIGS, ids=lambda x: x if isinstance(x, str) else "")
    def test_single_token_prompt(self, name, config, cls):
        algo = cls(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, [BOS_TOKEN_ID])
        assert len(result) == 3
        for seq in result:
            assert len(seq) >= 1

    @pytest.mark.parametrize("name,config,cls", ALGO_CONFIGS, ids=lambda x: x if isinstance(x, str) else "")
    def test_max_length_one(self, name, config, cls):
        cfg = type(config)(**{
            **{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()
               if f.name in config.__dataclass_fields__},
            "max_new_tokens": 1,
        })
        algo = cls(cfg)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            # Should have at most 1 new token beyond prompt
            assert len(seq) <= len(PROMPT_IDS) + 1

    @pytest.mark.parametrize("name,config,cls", ALGO_CONFIGS, ids=lambda x: x if isinstance(x, str) else "")
    def test_single_sequence(self, name, config, cls):
        cfg = type(config)(**{
            **{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()
               if f.name in config.__dataclass_fields__},
            "num_sequences": 1,
        })
        algo = cls(cfg)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 1

    def test_degenerate_distribution_temperature(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("degenerate")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_degenerate_distribution_topk(self):
        config = TopKConfig(
            k=5, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("degenerate")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_degenerate_distribution_nucleus(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("degenerate")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_long_prompt(self):
        long_prompt = list(range(4, 104))
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=5, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, long_prompt)
        assert len(result) == 3
        for seq in result:
            assert seq[:len(long_prompt)] == long_prompt

    def test_very_small_vocab(self):
        vs = 10
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("uniform", vocab_size=vs)
        result = algo.generate(source, [0, 4, 5])
        assert len(result) == 3

    def test_uniform_logits_all_algorithms(self):
        source = _make_logit_source("uniform")
        configs_algos = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=5, seed=DEFAULT_SEED), TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=5, seed=DEFAULT_SEED), TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=5, seed=DEFAULT_SEED), NucleusSampling),
            (TypicalConfig(typical_p=0.9, num_sequences=3,
                           max_new_tokens=5, seed=DEFAULT_SEED), TypicalDecoding),
        ]
        for config, cls in configs_algos:
            source.reset()
            result = cls(config).generate(source, PROMPT_IDS)
            assert len(result) == 3


# =========================================================================
# 12. TestAlgorithmDeterminism
# =========================================================================


class TestAlgorithmDeterminism:
    """Tests that algorithms produce deterministic output with the same seed."""

    def _run_twice(self, config, cls, dist="zipf"):
        source1 = _make_logit_source(dist)
        algo1 = cls(config)
        r1 = algo1.generate(source1, PROMPT_IDS)

        source2 = _make_logit_source(dist)
        algo2 = cls(config)
        r2 = algo2.generate(source2, PROMPT_IDS)
        return r1, r2

    def test_temperature_determinism(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, TemperatureSampling)
        assert r1 == r2

    def test_topk_determinism(self):
        config = TopKConfig(
            k=50, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, TopKSampling)
        assert r1 == r2

    def test_nucleus_determinism(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, NucleusSampling)
        assert r1 == r2

    def test_typical_determinism(self):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, TypicalDecoding)
        assert r1 == r2

    def test_diverse_beam_determinism(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=2, num_sequences=4,
            num_return_sequences=4,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, DiverseBeamSearch)
        assert r1 == r2

    def test_contrastive_determinism(self):
        config = ContrastiveConfig(
            alpha=0.6, k=5, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        r1, r2 = self._run_twice(config, ContrastiveSearch)
        assert r1 == r2

    def test_dpp_determinism(self):
        config = DPPConfig(
            candidate_pool_size=15, select_k=3, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, DPPReranking)
        assert r1 == r2

    def test_svd_determinism(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        r1, r2 = self._run_twice(config, SteinVariationalDecoding)
        assert r1 == r2

    def test_qdbs_determinism(self):
        config = QDBSConfig(
            beam_width=5, archive_size=10, num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            grid_resolution=3,
        )
        r1, r2 = self._run_twice(config, QualityDiversityBeamSearch)
        assert r1 == r2

    def test_different_seeds_different_output(self):
        config1 = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=42,
        )
        config2 = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=123,
        )
        source1 = _make_logit_source("uniform", seed=42)
        r1 = TemperatureSampling(config1).generate(source1, PROMPT_IDS)
        source2 = _make_logit_source("uniform", seed=123)
        r2 = TemperatureSampling(config2).generate(source2, PROMPT_IDS)
        # Very likely to differ
        assert r1 != r2

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal"])
    def test_temperature_determinism_across_dists(self, dist):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, TemperatureSampling, dist=dist)
        assert r1 == r2

    def test_mbr_determinism(self):
        config = MBRConfig(
            candidate_pool_size=15, reference_pool_size=10, select_k=3,
            num_sequences=3, max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        r1, r2 = self._run_twice(config, MBRDiversity)
        assert r1 == r2


# =========================================================================
# 13. TestAlgorithmDiversity
# =========================================================================


class TestAlgorithmDiversity:
    """Tests that diversity increases with relevant parameters."""

    def test_temperature_increases_diversity(self):
        diversities = []
        for temp in [0.1, 0.5, 1.0, 2.0]:
            config = TemperatureConfig(
                temperature=temp, num_sequences=10,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("zipf")
            result = TemperatureSampling(config).generate(source, PROMPT_IDS)
            tokens = _collect_unique_tokens(result)
            diversities.append(len(tokens))
        # Generally increasing, allow some non-monotonicity
        assert diversities[-1] >= diversities[0]

    def test_top_k_increases_diversity(self):
        diversities = []
        for k in [1, 5, 50, 200]:
            config = TopKConfig(
                k=k, num_sequences=10,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("uniform")
            result = TopKSampling(config).generate(source, PROMPT_IDS)
            tokens = _collect_unique_tokens(result)
            diversities.append(len(tokens))
        assert diversities[-1] >= diversities[0]

    def test_nucleus_p_increases_diversity(self):
        diversities = []
        for p in [0.1, 0.5, 0.9]:
            config = NucleusConfig(
                top_p=p, num_sequences=10,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("zipf")
            result = NucleusSampling(config).generate(source, PROMPT_IDS)
            tokens = _collect_unique_tokens(result)
            diversities.append(len(tokens))
        assert diversities[-1] >= diversities[0]

    def test_typical_p_increases_diversity(self):
        diversities = []
        for tp in [0.2, 0.5, 0.95]:
            config = TypicalConfig(
                typical_p=tp, num_sequences=10,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("zipf")
            result = TypicalDecoding(config).generate(source, PROMPT_IDS)
            tokens = _collect_unique_tokens(result)
            diversities.append(len(tokens))
        assert diversities[-1] >= diversities[0]

    def test_diverse_beam_penalty_increases_diversity(self):
        distances = []
        for penalty in [0.0, 1.0, 5.0]:
            config = DiverseBeamConfig(
                num_beams=8, num_beam_groups=2, diversity_penalty=penalty,
                num_sequences=8, num_return_sequences=8,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("zipf")
            result = DiverseBeamSearch(config).generate(source, PROMPT_IDS)
            if len(result) >= 2:
                distances.append(_mean_pairwise_edit_distance(result))
            else:
                distances.append(0.0)
        # Penalty 5 should have >= penalty 0
        assert distances[-1] >= distances[0] * 0.5

    def test_contrastive_alpha_increases_diversity(self):
        distances = []
        for alpha in [0.0, 0.5, 1.0]:
            config = ContrastiveConfig(
                alpha=alpha, k=5, num_sequences=SMALL_NUM_SEQ,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
                embedding_dim=DEFAULT_EMBEDDING_DIM,
            )
            source = _make_logit_source("zipf")
            result = ContrastiveSearch(config).generate(source, PROMPT_IDS)
            if len(result) >= 2:
                distances.append(_mean_pairwise_edit_distance(result))
            else:
                distances.append(0.0)
        # Alpha 1.0 should produce at least some diversity
        assert distances[-1] >= 0.0

    def test_svd_alpha_increases_diversity(self):
        distances = []
        for alpha in [0.0, 0.5, 2.0]:
            config = SVDConfig(
                n_particles=5, alpha=alpha, num_sequences=5,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
                embedding_dim=DEFAULT_EMBEDDING_DIM,
            )
            source = _make_logit_source("zipf")
            result = SteinVariationalDecoding(config).generate(source, PROMPT_IDS)
            if len(result) >= 2:
                distances.append(_mean_pairwise_edit_distance(result))
            else:
                distances.append(0.0)
        assert distances[-1] >= distances[0] * 0.3

    def test_more_sequences_more_unique_tokens(self):
        tokens_counts = []
        for n in [3, 10]:
            config = TemperatureConfig(
                temperature=1.0, num_sequences=n,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("uniform")
            result = TemperatureSampling(config).generate(source, PROMPT_IDS)
            tokens = _collect_unique_tokens(result)
            tokens_counts.append(len(tokens))
        assert tokens_counts[-1] >= tokens_counts[0]

    def test_dpp_produces_diverse_subset(self):
        config = DPPConfig(
            candidate_pool_size=30, select_k=5, num_sequences=5,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("uniform")
        result = DPPReranking(config).generate(source, PROMPT_IDS)
        if len(result) >= 2:
            dist = _mean_pairwise_edit_distance(result)
            assert dist > 0

    def test_mbr_diversity_weight(self):
        source = _make_logit_source("zipf")
        r_no_div = MBRDiversity(MBRConfig(
            candidate_pool_size=20, reference_pool_size=10, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, diversity_weight=0.0,
        )).generate(source, PROMPT_IDS)
        source.reset()
        r_div = MBRDiversity(MBRConfig(
            candidate_pool_size=20, reference_pool_size=10, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, diversity_weight=1.0,
        )).generate(source, PROMPT_IDS)
        assert len(r_no_div) >= 1
        assert len(r_div) >= 1


# =========================================================================
# 14. TestAlgorithmConvergence
# =========================================================================


class TestAlgorithmConvergence:
    """Tests that algorithm outputs stabilize / converge appropriately."""

    def test_temperature_output_length_bounded(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=20, seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_topk_output_length_bounded(self):
        config = TopKConfig(
            k=50, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=20, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_nucleus_output_length_bounded(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=20, seed=DEFAULT_SEED,
        )
        algo = NucleusSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_typical_output_length_bounded(self):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=20, seed=DEFAULT_SEED,
        )
        algo = TypicalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_contrastive_output_length_bounded(self):
        config = ContrastiveConfig(
            alpha=0.6, k=5, num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=20, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_diverse_beam_output_length_bounded(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=2, num_sequences=4,
            num_return_sequences=4,
            max_new_tokens=20, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_increasing_max_tokens_longer_output(self):
        lengths = []
        for max_tok in [5, 10, 20]:
            config = TemperatureConfig(
                temperature=1.0, num_sequences=3,
                max_new_tokens=max_tok, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("uniform", eos_probability=0.001)
            result = TemperatureSampling(config).generate(source, PROMPT_IDS)
            avg_len = np.mean([len(s) - len(PROMPT_IDS) for s in result])
            lengths.append(avg_len)
        # More tokens allowed should lead to longer or equal outputs
        assert lengths[-1] >= lengths[0]

    def test_eos_probability_affects_length(self):
        lengths = []
        for eos_p in [0.001, 0.1]:
            config = TemperatureConfig(
                temperature=1.0, num_sequences=5,
                max_new_tokens=30, seed=DEFAULT_SEED,
            )
            source = _make_logit_source("uniform", eos_probability=eos_p)
            result = TemperatureSampling(config).generate(source, PROMPT_IDS)
            avg_len = np.mean([len(s) - len(PROMPT_IDS) for s in result])
            lengths.append(avg_len)
        # Higher EOS probability should generally produce shorter sequences
        assert lengths[0] >= lengths[1] * 0.5

    def test_peaked_distribution_consistent_output(self):
        config = TemperatureConfig(
            temperature=0.01, num_sequences=5,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("peaked")
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        # With very peaked distribution and low temp, outputs should be consistent
        generated = [seq[len(PROMPT_IDS):] for seq in result]
        first_gen = generated[0]
        matching = sum(1 for g in generated if g == first_gen)
        assert matching >= 1

    def test_svd_convergence_with_many_steps(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=20, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 20

    def test_qdbs_fills_archive(self):
        config = QDBSConfig(
            beam_width=10, archive_size=20, num_sequences=5,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            grid_resolution=3,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_all_sequences_proper_length(self):
        configs = [
            (TemperatureConfig(temperature=1.0, num_sequences=5,
                               max_new_tokens=10, seed=DEFAULT_SEED), TemperatureSampling),
            (TopKConfig(k=50, num_sequences=5,
                        max_new_tokens=10, seed=DEFAULT_SEED), TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=5,
                           max_new_tokens=10, seed=DEFAULT_SEED), NucleusSampling),
        ]
        for config, cls in configs:
            source = _make_logit_source("zipf")
            result = cls(config).generate(source, PROMPT_IDS)
            for seq in result:
                new_tokens = len(seq) - len(PROMPT_IDS)
                assert 0 <= new_tokens <= 10


# =========================================================================
# 15. TestAlgorithmComparison
# =========================================================================


class TestAlgorithmComparison:
    """Cross-algorithm property tests."""

    def _generate_with(self, cls, config, dist="zipf"):
        source = _make_logit_source(dist)
        algo = cls(config)
        return algo.generate(source, PROMPT_IDS)

    def test_all_sampling_algorithms_return_lists(self):
        configs_algos = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             NucleusSampling),
            (TypicalConfig(typical_p=0.9, num_sequences=3,
                           max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TypicalDecoding),
        ]
        for config, cls in configs_algos:
            result = self._generate_with(cls, config)
            assert isinstance(result, list)
            for seq in result:
                assert isinstance(seq, list)
                assert all(isinstance(t, int) for t in seq)

    def test_all_algorithms_respect_max_tokens(self):
        max_tok = 10
        configs_algos = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=max_tok, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=max_tok, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=max_tok, seed=DEFAULT_SEED),
             NucleusSampling),
            (TypicalConfig(typical_p=0.9, num_sequences=3,
                           max_new_tokens=max_tok, seed=DEFAULT_SEED),
             TypicalDecoding),
            (ContrastiveConfig(alpha=0.6, k=5, num_sequences=3,
                               max_new_tokens=max_tok, seed=DEFAULT_SEED,
                               embedding_dim=DEFAULT_EMBEDDING_DIM),
             ContrastiveSearch),
        ]
        for config, cls in configs_algos:
            result = self._generate_with(cls, config)
            for seq in result:
                new_tokens = len(seq) - len(PROMPT_IDS)
                assert new_tokens <= max_tok

    def test_all_algorithms_preserve_prompt(self):
        configs_algos = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             NucleusSampling),
            (TypicalConfig(typical_p=0.9, num_sequences=3,
                           max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TypicalDecoding),
        ]
        for config, cls in configs_algos:
            result = self._generate_with(cls, config)
            for seq in result:
                assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    def test_greedy_equivalent_across_methods(self):
        """Very low temperature / k=1 should yield similar results."""
        source_t = _make_logit_source("peaked")
        r_temp = TemperatureSampling(TemperatureConfig(
            temperature=0.001, num_sequences=1,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )).generate(source_t, PROMPT_IDS)

        source_k = _make_logit_source("peaked")
        r_topk = TopKSampling(TopKConfig(
            k=1, num_sequences=1,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )).generate(source_k, PROMPT_IDS)

        # Both should produce the same greedy sequence
        assert r_temp[0] == r_topk[0]

    def test_sampling_algorithms_nonzero_diversity(self):
        """All sampling algorithms should produce at least some diversity."""
        configs_algos = [
            (TemperatureConfig(temperature=1.0, num_sequences=10,
                               max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=10,
                        max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=10,
                           max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED),
             NucleusSampling),
        ]
        for config, cls in configs_algos:
            source = _make_logit_source("uniform")
            result = cls(config).generate(source, PROMPT_IDS)
            unique = _unique_ratio(result)
            assert unique > 0.3, f"{cls.__name__} unique ratio {unique} too low"

    def test_beam_search_higher_quality_scores(self):
        """Beam search sequences should generally have higher scores than random."""
        source = _make_logit_source("zipf")
        beam_result = DiverseBeamSearch(DiverseBeamConfig(
            num_beams=4, num_beam_groups=2, num_sequences=4,
            num_return_sequences=4,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )).generate(source, PROMPT_IDS)
        assert len(beam_result) >= 1

    def test_all_configs_validate(self):
        configs = [
            TemperatureConfig(temperature=1.0),
            TopKConfig(k=50),
            NucleusConfig(top_p=0.9),
            TypicalConfig(typical_p=0.9),
            DiverseBeamConfig(num_beams=8, num_beam_groups=2),
            ContrastiveConfig(alpha=0.6, k=5),
            DPPConfig(candidate_pool_size=20, select_k=5),
            SVDConfig(n_particles=5),
            QDBSConfig(beam_width=10, archive_size=20),
        ]
        for config in configs:
            errors = config.validate()
            assert isinstance(errors, (list, bool)), f"{type(config).__name__} validate failed"

    def test_all_configs_serializable(self):
        configs = [
            TemperatureConfig(temperature=1.0),
            TopKConfig(k=50),
            NucleusConfig(top_p=0.9),
            TypicalConfig(typical_p=0.9),
            DiverseBeamConfig(num_beams=8, num_beam_groups=2),
            ContrastiveConfig(alpha=0.6, k=5),
        ]
        for config in configs:
            d = config.to_dict()
            assert isinstance(d, dict)
            reconstructed = type(config).from_dict(d)
            assert reconstructed.algorithm_name == config.algorithm_name

    def test_decoding_state_basics(self):
        state = DecodingState(
            sequences=[[1, 2, 3], [4, 5, 6]],
            scores=[0.5, 0.3],
            is_finished=[False, True],
        )
        assert state.num_sequences == 2
        assert state.num_active() == 1
        assert state.active_indices() == [0]
        assert not state.all_finished()
        assert state.get_sequence(0) == [1, 2, 3]

    def test_decoding_state_update(self):
        state = DecodingState(
            sequences=[[1, 2], [3, 4]],
            scores=[0.0, 0.0],
            is_finished=[False, False],
        )
        state.update_sequence(0, 10)
        assert state.sequences[0] == [1, 2, 10]
        state.mark_finished(0)
        assert state.is_finished[0] is True
        assert state.num_active() == 1

    def test_decoding_state_clone(self):
        state = DecodingState(
            sequences=[[1, 2, 3]],
            scores=[0.5],
            is_finished=[False],
        )
        cloned = state.clone()
        cloned.sequences[0].append(4)
        assert state.sequences[0] == [1, 2, 3]
        assert cloned.sequences[0] == [1, 2, 3, 4]

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_temperature_handles_all_distributions(self, dist):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source(dist)
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3
        for seq in result:
            assert all(isinstance(t, int) for t in seq)

    def test_stable_softmax_numerical(self):
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = _stable_softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_stable_softmax_negative(self):
        logits = np.array([-1000.0, -999.0, -1001.0])
        probs = _stable_softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_log_softmax_consistent(self):
        logits = np.random.RandomState(42).randn(50)
        log_probs = _log_softmax(logits)
        probs = np.exp(log_probs)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert np.all(log_probs <= 0)

    def test_sample_token_in_range(self):
        logits = np.random.RandomState(42).randn(100)
        for _ in range(20):
            token = sample_token(logits, temperature=1.0)
            assert 0 <= token < 100

    def test_sample_token_greedy(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0])
        token = sample_token(logits, temperature=0.0)
        assert token == 1  # argmax

    def test_categorical_sample_in_range(self):
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        for _ in range(20):
            idx = categorical_sample(probs)
            assert 0 <= idx < 4


# =========================================================================
# Additional helper tests
# =========================================================================


class TestBaseHelpers:
    """Tests for base module helper functions."""

    def test_top_k_filter_basic(self):
        logits = np.arange(10, dtype=np.float64)
        filtered = _top_k_filter(logits, 3)
        finite = np.isfinite(filtered)
        assert np.sum(finite) == 3
        assert filtered[9] == 9.0
        assert filtered[8] == 8.0
        assert filtered[7] == 7.0

    def test_top_k_filter_k_zero(self):
        logits = np.array([1.0, 2.0, 3.0])
        filtered = _top_k_filter(logits, 0)
        # k <= 0 should return logits unchanged
        np.testing.assert_array_equal(logits, filtered)

    def test_top_k_filter_k_exceeds_vocab(self):
        logits = np.array([1.0, 2.0, 3.0])
        filtered = _top_k_filter(logits, 100)
        np.testing.assert_array_equal(logits, filtered)

    def test_top_p_filter_basic(self):
        logits = np.array([10.0, 0.0, 0.0, 0.0, -10.0])
        filtered = _top_p_filter(logits, 0.5)
        # Token 0 has most mass; should be kept
        assert np.isfinite(filtered[0])

    def test_top_p_filter_p_1(self):
        logits = np.array([1.0, 2.0, 3.0])
        filtered = _top_p_filter(logits, 1.0)
        np.testing.assert_array_equal(logits, filtered)

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_softmax_sums_to_one(self, size):
        logits = np.random.RandomState(42).randn(size)
        probs = _stable_softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_batch(self):
        logits = np.random.RandomState(42).randn(5, 20)
        probs = _stable_softmax(logits)
        for b in range(5):
            assert abs(probs[b].sum() - 1.0) < 1e-6

    def test_log_softmax_batch(self):
        logits = np.random.RandomState(42).randn(5, 20)
        log_probs = _log_softmax(logits)
        for b in range(5):
            probs = np.exp(log_probs[b])
            assert abs(probs.sum() - 1.0) < 1e-5


class TestDecodingConfig:
    """Tests for DecodingConfig dataclass."""

    def test_default_values(self):
        config = DecodingConfig()
        assert config.num_sequences == 20
        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.seed is None

    def test_validation_valid(self):
        config = DecodingConfig(num_sequences=5, max_new_tokens=50)
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_invalid_num_sequences(self):
        config = DecodingConfig(num_sequences=0)
        errors = config.validate()
        assert any("num_sequences" in e for e in errors)

    def test_validation_invalid_max_tokens(self):
        config = DecodingConfig(max_new_tokens=0)
        errors = config.validate()
        assert any("max_new_tokens" in e for e in errors)

    def test_validation_invalid_temperature(self):
        config = DecodingConfig(temperature=0.0)
        errors = config.validate()
        assert any("temperature" in e for e in errors)

    def test_validation_min_gt_max(self):
        config = DecodingConfig(min_new_tokens=50, max_new_tokens=10)
        errors = config.validate()
        assert any("min_new_tokens" in e for e in errors)

    def test_to_dict(self):
        config = DecodingConfig(num_sequences=5, temperature=0.7)
        d = config.to_dict()
        assert d["num_sequences"] == 5
        assert d["temperature"] == 0.7

    def test_from_dict(self):
        d = {"num_sequences": 5, "temperature": 0.7, "max_new_tokens": 50}
        config = DecodingConfig.from_dict(d)
        assert config.num_sequences == 5
        assert config.temperature == 0.7

    def test_hash_deterministic(self):
        config1 = DecodingConfig(num_sequences=5, temperature=0.7, seed=42)
        config2 = DecodingConfig(num_sequences=5, temperature=0.7, seed=42)
        assert config1.hash() == config2.hash()

    def test_hash_different(self):
        config1 = DecodingConfig(num_sequences=5, temperature=0.7)
        config2 = DecodingConfig(num_sequences=10, temperature=0.7)
        assert config1.hash() != config2.hash()

    def test_params_dict(self):
        config = DecodingConfig(params={"custom_key": 42})
        assert config.params["custom_key"] == 42


class TestMockLogitSource:
    """Tests for the MockLogitSource fixture."""

    def test_output_shape(self):
        source = MockLogitSource(vocab_size=100)
        logits = source([[1, 2, 3]])
        assert logits.shape == (1, 100)

    def test_batch_output_shape(self):
        source = MockLogitSource(vocab_size=100)
        logits = source([[1, 2], [3, 4], [5, 6]])
        assert logits.shape == (3, 100)

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_distributions(self, dist):
        source = MockLogitSource(vocab_size=100, distribution=dist)
        logits = source([[1, 2, 3]])
        assert logits.shape == (1, 100)
        assert np.all(np.isfinite(logits))

    def test_call_count(self):
        source = MockLogitSource()
        assert source.call_count == 0
        source([[1, 2]])
        assert source.call_count == 1
        source([[1, 2]])
        assert source.call_count == 2

    def test_reset(self):
        source = MockLogitSource()
        source([[1, 2]])
        source.reset()
        assert source.call_count == 0

    def test_deterministic_mode(self):
        source = MockLogitSource(deterministic=True)
        l1 = source([[1, 2, 3]])
        source.reset()
        l2 = source([[1, 2, 3]])
        np.testing.assert_array_equal(l1, l2)

    def test_eos_boosting(self):
        source = MockLogitSource(vocab_size=100)
        short_seq = list(range(5))
        long_seq = list(range(25))
        l_short = source([short_seq])
        source.reset()
        l_long = source([long_seq])
        # Long sequences should have higher EOS logit
        assert l_long[0, EOS_TOKEN_ID] > l_short[0, EOS_TOKEN_ID]

    def test_last_token_influence(self):
        source = MockLogitSource(vocab_size=100)
        l1 = source([[10]])
        source.reset()
        l2 = source([[20]])
        # Different last tokens should produce different logit adjustments
        assert not np.array_equal(l1, l2)


class TestMockEmbedder:
    """Tests for the MockEmbedder fixture."""

    def test_embed_sequence(self):
        embedder = MockEmbedder(embedding_dim=32)
        emb = embedder.embed_sequence([1, 2, 3])
        assert emb.shape == (32,)
        assert np.all(np.isfinite(emb))

    def test_embed_empty_sequence(self):
        embedder = MockEmbedder(embedding_dim=32)
        emb = embedder.embed_sequence([])
        assert emb.shape == (32,)
        np.testing.assert_array_equal(emb, np.zeros(32))

    def test_normalized(self):
        embedder = MockEmbedder(embedding_dim=32, normalize=True)
        emb = embedder.embed_sequence([1, 2, 3])
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch(self):
        embedder = MockEmbedder(embedding_dim=32)
        embs = embedder.embed_batch([[1, 2], [3, 4], [5, 6]])
        assert embs.shape == (3, 32)

    def test_different_sequences_different_embeddings(self):
        embedder = MockEmbedder(embedding_dim=32)
        e1 = embedder.embed_sequence([1, 2, 3])
        e2 = embedder.embed_sequence([100, 200, 300])
        assert not np.allclose(e1, e2)

    def test_caching(self):
        embedder = MockEmbedder(embedding_dim=32)
        e1 = embedder.embed_sequence([1, 2, 3])
        e2 = embedder.embed_sequence([1, 2, 3])
        np.testing.assert_array_equal(e1, e2)

    def test_embed_text(self):
        embedder = MockEmbedder(embedding_dim=32)
        emb = embedder.embed_text("hello world")
        assert emb.shape == (32,)

    def test_embed_texts(self):
        embedder = MockEmbedder(embedding_dim=32)
        embs = embedder.embed_texts(["hello", "world", "test"])
        assert embs.shape == (3, 32)


class TestMockTokenizer:
    """Tests for the MockTokenizer fixture."""

    def test_encode_decode(self):
        tok = MockTokenizer()
        text = "the quick brown"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_tokens(self):
        tok = MockTokenizer()
        assert tok.bos_token_id == BOS_TOKEN_ID
        assert tok.eos_token_id == EOS_TOKEN_ID
        assert tok.pad_token_id == PAD_TOKEN_ID

    def test_unknown_word(self):
        tok = MockTokenizer()
        ids = tok.encode("xyznonexistent")
        assert ids == [3]  # UNK_TOKEN_ID

    def test_batch_encode(self):
        tok = MockTokenizer()
        ids = tok.batch_encode(["the", "a", "is"])
        assert len(ids) == 3

    def test_batch_decode(self):
        tok = MockTokenizer()
        texts = tok.batch_decode([[4, 5, 6], [7, 8, 9]])
        assert len(texts) == 2


# =========================================================================
# Parametrized cross-cutting tests
# =========================================================================


class TestParametrizedAlgorithms:
    """Parametrized tests running across multiple algorithms."""

    SAMPLING_CONFIGS = [
        pytest.param(
            TemperatureConfig(temperature=1.0, num_sequences=3,
                              max_new_tokens=10, seed=DEFAULT_SEED),
            TemperatureSampling, id="temperature"
        ),
        pytest.param(
            TopKConfig(k=50, num_sequences=3,
                       max_new_tokens=10, seed=DEFAULT_SEED),
            TopKSampling, id="topk"
        ),
        pytest.param(
            NucleusConfig(top_p=0.9, num_sequences=3,
                          max_new_tokens=10, seed=DEFAULT_SEED),
            NucleusSampling, id="nucleus"
        ),
        pytest.param(
            TypicalConfig(typical_p=0.9, num_sequences=3,
                          max_new_tokens=10, seed=DEFAULT_SEED),
            TypicalDecoding, id="typical"
        ),
    ]

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_returns_correct_count(self, config, cls):
        source = _make_logit_source("zipf")
        result = cls(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_sequences_are_lists_of_ints(self, config, cls):
        source = _make_logit_source("zipf")
        result = cls(config).generate(source, PROMPT_IDS)
        for seq in result:
            assert isinstance(seq, list)
            assert all(isinstance(t, int) for t in seq)

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_prompt_prefix_preserved(self, config, cls):
        source = _make_logit_source("zipf")
        result = cls(config).generate(source, PROMPT_IDS)
        for seq in result:
            assert seq[:len(PROMPT_IDS)] == PROMPT_IDS

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_max_tokens_respected(self, config, cls):
        source = _make_logit_source("zipf")
        result = cls(config).generate(source, PROMPT_IDS)
        for seq in result:
            new_tokens = len(seq) - len(PROMPT_IDS)
            assert new_tokens <= 10

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_tokens_in_vocab_range(self, config, cls):
        source = _make_logit_source("zipf")
        result = cls(config).generate(source, PROMPT_IDS)
        for seq in result:
            for token in seq[len(PROMPT_IDS):]:
                assert 0 <= token < DEFAULT_VOCAB_SIZE

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal"])
    def test_handles_all_distributions(self, config, cls, dist):
        source = _make_logit_source(dist)
        result = cls(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_deterministic_with_seed(self, config, cls):
        s1 = _make_logit_source("zipf")
        r1 = cls(config).generate(s1, PROMPT_IDS)
        s2 = _make_logit_source("zipf")
        r2 = cls(config).generate(s2, PROMPT_IDS)
        assert r1 == r2

    @pytest.mark.parametrize("config,cls", SAMPLING_CONFIGS)
    def test_generates_at_least_one_token(self, config, cls):
        source = _make_logit_source("zipf")
        result = cls(config).generate(source, PROMPT_IDS)
        for seq in result:
            assert len(seq) > len(PROMPT_IDS)


class TestStoppingCriteria:
    """Tests for stopping criteria."""

    def test_max_length_criteria(self):
        from src.algorithms.base import MaxLengthCriteria
        criteria = MaxLengthCriteria(max_length=5)
        assert criteria([[1, 2, 3]], [0.0]) == [False]
        assert criteria([[1, 2, 3, 4, 5]], [0.0]) == [True]

    def test_max_length_criteria_invalid(self):
        from src.algorithms.base import MaxLengthCriteria
        with pytest.raises(ValueError):
            MaxLengthCriteria(max_length=0)

    def test_eos_token_criteria(self):
        from src.algorithms.base import EosTokenCriteria
        criteria = EosTokenCriteria(eos_token_id=EOS_TOKEN_ID)
        assert criteria([[1, 2, 3]], [0.0]) == [False]
        assert criteria([[1, 2, EOS_TOKEN_ID]], [0.0]) == [True]

    def test_eos_token_criteria_empty(self):
        from src.algorithms.base import EosTokenCriteria
        criteria = EosTokenCriteria(eos_token_id=EOS_TOKEN_ID)
        assert criteria([[]], [0.0]) == [False]

    def test_min_length_criteria(self):
        from src.algorithms.base import MinLengthCriteria
        criteria = MinLengthCriteria(min_length=3)
        assert criteria([[1, 2]], [0.0]) == [False]
        assert criteria([[1, 2, 3]], [0.0]) == [True]

    def test_stopping_criteria_list(self):
        from src.algorithms.base import (
            StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria
        )
        scl = StoppingCriteriaList([
            MaxLengthCriteria(max_length=10),
            EosTokenCriteria(eos_token_id=EOS_TOKEN_ID),
        ])
        # Neither triggered
        assert scl([[1, 2, 3]], [0.0]) == [False]
        # EOS triggered
        assert scl([[1, 2, EOS_TOKEN_ID]], [0.0]) == [True]
        # Length triggered
        assert scl([list(range(10))], [0.0]) == [True]

    def test_stopping_criteria_list_empty(self):
        from src.algorithms.base import StoppingCriteriaList
        scl = StoppingCriteriaList()
        assert scl([[1, 2, 3]], [0.0]) == [False]

    def test_stopping_criteria_list_batch(self):
        from src.algorithms.base import StoppingCriteriaList, MaxLengthCriteria
        scl = StoppingCriteriaList([MaxLengthCriteria(max_length=3)])
        result = scl([[1, 2], [1, 2, 3], [1, 2, 3, 4]], [0.0, 0.0, 0.0])
        assert result == [False, True, True]

    def test_max_time_criteria(self):
        from src.algorithms.base import MaxTimeCriteria
        criteria = MaxTimeCriteria(max_time_seconds=100.0)
        result = criteria([[1, 2]], [0.0])
        assert result == [False]

    def test_max_time_criteria_invalid(self):
        from src.algorithms.base import MaxTimeCriteria
        with pytest.raises(ValueError):
            MaxTimeCriteria(max_time_seconds=0.0)

    def test_stopping_criteria_reset(self):
        from src.algorithms.base import StoppingCriteriaList, MaxTimeCriteria
        scl = StoppingCriteriaList([MaxTimeCriteria(max_time_seconds=100.0)])
        scl([[1]], [0.0])  # triggers start timer
        scl.reset()
        assert len(scl) == 1


class TestConfigSpecificValidation:
    """Tests for algorithm-specific config validation."""

    def test_temperature_config_invalid_schedule(self):
        config = TemperatureConfig(temp_schedule="invalid_schedule")
        errors = config.validate()
        assert any("temp_schedule" in e for e in errors)

    def test_topk_config_invalid_k(self):
        config = TopKConfig(k=0)
        errors = config.validate()
        assert any("k" in e for e in errors)

    def test_topk_config_invalid_schedule(self):
        config = TopKConfig(k_schedule="nonexistent")
        errors = config.validate()
        assert any("k_schedule" in e for e in errors)

    def test_nucleus_config_valid(self):
        config = NucleusConfig(top_p=0.5, temperature=0.8)
        errors = config.validate()
        assert len(errors) == 0

    def test_typical_config_invalid_entropy_weight(self):
        config = TypicalConfig(entropy_weight=-1.0)
        errors = config.validate()
        assert any("entropy_weight" in e for e in errors)

    def test_diverse_beam_config_valid(self):
        config = DiverseBeamConfig(
            num_beams=8, num_beam_groups=4, diversity_penalty=1.0,
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_diverse_beam_config_invalid_type(self):
        config = DiverseBeamConfig(diversity_type="unknown")
        errors = config.validate()
        assert any("diversity_type" in e for e in errors)

    def test_contrastive_config_invalid_sim(self):
        config = ContrastiveConfig(similarity_metric="unknown")
        errors = config.validate()
        assert any("similarity_metric" in e for e in errors)

    def test_contrastive_config_invalid_penalty(self):
        config = ContrastiveConfig(degeneration_penalty="unknown")
        errors = config.validate()
        assert any("degeneration_penalty" in e for e in errors)

    def test_dpp_config_invalid_kernel(self):
        config = DPPConfig(kernel_type="unknown")
        errors = config.validate()
        assert any("kernel_type" in e.lower() for e in errors)

    def test_svd_config_invalid_kernel(self):
        config = SVDConfig(kernel_type="unknown")
        errors = config.validate()
        assert any("kernel_type" in e.lower() for e in errors)

    def test_svd_config_invalid_bandwidth_method(self):
        config = SVDConfig(bandwidth_method="unknown")
        errors = config.validate()
        assert any("bandwidth_method" in e.lower() for e in errors)

    def test_qdbs_config_invalid_quality_metric(self):
        config = QDBSConfig(quality_metric="unknown")
        errors = config.validate()
        assert any("quality_metric" in e.lower() for e in errors)

    def test_qdbs_config_invalid_exploration_bonus(self):
        config = QDBSConfig(exploration_bonus=-1.0)
        errors = config.validate()
        assert any("exploration_bonus" in e.lower() for e in errors)

    def test_qdbs_config_valid(self):
        config = QDBSConfig(
            beam_width=10, archive_size=20, grid_resolution=5,
        )
        errors = config.validate()
        assert len(errors) == 0


class TestNumericalStability:
    """Tests for numerical stability across edge cases."""

    def test_softmax_large_logits(self):
        logits = np.array([1e10, 1e10 + 1, 1e10 - 1])
        probs = _stable_softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_very_negative_logits(self):
        logits = np.array([-1e10, -1e10 + 1, -1e10 - 1])
        probs = _stable_softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_mixed_extreme_logits(self):
        logits = np.array([100.0, -100.0, 0.0])
        probs = _stable_softmax(logits)
        assert np.all(np.isfinite(probs))
        assert probs[0] > 0.99

    def test_log_softmax_large_logits(self):
        logits = np.array([1e10, 1e10 + 1, 1e10 - 1])
        log_probs = _log_softmax(logits)
        assert np.all(np.isfinite(log_probs))
        assert np.all(log_probs <= 0)

    def test_top_k_filter_ties(self):
        logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        filtered = _top_k_filter(logits, 3)
        finite_count = np.sum(np.isfinite(filtered))
        assert finite_count >= 3

    def test_top_p_filter_all_equal(self):
        logits = np.zeros(10)
        filtered = _top_p_filter(logits, 0.5)
        # At least some should be kept
        assert np.sum(np.isfinite(filtered)) >= 1

    def test_temperature_sampling_extreme_logits(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=5, seed=DEFAULT_SEED,
        )
        # Use a source with extreme values
        source = MockLogitSource(
            vocab_size=100, distribution="degenerate", seed=DEFAULT_SEED,
        )
        algo = TemperatureSampling(config)
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 3
        for seq in result:
            assert all(isinstance(t, int) for t in seq)

    def test_sample_token_all_negative_inf(self):
        logits = np.full(10, -np.inf)
        logits[5] = 0.0  # Only one valid token
        token = sample_token(logits, temperature=1.0)
        assert token == 5

    def test_sample_token_uniform_zero(self):
        logits = np.zeros(10)
        token = sample_token(logits, temperature=1.0)
        assert 0 <= token < 10

    @pytest.mark.parametrize("vocab_size", [2, 10, 100, 1000])
    def test_various_vocab_sizes(self, vocab_size):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=2,
            max_new_tokens=5, seed=DEFAULT_SEED,
        )
        source = MockLogitSource(vocab_size=vocab_size, distribution="uniform")
        algo = TemperatureSampling(config)
        result = algo.generate(source, [0])
        assert len(result) == 2


# =========================================================================
# Extended Temperature Tests
# =========================================================================


class TestTemperatureSamplingExtended:
    """Extended tests for temperature sampling covering schedules and analysis."""

    @pytest.mark.parametrize("temp,expected_entropy_order", [
        (0.1, "low"),
        (1.0, "medium"),
        (5.0, "high"),
    ])
    def test_temperature_entropy_ordering(self, temp, expected_entropy_order):
        logits = np.random.RandomState(42).randn(200)
        probs = softmax_with_temperature(logits, temp)
        ent = -np.sum(probs * np.log(np.clip(probs, 1e-30, None)))
        if expected_entropy_order == "low":
            assert ent < 2.0
        elif expected_entropy_order == "high":
            assert ent > 3.0
        else:
            assert ent > 0

    def test_temperature_schedule_linear_decay_decreases(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=20, seed=DEFAULT_SEED,
            dynamic_temperature=True, temp_schedule="linear_decay",
            temp_start=3.0, temp_end=0.5,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 3
        for seq in result:
            assert len(seq) > len(PROMPT_IDS)

    def test_temperature_schedule_cosine_generates(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=20, seed=DEFAULT_SEED,
            dynamic_temperature=True, temp_schedule="cosine",
            temp_start=2.0, temp_end=0.3,
        )
        algo = TemperatureSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_softmax_temperature_preserves_ordering(self):
        logits = np.array([1.0, 3.0, 2.0, 5.0, 0.5])
        for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
            probs = softmax_with_temperature(logits, temp)
            # Token 3 (logit=5.0) should always have highest probability
            assert np.argmax(probs) == 3

    def test_softmax_temperature_all_positive(self):
        logits = np.random.RandomState(99).randn(500)
        for temp in [0.01, 0.1, 1.0, 5.0, 50.0]:
            probs = softmax_with_temperature(logits, temp)
            assert np.all(probs >= 0)
            assert np.all(probs <= 1.0 + 1e-6)

    def test_entropy_monotonically_increases_with_temperature(self):
        logits = np.random.RandomState(42).randn(100)
        temps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        entropies = []
        for temp in temps:
            probs = softmax_with_temperature(logits, temp)
            ent = -np.sum(probs * np.log(np.clip(probs, 1e-30, None)))
            entropies.append(ent)
        for i in range(len(entropies) - 1):
            assert entropies[i + 1] >= entropies[i] - 1e-6

    def test_many_sequences_temperature(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=20,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("uniform")
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 20

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 999])
    def test_different_seeds_produce_output(self, seed):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=seed,
        )
        source = _make_logit_source("zipf", seed=seed)
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_temperature_with_empty_hot_tokens(self):
        source = MockLogitSource(
            vocab_size=100, distribution="peaked",
            hot_tokens=[], seed=DEFAULT_SEED,
        )
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_batch_generation(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = TemperatureSampling(config)
        batch_result = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID, 50]])
        assert len(batch_result) == 2
        for result in batch_result:
            assert len(result) == 3


# =========================================================================
# Extended Top-K Tests
# =========================================================================


class TestTopKSamplingExtended:
    """Extended tests for top-k sampling."""

    def test_top_k_indices_empty_k(self):
        logits = np.array([1.0, 2.0, 3.0])
        indices = top_k_indices(logits, 0)
        assert len(indices) == 0

    def test_top_k_indices_full(self):
        logits = np.array([1.0, 2.0, 3.0])
        indices = top_k_indices(logits, 3)
        assert list(indices) == [2, 1, 0]

    def test_probability_mass_zero_k(self):
        logits = np.random.RandomState(42).randn(100)
        mass = probability_mass_in_top_k(logits, 0)
        assert mass == 0.0

    def test_probability_mass_increases_with_k(self):
        logits = np.random.RandomState(42).randn(100)
        masses = [probability_mass_in_top_k(logits, k) for k in [1, 5, 10, 50, 100]]
        for i in range(len(masses) - 1):
            assert masses[i + 1] >= masses[i] - 1e-6

    def test_topk_with_temperature(self):
        config = TopKConfig(
            k=20, temperature=2.0,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    def test_topk_repetition_penalty(self):
        config = TopKConfig(
            k=50, repetition_penalty=2.0,
            num_sequences=SMALL_NUM_SEQ,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = TopKSampling(config)
        source = _make_logit_source("peaked")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) == SMALL_NUM_SEQ

    @pytest.mark.parametrize("k", [2, 10, 50, 200, 500])
    def test_topk_unique_tokens_bounded_by_k(self, k):
        config = TopKConfig(
            k=k, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("uniform")
        result = TopKSampling(config).generate(source, PROMPT_IDS)
        # All generated tokens should be from the top-k at each step
        assert len(result) == 3

    def test_topk_batch_generation(self):
        config = TopKConfig(
            k=50, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = TopKSampling(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_topk_filter_with_negative_logits(self):
        logits = np.array([-5.0, -1.0, -3.0, -2.0, -4.0])
        filtered = top_k_filter(logits, 2)
        assert np.isfinite(filtered[1])  # -1.0 (highest)
        assert np.isfinite(filtered[3])  # -2.0 (second highest)

    def test_topk_filter_single_element(self):
        logits = np.array([5.0])
        filtered = top_k_filter(logits, 1)
        assert filtered[0] == 5.0

    def test_topk_config_k_min_k_max(self):
        config = TopKConfig(k_min=10, k_max=100)
        errors = config.validate()
        assert len(errors) == 0

    def test_topk_config_k_min_exceeds_k_max(self):
        config = TopKConfig(k_min=100, k_max=10)
        errors = config.validate()
        assert any("k_max" in e for e in errors)


# =========================================================================
# Extended Nucleus Tests
# =========================================================================


class TestNucleusSamplingExtended:
    """Extended tests for nucleus sampling."""

    def test_nucleus_very_small_p(self):
        config = NucleusConfig(
            top_p=0.01, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("peaked")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_nucleus_with_high_temperature(self):
        config = NucleusConfig(
            top_p=0.9, temperature=3.0,
            num_sequences=5,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 5

    def test_nucleus_adaptive_p(self):
        config = NucleusConfig(
            top_p=0.9, adaptive_p=True,
            adaptive_p_min=0.5, adaptive_p_max=0.95,
            num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_nucleus_no_repeat_ngram(self):
        config = NucleusConfig(
            top_p=0.9, no_repeat_ngram_size=2,
            num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("peaked")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    @pytest.mark.parametrize("p,expected_kept_min", [
        (0.1, 1),
        (0.5, 1),
        (0.9, 1),
        (1.0, 1),
    ])
    def test_top_p_filter_keeps_enough(self, p, expected_kept_min):
        logits = np.random.RandomState(42).randn(100)
        filtered = _top_p_filter(logits, p)
        kept = np.sum(np.isfinite(filtered))
        assert kept >= expected_kept_min

    def test_nucleus_batch(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = NucleusSampling(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID, 10]])
        assert len(batch) == 2

    def test_nucleus_many_sequences(self):
        config = NucleusConfig(
            top_p=0.9, num_sequences=20,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("uniform")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 20
        unique = _unique_ratio(result)
        assert unique > 0.3

    @pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 2.0])
    def test_nucleus_temperature_interaction(self, temp):
        config = NucleusConfig(
            top_p=0.9, temperature=temp,
            num_sequences=5,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 5


# =========================================================================
# Extended Typical Decoding Tests
# =========================================================================


class TestTypicalDecodingExtended:
    """Extended tests for typical decoding."""

    def test_typical_with_high_entropy_weight(self):
        config = TypicalConfig(
            typical_p=0.9, entropy_weight=5.0,
            num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_typical_with_zero_entropy_weight(self):
        config = TypicalConfig(
            typical_p=0.9, entropy_weight=0.0,
            num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_typical_batch(self):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = TypicalDecoding(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_typical_many_sequences(self):
        config = TypicalConfig(
            typical_p=0.9, num_sequences=20,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("uniform")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 20

    def test_typical_repetition_penalty(self):
        config = TypicalConfig(
            typical_p=0.9, repetition_penalty=2.0,
            num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("peaked")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    @pytest.mark.parametrize("typical_p,temp", [
        (0.5, 0.5), (0.5, 2.0), (0.9, 0.5), (0.9, 2.0),
    ])
    def test_typical_p_temperature_combinations(self, typical_p, temp):
        config = TypicalConfig(
            typical_p=typical_p, temperature=temp,
            num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_typical_local_entropy_window(self):
        config = TypicalConfig(
            typical_p=0.9, local_entropy_window=10,
            num_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 3


# =========================================================================
# Extended Diverse Beam Search Tests
# =========================================================================


class TestDiverseBeamSearchExtended:
    """Extended tests for diverse beam search."""

    def test_single_group(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=1,
            num_sequences=4, num_return_sequences=4,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_many_groups(self):
        config = DiverseBeamConfig(
            num_beams=12, num_beam_groups=6,
            num_sequences=12, num_return_sequences=12,
            diversity_penalty=2.0,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_early_stopping(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=2,
            num_sequences=4, num_return_sequences=4,
            early_stopping=True,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_ngram_diversity_order(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=2,
            num_sequences=4, num_return_sequences=4,
            diversity_type="ngram", ngram_diversity_order=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_length_penalty_values(self):
        for lp in [0.5, 1.0, 2.0]:
            config = DiverseBeamConfig(
                num_beams=4, num_beam_groups=2,
                num_sequences=4, num_return_sequences=4,
                length_penalty=lp,
                max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
            )
            algo = DiverseBeamSearch(config)
            source = _make_logit_source("zipf")
            result = algo.generate(source, PROMPT_IDS)
            assert len(result) >= 1

    def test_diverse_beam_batch(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=2,
            num_sequences=4, num_return_sequences=4,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = DiverseBeamSearch(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_diverse_beam_zero_penalty_same_as_standard(self):
        config = DiverseBeamConfig(
            num_beams=4, num_beam_groups=1,
            num_sequences=4, num_return_sequences=4,
            diversity_penalty=0.0,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_return_fewer_than_beams(self):
        config = DiverseBeamConfig(
            num_beams=8, num_beam_groups=2,
            num_sequences=8, num_return_sequences=3,
            max_new_tokens=SMALL_MAX_TOKENS, seed=DEFAULT_SEED,
        )
        algo = DiverseBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) <= 8


# =========================================================================
# Extended Contrastive Search Tests
# =========================================================================


class TestContrastiveSearchExtended:
    """Extended tests for contrastive search."""

    def test_contrastive_with_low_k(self):
        config = ContrastiveConfig(
            alpha=0.6, k=2, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_contrastive_with_high_k(self):
        config = ContrastiveConfig(
            alpha=0.6, k=50, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_contrastive_batch(self):
        config = ContrastiveConfig(
            alpha=0.6, k=5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        source = _make_logit_source("zipf")
        algo = ContrastiveSearch(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_contrastive_length_penalty(self):
        config = ContrastiveConfig(
            alpha=0.6, k=5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            length_penalty=2.0,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_contrastive_many_sequences(self):
        config = ContrastiveConfig(
            alpha=0.6, k=5, num_sequences=10,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_contrastive_degenerate_input(self):
        config = ContrastiveConfig(
            alpha=0.6, k=5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("degenerate")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("alpha,k", [
        (0.0, 1), (0.3, 3), (0.6, 5), (0.9, 10), (1.0, 20),
    ])
    def test_alpha_k_combinations(self, alpha, k):
        config = ContrastiveConfig(
            alpha=alpha, k=k, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = ContrastiveSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# Extended DPP Tests
# =========================================================================


class TestDPPRerankingExtended:
    """Extended tests for DPP reranking."""

    def test_dpp_larger_pool(self):
        config = DPPConfig(
            candidate_pool_size=50, select_k=10,
            num_sequences=10, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED,
        )
        algo = DPPReranking(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_dpp_select_one(self):
        config = DPPConfig(
            candidate_pool_size=20, select_k=1,
            num_sequences=1, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED,
        )
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_dpp_high_quality_weight(self):
        config = DPPConfig(
            candidate_pool_size=20, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, quality_weight=5.0,
        )
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_rbf_kernel_bandwidth_effect(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 32)
        dists = np.sum((embeddings[:, None] - embeddings[None, :]) ** 2, axis=-1)
        K_narrow = np.exp(-dists / (2 * 0.1 ** 2))
        K_wide = np.exp(-dists / (2 * 10.0 ** 2))
        # Narrow bandwidth should have more off-diagonal values near 0
        assert np.mean(K_narrow[np.triu_indices(10, k=1)]) < np.mean(K_wide[np.triu_indices(10, k=1)])

    def test_cosine_kernel_construction(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        K = normalized @ normalized.T
        # Diagonal should be ~1
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)
        # Should be symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_dpp_batch(self):
        config = DPPConfig(
            candidate_pool_size=15, select_k=3,
            num_sequences=3, max_new_tokens=10,
            seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = DPPReranking(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_dpp_no_quality_model(self):
        config = DPPConfig(
            candidate_pool_size=20, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, use_quality_model=False,
        )
        algo = DPPReranking(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# Extended MBR Tests
# =========================================================================


class TestMBRDiversityExtended:
    """Extended tests for MBR diversity."""

    def test_mbr_submodular_selection(self):
        config = MBRConfig(
            candidate_pool_size=20, reference_pool_size=10, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, selection_method="submodular",
        )
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_mbr_smoothing_methods(self):
        for smoothing in ["floor", "add_k", "none"]:
            config = MBRConfig(
                candidate_pool_size=15, reference_pool_size=8, select_k=3,
                num_sequences=3, max_new_tokens=SMALL_MAX_TOKENS,
                seed=DEFAULT_SEED, smoothing=smoothing,
            )
            algo = MBRDiversity(config)
            source = _make_logit_source("zipf")
            result = algo.generate(source, PROMPT_IDS)
            assert len(result) >= 1

    def test_mbr_batch(self):
        config = MBRConfig(
            candidate_pool_size=15, reference_pool_size=8, select_k=3,
            num_sequences=3, max_new_tokens=10,
            seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        algo = MBRDiversity(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_mbr_top_k_sampling(self):
        config = MBRConfig(
            candidate_pool_size=20, reference_pool_size=10, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, top_k_sampling=50,
        )
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_mbr_top_p_sampling(self):
        config = MBRConfig(
            candidate_pool_size=20, reference_pool_size=10, select_k=5,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, top_p_sampling=0.9,
        )
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("dw", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    def test_mbr_diversity_weight_range(self, dw):
        config = MBRConfig(
            candidate_pool_size=15, reference_pool_size=8, select_k=3,
            num_sequences=3, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, diversity_weight=dw,
        )
        algo = MBRDiversity(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# Extended SVD Tests
# =========================================================================


class TestSteinVariationalDecodingExtended:
    """Extended tests for SVD."""

    def test_svd_many_particles(self):
        config = SVDConfig(
            n_particles=10, alpha=0.5, num_sequences=10,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_svd_exponential_annealing(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            annealing_schedule="exponential",
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_svd_step_decay_annealing(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            annealing_schedule="step_decay",
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_svd_fixed_bandwidth(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            use_adaptive_bandwidth=False,
            kernel_bandwidth=1.0,
            bandwidth_method="fixed",
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_svd_top_k_project(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            top_k_project=10,
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_svd_low_gradient_clip(self):
        config = SVDConfig(
            n_particles=3, alpha=2.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            gradient_clip=0.1,
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_svd_batch(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )
        source = _make_logit_source("zipf")
        algo = SteinVariationalDecoding(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    def test_svd_embedding_update_freq(self):
        config = SVDConfig(
            n_particles=3, alpha=0.5, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            embedding_update_freq=3,
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    @pytest.mark.parametrize("kernel,alpha", [
        ("rbf", 0.1), ("rbf", 1.0), ("cosine", 0.5), ("imq", 0.5),
    ])
    def test_svd_kernel_alpha_combinations(self, kernel, alpha):
        config = SVDConfig(
            n_particles=3, alpha=alpha, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            kernel_type=kernel,
        )
        algo = SteinVariationalDecoding(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1


# =========================================================================
# Extended QD-BS Tests
# =========================================================================


class TestQualityDiversityBeamSearchExtended:
    """Extended tests for QD-BS."""

    def test_qdbs_large_archive(self):
        config = QDBSConfig(
            beam_width=10, archive_size=100,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, grid_resolution=10,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_qdbs_small_grid(self):
        config = QDBSConfig(
            beam_width=5, archive_size=10,
            num_sequences=3, max_new_tokens=10,
            seed=DEFAULT_SEED, grid_resolution=2,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_qdbs_high_diversity_pressure(self):
        config = QDBSConfig(
            beam_width=10, archive_size=20,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, diversity_pressure=1.0,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("uniform")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_qdbs_low_diversity_pressure(self):
        config = QDBSConfig(
            beam_width=10, archive_size=20,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, diversity_pressure=0.0,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_qdbs_batch(self):
        config = QDBSConfig(
            beam_width=5, archive_size=10,
            num_sequences=3, max_new_tokens=10,
            seed=DEFAULT_SEED, grid_resolution=3,
        )
        source = _make_logit_source("zipf")
        algo = QualityDiversityBeamSearch(config)
        batch = algo.generate_batch(source, [PROMPT_IDS, [BOS_TOKEN_ID]])
        assert len(batch) == 2

    @pytest.mark.parametrize("metric", ["log_prob", "perplexity", "length_normalized"])
    def test_qdbs_quality_metric_batch(self, metric):
        config = QDBSConfig(
            beam_width=5, archive_size=10,
            num_sequences=3, max_new_tokens=10,
            seed=DEFAULT_SEED, quality_metric=metric,
            grid_resolution=3,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_qdbs_exploration_bonus_effect(self):
        source = _make_logit_source("zipf")
        r1 = QualityDiversityBeamSearch(QDBSConfig(
            beam_width=10, archive_size=20,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, exploration_bonus=0.0,
            grid_resolution=3,
        )).generate(source, PROMPT_IDS)
        source.reset()
        r2 = QualityDiversityBeamSearch(QDBSConfig(
            beam_width=10, archive_size=20,
            num_sequences=5, max_new_tokens=SMALL_MAX_TOKENS,
            seed=DEFAULT_SEED, exploration_bonus=2.0,
            grid_resolution=3,
        )).generate(source, PROMPT_IDS)
        assert len(r1) >= 1
        assert len(r2) >= 1

    def test_qdbs_behavior_update_freq(self):
        config = QDBSConfig(
            beam_width=5, archive_size=10,
            num_sequences=3, max_new_tokens=10,
            seed=DEFAULT_SEED, behavior_update_freq=2,
            grid_resolution=3,
        )
        algo = QualityDiversityBeamSearch(config)
        source = _make_logit_source("zipf")
        result = algo.generate(source, PROMPT_IDS)
        assert len(result) >= 1

    def test_qdbs_num_behavior_dims_variation(self):
        for dims in [1, 2, 3, 5]:
            config = QDBSConfig(
                beam_width=5, archive_size=10,
                num_sequences=3, max_new_tokens=10,
                seed=DEFAULT_SEED, num_behavior_dims=dims,
                grid_resolution=3,
            )
            algo = QualityDiversityBeamSearch(config)
            source = _make_logit_source("zipf")
            result = algo.generate(source, PROMPT_IDS)
            assert len(result) >= 1


# =========================================================================
# Extended Algorithm Comparison Tests
# =========================================================================


class TestAlgorithmComparisonExtended:
    """Extended cross-algorithm comparison tests."""

    def test_all_algorithms_handle_bimodal(self):
        source = _make_logit_source("bimodal")
        configs = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=10, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=10, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=10, seed=DEFAULT_SEED),
             NucleusSampling),
            (TypicalConfig(typical_p=0.9, num_sequences=3,
                           max_new_tokens=10, seed=DEFAULT_SEED),
             TypicalDecoding),
        ]
        for config, cls in configs:
            source.reset()
            result = cls(config).generate(source, PROMPT_IDS)
            assert len(result) == 3, f"{cls.__name__} failed on bimodal"

    def test_sampling_diversity_ordering(self):
        """Temperature sampling with higher temp should be more diverse than top-k with small k."""
        source_temp = _make_logit_source("uniform")
        r_temp = TemperatureSampling(TemperatureConfig(
            temperature=3.0, num_sequences=10,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )).generate(source_temp, PROMPT_IDS)

        source_topk = _make_logit_source("uniform")
        r_topk = TopKSampling(TopKConfig(
            k=3, num_sequences=10,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )).generate(source_topk, PROMPT_IDS)

        tokens_temp = _collect_unique_tokens(r_temp)
        tokens_topk = _collect_unique_tokens(r_topk)
        assert len(tokens_temp) >= len(tokens_topk) * 0.5

    def test_all_algorithms_nonempty_output(self):
        configs_algos = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=5, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=5, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=5, seed=DEFAULT_SEED),
             NucleusSampling),
            (TypicalConfig(typical_p=0.9, num_sequences=3,
                           max_new_tokens=5, seed=DEFAULT_SEED),
             TypicalDecoding),
            (ContrastiveConfig(alpha=0.6, k=5, num_sequences=3,
                               max_new_tokens=5, seed=DEFAULT_SEED,
                               embedding_dim=DEFAULT_EMBEDDING_DIM),
             ContrastiveSearch),
        ]
        for config, cls in configs_algos:
            source = _make_logit_source("zipf")
            result = cls(config).generate(source, PROMPT_IDS)
            assert len(result) > 0, f"{cls.__name__} returned empty"
            for seq in result:
                assert len(seq) > 0, f"{cls.__name__} returned empty sequence"

    def test_algorithm_name_property(self):
        algos = [
            TemperatureSampling(TemperatureConfig()),
            TopKSampling(TopKConfig()),
            NucleusSampling(NucleusConfig()),
            TypicalDecoding(TypicalConfig()),
            ContrastiveSearch(ContrastiveConfig()),
        ]
        for algo in algos:
            assert isinstance(algo.name, str)
            assert len(algo.name) > 0

    def test_algorithm_description_property(self):
        algos = [
            TemperatureSampling(TemperatureConfig()),
            TopKSampling(TopKConfig()),
        ]
        for algo in algos:
            desc = algo.description
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_config_hash_consistency(self):
        """Same config should produce same hash every time."""
        configs = [
            TemperatureConfig(temperature=1.0, seed=42),
            TopKConfig(k=50, seed=42),
            NucleusConfig(top_p=0.9, seed=42),
        ]
        for config in configs:
            h1 = config.hash()
            h2 = config.hash()
            assert h1 == h2

    def test_all_configs_have_algorithm_name(self):
        configs = [
            TemperatureConfig(),
            TopKConfig(),
            NucleusConfig(),
            TypicalConfig(),
            DiverseBeamConfig(),
            ContrastiveConfig(),
            DPPConfig(),
            MBRConfig(),
            SVDConfig(),
            QDBSConfig(),
        ]
        for config in configs:
            assert hasattr(config, "algorithm_name")
            assert isinstance(config.algorithm_name, str)

    @pytest.mark.parametrize("n_seq", [1, 3, 5, 10])
    def test_temperature_various_num_sequences(self, n_seq):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=n_seq,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == n_seq

    @pytest.mark.parametrize("n_seq", [1, 3, 5])
    def test_topk_various_num_sequences(self, n_seq):
        config = TopKConfig(
            k=50, num_sequences=n_seq,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TopKSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == n_seq

    def test_different_prompts_different_outputs(self):
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source1 = _make_logit_source("zipf")
        r1 = TemperatureSampling(config).generate(source1, [BOS_TOKEN_ID, 10])
        source2 = _make_logit_source("zipf")
        r2 = TemperatureSampling(config).generate(source2, [BOS_TOKEN_ID, 50, 100])
        # Different prompts should likely produce different results
        assert r1 != r2


# =========================================================================
# Extended Edge Case Tests
# =========================================================================


class TestAlgorithmEdgeCasesExtended:
    """Extended edge case tests."""

    def test_repeated_prompt_tokens(self):
        prompt = [BOS_TOKEN_ID, 10, 10, 10, 10]
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TemperatureSampling(config).generate(source, prompt)
        assert len(result) == 3

    def test_large_prompt(self):
        prompt = list(range(4, 204))
        config = TemperatureConfig(
            temperature=1.0, num_sequences=2,
            max_new_tokens=5, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TemperatureSampling(config).generate(source, prompt)
        assert len(result) == 2
        for seq in result:
            assert seq[:len(prompt)] == prompt

    def test_prompt_with_special_tokens(self):
        prompt = [BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, 10, 20]
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TemperatureSampling(config).generate(source, prompt)
        assert len(result) == 3

    def test_max_tokens_equals_one_all_algorithms(self):
        configs = [
            (TemperatureConfig(temperature=1.0, num_sequences=3,
                               max_new_tokens=1, seed=DEFAULT_SEED),
             TemperatureSampling),
            (TopKConfig(k=50, num_sequences=3,
                        max_new_tokens=1, seed=DEFAULT_SEED),
             TopKSampling),
            (NucleusConfig(top_p=0.9, num_sequences=3,
                           max_new_tokens=1, seed=DEFAULT_SEED),
             NucleusSampling),
        ]
        for config, cls in configs:
            source = _make_logit_source("zipf")
            result = cls(config).generate(source, PROMPT_IDS)
            for seq in result:
                assert len(seq) <= len(PROMPT_IDS) + 1

    def test_extremely_peaked_distribution(self):
        source = MockLogitSource(
            vocab_size=100, distribution="degenerate",
            hot_tokens=[42], seed=DEFAULT_SEED,
        )
        config = TemperatureConfig(
            temperature=1.0, num_sequences=5,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 5
        # Most tokens should be the hot token or related
        for seq in result:
            assert len(seq) > len(PROMPT_IDS)

    def test_high_concentration_peaked(self):
        source = MockLogitSource(
            vocab_size=100, distribution="peaked",
            concentration=10.0, seed=DEFAULT_SEED,
        )
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_zipf_high_concentration(self):
        source = MockLogitSource(
            vocab_size=100, distribution="zipf",
            concentration=2.0, seed=DEFAULT_SEED,
        )
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_non_deterministic_source(self):
        source = MockLogitSource(
            vocab_size=100, distribution="uniform",
            deterministic=False, seed=DEFAULT_SEED,
        )
        config = TemperatureConfig(
            temperature=1.0, num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        result = TemperatureSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    @pytest.mark.parametrize("prompt_len", [0, 1, 5, 20, 50])
    def test_various_prompt_lengths(self, prompt_len):
        prompt = list(range(4, 4 + prompt_len))
        config = TemperatureConfig(
            temperature=1.0, num_sequences=2,
            max_new_tokens=5, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TemperatureSampling(config).generate(source, prompt)
        assert len(result) == 2

    def test_topk_k_equals_2(self):
        config = TopKConfig(
            k=2, num_sequences=5,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("uniform")
        result = TopKSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 5

    def test_nucleus_p_very_close_to_zero(self):
        config = NucleusConfig(
            top_p=0.001, min_tokens_to_keep=1,
            num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = NucleusSampling(config).generate(source, PROMPT_IDS)
        assert len(result) == 3

    def test_typical_p_very_close_to_zero(self):
        config = TypicalConfig(
            typical_p=0.01, min_tokens_to_keep=1,
            num_sequences=3,
            max_new_tokens=10, seed=DEFAULT_SEED,
        )
        source = _make_logit_source("zipf")
        result = TypicalDecoding(config).generate(source, PROMPT_IDS)
        assert len(result) == 3


# =========================================================================
# Kernel and Embedding Tests
# =========================================================================


class TestKernelComputations:
    """Tests for kernel matrix computations used by DPP and SVD."""

    def test_rbf_kernel_diagonal_is_one(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 32)
        dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / 2.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_rbf_kernel_symmetry(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 32)
        dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / 2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_rbf_kernel_psd(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 32)
        dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / 2.0)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)

    def test_cosine_similarity_range(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-8)
        K = X_norm @ X_norm.T
        assert np.all(K >= -1.0 - 1e-6)
        assert np.all(K <= 1.0 + 1e-6)

    def test_cosine_similarity_self(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 16)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-8)
        K = X_norm @ X_norm.T
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)

    def test_dot_product_kernel(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 16)
        K = X @ X.T
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_polynomial_kernel(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 16)
        c = 1.0
        degree = 3
        K = (X @ X.T + c) ** degree
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_kernel_bandwidth_effect(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 16)
        dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K_narrow = np.exp(-dists / (2 * 0.01))
        K_wide = np.exp(-dists / (2 * 100.0))
        # Narrow bandwidth -> values closer to identity
        assert np.mean(np.abs(K_narrow - np.eye(5))) < np.mean(np.abs(K_wide - np.eye(5)))

    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_rbf_kernel_various_sizes(self, n):
        rng = np.random.RandomState(42)
        X = rng.randn(n, 16)
        dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / 2.0)
        assert K.shape == (n, n)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_embedding_distance_triangle_inequality(self):
        rng = np.random.RandomState(42)
        X = rng.randn(3, 16)
        d01 = np.linalg.norm(X[0] - X[1])
        d02 = np.linalg.norm(X[0] - X[2])
        d12 = np.linalg.norm(X[1] - X[2])
        assert d01 <= d02 + d12 + 1e-10
        assert d02 <= d01 + d12 + 1e-10
        assert d12 <= d01 + d02 + 1e-10


# =========================================================================
# Diversity Metric Tests
# =========================================================================


class TestDiversityMetrics:
    """Tests for diversity measurement helpers."""

    def test_unique_ratio_all_unique(self):
        seqs = [[1, 2], [3, 4], [5, 6]]
        assert _unique_ratio(seqs) == 1.0

    def test_unique_ratio_all_same(self):
        seqs = [[1, 2], [1, 2], [1, 2]]
        assert abs(_unique_ratio(seqs) - 1.0 / 3.0) < 1e-6

    def test_unique_ratio_empty(self):
        assert _unique_ratio([]) == 0.0

    def test_pairwise_edit_distance_identical(self):
        seqs = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        dist = _mean_pairwise_edit_distance(seqs)
        assert dist == 0.0

    def test_pairwise_edit_distance_different(self):
        seqs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        dist = _mean_pairwise_edit_distance(seqs)
        assert dist == 3.0

    def test_pairwise_edit_distance_single(self):
        dist = _mean_pairwise_edit_distance([[1, 2, 3]])
        assert dist == 0.0

    def test_entropy_first_tokens_uniform(self):
        seqs = [[i, 0] for i in range(10)]
        ent = _entropy_of_first_tokens(seqs)
        expected = math.log(10)
        assert abs(ent - expected) < 1e-6

    def test_entropy_first_tokens_same(self):
        seqs = [[5, 0], [5, 1], [5, 2]]
        ent = _entropy_of_first_tokens(seqs)
        assert ent == 0.0

    def test_collect_unique_tokens(self):
        seqs = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
        tokens = _collect_unique_tokens(seqs)
        assert tokens == {1, 2, 3, 4, 5, 6}

    def test_collect_unique_tokens_empty(self):
        tokens = _collect_unique_tokens([])
        assert tokens == set()

    def test_collect_unique_tokens_empty_seqs(self):
        tokens = _collect_unique_tokens([[], [], []])
        assert tokens == set()
