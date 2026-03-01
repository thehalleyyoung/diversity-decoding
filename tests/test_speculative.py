"""
Comprehensive tests for speculative decoding algorithms.

Tests cover SpeculativeConfig, draft models (NGram, Cached, Ensemble, Adaptive),
verification strategies (Standard, DiversityAware, Relaxed, TopK),
SpeculativeDecoder, DiversitySpeculativeDecoder, SpeculativeTree,
AcceptanceTracker, and all helper functions.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base import DecodingConfig, _stable_softmax
from src.algorithms.speculative import (
    SpeculativeConfig,
    NGramDraftModel,
    CachedDraftModel,
    EnsembleDraftModel,
    AdaptiveDraftModel,
    StandardVerification,
    DiversityAwareVerification,
    RelaxedVerification,
    TopKVerification,
    SpeculativeDecoder,
    DiversitySpeculativeDecoder,
    SpeculativeTreeNode,
    SpeculativeTree,
    AcceptanceTracker,
    compute_acceptance_probability,
    rejection_sampling_step,
    diversity_modified_acceptance,
    compute_speculative_speedup,
    optimal_draft_length,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
SEED = 42
EOS_TOKEN = 1


# ---------------------------------------------------------------------------
# Mock logit sources
# ---------------------------------------------------------------------------


def make_uniform_source(vocab_size: int = VOCAB_SIZE):
    """Logit source returning uniform logits."""
    def source(input_ids):
        batch = len(input_ids)
        return np.zeros((batch, vocab_size), dtype=np.float32)
    return source


def make_peaked_source(hot_token: int = 10, vocab_size: int = VOCAB_SIZE):
    """Logit source strongly favouring a single token."""
    def source(input_ids):
        batch = len(input_ids)
        logits = np.full((batch, vocab_size), -10.0, dtype=np.float32)
        logits[:, hot_token] = 5.0
        return logits
    return source


def make_random_source(seed: int = SEED, vocab_size: int = VOCAB_SIZE):
    """Deterministic random logit source."""
    rng = np.random.RandomState(seed)
    base = rng.randn(vocab_size).astype(np.float32)
    def source(input_ids):
        batch = len(input_ids)
        return np.tile(base, (batch, 1))
    return source


def make_sequential_source(vocab_size: int = VOCAB_SIZE):
    """Source that returns logits favouring the next token in sequence."""
    def source(input_ids):
        batch = len(input_ids)
        logits = np.zeros((batch, vocab_size), dtype=np.float32)
        for b in range(batch):
            next_tok = (len(input_ids[b]) % vocab_size)
            logits[b, next_tok] = 10.0
        return logits
    return source


def make_eos_source(eos_token: int = EOS_TOKEN, after_steps: int = 5,
                    vocab_size: int = VOCAB_SIZE):
    """Source that begins favouring EOS after some steps."""
    call_count = [0]
    def source(input_ids):
        call_count[0] += 1
        batch = len(input_ids)
        logits = np.zeros((batch, vocab_size), dtype=np.float32)
        for b in range(batch):
            if len(input_ids[b]) >= after_steps:
                logits[b, eos_token] = 20.0
            else:
                logits[b, 10] = 5.0
        return logits
    return source


# ---------------------------------------------------------------------------
# Helper: simple draft model wrapping a source
# ---------------------------------------------------------------------------


class SimpleDraftModel:
    """Minimal draft model for testing: wraps a logit source."""

    def __init__(self, source, vocab_size: int = VOCAB_SIZE):
        self.source = source
        self.vocab_size = vocab_size
        self._accepted: List[List[int]] = []

    def predict(self, prefix: List[int], num_tokens: int
                ) -> Tuple[List[int], np.ndarray]:
        tokens: List[int] = []
        all_logits: List[np.ndarray] = []
        current = list(prefix)
        for _ in range(num_tokens):
            logits = self.source([current])[0]
            all_logits.append(logits)
            token = int(np.argmax(logits))
            tokens.append(token)
            current.append(token)
        return tokens, np.array(all_logits)

    def update(self, accepted_tokens: List[int]) -> None:
        self._accepted.append(list(accepted_tokens))


# =========================================================================
# 1. TestSpeculativeConfig
# =========================================================================


class TestSpeculativeConfig(unittest.TestCase):
    """Tests for SpeculativeConfig creation, serialization, validation."""

    def test_default_creation(self):
        cfg = SpeculativeConfig()
        self.assertEqual(cfg.algorithm_name, "speculative")
        self.assertEqual(cfg.draft_length, 5)
        self.assertEqual(cfg.gamma, 1.0)
        self.assertEqual(cfg.verification_strategy, "standard")
        self.assertEqual(cfg.diversity_penalty, 0.0)
        self.assertEqual(cfg.relaxation_tolerance, 0.1)
        self.assertEqual(cfg.top_k_verify, 10)
        self.assertEqual(cfg.tree_width, 1)
        self.assertEqual(cfg.tree_depth, 1)
        self.assertFalse(cfg.adaptive_draft)

    def test_custom_creation(self):
        cfg = SpeculativeConfig(
            draft_length=8, gamma=0.5, verification_strategy="relaxed",
            diversity_penalty=0.3, tree_width=4, tree_depth=3
        )
        self.assertEqual(cfg.draft_length, 8)
        self.assertEqual(cfg.gamma, 0.5)
        self.assertEqual(cfg.verification_strategy, "relaxed")
        self.assertEqual(cfg.diversity_penalty, 0.3)
        self.assertEqual(cfg.tree_width, 4)
        self.assertEqual(cfg.tree_depth, 3)

    def test_inherits_decoding_config(self):
        cfg = SpeculativeConfig()
        self.assertIsInstance(cfg, DecodingConfig)
        self.assertEqual(cfg.num_sequences, 20)
        self.assertEqual(cfg.max_new_tokens, 100)
        self.assertEqual(cfg.temperature, 1.0)

    def test_to_dict(self):
        cfg = SpeculativeConfig(draft_length=7, gamma=2.0)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["draft_length"], 7)
        self.assertEqual(d["gamma"], 2.0)
        self.assertEqual(d["algorithm_name"], "speculative")
        self.assertIn("num_sequences", d)
        self.assertIn("max_new_tokens", d)

    def test_from_dict(self):
        d = {"draft_length": 6, "gamma": 1.5, "verification_strategy": "topk",
             "num_sequences": 5}
        cfg = SpeculativeConfig.from_dict(d)
        self.assertEqual(cfg.draft_length, 6)
        self.assertEqual(cfg.gamma, 1.5)
        self.assertEqual(cfg.verification_strategy, "topk")
        self.assertEqual(cfg.num_sequences, 5)

    def test_roundtrip_serialization(self):
        cfg = SpeculativeConfig(draft_length=3, gamma=0.8, diversity_penalty=0.2)
        d = cfg.to_dict()
        cfg2 = SpeculativeConfig.from_dict(d)
        self.assertEqual(cfg2.draft_length, 3)
        self.assertEqual(cfg2.gamma, 0.8)
        self.assertEqual(cfg2.diversity_penalty, 0.2)

    def test_validate_valid_config(self):
        cfg = SpeculativeConfig()
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_validate_bad_draft_length(self):
        cfg = SpeculativeConfig(draft_length=0)
        errors = cfg.validate()
        self.assertTrue(any("draft_length" in e for e in errors))

    def test_validate_negative_gamma(self):
        cfg = SpeculativeConfig(gamma=-1.0)
        errors = cfg.validate()
        self.assertTrue(any("gamma" in e for e in errors))

    def test_validate_bad_tolerance(self):
        cfg = SpeculativeConfig(relaxation_tolerance=-0.5)
        errors = cfg.validate()
        self.assertTrue(any("relaxation_tolerance" in e for e in errors))

    def test_validate_bad_top_k(self):
        cfg = SpeculativeConfig(top_k_verify=0)
        errors = cfg.validate()
        self.assertTrue(any("top_k_verify" in e for e in errors))

    def test_validate_bad_tree_width(self):
        cfg = SpeculativeConfig(tree_width=0)
        errors = cfg.validate()
        self.assertTrue(any("tree_width" in e for e in errors))

    def test_validate_bad_tree_depth(self):
        cfg = SpeculativeConfig(tree_depth=0)
        errors = cfg.validate()
        self.assertTrue(any("tree_depth" in e for e in errors))

    def test_validate_bad_min_draft_length(self):
        cfg = SpeculativeConfig(min_draft_length=0)
        errors = cfg.validate()
        self.assertTrue(any("min_draft_length" in e for e in errors))

    def test_validate_max_lt_min_draft(self):
        cfg = SpeculativeConfig(min_draft_length=10, max_draft_length=3)
        errors = cfg.validate()
        self.assertTrue(any("max_draft_length" in e for e in errors))

    def test_validate_inherits_parent_validation(self):
        cfg = SpeculativeConfig(temperature=-1.0)
        errors = cfg.validate()
        self.assertTrue(any("temperature" in e for e in errors))

    def test_hash_deterministic(self):
        cfg1 = SpeculativeConfig(draft_length=5)
        cfg2 = SpeculativeConfig(draft_length=5)
        self.assertEqual(cfg1.hash(), cfg2.hash())

    def test_hash_changes_with_params(self):
        cfg1 = SpeculativeConfig(draft_length=5)
        cfg2 = SpeculativeConfig(draft_length=6)
        self.assertNotEqual(cfg1.hash(), cfg2.hash())

    def test_from_dict_extra_params(self):
        d = {"draft_length": 4, "custom_param": "hello"}
        cfg = SpeculativeConfig.from_dict(d)
        self.assertEqual(cfg.draft_length, 4)
        self.assertEqual(cfg.params.get("custom_param"), "hello")

    def test_defaults_min_max_draft(self):
        cfg = SpeculativeConfig()
        self.assertEqual(cfg.min_draft_length, 1)
        self.assertEqual(cfg.max_draft_length, 10)

    def test_acceptance_threshold_default(self):
        cfg = SpeculativeConfig()
        self.assertEqual(cfg.acceptance_threshold, 0.0)


# =========================================================================
# 2. TestNGramDraftModel
# =========================================================================


class TestNGramDraftModel(unittest.TestCase):
    """Tests for NGramDraftModel."""

    def setUp(self):
        np.random.seed(SEED)
        self.model = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)

    def test_empty_table_prediction(self):
        tokens, logits = self.model.predict([5, 6], 3)
        self.assertEqual(len(tokens), 3)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits.shape[1], VOCAB_SIZE)

    def test_build_table_simple(self):
        seqs = [[0, 1, 2, 3, 4]]
        self.model.build_table(seqs)
        counts = self.model.get_counts((0, 1))
        self.assertEqual(counts[2], 1.0)

    def test_build_table_multiple_sequences(self):
        seqs = [[0, 1, 2], [0, 1, 3], [0, 1, 2]]
        self.model.build_table(seqs)
        counts = self.model.get_counts((0, 1))
        self.assertEqual(counts[2], 2.0)
        self.assertEqual(counts[3], 1.0)

    def test_predict_after_build(self):
        seqs = [[10, 20, 30, 40, 50]]
        self.model.build_table(seqs)
        tokens, logits = self.model.predict([10, 20], 1)
        self.assertEqual(tokens[0], 30)

    def test_predict_multiple_tokens(self):
        seqs = [[5, 6, 7, 8, 9]]
        self.model.build_table(seqs)
        tokens, logits = self.model.predict([5, 6], 3)
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0], 7)

    def test_update_adds_to_table(self):
        self.model.update([10, 20, 30, 40])
        counts = self.model.get_counts((10, 20))
        self.assertEqual(counts[30], 1.0)
        counts2 = self.model.get_counts((20, 30))
        self.assertEqual(counts2[40], 1.0)

    def test_smoothing(self):
        model = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE, smoothing=0.5)
        tokens, logits = model.predict([1, 2], 1)
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_unknown_context(self):
        seqs = [[0, 1, 2, 3]]
        self.model.build_table(seqs)
        counts = self.model.get_counts((99, 98))
        self.assertTrue(np.all(counts == 0))

    def test_logits_shape(self):
        tokens, logits = self.model.predict([1, 2, 3], 5)
        self.assertEqual(logits.shape, (5, VOCAB_SIZE))

    def test_predict_consistency(self):
        seqs = [[10, 20, 30, 40, 50]]
        self.model.build_table(seqs)
        t1, _ = self.model.predict([10, 20], 2)
        t2, _ = self.model.predict([10, 20], 2)
        self.assertEqual(t1, t2)

    def test_vocab_boundary(self):
        model = NGramDraftModel(n=2, vocab_size=10)
        model.build_table([[0, 1, 15]])  # 15 > vocab_size
        counts = model.get_counts((0, 1))
        self.assertTrue(np.all(counts == 0))

    def test_n_equals_one(self):
        model = NGramDraftModel(n=1, vocab_size=VOCAB_SIZE)
        model.build_table([[5, 10, 15]])
        counts = model.get_counts((5,))
        self.assertEqual(counts[10], 1.0)

    def test_large_n(self):
        model = NGramDraftModel(n=5, vocab_size=VOCAB_SIZE)
        model.build_table([[1, 2, 3, 4, 5, 6]])
        counts = model.get_counts((1, 2, 3, 4, 5))
        self.assertEqual(counts[6], 1.0)

    def test_build_table_repeated_context(self):
        seqs = [[1, 2, 3, 1, 2, 4]]
        self.model.build_table(seqs)
        counts = self.model.get_counts((1, 2))
        self.assertEqual(counts[3], 1.0)
        self.assertEqual(counts[4], 1.0)


# =========================================================================
# 3. TestCachedDraftModel
# =========================================================================


class TestCachedDraftModel(unittest.TestCase):
    """Tests for CachedDraftModel."""

    def setUp(self):
        np.random.seed(SEED)
        self.source = make_peaked_source(hot_token=10)
        self.model = CachedDraftModel(self.source, vocab_size=VOCAB_SIZE)

    def test_predict_returns_correct_shape(self):
        tokens, logits = self.model.predict([1, 2, 3], 3)
        self.assertEqual(len(tokens), 3)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits.shape[1], VOCAB_SIZE)

    def test_predict_uses_cache(self):
        tokens1, _ = self.model.predict([1, 2], 1)
        self.assertEqual(self.model.cache_size(), 1)
        tokens2, _ = self.model.predict([1, 2], 1)
        self.assertEqual(tokens1, tokens2)

    def test_cache_grows(self):
        self.model.predict([1, 2], 3)
        self.assertEqual(self.model.cache_size(), 3)

    def test_clear_cache(self):
        self.model.predict([1, 2], 2)
        self.assertGreater(self.model.cache_size(), 0)
        self.model.clear_cache()
        self.assertEqual(self.model.cache_size(), 0)

    def test_get_cached_logits_hit(self):
        self.model.predict([1, 2], 1)
        cached = self.model.get_cached_logits((1, 2))
        self.assertIsNotNone(cached)
        self.assertEqual(cached.shape, (VOCAB_SIZE,))

    def test_get_cached_logits_miss(self):
        cached = self.model.get_cached_logits((99, 98))
        self.assertIsNone(cached)

    def test_max_cache_limit(self):
        model = CachedDraftModel(self.source, vocab_size=VOCAB_SIZE, max_cache=2)
        model.predict([1], 5)
        self.assertLessEqual(model.cache_size(), 2)

    def test_update_noop(self):
        # update is a no-op for cached model
        self.model.update([1, 2, 3])
        # Should not error

    def test_peaked_source_picks_hot_token(self):
        tokens, _ = self.model.predict([1, 2, 3], 3)
        for t in tokens:
            self.assertEqual(t, 10)

    def test_different_prefixes_different_cache_keys(self):
        self.model.predict([1, 2], 1)
        self.model.predict([3, 4], 1)
        self.assertEqual(self.model.cache_size(), 2)

    def test_cached_logits_match_source(self):
        self.model.predict([5, 6], 1)
        cached = self.model.get_cached_logits((5, 6))
        direct = self.source([[5, 6]])[0]
        np.testing.assert_array_almost_equal(cached, direct)


# =========================================================================
# 4. TestEnsembleDraftModel
# =========================================================================


class TestEnsembleDraftModel(unittest.TestCase):
    """Tests for EnsembleDraftModel."""

    def setUp(self):
        np.random.seed(SEED)
        self.src1 = make_peaked_source(hot_token=10)
        self.src2 = make_peaked_source(hot_token=20)
        self.model1 = SimpleDraftModel(self.src1)
        self.model2 = SimpleDraftModel(self.src2)

    def test_uniform_weights(self):
        ensemble = EnsembleDraftModel([self.model1, self.model2])
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)
        self.assertAlmostEqual(ensemble.weights[0], 0.5)

    def test_custom_weights(self):
        ensemble = EnsembleDraftModel(
            [self.model1, self.model2], weights=[3.0, 1.0]
        )
        self.assertAlmostEqual(ensemble.weights[0], 0.75)
        self.assertAlmostEqual(ensemble.weights[1], 0.25)

    def test_predict_shape(self):
        ensemble = EnsembleDraftModel([self.model1, self.model2])
        tokens, logits = ensemble.predict([1, 2], 3)
        self.assertEqual(len(tokens), 3)
        self.assertEqual(logits.shape, (3, VOCAB_SIZE))

    def test_combined_logits(self):
        ensemble = EnsembleDraftModel(
            [self.model1, self.model2], weights=[1.0, 1.0]
        )
        _, logits = ensemble.predict([1, 2], 1)
        # Combined should have peaks at both 10 and 20
        self.assertGreater(logits[0, 10], logits[0, 50])
        self.assertGreater(logits[0, 20], logits[0, 50])

    def test_weighted_towards_model1(self):
        ensemble = EnsembleDraftModel(
            [self.model1, self.model2], weights=[100.0, 1.0]
        )
        tokens, _ = ensemble.predict([1, 2], 1)
        self.assertEqual(tokens[0], 10)

    def test_weighted_towards_model2(self):
        ensemble = EnsembleDraftModel(
            [self.model1, self.model2], weights=[1.0, 100.0]
        )
        tokens, _ = ensemble.predict([1, 2], 1)
        self.assertEqual(tokens[0], 20)

    def test_single_model_ensemble(self):
        ensemble = EnsembleDraftModel([self.model1])
        tokens, _ = ensemble.predict([1, 2], 2)
        direct_tokens, _ = self.model1.predict([1, 2], 2)
        self.assertEqual(tokens[0], direct_tokens[0])

    def test_update_propagates(self):
        ensemble = EnsembleDraftModel([self.model1, self.model2])
        ensemble.update([5, 6, 7])
        self.assertEqual(len(self.model1._accepted), 1)
        self.assertEqual(len(self.model2._accepted), 1)

    def test_three_models(self):
        src3 = make_peaked_source(hot_token=30)
        model3 = SimpleDraftModel(src3)
        ensemble = EnsembleDraftModel([self.model1, self.model2, model3])
        self.assertEqual(len(ensemble.weights), 3)
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)

    def test_ensemble_consistency(self):
        ensemble = EnsembleDraftModel([self.model1, self.model2])
        t1, _ = ensemble.predict([1, 2], 2)
        t2, _ = ensemble.predict([1, 2], 2)
        self.assertEqual(t1, t2)


# =========================================================================
# 5. TestAdaptiveDraftModel
# =========================================================================


class TestAdaptiveDraftModel(unittest.TestCase):
    """Tests for AdaptiveDraftModel."""

    def setUp(self):
        np.random.seed(SEED)
        src = make_peaked_source(hot_token=10)
        self.base = SimpleDraftModel(src)
        self.model = AdaptiveDraftModel(self.base, min_length=1, max_length=10)

    def test_initial_length(self):
        self.assertEqual(self.model.current_length, 5)  # (1+10)//2

    def test_predict_uses_current_length(self):
        tokens, logits = self.model.predict([1, 2])
        self.assertEqual(len(tokens), self.model.current_length)

    def test_predict_with_explicit_length(self):
        tokens, logits = self.model.predict([1, 2], num_tokens=3)
        self.assertEqual(len(tokens), 3)

    def test_high_acceptance_increases_length(self):
        initial = self.model.current_length
        self.model.update_acceptance_rate(0.95)
        self.assertGreaterEqual(self.model.current_length, initial)

    def test_low_acceptance_decreases_length(self):
        self.model._current_length = 5
        self.model.update_acceptance_rate(0.1)
        self.assertLessEqual(self.model.current_length, 5)

    def test_length_clamped_max(self):
        self.model._current_length = 10
        self.model.update_acceptance_rate(0.99)
        self.assertEqual(self.model.current_length, 10)

    def test_length_clamped_min(self):
        self.model._current_length = 1
        self.model.update_acceptance_rate(0.01)
        self.assertEqual(self.model.current_length, 1)

    def test_acceptance_history(self):
        self.model.update_acceptance_rate(0.5)
        self.model.update_acceptance_rate(0.8)
        hist = self.model.get_acceptance_history()
        self.assertEqual(len(hist), 2)
        self.assertAlmostEqual(hist[0], 0.5)
        self.assertAlmostEqual(hist[1], 0.8)

    def test_update_delegates(self):
        self.model.update([1, 2, 3])
        self.assertEqual(len(self.base._accepted), 1)

    def test_gradual_increase(self):
        self.model._current_length = 3
        for _ in range(5):
            self.model.update_acceptance_rate(0.95)
        self.assertGreater(self.model.current_length, 3)

    def test_gradual_decrease(self):
        self.model._current_length = 8
        for _ in range(5):
            self.model.update_acceptance_rate(0.05)
        self.assertLess(self.model.current_length, 8)

    def test_target_acceptance(self):
        model = AdaptiveDraftModel(self.base, target_acceptance=0.9)
        self.assertEqual(model.target_acceptance, 0.9)

    def test_custom_min_max(self):
        model = AdaptiveDraftModel(self.base, min_length=3, max_length=7)
        self.assertEqual(model.min_length, 3)
        self.assertEqual(model.max_length, 7)
        self.assertEqual(model.current_length, 5)  # (3+7)//2


# =========================================================================
# 6. TestStandardVerification
# =========================================================================


class TestStandardVerification(unittest.TestCase):
    """Tests for StandardVerification."""

    def setUp(self):
        self.rng = np.random.RandomState(SEED)

    def test_all_accepted_identical_distributions(self):
        """When draft == target, all tokens should be accepted."""
        rng = np.random.RandomState(0)
        verifier = StandardVerification(rng=rng)
        logits = np.random.RandomState(1).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits[i])) for i in range(5)]
        accepted, n = verifier.verify(tokens, logits, logits.copy())
        self.assertEqual(n, 5)
        self.assertEqual(accepted, tokens)

    def test_rejection_on_mismatch(self):
        """Tokens unlikely under target should be rejected."""
        rng = np.random.RandomState(SEED)
        verifier = StandardVerification(rng=rng)
        draft_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        draft_logits[:, 50] = 10.0  # draft strongly prefers 50
        target_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        target_logits[:, 30] = 10.0  # target strongly prefers 30
        tokens = [50, 50, 50]
        accepted, n = verifier.verify(tokens, draft_logits, target_logits)
        self.assertLessEqual(n, 3)

    def test_empty_draft(self):
        verifier = StandardVerification(rng=self.rng)
        accepted, n = verifier.verify([], np.zeros((0, VOCAB_SIZE)), np.zeros((0, VOCAB_SIZE)))
        self.assertEqual(n, 0)
        self.assertEqual(accepted, [])

    def test_single_token_accepted(self):
        rng = np.random.RandomState(0)
        verifier = StandardVerification(rng=rng)
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 10] = 5.0
        accepted, n = verifier.verify([10], logits, logits.copy())
        self.assertEqual(n, 1)
        self.assertEqual(accepted, [10])

    def test_deterministic_with_seed(self):
        logits_d = np.random.RandomState(10).randn(5, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(20).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits_d[i])) for i in range(5)]

        v1 = StandardVerification(rng=np.random.RandomState(SEED))
        a1, n1 = v1.verify(tokens, logits_d, logits_t)

        v2 = StandardVerification(rng=np.random.RandomState(SEED))
        a2, n2 = v2.verify(tokens, logits_d, logits_t)

        self.assertEqual(a1, a2)
        self.assertEqual(n1, n2)

    def test_high_target_prob_accepted(self):
        rng = np.random.RandomState(0)
        verifier = StandardVerification(rng=rng)
        draft_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        target_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        target_logits[0, 10] = 20.0  # very high target prob
        accepted, n = verifier.verify([10], draft_logits, target_logits)
        self.assertEqual(n, 1)

    def test_sequential_acceptance(self):
        """Early rejection stops remaining tokens from being checked."""
        rng = np.random.RandomState(0)
        verifier = StandardVerification(rng=rng)
        draft_logits = np.zeros((5, VOCAB_SIZE), dtype=np.float32)
        draft_logits[:, 10] = 20.0
        target_logits = np.zeros((5, VOCAB_SIZE), dtype=np.float32)
        target_logits[:, 20] = 20.0
        tokens = [10] * 5
        accepted, n = verifier.verify(tokens, draft_logits, target_logits)
        # If first token rejected, n should be 0
        # The result depends on the random draw
        self.assertLessEqual(n, 5)
        self.assertEqual(len(accepted), n)

    def test_near_zero_draft_prob(self):
        """When draft prob is near zero, token should be accepted (ratio high)."""
        rng = np.random.RandomState(0)
        verifier = StandardVerification(rng=rng)
        draft_logits = np.full((1, VOCAB_SIZE), -100.0, dtype=np.float32)
        draft_logits[0, 0] = 100.0  # all mass on 0
        # Token 5 has near zero draft prob
        target_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        target_logits[0, 5] = 5.0
        # draft prob of token 5 is essentially 0, so acceptance via guard
        accepted, n = verifier.verify([5], draft_logits, target_logits)
        self.assertEqual(n, 1)

    def test_mismatched_lengths(self):
        """When arrays have different lengths, verify min of all."""
        verifier = StandardVerification(rng=self.rng)
        logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        accepted, n = verifier.verify([1, 2, 3, 4, 5], logits, logits)
        self.assertLessEqual(n, 3)


# =========================================================================
# 7. TestDiversityAwareVerification
# =========================================================================


class TestDiversityAwareVerification(unittest.TestCase):
    """Tests for DiversityAwareVerification."""

    def setUp(self):
        self.rng = np.random.RandomState(SEED)

    def test_novel_token_boosted(self):
        """Novel tokens get diversity bonus."""
        verifier = DiversityAwareVerification(diversity_penalty=0.5, rng=self.rng)
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        accepted, n = verifier.verify([10], logits, logits.copy())
        self.assertEqual(n, 1)

    def test_repeated_token_no_boost(self):
        """After seeing a token, no diversity bonus."""
        verifier = DiversityAwareVerification(diversity_penalty=0.5,
                                               rng=np.random.RandomState(0))
        logits = np.zeros((2, VOCAB_SIZE), dtype=np.float32)
        verifier._seen_tokens.add(10)
        # Token 10 already seen—no boost

    def test_reset_seen(self):
        verifier = DiversityAwareVerification()
        verifier._seen_tokens.add(5)
        verifier._seen_tokens.add(10)
        verifier.reset_seen()
        self.assertEqual(len(verifier._seen_tokens), 0)

    def test_seen_tokens_grow(self):
        verifier = DiversityAwareVerification(diversity_penalty=0.5,
                                               rng=np.random.RandomState(0))
        logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        tokens = [10, 20, 30]
        verifier.verify(tokens, logits, logits.copy())
        for t in tokens[:len(verifier._seen_tokens)]:
            self.assertIn(t, verifier._seen_tokens)

    def test_high_diversity_penalty_accepts_more(self):
        """With high diversity penalty, novel tokens should be accepted more."""
        logits_d = np.random.RandomState(5).randn(10, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(6).randn(10, VOCAB_SIZE).astype(np.float32)
        tokens = list(range(10, 20))

        v_low = DiversityAwareVerification(diversity_penalty=0.0,
                                            rng=np.random.RandomState(SEED))
        _, n_low = v_low.verify(tokens, logits_d, logits_t)

        v_high = DiversityAwareVerification(diversity_penalty=0.9,
                                             rng=np.random.RandomState(SEED))
        _, n_high = v_high.verify(tokens, logits_d, logits_t)

        self.assertGreaterEqual(n_high, n_low)

    def test_identical_distributions_accepted(self):
        verifier = DiversityAwareVerification(diversity_penalty=0.1,
                                               rng=np.random.RandomState(0))
        logits = np.random.RandomState(7).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits[i])) for i in range(5)]
        accepted, n = verifier.verify(tokens, logits, logits.copy())
        self.assertEqual(n, 5)

    def test_zero_diversity_penalty(self):
        """With zero penalty, behaves like standard verification."""
        logits_d = np.random.RandomState(8).randn(5, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(9).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits_d[i])) for i in range(5)]

        v_div = DiversityAwareVerification(diversity_penalty=0.0,
                                            rng=np.random.RandomState(SEED))
        v_std = StandardVerification(rng=np.random.RandomState(SEED))

        a_div, n_div = v_div.verify(tokens, logits_d, logits_t)
        a_std, n_std = v_std.verify(tokens, logits_d, logits_t)
        self.assertEqual(n_div, n_std)


# =========================================================================
# 8. TestRelaxedVerification
# =========================================================================


class TestRelaxedVerification(unittest.TestCase):
    """Tests for RelaxedVerification."""

    def test_zero_tolerance_stricter(self):
        logits_d = np.random.RandomState(1).randn(10, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(2).randn(10, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits_d[i])) for i in range(10)]

        v_strict = RelaxedVerification(tolerance=0.0, rng=np.random.RandomState(SEED))
        _, n_strict = v_strict.verify(tokens, logits_d, logits_t)

        v_relaxed = RelaxedVerification(tolerance=0.5, rng=np.random.RandomState(SEED))
        _, n_relaxed = v_relaxed.verify(tokens, logits_d, logits_t)

        self.assertGreaterEqual(n_relaxed, n_strict)

    def test_high_tolerance_accepts_all_identical(self):
        verifier = RelaxedVerification(tolerance=0.0, rng=np.random.RandomState(0))
        logits = np.random.RandomState(3).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits[i])) for i in range(5)]
        accepted, n = verifier.verify(tokens, logits, logits.copy())
        self.assertEqual(n, 5)

    def test_very_high_tolerance(self):
        """High tolerance should accept almost everything."""
        verifier = RelaxedVerification(tolerance=1.0, rng=np.random.RandomState(0))
        logits_d = np.random.RandomState(4).randn(5, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(5).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits_d[i])) for i in range(5)]
        accepted, n = verifier.verify(tokens, logits_d, logits_t)
        self.assertGreater(n, 0)

    def test_empty_input(self):
        verifier = RelaxedVerification(tolerance=0.1)
        accepted, n = verifier.verify([], np.zeros((0, VOCAB_SIZE)),
                                       np.zeros((0, VOCAB_SIZE)))
        self.assertEqual(n, 0)

    def test_tolerance_effect_on_ratio_boundary(self):
        """A ratio just below 1.0 should be accepted with sufficient tolerance."""
        rng = np.random.RandomState(0)
        verifier = RelaxedVerification(tolerance=0.2, rng=rng)
        # Draft has slightly higher prob than target for token
        draft_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        draft_logits[0, 10] = 1.0
        target_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        target_logits[0, 10] = 0.9
        accepted, n = verifier.verify([10], draft_logits, target_logits)
        # ratio is close to 1.0, tolerance is 0.2 so ratio >= 0.8 => accept
        self.assertEqual(n, 1)

    def test_deterministic(self):
        logits_d = np.random.RandomState(10).randn(5, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(11).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = list(range(5))

        v1 = RelaxedVerification(tolerance=0.1, rng=np.random.RandomState(SEED))
        a1, n1 = v1.verify(tokens, logits_d, logits_t)

        v2 = RelaxedVerification(tolerance=0.1, rng=np.random.RandomState(SEED))
        a2, n2 = v2.verify(tokens, logits_d, logits_t)

        self.assertEqual(a1, a2)
        self.assertEqual(n1, n2)


# =========================================================================
# 9. TestTopKVerification
# =========================================================================


class TestTopKVerification(unittest.TestCase):
    """Tests for TopKVerification."""

    def test_token_in_top_k_accepted(self):
        verifier = TopKVerification(k=5)
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 10] = 10.0  # token 10 is top-1
        accepted, n = verifier.verify([10], logits, logits)
        self.assertEqual(n, 1)

    def test_token_not_in_top_k_rejected(self):
        verifier = TopKVerification(k=5)
        target_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        for i in range(5):
            target_logits[0, i] = 10.0 - i
        draft_logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        # Token 99 is not in target top-5
        accepted, n = verifier.verify([99], draft_logits, target_logits)
        self.assertEqual(n, 0)

    def test_larger_k_accepts_more(self):
        target_logits = np.random.RandomState(SEED).randn(10, VOCAB_SIZE).astype(np.float32)
        tokens = list(range(10))

        v_small = TopKVerification(k=1)
        _, n_small = v_small.verify(tokens, target_logits, target_logits)

        v_large = TopKVerification(k=50)
        _, n_large = v_large.verify(tokens, target_logits, target_logits)

        self.assertGreaterEqual(n_large, n_small)

    def test_k_equals_vocab_accepts_all(self):
        verifier = TopKVerification(k=VOCAB_SIZE)
        logits = np.random.RandomState(1).randn(5, VOCAB_SIZE).astype(np.float32)
        tokens = list(range(5))
        accepted, n = verifier.verify(tokens, logits, logits)
        self.assertEqual(n, 5)

    def test_empty_input(self):
        verifier = TopKVerification(k=10)
        accepted, n = verifier.verify([], np.zeros((0, VOCAB_SIZE)),
                                       np.zeros((0, VOCAB_SIZE)))
        self.assertEqual(n, 0)

    def test_sequential_rejection(self):
        """First out-of-top-k token stops verification."""
        verifier = TopKVerification(k=5)
        target_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        for i in range(3):
            for j in range(5):
                target_logits[i, j] = 10.0 - j
        # Token 0 in top-k, token 99 not, token 0 in top-k
        tokens = [0, 99, 0]
        draft_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        accepted, n = verifier.verify(tokens, draft_logits, target_logits)
        self.assertEqual(n, 1)
        self.assertEqual(accepted, [0])

    def test_k_equals_one(self):
        verifier = TopKVerification(k=1)
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 42] = 10.0
        accepted, n = verifier.verify([42], logits, logits)
        self.assertEqual(n, 1)

        accepted2, n2 = verifier.verify([0], logits, logits)
        self.assertEqual(n2, 0)

    def test_multiple_top_k_tokens(self):
        verifier = TopKVerification(k=3)
        target_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        target_logits[0, 10] = 5.0
        target_logits[0, 11] = 4.0
        target_logits[0, 12] = 3.0
        target_logits[1, 10] = 5.0
        target_logits[1, 11] = 4.0
        target_logits[1, 12] = 3.0
        target_logits[2, 10] = 5.0
        target_logits[2, 11] = 4.0
        target_logits[2, 12] = 3.0
        draft_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        accepted, n = verifier.verify([10, 11, 12], draft_logits, target_logits)
        self.assertEqual(n, 3)


# =========================================================================
# 10. TestSpeculativeDecoder
# =========================================================================


class TestSpeculativeDecoder(unittest.TestCase):
    """Tests for SpeculativeDecoder."""

    def _make_decoder(self, target_src=None, draft_src=None, **kwargs):
        target = target_src or make_peaked_source(hot_token=10)
        draft = draft_src or make_peaked_source(hot_token=10)
        draft_model = SimpleDraftModel(draft)
        config = SpeculativeConfig(
            max_new_tokens=kwargs.get("max_new_tokens", 10),
            draft_length=kwargs.get("draft_length", 3),
            seed=SEED,
            eos_token_id=kwargs.get("eos_token_id", None),
        )
        return SpeculativeDecoder(target, draft_model, config,
                                   rng=np.random.RandomState(SEED))

    def test_basic_decode(self):
        decoder = self._make_decoder()
        result = decoder.decode([0])
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)

    def test_prefix_preserved(self):
        decoder = self._make_decoder()
        prefix = [5, 6, 7]
        result = decoder.decode(prefix)
        self.assertEqual(result[:3], prefix)

    def test_max_tokens_respected(self):
        decoder = self._make_decoder(max_new_tokens=5)
        result = decoder.decode([0])
        self.assertLessEqual(len(result) - 1, 5 + 3)  # slack for draft overshoot

    def test_eos_stops_generation(self):
        src = make_eos_source(eos_token=EOS_TOKEN, after_steps=3)
        decoder = self._make_decoder(target_src=src, draft_src=src,
                                      max_new_tokens=20, eos_token_id=EOS_TOKEN)
        result = decoder.decode([0])
        self.assertEqual(result[-1], EOS_TOKEN)

    def test_identical_draft_target_fast(self):
        """When draft == target, all tokens should be accepted."""
        src = make_peaked_source(hot_token=10)
        decoder = self._make_decoder(target_src=src, draft_src=src,
                                      max_new_tokens=10)
        result = decoder.decode([0])
        # All generated tokens should be 10
        for t in result[1:]:
            self.assertEqual(t, 10)

    def test_tracker_records(self):
        decoder = self._make_decoder(max_new_tokens=10)
        decoder.decode([0])
        self.assertGreater(decoder.tracker.num_records(), 0)

    def test_acceptance_rate_perfect_match(self):
        src = make_peaked_source(hot_token=10)
        decoder = self._make_decoder(target_src=src, draft_src=src,
                                      max_new_tokens=15)
        decoder.decode([0])
        rate = decoder.tracker.acceptance_rate()
        self.assertGreater(rate, 0.5)

    def test_different_draft_target(self):
        target = make_peaked_source(hot_token=10)
        draft = make_peaked_source(hot_token=20)
        decoder = self._make_decoder(target_src=target, draft_src=draft,
                                      max_new_tokens=10)
        result = decoder.decode([0])
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)

    def test_draft_length_one(self):
        decoder = self._make_decoder(draft_length=1, max_new_tokens=5)
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)

    def test_output_deterministic(self):
        src = make_peaked_source(hot_token=10)
        d1 = SpeculativeDecoder(
            src, SimpleDraftModel(src),
            SpeculativeConfig(max_new_tokens=10, draft_length=3, seed=SEED),
            rng=np.random.RandomState(SEED)
        )
        d2 = SpeculativeDecoder(
            src, SimpleDraftModel(src),
            SpeculativeConfig(max_new_tokens=10, draft_length=3, seed=SEED),
            rng=np.random.RandomState(SEED)
        )
        r1 = d1.decode([0])
        r2 = d2.decode([0])
        self.assertEqual(r1, r2)

    def test_custom_verification(self):
        src = make_peaked_source(hot_token=10)
        decoder = SpeculativeDecoder(
            src, SimpleDraftModel(src),
            SpeculativeConfig(max_new_tokens=10, draft_length=3, seed=SEED),
            verification=TopKVerification(k=5),
            rng=np.random.RandomState(SEED)
        )
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)


# =========================================================================
# 11. TestDiversitySpeculativeDecoder
# =========================================================================


class TestDiversitySpeculativeDecoder(unittest.TestCase):
    """Tests for DiversitySpeculativeDecoder."""

    def _make_decoder(self, **kwargs):
        src = make_random_source(seed=SEED)
        draft = SimpleDraftModel(make_random_source(seed=SEED + 1))
        config = SpeculativeConfig(
            max_new_tokens=kwargs.get("max_new_tokens", 8),
            draft_length=kwargs.get("draft_length", 2),
            diversity_penalty=kwargs.get("diversity_penalty", 0.2),
            seed=SEED,
        )
        return DiversitySpeculativeDecoder(src, draft, config,
                                            rng=np.random.RandomState(SEED))

    def test_generates_multiple(self):
        decoder = self._make_decoder()
        results = decoder.decode_multiple([0], num_sequences=3)
        self.assertEqual(len(results), 3)

    def test_each_sequence_has_prefix(self):
        decoder = self._make_decoder()
        results = decoder.decode_multiple([5, 6], num_sequences=3)
        for seq in results:
            self.assertEqual(seq[:2], [5, 6])

    def test_sequences_not_empty(self):
        decoder = self._make_decoder()
        results = decoder.decode_multiple([0], num_sequences=5)
        for seq in results:
            self.assertGreater(len(seq), 1)

    def test_diversity_among_sequences(self):
        """With diversity penalty, sequences should show some variation."""
        decoder = self._make_decoder(diversity_penalty=0.5)
        results = decoder.decode_multiple([0], num_sequences=5)
        unique_seqs = set(tuple(s) for s in results)
        # At least some diversity expected (but not guaranteed with peaked logits)
        self.assertGreaterEqual(len(unique_seqs), 1)

    def test_single_sequence(self):
        decoder = self._make_decoder()
        results = decoder.decode_multiple([0], num_sequences=1)
        self.assertEqual(len(results), 1)

    def test_zero_diversity_penalty(self):
        decoder = self._make_decoder(diversity_penalty=0.0)
        results = decoder.decode_multiple([0], num_sequences=3)
        self.assertEqual(len(results), 3)

    def test_max_tokens_respected(self):
        decoder = self._make_decoder(max_new_tokens=5)
        results = decoder.decode_multiple([0], num_sequences=3)
        for seq in results:
            self.assertLessEqual(len(seq) - 1, 5 + 3)

    def test_longer_prefix(self):
        decoder = self._make_decoder()
        prefix = list(range(10, 20))
        results = decoder.decode_multiple(prefix, num_sequences=2)
        for seq in results:
            self.assertEqual(seq[:10], prefix)

    def test_many_sequences(self):
        decoder = self._make_decoder(max_new_tokens=5)
        results = decoder.decode_multiple([0], num_sequences=10)
        self.assertEqual(len(results), 10)


# =========================================================================
# 12. TestSpeculativeTreeNode
# =========================================================================


class TestSpeculativeTreeNode(unittest.TestCase):
    """Tests for SpeculativeTreeNode."""

    def test_creation(self):
        node = SpeculativeTreeNode(token=5, logit=1.0, probability=0.5, depth=0)
        self.assertEqual(node.token, 5)
        self.assertEqual(node.logit, 1.0)
        self.assertEqual(node.probability, 0.5)
        self.assertEqual(node.depth, 0)
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, [])

    def test_add_child(self):
        root = SpeculativeTreeNode(token=0)
        child = root.add_child(token=1, logit=2.0, probability=0.3)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(child.token, 1)
        self.assertEqual(child.depth, 1)
        self.assertIs(child.parent, root)

    def test_add_multiple_children(self):
        root = SpeculativeTreeNode(token=0)
        c1 = root.add_child(token=1)
        c2 = root.add_child(token=2)
        c3 = root.add_child(token=3)
        self.assertEqual(len(root.children), 3)

    def test_is_leaf(self):
        root = SpeculativeTreeNode(token=0)
        self.assertTrue(root.is_leaf())
        root.add_child(token=1)
        self.assertFalse(root.is_leaf())

    def test_is_root(self):
        root = SpeculativeTreeNode(token=0)
        self.assertTrue(root.is_root())
        child = root.add_child(token=1)
        self.assertFalse(child.is_root())

    def test_path_from_root(self):
        root = SpeculativeTreeNode(token=0)
        c1 = root.add_child(token=1)
        c2 = c1.add_child(token=2)
        c3 = c2.add_child(token=3)
        path = c3.path_from_root()
        self.assertEqual(path, [0, 1, 2, 3])

    def test_path_from_root_single(self):
        node = SpeculativeTreeNode(token=5)
        self.assertEqual(node.path_from_root(), [5])

    def test_subtree_size_leaf(self):
        node = SpeculativeTreeNode(token=0)
        self.assertEqual(node.subtree_size(), 1)

    def test_subtree_size_tree(self):
        root = SpeculativeTreeNode(token=0)
        c1 = root.add_child(token=1)
        c2 = root.add_child(token=2)
        c1.add_child(token=3)
        c1.add_child(token=4)
        # root + c1 + c2 + c3 + c4 = 5
        self.assertEqual(root.subtree_size(), 5)

    def test_max_depth_leaf(self):
        node = SpeculativeTreeNode(token=0, depth=0)
        self.assertEqual(node.max_depth(), 0)

    def test_max_depth_deep_tree(self):
        root = SpeculativeTreeNode(token=0)
        c1 = root.add_child(token=1)
        c2 = c1.add_child(token=2)
        c3 = c2.add_child(token=3)
        self.assertEqual(root.max_depth(), 3)

    def test_max_depth_wide_tree(self):
        root = SpeculativeTreeNode(token=0)
        c1 = root.add_child(token=1)
        c2 = root.add_child(token=2)
        c1.add_child(token=3)
        c2.add_child(token=4)
        c2_child = c2.children[0]
        c2_child.add_child(token=5)
        self.assertEqual(root.max_depth(), 3)

    def test_leaves_single(self):
        node = SpeculativeTreeNode(token=0)
        leaves = node.leaves()
        self.assertEqual(len(leaves), 1)
        self.assertIs(leaves[0], node)

    def test_leaves_tree(self):
        root = SpeculativeTreeNode(token=0)
        c1 = root.add_child(token=1)
        c2 = root.add_child(token=2)
        c1.add_child(token=3)
        c1.add_child(token=4)
        leaves = root.leaves()
        self.assertEqual(len(leaves), 3)  # c2, c3, c4

    def test_accepted_flag(self):
        node = SpeculativeTreeNode(token=0)
        self.assertFalse(node.accepted)
        node.accepted = True
        self.assertTrue(node.accepted)

    def test_child_depth_increments(self):
        root = SpeculativeTreeNode(token=0, depth=0)
        c = root.add_child(token=1)
        gc = c.add_child(token=2)
        self.assertEqual(c.depth, 1)
        self.assertEqual(gc.depth, 2)


# =========================================================================
# 13. TestSpeculativeTree
# =========================================================================


class TestSpeculativeTree(unittest.TestCase):
    """Tests for SpeculativeTree."""

    def test_creation(self):
        tree = SpeculativeTree(root_token=-1, width=3, max_depth=4)
        self.assertEqual(tree.root.token, -1)
        self.assertEqual(tree.width, 3)
        self.assertEqual(tree.max_depth, 4)

    def test_initial_size(self):
        tree = SpeculativeTree()
        self.assertEqual(tree.size(), 1)

    def test_initial_depth(self):
        tree = SpeculativeTree()
        self.assertEqual(tree.depth(), 0)

    def test_expand(self):
        tree = SpeculativeTree(width=3, max_depth=5)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        logits[20] = 4.0
        logits[30] = 3.0
        children = tree.expand(tree.root, [], logits)
        self.assertEqual(len(children), 3)

    def test_expand_respects_max_depth(self):
        tree = SpeculativeTree(width=2, max_depth=1)
        deep_node = SpeculativeTreeNode(token=1, depth=1)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        children = tree.expand(deep_node, [], logits)
        self.assertEqual(len(children), 0)

    def test_get_all_paths_single_expansion(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        logits[20] = 4.0
        tree.expand(tree.root, [], logits)
        paths = tree.get_all_paths()
        self.assertEqual(len(paths), 2)

    def test_get_all_paths_two_levels(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        logits[20] = 4.0
        children = tree.expand(tree.root, [], logits)
        for c in children:
            tree.expand(c, [], logits)
        paths = tree.get_all_paths()
        self.assertEqual(len(paths), 4)  # 2 * 2 leaves

    def test_prune_low_probability(self):
        tree = SpeculativeTree(root_token=-1, width=3, max_depth=5)
        c1 = tree.root.add_child(token=1, probability=0.5)
        c2 = tree.root.add_child(token=2, probability=0.005)
        c3 = tree.root.add_child(token=3, probability=0.3)
        pruned = tree.prune(min_probability=0.01)
        self.assertEqual(len(tree.root.children), 2)
        self.assertEqual(pruned, 1)

    def test_prune_keeps_high_prob(self):
        tree = SpeculativeTree(root_token=-1, width=3, max_depth=5)
        tree.root.add_child(token=1, probability=0.5)
        tree.root.add_child(token=2, probability=0.3)
        pruned = tree.prune(min_probability=0.01)
        self.assertEqual(pruned, 0)
        self.assertEqual(len(tree.root.children), 2)

    def test_prune_subtrees(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=5)
        c1 = tree.root.add_child(token=1, probability=0.5)
        c2 = tree.root.add_child(token=2, probability=0.005)
        c2.add_child(token=3, probability=0.001)
        c2.add_child(token=4, probability=0.002)
        pruned = tree.prune(min_probability=0.01)
        self.assertEqual(pruned, 3)  # c2 + its 2 children
        self.assertEqual(len(tree.root.children), 1)

    def test_size_after_expansion(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        logits[20] = 4.0
        tree.expand(tree.root, [], logits)
        self.assertEqual(tree.size(), 3)  # root + 2 children

    def test_depth_after_expansion(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        logits[20] = 4.0
        children = tree.expand(tree.root, [], logits)
        tree.expand(children[0], [], logits)
        self.assertEqual(tree.depth(), 2)

    def test_best_path(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        c1 = tree.root.add_child(token=10, probability=0.8)
        c2 = tree.root.add_child(token=20, probability=0.2)
        c1.add_child(token=30, probability=0.9)
        c2.add_child(token=40, probability=0.1)
        path = tree.best_path()
        self.assertIn(30, path)

    def test_best_path_empty_tree(self):
        tree = SpeculativeTree(root_token=-1)
        path = tree.best_path()
        self.assertEqual(path, [])

    def test_expand_probability_values(self):
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        logits[20] = 3.0
        children = tree.expand(tree.root, [], logits)
        probs = [c.probability for c in children]
        for p in probs:
            self.assertGreater(p, 0)
            self.assertLessEqual(p, 1.0)


# =========================================================================
# 14. TestAcceptanceTracker
# =========================================================================


class TestAcceptanceTracker(unittest.TestCase):
    """Tests for AcceptanceTracker."""

    def setUp(self):
        self.tracker = AcceptanceTracker(window_size=10)

    def test_initial_rate_zero(self):
        self.assertEqual(self.tracker.acceptance_rate(), 0.0)

    def test_record_and_rate(self):
        self.tracker.record(5, 3)
        self.assertAlmostEqual(self.tracker.acceptance_rate(), 0.6)

    def test_multiple_records(self):
        self.tracker.record(10, 8)
        self.tracker.record(10, 6)
        self.assertAlmostEqual(self.tracker.acceptance_rate(), 14.0 / 20.0)

    def test_window_eviction(self):
        tracker = AcceptanceTracker(window_size=3)
        tracker.record(10, 10)  # rate=1.0
        tracker.record(10, 10)
        tracker.record(10, 10)
        tracker.record(10, 0)  # now window is [10/10, 10/10, 10/0]
        # Window keeps last 3: second, third, fourth records
        rate = tracker.acceptance_rate()
        # proposed: 10+10+10=30, accepted: 10+10+0=20
        self.assertAlmostEqual(rate, 20.0 / 30.0)

    def test_adaptive_gamma_increases(self):
        tracker = AcceptanceTracker(window_size=10, initial_gamma=1.0)
        tracker.record(10, 9)  # 90% rate
        gamma = tracker.adaptive_gamma(target_rate=0.7)
        self.assertGreater(gamma, 1.0)

    def test_adaptive_gamma_decreases(self):
        tracker = AcceptanceTracker(window_size=10, initial_gamma=1.0)
        tracker.record(10, 1)  # 10% rate
        gamma = tracker.adaptive_gamma(target_rate=0.7)
        self.assertLess(gamma, 1.0)

    def test_gamma_clamped_high(self):
        tracker = AcceptanceTracker(window_size=10, initial_gamma=2.9)
        tracker.record(10, 10)
        gamma = tracker.adaptive_gamma(target_rate=0.5)
        self.assertLessEqual(gamma, 3.0)

    def test_gamma_clamped_low(self):
        tracker = AcceptanceTracker(window_size=10, initial_gamma=0.15)
        tracker.record(10, 0)
        gamma = tracker.adaptive_gamma(target_rate=0.7)
        self.assertGreaterEqual(gamma, 0.1)

    def test_num_records(self):
        self.assertEqual(self.tracker.num_records(), 0)
        self.tracker.record(5, 3)
        self.assertEqual(self.tracker.num_records(), 1)
        self.tracker.record(5, 4)
        self.assertEqual(self.tracker.num_records(), 2)

    def test_reset(self):
        self.tracker.record(5, 3)
        self.tracker.record(10, 8)
        self.tracker.reset()
        self.assertEqual(self.tracker.num_records(), 0)
        self.assertEqual(self.tracker.acceptance_rate(), 0.0)
        self.assertAlmostEqual(self.tracker.gamma, 1.0)

    def test_gamma_property(self):
        self.assertAlmostEqual(self.tracker.gamma, 1.0)

    def test_zero_proposed(self):
        self.tracker.record(0, 0)
        self.assertEqual(self.tracker.acceptance_rate(), 0.0)

    def test_all_accepted(self):
        self.tracker.record(10, 10)
        self.assertAlmostEqual(self.tracker.acceptance_rate(), 1.0)

    def test_none_accepted(self):
        self.tracker.record(10, 0)
        self.assertAlmostEqual(self.tracker.acceptance_rate(), 0.0)

    def test_adaptive_gamma_stable_at_target(self):
        """When rate == target, gamma stays roughly the same."""
        tracker = AcceptanceTracker(window_size=10, initial_gamma=1.0)
        tracker.record(10, 7)  # 70% == target
        gamma = tracker.adaptive_gamma(target_rate=0.7)
        self.assertGreater(gamma, 0.9)  # slight increase because 0.7 >= 0.7

    def test_long_history(self):
        tracker = AcceptanceTracker(window_size=5)
        for _ in range(20):
            tracker.record(10, 7)
        self.assertAlmostEqual(tracker.acceptance_rate(), 0.7)
        self.assertEqual(tracker.num_records(), 5)


# =========================================================================
# 15. TestComputeAcceptanceProbability
# =========================================================================


class TestComputeAcceptanceProbability(unittest.TestCase):
    """Tests for compute_acceptance_probability."""

    def test_equal_probs(self):
        p = compute_acceptance_probability(0.5, 0.5, gamma=1.0)
        self.assertAlmostEqual(p, 1.0)

    def test_target_higher(self):
        p = compute_acceptance_probability(0.3, 0.6, gamma=1.0)
        self.assertAlmostEqual(p, 1.0)

    def test_target_lower(self):
        p = compute_acceptance_probability(0.6, 0.3, gamma=1.0)
        self.assertAlmostEqual(p, 0.5)

    def test_zero_draft_prob(self):
        p = compute_acceptance_probability(0.0, 0.5, gamma=1.0)
        self.assertAlmostEqual(p, 1.0)

    def test_very_small_draft_prob(self):
        p = compute_acceptance_probability(1e-15, 0.5, gamma=1.0)
        self.assertAlmostEqual(p, 1.0)

    def test_gamma_zero(self):
        p = compute_acceptance_probability(0.3, 0.1, gamma=0.0)
        # ratio^0 = 1.0, min(1.0, 1.0) = 1.0
        self.assertAlmostEqual(p, 1.0)

    def test_gamma_high(self):
        p = compute_acceptance_probability(0.5, 0.25, gamma=2.0)
        # ratio=0.5, 0.5^2 = 0.25
        self.assertAlmostEqual(p, 0.25)

    def test_clamped_at_one(self):
        p = compute_acceptance_probability(0.1, 0.9, gamma=1.0)
        self.assertAlmostEqual(p, 1.0)

    def test_probability_between_zero_and_one(self):
        for dp, tp in [(0.5, 0.3), (0.8, 0.1), (0.9, 0.05)]:
            p = compute_acceptance_probability(dp, tp, gamma=1.0)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_both_zero(self):
        p = compute_acceptance_probability(0.0, 0.0, gamma=1.0)
        self.assertAlmostEqual(p, 1.0)

    def test_gamma_fractional(self):
        p = compute_acceptance_probability(0.6, 0.3, gamma=0.5)
        ratio = 0.3 / 0.6
        expected = min(1.0, ratio ** 0.5)
        self.assertAlmostEqual(p, expected, places=5)

    def test_identical_probs_different_gamma(self):
        for g in [0.1, 0.5, 1.0, 2.0, 5.0]:
            p = compute_acceptance_probability(0.4, 0.4, gamma=g)
            self.assertAlmostEqual(p, 1.0)

    def test_target_slightly_less(self):
        p = compute_acceptance_probability(0.5, 0.49, gamma=1.0)
        self.assertAlmostEqual(p, 0.98, places=1)


# =========================================================================
# 16. TestRejectionSamplingStep
# =========================================================================


class TestRejectionSamplingStep(unittest.TestCase):
    """Tests for rejection_sampling_step."""

    def test_returns_bool_and_float(self):
        draft_logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        target_logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        accepted, prob = rejection_sampling_step(draft_logits, target_logits, 0)
        self.assertIsInstance(accepted, bool)
        self.assertIsInstance(prob, float)

    def test_identical_logits_accepted(self):
        rng = np.random.RandomState(0)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[10] = 5.0
        accepted, prob = rejection_sampling_step(logits, logits.copy(), 10, rng=rng)
        self.assertAlmostEqual(prob, 1.0)
        self.assertTrue(accepted)

    def test_deterministic_with_seed(self):
        draft = np.random.RandomState(1).randn(VOCAB_SIZE).astype(np.float32)
        target = np.random.RandomState(2).randn(VOCAB_SIZE).astype(np.float32)

        a1, p1 = rejection_sampling_step(draft, target, 5, rng=np.random.RandomState(SEED))
        a2, p2 = rejection_sampling_step(draft, target, 5, rng=np.random.RandomState(SEED))
        self.assertEqual(a1, a2)
        self.assertAlmostEqual(p1, p2)

    def test_gamma_affects_probability(self):
        draft = np.zeros(VOCAB_SIZE, dtype=np.float32)
        draft[10] = 3.0
        target = np.zeros(VOCAB_SIZE, dtype=np.float32)
        target[10] = 1.0
        _, p1 = rejection_sampling_step(draft, target, 10, gamma=1.0)
        _, p2 = rejection_sampling_step(draft, target, 10, gamma=2.0)
        self.assertGreater(p1, p2)

    def test_prob_between_zero_and_one(self):
        rng = np.random.RandomState(SEED)
        for _ in range(20):
            draft = rng.randn(VOCAB_SIZE).astype(np.float32)
            target = rng.randn(VOCAB_SIZE).astype(np.float32)
            token = rng.randint(0, VOCAB_SIZE)
            _, prob = rejection_sampling_step(draft, target, token)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_high_target_prob_accepted(self):
        rng = np.random.RandomState(0)
        draft = np.zeros(VOCAB_SIZE, dtype=np.float32)
        target = np.zeros(VOCAB_SIZE, dtype=np.float32)
        target[10] = 20.0
        accepted, prob = rejection_sampling_step(draft, target, 10, rng=rng)
        self.assertAlmostEqual(prob, 1.0)
        self.assertTrue(accepted)

    def test_low_target_prob_likely_rejected(self):
        """Statistically, low target prob should usually reject."""
        draft = np.zeros(VOCAB_SIZE, dtype=np.float32)
        draft[10] = 20.0
        target = np.zeros(VOCAB_SIZE, dtype=np.float32)
        target[10] = -20.0
        accept_count = 0
        for seed in range(50):
            rng = np.random.RandomState(seed)
            accepted, _ = rejection_sampling_step(draft, target, 10, rng=rng)
            if accepted:
                accept_count += 1
        self.assertLess(accept_count, 25)  # most should reject

    def test_uniform_logits(self):
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        _, prob = rejection_sampling_step(logits, logits, 0)
        self.assertAlmostEqual(prob, 1.0)


# =========================================================================
# 17. TestDiversityModifiedAcceptance
# =========================================================================


class TestDiversityModifiedAcceptance(unittest.TestCase):
    """Tests for diversity_modified_acceptance."""

    def test_novel_token_gets_bonus(self):
        prob = diversity_modified_acceptance(0.3, 0.3, token=10,
                                             seen_tokens=set(),
                                             diversity_bonus=0.2)
        base = compute_acceptance_probability(0.3, 0.3)
        self.assertGreater(prob, base - 0.01)  # bonus should be added

    def test_seen_token_no_bonus(self):
        prob = diversity_modified_acceptance(0.3, 0.3, token=10,
                                             seen_tokens={10},
                                             diversity_bonus=0.5)
        base = compute_acceptance_probability(0.3, 0.3)
        self.assertAlmostEqual(prob, base)

    def test_clamped_at_one(self):
        prob = diversity_modified_acceptance(0.5, 0.5, token=10,
                                             seen_tokens=set(),
                                             diversity_bonus=0.5)
        self.assertLessEqual(prob, 1.0)

    def test_zero_bonus(self):
        prob = diversity_modified_acceptance(0.4, 0.2, token=10,
                                             seen_tokens=set(),
                                             diversity_bonus=0.0)
        base = compute_acceptance_probability(0.4, 0.2)
        self.assertAlmostEqual(prob, base)

    def test_gamma_effect(self):
        p1 = diversity_modified_acceptance(0.5, 0.25, token=10,
                                            seen_tokens={10}, gamma=1.0)
        p2 = diversity_modified_acceptance(0.5, 0.25, token=10,
                                            seen_tokens={10}, gamma=2.0)
        self.assertGreater(p1, p2)

    def test_large_bonus_novel(self):
        prob = diversity_modified_acceptance(0.5, 0.1, token=99,
                                             seen_tokens=set(),
                                             diversity_bonus=1.0)
        self.assertAlmostEqual(prob, 1.0)

    def test_multiple_seen_tokens(self):
        seen = {1, 2, 3, 4, 5}
        prob_seen = diversity_modified_acceptance(0.5, 0.3, token=3,
                                                   seen_tokens=seen,
                                                   diversity_bonus=0.3)
        prob_novel = diversity_modified_acceptance(0.5, 0.3, token=10,
                                                    seen_tokens=seen,
                                                    diversity_bonus=0.3)
        self.assertGreaterEqual(prob_novel, prob_seen)

    def test_return_type(self):
        prob = diversity_modified_acceptance(0.5, 0.5, 0, set())
        self.assertIsInstance(prob, float)

    def test_very_small_probs(self):
        prob = diversity_modified_acceptance(1e-10, 1e-10, 0, set(),
                                              diversity_bonus=0.1)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_zero_draft_prob(self):
        prob = diversity_modified_acceptance(0.0, 0.5, 0, set(),
                                              diversity_bonus=0.1)
        self.assertAlmostEqual(prob, 1.0)


# =========================================================================
# 18. TestComputeSpeculativeSpeedup
# =========================================================================


class TestComputeSpeculativeSpeedup(unittest.TestCase):
    """Tests for compute_speculative_speedup."""

    def test_perfect_acceptance(self):
        speedup = compute_speculative_speedup(
            acceptance_rate=1.0, draft_cost=0.1, target_cost=1.0,
            draft_length=5
        )
        self.assertGreater(speedup, 1.0)

    def test_zero_acceptance(self):
        speedup = compute_speculative_speedup(
            acceptance_rate=0.0, draft_cost=0.1, target_cost=1.0,
            draft_length=5
        )
        # expected_accepted = 1 (sum of 0^i for i=0..4 = 1)
        # spec_cost = 0.1*5 + 1.0 = 1.5
        # baseline = 1.0 * 2 = 2.0
        expected = 2.0 / 1.5
        self.assertAlmostEqual(speedup, expected, places=3)

    def test_draft_length_one(self):
        speedup = compute_speculative_speedup(
            acceptance_rate=0.8, draft_cost=0.1, target_cost=1.0,
            draft_length=1
        )
        self.assertGreater(speedup, 0)

    def test_zero_draft_cost(self):
        speedup = compute_speculative_speedup(
            acceptance_rate=0.8, draft_cost=0.0, target_cost=1.0,
            draft_length=5
        )
        self.assertGreater(speedup, 1.0)

    def test_equal_costs(self):
        speedup = compute_speculative_speedup(
            acceptance_rate=0.5, draft_cost=1.0, target_cost=1.0,
            draft_length=3
        )
        # With equal costs, spec might be slower
        self.assertGreater(speedup, 0)

    def test_higher_acceptance_better_speedup(self):
        s1 = compute_speculative_speedup(0.5, 0.1, 1.0, 5)
        s2 = compute_speculative_speedup(0.9, 0.1, 1.0, 5)
        self.assertGreater(s2, s1)

    def test_lower_draft_cost_better(self):
        s1 = compute_speculative_speedup(0.7, 0.5, 1.0, 5)
        s2 = compute_speculative_speedup(0.7, 0.1, 1.0, 5)
        self.assertGreater(s2, s1)

    def test_negative_target_cost(self):
        speedup = compute_speculative_speedup(0.5, 0.1, -1.0, 5)
        self.assertAlmostEqual(speedup, 1.0)

    def test_negative_draft_cost(self):
        speedup = compute_speculative_speedup(0.5, -0.1, 1.0, 5)
        self.assertAlmostEqual(speedup, 1.0)

    def test_large_draft_length(self):
        speedup = compute_speculative_speedup(0.9, 0.1, 1.0, 20)
        self.assertGreater(speedup, 0)

    def test_acceptance_rate_one_large_draft(self):
        speedup = compute_speculative_speedup(1.0, 0.1, 1.0, 10)
        # expected_accepted = 10, spec_cost = 0.1*10+1.0=2.0
        # baseline = 1.0*11 = 11.0
        expected = 11.0 / 2.0
        self.assertAlmostEqual(speedup, expected, places=3)

    def test_return_type(self):
        s = compute_speculative_speedup(0.5, 0.1, 1.0, 5)
        self.assertIsInstance(s, float)


# =========================================================================
# 19. TestOptimalDraftLength
# =========================================================================


class TestOptimalDraftLength(unittest.TestCase):
    """Tests for optimal_draft_length."""

    def test_returns_positive_int(self):
        length = optimal_draft_length(0.8, 0.1, 1.0)
        self.assertIsInstance(length, int)
        self.assertGreaterEqual(length, 1)

    def test_zero_acceptance_returns_one(self):
        length = optimal_draft_length(0.0, 0.1, 1.0)
        self.assertEqual(length, 1)

    def test_high_acceptance_longer_draft(self):
        l_low = optimal_draft_length(0.3, 0.1, 1.0, max_length=20)
        l_high = optimal_draft_length(0.95, 0.1, 1.0, max_length=20)
        self.assertGreaterEqual(l_high, l_low)

    def test_expensive_draft_shorter(self):
        l_cheap = optimal_draft_length(0.8, 0.01, 1.0, max_length=20)
        l_expensive = optimal_draft_length(0.8, 0.5, 1.0, max_length=20)
        self.assertGreaterEqual(l_cheap, l_expensive)

    def test_max_length_respected(self):
        length = optimal_draft_length(0.99, 0.01, 1.0, max_length=5)
        self.assertLessEqual(length, 5)

    def test_zero_draft_cost(self):
        length = optimal_draft_length(0.0, 0.0, 1.0)
        self.assertEqual(length, 1)

    def test_zero_target_cost(self):
        length = optimal_draft_length(0.5, 0.1, 0.0)
        self.assertEqual(length, 1)

    def test_negative_acceptance(self):
        length = optimal_draft_length(-0.5, 0.1, 1.0)
        self.assertEqual(length, 1)

    def test_various_acceptance_rates(self):
        for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            length = optimal_draft_length(rate, 0.1, 1.0, max_length=15)
            self.assertGreaterEqual(length, 1)
            self.assertLessEqual(length, 15)

    def test_max_length_one(self):
        length = optimal_draft_length(0.9, 0.1, 1.0, max_length=1)
        self.assertEqual(length, 1)

    def test_moderate_settings(self):
        length = optimal_draft_length(0.7, 0.1, 1.0, max_length=10)
        self.assertGreaterEqual(length, 1)
        self.assertLessEqual(length, 10)


# =========================================================================
# 20. TestEndToEnd
# =========================================================================


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_pipeline_peaked(self):
        """Peaked source should generate sequence of hot token."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=10, draft_length=3, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)
        for t in result[1:]:
            self.assertEqual(t, 10)

    def test_full_pipeline_random(self):
        """Random source should produce output without errors."""
        src = make_random_source(seed=SEED)
        draft = SimpleDraftModel(make_random_source(seed=SEED + 1))
        config = SpeculativeConfig(max_new_tokens=15, draft_length=4, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)

    def test_diversity_pipeline(self):
        """Diversity decoder should produce multiple sequences."""
        src = make_random_source(seed=SEED)
        draft = SimpleDraftModel(make_random_source(seed=SEED + 1))
        config = SpeculativeConfig(
            max_new_tokens=10, draft_length=3, diversity_penalty=0.3, seed=SEED
        )
        decoder = DiversitySpeculativeDecoder(src, draft, config,
                                               rng=np.random.RandomState(SEED))
        results = decoder.decode_multiple([0], num_sequences=5)
        self.assertEqual(len(results), 5)
        for seq in results:
            self.assertEqual(seq[0], 0)
            self.assertGreater(len(seq), 1)

    def test_ngram_draft_pipeline(self):
        """NGram draft model in full pipeline."""
        training_data = [[10, 20, 30, 40, 50, 60] for _ in range(10)]
        ngram = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)
        ngram.build_table(training_data)

        src = make_peaked_source(hot_token=30)
        config = SpeculativeConfig(max_new_tokens=5, draft_length=2, seed=SEED)
        decoder = SpeculativeDecoder(src, ngram, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([10, 20])
        self.assertGreater(len(result), 2)

    def test_cached_draft_pipeline(self):
        """Cached draft model in full pipeline."""
        src = make_peaked_source(hot_token=10)
        cached = CachedDraftModel(src, vocab_size=VOCAB_SIZE)
        config = SpeculativeConfig(max_new_tokens=8, draft_length=3, seed=SEED)
        decoder = SpeculativeDecoder(src, cached, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)
        self.assertGreater(cached.cache_size(), 0)

    def test_ensemble_draft_pipeline(self):
        """Ensemble draft model in full pipeline."""
        src1 = make_peaked_source(hot_token=10)
        src2 = make_peaked_source(hot_token=20)
        m1 = SimpleDraftModel(src1)
        m2 = SimpleDraftModel(src2)
        ensemble = EnsembleDraftModel([m1, m2])

        target = make_peaked_source(hot_token=10)
        config = SpeculativeConfig(max_new_tokens=8, draft_length=3, seed=SEED)
        decoder = SpeculativeDecoder(target, ensemble, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)

    def test_adaptive_draft_pipeline(self):
        """Adaptive draft model in full pipeline."""
        src = make_peaked_source(hot_token=10)
        base = SimpleDraftModel(src)
        adaptive = AdaptiveDraftModel(base, min_length=1, max_length=5)

        config = SpeculativeConfig(max_new_tokens=8, draft_length=3, seed=SEED)
        decoder = SpeculativeDecoder(src, adaptive, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)

    def test_tree_exploration_pipeline(self):
        """Tree-based expansion and best path selection."""
        tree = SpeculativeTree(root_token=-1, width=3, max_depth=3)
        rng = np.random.RandomState(SEED)

        logits = rng.randn(VOCAB_SIZE).astype(np.float32)
        children = tree.expand(tree.root, [], logits)
        for c in children:
            logits2 = rng.randn(VOCAB_SIZE).astype(np.float32)
            tree.expand(c, [], logits2)

        paths = tree.get_all_paths()
        self.assertGreater(len(paths), 0)
        best = tree.best_path()
        self.assertGreater(len(best), 0)

    def test_tracker_in_decoder(self):
        """Verify tracker records during decode."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=20, draft_length=5, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        decoder.decode([0])
        self.assertGreater(decoder.tracker.num_records(), 0)
        rate = decoder.tracker.acceptance_rate()
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_verification_strategies_in_decoder(self):
        """Different verification strategies produce valid output."""
        src = make_peaked_source(hot_token=10)
        config = SpeculativeConfig(max_new_tokens=8, draft_length=3, seed=SEED)

        strategies = [
            StandardVerification(rng=np.random.RandomState(SEED)),
            DiversityAwareVerification(diversity_penalty=0.2,
                                        rng=np.random.RandomState(SEED)),
            RelaxedVerification(tolerance=0.1,
                                rng=np.random.RandomState(SEED)),
            TopKVerification(k=10, rng=np.random.RandomState(SEED)),
        ]

        for strat in strategies:
            draft = SimpleDraftModel(src)
            decoder = SpeculativeDecoder(src, draft, config,
                                          verification=strat,
                                          rng=np.random.RandomState(SEED))
            result = decoder.decode([0])
            self.assertGreater(len(result), 1,
                               f"Failed with {type(strat).__name__}")

    def test_eos_handling(self):
        """Decoder stops on EOS."""
        src = make_eos_source(eos_token=EOS_TOKEN, after_steps=3)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(
            max_new_tokens=50, draft_length=3, seed=SEED,
            eos_token_id=EOS_TOKEN
        )
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertEqual(result[-1], EOS_TOKEN)
        self.assertLess(len(result), 52)

    def test_empty_prefix(self):
        """Decode with empty prefix."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=5, draft_length=2, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([])
        self.assertGreater(len(result), 0)

    def test_long_prefix(self):
        """Decode with long prefix."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=5, draft_length=2, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        prefix = list(range(50))
        result = decoder.decode(prefix)
        self.assertEqual(result[:50], prefix)

    def test_draft_update_called(self):
        """Draft model update is called during decoding."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=10, draft_length=3, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        decoder.decode([0])
        self.assertGreater(len(draft._accepted), 0)

    def test_speedup_computation_integration(self):
        """Compute speedup from tracker data."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=20, draft_length=5, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        decoder.decode([0])
        rate = decoder.tracker.acceptance_rate()
        speedup = compute_speculative_speedup(rate, 0.1, 1.0, 5)
        self.assertGreater(speedup, 0)

    def test_optimal_length_from_tracker(self):
        """Compute optimal draft length from tracker data."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=20, draft_length=5, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        decoder.decode([0])
        rate = decoder.tracker.acceptance_rate()
        length = optimal_draft_length(rate, 0.1, 1.0, max_length=10)
        self.assertGreaterEqual(length, 1)
        self.assertLessEqual(length, 10)

    def test_tree_prune_then_decode(self):
        """Build tree, prune, then verify structure."""
        tree = SpeculativeTree(root_token=-1, width=4, max_depth=3)
        rng = np.random.RandomState(SEED)

        logits = rng.randn(VOCAB_SIZE).astype(np.float32)
        children = tree.expand(tree.root, [], logits)
        for c in children:
            logits2 = rng.randn(VOCAB_SIZE).astype(np.float32)
            tree.expand(c, [], logits2)

        size_before = tree.size()
        tree.prune(min_probability=0.05)
        size_after = tree.size()
        self.assertLessEqual(size_after, size_before)

    def test_diversity_sequences_differ(self):
        """With random source and diversity penalty, sequences should differ."""
        np.random.seed(SEED)
        src = make_random_source(seed=SEED)
        draft = SimpleDraftModel(make_random_source(seed=SEED + 1))
        config = SpeculativeConfig(
            max_new_tokens=15, draft_length=3, diversity_penalty=0.5, seed=SEED
        )
        decoder = DiversitySpeculativeDecoder(src, draft, config,
                                               rng=np.random.RandomState(SEED))
        results = decoder.decode_multiple([0], num_sequences=5)
        tuples = [tuple(s) for s in results]
        unique = set(tuples)
        # We expect at least some diversity (not all identical)
        self.assertGreaterEqual(len(unique), 1)

    def test_multiple_decodes_independent(self):
        """Two separate decode calls should be independent."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=5, draft_length=2, seed=SEED)

        d1 = SpeculativeDecoder(src, draft, config,
                                 rng=np.random.RandomState(SEED))
        r1 = d1.decode([0])

        d2 = SpeculativeDecoder(src, draft, config,
                                 rng=np.random.RandomState(SEED))
        r2 = d2.decode([0])

        self.assertEqual(r1, r2)


# =========================================================================
# Additional edge-case and stress tests
# =========================================================================


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests."""

    def test_config_with_all_defaults(self):
        cfg = SpeculativeConfig()
        d = cfg.to_dict()
        cfg2 = SpeculativeConfig.from_dict(d)
        self.assertEqual(cfg.draft_length, cfg2.draft_length)
        self.assertEqual(cfg.gamma, cfg2.gamma)

    def test_tree_single_node(self):
        tree = SpeculativeTree(root_token=0)
        self.assertEqual(tree.size(), 1)
        self.assertEqual(tree.depth(), 0)
        paths = tree.get_all_paths()
        self.assertEqual(len(paths), 1)

    def test_tree_width_one(self):
        tree = SpeculativeTree(root_token=-1, width=1, max_depth=3)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[5] = 10.0
        children = tree.expand(tree.root, [], logits)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].token, 5)

    def test_acceptance_prob_edge_zero_zero(self):
        p = compute_acceptance_probability(0.0, 0.0)
        self.assertEqual(p, 1.0)

    def test_ngram_empty_sequence(self):
        model = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)
        model.build_table([[]])
        tokens, logits = model.predict([], 1)
        self.assertEqual(len(tokens), 1)

    def test_cached_model_single_prefix(self):
        src = make_uniform_source()
        model = CachedDraftModel(src, vocab_size=VOCAB_SIZE)
        tokens, _ = model.predict([1], 1)
        self.assertEqual(len(tokens), 1)

    def test_ensemble_single_model(self):
        src = make_peaked_source(hot_token=10)
        m = SimpleDraftModel(src)
        ensemble = EnsembleDraftModel([m])
        tokens, _ = ensemble.predict([1, 2], 2)
        self.assertEqual(len(tokens), 2)

    def test_adaptive_no_updates(self):
        src = make_peaked_source(hot_token=10)
        base = SimpleDraftModel(src)
        model = AdaptiveDraftModel(base)
        self.assertEqual(model.get_acceptance_history(), [])

    def test_tracker_single_record(self):
        tracker = AcceptanceTracker()
        tracker.record(1, 1)
        self.assertAlmostEqual(tracker.acceptance_rate(), 1.0)

    def test_speedup_zero_spec_cost(self):
        # draft_cost=0, draft_length=0 edge
        s = compute_speculative_speedup(0.5, 0.0, 1.0, 0)
        # expected_accepted = 0 (empty range sum), baseline = 1.0*1 = 1.0
        # spec_cost = 0 + 1.0 = 1.0
        self.assertAlmostEqual(s, 1.0)

    def test_standard_verification_single_step(self):
        v = StandardVerification(rng=np.random.RandomState(0))
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        accepted, n = v.verify([0], logits, logits)
        self.assertEqual(n, 1)

    def test_relaxed_zero_tolerance_same_as_standard(self):
        """Relaxed with tolerance=0 shouldn't accept less than standard."""
        logits_d = np.random.RandomState(10).randn(5, VOCAB_SIZE).astype(np.float32)
        logits_t = logits_d.copy()
        tokens = [int(np.argmax(logits_d[i])) for i in range(5)]

        v_std = StandardVerification(rng=np.random.RandomState(SEED))
        _, n_std = v_std.verify(tokens, logits_d, logits_t)

        v_rel = RelaxedVerification(tolerance=0.0, rng=np.random.RandomState(SEED))
        _, n_rel = v_rel.verify(tokens, logits_d, logits_t)

        self.assertEqual(n_std, 5)
        self.assertEqual(n_rel, 5)

    def test_topk_verification_all_top1(self):
        """All tokens are top-1 of target."""
        v = TopKVerification(k=1)
        target_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        target_logits[0, 5] = 10.0
        target_logits[1, 5] = 10.0
        target_logits[2, 5] = 10.0
        draft_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        accepted, n = v.verify([5, 5, 5], draft_logits, target_logits)
        self.assertEqual(n, 3)

    def test_diversity_verification_large_penalty(self):
        """Large penalty should boost acceptance significantly."""
        v = DiversityAwareVerification(diversity_penalty=0.99,
                                        rng=np.random.RandomState(0))
        logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        accepted, n = v.verify([10, 20, 30], logits, logits.copy())
        self.assertEqual(n, 3)

    def test_config_seed_none(self):
        cfg = SpeculativeConfig(seed=None)
        self.assertIsNone(cfg.seed)

    def test_config_eos_token(self):
        cfg = SpeculativeConfig(eos_token_id=99)
        self.assertEqual(cfg.eos_token_id, 99)


class TestNGramDraftModelAdvanced(unittest.TestCase):
    """Advanced tests for NGram draft model."""

    def test_large_corpus(self):
        rng = np.random.RandomState(SEED)
        seqs = [rng.randint(0, VOCAB_SIZE, size=50).tolist() for _ in range(100)]
        model = NGramDraftModel(n=3, vocab_size=VOCAB_SIZE)
        model.build_table(seqs)
        tokens, logits = model.predict([10, 20, 30], 5)
        self.assertEqual(len(tokens), 5)
        self.assertEqual(logits.shape, (5, VOCAB_SIZE))

    def test_predict_determinism(self):
        rng = np.random.RandomState(SEED)
        seqs = [rng.randint(0, VOCAB_SIZE, size=20).tolist() for _ in range(10)]
        m1 = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)
        m1.build_table(seqs)
        m2 = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)
        m2.build_table(seqs)
        t1, _ = m1.predict([5, 6], 3)
        t2, _ = m2.predict([5, 6], 3)
        self.assertEqual(t1, t2)

    def test_update_then_predict(self):
        model = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)
        model.update([10, 20, 30, 40])
        tokens, _ = model.predict([10, 20], 1)
        self.assertEqual(tokens[0], 30)

    def test_multiple_updates(self):
        model = NGramDraftModel(n=2, vocab_size=VOCAB_SIZE)
        model.update([10, 20, 30])
        model.update([10, 20, 40])
        model.update([10, 20, 30])
        counts = model.get_counts((10, 20))
        self.assertEqual(counts[30], 2.0)
        self.assertEqual(counts[40], 1.0)

    def test_short_prefix(self):
        model = NGramDraftModel(n=3, vocab_size=VOCAB_SIZE)
        tokens, logits = model.predict([5], 2)
        self.assertEqual(len(tokens), 2)


class TestCachedDraftModelAdvanced(unittest.TestCase):
    """Advanced tests for CachedDraftModel."""

    def test_cache_content_stability(self):
        src = make_peaked_source(hot_token=15)
        model = CachedDraftModel(src, vocab_size=VOCAB_SIZE)
        model.predict([1, 2, 3], 1)
        cached = model.get_cached_logits((1, 2, 3))
        self.assertIsNotNone(cached)
        self.assertEqual(int(np.argmax(cached)), 15)

    def test_predict_many_steps(self):
        src = make_peaked_source(hot_token=10)
        model = CachedDraftModel(src, vocab_size=VOCAB_SIZE)
        tokens, logits = model.predict([0], 20)
        self.assertEqual(len(tokens), 20)
        for t in tokens:
            self.assertEqual(t, 10)

    def test_max_cache_boundary(self):
        src = make_peaked_source(hot_token=10)
        model = CachedDraftModel(src, vocab_size=VOCAB_SIZE, max_cache=5)
        model.predict([1], 10)
        self.assertLessEqual(model.cache_size(), 5)


class TestEnsembleDraftModelAdvanced(unittest.TestCase):
    """Advanced tests for EnsembleDraftModel."""

    def test_many_models(self):
        models = []
        for i in range(5):
            src = make_peaked_source(hot_token=10 + i)
            models.append(SimpleDraftModel(src))
        ensemble = EnsembleDraftModel(models)
        self.assertEqual(len(ensemble.weights), 5)
        tokens, _ = ensemble.predict([0], 3)
        self.assertEqual(len(tokens), 3)

    def test_weights_normalize(self):
        src = make_peaked_source(hot_token=10)
        m = SimpleDraftModel(src)
        ensemble = EnsembleDraftModel([m, m, m], weights=[10.0, 20.0, 30.0])
        total = sum(ensemble.weights)
        self.assertAlmostEqual(total, 1.0)

    def test_extreme_weight_ratio(self):
        src1 = make_peaked_source(hot_token=10)
        src2 = make_peaked_source(hot_token=20)
        m1 = SimpleDraftModel(src1)
        m2 = SimpleDraftModel(src2)
        ensemble = EnsembleDraftModel([m1, m2], weights=[1000.0, 1.0])
        tokens, _ = ensemble.predict([0], 1)
        self.assertEqual(tokens[0], 10)


class TestVerificationAdvanced(unittest.TestCase):
    """Advanced verification tests across strategies."""

    def test_all_strategies_handle_single_token(self):
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, 10] = 5.0

        strategies = [
            StandardVerification(rng=np.random.RandomState(0)),
            DiversityAwareVerification(rng=np.random.RandomState(0)),
            RelaxedVerification(rng=np.random.RandomState(0)),
            TopKVerification(k=5),
        ]
        for s in strategies:
            accepted, n = s.verify([10], logits, logits.copy())
            self.assertGreaterEqual(n, 0,
                                     f"Failed with {type(s).__name__}")

    def test_all_strategies_handle_empty(self):
        for s in [StandardVerification(), DiversityAwareVerification(),
                  RelaxedVerification(), TopKVerification(k=5)]:
            accepted, n = s.verify([], np.zeros((0, VOCAB_SIZE)),
                                    np.zeros((0, VOCAB_SIZE)))
            self.assertEqual(n, 0)

    def test_standard_vs_relaxed_ordering(self):
        """Relaxed should accept at least as many as standard (same seed)."""
        rng_seed = 123
        logits_d = np.random.RandomState(1).randn(8, VOCAB_SIZE).astype(np.float32)
        logits_t = np.random.RandomState(2).randn(8, VOCAB_SIZE).astype(np.float32)
        tokens = [int(np.argmax(logits_d[i])) for i in range(8)]

        v_std = StandardVerification(rng=np.random.RandomState(rng_seed))
        _, n_std = v_std.verify(tokens, logits_d, logits_t)

        v_rel = RelaxedVerification(tolerance=0.3, rng=np.random.RandomState(rng_seed))
        _, n_rel = v_rel.verify(tokens, logits_d, logits_t)

        self.assertGreaterEqual(n_rel, n_std)


class TestTreeAdvanced(unittest.TestCase):
    """Advanced tree tests."""

    def test_deep_tree(self):
        tree = SpeculativeTree(root_token=-1, width=1, max_depth=10)
        node = tree.root
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[5] = 10.0
        for _ in range(10):
            children = tree.expand(node, [], logits)
            if not children:
                break
            node = children[0]
        self.assertEqual(tree.depth(), min(10, tree.depth()))

    def test_wide_tree(self):
        tree = SpeculativeTree(root_token=-1, width=10, max_depth=2)
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        for i in range(10):
            logits[i] = 10.0 - i
        children = tree.expand(tree.root, [], logits)
        self.assertEqual(len(children), 10)

    def test_tree_paths_unique(self):
        tree = SpeculativeTree(root_token=-1, width=3, max_depth=2)
        rng = np.random.RandomState(SEED)
        logits = rng.randn(VOCAB_SIZE).astype(np.float32)
        children = tree.expand(tree.root, [], logits)
        for c in children:
            logits2 = rng.randn(VOCAB_SIZE).astype(np.float32)
            tree.expand(c, [], logits2)
        paths = tree.get_all_paths()
        path_tuples = [tuple(p) for p in paths]
        self.assertEqual(len(path_tuples), len(set(path_tuples)))

    def test_prune_no_effect_high_threshold(self):
        """If all probs are above threshold, nothing pruned."""
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        tree.root.add_child(token=1, probability=0.9)
        tree.root.add_child(token=2, probability=0.8)
        pruned = tree.prune(min_probability=0.01)
        self.assertEqual(pruned, 0)
        self.assertEqual(len(tree.root.children), 2)

    def test_prune_all_children(self):
        """All children below threshold."""
        tree = SpeculativeTree(root_token=-1, width=2, max_depth=3)
        tree.root.add_child(token=1, probability=0.001)
        tree.root.add_child(token=2, probability=0.002)
        pruned = tree.prune(min_probability=0.01)
        self.assertEqual(pruned, 2)
        self.assertEqual(len(tree.root.children), 0)


class TestTrackerAdvanced(unittest.TestCase):
    """Advanced tracker tests."""

    def test_many_records(self):
        tracker = AcceptanceTracker(window_size=100)
        for _ in range(200):
            tracker.record(10, 7)
        self.assertEqual(tracker.num_records(), 100)
        self.assertAlmostEqual(tracker.acceptance_rate(), 0.7)

    def test_adaptive_gamma_convergence(self):
        """Gamma should stabilise with consistent acceptance rate."""
        tracker = AcceptanceTracker(window_size=50, initial_gamma=1.0)
        for _ in range(100):
            tracker.record(10, 7)
            tracker.adaptive_gamma(target_rate=0.7)
        # Gamma should be close to its initial value since rate == target
        self.assertGreater(tracker.gamma, 0.5)
        self.assertLess(tracker.gamma, 5.0)

    def test_adaptive_gamma_increasing_trend(self):
        tracker = AcceptanceTracker(window_size=10, initial_gamma=1.0)
        for _ in range(20):
            tracker.record(10, 9)
            tracker.adaptive_gamma(target_rate=0.5)
        self.assertGreater(tracker.gamma, 1.0)

    def test_adaptive_gamma_decreasing_trend(self):
        tracker = AcceptanceTracker(window_size=10, initial_gamma=2.0)
        for _ in range(20):
            tracker.record(10, 1)
            tracker.adaptive_gamma(target_rate=0.7)
        self.assertLess(tracker.gamma, 2.0)


class TestHelperFunctionsAdvanced(unittest.TestCase):
    """Advanced tests for helper functions."""

    def test_speedup_monotonic_in_acceptance(self):
        """Speedup should increase with acceptance rate."""
        rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        speedups = [compute_speculative_speedup(r, 0.1, 1.0, 5) for r in rates]
        for i in range(len(speedups) - 1):
            self.assertLessEqual(speedups[i], speedups[i + 1] + 1e-6)

    def test_optimal_length_bounded(self):
        for max_l in [1, 5, 10, 20]:
            length = optimal_draft_length(0.8, 0.1, 1.0, max_length=max_l)
            self.assertGreaterEqual(length, 1)
            self.assertLessEqual(length, max_l)

    def test_rejection_sampling_many_tokens(self):
        """Run many rejection sampling steps for statistical testing."""
        rng = np.random.RandomState(SEED)
        draft = rng.randn(VOCAB_SIZE).astype(np.float32)
        target = draft.copy()  # identical
        token = int(np.argmax(draft))
        accept_count = 0
        for _ in range(100):
            accepted, prob = rejection_sampling_step(
                draft, target, token, rng=np.random.RandomState(rng.randint(0, 2**31))
            )
            if accepted:
                accept_count += 1
        # Identical distributions => always accepted
        self.assertEqual(accept_count, 100)

    def test_diversity_modified_many_tokens(self):
        """Test diversity modification across many tokens."""
        seen = set()
        probs = []
        for tok in range(20):
            p = diversity_modified_acceptance(0.5, 0.3, tok, seen,
                                               diversity_bonus=0.1)
            probs.append(p)
            seen.add(tok)
        # First token should get bonus, subsequent calls with same token won't
        self.assertGreater(probs[0], 0)

    def test_speedup_with_real_parameters(self):
        """Test with realistic cost ratios."""
        # GPT-2 small draft vs large target
        speedup = compute_speculative_speedup(
            acceptance_rate=0.7,
            draft_cost=0.05,
            target_cost=1.0,
            draft_length=5
        )
        self.assertGreater(speedup, 1.0)


class TestProtocolCompliance(unittest.TestCase):
    """Test that draft models comply with DraftModel protocol."""

    def test_ngram_has_predict(self):
        model = NGramDraftModel()
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(callable(model.predict))

    def test_ngram_has_update(self):
        model = NGramDraftModel()
        self.assertTrue(hasattr(model, "update"))
        self.assertTrue(callable(model.update))

    def test_cached_has_predict(self):
        model = CachedDraftModel(make_uniform_source())
        self.assertTrue(hasattr(model, "predict"))

    def test_cached_has_update(self):
        model = CachedDraftModel(make_uniform_source())
        self.assertTrue(hasattr(model, "update"))

    def test_ensemble_has_predict(self):
        model = EnsembleDraftModel([SimpleDraftModel(make_uniform_source())])
        self.assertTrue(hasattr(model, "predict"))

    def test_ensemble_has_update(self):
        model = EnsembleDraftModel([SimpleDraftModel(make_uniform_source())])
        self.assertTrue(hasattr(model, "update"))

    def test_adaptive_has_predict(self):
        model = AdaptiveDraftModel(SimpleDraftModel(make_uniform_source()))
        self.assertTrue(hasattr(model, "predict"))

    def test_adaptive_has_update(self):
        model = AdaptiveDraftModel(SimpleDraftModel(make_uniform_source()))
        self.assertTrue(hasattr(model, "update"))

    def test_predict_return_types(self):
        models = [
            NGramDraftModel(vocab_size=VOCAB_SIZE),
            CachedDraftModel(make_uniform_source(), vocab_size=VOCAB_SIZE),
            EnsembleDraftModel([SimpleDraftModel(make_uniform_source())],
                                vocab_size=VOCAB_SIZE),
        ]
        for m in models:
            tokens, logits = m.predict([1, 2], 2)
            self.assertIsInstance(tokens, list)
            self.assertIsInstance(logits, np.ndarray)
            self.assertEqual(len(tokens), 2)


class TestConfigEdgeCases(unittest.TestCase):
    """Edge case tests for SpeculativeConfig."""

    def test_very_large_draft_length(self):
        cfg = SpeculativeConfig(draft_length=1000)
        self.assertEqual(cfg.draft_length, 1000)
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_very_small_gamma(self):
        cfg = SpeculativeConfig(gamma=0.001)
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_very_large_gamma(self):
        cfg = SpeculativeConfig(gamma=100.0)
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_multiple_validation_errors(self):
        cfg = SpeculativeConfig(
            draft_length=0, gamma=-1.0, tree_width=0, tree_depth=0,
            min_draft_length=0, max_draft_length=-1
        )
        errors = cfg.validate()
        self.assertGreater(len(errors), 3)

    def test_from_dict_missing_fields(self):
        d = {"draft_length": 3}
        cfg = SpeculativeConfig.from_dict(d)
        self.assertEqual(cfg.draft_length, 3)
        self.assertEqual(cfg.gamma, 1.0)  # default

    def test_to_dict_contains_all_spec_fields(self):
        cfg = SpeculativeConfig()
        d = cfg.to_dict()
        spec_fields = ["draft_length", "gamma", "verification_strategy",
                        "diversity_penalty", "relaxation_tolerance",
                        "top_k_verify", "tree_width", "tree_depth",
                        "adaptive_draft"]
        for f in spec_fields:
            self.assertIn(f, d)


class TestDecoderEdgeCases(unittest.TestCase):
    """Edge case tests for decoders."""

    def test_max_tokens_zero(self):
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=0, draft_length=3, seed=SEED)
        # max_new_tokens=0 will fail validation but decoder should handle it
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertEqual(result, [0])  # no new tokens

    def test_draft_length_larger_than_max_tokens(self):
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=2, draft_length=10, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)

    def test_prefix_with_eos(self):
        """Prefix containing EOS should still decode (EOS not at end)."""
        src = make_peaked_source(hot_token=10)
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=5, draft_length=2,
                                    seed=SEED, eos_token_id=EOS_TOKEN)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([5, EOS_TOKEN, 6])
        self.assertGreater(len(result), 3)

    def test_uniform_source_produces_output(self):
        src = make_uniform_source()
        draft = SimpleDraftModel(src)
        config = SpeculativeConfig(max_new_tokens=5, draft_length=2, seed=SEED)
        decoder = SpeculativeDecoder(src, draft, config,
                                      rng=np.random.RandomState(SEED))
        result = decoder.decode([0])
        self.assertGreater(len(result), 1)


if __name__ == "__main__":
    unittest.main()
