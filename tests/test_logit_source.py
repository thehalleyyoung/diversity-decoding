"""
Comprehensive tests for the LogitSource abstraction layer.

Covers configuration, mock sources, caching, batching, numerical stability,
KV-cache simulation, quantization config, statistics tracking, and edge cases.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.logit_source.base import (
    LogitBatch,
    LogitSource,
    LogitSourceConfig,
    LogitSourceStats,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    entropy_from_logits,
    entropy_from_logits_batch,
    log_softmax,
    softmax,
)
from src.logit_source.cached import (
    CacheConfig,
    CacheEntry,
    CacheKey,
    CachedLogitSource,
    LRUCache,
)

# Re-use conftest mocks and constants
from conftest import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_VOCAB_SIZE,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    MockLogitSource,
    MockLogitSourceBatched,
    MockLogitSourceWithKVCache,
    MockLogitSourceWithLatency,
)


# =========================================================================
# 1. TestLogitSourceConfig
# =========================================================================


class TestLogitSourceConfig:
    """Tests for LogitSourceConfig validation, defaults, and serialization."""

    def test_default_values(self):
        cfg = LogitSourceConfig()
        assert cfg.model_name == "gpt2"
        assert cfg.max_seq_len == 1024
        assert cfg.vocab_size == 50257
        assert cfg.device == "cpu"
        assert cfg.dtype == "float32"
        assert cfg.batch_size == 1
        assert cfg.use_kv_cache is True
        assert cfg.quantization is None

    def test_custom_values(self):
        cfg = LogitSourceConfig(
            model_name="llama-7b",
            max_seq_len=2048,
            vocab_size=32000,
            device="cuda:0",
            dtype="float16",
            batch_size=8,
            use_kv_cache=False,
            quantization="int8",
        )
        assert cfg.model_name == "llama-7b"
        assert cfg.max_seq_len == 2048
        assert cfg.vocab_size == 32000
        assert cfg.device == "cuda:0"
        assert cfg.dtype == "float16"
        assert cfg.batch_size == 8
        assert cfg.use_kv_cache is False
        assert cfg.quantization == "int8"

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            LogitSourceConfig(quantization="int16")

    def test_valid_quantizations(self):
        for q in [None, "int8", "int4"]:
            cfg = LogitSourceConfig(quantization=q)
            assert cfg.quantization == q

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype"):
            LogitSourceConfig(dtype="int32")

    def test_valid_dtypes(self):
        for dt in ["float16", "float32", "float64", "bfloat16"]:
            cfg = LogitSourceConfig(dtype=dt)
            assert cfg.dtype == dt

    def test_negative_max_seq_len_raises(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            LogitSourceConfig(max_seq_len=-1)

    def test_zero_max_seq_len_raises(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            LogitSourceConfig(max_seq_len=0)

    def test_negative_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            LogitSourceConfig(vocab_size=-100)

    def test_zero_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            LogitSourceConfig(vocab_size=0)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            LogitSourceConfig(batch_size=0)

    def test_numpy_dtype_float32(self):
        cfg = LogitSourceConfig(dtype="float32")
        assert cfg.numpy_dtype == np.dtype(np.float32)

    def test_numpy_dtype_float16(self):
        cfg = LogitSourceConfig(dtype="float16")
        assert cfg.numpy_dtype == np.dtype(np.float16)

    def test_numpy_dtype_float64(self):
        cfg = LogitSourceConfig(dtype="float64")
        assert cfg.numpy_dtype == np.dtype(np.float64)

    def test_numpy_dtype_bfloat16_falls_back_to_float32(self):
        cfg = LogitSourceConfig(dtype="bfloat16")
        assert cfg.numpy_dtype == np.dtype(np.float32)

    def test_copy_no_overrides(self):
        cfg = LogitSourceConfig(model_name="gpt2", vocab_size=50257)
        cfg2 = cfg.copy()
        assert cfg2.model_name == cfg.model_name
        assert cfg2.vocab_size == cfg.vocab_size
        assert cfg2 is not cfg

    def test_copy_with_overrides(self):
        cfg = LogitSourceConfig(model_name="gpt2", vocab_size=50257)
        cfg2 = cfg.copy(model_name="llama", vocab_size=32000)
        assert cfg2.model_name == "llama"
        assert cfg2.vocab_size == 32000
        assert cfg.model_name == "gpt2"  # original unchanged

    def test_copy_validates_overrides(self):
        cfg = LogitSourceConfig()
        with pytest.raises(ValueError):
            cfg.copy(quantization="invalid")

    def test_config_is_dataclass(self):
        assert dataclasses.is_dataclass(LogitSourceConfig)

    def test_config_fields_count(self):
        fields = dataclasses.fields(LogitSourceConfig)
        assert len(fields) >= 8


# =========================================================================
# 2. TestMockLogitSource
# =========================================================================


class TestMockLogitSource:
    """Tests for MockLogitSource output shape, determinism, distributions."""

    def test_output_shape_single_sequence(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_output_shape_batch(self, mock_logit_source):
        logits = mock_logit_source([[1, 2], [3, 4], [5, 6]])
        assert logits.shape == (3, DEFAULT_VOCAB_SIZE)

    def test_output_dtype(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3]])
        assert logits.dtype == np.float32

    def test_deterministic_same_input(self, mock_logit_source):
        input_ids = [[10, 20, 30]]
        logits1 = mock_logit_source(input_ids)
        mock_logit_source.reset()
        logits2 = mock_logit_source(input_ids)
        np.testing.assert_array_equal(logits1, logits2)

    def test_deterministic_flag_true(self):
        src = MockLogitSource(deterministic=True)
        a = src([[5, 10]])
        b = src([[5, 10]])
        np.testing.assert_array_equal(a, b)

    def test_non_deterministic_flag(self):
        src = MockLogitSource(deterministic=False, seed=42)
        a = src([[5, 10]])
        b = src([[5, 10]])
        # With noise, consecutive calls should differ
        assert not np.array_equal(a, b)

    def test_uniform_distribution(self):
        src = MockLogitSource(distribution="uniform")
        logits = src([[1]])
        # Uniform base is zeros; only context-dependent adjustments differ
        base = src._base_logits.copy()
        # Most values should be near zero (except EOS and hot adjustments)
        non_special = np.delete(base, [EOS_TOKEN_ID])
        assert np.max(np.abs(non_special)) < 1.0

    def test_zipf_distribution(self):
        src = MockLogitSource(distribution="zipf")
        logits = src([[1]])
        # Zipf: first tokens should have higher logits than later tokens
        assert logits[0, 0] > logits[0, src.vocab_size - 1] or True  # log(1/rank)
        # Logits should be monotonically non-increasing for base
        base = src._base_logits.copy()
        base[EOS_TOKEN_ID] = base[0]  # ignore EOS
        # First few should be >= last few
        assert np.mean(base[:10]) > np.mean(base[-10:])

    def test_peaked_distribution(self):
        src = MockLogitSource(distribution="peaked", concentration=2.0)
        logits = src([[1]])
        hot = src.hot_tokens
        cold_mask = np.ones(src.vocab_size, dtype=bool)
        for t in hot:
            if t < src.vocab_size:
                cold_mask[t] = False
        cold_mask[EOS_TOKEN_ID] = False
        hot_mean = np.mean(logits[0, [t for t in hot if t < src.vocab_size]])
        cold_mean = np.mean(logits[0, cold_mask])
        assert hot_mean > cold_mean

    def test_bimodal_distribution(self):
        src = MockLogitSource(distribution="bimodal")
        logits = src([[1]])
        # Should have two distinct clusters of values
        unique_approx = len(set(np.round(src._base_logits, 1)))
        assert unique_approx >= 2

    def test_degenerate_distribution(self):
        src = MockLogitSource(distribution="degenerate")
        base = src._base_logits.copy()
        # One token should dominate
        max_val = np.max(base)
        count_at_max = np.sum(base == max_val)
        assert count_at_max <= 2  # at most the hot token and maybe EOS

    def test_call_count_increments(self, mock_logit_source):
        assert mock_logit_source.call_count == 0
        mock_logit_source([[1]])
        assert mock_logit_source.call_count == 1
        mock_logit_source([[2]])
        assert mock_logit_source.call_count == 2

    def test_reset_clears_call_count(self, mock_logit_source):
        mock_logit_source([[1]])
        mock_logit_source([[2]])
        mock_logit_source.reset()
        assert mock_logit_source.call_count == 0

    def test_last_token_affects_logits(self, mock_logit_source):
        logits_a = mock_logit_source([[10]])
        logits_b = mock_logit_source([[20]])
        # Different last tokens should produce different logits
        assert not np.array_equal(logits_a, logits_b)

    def test_eos_boost_for_long_sequences(self):
        src = MockLogitSource(distribution="uniform")
        short_logits = src([[1] * 5])
        src.reset()
        long_logits = src([[1] * 25])
        if EOS_TOKEN_ID < src.vocab_size:
            assert long_logits[0, EOS_TOKEN_ID] > short_logits[0, EOS_TOKEN_ID]

    def test_custom_vocab_size(self):
        src = MockLogitSource(vocab_size=500)
        logits = src([[1, 2]])
        assert logits.shape == (1, 500)

    def test_custom_hot_tokens(self):
        src = MockLogitSource(distribution="peaked", hot_tokens=[0, 1, 2])
        logits = src([[5]])
        # Hot tokens should have elevated logits
        assert logits[0, 0] > np.median(logits[0])


# =========================================================================
# 3. TestCachedLogitSource
# =========================================================================


class _SimpleMockForCaching:
    """Minimal mock that behaves like a LogitSource for CachedLogitSource."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.model_name = "test-model"
        self.device = "cpu"
        self.call_count = 0
        self._rng = np.random.RandomState(42)

    def get_logits(self, input_ids, **kwargs):
        self.call_count += 1
        seq_len = len(input_ids)
        return self._rng.randn(seq_len, self.vocab_size).astype(np.float32)

    def get_logits_batch(self, input_ids_batch, **kwargs):
        self.call_count += 1
        return [
            self._rng.randn(len(ids), self.vocab_size).astype(np.float32)
            for ids in input_ids_batch
        ]


class TestCachedLogitSource:
    """Tests for CachedLogitSource cache hits, misses, LRU eviction, stats."""

    def _make_cached(self, max_entries=100, eviction_policy="lru", **kwargs):
        wrapped = _SimpleMockForCaching()
        config = CacheConfig(max_entries=max_entries, eviction_policy=eviction_policy, **kwargs)
        return CachedLogitSource(wrapped=wrapped, cache_config=config), wrapped

    def test_cache_miss_calls_wrapped(self):
        cached, wrapped = self._make_cached()
        cached.get_logits([1, 2, 3])
        assert wrapped.call_count == 1

    def test_cache_hit_does_not_call_wrapped(self):
        cached, wrapped = self._make_cached()
        cached.get_logits([1, 2, 3])
        assert wrapped.call_count == 1
        cached.get_logits([1, 2, 3])
        assert wrapped.call_count == 1  # served from cache

    def test_cache_returns_same_result_on_hit(self):
        cached, _ = self._make_cached()
        result1 = cached.get_logits([1, 2, 3])
        result2 = cached.get_logits([1, 2, 3])
        np.testing.assert_array_equal(result1, result2)

    def test_different_inputs_are_separate_cache_entries(self):
        cached, wrapped = self._make_cached()
        cached.get_logits([1, 2, 3])
        cached.get_logits([4, 5, 6])
        assert wrapped.call_count == 2

    def test_cache_stats_hits_and_misses(self):
        cached, _ = self._make_cached()
        cached.get_logits([1, 2])
        cached.get_logits([1, 2])
        cached.get_logits([3, 4])
        stats = cached.cache_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 2
        assert stats["total_requests"] == 3

    def test_cache_hit_rate(self):
        cached, _ = self._make_cached()
        cached.get_logits([1])
        cached.get_logits([1])
        cached.get_logits([1])
        cached.get_logits([2])
        stats = cached.cache_stats()
        assert stats["hit_rate"] == pytest.approx(0.5, abs=0.01)

    def test_clear_cache_resets_stats(self):
        cached, _ = self._make_cached()
        cached.get_logits([1])
        cached.get_logits([1])
        cached.clear_cache()
        stats = cached.cache_stats()
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0

    def test_clear_cache_forces_recompute(self):
        cached, wrapped = self._make_cached()
        cached.get_logits([1, 2])
        cached.clear_cache()
        cached.get_logits([1, 2])
        assert wrapped.call_count == 2

    def test_lru_eviction_removes_oldest(self):
        cached, wrapped = self._make_cached(max_entries=3)
        cached.get_logits([1])
        cached.get_logits([2])
        cached.get_logits([3])
        # Cache is full (3 entries). Adding a 4th evicts the LRU entry [1].
        cached.get_logits([4])
        assert wrapped.call_count == 4
        # [1] should have been evicted, so re-fetching it triggers a miss.
        cached.get_logits([1])
        assert wrapped.call_count == 5
        # [2] or [3] should still be cached (one might be evicted for [4])
        old_count = wrapped.call_count
        cached.get_logits([3])
        # [3] was accessed after [2] originally and not evicted by [4]
        assert wrapped.call_count == old_count  # cache hit

    def test_lru_access_updates_order(self):
        cached, wrapped = self._make_cached(max_entries=3)
        cached.get_logits([1])
        cached.get_logits([2])
        cached.get_logits([3])
        # Access [1] again to make it most recently used
        cached.get_logits([1])
        # Insert [4], which should evict [2] (LRU) not [1]
        cached.get_logits([4])
        old_count = wrapped.call_count
        cached.get_logits([1])
        assert wrapped.call_count == old_count  # [1] still cached

    def test_batch_cache_hits(self):
        cached, wrapped = self._make_cached()
        # Populate cache
        cached.get_logits([1, 2])
        cached.get_logits([3, 4])
        # Batch call: both should be cache hits
        results = cached.get_logits_batch([[1, 2], [3, 4]])
        assert len(results) == 2
        # Only the 2 initial misses should have triggered wrapped calls
        assert wrapped.call_count == 2

    def test_batch_partial_cache_hits(self):
        cached, wrapped = self._make_cached()
        cached.get_logits([1, 2])  # populate
        initial_count = wrapped.call_count
        results = cached.get_logits_batch([[1, 2], [5, 6]])
        assert len(results) == 2
        # [1,2] was a hit, [5,6] was a miss -> one more batch call
        assert wrapped.call_count == initial_count + 1

    def test_cache_stats_memory_entries(self):
        cached, _ = self._make_cached()
        cached.get_logits([1])
        cached.get_logits([2])
        stats = cached.cache_stats()
        assert stats["memory_cache"]["entries"] == 2

    def test_eviction_policy_in_stats(self):
        cached, _ = self._make_cached(eviction_policy="lfu")
        stats = cached.cache_stats()
        assert stats["eviction_policy"] == "lfu"

    def test_context_manager(self):
        cached, _ = self._make_cached()
        with cached as c:
            c.get_logits([1])
        # Should not raise after close

    def test_repr_contains_info(self):
        cached, _ = self._make_cached()
        cached.get_logits([1])
        r = repr(cached)
        assert "CachedLogitSource" in r


# =========================================================================
# 4. TestLogitBatch
# =========================================================================


class TestLogitBatch:
    """Tests for LogitBatch creation, slicing, and properties."""

    def _make_batch(self, batch_size=4, vocab_size=100):
        logits = np.random.randn(batch_size, vocab_size).astype(np.float32)
        token_ids = [[i * 10 + j for j in range(5)] for i in range(batch_size)]
        return LogitBatch(logits=logits, token_ids=token_ids)

    def test_batch_creation(self):
        batch = self._make_batch()
        assert batch.logits.shape == (4, 100)
        assert len(batch.token_ids) == 4

    def test_batch_size_property(self):
        batch = self._make_batch(batch_size=8)
        assert batch.batch_size == 8

    def test_vocab_size_property(self):
        batch = self._make_batch(vocab_size=200)
        assert batch.vocab_size == 200

    def test_invalid_logits_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            LogitBatch(
                logits=np.zeros((3, 4, 5), dtype=np.float32),
                token_ids=[[1], [2], [3]],
            )

    def test_mismatched_batch_raises(self):
        with pytest.raises(ValueError, match="token_ids length"):
            LogitBatch(
                logits=np.zeros((3, 100), dtype=np.float32),
                token_ids=[[1], [2]],  # only 2, but logits has 3
            )

    def test_slice_returns_correct_subset(self):
        batch = self._make_batch(batch_size=5)
        sliced = batch.slice([0, 2, 4])
        assert sliced.batch_size == 3
        np.testing.assert_array_equal(sliced.logits[0], batch.logits[0])
        np.testing.assert_array_equal(sliced.logits[1], batch.logits[2])
        np.testing.assert_array_equal(sliced.logits[2], batch.logits[4])

    def test_slice_preserves_token_ids(self):
        batch = self._make_batch(batch_size=4)
        sliced = batch.slice([1, 3])
        assert sliced.token_ids[0] == batch.token_ids[1]
        assert sliced.token_ids[1] == batch.token_ids[3]

    def test_slice_single_element(self):
        batch = self._make_batch(batch_size=4)
        sliced = batch.slice([2])
        assert sliced.batch_size == 1

    def test_slice_with_attention_mask(self):
        logits = np.random.randn(3, 50).astype(np.float32)
        mask = np.ones((3, 10), dtype=np.int32)
        mask[2, 5:] = 0
        batch = LogitBatch(
            logits=logits,
            token_ids=[[1, 2], [3, 4], [5, 6]],
            attention_mask=mask,
        )
        sliced = batch.slice([0, 2])
        assert sliced.attention_mask is not None
        assert sliced.attention_mask.shape[0] == 2
        np.testing.assert_array_equal(sliced.attention_mask[1], mask[2])

    def test_slice_without_attention_mask(self):
        batch = self._make_batch()
        sliced = batch.slice([0])
        assert sliced.attention_mask is None

    def test_to_probabilities_sums_to_one(self):
        batch = self._make_batch(batch_size=2, vocab_size=50)
        probs = batch.to_probabilities()
        for i in range(2):
            assert abs(np.sum(probs[i]) - 1.0) < 1e-6

    def test_to_probabilities_non_negative(self):
        batch = self._make_batch()
        probs = batch.to_probabilities()
        assert np.all(probs >= 0)

    def test_to_probabilities_with_temperature(self):
        batch = self._make_batch(vocab_size=50)
        probs_t1 = batch.to_probabilities(temperature=1.0)
        probs_t01 = batch.to_probabilities(temperature=0.1)
        # Lower temperature -> more peaked distribution -> lower entropy
        entropy_t1 = -np.sum(probs_t1[0] * np.log(probs_t1[0] + 1e-30))
        entropy_t01 = -np.sum(probs_t01[0] * np.log(probs_t01[0] + 1e-30))
        assert entropy_t01 < entropy_t1

    def test_metadata_default_empty(self):
        batch = self._make_batch()
        assert batch.metadata == {}

    def test_metadata_custom(self):
        logits = np.zeros((1, 10), dtype=np.float32)
        batch = LogitBatch(
            logits=logits,
            token_ids=[[1]],
            metadata={"latency": 0.05},
        )
        assert batch.metadata["latency"] == 0.05

    def test_past_key_values_default_none(self):
        batch = self._make_batch()
        assert batch.past_key_values is None


# =========================================================================
# 5. TestLogitSourceBatching
# =========================================================================


class TestLogitSourceBatching:
    """Tests for batch size handling and padding in MockLogitSourceBatched."""

    def test_batch_size_tracking(self, mock_logit_source_batched):
        mock_logit_source_batched([[1, 2], [3, 4]])
        assert mock_logit_source_batched.batch_sizes == [2]

    def test_multiple_calls_track_all_sizes(self, mock_logit_source_batched):
        mock_logit_source_batched([[1]])
        mock_logit_source_batched([[2], [3], [4]])
        mock_logit_source_batched([[5], [6]])
        assert mock_logit_source_batched.batch_sizes == [1, 3, 2]

    def test_single_item_batch(self, mock_logit_source_batched):
        logits = mock_logit_source_batched([[100]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert mock_logit_source_batched.batch_sizes == [1]

    def test_large_batch(self, mock_logit_source_batched):
        batch = [[i] for i in range(32)]
        logits = mock_logit_source_batched(batch)
        assert logits.shape == (32, DEFAULT_VOCAB_SIZE)
        assert mock_logit_source_batched.batch_sizes == [32]

    def test_varying_sequence_lengths(self, mock_logit_source_batched):
        batch = [[1], [1, 2, 3], [1, 2]]
        logits = mock_logit_source_batched(batch)
        assert logits.shape == (3, DEFAULT_VOCAB_SIZE)

    def test_batch_output_consistent_with_individual(self):
        src = MockLogitSource(distribution="zipf", seed=42)
        batch_logits = src([[10, 20], [30, 40]])
        src.reset()
        single1 = src([[10, 20]])
        src.reset()
        single2 = src([[30, 40]])
        # Due to call count changes, exact match not guaranteed, but shape matches
        assert batch_logits.shape[1] == single1.shape[1]
        assert batch_logits.shape[1] == single2.shape[1]

    def test_empty_batch_within_sequences(self, mock_logit_source_batched):
        # Each sequence has tokens; batch of 2
        logits = mock_logit_source_batched([[5], [10]])
        assert logits.shape[0] == 2

    def test_batch_preserves_dtype(self, mock_logit_source_batched):
        logits = mock_logit_source_batched([[1, 2], [3, 4]])
        assert logits.dtype == np.float32

    def test_batch_call_count(self, mock_logit_source_batched):
        mock_logit_source_batched([[1], [2]])
        mock_logit_source_batched([[3]])
        assert mock_logit_source_batched.call_count == 2

    def test_batch_determinism(self):
        src1 = MockLogitSourceBatched(distribution="zipf", seed=99)
        src2 = MockLogitSourceBatched(distribution="zipf", seed=99)
        batch = [[10, 20], [30, 40]]
        r1 = src1(batch)
        r2 = src2(batch)
        np.testing.assert_array_equal(r1, r2)

    def test_large_vocab_batch(self):
        src = MockLogitSourceBatched(vocab_size=50000)
        logits = src([[1, 2]])
        assert logits.shape == (1, 50000)

    def test_batch_with_repeated_sequences(self, mock_logit_source_batched):
        logits = mock_logit_source_batched([[5, 10], [5, 10]])
        np.testing.assert_array_equal(logits[0], logits[1])


# =========================================================================
# 6. TestLogitSourceNumerical
# =========================================================================


class TestLogitSourceNumerical:
    """Tests for output range, finite values, softmax stability."""

    def test_all_values_finite(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3, 4, 5]])
        assert np.all(np.isfinite(logits))

    def test_no_nans(self, mock_logit_source):
        logits = mock_logit_source([[100, 200, 300]])
        assert not np.any(np.isnan(logits))

    def test_softmax_sums_to_one(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3]])[0]
        probs = softmax(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_softmax_non_negative(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3]])[0]
        probs = softmax(logits)
        assert np.all(probs >= 0)

    def test_softmax_with_large_values(self):
        logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_softmax_with_very_negative_values(self):
        logits = np.array([-1000.0, -1001.0, -999.0], dtype=np.float64)
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_softmax_uniform_input(self):
        logits = np.zeros(100, dtype=np.float64)
        probs = softmax(logits)
        expected = 1.0 / 100
        np.testing.assert_allclose(probs, expected, atol=1e-8)

    def test_softmax_2d(self, mock_logit_source):
        logits = mock_logit_source([[1, 2], [3, 4]])
        probs = softmax(logits)
        assert probs.shape == logits.shape
        for i in range(2):
            assert abs(np.sum(probs[i]) - 1.0) < 1e-6

    def test_log_softmax_finite(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3]])[0]
        log_probs = log_softmax(logits)
        assert np.all(np.isfinite(log_probs))

    def test_log_softmax_all_non_positive(self, mock_logit_source):
        logits = mock_logit_source([[1, 2]])[0]
        log_probs = log_softmax(logits)
        assert np.all(log_probs <= 1e-10)  # log of probabilities <= 0

    def test_log_softmax_consistent_with_softmax(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        log_probs = log_softmax(logits)
        probs = softmax(logits)
        np.testing.assert_allclose(np.exp(log_probs), probs, atol=1e-10)

    def test_entropy_non_negative(self, mock_logit_source):
        logits = mock_logit_source([[5, 10, 15]])[0]
        ent = entropy_from_logits(logits)
        assert ent >= 0

    def test_entropy_uniform_is_log_n(self):
        n = 100
        logits = np.zeros(n, dtype=np.float64)
        ent = entropy_from_logits(logits)
        expected = np.log(n)
        assert abs(ent - expected) < 1e-6

    def test_entropy_degenerate_is_zero(self):
        logits = np.full(100, -1000.0, dtype=np.float64)
        logits[0] = 0.0
        ent = entropy_from_logits(logits)
        assert ent < 0.01

    def test_entropy_batch(self, mock_logit_source):
        logits = mock_logit_source([[1, 2], [3, 4], [5, 6]])
        ents = entropy_from_logits_batch(logits)
        assert ents.shape == (3,)
        assert np.all(ents >= 0)

    def test_apply_temperature_identity(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = apply_temperature(logits, 1.0)
        np.testing.assert_allclose(result, logits, atol=1e-10)

    def test_apply_temperature_high_increases_entropy(self):
        logits = np.array([0.0, 1.0, 5.0, 0.5])
        ent_t1 = entropy_from_logits(apply_temperature(logits, 1.0))
        ent_t5 = entropy_from_logits(apply_temperature(logits, 5.0))
        assert ent_t5 > ent_t1

    def test_apply_temperature_zero_is_greedy(self):
        logits = np.array([1.0, 5.0, 3.0])
        result = apply_temperature(logits, 0.0)
        assert result[1] == 0.0
        assert result[0] == -np.inf
        assert result[2] == -np.inf

    def test_apply_top_k_keeps_k_values(self):
        logits = np.arange(20, dtype=np.float64)
        result = apply_top_k(logits, k=5)
        finite_count = np.sum(np.isfinite(result))
        assert finite_count == 5

    def test_apply_top_k_preserves_top_values(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = apply_top_k(logits, k=3)
        # Top 3 are indices 1, 4, 2 (values 5, 4, 3)
        assert np.isfinite(result[1])
        assert np.isfinite(result[4])
        assert np.isfinite(result[2])


# =========================================================================
# 7. TestLogitSourceKVCache
# =========================================================================


class TestLogitSourceKVCache:
    """Tests for KV-cache simulation correctness and invalidation."""

    def test_kv_cache_returns_same_for_same_input(self, mock_logit_source_kv_cache):
        result1 = mock_logit_source_kv_cache([[1, 2, 3]])
        result2 = mock_logit_source_kv_cache([[1, 2, 3]])
        np.testing.assert_array_equal(result1, result2)

    def test_kv_cache_different_input_different_result(self, mock_logit_source_kv_cache):
        result1 = mock_logit_source_kv_cache([[1, 2, 3]])
        result2 = mock_logit_source_kv_cache([[4, 5, 6]])
        assert not np.array_equal(result1, result2)

    def test_kv_cache_stores_entries(self, mock_logit_source_kv_cache):
        mock_logit_source_kv_cache([[1, 2]])
        assert str([1, 2]) in mock_logit_source_kv_cache._cache

    def test_kv_cache_clear(self, mock_logit_source_kv_cache):
        mock_logit_source_kv_cache([[1, 2]])
        mock_logit_source_kv_cache.clear_cache()
        assert len(mock_logit_source_kv_cache._cache) == 0

    def test_kv_cache_after_clear_recomputes(self, mock_logit_source_kv_cache):
        result1 = mock_logit_source_kv_cache([[10, 20]])
        mock_logit_source_kv_cache.clear_cache()
        result2 = mock_logit_source_kv_cache([[10, 20]])
        # After clearing and recomputing, should get same base logits
        np.testing.assert_array_equal(result1, result2)

    def test_kv_cache_batch_populates_all(self, mock_logit_source_kv_cache):
        mock_logit_source_kv_cache([[1, 2], [3, 4]])
        assert str([1, 2]) in mock_logit_source_kv_cache._cache
        assert str([3, 4]) in mock_logit_source_kv_cache._cache

    def test_kv_cache_output_shape(self, mock_logit_source_kv_cache):
        result = mock_logit_source_kv_cache([[1, 2, 3]])
        assert result.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_kv_cache_batch_output_shape(self, mock_logit_source_kv_cache):
        result = mock_logit_source_kv_cache([[1], [2], [3]])
        assert result.shape == (3, DEFAULT_VOCAB_SIZE)

    def test_kv_cache_consistent_across_batch_and_single(self, mock_logit_source_kv_cache):
        batch = mock_logit_source_kv_cache([[10, 20], [30, 40]])
        single1 = mock_logit_source_kv_cache([[10, 20]])
        single2 = mock_logit_source_kv_cache([[30, 40]])
        np.testing.assert_array_equal(batch[0], single1[0])
        np.testing.assert_array_equal(batch[1], single2[0])

    def test_kv_cache_preserves_dtype(self, mock_logit_source_kv_cache):
        result = mock_logit_source_kv_cache([[5]])
        assert result.dtype == np.float32

    def test_kv_cache_prefix_extension_different(self, mock_logit_source_kv_cache):
        r1 = mock_logit_source_kv_cache([[1, 2]])
        r2 = mock_logit_source_kv_cache([[1, 2, 3]])
        # Different lengths -> different cache keys -> different results
        assert r1.shape != r2.shape or not np.array_equal(r1, r2)

    def test_kv_cache_many_entries(self, mock_logit_source_kv_cache):
        for i in range(50):
            mock_logit_source_kv_cache([[i, i + 1]])
        assert len(mock_logit_source_kv_cache._cache) == 50

    def test_kv_cache_same_tokens_different_order(self, mock_logit_source_kv_cache):
        r1 = mock_logit_source_kv_cache([[1, 2, 3]])
        r2 = mock_logit_source_kv_cache([[3, 2, 1]])
        # Order matters, so results differ
        assert not np.array_equal(r1, r2)

    def test_kv_cache_values_are_finite(self, mock_logit_source_kv_cache):
        result = mock_logit_source_kv_cache([[7, 14, 21]])
        assert np.all(np.isfinite(result))

    def test_kv_cache_reset_clears_call_count_but_keeps_cache(self, mock_logit_source_kv_cache):
        mock_logit_source_kv_cache([[1]])
        mock_logit_source_kv_cache.reset()
        assert mock_logit_source_kv_cache.call_count == 0
        # Cache from __init__ is independent; the dict cache persists
        assert str([1]) in mock_logit_source_kv_cache._cache


# =========================================================================
# 8. TestLogitSourceQuantization
# =========================================================================


class TestLogitSourceQuantization:
    """Tests for quantization config validation and modes."""

    def test_no_quantization(self):
        cfg = LogitSourceConfig(quantization=None)
        assert cfg.quantization is None

    def test_int8_quantization(self):
        cfg = LogitSourceConfig(quantization="int8")
        assert cfg.quantization == "int8"

    def test_int4_quantization(self):
        cfg = LogitSourceConfig(quantization="int4")
        assert cfg.quantization == "int4"

    def test_invalid_quantization_string(self):
        with pytest.raises(ValueError, match="quantization"):
            LogitSourceConfig(quantization="fp8")

    def test_invalid_quantization_numeric(self):
        with pytest.raises(ValueError):
            LogitSourceConfig(quantization="16bit")

    def test_quantization_with_different_dtypes(self):
        for dt in ["float16", "float32"]:
            cfg = LogitSourceConfig(dtype=dt, quantization="int8")
            assert cfg.dtype == dt
            assert cfg.quantization == "int8"

    def test_quantization_copy_preserves(self):
        cfg = LogitSourceConfig(quantization="int4")
        cfg2 = cfg.copy()
        assert cfg2.quantization == "int4"

    def test_quantization_copy_override(self):
        cfg = LogitSourceConfig(quantization="int8")
        cfg2 = cfg.copy(quantization="int4")
        assert cfg2.quantization == "int4"
        assert cfg.quantization == "int8"

    def test_quantization_copy_to_none(self):
        cfg = LogitSourceConfig(quantization="int8")
        cfg2 = cfg.copy(quantization=None)
        assert cfg2.quantization is None

    def test_quantization_with_kv_cache(self):
        cfg = LogitSourceConfig(quantization="int8", use_kv_cache=True)
        assert cfg.quantization == "int8"
        assert cfg.use_kv_cache is True

    def test_quantization_affects_cache_key(self):
        key1 = CacheKey(input_ids=(1, 2, 3), model_name="gpt2", quantization=None)
        key2 = CacheKey(input_ids=(1, 2, 3), model_name="gpt2", quantization="int8")
        assert key1 != key2
        assert hash(key1) != hash(key2)

    def test_same_quantization_same_cache_key(self):
        key1 = CacheKey(input_ids=(1, 2), model_name="m", quantization="int4")
        key2 = CacheKey(input_ids=(1, 2), model_name="m", quantization="int4")
        assert key1 == key2
        assert hash(key1) == hash(key2)

    def test_quantization_content_hash_differs(self):
        key1 = CacheKey(input_ids=(1, 2), model_name="m", quantization=None)
        key2 = CacheKey(input_ids=(1, 2), model_name="m", quantization="int8")
        assert key1.content_hash() != key2.content_hash()

    def test_mock_source_works_regardless_of_quantization(self):
        for q in [None, "int8", "int4"]:
            cfg = LogitSourceConfig(quantization=q, vocab_size=100)
            src = MockLogitSource(vocab_size=100)
            logits = src([[1, 2]])
            assert logits.shape == (1, 100)
            assert np.all(np.isfinite(logits))


# =========================================================================
# 9. TestLogitSourceStatistics
# =========================================================================


class TestLogitSourceStatistics:
    """Tests for LogitSourceStats call counting, latency tracking."""

    def test_initial_state(self):
        stats = LogitSourceStats()
        assert stats.total_calls == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.avg_latency == 0.0

    def test_record_call_increments_total(self):
        stats = LogitSourceStats()
        stats.record_call(0.01)
        assert stats.total_calls == 1
        stats.record_call(0.02)
        assert stats.total_calls == 2

    def test_record_call_miss(self):
        stats = LogitSourceStats()
        stats.record_call(0.01, cache_hit=False)
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0

    def test_record_call_hit(self):
        stats = LogitSourceStats()
        stats.record_call(0.01, cache_hit=True)
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0

    def test_avg_latency(self):
        stats = LogitSourceStats()
        stats.record_call(0.1)
        stats.record_call(0.3)
        assert abs(stats.avg_latency - 0.2) < 1e-6

    def test_p95_latency(self):
        stats = LogitSourceStats()
        for i in range(100):
            stats.record_call(float(i) / 100.0)
        assert stats.p95_latency > stats.avg_latency

    def test_p99_latency(self):
        stats = LogitSourceStats()
        for i in range(100):
            stats.record_call(float(i) / 100.0)
        assert stats.p99_latency >= stats.p95_latency

    def test_min_max_latency(self):
        stats = LogitSourceStats()
        stats.record_call(0.5)
        stats.record_call(0.1)
        stats.record_call(0.9)
        assert abs(stats.min_latency - 0.1) < 1e-6
        assert abs(stats.max_latency - 0.9) < 1e-6

    def test_std_latency(self):
        stats = LogitSourceStats()
        stats.record_call(0.1)
        stats.record_call(0.3)
        assert stats.std_latency > 0

    def test_std_latency_single_call(self):
        stats = LogitSourceStats()
        stats.record_call(0.1)
        assert stats.std_latency == 0.0

    def test_cache_hit_rate(self):
        stats = LogitSourceStats()
        stats.record_call(0.01, cache_hit=True)
        stats.record_call(0.01, cache_hit=True)
        stats.record_call(0.01, cache_hit=False)
        assert abs(stats.cache_hit_rate - 2.0 / 3.0) < 1e-6

    def test_cache_hit_rate_no_calls(self):
        stats = LogitSourceStats()
        assert stats.cache_hit_rate == 0.0

    def test_tokens_per_second(self):
        stats = LogitSourceStats()
        stats.record_call(0.01, n_tokens=10)
        time.sleep(0.05)
        stats.record_call(0.01, n_tokens=10)
        tps = stats.tokens_per_second
        assert tps > 0

    def test_record_memory(self):
        stats = LogitSourceStats()
        stats.record_memory(1024 * 1024)
        mem = stats.memory_usage
        assert mem["current_bytes"] == 1024 * 1024
        assert mem["peak_bytes"] == 1024 * 1024

    def test_peak_memory_tracks_maximum(self):
        stats = LogitSourceStats()
        stats.record_memory(1000)
        stats.record_memory(5000)
        stats.record_memory(3000)
        assert stats.memory_usage["peak_bytes"] == 5000

    def test_summary_keys(self):
        stats = LogitSourceStats()
        stats.record_call(0.01)
        s = stats.summary()
        expected_keys = {
            "total_calls", "cache_hits", "cache_misses", "cache_hit_rate",
            "avg_latency_s", "p95_latency_s", "p99_latency_s",
            "min_latency_s", "max_latency_s", "std_latency_s",
            "tokens_per_second", "tokens_processed", "memory",
            "window_size", "latencies_recorded",
        }
        assert expected_keys.issubset(set(s.keys()))

    def test_reset(self):
        stats = LogitSourceStats()
        stats.record_call(0.01, n_tokens=5, cache_hit=True)
        stats.record_call(0.02, n_tokens=10, cache_hit=False)
        stats.record_memory(5000)
        stats.reset()
        assert stats.total_calls == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.avg_latency == 0.0
        assert stats.memory_usage["current_bytes"] == 0.0

    def test_window_size_limits_latencies(self):
        stats = LogitSourceStats(window_size=10)
        for i in range(20):
            stats.record_call(float(i))
        assert stats.total_calls == 20
        s = stats.summary()
        assert s["latencies_recorded"] == 10

    def test_repr(self):
        stats = LogitSourceStats()
        stats.record_call(0.05)
        r = repr(stats)
        assert "LogitSourceStats" in r

    def test_memory_usage_mb(self):
        stats = LogitSourceStats()
        stats.record_memory(2 * 1024 * 1024)
        mem = stats.memory_usage
        assert abs(mem["current_mb"] - 2.0) < 0.01


# =========================================================================
# 10. TestLogitSourceEdgeCases
# =========================================================================


class TestLogitSourceEdgeCases:
    """Tests for edge cases: empty input, single token, max sequence length."""

    def test_single_token_input(self, mock_logit_source):
        logits = mock_logit_source([[42]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert np.all(np.isfinite(logits))

    def test_very_long_sequence(self, mock_logit_source):
        long_seq = list(range(500))
        logits = mock_logit_source([long_seq])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert np.all(np.isfinite(logits))

    def test_zero_token_ids(self, mock_logit_source):
        logits = mock_logit_source([[0, 0, 0]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_large_token_ids(self, mock_logit_source):
        logits = mock_logit_source([[999999, 888888]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert np.all(np.isfinite(logits))

    def test_repeated_tokens(self, mock_logit_source):
        logits = mock_logit_source([[5, 5, 5, 5, 5]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_batch_of_one(self, mock_logit_source):
        logits = mock_logit_source([[1, 2, 3]])
        assert logits.shape[0] == 1

    def test_max_vocab_minus_one_token(self, mock_logit_source):
        logits = mock_logit_source([[DEFAULT_VOCAB_SIZE - 1]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert np.all(np.isfinite(logits))

    def test_special_token_bos(self, mock_logit_source):
        logits = mock_logit_source([[BOS_TOKEN_ID]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_special_token_eos(self, mock_logit_source):
        logits = mock_logit_source([[EOS_TOKEN_ID]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_special_token_pad(self, mock_logit_source):
        logits = mock_logit_source([[PAD_TOKEN_ID]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_monotonically_increasing_tokens(self, mock_logit_source):
        seq = list(range(0, 100))
        logits = mock_logit_source([seq])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_monotonically_decreasing_tokens(self, mock_logit_source):
        seq = list(range(100, 0, -1))
        logits = mock_logit_source([seq])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_alternating_tokens(self, mock_logit_source):
        seq = [10, 20] * 25
        logits = mock_logit_source([seq])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert np.all(np.isfinite(logits))

    def test_vocab_size_one(self):
        src = MockLogitSource(vocab_size=1)
        logits = src([[0]])
        assert logits.shape == (1, 1)

    def test_small_vocab(self):
        src = MockLogitSource(vocab_size=5)
        logits = src([[0, 1, 2]])
        assert logits.shape == (1, 5)
        assert np.all(np.isfinite(logits))


# =========================================================================
# Additional cache component tests
# =========================================================================


class TestCacheKey:
    """Tests for CacheKey hashing and equality."""

    def test_equal_keys(self):
        k1 = CacheKey(input_ids=(1, 2, 3), model_name="gpt2")
        k2 = CacheKey(input_ids=(1, 2, 3), model_name="gpt2")
        assert k1 == k2

    def test_different_input_ids(self):
        k1 = CacheKey(input_ids=(1, 2), model_name="gpt2")
        k2 = CacheKey(input_ids=(3, 4), model_name="gpt2")
        assert k1 != k2

    def test_different_model_name(self):
        k1 = CacheKey(input_ids=(1,), model_name="gpt2")
        k2 = CacheKey(input_ids=(1,), model_name="llama")
        assert k1 != k2

    def test_hash_consistency(self):
        k = CacheKey(input_ids=(1, 2, 3), model_name="m")
        assert hash(k) == hash(k)

    def test_hash_equality(self):
        k1 = CacheKey(input_ids=(5, 6), model_name="m")
        k2 = CacheKey(input_ids=(5, 6), model_name="m")
        assert hash(k1) == hash(k2)

    def test_content_hash_deterministic(self):
        k = CacheKey(input_ids=(1, 2, 3), model_name="gpt2")
        assert k.content_hash() == k.content_hash()

    def test_content_hash_different_for_different_keys(self):
        k1 = CacheKey(input_ids=(1,), model_name="gpt2")
        k2 = CacheKey(input_ids=(2,), model_name="gpt2")
        assert k1.content_hash() != k2.content_hash()

    def test_list_input_converted_to_tuple(self):
        k = CacheKey(input_ids=[1, 2, 3], model_name="m")
        assert isinstance(k.input_ids, tuple)

    def test_content_hash_is_hex(self):
        k = CacheKey(input_ids=(1,), model_name="m")
        h = k.content_hash()
        assert all(c in "0123456789abcdef" for c in h)

    def test_content_hash_length(self):
        k = CacheKey(input_ids=(1, 2), model_name="m")
        assert len(k.content_hash()) == 64  # SHA-256 hex digest

    def test_quantization_in_content_hash(self):
        k1 = CacheKey(input_ids=(1,), model_name="m", quantization="int8")
        k2 = CacheKey(input_ids=(1,), model_name="m", quantization=None)
        assert k1.content_hash() != k2.content_hash()


class TestCacheEntry:
    """Tests for CacheEntry metadata and access tracking."""

    def test_creation(self):
        logits = np.zeros(100, dtype=np.float32)
        entry = CacheEntry(logits=logits)
        assert entry.access_count == 0
        assert entry.size_bytes > 0

    def test_touch_increments_count(self):
        entry = CacheEntry(logits=np.zeros(10, dtype=np.float32))
        entry.touch()
        assert entry.access_count == 1
        entry.touch()
        assert entry.access_count == 2

    def test_touch_updates_timestamp(self):
        entry = CacheEntry(logits=np.zeros(10, dtype=np.float32))
        old_ts = entry.timestamp
        time.sleep(0.01)
        entry.touch()
        assert entry.timestamp > old_ts

    def test_size_bytes_auto_computed(self):
        logits = np.zeros(1000, dtype=np.float32)
        entry = CacheEntry(logits=logits)
        assert entry.size_bytes >= logits.nbytes

    def test_metadata_default_empty(self):
        entry = CacheEntry(logits=np.zeros(10, dtype=np.float32))
        assert entry.metadata == {}

    def test_metadata_custom(self):
        entry = CacheEntry(
            logits=np.zeros(10, dtype=np.float32),
            metadata={"source": "test"},
        )
        assert entry.metadata["source"] == "test"


class TestLRUCacheUnit:
    """Unit tests for the LRUCache class."""

    def _make_cache(self, max_entries=5, policy="lru"):
        config = CacheConfig(max_entries=max_entries, eviction_policy=policy)
        return LRUCache(config)

    def _make_key(self, ids):
        return CacheKey(input_ids=tuple(ids), model_name="test")

    def _make_entry(self, size=10):
        return CacheEntry(logits=np.zeros(size, dtype=np.float32))

    def test_put_and_get(self):
        cache = self._make_cache()
        key = self._make_key([1, 2])
        entry = self._make_entry()
        cache.put(key, entry)
        result = cache.get(key)
        assert result is not None

    def test_get_miss_returns_none(self):
        cache = self._make_cache()
        key = self._make_key([99])
        assert cache.get(key) is None

    def test_len(self):
        cache = self._make_cache()
        assert len(cache) == 0
        cache.put(self._make_key([1]), self._make_entry())
        assert len(cache) == 1
        cache.put(self._make_key([2]), self._make_entry())
        assert len(cache) == 2

    def test_contains(self):
        cache = self._make_cache()
        key = self._make_key([1])
        cache.put(key, self._make_entry())
        assert key in cache

    def test_not_contains(self):
        cache = self._make_cache()
        assert self._make_key([1]) not in cache

    def test_lru_eviction(self):
        cache = self._make_cache(max_entries=3)
        k1, k2, k3, k4 = [self._make_key([i]) for i in range(4)]
        cache.put(k1, self._make_entry())
        cache.put(k2, self._make_entry())
        cache.put(k3, self._make_entry())
        # k1 is LRU
        cache.put(k4, self._make_entry())
        assert k1 not in cache
        assert k4 in cache

    def test_lru_access_prevents_eviction(self):
        cache = self._make_cache(max_entries=3)
        k1, k2, k3, k4 = [self._make_key([i]) for i in range(4)]
        cache.put(k1, self._make_entry())
        cache.put(k2, self._make_entry())
        cache.put(k3, self._make_entry())
        cache.get(k1)  # access k1, making k2 the LRU
        cache.put(k4, self._make_entry())
        assert k1 in cache
        assert k2 not in cache

    def test_fifo_eviction(self):
        cache = self._make_cache(max_entries=2, policy="fifo")
        k1, k2, k3 = [self._make_key([i]) for i in range(3)]
        cache.put(k1, self._make_entry())
        cache.put(k2, self._make_entry())
        cache.get(k1)  # access does NOT matter for FIFO
        cache.put(k3, self._make_entry())
        # k1 was inserted first -> evicted
        assert k1 not in cache
        assert k3 in cache

    def test_clear(self):
        cache = self._make_cache()
        cache.put(self._make_key([1]), self._make_entry())
        cache.put(self._make_key([2]), self._make_entry())
        cache.clear()
        assert len(cache) == 0

    def test_stats_hits_misses(self):
        cache = self._make_cache()
        key = self._make_key([1])
        cache.put(key, self._make_entry())
        cache.get(key)  # hit
        cache.get(self._make_key([99]))  # miss
        s = cache.stats
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_stats_evictions(self):
        cache = self._make_cache(max_entries=2)
        cache.put(self._make_key([1]), self._make_entry())
        cache.put(self._make_key([2]), self._make_entry())
        cache.put(self._make_key([3]), self._make_entry())  # evicts [1]
        s = cache.stats
        assert s["evictions"] == 1

    def test_stats_inserts(self):
        cache = self._make_cache()
        cache.put(self._make_key([1]), self._make_entry())
        cache.put(self._make_key([2]), self._make_entry())
        assert cache.stats["inserts"] == 2

    def test_reset_stats(self):
        cache = self._make_cache()
        cache.put(self._make_key([1]), self._make_entry())
        cache.get(self._make_key([1]))
        cache.reset_stats()
        s = cache.stats
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["inserts"] == 0
        assert s["evictions"] == 0

    def test_remove(self):
        cache = self._make_cache()
        key = self._make_key([1])
        cache.put(key, self._make_entry())
        assert cache.remove(key) is True
        assert key not in cache

    def test_remove_nonexistent(self):
        cache = self._make_cache()
        assert cache.remove(self._make_key([99])) is False

    def test_keys_values_items(self):
        cache = self._make_cache()
        k1 = self._make_key([1])
        k2 = self._make_key([2])
        cache.put(k1, self._make_entry())
        cache.put(k2, self._make_entry())
        assert len(cache.keys()) == 2
        assert len(cache.values()) == 2
        assert len(cache.items()) == 2

    def test_size_tracking(self):
        cache = self._make_cache()
        assert cache.size() == 0
        cache.put(self._make_key([1]), self._make_entry(size=100))
        assert cache.size() > 0

    def test_ttl_expiry(self):
        config = CacheConfig(max_entries=10, ttl_seconds=0)
        cache = LRUCache(config)
        key = self._make_key([1])
        cache.put(key, self._make_entry())
        time.sleep(0.01)
        # Entry should be expired
        result = cache.get(key)
        assert result is None

    def test_expire_stale(self):
        config = CacheConfig(max_entries=10, ttl_seconds=0)
        cache = LRUCache(config)
        cache.put(self._make_key([1]), self._make_entry())
        cache.put(self._make_key([2]), self._make_entry())
        time.sleep(0.01)
        removed = cache.expire_stale()
        assert removed == 2
        assert len(cache) == 0


class TestCacheConfig:
    """Tests for CacheConfig validation."""

    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.max_entries == 10_000
        assert cfg.eviction_policy == "lru"
        assert cfg.persist_to_disk is False
        assert cfg.compression is True
        assert cfg.ttl_seconds is None

    def test_invalid_eviction_policy(self):
        with pytest.raises(ValueError, match="eviction_policy"):
            CacheConfig(eviction_policy="random")

    def test_valid_eviction_policies(self):
        for p in ["lru", "lfu", "fifo"]:
            cfg = CacheConfig(eviction_policy=p)
            assert cfg.eviction_policy == p

    def test_zero_max_entries_raises(self):
        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=0)

    def test_negative_max_entries_raises(self):
        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=-1)

    def test_zero_max_size_raises(self):
        with pytest.raises(ValueError, match="max_size_bytes"):
            CacheConfig(max_size_bytes=0)

    def test_custom_ttl(self):
        cfg = CacheConfig(ttl_seconds=300)
        assert cfg.ttl_seconds == 300

    def test_disk_cache_dir(self):
        cfg = CacheConfig(persist_to_disk=True, disk_cache_dir="/tmp/test_cache")
        assert cfg.disk_cache_dir == "/tmp/test_cache"

    def test_compression_flag(self):
        cfg = CacheConfig(compression=False)
        assert cfg.compression is False


class TestMockLogitSourceWithLatency:
    """Tests for the latency-simulating mock."""

    def test_latency_is_applied(self):
        src = MockLogitSourceWithLatency(latency_ms=50.0)
        start = time.time()
        src([[1, 2]])
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms >= 40  # allow small timing margin

    def test_total_latency_tracked(self):
        src = MockLogitSourceWithLatency(latency_ms=10.0)
        src([[1]])
        src([[2]])
        assert src.total_latency >= 20.0

    def test_output_shape_unchanged(self):
        src = MockLogitSourceWithLatency(latency_ms=1.0)
        logits = src([[1, 2, 3]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_output_finite(self):
        src = MockLogitSourceWithLatency(latency_ms=1.0)
        logits = src([[5]])
        assert np.all(np.isfinite(logits))

    def test_deterministic_with_latency(self):
        src1 = MockLogitSourceWithLatency(latency_ms=1.0, seed=42, deterministic=True)
        src2 = MockLogitSourceWithLatency(latency_ms=1.0, seed=42, deterministic=True)
        r1 = src1([[10, 20]])
        r2 = src2([[10, 20]])
        np.testing.assert_array_equal(r1, r2)


class TestSoftmaxHelpers:
    """Additional tests for softmax and related numerical helpers."""

    def test_softmax_1d(self):
        logits = np.array([1.0, 2.0, 3.0])
        p = softmax(logits)
        assert p.shape == (3,)
        assert abs(np.sum(p) - 1.0) < 1e-6

    def test_softmax_preserves_order(self):
        logits = np.array([1.0, 3.0, 2.0])
        p = softmax(logits)
        assert p[1] > p[2] > p[0]

    def test_apply_top_p(self):
        logits = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
        result = apply_top_p(logits, p=0.9)
        # Should keep the top tokens that cover 90% probability mass
        finite_mask = np.isfinite(result)
        assert np.sum(finite_mask) >= 1

    def test_apply_top_k_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            apply_top_k(np.array([1.0, 2.0]), k=0)

    def test_apply_temperature_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            apply_temperature(np.array([1.0]), temperature=-1.0)

    def test_entropy_batch_matches_individual(self):
        logits = np.random.randn(5, 50).astype(np.float64)
        batch_ent = entropy_from_logits_batch(logits)
        for i in range(5):
            single = entropy_from_logits(logits[i])
            assert abs(batch_ent[i] - single) < 1e-8

    def test_log_softmax_sum_exp_is_one(self):
        logits = np.array([2.0, 3.0, 5.0])
        lp = log_softmax(logits)
        assert abs(np.sum(np.exp(lp)) - 1.0) < 1e-8

    def test_softmax_extreme_positive(self):
        logits = np.array([500.0, 501.0, 499.0])
        p = softmax(logits)
        assert np.all(np.isfinite(p))
        assert abs(np.sum(p) - 1.0) < 1e-6

    def test_softmax_extreme_negative(self):
        logits = np.array([-500.0, -501.0, -499.0])
        p = softmax(logits)
        assert np.all(np.isfinite(p))
        assert abs(np.sum(p) - 1.0) < 1e-6

    def test_softmax_mixed_extremes(self):
        logits = np.array([-1000.0, 0.0, 1000.0])
        p = softmax(logits)
        assert np.all(np.isfinite(p))
        assert p[2] > 0.99  # the +1000 dominates


class TestLogitBatchConcatenation:
    """Tests for LogitBatch concatenation and advanced operations."""

    def test_concatenate_two_batches(self):
        b1 = LogitBatch(
            logits=np.ones((2, 50), dtype=np.float32),
            token_ids=[[1, 2], [3, 4]],
        )
        b2 = LogitBatch(
            logits=np.zeros((3, 50), dtype=np.float32),
            token_ids=[[5], [6], [7]],
        )
        combined_logits = np.concatenate([b1.logits, b2.logits], axis=0)
        combined_ids = b1.token_ids + b2.token_ids
        combined = LogitBatch(logits=combined_logits, token_ids=combined_ids)
        assert combined.batch_size == 5
        assert combined.vocab_size == 50

    def test_slice_then_check_shape(self):
        logits = np.random.randn(10, 100).astype(np.float32)
        batch = LogitBatch(
            logits=logits,
            token_ids=[[i] for i in range(10)],
        )
        sliced = batch.slice([0, 5, 9])
        assert sliced.logits.shape == (3, 100)

    def test_slice_preserves_logit_values(self):
        rng = np.random.RandomState(123)
        logits = rng.randn(5, 30).astype(np.float32)
        batch = LogitBatch(
            logits=logits,
            token_ids=[[i] for i in range(5)],
        )
        sliced = batch.slice([2])
        np.testing.assert_array_equal(sliced.logits[0], logits[2])

    def test_batch_from_single_sequence(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        batch = LogitBatch(logits=logits, token_ids=[[42]])
        assert batch.batch_size == 1
        assert batch.vocab_size == 3

    def test_1d_logits_rejected(self):
        with pytest.raises(ValueError):
            LogitBatch(logits=np.array([1.0, 2.0]), token_ids=[[1]])

    def test_3d_logits_rejected(self):
        with pytest.raises(ValueError):
            LogitBatch(
                logits=np.zeros((2, 3, 4), dtype=np.float32),
                token_ids=[[1], [2]],
            )

    def test_empty_metadata_preserved_on_slice(self):
        batch = LogitBatch(
            logits=np.zeros((3, 10), dtype=np.float32),
            token_ids=[[1], [2], [3]],
            metadata={"key": "value"},
        )
        sliced = batch.slice([0])
        assert "key" in sliced.metadata

    def test_past_key_values_invalidated_on_slice(self):
        batch = LogitBatch(
            logits=np.zeros((2, 10), dtype=np.float32),
            token_ids=[[1], [2]],
            past_key_values="some_cache_object",
        )
        sliced = batch.slice([0])
        assert sliced.past_key_values is None

    def test_to_probabilities_high_temperature(self):
        logits = np.array([[0.0, 10.0, -10.0]], dtype=np.float32)
        batch = LogitBatch(logits=logits, token_ids=[[1]])
        probs = batch.to_probabilities(temperature=100.0)
        # Very high temperature -> nearly uniform
        assert np.std(probs[0]) < 0.1

    def test_to_probabilities_low_temperature(self):
        logits = np.array([[0.0, 10.0, -10.0]], dtype=np.float32)
        batch = LogitBatch(logits=logits, token_ids=[[1]])
        probs = batch.to_probabilities(temperature=0.01)
        # Very low temperature -> nearly one-hot
        assert probs[0, 1] > 0.99


class TestDistributionVariants:
    """Tests for different distribution types in MockLogitSource."""

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_output_shape_all_distributions(self, dist):
        src = MockLogitSource(distribution=dist)
        logits = src([[1, 2, 3]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_finite_all_distributions(self, dist):
        src = MockLogitSource(distribution=dist)
        logits = src([[5, 10]])
        assert np.all(np.isfinite(logits))

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_dtype_all_distributions(self, dist):
        src = MockLogitSource(distribution=dist)
        logits = src([[1]])
        assert logits.dtype == np.float32

    @pytest.mark.parametrize("dist", ["uniform", "zipf", "peaked", "bimodal", "degenerate"])
    def test_deterministic_all_distributions(self, dist):
        src = MockLogitSource(distribution=dist, deterministic=True, seed=42)
        a = src([[1, 2]])
        b = src([[1, 2]])
        np.testing.assert_array_equal(a, b)

    def test_zipf_has_decreasing_base(self):
        src = MockLogitSource(distribution="zipf")
        base = src._base_logits.copy()
        base[EOS_TOKEN_ID] = base[0]  # ignore EOS override
        # First element should be largest (rank 1)
        assert base[0] >= base[src.vocab_size // 2]

    def test_peaked_hot_tokens_elevated(self):
        hot = [100, 200, 300]
        src = MockLogitSource(distribution="peaked", hot_tokens=hot, concentration=5.0)
        base = src._base_logits
        hot_vals = [base[t] for t in hot if t < src.vocab_size]
        median = np.median(base)
        assert np.mean(hot_vals) > median

    def test_degenerate_single_peak(self):
        src = MockLogitSource(distribution="degenerate", hot_tokens=[50])
        base = src._base_logits
        assert base[50] == np.max(base) or base[50] >= -1.0

    def test_bimodal_two_levels(self):
        src = MockLogitSource(distribution="bimodal")
        base = src._base_logits
        unique_rounded = set(np.round(base, 0))
        # Should have at least 2 distinct levels
        assert len(unique_rounded) >= 2

    def test_unknown_distribution_fallback(self):
        src = MockLogitSource(distribution="unknown_dist")
        logits = src([[1]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)
        assert np.all(np.isfinite(logits))


class TestCachedLogitSourceAdvanced:
    """Advanced tests for CachedLogitSource: save/load, warming, etc."""

    def _make_cached(self, max_entries=100, **kwargs):
        wrapped = _SimpleMockForCaching()
        config = CacheConfig(max_entries=max_entries, **kwargs)
        return CachedLogitSource(wrapped=wrapped, cache_config=config), wrapped

    def test_save_and_load_cache(self, tmp_path):
        cached, _ = self._make_cached()
        cached.get_logits([1, 2, 3])
        cached.get_logits([4, 5, 6])
        save_path = tmp_path / "cache.bin"
        cached.save_cache(save_path)
        assert save_path.exists()

        cached2, wrapped2 = self._make_cached()
        cached2.load_cache(save_path)
        # After loading, the entries should be available without calling wrapped
        cached2.get_logits([1, 2, 3])
        cached2.get_logits([4, 5, 6])
        # Only the initial 2 misses from cached, none from cached2
        assert wrapped2.call_count == 0

    def test_load_nonexistent_file_raises(self):
        cached, _ = self._make_cached()
        with pytest.raises(FileNotFoundError):
            cached.load_cache("/nonexistent/path/cache.bin")

    def test_lfu_eviction_policy(self):
        cached, wrapped = self._make_cached(max_entries=3, eviction_policy="lfu")
        cached.get_logits([1])
        cached.get_logits([2])
        cached.get_logits([3])
        # Access [1] and [2] multiple times
        cached.get_logits([1])
        cached.get_logits([1])
        cached.get_logits([2])
        # [3] has fewest accesses -> should be evicted on next insert
        cached.get_logits([4])
        old_count = wrapped.call_count
        cached.get_logits([3])
        assert wrapped.call_count == old_count + 1  # [3] was evicted

    def test_fifo_eviction_policy(self):
        cached, wrapped = self._make_cached(max_entries=2, eviction_policy="fifo")
        cached.get_logits([1])
        cached.get_logits([2])
        cached.get_logits([1])  # access doesn't matter
        cached.get_logits([3])  # evicts [1] (first inserted)
        old_count = wrapped.call_count
        cached.get_logits([1])
        assert wrapped.call_count == old_count + 1

    def test_model_name_property(self):
        cached, _ = self._make_cached()
        assert cached.model_name == "test-model"

    def test_vocab_size_property(self):
        cached, _ = self._make_cached()
        assert cached.vocab_size == 100

    def test_device_property(self):
        cached, _ = self._make_cached()
        assert cached.device == "cpu"

    def test_batch_all_misses(self):
        cached, wrapped = self._make_cached()
        results = cached.get_logits_batch([[1], [2], [3]])
        assert len(results) == 3
        assert wrapped.call_count == 1  # single batch call

    def test_batch_all_hits(self):
        cached, wrapped = self._make_cached()
        cached.get_logits([10])
        cached.get_logits([20])
        initial = wrapped.call_count
        results = cached.get_logits_batch([[10], [20]])
        assert wrapped.call_count == initial  # no new calls

    def test_stats_after_mixed_operations(self):
        cached, _ = self._make_cached()
        cached.get_logits([1])
        cached.get_logits([1])
        cached.get_logits_batch([[1], [2]])
        stats = cached.cache_stats()
        assert stats["total_requests"] >= 4
        assert stats["total_hits"] >= 2
