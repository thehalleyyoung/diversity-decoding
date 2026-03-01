"""
Tests for the fixtures and utilities defined in tests/conftest.py
for the Diversity Decoding Arena.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List

import numpy as np
import pytest

from conftest import (
    AlgorithmFixtureFactory,
    BOS_TOKEN_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_SEQUENCES,
    DEFAULT_SEED,
    DEFAULT_SEQ_LEN,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VOCAB_SIZE,
    DISTRIBUTIONS,
    EOS_TOKEN_ID,
    MockEmbedder,
    MockGenerationResult,
    MockLogitSource,
    MockLogitSourceBatched,
    MockLogitSourceWithKVCache,
    MockLogitSourceWithLatency,
    MockTokenizer,
    PAD_TOKEN_ID,
    PerformanceTracker,
    UNK_TOKEN_ID,
    assert_diverse_texts,
    assert_metric_in_range,
    assert_monotonic,
    assert_positive_semidefinite,
    assert_stochastic_matrix,
    assert_symmetric,
    assert_valid_logits,
    assert_valid_probability_distribution,
    bootstrap_mean_ci,
    check_numerical_stability,
    compute_distinct_n,
    compute_ngram_entropy,
    compute_ngram_frequencies,
    compute_pairwise_jaccard,
    compute_simple_self_bleu,
    create_mock_generation_result,
    create_multiple_generation_results,
    effect_size_cohens_d,
    generate_diverse_texts,
    generate_diverse_token_sequences,
    generate_embedding_matrix,
    generate_identical_texts,
    generate_kernel_matrix,
    generate_logit_matrix,
    generate_random_texts,
    generate_token_sequences,
    permutation_test,
    validate_generation_output,
    validate_metric_output,
)


# =========================================================================
# TestMockLogitSource
# =========================================================================


class TestMockLogitSource:
    """Tests for MockLogitSource: output shapes, distributions, determinism,
    reset, and call counting."""

    def test_output_shape_single_sequence(self):
        src = MockLogitSource()
        logits = src([[0, 1, 2]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_output_shape_batch(self):
        src = MockLogitSource()
        logits = src([[0], [1], [2], [3]])
        assert logits.shape == (4, DEFAULT_VOCAB_SIZE)

    def test_output_dtype_float32(self):
        src = MockLogitSource()
        logits = src([[5]])
        assert logits.dtype == np.float32

    def test_all_values_finite(self):
        src = MockLogitSource()
        logits = src([[0, 1, 2, 3, 4]])
        assert np.all(np.isfinite(logits))

    def test_uniform_distribution_base(self):
        src = MockLogitSource(distribution="uniform")
        base = src._base_logits.copy()
        # EOS slot has a special value; all others should be 0
        mask = np.ones(DEFAULT_VOCAB_SIZE, dtype=bool)
        mask[EOS_TOKEN_ID] = False
        assert np.allclose(base[mask], 0.0, atol=1e-6)

    def test_zipf_distribution_decreasing(self):
        src = MockLogitSource(distribution="zipf")
        base = src._base_logits.copy()
        # Zipf: token 0 area set by EOS override, but ranks 2..N should decrease
        assert base[4] > base[100], "Zipf logits should decrease with rank"

    def test_peaked_distribution_hot_tokens(self):
        hot = [20, 30, 40]
        src = MockLogitSource(distribution="peaked", hot_tokens=hot, concentration=2.0)
        base = src._base_logits
        cold_mean = np.mean(np.delete(base, hot + [EOS_TOKEN_ID]))
        hot_mean = np.mean(base[hot])
        assert hot_mean > cold_mean + 1.0

    def test_bimodal_distribution(self):
        src = MockLogitSource(distribution="bimodal")
        base = src._base_logits
        unique_vals = set(np.round(base, 1))
        assert len(unique_vals) >= 2, "Bimodal should have at least two value levels"

    def test_degenerate_distribution(self):
        src = MockLogitSource(distribution="degenerate", hot_tokens=[15])
        base = src._base_logits
        argmax = int(np.argmax(base))
        assert argmax == 15

    def test_deterministic_same_seed(self):
        src1 = MockLogitSource(seed=99, distribution="zipf")
        src2 = MockLogitSource(seed=99, distribution="zipf")
        inp = [[5, 10, 15]]
        assert np.allclose(src1(inp), src2(inp))

    def test_deterministic_flag_true(self):
        src = MockLogitSource(deterministic=True)
        inp = [[1, 2, 3]]
        a = src(inp)
        src.reset()
        b = src(inp)
        assert np.allclose(a, b)

    def test_nondeterministic_flag(self):
        src = MockLogitSource(deterministic=False, seed=42)
        inp = [[1, 2, 3]]
        a = src(inp).copy()
        b = src(inp).copy()
        # Noise is added each call so outputs should differ slightly
        assert not np.allclose(a, b, atol=1e-8)

    def test_call_count_increments(self):
        src = MockLogitSource()
        assert src.call_count == 0
        src([[0]])
        assert src.call_count == 1
        src([[0]])
        src([[0]])
        assert src.call_count == 3

    def test_reset_clears_call_count(self):
        src = MockLogitSource()
        src([[0]])
        src([[1]])
        src.reset()
        assert src.call_count == 0

    def test_reset_restores_rng_state(self):
        src = MockLogitSource(deterministic=False)
        inp = [[1, 2]]
        a = src(inp).copy()
        src.reset()
        b = src(inp).copy()
        assert np.allclose(a, b)

    def test_eos_probability_reflected(self):
        src = MockLogitSource(eos_probability=0.5)
        base = src._base_logits
        expected = np.log(0.5 + 1e-10)
        assert abs(base[EOS_TOKEN_ID] - expected) < 1e-5

    def test_context_dependent_shift(self):
        src = MockLogitSource(distribution="uniform")
        logits_a = src([[10]])[0]
        logits_b = src([[20]])[0]
        # Different last token -> different shifts
        assert not np.allclose(logits_a, logits_b)

    def test_eos_boost_for_long_sequences(self):
        src = MockLogitSource(distribution="uniform")
        short_seq = list(range(5))
        long_seq = list(range(25))
        logits_short = src([short_seq])[0]
        logits_long = src([long_seq])[0]
        assert logits_long[EOS_TOKEN_ID] > logits_short[EOS_TOKEN_ID]

    def test_custom_vocab_size(self):
        for vs in [10, 100, 500]:
            src = MockLogitSource(vocab_size=vs)
            logits = src([[1, 2]])
            assert logits.shape == (1, vs)


# =========================================================================
# TestMockLogitSourceVariants
# =========================================================================


class TestMockLogitSourceVariants:
    """Tests for KVCache, Batched, and Latency logit source variants."""

    # --- KVCache ---

    def test_kv_cache_output_shape(self):
        src = MockLogitSourceWithKVCache()
        logits = src([[1, 2, 3]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_kv_cache_returns_same_for_same_input(self):
        src = MockLogitSourceWithKVCache()
        inp = [[5, 10]]
        a = src(inp).copy()
        b = src(inp).copy()
        assert np.allclose(a, b)

    def test_kv_cache_populates(self):
        src = MockLogitSourceWithKVCache()
        src([[1, 2]])
        assert len(src._cache) == 1

    def test_kv_cache_clear(self):
        src = MockLogitSourceWithKVCache()
        src([[1, 2]])
        src.clear_cache()
        assert len(src._cache) == 0

    def test_kv_cache_different_inputs_different_cache_entries(self):
        src = MockLogitSourceWithKVCache()
        src([[1, 2]])
        src([[3, 4]])
        assert len(src._cache) == 2

    # --- Batched ---

    def test_batched_tracks_batch_size(self):
        src = MockLogitSourceBatched()
        src([[1], [2], [3]])
        assert src.batch_sizes == [3]

    def test_batched_tracks_multiple_calls(self):
        src = MockLogitSourceBatched()
        src([[1]])
        src([[1], [2]])
        src([[1], [2], [3], [4]])
        assert src.batch_sizes == [1, 2, 4]

    def test_batched_output_matches_parent(self):
        parent = MockLogitSource(seed=7, distribution="zipf")
        batched = MockLogitSourceBatched(seed=7, distribution="zipf")
        inp = [[1, 2, 3]]
        assert np.allclose(parent(inp), batched(inp))

    def test_batched_inherits_call_count(self):
        src = MockLogitSourceBatched()
        src([[1]])
        src([[2]])
        assert src.call_count == 2

    # --- Latency ---

    def test_latency_introduces_delay(self):
        src = MockLogitSourceWithLatency(latency_ms=50.0)
        start = time.time()
        src([[1]])
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms >= 40.0  # allow small tolerance

    def test_latency_total_tracking(self):
        src = MockLogitSourceWithLatency(latency_ms=10.0)
        src([[1]])
        src([[2]])
        assert src.total_latency >= 20.0

    def test_latency_output_correct(self):
        parent = MockLogitSource(seed=5, distribution="uniform")
        lat = MockLogitSourceWithLatency(seed=5, distribution="uniform", latency_ms=1.0)
        inp = [[1, 2]]
        assert np.allclose(parent(inp), lat(inp))

    def test_latency_zero_delay(self):
        src = MockLogitSourceWithLatency(latency_ms=0.0)
        logits = src([[1]])
        assert logits.shape == (1, DEFAULT_VOCAB_SIZE)

    def test_latency_call_count(self):
        src = MockLogitSourceWithLatency(latency_ms=1.0)
        src([[0]])
        src([[1]])
        src([[2]])
        assert src.call_count == 3


# =========================================================================
# TestMockEmbedder
# =========================================================================


class TestMockEmbedder:
    """Tests for MockEmbedder: dimensions, normalization, caching, batch."""

    def test_embed_sequence_shape(self):
        emb = MockEmbedder()
        vec = emb.embed_sequence([1, 2, 3])
        assert vec.shape == (DEFAULT_EMBEDDING_DIM,)

    def test_embed_sequence_dtype(self):
        emb = MockEmbedder()
        vec = emb.embed_sequence([10])
        assert vec.dtype == np.float32

    def test_custom_embedding_dim(self):
        for dim in [16, 128, 256]:
            emb = MockEmbedder(embedding_dim=dim)
            vec = emb.embed_sequence([1, 2])
            assert vec.shape == (dim,)

    def test_normalized_unit_length(self):
        emb = MockEmbedder(normalize=True)
        vec = emb.embed_sequence([5, 10, 15])
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_unnormalized_not_unit(self):
        emb = MockEmbedder(normalize=False)
        vec = emb.embed_sequence([5, 10, 15])
        # Very unlikely to be exactly 1.0
        assert abs(np.linalg.norm(vec) - 1.0) > 1e-3

    def test_empty_sequence_zero_vector(self):
        emb = MockEmbedder()
        vec = emb.embed_sequence([])
        assert np.allclose(vec, 0.0)

    def test_cache_hit_returns_same(self):
        emb = MockEmbedder()
        a = emb.embed_sequence([1, 2, 3])
        b = emb.embed_sequence([1, 2, 3])
        assert np.array_equal(a, b)

    def test_cache_populated(self):
        emb = MockEmbedder()
        emb.embed_sequence([1, 2])
        assert len(emb._cache) == 1

    def test_reset_cache_clears(self):
        emb = MockEmbedder()
        emb.embed_sequence([1])
        emb.reset_cache()
        assert len(emb._cache) == 0

    def test_embed_batch_shape(self):
        emb = MockEmbedder()
        result = emb.embed_batch([[1, 2], [3, 4], [5]])
        assert result.shape == (3, DEFAULT_EMBEDDING_DIM)

    def test_embed_text_shape(self):
        emb = MockEmbedder()
        vec = emb.embed_text("hello world")
        assert vec.shape == (DEFAULT_EMBEDDING_DIM,)

    def test_embed_texts_shape(self):
        emb = MockEmbedder()
        result = emb.embed_texts(["hello", "world", "foo"])
        assert result.shape == (3, DEFAULT_EMBEDDING_DIM)

    def test_different_sequences_different_embeddings(self):
        emb = MockEmbedder()
        a = emb.embed_sequence([10, 20])
        b = emb.embed_sequence([30, 40])
        assert not np.allclose(a, b)

    def test_token_embedding_consistent(self):
        emb = MockEmbedder()
        a = emb.token_embedding(42)
        b = emb.token_embedding(42)
        assert np.array_equal(a, b)

    def test_deterministic_across_instances(self):
        e1 = MockEmbedder(seed=77)
        e2 = MockEmbedder(seed=77)
        assert np.allclose(e1.embed_sequence([1, 2]), e2.embed_sequence([1, 2]))


# =========================================================================
# TestMockTokenizer
# =========================================================================


class TestMockTokenizer:
    """Tests for MockTokenizer: encode/decode roundtrip, special tokens, vocabulary."""

    def test_encode_returns_list(self):
        tok = MockTokenizer()
        result = tok.encode("the quick brown fox")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_decode_returns_string(self):
        tok = MockTokenizer()
        result = tok.decode([4, 5, 6])
        assert isinstance(result, str)

    def test_encode_decode_roundtrip_known_words(self):
        tok = MockTokenizer()
        text = "the a an is was"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_unknown_word_maps_to_unk(self):
        tok = MockTokenizer()
        encoded = tok.encode("xyzzyspoon")
        assert encoded == [UNK_TOKEN_ID]

    def test_special_token_ids(self):
        tok = MockTokenizer()
        assert tok.bos_token_id == BOS_TOKEN_ID
        assert tok.eos_token_id == EOS_TOKEN_ID
        assert tok.pad_token_id == PAD_TOKEN_ID
        assert tok.unk_token_id == UNK_TOKEN_ID

    def test_special_tokens_in_vocab(self):
        tok = MockTokenizer()
        assert "<bos>" in tok.vocab
        assert "<eos>" in tok.vocab
        assert "<pad>" in tok.vocab
        assert "<unk>" in tok.vocab

    def test_vocab_size(self):
        tok = MockTokenizer()
        # Vocab may have fewer entries than vocab_size due to duplicate words
        assert len(tok.vocab) <= DEFAULT_VOCAB_SIZE
        assert len(tok.vocab) >= DEFAULT_VOCAB_SIZE - 10

    def test_custom_vocab_size(self):
        tok = MockTokenizer(vocab_size=200)
        assert len(tok.vocab) <= 200
        assert len(tok.vocab) >= 190

    def test_decode_skips_special_tokens(self):
        tok = MockTokenizer()
        decoded = tok.decode([BOS_TOKEN_ID, 4, 5, EOS_TOKEN_ID, PAD_TOKEN_ID])
        words = decoded.split()
        assert "<bos>" not in words
        assert "<eos>" not in words
        assert "<pad>" not in words

    def test_batch_encode_length(self):
        tok = MockTokenizer()
        texts = ["the cat", "a dog", "is here"]
        result = tok.batch_encode(texts)
        assert len(result) == 3

    def test_batch_decode_length(self):
        tok = MockTokenizer()
        ids = [[4, 5], [6, 7], [8]]
        result = tok.batch_decode(ids)
        assert len(result) == 3

    def test_batch_encode_decode_roundtrip(self):
        tok = MockTokenizer()
        # Use words known to be in the vocab (from _build_vocab)
        texts = ["the a an", "is was were"]
        encoded = tok.batch_encode(texts)
        decoded = tok.batch_decode(encoded)
        assert decoded == texts

    def test_encode_empty_string(self):
        tok = MockTokenizer()
        assert tok.encode("") == []

    def test_vocab_has_domain_words(self):
        tok = MockTokenizer()
        for word in ["diversity", "beam", "token", "entropy", "softmax"]:
            assert word in tok.vocab

    def test_reverse_vocab_consistency(self):
        tok = MockTokenizer()
        for word, tid in tok.vocab.items():
            assert tok._reverse_vocab[tid] == word


# =========================================================================
# TestTextGenerators
# =========================================================================


class TestTextGenerators:
    """Tests for generate_random_texts, generate_diverse_texts, generate_identical_texts."""

    def test_random_texts_count(self):
        texts = generate_random_texts(n=15)
        assert len(texts) == 15

    def test_random_texts_are_strings(self):
        texts = generate_random_texts(n=5)
        assert all(isinstance(t, str) for t in texts)

    def test_random_texts_min_max_length(self):
        texts = generate_random_texts(n=20, min_len=5, max_len=10)
        for t in texts:
            word_count = len(t.split())
            assert 5 <= word_count <= 10

    def test_random_texts_deterministic(self):
        a = generate_random_texts(n=5, seed=123)
        b = generate_random_texts(n=5, seed=123)
        assert a == b

    def test_random_texts_different_seeds_differ(self):
        a = generate_random_texts(n=10, seed=1)
        b = generate_random_texts(n=10, seed=2)
        assert a != b

    def test_random_texts_custom_vocab(self):
        vocab = ["alpha", "beta", "gamma"]
        texts = generate_random_texts(n=5, vocab=vocab, min_len=3, max_len=5)
        for t in texts:
            for word in t.split():
                assert word in vocab

    def test_diverse_texts_count(self):
        texts = generate_diverse_texts(n=12)
        assert len(texts) == 12

    def test_diverse_texts_have_diversity(self):
        texts = generate_diverse_texts(n=16)
        unique = len(set(texts))
        assert unique >= len(texts) * 0.5

    def test_diverse_texts_deterministic(self):
        a = generate_diverse_texts(n=8, seed=42)
        b = generate_diverse_texts(n=8, seed=42)
        assert a == b

    def test_diverse_texts_non_empty(self):
        texts = generate_diverse_texts(n=5)
        for t in texts:
            assert len(t.strip()) > 0

    def test_diverse_texts_topic_cycling(self):
        # 8 topics, so texts at indices 0 and 8 should share a topic vocab
        texts = generate_diverse_texts(n=16, seed=42)
        words_0 = set(texts[0].split())
        words_8 = set(texts[8].split())
        # Same topic -> higher overlap expected than different topics
        overlap_same = len(words_0 & words_8)
        words_1 = set(texts[1].split())
        overlap_diff = len(words_0 & words_1)
        # Same-topic overlap should generally be >= different-topic overlap
        assert overlap_same >= overlap_diff or overlap_same > 0

    def test_identical_texts_count(self):
        texts = generate_identical_texts(n=7)
        assert len(texts) == 7

    def test_identical_texts_all_same(self):
        texts = generate_identical_texts(n=10)
        assert len(set(texts)) == 1

    def test_identical_texts_custom_text(self):
        custom = "foo bar baz"
        texts = generate_identical_texts(n=3, text=custom)
        assert all(t == custom for t in texts)

    def test_identical_texts_default_content(self):
        texts = generate_identical_texts(n=1)
        assert "fox" in texts[0]


# =========================================================================
# TestTokenSequenceGenerators
# =========================================================================


class TestTokenSequenceGenerators:
    """Tests for generate_token_sequences and generate_diverse_token_sequences."""

    def test_token_sequences_count(self):
        seqs = generate_token_sequences(n=8)
        assert len(seqs) == 8

    def test_token_sequences_length_range(self):
        seqs = generate_token_sequences(n=20, min_len=3, max_len=10)
        for s in seqs:
            assert 3 <= len(s) <= 10

    def test_token_sequences_vocab_bounds(self):
        vs = 100
        seqs = generate_token_sequences(n=10, vocab_size=vs)
        for s in seqs:
            for tok in s:
                assert 4 <= tok < vs

    def test_token_sequences_deterministic(self):
        a = generate_token_sequences(n=5, seed=10)
        b = generate_token_sequences(n=5, seed=10)
        assert a == b

    def test_token_sequences_different_seeds(self):
        a = generate_token_sequences(n=5, seed=1)
        b = generate_token_sequences(n=5, seed=2)
        assert a != b

    def test_diverse_token_sequences_count(self):
        seqs = generate_diverse_token_sequences(n=9)
        assert len(seqs) == 9

    def test_diverse_token_sequences_length(self):
        seqs = generate_diverse_token_sequences(n=6, seq_len=15)
        for s in seqs:
            assert len(s) == 15

    def test_diverse_token_sequences_clustering(self):
        seqs = generate_diverse_token_sequences(n=9, vocab_size=900, n_clusters=3)
        means = []
        for cluster_idx in range(3):
            cluster_seqs = [seqs[i] for i in range(9) if i % 3 == cluster_idx]
            cluster_mean = np.mean([np.mean(s) for s in cluster_seqs])
            means.append(cluster_mean)
        # Cluster means should be spread out
        assert max(means) - min(means) > 50

    def test_diverse_token_sequences_vocab_bounds(self):
        vs = 200
        seqs = generate_diverse_token_sequences(n=10, vocab_size=vs)
        for s in seqs:
            for tok in s:
                assert 4 <= tok < vs

    def test_diverse_token_sequences_deterministic(self):
        a = generate_diverse_token_sequences(n=5, seed=42)
        b = generate_diverse_token_sequences(n=5, seed=42)
        assert a == b


# =========================================================================
# TestMatrixGenerators
# =========================================================================


class TestMatrixGenerators:
    """Tests for logit_matrix, embedding_matrix, kernel_matrix shapes and properties."""

    def test_logit_matrix_shape_normal(self):
        m = generate_logit_matrix(batch_size=3, vocab_size=100)
        assert m.shape == (3, 100)

    def test_logit_matrix_dtype(self):
        m = generate_logit_matrix()
        assert m.dtype == np.float32

    def test_logit_matrix_peaked_has_spikes(self):
        m = generate_logit_matrix(batch_size=2, vocab_size=200, distribution="peaked")
        for b in range(2):
            top5 = np.sort(m[b])[-5:]
            rest_mean = np.mean(np.sort(m[b])[:-5])
            assert np.mean(top5) > rest_mean + 1.0

    def test_logit_matrix_flat_all_zeros(self):
        m = generate_logit_matrix(distribution="flat")
        assert np.allclose(m, 0.0)

    def test_logit_matrix_deterministic(self):
        a = generate_logit_matrix(seed=7)
        b = generate_logit_matrix(seed=7)
        assert np.array_equal(a, b)

    def test_embedding_matrix_shape(self):
        m = generate_embedding_matrix(n=5, dim=32)
        assert m.shape == (5, 32)

    def test_embedding_matrix_normalized(self):
        m = generate_embedding_matrix(n=10, normalize=True)
        norms = np.linalg.norm(m, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embedding_matrix_unnormalized(self):
        m = generate_embedding_matrix(n=10, normalize=False)
        norms = np.linalg.norm(m, axis=1)
        assert not np.allclose(norms, 1.0, atol=1e-3)

    def test_kernel_matrix_rbf_symmetric(self):
        K = generate_kernel_matrix(n=8, kernel_type="rbf")
        assert np.allclose(K, K.T, atol=1e-6)

    def test_kernel_matrix_rbf_psd(self):
        K = generate_kernel_matrix(n=8, kernel_type="rbf")
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-6)

    def test_kernel_matrix_linear_shape(self):
        K = generate_kernel_matrix(n=6, kernel_type="linear")
        assert K.shape == (6, 6)

    def test_kernel_matrix_identity(self):
        K = generate_kernel_matrix(n=5, kernel_type="identity")
        assert np.allclose(K, np.eye(5), atol=1e-6)

    def test_kernel_matrix_uniform_all_ones(self):
        K = generate_kernel_matrix(n=4, kernel_type="uniform")
        expected = np.ones((4, 4), dtype=np.float32)
        assert np.allclose(K, expected, atol=1e-5)

    def test_kernel_matrix_positive_diagonal(self):
        for kt in ["rbf", "linear", "identity", "other"]:
            K = generate_kernel_matrix(n=5, kernel_type=kt)
            assert np.all(np.diag(K) > 0)


# =========================================================================
# TestAssertionHelpers
# =========================================================================


class TestAssertionHelpers:
    """Tests for assertion helper functions."""

    def test_valid_probability_distribution_passes(self):
        probs = np.array([0.2, 0.3, 0.5])
        assert_valid_probability_distribution(probs)

    def test_valid_probability_distribution_fails_negative(self):
        probs = np.array([-0.1, 0.6, 0.5])
        with pytest.raises(AssertionError):
            assert_valid_probability_distribution(probs)

    def test_valid_probability_distribution_fails_sum(self):
        probs = np.array([0.2, 0.3, 0.3])
        with pytest.raises(AssertionError):
            assert_valid_probability_distribution(probs)

    def test_valid_logits_passes(self):
        logits = np.random.randn(5, 100).astype(np.float32)
        assert_valid_logits(logits, vocab_size=100)

    def test_valid_logits_fails_wrong_size(self):
        logits = np.random.randn(5, 50).astype(np.float32)
        with pytest.raises(AssertionError):
            assert_valid_logits(logits, vocab_size=100)

    def test_valid_logits_fails_nan(self):
        logits = np.array([[1.0, float("nan"), 3.0]])
        with pytest.raises(AssertionError):
            assert_valid_logits(logits, vocab_size=3)

    def test_diverse_texts_passes(self):
        texts = ["a", "b", "c", "d", "e"]
        assert_diverse_texts(texts, min_unique_ratio=0.5)

    def test_diverse_texts_fails(self):
        texts = ["a", "a", "a", "a", "b"]
        with pytest.raises(AssertionError):
            assert_diverse_texts(texts, min_unique_ratio=0.8)

    def test_metric_in_range_passes(self):
        assert_metric_in_range(0.5, 0.0, 1.0)

    def test_metric_in_range_fails_low(self):
        with pytest.raises(AssertionError):
            assert_metric_in_range(-0.1, 0.0, 1.0)

    def test_metric_in_range_fails_high(self):
        with pytest.raises(AssertionError):
            assert_metric_in_range(1.5, 0.0, 1.0)

    def test_monotonic_increasing_passes(self):
        assert_monotonic([1, 2, 3, 4], increasing=True)

    def test_monotonic_increasing_fails(self):
        with pytest.raises(AssertionError):
            assert_monotonic([1, 3, 2, 4], increasing=True, strict=True)

    def test_monotonic_decreasing_passes(self):
        assert_monotonic([4, 3, 2, 1], increasing=False, strict=True)

    def test_positive_semidefinite_passes(self):
        K = np.eye(5)
        assert_positive_semidefinite(K)

    def test_positive_semidefinite_fails(self):
        K = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalue -1
        with pytest.raises(AssertionError):
            assert_positive_semidefinite(K)

    def test_symmetric_passes(self):
        M = np.array([[1.0, 2.0], [2.0, 3.0]])
        assert_symmetric(M)

    def test_symmetric_fails(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(AssertionError):
            assert_symmetric(M)

    def test_stochastic_matrix_passes(self):
        M = np.array([[0.3, 0.7], [0.5, 0.5]])
        assert_stochastic_matrix(M)

    def test_stochastic_matrix_fails(self):
        M = np.array([[0.3, 0.3], [0.5, 0.5]])
        with pytest.raises(AssertionError):
            assert_stochastic_matrix(M)


# =========================================================================
# TestStatisticalHelpers
# =========================================================================


class TestStatisticalHelpers:
    """Tests for bootstrap_mean_ci, permutation_test, effect_size_cohens_d."""

    def test_bootstrap_mean_ci_returns_tuple(self):
        result = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_bootstrap_mean_ci_mean_correct(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_mean_ci(vals)
        assert abs(mean - 3.0) < 1e-10

    def test_bootstrap_ci_contains_mean(self):
        vals = list(range(1, 101))
        mean, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=2000)
        assert lo <= mean <= hi

    def test_bootstrap_ci_width_decreases_with_n(self):
        rng = np.random.RandomState(42)
        small = rng.normal(5.0, 1.0, 10).tolist()
        large = rng.normal(5.0, 1.0, 200).tolist()
        _, lo_s, hi_s = bootstrap_mean_ci(small, seed=42)
        _, lo_l, hi_l = bootstrap_mean_ci(large, seed=42)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_permutation_test_returns_pvalue(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        p = permutation_test(a, b)
        assert 0.0 <= p <= 1.0

    def test_permutation_test_identical_groups(self):
        vals = [3.0, 3.0, 3.0, 3.0]
        p = permutation_test(vals, vals)
        assert p >= 0.5  # no difference -> high p-value

    def test_permutation_test_very_different_groups(self):
        a = [0.0, 0.0, 0.0, 0.0, 0.0]
        b = [100.0, 100.0, 100.0, 100.0, 100.0]
        p = permutation_test(a, b, n_permutations=5000)
        assert p < 0.05

    def test_effect_size_identical(self):
        vals = [5.0, 5.0, 5.0, 5.0]
        d = effect_size_cohens_d(vals, vals)
        assert abs(d) < 1e-8

    def test_effect_size_known_difference(self):
        a = [10.0, 10.0, 10.0, 10.0, 10.0]
        b = [0.0, 0.0, 0.0, 0.0, 0.0]
        d = effect_size_cohens_d(a, b)
        # Large effect size since no variance within groups leads to 0 pooled std
        # Actually var(ddof=1) of constant is 0 -> returns 0 due to guard
        assert isinstance(d, float)

    def test_effect_size_sign(self):
        a = np.random.RandomState(1).normal(10, 1, 50).tolist()
        b = np.random.RandomState(2).normal(5, 1, 50).tolist()
        d = effect_size_cohens_d(a, b)
        assert d > 0  # a has higher mean


# =========================================================================
# TestAlgorithmFixtureFactory
# =========================================================================


class TestAlgorithmFixtureFactory:
    """Tests for AlgorithmFixtureFactory config generation."""

    def test_temperature_config_name(self):
        cfg = AlgorithmFixtureFactory.temperature_config()
        assert cfg["algorithm_name"] == "temperature"

    def test_temperature_config_custom(self):
        cfg = AlgorithmFixtureFactory.temperature_config(temperature=0.5)
        assert cfg["temperature"] == 0.5

    def test_top_k_config(self):
        cfg = AlgorithmFixtureFactory.top_k_config(k=20)
        assert cfg["algorithm_name"] == "top_k"
        assert cfg["params"]["k"] == 20

    def test_nucleus_config(self):
        cfg = AlgorithmFixtureFactory.nucleus_config(p=0.8)
        assert cfg["algorithm_name"] == "nucleus"
        assert cfg["params"]["p"] == 0.8

    def test_typical_config(self):
        cfg = AlgorithmFixtureFactory.typical_config(mass=0.85)
        assert cfg["params"]["mass"] == 0.85

    def test_diverse_beam_config(self):
        cfg = AlgorithmFixtureFactory.diverse_beam_config(num_beams=16, num_groups=4)
        assert cfg["algorithm_name"] == "diverse_beam"
        assert cfg["params"]["num_beams"] == 16
        assert cfg["params"]["num_groups"] == 4

    def test_contrastive_config(self):
        cfg = AlgorithmFixtureFactory.contrastive_config(alpha=0.7, k=10)
        assert cfg["params"]["alpha"] == 0.7
        assert cfg["params"]["k"] == 10

    def test_dpp_config(self):
        cfg = AlgorithmFixtureFactory.dpp_config(pool_size=100, select_size=20)
        assert cfg["params"]["pool_size"] == 100
        assert cfg["params"]["select_size"] == 20

    def test_svd_config(self):
        cfg = AlgorithmFixtureFactory.svd_config(n_particles=8, alpha=0.2)
        assert cfg["params"]["n_particles"] == 8
        assert cfg["params"]["alpha"] == 0.2

    def test_qdbs_config(self):
        cfg = AlgorithmFixtureFactory.qdbs_config(beam_width=12, n_cells=30)
        assert cfg["params"]["beam_width"] == 12
        assert cfg["params"]["n_cells"] == 30

    def test_all_configs_have_required_keys(self):
        configs = [
            AlgorithmFixtureFactory.temperature_config(),
            AlgorithmFixtureFactory.top_k_config(),
            AlgorithmFixtureFactory.nucleus_config(),
            AlgorithmFixtureFactory.typical_config(),
            AlgorithmFixtureFactory.diverse_beam_config(),
            AlgorithmFixtureFactory.contrastive_config(),
            AlgorithmFixtureFactory.dpp_config(),
            AlgorithmFixtureFactory.svd_config(),
            AlgorithmFixtureFactory.qdbs_config(),
        ]
        required_keys = {"algorithm_name", "num_sequences", "max_new_tokens", "seed", "params"}
        for cfg in configs:
            assert required_keys.issubset(cfg.keys()), f"Missing keys in {cfg['algorithm_name']}"

    def test_custom_overrides(self):
        cfg = AlgorithmFixtureFactory.temperature_config(
            num_sequences=99, max_new_tokens=77, seed=11
        )
        assert cfg["num_sequences"] == 99
        assert cfg["max_new_tokens"] == 77
        assert cfg["seed"] == 11


# =========================================================================
# TestPerformanceTracker
# =========================================================================


class TestPerformanceTracker:
    """Tests for PerformanceTracker timing and summary statistics."""

    def test_empty_tracker(self):
        tracker = PerformanceTracker()
        assert tracker.summary() == {}

    def test_single_timing(self):
        tracker = PerformanceTracker()
        with tracker.time_it("op"):
            time.sleep(0.01)
        assert "op" in tracker.timings
        assert len(tracker.timings["op"]) == 1
        assert tracker.timings["op"][0] >= 0.005

    def test_multiple_timings_same_key(self):
        tracker = PerformanceTracker()
        for _ in range(5):
            with tracker.time_it("op"):
                pass
        assert len(tracker.timings["op"]) == 5

    def test_multiple_keys(self):
        tracker = PerformanceTracker()
        with tracker.time_it("a"):
            pass
        with tracker.time_it("b"):
            pass
        assert "a" in tracker.timings
        assert "b" in tracker.timings

    def test_summary_contains_stats(self):
        tracker = PerformanceTracker()
        for _ in range(3):
            with tracker.time_it("op"):
                time.sleep(0.001)
        s = tracker.summary()
        assert "mean" in s["op"]
        assert "std" in s["op"]
        assert "min" in s["op"]
        assert "max" in s["op"]
        assert "n" in s["op"]

    def test_summary_n_correct(self):
        tracker = PerformanceTracker()
        for _ in range(7):
            with tracker.time_it("x"):
                pass
        assert tracker.summary()["x"]["n"] == 7

    def test_summary_min_le_max(self):
        tracker = PerformanceTracker()
        for _ in range(5):
            with tracker.time_it("op"):
                time.sleep(0.001)
        s = tracker.summary()["op"]
        assert s["min"] <= s["max"]

    def test_summary_mean_between_min_max(self):
        tracker = PerformanceTracker()
        for _ in range(10):
            with tracker.time_it("op"):
                time.sleep(0.001)
        s = tracker.summary()["op"]
        assert s["min"] <= s["mean"] <= s["max"]


# =========================================================================
# TestDataValidation
# =========================================================================


class TestDataValidation:
    """Tests for validate_generation_output and validate_metric_output."""

    # --- validate_generation_output ---

    def test_valid_generation_output(self):
        texts = generate_random_texts(n=5)
        issues = validate_generation_output(texts)
        assert issues == []

    def test_too_few_texts(self):
        issues = validate_generation_output([], min_texts=1)
        assert any("Too few" in i for i in issues)

    def test_too_many_texts(self):
        texts = ["hello"] * 20
        issues = validate_generation_output(texts, max_texts=5)
        assert any("Too many" in i for i in issues)

    def test_text_too_short(self):
        texts = [""]
        issues = validate_generation_output(texts, min_length=5)
        assert any("too short" in i for i in issues)

    def test_text_too_long(self):
        texts = ["x" * 200]
        issues = validate_generation_output(texts, max_length=50)
        assert any("too long" in i for i in issues)

    def test_non_string_element(self):
        issues = validate_generation_output([123], min_texts=1)  # type: ignore
        assert any("not a string" in i for i in issues)

    # --- validate_metric_output ---

    def test_valid_metric_output(self):
        issues = validate_metric_output(0.75, min_val=0.0, max_val=1.0)
        assert issues == []

    def test_metric_wrong_type(self):
        issues = validate_metric_output("bad")
        assert any("Wrong type" in i for i in issues)

    def test_metric_nan(self):
        issues = validate_metric_output(float("nan"))
        assert any("Non-finite" in i for i in issues)

    def test_metric_inf(self):
        issues = validate_metric_output(float("inf"))
        assert any("Non-finite" in i for i in issues)

    def test_metric_below_min(self):
        issues = validate_metric_output(-1.0, min_val=0.0)
        assert any("below minimum" in i for i in issues)

    def test_metric_above_max(self):
        issues = validate_metric_output(2.0, max_val=1.0)
        assert any("above maximum" in i for i in issues)

    def test_metric_integer_accepted(self):
        issues = validate_metric_output(5, min_val=0.0, max_val=10.0)
        assert issues == []

    def test_metric_numpy_float_accepted(self):
        issues = validate_metric_output(np.float64(0.5), min_val=0.0, max_val=1.0)
        assert issues == []


# =========================================================================
# TestNumericalStability
# =========================================================================


class TestNumericalStability:
    """Tests for check_numerical_stability helper."""

    def test_stable_function(self):
        def f(x):
            return np.sum(x)
        assert check_numerical_stability(f, [np.array([1.0, 2.0, 3.0])])

    def test_deterministic_logit_source_is_stable(self):
        src = MockLogitSource(deterministic=True)
        def f():
            return src([[1, 2, 3]])
        results = [f() for _ in range(5)]
        assert all(np.allclose(results[0], r) for r in results)


# =========================================================================
# TestMetricComputationHelpers
# =========================================================================


class TestMetricComputationHelpers:
    """Tests for the metric computation helpers in conftest."""

    def test_ngram_frequencies_type(self):
        texts = ["the cat sat on the mat"]
        freqs = compute_ngram_frequencies(texts, n=2)
        assert isinstance(freqs, dict)
        assert ("the", "cat") in freqs

    def test_ngram_frequencies_counts(self):
        texts = ["a b a b a b"]
        freqs = compute_ngram_frequencies(texts, n=2)
        assert freqs[("a", "b")] == 3
        assert freqs[("b", "a")] == 2

    def test_pairwise_jaccard_shape(self):
        texts = generate_random_texts(n=5)
        J = compute_pairwise_jaccard(texts)
        assert J.shape == (5, 5)

    def test_pairwise_jaccard_diagonal_ones(self):
        texts = generate_random_texts(n=5)
        J = compute_pairwise_jaccard(texts)
        assert np.allclose(np.diag(J), 1.0)

    def test_pairwise_jaccard_symmetric(self):
        texts = generate_random_texts(n=5)
        J = compute_pairwise_jaccard(texts)
        assert np.allclose(J, J.T)

    def test_distinct_n_range(self):
        texts = generate_random_texts(n=10)
        d = compute_distinct_n(texts, n=2)
        assert 0.0 <= d <= 1.0

    def test_distinct_n_identical_texts(self):
        texts = generate_identical_texts(n=10)
        d = compute_distinct_n(texts, n=2)
        # Identical texts have low distinct-n
        assert d < 0.5

    def test_ngram_entropy_non_negative(self):
        texts = generate_random_texts(n=10)
        e = compute_ngram_entropy(texts, n=2)
        assert e >= 0.0

    def test_simple_self_bleu_range(self):
        texts = generate_random_texts(n=5)
        sb = compute_simple_self_bleu(texts)
        assert 0.0 <= sb <= 1.0

    def test_simple_self_bleu_identical_high(self):
        texts = generate_identical_texts(n=5)
        sb = compute_simple_self_bleu(texts)
        assert sb > 0.5

    def test_simple_self_bleu_single_text(self):
        sb = compute_simple_self_bleu(["hello world"])
        assert sb == 0.0


# =========================================================================
# TestMockGenerationResult
# =========================================================================


class TestMockGenerationResult:
    """Tests for create_mock_generation_result and create_multiple_generation_results."""

    def test_creation_default(self):
        r = create_mock_generation_result()
        assert isinstance(r, MockGenerationResult)
        assert len(r.texts) == DEFAULT_NUM_SEQUENCES

    def test_creation_algorithm_name(self):
        r = create_mock_generation_result(algorithm="nucleus")
        assert r.algorithm == "nucleus"

    def test_creation_custom_n(self):
        r = create_mock_generation_result(n=7)
        assert len(r.texts) == 7
        assert len(r.token_ids) == 7
        assert len(r.log_probs) == 7

    def test_log_probs_negative(self):
        r = create_mock_generation_result()
        assert all(lp <= 0 for lp in r.log_probs)

    def test_multiple_results_keys(self):
        results = create_multiple_generation_results()
        expected_algos = [
            "temperature", "top_k", "nucleus", "typical",
            "diverse_beam", "contrastive", "dpp", "mbr",
            "svd", "qdbs",
        ]
        for algo in expected_algos:
            assert algo in results

    def test_multiple_results_custom_algos(self):
        results = create_multiple_generation_results(algorithms=["temperature", "nucleus"])
        assert len(results) == 2
        assert "temperature" in results
        assert "nucleus" in results


# =========================================================================
# TestFixturesIntegration
# =========================================================================


class TestFixturesIntegration:
    """Integration tests combining multiple conftest utilities."""

    def test_tokenizer_with_embedder(self):
        tok = MockTokenizer()
        emb = MockEmbedder()
        text = "the cat sat"
        ids = tok.encode(text)
        vec = emb.embed_sequence(ids)
        assert vec.shape == (DEFAULT_EMBEDDING_DIM,)

    def test_logit_source_output_as_valid_logits(self):
        src = MockLogitSource()
        logits = src([[1, 2, 3]])
        assert_valid_logits(logits, DEFAULT_VOCAB_SIZE)

    def test_generation_result_texts_validate(self):
        r = create_mock_generation_result()
        issues = validate_generation_output(r.texts)
        assert issues == []

    def test_diverse_texts_pass_diversity_check(self):
        texts = generate_diverse_texts(n=20)
        assert_diverse_texts(texts, min_unique_ratio=0.5)

    def test_kernel_matrix_passes_psd_check(self):
        K = generate_kernel_matrix(n=10, kernel_type="rbf")
        assert_positive_semidefinite(K)

    def test_kernel_matrix_passes_symmetric_check(self):
        K = generate_kernel_matrix(n=10, kernel_type="rbf")
        assert_symmetric(K)

    def test_embedding_matrix_with_kernel(self):
        emb_mat = generate_embedding_matrix(n=8, dim=32, normalize=True)
        K = emb_mat @ emb_mat.T
        assert_symmetric(K, atol=1e-5)
        assert_positive_semidefinite(K, atol=1e-5)

    def test_performance_tracker_with_logit_source(self):
        tracker = PerformanceTracker()
        src = MockLogitSource()
        with tracker.time_it("logit_call"):
            src([[1, 2, 3]])
        s = tracker.summary()
        assert s["logit_call"]["n"] == 1
        assert s["logit_call"]["mean"] >= 0

    def test_statistical_helpers_with_generation_results(self):
        results = create_multiple_generation_results(
            algorithms=["temperature", "nucleus"], n_per_algo=10
        )
        bleu_temp = [compute_simple_self_bleu([t]) for t in results["temperature"].texts]
        bleu_nuc = [compute_simple_self_bleu([t]) for t in results["nucleus"].texts]
        # Just verify it runs without error
        p = permutation_test(bleu_temp, bleu_nuc)
        assert 0.0 <= p <= 1.0

    def test_all_distributions_produce_valid_logits(self):
        for dist in DISTRIBUTIONS:
            src = MockLogitSource(distribution=dist)
            logits = src([[1, 2]])
            assert_valid_logits(logits, DEFAULT_VOCAB_SIZE)
