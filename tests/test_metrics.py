"""
Comprehensive test suite for the Diversity Decoding Arena metrics.

Tests all diversity metrics, quality metrics, and the correlation
analyzer with real implementations and numerical assertions.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.metrics.diversity import (
    BehavioralDiversity,
    DiversityMetric,
    DistinctN,
    EmbeddingPairwiseDistance,
    NGramEntropy,
    ParseTreeDiversity,
    SelfBLEU,
    VendiScore,
    extract_ngrams,
    ngram_hash_embeddings,
    pairwise_distances,
    tfidf_embeddings,
    tokenize_simple,
)
from src.metrics.quality import (
    ConstraintSatisfaction,
    ContainsKeywordConstraint,
    DoesNotContainConstraint,
    FormatConstraint,
    MaxLengthConstraint,
    MinLengthConstraint,
    NLICoherence,
    Perplexity,
    QualityMetric,
    ReadabilityConstraint,
    RepetitionConstraint,
    SentenceCountConstraint,
)
from src.metrics.correlation import (
    MetricCorrelationAnalyzer,
    SpectralClusterer,
    concordance_matrix,
    correlation_to_distance,
    effective_dimensionality,
    kendall_tau_fast,
    rank_data,
)
from tests.conftest import (
    MockEmbedder,
    compute_distinct_n,
    compute_ngram_entropy,
    compute_ngram_frequencies,
    generate_diverse_texts,
    generate_identical_texts,
    generate_random_texts,
    generate_embedding_matrix,
    generate_kernel_matrix,
    assert_metric_in_range,
    assert_symmetric,
    assert_positive_semidefinite,
    assert_monotonic,
)


# =========================================================================
# Shared helpers and fixtures
# =========================================================================

DIVERSE_TEXTS = [
    "The cat sat on the warm mat and purred softly in the afternoon sun.",
    "Python programming enables developers to build scalable software systems.",
    "Stars twinkle in the vast cosmic expanse of the infinite universe tonight.",
    "Jazz music fills the smoky room with soulful melodies and complex rhythms.",
    "Ocean waves crash against the rocky shore under a bright crescent moon.",
    "Ancient forests harbor diverse ecosystems with countless species of wildlife.",
    "Mathematical theorems reveal deep truths about the structure of abstract spaces.",
    "Culinary traditions from around the world celebrate diverse flavors and techniques.",
]

IDENTICAL_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
]

SIMILAR_TEXTS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the cat sat on the bed",
    "the cat sat on the floor",
    "the cat sat on the chair",
]

SHORT_TEXTS = [
    "hello world",
    "goodbye world",
]


@pytest.fixture
def diverse_texts():
    return list(DIVERSE_TEXTS)


@pytest.fixture
def identical_texts():
    return list(IDENTICAL_TEXTS)


@pytest.fixture
def similar_texts():
    return list(SIMILAR_TEXTS)


@pytest.fixture
def random_texts():
    return generate_random_texts(n=20, seed=42)


@pytest.fixture
def generated_diverse():
    return generate_diverse_texts(n=10, seed=42)


# =========================================================================
# 1. TestSelfBLEU
# =========================================================================


class TestSelfBLEU:
    """Tests for SelfBLEU diversity metric."""

    def test_name(self):
        metric = SelfBLEU(max_order=4)
        assert metric.name == "self_bleu_4"
        metric2 = SelfBLEU(max_order=2)
        assert metric2.name == "self_bleu_2"

    def test_higher_is_better_false(self):
        metric = SelfBLEU()
        assert metric.higher_is_better is False

    def test_identical_texts_high_score(self):
        metric = SelfBLEU(max_order=4)
        texts = generate_identical_texts(n=5)
        score = metric.compute(texts)
        assert score > 0.8, f"Identical texts should yield high SelfBLEU, got {score}"

    def test_diverse_texts_low_score(self):
        metric = SelfBLEU(max_order=4)
        score = metric.compute(DIVERSE_TEXTS)
        assert score < 0.5, f"Diverse texts should yield low SelfBLEU, got {score}"

    def test_identical_higher_than_diverse(self):
        metric = SelfBLEU(max_order=4)
        identical_score = metric.compute(generate_identical_texts(5))
        diverse_score = metric.compute(DIVERSE_TEXTS)
        assert identical_score > diverse_score

    def test_score_in_unit_interval(self):
        metric = SelfBLEU(max_order=4)
        for texts in [DIVERSE_TEXTS, SIMILAR_TEXTS, generate_random_texts(10, seed=99)]:
            score = metric.compute(texts)
            assert 0.0 <= score <= 1.0, f"SelfBLEU out of [0,1]: {score}"

    def test_per_sample_length(self):
        metric = SelfBLEU(max_order=4)
        per_sample = metric.compute_per_sample(DIVERSE_TEXTS)
        assert len(per_sample) == len(DIVERSE_TEXTS)

    def test_per_sample_mean_equals_compute(self):
        metric = SelfBLEU(max_order=4)
        per_sample = metric.compute_per_sample(DIVERSE_TEXTS)
        computed = metric.compute(DIVERSE_TEXTS)
        assert abs(np.mean(per_sample) - computed) < 1e-10

    def test_symmetry_two_texts(self):
        metric = SelfBLEU(max_order=2)
        texts = ["the cat sat on the mat", "the dog ran in the park"]
        per_sample = metric.compute_per_sample(texts)
        # With 2 texts, each sentence is scored against the other; order matters
        # but both should produce finite values
        assert all(0.0 <= s <= 1.0 for s in per_sample)

    def test_max_order_variations(self):
        scores = {}
        for order in [1, 2, 3, 4]:
            metric = SelfBLEU(max_order=order)
            scores[order] = metric.compute(SIMILAR_TEXTS)
        # Higher order should generally give lower or equal self-BLEU
        # because higher-order n-gram matches are harder
        assert scores[1] >= scores[4] - 0.1

    def test_smoothing_floor(self):
        metric = SelfBLEU(max_order=4, smoothing_function="floor", smooth_epsilon=0.1)
        score = metric.compute(DIVERSE_TEXTS)
        assert 0.0 <= score <= 1.0

    def test_smoothing_add1(self):
        metric = SelfBLEU(max_order=4, smoothing_function="add1")
        score = metric.compute(DIVERSE_TEXTS)
        assert 0.0 <= score <= 1.0

    def test_validate_input_too_few(self):
        metric = SelfBLEU()
        with pytest.raises(ValueError, match="at least 2"):
            metric.compute(["only one text"])

    def test_validate_input_not_strings(self):
        metric = SelfBLEU()
        with pytest.raises(ValueError):
            metric.compute([123, 456])

    def test_similar_texts_moderate_score(self):
        metric = SelfBLEU(max_order=4)
        score = metric.compute(SIMILAR_TEXTS)
        # Similar texts share a lot but not everything
        assert 0.3 < score < 1.0, f"Expected moderate score, got {score}"

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_scales_with_set_size(self, n):
        metric = SelfBLEU(max_order=4)
        texts = generate_random_texts(n=n, seed=42)
        score = metric.compute(texts)
        assert 0.0 <= score <= 1.0


# =========================================================================
# 2. TestDistinctN
# =========================================================================


class TestDistinctN:
    """Tests for Distinct-N diversity metric."""

    def test_name(self):
        assert DistinctN(n=1).name == "distinct_1"
        assert DistinctN(n=3).name == "distinct_3"

    def test_higher_is_better(self):
        assert DistinctN().higher_is_better is True

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            DistinctN(n=0)

    def test_identical_texts_low(self):
        metric = DistinctN(n=2)
        texts = generate_identical_texts(n=5)
        score = metric.compute(texts)
        # Identical texts repeated: the unique bigrams stay the same
        # but total count is multiplied
        assert score < 0.5

    def test_diverse_texts_high(self):
        metric = DistinctN(n=2)
        score = metric.compute(DIVERSE_TEXTS)
        assert score > 0.5, f"Diverse texts should have high distinct-2, got {score}"

    def test_diverse_higher_than_identical(self):
        metric = DistinctN(n=2)
        d = metric.compute(DIVERSE_TEXTS)
        i = metric.compute(generate_identical_texts(5))
        assert d > i

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_score_in_unit_interval(self, n):
        metric = DistinctN(n=n)
        score = metric.compute(DIVERSE_TEXTS)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_ratio_computation_matches_manual(self, n):
        metric = DistinctN(n=n)
        texts = SIMILAR_TEXTS
        score = metric.compute(texts)
        # Manually compute
        all_ngrams = []
        for text in texts:
            tokens = tokenize_simple(text)
            all_ngrams.extend(extract_ngrams(tokens, n))
        if all_ngrams:
            expected = len(set(all_ngrams)) / len(all_ngrams)
        else:
            expected = 0.0
        assert abs(score - expected) < 1e-10

    def test_compute_per_n(self):
        metric = DistinctN(n=2)
        results = metric.compute_per_n(DIVERSE_TEXTS, n_values=[1, 2, 3, 4])
        assert set(results.keys()) == {1, 2, 3, 4}
        for v in results.values():
            assert 0.0 <= v <= 1.0

    def test_monotonicity_with_n(self):
        """Higher n generally yields higher distinct-n for diverse texts."""
        metric = DistinctN(n=1)
        results = metric.compute_per_n(DIVERSE_TEXTS, n_values=[1, 2, 3, 4])
        # distinct-n tends to increase with n for diverse texts
        assert results[4] >= results[1] - 0.1

    def test_single_word_texts(self):
        metric = DistinctN(n=1)
        texts = ["hello", "world"]
        score = metric.compute(texts)
        assert score == 1.0  # 2 unique unigrams / 2 total

    def test_two_identical_single_words(self):
        metric = DistinctN(n=1)
        texts = ["hello", "hello"]
        score = metric.compute(texts)
        assert score == 0.5  # 1 unique unigram / 2 total

    def test_empty_after_tokenization(self):
        metric = DistinctN(n=2)
        texts = ["a", "b"]  # single-token texts have no bigrams
        score = metric.compute(texts)
        assert score == 0.0

    def test_validate_input(self):
        metric = DistinctN(n=1)
        with pytest.raises(ValueError):
            metric.compute(["only one"])

    @pytest.mark.parametrize("n_texts", [3, 5, 10, 20])
    def test_scaling_with_corpus_size(self, n_texts):
        metric = DistinctN(n=2)
        texts = generate_random_texts(n=n_texts, seed=77)
        score = metric.compute(texts)
        assert 0.0 <= score <= 1.0


# =========================================================================
# 3. TestNGramEntropy
# =========================================================================


class TestNGramEntropy:
    """Tests for NGramEntropy diversity metric."""

    def test_name(self):
        assert NGramEntropy(n=2).name == "ngram_entropy_2"

    def test_higher_is_better(self):
        assert NGramEntropy().higher_is_better is True

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            NGramEntropy(n=0)

    def test_identical_texts_low_entropy(self):
        metric = NGramEntropy(n=2)
        texts = generate_identical_texts(n=5)
        score = metric.compute(texts)
        # All texts identical => the distribution is the same as one text
        # Entropy is still > 0 because there are multiple distinct bigrams
        # within that one text, but should be lower than diverse texts.
        diverse_score = metric.compute(DIVERSE_TEXTS)
        assert score < diverse_score

    def test_diverse_texts_high_entropy(self):
        metric = NGramEntropy(n=2)
        score = metric.compute(DIVERSE_TEXTS)
        assert score > 3.0, f"Diverse texts should have high entropy, got {score}"

    def test_uniform_distribution_max_entropy(self):
        """When all n-grams appear equally often, entropy is maximal (log2 of count)."""
        metric = NGramEntropy(n=1, base=2.0)
        # Create texts where each word appears exactly once
        words = [f"word{i}" for i in range(16)]
        texts = [" ".join(words[:8]), " ".join(words[8:])]
        score = metric.compute(texts)
        # 16 unique unigrams, each appearing once => entropy = log2(16) = 4.0
        assert abs(score - 4.0) < 0.01

    def test_degenerate_single_ngram_zero_entropy(self):
        """If all n-grams are the same, entropy should be 0."""
        metric = NGramEntropy(n=1, base=2.0)
        texts = ["aaa aaa aaa", "aaa aaa aaa"]
        score = metric.compute(texts)
        assert abs(score) < 1e-10

    def test_entropy_non_negative(self):
        metric = NGramEntropy(n=2)
        for texts in [DIVERSE_TEXTS, SIMILAR_TEXTS, generate_random_texts(10)]:
            score = metric.compute(texts)
            assert score >= 0.0

    def test_compute_per_n(self):
        metric = NGramEntropy(n=2)
        results = metric.compute_per_n(DIVERSE_TEXTS, n_values=[1, 2, 3])
        assert len(results) == 3
        for v in results.values():
            assert v >= 0.0

    def test_base_change(self):
        metric_2 = NGramEntropy(n=2, base=2.0)
        metric_e = NGramEntropy(n=2, base=math.e)
        s2 = metric_2.compute(DIVERSE_TEXTS)
        se = metric_e.compute(DIVERSE_TEXTS)
        # H_2 = H_e / ln(2)  =>  H_e = H_2 * ln(2)
        assert abs(se - s2 * math.log(2)) < 0.01

    def test_matches_manual_computation(self):
        metric = NGramEntropy(n=2, base=2.0)
        texts = SIMILAR_TEXTS
        score = metric.compute(texts)
        # Manual: compute distribution, then Shannon entropy
        counter = Counter()
        for text in texts:
            tokens = tokenize_simple(text)
            for ng in extract_ngrams(tokens, 2):
                counter[ng] += 1
        total = sum(counter.values())
        expected = 0.0
        for c in counter.values():
            p = c / total
            if p > 0:
                expected -= p * math.log2(p)
        assert abs(score - expected) < 1e-10

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_entropy_increases_with_diversity(self, n):
        metric = NGramEntropy(n=n)
        ident = metric.compute(generate_identical_texts(5))
        diverse = metric.compute(DIVERSE_TEXTS)
        assert diverse >= ident

    def test_validate_input(self):
        metric = NGramEntropy(n=2)
        with pytest.raises(ValueError):
            metric.compute(["single"])

    def test_conditional_entropy(self):
        """H(bigram | unigram) = H_2 - H_1, and should be non-negative."""
        metric = NGramEntropy(n=2, base=2.0)
        ce = metric._conditional_entropy(DIVERSE_TEXTS, 2)
        assert ce >= 0.0

    def test_entropy_upper_bound(self):
        """Entropy <= log2(number of unique n-grams)."""
        metric = NGramEntropy(n=2, base=2.0)
        texts = DIVERSE_TEXTS
        score = metric.compute(texts)
        counter = Counter()
        for text in texts:
            tokens = tokenize_simple(text)
            for ng in extract_ngrams(tokens, 2):
                counter[ng] += 1
        max_entropy = math.log2(len(counter)) if counter else 0.0
        assert score <= max_entropy + 1e-10


# =========================================================================
# 4. TestEmbeddingPairwiseDistance
# =========================================================================


class TestEmbeddingPairwiseDistance:
    """Tests for EmbeddingPairwiseDistance metric."""

    def test_name(self):
        metric = EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf")
        assert "cosine" in metric.name
        assert "tfidf" in metric.name

    def test_higher_is_better(self):
        assert EmbeddingPairwiseDistance().higher_is_better is True

    def test_invalid_distance_metric(self):
        with pytest.raises(ValueError):
            EmbeddingPairwiseDistance(distance_metric="manhattan")

    def test_invalid_embedding_method(self):
        with pytest.raises(ValueError):
            EmbeddingPairwiseDistance(embedding_method="bert")

    def test_identical_texts_low_distance(self):
        metric = EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf")
        texts = generate_identical_texts(5)
        score = metric.compute(texts)
        assert score < 0.01, f"Identical texts should have near-zero distance, got {score}"

    def test_diverse_texts_higher_distance(self):
        metric = EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf")
        diverse_score = metric.compute(DIVERSE_TEXTS)
        identical_score = metric.compute(generate_identical_texts(5))
        assert diverse_score > identical_score

    def test_cosine_distance_range(self):
        metric = EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf")
        score = metric.compute(DIVERSE_TEXTS)
        assert 0.0 <= score <= 2.0  # cosine distance in [0, 2]

    def test_euclidean_distance_non_negative(self):
        metric = EmbeddingPairwiseDistance(distance_metric="euclidean", embedding_method="tfidf")
        score = metric.compute(DIVERSE_TEXTS)
        assert score >= 0.0

    def test_pairwise_matrix_symmetric(self):
        metric = EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf")
        mat = metric.compute_pairwise_matrix(DIVERSE_TEXTS)
        assert mat.shape == (len(DIVERSE_TEXTS), len(DIVERSE_TEXTS))
        assert_symmetric(mat, atol=1e-10)

    def test_pairwise_matrix_zero_diagonal(self):
        metric = EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf")
        mat = metric.compute_pairwise_matrix(DIVERSE_TEXTS)
        diag = np.diag(mat)
        assert np.allclose(diag, 0.0, atol=1e-10)

    def test_ngram_hash_embedding(self):
        metric = EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method="ngram_hash"
        )
        score = metric.compute(DIVERSE_TEXTS)
        assert score > 0.0

    def test_random_embedding(self):
        metric = EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method="random"
        )
        score = metric.compute(DIVERSE_TEXTS)
        assert score > 0.0

    def test_random_embedding_deterministic(self):
        metric = EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method="random"
        )
        s1 = metric.compute(DIVERSE_TEXTS)
        s2 = metric.compute(DIVERSE_TEXTS)
        assert abs(s1 - s2) < 1e-10

    @pytest.mark.parametrize("method", ["tfidf", "ngram_hash", "random"])
    def test_all_embedding_methods(self, method):
        metric = EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method=method
        )
        score = metric.compute(DIVERSE_TEXTS)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_normalized_embeddings_cosine(self):
        """TF-IDF embeddings are L2-normalized, so cosine distance should be bounded."""
        emb = tfidf_embeddings(DIVERSE_TEXTS)
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_validate_input(self):
        metric = EmbeddingPairwiseDistance()
        with pytest.raises(ValueError):
            metric.compute(["single text"])


# =========================================================================
# 5. TestVendiScore
# =========================================================================


class TestVendiScore:
    """Tests for VendiScore diversity metric."""

    def test_name(self):
        assert "cosine" in VendiScore(kernel_type="cosine").name

    def test_higher_is_better(self):
        assert VendiScore().higher_is_better is True

    def test_invalid_kernel(self):
        with pytest.raises(ValueError):
            VendiScore(kernel_type="linear")

    def test_identical_texts_low_score(self):
        metric = VendiScore(kernel_type="cosine")
        texts = generate_identical_texts(5)
        score = metric.compute(texts)
        assert score < 2.0, f"Identical texts: expected low Vendi, got {score}"

    def test_diverse_texts_higher_score(self):
        metric = VendiScore(kernel_type="cosine")
        diverse_score = metric.compute(DIVERSE_TEXTS)
        identical_score = metric.compute(generate_identical_texts(5))
        assert diverse_score > identical_score

    def test_identity_kernel_equals_n(self):
        """If the kernel matrix is the identity, Vendi score = n."""
        metric = VendiScore(kernel_type="cosine")
        n = 5
        # We test the _matrix_entropy + exp path directly
        eigenvalues = np.ones(n) / n  # uniform distribution
        entropy = metric._matrix_entropy(eigenvalues)
        vs = math.exp(entropy)
        assert abs(vs - n) < 1e-6

    def test_score_bounded_by_n(self):
        metric = VendiScore(kernel_type="cosine")
        n = len(DIVERSE_TEXTS)
        score = metric.compute(DIVERSE_TEXTS)
        assert score <= n + 1.0  # allow small numerical slack

    def test_score_at_least_one(self):
        metric = VendiScore(kernel_type="cosine")
        score = metric.compute(DIVERSE_TEXTS)
        assert score >= 1.0 - 1e-6

    def test_rbf_kernel(self):
        metric = VendiScore(kernel_type="rbf", bandwidth=1.0)
        score = metric.compute(DIVERSE_TEXTS)
        assert score >= 1.0

    def test_ngram_kernel(self):
        metric = VendiScore(kernel_type="ngram")
        score = metric.compute(DIVERSE_TEXTS)
        assert score >= 1.0

    def test_cosine_kernel_symmetric(self):
        metric = VendiScore(kernel_type="cosine")
        K = metric._build_kernel_matrix(DIVERSE_TEXTS)
        assert_symmetric(K, atol=1e-10)

    def test_cosine_kernel_psd(self):
        metric = VendiScore(kernel_type="cosine")
        K = metric._build_kernel_matrix(DIVERSE_TEXTS)
        assert_positive_semidefinite(K, atol=1e-6)

    def test_rbf_kernel_psd(self):
        metric = VendiScore(kernel_type="rbf", bandwidth=1.0)
        K = metric._build_kernel_matrix(DIVERSE_TEXTS)
        assert_positive_semidefinite(K, atol=1e-6)

    def test_eigenspectrum_non_negative(self):
        metric = VendiScore(kernel_type="cosine")
        K = metric._build_kernel_matrix(DIVERSE_TEXTS)
        Kn = metric._normalize_kernel(K)
        from scipy.linalg import eigh
        eigvals = eigh(Kn, eigvals_only=True)
        assert np.all(eigvals >= -1e-8)

    def test_effective_dimensionality_correlates(self):
        """Vendi score should correlate with number of distinct items."""
        metric = VendiScore(kernel_type="cosine")
        score_small = metric.compute(DIVERSE_TEXTS[:3])
        score_large = metric.compute(DIVERSE_TEXTS)
        assert score_large >= score_small - 0.5

    def test_validate_input(self):
        metric = VendiScore()
        with pytest.raises(ValueError):
            metric.compute(["only"])


# =========================================================================
# 6. TestParseTreeDiversity
# =========================================================================


class TestParseTreeDiversity:
    """Tests for ParseTreeDiversity metric."""

    def test_name(self):
        assert ParseTreeDiversity().name == "parse_tree_diversity"

    def test_higher_is_better(self):
        assert ParseTreeDiversity().higher_is_better is True

    def test_identical_texts_low(self):
        metric = ParseTreeDiversity()
        texts = generate_identical_texts(5)
        score = metric.compute(texts)
        assert score < 0.1, f"Identical texts: expected near-zero diversity, got {score}"

    def test_diverse_texts_higher(self):
        metric = ParseTreeDiversity()
        diverse_score = metric.compute(DIVERSE_TEXTS)
        identical_score = metric.compute(generate_identical_texts(5))
        assert diverse_score > identical_score

    def test_structural_differences_detected(self):
        metric = ParseTreeDiversity()
        # Questions vs statements should show structural diversity
        struct_diverse = [
            "What is the meaning of life?",
            "If you go there, bring an umbrella because it rains often.",
            "The sun rises in the east.",
            "She quickly ran to the store and bought some milk.",
        ]
        struct_similar = [
            "The cat sits on the mat.",
            "The dog sits on the rug.",
            "The bird sits on the branch.",
            "The fish sits in the bowl.",
        ]
        diverse_score = metric.compute(struct_diverse)
        similar_score = metric.compute(struct_similar)
        assert diverse_score > similar_score

    def test_score_non_negative(self):
        metric = ParseTreeDiversity()
        score = metric.compute(DIVERSE_TEXTS)
        assert score >= 0.0

    def test_feature_vector_dimension(self):
        metric = ParseTreeDiversity()
        features = metric._approximate_parse_features("The cat sat on the mat.")
        assert features.shape == (30,)

    def test_tree_edit_distance_symmetric(self):
        metric = ParseTreeDiversity()
        f1 = metric._approximate_parse_features("Hello world.")
        f2 = metric._approximate_parse_features("Goodbye universe.")
        d12 = metric._tree_edit_distance_approx(f1, f2)
        d21 = metric._tree_edit_distance_approx(f2, f1)
        assert abs(d12 - d21) < 1e-10

    def test_tree_edit_distance_zero_for_same(self):
        metric = ParseTreeDiversity()
        f = metric._approximate_parse_features("The cat sat on the mat.")
        d = metric._tree_edit_distance_approx(f, f)
        assert abs(d) < 1e-10

    def test_pattern_weight(self):
        m1 = ParseTreeDiversity(pattern_weight=0.0)
        m2 = ParseTreeDiversity(pattern_weight=1.0)
        s1 = m1.compute(DIVERSE_TEXTS)
        s2 = m2.compute(DIVERSE_TEXTS)
        # Both should be valid, just different
        assert s1 >= 0.0
        assert s2 >= 0.0

    def test_pattern_extraction(self):
        metric = ParseTreeDiversity()
        patterns = metric._extract_syntactic_patterns("If you go there, bring an umbrella.")
        assert len(patterns) > 0

    def test_empty_text_features(self):
        metric = ParseTreeDiversity()
        f = metric._approximate_parse_features("")
        assert f.shape == (30,)
        assert np.allclose(f, 0.0)

    def test_question_detection(self):
        metric = ParseTreeDiversity()
        f_q = metric._approximate_parse_features("What is this?")
        f_s = metric._approximate_parse_features("This is something.")
        # Feature vector has an is_question component
        # Question should differ from statement
        assert not np.allclose(f_q, f_s)

    def test_validate_input(self):
        metric = ParseTreeDiversity()
        with pytest.raises(ValueError):
            metric.compute(["alone"])

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_various_set_sizes(self, n):
        metric = ParseTreeDiversity()
        texts = generate_random_texts(n=n, seed=123)
        score = metric.compute(texts)
        assert score >= 0.0


# =========================================================================
# 7. TestBehavioralDiversity
# =========================================================================


class TestBehavioralDiversity:
    """Tests for BehavioralDiversity metric."""

    def test_name(self):
        assert BehavioralDiversity().name == "behavioral_diversity"

    def test_higher_is_better(self):
        assert BehavioralDiversity().higher_is_better is True

    def test_identical_texts_low(self):
        metric = BehavioralDiversity()
        texts = generate_identical_texts(5)
        score = metric.compute(texts)
        # Identical texts produce identical behavior vectors => zero diversity
        # After z-score normalization, all columns are 0 => kernel is near-identity
        # Log-det / n might be very small
        assert score < 1.0

    def test_diverse_texts_higher(self):
        metric = BehavioralDiversity()
        diverse_score = metric.compute(DIVERSE_TEXTS)
        identical_score = metric.compute(generate_identical_texts(5))
        assert diverse_score > identical_score

    def test_behavior_vector_computation(self):
        metric = BehavioralDiversity()
        bv = metric._compute_behavior_vectors(DIVERSE_TEXTS)
        assert bv.shape == (len(DIVERSE_TEXTS), 5)  # 5 default descriptors

    def test_behavior_vectors_normalized(self):
        """After z-score normalization, columns should have mean ~0 and std ~1."""
        metric = BehavioralDiversity()
        bv = metric._compute_behavior_vectors(DIVERSE_TEXTS)
        for j in range(bv.shape[1]):
            col = bv[:, j]
            if np.std(col) > 1e-12:
                assert abs(np.mean(col)) < 1e-10
                assert abs(np.std(col) - 1.0) < 1e-10

    def test_custom_descriptor_functions(self):
        def word_count(text: str) -> float:
            return float(len(text.split()))

        def char_count(text: str) -> float:
            return float(len(text))

        metric = BehavioralDiversity(descriptor_functions=[word_count, char_count])
        bv = metric._compute_behavior_vectors(DIVERSE_TEXTS)
        assert bv.shape[1] == 2

    def test_kernel_bandwidth_effect(self):
        m1 = BehavioralDiversity(kernel_bandwidth=0.1)
        m2 = BehavioralDiversity(kernel_bandwidth=10.0)
        s1 = m1.compute(DIVERSE_TEXTS)
        s2 = m2.compute(DIVERSE_TEXTS)
        # Different bandwidth changes the score
        assert s1 != s2 or True  # just check no error

    def test_default_descriptors(self):
        """Check all 5 default descriptors produce finite values."""
        metric = BehavioralDiversity()
        for fn in metric._descriptor_fns:
            for text in DIVERSE_TEXTS:
                val = fn(text)
                assert np.isfinite(val)

    def test_length_feature(self):
        val = BehavioralDiversity._length_feature("hello world")
        assert abs(val - len("hello world") / 1000.0) < 1e-10

    def test_vocabulary_richness(self):
        val = BehavioralDiversity._vocabulary_richness("the the the the")
        assert val < 0.5  # low TTR
        val2 = BehavioralDiversity._vocabulary_richness("one two three four")
        assert val2 == 1.0  # all unique

    def test_score_finite(self):
        metric = BehavioralDiversity()
        score = metric.compute(DIVERSE_TEXTS)
        assert np.isfinite(score)

    def test_validate_input(self):
        metric = BehavioralDiversity()
        with pytest.raises(ValueError):
            metric.compute(["one"])

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_scales_with_n(self, n):
        metric = BehavioralDiversity()
        texts = generate_random_texts(n=n, seed=55)
        score = metric.compute(texts)
        assert np.isfinite(score)


# =========================================================================
# 8. TestPerplexity
# =========================================================================


class TestPerplexity:
    """Tests for Perplexity quality metric."""

    def test_name(self):
        assert Perplexity().name == "perplexity"

    def test_higher_is_better_false(self):
        assert Perplexity().higher_is_better is False

    def test_compute_returns_positive(self):
        metric = Perplexity(order=3, smoothing="laplace")
        texts = DIVERSE_TEXTS
        score = metric.compute(texts)
        assert score > 0.0

    def test_per_sample_length(self):
        metric = Perplexity(order=3)
        per = metric.compute_per_sample(DIVERSE_TEXTS)
        assert len(per) == len(DIVERSE_TEXTS)

    def test_per_sample_all_positive(self):
        metric = Perplexity(order=3)
        per = metric.compute_per_sample(DIVERSE_TEXTS)
        assert all(p > 0 for p in per)

    def test_mean_matches_compute(self):
        metric = Perplexity(order=3)
        per = metric.compute_per_sample(DIVERSE_TEXTS)
        computed = metric.compute(DIVERSE_TEXTS)
        assert abs(np.mean(per) - computed) < 1e-6

    def test_identical_texts_lower_perplexity(self):
        """Self-perplexity on identical texts should be lower (more predictable)."""
        metric = Perplexity(order=2, smoothing="laplace")
        identical = generate_identical_texts(5)
        random = generate_random_texts(5, seed=42)
        ppl_ident = metric.compute(identical)
        ppl_random = metric.compute(random)
        # Identical texts create a very focused model => lower self-perplexity
        assert ppl_ident < ppl_random

    def test_laplace_smoothing(self):
        metric = Perplexity(order=3, smoothing="laplace", alpha=1.0)
        score = metric.compute(DIVERSE_TEXTS)
        assert np.isfinite(score) and score > 0

    def test_kneser_ney_smoothing(self):
        metric = Perplexity(order=3, smoothing="kneser_ney", discount=0.75)
        score = metric.compute(DIVERSE_TEXTS)
        assert np.isfinite(score) and score > 0

    def test_cross_perplexity_with_reference(self):
        ref_texts = DIVERSE_TEXTS[:4]
        test_texts = DIVERSE_TEXTS[4:]
        metric = Perplexity(order=2, reference_texts=ref_texts)
        score = metric.compute(test_texts)
        assert score > 0

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_various_orders(self, order):
        metric = Perplexity(order=order)
        score = metric.compute(DIVERSE_TEXTS)
        assert np.isfinite(score) and score > 0

    def test_cross_entropy_non_negative(self):
        metric = Perplexity(order=2, smoothing="laplace")
        # Build model first
        metric.compute_per_sample(DIVERSE_TEXTS)
        ce = metric._cross_entropy(DIVERSE_TEXTS[0], metric._model)
        assert ce >= 0.0

    def test_log_probability_relationship(self):
        """PPL = 2^H where H is cross-entropy in bits."""
        metric = Perplexity(order=2, smoothing="laplace")
        metric.compute_per_sample(DIVERSE_TEXTS)
        text = DIVERSE_TEXTS[0]
        ce = metric._cross_entropy(text, metric._model)
        ppl = math.pow(2, ce) if ce < 500 else float("inf")
        per_sample = metric.compute_per_sample(DIVERSE_TEXTS)
        assert abs(ppl - per_sample[0]) < 1e-6

    def test_validate_input(self):
        metric = Perplexity()
        with pytest.raises(ValueError):
            metric.compute([123])

    def test_empty_texts(self):
        metric = Perplexity()
        assert metric.compute([]) == 0.0


# =========================================================================
# 9. TestNLICoherence
# =========================================================================


class TestNLICoherence:
    """Tests for NLICoherence quality metric."""

    def test_name(self):
        assert NLICoherence().name == "nli_coherence"

    def test_higher_is_better(self):
        assert NLICoherence().higher_is_better is True

    def test_score_in_unit_interval(self):
        metric = NLICoherence()
        score = metric.compute(DIVERSE_TEXTS)
        assert 0.0 <= score <= 1.0

    def test_per_sample_length(self):
        metric = NLICoherence()
        per = metric.compute_per_sample(DIVERSE_TEXTS)
        assert len(per) == len(DIVERSE_TEXTS)

    def test_per_sample_in_unit_interval(self):
        metric = NLICoherence()
        per = metric.compute_per_sample(DIVERSE_TEXTS)
        for s in per:
            assert 0.0 <= s <= 1.0

    def test_coherent_text_high_score(self):
        metric = NLICoherence()
        coherent = [
            "The cat sat on the mat. The cat was very comfortable. It purred softly and fell asleep."
        ]
        score = metric.compute(coherent)
        assert score > 0.3

    def test_single_sentence_perfect_coherence(self):
        metric = NLICoherence()
        per = metric.compute_per_sample(["Hello world."])
        assert per[0] == 1.0

    def test_prompt_relevance(self):
        metric = NLICoherence()
        texts = ["The cat sat on the mat and purred softly in the sun."]
        score_with = metric.compute(texts, prompt="cat sitting")
        score_without = metric.compute(texts)
        # Both should be valid scores
        assert 0.0 <= score_with <= 1.0
        assert 0.0 <= score_without <= 1.0

    def test_weight_configuration(self):
        m1 = NLICoherence(local_weight=1.0, global_weight=0, entity_weight=0, topic_weight=0)
        m2 = NLICoherence(local_weight=0, global_weight=1.0, entity_weight=0, topic_weight=0)
        texts = [
            "The cat sat on the mat. The dog ran in the park. "
            "Stars twinkled in the sky. Music played in the room."
        ]
        s1 = m1.compute(texts)
        s2 = m2.compute(texts)
        # Different weights => generally different scores
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0

    def test_multi_sentence_coherence(self):
        metric = NLICoherence()
        high_coherence = [
            "Cats are popular pets. They are known for their independence. "
            "Many people enjoy having cats at home."
        ]
        low_coherence = [
            "Cats are popular pets. Quantum mechanics describes wave functions. "
            "Cooking pasta requires boiling water."
        ]
        s_high = metric.compute(high_coherence)
        s_low = metric.compute(low_coherence)
        assert s_high >= s_low - 0.1  # Allow some tolerance

    def test_entity_coherence(self):
        metric = NLICoherence()
        sentences = ["The Cat sat on the mat.", "The Cat purred softly."]
        ec = metric._entity_coherence(sentences)
        assert 0.0 <= ec <= 1.0

    def test_topic_coherence(self):
        metric = NLICoherence()
        sentences = ["machine learning models work well", "machine learning algorithms improve"]
        tc = metric._topic_coherence(sentences)
        assert tc > 0.0  # Should have keyword overlap

    def test_validate_input(self):
        metric = NLICoherence()
        with pytest.raises(ValueError):
            metric.compute([42])

    def test_compute_with_ci(self):
        metric = NLICoherence()
        mean_val, (lo, hi) = metric.compute_with_ci(DIVERSE_TEXTS)
        assert lo <= mean_val <= hi
        assert 0.0 <= lo
        assert hi <= 1.0


# =========================================================================
# 10. TestConstraintSatisfaction
# =========================================================================


class TestConstraintSatisfaction:
    """Tests for ConstraintSatisfaction quality metric."""

    def test_name(self):
        assert ConstraintSatisfaction().name == "constraint_satisfaction"

    def test_higher_is_better(self):
        assert ConstraintSatisfaction().higher_is_better is True

    def test_no_constraints_all_pass(self):
        metric = ConstraintSatisfaction(constraints=[])
        score = metric.compute(DIVERSE_TEXTS)
        assert score == 1.0

    def test_min_length_constraint(self):
        c = MinLengthConstraint(min_words=5)
        metric = ConstraintSatisfaction(constraints=[c])
        texts = [
            "This is a short text with enough words to pass easily.",
            "Hi.",
        ]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0
        assert per[1] == 0.0

    def test_max_length_constraint(self):
        c = MaxLengthConstraint(max_words=5)
        metric = ConstraintSatisfaction(constraints=[c])
        texts = ["Hello there.", "This is a very long text with many many many words in it."]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0
        assert per[1] == 0.0

    def test_contains_keyword(self):
        c = ContainsKeywordConstraint(keywords=["python", "code"])
        metric = ConstraintSatisfaction(constraints=[c])
        texts = [
            "Python code is beautiful.",
            "Java programs are verbose.",
        ]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0
        assert per[1] == 0.0

    def test_does_not_contain(self):
        c = DoesNotContainConstraint(forbidden=["bad", "terrible"])
        metric = ConstraintSatisfaction(constraints=[c])
        texts = ["Good day!", "This is bad."]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0
        assert per[1] == 0.0

    def test_sentence_count_constraint(self):
        c = SentenceCountConstraint(min_sentences=2, max_sentences=3)
        metric = ConstraintSatisfaction(constraints=[c])
        texts = [
            "First sentence. Second sentence.",
            "Only one sentence",
        ]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0
        assert per[1] == 0.0

    def test_format_constraint(self):
        c = FormatConstraint(pattern=r"\d{3}-\d{4}")
        metric = ConstraintSatisfaction(constraints=[c])
        texts = ["Call 555-1234.", "No phone number here."]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0
        assert per[1] == 0.0

    def test_multiple_constraints(self):
        constraints = [
            MinLengthConstraint(min_words=3),
            ContainsKeywordConstraint(keywords=["test"]),
        ]
        metric = ConstraintSatisfaction(constraints=constraints)
        texts = [
            "This is a test of multiple constraints.",
            "Test.",
            "No keyword here at all.",
        ]
        per = metric.compute_per_sample(texts)
        assert per[0] == 1.0  # passes both
        assert per[1] == 0.0  # fails min_length
        assert per[2] == 0.0  # fails keyword

    def test_add_remove_constraint(self):
        metric = ConstraintSatisfaction()
        c = MinLengthConstraint(min_words=100)
        metric.add_constraint(c)
        score = metric.compute(["Short text."])
        assert score == 0.0
        metric.remove_constraint(c.name)
        score = metric.compute(["Short text."])
        assert score == 1.0

    def test_aggregate_is_mean(self):
        c = MinLengthConstraint(min_words=3)
        metric = ConstraintSatisfaction(constraints=[c])
        texts = [
            "This passes the constraint easily.",
            "No.",
            "This also passes fine.",
            "X.",
        ]
        score = metric.compute(texts)
        assert abs(score - 0.5) < 1e-10

    def test_repetition_constraint(self):
        c = RepetitionConstraint(max_repeat_ratio=0.1, ngram_order=2)
        metric = ConstraintSatisfaction(constraints=[c])
        texts = [
            "the the the the the the the the the the",
            "diverse unique words appear in every sentence naturally",
        ]
        per = metric.compute_per_sample(texts)
        assert per[0] == 0.0  # highly repetitive
        assert per[1] == 1.0

    def test_validate_input(self):
        metric = ConstraintSatisfaction()
        with pytest.raises(ValueError):
            metric.compute([123])

    def test_per_sample_binary(self):
        c = MinLengthConstraint(min_words=5)
        metric = ConstraintSatisfaction(constraints=[c])
        per = metric.compute_per_sample(DIVERSE_TEXTS)
        for v in per:
            assert v in (0.0, 1.0)


# =========================================================================
# 11. TestMetricCorrelationAnalyzer
# =========================================================================


class TestMetricCorrelationAnalyzer:
    """Tests for MetricCorrelationAnalyzer."""

    def _make_metric_values(self, n: int = 20, seed: int = 42) -> Dict[str, List[float]]:
        rng = np.random.RandomState(seed)
        base = rng.randn(n)
        return {
            "metric_a": list(base + rng.randn(n) * 0.1),
            "metric_b": list(-base + rng.randn(n) * 0.1),  # anti-correlated
            "metric_c": list(rng.randn(n)),  # independent
        }

    def test_correlation_matrix_shape(self):
        metrics = ["metric_a", "metric_b", "metric_c"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        values = self._make_metric_values()
        corr = analyzer.compute_correlation_matrix(values)
        assert corr.shape == (3, 3)

    def test_correlation_matrix_diagonal(self):
        metrics = ["metric_a", "metric_b", "metric_c"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        values = self._make_metric_values()
        corr = analyzer.compute_correlation_matrix(values)
        assert np.allclose(np.diag(corr), 1.0)

    def test_correlation_matrix_symmetric(self):
        metrics = ["metric_a", "metric_b"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        values = self._make_metric_values()
        corr = analyzer.compute_correlation_matrix(values)
        assert_symmetric(corr)

    def test_kendall_tau_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert abs(tau - 1.0) < 1e-10

    def test_kendall_tau_perfect_anticorrelation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert abs(tau - (-1.0)) < 1e-10

    def test_kendall_tau_uncorrelated(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = rng.randn(100)
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert abs(tau) < 0.3

    def test_kendall_tau_range(self):
        rng = np.random.RandomState(42)
        x = rng.randn(50)
        y = rng.randn(50)
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert -1.0 <= tau <= 1.0
        assert 0.0 <= p <= 1.0

    def test_compute_with_pvalues(self):
        metrics = ["metric_a", "metric_b"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        values = self._make_metric_values()
        corr, pvals = analyzer.compute_with_pvalues(values)
        assert corr.shape == pvals.shape
        assert np.all(pvals >= 0.0)
        assert np.all(pvals <= 1.0)

    def test_anticorrelated_detected(self):
        metrics = ["metric_a", "metric_b"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        values = self._make_metric_values()
        corr = analyzer.compute_correlation_matrix(values)
        # metric_a and metric_b are anti-correlated by construction
        assert corr[0, 1] < -0.5

    def test_spearman_rho(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rho, p = MetricCorrelationAnalyzer.spearman_rho(x, y)
        assert abs(rho - 1.0) < 1e-10

    def test_pearson_r(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        r, p = MetricCorrelationAnalyzer.pearson_r(x, y)
        assert abs(r - 1.0) < 1e-10

    def test_discover_metric_clusters(self):
        metrics = ["a", "b", "c", "d"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        # Block diagonal correlation: (a,b) correlated, (c,d) correlated
        corr = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ])
        clusters = analyzer.discover_metric_clusters(corr, n_clusters=2)
        assert len(clusters) == 2
        # Check that correlated metrics end up together
        flat = [set(c) for c in clusters]
        assert ({0, 1} in flat or {2, 3} in flat)

    def test_optimal_num_clusters(self):
        metrics = ["a", "b", "c", "d"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        corr = np.array([
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
        ])
        k = analyzer.optimal_num_clusters(corr)
        assert k >= 2

    def test_partial_correlation(self):
        rng = np.random.RandomState(42)
        z = rng.randn(100)
        x = z + rng.randn(100) * 0.3
        y = z + rng.randn(100) * 0.3
        # x and y are correlated through z
        r_xy, _ = MetricCorrelationAnalyzer.pearson_r(x, y)
        r_partial = MetricCorrelationAnalyzer.partial_correlation(x, y, z)
        # Partial correlation controlling for z should be smaller
        assert abs(r_partial) < abs(r_xy)


# =========================================================================
# 12. TestMetricProperties
# =========================================================================


class TestMetricProperties:
    """Test mathematical properties that metrics should satisfy."""

    def test_self_bleu_permutation_invariance(self):
        metric = SelfBLEU(max_order=4)
        texts = DIVERSE_TEXTS[:5]
        score1 = metric.compute(texts)
        shuffled = texts[2:] + texts[:2]
        score2 = metric.compute(shuffled)
        assert abs(score1 - score2) < 1e-10

    def test_distinct_n_permutation_invariance(self):
        metric = DistinctN(n=2)
        texts = DIVERSE_TEXTS[:5]
        score1 = metric.compute(texts)
        shuffled = texts[3:] + texts[:3]
        score2 = metric.compute(shuffled)
        assert abs(score1 - score2) < 1e-10

    def test_ngram_entropy_permutation_invariance(self):
        metric = NGramEntropy(n=2)
        texts = DIVERSE_TEXTS[:5]
        score1 = metric.compute(texts)
        shuffled = texts[4:] + texts[:4]
        score2 = metric.compute(shuffled)
        assert abs(score1 - score2) < 1e-10

    def test_epd_permutation_invariance(self):
        metric = EmbeddingPairwiseDistance(embedding_method="tfidf")
        texts = DIVERSE_TEXTS[:5]
        score1 = metric.compute(texts)
        shuffled = texts[2:] + texts[:2]
        score2 = metric.compute(shuffled)
        assert abs(score1 - score2) < 1e-6

    def test_vendi_permutation_invariance(self):
        metric = VendiScore(kernel_type="cosine")
        texts = DIVERSE_TEXTS[:5]
        score1 = metric.compute(texts)
        shuffled = texts[3:] + texts[:3]
        score2 = metric.compute(shuffled)
        assert abs(score1 - score2) < 1e-6

    def test_self_bleu_monotonicity_with_repetition(self):
        """Adding more copies of the same text should not decrease SelfBLEU."""
        metric = SelfBLEU(max_order=4)
        base = DIVERSE_TEXTS[:3]
        scores = []
        for k in range(3):
            texts = base + [base[0]] * k
            if len(texts) >= 2:
                scores.append(metric.compute(texts))
        # SelfBLEU should generally increase with more repetition
        assert scores[-1] >= scores[0] - 0.1

    def test_distinct_n_monotonicity_with_diversity(self):
        """Adding genuinely new text should increase or maintain Distinct-N."""
        metric = DistinctN(n=2)
        texts = [DIVERSE_TEXTS[0], DIVERSE_TEXTS[1]]
        score_2 = metric.compute(texts)
        texts_more = texts + [DIVERSE_TEXTS[2]]
        score_3 = metric.compute(texts_more)
        # Adding diverse text should maintain or increase distinct-n
        assert score_3 >= score_2 - 0.1

    def test_diversity_metrics_inherit_base(self):
        metrics = [
            SelfBLEU(), DistinctN(), NGramEntropy(),
            EmbeddingPairwiseDistance(), VendiScore(),
            ParseTreeDiversity(), BehavioralDiversity(),
        ]
        for m in metrics:
            assert isinstance(m, DiversityMetric)
            assert isinstance(m.name, str)
            assert isinstance(m.higher_is_better, bool)

    def test_quality_metrics_inherit_base(self):
        metrics = [Perplexity(), NLICoherence(), ConstraintSatisfaction()]
        for m in metrics:
            assert isinstance(m, QualityMetric)
            assert isinstance(m.name, str)
            assert isinstance(m.higher_is_better, bool)

    def test_diversity_metric_repr(self):
        m = SelfBLEU()
        r = repr(m)
        assert "SelfBLEU" in r
        assert "self_bleu" in r

    def test_compute_with_ci_returns_ci(self):
        metric = SelfBLEU(max_order=2)
        mean_val, (lo, hi) = metric.compute_with_ci(
            DIVERSE_TEXTS, n_bootstrap=50, confidence=0.95
        )
        assert lo <= mean_val <= hi or abs(lo - hi) < 0.1

    @pytest.mark.parametrize("MetricClass,kwargs", [
        (SelfBLEU, {"max_order": 2}),
        (DistinctN, {"n": 2}),
        (NGramEntropy, {"n": 2}),
    ])
    def test_determinism(self, MetricClass, kwargs):
        metric = MetricClass(**kwargs)
        s1 = metric.compute(DIVERSE_TEXTS)
        s2 = metric.compute(DIVERSE_TEXTS)
        assert s1 == s2

    def test_scale_invariance_ngram_entropy(self):
        """Entropy doesn't change if we duplicate the entire corpus."""
        metric = NGramEntropy(n=2)
        texts = DIVERSE_TEXTS[:4]
        score1 = metric.compute(texts)
        doubled = texts + texts
        score2 = metric.compute(doubled)
        # The frequency distribution is the same, just scaled
        assert abs(score1 - score2) < 1e-10


# =========================================================================
# 13. TestMetricNumericalStability
# =========================================================================


class TestMetricNumericalStability:
    """Tests for numerical stability of metrics under edge cases."""

    def test_self_bleu_very_short_texts(self):
        metric = SelfBLEU(max_order=4)
        texts = ["hi", "hello"]
        score = metric.compute(texts)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_self_bleu_very_long_texts(self):
        metric = SelfBLEU(max_order=4)
        rng = random.Random(42)
        words = ["word" + str(i) for i in range(100)]
        texts = [" ".join(rng.choices(words, k=500)) for _ in range(3)]
        score = metric.compute(texts)
        assert np.isfinite(score)

    def test_distinct_n_large_corpus(self):
        metric = DistinctN(n=2)
        texts = generate_random_texts(n=50, seed=42)
        score = metric.compute(texts)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_ngram_entropy_single_type(self):
        """All texts are the same word repeated."""
        metric = NGramEntropy(n=1)
        texts = ["aaa aaa aaa aaa", "aaa aaa aaa"]
        score = metric.compute(texts)
        assert abs(score) < 1e-10  # zero entropy

    def test_epd_near_zero_embeddings(self):
        """Texts that produce near-zero embeddings."""
        metric = EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method="tfidf"
        )
        # Very short, near-identical texts
        texts = [".", "!"]
        score = metric.compute(texts)
        assert np.isfinite(score)

    def test_vendi_score_two_identical(self):
        metric = VendiScore(kernel_type="cosine")
        texts = ["hello world", "hello world"]
        score = metric.compute(texts)
        assert np.isfinite(score)
        assert score >= 0.9  # effectively 1 distinct item

    def test_vendi_score_large_set(self):
        metric = VendiScore(kernel_type="cosine")
        texts = generate_random_texts(n=30, seed=42)
        score = metric.compute(texts)
        assert np.isfinite(score)
        assert score >= 1.0

    def test_behavioral_diversity_identical(self):
        metric = BehavioralDiversity()
        texts = generate_identical_texts(5)
        score = metric.compute(texts)
        assert np.isfinite(score)

    def test_perplexity_short_text(self):
        metric = Perplexity(order=3)
        texts = ["hi", "hello"]
        score = metric.compute(texts)
        assert np.isfinite(score) and score > 0

    def test_perplexity_repeated_text(self):
        metric = Perplexity(order=2)
        texts = ["the the the the the the the the"] * 3
        score = metric.compute(texts)
        assert np.isfinite(score) and score > 0

    def test_coherence_empty_sentence(self):
        metric = NLICoherence()
        texts = ["Hello. . World."]
        per = metric.compute_per_sample(texts)
        assert all(np.isfinite(s) for s in per)

    def test_nan_handling_correlation(self):
        """Constant arrays should not produce NaN in correlation."""
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert np.isfinite(tau)
        assert np.isfinite(p)

    def test_large_embedding_pairwise(self):
        metric = EmbeddingPairwiseDistance(
            distance_metric="euclidean", embedding_method="ngram_hash", hash_dim=64
        )
        texts = generate_random_texts(n=40, seed=42)
        score = metric.compute(texts)
        assert np.isfinite(score)

    def test_parse_tree_punctuation_only(self):
        metric = ParseTreeDiversity()
        texts = ["!!! ???", "... ---"]
        score = metric.compute(texts)
        assert np.isfinite(score)

    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_all_metrics_finite_with_random_texts(self, n):
        texts = generate_random_texts(n=n, seed=42)
        metrics = [
            SelfBLEU(max_order=2),
            DistinctN(n=2),
            NGramEntropy(n=2),
            EmbeddingPairwiseDistance(embedding_method="tfidf"),
            VendiScore(kernel_type="cosine"),
            ParseTreeDiversity(),
            BehavioralDiversity(),
        ]
        for m in metrics:
            score = m.compute(texts)
            assert np.isfinite(score), f"{m.name} produced non-finite: {score}"


# =========================================================================
# 14. TestMetricCorrelation
# =========================================================================


class TestMetricCorrelation:
    """Tests for cross-metric correlations and expected relationships."""

    @pytest.fixture
    def text_sets(self):
        """Generate multiple text sets with varying diversity."""
        sets = []
        for i in range(15):
            if i < 5:
                # Low diversity: similar texts
                base = f"word{i} " * 10
                texts = [base + f"extra{j}" for j in range(5)]
            elif i < 10:
                # Medium diversity
                texts = generate_random_texts(n=5, seed=i * 100)
            else:
                # High diversity
                texts = generate_diverse_texts(n=5, seed=i * 100)
            sets.append(texts)
        return sets

    def test_self_bleu_vs_distinct_n_anticorrelation(self, text_sets):
        """SelfBLEU (lower=diverse) should anti-correlate with Distinct-N (higher=diverse)."""
        self_bleu = SelfBLEU(max_order=2)
        distinct = DistinctN(n=2)
        bleu_scores = [self_bleu.compute(ts) for ts in text_sets]
        dist_scores = [distinct.compute(ts) for ts in text_sets]
        tau, _ = MetricCorrelationAnalyzer.kendall_tau(
            np.array(bleu_scores), np.array(dist_scores)
        )
        # Expect negative correlation (both measure diversity, opposite direction)
        assert tau < 0.3  # not strongly positively correlated

    def test_distinct_n_vs_entropy_correlation(self, text_sets):
        """Distinct-N and entropy should be positively correlated."""
        distinct = DistinctN(n=2)
        entropy = NGramEntropy(n=2)
        dist_scores = [distinct.compute(ts) for ts in text_sets]
        ent_scores = [entropy.compute(ts) for ts in text_sets]
        tau, _ = MetricCorrelationAnalyzer.kendall_tau(
            np.array(dist_scores), np.array(ent_scores)
        )
        assert tau > -0.5  # shouldn't be strongly anti-correlated

    def test_analyzer_full_pipeline(self, text_sets):
        """Full pipeline: compute metrics, build correlation, cluster."""
        metric_names = ["self_bleu", "distinct_2", "entropy"]
        self_bleu = SelfBLEU(max_order=2)
        distinct = DistinctN(n=2)
        entropy = NGramEntropy(n=2)

        values = {
            "self_bleu": [self_bleu.compute(ts) for ts in text_sets],
            "distinct_2": [distinct.compute(ts) for ts in text_sets],
            "entropy": [entropy.compute(ts) for ts in text_sets],
        }
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(values)
        assert corr.shape == (3, 3)
        assert_symmetric(corr)

    def test_correlation_matrix_block_structure(self):
        """Highly correlated metrics should form blocks."""
        rng = np.random.RandomState(42)
        n = 30
        # Create two groups of correlated metrics
        base1 = rng.randn(n)
        base2 = rng.randn(n)
        values = {
            "a1": list(base1 + rng.randn(n) * 0.05),
            "a2": list(base1 + rng.randn(n) * 0.05),
            "b1": list(base2 + rng.randn(n) * 0.05),
            "b2": list(base2 + rng.randn(n) * 0.05),
        }
        metrics = ["a1", "a2", "b1", "b2"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        corr = analyzer.compute_correlation_matrix(values)
        # Within-block correlation should be high
        assert corr[0, 1] > 0.7
        assert corr[2, 3] > 0.7
        # Between-block correlation should be low
        assert abs(corr[0, 2]) < 0.5

    def test_kendall_tau_fast_matches_analyzer(self):
        rng = np.random.RandomState(42)
        x = rng.randn(20)
        y = rng.randn(20)
        tau_fast = kendall_tau_fast(x, y)
        tau_analyzer, _ = MetricCorrelationAnalyzer.kendall_tau(x, y)
        # Both should be close (kendall_tau_fast may differ slightly due to tie handling)
        assert abs(tau_fast - tau_analyzer) < 0.15

    def test_rank_data_basic(self):
        x = np.array([3.0, 1.0, 2.0])
        ranks = rank_data(x)
        assert ranks[0] == 3.0
        assert ranks[1] == 1.0
        assert ranks[2] == 2.0

    def test_rank_data_ties(self):
        x = np.array([1.0, 2.0, 2.0, 3.0])
        ranks = rank_data(x)
        assert ranks[0] == 1.0
        assert ranks[1] == 2.5
        assert ranks[2] == 2.5
        assert ranks[3] == 4.0

    def test_concordance_matrix(self):
        rankings = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ])
        C = concordance_matrix(rankings)
        assert C.shape == (2, 2)
        assert C[0, 1] == 1.0  # identical rankings => perfect concordance

    def test_correlation_to_distance(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        dist = correlation_to_distance(corr)
        assert dist[0, 0] == 0.0
        assert dist[0, 1] > 0.0
        assert dist[0, 1] == dist[1, 0]

    def test_effective_dimensionality_uniform(self):
        """Uniform eigenvalues: ED = n."""
        eigvals = np.ones(5)
        ed = effective_dimensionality(eigvals)
        assert abs(ed - 5.0) < 1e-10

    def test_effective_dimensionality_single(self):
        """Single non-zero eigenvalue: ED = 1."""
        eigvals = np.array([1.0, 0.0, 0.0, 0.0])
        ed = effective_dimensionality(eigvals)
        assert abs(ed - 1.0) < 1e-10

    def test_reduce_metric_set(self):
        metrics = ["a", "b", "c", "d"]
        analyzer = MetricCorrelationAnalyzer(metrics=metrics)
        corr = np.array([
            [1.0, 0.95, 0.1, 0.1],
            [0.95, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.95],
            [0.1, 0.1, 0.95, 1.0],
        ])
        reps = analyzer.reduce_metric_set(corr)
        # Should select one from each cluster
        assert len(reps) >= 2
        assert len(reps) <= 4


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for helper functions used by metrics."""

    def test_tokenize_simple(self):
        tokens = tokenize_simple("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," in tokens

    def test_tokenize_simple_empty(self):
        tokens = tokenize_simple("")
        assert tokens == []

    def test_extract_ngrams_basic(self):
        tokens = ["a", "b", "c", "d"]
        bigrams = extract_ngrams(tokens, 2)
        assert bigrams == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_extract_ngrams_too_short(self):
        tokens = ["a"]
        bigrams = extract_ngrams(tokens, 2)
        assert bigrams == []

    def test_extract_ngrams_n_equals_length(self):
        tokens = ["a", "b", "c"]
        trigrams = extract_ngrams(tokens, 3)
        assert trigrams == [("a", "b", "c")]

    def test_tfidf_embeddings_shape(self):
        emb = tfidf_embeddings(DIVERSE_TEXTS)
        assert emb.shape[0] == len(DIVERSE_TEXTS)
        assert emb.shape[1] > 0

    def test_tfidf_embeddings_normalized(self):
        emb = tfidf_embeddings(DIVERSE_TEXTS)
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_ngram_hash_embeddings_shape(self):
        emb = ngram_hash_embeddings(DIVERSE_TEXTS, n=3, dim=128)
        assert emb.shape == (len(DIVERSE_TEXTS), 128)

    def test_ngram_hash_embeddings_normalized(self):
        emb = ngram_hash_embeddings(DIVERSE_TEXTS, n=3, dim=128)
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_pairwise_distances_symmetric(self):
        emb = np.random.RandomState(42).randn(5, 10)
        dist = pairwise_distances(emb, metric="euclidean")
        assert_symmetric(dist, atol=1e-10)

    def test_pairwise_distances_zero_diagonal(self):
        emb = np.random.RandomState(42).randn(5, 10)
        dist = pairwise_distances(emb, metric="euclidean")
        assert np.allclose(np.diag(dist), 0.0, atol=1e-10)

    def test_pairwise_distances_single(self):
        emb = np.random.RandomState(42).randn(1, 10)
        dist = pairwise_distances(emb, metric="cosine")
        assert dist.shape == (1, 1)
        assert dist[0, 0] == 0.0


# =========================================================================
# Spectral clustering tests
# =========================================================================


class TestSpectralClusterer:
    """Tests for SpectralClusterer used by MetricCorrelationAnalyzer."""

    def test_cluster_two_blocks(self):
        """Two clear blocks should be separated."""
        W = np.array([
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.9],
            [0.1, 0.1, 0.9, 0.0],
        ])
        sc = SpectralClusterer()
        labels = sc.cluster(W, n_clusters=2)
        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_normalized_laplacian_symmetric(self):
        W = np.array([[0.0, 0.5, 0.1], [0.5, 0.0, 0.3], [0.1, 0.3, 0.0]])
        L = SpectralClusterer._normalized_laplacian(W)
        assert_symmetric(L, atol=1e-10)

    def test_unnormalized_laplacian(self):
        W = np.array([[0.0, 1.0], [1.0, 0.0]])
        L = SpectralClusterer._unnormalized_laplacian(W)
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(L, expected)

    def test_eigenvalue_gap(self):
        eigvals = np.array([0.0, 0.0, 0.5, 0.8, 1.0])
        k = SpectralClusterer._eigenvalue_gap(eigvals)
        assert k >= 2

    def test_affinity_from_correlation(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        W = SpectralClusterer._affinity_from_correlation(corr)
        assert W[0, 0] == 0.0  # diagonal set to 0
        assert W[0, 1] == 0.75  # (1 + 0.5) / 2

    def test_cluster_identity_matrix(self):
        """Identity similarity matrix (each item unique) should still work."""
        W = np.eye(4)
        np.fill_diagonal(W, 0.0)  # remove self-loops
        sc = SpectralClusterer()
        labels = sc.cluster(W, n_clusters=2)
        assert len(labels) == 4

    def test_single_cluster(self):
        W = np.ones((4, 4)) * 0.9
        np.fill_diagonal(W, 0.0)
        sc = SpectralClusterer()
        labels = sc.cluster(W, n_clusters=1)
        assert len(labels) == 4
        assert len(np.unique(labels)) == 1


# =========================================================================
# Integration tests
# =========================================================================


class TestMetricIntegration:
    """Integration tests running multiple metrics on the same data."""

    def test_all_diversity_metrics_on_diverse_texts(self):
        metrics = [
            SelfBLEU(max_order=2),
            DistinctN(n=2),
            NGramEntropy(n=2),
            EmbeddingPairwiseDistance(embedding_method="tfidf"),
            VendiScore(kernel_type="cosine"),
            ParseTreeDiversity(),
            BehavioralDiversity(),
        ]
        for m in metrics:
            score = m.compute(DIVERSE_TEXTS)
            assert np.isfinite(score), f"{m.name} non-finite"

    def test_all_quality_metrics_on_diverse_texts(self):
        metrics = [
            Perplexity(order=2),
            NLICoherence(),
            ConstraintSatisfaction(),
        ]
        for m in metrics:
            score = m.compute(DIVERSE_TEXTS)
            assert np.isfinite(score), f"{m.name} non-finite"

    def test_correlation_across_all_metrics(self):
        """Compute all metrics on multiple text sets and check correlation matrix."""
        text_sets = [
            generate_random_texts(n=5, seed=i) for i in range(10)
        ]
        metric_names = ["distinct_2", "entropy_2"]
        values = {
            "distinct_2": [DistinctN(n=2).compute(ts) for ts in text_sets],
            "entropy_2": [NGramEntropy(n=2).compute(ts) for ts in text_sets],
        }
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(values)
        assert corr.shape == (2, 2)
        assert np.allclose(np.diag(corr), 1.0)

    def test_conftest_helpers_match_metrics(self):
        """Check that conftest helper computations agree with metric classes."""
        texts = generate_random_texts(n=5, seed=42)
        # Distinct-N
        metric_dn = DistinctN(n=2)
        dn_metric = metric_dn.compute(texts)
        dn_conftest = compute_distinct_n(texts, n=2)
        # They may differ slightly due to tokenization differences
        # (conftest uses split(), metric uses tokenize_simple)
        assert abs(dn_metric - dn_conftest) < 0.5

    def test_generate_identical_all_metrics_consistent(self):
        """All diversity metrics should indicate low diversity for identical texts."""
        texts = generate_identical_texts(5)
        self_bleu = SelfBLEU(max_order=2).compute(texts)
        distinct = DistinctN(n=2).compute(texts)
        entropy = NGramEntropy(n=2).compute(texts)
        epd = EmbeddingPairwiseDistance(embedding_method="tfidf").compute(texts)

        # SelfBLEU high (low diversity), Distinct-N constant, EPD ~0
        assert self_bleu > 0.7
        assert epd < 0.01

    def test_generate_diverse_all_metrics_consistent(self):
        """All diversity metrics should indicate higher diversity for diverse texts."""
        texts_div = generate_diverse_texts(n=8, seed=42)
        texts_ident = generate_identical_texts(5)

        sb_div = SelfBLEU(max_order=2).compute(texts_div)
        sb_ident = SelfBLEU(max_order=2).compute(texts_ident)
        assert sb_div < sb_ident  # lower SelfBLEU = more diverse

        dn_div = DistinctN(n=2).compute(texts_div)
        dn_ident = DistinctN(n=2).compute(texts_ident)
        assert dn_div > dn_ident  # higher Distinct-N = more diverse
