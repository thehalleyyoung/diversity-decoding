"""Tests for neural diversity metrics: MAUVE, BERTScore diversity, STS diversity, CRD."""

import math
import sys
import os
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.neural_diversity import (
    MAUVE,
    BERTScoreDiversity,
    STSDiversity,
    CompressionRatioDiversity,
    NeuralDiversitySuite,
    _compute_divergence_curve,
    _compute_mauve_area,
    _kmeans_cluster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def diverse_texts():
    return [
        "The cat sat on the mat and watched birds fly by.",
        "Quantum computing leverages superposition and entanglement.",
        "She cooked a delicious Italian pasta with fresh basil.",
        "The stock market rallied after positive earnings reports.",
        "A thunderstorm rolled through the valley at midnight.",
    ]


@pytest.fixture
def identical_texts():
    return [
        "The cat sat on the mat.",
        "The cat sat on the mat.",
        "The cat sat on the mat.",
        "The cat sat on the mat.",
    ]


@pytest.fixture
def similar_texts():
    return [
        "The cat sat on the mat.",
        "The cat was sitting on the mat.",
        "A cat sat upon the mat.",
        "The cat rested on the mat.",
    ]


# ---------------------------------------------------------------------------
# CRD Tests — verifying consistency with paper formula
# ---------------------------------------------------------------------------

class TestCRD:
    """CRD: |compress(concat(G))| / Σ|compress(yi)|"""

    def test_formula_matches_canonical(self):
        """Verify CRD implementation matches canonical formula."""
        texts = ["hello world", "foo bar baz", "unique sentence here"]
        crd = CompressionRatioDiversity()
        score = crd.compute(texts)

        # Manual calculation
        individual = sum(len(zlib.compress(t.encode("utf-8"), 9)) for t in texts)
        concat_compressed = len(zlib.compress("\n".join(texts).encode("utf-8"), 9))
        expected = min(concat_compressed / individual, 1.0)

        assert abs(score - expected) < 1e-10, (
            f"CRD score {score} != manual {expected}"
        )

    def test_identical_texts_low_diversity(self):
        """Identical texts should have low CRD (high cross-text redundancy)."""
        crd = CompressionRatioDiversity()
        # Use enough repetitions so compression exploits redundancy
        identical = ["The cat sat on the mat. This is a test sentence."] * 10
        score = crd.compute(identical)
        # concat of identical texts compresses far below sum of individuals
        assert score < 0.5, f"Identical texts should have low CRD, got {score}"

    def test_diverse_texts_higher_diversity(self):
        """Diverse texts should have higher CRD than identical texts."""
        crd = CompressionRatioDiversity()
        identical = ["The cat sat on the mat. This is a test sentence."] * 10
        diverse = [
            "The cat sat on the mat and watched birds fly by the window.",
            "Quantum computing leverages superposition and entanglement.",
            "She cooked a delicious Italian pasta with fresh basil.",
            "The stock market rallied after positive earnings reports.",
            "A thunderstorm rolled through the valley at midnight.",
            "The mathematician proved the theorem using novel methods.",
            "Deep sea creatures have adapted to extreme pressure.",
            "The Renaissance brought unprecedented changes to art.",
            "Renewable energy sources include solar and wind power.",
            "Ancient Egyptian pyramids were built as monumental tombs.",
        ]
        diverse_score = crd.compute(diverse)
        identical_score = crd.compute(identical)
        assert diverse_score > identical_score, (
            f"Diverse CRD {diverse_score} should > identical CRD {identical_score}"
        )

    def test_crd_range(self, diverse_texts):
        """CRD should be in [0, 1]."""
        crd = CompressionRatioDiversity()
        score = crd.compute(diverse_texts)
        assert 0.0 <= score <= 1.0

    def test_minimum_texts_required(self):
        """CRD needs at least 2 texts."""
        crd = CompressionRatioDiversity()
        with pytest.raises(ValueError):
            crd.compute(["only one"])

    def test_name_and_properties(self):
        crd = CompressionRatioDiversity()
        assert crd.name == "CRD"
        assert crd.higher_is_better is True

    def test_appendix_old_formula_differs(self):
        """Verify the OLD appendix formula (1 - compress/raw) gives a DIFFERENT
        value than the canonical formula, confirming the fix was needed."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A completely different sentence about quantum physics.",
            "Yet another unique text about cooking recipes.",
        ]
        # Canonical formula: compress(concat) / sum(compress(yi))
        crd = CompressionRatioDiversity()
        canonical = crd.compute(texts)

        # Old appendix formula: 1 - compress(concat) / len(concat)
        concatenated = "\n".join(texts)
        raw = concatenated.encode("utf-8")
        compressed = zlib.compress(raw, 9)
        old_appendix = 1.0 - len(compressed) / len(raw)

        # They should be different quantities
        assert abs(canonical - old_appendix) > 0.01, (
            f"Canonical CRD {canonical} and old appendix {old_appendix} should differ"
        )


# ---------------------------------------------------------------------------
# MAUVE Tests
# ---------------------------------------------------------------------------

class TestMAUVE:
    def test_basic_computation(self, diverse_texts):
        mauve = MAUVE(n_clusters=3, backend="tfidf")
        score = mauve.compute(diverse_texts)
        assert 0.0 <= score <= 1.0

    def test_identical_texts(self, identical_texts):
        mauve = MAUVE(n_clusters=2, backend="tfidf")
        score = mauve.compute(identical_texts)
        assert 0.0 <= score <= 1.0

    def test_between_distributions(self, diverse_texts, similar_texts):
        mauve = MAUVE(n_clusters=3, backend="tfidf")
        score = mauve.compute_between(diverse_texts, similar_texts)
        assert 0.0 <= score <= 1.0

    def test_detailed(self, diverse_texts):
        mauve = MAUVE(n_clusters=3, backend="tfidf")
        result = mauve.compute_detailed(diverse_texts)
        assert "mauve_score" in result
        assert "n_texts" in result

    def test_name_and_properties(self):
        mauve = MAUVE()
        assert mauve.name == "MAUVE"
        assert mauve.higher_is_better is True


# ---------------------------------------------------------------------------
# BERTScore Diversity Tests
# ---------------------------------------------------------------------------

class TestBERTScoreDiversity:
    def test_diverse_texts_high_diversity(self, diverse_texts):
        bert_div = BERTScoreDiversity(backend="tfidf")
        score = bert_div.compute(diverse_texts)
        assert 0.0 <= score <= 1.0

    def test_identical_texts_low_diversity(self, identical_texts):
        bert_div = BERTScoreDiversity(backend="tfidf")
        score = bert_div.compute(identical_texts)
        # Identical texts should have low diversity (high similarity)
        assert score < 0.1, f"Identical texts should have near-zero diversity, got {score}"

    def test_diverse_higher_than_identical(self, diverse_texts, identical_texts):
        bert_div = BERTScoreDiversity(backend="tfidf")
        diverse_score = bert_div.compute(diverse_texts)
        identical_score = bert_div.compute(identical_texts)
        assert diverse_score > identical_score

    def test_detailed(self, diverse_texts):
        bert_div = BERTScoreDiversity(backend="tfidf")
        result = bert_div.compute_detailed(diverse_texts)
        assert "diversity" in result
        assert "mean_pairwise_similarity" in result
        assert "n_pairs" in result

    def test_name_and_properties(self):
        bert_div = BERTScoreDiversity()
        assert bert_div.name == "BERTScore-Diversity"
        assert bert_div.higher_is_better is True


# ---------------------------------------------------------------------------
# STS Diversity Tests
# ---------------------------------------------------------------------------

class TestSTSDiversity:
    def test_diverse_texts(self, diverse_texts):
        sts = STSDiversity(backend="tfidf")
        score = sts.compute(diverse_texts)
        assert 0.0 <= score <= 1.0

    def test_identical_texts_low(self, identical_texts):
        sts = STSDiversity(backend="tfidf")
        score = sts.compute(identical_texts)
        assert score < 0.2, f"Identical texts should have low STS diversity, got {score}"

    def test_detailed(self, diverse_texts):
        sts = STSDiversity(backend="tfidf")
        result = sts.compute_detailed(diverse_texts)
        assert "sts_diversity" in result
        assert "mean_distance" in result
        assert "sim_entropy" in result

    def test_name_and_properties(self):
        sts = STSDiversity()
        assert sts.name == "STS-Diversity"
        assert sts.higher_is_better is True


# ---------------------------------------------------------------------------
# Suite Tests
# ---------------------------------------------------------------------------

class TestNeuralDiversitySuite:
    def test_compute_all(self, diverse_texts):
        suite = NeuralDiversitySuite(backend="tfidf")
        results = suite.compute_all(diverse_texts)
        assert "MAUVE" in results
        assert "BERTScore-Diversity" in results
        assert "STS-Diversity" in results
        assert "CRD" in results
        for k, v in results.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_compute_all_detailed(self, diverse_texts):
        suite = NeuralDiversitySuite(backend="tfidf")
        results = suite.compute_all_detailed(diverse_texts)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_kmeans_cluster(self):
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]], dtype=float)
        labels, centroids = _kmeans_cluster(X, 2)
        assert len(labels) == 4
        assert centroids.shape == (2, 2)
        # Points 0,1 should be in same cluster, 2,3 in another
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_divergence_curve(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.3, 0.6])
        kl_p, kl_q = _compute_divergence_curve(p, q, num_points=50)
        assert len(kl_p) == 50
        assert len(kl_q) == 50
        assert all(v >= 0 for v in kl_p)
        assert all(v >= 0 for v in kl_q)

    def test_mauve_area(self):
        # Non-zero divergence: area should be positive
        kl_p = np.linspace(0, 1, 10)
        kl_q = np.linspace(1, 0, 10)
        area = _compute_mauve_area(kl_p, kl_q)
        assert area > 0.0
        assert area <= 1.0
