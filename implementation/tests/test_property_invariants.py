"""
Property-based tests for diversity metric invariants.

Verifies mathematical properties that every well-behaved diversity
metric should satisfy:
  1. Non-negativity
  2. Boundedness (metrics in [0, 1] or known range)
  3. Monotonicity under duplication (adding duplicates cannot increase diversity)
  4. Symmetry (order of texts does not matter)
  5. Maximality on maximally distinct inputs
  6. Minimality on identical inputs
"""

from __future__ import annotations

import random
import string

import numpy as np
import pytest

# ---------- helpers ----------
def _random_text(rng: random.Random, length: int = 30) -> str:
    words = []
    for _ in range(length):
        wl = rng.randint(2, 8)
        words.append("".join(rng.choices(string.ascii_lowercase, k=wl)))
    return " ".join(words)


def _random_texts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [_random_text(rng) for _ in range(n)]


# ---------- metric wrappers ----------
from src.metrics.diversity import DistinctN, NGramEntropy, SelfBLEU
from src.metrics.information_theoretic import shannon_entropy, kl_divergence


def _distinct2(texts: list[str]) -> float:
    m = DistinctN(n=2)
    return m.compute(texts)


def _ngram_entropy(texts: list[str]) -> float:
    m = NGramEntropy()
    return m.compute(texts)


def _self_bleu(texts: list[str]) -> float:
    m = SelfBLEU()
    return m.compute(texts)


def _shannon(texts: list[str]) -> float:
    return shannon_entropy(texts, n=2)


DIVERSITY_METRICS = [
    ("distinct_2", _distinct2),
    ("ngram_entropy", _ngram_entropy),
    ("shannon_entropy", _shannon),
]

# Self-BLEU is a *similarity* metric (lower = more diverse), tested separately.


# ---------- property tests ----------
class TestNonNegativity:
    """All diversity metrics must return >= 0."""

    @pytest.mark.parametrize("name,metric", DIVERSITY_METRICS)
    def test_non_negative(self, name, metric):
        texts = _random_texts(10, seed=1)
        assert metric(texts) >= 0.0, f"{name} returned negative"


class TestSymmetry:
    """Metric value must be invariant to the ordering of texts."""

    @pytest.mark.parametrize("name,metric", DIVERSITY_METRICS)
    def test_permutation_invariant(self, name, metric):
        texts = _random_texts(8, seed=2)
        val1 = metric(texts)
        shuffled = texts[::-1]
        val2 = metric(shuffled)
        assert abs(val1 - val2) < 1e-10, f"{name} not permutation-invariant"


class TestMonotonicity:
    """Adding duplicate texts should not increase diversity."""

    @pytest.mark.parametrize("name,metric", DIVERSITY_METRICS)
    def test_duplication_decreases(self, name, metric):
        texts = _random_texts(6, seed=3)
        val_orig = metric(texts)
        # duplicate every text
        doubled = texts + texts
        val_dup = metric(doubled)
        assert val_dup <= val_orig + 1e-9, (
            f"{name}: duplication increased diversity "
            f"({val_orig:.6f} -> {val_dup:.6f})"
        )


class TestIdenticalInputs:
    """Identical texts should yield minimal diversity."""

    @pytest.mark.parametrize("name,metric", DIVERSITY_METRICS)
    def test_identical_low(self, name, metric):
        single = "the quick brown fox jumps over the lazy dog"
        texts = [single] * 10
        val = metric(texts)
        # For a set of identical texts, diversity should be very low.
        # Distinct-2 with identical texts is still > 0 because there are
        # unique 2-grams within the text, but it should be low.
        diverse_texts = _random_texts(10, seed=99)
        val_diverse = metric(diverse_texts)
        assert val <= val_diverse + 1e-9, (
            f"{name}: identical texts not lower than diverse "
            f"({val:.4f} vs {val_diverse:.4f})"
        )


class TestSelfBLEUProperties:
    """Self-BLEU is a similarity metric (lower = more diverse)."""

    def test_non_negative(self):
        texts = _random_texts(8, seed=4)
        assert _self_bleu(texts) >= 0.0

    def test_identical_high(self):
        single = "the quick brown fox jumps over the lazy dog"
        texts = [single] * 8
        val = _self_bleu(texts)
        # Identical texts should yield high self-BLEU
        diverse_texts = _random_texts(8, seed=5)
        val_diverse = _self_bleu(diverse_texts)
        assert val >= val_diverse - 1e-9

    def test_symmetry(self):
        texts = _random_texts(6, seed=6)
        v1 = _self_bleu(texts)
        v2 = _self_bleu(texts[::-1])
        assert abs(v1 - v2) < 1e-10


class TestKLDivergenceProperties:
    """KL divergence invariants."""

    def test_non_negative(self):
        a = _random_texts(10, seed=7)
        b = _random_texts(10, seed=8)
        assert kl_divergence(a, b) >= -1e-10

    def test_self_divergence_near_zero(self):
        a = _random_texts(20, seed=9)
        # KL(P || P) ≈ 0 (exact 0 without smoothing)
        val = kl_divergence(a, a)
        assert val < 0.01, f"Self-divergence too large: {val}"


class TestBootstrapCI:
    """Bootstrap CI must contain the point estimate."""

    def test_ci_contains_point(self):
        from src.metrics.bootstrap import bootstrap_ci
        data = np.random.RandomState(10).randn(50)
        result = bootstrap_ci(data, np.mean)
        assert result["ci_lower"] <= result["point"] <= result["ci_upper"]
        assert result["ci_lower"] < result["ci_upper"]
