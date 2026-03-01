"""
Comprehensive utilities module for the Diversity Decoding Arena.

Provides math, sampling, text, caching, IO, timing, numpy, and config utilities
used throughout the decoding and evaluation pipeline.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import time
import zlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np


# ---------------------------------------------------------------------------
# 1. MathUtils
# ---------------------------------------------------------------------------

class MathUtils:
    """Numerical routines for probability distributions, kernels, and distances."""

    # -- softmax / log-softmax ------------------------------------------------

    @staticmethod
    def log_softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable log-softmax.

        Subtracts the maximum before exponentiation to avoid overflow, then
        computes log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x)))).

        Parameters
        ----------
        logits : np.ndarray
            Raw (unnormalised) log-probabilities.  Supports 1-D or 2-D input;
            when 2-D the operation is applied row-wise.

        Returns
        -------
        np.ndarray
            Log-probabilities of the same shape as *logits*.
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            c = logits.max()
            shifted = logits - c
            log_z = np.log(np.sum(np.exp(shifted)))
            return shifted - log_z
        elif logits.ndim == 2:
            c = logits.max(axis=1, keepdims=True)
            shifted = logits - c
            log_z = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            return shifted - log_z
        else:
            raise ValueError(
                f"log_softmax expects 1-D or 2-D input, got {logits.ndim}-D"
            )

    @staticmethod
    def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Temperature-scaled softmax.

        Parameters
        ----------
        logits : np.ndarray
            Raw log-probabilities (1-D or 2-D).
        temperature : float
            Softmax temperature.  Values < 1 sharpen the distribution; > 1
            flatten it.  Must be positive.

        Returns
        -------
        np.ndarray
            Probability distribution(s) summing to 1.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = np.asarray(logits, dtype=np.float64)
        scaled = logits / temperature
        log_p = MathUtils.log_softmax(scaled)
        return np.exp(log_p)

    # -- log-sum-exp ----------------------------------------------------------

    @staticmethod
    def log_sum_exp(values: np.ndarray) -> float:
        """Numerically stable log-sum-exp.

        Parameters
        ----------
        values : array-like
            Sequence of log-values.

        Returns
        -------
        float
            log(sum(exp(values)))
        """
        values = np.asarray(values, dtype=np.float64)
        c = values.max()
        return float(c + np.log(np.sum(np.exp(values - c))))

    # -- divergences & entropies ---------------------------------------------

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Kullback–Leibler divergence D_KL(p || q).

        Parameters
        ----------
        p, q : np.ndarray
            Discrete probability distributions (must be non-negative and sum
            to 1).  Entries where *p* == 0 are ignored (0 log 0 := 0).  If
            *q* == 0 where *p* > 0 the result is +inf.

        Returns
        -------
        float
            KL divergence in nats.
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if p.shape != q.shape:
            raise ValueError("p and q must have the same shape")
        mask = p > 0
        if np.any(q[mask] <= 0):
            return float("inf")
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen–Shannon divergence (symmetric, bounded in [0, ln 2]).

        JSD(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)  with m = (p+q)/2.

        Parameters
        ----------
        p, q : np.ndarray
            Discrete probability distributions.

        Returns
        -------
        float
            JS divergence in nats.
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        m = 0.5 * (p + q)
        return 0.5 * MathUtils.kl_divergence(p, m) + 0.5 * MathUtils.kl_divergence(q, m)

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        """Shannon entropy H(p) = -sum(p_i * log(p_i)).

        Parameters
        ----------
        p : np.ndarray
            Probability distribution.

        Returns
        -------
        float
            Entropy in nats.
        """
        p = np.asarray(p, dtype=np.float64)
        mask = p > 0
        return float(-np.sum(p[mask] * np.log(p[mask])))

    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """Cross-entropy H(p, q) = -sum(p_i * log(q_i)).

        Parameters
        ----------
        p, q : np.ndarray
            Probability distributions.

        Returns
        -------
        float
            Cross-entropy in nats.  Returns +inf if q_i == 0 where p_i > 0.
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        mask = p > 0
        if np.any(q[mask] <= 0):
            return float("inf")
        return float(-np.sum(p[mask] * np.log(q[mask])))

    # -- inequality & similarity measures ------------------------------------

    @staticmethod
    def gini_coefficient(values: np.ndarray) -> float:
        """Gini coefficient measuring inequality of a distribution.

        Parameters
        ----------
        values : np.ndarray
            Non-negative values.

        Returns
        -------
        float
            Gini index in [0, 1].  0 = perfect equality, 1 = maximal
            inequality.
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        if len(values) == 0:
            return 0.0
        sorted_v = np.sort(values)
        n = len(sorted_v)
        total = sorted_v.sum()
        if total == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * sorted_v) / (n * total)) - (n + 1) / n)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors.

        Parameters
        ----------
        a, b : np.ndarray
            1-D vectors.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # -- pairwise distance / kernel matrices ---------------------------------

    @staticmethod
    def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
        """Pairwise cosine distance matrix.

        Parameters
        ----------
        X : np.ndarray
            Matrix of shape (n, d) where each row is a vector.

        Returns
        -------
        np.ndarray
            Symmetric (n, n) matrix of cosine distances in [0, 2].
        """
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_normed = X / norms
        sim = X_normed @ X_normed.T
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim

    @staticmethod
    def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
        """Pairwise Euclidean distance matrix.

        Uses the expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b for speed.

        Parameters
        ----------
        X : np.ndarray
            (n, d) matrix.

        Returns
        -------
        np.ndarray
            (n, n) distance matrix.
        """
        X = np.asarray(X, dtype=np.float64)
        sq_norms = np.sum(X ** 2, axis=1)
        D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
        np.maximum(D_sq, 0.0, out=D_sq)
        return np.sqrt(D_sq)

    @staticmethod
    def rbf_kernel(X: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
        """Radial basis function (Gaussian) kernel matrix.

        K(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))

        Parameters
        ----------
        X : np.ndarray
            (n, d) data matrix.
        bandwidth : float, optional
            Kernel bandwidth.  If *None*, the median heuristic is used.

        Returns
        -------
        np.ndarray
            (n, n) kernel matrix with values in (0, 1].
        """
        X = np.asarray(X, dtype=np.float64)
        if bandwidth is None:
            bandwidth = MathUtils.median_heuristic(X)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        D = MathUtils.euclidean_distance_matrix(X)
        return np.exp(-D ** 2 / (2.0 * bandwidth ** 2))

    @staticmethod
    def median_heuristic(X: np.ndarray) -> float:
        """Median heuristic for RBF kernel bandwidth selection.

        Returns the median of all pairwise Euclidean distances (excluding
        self-distances).

        Parameters
        ----------
        X : np.ndarray
            (n, d) data matrix.

        Returns
        -------
        float
            Bandwidth estimate.
        """
        D = MathUtils.euclidean_distance_matrix(X)
        n = D.shape[0]
        # upper triangle (no diagonal)
        upper = D[np.triu_indices(n, k=1)]
        if len(upper) == 0:
            return 1.0
        med = float(np.median(upper))
        return med if med > 0 else 1.0

    @staticmethod
    def normalize_rows(X: np.ndarray) -> np.ndarray:
        """L2-normalise each row of *X*.

        Zero rows are left as zero.

        Parameters
        ----------
        X : np.ndarray
            (n, d) matrix.

        Returns
        -------
        np.ndarray
            Row-normalised matrix.
        """
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms


# ---------------------------------------------------------------------------
# 2. SamplingUtils
# ---------------------------------------------------------------------------

class SamplingUtils:
    """Sampling helpers for autoregressive language-model decoding."""

    @staticmethod
    def categorical_sample(
        probs: np.ndarray,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """Draw *n* independent samples from a categorical distribution.

        Parameters
        ----------
        probs : np.ndarray
            1-D probability vector (must sum to 1).
        n : int
            Number of samples to draw.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        List[int]
            Sampled indices.
        """
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum()  # re-normalise for safety
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(len(probs), size=n, p=probs).tolist()

    @staticmethod
    def gumbel_max_sample(
        logits: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Gumbel-max trick: sample from categorical without computing softmax.

        Parameters
        ----------
        logits : np.ndarray
            Unnormalised log-probabilities (1-D).
        rng : np.random.Generator, optional

        Returns
        -------
        int
            Sampled index.
        """
        logits = np.asarray(logits, dtype=np.float64)
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(size=logits.shape)
        u = np.clip(u, 1e-20, 1.0)
        g = -np.log(-np.log(u))
        return int(np.argmax(logits + g))

    # -- masking strategies ---------------------------------------------------

    @staticmethod
    def top_k_mask(logits: np.ndarray, k: int) -> np.ndarray:
        """Mask all but the *k* highest logits with -inf.

        Parameters
        ----------
        logits : np.ndarray
            1-D logits.
        k : int
            Number of entries to keep.

        Returns
        -------
        np.ndarray
            Masked logits (same shape).
        """
        logits = np.asarray(logits, dtype=np.float64).copy()
        if k <= 0:
            raise ValueError("k must be positive")
        if k >= len(logits):
            return logits
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -np.inf
        return logits

    @staticmethod
    def top_p_mask(logits: np.ndarray, p: float) -> np.ndarray:
        """Nucleus (top-p) sampling mask.

        Keeps the smallest set of tokens whose cumulative probability ≥ *p*.

        Parameters
        ----------
        logits : np.ndarray
            1-D logits.
        p : float
            Probability threshold in (0, 1].

        Returns
        -------
        np.ndarray
            Masked logits.
        """
        logits = np.asarray(logits, dtype=np.float64).copy()
        if p <= 0.0 or p > 1.0:
            raise ValueError("p must be in (0, 1]")

        probs = MathUtils.softmax(logits)
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        # Find cutoff: keep tokens up to cumulative >= p
        cutoff_idx = np.searchsorted(cumulative, p)
        if cutoff_idx < len(cumulative):
            cutoff_idx += 1  # include the token that crosses the threshold
        keep_indices = sorted_indices[:cutoff_idx]

        mask = np.full(logits.shape, -np.inf)
        mask[keep_indices] = logits[keep_indices]
        return mask

    @staticmethod
    def typical_mask(logits: np.ndarray, mass: float = 0.9) -> np.ndarray:
        """Typical sampling mask (Meister et al., 2023).

        Keeps tokens whose information content is close to the entropy of the
        distribution, covering cumulative probability *mass*.

        Parameters
        ----------
        logits : np.ndarray
            1-D logits.
        mass : float
            Cumulative probability target.

        Returns
        -------
        np.ndarray
            Masked logits.
        """
        logits = np.asarray(logits, dtype=np.float64).copy()
        probs = MathUtils.softmax(logits)
        log_probs = np.log(np.clip(probs, 1e-20, None))
        ent = MathUtils.entropy(probs)

        # Surprise (negative log-prob) minus entropy gives "typicality" shift
        surprisal = -log_probs
        shift = np.abs(surprisal - ent)

        # Sort by shift (most typical first)
        sorted_indices = np.argsort(shift)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        cutoff_idx = np.searchsorted(cumulative, mass)
        if cutoff_idx < len(cumulative):
            cutoff_idx += 1
        keep_indices = sorted_indices[:cutoff_idx]

        mask = np.full(logits.shape, -np.inf)
        mask[keep_indices] = logits[keep_indices]
        return mask

    @staticmethod
    def temperature_scale(logits: np.ndarray, temp: float) -> np.ndarray:
        """Scale logits by temperature.

        Parameters
        ----------
        logits : np.ndarray
            1-D or 2-D logits.
        temp : float
            Temperature (must be positive).

        Returns
        -------
        np.ndarray
            Scaled logits.
        """
        if temp <= 0:
            raise ValueError("temp must be positive")
        logits = np.asarray(logits, dtype=np.float64)
        return logits / temp

    @staticmethod
    def repetition_penalty(
        logits: np.ndarray,
        prev_tokens: Sequence[int],
        penalty: float = 1.2,
    ) -> np.ndarray:
        """Apply repetition penalty to logits for previously generated tokens.

        For each token in *prev_tokens*, if the logit is positive it is divided
        by *penalty*; if negative it is multiplied by *penalty*.  This reduces
        the probability of repeating tokens.

        Parameters
        ----------
        logits : np.ndarray
            1-D logits.
        prev_tokens : Sequence[int]
            Token indices that have already been generated.
        penalty : float
            Repetition penalty factor (≥ 1.0).

        Returns
        -------
        np.ndarray
            Penalised logits.
        """
        logits = np.asarray(logits, dtype=np.float64).copy()
        if penalty < 1.0:
            raise ValueError("penalty must be >= 1.0")
        for tok in set(prev_tokens):
            if 0 <= tok < len(logits):
                if logits[tok] > 0:
                    logits[tok] /= penalty
                else:
                    logits[tok] *= penalty
        return logits

    @staticmethod
    def no_repeat_ngram(
        logits: np.ndarray,
        prev_tokens: Sequence[int],
        n: int,
    ) -> np.ndarray:
        """Ban any token that would create a repeated n-gram.

        Scans the generated prefix for all (n-1)-grams and, if the current
        trailing (n-1) tokens match a previous (n-1)-gram, sets the logit of
        the token that would complete the repeat to -inf.

        Parameters
        ----------
        logits : np.ndarray
            1-D logits.
        prev_tokens : Sequence[int]
            Previously generated token indices.
        n : int
            N-gram size to avoid repeating (e.g. 3 for tri-grams).

        Returns
        -------
        np.ndarray
            Logits with banned positions set to -inf.
        """
        logits = np.asarray(logits, dtype=np.float64).copy()
        if n < 2 or len(prev_tokens) < n - 1:
            return logits

        prev = list(prev_tokens)
        # Build dict: (n-1)-gram prefix -> set of next tokens seen
        ngram_dict: Dict[Tuple[int, ...], Set[int]] = {}
        for i in range(len(prev) - n + 1):
            prefix = tuple(prev[i : i + n - 1])
            continuation = prev[i + n - 1]
            ngram_dict.setdefault(prefix, set()).add(continuation)

        # Current trailing (n-1) tokens
        tail = tuple(prev[-(n - 1) :])
        banned = ngram_dict.get(tail, set())
        for tok in banned:
            if 0 <= tok < len(logits):
                logits[tok] = -np.inf
        return logits


# ---------------------------------------------------------------------------
# 3. TextUtils
# ---------------------------------------------------------------------------

class TextUtils:
    """Lightweight NLP helpers (no heavy dependencies)."""

    _SENTENCE_RE = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z"])|(?<=[.!?])$', re.MULTILINE
    )
    _WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?")

    # -- tokenisation ---------------------------------------------------------

    @staticmethod
    def word_tokenize(text: str) -> List[str]:
        """Simple regex-based word tokeniser.

        Splits on whitespace / punctuation, keeps contractions and numbers.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[str]
            Tokens in order.
        """
        return TextUtils._WORD_RE.findall(text)

    @staticmethod
    def sentence_split(text: str) -> List[str]:
        """Heuristic sentence splitter.

        Splits on sentence-ending punctuation followed by whitespace and a
        capital letter.  Not perfect but adequate for readability metrics.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[str]
            Sentences.
        """
        text = text.strip()
        if not text:
            return []
        parts = TextUtils._SENTENCE_RE.split(text)
        return [s.strip() for s in parts if s.strip()]

    # -- n-grams --------------------------------------------------------------

    @staticmethod
    def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from a token sequence.

        Parameters
        ----------
        tokens : Sequence[str]
        n : int

        Returns
        -------
        List[Tuple[str, ...]]
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    # -- lexical diversity ----------------------------------------------------

    @staticmethod
    def type_token_ratio(text: str) -> float:
        """Type-token ratio (TTR) – unique tokens / total tokens.

        Parameters
        ----------
        text : str

        Returns
        -------
        float
            TTR in [0, 1].  Returns 0 for empty text.
        """
        tokens = TextUtils.word_tokenize(text.lower())
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def hapax_ratio(text: str) -> float:
        """Fraction of tokens that occur exactly once (hapax legomena).

        Parameters
        ----------
        text : str

        Returns
        -------
        float
        """
        tokens = TextUtils.word_tokenize(text.lower())
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        hapax = sum(1 for c in freq.values() if c == 1)
        return hapax / len(tokens)

    # -- readability ----------------------------------------------------------

    @staticmethod
    def syllable_count(word: str) -> int:
        """Estimate syllable count for an English word.

        Uses a simple vowel-group heuristic with adjustments for silent-e and
        common patterns.

        Parameters
        ----------
        word : str

        Returns
        -------
        int
            Estimated syllable count (minimum 1).
        """
        word = word.lower().strip()
        if not word:
            return 0
        # Remove trailing silent e
        if word.endswith("e") and len(word) > 2 and word[-2] not in "aeiou":
            word = word[:-1]
        # Count vowel groups
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in "aeiouy"
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)

    @staticmethod
    def flesch_reading_ease(text: str) -> float:
        """Flesch reading-ease score.

        206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

        Parameters
        ----------
        text : str

        Returns
        -------
        float
            Score (higher = easier to read).  Typical range 0–100.
        """
        sentences = TextUtils.sentence_split(text)
        words = TextUtils.word_tokenize(text)
        if not sentences or not words:
            return 0.0
        total_syllables = sum(TextUtils.syllable_count(w) for w in words)
        asl = len(words) / len(sentences)
        asw = total_syllables / len(words)
        return 206.835 - 1.015 * asl - 84.6 * asw

    # -- frequency distribution -----------------------------------------------

    @staticmethod
    def word_frequency_distribution(text: str) -> Counter:
        """Case-insensitive word frequency counter.

        Parameters
        ----------
        text : str

        Returns
        -------
        Counter
        """
        return Counter(TextUtils.word_tokenize(text.lower()))

    # -- set / string similarity ----------------------------------------------

    @staticmethod
    def jaccard_similarity(set_a: Set, set_b: Set) -> float:
        """Jaccard similarity |A ∩ B| / |A ∪ B|.

        Parameters
        ----------
        set_a, set_b : Set

        Returns
        -------
        float
            Similarity in [0, 1].
        """
        set_a = set(set_a)
        set_b = set(set_b)
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Levenshtein (edit) distance via dynamic programming.

        Parameters
        ----------
        s1, s2 : str

        Returns
        -------
        int
        """
        if len(s1) < len(s2):
            return TextUtils.levenshtein_distance(s2, s1)
        if not s2:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insert = prev_row[j + 1] + 1
                delete = curr_row[j] + 1
                substitute = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insert, delete, substitute))
            prev_row = curr_row
        return prev_row[-1]

    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> str:
        """Longest common subsequence (LCS) via DP.

        Parameters
        ----------
        s1, s2 : str

        Returns
        -------
        str
            The LCS string.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Back-track to recover the subsequence
        lcs_chars: List[str] = []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs_chars.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return "".join(reversed(lcs_chars))

    @staticmethod
    def compression_ratio(text: str) -> float:
        """Compression ratio = len(compressed) / len(original).

        Uses zlib (deflate).  A lower ratio indicates more redundancy.

        Parameters
        ----------
        text : str

        Returns
        -------
        float
            Ratio in (0, ∞).  Returns 0.0 for empty text.
        """
        raw = text.encode("utf-8")
        if not raw:
            return 0.0
        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)


# ---------------------------------------------------------------------------
# 4. CacheUtils
# ---------------------------------------------------------------------------

class LRUCache:
    """Thread-safe Least Recently Used cache with statistics.

    Parameters
    ----------
    max_size : int
        Maximum number of entries.
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -- public API -----------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value for *key*, moving it to the front."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Insert or update *key*.  Evicts LRU entry if at capacity."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                    self._evictions += 1
                self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def invalidate(self, key: str) -> bool:
        """Remove *key* from cache.  Returns True if it was present."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    @property
    def stats(self) -> Dict[str, int]:
        """Return hit/miss/eviction statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._cache),
            "max_size": self.max_size,
        }

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[Any]:
        with self._lock:
            return list(self._cache.values())

    def items(self) -> List[Tuple[str, Any]]:
        with self._lock:
            return list(self._cache.items())

    def __repr__(self) -> str:
        return (
            f"LRUCache(size={len(self._cache)}, max_size={self.max_size}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


class ContentAddressedCache:
    """Cache keyed by the SHA-256 hash of the content.

    Useful when the same computation can be identified purely by its inputs.

    Parameters
    ----------
    max_size : int
        Maximum entries.
    """

    def __init__(self, max_size: int = 4096) -> None:
        self._store = LRUCache(max_size)

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, data: bytes) -> Optional[Any]:
        key = self._hash(data)
        return self._store.get(key)

    def put(self, data: bytes, value: Any) -> str:
        """Store *value* keyed by hash of *data*.  Returns the key."""
        key = self._hash(data)
        self._store.put(key, value)
        return key

    def get_by_key(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def __contains__(self, data: bytes) -> bool:
        return self._hash(data) in self._store

    def __len__(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> Dict[str, int]:
        return self._store.stats

    def clear(self) -> None:
        self._store.clear()

    def __repr__(self) -> str:
        return f"ContentAddressedCache({self._store!r})"


class TTLCache:
    """Cache with per-entry time-to-live.

    Parameters
    ----------
    max_size : int
        Maximum entries (oldest expired entries are evicted first).
    default_ttl : float
        Default time-to-live in seconds.
    """

    def __init__(self, max_size: int = 1024, default_ttl: float = 300.0) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._expirations = 0

    def _is_expired(self, expiry: float) -> bool:
        return time.monotonic() > expiry

    def _evict_expired(self) -> None:
        """Remove all expired entries (call while holding lock)."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
            self._expirations += 1

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._evict_expired()
            if key in self._store:
                value, expiry = self._store[key]
                if not self._is_expired(expiry):
                    self._store.move_to_end(key)
                    self._hits += 1
                    return value
                del self._store[key]
                self._expirations += 1
            self._misses += 1
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + ttl
        with self._lock:
            self._evict_expired()
            if key in self._store:
                self._store.move_to_end(key)
            elif len(self._store) >= self.max_size:
                self._store.popitem(last=False)
            self._store[key] = (value, expiry)

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                _, expiry = self._store[key]
                return not self._is_expired(expiry)
            return False

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "expirations": self._expirations,
            "size": len(self),
            "max_size": self.max_size,
        }

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def __repr__(self) -> str:
        return (
            f"TTLCache(size={len(self)}, max_size={self.max_size}, "
            f"default_ttl={self.default_ttl:.1f}s, hit_rate={self.hit_rate:.2%})"
        )


class CacheUtils:
    """Factory and helpers for the caching subsystem."""

    @staticmethod
    def cache_key(*args: Any, **kwargs: Any) -> str:
        """Deterministic cache key from positional and keyword arguments.

        Converts arguments to their ``repr`` and hashes the result with
        SHA-256 (truncated to 16 hex chars for readability).

        Parameters
        ----------
        args, kwargs
            Arbitrary hashable/reprable arguments.

        Returns
        -------
        str
            Hex digest key.
        """
        parts: List[str] = [repr(a) for a in args]
        for k in sorted(kwargs):
            parts.append(f"{k}={kwargs[k]!r}")
        raw = "|".join(parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def memoize(
        max_size: int = 512,
    ) -> Callable:
        """Decorator that memoises function results in an LRU cache.

        Parameters
        ----------
        max_size : int
            Maximum cached results.

        Returns
        -------
        Callable
            Decorator.
        """

        def decorator(func: Callable) -> Callable:
            cache = LRUCache(max_size)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = CacheUtils.cache_key(*args, **kwargs)
                result = cache.get(key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                cache.put(key, result)
                return result

            wrapper.cache = cache  # type: ignore[attr-defined]
            wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
            return wrapper

        return decorator

    @staticmethod
    def make_lru(max_size: int = 1024) -> LRUCache:
        return LRUCache(max_size)

    @staticmethod
    def make_content_addressed(max_size: int = 4096) -> ContentAddressedCache:
        return ContentAddressedCache(max_size)

    @staticmethod
    def make_ttl(
        max_size: int = 1024, default_ttl: float = 300.0
    ) -> TTLCache:
        return TTLCache(max_size, default_ttl)


# ---------------------------------------------------------------------------
# 5. IOUtils
# ---------------------------------------------------------------------------

class IOUtils:
    """File I/O helpers with safety features (atomic writes, checksums)."""

    @staticmethod
    def save_json(
        data: Any,
        path: Union[str, Path],
        indent: int = 2,
        sort_keys: bool = False,
    ) -> None:
        """Atomically write JSON to *path*.

        Parameters
        ----------
        data : Any
            JSON-serialisable object.
        path : str | Path
            Destination file.
        indent : int
            Indentation width.
        sort_keys : bool
            Whether to sort dictionary keys.
        """
        path = Path(path)
        IOUtils.ensure_dir(path.parent)
        content = json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)
        IOUtils.atomic_write(path, content)

    @staticmethod
    def load_json(path: Union[str, Path]) -> Any:
        """Load JSON from *path*.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        Any
            Parsed JSON object.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def save_numpy(arr: np.ndarray, path: Union[str, Path]) -> None:
        """Save a NumPy array to disk (.npy).

        Parameters
        ----------
        arr : np.ndarray
        path : str | Path
        """
        path = Path(path)
        IOUtils.ensure_dir(path.parent)
        np.save(str(path), arr)

    @staticmethod
    def load_numpy(path: Union[str, Path]) -> np.ndarray:
        """Load a NumPy array from disk.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        np.ndarray
        """
        return np.load(str(path))

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Create directory (and parents) if it doesn't exist.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        Path
            The ensured directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def atomic_write(path: Union[str, Path], content: str) -> None:
        """Write *content* to *path* atomically via a temp file + rename.

        This avoids leaving a partially-written file if the process is
        interrupted.

        Parameters
        ----------
        path : str | Path
        content : str
        """
        path = Path(path)
        IOUtils.ensure_dir(path.parent)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @staticmethod
    def checksum(path: Union[str, Path], algorithm: str = "sha256") -> str:
        """Compute hex-digest checksum of a file.

        Parameters
        ----------
        path : str | Path
        algorithm : str
            Hash algorithm name (anything accepted by ``hashlib.new``).

        Returns
        -------
        str
            Hex digest.
        """
        h = hashlib.new(algorithm)
        with open(str(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def file_size_human(size_bytes: Union[int, float]) -> str:
        """Convert byte count to human-readable string.

        Parameters
        ----------
        size_bytes : int | float

        Returns
        -------
        str
            e.g. "1.23 MB"
        """
        size_bytes = float(size_bytes)
        if size_bytes < 0:
            return f"-{IOUtils.file_size_human(-size_bytes)}"
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        for unit in units[:-1]:
            if abs(size_bytes) < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {units[-1]}"

    @staticmethod
    def read_text(path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read entire text file."""
        with open(str(path), "r", encoding=encoding) as fh:
            return fh.read()

    @staticmethod
    def write_text(
        path: Union[str, Path], content: str, encoding: str = "utf-8"
    ) -> None:
        """Write text file (non-atomic)."""
        path = Path(path)
        IOUtils.ensure_dir(path.parent)
        with open(str(path), "w", encoding=encoding) as fh:
            fh.write(content)

    @staticmethod
    def list_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """List files matching *pattern* in *directory*.

        Parameters
        ----------
        directory : str | Path
        pattern : str
            Glob pattern.
        recursive : bool
            If True, search recursively.

        Returns
        -------
        List[Path]
        """
        d = Path(directory)
        if recursive:
            return sorted(d.rglob(pattern))
        return sorted(d.glob(pattern))


# ---------------------------------------------------------------------------
# 6. TimingUtils
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks.

    Usage::

        with Timer("training") as t:
            train_model()
        print(t.elapsed)
    """

    def __init__(self, name: str = "block") -> None:
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"Timer({self.name!r}, elapsed={self.elapsed:.4f}s)"


class RateLimiter:
    """Simple token-bucket rate limiter.

    Parameters
    ----------
    calls_per_sec : float
        Maximum number of calls per second.
    """

    def __init__(self, calls_per_sec: float) -> None:
        if calls_per_sec <= 0:
            raise ValueError("calls_per_sec must be positive")
        self.calls_per_sec = calls_per_sec
        self._interval = 1.0 / calls_per_sec
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the next call is allowed."""
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator to rate-limit a function."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.acquire()
            return func(*args, **kwargs)

        return wrapper

    def __repr__(self) -> str:
        return f"RateLimiter({self.calls_per_sec} calls/s)"


class ProgressTracker:
    """Lightweight progress tracker with callback support.

    Parameters
    ----------
    total : int
        Total number of items.
    callback : Callable, optional
        Called with ``(done, total, elapsed, eta)`` on each update.
    """

    def __init__(
        self,
        total: int,
        callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> None:
        self.total = total
        self.callback = callback
        self.done = 0
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def eta(self) -> float:
        if self.done == 0:
            return float("inf")
        return TimingUtils.eta_estimator(self.done, self.total, self.elapsed)

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    def update(self, n: int = 1) -> None:
        """Record *n* completed items."""
        self.done += n
        if self.callback is not None:
            self.callback(self.done, self.total, self.elapsed, self.eta)

    def __repr__(self) -> str:
        return (
            f"ProgressTracker({self.done}/{self.total}, "
            f"elapsed={self.elapsed:.1f}s, eta={self.eta:.1f}s)"
        )


class TimingUtils:
    """Factory methods for timing utilities."""

    Timer = Timer  # re-export for convenience

    @staticmethod
    def timer(name: str = "block") -> Timer:
        """Create a Timer context manager."""
        return Timer(name)

    @staticmethod
    def rate_limiter(calls_per_sec: float) -> RateLimiter:
        """Create a RateLimiter."""
        return RateLimiter(calls_per_sec)

    @staticmethod
    def progress_tracker(
        total: int,
        callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> ProgressTracker:
        """Create a ProgressTracker."""
        return ProgressTracker(total, callback)

    @staticmethod
    def eta_estimator(done: int, total: int, elapsed: float) -> float:
        """Estimate remaining time assuming constant throughput.

        Parameters
        ----------
        done : int
            Items completed so far.
        total : int
            Total items.
        elapsed : float
            Time elapsed so far (seconds).

        Returns
        -------
        float
            Estimated seconds remaining.
        """
        if done <= 0:
            return float("inf")
        rate = done / elapsed
        remaining = total - done
        return remaining / rate

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as ``HH:MM:SS.mmm``."""
        if seconds < 0:
            return "-" + TimingUtils.format_duration(-seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours:
            return f"{hours}:{minutes:02d}:{secs:06.3f}"
        if minutes:
            return f"{minutes}:{secs:06.3f}"
        return f"{secs:.3f}s"


# ---------------------------------------------------------------------------
# 7. NumpyUtils
# ---------------------------------------------------------------------------

class NumpyUtils:
    """Array manipulation helpers built on NumPy."""

    @staticmethod
    def safe_divide(
        a: np.ndarray,
        b: np.ndarray,
        default: float = 0.0,
    ) -> np.ndarray:
        """Element-wise division that returns *default* where *b* is zero.

        Parameters
        ----------
        a, b : np.ndarray
        default : float

        Returns
        -------
        np.ndarray
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        out = np.full_like(a, default)
        mask = b != 0
        out[mask] = a[mask] / b[mask]
        return out

    @staticmethod
    def clip_to_range(
        arr: np.ndarray,
        lo: float,
        hi: float,
    ) -> np.ndarray:
        """Clip array values to ``[lo, hi]``.

        Parameters
        ----------
        arr : np.ndarray
        lo, hi : float

        Returns
        -------
        np.ndarray
        """
        return np.clip(np.asarray(arr, dtype=np.float64), lo, hi)

    @staticmethod
    def running_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute running (rolling) mean with the given window size.

        Uses a cumulative-sum approach for O(n) performance.

        Parameters
        ----------
        arr : np.ndarray
            1-D input.
        window : int
            Window size.

        Returns
        -------
        np.ndarray
            Array of length ``len(arr) - window + 1``.
        """
        arr = np.asarray(arr, dtype=np.float64)
        if window < 1:
            raise ValueError("window must be >= 1")
        if window > len(arr):
            raise ValueError("window larger than array length")
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        return (cs[window:] - cs[:-window]) / window

    @staticmethod
    def running_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute running standard deviation with the given window size.

        Parameters
        ----------
        arr : np.ndarray
            1-D input.
        window : int
            Window size.

        Returns
        -------
        np.ndarray
            Array of length ``len(arr) - window + 1``.
        """
        arr = np.asarray(arr, dtype=np.float64)
        if window < 1:
            raise ValueError("window must be >= 1")
        if window > len(arr):
            raise ValueError("window larger than array length")
        mean = NumpyUtils.running_mean(arr, window)
        sq_mean = NumpyUtils.running_mean(arr ** 2, window)
        variance = sq_mean - mean ** 2
        np.maximum(variance, 0.0, out=variance)
        return np.sqrt(variance)

    @staticmethod
    def percentile_rank(arr: np.ndarray) -> np.ndarray:
        """Percentile rank of each element (fraction of values ≤ element).

        Parameters
        ----------
        arr : np.ndarray
            1-D input.

        Returns
        -------
        np.ndarray
            Values in [0, 1].
        """
        arr = np.asarray(arr, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.float64)
        ranks = NumpyUtils.rank_data(arr)
        return ranks / n

    @staticmethod
    def rank_data(arr: np.ndarray) -> np.ndarray:
        """Assign average ranks to data (1-based).

        Ties are broken by averaging the ranks of tied elements.

        Parameters
        ----------
        arr : np.ndarray
            1-D input.

        Returns
        -------
        np.ndarray
            Float array of ranks.
        """
        arr = np.asarray(arr, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.float64)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)

        # Average ranks for ties
        sorted_arr = arr[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_arr[j] == sorted_arr[i]:
                j += 1
            if j > i + 1:
                avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
                for k in range(i, j):
                    ranks[order[k]] = avg_rank
            i = j
        return ranks

    @staticmethod
    def bootstrap_sample(
        arr: np.ndarray,
        n: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw a bootstrap sample (with replacement).

        Parameters
        ----------
        arr : np.ndarray
            Source array (1-D).
        n : int
            Number of samples.
        seed : int, optional
            RNG seed.

        Returns
        -------
        np.ndarray
            Bootstrap sample of length *n*.
        """
        arr = np.asarray(arr)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(arr), size=n, replace=True)
        return arr[indices]

    @staticmethod
    def moving_max(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute the running maximum over a sliding window.

        Parameters
        ----------
        arr : np.ndarray
            1-D input.
        window : int

        Returns
        -------
        np.ndarray
            Array of length ``len(arr) - window + 1``.
        """
        arr = np.asarray(arr, dtype=np.float64)
        if window < 1 or window > len(arr):
            raise ValueError("invalid window size")
        n = len(arr) - window + 1
        result = np.empty(n, dtype=np.float64)
        for i in range(n):
            result[i] = arr[i : i + window].max()
        return result

    @staticmethod
    def moving_min(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute the running minimum over a sliding window."""
        arr = np.asarray(arr, dtype=np.float64)
        if window < 1 or window > len(arr):
            raise ValueError("invalid window size")
        n = len(arr) - window + 1
        result = np.empty(n, dtype=np.float64)
        for i in range(n):
            result[i] = arr[i : i + window].min()
        return result

    @staticmethod
    def z_score_normalize(arr: np.ndarray) -> np.ndarray:
        """Z-score normalisation (zero mean, unit variance).

        Parameters
        ----------
        arr : np.ndarray

        Returns
        -------
        np.ndarray
        """
        arr = np.asarray(arr, dtype=np.float64)
        mu = arr.mean()
        std = arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mu) / std

    @staticmethod
    def min_max_normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalisation to [0, 1]."""
        arr = np.asarray(arr, dtype=np.float64)
        lo, hi = arr.min(), arr.max()
        if lo == hi:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def cumulative_mean(arr: np.ndarray) -> np.ndarray:
        """Cumulative (expanding) mean."""
        arr = np.asarray(arr, dtype=np.float64)
        cs = np.cumsum(arr)
        counts = np.arange(1, len(arr) + 1, dtype=np.float64)
        return cs / counts

    @staticmethod
    def ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
        """Exponentially weighted moving average.

        Parameters
        ----------
        arr : np.ndarray
            1-D input.
        alpha : float
            Smoothing factor in (0, 1].

        Returns
        -------
        np.ndarray
        """
        arr = np.asarray(arr, dtype=np.float64)
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        result = np.empty_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
        return result


# ---------------------------------------------------------------------------
# 8. ConfigUtils
# ---------------------------------------------------------------------------

class ConfigUtils:
    """Utilities for hierarchical configuration management."""

    @staticmethod
    def merge_configs(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep-merge *override* into *base* (non-destructive).

        Nested dicts are merged recursively.  Non-dict values in *override*
        replace those in *base*.

        Parameters
        ----------
        base : dict
        override : dict

        Returns
        -------
        dict
            Merged configuration.
        """
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigUtils.merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def flatten_config(
        nested: Dict[str, Any],
        separator: str = ".",
        _prefix: str = "",
    ) -> Dict[str, Any]:
        """Flatten a nested dict to dot-separated keys.

        Example::

            {"a": {"b": 1, "c": 2}} -> {"a.b": 1, "a.c": 2}

        Parameters
        ----------
        nested : dict
        separator : str
        _prefix : str
            Internal; do not set.

        Returns
        -------
        dict
        """
        flat: Dict[str, Any] = {}
        for key, value in nested.items():
            full_key = f"{_prefix}{separator}{key}" if _prefix else key
            if isinstance(value, dict):
                flat.update(
                    ConfigUtils.flatten_config(value, separator, full_key)
                )
            else:
                flat[full_key] = value
        return flat

    @staticmethod
    def unflatten_config(
        flat: Dict[str, Any],
        separator: str = ".",
    ) -> Dict[str, Any]:
        """Unflatten dot-separated keys into a nested dict.

        Inverse of :meth:`flatten_config`.

        Parameters
        ----------
        flat : dict
        separator : str

        Returns
        -------
        dict
        """
        nested: Dict[str, Any] = {}
        for compound_key, value in flat.items():
            parts = compound_key.split(separator)
            d = nested
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        return nested

    @staticmethod
    def config_hash(config: Dict[str, Any]) -> str:
        """Deterministic hash of a configuration dict.

        The dict is serialised with sorted keys to ensure stability.

        Parameters
        ----------
        config : dict

        Returns
        -------
        str
            SHA-256 hex digest (first 16 chars).
        """
        serialised = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def validate_config_schema(
        config: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[str]:
        """Validate *config* against a simple schema specification.

        The *schema* dict mirrors the expected structure of *config*.  Leaf
        values are either:
        - A Python type (``int``, ``str``, ``float``, ``bool``, ``list``,
          ``dict``) – checks ``isinstance``.
        - A dict with keys ``"type"`` (required), ``"required"`` (bool,
          default True), ``"min"`` / ``"max"`` (numeric bounds), and
          ``"choices"`` (allowed values).

        Parameters
        ----------
        config : dict
        schema : dict

        Returns
        -------
        List[str]
            Error messages.  Empty list means valid.
        """
        errors: List[str] = []
        ConfigUtils._validate_recursive(config, schema, "", errors)
        return errors

    @staticmethod
    def _validate_recursive(
        config: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        for key, spec in schema.items():
            full_key = f"{path}.{key}" if path else key
            # Determine requirement
            required = True
            if isinstance(spec, dict) and "type" in spec:
                required = spec.get("required", True)

            if key not in config:
                if required:
                    errors.append(f"Missing required key: {full_key}")
                continue

            value = config[key]

            # If spec is a bare type
            if isinstance(spec, type):
                if not isinstance(value, spec):
                    errors.append(
                        f"{full_key}: expected {spec.__name__}, "
                        f"got {type(value).__name__}"
                    )
                continue

            # If spec is a nested dict that is itself a sub-schema (no "type" key)
            if isinstance(spec, dict) and "type" not in spec:
                if not isinstance(value, dict):
                    errors.append(
                        f"{full_key}: expected dict, got {type(value).__name__}"
                    )
                else:
                    ConfigUtils._validate_recursive(value, spec, full_key, errors)
                continue

            # Detailed spec dict
            if isinstance(spec, dict) and "type" in spec:
                expected_type = spec["type"]
                if isinstance(expected_type, type):
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"{full_key}: expected {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                        continue
                elif isinstance(expected_type, str):
                    type_map = {
                        "int": int,
                        "float": (int, float),
                        "str": str,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                    }
                    expected = type_map.get(expected_type)
                    if expected and not isinstance(value, expected):
                        errors.append(
                            f"{full_key}: expected {expected_type}, "
                            f"got {type(value).__name__}"
                        )
                        continue

                if "min" in spec and isinstance(value, (int, float)):
                    if value < spec["min"]:
                        errors.append(
                            f"{full_key}: value {value} < minimum {spec['min']}"
                        )
                if "max" in spec and isinstance(value, (int, float)):
                    if value > spec["max"]:
                        errors.append(
                            f"{full_key}: value {value} > maximum {spec['max']}"
                        )
                if "choices" in spec:
                    if value not in spec["choices"]:
                        errors.append(
                            f"{full_key}: value {value!r} not in "
                            f"allowed choices {spec['choices']}"
                        )

        # Warn about unknown keys
        schema_keys = set(schema.keys())
        config_keys = set(config.keys())
        for extra in config_keys - schema_keys:
            full_key = f"{path}.{extra}" if path else extra
            errors.append(f"Unknown key: {full_key}")

    @staticmethod
    def get_nested(
        config: Dict[str, Any],
        dotted_key: str,
        default: Any = None,
        separator: str = ".",
    ) -> Any:
        """Retrieve a value from a nested dict using a dot-separated key.

        Parameters
        ----------
        config : dict
        dotted_key : str
        default : Any
        separator : str

        Returns
        -------
        Any
        """
        parts = dotted_key.split(separator)
        current: Any = config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @staticmethod
    def set_nested(
        config: Dict[str, Any],
        dotted_key: str,
        value: Any,
        separator: str = ".",
    ) -> None:
        """Set a value in a nested dict using a dot-separated key.

        Intermediate dicts are created as needed.

        Parameters
        ----------
        config : dict
        dotted_key : str
        value : Any
        separator : str
        """
        parts = dotted_key.split(separator)
        d = config
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    @staticmethod
    def diff_configs(
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
    ) -> Dict[str, Tuple[Any, Any]]:
        """Return keys where *config_a* and *config_b* differ.

        Both configs are flattened first.  The result maps each differing key
        to ``(value_in_a_or_MISSING, value_in_b_or_MISSING)``.

        Parameters
        ----------
        config_a, config_b : dict

        Returns
        -------
        dict
        """
        _MISSING = object()
        flat_a = ConfigUtils.flatten_config(config_a)
        flat_b = ConfigUtils.flatten_config(config_b)
        all_keys = set(flat_a) | set(flat_b)
        diffs: Dict[str, Tuple[Any, Any]] = {}
        for k in sorted(all_keys):
            va = flat_a.get(k, _MISSING)
            vb = flat_b.get(k, _MISSING)
            if va != vb:
                diffs[k] = (
                    va if va is not _MISSING else "<MISSING>",
                    vb if vb is not _MISSING else "<MISSING>",
                )
        return diffs


# ---------------------------------------------------------------------------
# Convenience / top-level exports
# ---------------------------------------------------------------------------

def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Module-level shortcut for :meth:`MathUtils.log_softmax`."""
    return MathUtils.log_softmax(logits)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Module-level shortcut for :meth:`MathUtils.softmax`."""
    return MathUtils.softmax(logits, temperature)


def top_k_sample(logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
    """Sample from top-k filtered distribution."""
    masked = SamplingUtils.top_k_mask(logits, k)
    probs = MathUtils.softmax(masked, temperature)
    return SamplingUtils.categorical_sample(probs, 1)[0]


def top_p_sample(logits: np.ndarray, p: float, temperature: float = 1.0) -> int:
    """Sample from nucleus-filtered distribution."""
    masked = SamplingUtils.top_p_mask(logits, p)
    probs = MathUtils.softmax(masked, temperature)
    return SamplingUtils.categorical_sample(probs, 1)[0]


def typical_sample(
    logits: np.ndarray, mass: float = 0.9, temperature: float = 1.0
) -> int:
    """Sample from typical-filtered distribution."""
    masked = SamplingUtils.typical_mask(logits, mass)
    probs = MathUtils.softmax(masked, temperature)
    return SamplingUtils.categorical_sample(probs, 1)[0]


# ---------------------------------------------------------------------------
# ALL public names
# ---------------------------------------------------------------------------

__all__ = [
    # classes
    "MathUtils",
    "SamplingUtils",
    "TextUtils",
    "CacheUtils",
    "LRUCache",
    "ContentAddressedCache",
    "TTLCache",
    "IOUtils",
    "TimingUtils",
    "Timer",
    "RateLimiter",
    "ProgressTracker",
    "NumpyUtils",
    "ConfigUtils",
    # convenience functions
    "log_softmax",
    "softmax",
    "top_k_sample",
    "top_p_sample",
    "typical_sample",
]
