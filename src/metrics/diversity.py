"""
Comprehensive diversity metrics for the Diversity Decoding Arena.

Provides a suite of metrics that measure lexical, syntactic, semantic,
and behavioral diversity across sets of generated texts. All metrics
are self-contained with no external NLP library dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import string
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist, squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WHITESPACE_RE = re.compile(r"\s+")


def tokenize_simple(text: str) -> List[str]:
    """Whitespace + punctuation tokenisation (lowercased)."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \1 ", text)
    tokens = _WHITESPACE_RE.split(text)
    return [t for t in tokens if t]


def extract_ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    """Return a list of n-gram tuples from *tokens*."""
    if n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """Jaccard overlap of n-grams between two texts."""
    tokens_a = tokenize_simple(text_a)
    tokens_b = tokenize_simple(text_b)
    set_a = set(extract_ngrams(tokens_a, n))
    set_b = set(extract_ngrams(tokens_b, n))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def pairwise_distances(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Return a square pairwise-distance matrix."""
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], embeddings.shape[0]))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_mask = (norms.flatten() < 1e-12)
    if metric == "cosine" and np.any(zero_mask):
        embeddings = embeddings.copy()
        embeddings[zero_mask] = 1e-12
    condensed = pdist(embeddings, metric=metric)
    return squareform(condensed)


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Return (mean, (lower, upper)) via bootstrap resampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, (0.0, 0.0)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), (lower, upper)


def tfidf_embeddings(texts: List[str], max_features: int = 5000) -> np.ndarray:
    """Compute TF-IDF embeddings without sklearn.

    Returns an (n_texts, n_features) matrix.
    """
    tokenised = [tokenize_simple(t) for t in texts]
    # Build vocabulary from unigrams + bigrams, capped at *max_features*
    doc_freq: Counter = Counter()
    term_freq_per_doc: List[Counter] = []
    for tokens in tokenised:
        tf: Counter = Counter()
        seen: Set[str] = set()
        for i, tok in enumerate(tokens):
            tf[tok] += 1
            if tok not in seen:
                doc_freq[tok] += 1
                seen.add(tok)
            if i > 0:
                bigram = tokens[i - 1] + " " + tok
                tf[bigram] += 1
                if bigram not in seen:
                    doc_freq[bigram] += 1
                    seen.add(bigram)
        term_freq_per_doc.append(tf)

    # Select top-k features by document frequency
    vocab_items = doc_freq.most_common(max_features)
    vocab: Dict[str, int] = {term: idx for idx, (term, _) in enumerate(vocab_items)}
    n_docs = len(texts)
    n_feat = len(vocab)
    if n_feat == 0:
        return np.zeros((n_docs, 1), dtype=np.float64)

    mat = np.zeros((n_docs, n_feat), dtype=np.float64)
    for doc_idx, tf in enumerate(term_freq_per_doc):
        doc_len = sum(tf.values()) or 1
        for term, count in tf.items():
            if term in vocab:
                col = vocab[term]
                tf_val = count / doc_len
                idf_val = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
                mat[doc_idx, col] = tf_val * idf_val
    # L2-normalise rows
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    row_norms[row_norms < 1e-12] = 1.0
    mat /= row_norms
    return mat


def ngram_hash_embeddings(
    texts: List[str], n: int = 3, dim: int = 256
) -> np.ndarray:
    """Character n-gram hashing-trick embeddings.

    Each text is represented as a fixed-dim vector via the hashing trick
    over character n-grams.
    """
    mat = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        text_lower = text.lower()
        padded = f"<{text_lower}>"
        for j in range(len(padded) - n + 1):
            gram = padded[j : j + n]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1 if (h // dim) % 2 == 0 else -1
            mat[i, idx] += sign
    # L2-normalise
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    row_norms[row_norms < 1e-12] = 1.0
    mat /= row_norms
    return mat


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DiversityMetric(ABC):
    """Abstract base class for all diversity metrics."""

    @abstractmethod
    def compute(self, generation_set: List[str]) -> float:
        """Compute the metric on a set of generated texts."""
        ...

    def compute_with_ci(
        self,
        generation_set: List[str],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, Tuple[float, float]]:
        """Compute the metric with a bootstrap confidence interval.

        Default implementation: bootstrap over leave-one-out subsets.
        """
        self.validate_input(generation_set)
        n = len(generation_set)
        rng = np.random.default_rng(42)
        scores: List[float] = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            subset = [generation_set[i] for i in indices]
            try:
                scores.append(self.compute(subset))
            except Exception:
                continue
        if not scores:
            val = self.compute(generation_set)
            return val, (val, val)
        mean_val = float(np.mean(scores))
        alpha = 1.0 - confidence
        lo = float(np.percentile(scores, 100 * alpha / 2))
        hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        return mean_val, (lo, hi)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        ...

    @property
    def description(self) -> str:
        return ""

    def validate_input(self, generation_set: List[str]) -> None:
        """Raise ``ValueError`` if the input is invalid."""
        if not isinstance(generation_set, (list, tuple)):
            raise ValueError("generation_set must be a list of strings.")
        if len(generation_set) < 2:
            raise ValueError("Need at least 2 texts to measure diversity.")
        for i, t in enumerate(generation_set):
            if not isinstance(t, str):
                raise ValueError(f"Element {i} is not a string: {type(t)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# 1. Self-BLEU
# ---------------------------------------------------------------------------


class SelfBLEU(DiversityMetric):
    """Self-BLEU: average BLEU of each sentence against the rest.

    SelfBLEU(G) = (1/n) Σ_i BLEU(y_i, G \\ {y_i})

    Lower values indicate more diverse outputs.
    """

    def __init__(
        self,
        max_order: int = 4,
        smoothing_function: Optional[str] = "floor",
        smooth_epsilon: float = 0.1,
    ) -> None:
        self.max_order = max_order
        self.smoothing_function = smoothing_function
        self.smooth_epsilon = smooth_epsilon

    # -- public API --

    @property
    def name(self) -> str:
        return f"self_bleu_{self.max_order}"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def description(self) -> str:
        return (
            "Average BLEU score of each generation against the rest. "
            "Lower values indicate more lexical diversity."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        per_sample = self.compute_per_sample(texts)
        return float(np.mean(per_sample))

    def compute_per_sample(self, texts: List[str]) -> List[float]:
        """Return per-sequence self-BLEU scores."""
        self.validate_input(texts)
        tokenised = [self._tokenize(t) for t in texts]
        n = len(tokenised)
        scores: List[float] = []
        for i in range(n):
            hypothesis = tokenised[i]
            references = [tokenised[j] for j in range(n) if j != i]
            scores.append(self._sentence_bleu(hypothesis, references, self.max_order))
        return scores

    # -- BLEU internals --

    def _sentence_bleu(
        self,
        hypothesis: List[str],
        references: List[List[str]],
        max_order: int,
    ) -> float:
        if not hypothesis:
            return 0.0
        if not references:
            return 0.0

        hyp_len = len(hypothesis)
        # Closest reference length for brevity penalty
        ref_lens = [len(r) for r in references]
        closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))

        log_bleu = 0.0
        all_positive = True
        for n in range(1, max_order + 1):
            hyp_ngrams = self._extract_ngrams(hypothesis, n)
            ref_ngrams_union: Counter = Counter()
            for ref in references:
                ref_ng = self._extract_ngrams(ref, n)
                for ng, cnt in ref_ng.items():
                    ref_ngrams_union[ng] = max(ref_ngrams_union[ng], cnt)
            clipped, total = self._modified_precision(hyp_ngrams, ref_ngrams_union, n)

            if total == 0:
                precision = 0.0
            else:
                precision = clipped / total

            # Smoothing
            if precision == 0.0:
                all_positive = False
                if self.smoothing_function == "floor":
                    precision = self.smooth_epsilon / (total if total > 0 else 1)
                elif self.smoothing_function == "add1":
                    precision = (clipped + 1) / (total + 1)
                else:
                    precision = 1e-12

            if precision > 0:
                log_bleu += (1.0 / max_order) * math.log(precision)
            else:
                log_bleu += (1.0 / max_order) * math.log(1e-12)

        bp = self._brevity_penalty(hyp_len, closest_ref_len)
        return bp * math.exp(log_bleu)

    @staticmethod
    def _modified_precision(
        hyp_ngrams: Counter, ref_ngrams: Counter, n: int
    ) -> Tuple[int, int]:
        """Clipped precision counts for order *n*."""
        clipped = 0
        total = 0
        for ng, count in hyp_ngrams.items():
            total += count
            clipped += min(count, ref_ngrams.get(ng, 0))
        return clipped, total

    @staticmethod
    def _brevity_penalty(hyp_len: int, closest_ref_len: int) -> float:
        if hyp_len == 0:
            return 0.0
        if hyp_len >= closest_ref_len:
            return 1.0
        return math.exp(1 - closest_ref_len / hyp_len)

    @staticmethod
    def _extract_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(extract_ngrams(tokens, n))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return tokenize_simple(text)


# ---------------------------------------------------------------------------
# 2. Distinct-N
# ---------------------------------------------------------------------------


class DistinctN(DiversityMetric):
    """Distinct-N: ratio of unique n-grams to total n-grams.

    Higher values indicate more lexical diversity.
    """

    def __init__(self, n: int = 2) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

    @property
    def name(self) -> str:
        return f"distinct_{self.n}"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            f"Ratio of unique {self.n}-grams to total {self.n}-grams. "
            "Higher means more diverse."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        unique, total = self._count_ngrams(texts, self.n)
        if total == 0:
            return 0.0
        return unique / total

    def compute_per_n(
        self, texts: List[str], n_values: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """Compute Distinct-N for multiple values of *n*."""
        self.validate_input(texts)
        if n_values is None:
            n_values = [1, 2, 3, 4]
        results: Dict[int, float] = {}
        for n_val in n_values:
            unique, total = self._count_ngrams(texts, n_val)
            results[n_val] = unique / total if total > 0 else 0.0
        return results

    @staticmethod
    def _count_ngrams(texts: List[str], n: int) -> Tuple[int, int]:
        """Return (unique_count, total_count) of n-grams across *texts*."""
        all_ngrams: Counter = Counter()
        for text in texts:
            tokens = tokenize_simple(text)
            for ng in extract_ngrams(tokens, n):
                all_ngrams[ng] += 1
        unique = len(all_ngrams)
        total = sum(all_ngrams.values())
        return unique, total


# ---------------------------------------------------------------------------
# 3. N-Gram Entropy
# ---------------------------------------------------------------------------


class NGramEntropy(DiversityMetric):
    """Shannon entropy over the n-gram frequency distribution.

    H_n = -Σ p(g) log₂ p(g)
    """

    def __init__(self, n: int = 2, base: float = 2.0) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.base = base

    @property
    def name(self) -> str:
        return f"ngram_entropy_{self.n}"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            f"Shannon entropy of the {self.n}-gram distribution. "
            "Higher means more diverse."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        dist = self._build_ngram_distribution(texts, self.n)
        return self._shannon_entropy(dist)

    def compute_per_n(
        self, texts: List[str], n_values: Optional[List[int]] = None
    ) -> Dict[int, float]:
        self.validate_input(texts)
        if n_values is None:
            n_values = [1, 2, 3, 4]
        results: Dict[int, float] = {}
        for n_val in n_values:
            dist = self._build_ngram_distribution(texts, n_val)
            results[n_val] = self._shannon_entropy(dist)
        return results

    def _build_ngram_distribution(
        self, texts: List[str], n: int
    ) -> Dict[tuple, float]:
        counts: Counter = Counter()
        for text in texts:
            tokens = tokenize_simple(text)
            for ng in extract_ngrams(tokens, n):
                counts[ng] += 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {ng: c / total for ng, c in counts.items()}

    def _shannon_entropy(self, distribution: Dict[Any, float]) -> float:
        if not distribution:
            return 0.0
        ent = 0.0
        log_base = math.log(self.base)
        for p in distribution.values():
            if p > 0:
                ent -= p * (math.log(p) / log_base)
        return ent

    def _conditional_entropy(self, texts: List[str], n: int) -> float:
        """H(n-gram | (n-1)-gram) = H_n - H_{n-1}."""
        if n < 2:
            return self.compute(texts)
        h_n = self._shannon_entropy(self._build_ngram_distribution(texts, n))
        h_n1 = self._shannon_entropy(self._build_ngram_distribution(texts, n - 1))
        return max(h_n - h_n1, 0.0)


# ---------------------------------------------------------------------------
# 4. Embedding Pairwise Distance
# ---------------------------------------------------------------------------


class EmbeddingPairwiseDistance(DiversityMetric):
    """Average pairwise distance in embedding space.

    EPD = (2 / n(n-1)) Σ_{i<j} d(e(y_i), e(y_j))
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_method: str = "tfidf",
        max_features: int = 5000,
        hash_dim: int = 256,
        ngram_n: int = 3,
    ) -> None:
        if distance_metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        if embedding_method not in ("tfidf", "ngram_hash", "random"):
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
        self.distance_metric = distance_metric
        self.embedding_method = embedding_method
        self.max_features = max_features
        self.hash_dim = hash_dim
        self.ngram_n = ngram_n

    @property
    def name(self) -> str:
        return f"epd_{self.embedding_method}_{self.distance_metric}"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Average pairwise distance in embedding space. "
            "Higher means more semantic diversity."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        mat = self.compute_pairwise_matrix(texts)
        n = len(texts)
        if n < 2:
            return 0.0
        upper = mat[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    def compute_pairwise_matrix(self, texts: List[str]) -> np.ndarray:
        embeddings = self._compute_embeddings(texts)
        return pairwise_distances(embeddings, metric=self.distance_metric)

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.embedding_method == "tfidf":
            return tfidf_embeddings(texts, max_features=self.max_features)
        elif self.embedding_method == "ngram_hash":
            return ngram_hash_embeddings(texts, n=self.ngram_n, dim=self.hash_dim)
        elif self.embedding_method == "random":
            return self._random_embeddings(texts)
        raise ValueError(f"Unknown embedding method: {self.embedding_method}")

    @staticmethod
    def _random_embeddings(texts: List[str], dim: int = 128) -> np.ndarray:
        """Deterministic pseudo-random embeddings seeded by text content."""
        mat = np.zeros((len(texts), dim), dtype=np.float64)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            mat[i] = rng.standard_normal(dim)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        mat /= norms
        return mat

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 1.0
        return 1.0 - dot / (na * nb)

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# 5. Vendi Score
# ---------------------------------------------------------------------------


class VendiScore(DiversityMetric):
    """Vendi Score — effective number of distinct items.

    VS = exp(-Σ λ_i log λ_i) where λ_i are eigenvalues of the
    normalised kernel matrix K/n.
    """

    def __init__(
        self,
        kernel_type: str = "cosine",
        bandwidth: float = 1.0,
        embedding_method: str = "tfidf",
        max_features: int = 5000,
        ngram_n: int = 3,
        hash_dim: int = 256,
    ) -> None:
        if kernel_type not in ("cosine", "rbf", "ngram"):
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.embedding_method = embedding_method
        self.max_features = max_features
        self.ngram_n = ngram_n
        self.hash_dim = hash_dim

    @property
    def name(self) -> str:
        return f"vendi_score_{self.kernel_type}"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Effective number of distinct items via matrix entropy. "
            "Higher means more diversity."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        K = self._build_kernel_matrix(texts)
        Kn = self._normalize_kernel(K)
        eigvals = eigh(Kn, eigvals_only=True)
        eigvals = np.clip(eigvals, 0.0, None)
        eigvals = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
        entropy = self._matrix_entropy(eigvals)
        return math.exp(entropy)

    def _build_kernel_matrix(self, texts: List[str]) -> np.ndarray:
        if self.kernel_type == "ngram":
            return self._ngram_kernel(texts, self.ngram_n)
        emb = self._get_embeddings(texts)
        if self.kernel_type == "cosine":
            return self._cosine_kernel(emb)
        elif self.kernel_type == "rbf":
            return self._rbf_kernel(emb, self.bandwidth)
        raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.embedding_method == "tfidf":
            return tfidf_embeddings(texts, self.max_features)
        elif self.embedding_method == "ngram_hash":
            return ngram_hash_embeddings(texts, self.ngram_n, self.hash_dim)
        raise ValueError(f"Unknown embedding method: {self.embedding_method}")

    @staticmethod
    def _normalize_kernel(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        if n == 0:
            return K
        return K / n

    @staticmethod
    def _matrix_entropy(eigenvalues: np.ndarray) -> float:
        """Von Neumann entropy: -Σ λ_i log λ_i."""
        ent = 0.0
        for lam in eigenvalues:
            if lam > 1e-12:
                ent -= lam * math.log(lam)
        return ent

    @staticmethod
    def _rbf_kernel(embeddings: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
        sq_dists = squareform(pdist(embeddings, metric="sqeuclidean"))
        return np.exp(-sq_dists / (2.0 * bandwidth**2))

    @staticmethod
    def _cosine_kernel(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normed = embeddings / norms
        K = normed @ normed.T
        np.clip(K, -1.0, 1.0, out=K)
        return K

    @staticmethod
    def _ngram_kernel(texts: List[str], n: int = 3) -> np.ndarray:
        """Jaccard-similarity kernel over character n-grams."""
        n_texts = len(texts)
        ngram_sets: List[Set[str]] = []
        for text in texts:
            t = text.lower()
            padded = f"<{t}>"
            grams = {padded[i : i + n] for i in range(len(padded) - n + 1)}
            ngram_sets.append(grams)
        K = np.zeros((n_texts, n_texts), dtype=np.float64)
        for i in range(n_texts):
            K[i, i] = 1.0
            for j in range(i + 1, n_texts):
                inter = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                sim = inter / union if union > 0 else 1.0
                K[i, j] = sim
                K[j, i] = sim
        return K


# ---------------------------------------------------------------------------
# 6. Parse-Tree Diversity (heuristic approximation)
# ---------------------------------------------------------------------------

# Simple POS-tag-like heuristics (no external tagger required)
_DETERMINERS = frozenset(
    "a an the this that these those my your his her its our their some any no every each".split()
)
_PREPOSITIONS = frozenset(
    "in on at to for with by from of into through during before after above below between among about against".split()
)
_CONJUNCTIONS = frozenset("and but or nor for yet so".split())
_PRONOUNS = frozenset(
    "i me my mine we us our ours you your yours he him his she her hers it its they them their theirs".split()
)
_AUX_VERBS = frozenset(
    "is am are was were be been being have has had do does did will would shall should may might can could must".split()
)
_QUESTION_WORDS = frozenset("who whom whose what which when where why how".split())
_ADVERBS_COMMON = frozenset(
    "very really quite rather too also just still already even never always often sometimes usually".split()
)

_CLAUSE_PATTERNS = [
    re.compile(r"\b(if|unless|although|though|while|because|since|when|where|after|before)\b", re.I),
    re.compile(r"\b(that|which|who|whom|whose)\b", re.I),
    re.compile(r",\s*(and|but|or|yet|so)\s+", re.I),
    re.compile(r"[;:]", re.I),
]

_SENTENCE_END_RE = re.compile(r"[.!?]+")


def _heuristic_pos(token: str) -> str:
    """Very rough POS approximation for a lowercased token."""
    t = token.lower()
    if t in _DETERMINERS:
        return "DET"
    if t in _PREPOSITIONS:
        return "PREP"
    if t in _CONJUNCTIONS:
        return "CONJ"
    if t in _PRONOUNS:
        return "PRON"
    if t in _AUX_VERBS:
        return "AUX"
    if t in _QUESTION_WORDS:
        return "WH"
    if t in _ADVERBS_COMMON:
        return "ADV"
    if t.endswith("ly") and len(t) > 3:
        return "ADV"
    if t.endswith("ing") and len(t) > 4:
        return "VERB_ING"
    if t.endswith("ed") and len(t) > 3:
        return "VERB_ED"
    if t.endswith("tion") or t.endswith("ment") or t.endswith("ness") or t.endswith("ity"):
        return "NOUN_SUFFIX"
    if t.endswith("ous") or t.endswith("ive") or t.endswith("ful") or t.endswith("less") or t.endswith("able"):
        return "ADJ_SUFFIX"
    if t[0].isupper() if t else False:
        return "PROPER"
    if all(c in string.punctuation for c in t):
        return "PUNCT"
    if t.isdigit():
        return "NUM"
    return "WORD"


class ParseTreeDiversity(DiversityMetric):
    """Syntactic structural diversity via heuristic parse features."""

    def __init__(self, pattern_weight: float = 0.5) -> None:
        self.pattern_weight = pattern_weight

    @property
    def name(self) -> str:
        return "parse_tree_diversity"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Syntactic diversity measured via approximate parse-tree features. "
            "Higher means more structurally diverse."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        n = len(texts)

        # Feature-based diversity
        features = [self._approximate_parse_features(t) for t in texts]
        feat_dists: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                feat_dists.append(self._tree_edit_distance_approx(features[i], features[j]))
        feat_div = float(np.mean(feat_dists)) if feat_dists else 0.0

        # Pattern-based diversity
        pat_div = self._pattern_based_diversity(texts)

        return (1 - self.pattern_weight) * feat_div + self.pattern_weight * pat_div

    def _approximate_parse_features(self, text: str) -> np.ndarray:
        """Construct a feature vector approximating syntactic structure."""
        tokens = tokenize_simple(text)
        if not tokens:
            return np.zeros(30, dtype=np.float64)

        pos_tags = [_heuristic_pos(t) for t in tokens]
        tag_set = [
            "DET", "PREP", "CONJ", "PRON", "AUX", "WH", "ADV",
            "VERB_ING", "VERB_ED", "NOUN_SUFFIX", "ADJ_SUFFIX",
            "PROPER", "PUNCT", "NUM", "WORD",
        ]
        tag_counts = Counter(pos_tags)
        total_tokens = len(tokens)

        # POS distribution features (15)
        pos_feats = np.array(
            [tag_counts.get(t, 0) / total_tokens for t in tag_set],
            dtype=np.float64,
        )

        # POS bigram features (top-5 diversity measure)
        pos_bigrams = extract_ngrams(pos_tags, 2)
        unique_pos_bi = len(set(pos_bigrams))
        total_pos_bi = max(len(pos_bigrams), 1)

        # Sentence-level features
        sentences = _SENTENCE_END_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = max(len(sentences), 1)
        avg_sent_len = total_tokens / n_sentences

        # Clause markers
        clause_count = sum(
            len(pat.findall(text)) for pat in _CLAUSE_PATTERNS
        )

        # Depth proxy: nested punctuation pairs
        depth = 0
        max_depth = 0
        for ch in text:
            if ch in "([{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch in ")]}":
                depth = max(0, depth - 1)

        # Question structures
        is_question = 1.0 if text.strip().endswith("?") else 0.0
        wh_count = tag_counts.get("WH", 0)

        structural_feats = np.array([
            unique_pos_bi / total_pos_bi,        # POS bigram diversity
            avg_sent_len / 50.0,                  # normalised avg sentence length
            clause_count / n_sentences,            # clauses per sentence
            max_depth / 5.0,                       # nesting depth
            n_sentences / 20.0,                    # number of sentences (normalised)
            is_question,
            wh_count / total_tokens,
            tag_counts.get("CONJ", 0) / total_tokens,
            tag_counts.get("PREP", 0) / total_tokens,
            tag_counts.get("AUX", 0) / total_tokens,
            tag_counts.get("PUNCT", 0) / total_tokens,
            len(set(tokens)) / total_tokens,       # type-token ratio
            total_tokens / 100.0,                  # length feature
            sum(1 for t in tokens if len(t) > 6) / total_tokens,  # long-word ratio
            np.std([len(t) for t in tokens]) / 10.0 if len(tokens) > 1 else 0.0,  # word-len variability
        ], dtype=np.float64)

        return np.concatenate([pos_feats, structural_feats])

    @staticmethod
    def _tree_edit_distance_approx(features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Approximate tree edit distance via feature-vector distance."""
        diff = features_a - features_b
        return float(np.sqrt(np.dot(diff, diff)))

    @staticmethod
    def _extract_syntactic_patterns(text: str) -> List[str]:
        """Extract regex-based clause / structure patterns."""
        patterns_found: List[str] = []
        tokens = tokenize_simple(text)
        pos_tags = [_heuristic_pos(t) for t in tokens]

        # POS trigram patterns
        for tri in extract_ngrams(pos_tags, 3):
            patterns_found.append("_".join(tri))

        # Clause boundary patterns
        for pat in _CLAUSE_PATTERNS:
            for m in pat.finditer(text):
                patterns_found.append(f"CLAUSE:{m.group().strip().lower()}")

        # Sentence-shape patterns (first 3 POS tags of each sentence)
        sents = _SENTENCE_END_RE.split(text)
        for sent in sents:
            s_tokens = tokenize_simple(sent)
            s_pos = [_heuristic_pos(t) for t in s_tokens[:3]]
            if s_pos:
                patterns_found.append("START:" + "_".join(s_pos))

        return patterns_found

    def _pattern_based_diversity(self, texts: List[str]) -> float:
        """Diversity measured by Jaccard distance over syntactic patterns."""
        pattern_sets = [set(self._extract_syntactic_patterns(t)) for t in texts]
        n = len(texts)
        dists: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                union = len(pattern_sets[i] | pattern_sets[j])
                if union == 0:
                    dists.append(0.0)
                else:
                    inter = len(pattern_sets[i] & pattern_sets[j])
                    dists.append(1.0 - inter / union)
        return float(np.mean(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# 7. Behavioral Diversity
# ---------------------------------------------------------------------------

_CONTENT_POS = frozenset(["WORD", "VERB_ING", "VERB_ED", "NOUN_SUFFIX", "ADJ_SUFFIX", "PROPER", "NUM"])


class BehavioralDiversity(DiversityMetric):
    """Behavioral diversity via determinant of kernel over behaviour descriptors.

    BD = det(K_β) where K_β is a kernel matrix over behaviour vectors.
    """

    def __init__(
        self,
        descriptor_functions: Optional[List[Callable[[str], float]]] = None,
        kernel_bandwidth: float = 1.0,
    ) -> None:
        if descriptor_functions is not None:
            self._descriptor_fns = descriptor_functions
        else:
            self._descriptor_fns = [
                self._length_feature,
                self._vocabulary_richness,
                self._sentence_complexity,
                self._lexical_density,
                self._readability_score,
            ]
        self.kernel_bandwidth = kernel_bandwidth

    @property
    def name(self) -> str:
        return "behavioral_diversity"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Functional diversity of outputs measured via behavioural descriptors. "
            "Higher means more behaviourally diverse."
        )

    def compute(self, texts: List[str]) -> float:
        self.validate_input(texts)
        bv = self._compute_behavior_vectors(texts)
        n = len(texts)
        # Build RBF kernel matrix
        sq_dists = squareform(pdist(bv, metric="sqeuclidean"))
        K = np.exp(-sq_dists / (2.0 * self.kernel_bandwidth**2))
        # Regularise slightly for numerical stability
        K += np.eye(n) * 1e-8
        # log-det for numerical stability
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            return 0.0
        # Normalise by n so comparable across set sizes
        return float(logdet / n)

    def _compute_behavior_vectors(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        d = len(self._descriptor_fns)
        mat = np.zeros((n, d), dtype=np.float64)
        for i, text in enumerate(texts):
            for j, fn in enumerate(self._descriptor_fns):
                try:
                    mat[i, j] = fn(text)
                except Exception:
                    mat[i, j] = 0.0
        # z-score normalise columns
        for j in range(d):
            col = mat[:, j]
            std = np.std(col)
            if std > 1e-12:
                mat[:, j] = (col - np.mean(col)) / std
            else:
                mat[:, j] = 0.0
        return mat

    @staticmethod
    def _length_feature(text: str) -> float:
        """Normalised character length."""
        return len(text) / 1000.0

    @staticmethod
    def _vocabulary_richness(text: str) -> float:
        """Type-token ratio."""
        tokens = tokenize_simple(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def _sentence_complexity(text: str) -> float:
        """Average sentence length × clause density."""
        sentences = _SENTENCE_END_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        total_tokens = sum(len(tokenize_simple(s)) for s in sentences)
        avg_sent_len = total_tokens / len(sentences)
        clause_markers = sum(len(pat.findall(text)) for pat in _CLAUSE_PATTERNS)
        clause_density = clause_markers / max(len(sentences), 1)
        return avg_sent_len * (1 + clause_density) / 100.0

    @staticmethod
    def _lexical_density(text: str) -> float:
        """Ratio of content words to total words."""
        tokens = tokenize_simple(text)
        if not tokens:
            return 0.0
        content = sum(1 for t in tokens if _heuristic_pos(t) in _CONTENT_POS)
        return content / len(tokens)

    @staticmethod
    def _readability_score(text: str) -> float:
        """Flesch-Kincaid approximation (normalised to ~[0, 1])."""
        tokens = tokenize_simple(text)
        sentences = _SENTENCE_END_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_words = len(tokens) or 1
        n_sents = max(len(sentences), 1)
        # Approximate syllable count: count vowel groups per word
        n_syllables = 0
        for tok in tokens:
            vowels = re.findall(r"[aeiouy]+", tok.lower())
            n_syllables += max(len(vowels), 1)
        # Flesch Reading Ease (0-100 scale, we normalise)
        fre = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syllables / n_words)
        return max(0.0, min(fre / 100.0, 1.0))


# ---------------------------------------------------------------------------
# 8. Diversity Metric Suite
# ---------------------------------------------------------------------------


class DiversityMetricSuite:
    """A collection of diversity metrics that can be evaluated together."""

    def __init__(self, metrics: Optional[List[DiversityMetric]] = None) -> None:
        self._metrics: Dict[str, DiversityMetric] = {}
        if metrics:
            for m in metrics:
                self._metrics[m.name] = m

    def add_metric(self, metric: DiversityMetric) -> None:
        self._metrics[metric.name] = metric

    def remove_metric(self, name: str) -> None:
        if name in self._metrics:
            del self._metrics[name]
        else:
            raise KeyError(f"No metric named {name!r}")

    def get_metric(self, name: str) -> DiversityMetric:
        if name not in self._metrics:
            raise KeyError(f"No metric named {name!r}")
        return self._metrics[name]

    @property
    def metric_names(self) -> List[str]:
        return list(self._metrics.keys())

    def compute_all(self, texts: List[str]) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute(texts)
            except Exception as e:
                logger.warning("Metric %s failed: %s", name, e)
                results[name] = float("nan")
        return results

    def compute_all_with_ci(
        self,
        texts: List[str],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, Tuple[float, Tuple[float, float]]]:
        results: Dict[str, Tuple[float, Tuple[float, float]]] = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute_with_ci(texts, n_bootstrap, confidence)
            except Exception as e:
                logger.warning("Metric %s CI failed: %s", name, e)
                results[name] = (float("nan"), (float("nan"), float("nan")))
        return results

    def summary(self, texts: List[str]) -> dict:
        """Return a formatted summary dictionary."""
        raw = self.compute_all(texts)
        entries: List[dict] = []
        for name, value in raw.items():
            metric = self._metrics[name]
            entries.append({
                "metric": name,
                "value": round(value, 6),
                "higher_is_better": metric.higher_is_better,
                "description": metric.description,
            })
        return {
            "n_texts": len(texts),
            "n_metrics": len(self._metrics),
            "results": entries,
        }

    def __repr__(self) -> str:
        names = ", ".join(self._metrics.keys())
        return f"DiversityMetricSuite([{names}])"

    def __len__(self) -> int:
        return len(self._metrics)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def default_suite() -> DiversityMetricSuite:
    """Create a suite with sensible defaults for all built-in metrics."""
    return DiversityMetricSuite([
        SelfBLEU(max_order=4),
        DistinctN(n=1),
        DistinctN(n=2),
        DistinctN(n=3),
        NGramEntropy(n=2),
        EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf"),
        VendiScore(kernel_type="cosine"),
        ParseTreeDiversity(),
        BehavioralDiversity(),
    ])
