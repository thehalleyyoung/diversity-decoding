"""
Production-grade deduplication for text collections.

Provides exact, near-duplicate, and semantic deduplication using MinHash
LSH and embedding-based approaches. All implementations rely only on
numpy/scipy and the standard library.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import string
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \\1 ", text)
    return [t for t in _WS_RE.split(text) if t]


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _word_ngrams(text: str, n: int = 2) -> Set[Tuple[str, ...]]:
    tokens = _tokenize(text)
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------
# Simple TF-IDF vectorizer (self-contained)
# ---------------------------------------------------------------------------


class _SimpleTFIDF:
    """Minimal TF-IDF vectorizer for embedding-based deduplication."""

    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None

    def fit(self, texts: Sequence[str]) -> "_SimpleTFIDF":
        doc_freq: Counter = Counter()
        for text in texts:
            tokens = set(_tokenize(text))
            for t in tokens:
                doc_freq[t] += 1
        # Build vocabulary from tokens appearing in >= 2 docs (or all if small)
        min_df = 2 if len(texts) > 10 else 1
        self._vocab = {
            tok: i
            for i, (tok, df) in enumerate(
                sorted(doc_freq.items())
            )
            if df >= min_df
        }
        n = len(texts)
        self._idf = np.zeros(len(self._vocab))
        for tok, idx in self._vocab.items():
            self._idf[idx] = math.log((n + 1) / (doc_freq[tok] + 1)) + 1
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), len(self._vocab)))
        for i, text in enumerate(texts):
            tf = Counter(_tokenize(text))
            for tok, count in tf.items():
                if tok in self._vocab:
                    idx = self._vocab[tok]
                    vecs[i, idx] = count * self._idf[idx]
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# ---------------------------------------------------------------------------
# Core deduplication functions
# ---------------------------------------------------------------------------


def deduplicate(
    texts: List[str],
    threshold: float = 0.85,
    method: str = "ngram",
) -> List[str]:
    """Remove near-duplicates, keeping the first occurrence.

    Parameters
    ----------
    texts : list of str
        Input texts.
    threshold : float
        Jaccard similarity above which two texts are considered duplicates.
    method : str
        ``"ngram"`` (word bigram Jaccard) or ``"char"`` (character trigram).

    Returns
    -------
    list of str
        Deduplicated texts in original order.
    """
    if not texts:
        return []

    if method == "char":
        shingles = [_char_ngrams(t, 3) for t in texts]
    else:
        shingles = [_word_ngrams(t, 2) for t in texts]

    keep: List[int] = [0]
    for i in range(1, len(texts)):
        is_dup = False
        for j in keep:
            if _jaccard(shingles[i], shingles[j]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    return [texts[i] for i in keep]


def near_duplicate_clusters(
    texts: List[str],
    threshold: float = 0.9,
    method: str = "ngram",
) -> List[List[int]]:
    """Cluster texts into groups of near-duplicates.

    Returns a list of clusters, where each cluster is a list of indices
    into the original *texts* list.
    """
    if not texts:
        return []

    if method == "char":
        shingles = [_char_ngrams(t, 3) for t in texts]
    else:
        shingles = [_word_ngrams(t, 2) for t in texts]

    n = len(texts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(shingles[i], shingles[j]) >= threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    return list(clusters.values())


def incremental_dedup(
    existing: List[str],
    new: str,
    threshold: float = 0.85,
    method: str = "ngram",
) -> bool:
    """Check whether *new* is novel relative to *existing* texts.

    Returns ``True`` if the new text is sufficiently different from all
    existing texts (i.e., should be kept).
    """
    if not existing:
        return True

    if method == "char":
        new_sh = _char_ngrams(new, 3)
        for ex in existing:
            if _jaccard(new_sh, _char_ngrams(ex, 3)) >= threshold:
                return False
    else:
        new_sh = _word_ngrams(new, 2)
        for ex in existing:
            if _jaccard(new_sh, _word_ngrams(ex, 2)) >= threshold:
                return False
    return True


# ---------------------------------------------------------------------------
# MinHashDeduplicator — LSH-based for large scale
# ---------------------------------------------------------------------------


class MinHashDeduplicator:
    """Locality-Sensitive Hashing deduplicator using MinHash signatures.

    Efficient for large collections where O(n²) pairwise comparison is
    infeasible.

    Parameters
    ----------
    num_perm : int
        Number of hash permutations (signature length).
    bands : int
        Number of LSH bands. More bands = higher recall, lower precision.
    threshold : float
        Jaccard similarity threshold for duplicate detection.
    ngram_size : int
        Character n-gram size for shingling.
    """

    def __init__(
        self,
        num_perm: int = 128,
        bands: int = 16,
        threshold: float = 0.85,
        ngram_size: int = 3,
    ):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self.threshold = threshold
        self.ngram_size = ngram_size

        # Generate hash function params: h(x) = (a*x + b) % p
        self._max_hash = (1 << 32) - 1
        self._prime = 4294967311  # next prime after 2^32
        rng = np.random.RandomState(42)
        self._a = rng.randint(1, self._prime, size=num_perm).astype(np.uint64)
        self._b = rng.randint(0, self._prime, size=num_perm).astype(np.uint64)

        # LSH index: band -> {bucket_hash: set of doc ids}
        self._buckets: List[Dict[int, Set[int]]] = [
            defaultdict(set) for _ in range(bands)
        ]
        self._signatures: List[np.ndarray] = []
        self._texts: List[str] = []

    def _shingle(self, text: str) -> Set[int]:
        """Convert text to a set of hashed shingles."""
        text = text.lower().strip()
        n = self.ngram_size
        shingles = set()
        for i in range(len(text) - n + 1):
            h = struct.unpack(
                "<I",
                hashlib.md5(text[i : i + n].encode()).digest()[:4],
            )[0]
            shingles.add(h)
        return shingles

    def _minhash(self, shingles: Set[int]) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""
        sig = np.full(self.num_perm, self._max_hash, dtype=np.uint64)
        for s in shingles:
            hashes = (self._a * np.uint64(s) + self._b) % np.uint64(self._prime)
            sig = np.minimum(sig, hashes)
        return sig

    def _lsh_insert(self, doc_id: int, sig: np.ndarray) -> None:
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(sig[start:end].tobytes())
            self._buckets[band_idx][band_hash].add(doc_id)

    def _lsh_query(self, sig: np.ndarray) -> Set[int]:
        """Find candidate duplicates via LSH."""
        candidates: Set[int] = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(sig[start:end].tobytes())
            candidates |= self._buckets[band_idx].get(band_hash, set())
        return candidates

    def _jaccard_estimate(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))

    def fit(self, texts: List[str]) -> "MinHashDeduplicator":
        """Index all texts for deduplication queries."""
        self._texts = list(texts)
        self._signatures = []
        self._buckets = [defaultdict(set) for _ in range(self.bands)]

        for i, text in enumerate(texts):
            shingles = self._shingle(text)
            sig = self._minhash(shingles)
            self._signatures.append(sig)
            self._lsh_insert(i, sig)
        return self

    def is_duplicate(self, text: str) -> bool:
        """Check if *text* is a near-duplicate of any indexed text."""
        shingles = self._shingle(text)
        sig = self._minhash(shingles)
        candidates = self._lsh_query(sig)
        for cand_id in candidates:
            if self._jaccard_estimate(sig, self._signatures[cand_id]) >= self.threshold:
                return True
        return False

    def add(self, text: str) -> bool:
        """Add *text* to the index. Returns True if novel (not a duplicate)."""
        is_dup = self.is_duplicate(text)
        doc_id = len(self._texts)
        self._texts.append(text)
        shingles = self._shingle(text)
        sig = self._minhash(shingles)
        self._signatures.append(sig)
        self._lsh_insert(doc_id, sig)
        return not is_dup

    def deduplicate(self) -> List[str]:
        """Return deduplicated texts from the indexed collection."""
        if not self._texts:
            return []
        keep: List[int] = []
        seen_sigs: List[np.ndarray] = []
        for i in range(len(self._texts)):
            is_dup = False
            for prev_sig in seen_sigs:
                if self._jaccard_estimate(self._signatures[i], prev_sig) >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(i)
                seen_sigs.append(self._signatures[i])
        return [self._texts[i] for i in keep]

    def find_duplicates(self) -> List[List[int]]:
        """Return clusters of duplicate indices."""
        return near_duplicate_clusters(
            self._texts, threshold=self.threshold, method="char"
        )


# ---------------------------------------------------------------------------
# EmbeddingDeduplicator — semantic dedup via TF-IDF cosine
# ---------------------------------------------------------------------------


class EmbeddingDeduplicator:
    """Semantic deduplication using TF-IDF cosine similarity.

    For production use with real embeddings, pass a custom ``embed_fn``.

    Parameters
    ----------
    threshold : float
        Cosine similarity above which texts are considered duplicates.
    embed_fn : callable, optional
        Custom embedding function ``List[str] -> np.ndarray``.
        If None, uses built-in TF-IDF.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        self.threshold = threshold
        self._embed_fn = embed_fn
        self._tfidf = _SimpleTFIDF()
        self._embeddings: Optional[np.ndarray] = None
        self._texts: List[str] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        self._tfidf.fit(texts)
        return self._tfidf.transform(texts)

    def fit(self, texts: List[str]) -> "EmbeddingDeduplicator":
        """Compute embeddings for all texts."""
        self._texts = list(texts)
        self._embeddings = self._embed(texts)
        return self

    def deduplicate(self) -> List[str]:
        """Return deduplicated texts using cosine similarity."""
        if not self._texts:
            return []
        if self._embeddings is None:
            self.fit(self._texts)

        emb = self._embeddings
        n = emb.shape[0]
        keep: List[int] = [0]

        for i in range(1, n):
            is_dup = False
            for j in keep:
                sim = float(np.dot(emb[i], emb[j]))
                if sim >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(i)

        return [self._texts[i] for i in keep]

    def is_duplicate(self, text: str) -> bool:
        """Check if *text* is a semantic duplicate of any indexed text."""
        if not self._texts or self._embeddings is None:
            return False
        new_emb = self._embed([text] + self._texts)
        query_vec = new_emb[0]
        for i in range(self._embeddings.shape[0]):
            sim = float(np.dot(query_vec, self._embeddings[i]))
            if sim >= self.threshold:
                return True
        return False

    def find_clusters(self) -> List[List[int]]:
        """Return clusters of semantically similar texts."""
        if not self._texts or self._embeddings is None:
            return []

        emb = self._embeddings
        n = emb.shape[0]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(emb[i], emb[j]))
                if sim >= self.threshold:
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        return list(clusters.values())
