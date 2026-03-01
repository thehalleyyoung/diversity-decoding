"""
Embedding-based diversity metrics for the Diversity Decoding Arena.

Provides self-contained embedding methods (TF-IDF, BoW, hashing, n-gram,
random projection) and a rich set of diversity metrics built on top of them
including pairwise distances, clustering diversity, manifold diversity, and
a full embedding-space analyser.

No external NLP libraries are required – only numpy and scipy.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import string
import warnings
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
from scipy.linalg import eigh, svd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenisation helpers (mirrors diversity.py helpers)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenisation (lowercased)."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \\1 ", text)
    tokens = _WHITESPACE_RE.split(text)
    return [t for t in tokens if t]


def _extract_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Return n-gram tuples from *tokens*."""
    if n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _char_ngrams(text: str, n: int) -> List[str]:
    """Return character n-grams from *text*."""
    text = text.lower().strip()
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


# ---------------------------------------------------------------------------
# 1. Sentence Embedder hierarchy
# ---------------------------------------------------------------------------


class SentenceEmbedder(ABC):
    """Abstract base for all self-contained sentence embedding methods."""

    @abstractmethod
    def fit(self, texts: List[str]) -> "SentenceEmbedder":
        """Learn parameters from the corpus (if needed)."""
        ...

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) embedding matrix."""
        ...

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------------------
    # Utility helpers shared by sub-classes
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_rows(mat: np.ndarray) -> np.ndarray:
        """L2-normalise each row (zero rows stay zero)."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    @staticmethod
    def _clip_dim(mat: np.ndarray, max_dim: int) -> np.ndarray:
        """Truncate or pad columns to *max_dim*."""
        if mat.shape[1] >= max_dim:
            return mat[:, :max_dim]
        pad = np.zeros((mat.shape[0], max_dim - mat.shape[1]))
        return np.hstack([mat, pad])


# ---- TFIDFEmbedder -------------------------------------------------------


class TFIDFEmbedder(SentenceEmbedder):
    """TF-IDF vectorisation without external libraries.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size (most frequent terms kept).
    min_df : int
        Minimum document frequency for a term to be retained.
    max_df_ratio : float
        Maximum document-frequency ratio (0-1).  Terms appearing in more
        than this fraction of documents are dropped.
    sublinear_tf : bool
        If True, apply ``1 + log(tf)`` instead of raw term frequency.
    norm : str
        Row normalisation: ``"l2"`` (default) or ``"l1"`` or ``None``.
    ngram_range : tuple[int, int]
        Range of word n-grams to include, e.g. ``(1, 2)``.
    """

    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 1,
        max_df_ratio: float = 1.0,
        sublinear_tf: bool = True,
        norm: Optional[str] = "l2",
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.ngram_range = ngram_range

        # Learned state
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._fitted = False

    # ---- internal helpers ------------------------------------------------

    def _tokenize_doc(self, text: str) -> List[str]:
        """Tokenise and optionally generate n-grams."""
        tokens = _tokenize(text)
        all_terms: List[str] = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if n == 1:
                all_terms.extend(tokens)
            else:
                all_terms.extend("_".join(ng) for ng in _extract_ngrams(tokens, n))
        return all_terms

    def _build_vocab(self, tokenised_docs: List[List[str]]) -> Dict[str, int]:
        """Build vocabulary respecting min_df / max_df_ratio."""
        n_docs = len(tokenised_docs)
        doc_freq: Counter = Counter()
        total_freq: Counter = Counter()
        for doc_tokens in tokenised_docs:
            unique = set(doc_tokens)
            for t in unique:
                doc_freq[t] += 1
            total_freq.update(doc_tokens)

        max_df_abs = max(1, int(self.max_df_ratio * n_docs))
        filtered = {
            t: f
            for t, f in total_freq.items()
            if self.min_df <= doc_freq[t] <= max_df_abs
        }
        top = sorted(filtered.items(), key=lambda x: -x[1])[: self.max_features]
        return {term: idx for idx, (term, _) in enumerate(top)}

    def _compute_idf(
        self, tokenised_docs: List[List[str]], vocab: Dict[str, int]
    ) -> np.ndarray:
        """Standard IDF: log((1 + N) / (1 + df)) + 1."""
        n_docs = len(tokenised_docs)
        df = np.zeros(len(vocab))
        for doc_tokens in tokenised_docs:
            seen: Set[str] = set()
            for t in doc_tokens:
                if t in vocab and t not in seen:
                    df[vocab[t]] += 1
                    seen.add(t)
        return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    def _tf_vector(self, tokens: List[str]) -> np.ndarray:
        """Build a raw TF vector for one document."""
        vec = np.zeros(len(self._vocab))
        counts = Counter(tokens)
        for t, c in counts.items():
            if t in self._vocab:
                if self.sublinear_tf:
                    vec[self._vocab[t]] = 1.0 + math.log(c) if c > 0 else 0.0
                else:
                    vec[self._vocab[t]] = c
        return vec

    # ---- public API ------------------------------------------------------

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        tokenised = [self._tokenize_doc(t) for t in texts]
        self._vocab = self._build_vocab(tokenised)
        if not self._vocab:
            self._vocab = {"<UNK>": 0}
        self._idf = self._compute_idf(tokenised, self._vocab)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        n = len(texts)
        dim = len(self._vocab)
        mat = np.zeros((n, dim))
        for i, text in enumerate(texts):
            tokens = self._tokenize_doc(text)
            mat[i] = self._tf_vector(tokens) * self._idf  # type: ignore[operator]

        if self.norm == "l2":
            mat = self._normalise_rows(mat)
        elif self.norm == "l1":
            row_sums = np.abs(mat).sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            mat = mat / row_sums
        return mat

    @property
    def vocabulary(self) -> Dict[str, int]:
        return dict(self._vocab)

    @property
    def idf_vector(self) -> Optional[np.ndarray]:
        return self._idf.copy() if self._idf is not None else None


# ---- BagOfWordsEmbedder --------------------------------------------------


class BagOfWordsEmbedder(SentenceEmbedder):
    """Simple count-based bag-of-words embeddings.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    binary : bool
        If True, use binary (presence/absence) counts.
    norm : str | None
        Row normalisation (``"l2"``, ``"l1"``, or ``None``).
    ngram_range : tuple[int, int]
        Word n-gram range.
    """

    def __init__(
        self,
        max_features: int = 5000,
        binary: bool = False,
        norm: Optional[str] = "l2",
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> None:
        self.max_features = max_features
        self.binary = binary
        self.norm = norm
        self.ngram_range = ngram_range
        self._vocab: Dict[str, int] = {}
        self._fitted = False

    def _tokenize_doc(self, text: str) -> List[str]:
        tokens = _tokenize(text)
        terms: List[str] = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if n == 1:
                terms.extend(tokens)
            else:
                terms.extend("_".join(ng) for ng in _extract_ngrams(tokens, n))
        return terms

    def fit(self, texts: List[str]) -> "BagOfWordsEmbedder":
        freq: Counter = Counter()
        for t in texts:
            freq.update(self._tokenize_doc(t))
        top = sorted(freq.items(), key=lambda x: -x[1])[: self.max_features]
        self._vocab = {term: idx for idx, (term, _) in enumerate(top)}
        if not self._vocab:
            self._vocab = {"<UNK>": 0}
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        dim = len(self._vocab)
        mat = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            counts = Counter(self._tokenize_doc(text))
            for term, cnt in counts.items():
                if term in self._vocab:
                    mat[i, self._vocab[term]] = 1.0 if self.binary else float(cnt)
        if self.norm == "l2":
            mat = self._normalise_rows(mat)
        elif self.norm == "l1":
            row_sums = np.abs(mat).sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            mat = mat / row_sums
        return mat

    @property
    def vocabulary(self) -> Dict[str, int]:
        return dict(self._vocab)


# ---- HashEmbedder --------------------------------------------------------


class HashEmbedder(SentenceEmbedder):
    """Feature-hashing embedder for fixed-dimension representations.

    Uses the *hashing trick* to map arbitrary token strings to a
    fixed-width vector without maintaining a vocabulary.

    Parameters
    ----------
    n_features : int
        Dimensionality of the output vector.
    alternate_sign : bool
        If True, the sign of the hash is used to reduce collision bias.
    norm : str | None
        Row normalisation.
    ngram_range : tuple[int, int]
        Word n-gram range.
    """

    def __init__(
        self,
        n_features: int = 1024,
        alternate_sign: bool = True,
        norm: Optional[str] = "l2",
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> None:
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.norm = norm
        self.ngram_range = ngram_range

    @staticmethod
    def _hash_token(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def _tokenize_doc(self, text: str) -> List[str]:
        tokens = _tokenize(text)
        terms: List[str] = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if n == 1:
                terms.extend(tokens)
            else:
                terms.extend("_".join(ng) for ng in _extract_ngrams(tokens, n))
        return terms

    def fit(self, texts: List[str]) -> "HashEmbedder":
        # No fitting needed for hashing.
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.n_features))
        for i, text in enumerate(texts):
            for token in self._tokenize_doc(text):
                h = self._hash_token(token)
                idx = h % self.n_features
                if self.alternate_sign:
                    sign = 1.0 if (h // self.n_features) % 2 == 0 else -1.0
                else:
                    sign = 1.0
                mat[i, idx] += sign

        if self.norm == "l2":
            mat = self._normalise_rows(mat)
        elif self.norm == "l1":
            row_sums = np.abs(mat).sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            mat = mat / row_sums
        return mat


# ---- NGramEmbedder -------------------------------------------------------


class NGramEmbedder(SentenceEmbedder):
    """Character and/or word n-gram based embeddings.

    Supports character-level n-grams, word-level n-grams, or both.

    Parameters
    ----------
    char_ngram_range : tuple[int, int] | None
        Range of character n-gram sizes.  ``None`` disables char n-grams.
    word_ngram_range : tuple[int, int] | None
        Range of word n-gram sizes.  ``None`` disables word n-grams.
    max_features : int
        Maximum vocabulary size across all n-gram types.
    norm : str | None
        Row normalisation.
    use_idf : bool
        Weight n-gram counts by IDF.
    """

    def __init__(
        self,
        char_ngram_range: Optional[Tuple[int, int]] = (2, 4),
        word_ngram_range: Optional[Tuple[int, int]] = None,
        max_features: int = 5000,
        norm: Optional[str] = "l2",
        use_idf: bool = False,
    ) -> None:
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.max_features = max_features
        self.norm = norm
        self.use_idf = use_idf

        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._fitted = False

    def _extract_features(self, text: str) -> List[str]:
        """Extract all configured n-gram features from *text*."""
        features: List[str] = []
        if self.char_ngram_range is not None:
            lo, hi = self.char_ngram_range
            for n in range(lo, hi + 1):
                features.extend(f"c{n}:{g}" for g in _char_ngrams(text, n))
        if self.word_ngram_range is not None:
            tokens = _tokenize(text)
            lo, hi = self.word_ngram_range
            for n in range(lo, hi + 1):
                features.extend(
                    f"w{n}:{'_'.join(ng)}" for ng in _extract_ngrams(tokens, n)
                )
        return features

    def fit(self, texts: List[str]) -> "NGramEmbedder":
        freq: Counter = Counter()
        doc_freq: Counter = Counter()
        for text in texts:
            feats = self._extract_features(text)
            freq.update(feats)
            doc_freq.update(set(feats))

        top = sorted(freq.items(), key=lambda x: -x[1])[: self.max_features]
        self._vocab = {feat: idx for idx, (feat, _) in enumerate(top)}
        if not self._vocab:
            self._vocab = {"<UNK>": 0}

        if self.use_idf:
            n_docs = len(texts)
            idf = np.zeros(len(self._vocab))
            for feat, idx in self._vocab.items():
                df = doc_freq.get(feat, 0)
                idf[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            self._idf = idf

        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        dim = len(self._vocab)
        mat = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            counts = Counter(self._extract_features(text))
            for feat, cnt in counts.items():
                if feat in self._vocab:
                    mat[i, self._vocab[feat]] = float(cnt)

        if self._idf is not None:
            mat *= self._idf

        if self.norm == "l2":
            mat = self._normalise_rows(mat)
        elif self.norm == "l1":
            row_sums = np.abs(mat).sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            mat = mat / row_sums
        return mat

    @property
    def vocabulary(self) -> Dict[str, int]:
        return dict(self._vocab)


# ---- RandomProjectionEmbedder -------------------------------------------


class RandomProjectionEmbedder(SentenceEmbedder):
    """Projects a sparse high-dimensional representation to a dense
    low-dimensional one via a random Gaussian matrix (Johnson-Lindenstrauss).

    Parameters
    ----------
    base_embedder : SentenceEmbedder
        Source embedder producing the high-dimensional input.
    n_components : int
        Target dimensionality.
    density : float
        Fraction of non-zero entries in the random matrix.  ``1.0`` gives
        a dense Gaussian matrix; lower values give sparse random projection.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_embedder: Optional[SentenceEmbedder] = None,
        n_components: int = 128,
        density: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.base_embedder = base_embedder or TFIDFEmbedder()
        self.n_components = n_components
        self.density = max(0.01, min(1.0, density))
        self.seed = seed
        self._projection: Optional[np.ndarray] = None
        self._fitted = False

    def _make_projection(self, input_dim: int) -> np.ndarray:
        """Build the random projection matrix."""
        rng = np.random.RandomState(self.seed)
        if self.density >= 1.0:
            proj = rng.randn(input_dim, self.n_components)
        else:
            proj = np.zeros((input_dim, self.n_components))
            mask = rng.rand(input_dim, self.n_components) < self.density
            proj[mask] = rng.randn(int(mask.sum()))
        proj /= np.sqrt(self.n_components)
        return proj

    def fit(self, texts: List[str]) -> "RandomProjectionEmbedder":
        self.base_embedder.fit(texts)
        probe = self.base_embedder.transform(texts[:1])
        input_dim = probe.shape[1]
        self._projection = self._make_projection(input_dim)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted or self._projection is None:
            raise RuntimeError("Call fit() before transform().")
        sparse_mat = self.base_embedder.transform(texts)
        dense = sparse_mat @ self._projection
        return self._normalise_rows(dense)


# ---- WeightedEmbedder ----------------------------------------------------


class WeightedEmbedder(SentenceEmbedder):
    """Combine multiple embedding methods via weighted concatenation.

    Parameters
    ----------
    embedders : list[tuple[SentenceEmbedder, float]]
        List of ``(embedder, weight)`` pairs.
    norm : str | None
        Final row normalisation.
    """

    def __init__(
        self,
        embedders: Optional[List[Tuple[SentenceEmbedder, float]]] = None,
        norm: Optional[str] = "l2",
    ) -> None:
        if embedders is None:
            embedders = [
                (TFIDFEmbedder(), 1.0),
                (HashEmbedder(n_features=256), 0.5),
            ]
        self.embedders = embedders
        self.norm = norm
        self._fitted = False

    def fit(self, texts: List[str]) -> "WeightedEmbedder":
        for emb, _ in self.embedders:
            emb.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        parts: List[np.ndarray] = []
        for emb, w in self.embedders:
            part = emb.transform(texts)
            parts.append(part * w)
        mat = np.hstack(parts)
        if self.norm == "l2":
            mat = self._normalise_rows(mat)
        elif self.norm == "l1":
            row_sums = np.abs(mat).sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            mat = mat / row_sums
        return mat


# ---------------------------------------------------------------------------
# 2. Embedding Diversity Metrics
# ---------------------------------------------------------------------------


class EmbeddingDiversityMetric(ABC):
    """Abstract base for diversity metrics that operate on embedding matrices.

    All metrics accept an ``embedder`` that is used to convert raw texts to
    vectors.  If embeddings are already available they can be passed directly
    via :meth:`compute_from_embeddings`.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self.embedder = embedder or TFIDFEmbedder()

    def compute(self, texts: List[str]) -> float:
        """End-to-end: embed *texts* then compute metric."""
        embs = self.embedder.fit_transform(texts)
        return self.compute_from_embeddings(embs)

    @abstractmethod
    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        """Compute the metric from a pre-computed (n, d) embedding matrix."""
        ...


# ---- MeanPairwiseCosineDistance ------------------------------------------


class MeanPairwiseCosineDistance(EmbeddingDiversityMetric):
    """Average cosine distance between all pairs of embeddings.

    Cosine distance = 1 − cosine_similarity.  Higher values indicate
    greater diversity.
    """

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        # Normalise for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        # Pairwise cosine similarity via dot product
        sim = normed @ normed.T
        np.clip(sim, -1.0, 1.0, out=sim)
        dist = 1.0 - sim

        # Extract upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        return float(np.mean(dist[iu]))


# ---- MeanPairwiseEuclideanDistance ---------------------------------------


class MeanPairwiseEuclideanDistance(EmbeddingDiversityMetric):
    """Average L2 (Euclidean) distance between all pairs."""

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        dists = pdist(embeddings, metric="euclidean")
        return float(np.mean(dists))


# ---- MaxPairwiseDistance -------------------------------------------------


class MaxPairwiseDistance(EmbeddingDiversityMetric):
    """Maximum pairwise distance (diameter of the embedding cloud).

    Parameters
    ----------
    metric : str
        Distance metric passed to ``scipy.spatial.distance.pdist``.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metric: str = "cosine",
    ) -> None:
        super().__init__(embedder)
        self.metric = metric

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        dists = pdist(embeddings, metric=self.metric)
        return float(np.max(dists))


# ---- MinPairwiseDistance -------------------------------------------------


class MinPairwiseDistance(EmbeddingDiversityMetric):
    """Minimum pairwise distance – the *bottleneck* diversity.

    A high minimum pairwise distance means every pair of texts is at
    least somewhat different.

    Parameters
    ----------
    metric : str
        Distance metric for ``pdist``.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metric: str = "cosine",
    ) -> None:
        super().__init__(embedder)
        self.metric = metric

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        dists = pdist(embeddings, metric=self.metric)
        return float(np.min(dists))


# ---- CentroidDistance ----------------------------------------------------


class CentroidDistance(EmbeddingDiversityMetric):
    """Average distance from each embedding to the centroid.

    Parameters
    ----------
    metric : str
        ``"euclidean"`` or ``"cosine"``.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metric: str = "euclidean",
    ) -> None:
        super().__init__(embedder)
        self.metric = metric

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 1:
            return 0.0
        centroid = embeddings.mean(axis=0, keepdims=True)
        dists = cdist(embeddings, centroid, metric=self.metric).ravel()
        return float(np.mean(dists))


# ---- CoverageScore ------------------------------------------------------


class CoverageScore(EmbeddingDiversityMetric):
    """Fraction of the embedding space covered by the set of texts.

    The space is divided into a grid of hyper-cells (after PCA to a
    manageable number of dimensions) and the fraction of occupied cells
    is reported.

    Parameters
    ----------
    n_pca_dims : int
        Number of PCA dimensions for the grid.
    n_bins : int
        Number of bins per dimension.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_pca_dims: int = 3,
        n_bins: int = 10,
    ) -> None:
        super().__init__(embedder)
        self.n_pca_dims = n_pca_dims
        self.n_bins = n_bins

    @staticmethod
    def _pca_reduce(data: np.ndarray, k: int) -> np.ndarray:
        """Mean-centre and project onto the top-*k* principal components."""
        if data.shape[1] <= k:
            return data
        mean = data.mean(axis=0)
        centred = data - mean
        # Use SVD for numerical stability
        U, S, Vt = svd(centred, full_matrices=False)
        return centred @ Vt[:k].T

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        k = min(self.n_pca_dims, embeddings.shape[1], n)
        reduced = self._pca_reduce(embeddings, k)

        # Bin each dimension into n_bins quantile-based bins
        occupied: Set[Tuple[int, ...]] = set()
        bin_edges: List[np.ndarray] = []
        for d in range(reduced.shape[1]):
            col = reduced[:, d]
            edges = np.linspace(col.min(), col.max() + 1e-12, self.n_bins + 1)
            bin_edges.append(edges)

        for i in range(n):
            cell = tuple(
                int(np.searchsorted(bin_edges[d], reduced[i, d]) - 1)
                for d in range(reduced.shape[1])
            )
            occupied.add(cell)

        total_cells = self.n_bins ** reduced.shape[1]
        return float(len(occupied) / total_cells)


# ---- RadiusOfGyration ---------------------------------------------------


class RadiusOfGyration(EmbeddingDiversityMetric):
    """Root-mean-square distance of embeddings from their centroid.

    Also known as *radius of gyration* in polymer physics / information
    retrieval.  Measures how spread-out the embeddings are.
    """

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        centroid = embeddings.mean(axis=0)
        sq_dists = np.sum((embeddings - centroid) ** 2, axis=1)
        return float(np.sqrt(np.mean(sq_dists)))


# ---- ConvexHullVolume ---------------------------------------------------


class ConvexHullVolume(EmbeddingDiversityMetric):
    """Volume (or area) of the convex hull of embeddings in PCA-reduced space.

    Parameters
    ----------
    n_pca_dims : int
        Dimensionality for PCA reduction before computing the hull.
        Must be ≥ 2.
    log_scale : bool
        Return ``log(1 + volume)`` to keep numbers manageable.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_pca_dims: int = 3,
        log_scale: bool = True,
    ) -> None:
        super().__init__(embedder)
        self.n_pca_dims = max(2, n_pca_dims)
        self.log_scale = log_scale

    @staticmethod
    def _pca_reduce(data: np.ndarray, k: int) -> np.ndarray:
        if data.shape[1] <= k:
            return data
        mean = data.mean(axis=0)
        centred = data - mean
        U, S, Vt = svd(centred, full_matrices=False)
        return centred @ Vt[:k].T

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        k = min(self.n_pca_dims, embeddings.shape[1])
        # Need at least k+1 points for a k-dimensional hull
        if n < k + 1:
            return 0.0

        reduced = self._pca_reduce(embeddings, k)

        # Add tiny jitter to avoid degenerate configurations
        rng = np.random.RandomState(0)
        reduced = reduced + rng.randn(*reduced.shape) * 1e-10

        try:
            hull = ConvexHull(reduced)
            vol = hull.volume  # area in 2-D, volume in 3-D
        except Exception:
            return 0.0

        if self.log_scale:
            return float(math.log1p(abs(vol)))
        return float(vol)


# ---------------------------------------------------------------------------
# Distance matrix utilities
# ---------------------------------------------------------------------------


def _safe_pdist(embeddings: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distances, handling edge cases."""
    if embeddings.shape[0] < 2:
        return np.array([0.0])
    return pdist(embeddings, metric=metric)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return an (n, n) cosine-similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def _rbf_kernel(embeddings: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """Radial-basis-function (Gaussian) kernel matrix."""
    sq_dists = squareform(pdist(embeddings, "sqeuclidean"))
    if gamma is None:
        median_dist = np.median(sq_dists[sq_dists > 0]) if np.any(sq_dists > 0) else 1.0
        gamma = 1.0 / (2.0 * median_dist)
    return np.exp(-gamma * sq_dists)


# ---------------------------------------------------------------------------
# 3. Clustering Diversity
# ---------------------------------------------------------------------------


class ClusteringDiversityMetric(EmbeddingDiversityMetric):
    """Base for clustering-based diversity metrics."""
    pass


# ---- K-Means helpers (self-contained) ------------------------------------


def _kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple K-Means clustering.

    Returns
    -------
    labels : np.ndarray, shape (n,)
    centroids : np.ndarray, shape (k, d)
    """
    n, d = data.shape
    rng = np.random.RandomState(seed)

    # K-Means++ initialisation
    centroids = np.empty((k, d))
    idx = rng.randint(n)
    centroids[0] = data[idx]
    for c in range(1, k):
        sq_dist = cdist(data, centroids[:c], "sqeuclidean").min(axis=1)
        probs = sq_dist / sq_dist.sum()
        idx = rng.choice(n, p=probs)
        centroids[c] = data[idx]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = cdist(data, centroids, "sqeuclidean")
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = data[labels == c]
            if len(members) > 0:
                new_centroid = members.mean(axis=0)
                if np.linalg.norm(new_centroid - centroids[c]) < tol:
                    continue
                centroids[c] = new_centroid

    return labels, centroids


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score."""
    n = data.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= n:
        return 0.0

    dist_matrix = squareform(pdist(data, "euclidean"))
    silhouettes = np.zeros(n)

    for i in range(n):
        own_cluster = labels[i]
        same = labels == own_cluster
        if same.sum() <= 1:
            silhouettes[i] = 0.0
            continue
        a_i = dist_matrix[i, same].sum() / (same.sum() - 1)

        b_i = np.inf
        for lbl in unique_labels:
            if lbl == own_cluster:
                continue
            other = labels == lbl
            if other.sum() == 0:
                continue
            b_candidate = dist_matrix[i, other].mean()
            if b_candidate < b_i:
                b_i = b_candidate

        denom = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(silhouettes))


# ---- DBSCAN (self-contained) --------------------------------------------


def _dbscan(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 3,
) -> np.ndarray:
    """Simple DBSCAN implementation.

    Returns
    -------
    labels : np.ndarray, shape (n,)
        Cluster labels; ``-1`` means noise.
    """
    n = data.shape[0]
    dist_matrix = squareform(pdist(data, "euclidean"))
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = np.where(dist_matrix[i] <= eps)[0]

        if len(neighbours) < min_samples:
            continue  # noise

        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbours = np.where(dist_matrix[q] <= eps)[0]
                if len(q_neighbours) >= min_samples:
                    seed_set.extend(
                        int(x)
                        for x in q_neighbours
                        if x not in seed_set  # type: ignore[arg-type]
                    )
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


# ---- KMeansClusterDiversity ---------------------------------------------


class KMeansClusterDiversity(ClusteringDiversityMetric):
    """Diversity measured via effective number of K-Means clusters.

    Uses the *gap statistic* idea: given a fixed ``k``, the more evenly
    distributed points are across clusters, the higher the diversity.

    The score is the *entropy* of the cluster-size distribution normalised
    by ``log(k)`` (yielding a 0-1 score).

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iter : int
        K-Means iterations.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        k: int = 5,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(embedder)
        self.k = k
        self.max_iter = max_iter
        self.seed = seed

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        k = min(self.k, n)
        labels, _ = _kmeans(embeddings, k, self.max_iter, self.seed)

        counts = np.bincount(labels, minlength=k).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(k) if k > 1 else 1.0
        return entropy / max_entropy


# ---- SilhouetteDiversity ------------------------------------------------


class SilhouetteDiversity(ClusteringDiversityMetric):
    """Diversity as ``1 − silhouette_score``.

    A *low* silhouette score means clusters are not well separated, which
    correlates with *high* diversity (texts are spread out rather than
    forming tight groups).

    Parameters
    ----------
    k : int
        Number of K-Means clusters for the silhouette computation.
    seed : int
        Random seed for K-Means.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        k: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(embedder)
        self.k = k
        self.seed = seed

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 3:
            return 0.0
        k = min(self.k, n - 1)
        if k < 2:
            return 0.0
        labels, _ = _kmeans(embeddings, k, seed=self.seed)
        sil = _silhouette_score(embeddings, labels)
        # silhouette in [-1, 1]; map to diversity in [0, 1]
        return float((1.0 - sil) / 2.0)


# ---- DBSCANDiversity ----------------------------------------------------


class DBSCANDiversity(ClusteringDiversityMetric):
    """Diversity based on the number of clusters discovered by DBSCAN.

    More clusters ⇒ more diverse regions of the embedding space are
    populated.  The score is normalised as ``n_clusters / n``.

    Parameters
    ----------
    eps : float | None
        Neighbourhood radius.  If ``None``, it is estimated from the data
        (median of 3-NN distances).
    min_samples : int
        Minimum neighbours to form a core point.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        eps: Optional[float] = None,
        min_samples: int = 3,
    ) -> None:
        super().__init__(embedder)
        self.eps = eps
        self.min_samples = min_samples

    def _estimate_eps(self, embeddings: np.ndarray) -> float:
        """Heuristic: median of k-nearest-neighbour distances."""
        dists = squareform(pdist(embeddings, "euclidean"))
        np.fill_diagonal(dists, np.inf)
        k = min(self.min_samples, embeddings.shape[0] - 1)
        knn_dists = np.sort(dists, axis=1)[:, :k]
        return float(np.median(knn_dists[:, -1]))

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 3:
            return 0.0
        eps = self.eps if self.eps is not None else self._estimate_eps(embeddings)
        labels = _dbscan(embeddings, eps=eps, min_samples=self.min_samples)
        n_clusters = len(set(labels) - {-1})
        # Normalise: more clusters relative to n ⇒ higher diversity
        return float(n_clusters / n) if n > 0 else 0.0


# ---- SpectralClusterDiversity -------------------------------------------


class SpectralClusterDiversity(ClusteringDiversityMetric):
    """Eigenvalue-based diversity from a similarity kernel.

    Computes the RBF kernel matrix, finds its eigenvalues, and reports
    the *effective rank* (exponential of the Shannon entropy of the
    normalised eigenvalue spectrum).  Higher effective rank means the
    data spans more independent directions – higher diversity.

    Parameters
    ----------
    gamma : float | None
        RBF kernel parameter.  ``None`` for automatic median heuristic.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        gamma: Optional[float] = None,
    ) -> None:
        super().__init__(embedder)
        self.gamma = gamma

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        K = _rbf_kernel(embeddings, self.gamma)
        eigvals = eigh(K, eigvals_only=True)
        eigvals = np.maximum(eigvals, 0.0)

        total = eigvals.sum()
        if total < 1e-14:
            return 0.0
        probs = eigvals / total
        probs = probs[probs > 1e-14]
        entropy = -float(np.sum(probs * np.log(probs)))
        effective_rank = math.exp(entropy)
        # Normalise by n
        return float(effective_rank / n)


# ---------------------------------------------------------------------------
# 4. Manifold Diversity
# ---------------------------------------------------------------------------


class ManifoldDiversityMetric(EmbeddingDiversityMetric):
    """Base for manifold-based diversity metrics."""
    pass


# ---- IntrinsicDimensionEstimator -----------------------------------------


class IntrinsicDimensionEstimator(ManifoldDiversityMetric):
    """Estimate the intrinsic dimensionality of the embedding manifold.

    Uses two methods:
    - **MLE** (Levina & Bickel, 2004): local intrinsic dimension via
      maximum likelihood on nearest-neighbour distances.
    - **PCA eigenvalue ratio**: fraction of variance explained.

    The final score is the estimated dimension normalised by the ambient
    dimension, yielding a value in (0, 1].

    Parameters
    ----------
    method : str
        ``"mle"`` or ``"pca"`` (default ``"mle"``).
    k : int
        Number of nearest neighbours for the MLE method.
    variance_threshold : float
        Cumulative variance threshold for the PCA method.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        method: str = "mle",
        k: int = 5,
        variance_threshold: float = 0.95,
    ) -> None:
        super().__init__(embedder)
        self.method = method
        self.k = k
        self.variance_threshold = variance_threshold

    def _mle_id(self, embeddings: np.ndarray) -> float:
        """Maximum-likelihood intrinsic dimension (Levina & Bickel)."""
        n, d = embeddings.shape
        k = min(self.k, n - 1)
        if k < 2:
            return 1.0

        dist_matrix = squareform(pdist(embeddings, "euclidean"))
        np.fill_diagonal(dist_matrix, np.inf)

        estimates: List[float] = []
        for i in range(n):
            sorted_dists = np.sort(dist_matrix[i])[:k]
            sorted_dists = sorted_dists[sorted_dists > 1e-14]
            if len(sorted_dists) < 2:
                continue
            T_k = sorted_dists[-1]
            if T_k < 1e-14:
                continue
            log_ratios = np.log(T_k / sorted_dists[:-1])
            if log_ratios.sum() < 1e-14:
                continue
            m_hat = (len(log_ratios)) / log_ratios.sum()
            estimates.append(m_hat)

        if not estimates:
            return 1.0
        return float(np.mean(estimates))

    def _pca_id(self, embeddings: np.ndarray) -> float:
        """Intrinsic dimension via PCA cumulative variance."""
        mean = embeddings.mean(axis=0)
        centred = embeddings - mean
        cov = np.cov(centred.T)
        if cov.ndim < 2:
            return 1.0
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 0.0)
        total = eigvals.sum()
        if total < 1e-14:
            return 1.0
        cum = np.cumsum(eigvals) / total
        dim = int(np.searchsorted(cum, self.variance_threshold)) + 1
        return float(dim)

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n, d = embeddings.shape
        if n < 3:
            return 0.0

        if self.method == "pca":
            id_est = self._pca_id(embeddings)
        else:
            id_est = self._mle_id(embeddings)

        # Normalise by ambient dimension
        return float(min(id_est / d, 1.0))


# ---- LocalLinearityScore -------------------------------------------------


class LocalLinearityScore(ManifoldDiversityMetric):
    """Measure how *non-linearly* the embeddings are distributed.

    For each point, fit a local linear model to its *k* nearest neighbours
    and compute the reconstruction error.  High average error means the
    manifold is highly curved – suggesting diverse content.

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        k: int = 5,
    ) -> None:
        super().__init__(embedder)
        self.k = k

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n, d = embeddings.shape
        k = min(self.k, n - 1)
        if n < 3 or k < 2:
            return 0.0

        dist_matrix = squareform(pdist(embeddings, "euclidean"))
        np.fill_diagonal(dist_matrix, np.inf)

        errors: List[float] = []
        for i in range(n):
            nn_idx = np.argsort(dist_matrix[i])[:k]
            neighbours = embeddings[nn_idx]
            centre = neighbours.mean(axis=0)
            diff = neighbours - centre
            # Local PCA: fit 1-D linear model
            try:
                _, S, Vt = svd(diff, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            # Reconstruction with top component
            if len(S) < 1:
                continue
            proj = diff @ Vt[0]
            reconstructed = np.outer(proj, Vt[0]) + centre
            err = np.mean(np.linalg.norm(neighbours - reconstructed, axis=1))
            # Normalise by mean distance from centre
            avg_dist = np.mean(np.linalg.norm(diff, axis=1))
            if avg_dist > 1e-14:
                errors.append(err / avg_dist)

        if not errors:
            return 0.0
        # Higher error ⇒ more non-linear ⇒ higher diversity
        return float(np.mean(errors))


# ---- NeighborhoodDiversity ----------------------------------------------


class NeighborhoodDiversity(ManifoldDiversityMetric):
    """Diversity in local k-nearest neighbourhoods.

    Computes the average ratio of *unique* neighbours across all points.
    If every point shares the same neighbours, diversity is low; if
    neighbourhoods vary widely, diversity is high.

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        k: int = 5,
    ) -> None:
        super().__init__(embedder)
        self.k = k

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        k = min(self.k, n - 1)
        if n < 3 or k < 1:
            return 0.0

        dist_matrix = squareform(pdist(embeddings, "euclidean"))
        np.fill_diagonal(dist_matrix, np.inf)

        # Gather neighbourhood sets
        hoods: List[Set[int]] = []
        for i in range(n):
            nn = set(np.argsort(dist_matrix[i])[:k].tolist())
            hoods.append(nn)

        # Average Jaccard distance between neighbourhood sets
        total_jaccard = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(hoods[i] & hoods[j])
                union = len(hoods[i] | hoods[j])
                if union > 0:
                    total_jaccard += 1.0 - inter / union
                count += 1

        return float(total_jaccard / count) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Additional pairwise and distribution metrics
# ---------------------------------------------------------------------------


class MedianPairwiseDistance(EmbeddingDiversityMetric):
    """Median of all pairwise distances.

    More robust to outliers than the mean.

    Parameters
    ----------
    metric : str
        Distance metric for ``pdist``.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metric: str = "cosine",
    ) -> None:
        super().__init__(embedder)
        self.metric = metric

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        return float(np.median(pdist(embeddings, metric=self.metric)))


class DistanceVariance(EmbeddingDiversityMetric):
    """Variance of the pairwise distance distribution.

    High variance means some pairs are very similar while others are very
    different – indicating multi-modal diversity.

    Parameters
    ----------
    metric : str
        Distance metric.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metric: str = "cosine",
    ) -> None:
        super().__init__(embedder)
        self.metric = metric

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        dists = pdist(embeddings, metric=self.metric)
        return float(np.var(dists))


class DistanceEntropy(EmbeddingDiversityMetric):
    """Shannon entropy of the discretised pairwise distance distribution.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    metric : str
        Distance metric.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_bins: int = 30,
        metric: str = "cosine",
    ) -> None:
        super().__init__(embedder)
        self.n_bins = n_bins
        self.metric = metric

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        dists = pdist(embeddings, metric=self.metric)
        counts, _ = np.histogram(dists, bins=self.n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(self.n_bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0


class VendiScoreEmbedding(EmbeddingDiversityMetric):
    """Vendi Score computed from the embedding similarity kernel.

    The Vendi Score is ``exp(H(eigenvalues))`` where *H* is the Shannon
    entropy of the eigenvalue spectrum of the kernel matrix.

    Parameters
    ----------
    kernel : str
        ``"cosine"`` or ``"rbf"``.
    gamma : float | None
        RBF gamma (auto if ``None``).
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        kernel: str = "cosine",
        gamma: Optional[float] = None,
    ) -> None:
        super().__init__(embedder)
        self.kernel = kernel
        self.gamma = gamma

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 1.0

        if self.kernel == "cosine":
            K = _cosine_similarity_matrix(embeddings)
        else:
            K = _rbf_kernel(embeddings, self.gamma)

        eigvals = eigh(K, eigvals_only=True)
        eigvals = np.maximum(eigvals, 0.0)
        total = eigvals.sum()
        if total < 1e-14:
            return 1.0
        probs = eigvals / total
        probs = probs[probs > 1e-14]
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(math.exp(entropy))


class DeterminantalDiversity(EmbeddingDiversityMetric):
    """Log-determinant of the kernel matrix as a diversity measure.

    The determinant of a positive-definite kernel matrix is maximised
    when points are as spread out as possible (related to DPPs).

    Parameters
    ----------
    kernel : str
        ``"cosine"`` or ``"rbf"``.
    reg : float
        Regularisation added to the diagonal for numerical stability.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        kernel: str = "rbf",
        reg: float = 1e-6,
    ) -> None:
        super().__init__(embedder)
        self.kernel = kernel
        self.reg = reg

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        if self.kernel == "cosine":
            K = _cosine_similarity_matrix(embeddings)
        else:
            K = _rbf_kernel(embeddings)

        K += self.reg * np.eye(n)
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            return 0.0
        # Normalise by n to make comparable across set sizes
        return float(logdet / n)


class PairwiseManhattanDistance(EmbeddingDiversityMetric):
    """Mean pairwise Manhattan (L1) distance."""

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        return float(np.mean(pdist(embeddings, metric="cityblock")))


class PairwiseChebyshevDistance(EmbeddingDiversityMetric):
    """Mean pairwise Chebyshev (L∞) distance."""

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        return float(np.mean(pdist(embeddings, metric="chebyshev")))


# ---------------------------------------------------------------------------
# 5. Embedding Analyzer
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Result of PCA analysis."""
    coordinates: np.ndarray  # (n, n_components) projected data
    explained_variance_ratio: np.ndarray  # per-component ratio
    cumulative_variance: np.ndarray  # cumulative ratios
    components: np.ndarray  # (n_components, d) principal components
    singular_values: np.ndarray


@dataclass
class ClusterResult:
    """Result of clustering analysis."""
    labels: np.ndarray  # (n,) cluster assignments
    centroids: np.ndarray  # (k, d)
    sizes: np.ndarray  # (k,) number of members per cluster
    silhouette: float
    inertia: float  # within-cluster sum of squares


@dataclass
class OutlierResult:
    """Result of outlier detection."""
    is_outlier: np.ndarray  # (n,) boolean mask
    scores: np.ndarray  # (n,) outlierness scores
    threshold: float
    n_outliers: int


@dataclass
class EmbeddingStats:
    """Descriptive statistics of an embedding matrix."""
    n_samples: int
    n_dims: int
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float
    mean_pairwise_cosine: float
    mean_pairwise_euclidean: float
    effective_rank: float
    sparsity: float  # fraction of near-zero entries


@dataclass
class AnalysisReport:
    """Full analysis report from EmbeddingAnalyzer."""
    stats: EmbeddingStats
    pca: PCAResult
    clusters: ClusterResult
    outliers: OutlierResult
    diversity_scores: Dict[str, float]


class EmbeddingAnalyzer:
    """Comprehensive analyser for embedding spaces.

    Provides PCA visualisation coordinates, clustering analysis, outlier
    detection, and descriptive statistics – all without external ML
    libraries.

    Parameters
    ----------
    embedder : SentenceEmbedder
        Embedder used to convert texts to vectors.
    n_pca_components : int
        Number of PCA components for visualisation / analysis.
    n_clusters : int
        Number of clusters for K-Means.
    outlier_method : str
        ``"zscore"`` or ``"iqr"`` for outlier detection.
    outlier_threshold : float
        Threshold for the chosen outlier method.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_pca_components: int = 3,
        n_clusters: int = 5,
        outlier_method: str = "zscore",
        outlier_threshold: float = 2.5,
        seed: int = 42,
    ) -> None:
        self.embedder = embedder or TFIDFEmbedder()
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.seed = seed

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def pca(self, embeddings: np.ndarray) -> PCAResult:
        """Compute PCA decomposition."""
        n, d = embeddings.shape
        k = min(self.n_pca_components, d, n)
        mean = embeddings.mean(axis=0)
        centred = embeddings - mean

        U, S, Vt = svd(centred, full_matrices=False)
        total_var = np.sum(S ** 2) / (n - 1) if n > 1 else 1.0
        explained = (S[:k] ** 2) / (n - 1) if n > 1 else S[:k] ** 2
        evr = explained / total_var if total_var > 0 else np.zeros(k)

        coords = centred @ Vt[:k].T
        return PCAResult(
            coordinates=coords,
            explained_variance_ratio=evr,
            cumulative_variance=np.cumsum(evr),
            components=Vt[:k],
            singular_values=S[:k],
        )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster(self, embeddings: np.ndarray) -> ClusterResult:
        """K-Means clustering with diagnostics."""
        n = embeddings.shape[0]
        k = min(self.n_clusters, n)
        if k < 2:
            labels = np.zeros(n, dtype=int)
            centroids = embeddings.mean(axis=0, keepdims=True)
            return ClusterResult(
                labels=labels,
                centroids=centroids,
                sizes=np.array([n]),
                silhouette=0.0,
                inertia=0.0,
            )

        labels, centroids = _kmeans(embeddings, k, seed=self.seed)
        sizes = np.bincount(labels, minlength=k)
        sil = _silhouette_score(embeddings, labels) if n > k else 0.0

        # Inertia: sum of squared distances to assigned centroid
        inertia = 0.0
        for c in range(k):
            members = embeddings[labels == c]
            if len(members) > 0:
                inertia += float(np.sum((members - centroids[c]) ** 2))

        return ClusterResult(
            labels=labels,
            centroids=centroids,
            sizes=sizes,
            silhouette=sil,
            inertia=inertia,
        )

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers(self, embeddings: np.ndarray) -> OutlierResult:
        """Detect outliers by distance from centroid."""
        centroid = embeddings.mean(axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)

        if self.outlier_method == "iqr":
            q1 = np.percentile(dists, 25)
            q3 = np.percentile(dists, 75)
            iqr = q3 - q1
            threshold = q3 + self.outlier_threshold * iqr
        else:
            # z-score method
            mean_d = np.mean(dists)
            std_d = np.std(dists)
            threshold = mean_d + self.outlier_threshold * std_d if std_d > 0 else np.inf

        is_outlier = dists > threshold
        return OutlierResult(
            is_outlier=is_outlier,
            scores=dists,
            threshold=float(threshold),
            n_outliers=int(is_outlier.sum()),
        )

    # ------------------------------------------------------------------
    # Embedding statistics
    # ------------------------------------------------------------------

    def compute_stats(self, embeddings: np.ndarray) -> EmbeddingStats:
        """Compute descriptive statistics of the embedding matrix."""
        n, d = embeddings.shape
        norms = np.linalg.norm(embeddings, axis=1)

        # Effective rank via eigenvalue entropy
        if n > 1:
            cov = np.cov(embeddings.T)
            if cov.ndim < 2:
                eff_rank = 1.0
            else:
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.maximum(eigvals, 0.0)
                total = eigvals.sum()
                if total > 1e-14:
                    probs = eigvals / total
                    probs = probs[probs > 1e-14]
                    eff_rank = float(math.exp(-np.sum(probs * np.log(probs))))
                else:
                    eff_rank = 1.0
        else:
            eff_rank = 1.0

        # Pairwise distances
        if n >= 2:
            cos_dist = float(np.mean(pdist(embeddings, "cosine")))
            euc_dist = float(np.mean(pdist(embeddings, "euclidean")))
        else:
            cos_dist = 0.0
            euc_dist = 0.0

        # Sparsity
        sparsity = float(np.mean(np.abs(embeddings) < 1e-8))

        return EmbeddingStats(
            n_samples=n,
            n_dims=d,
            mean_norm=float(np.mean(norms)),
            std_norm=float(np.std(norms)),
            min_norm=float(np.min(norms)),
            max_norm=float(np.max(norms)),
            mean_pairwise_cosine=cos_dist,
            mean_pairwise_euclidean=euc_dist,
            effective_rank=eff_rank,
            sparsity=sparsity,
        )

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyse(self, texts: List[str]) -> AnalysisReport:
        """Run full analysis pipeline on raw texts."""
        embeddings = self.embedder.fit_transform(texts)
        return self.analyse_embeddings(embeddings)

    def analyse_embeddings(self, embeddings: np.ndarray) -> AnalysisReport:
        """Run full analysis pipeline on pre-computed embeddings."""
        stats = self.compute_stats(embeddings)
        pca_res = self.pca(embeddings)
        cluster_res = self.cluster(embeddings)
        outlier_res = self.detect_outliers(embeddings)

        # Compute a suite of diversity scores
        diversity_scores: Dict[str, float] = {}
        for name, MetricCls in [
            ("mean_cosine_distance", MeanPairwiseCosineDistance),
            ("mean_euclidean_distance", MeanPairwiseEuclideanDistance),
            ("max_distance", MaxPairwiseDistance),
            ("min_distance", MinPairwiseDistance),
            ("centroid_distance", CentroidDistance),
            ("radius_of_gyration", RadiusOfGyration),
        ]:
            try:
                m = MetricCls()
                diversity_scores[name] = m.compute_from_embeddings(embeddings)
            except Exception as exc:
                logger.warning("Metric %s failed: %s", name, exc)
                diversity_scores[name] = float("nan")

        return AnalysisReport(
            stats=stats,
            pca=pca_res,
            clusters=cluster_res,
            outliers=outlier_res,
            diversity_scores=diversity_scores,
        )

    # ------------------------------------------------------------------
    # Elbow / gap analysis
    # ------------------------------------------------------------------

    def elbow_analysis(
        self,
        embeddings: np.ndarray,
        k_range: Optional[range] = None,
    ) -> Dict[str, List[float]]:
        """Run K-Means for a range of *k* values and return inertias
        and silhouette scores – useful for choosing the number of clusters.

        Returns
        -------
        dict with keys ``"k"``, ``"inertia"``, ``"silhouette"``.
        """
        if k_range is None:
            max_k = min(10, embeddings.shape[0] - 1)
            k_range = range(2, max(3, max_k + 1))

        ks: List[float] = []
        inertias: List[float] = []
        silhouettes: List[float] = []

        for k in k_range:
            if k >= embeddings.shape[0]:
                break
            labels, centroids = _kmeans(embeddings, k, seed=self.seed)
            inertia = 0.0
            for c in range(k):
                members = embeddings[labels == c]
                if len(members) > 0:
                    inertia += float(np.sum((members - centroids[c]) ** 2))
            sil = _silhouette_score(embeddings, labels) if k >= 2 else 0.0

            ks.append(float(k))
            inertias.append(inertia)
            silhouettes.append(sil)

        return {"k": ks, "inertia": inertias, "silhouette": silhouettes}


# ---------------------------------------------------------------------------
# Composite / convenience classes
# ---------------------------------------------------------------------------


class EmbeddingDiversitySuite:
    """Run a full suite of embedding-based diversity metrics.

    Parameters
    ----------
    embedder : SentenceEmbedder
        Shared embedder for all metrics.
    metrics : dict[str, EmbeddingDiversityMetric] | None
        Custom metric set.  If ``None`` a default set is used.
    """

    DEFAULT_METRICS: Dict[str, type] = {
        "mean_cosine_dist": MeanPairwiseCosineDistance,
        "mean_euclidean_dist": MeanPairwiseEuclideanDistance,
        "max_pairwise_dist": MaxPairwiseDistance,
        "min_pairwise_dist": MinPairwiseDistance,
        "centroid_dist": CentroidDistance,
        "coverage": CoverageScore,
        "radius_of_gyration": RadiusOfGyration,
        "convex_hull_volume": ConvexHullVolume,
        "kmeans_diversity": KMeansClusterDiversity,
        "silhouette_diversity": SilhouetteDiversity,
        "dbscan_diversity": DBSCANDiversity,
        "spectral_diversity": SpectralClusterDiversity,
        "intrinsic_dimension": IntrinsicDimensionEstimator,
        "local_linearity": LocalLinearityScore,
        "neighborhood_diversity": NeighborhoodDiversity,
        "median_pairwise_dist": MedianPairwiseDistance,
        "distance_variance": DistanceVariance,
        "distance_entropy": DistanceEntropy,
        "vendi_score": VendiScoreEmbedding,
        "determinantal_diversity": DeterminantalDiversity,
        "manhattan_dist": PairwiseManhattanDistance,
        "chebyshev_dist": PairwiseChebyshevDistance,
    }

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metrics: Optional[Dict[str, EmbeddingDiversityMetric]] = None,
    ) -> None:
        self.embedder = embedder or TFIDFEmbedder()
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = {
                name: cls(embedder=self.embedder)
                for name, cls in self.DEFAULT_METRICS.items()
            }

    def compute(self, texts: List[str]) -> Dict[str, float]:
        """Embed texts once and compute all metrics."""
        embeddings = self.embedder.fit_transform(texts)
        return self.compute_from_embeddings(embeddings)

    def compute_from_embeddings(self, embeddings: np.ndarray) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute_from_embeddings(embeddings)
            except Exception as exc:
                logger.warning("Metric %s failed: %s", name, exc)
                results[name] = float("nan")
        return results


# ---------------------------------------------------------------------------
# Embedding comparison utilities
# ---------------------------------------------------------------------------


class EmbeddingComparator:
    """Compare two sets of texts in embedding space.

    Useful for measuring how differently two decoding strategies behave.

    Parameters
    ----------
    embedder : SentenceEmbedder
        Shared embedder (fit on the union of both text sets).
    """

    def __init__(self, embedder: Optional[SentenceEmbedder] = None) -> None:
        self.embedder = embedder or TFIDFEmbedder()

    def compare(
        self,
        texts_a: List[str],
        texts_b: List[str],
    ) -> Dict[str, float]:
        """Return a dict of comparison statistics.

        Keys
        ----
        centroid_distance : float
            Distance between the centroids of the two sets.
        mean_cross_distance : float
            Average distance between pairs drawn from different sets.
        mean_intra_a / mean_intra_b : float
            Average intra-set distances.
        overlap_score : float
            Fraction of nearest-neighbours that are cross-set.
        mmd : float
            Maximum Mean Discrepancy (RBF kernel).
        """
        all_texts = texts_a + texts_b
        self.embedder.fit(all_texts)
        emb_a = self.embedder.transform(texts_a)
        emb_b = self.embedder.transform(texts_b)

        results: Dict[str, float] = {}

        # Centroid distance
        c_a = emb_a.mean(axis=0)
        c_b = emb_b.mean(axis=0)
        results["centroid_distance"] = float(np.linalg.norm(c_a - c_b))

        # Cross / intra distances
        cross_dists = cdist(emb_a, emb_b, "euclidean")
        results["mean_cross_distance"] = float(np.mean(cross_dists))

        if emb_a.shape[0] >= 2:
            results["mean_intra_a"] = float(np.mean(pdist(emb_a, "euclidean")))
        else:
            results["mean_intra_a"] = 0.0

        if emb_b.shape[0] >= 2:
            results["mean_intra_b"] = float(np.mean(pdist(emb_b, "euclidean")))
        else:
            results["mean_intra_b"] = 0.0

        # Overlap: fraction of 1-NN that come from the other set
        all_emb = np.vstack([emb_a, emb_b])
        n_a = emb_a.shape[0]
        dist_full = squareform(pdist(all_emb, "euclidean"))
        np.fill_diagonal(dist_full, np.inf)
        nn = np.argmin(dist_full, axis=1)

        cross_nn = 0
        n_total = all_emb.shape[0]
        for i in range(n_total):
            in_a = i < n_a
            nn_in_a = nn[i] < n_a
            if in_a != nn_in_a:
                cross_nn += 1
        results["overlap_score"] = float(cross_nn / n_total)

        # Maximum Mean Discrepancy (unbiased estimator with RBF kernel)
        results["mmd"] = self._mmd_rbf(emb_a, emb_b)

        return results

    @staticmethod
    def _mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
        """Unbiased MMD² estimate with RBF kernel (median heuristic)."""
        XY = np.vstack([X, Y])
        sq_dists = squareform(pdist(XY, "sqeuclidean"))
        median_sq = np.median(sq_dists[sq_dists > 0]) if np.any(sq_dists > 0) else 1.0
        gamma = 1.0 / (2.0 * median_sq)
        K = np.exp(-gamma * sq_dists)

        n = X.shape[0]
        m = Y.shape[0]

        K_xx = K[:n, :n]
        K_yy = K[n:, n:]
        K_xy = K[:n, n:]

        # Unbiased estimate
        np.fill_diagonal(K_xx, 0.0)
        np.fill_diagonal(K_yy, 0.0)

        mmd_sq = (
            K_xx.sum() / max(n * (n - 1), 1)
            + K_yy.sum() / max(m * (m - 1), 1)
            - 2.0 * K_xy.sum() / max(n * m, 1)
        )
        return float(max(mmd_sq, 0.0) ** 0.5)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for embedding metrics
# ---------------------------------------------------------------------------


def bootstrap_metric(
    embeddings: np.ndarray,
    metric: EmbeddingDiversityMetric,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals for an embedding metric.

    Returns
    -------
    dict with keys ``"mean"``, ``"std"``, ``"lower"``, ``"upper"``.
    """
    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]
    scores: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = embeddings[idx]
        try:
            scores.append(metric.compute_from_embeddings(sample))
        except Exception:
            continue

    if not scores:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "lower": lower,
        "upper": upper,
    }


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------


def pca_reduce(
    embeddings: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA-reduce embeddings, returning (projected, explained_variance_ratio)."""
    n, d = embeddings.shape
    k = min(n_components, d, n)
    mean = embeddings.mean(axis=0)
    centred = embeddings - mean
    U, S, Vt = svd(centred, full_matrices=False)
    projected = centred @ Vt[:k].T
    total_var = np.sum(S ** 2) / max(n - 1, 1)
    evr = (S[:k] ** 2 / max(n - 1, 1)) / total_var if total_var > 0 else np.zeros(k)
    return projected, evr


def tsne_reduce(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 500,
    learning_rate: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Simplified Barnes-Hut-free t-SNE (exact, small-scale only).

    For large datasets, use PCA instead.  This is provided for
    small embedding sets (< 200 points) where a non-linear projection
    is useful for visualisation.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
    n_components : int
    perplexity : float
    n_iter : int
    learning_rate : float
    seed : int

    Returns
    -------
    np.ndarray, shape (n, n_components)
    """
    n = embeddings.shape[0]
    if n < 4:
        return embeddings[:, :n_components] if embeddings.shape[1] >= n_components else embeddings

    rng = np.random.RandomState(seed)

    # Compute pairwise squared Euclidean distances
    sq_dists = squareform(pdist(embeddings, "sqeuclidean"))

    # Binary search for sigma to match target perplexity
    target_entropy = math.log(perplexity)
    P = np.zeros((n, n))
    for i in range(n):
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2.0
            pi = np.exp(-sq_dists[i] / (2.0 * sigma ** 2))
            pi[i] = 0.0
            sum_pi = pi.sum()
            if sum_pi < 1e-14:
                lo = sigma
                continue
            pi /= sum_pi
            H = -float(np.sum(pi[pi > 0] * np.log(pi[pi > 0])))
            if H > target_entropy:
                hi = sigma
            else:
                lo = sigma
        P[i] = pi

    # Symmetrise
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)

    # Initialise low-dimensional points
    Y = rng.randn(n, n_components) * 0.01
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    momentum = 0.5

    for iteration in range(n_iter):
        # Student-t kernel in low-dimensional space
        sum_Y = np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + sum_Y[:, None] + sum_Y[None, :] - 2.0 * Y @ Y.T)
        np.fill_diagonal(num, 0.0)
        Q = num / num.sum()
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4.0 * np.sum((PQ[i] * num[i])[:, None] * diff, axis=0)

        # Adaptive gains
        gains = np.where(
            np.sign(grad) != np.sign(dY),
            gains + 0.2,
            gains * 0.8,
        )
        gains = np.maximum(gains, 0.01)

        if iteration >= 250:
            momentum = 0.8

        dY = momentum * dY - learning_rate * gains * grad
        Y += dY
        Y -= Y.mean(axis=0)

    return Y


# ---------------------------------------------------------------------------
# Kernel / similarity matrix builders
# ---------------------------------------------------------------------------


def build_kernel_matrix(
    embeddings: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> np.ndarray:
    """Build a kernel matrix from embeddings.

    Parameters
    ----------
    kernel : str
        ``"rbf"``, ``"cosine"``, ``"linear"``, ``"polynomial"``.
    gamma : float | None
        RBF gamma; auto if ``None``.
    """
    if kernel == "cosine":
        return _cosine_similarity_matrix(embeddings)
    elif kernel == "linear":
        return embeddings @ embeddings.T
    elif kernel == "polynomial":
        degree = 3
        c = 1.0
        return (embeddings @ embeddings.T + c) ** degree
    else:
        return _rbf_kernel(embeddings, gamma)


def build_distance_matrix(
    embeddings: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Build a full (n, n) distance matrix."""
    return squareform(pdist(embeddings, metric=metric))


# ---------------------------------------------------------------------------
# Nearest-neighbour graph utilities
# ---------------------------------------------------------------------------


class NearestNeighbourGraph:
    """Build and query a nearest-neighbour graph from embeddings.

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    metric : str
        Distance metric.
    """

    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric
        self._indices: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        self._n: int = 0

    def fit(self, embeddings: np.ndarray) -> "NearestNeighbourGraph":
        """Build the graph."""
        n = embeddings.shape[0]
        self._n = n
        k = min(self.k, n - 1)
        dist_matrix = squareform(pdist(embeddings, self.metric))
        np.fill_diagonal(dist_matrix, np.inf)

        self._indices = np.argsort(dist_matrix, axis=1)[:, :k]
        self._distances = np.sort(dist_matrix, axis=1)[:, :k]
        return self

    @property
    def indices(self) -> np.ndarray:
        if self._indices is None:
            raise RuntimeError("Call fit() first.")
        return self._indices

    @property
    def distances(self) -> np.ndarray:
        if self._distances is None:
            raise RuntimeError("Call fit() first.")
        return self._distances

    def knn_graph_adjacency(self) -> np.ndarray:
        """Return a (n, n) binary adjacency matrix of the kNN graph."""
        adj = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in self.indices[i]:
                adj[i, j] = 1.0
        return adj

    def mean_knn_distance(self) -> float:
        """Average k-NN distance (global density proxy)."""
        return float(np.mean(self.distances))

    def knn_distance_variance(self) -> float:
        """Variance of k-NN distances – higher means uneven density."""
        return float(np.var(self.distances))

    def hub_score(self) -> np.ndarray:
        """Count how often each point appears as a neighbour (hubness)."""
        counts = np.zeros(self._n)
        for i in range(self._n):
            for j in self.indices[i]:
                counts[j] += 1
        return counts

    def local_density(self) -> np.ndarray:
        """Local density estimate as 1 / mean kNN distance per point."""
        mean_d = self.distances.mean(axis=1)
        mean_d = np.where(mean_d < 1e-14, 1e-14, mean_d)
        return 1.0 / mean_d


# ---------------------------------------------------------------------------
# Embedding stability / robustness
# ---------------------------------------------------------------------------


class EmbeddingStability:
    """Measure how stable diversity scores are under perturbation.

    Adds Gaussian noise at increasing scales and re-computes metrics.

    Parameters
    ----------
    metric : EmbeddingDiversityMetric
        Metric to test.
    noise_scales : list[float]
        Standard deviations of Gaussian noise.
    n_trials : int
        Number of repetitions per noise scale.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        metric: Optional[EmbeddingDiversityMetric] = None,
        noise_scales: Optional[List[float]] = None,
        n_trials: int = 10,
        seed: int = 42,
    ) -> None:
        self.metric = metric or MeanPairwiseCosineDistance()
        self.noise_scales = noise_scales or [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        self.n_trials = n_trials
        self.seed = seed

    def evaluate(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Return per-scale mean and std of the metric.

        Returns
        -------
        dict with keys ``"scales"``, ``"means"``, ``"stds"``.
        """
        rng = np.random.RandomState(self.seed)
        scales: List[float] = []
        means: List[float] = []
        stds: List[float] = []

        for sigma in self.noise_scales:
            trial_scores: List[float] = []
            for _ in range(self.n_trials):
                noisy = embeddings + rng.randn(*embeddings.shape) * sigma
                try:
                    trial_scores.append(
                        self.metric.compute_from_embeddings(noisy)
                    )
                except Exception:
                    continue
            if trial_scores:
                scales.append(sigma)
                means.append(float(np.mean(trial_scores)))
                stds.append(float(np.std(trial_scores)))

        return {"scales": scales, "means": means, "stds": stds}


# ---------------------------------------------------------------------------
# Embedding set operations
# ---------------------------------------------------------------------------


def embedding_union_diversity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    metric: Optional[EmbeddingDiversityMetric] = None,
) -> Dict[str, float]:
    """Compare diversity of two sets and their union.

    Returns dict with ``"div_a"``, ``"div_b"``, ``"div_union"``,
    ``"synergy"`` (= div_union − max(div_a, div_b)).
    """
    metric = metric or MeanPairwiseCosineDistance()
    d_a = metric.compute_from_embeddings(emb_a)
    d_b = metric.compute_from_embeddings(emb_b)
    d_union = metric.compute_from_embeddings(np.vstack([emb_a, emb_b]))
    return {
        "div_a": d_a,
        "div_b": d_b,
        "div_union": d_union,
        "synergy": d_union - max(d_a, d_b),
    }


# ---------------------------------------------------------------------------
# Greedy subset selection for maximum diversity
# ---------------------------------------------------------------------------


def greedy_max_diversity_subset(
    embeddings: np.ndarray,
    k: int,
    metric: str = "cosine",
    seed: int = 42,
) -> List[int]:
    """Greedy farthest-point selection for a diverse subset of size *k*.

    At each step, the point maximising the minimum distance to the
    already-selected set is added.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
    k : int
        Desired subset size.
    metric : str
        Distance metric.
    seed : int
        Random seed for the initial point.

    Returns
    -------
    list[int]
        Indices of the selected points.
    """
    n = embeddings.shape[0]
    k = min(k, n)
    if k <= 0:
        return []

    dist_mat = squareform(pdist(embeddings, metric=metric))
    rng = np.random.RandomState(seed)
    selected = [rng.randint(n)]
    remaining = set(range(n)) - set(selected)

    for _ in range(k - 1):
        if not remaining:
            break
        best_idx = -1
        best_min_dist = -1.0
        for idx in remaining:
            min_dist = min(dist_mat[idx, s] for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Embedding-based text deduplication
# ---------------------------------------------------------------------------


def deduplicate_by_embedding(
    texts: List[str],
    embedder: Optional[SentenceEmbedder] = None,
    threshold: float = 0.1,
    metric: str = "cosine",
) -> List[int]:
    """Return indices of texts that are sufficiently distinct.

    Parameters
    ----------
    texts : list[str]
    embedder : SentenceEmbedder
    threshold : float
        Minimum distance for a text to be considered distinct.
    metric : str
        Distance metric.

    Returns
    -------
    list[int]
        Indices of retained texts.
    """
    if not texts:
        return []
    embedder = embedder or TFIDFEmbedder()
    embeddings = embedder.fit_transform(texts)
    n = embeddings.shape[0]

    kept = [0]
    for i in range(1, n):
        dists = cdist(embeddings[i : i + 1], embeddings[kept], metric=metric).ravel()
        if np.min(dists) >= threshold:
            kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# Batch processing utilities
# ---------------------------------------------------------------------------


class BatchEmbeddingProcessor:
    """Process multiple text sets and aggregate diversity metrics.

    Parameters
    ----------
    embedder : SentenceEmbedder
    metrics : dict[str, EmbeddingDiversityMetric] | None
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        metrics: Optional[Dict[str, EmbeddingDiversityMetric]] = None,
    ) -> None:
        self.embedder = embedder or TFIDFEmbedder()
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = {
                "mean_cosine": MeanPairwiseCosineDistance(self.embedder),
                "mean_euclidean": MeanPairwiseEuclideanDistance(self.embedder),
                "radius_of_gyration": RadiusOfGyration(self.embedder),
                "coverage": CoverageScore(self.embedder),
            }

    def process_batch(
        self, text_sets: List[List[str]]
    ) -> List[Dict[str, float]]:
        """Compute metrics for each text set in the batch."""
        results: List[Dict[str, float]] = []
        for texts in text_sets:
            embeddings = self.embedder.fit_transform(texts)
            row: Dict[str, float] = {}
            for name, metric in self.metrics.items():
                try:
                    row[name] = metric.compute_from_embeddings(embeddings)
                except Exception as exc:
                    logger.warning("Metric %s failed: %s", name, exc)
                    row[name] = float("nan")
            results.append(row)
        return results

    def aggregate(
        self, results: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate batch results: mean, std, min, max per metric."""
        if not results:
            return {}
        keys = results[0].keys()
        agg: Dict[str, Dict[str, float]] = {}
        for key in keys:
            vals = [r[key] for r in results if not math.isnan(r.get(key, float("nan")))]
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            else:
                agg[key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return agg


# ---------------------------------------------------------------------------
# Correlation between embedding metrics
# ---------------------------------------------------------------------------


def metric_correlation_matrix(
    embeddings: np.ndarray,
    metrics: Optional[Dict[str, EmbeddingDiversityMetric]] = None,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Compute Pearson correlation between metrics over bootstrap samples.

    Returns
    -------
    corr_matrix : np.ndarray, shape (n_metrics, n_metrics)
    metric_names : list[str]
    """
    if metrics is None:
        metrics = {
            "cosine": MeanPairwiseCosineDistance(),
            "euclidean": MeanPairwiseEuclideanDistance(),
            "centroid": CentroidDistance(),
            "gyration": RadiusOfGyration(),
        }

    names = list(metrics.keys())
    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]

    scores = {name: [] for name in names}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = embeddings[idx]
        for name, m in metrics.items():
            try:
                scores[name].append(m.compute_from_embeddings(sample))
            except Exception:
                scores[name].append(float("nan"))

    mat = np.array([scores[name] for name in names])
    # Filter out NaN columns
    valid = ~np.any(np.isnan(mat), axis=0)
    mat = mat[:, valid]

    if mat.shape[1] < 2:
        return np.eye(len(names)), names

    corr = np.corrcoef(mat)
    return corr, names


# ---------------------------------------------------------------------------
# Token-level embedding pooling strategies
# ---------------------------------------------------------------------------


class PoolingStrategy(ABC):
    """Strategy for combining token embeddings into a sentence embedding."""

    @abstractmethod
    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Combine (n_tokens, d) into (d,)."""
        ...


class MeanPooling(PoolingStrategy):
    """Average of token embeddings."""

    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        if token_embeddings.shape[0] == 0:
            return np.zeros(token_embeddings.shape[1])
        return token_embeddings.mean(axis=0)


class MaxPooling(PoolingStrategy):
    """Element-wise maximum of token embeddings."""

    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        if token_embeddings.shape[0] == 0:
            return np.zeros(token_embeddings.shape[1])
        return token_embeddings.max(axis=0)


class MinPooling(PoolingStrategy):
    """Element-wise minimum of token embeddings."""

    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        if token_embeddings.shape[0] == 0:
            return np.zeros(token_embeddings.shape[1])
        return token_embeddings.min(axis=0)


class AttentionPooling(PoolingStrategy):
    """Self-attention pooling: weighted average by L2 norm of each token.

    Higher-norm tokens get more weight, capturing salience.
    """

    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        n = token_embeddings.shape[0]
        if n == 0:
            return np.zeros(token_embeddings.shape[1])
        norms = np.linalg.norm(token_embeddings, axis=1)
        weights = np.exp(norms - norms.max())  # softmax-like
        weights /= weights.sum()
        return (token_embeddings.T @ weights)


class ConcatPooling(PoolingStrategy):
    """Concatenate mean, max, and min pooling."""

    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        if token_embeddings.shape[0] == 0:
            d = token_embeddings.shape[1]
            return np.zeros(3 * d)
        return np.concatenate([
            token_embeddings.mean(axis=0),
            token_embeddings.max(axis=0),
            token_embeddings.min(axis=0),
        ])


# ---------------------------------------------------------------------------
# Token embedding via hash / random vectors
# ---------------------------------------------------------------------------


class HashTokenEmbedder:
    """Map tokens to fixed random vectors via hashing.

    Each unique token always maps to the same vector (deterministic).

    Parameters
    ----------
    dim : int
        Embedding dimension.
    seed : int
        Base seed.
    pooling : PoolingStrategy
        Strategy for combining token vectors.
    """

    def __init__(
        self,
        dim: int = 64,
        seed: int = 42,
        pooling: Optional[PoolingStrategy] = None,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.pooling = pooling or MeanPooling()

    def _token_vector(self, token: str) -> np.ndarray:
        """Deterministic random vector for a token."""
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        rng = np.random.RandomState(h % (2**31) ^ self.seed)
        vec = rng.randn(self.dim)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(self.dim if isinstance(self.pooling, (MeanPooling, MaxPooling, MinPooling, AttentionPooling)) else 3 * self.dim)
        token_vecs = np.array([self._token_vector(t) for t in tokens])
        return self.pooling.pool(token_vecs)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        vecs = [self.embed_text(t) for t in texts]
        return np.array(vecs)


# ---------------------------------------------------------------------------
# Sparse-to-dense wrapper as SentenceEmbedder
# ---------------------------------------------------------------------------


class HashTokenSentenceEmbedder(SentenceEmbedder):
    """SentenceEmbedder wrapper around HashTokenEmbedder."""

    def __init__(
        self,
        dim: int = 64,
        seed: int = 42,
        pooling: Optional[PoolingStrategy] = None,
    ) -> None:
        self._inner = HashTokenEmbedder(dim=dim, seed=seed, pooling=pooling)

    def fit(self, texts: List[str]) -> "HashTokenSentenceEmbedder":
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._inner.embed_batch(texts)


# ---------------------------------------------------------------------------
# Positional-aware embedder
# ---------------------------------------------------------------------------


class PositionalEmbedder(SentenceEmbedder):
    """Adds sinusoidal positional encoding to token hash vectors.

    Captures token order in the embedding.

    Parameters
    ----------
    dim : int
        Per-token embedding dimension.
    max_len : int
        Maximum sequence length.
    seed : int
        Random seed for token vectors.
    """

    def __init__(
        self,
        dim: int = 64,
        max_len: int = 512,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.max_len = max_len
        self.seed = seed
        self._token_emb = HashTokenEmbedder(dim=dim, seed=seed)
        self._pos_enc = self._build_positional_encoding(max_len, dim)

    @staticmethod
    def _build_positional_encoding(max_len: int, dim: int) -> np.ndarray:
        """Sinusoidal positional encoding (Vaswani et al.)."""
        pe = np.zeros((max_len, dim))
        position = np.arange(max_len)[:, None]
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = np.cos(position * div_term[: dim // 2])
        return pe

    def fit(self, texts: List[str]) -> "PositionalEmbedder":
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        results: List[np.ndarray] = []
        for text in texts:
            tokens = _tokenize(text)
            if not tokens:
                results.append(np.zeros(self.dim))
                continue
            vecs = np.array([self._token_emb._token_vector(t) for t in tokens])
            length = min(len(tokens), self.max_len)
            vecs[:length] += self._pos_enc[:length]
            results.append(vecs.mean(axis=0))
        mat = np.array(results)
        return self._normalise_rows(mat)


# ---------------------------------------------------------------------------
# Subspace angle diversity
# ---------------------------------------------------------------------------


class SubspaceAngleDiversity(EmbeddingDiversityMetric):
    """Diversity measured by principal angles between subspaces.

    Splits texts into two halves and measures the angle between the
    subspaces spanned by each half's embeddings.  Large angles indicate
    that different subsets occupy different regions.

    Parameters
    ----------
    n_components : int
        Subspace dimension.
    n_splits : int
        Number of random splits to average over.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_components: int = 3,
        n_splits: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(embedder)
        self.n_components = n_components
        self.n_splits = n_splits
        self.seed = seed

    @staticmethod
    def _subspace_basis(data: np.ndarray, k: int) -> np.ndarray:
        """Top-k right singular vectors."""
        mean = data.mean(axis=0)
        centred = data - mean
        _, _, Vt = svd(centred, full_matrices=False)
        return Vt[: min(k, Vt.shape[0])]

    @staticmethod
    def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute principal angles between subspaces spanned by rows of A and B."""
        # Use SVD of A^T B
        M = A @ B.T
        svals = np.linalg.svd(M, compute_uv=False)
        svals = np.clip(svals, 0.0, 1.0)
        return np.arccos(svals)

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        k = min(self.n_components, embeddings.shape[1])
        if n < 4 or k < 1:
            return 0.0

        rng = np.random.RandomState(self.seed)
        angles_all: List[float] = []

        for _ in range(self.n_splits):
            perm = rng.permutation(n)
            half = n // 2
            A = embeddings[perm[:half]]
            B = embeddings[perm[half : 2 * half]]
            if A.shape[0] < k + 1 or B.shape[0] < k + 1:
                continue
            basis_a = self._subspace_basis(A, k)
            basis_b = self._subspace_basis(B, k)
            pa = self._principal_angles(basis_a, basis_b)
            angles_all.append(float(np.mean(pa)))

        if not angles_all:
            return 0.0
        # Normalise by pi/2 (max possible principal angle)
        return float(np.mean(angles_all) / (math.pi / 2.0))


# ---------------------------------------------------------------------------
# Mahalanobis-based diversity
# ---------------------------------------------------------------------------


class MahalanobisDiversity(EmbeddingDiversityMetric):
    """Mean Mahalanobis distance from the centroid.

    Accounts for the covariance structure of the embedding distribution,
    so elongated distributions are penalised less than radially isotropic
    ones.

    Parameters
    ----------
    reg : float
        Regularisation for the covariance inverse.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        reg: float = 1e-4,
    ) -> None:
        super().__init__(embedder)
        self.reg = reg

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n, d = embeddings.shape
        if n < 3:
            return 0.0

        mean = embeddings.mean(axis=0)
        centred = embeddings - mean
        cov = (centred.T @ centred) / (n - 1) + self.reg * np.eye(d)

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 0.0

        # Mahalanobis distance for each point
        dists = np.sqrt(np.sum(centred @ cov_inv * centred, axis=1))
        return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Entropy of angular distribution
# ---------------------------------------------------------------------------


class AngularEntropy(EmbeddingDiversityMetric):
    """Entropy of the angular distribution of embeddings around the centroid.

    Projects each embedding direction into angular bins and computes
    Shannon entropy of the resulting distribution.

    Parameters
    ----------
    n_bins : int
        Number of angular bins.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_bins: int = 36,
    ) -> None:
        super().__init__(embedder)
        self.n_bins = n_bins

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n, d = embeddings.shape
        if n < 3 or d < 2:
            return 0.0

        # PCA to 2-D for angular distribution
        mean = embeddings.mean(axis=0)
        centred = embeddings - mean
        _, _, Vt = svd(centred, full_matrices=False)
        proj = centred @ Vt[:2].T  # (n, 2)

        angles = np.arctan2(proj[:, 1], proj[:, 0])  # in [-pi, pi]
        angles = (angles + np.pi) / (2 * np.pi)  # map to [0, 1]

        counts, _ = np.histogram(angles, bins=self.n_bins, range=(0, 1))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(self.n_bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Fractal dimension estimator
# ---------------------------------------------------------------------------


class FractalDimensionDiversity(EmbeddingDiversityMetric):
    """Correlation (fractal) dimension of the embedding point cloud.

    Uses the Grassberger-Procaccia algorithm: count pairs within
    distance *r* for several values of *r*, then estimate the slope of
    log C(r) vs log r.

    Parameters
    ----------
    n_radii : int
        Number of radius values to evaluate.
    """

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        n_radii: int = 20,
    ) -> None:
        super().__init__(embedder)
        self.n_radii = n_radii

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n = embeddings.shape[0]
        if n < 5:
            return 0.0

        dists = pdist(embeddings, "euclidean")
        if len(dists) == 0 or np.max(dists) < 1e-14:
            return 0.0

        r_min = np.percentile(dists[dists > 0], 5)
        r_max = np.percentile(dists, 95)
        if r_min <= 0 or r_max <= r_min:
            return 0.0

        radii = np.logspace(np.log10(r_min), np.log10(r_max), self.n_radii)
        n_pairs = len(dists)
        log_r: List[float] = []
        log_c: List[float] = []

        for r in radii:
            count = np.sum(dists < r)
            if count > 0:
                log_r.append(math.log(r))
                log_c.append(math.log(count / n_pairs))

        if len(log_r) < 3:
            return 0.0

        # Linear regression for slope
        log_r_arr = np.array(log_r)
        log_c_arr = np.array(log_c)
        A = np.vstack([log_r_arr, np.ones(len(log_r_arr))]).T
        result = np.linalg.lstsq(A, log_c_arr, rcond=None)
        slope = result[0][0]

        # Normalise by ambient dimension
        d = embeddings.shape[1]
        return float(min(max(slope, 0.0) / d, 1.0))


# ---------------------------------------------------------------------------
# Dispersion metrics
# ---------------------------------------------------------------------------


class Dispersion(EmbeddingDiversityMetric):
    """Simple dispersion: trace of the covariance matrix divided by
    the number of dimensions.
    """

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n, d = embeddings.shape
        if n < 2 or d < 1:
            return 0.0
        cov = np.cov(embeddings.T)
        if cov.ndim < 2:
            return float(cov)
        return float(np.trace(cov) / d)


class SpreadRatio(EmbeddingDiversityMetric):
    """Ratio of largest to smallest eigenvalue of the covariance matrix.

    Measures isotropy: a ratio near 1 means uniform spread; a high
    ratio means the distribution is elongated along one axis.

    Returns ``1 / ratio`` so that higher is more diverse (isotropic).
    """

    def compute_from_embeddings(self, embeddings: np.ndarray) -> float:
        n, d = embeddings.shape
        if n < 3 or d < 2:
            return 0.0
        cov = np.cov(embeddings.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        max_eig = eigvals[-1]
        min_nonzero = eigvals[eigvals > 1e-12]
        if len(min_nonzero) == 0 or max_eig < 1e-12:
            return 0.0
        ratio = max_eig / min_nonzero[0]
        return float(1.0 / ratio) if ratio > 0 else 0.0


# ---------------------------------------------------------------------------
# Sample efficiency curve
# ---------------------------------------------------------------------------


def diversity_curve(
    embeddings: np.ndarray,
    metric: Optional[EmbeddingDiversityMetric] = None,
    fractions: Optional[List[float]] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Compute diversity as a function of sample size.

    Useful for understanding how diversity scales with the number of
    generated texts.

    Returns
    -------
    dict with ``"fractions"``, ``"means"``, ``"stds"``.
    """
    metric = metric or MeanPairwiseCosineDistance()
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]
    means: List[float] = []
    stds: List[float] = []

    for frac in fractions:
        k = max(2, int(n * frac))
        scores: List[float] = []
        for _ in range(n_trials):
            idx = rng.choice(n, size=min(k, n), replace=False)
            try:
                scores.append(metric.compute_from_embeddings(embeddings[idx]))
            except Exception:
                continue
        means.append(float(np.mean(scores)) if scores else 0.0)
        stds.append(float(np.std(scores)) if scores else 0.0)

    return {"fractions": fractions, "means": means, "stds": stds}


# ---------------------------------------------------------------------------
# Public API summary
# ---------------------------------------------------------------------------

__all__ = [
    # Embedders
    "SentenceEmbedder",
    "TFIDFEmbedder",
    "BagOfWordsEmbedder",
    "HashEmbedder",
    "NGramEmbedder",
    "RandomProjectionEmbedder",
    "WeightedEmbedder",
    "HashTokenSentenceEmbedder",
    "PositionalEmbedder",
    # Diversity metrics
    "EmbeddingDiversityMetric",
    "MeanPairwiseCosineDistance",
    "MeanPairwiseEuclideanDistance",
    "MaxPairwiseDistance",
    "MinPairwiseDistance",
    "CentroidDistance",
    "CoverageScore",
    "RadiusOfGyration",
    "ConvexHullVolume",
    "MedianPairwiseDistance",
    "DistanceVariance",
    "DistanceEntropy",
    "VendiScoreEmbedding",
    "DeterminantalDiversity",
    "PairwiseManhattanDistance",
    "PairwiseChebyshevDistance",
    "SubspaceAngleDiversity",
    "MahalanobisDiversity",
    "AngularEntropy",
    "FractalDimensionDiversity",
    "Dispersion",
    "SpreadRatio",
    # Clustering diversity
    "ClusteringDiversityMetric",
    "KMeansClusterDiversity",
    "SilhouetteDiversity",
    "DBSCANDiversity",
    "SpectralClusterDiversity",
    # Manifold diversity
    "ManifoldDiversityMetric",
    "IntrinsicDimensionEstimator",
    "LocalLinearityScore",
    "NeighborhoodDiversity",
    # Analyser
    "EmbeddingAnalyzer",
    "AnalysisReport",
    "PCAResult",
    "ClusterResult",
    "OutlierResult",
    "EmbeddingStats",
    # Suite & comparator
    "EmbeddingDiversitySuite",
    "EmbeddingComparator",
    # Utilities
    "bootstrap_metric",
    "pca_reduce",
    "tsne_reduce",
    "build_kernel_matrix",
    "build_distance_matrix",
    "NearestNeighbourGraph",
    "EmbeddingStability",
    "embedding_union_diversity",
    "greedy_max_diversity_subset",
    "deduplicate_by_embedding",
    "BatchEmbeddingProcessor",
    "metric_correlation_matrix",
    "diversity_curve",
    # Pooling
    "PoolingStrategy",
    "MeanPooling",
    "MaxPooling",
    "MinPooling",
    "AttentionPooling",
    "ConcatPooling",
    "HashTokenEmbedder",
]
