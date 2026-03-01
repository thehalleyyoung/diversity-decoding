"""
Semantic similarity matrix computation for the Diversity Decoding Arena.

Provides comprehensive tools for computing pairwise semantic similarity
between generated texts, clustering outputs by semantic content, analyzing
coverage of the semantic space, and reducing dimensionality for
visualization. All implementations use only numpy/scipy — no external
NLP libraries.
"""

from __future__ import annotations

import logging
import math
import re
import string
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
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd as scipy_svd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "this", "that", "these", "those", "am", "s", "t", "d", "ll", "ve",
    "re", "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
}

_PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")


# =========================================================================
# Helper functions
# =========================================================================

def tokenize(text: str, lower: bool = True, remove_stopwords: bool = False) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    if lower:
        text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def build_vocabulary(
    documents: Sequence[str],
    min_df: int = 1,
    max_df_ratio: float = 1.0,
    remove_stopwords: bool = True,
) -> Tuple[Dict[str, int], List[str]]:
    """Build a word-to-index vocabulary from a collection of documents.

    Returns
    -------
    word2idx : dict mapping word -> column index
    idx2word : list mapping column index -> word
    """
    doc_freq: Counter = Counter()
    n_docs = len(documents)

    for doc in documents:
        tokens = set(tokenize(doc, remove_stopwords=remove_stopwords))
        for tok in tokens:
            doc_freq[tok] += 1

    max_df = int(max_df_ratio * n_docs) if max_df_ratio < 1.0 else n_docs
    vocab = sorted(
        w for w, c in doc_freq.items() if min_df <= c <= max_df
    )
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, vocab


def compute_idf_weights(
    documents: Sequence[str],
    word2idx: Dict[str, int],
    smooth: bool = True,
    remove_stopwords: bool = True,
) -> np.ndarray:
    """Compute inverse-document-frequency weights for each vocabulary term.

    Uses the standard formula:  idf(t) = log((1 + N) / (1 + df(t))) + 1
    when *smooth* is True, otherwise idf(t) = log(N / df(t)) + 1.
    """
    n_docs = len(documents)
    vocab_size = len(word2idx)
    df = np.zeros(vocab_size, dtype=np.float64)

    for doc in documents:
        seen: Set[str] = set()
        for tok in tokenize(doc, remove_stopwords=remove_stopwords):
            if tok in word2idx and tok not in seen:
                df[word2idx[tok]] += 1.0
                seen.add(tok)

    if smooth:
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    else:
        # Avoid division by zero
        df = np.maximum(df, 1e-10)
        idf = np.log(n_docs / df) + 1.0

    return idf


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for row-vectors in *X*.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    S : ndarray of shape (n_samples, n_samples) with values in [-1, 1].
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_normed = X / norms
    S = X_normed @ X_normed.T
    # Clip for numerical stability
    np.clip(S, -1.0, 1.0, out=S)
    return S


def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for row-vectors in *X*.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples) with non-negative values.
    """
    sq_norms = np.sum(X ** 2, axis=1)
    D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    np.maximum(D_sq, 0.0, out=D_sq)
    return np.sqrt(D_sq)


def normalize_matrix(M: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize a matrix to [0, 1] or zero-mean/unit-variance.

    Parameters
    ----------
    M : ndarray
    method : 'minmax' | 'zscore'
    """
    if method == "minmax":
        lo, hi = M.min(), M.max()
        if hi - lo < 1e-12:
            return np.zeros_like(M)
        return (M - lo) / (hi - lo)
    elif method == "zscore":
        mu = M.mean()
        sigma = M.std()
        if sigma < 1e-12:
            return np.zeros_like(M)
        return (M - mu) / sigma
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def symmetrize_matrix(M: np.ndarray) -> np.ndarray:
    """Return (M + M^T) / 2."""
    return (M + M.T) / 2.0


def soft_cosine_similarity(
    v1: np.ndarray,
    v2: np.ndarray,
    S: np.ndarray,
) -> float:
    """Compute the soft cosine similarity between two vectors given a
    word-similarity matrix *S*.

    soft_cos(a, b) = (a^T S b) / sqrt((a^T S a)(b^T S b))
    """
    num = float(v1 @ S @ v2)
    d1 = float(v1 @ S @ v1)
    d2 = float(v2 @ S @ v2)
    denom = math.sqrt(max(d1, 0.0) * max(d2, 0.0))
    if denom < 1e-12:
        return 0.0
    return num / denom


def _character_ngrams(text: str, n: int) -> List[str]:
    """Extract character n-grams from *text*."""
    text = text.lower()
    return [text[i: i + n] for i in range(len(text) - n + 1)]


def _word_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract word n-grams from a token list."""
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


# =========================================================================
# Data-classes
# =========================================================================

@dataclass
class SimilarityConfig:
    """Configuration for semantic similarity computation."""

    method: str = "tfidf"
    n_components: int = 50
    n_clusters: int = 5
    min_cluster_size: int = 2
    clustering_method: str = "kmeans"
    ngram_range: Tuple[int, int] = (1, 2)
    remove_stopwords: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    min_df: int = 1
    max_df_ratio: float = 1.0
    combined_weights: Dict[str, float] = field(default_factory=lambda: {
        "tfidf": 0.4,
        "bm25": 0.2,
        "jaccard": 0.2,
        "ngram": 0.2,
    })
    random_state: int = 42
    max_kmeans_iter: int = 300
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3
    linkage_method: str = "average"
    spectral_n_neighbors: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_iter: int = 200

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return {
            "method": self.method,
            "n_components": self.n_components,
            "n_clusters": self.n_clusters,
            "min_cluster_size": self.min_cluster_size,
            "clustering_method": self.clustering_method,
            "ngram_range": list(self.ngram_range),
            "remove_stopwords": self.remove_stopwords,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "combined_weights": dict(self.combined_weights),
            "random_state": self.random_state,
            "max_kmeans_iter": self.max_kmeans_iter,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "linkage_method": self.linkage_method,
            "spectral_n_neighbors": self.spectral_n_neighbors,
            "umap_n_neighbors": self.umap_n_neighbors,
            "umap_min_dist": self.umap_min_dist,
            "umap_n_iter": self.umap_n_iter,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimilarityConfig":
        """Deserialize a configuration from a plain dictionary."""
        d = dict(d)
        if "ngram_range" in d:
            d["ngram_range"] = tuple(d["ngram_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimilarityResult:
    """Result of a semantic similarity computation."""

    matrix: np.ndarray
    method: str = ""
    clusters: Optional[np.ndarray] = None
    coverage: Optional[float] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    documents: Optional[List[str]] = None
    vocabulary_size: int = 0

    # Convenience ---------------------------------------------------------

    @property
    def n_documents(self) -> int:
        return self.matrix.shape[0]

    @property
    def mean_similarity(self) -> float:
        n = self.n_documents
        if n < 2:
            return 1.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(self.matrix[mask]))

    @property
    def diversity_score(self) -> float:
        return 1.0 - self.mean_similarity

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "method": self.method,
            "n_documents": self.n_documents,
            "mean_similarity": self.mean_similarity,
            "diversity_score": self.diversity_score,
            "vocabulary_size": self.vocabulary_size,
            "statistics": self.statistics,
        }
        if self.coverage is not None:
            d["coverage"] = self.coverage
        if self.clusters is not None:
            d["n_clusters"] = int(self.clusters.max() + 1)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimilarityResult":
        """Minimal reconstruction (matrix must be supplied separately)."""
        return cls(
            matrix=np.array(d.get("matrix", [[]])),
            method=d.get("method", ""),
            coverage=d.get("coverage"),
            vocabulary_size=d.get("vocabulary_size", 0),
            statistics=d.get("statistics", {}),
        )


# =========================================================================
# SemanticSimilarityComputer
# =========================================================================

class SemanticSimilarityComputer:
    """Compute semantic similarity matrices between documents using
    various information-retrieval and text-based methods.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None) -> None:
        self.config = config or SimilarityConfig()
        self._rng = np.random.RandomState(self.config.random_state)

    # -----------------------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------------------

    def compute_tfidf_vectors(
        self,
        documents: Sequence[str],
        word2idx: Optional[Dict[str, int]] = None,
        idf: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
        """Build TF-IDF vectors for each document.

        Returns (tfidf_matrix, word2idx, idf).
        """
        if word2idx is None:
            word2idx, _ = build_vocabulary(
                documents,
                min_df=self.config.min_df,
                max_df_ratio=self.config.max_df_ratio,
                remove_stopwords=self.config.remove_stopwords,
            )
        vocab_size = len(word2idx)
        if vocab_size == 0:
            logger.warning("Empty vocabulary; returning zero matrix.")
            return (
                np.zeros((len(documents), 1)),
                word2idx,
                np.ones(1),
            )

        if idf is None:
            idf = compute_idf_weights(
                documents, word2idx,
                remove_stopwords=self.config.remove_stopwords,
            )

        n_docs = len(documents)
        tf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        for i, doc in enumerate(documents):
            counts: Counter = Counter()
            tokens = tokenize(doc, remove_stopwords=self.config.remove_stopwords)
            for tok in tokens:
                if tok in word2idx:
                    counts[tok] += 1
            doc_len = max(len(tokens), 1)
            for w, c in counts.items():
                tf_matrix[i, word2idx[w]] = c / doc_len

        tfidf = tf_matrix * idf[None, :]
        # L2-normalize rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tfidf /= norms

        return tfidf, word2idx, idf

    def compute_tfidf_similarity(
        self,
        documents: Sequence[str],
    ) -> SimilarityResult:
        """Compute TF-IDF cosine similarity matrix."""
        logger.debug("Computing TF-IDF similarity for %d documents.", len(documents))
        tfidf, word2idx, idf = self.compute_tfidf_vectors(documents)
        sim = cosine_similarity_matrix(tfidf)
        np.fill_diagonal(sim, 1.0)

        mask = ~np.eye(len(documents), dtype=bool)
        stats: Dict[str, Any] = {
            "mean": float(np.mean(sim[mask])) if len(documents) > 1 else 1.0,
            "std": float(np.std(sim[mask])) if len(documents) > 1 else 0.0,
            "min": float(np.min(sim[mask])) if len(documents) > 1 else 1.0,
            "max": float(np.max(sim[mask])) if len(documents) > 1 else 1.0,
            "median": float(np.median(sim[mask])) if len(documents) > 1 else 1.0,
        }

        return SimilarityResult(
            matrix=sim,
            method="tfidf",
            statistics=stats,
            documents=list(documents),
            vocabulary_size=len(word2idx),
        )

    # -----------------------------------------------------------------
    # BM25
    # -----------------------------------------------------------------

    def compute_bm25_similarity(
        self,
        documents: Sequence[str],
    ) -> SimilarityResult:
        """Compute BM25-based similarity matrix.

        For each pair (i, j) we score document j as a "query" against
        document i, then symmetrise.
        """
        logger.debug("Computing BM25 similarity for %d documents.", len(documents))
        k1 = self.config.bm25_k1
        b = self.config.bm25_b

        word2idx, _ = build_vocabulary(
            documents,
            min_df=self.config.min_df,
            max_df_ratio=self.config.max_df_ratio,
            remove_stopwords=self.config.remove_stopwords,
        )
        vocab_size = len(word2idx)
        if vocab_size == 0:
            z = np.ones((len(documents), len(documents)))
            return SimilarityResult(matrix=z, method="bm25",
                                    documents=list(documents))

        n_docs = len(documents)
        idf = compute_idf_weights(
            documents, word2idx, smooth=True,
            remove_stopwords=self.config.remove_stopwords,
        )

        # Build raw TF counts and document lengths
        tf_counts = np.zeros((n_docs, vocab_size), dtype=np.float64)
        doc_lens = np.zeros(n_docs, dtype=np.float64)
        tokenized_docs: List[List[str]] = []

        for i, doc in enumerate(documents):
            tokens = tokenize(doc, remove_stopwords=self.config.remove_stopwords)
            tokenized_docs.append(tokens)
            doc_lens[i] = len(tokens)
            for tok in tokens:
                if tok in word2idx:
                    tf_counts[i, word2idx[tok]] += 1

        avgdl = float(np.mean(doc_lens)) if n_docs > 0 else 1.0

        # BM25 score: for document d scored against query q
        # score(d, q) = sum_t idf(t) * (tf_d(t) * (k1+1)) / (tf_d(t) + k1*(1 - b + b*|d|/avgdl))
        #   where the sum is over terms t in q
        # We treat each document as a query (using its unique terms).

        scores = np.zeros((n_docs, n_docs), dtype=np.float64)
        for qi in range(n_docs):
            query_terms = set(tokenized_docs[qi]) & set(word2idx)
            if not query_terms:
                continue
            q_indices = np.array([word2idx[t] for t in query_terms])
            for di in range(n_docs):
                tf_d = tf_counts[di, q_indices]
                idf_q = idf[q_indices]
                dl = doc_lens[di]
                denom = tf_d + k1 * (1.0 - b + b * dl / max(avgdl, 1e-12))
                numer = tf_d * (k1 + 1.0)
                scores[qi, di] = float(np.sum(idf_q * numer / np.maximum(denom, 1e-12)))

        # Symmetrise and normalise to [0, 1]
        scores = symmetrize_matrix(scores)
        scores = normalize_matrix(scores, method="minmax")
        np.fill_diagonal(scores, 1.0)

        mask = ~np.eye(n_docs, dtype=bool)
        stats = {
            "mean": float(np.mean(scores[mask])) if n_docs > 1 else 1.0,
            "std": float(np.std(scores[mask])) if n_docs > 1 else 0.0,
            "min": float(np.min(scores[mask])) if n_docs > 1 else 1.0,
            "max": float(np.max(scores[mask])) if n_docs > 1 else 1.0,
        }
        return SimilarityResult(
            matrix=scores,
            method="bm25",
            statistics=stats,
            documents=list(documents),
            vocabulary_size=vocab_size,
        )

    # -----------------------------------------------------------------
    # Jaccard
    # -----------------------------------------------------------------

    def compute_jaccard_similarity_matrix(
        self,
        documents: Sequence[str],
    ) -> SimilarityResult:
        """Compute Jaccard similarity between token sets."""
        logger.debug("Computing Jaccard similarity for %d documents.", len(documents))
        n = len(documents)
        token_sets: List[Set[str]] = []
        for doc in documents:
            token_sets.append(
                set(tokenize(doc, remove_stopwords=self.config.remove_stopwords))
            )

        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                inter = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                val = inter / max(union, 1)
                sim[i, j] = val
                sim[j, i] = val

        mask = ~np.eye(n, dtype=bool)
        stats = {
            "mean": float(np.mean(sim[mask])) if n > 1 else 1.0,
            "std": float(np.std(sim[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=sim, method="jaccard", statistics=stats,
            documents=list(documents),
        )

    # -----------------------------------------------------------------
    # Overlap coefficient
    # -----------------------------------------------------------------

    def compute_overlap_coefficient_matrix(
        self,
        documents: Sequence[str],
    ) -> SimilarityResult:
        """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
        logger.debug("Computing overlap coefficient for %d documents.", len(documents))
        n = len(documents)
        token_sets: List[Set[str]] = [
            set(tokenize(doc, remove_stopwords=self.config.remove_stopwords))
            for doc in documents
        ]

        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                inter = len(token_sets[i] & token_sets[j])
                denom = min(len(token_sets[i]), len(token_sets[j]))
                val = inter / max(denom, 1)
                sim[i, j] = val
                sim[j, i] = val

        mask = ~np.eye(n, dtype=bool)
        stats = {
            "mean": float(np.mean(sim[mask])) if n > 1 else 1.0,
            "std": float(np.std(sim[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=sim, method="overlap", statistics=stats,
            documents=list(documents),
        )

    # -----------------------------------------------------------------
    # Dice coefficient
    # -----------------------------------------------------------------

    def compute_dice_similarity_matrix(
        self,
        documents: Sequence[str],
    ) -> SimilarityResult:
        """Sørensen–Dice coefficient: 2|A ∩ B| / (|A| + |B|)."""
        logger.debug("Computing Dice similarity for %d documents.", len(documents))
        n = len(documents)
        token_sets: List[Set[str]] = [
            set(tokenize(doc, remove_stopwords=self.config.remove_stopwords))
            for doc in documents
        ]

        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                inter = len(token_sets[i] & token_sets[j])
                denom = len(token_sets[i]) + len(token_sets[j])
                val = 2.0 * inter / max(denom, 1)
                sim[i, j] = val
                sim[j, i] = val

        mask = ~np.eye(n, dtype=bool)
        stats = {
            "mean": float(np.mean(sim[mask])) if n > 1 else 1.0,
            "std": float(np.std(sim[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=sim, method="dice", statistics=stats,
            documents=list(documents),
        )

    # -----------------------------------------------------------------
    # LSI (Latent Semantic Indexing via truncated SVD)
    # -----------------------------------------------------------------

    def compute_lsi_similarity(
        self,
        documents: Sequence[str],
        n_components: Optional[int] = None,
    ) -> SimilarityResult:
        """Latent Semantic Indexing: TF-IDF → truncated SVD → cosine sim."""
        logger.debug("Computing LSI similarity for %d documents.", len(documents))
        n_comp = n_components or self.config.n_components
        tfidf, word2idx, idf = self.compute_tfidf_vectors(documents)

        actual_components = min(n_comp, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if actual_components < 1:
            actual_components = 1

        # Truncated SVD
        U, sigma, Vt = scipy_svd(tfidf, full_matrices=False)
        U_k = U[:, :actual_components]
        sigma_k = sigma[:actual_components]
        # Project documents into latent space
        projected = U_k * sigma_k[None, :]

        sim = cosine_similarity_matrix(projected)
        np.fill_diagonal(sim, 1.0)
        # Clamp to [0, 1] since we are measuring similarity
        np.clip(sim, 0.0, 1.0, out=sim)

        mask = ~np.eye(len(documents), dtype=bool)
        n = len(documents)
        stats = {
            "n_components": actual_components,
            "explained_variance_ratio": float(
                np.sum(sigma_k ** 2) / max(np.sum(sigma ** 2), 1e-12)
            ),
            "mean": float(np.mean(sim[mask])) if n > 1 else 1.0,
            "std": float(np.std(sim[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=sim, method="lsi", statistics=stats,
            documents=list(documents), vocabulary_size=len(word2idx),
        )

    # -----------------------------------------------------------------
    # Word Mover Approximation
    # -----------------------------------------------------------------

    def compute_word_mover_approximation(
        self,
        documents: Sequence[str],
    ) -> SimilarityResult:
        """Approximate Word Mover's Distance using word-frequency vectors
        and the relaxed dual bound (sum of minimum per-word transport costs).

        Since we lack real embeddings we use IDF-weighted bag-of-words
        and Euclidean distance in that space as a proxy.  The result is
        normalised and converted to a similarity.
        """
        logger.debug("Computing WMD approximation for %d documents.", len(documents))
        word2idx, idx2word = build_vocabulary(
            documents,
            min_df=self.config.min_df,
            max_df_ratio=self.config.max_df_ratio,
            remove_stopwords=self.config.remove_stopwords,
        )
        vocab_size = len(word2idx)
        if vocab_size == 0:
            return SimilarityResult(
                matrix=np.ones((len(documents), len(documents))),
                method="wmd_approx",
                documents=list(documents),
            )

        idf = compute_idf_weights(
            documents, word2idx, remove_stopwords=self.config.remove_stopwords,
        )
        n = len(documents)

        # Build normalised word-frequency (nBOW) vectors
        nbow = np.zeros((n, vocab_size), dtype=np.float64)
        for i, doc in enumerate(documents):
            tokens = tokenize(doc, remove_stopwords=self.config.remove_stopwords)
            for tok in tokens:
                if tok in word2idx:
                    nbow[i, word2idx[tok]] += 1.0
            row_sum = nbow[i].sum()
            if row_sum > 0:
                nbow[i] /= row_sum

        # Weight by IDF to give more importance to rare words
        weighted = nbow * idf[None, :]

        # Relaxed WMD lower bound: for each word in doc_i, find cheapest
        # transport to any word in doc_j.  We approximate word-word cost
        # as absolute difference in IDF (since we lack real embeddings).
        # Instead, we use a simpler proxy: Euclidean distance between
        # the weighted nBOW vectors.
        dist = euclidean_distance_matrix(weighted)

        # Convert distance to similarity in [0, 1]
        max_dist = dist.max()
        if max_dist < 1e-12:
            sim = np.ones((n, n))
        else:
            sim = 1.0 - dist / max_dist
        np.fill_diagonal(sim, 1.0)

        mask = ~np.eye(n, dtype=bool)
        stats = {
            "mean": float(np.mean(sim[mask])) if n > 1 else 1.0,
            "std": float(np.std(sim[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=sim, method="wmd_approx", statistics=stats,
            documents=list(documents), vocabulary_size=vocab_size,
        )

    # -----------------------------------------------------------------
    # N-gram similarity
    # -----------------------------------------------------------------

    def compute_ngram_similarity_matrix(
        self,
        documents: Sequence[str],
        char_ngram_n: int = 3,
        word_ngram_n: int = 2,
        char_weight: float = 0.5,
    ) -> SimilarityResult:
        """Compute similarity based on character and word n-gram overlap.

        The final similarity is a weighted average of the Jaccard similarity
        of character n-gram multisets and word n-gram multisets.
        """
        logger.debug("Computing n-gram similarity for %d documents.", len(documents))
        n = len(documents)

        # Precompute n-gram sets
        char_ngram_sets: List[Counter] = []
        word_ngram_sets: List[Counter] = []
        for doc in documents:
            cng = Counter(_character_ngrams(doc, char_ngram_n))
            char_ngram_sets.append(cng)

            tokens = tokenize(doc, remove_stopwords=False)
            wng = Counter(_word_ngrams(tokens, word_ngram_n))
            word_ngram_sets.append(wng)

        def _counter_jaccard(a: Counter, b: Counter) -> float:
            if not a and not b:
                return 1.0
            keys = set(a) | set(b)
            inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
            union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
            return inter / max(union, 1)

        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                char_sim = _counter_jaccard(char_ngram_sets[i], char_ngram_sets[j])
                word_sim = _counter_jaccard(word_ngram_sets[i], word_ngram_sets[j])
                combined = char_weight * char_sim + (1.0 - char_weight) * word_sim
                sim[i, j] = combined
                sim[j, i] = combined

        mask = ~np.eye(n, dtype=bool)
        stats = {
            "char_ngram_n": char_ngram_n,
            "word_ngram_n": word_ngram_n,
            "mean": float(np.mean(sim[mask])) if n > 1 else 1.0,
            "std": float(np.std(sim[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=sim, method="ngram", statistics=stats,
            documents=list(documents),
        )

    # -----------------------------------------------------------------
    # Combined similarity
    # -----------------------------------------------------------------

    def compute_combined_similarity(
        self,
        documents: Sequence[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> SimilarityResult:
        """Compute a weighted combination of multiple similarity methods.

        Parameters
        ----------
        documents : sequence of strings
        weights : dict mapping method name to weight.  Defaults to
            ``self.config.combined_weights``.
        """
        logger.debug("Computing combined similarity for %d documents.", len(documents))
        weights = weights or dict(self.config.combined_weights)

        method_map: Dict[str, Callable[..., SimilarityResult]] = {
            "tfidf": self.compute_tfidf_similarity,
            "bm25": self.compute_bm25_similarity,
            "jaccard": self.compute_jaccard_similarity_matrix,
            "overlap": self.compute_overlap_coefficient_matrix,
            "dice": self.compute_dice_similarity_matrix,
            "lsi": self.compute_lsi_similarity,
            "wmd_approx": self.compute_word_mover_approximation,
            "ngram": self.compute_ngram_similarity_matrix,
        }

        n = len(documents)
        combined = np.zeros((n, n), dtype=np.float64)
        total_weight = 0.0
        individual_stats: Dict[str, Dict[str, Any]] = {}

        for name, w in weights.items():
            if w <= 0 or name not in method_map:
                continue
            result = method_map[name](documents)
            combined += w * result.matrix
            total_weight += w
            individual_stats[name] = result.statistics

        if total_weight > 0:
            combined /= total_weight
        np.fill_diagonal(combined, 1.0)
        np.clip(combined, 0.0, 1.0, out=combined)

        mask = ~np.eye(n, dtype=bool)
        stats: Dict[str, Any] = {
            "weights": weights,
            "individual": individual_stats,
            "mean": float(np.mean(combined[mask])) if n > 1 else 1.0,
            "std": float(np.std(combined[mask])) if n > 1 else 0.0,
        }
        return SimilarityResult(
            matrix=combined, method="combined", statistics=stats,
            documents=list(documents),
        )


# =========================================================================
# PairwiseSimilarityMatrix
# =========================================================================

class PairwiseSimilarityMatrix:
    """Operate on a pre-computed NxN similarity matrix."""

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        documents: Optional[Sequence[str]] = None,
        config: Optional[SimilarityConfig] = None,
    ) -> None:
        self.config = config or SimilarityConfig()
        self._matrix: Optional[np.ndarray] = matrix
        self._documents = list(documents) if documents else []
        self._computer = SemanticSimilarityComputer(self.config)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("Matrix not yet computed; call compute() first.")
        return self._matrix

    # -----------------------------------------------------------------
    # compute
    # -----------------------------------------------------------------

    def compute(
        self,
        documents: Optional[Sequence[str]] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Build the full NxN similarity matrix.

        Parameters
        ----------
        documents : optional new document set (replaces stored docs).
        method : similarity method name (default from config).

        Returns
        -------
        S : ndarray of shape (N, N)
        """
        if documents is not None:
            self._documents = list(documents)
        method = method or self.config.method

        method_map: Dict[str, Callable[..., SimilarityResult]] = {
            "tfidf": self._computer.compute_tfidf_similarity,
            "bm25": self._computer.compute_bm25_similarity,
            "jaccard": self._computer.compute_jaccard_similarity_matrix,
            "overlap": self._computer.compute_overlap_coefficient_matrix,
            "dice": self._computer.compute_dice_similarity_matrix,
            "lsi": self._computer.compute_lsi_similarity,
            "wmd_approx": self._computer.compute_word_mover_approximation,
            "ngram": self._computer.compute_ngram_similarity_matrix,
            "combined": self._computer.compute_combined_similarity,
        }

        if method not in method_map:
            raise ValueError(f"Unknown similarity method: {method}")

        result = method_map[method](self._documents)
        self._matrix = result.matrix
        return self._matrix

    # -----------------------------------------------------------------
    # Pair queries
    # -----------------------------------------------------------------

    def get_most_similar_pairs(
        self,
        k: int = 5,
        exclude_diagonal: bool = True,
    ) -> List[Tuple[int, int, float]]:
        """Return the *k* most similar (i, j) pairs (i < j)."""
        S = self.matrix.copy()
        if exclude_diagonal:
            np.fill_diagonal(S, -np.inf)
        n = S.shape[0]
        # Only upper triangle
        upper = np.triu_indices(n, k=1)
        vals = S[upper]
        top_k = min(k, len(vals))
        if top_k == 0:
            return []
        indices = np.argpartition(vals, -top_k)[-top_k:]
        indices = indices[np.argsort(-vals[indices])]
        return [
            (int(upper[0][idx]), int(upper[1][idx]), float(vals[idx]))
            for idx in indices
        ]

    def get_least_similar_pairs(
        self,
        k: int = 5,
    ) -> List[Tuple[int, int, float]]:
        """Return the *k* least similar (i, j) pairs (i < j)."""
        S = self.matrix.copy()
        np.fill_diagonal(S, np.inf)
        n = S.shape[0]
        upper = np.triu_indices(n, k=1)
        vals = S[upper]
        top_k = min(k, len(vals))
        if top_k == 0:
            return []
        indices = np.argpartition(vals, top_k)[:top_k]
        indices = indices[np.argsort(vals[indices])]
        return [
            (int(upper[0][idx]), int(upper[1][idx]), float(vals[idx]))
            for idx in indices
        ]

    # -----------------------------------------------------------------
    # Distribution & diversity
    # -----------------------------------------------------------------

    def compute_similarity_distribution(self) -> Dict[str, float]:
        """Compute statistics over off-diagonal similarity entries."""
        n = self.matrix.shape[0]
        if n < 2:
            return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0,
                    "median": 1.0, "q25": 1.0, "q75": 1.0, "skewness": 0.0,
                    "kurtosis": 0.0}
        mask = ~np.eye(n, dtype=bool)
        vals = self.matrix[mask]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        skew = 0.0
        kurt = 0.0
        if sigma > 1e-12:
            centered = vals - mu
            skew = float(np.mean(centered ** 3) / sigma ** 3)
            kurt = float(np.mean(centered ** 4) / sigma ** 4 - 3.0)
        return {
            "mean": mu,
            "std": sigma,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "skewness": skew,
            "kurtosis": kurt,
        }

    def compute_diversity_from_similarity(self) -> float:
        """Diversity = 1 − mean off-diagonal similarity."""
        n = self.matrix.shape[0]
        if n < 2:
            return 0.0
        mask = ~np.eye(n, dtype=bool)
        return 1.0 - float(np.mean(self.matrix[mask]))

    # -----------------------------------------------------------------
    # Matrix transforms
    # -----------------------------------------------------------------

    def to_distance_matrix(self) -> np.ndarray:
        """Convert similarity to distance: D = 1 − S, clamped ≥ 0."""
        D = 1.0 - self.matrix
        np.maximum(D, 0.0, out=D)
        np.fill_diagonal(D, 0.0)
        return D

    def to_kernel_matrix(self, gamma: float = 1.0) -> np.ndarray:
        """Convert similarity to an RBF-style kernel matrix.

        K_ij = exp(−gamma * D_ij^2)  where D = 1 − S.
        """
        D = self.to_distance_matrix()
        K = np.exp(-gamma * D ** 2)
        return K

    # -----------------------------------------------------------------
    # Spectral analysis
    # -----------------------------------------------------------------

    def spectral_analysis(
        self,
        n_eigenvalues: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Eigenvalue decomposition of the similarity matrix.

        Returns eigenvalues (descending), eigenvectors, and derived
        diversity statistics.
        """
        S = self.matrix.copy()
        S = symmetrize_matrix(S)
        n = S.shape[0]
        n_eig = n_eigenvalues or n

        if n <= 0:
            return {"eigenvalues": np.array([]), "eigenvectors": np.array([])}

        eigenvalues, eigenvectors = np.linalg.eigh(S)
        # Sort descending
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx][:n_eig]
        eigenvectors = eigenvectors[:, idx][:, :n_eig]

        # Fraction of variance explained by top-k eigenvalues
        total = float(np.sum(np.abs(eigenvalues)))
        cumulative = np.cumsum(np.abs(eigenvalues)) / max(total, 1e-12)

        positive = eigenvalues[eigenvalues > 1e-12]
        shannon_entropy = 0.0
        if len(positive) > 0:
            p = positive / positive.sum()
            shannon_entropy = -float(np.sum(p * np.log(p + 1e-30)))

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "cumulative_variance": cumulative,
            "spectral_entropy": shannon_entropy,
            "spectral_gap": float(eigenvalues[0] - eigenvalues[1]) if n >= 2 else 0.0,
            "rank": int(np.sum(np.abs(eigenvalues) > 1e-8)),
        }

    def effective_diversity(self) -> float:
        """Compute effective diversity (Vendi-score style).

        The Vendi score is defined as exp(H) where H is the von Neumann
        entropy of the normalised kernel matrix:
          H = -Σ λ_i log(λ_i)
        with λ_i the eigenvalues of K / tr(K).
        """
        S = self.matrix.copy()
        S = symmetrize_matrix(S)
        trace = np.trace(S)
        if trace < 1e-12:
            return 1.0
        K_norm = S / trace
        eigenvalues = np.linalg.eigvalsh(K_norm)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            return 1.0
        entropy = -float(np.sum(eigenvalues * np.log(eigenvalues + 1e-30)))
        return float(np.exp(entropy))


# =========================================================================
# SemanticClustering
# =========================================================================

class SemanticClustering:
    """Clustering algorithms operating on similarity / distance matrices.

    All algorithms are implemented from scratch using numpy/scipy.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None) -> None:
        self.config = config or SimilarityConfig()
        self._rng = np.random.RandomState(self.config.random_state)

    # -----------------------------------------------------------------
    # K-means (from scratch)
    # -----------------------------------------------------------------

    def kmeans_clustering(
        self,
        X: np.ndarray,
        n_clusters: Optional[int] = None,
        max_iter: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-means clustering implemented from scratch.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        n_clusters : number of clusters (default from config)
        max_iter : maximum iterations

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        centroids : ndarray of shape (n_clusters, n_features)
        """
        k = n_clusters or self.config.n_clusters
        max_it = max_iter or self.config.max_kmeans_iter
        n_samples, n_features = X.shape
        k = min(k, n_samples)

        # K-means++ initialisation
        centroids = np.empty((k, n_features), dtype=np.float64)
        idx = self._rng.randint(n_samples)
        centroids[0] = X[idx]

        for c in range(1, k):
            dists = np.min(
                np.sum((X[:, None, :] - centroids[None, :c, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / max(dists.sum(), 1e-12)
            idx = self._rng.choice(n_samples, p=probs)
            centroids[c] = X[idx]

        labels = np.zeros(n_samples, dtype=np.int64)

        for iteration in range(max_it):
            # Assignment step
            dist_to_centroids = np.sum(
                (X[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(dist_to_centroids, axis=1)

            # Update step
            new_centroids = np.empty_like(centroids)
            for c in range(k):
                members = X[new_labels == c]
                if len(members) == 0:
                    new_centroids[c] = X[self._rng.randint(n_samples)]
                else:
                    new_centroids[c] = members.mean(axis=0)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centroids = new_centroids

        return labels, centroids

    # -----------------------------------------------------------------
    # Spectral clustering
    # -----------------------------------------------------------------

    def spectral_clustering(
        self,
        similarity_matrix: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """Spectral clustering using the similarity matrix.

        1. Build the normalised graph Laplacian.
        2. Embed in the space of the first k eigenvectors.
        3. Run k-means in that space.
        """
        k = n_clusters or self.config.n_clusters
        S = similarity_matrix.copy()
        np.fill_diagonal(S, 0.0)
        S = np.maximum(S, 0.0)
        n = S.shape[0]
        k = min(k, n)

        # Degree matrix
        D = np.diag(S.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))

        # Normalised Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}
        L_sym = np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt

        # Eigen-decomposition (smallest eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        # Take the first k eigenvectors (smallest eigenvalues)
        embedding = eigenvectors[:, :k].copy()

        # Row-normalise
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embedding /= norms

        labels, _ = self.kmeans_clustering(embedding, n_clusters=k)
        return labels

    # -----------------------------------------------------------------
    # Hierarchical / agglomerative clustering
    # -----------------------------------------------------------------

    def hierarchical_clustering(
        self,
        distance_matrix: np.ndarray,
        n_clusters: Optional[int] = None,
        linkage: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Agglomerative hierarchical clustering from a distance matrix.

        Supports 'single', 'complete', 'average', and 'ward' linkage
        (ward requires feature vectors; here we approximate with average
        if a pure distance matrix is given).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        merge_distances : ndarray of merge distances at each step
        """
        k = n_clusters or self.config.n_clusters
        link = linkage or self.config.linkage_method
        n = distance_matrix.shape[0]
        k = min(k, n)

        # Each sample starts in its own cluster
        cluster_ids: List[Set[int]] = [{i} for i in range(n)]
        active = list(range(n))
        D = distance_matrix.copy().astype(np.float64)
        np.fill_diagonal(D, np.inf)
        merge_distances: List[float] = []

        while len(active) > k:
            # Find closest pair among active clusters
            min_dist = np.inf
            mi, mj = -1, -1
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    ci, cj = active[ii], active[jj]
                    d = self._cluster_distance(
                        cluster_ids[ci], cluster_ids[cj], distance_matrix, link,
                    )
                    if d < min_dist:
                        min_dist = d
                        mi, mj = ii, jj

            if mi < 0:
                break

            merge_distances.append(min_dist)
            ci_idx, cj_idx = active[mi], active[mj]
            # Merge cj into ci
            cluster_ids[ci_idx] = cluster_ids[ci_idx] | cluster_ids[cj_idx]
            active.pop(mj)

        # Build label array
        labels = np.full(n, -1, dtype=np.int64)
        for label, a_idx in enumerate(active):
            for member in cluster_ids[a_idx]:
                labels[member] = label

        return labels, np.array(merge_distances, dtype=np.float64)

    @staticmethod
    def _cluster_distance(
        c1: Set[int],
        c2: Set[int],
        D: np.ndarray,
        linkage: str,
    ) -> float:
        dists = [D[i, j] for i in c1 for j in c2]
        if not dists:
            return np.inf
        if linkage == "single":
            return float(min(dists))
        elif linkage == "complete":
            return float(max(dists))
        elif linkage == "average":
            return float(np.mean(dists))
        elif linkage == "ward":
            # Ward approximation via average for distance-matrix-only case
            return float(np.mean(dists))
        else:
            return float(np.mean(dists))

    # -----------------------------------------------------------------
    # DBSCAN (from scratch)
    # -----------------------------------------------------------------

    def dbscan_clustering(
        self,
        distance_matrix: np.ndarray,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> np.ndarray:
        """DBSCAN clustering implemented from scratch on a distance matrix.

        Returns
        -------
        labels : ndarray where −1 indicates noise.
        """
        _eps = eps if eps is not None else self.config.dbscan_eps
        _min_samples = min_samples if min_samples is not None else self.config.dbscan_min_samples
        n = distance_matrix.shape[0]
        labels = np.full(n, -1, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        def _region_query(idx: int) -> List[int]:
            return [j for j in range(n) if distance_matrix[idx, j] <= _eps]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbours = _region_query(i)
            if len(neighbours) < _min_samples:
                # Mark as noise (may be claimed later)
                continue

            labels[i] = cluster_id
            seed_set = list(neighbours)
            ptr = 0
            while ptr < len(seed_set):
                q = seed_set[ptr]
                ptr += 1
                if not visited[q]:
                    visited[q] = True
                    q_neighbours = _region_query(q)
                    if len(q_neighbours) >= _min_samples:
                        seed_set.extend(q_neighbours)
                if labels[q] == -1:
                    labels[q] = cluster_id
            cluster_id += 1

        return labels

    # -----------------------------------------------------------------
    # Cluster quality metrics
    # -----------------------------------------------------------------

    def cluster_quality_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute silhouette, Davies-Bouldin, and Calinski-Harabasz indices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)
        """
        unique_labels = sorted(set(labels))
        # Remove noise label if present
        unique_labels = [l for l in unique_labels if l >= 0]
        n_clusters = len(unique_labels)
        n = X.shape[0]

        if n_clusters < 2 or n_clusters >= n:
            return {
                "silhouette": 0.0,
                "davies_bouldin": float("inf"),
                "calinski_harabasz": 0.0,
            }

        # Precompute distance matrix
        D = euclidean_distance_matrix(X)

        # --- Silhouette ---
        silhouette_vals = np.zeros(n)
        for i in range(n):
            li = labels[i]
            if li < 0:
                silhouette_vals[i] = 0.0
                continue
            same = [j for j in range(n) if labels[j] == li and j != i]
            if not same:
                silhouette_vals[i] = 0.0
                continue
            a_i = np.mean(D[i, same])
            b_i = np.inf
            for other_label in unique_labels:
                if other_label == li:
                    continue
                diff = [j for j in range(n) if labels[j] == other_label]
                if diff:
                    b_i = min(b_i, np.mean(D[i, diff]))
            silhouette_vals[i] = (b_i - a_i) / max(max(a_i, b_i), 1e-12)

        silhouette = float(np.mean(silhouette_vals[labels >= 0]))

        # --- Davies-Bouldin ---
        centroids = np.array([
            X[labels == l].mean(axis=0) for l in unique_labels
        ])
        scatter = np.array([
            np.mean(np.linalg.norm(X[labels == l] - centroids[ci], axis=1))
            for ci, l in enumerate(unique_labels)
        ])
        db = 0.0
        for ci in range(n_clusters):
            max_ratio = 0.0
            for cj in range(n_clusters):
                if ci == cj:
                    continue
                dist_ij = np.linalg.norm(centroids[ci] - centroids[cj])
                ratio = (scatter[ci] + scatter[cj]) / max(dist_ij, 1e-12)
                max_ratio = max(max_ratio, ratio)
            db += max_ratio
        db /= max(n_clusters, 1)

        # --- Calinski-Harabasz ---
        overall_center = X[labels >= 0].mean(axis=0) if np.any(labels >= 0) else X.mean(axis=0)
        n_valid = int(np.sum(labels >= 0))
        bgss = sum(
            np.sum(labels == l) * np.sum((centroids[ci] - overall_center) ** 2)
            for ci, l in enumerate(unique_labels)
        )
        wgss = sum(
            np.sum((X[labels == l] - centroids[ci]) ** 2)
            for ci, l in enumerate(unique_labels)
        )
        ch = 0.0
        if wgss > 1e-12 and n_valid > n_clusters:
            ch = (bgss / max(n_clusters - 1, 1)) / (wgss / max(n_valid - n_clusters, 1))

        return {
            "silhouette": silhouette,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
        }

    # -----------------------------------------------------------------
    # Optimal number of clusters
    # -----------------------------------------------------------------

    def optimal_num_clusters(
        self,
        X: np.ndarray,
        max_k: int = 10,
        method: str = "elbow",
    ) -> Dict[str, Any]:
        """Determine optimal number of clusters via elbow or gap statistic.

        Parameters
        ----------
        X : feature matrix
        max_k : maximum k to try
        method : 'elbow' or 'gap'

        Returns
        -------
        dict with 'optimal_k', 'inertias' / 'gap_values', etc.
        """
        n = X.shape[0]
        max_k = min(max_k, n - 1) if n > 2 else 1
        if max_k < 2:
            return {"optimal_k": 1, "inertias": [], "method": method}

        ks = list(range(1, max_k + 1))
        inertias: List[float] = []

        for k in ks:
            labels, centroids = self.kmeans_clustering(X, n_clusters=k)
            inertia = 0.0
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    inertia += float(np.sum((members - centroids[c]) ** 2))
            inertias.append(inertia)

        if method == "elbow":
            # Elbow detection via maximum second derivative
            if len(inertias) < 3:
                optimal_k = 1
            else:
                arr = np.array(inertias)
                second_deriv = np.diff(arr, n=2)
                optimal_k = int(np.argmax(second_deriv)) + 2
                optimal_k = max(2, min(optimal_k, max_k))
            return {
                "optimal_k": optimal_k,
                "inertias": inertias,
                "ks": ks,
                "method": "elbow",
            }

        elif method == "gap":
            # Gap statistic
            gap_values: List[float] = []
            gap_stds: List[float] = []
            n_ref = 10
            for ki, k in enumerate(ks):
                log_wk = math.log(max(inertias[ki], 1e-12))
                ref_log_wks: List[float] = []
                for _ in range(n_ref):
                    X_ref = self._rng.uniform(
                        X.min(axis=0), X.max(axis=0), size=X.shape,
                    )
                    ref_labels, ref_centroids = self.kmeans_clustering(X_ref, n_clusters=k)
                    ref_inertia = 0.0
                    for c in range(k):
                        members = X_ref[ref_labels == c]
                        if len(members) > 0:
                            ref_inertia += float(np.sum((members - ref_centroids[c]) ** 2))
                    ref_log_wks.append(math.log(max(ref_inertia, 1e-12)))
                mean_ref = float(np.mean(ref_log_wks))
                std_ref = float(np.std(ref_log_wks)) * math.sqrt(1 + 1.0 / n_ref)
                gap_values.append(mean_ref - log_wk)
                gap_stds.append(std_ref)

            # Optimal k: first k where gap(k) >= gap(k+1) - s(k+1)
            optimal_k = ks[-1]
            for ki in range(len(ks) - 1):
                if gap_values[ki] >= gap_values[ki + 1] - gap_stds[ki + 1]:
                    optimal_k = ks[ki]
                    break

            return {
                "optimal_k": optimal_k,
                "gap_values": gap_values,
                "gap_stds": gap_stds,
                "inertias": inertias,
                "ks": ks,
                "method": "gap",
            }

        raise ValueError(f"Unknown method: {method}")

    # -----------------------------------------------------------------
    # Cluster diversity
    # -----------------------------------------------------------------

    def compute_cluster_diversity(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute inter-cluster and intra-cluster distance statistics.

        Returns
        -------
        dict with 'intra_mean', 'inter_mean', 'ratio', per-cluster stats.
        """
        unique_labels = sorted(set(labels))
        unique_labels = [l for l in unique_labels if l >= 0]
        n_clusters = len(unique_labels)

        if n_clusters < 1:
            return {"intra_mean": 0.0, "inter_mean": 0.0, "ratio": 0.0}

        centroids = np.array([X[labels == l].mean(axis=0) for l in unique_labels])

        intra_dists: List[float] = []
        per_cluster: List[Dict[str, Any]] = []
        for ci, l in enumerate(unique_labels):
            members = X[labels == l]
            if len(members) < 2:
                intra_dists.append(0.0)
                per_cluster.append({
                    "label": int(l),
                    "size": len(members),
                    "intra_mean_dist": 0.0,
                    "intra_max_dist": 0.0,
                })
                continue
            dists = pdist(members, metric="euclidean")
            intra_dists.append(float(np.mean(dists)))
            per_cluster.append({
                "label": int(l),
                "size": len(members),
                "intra_mean_dist": float(np.mean(dists)),
                "intra_max_dist": float(np.max(dists)),
            })

        inter_dists: List[float] = []
        if n_clusters >= 2:
            centroid_dists = pdist(centroids, metric="euclidean")
            inter_dists = centroid_dists.tolist()

        intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
        inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0

        ratio = inter_mean / max(intra_mean, 1e-12) if intra_mean > 0 else 0.0

        return {
            "intra_mean": intra_mean,
            "inter_mean": inter_mean,
            "ratio": ratio,
            "n_clusters": n_clusters,
            "per_cluster": per_cluster,
        }


# =========================================================================
# CoverageAnalyzer
# =========================================================================

class CoverageAnalyzer:
    """Analyze how well a set of outputs covers the semantic space."""

    def __init__(self, config: Optional[SimilarityConfig] = None) -> None:
        self.config = config or SimilarityConfig()
        self._rng = np.random.RandomState(self.config.random_state)

    # -----------------------------------------------------------------
    # Semantic coverage
    # -----------------------------------------------------------------

    def compute_semantic_coverage(
        self,
        similarity_matrix: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Measure how well outputs cover the semantic space.

        For each document, we check whether at least one other document
        is *dissimilar enough* (similarity < threshold) to indicate
        that different regions of the space are being explored.

        We also compute a coverage radius: the mean minimum distance to
        any other point, which indicates how spread out the points are.
        """
        n = similarity_matrix.shape[0]
        if n < 2:
            return {"coverage_score": 0.0, "coverage_radius": 0.0,
                    "n_covered_regions": 1}

        D = 1.0 - similarity_matrix
        np.fill_diagonal(D, np.inf)

        # Minimum distance to any neighbour
        min_dists = np.min(D, axis=1)
        np.clip(min_dists, 0, None, out=min_dists)
        coverage_radius = float(np.mean(min_dists))

        # Count how many "distinct" regions exist at the given threshold
        # Two documents are in the same region if similarity > threshold
        visited = np.zeros(n, dtype=bool)
        n_regions = 0
        for i in range(n):
            if visited[i]:
                continue
            n_regions += 1
            stack = [i]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                for j in range(n):
                    if not visited[j] and similarity_matrix[node, j] >= threshold:
                        stack.append(j)

        # Coverage score: higher when more distinct regions
        # Normalise by n so perfect coverage (every doc is its own region) = 1
        coverage_score = n_regions / n

        return {
            "coverage_score": coverage_score,
            "coverage_radius": coverage_radius,
            "n_covered_regions": n_regions,
            "threshold": threshold,
        }

    # -----------------------------------------------------------------
    # Novelty scores
    # -----------------------------------------------------------------

    def compute_novelty_scores(
        self,
        similarity_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute a novelty score for each document: how dissimilar it is
        from all other documents.

        novelty_i = 1 − max_{j≠i} sim(i, j)

        High novelty means the document is unlike anything else.
        """
        n = similarity_matrix.shape[0]
        if n < 2:
            return np.zeros(n)
        S = similarity_matrix.copy()
        np.fill_diagonal(S, -np.inf)
        max_sim = np.max(S, axis=1)
        novelty = 1.0 - max_sim
        np.clip(novelty, 0.0, 1.0, out=novelty)
        return novelty

    # -----------------------------------------------------------------
    # Redundancy scores
    # -----------------------------------------------------------------

    def compute_redundancy_scores(
        self,
        similarity_matrix: np.ndarray,
        threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """Identify near-duplicate documents.

        A document is "redundant" if its similarity to at least one other
        document exceeds *threshold*.

        Returns per-document redundancy flag, redundant pairs, and an
        overall redundancy ratio.
        """
        n = similarity_matrix.shape[0]
        is_redundant = np.zeros(n, dtype=bool)
        redundant_pairs: List[Tuple[int, int, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    is_redundant[i] = True
                    is_redundant[j] = True
                    redundant_pairs.append((i, j, float(similarity_matrix[i, j])))

        redundancy_ratio = float(np.mean(is_redundant)) if n > 0 else 0.0

        # Per-document redundancy score: average similarity to k nearest
        per_doc_scores = np.zeros(n)
        if n >= 2:
            S = similarity_matrix.copy()
            np.fill_diagonal(S, 0.0)
            per_doc_scores = np.mean(np.sort(S, axis=1)[:, -min(3, n - 1):], axis=1)

        return {
            "is_redundant": is_redundant,
            "redundant_pairs": redundant_pairs,
            "redundancy_ratio": redundancy_ratio,
            "per_document_scores": per_doc_scores,
            "threshold": threshold,
        }

    # -----------------------------------------------------------------
    # Coverage uniformity
    # -----------------------------------------------------------------

    def compute_coverage_uniformity(
        self,
        X: np.ndarray,
    ) -> Dict[str, float]:
        """Measure how uniformly the points cover the space.

        Computes the coefficient of variation of nearest-neighbour distances,
        plus a uniformity index based on the Gini coefficient of pairwise
        distances.
        """
        n = X.shape[0]
        if n < 2:
            return {"uniformity": 1.0, "cv_nn_dist": 0.0, "gini": 0.0}

        D = euclidean_distance_matrix(X)
        np.fill_diagonal(D, np.inf)
        nn_dists = np.min(D, axis=1)

        mean_nn = float(np.mean(nn_dists))
        std_nn = float(np.std(nn_dists))
        cv = std_nn / max(mean_nn, 1e-12)

        # Gini coefficient of pairwise distances
        pw = squareform(D, checks=False) if n > 1 else np.array([0.0])
        # D has np.inf on diagonal; extract upper triangle manually
        pw_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                pw_vals.append(D[i, j])
        pw_arr = np.array(pw_vals)
        if len(pw_arr) == 0 or pw_arr.sum() < 1e-12:
            gini = 0.0
        else:
            sorted_pw = np.sort(pw_arr)
            m = len(sorted_pw)
            cum = np.cumsum(sorted_pw)
            gini = float(
                (2.0 * np.sum((np.arange(1, m + 1) * sorted_pw)) / (m * cum[-1])) - (m + 1) / m
            )
            gini = max(0.0, min(gini, 1.0))

        uniformity = 1.0 - gini

        return {
            "uniformity": uniformity,
            "cv_nn_dist": cv,
            "gini": gini,
            "mean_nn_dist": mean_nn,
            "std_nn_dist": std_nn,
        }

    # -----------------------------------------------------------------
    # Greedy max-coverage subset
    # -----------------------------------------------------------------

    def greedy_max_coverage_subset(
        self,
        similarity_matrix: np.ndarray,
        subset_size: int,
    ) -> Tuple[List[int], float]:
        """Select a subset of documents that maximises coverage.

        Uses a greedy algorithm that iteratively picks the document most
        dissimilar from the already-selected set (facility-location style).

        Returns
        -------
        selected : list of selected indices
        coverage : coverage score of the selected subset
        """
        n = similarity_matrix.shape[0]
        subset_size = min(subset_size, n)
        if subset_size <= 0:
            return [], 0.0

        D = 1.0 - similarity_matrix
        np.maximum(D, 0.0, out=D)

        # Start with the document that has the highest mean distance
        mean_dists = D.sum(axis=1) / max(n - 1, 1)
        first = int(np.argmax(mean_dists))
        selected = [first]

        min_dist_to_selected = D[first].copy()
        min_dist_to_selected[first] = 0.0

        for _ in range(1, subset_size):
            # Pick the document maximising min distance to selected set
            candidates = np.copy(min_dist_to_selected)
            for s in selected:
                candidates[s] = -np.inf
            next_idx = int(np.argmax(candidates))
            selected.append(next_idx)
            # Update min distances
            np.minimum(min_dist_to_selected, D[next_idx], out=min_dist_to_selected)
            min_dist_to_selected[next_idx] = 0.0

        # Coverage: mean min distance from non-selected to selected
        remaining = [i for i in range(n) if i not in selected]
        if remaining:
            coverage = float(np.mean([
                min(D[i, s] for s in selected) for i in remaining
            ]))
        else:
            coverage = 0.0  # everything is selected

        return selected, coverage

    # -----------------------------------------------------------------
    # Volume coverage (convex hull)
    # -----------------------------------------------------------------

    def compute_volume_coverage(
        self,
        X: np.ndarray,
        n_components: int = 3,
    ) -> Dict[str, float]:
        """Estimate convex hull volume coverage in a reduced space.

        Uses PCA to reduce to *n_components* dimensions, then computes
        the convex hull volume and compares it with the bounding box.
        """
        n, d = X.shape
        if n < d + 1 or n < 4:
            return {"volume": 0.0, "bbox_volume": 0.0, "ratio": 0.0}

        reducer = DimensionalityReducer(self.config)
        X_red, _ = reducer.pca_reduce(X, n_components=min(n_components, d, n - 1))

        # Bounding box volume
        ranges = X_red.max(axis=0) - X_red.min(axis=0)
        bbox_vol = float(np.prod(np.maximum(ranges, 1e-12)))

        # Convex hull volume via Delaunay-like simplex decomposition
        # Simplified: use the determinant-based formula for small dimensions
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(X_red)
            hull_vol = float(hull.volume)
        except Exception:
            hull_vol = 0.0

        ratio = hull_vol / max(bbox_vol, 1e-12)

        return {
            "volume": hull_vol,
            "bbox_volume": bbox_vol,
            "ratio": min(ratio, 1.0),
            "n_components": X_red.shape[1],
        }


# =========================================================================
# DimensionalityReducer
# =========================================================================

class DimensionalityReducer:
    """Reduce dimensionality of feature matrices for visualisation or
    further analysis.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None) -> None:
        self.config = config or SimilarityConfig()
        self._rng = np.random.RandomState(self.config.random_state)

    # -----------------------------------------------------------------
    # PCA
    # -----------------------------------------------------------------

    def pca_reduce(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """PCA via eigendecomposition of the covariance matrix.

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, n_components)
        info : dict with eigenvalues, explained variance, etc.
        """
        k = n_components or self.config.n_components
        n, d = X.shape
        k = min(k, n, d)

        mean = X.mean(axis=0)
        X_centered = X - mean

        # Covariance matrix
        cov = (X_centered.T @ X_centered) / max(n - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx][:k]
        eigenvectors = eigenvectors[:, idx][:, :k]

        X_reduced = X_centered @ eigenvectors

        total_var = float(np.sum(np.abs(eigenvalues)))
        explained = eigenvalues / max(total_var, 1e-12) if total_var > 0 else eigenvalues

        info = {
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained,
            "cumulative_variance": np.cumsum(explained),
            "n_components": k,
            "mean": mean,
            "components": eigenvectors,
        }
        return X_reduced, info

    # -----------------------------------------------------------------
    # Random projection
    # -----------------------------------------------------------------

    def random_projection(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian random projection (Johnson-Lindenstrauss style).

        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_components)
        projection_matrix : ndarray of shape (n_features, n_components)
        """
        k = n_components or self.config.n_components
        d = X.shape[1]
        k = min(k, d)

        # Random Gaussian matrix, scaled by 1/sqrt(k)
        R = self._rng.randn(d, k) / math.sqrt(k)
        X_projected = X @ R

        return X_projected, R

    # -----------------------------------------------------------------
    # UMAP approximation
    # -----------------------------------------------------------------

    def umap_approximation(
        self,
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        n_iter: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simplified UMAP-style layout using neighbour-graph construction
        and force-directed optimisation.

        This is an approximation of the UMAP algorithm:
        1. Build a k-nearest-neighbour graph.
        2. Compute fuzzy set membership strengths.
        3. Optimise a low-dimensional layout via stochastic gradient descent
           with attractive/repulsive forces.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
        info : dict with graph info
        """
        _n_neighbors = n_neighbors or self.config.umap_n_neighbors
        _min_dist = min_dist if min_dist is not None else self.config.umap_min_dist
        _n_iter = n_iter or self.config.umap_n_iter
        n = X.shape[0]
        _n_neighbors = min(_n_neighbors, n - 1)

        if n <= n_components:
            return X[:, :n_components].copy() if X.shape[1] >= n_components else \
                np.hstack([X, np.zeros((n, n_components - X.shape[1]))]), {}

        # 1. Build distance matrix & k-NN graph
        D = euclidean_distance_matrix(X)
        # For each point, find k nearest neighbours
        knn_indices = np.zeros((n, _n_neighbors), dtype=np.int64)
        knn_dists = np.zeros((n, _n_neighbors), dtype=np.float64)
        for i in range(n):
            d_i = D[i].copy()
            d_i[i] = np.inf
            nn_idx = np.argpartition(d_i, _n_neighbors)[:_n_neighbors]
            nn_idx = nn_idx[np.argsort(d_i[nn_idx])]
            knn_indices[i] = nn_idx
            knn_dists[i] = d_i[nn_idx]

        # 2. Compute fuzzy set membership strengths
        # sigma_i chosen so that sum of exp(-d/sigma) = log2(k)
        sigmas = np.ones(n, dtype=np.float64)
        target = math.log2(max(_n_neighbors, 2))
        for i in range(n):
            lo, hi = 1e-5, 1000.0
            for _ in range(64):
                mid = (lo + hi) / 2.0
                val = np.sum(np.exp(-knn_dists[i] / max(mid, 1e-12)))
                if val > target:
                    hi = mid
                else:
                    lo = mid
            sigmas[i] = (lo + hi) / 2.0

        # Build symmetrised weight matrix (fuzzy union)
        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for ki in range(_n_neighbors):
                j = knn_indices[i, ki]
                w = math.exp(-knn_dists[i, ki] / max(sigmas[i], 1e-12))
                W[i, j] = max(W[i, j], w)
        # Symmetrise: W_sym = W + W^T - W * W^T
        W_sym = W + W.T - W * W.T
        np.fill_diagonal(W_sym, 0.0)

        # 3. Initialise embedding with PCA
        embedding, _ = self.pca_reduce(X, n_components=n_components)
        # Scale down
        embedding *= 0.01

        # 4. SGD optimisation
        a = 1.0
        b = 1.0
        if _min_dist < 1.0:
            b = 1.0 / max(_min_dist, 1e-3)

        edges_i: List[int] = []
        edges_j: List[int] = []
        edges_w: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                if W_sym[i, j] > 0.01:
                    edges_i.append(i)
                    edges_j.append(j)
                    edges_w.append(W_sym[i, j])

        n_edges = len(edges_i)
        if n_edges == 0:
            return embedding, {"n_edges": 0}

        edges_i_arr = np.array(edges_i)
        edges_j_arr = np.array(edges_j)
        edges_w_arr = np.array(edges_w)

        lr = 1.0
        for epoch in range(_n_iter):
            alpha = 1.0 - epoch / _n_iter
            lr_t = lr * alpha
            if lr_t < 1e-6:
                break

            # Attractive forces along edges
            for ei in range(n_edges):
                ii = edges_i_arr[ei]
                jj = edges_j_arr[ei]
                diff = embedding[ii] - embedding[jj]
                dist_sq = float(np.sum(diff ** 2))
                grad_coeff = -2.0 * a * b * edges_w_arr[ei]
                grad_coeff /= (1.0 + a * dist_sq ** b) * max(dist_sq, 1e-4)
                grad = grad_coeff * diff
                embedding[ii] += lr_t * grad
                embedding[jj] -= lr_t * grad

            # Repulsive forces: sample negative pairs
            n_neg = min(5 * n_edges, n * n)
            neg_i = self._rng.randint(0, n, size=min(n_neg, n * 3))
            neg_j = self._rng.randint(0, n, size=min(n_neg, n * 3))
            for ni in range(len(neg_i)):
                ii = neg_i[ni]
                jj = neg_j[ni]
                if ii == jj:
                    continue
                diff = embedding[ii] - embedding[jj]
                dist_sq = float(np.sum(diff ** 2))
                if dist_sq < 1e-4:
                    continue
                grad_coeff = 2.0 * b / ((1.0 + a * dist_sq ** b) * max(dist_sq, 1e-4))
                grad_coeff *= (1.0 - W_sym[ii, jj])
                grad = grad_coeff * diff
                embedding[ii] += lr_t * grad * 0.1
                embedding[jj] -= lr_t * grad * 0.1

        info = {
            "n_edges": n_edges,
            "n_iter": _n_iter,
            "n_neighbors": _n_neighbors,
        }
        return embedding, info

    # -----------------------------------------------------------------
    # Intrinsic dimensionality
    # -----------------------------------------------------------------

    def compute_intrinsic_dimensionality(
        self,
        X: np.ndarray,
        method: str = "mle",
        k: int = 10,
    ) -> Dict[str, float]:
        """Estimate the intrinsic dimensionality of the data.

        Supports:
        - 'mle': Maximum likelihood estimator (Levina-Bickel).
        - 'correlation': Correlation dimension estimate.
        - 'pca': Based on eigenvalue decay.

        Returns a dict with the estimate(s).
        """
        n, d = X.shape
        k = min(k, n - 1)

        results: Dict[str, float] = {}

        if method in ("mle", "all"):
            # Levina-Bickel MLE estimator
            D = euclidean_distance_matrix(X)
            np.fill_diagonal(D, np.inf)
            dims: List[float] = []
            for i in range(n):
                sorted_dists = np.sort(D[i])[:k]
                sorted_dists = sorted_dists[sorted_dists > 1e-12]
                if len(sorted_dists) < 2:
                    continue
                Tk = sorted_dists[-1]
                if Tk < 1e-12:
                    continue
                log_ratios = np.log(Tk / sorted_dists[:-1])
                m_hat = float(len(log_ratios)) / max(float(np.sum(log_ratios)), 1e-12)
                dims.append(m_hat)
            results["mle"] = float(np.median(dims)) if dims else float(d)

        if method in ("correlation", "all"):
            # Correlation dimension: slope of log C(r) vs log r
            D = euclidean_distance_matrix(X)
            upper = D[np.triu_indices(n, k=1)]
            if len(upper) == 0:
                results["correlation"] = float(d)
            else:
                radii = np.percentile(upper, np.linspace(10, 90, 20))
                radii = np.unique(radii[radii > 1e-12])
                if len(radii) < 3:
                    results["correlation"] = float(d)
                else:
                    n_pairs = len(upper)
                    log_r = np.log(radii)
                    log_c = np.array([
                        math.log(max(np.sum(upper < r) / n_pairs, 1e-12))
                        for r in radii
                    ])
                    # Linear regression
                    coeffs = np.polyfit(log_r, log_c, 1)
                    results["correlation"] = max(float(coeffs[0]), 0.0)

        if method in ("pca", "all"):
            _, info = self.pca_reduce(X, n_components=min(d, n - 1))
            evals = info["eigenvalues"]
            # Find the "knee" — number of components explaining 90% variance
            cumvar = info["cumulative_variance"]
            dims_90 = int(np.searchsorted(cumvar, 0.9)) + 1
            results["pca"] = float(dims_90)
            # Also compute participation ratio: (sum λ)^2 / sum λ^2
            total = float(np.sum(evals))
            sum_sq = float(np.sum(evals ** 2))
            results["pca_participation_ratio"] = total ** 2 / max(sum_sq, 1e-12)

        return results


# =========================================================================
# Convenience / pipeline functions
# =========================================================================

def compute_full_analysis(
    documents: Sequence[str],
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, Any]:
    """Run a full semantic similarity analysis pipeline.

    Returns a comprehensive dictionary with similarity matrices,
    clustering, coverage, and diversity statistics.
    """
    config = config or SimilarityConfig()
    logger.info("Starting full semantic similarity analysis on %d documents.", len(documents))

    computer = SemanticSimilarityComputer(config)
    psm = PairwiseSimilarityMatrix(documents=documents, config=config)
    clusterer = SemanticClustering(config)
    coverage_analyzer = CoverageAnalyzer(config)
    reducer = DimensionalityReducer(config)

    # --- Similarity ---
    tfidf_result = computer.compute_tfidf_similarity(documents)
    psm._matrix = tfidf_result.matrix

    sim_dist = psm.compute_similarity_distribution()
    diversity = psm.compute_diversity_from_similarity()
    most_sim = psm.get_most_similar_pairs(k=5)
    least_sim = psm.get_least_similar_pairs(k=5)
    eff_div = psm.effective_diversity()
    spectral = psm.spectral_analysis(n_eigenvalues=min(10, len(documents)))

    # --- Feature vectors for clustering ---
    tfidf_vecs, _, _ = computer.compute_tfidf_vectors(documents)

    # --- Clustering ---
    n = len(documents)
    cluster_labels = np.zeros(n, dtype=np.int64)
    cluster_quality: Dict[str, float] = {}
    cluster_div: Dict[str, Any] = {}
    if n >= 3:
        k = min(config.n_clusters, n - 1)
        cluster_labels, _ = clusterer.kmeans_clustering(tfidf_vecs, n_clusters=k)
        cluster_quality = clusterer.cluster_quality_metrics(tfidf_vecs, cluster_labels)
        cluster_div = clusterer.compute_cluster_diversity(tfidf_vecs, cluster_labels)

    # --- Coverage ---
    sem_cov = coverage_analyzer.compute_semantic_coverage(tfidf_result.matrix)
    novelty = coverage_analyzer.compute_novelty_scores(tfidf_result.matrix)
    redundancy = coverage_analyzer.compute_redundancy_scores(tfidf_result.matrix)

    # --- Dimensionality ---
    if n >= 3 and tfidf_vecs.shape[1] >= 2:
        reduced, pca_info = reducer.pca_reduce(tfidf_vecs, n_components=min(3, tfidf_vecs.shape[1]))
        uniformity = coverage_analyzer.compute_coverage_uniformity(reduced)
        vol_coverage = coverage_analyzer.compute_volume_coverage(tfidf_vecs, n_components=min(3, tfidf_vecs.shape[1]))
    else:
        uniformity = {"uniformity": 1.0}
        vol_coverage = {"volume": 0.0, "ratio": 0.0}

    return {
        "similarity": {
            "method": "tfidf",
            "distribution": sim_dist,
            "diversity": diversity,
            "effective_diversity": eff_div,
            "most_similar_pairs": most_sim,
            "least_similar_pairs": least_sim,
            "spectral": {
                "entropy": spectral.get("spectral_entropy", 0.0),
                "gap": spectral.get("spectral_gap", 0.0),
                "rank": spectral.get("rank", 0),
            },
        },
        "clustering": {
            "labels": cluster_labels.tolist(),
            "quality": cluster_quality,
            "diversity": cluster_div,
        },
        "coverage": {
            "semantic": sem_cov,
            "novelty_scores": novelty.tolist(),
            "redundancy": {
                "ratio": redundancy["redundancy_ratio"],
                "n_redundant_pairs": len(redundancy["redundant_pairs"]),
            },
            "uniformity": uniformity,
            "volume": vol_coverage,
        },
        "n_documents": n,
    }


def compute_pairwise_similarity(
    documents: Sequence[str],
    method: str = "tfidf",
    config: Optional[SimilarityConfig] = None,
) -> SimilarityResult:
    """Convenience function to compute a similarity matrix."""
    config = config or SimilarityConfig(method=method)
    computer = SemanticSimilarityComputer(config)

    dispatch: Dict[str, Callable[..., SimilarityResult]] = {
        "tfidf": computer.compute_tfidf_similarity,
        "bm25": computer.compute_bm25_similarity,
        "jaccard": computer.compute_jaccard_similarity_matrix,
        "overlap": computer.compute_overlap_coefficient_matrix,
        "dice": computer.compute_dice_similarity_matrix,
        "lsi": computer.compute_lsi_similarity,
        "wmd_approx": computer.compute_word_mover_approximation,
        "ngram": computer.compute_ngram_similarity_matrix,
        "combined": computer.compute_combined_similarity,
    }

    if method not in dispatch:
        raise ValueError(f"Unknown method: {method}. Available: {list(dispatch)}")

    return dispatch[method](documents)


def select_diverse_subset(
    documents: Sequence[str],
    subset_size: int,
    method: str = "tfidf",
    config: Optional[SimilarityConfig] = None,
) -> Tuple[List[int], List[str], float]:
    """Select a maximally diverse subset of documents.

    Returns (selected_indices, selected_documents, coverage_score).
    """
    config = config or SimilarityConfig(method=method)
    result = compute_pairwise_similarity(documents, method=method, config=config)
    analyzer = CoverageAnalyzer(config)
    indices, coverage = analyzer.greedy_max_coverage_subset(
        result.matrix, subset_size,
    )
    selected_docs = [documents[i] for i in indices]
    return indices, selected_docs, coverage


def compute_diversity_spectrum(
    documents: Sequence[str],
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, float]:
    """Compute diversity scores using multiple methods and return a
    summary "spectrum" of diversity values.
    """
    config = config or SimilarityConfig()
    computer = SemanticSimilarityComputer(config)

    methods = ["tfidf", "bm25", "jaccard", "dice", "ngram"]
    scores: Dict[str, float] = {}

    for method in methods:
        try:
            result = compute_pairwise_similarity(documents, method=method, config=config)
            scores[f"{method}_diversity"] = result.diversity_score
            scores[f"{method}_mean_sim"] = result.mean_similarity
        except Exception as e:
            logger.warning("Method %s failed: %s", method, e)
            scores[f"{method}_diversity"] = 0.0
            scores[f"{method}_mean_sim"] = 1.0

    # Aggregate
    div_values = [v for k, v in scores.items() if k.endswith("_diversity")]
    scores["mean_diversity"] = float(np.mean(div_values)) if div_values else 0.0
    scores["min_diversity"] = float(np.min(div_values)) if div_values else 0.0
    scores["max_diversity"] = float(np.max(div_values)) if div_values else 0.0

    return scores


def compare_document_sets(
    set_a: Sequence[str],
    set_b: Sequence[str],
    method: str = "tfidf",
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, Any]:
    """Compare two sets of documents in terms of diversity and coverage.

    Returns cross-set similarity statistics and per-set diversity.
    """
    config = config or SimilarityConfig(method=method)
    computer = SemanticSimilarityComputer(config)

    result_a = compute_pairwise_similarity(set_a, method=method, config=config)
    result_b = compute_pairwise_similarity(set_b, method=method, config=config)

    # Cross-set similarity
    all_docs = list(set_a) + list(set_b)
    result_all = compute_pairwise_similarity(all_docs, method=method, config=config)
    n_a = len(set_a)
    n_b = len(set_b)
    cross_block = result_all.matrix[:n_a, n_a:]

    return {
        "set_a": {
            "n_documents": n_a,
            "diversity": result_a.diversity_score,
            "mean_similarity": result_a.mean_similarity,
        },
        "set_b": {
            "n_documents": n_b,
            "diversity": result_b.diversity_score,
            "mean_similarity": result_b.mean_similarity,
        },
        "cross_set": {
            "mean_similarity": float(np.mean(cross_block)),
            "max_similarity": float(np.max(cross_block)),
            "min_similarity": float(np.min(cross_block)),
        },
        "combined_diversity": result_all.diversity_score,
        "method": method,
    }


def batch_similarity_analysis(
    document_groups: Dict[str, Sequence[str]],
    method: str = "tfidf",
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, Any]:
    """Run similarity analysis on multiple named groups of documents.

    Returns per-group diversity and cross-group comparison.
    """
    config = config or SimilarityConfig(method=method)
    per_group: Dict[str, Dict[str, Any]] = {}
    all_documents: List[str] = []
    group_labels: List[str] = []

    for name, docs in document_groups.items():
        result = compute_pairwise_similarity(docs, method=method, config=config)
        per_group[name] = {
            "n_documents": len(docs),
            "diversity": result.diversity_score,
            "mean_similarity": result.mean_similarity,
            "statistics": result.statistics,
        }
        all_documents.extend(docs)
        group_labels.extend([name] * len(docs))

    # Cross-group analysis
    cross_group: Dict[str, Dict[str, float]] = {}
    group_names = list(document_groups.keys())
    if len(group_names) >= 2:
        all_result = compute_pairwise_similarity(
            all_documents, method=method, config=config,
        )
        offsets: Dict[str, Tuple[int, int]] = {}
        idx = 0
        for name, docs in document_groups.items():
            offsets[name] = (idx, idx + len(docs))
            idx += len(docs)

        for i, name_a in enumerate(group_names):
            for j, name_b in enumerate(group_names):
                if i >= j:
                    continue
                sa, ea = offsets[name_a]
                sb, eb = offsets[name_b]
                block = all_result.matrix[sa:ea, sb:eb]
                key = f"{name_a}_vs_{name_b}"
                cross_group[key] = {
                    "mean_similarity": float(np.mean(block)),
                    "std_similarity": float(np.std(block)),
                }

    return {
        "per_group": per_group,
        "cross_group": cross_group,
        "method": method,
    }
