"""
Search Diversity — Diverse search and retrieval.

Implements diversity-aware search algorithms including MMR-based diverse
search, faceted search, exploratory multi-round search, query expansion,
and result clustering. All operate on generic text items with TF-IDF
similarity — no external vector-DB dependencies required.
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "was", "are", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "so", "if",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [t for t in _WS_RE.split(text) if t and t not in _STOPWORDS and len(t) > 1]


def _tfidf_vectors(texts: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build TF-IDF matrix and vocabulary mapping."""
    docs = [_tokenize(t) for t in texts]
    vocab: Dict[str, int] = {}
    for doc in docs:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    n_docs, n_vocab = len(docs), len(vocab)
    if n_vocab == 0:
        return np.zeros((n_docs, 1)), {}
    tf = np.zeros((n_docs, n_vocab))
    for i, doc in enumerate(docs):
        for tok in doc:
            tf[i, vocab[tok]] += 1
    row_sums = tf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tf /= row_sums
    df = (tf > 0).sum(axis=0).astype(float)
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    return tf * idf, vocab


def _cosine_sim_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = mat / norms
    return normed @ normed.T


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Index abstraction
# ---------------------------------------------------------------------------


class TextIndex:
    """
    Simple in-memory text index using TF-IDF.

    Supports cosine-similarity search, faceted retrieval, and metadata
    filtering. Drop-in replacement for external vector stores in
    prototyping scenarios.
    """

    def __init__(self, documents: Optional[List[str]] = None,
                 metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        self.documents: List[str] = list(documents) if documents else []
        self.metadata: List[Dict[str, Any]] = list(metadata) if metadata else []
        self._tfidf: Optional[np.ndarray] = None
        self._vocab: Dict[str, int] = {}
        if self.documents:
            self._rebuild()

    def _rebuild(self) -> None:
        self._tfidf, self._vocab = _tfidf_vectors(self.documents)

    def add(self, doc: str, meta: Optional[Dict[str, Any]] = None) -> int:
        idx = len(self.documents)
        self.documents.append(doc)
        self.metadata.append(meta or {})
        self._rebuild()
        return idx

    def add_many(self, docs: List[str],
                 metas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        start = len(self.documents)
        self.documents.extend(docs)
        if metas:
            self.metadata.extend(metas)
        else:
            self.metadata.extend([{}] * len(docs))
        self._rebuild()
        return list(range(start, start + len(docs)))

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Return top-k (index, similarity) pairs for *query*."""
        if not self.documents or self._tfidf is None:
            return []
        all_texts = self.documents + [query]
        mat, _ = _tfidf_vectors(all_texts)
        q_vec = mat[-1]
        doc_mat = mat[:-1]
        sims = np.array([_cosine_sim(q_vec, doc_mat[i]) for i in range(len(self.documents))])
        top_k = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top_k if sims[i] > 0]

    def __len__(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """A single search result."""
    index: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Result(idx={self.index}, score={self.score:.3f}, text={self.text[:50]!r})"


@dataclass
class Cluster:
    """A cluster of search results."""
    cluster_id: int
    label: str
    results: List[Result]
    centroid_index: int

    @property
    def size(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return f"Cluster({self.label!r}, n={self.size})"


@dataclass
class ExplorationResult:
    """Result of multi-round exploratory search."""
    rounds: List[List[Result]]
    expanded_queries: List[str]
    all_results: List[Result]
    coverage_score: float

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def n_total(self) -> int:
        return len(self.all_results)


# ---------------------------------------------------------------------------
# Core: diverse_search (MMR-based)
# ---------------------------------------------------------------------------


def diverse_search(
    query: str,
    index: TextIndex,
    k: int = 10,
    lambda_: float = 0.5,
) -> List[Result]:
    """
    MMR-based diverse search: balances relevance to *query* with
    diversity among returned results.

    Parameters
    ----------
    query : str
    index : TextIndex
    k : int
        Number of results to return.
    lambda_ : float
        Trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    if len(index) == 0:
        return []

    # Compute query-doc similarities
    all_texts = index.documents + [query]
    mat, _ = _tfidf_vectors(all_texts)
    q_vec = mat[-1]
    doc_mat = mat[:-1]
    n = len(index.documents)

    query_sims = np.array([_cosine_sim(q_vec, doc_mat[i]) for i in range(n)])
    doc_sim = _cosine_sim_matrix(doc_mat)

    # candidate pool: top 4*k by relevance
    pool_size = min(n, 4 * k)
    pool = np.argsort(-query_sims)[:pool_size].tolist()

    selected: List[int] = []
    remaining = set(pool)

    for _ in range(min(k, len(pool))):
        best_idx = -1
        best_score = -np.inf

        for idx in remaining:
            rel = query_sims[idx]
            if selected:
                max_sim = max(doc_sim[idx, s] for s in selected)
            else:
                max_sim = 0.0
            score = lambda_ * rel - (1 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [
        Result(
            index=idx,
            text=index.documents[idx],
            score=float(query_sims[idx]),
            metadata=index.metadata[idx] if idx < len(index.metadata) else {},
        )
        for idx in selected
    ]


# ---------------------------------------------------------------------------
# Faceted search
# ---------------------------------------------------------------------------


def faceted_search(
    query: str,
    index: TextIndex,
    facets: List[str],
    k_per_facet: int = 3,
) -> Dict[str, List[Result]]:
    """
    Search with diversity across facets. Each facet is a category label
    that must appear in document metadata under key ``"facet"``.

    Falls back to keyword matching against document text if metadata
    facets are not set.

    Parameters
    ----------
    query : str
    index : TextIndex
    facets : list of facet labels
    k_per_facet : int
        Results per facet.
    """
    # Get all results ranked by relevance
    all_results = diverse_search(query, index, k=len(index), lambda_=1.0)

    facet_results: Dict[str, List[Result]] = {f: [] for f in facets}

    for result in all_results:
        meta_facet = result.metadata.get("facet", "")

        for facet in facets:
            if len(facet_results[facet]) >= k_per_facet:
                continue

            # match by metadata or text content
            if meta_facet.lower() == facet.lower():
                facet_results[facet].append(result)
            elif facet.lower() in result.text.lower():
                facet_results[facet].append(result)

    return facet_results


# ---------------------------------------------------------------------------
# Exploratory search (multi-round)
# ---------------------------------------------------------------------------


def exploratory_search(
    query: str,
    index: TextIndex,
    rounds: int = 3,
    k_per_round: int = 5,
    *,
    expand_fn: Optional[Callable[[str, List[Result]], List[str]]] = None,
) -> ExplorationResult:
    """
    Multi-round exploratory search that expands the query based on
    retrieved results, discovering diverse facets of the information space.

    Parameters
    ----------
    query : str
        Initial query.
    index : TextIndex
    rounds : int
        Number of search rounds.
    k_per_round : int
        Results per round.
    expand_fn : callable(query, results) -> List[str], optional
        Custom query expansion function. Defaults to keyword extraction.
    """
    all_results: List[Result] = []
    round_results: List[List[Result]] = []
    queries: List[str] = [query]
    seen_indices: Set[int] = set()

    current_query = query

    for r in range(rounds):
        results = diverse_search(current_query, index, k=k_per_round * 2, lambda_=0.4)

        # filter already-seen
        new_results: List[Result] = []
        for res in results:
            if res.index not in seen_indices:
                seen_indices.add(res.index)
                new_results.append(res)
            if len(new_results) >= k_per_round:
                break

        round_results.append(new_results)
        all_results.extend(new_results)

        if r < rounds - 1 and new_results:
            # expand query for next round
            if expand_fn is not None:
                expanded = expand_fn(current_query, new_results)
            else:
                expanded = _auto_expand_query(current_query, new_results)

            if expanded:
                current_query = expanded[0]
                queries.extend(expanded)

    # coverage: fraction of unique tokens in results vs. index
    result_tokens: Set[str] = set()
    for res in all_results:
        result_tokens.update(_tokenize(res.text))
    index_tokens: Set[str] = set()
    for doc in index.documents:
        index_tokens.update(_tokenize(doc))
    coverage = len(result_tokens & index_tokens) / max(len(index_tokens), 1)

    return ExplorationResult(
        rounds=round_results,
        expanded_queries=queries,
        all_results=all_results,
        coverage_score=coverage,
    )


def _auto_expand_query(query: str, results: List[Result]) -> List[str]:
    """Extract novel terms from results to form expanded queries."""
    query_tokens = set(_tokenize(query))
    result_tokens: Counter = Counter()
    for res in results:
        for tok in _tokenize(res.text):
            if tok not in query_tokens:
                result_tokens[tok] += 1

    if not result_tokens:
        return []

    # pick top novel terms
    top_terms = [term for term, _ in result_tokens.most_common(3)]
    expanded = query + " " + " ".join(top_terms)
    return [expanded]


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------


def query_expansion(
    query: str,
    diversity_target: float = 0.7,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    max_expansions: int = 10,
    seed: int = 42,
) -> List[str]:
    """
    Expand a query into multiple diverse reformulations.

    Parameters
    ----------
    query : str
        Original query.
    diversity_target : float
        Target pairwise diversity among expanded queries (0-1).
    gen_fn : callable(prompt) -> str, optional
        LLM generation function for semantic expansion.
    max_expansions : int
        Maximum number of expanded queries.
    """
    import random as _random
    rng = _random.Random(seed)

    tokens = _tokenize(query)
    expansions: List[str] = [query]

    # Strategy 1: synonym-like reordering
    if len(tokens) >= 2:
        for _ in range(min(3, max_expansions)):
            shuffled = list(tokens)
            rng.shuffle(shuffled)
            expansions.append(" ".join(shuffled))

    # Strategy 2: sub-queries (dropping terms)
    if len(tokens) >= 3:
        for i in range(len(tokens)):
            sub = [t for j, t in enumerate(tokens) if j != i]
            expansions.append(" ".join(sub))

    # Strategy 3: bigram expansion
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            expansions.append(f"{tokens[i]} {tokens[i + 1]}")

    # Strategy 4: LLM-based expansion
    if gen_fn is not None:
        strategies = [
            f"Rephrase this query differently: {query}",
            f"What is another way to ask: {query}",
            f"Broaden this query: {query}",
            f"Make this query more specific: {query}",
        ]
        for prompt in strategies[:max(1, max_expansions - len(expansions))]:
            expansions.append(gen_fn(prompt))

    # Deduplicate
    seen: Set[str] = set()
    unique: List[str] = []
    for exp in expansions:
        key = exp.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(exp.strip())

    # Greedily select for diversity
    if len(unique) <= max_expansions:
        return unique

    selected = [unique[0]]
    remaining = unique[1:]

    while len(selected) < max_expansions and remaining:
        best_idx = -1
        best_div = -1.0
        for i, candidate in enumerate(remaining):
            trial = selected + [candidate]
            div = _avg_pairwise_distance(trial)
            if div > best_div:
                best_div = div
                best_idx = i
        if best_idx < 0:
            break
        selected.append(remaining.pop(best_idx))

    return selected


def _avg_pairwise_distance(texts: List[str]) -> float:
    if len(texts) < 2:
        return 0.0
    mat, _ = _tfidf_vectors(texts)
    sim = _cosine_sim_matrix(mat)
    n = len(texts)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - sim[i, j]
            count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Result clustering
# ---------------------------------------------------------------------------


def result_clustering(
    results: List[Result],
    n_clusters: Union[int, str] = "auto",
    *,
    max_iter: int = 50,
    seed: int = 42,
) -> List[Cluster]:
    """
    Cluster search results into topically coherent groups.

    Parameters
    ----------
    results : list of Result
    n_clusters : int or "auto"
        Number of clusters. "auto" uses a heuristic based on result count.
    max_iter : int
        Max k-means iterations.
    """
    if not results:
        return []

    texts = [r.text for r in results]
    n = len(texts)

    # determine number of clusters
    if n_clusters == "auto":
        k = max(2, min(int(math.sqrt(n)), 10))
    else:
        k = int(n_clusters)  # type: ignore[arg-type]
    k = min(k, n)

    mat, vocab = _tfidf_vectors(texts)
    rng = np.random.default_rng(seed)

    # k-means clustering
    # initialize centroids with k-means++
    centroids = np.zeros((k, mat.shape[1]))
    idx0 = rng.integers(n)
    centroids[0] = mat[idx0]

    for c in range(1, k):
        # distance to nearest centroid
        dists = np.full(n, np.inf)
        for prev in range(c):
            d = np.linalg.norm(mat - centroids[prev], axis=1)
            dists = np.minimum(dists, d)
        dists_sq = dists ** 2
        probs = dists_sq / (dists_sq.sum() + 1e-12)
        centroids[c] = mat[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # assign
        new_labels = np.zeros(n, dtype=int)
        for i in range(n):
            dists = [np.linalg.norm(mat[i] - centroids[c]) for c in range(k)]
            new_labels[i] = int(np.argmin(dists))

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # update centroids
        for c in range(k):
            members = mat[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    # build clusters
    clusters: List[Cluster] = []
    inv_vocab = {v: k_ for k_, v in vocab.items()} if vocab else {}

    for c in range(k):
        member_indices = [i for i in range(n) if labels[i] == c]
        if not member_indices:
            continue

        cluster_results = [results[i] for i in member_indices]

        # label from top TF-IDF terms of centroid
        top_dims = np.argsort(-centroids[c])[:3]
        label_terms = [inv_vocab.get(int(d), f"dim{d}") for d in top_dims if int(d) in inv_vocab]
        label = " / ".join(label_terms) if label_terms else f"cluster_{c}"

        # centroid: member closest to centroid vector
        dists_to_centroid = [np.linalg.norm(mat[i] - centroids[c]) for i in member_indices]
        centroid_member = member_indices[int(np.argmin(dists_to_centroid))]

        clusters.append(Cluster(
            cluster_id=c,
            label=label,
            results=cluster_results,
            centroid_index=centroid_member,
        ))

    clusters.sort(key=lambda c: -c.size)
    return clusters


# ---------------------------------------------------------------------------
# Diversity-aware re-ranking
# ---------------------------------------------------------------------------


def diversity_rerank(
    results: List[Result],
    lambda_: float = 0.5,
    k: Optional[int] = None,
) -> List[Result]:
    """
    Re-rank results to maximize diversity while preserving relevance.

    Uses a greedy MMR-like approach over the already-retrieved result set.
    """
    if not results:
        return []

    n = len(results)
    k = k or n
    k = min(k, n)

    texts = [r.text for r in results]
    mat, _ = _tfidf_vectors(texts)
    sim = _cosine_sim_matrix(mat)
    relevance = np.array([r.score for r in results])

    selected: List[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            rel = relevance[idx]
            if selected:
                max_sim = max(sim[idx, s] for s in selected)
            else:
                max_sim = 0.0
            score = lambda_ * rel - (1 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [results[i] for i in selected]
