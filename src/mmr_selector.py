"""
Maximal Marginal Relevance (MMR) for diverse retrieval.

Implements standard MMR, batch MMR, adaptive MMR, constrained MMR,
diversity-aware reranking, fairness-aware MMR, and multiple similarity functions.
"""

import numpy as np
from typing import Optional, List, Callable, Dict, Tuple, Any, Union


# ======================================================================
# Similarity functions
# ======================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of X and rows of Y."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    norms_X = np.linalg.norm(X, axis=1, keepdims=True)
    norms_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    norms_X = np.maximum(norms_X, 1e-12)
    norms_Y = np.maximum(norms_Y, 1e-12)
    return (X / norms_X) @ (Y / norms_Y).T


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity for binary or set-like vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    if union < 1e-12:
        return 0.0
    return float(intersection / union)


def edit_distance_similarity(a: str, b: str) -> float:
    """Normalized edit distance similarity between two strings.

    Returns 1 - (edit_distance / max_length).
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    # Standard DP for Levenshtein distance
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,
                           dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + cost)
    dist = dp[n, m]
    return 1.0 - dist / max(n, m)


def tfidf_vectors(texts: List[str]) -> np.ndarray:
    """Compute TF-IDF vectors from a list of texts. Numpy-only.

    Returns:
        (n_texts, vocab_size) TF-IDF matrix.
    """
    # Tokenize
    docs = []
    vocab = {}
    for text in texts:
        tokens = text.lower().split()
        doc_tokens = []
        for t in tokens:
            # Strip punctuation
            t = ''.join(c for c in t if c.isalnum())
            if t:
                if t not in vocab:
                    vocab[t] = len(vocab)
                doc_tokens.append(vocab[t])
        docs.append(doc_tokens)

    n = len(texts)
    v = len(vocab)
    if v == 0:
        return np.zeros((n, 1), dtype=np.float64)

    # Term frequency
    tf = np.zeros((n, v), dtype=np.float64)
    for i, doc in enumerate(docs):
        for token_id in doc:
            tf[i, token_id] += 1.0
        if len(doc) > 0:
            tf[i] /= len(doc)

    # Inverse document frequency
    df = np.sum(tf > 0, axis=0).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    tfidf = tf * idf
    return tfidf


def get_similarity_function(name: str) -> Callable:
    """Get a similarity function by name."""
    funcs = {
        'cosine': cosine_similarity,
        'jaccard': jaccard_similarity,
    }
    if name not in funcs:
        raise ValueError(f"Unknown similarity: {name}. Choose from {list(funcs.keys())}")
    return funcs[name]


# ======================================================================
# MMR Selector
# ======================================================================

class MMRSelector:
    """Maximal Marginal Relevance for diverse retrieval.

    MMR balances relevance to a query with diversity among selected items:
        MMR = arg max_{d_i ∈ R\\S} [λ sim(d_i, q) - (1-λ) max_{d_j ∈ S} sim(d_i, d_j)]
    """

    def __init__(self, similarity: str = 'cosine'):
        """Initialize MMR selector.

        Args:
            similarity: Similarity function name ('cosine', 'jaccard').
        """
        self.similarity = similarity

    def select(self, items: np.ndarray, query: np.ndarray, k: int,
               lambda_param: float = 0.5,
               relevance_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Standard MMR selection.

        Args:
            items: (n, d) item feature vectors.
            query: (d,) query feature vector.
            k: Number of items to select.
            lambda_param: Trade-off parameter [0, 1]. Higher = more relevant.
            relevance_scores: Optional precomputed relevance scores (n,).

        Returns:
            Array of k selected indices (in selection order).
        """
        items = np.asarray(items, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64).ravel()
        n = items.shape[0]
        k = min(k, n)

        # Precompute relevance scores
        if relevance_scores is not None:
            rel = np.asarray(relevance_scores, dtype=np.float64)
        else:
            if self.similarity == 'cosine':
                rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
            else:
                sim_fn = get_similarity_function(self.similarity)
                rel = np.array([sim_fn(items[i], query) for i in range(n)])

        # Precompute pairwise similarity matrix
        if self.similarity == 'cosine':
            sim_matrix = cosine_similarity_matrix(items, items)
        else:
            sim_fn = get_similarity_function(self.similarity)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    s = sim_fn(items[i], items[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s

        selected = []
        remaining = set(range(n))
        max_sim_to_selected = np.full(n, -np.inf)

        for _ in range(k):
            best_idx = -1
            best_score = -np.inf

            for i in remaining:
                if len(selected) == 0:
                    div_penalty = 0.0
                else:
                    div_penalty = max_sim_to_selected[i]
                    if div_penalty == -np.inf:
                        div_penalty = 0.0

                mmr_score = lambda_param * rel[i] - (1 - lambda_param) * div_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)

            # Update max similarities
            for i in remaining:
                max_sim_to_selected[i] = max(max_sim_to_selected[i],
                                             sim_matrix[i, best_idx])

        return np.array(selected, dtype=np.intp)

    def select_fast(self, items: np.ndarray, query: np.ndarray, k: int,
                    lambda_param: float = 0.5) -> np.ndarray:
        """Vectorized MMR selection (faster for large n with cosine similarity).

        Args:
            items: (n, d) item vectors.
            query: (d,) query vector.
            k: Number of items.
            lambda_param: Trade-off.

        Returns:
            Array of k selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64).ravel()
        n = items.shape[0]
        k = min(k, n)

        # Relevance
        rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
        # Pairwise sim
        sim = cosine_similarity_matrix(items, items)

        selected = []
        mask = np.ones(n, dtype=bool)
        max_sim = np.full(n, -np.inf)

        for _ in range(k):
            if len(selected) == 0:
                div_penalty = np.zeros(n)
            else:
                div_penalty = np.where(max_sim > -np.inf, max_sim, 0.0)

            scores = lambda_param * rel - (1 - lambda_param) * div_penalty
            scores[~mask] = -np.inf

            best = np.argmax(scores)
            selected.append(best)
            mask[best] = False

            # Update max similarities
            np.maximum(max_sim, sim[:, best], out=max_sim)

        return np.array(selected, dtype=np.intp)

    # ------------------------------------------------------------------
    # Batch MMR
    # ------------------------------------------------------------------

    def select_batch(self, items: np.ndarray, query: np.ndarray,
                     k: int, batch_size: int = 5,
                     lambda_param: float = 0.5) -> np.ndarray:
        """Batch MMR: select multiple items per iteration.

        Selects top-batch_size items by MMR score at each step,
        then updates diversity penalties. Faster but slightly
        less diverse than standard MMR.

        Args:
            items: (n, d) item vectors.
            query: (d,) query vector.
            k: Total items to select.
            batch_size: Items per iteration.
            lambda_param: Trade-off.

        Returns:
            Array of k selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64).ravel()
        n = items.shape[0]
        k = min(k, n)

        rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
        sim = cosine_similarity_matrix(items, items)

        selected = []
        mask = np.ones(n, dtype=bool)
        max_sim = np.full(n, -np.inf)

        while len(selected) < k:
            div_penalty = np.where(max_sim > -np.inf, max_sim, 0.0)
            scores = lambda_param * rel - (1 - lambda_param) * div_penalty
            scores[~mask] = -np.inf

            this_batch = min(batch_size, k - len(selected))
            top_indices = np.argsort(-scores)[:this_batch]

            for idx in top_indices:
                if mask[idx]:
                    selected.append(idx)
                    mask[idx] = False
                    np.maximum(max_sim, sim[:, idx], out=max_sim)

        return np.array(selected[:k], dtype=np.intp)

    # ------------------------------------------------------------------
    # Adaptive MMR
    # ------------------------------------------------------------------

    def select_adaptive(self, items: np.ndarray, query: np.ndarray,
                        k: int, n_trials: int = 10,
                        metric: str = 'combined') -> Tuple[np.ndarray, float]:
        """Adaptive MMR: automatically tune λ.

        Tries multiple λ values and picks the one optimizing a combined
        relevance-diversity metric.

        Args:
            items: (n, d) item vectors.
            query: (d,) query vector.
            k: Number of items.
            n_trials: Number of λ values to try.
            metric: 'combined', 'diversity', or 'relevance'.

        Returns:
            (selected_indices, best_lambda)
        """
        items = np.asarray(items, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64).ravel()

        rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
        sim = cosine_similarity_matrix(items, items)

        lambdas = np.linspace(0.0, 1.0, n_trials)
        best_score = -np.inf
        best_selected = None
        best_lam = 0.5

        for lam in lambdas:
            sel = self.select_fast(items, query, k, lambda_param=lam)

            # Compute quality metrics
            avg_rel = np.mean(rel[sel])
            if len(sel) > 1:
                sel_sim = sim[np.ix_(sel, sel)]
                # Mean off-diagonal similarity (lower = more diverse)
                mask_offdiag = ~np.eye(len(sel), dtype=bool)
                avg_div = 1.0 - np.mean(sel_sim[mask_offdiag])
            else:
                avg_div = 1.0

            if metric == 'combined':
                score = avg_rel * avg_div
            elif metric == 'diversity':
                score = avg_div
            elif metric == 'relevance':
                score = avg_rel
            else:
                score = avg_rel * avg_div

            if score > best_score:
                best_score = score
                best_selected = sel
                best_lam = lam

        return best_selected, float(best_lam)

    # ------------------------------------------------------------------
    # Constrained MMR
    # ------------------------------------------------------------------

    def select_constrained(self, items: np.ndarray, query: np.ndarray,
                           k: int, lambda_param: float = 0.5,
                           constraints: Optional[Callable[[int, List[int]], bool]] = None
                           ) -> np.ndarray:
        """MMR with hard constraints.

        Args:
            items: (n, d) item vectors.
            query: (d,) query vector.
            k: Number of items.
            lambda_param: Trade-off.
            constraints: Function(candidate_idx, selected_so_far) -> bool.
                Returns True if candidate can be added.

        Returns:
            Array of selected indices (may be < k if constraints too tight).
        """
        items = np.asarray(items, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64).ravel()
        n = items.shape[0]
        k = min(k, n)

        rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
        sim = cosine_similarity_matrix(items, items)

        selected = []
        mask = np.ones(n, dtype=bool)
        max_sim = np.full(n, -np.inf)

        for _ in range(k):
            div_penalty = np.where(max_sim > -np.inf, max_sim, 0.0)
            scores = lambda_param * rel - (1 - lambda_param) * div_penalty
            scores[~mask] = -np.inf

            # Sort candidates by score
            order = np.argsort(-scores)
            found = False
            for idx in order:
                if not mask[idx]:
                    continue
                if scores[idx] == -np.inf:
                    break
                if constraints is None or constraints(int(idx), selected):
                    selected.append(int(idx))
                    mask[idx] = False
                    np.maximum(max_sim, sim[:, idx], out=max_sim)
                    found = True
                    break
            if not found:
                break

        return np.array(selected, dtype=np.intp)

    # ------------------------------------------------------------------
    # Diversity-aware reranking
    # ------------------------------------------------------------------

    @staticmethod
    def diversity_rerank(items: np.ndarray, scores: np.ndarray,
                         k: int, lambda_param: float = 0.5) -> np.ndarray:
        """Rerank a scored list to improve diversity.

        Takes a ranked list (by scores) and reranks to balance
        original score with diversity.

        Args:
            items: (n, d) item vectors.
            scores: (n,) original relevance scores.
            k: Number to select.
            lambda_param: Original score vs diversity.

        Returns:
            Reranked indices.
        """
        items = np.asarray(items, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        n = items.shape[0]
        k = min(k, n)

        # Normalize scores to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-12:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones(n) * 0.5

        sim = cosine_similarity_matrix(items, items)

        selected = []
        mask = np.ones(n, dtype=bool)
        max_sim = np.full(n, -np.inf)

        for _ in range(k):
            div_penalty = np.where(max_sim > -np.inf, max_sim, 0.0)
            rerank_score = lambda_param * norm_scores - (1 - lambda_param) * div_penalty
            rerank_score[~mask] = -np.inf

            best = np.argmax(rerank_score)
            selected.append(best)
            mask[best] = False
            np.maximum(max_sim, sim[:, best], out=max_sim)

        return np.array(selected, dtype=np.intp)

    # ------------------------------------------------------------------
    # Fairness-aware MMR
    # ------------------------------------------------------------------

    @staticmethod
    def fairness_mmr(items: np.ndarray, query: np.ndarray,
                     groups: np.ndarray, k: int,
                     lambda_param: float = 0.5,
                     min_per_group: Optional[Dict[int, int]] = None) -> np.ndarray:
        """Fairness-aware MMR ensuring representation across groups.

        First satisfies minimum group constraints, then fills remaining
        slots with standard MMR.

        Args:
            items: (n, d) item vectors.
            query: (d,) query vector.
            groups: (n,) integer group labels.
            k: Number to select.
            lambda_param: Relevance-diversity trade-off.
            min_per_group: Dict mapping group_id -> min count.

        Returns:
            Selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64).ravel()
        groups = np.asarray(groups, dtype=np.intp)
        n = items.shape[0]
        k = min(k, n)

        rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
        sim = cosine_similarity_matrix(items, items)

        selected = []
        mask = np.ones(n, dtype=bool)
        max_sim = np.full(n, -np.inf)
        group_counts = {}

        if min_per_group is None:
            min_per_group = {}

        # Phase 1: satisfy minimum group constraints
        for group_id, min_count in min_per_group.items():
            group_mask = (groups == group_id) & mask
            group_indices = np.where(group_mask)[0]
            if len(group_indices) == 0:
                continue

            needed = min_count - group_counts.get(group_id, 0)
            for _ in range(needed):
                if len(selected) >= k:
                    break
                group_mask = (groups == group_id) & mask
                group_indices = np.where(group_mask)[0]
                if len(group_indices) == 0:
                    break

                # Use MMR within group
                div_penalty = np.where(max_sim > -np.inf, max_sim, 0.0)
                scores = lambda_param * rel - (1 - lambda_param) * div_penalty
                group_scores = scores[group_indices]
                best_local = np.argmax(group_scores)
                best_idx = group_indices[best_local]

                selected.append(int(best_idx))
                mask[best_idx] = False
                np.maximum(max_sim, sim[:, best_idx], out=max_sim)
                group_counts[group_id] = group_counts.get(group_id, 0) + 1

        # Phase 2: fill remaining with standard MMR
        while len(selected) < k:
            div_penalty = np.where(max_sim > -np.inf, max_sim, 0.0)
            scores = lambda_param * rel - (1 - lambda_param) * div_penalty
            scores[~mask] = -np.inf

            best = np.argmax(scores)
            if scores[best] == -np.inf:
                break

            selected.append(int(best))
            mask[best] = False
            np.maximum(max_sim, sim[:, best], out=max_sim)

        return np.array(selected, dtype=np.intp)


class MMRWithTFIDF:
    """MMR selector that works directly with text using TF-IDF."""

    def __init__(self):
        self.selector = MMRSelector(similarity='cosine')

    def select(self, texts: List[str], query_text: str, k: int,
               lambda_param: float = 0.5) -> np.ndarray:
        """Select diverse and relevant texts using TF-IDF + MMR.

        Args:
            texts: List of text strings.
            query_text: Query string.
            k: Number to select.
            lambda_param: Trade-off.

        Returns:
            Selected indices.
        """
        all_texts = texts + [query_text]
        tfidf = tfidf_vectors(all_texts)
        items = tfidf[:-1]
        query = tfidf[-1]
        return self.selector.select_fast(items, query, k, lambda_param)


class InterpolatedMMR:
    """MMR with interpolated similarity from multiple sources."""

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights

    def select(self, feature_sets: List[np.ndarray], query_features: List[np.ndarray],
               k: int, lambda_param: float = 0.5) -> np.ndarray:
        """MMR with multiple feature representations.

        Args:
            feature_sets: List of (n, d_i) feature matrices.
            query_features: List of (d_i,) query vectors.
            k: Number to select.
            lambda_param: Trade-off.

        Returns:
            Selected indices.
        """
        n_sources = len(feature_sets)
        n = feature_sets[0].shape[0]
        k = min(k, n)

        if self.weights is None:
            w = np.ones(n_sources) / n_sources
        else:
            w = np.asarray(self.weights) / np.sum(self.weights)

        # Compute blended relevance and similarity
        rel = np.zeros(n)
        sim = np.zeros((n, n))
        for i, (feats, qf) in enumerate(zip(feature_sets, query_features)):
            feats = np.asarray(feats, dtype=np.float64)
            qf = np.asarray(qf, dtype=np.float64).ravel()
            rel += w[i] * cosine_similarity_matrix(feats, qf.reshape(1, -1)).ravel()
            sim += w[i] * cosine_similarity_matrix(feats, feats)

        selected = []
        mask = np.ones(n, dtype=bool)
        max_sim_to_sel = np.full(n, -np.inf)

        for _ in range(k):
            div_penalty = np.where(max_sim_to_sel > -np.inf, max_sim_to_sel, 0.0)
            scores = lambda_param * rel - (1 - lambda_param) * div_penalty
            scores[~mask] = -np.inf

            best = np.argmax(scores)
            selected.append(best)
            mask[best] = False
            np.maximum(max_sim_to_sel, sim[:, best], out=max_sim_to_sel)

        return np.array(selected, dtype=np.intp)


def demo_mmr():
    """Demonstrate MMR selection."""
    rng = np.random.RandomState(42)
    n, d = 100, 10
    items = rng.randn(n, d)
    query = rng.randn(d)

    selector = MMRSelector()

    # Standard MMR
    sel1 = selector.select(items, query, k=10, lambda_param=0.5)
    print(f"Standard MMR (k=10): {sel1}")

    # Fast MMR
    sel2 = selector.select_fast(items, query, k=10, lambda_param=0.5)
    print(f"Fast MMR (k=10): {sel2}")

    # Batch MMR
    sel3 = selector.select_batch(items, query, k=10, batch_size=3)
    print(f"Batch MMR (k=10, batch=3): {sel3}")

    # Adaptive MMR
    sel4, best_lam = selector.select_adaptive(items, query, k=10)
    print(f"Adaptive MMR (k=10): {sel4}, best λ={best_lam:.2f}")

    # Reranking
    scores = rng.rand(n)
    sel5 = MMRSelector.diversity_rerank(items, scores, k=10)
    print(f"Diversity rerank (k=10): {sel5}")

    # Fairness MMR
    groups = rng.choice(3, size=n)
    sel6 = MMRSelector.fairness_mmr(items, query, groups, k=10,
                                    min_per_group={0: 2, 1: 2, 2: 2})
    print(f"Fair MMR (k=10): {sel6}")
    for g in range(3):
        cnt = np.sum(groups[sel6] == g)
        print(f"  Group {g}: {cnt} items")


if __name__ == '__main__':
    demo_mmr()
