"""
Metric correlation analysis for the Diversity Decoding Arena.

Computes Kendall τ correlation matrices between diversity metrics and
performs spectral clustering to discover metric taxonomy.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scipy imports with pure-numpy fallbacks
# ---------------------------------------------------------------------------
try:
    from scipy.stats import kendalltau as _scipy_kendalltau
    from scipy.stats import pearsonr as _scipy_pearsonr
    from scipy.stats import spearmanr as _scipy_spearmanr

    _HAS_SCIPY_STATS = True
except ImportError:
    _HAS_SCIPY_STATS = False

try:
    from scipy.linalg import eigh as _scipy_eigh

    _HAS_SCIPY_LINALG = True
except ImportError:
    _HAS_SCIPY_LINALG = False


# ===================================================================
# Helper functions
# ===================================================================

def rank_data(x: np.ndarray) -> np.ndarray:
    """Assign average ranks to data values (handles ties)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[sorter] = np.arange(1, n + 1, dtype=float)

    # Average ranks for ties
    sorted_x = x[sorter]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg_rank = 0.5 * (i + 1 + j)
            for k in range(i, j):
                ranks[sorter[k]] = avg_rank
        i = j
    return ranks


def _merge_count_inversions(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Merge-sort based inversion count — O(n log n)."""
    n = len(arr)
    if n <= 1:
        return arr.copy(), 0
    mid = n // 2
    left, left_inv = _merge_count_inversions(arr[:mid])
    right, right_inv = _merge_count_inversions(arr[mid:])
    merged = np.empty(n, dtype=arr.dtype)
    inversions = left_inv + right_inv
    i = j = k = 0
    nl, nr = len(left), len(right)
    while i < nl and j < nr:
        if left[i] <= right[j]:
            merged[k] = left[i]
            i += 1
        else:
            merged[k] = right[j]
            inversions += nl - i
            j += 1
        k += 1
    while i < nl:
        merged[k] = left[i]
        i += 1
        k += 1
    while j < nr:
        merged[k] = right[j]
        j += 1
        k += 1
    return merged, inversions


def kendall_tau_fast(x: np.ndarray, y: np.ndarray) -> float:
    """O(n log n) Kendall tau using merge-sort inversion count.

    tau = (n_c - n_d) / (n_c + n_d)
    where n_c + n_d = n*(n-1)/2 for data without ties.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0

    # Sort by x, then count inversions in the induced y ordering
    order = np.argsort(x, kind="mergesort")
    y_sorted = y[order]
    ranks_y = rank_data(y_sorted)
    _, n_d = _merge_count_inversions(ranks_y)
    n_pairs = n * (n - 1) // 2
    n_c = n_pairs - n_d
    denom = n_c + n_d
    if denom == 0:
        return 0.0
    return (n_c - n_d) / denom


def concordance_matrix(rankings: np.ndarray) -> np.ndarray:
    """Compute pairwise concordance rates between multiple rankings.

    Parameters
    ----------
    rankings : np.ndarray
        Shape (m, n) — m rankings over n items.

    Returns
    -------
    np.ndarray
        Shape (m, m) concordance matrix.
    """
    rankings = np.asarray(rankings, dtype=float)
    m, n = rankings.shape
    C = np.ones((m, m), dtype=float)
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return C
    for i in range(m):
        for j in range(i + 1, m):
            nc, nd = _count_concordant_discordant_arrays(rankings[i], rankings[j])
            rate = nc / n_pairs if n_pairs > 0 else 0.5
            C[i, j] = rate
            C[j, i] = rate
    return C


def correlation_to_distance(correlation_matrix: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(2(1 - r))."""
    corr = np.asarray(correlation_matrix, dtype=float)
    inner = np.clip(2.0 * (1.0 - corr), 0.0, 4.0)
    return np.sqrt(inner)


def effective_dimensionality(eigenvalues: np.ndarray) -> float:
    """Participation ratio: (sum λ)^2 / sum λ^2.

    Estimates the effective number of independent dimensions.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    s1 = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    if s2 == 0:
        return 0.0
    return (s1 ** 2) / s2


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score.

    Parameters
    ----------
    X : np.ndarray
        (n, d) data matrix.
    labels : np.ndarray
        Cluster assignment for each sample.

    Returns
    -------
    float
        Mean silhouette coefficient in [-1, 1].
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(X)
    if n < 2:
        return 0.0
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Precompute pairwise distances
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    sil = np.zeros(n)
    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = own_mask.sum() - 1  # exclude self
        if own_count <= 0:
            sil[i] = 0.0
            continue
        a_i = dists[i][own_mask].sum() / own_count if own_count > 0 else 0.0

        b_i = np.inf
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = labels == lbl
            other_count = other_mask.sum()
            if other_count == 0:
                continue
            mean_dist = dists[i][other_mask].sum() / other_count
            if mean_dist < b_i:
                b_i = mean_dist

        if b_i == np.inf:
            sil[i] = 0.0
        else:
            sil[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(sil))


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute the Adjusted Rand Index between two clusterings.

    ARI = (RI - E[RI]) / (max(RI) - E[RI])
    """
    labels_a = np.asarray(labels_a, dtype=int)
    labels_b = np.asarray(labels_b, dtype=int)
    n = len(labels_a)
    if n < 2:
        return 0.0

    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)

    # Contingency table
    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=int)
    a_map = {v: i for i, v in enumerate(classes_a)}
    b_map = {v: i for i, v in enumerate(classes_b)}
    for idx in range(n):
        contingency[a_map[labels_a[idx]], b_map[labels_b[idx]]] += 1

    # Sum of C(n_ij, 2)
    sum_comb_ij = sum(
        int(contingency[i, j]) * (int(contingency[i, j]) - 1) // 2
        for i in range(len(classes_a))
        for j in range(len(classes_b))
    )

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    sum_comb_a = sum(int(s) * (int(s) - 1) // 2 for s in row_sums)
    sum_comb_b = sum(int(s) * (int(s) - 1) // 2 for s in col_sums)

    comb_n = n * (n - 1) // 2

    expected = sum_comb_a * sum_comb_b / comb_n if comb_n > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if abs(denom) < 1e-15:
        return 1.0 if sum_comb_ij == expected else 0.0
    return (sum_comb_ij - expected) / denom


def _count_concordant_discordant_arrays(
    x: np.ndarray, y: np.ndarray
) -> Tuple[int, int]:
    """Count concordant and discordant pairs between arrays x, y."""
    n = len(x)
    n_c = 0
    n_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            prod = dx * dy
            if prod > 0:
                n_c += 1
            elif prod < 0:
                n_d += 1
    return n_c, n_d


# ===================================================================
# KMeans — from-scratch implementation
# ===================================================================


class KMeans:
    """Simple k-means clustering with k-means++ initialization."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-6,
        n_init: int = 10,
        seed: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.seed = seed

        self.labels_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit KMeans to data *X* (n_samples, n_features).

        Runs *n_init* independent initialisations and keeps the best.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        k = min(self.n_clusters, n)
        if k <= 0:
            self.labels_ = np.zeros(n, dtype=int)
            self.centroids_ = X[:1].copy() if n > 0 else np.empty((0, X.shape[1]))
            self.inertia_ = 0.0
            return self

        rng = np.random.RandomState(self.seed)
        best_inertia = np.inf
        best_labels: Optional[np.ndarray] = None
        best_centroids: Optional[np.ndarray] = None

        for _ in range(self.n_init):
            centroids = self._init_centroids(X, k, rng)
            for _it in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels, k)
                shift = np.sqrt(((new_centroids - centroids) ** 2).sum())
                centroids = new_centroids
                if shift < self.tol:
                    break
            labels = self._assign_clusters(X, centroids)
            inertia = self._compute_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row in *X* to the nearest centroid."""
        if self.centroids_ is None:
            raise RuntimeError("KMeans has not been fitted yet.")
        return self._assign_clusters(np.asarray(X, dtype=float), self.centroids_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_centroids(
        X: np.ndarray, k: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """K-means++ initialization.

        1. Choose first centroid uniformly at random.
        2. For each subsequent centroid, choose with probability proportional
           to squared distance from the nearest existing centroid.
        """
        n, d = X.shape
        centroids = np.empty((k, d), dtype=float)
        idx = rng.randint(n)
        centroids[0] = X[idx]
        for c in range(1, k):
            dists = np.full(n, np.inf)
            for prev in range(c):
                d2 = ((X - centroids[prev]) ** 2).sum(axis=1)
                dists = np.minimum(dists, d2)
            total = dists.sum()
            if total == 0:
                centroids[c] = X[rng.randint(n)]
            else:
                probs = dists / total
                cum = np.cumsum(probs)
                r = rng.rand()
                idx = int(np.searchsorted(cum, r))
                idx = min(idx, n - 1)
                centroids[c] = X[idx]
        return centroids

    @staticmethod
    def _assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest centroid."""
        # Broadcast: (n, 1, d) - (1, k, d) -> (n, k, d) -> sum -> (n, k)
        diffs = X[:, None, :] - centroids[None, :, :]
        dists = (diffs ** 2).sum(axis=2)
        return np.argmin(dists, axis=1).astype(int)

    @staticmethod
    def _update_centroids(
        X: np.ndarray, labels: np.ndarray, k: int
    ) -> np.ndarray:
        """Recompute centroids as cluster means."""
        d = X.shape[1]
        centroids = np.zeros((k, d), dtype=float)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(axis=0)
            else:
                centroids[c] = X[np.random.randint(len(X))]
        return centroids

    @staticmethod
    def _compute_inertia(
        X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        """Sum of squared distances to assigned centroids."""
        total = 0.0
        for c in range(len(centroids)):
            mask = labels == c
            if mask.any():
                total += ((X[mask] - centroids[c]) ** 2).sum()
        return float(total)


# ===================================================================
# SpectralClusterer
# ===================================================================


class SpectralClusterer:
    """Spectral clustering on a similarity / correlation matrix."""

    def cluster(
        self, similarity_matrix: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Return cluster labels via normalised spectral clustering.

        Steps:
        1. Build normalised Laplacian L_sym.
        2. Compute the first *n_clusters* eigenvectors.
        3. Run k-means on the eigenvector embedding.
        """
        W = np.asarray(similarity_matrix, dtype=float)
        n = W.shape[0]
        if n_clusters <= 0 or n_clusters > n:
            n_clusters = min(max(n_clusters, 1), n)

        L = self._normalized_laplacian(W)
        eigenvalues, eigenvectors = self._eigen_decomposition(L, n_clusters)
        labels = self._kmeans(eigenvectors, n_clusters)
        return labels

    # ------------------------------------------------------------------
    # Laplacians
    # ------------------------------------------------------------------

    @staticmethod
    def _build_laplacian(W: np.ndarray) -> np.ndarray:
        """Build the normalised graph Laplacian (alias for _normalized_laplacian)."""
        return SpectralClusterer._normalized_laplacian(W)

    @staticmethod
    def _unnormalized_laplacian(W: np.ndarray) -> np.ndarray:
        """L = D - W."""
        D = np.diag(W.sum(axis=1))
        return D - W

    @staticmethod
    def _normalized_laplacian(W: np.ndarray) -> np.ndarray:
        """L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}."""
        d = W.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 1e-15
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        n = W.shape[0]
        return np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # ------------------------------------------------------------------
    # Eigen helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eigen_decomposition(
        L: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *k* smallest eigenvalues/vectors of symmetric *L*."""
        if _HAS_SCIPY_LINALG:
            eigenvalues, eigenvectors = _scipy_eigh(
                L, subset_by_index=[0, k - 1]
            )
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)[:k]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iter: int = 300) -> np.ndarray:
        """Run k-means on the row-normalised eigenvector matrix."""
        # Row-normalise
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        X_norm = X / norms

        km = KMeans(n_clusters=k, max_iter=max_iter, seed=42)
        km.fit(X_norm)
        return km.labels_ if km.labels_ is not None else np.zeros(X.shape[0], dtype=int)

    @staticmethod
    def _eigenvalue_gap(eigenvalues: np.ndarray) -> int:
        """Estimate optimal k from the largest eigenvalue gap.

        Returns the index of the gap + 1  (= number of clusters).
        Minimum returned value is 2.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        if len(eigenvalues) < 2:
            return 1
        gaps = np.diff(eigenvalues)
        # Ignore the first gap (near-zero eigenvalue to second)
        best = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
        return max(best, 2)

    @staticmethod
    def _affinity_from_correlation(correlation_matrix: np.ndarray) -> np.ndarray:
        """Convert correlation matrix to a non-negative affinity matrix.

        W_ij = (1 + corr_ij) / 2   ∈ [0, 1]
        Diagonal set to 0 to avoid self-loops.
        """
        W = (1.0 + np.asarray(correlation_matrix, dtype=float)) / 2.0
        np.fill_diagonal(W, 0.0)
        return W


# ===================================================================
# MetricCorrelationAnalyzer
# ===================================================================


class MetricCorrelationAnalyzer:
    """Compute pairwise correlations between diversity metrics."""

    def __init__(
        self,
        metrics: List[str],
        confidence_level: float = 0.95,
    ):
        self.metrics = list(metrics)
        self.confidence_level = confidence_level
        self._spectral = SpectralClusterer()

    # ------------------------------------------------------------------
    # Core correlations
    # ------------------------------------------------------------------

    def compute_correlation_matrix(
        self, metric_values: Dict[str, List[float]]
    ) -> np.ndarray:
        """Pairwise Kendall τ between all metric pairs.

        Parameters
        ----------
        metric_values : dict
            Maps metric name → list of observed values (same length for all).

        Returns
        -------
        np.ndarray
            (m, m) correlation matrix.
        """
        m = len(self.metrics)
        corr = np.eye(m, dtype=float)
        arrays = {
            name: np.asarray(metric_values[name], dtype=float)
            for name in self.metrics
        }
        for i in range(m):
            for j in range(i + 1, m):
                tau, _ = self.kendall_tau(arrays[self.metrics[i]], arrays[self.metrics[j]])
                corr[i, j] = tau
                corr[j, i] = tau
        return corr

    def compute_with_pvalues(
        self, metric_values: Dict[str, List[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (correlation_matrix, pvalue_matrix)."""
        m = len(self.metrics)
        corr = np.eye(m, dtype=float)
        pvals = np.zeros((m, m), dtype=float)
        arrays = {
            name: np.asarray(metric_values[name], dtype=float)
            for name in self.metrics
        }
        for i in range(m):
            for j in range(i + 1, m):
                tau, p = self.kendall_tau(arrays[self.metrics[i]], arrays[self.metrics[j]])
                corr[i, j] = corr[j, i] = tau
                pvals[i, j] = pvals[j, i] = p
        return corr, pvals

    # ------------------------------------------------------------------
    # Individual correlation methods
    # ------------------------------------------------------------------

    @staticmethod
    def kendall_tau(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """Kendall tau-b with p-value.

        tau = (n_c - n_d) / sqrt((n_0 - n_x)(n_0 - n_y))
        where n_0 = n(n-1)/2, n_x = ties in x, n_y = ties in y.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if _HAS_SCIPY_STATS:
            tau, p = _scipy_kendalltau(x, y)
            if np.isnan(tau):
                return 0.0, 1.0
            return float(tau), float(p)

        # Fallback pure-numpy implementation
        n = len(x)
        if n < 2:
            return 0.0, 1.0

        n_c, n_d = _count_concordant_discordant_arrays(x, y)
        n_0 = n * (n - 1) // 2

        # Count ties
        def _count_ties(arr: np.ndarray) -> int:
            vals, counts = np.unique(arr, return_counts=True)
            return sum(int(c) * (int(c) - 1) // 2 for c in counts if c > 1)

        n_x = _count_ties(x)
        n_y = _count_ties(y)

        denom = math.sqrt((n_0 - n_x) * (n_0 - n_y))
        if denom == 0:
            return 0.0, 1.0
        tau = (n_c - n_d) / denom

        # Approximate p-value (normal approximation)
        var = (2.0 * (2 * n + 5)) / (9.0 * n * (n - 1))
        if var <= 0:
            return float(tau), 1.0
        z = tau / math.sqrt(var)
        p = 2.0 * _normal_sf(abs(z))
        return float(tau), float(p)

    @staticmethod
    def _count_concordant_discordant(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[int, int]:
        """Count concordant and discordant pairs."""
        return _count_concordant_discordant_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )

    @staticmethod
    def spearman_rho(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """Spearman rank correlation with p-value."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if _HAS_SCIPY_STATS:
            rho, p = _scipy_spearmanr(x, y)
            if np.isnan(rho):
                return 0.0, 1.0
            return float(rho), float(p)
        # Fallback
        rx = rank_data(x)
        ry = rank_data(y)
        return MetricCorrelationAnalyzer.pearson_r(rx, ry)

    @staticmethod
    def pearson_r(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """Pearson correlation coefficient with p-value."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if _HAS_SCIPY_STATS:
            r, p = _scipy_pearsonr(x, y)
            if np.isnan(r):
                return 0.0, 1.0
            return float(r), float(p)
        # Fallback
        n = len(x)
        if n < 2:
            return 0.0, 1.0
        mx, my = x.mean(), y.mean()
        dx, dy = x - mx, y - my
        num = (dx * dy).sum()
        den = math.sqrt((dx ** 2).sum() * (dy ** 2).sum())
        if den == 0:
            return 0.0, 1.0
        r = num / den
        r = max(-1.0, min(1.0, r))
        # t-test
        if abs(r) >= 1.0:
            return float(r), 0.0
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        p = 2.0 * _t_sf(abs(t_stat), n - 2)
        return float(r), float(p)

    @staticmethod
    def partial_correlation(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Partial Pearson correlation between x and y, controlling for z.

        r_{xy.z} = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        r_xy, _ = MetricCorrelationAnalyzer.pearson_r(x, y)
        r_xz, _ = MetricCorrelationAnalyzer.pearson_r(x, z)
        r_yz, _ = MetricCorrelationAnalyzer.pearson_r(y, z)

        denom = math.sqrt(max(0.0, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
        if denom < 1e-15:
            return 0.0
        return (r_xy - r_xz * r_yz) / denom

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    def discover_metric_clusters(
        self,
        correlation_matrix: np.ndarray,
        n_clusters: int,
    ) -> List[List[int]]:
        """Group metrics into clusters via spectral clustering.

        Returns list of lists, each inner list contains metric indices.
        """
        W = self._spectral._affinity_from_correlation(correlation_matrix)
        labels = self._spectral.cluster(W, n_clusters)
        clusters: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(idx)
        return [clusters[k] for k in sorted(clusters)]

    def optimal_num_clusters(
        self, correlation_matrix: np.ndarray
    ) -> int:
        """Estimate the optimal number of metric clusters via eigenvalue gap."""
        W = self._spectral._affinity_from_correlation(correlation_matrix)
        L = self._spectral._normalized_laplacian(W)
        k = min(L.shape[0], 20)
        eigenvalues, _ = self._spectral._eigen_decomposition(L, k)
        return self._spectral._eigenvalue_gap(eigenvalues)

    def reduce_metric_set(
        self,
        correlation_matrix: np.ndarray,
        threshold: float = 0.8,
    ) -> List[int]:
        """Select one representative metric per cluster.

        First clusters metrics at the given redundancy threshold, then
        picks the metric with highest average absolute correlation within
        each cluster (most 'central' member).
        """
        n = correlation_matrix.shape[0]
        if n <= 1:
            return list(range(n))

        n_clusters = self.optimal_num_clusters(correlation_matrix)
        clusters = self.discover_metric_clusters(correlation_matrix, n_clusters)

        representatives: List[int] = []
        for cluster in clusters:
            if len(cluster) == 1:
                representatives.append(cluster[0])
                continue
            # Pick the metric with highest mean |correlation| to others in cluster
            best_idx = cluster[0]
            best_score = -1.0
            for idx in cluster:
                score = np.mean(
                    [abs(correlation_matrix[idx, j]) for j in cluster if j != idx]
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            representatives.append(best_idx)
        return sorted(representatives)

    # ------------------------------------------------------------------
    # Visualisation / reporting
    # ------------------------------------------------------------------

    def plot_data(
        self,
        correlation_matrix: np.ndarray,
        metric_names: Optional[List[str]] = None,
    ) -> dict:
        """Return data suitable for rendering a heatmap.

        Returns dict with keys: matrix, metric_names, vmin, vmax, annotations.
        """
        names = metric_names or self.metrics
        m = correlation_matrix.shape[0]
        annotations: List[List[str]] = []
        for i in range(m):
            row: List[str] = []
            for j in range(m):
                row.append(f"{correlation_matrix[i, j]:.2f}")
            annotations.append(row)
        return {
            "matrix": correlation_matrix.tolist(),
            "metric_names": list(names),
            "vmin": -1.0,
            "vmax": 1.0,
            "annotations": annotations,
        }

    def summary(
        self,
        correlation_matrix: np.ndarray,
        metric_names: Optional[List[str]] = None,
    ) -> str:
        """Generate a human-readable summary of correlation analysis."""
        names = metric_names or self.metrics
        m = correlation_matrix.shape[0]
        lines: List[str] = [
            "=== Metric Correlation Summary ===",
            f"Number of metrics: {m}",
        ]

        # Top-5 most correlated pairs
        pairs: List[Tuple[float, str, str]] = []
        for i in range(m):
            for j in range(i + 1, m):
                pairs.append((abs(correlation_matrix[i, j]), names[i], names[j]))
        pairs.sort(reverse=True)

        lines.append("\nMost correlated pairs:")
        for val, a, b in pairs[:5]:
            sign = correlation_matrix[
                names.index(a) if a in names else 0,
                names.index(b) if b in names else 0,
            ]
            lines.append(f"  {a} <-> {b}: τ = {sign:+.4f}")

        # Least correlated
        lines.append("\nLeast correlated (most independent) pairs:")
        for val, a, b in pairs[-5:]:
            sign = correlation_matrix[
                names.index(a) if a in names else 0,
                names.index(b) if b in names else 0,
            ]
            lines.append(f"  {a} <-> {b}: τ = {sign:+.4f}")

        # Effective dimensionality
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        eff_dim = effective_dimensionality(np.maximum(eigenvalues, 0))
        lines.append(f"\nEffective dimensionality: {eff_dim:.2f} / {m}")

        # Optimal clusters
        n_clust = self.optimal_num_clusters(correlation_matrix)
        lines.append(f"Suggested metric clusters: {n_clust}")

        return "\n".join(lines)


# ===================================================================
# CorrelationBootstrap
# ===================================================================


class CorrelationBootstrap:
    """Bootstrap confidence intervals for correlation estimates."""

    @staticmethod
    def bootstrap_correlation(
        x: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap a single Kendall tau and return (mean_tau, (lo, hi))."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.RandomState(seed)
        n = len(x)
        taus = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            taus[b] = CorrelationBootstrap._single_bootstrap(x, y, rng)
        alpha = 1.0 - confidence
        lo = float(np.percentile(taus, 100 * alpha / 2))
        hi = float(np.percentile(taus, 100 * (1 - alpha / 2)))
        return float(np.mean(taus)), (lo, hi)

    @staticmethod
    def bootstrap_correlation_matrix(
        metric_values: Dict[str, List[float]],
        n_bootstrap: int = 500,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap all pairwise Kendall tau correlations.

        Returns (mean_matrix, lower_ci_matrix, upper_ci_matrix).
        """
        names = sorted(metric_values.keys())
        m = len(names)
        mean_mat = np.eye(m, dtype=float)
        lo_mat = np.eye(m, dtype=float)
        hi_mat = np.eye(m, dtype=float)

        arrays = {n: np.asarray(metric_values[n], dtype=float) for n in names}
        rng = np.random.RandomState(seed)

        for i in range(m):
            for j in range(i + 1, m):
                mean_tau, (lo, hi) = CorrelationBootstrap.bootstrap_correlation(
                    arrays[names[i]],
                    arrays[names[j]],
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=rng.randint(2**31),
                )
                mean_mat[i, j] = mean_mat[j, i] = mean_tau
                lo_mat[i, j] = lo_mat[j, i] = lo
                hi_mat[i, j] = hi_mat[j, i] = hi
        return mean_mat, lo_mat, hi_mat

    @staticmethod
    def _single_bootstrap(
        x: np.ndarray, y: np.ndarray, rng: np.random.RandomState
    ) -> float:
        """Draw a single bootstrap sample and compute Kendall tau."""
        n = len(x)
        idx = rng.randint(0, n, size=n)
        tau = kendall_tau_fast(x[idx], y[idx])
        return tau


# ===================================================================
# MetricRedundancyAnalyzer
# ===================================================================


class MetricRedundancyAnalyzer:
    """Identify redundant and orthogonal metric pairs."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def find_redundant_pairs(
        self,
        correlation_matrix: np.ndarray,
        metric_names: List[str],
    ) -> List[Tuple[str, str, float]]:
        """Return pairs whose |correlation| ≥ threshold, sorted desc."""
        m = correlation_matrix.shape[0]
        result: List[Tuple[str, str, float]] = []
        for i in range(m):
            for j in range(i + 1, m):
                r = abs(correlation_matrix[i, j])
                if r >= self.threshold:
                    result.append((metric_names[i], metric_names[j], float(correlation_matrix[i, j])))
        result.sort(key=lambda t: abs(t[2]), reverse=True)
        return result

    def find_orthogonal_pairs(
        self,
        correlation_matrix: np.ndarray,
        metric_names: List[str],
        orthogonal_threshold: float = 0.1,
    ) -> List[Tuple[str, str, float]]:
        """Return pairs whose |correlation| ≤ orthogonal_threshold."""
        m = correlation_matrix.shape[0]
        result: List[Tuple[str, str, float]] = []
        for i in range(m):
            for j in range(i + 1, m):
                r = abs(correlation_matrix[i, j])
                if r <= orthogonal_threshold:
                    result.append((metric_names[i], metric_names[j], float(correlation_matrix[i, j])))
        result.sort(key=lambda t: abs(t[2]))
        return result

    def select_representative_metrics(
        self,
        correlation_matrix: np.ndarray,
        metric_names: List[str],
        n_representatives: Optional[int] = None,
    ) -> List[str]:
        """Greedily select a maximally independent subset.

        Algorithm:
        1. Start with the metric that has the lowest mean |correlation|.
        2. Iteratively add the metric with the lowest max |correlation|
           to any already-selected metric.
        """
        m = correlation_matrix.shape[0]
        if n_representatives is None:
            eigenvalues = np.linalg.eigvalsh(correlation_matrix)
            n_representatives = max(1, int(round(effective_dimensionality(np.maximum(eigenvalues, 0)))))
        n_representatives = min(n_representatives, m)

        abs_corr = np.abs(correlation_matrix).copy()
        np.fill_diagonal(abs_corr, 0.0)

        # Start with the most independent metric overall
        mean_corr = abs_corr.mean(axis=1)
        selected: List[int] = [int(np.argmin(mean_corr))]
        remaining = set(range(m)) - set(selected)

        while len(selected) < n_representatives and remaining:
            best_idx = -1
            best_score = np.inf
            for idx in remaining:
                score = max(abs_corr[idx, s] for s in selected)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return [metric_names[i] for i in sorted(selected)]

    def metric_independence_score(
        self, correlation_matrix: np.ndarray
    ) -> float:
        """Overall independence score in [0, 1].

        1.0 = all metrics perfectly independent (identity matrix).
        0.0 = all metrics perfectly correlated.
        """
        m = correlation_matrix.shape[0]
        if m <= 1:
            return 1.0
        abs_corr = np.abs(correlation_matrix)
        np.fill_diagonal(abs_corr, 0.0)
        off_diag_mean = abs_corr.sum() / (m * (m - 1))
        return float(1.0 - off_diag_mean)

    def information_content(
        self, correlation_matrix: np.ndarray
    ) -> float:
        """Effective dimensionality as a measure of information content."""
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        return effective_dimensionality(np.maximum(eigenvalues, 0))


# ===================================================================
# ConditionedCorrelation
# ===================================================================


class ConditionedCorrelation:
    """Compute correlations conditioned on a third variable."""

    @staticmethod
    def compute_conditioned(
        metric_values: Dict[str, List[float]],
        condition_variable: str,
        condition_bins: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Compute correlation matrices within bins of *condition_variable*.

        Parameters
        ----------
        metric_values : dict
            All metric values including the condition variable.
        condition_variable : str
            Name of the variable to condition on.
        condition_bins : int
            Number of quantile bins.

        Returns
        -------
        dict
            Maps bin label ("bin_0", …) → correlation matrix.
        """
        cond = np.asarray(metric_values[condition_variable], dtype=float)
        other_names = sorted(k for k in metric_values if k != condition_variable)
        m = len(other_names)
        n = len(cond)

        # Quantile bin edges
        percentiles = np.linspace(0, 100, condition_bins + 1)
        edges = np.percentile(cond, percentiles)
        edges[0] -= 1e-10
        edges[-1] += 1e-10

        result: Dict[str, np.ndarray] = {}
        for b in range(condition_bins):
            mask = (cond > edges[b]) & (cond <= edges[b + 1])
            if mask.sum() < 3:
                result[f"bin_{b}"] = np.eye(m, dtype=float)
                continue
            subset = {name: np.asarray(metric_values[name], dtype=float)[mask] for name in other_names}
            analyzer = MetricCorrelationAnalyzer(other_names)
            result[f"bin_{b}"] = analyzer.compute_correlation_matrix(subset)
        return result

    @staticmethod
    def compare_correlations(
        corr_a: np.ndarray, corr_b: np.ndarray
    ) -> np.ndarray:
        """Element-wise difference: corr_a - corr_b."""
        return np.asarray(corr_a, dtype=float) - np.asarray(corr_b, dtype=float)

    @staticmethod
    def stability_across_conditions(
        conditioned_correlations: Dict[str, np.ndarray],
    ) -> float:
        """Mean pairwise Frobenius distance across condition bins.

        Lower values indicate more stable correlations.
        Returns 0 if fewer than 2 bins.
        """
        matrices = list(conditioned_correlations.values())
        k = len(matrices)
        if k < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                diff = matrices[i] - matrices[j]
                total += float(np.sqrt((diff ** 2).sum()))
                count += 1
        return total / count if count > 0 else 0.0


# ===================================================================
# Private statistical helpers (fallback when scipy unavailable)
# ===================================================================


def _normal_sf(z: float) -> float:
    """Survival function for standard normal (1 - Φ(z)), Abramowitz & Stegun."""
    # Use the complementary error function approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = (
        0.319381530 * t
        - 0.356563782 * t ** 2
        + 1.781477937 * t ** 3
        - 1.821255978 * t ** 4
        + 1.330274429 * t ** 5
    )
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    sf = pdf * poly
    if z < 0:
        sf = 1.0 - sf
    return max(0.0, min(1.0, sf))


def _t_sf(t_val: float, df: int) -> float:
    """Approximate survival function for t-distribution.

    For df ≥ 30 uses normal approximation; otherwise uses a rough
    Cornish–Fisher expansion.
    """
    if df <= 0:
        return 0.5
    if df >= 30:
        return _normal_sf(t_val)
    # Approximation: z ≈ t * (1 - 1/(4*df))
    z = t_val * (1.0 - 1.0 / (4.0 * df))
    return _normal_sf(z)


# ===================================================================
# Module-level exports
# ===================================================================

__all__ = [
    "MetricCorrelationAnalyzer",
    "SpectralClusterer",
    "KMeans",
    "CorrelationBootstrap",
    "MetricRedundancyAnalyzer",
    "ConditionedCorrelation",
    "kendall_tau_fast",
    "rank_data",
    "concordance_matrix",
    "correlation_to_distance",
    "effective_dimensionality",
    "silhouette_score",
    "adjusted_rand_index",
]
