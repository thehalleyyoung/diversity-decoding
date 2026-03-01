"""
Diversity metrics for embedding spaces.

Implements volume-based diversity, spread, coverage, entropy-based,
Vendi score, representation diversity, projection-based, and
dimensional analysis.
"""

import numpy as np
from typing import Optional, List, Tuple, NamedTuple, Dict


class DiversityScores(NamedTuple):
    """Container for diversity metric results."""
    volume: float
    spread: float
    coverage: float
    entropy: float
    vendi_score: float
    representation: float
    projection: float
    intrinsic_dim: float


class EmbeddingDiversity:
    """Compute diversity metrics for sets of embeddings.

    All metrics operate on (n, d) embedding matrices where n is the
    number of items and d is the embedding dimensionality.
    """

    def __init__(self):
        pass

    def compute(self, embeddings: np.ndarray,
                n_bins: int = 20,
                n_projections: int = 50) -> DiversityScores:
        """Compute all diversity metrics.

        Args:
            embeddings: (n, d) embedding matrix.
            n_bins: Bins for entropy/coverage discretization.
            n_projections: Random projections for projection-based metric.

        Returns:
            DiversityScores with all metrics.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)

        vol = self.volume_diversity(embeddings)
        spr = self.spread(embeddings)
        cov = self.coverage(embeddings, n_bins=n_bins)
        ent = self.entropy_diversity(embeddings, n_bins=n_bins)
        ven = self.vendi_score(embeddings)
        rep = self.representation_diversity(embeddings)
        proj = self.projection_diversity(embeddings, n_projections=n_projections)
        dim = self.intrinsic_dimensionality(embeddings)

        return DiversityScores(
            volume=vol, spread=spr, coverage=cov, entropy=ent,
            vendi_score=ven, representation=rep, projection=proj,
            intrinsic_dim=dim
        )

    # ------------------------------------------------------------------
    # Volume-based diversity: log-det of Gram matrix
    # ------------------------------------------------------------------

    @staticmethod
    def volume_diversity(embeddings: np.ndarray) -> float:
        """Volume-based diversity: log-det of the Gram matrix.

        The Gram matrix G = X X^T captures the "volume" spanned by
        the embeddings. Higher volume = more diverse.

        For numerical stability, uses log-det with regularization.

        Args:
            embeddings: (n, d) matrix.

        Returns:
            Log-determinant of regularized Gram matrix.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n = X.shape[0]
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.log(np.dot(X[0], X[0]) + 1e-10))

        G = X @ X.T + 1e-6 * np.eye(n)
        sign, logdet = np.linalg.slogdet(G)
        if sign <= 0:
            return -np.inf
        return float(logdet)

    # ------------------------------------------------------------------
    # Spread: mean pairwise distance
    # ------------------------------------------------------------------

    @staticmethod
    def spread(embeddings: np.ndarray, metric: str = 'euclidean') -> float:
        """Spread: mean pairwise distance.

        Args:
            embeddings: (n, d) matrix.
            metric: 'euclidean' or 'cosine'.

        Returns:
            Mean pairwise distance.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n = X.shape[0]
        if n <= 1:
            return 0.0

        if metric == 'euclidean':
            sq = np.sum(X ** 2, axis=1)
            D_sq = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
            D = np.sqrt(np.maximum(D_sq, 0.0))
        elif metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms
            sim = Xn @ Xn.T
            np.clip(sim, -1.0, 1.0, out=sim)
            D = 1.0 - sim
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Mean of upper triangle
        upper_idx = np.triu_indices(n, k=1)
        return float(np.mean(D[upper_idx]))

    # ------------------------------------------------------------------
    # Coverage: fraction of Voronoi cells occupied
    # ------------------------------------------------------------------

    @staticmethod
    def coverage(embeddings: np.ndarray, n_bins: int = 20) -> float:
        """Coverage: fraction of discretized space cells occupied.

        Discretizes each dimension into n_bins bins and counts
        the fraction of unique cells occupied.

        Args:
            embeddings: (n, d) matrix.
            n_bins: Number of bins per dimension.

        Returns:
            Fraction of occupied cells (0 to 1).
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n, d = X.shape
        if n == 0:
            return 0.0

        # For high dimensions, project to lower dims first
        effective_d = min(d, 5)
        if d > effective_d:
            # PCA-like projection using SVD
            X_centered = X - X.mean(axis=0)
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            X_proj = U[:, :effective_d] * S[:effective_d]
        else:
            X_proj = X

        # Discretize each dimension
        bins_per_dim = min(n_bins, int(np.ceil(n ** (1.0 / effective_d))))
        bins_per_dim = max(bins_per_dim, 2)

        cell_ids = set()
        for i in range(n):
            cell = []
            for j in range(X_proj.shape[1]):
                col = X_proj[:, j]
                vmin, vmax = col.min(), col.max()
                if vmax - vmin < 1e-12:
                    b = 0
                else:
                    b = int((X_proj[i, j] - vmin) / (vmax - vmin) * (bins_per_dim - 1))
                    b = min(b, bins_per_dim - 1)
                cell.append(b)
            cell_ids.add(tuple(cell))

        total_cells = bins_per_dim ** X_proj.shape[1]
        return float(len(cell_ids)) / min(total_cells, n * 10)  # Normalize

    # ------------------------------------------------------------------
    # Entropy-based diversity
    # ------------------------------------------------------------------

    @staticmethod
    def entropy_diversity(embeddings: np.ndarray, n_bins: int = 20) -> float:
        """Entropy-based diversity: Shannon entropy of discretized distribution.

        Higher entropy = more spread out = more diverse.

        Args:
            embeddings: (n, d) matrix.
            n_bins: Number of bins for histogram.

        Returns:
            Shannon entropy (nats).
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n, d = X.shape
        if n <= 1:
            return 0.0

        # Compute entropy per dimension and average
        total_entropy = 0.0
        for j in range(d):
            col = X[:, j]
            # Create histogram
            hist, _ = np.histogram(col, bins=n_bins)
            probs = hist.astype(np.float64) / n
            probs = probs[probs > 0]
            total_entropy += -np.sum(probs * np.log(probs))

        return float(total_entropy / d)

    # ------------------------------------------------------------------
    # Vendi Score
    # ------------------------------------------------------------------

    @staticmethod
    def vendi_score(embeddings: np.ndarray, kernel: str = 'cosine') -> float:
        """Vendi score: exp(Shannon entropy of eigenvalues of normalized kernel).

        The Vendi score measures the effective number of unique items.
        It equals 1 when all items are identical and n when all are orthogonal.

        VS = exp(-Σ_i λ_i log λ_i) where λ_i are eigenvalues of K/n.

        Args:
            embeddings: (n, d) matrix.
            kernel: 'cosine' or 'rbf'.

        Returns:
            Vendi score (effective number of unique items).
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n = X.shape[0]
        if n <= 1:
            return float(n)

        # Compute kernel matrix
        if kernel == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms
            K = Xn @ Xn.T
        elif kernel == 'rbf':
            sq = np.sum(X ** 2, axis=1)
            D_sq = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
            D_sq = np.maximum(D_sq, 0.0)
            gamma = 1.0 / X.shape[1]
            K = np.exp(-gamma * D_sq)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Normalize kernel: K_normalized = K / n
        K_norm = K / n

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(K_norm)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Remove negligible eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        if len(eigenvalues) == 0:
            return 1.0

        # Shannon entropy of eigenvalue distribution
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

        return float(np.exp(entropy))

    # ------------------------------------------------------------------
    # Representation diversity: principal angles
    # ------------------------------------------------------------------

    @staticmethod
    def representation_diversity(embeddings: np.ndarray) -> float:
        """Representation diversity: how well embeddings span the space.

        Measured via the ratio of singular values. A set that spans
        many directions has more uniform singular values.

        Returns the normalized entropy of singular values, which ranges
        from 0 (all along one direction) to 1 (uniform in all directions).

        Args:
            embeddings: (n, d) matrix.

        Returns:
            Representation diversity score [0, 1].
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n, d = X.shape
        if n <= 1 or d <= 1:
            return 0.0

        X_centered = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)

        # Normalized singular values as "probabilities"
        S_pos = S[S > 1e-12]
        if len(S_pos) <= 1:
            return 0.0

        p = S_pos / S_pos.sum()
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(S_pos))

        if max_entropy < 1e-12:
            return 0.0

        return float(entropy / max_entropy)

    # ------------------------------------------------------------------
    # Projection-based diversity
    # ------------------------------------------------------------------

    @staticmethod
    def projection_diversity(embeddings: np.ndarray,
                             n_projections: int = 50,
                             rng: Optional[np.random.RandomState] = None) -> float:
        """Projection-based diversity: max spread in random projections.

        Projects embeddings onto random 1D directions and measures
        the spread (std dev) in each. Reports the average.
        This is related to the sliced Wasserstein distance approach.

        Args:
            embeddings: (n, d) matrix.
            n_projections: Number of random projections.
            rng: Random state.

        Returns:
            Mean standard deviation across random projections.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n, d = X.shape
        if n <= 1:
            return 0.0

        if rng is None:
            rng = np.random.RandomState(42)

        spreads = []
        for _ in range(n_projections):
            direction = rng.randn(d)
            direction /= max(np.linalg.norm(direction), 1e-12)
            projections = X @ direction
            spreads.append(np.std(projections))

        return float(np.mean(spreads))

    # ------------------------------------------------------------------
    # Intrinsic dimensionality
    # ------------------------------------------------------------------

    @staticmethod
    def intrinsic_dimensionality(embeddings: np.ndarray,
                                  threshold: float = 0.95) -> float:
        """Effective intrinsic dimensionality of the selected set.

        Uses PCA: the number of principal components needed to explain
        `threshold` fraction of variance.

        Args:
            embeddings: (n, d) matrix.
            threshold: Variance fraction threshold.

        Returns:
            Effective intrinsic dimensionality.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n, d = X.shape
        if n <= 1:
            return 1.0

        X_centered = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)

        variance = S ** 2
        total_var = variance.sum()
        if total_var < 1e-12:
            return 1.0

        cumulative = np.cumsum(variance) / total_var

        # Number of components to reach threshold
        n_components = int(np.searchsorted(cumulative, threshold) + 1)

        # Participation ratio (continuous estimate)
        p = variance / total_var
        p = p[p > 1e-12]
        participation_ratio = 1.0 / np.sum(p ** 2)

        # Return participation ratio as the continuous measure
        return float(participation_ratio)


# ======================================================================
# Additional specialized metrics
# ======================================================================

class PairwiseDiversityAnalyzer:
    """Analyze pairwise diversity structure."""

    @staticmethod
    def distance_distribution(embeddings: np.ndarray,
                              metric: str = 'euclidean') -> Dict:
        """Compute statistics of pairwise distance distribution.

        Args:
            embeddings: (n, d) matrix.
            metric: Distance metric.

        Returns:
            Dict with mean, std, min, max, median, skewness, kurtosis.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n = X.shape[0]
        if n <= 1:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                    'median': 0, 'skewness': 0, 'kurtosis': 0}

        if metric == 'euclidean':
            sq = np.sum(X ** 2, axis=1)
            D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            D = 1.0 - (X / norms) @ (X / norms).T

        upper = D[np.triu_indices(n, k=1)]

        mean = float(np.mean(upper))
        std = float(np.std(upper))
        if std < 1e-12:
            skewness = 0.0
            kurtosis = 0.0
        else:
            centered = upper - mean
            skewness = float(np.mean(centered ** 3) / (std ** 3))
            kurtosis = float(np.mean(centered ** 4) / (std ** 4) - 3.0)

        return {
            'mean': mean,
            'std': std,
            'min': float(np.min(upper)),
            'max': float(np.max(upper)),
            'median': float(np.median(upper)),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n_pairs': len(upper)
        }

    @staticmethod
    def nearest_neighbor_diversity(embeddings: np.ndarray) -> Dict:
        """Diversity based on nearest neighbor distances.

        Args:
            embeddings: (n, d) matrix.

        Returns:
            Dict with mean, std, min NN distances and uniformity.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n = X.shape[0]
        if n <= 1:
            return {'mean_nn_dist': 0, 'std_nn_dist': 0,
                    'min_nn_dist': 0, 'uniformity': 0}

        sq = np.sum(X ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        np.fill_diagonal(D, np.inf)

        nn_dists = np.min(D, axis=1)

        mean_nn = float(np.mean(nn_dists))
        std_nn = float(np.std(nn_dists))

        # Uniformity: low coefficient of variation means uniform spacing
        cv = std_nn / max(mean_nn, 1e-12)
        uniformity = 1.0 / (1.0 + cv)

        return {
            'mean_nn_dist': mean_nn,
            'std_nn_dist': std_nn,
            'min_nn_dist': float(np.min(nn_dists)),
            'max_nn_dist': float(np.max(nn_dists)),
            'uniformity': float(uniformity)
        }


class DiversityComparator:
    """Compare diversity of multiple embedding sets."""

    def __init__(self):
        self.metric_computer = EmbeddingDiversity()

    def compare(self, embedding_sets: List[np.ndarray],
                names: Optional[List[str]] = None) -> Dict:
        """Compare diversity across multiple sets.

        Args:
            embedding_sets: List of (n_i, d) embedding matrices.
            names: Optional names for each set.

        Returns:
            Dict with per-set scores and rankings.
        """
        if names is None:
            names = [f"set_{i}" for i in range(len(embedding_sets))]

        results = {}
        for i, (emb, name) in enumerate(zip(embedding_sets, names)):
            scores = self.metric_computer.compute(emb)
            results[name] = scores._asdict()

        # Rank by each metric
        metrics = ['volume', 'spread', 'coverage', 'entropy',
                   'vendi_score', 'representation', 'projection', 'intrinsic_dim']
        rankings = {}
        for metric in metrics:
            vals = [(name, results[name][metric]) for name in names]
            vals.sort(key=lambda x: -x[1])
            rankings[metric] = [name for name, _ in vals]

        return {
            'scores': results,
            'rankings': rankings,
            'overall_winner': self._overall_winner(results, names, metrics)
        }

    @staticmethod
    def _overall_winner(results: Dict, names: List[str],
                        metrics: List[str]) -> str:
        """Determine overall winner by counting metric wins."""
        win_counts = {name: 0 for name in names}
        for metric in metrics:
            vals = [(name, results[name][metric]) for name in names]
            vals.sort(key=lambda x: -x[1])
            if vals:
                win_counts[vals[0][0]] += 1

        return max(win_counts, key=win_counts.get)


def demo_embedding_diversity():
    """Demonstrate embedding diversity metrics."""
    rng = np.random.RandomState(42)

    # High diversity: random embeddings
    high_div = rng.randn(50, 10)

    # Low diversity: clustered embeddings
    center = rng.randn(10)
    low_div = center + rng.randn(50, 10) * 0.1

    # Medium diversity
    med_div = rng.randn(50, 10) * 0.5

    computer = EmbeddingDiversity()

    print("=== High Diversity ===")
    scores_h = computer.compute(high_div)
    for field in scores_h._fields:
        print(f"  {field}: {getattr(scores_h, field):.4f}")

    print("\n=== Low Diversity ===")
    scores_l = computer.compute(low_div)
    for field in scores_l._fields:
        print(f"  {field}: {getattr(scores_l, field):.4f}")

    print("\n=== Medium Diversity ===")
    scores_m = computer.compute(med_div)
    for field in scores_m._fields:
        print(f"  {field}: {getattr(scores_m, field):.4f}")

    # Compare
    comparator = DiversityComparator()
    comparison = comparator.compare(
        [high_div, low_div, med_div],
        ['high', 'low', 'medium']
    )
    print(f"\nOverall winner: {comparison['overall_winner']}")

    # Pairwise analysis
    analyzer = PairwiseDiversityAnalyzer()
    dist_stats = analyzer.distance_distribution(high_div)
    print(f"\nHigh-div distance stats: mean={dist_stats['mean']:.3f}, "
          f"std={dist_stats['std']:.3f}")

    nn_stats = analyzer.nearest_neighbor_diversity(high_div)
    print(f"High-div NN diversity: mean_nn={nn_stats['mean_nn_dist']:.3f}, "
          f"uniformity={nn_stats['uniformity']:.3f}")


if __name__ == '__main__':
    demo_embedding_diversity()
