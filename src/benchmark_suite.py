"""
Standardized benchmarks for diversity methods.

Implements synthetic datasets, evaluation metrics, statistical comparison,
scaling benchmarks, robustness benchmarks, and visualization data generation.
"""

import numpy as np
import time
from typing import Optional, List, Dict, Tuple, Callable, Any, NamedTuple


# ======================================================================
# Synthetic Dataset Generators
# ======================================================================

class SyntheticDatasets:
    """Generate synthetic datasets for benchmarking diversity methods."""

    @staticmethod
    def gaussian_clusters(n: int = 200, d: int = 10, n_clusters: int = 5,
                          cluster_std: float = 0.5,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Gaussian cluster data.

        Args:
            n: Total number of items.
            d: Dimensionality.
            n_clusters: Number of clusters.
            cluster_std: Standard deviation within clusters.
            random_state: Random seed.

        Returns:
            (items, labels) where items is (n, d) and labels is (n,).
        """
        rng = np.random.RandomState(random_state)
        centers = rng.randn(n_clusters, d) * 5.0
        items_per_cluster = n // n_clusters
        remainder = n % n_clusters

        items = []
        labels = []
        for c in range(n_clusters):
            nc = items_per_cluster + (1 if c < remainder else 0)
            cluster_items = centers[c] + rng.randn(nc, d) * cluster_std
            items.append(cluster_items)
            labels.extend([c] * nc)

        return np.vstack(items), np.array(labels)

    @staticmethod
    def manifold(n: int = 200, ambient_dim: int = 10,
                 intrinsic_dim: int = 2, noise: float = 0.01,
                 random_state: int = 42) -> np.ndarray:
        """Generate data on a low-dimensional manifold in high-dim space.

        Args:
            n: Number of items.
            ambient_dim: Ambient dimensionality.
            intrinsic_dim: Intrinsic dimensionality.
            noise: Noise level.
            random_state: Random seed.

        Returns:
            (n, ambient_dim) data matrix.
        """
        rng = np.random.RandomState(random_state)
        # Generate on low-dim manifold
        Z = rng.randn(n, intrinsic_dim)
        # Random projection to high dim
        A = rng.randn(intrinsic_dim, ambient_dim)
        X = Z @ A + noise * rng.randn(n, ambient_dim)
        return X

    @staticmethod
    def text_like(n: int = 100, vocab_size: int = 500,
                  doc_length: int = 50, n_topics: int = 5,
                  random_state: int = 42) -> np.ndarray:
        """Generate text-like TF-IDF vectors.

        Uses a latent topic model to generate realistic TF-IDF-like
        sparse vectors.

        Args:
            n: Number of documents.
            vocab_size: Vocabulary size.
            doc_length: Average document length.
            n_topics: Number of latent topics.
            random_state: Random seed.

        Returns:
            (n, vocab_size) TF-IDF-like matrix.
        """
        rng = np.random.RandomState(random_state)

        # Topic-word distribution (Dirichlet)
        topic_word = rng.dirichlet(np.ones(vocab_size) * 0.1, size=n_topics)

        # Document-topic distribution
        doc_topic = rng.dirichlet(np.ones(n_topics) * 0.5, size=n)

        # Generate TF-IDF-like vectors
        tfidf = np.zeros((n, vocab_size))
        for i in range(n):
            word_dist = doc_topic[i] @ topic_word
            # Sample word counts
            word_counts = rng.multinomial(doc_length, word_dist)
            tf = word_counts / max(doc_length, 1)
            tfidf[i] = tf

        # Apply IDF
        df = np.sum(tfidf > 0, axis=0)
        idf = np.log((n + 1) / (df + 1)) + 1
        tfidf *= idf

        return tfidf

    @staticmethod
    def uniform(n: int = 200, d: int = 10,
                random_state: int = 42) -> np.ndarray:
        """Generate uniformly distributed data.

        Args:
            n: Number of items.
            d: Dimensionality.
            random_state: Random seed.

        Returns:
            (n, d) uniform data.
        """
        rng = np.random.RandomState(random_state)
        return rng.rand(n, d)

    @staticmethod
    def controlled_diversity(n: int = 50, d: int = 10,
                              diversity_level: float = 1.0,
                              random_state: int = 42) -> np.ndarray:
        """Generate data with controlled diversity level.

        Args:
            n: Number of items.
            d: Dimensionality.
            diversity_level: 0.0 = identical, 1.0 = fully random.
            random_state: Random seed.

        Returns:
            (n, d) data.
        """
        rng = np.random.RandomState(random_state)
        base = rng.randn(1, d)
        noise = rng.randn(n, d) * diversity_level
        return base + noise


# ======================================================================
# Evaluation Metrics
# ======================================================================

class DiversityMetrics:
    """Standard evaluation metrics for diversity."""

    @staticmethod
    def coverage(items: np.ndarray, selected: np.ndarray,
                 radius: Optional[float] = None) -> float:
        """Coverage: fraction of items within radius of selected."""
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        selected = np.asarray(selected, dtype=np.intp)

        if len(selected) == 0:
            return 0.0

        sel_items = items[selected]
        # Distances from each item to nearest selected item
        dists = np.zeros(n)
        for i in range(n):
            dists[i] = np.min(np.sqrt(np.sum((sel_items - items[i]) ** 2, axis=1)))

        if radius is None:
            radius = np.median(dists)

        return float(np.mean(dists <= radius))

    @staticmethod
    def spread(items: np.ndarray, selected: np.ndarray) -> float:
        """Spread: mean pairwise distance among selected items."""
        items = np.asarray(items, dtype=np.float64)
        sel_items = items[selected]
        n = len(sel_items)
        if n <= 1:
            return 0.0

        sq = np.sum(sel_items ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * sel_items @ sel_items.T, 0))
        upper = D[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    @staticmethod
    def volume(items: np.ndarray, selected: np.ndarray) -> float:
        """Volume: log-det of Gram matrix of selected items."""
        items = np.asarray(items, dtype=np.float64)
        sel_items = items[selected]
        n = len(sel_items)
        if n == 0:
            return 0.0

        G = sel_items @ sel_items.T + 1e-6 * np.eye(n)
        sign, logdet = np.linalg.slogdet(G)
        return float(logdet) if sign > 0 else -np.inf

    @staticmethod
    def quality_diversity_score(items: np.ndarray, selected: np.ndarray,
                                 quality_scores: np.ndarray) -> float:
        """Combined quality-diversity score."""
        quality = float(np.mean(quality_scores[selected]))
        diversity = DiversityMetrics.spread(items, selected)
        return quality * diversity

    @staticmethod
    def min_pairwise_distance(items: np.ndarray, selected: np.ndarray) -> float:
        """Minimum pairwise distance among selected items."""
        sel_items = items[selected]
        n = len(sel_items)
        if n <= 1:
            return 0.0

        sq = np.sum(sel_items ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * sel_items @ sel_items.T, 0))
        np.fill_diagonal(D, np.inf)
        return float(np.min(D))


# ======================================================================
# Statistical Comparison
# ======================================================================

class StatisticalComparison:
    """Statistical tests for comparing diversity methods."""

    @staticmethod
    def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray
                      ) -> Tuple[float, float]:
        """Paired t-test.

        Args:
            scores_a: Scores from method A across trials.
            scores_b: Scores from method B across trials.

        Returns:
            (t_statistic, p_value)
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        diffs = scores_a - scores_b
        n = len(diffs)
        if n <= 1:
            return 0.0, 1.0

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        if std_diff < 1e-12:
            return float('inf') if mean_diff > 0 else float('-inf'), 0.0

        t_stat = mean_diff / (std_diff / np.sqrt(n))

        # Approximate p-value using normal distribution (for large n)
        # For small n, this is an approximation
        p_value = 2.0 * _normal_sf(abs(t_stat))

        return float(t_stat), float(p_value)

    @staticmethod
    def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray
                      ) -> Tuple[float, float]:
        """Wilcoxon signed-rank test (approximate).

        Non-parametric alternative to paired t-test.

        Args:
            scores_a: Scores from method A.
            scores_b: Scores from method B.

        Returns:
            (W_statistic, approximate_p_value)
        """
        diffs = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
        diffs = diffs[np.abs(diffs) > 1e-12]
        n = len(diffs)
        if n == 0:
            return 0.0, 1.0

        abs_diffs = np.abs(diffs)
        ranks = _rank_data(abs_diffs)
        signs = np.sign(diffs)

        W_plus = np.sum(ranks[signs > 0])
        W_minus = np.sum(ranks[signs < 0])
        W = min(W_plus, W_minus)

        # Normal approximation
        mean_W = n * (n + 1) / 4.0
        std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if std_W < 1e-12:
            return float(W), 1.0

        z = (W - mean_W) / std_W
        p_value = 2.0 * _normal_sf(abs(z))

        return float(W), float(p_value)

    @staticmethod
    def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Cohen's d effect size.

        Args:
            scores_a: Scores from method A.
            scores_b: Scores from method B.

        Returns:
            Cohen's d.
        """
        a = np.asarray(scores_a, dtype=np.float64)
        b = np.asarray(scores_b, dtype=np.float64)
        n_a, n_b = len(a), len(b)
        if n_a + n_b <= 2:
            return 0.0

        pooled_std = np.sqrt(
            ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
            / (n_a + n_b - 2)
        )
        if pooled_std < 1e-12:
            return 0.0

        return float((np.mean(a) - np.mean(b)) / pooled_std)


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) of standard normal, approximate."""
    # Abramowitz and Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * np.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    )
    if x >= 0:
        return p
    else:
        return 1.0 - p


def _rank_data(data: np.ndarray) -> np.ndarray:
    """Rank data with average ties."""
    n = len(data)
    order = np.argsort(data)
    ranks = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and np.abs(data[order[j + 1]] - data[order[j]]) < 1e-12:
            j += 1
        avg_rank = np.mean(np.arange(i + 1, j + 2, dtype=np.float64))
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


# ======================================================================
# BenchmarkReport
# ======================================================================

class BenchmarkReport:
    """Container for benchmark results."""

    def __init__(self):
        self.per_method_scores: Dict[str, Dict[str, float]] = {}
        self.statistical_comparisons: Dict[str, Dict] = {}
        self.scaling_results: Dict[str, List[Dict]] = {}
        self.robustness_results: Dict[str, Dict] = {}
        self.winner: str = ""

    def to_dict(self) -> Dict:
        return {
            'per_method_scores': self.per_method_scores,
            'statistical_comparisons': self.statistical_comparisons,
            'scaling_results': self.scaling_results,
            'robustness_results': self.robustness_results,
            'winner': self.winner
        }

    def summary(self) -> str:
        lines = ["=== Benchmark Report ==="]
        lines.append(f"Winner: {self.winner}")
        lines.append("")
        for method, scores in self.per_method_scores.items():
            lines.append(f"{method}:")
            for metric, val in scores.items():
                lines.append(f"  {metric}: {val:.4f}")
        return "\n".join(lines)


# ======================================================================
# DiversityBenchmark
# ======================================================================

class DiversityBenchmark:
    """Run standardized benchmarks comparing diversity methods."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
        self.metrics = DiversityMetrics()
        self.stats = StatisticalComparison()

    def run(self, methods: Dict[str, Callable],
            k: int = 10,
            datasets: Optional[Dict[str, np.ndarray]] = None,
            n_trials: int = 5) -> BenchmarkReport:
        """Run full benchmark suite.

        Args:
            methods: Dict mapping method_name -> function(items, k) -> selected_indices.
            k: Number of items to select.
            datasets: Dict of dataset_name -> (n, d) arrays. Auto-generated if None.
            n_trials: Number of random trials per dataset.

        Returns:
            BenchmarkReport.
        """
        if datasets is None:
            datasets = self._default_datasets()

        report = BenchmarkReport()

        # Collect scores per method
        method_scores = {name: {m: [] for m in ['spread', 'volume', 'coverage', 'runtime']}
                        for name in methods}

        for ds_name, items in datasets.items():
            for trial in range(n_trials):
                for method_name, method_fn in methods.items():
                    start = time.time()
                    try:
                        selected = method_fn(items, k)
                        selected = np.asarray(selected, dtype=np.intp)
                        runtime = time.time() - start

                        spread = self.metrics.spread(items, selected)
                        volume = self.metrics.volume(items, selected)
                        coverage = self.metrics.coverage(items, selected)

                        method_scores[method_name]['spread'].append(spread)
                        method_scores[method_name]['volume'].append(volume)
                        method_scores[method_name]['coverage'].append(coverage)
                        method_scores[method_name]['runtime'].append(runtime)
                    except Exception as e:
                        method_scores[method_name]['spread'].append(0.0)
                        method_scores[method_name]['volume'].append(0.0)
                        method_scores[method_name]['coverage'].append(0.0)
                        method_scores[method_name]['runtime'].append(float('inf'))

        # Aggregate
        for name in methods:
            report.per_method_scores[name] = {
                metric: float(np.mean(vals)) if vals else 0.0
                for metric, vals in method_scores[name].items()
            }

        # Statistical comparisons (all pairs)
        method_names = list(methods.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                key = f"{m1}_vs_{m2}"
                s1 = np.array(method_scores[m1]['spread'])
                s2 = np.array(method_scores[m2]['spread'])
                if len(s1) > 1 and len(s2) > 1 and len(s1) == len(s2):
                    t_stat, p_val = self.stats.paired_t_test(s1, s2)
                    d_effect = self.stats.cohens_d(s1, s2)
                    report.statistical_comparisons[key] = {
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'cohens_d': d_effect,
                        'significant': p_val < 0.05
                    }

        # Determine winner (highest average spread)
        if report.per_method_scores:
            report.winner = max(
                report.per_method_scores,
                key=lambda m: report.per_method_scores[m].get('spread', 0)
            )

        return report

    def scaling_benchmark(self, method_fn: Callable,
                          method_name: str,
                          sizes: Optional[List[int]] = None,
                          k: int = 10, d: int = 10
                          ) -> List[Dict]:
        """Measure runtime scaling with dataset size.

        Args:
            method_fn: Function(items, k) -> selected.
            method_name: Name for reporting.
            sizes: List of dataset sizes to test.
            k: Selection size.
            d: Dimensionality.

        Returns:
            List of dicts with 'n', 'k', 'd', 'runtime'.
        """
        if sizes is None:
            sizes = [50, 100, 200, 500]

        results = []
        for n in sizes:
            items = self.rng.randn(n, d)
            actual_k = min(k, n)

            start = time.time()
            try:
                method_fn(items, actual_k)
                runtime = time.time() - start
            except Exception:
                runtime = float('inf')

            results.append({
                'n': n,
                'k': actual_k,
                'd': d,
                'runtime': runtime,
                'method': method_name
            })

        return results

    def robustness_benchmark(self, method_fn: Callable,
                             method_name: str,
                             n: int = 100, d: int = 10, k: int = 10,
                             noise_levels: Optional[List[float]] = None
                             ) -> Dict:
        """Measure robustness to noise.

        Args:
            method_fn: Function(items, k) -> selected.
            method_name: Name.
            n: Number of items.
            d: Dimensionality.
            k: Selection size.
            noise_levels: List of noise magnitudes.

        Returns:
            Dict with noise levels and corresponding diversity scores.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]

        base_items = self.rng.randn(n, d)

        results = {'noise_levels': noise_levels, 'spreads': [], 'volumes': []}

        for noise in noise_levels:
            noisy_items = base_items + self.rng.randn(n, d) * noise
            try:
                selected = np.asarray(method_fn(noisy_items, k), dtype=np.intp)
                spread = self.metrics.spread(noisy_items, selected)
                volume = self.metrics.volume(noisy_items, selected)
            except Exception:
                spread = 0.0
                volume = -np.inf

            results['spreads'].append(spread)
            results['volumes'].append(volume)

        return results

    def _default_datasets(self) -> Dict[str, np.ndarray]:
        """Generate default benchmark datasets."""
        gen = SyntheticDatasets()
        items_clusters, _ = gen.gaussian_clusters(n=100, d=10)
        items_manifold = gen.manifold(n=100, ambient_dim=10)
        items_uniform = gen.uniform(n=100, d=10)
        return {
            'gaussian_clusters': items_clusters,
            'manifold': items_manifold,
            'uniform': items_uniform,
        }


# ======================================================================
# Visualization data generators
# ======================================================================

class VisualizationData:
    """Generate data structures for plotting (without matplotlib dependency)."""

    @staticmethod
    def quality_diversity_scatter(methods_results: Dict[str, Dict]
                                   ) -> Dict:
        """Generate data for quality-diversity scatter plot.

        Args:
            methods_results: Dict of method_name -> {'quality': float, 'diversity': float}.

        Returns:
            Dict with x, y, labels for plotting.
        """
        x = []
        y = []
        labels = []
        for name, res in methods_results.items():
            x.append(res.get('quality', 0.0))
            y.append(res.get('diversity', 0.0))
            labels.append(name)
        return {'x': x, 'y': y, 'labels': labels,
                'xlabel': 'Quality', 'ylabel': 'Diversity'}

    @staticmethod
    def method_comparison_heatmap(report: BenchmarkReport) -> Dict:
        """Generate data for method comparison heatmap.

        Args:
            report: BenchmarkReport with per_method_scores.

        Returns:
            Dict with matrix, row_labels, col_labels for heatmap.
        """
        methods = list(report.per_method_scores.keys())
        metrics = ['spread', 'volume', 'coverage', 'runtime']

        matrix = []
        for method in methods:
            row = []
            for metric in metrics:
                val = report.per_method_scores[method].get(metric, 0.0)
                row.append(val)
            matrix.append(row)

        return {
            'matrix': matrix,
            'row_labels': methods,
            'col_labels': metrics
        }

    @staticmethod
    def scaling_plot_data(scaling_results: List[Dict]) -> Dict:
        """Generate data for scaling plot.

        Args:
            scaling_results: List of dicts from scaling_benchmark.

        Returns:
            Dict with x (sizes), y (runtimes) for line plot.
        """
        sizes = [r['n'] for r in scaling_results]
        runtimes = [r['runtime'] for r in scaling_results]
        return {'x': sizes, 'y': runtimes,
                'xlabel': 'Dataset Size', 'ylabel': 'Runtime (s)'}


def demo_benchmark():
    """Demonstrate benchmark suite."""
    rng = np.random.RandomState(42)

    # Define simple methods
    def random_select(items, k):
        return rng.choice(len(items), size=min(k, len(items)), replace=False)

    def farthest_point(items, k):
        n = len(items)
        k = min(k, n)
        selected = [rng.randint(n)]
        for _ in range(1, k):
            sq = np.sum(items ** 2, axis=1)
            D = np.sqrt(np.maximum(
                sq[:, None] + sq[None, :] - 2 * items @ items.T, 0))
            sel_arr = np.array(selected)
            min_dists = D[:, sel_arr].min(axis=1)
            min_dists[selected] = -1
            selected.append(int(np.argmax(min_dists)))
        return np.array(selected)

    methods = {
        'random': random_select,
        'farthest_point': farthest_point,
    }

    bench = DiversityBenchmark()
    report = bench.run(methods, k=10, n_trials=3)
    print(report.summary())

    # Scaling
    scaling = bench.scaling_benchmark(farthest_point, 'farthest_point',
                                       sizes=[50, 100, 200])
    print(f"\nScaling results: {scaling}")

    # Robustness
    robust = bench.robustness_benchmark(farthest_point, 'farthest_point')
    print(f"Robustness spreads: {robust['spreads']}")


if __name__ == '__main__':
    demo_benchmark()
