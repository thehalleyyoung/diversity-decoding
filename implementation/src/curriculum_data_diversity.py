"""
Diversity in training data selection: facility location, coreset construction,
stratified sampling, data deduplication, difficulty-diversity balance,
active data selection, data augmentation diversity.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.vq import kmeans2
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import math


@dataclass
class DataSubsetResult:
    """Result of diverse data subset selection."""
    indices: List[int]
    coverage: float = 0.0
    class_balance: Dict[int, float] = field(default_factory=dict)
    diversity_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeduplicationResult:
    """Result of data deduplication."""
    kept_indices: List[int]
    removed_indices: List[int]
    n_clusters: int = 0
    reduction_ratio: float = 0.0
    diversity_before: float = 0.0
    diversity_after: float = 0.0


class FacilityLocation:
    """Select centers that maximize coverage (facility location function)."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def select(self, data: np.ndarray, budget: int,
               weights: Optional[np.ndarray] = None) -> List[int]:
        """Greedy facility location: maximize sum_v max_{s in S} sim(v,s)."""
        n = len(data)
        if budget >= n:
            return list(range(n))

        dist_matrix = cdist(data, data, metric=self.metric)
        max_dist = np.max(dist_matrix) + 1e-10
        sim_matrix = max_dist - dist_matrix

        if weights is not None:
            sim_matrix = sim_matrix * weights.reshape(-1, 1)

        current_max_sim = np.zeros(n)
        selected: List[int] = []

        for _ in range(budget):
            marginal_gains = np.zeros(n)
            for i in range(n):
                if i in selected:
                    marginal_gains[i] = -np.inf
                    continue
                new_sims = np.maximum(current_max_sim, sim_matrix[:, i])
                marginal_gains[i] = np.sum(new_sims) - np.sum(current_max_sim)

            best = int(np.argmax(marginal_gains))
            selected.append(best)
            current_max_sim = np.maximum(current_max_sim, sim_matrix[:, best])

        return selected

    def coverage_score(self, data: np.ndarray, centers: List[int]) -> float:
        """Compute facility location objective value."""
        if not centers:
            return 0.0
        dist_matrix = cdist(data, data[centers], metric=self.metric)
        max_dist = np.max(dist_matrix) + 1e-10
        sim_matrix = max_dist - dist_matrix
        return float(np.sum(np.max(sim_matrix, axis=1)))

    def lazy_greedy_select(self, data: np.ndarray, budget: int) -> List[int]:
        """Lazy greedy selection exploiting submodularity."""
        n = len(data)
        if budget >= n:
            return list(range(n))

        dist_matrix = cdist(data, data, metric=self.metric)
        max_dist = np.max(dist_matrix) + 1e-10
        sim_matrix = max_dist - dist_matrix

        upper_bounds = np.full(n, np.inf)
        current_max_sim = np.zeros(n)
        selected: List[int] = []
        in_selected = np.zeros(n, dtype=bool)

        for _ in range(budget):
            while True:
                candidate = -1
                best_ub = -np.inf
                for i in range(n):
                    if in_selected[i]:
                        continue
                    if upper_bounds[i] > best_ub:
                        best_ub = upper_bounds[i]
                        candidate = i

                if candidate < 0:
                    break

                new_sims = np.maximum(current_max_sim, sim_matrix[:, candidate])
                actual_gain = np.sum(new_sims) - np.sum(current_max_sim)
                upper_bounds[candidate] = actual_gain

                is_best = True
                for i in range(n):
                    if not in_selected[i] and i != candidate:
                        if upper_bounds[i] > actual_gain + 1e-12:
                            is_best = False
                            break
                if is_best:
                    break

            if candidate >= 0:
                selected.append(candidate)
                in_selected[candidate] = True
                current_max_sim = np.maximum(current_max_sim, sim_matrix[:, candidate])

        return selected


class CoresetConstruction:
    """Select representative subset via k-center (coreset)."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def k_center_greedy(self, data: np.ndarray, k: int) -> List[int]:
        """Greedy k-center: minimize maximum distance from any point to nearest center."""
        n = len(data)
        if k >= n:
            return list(range(n))

        first = np.random.randint(n)
        centers = [first]

        dists = cdist(data, data[first:first + 1], metric=self.metric).flatten()

        for _ in range(k - 1):
            farthest = int(np.argmax(dists))
            centers.append(farthest)
            new_dists = cdist(data, data[farthest:farthest + 1], metric=self.metric).flatten()
            dists = np.minimum(dists, new_dists)

        return centers

    def k_center_radius(self, data: np.ndarray, centers: List[int]) -> float:
        """Maximum distance from any point to its nearest center."""
        if not centers:
            return float('inf')
        dists = cdist(data, data[centers], metric=self.metric)
        min_dists = np.min(dists, axis=1)
        return float(np.max(min_dists))

    def coreset_with_weights(self, data: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
        """Return coreset with importance weights (proportional to coverage)."""
        centers = self.k_center_greedy(data, k)
        dists = cdist(data, data[centers], metric=self.metric)
        assignments = np.argmin(dists, axis=1)

        weights = np.zeros(k)
        for i in range(len(data)):
            weights[assignments[i]] += 1.0
        weights = weights / weights.sum()

        return centers, weights

    def streaming_coreset(self, data: np.ndarray, k: int,
                          chunk_size: int = 100) -> List[int]:
        """Streaming coreset construction: process data in chunks."""
        n = len(data)
        if k >= n:
            return list(range(n))

        current_centers = self.k_center_greedy(data[:min(chunk_size, n)], min(k, chunk_size))

        for start in range(chunk_size, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = data[start:end]

            combined_indices = current_centers + list(range(start, end))
            combined_data = np.vstack([data[current_centers], chunk])

            if len(combined_data) > k:
                local_centers = self.k_center_greedy(combined_data, k)
                current_centers = [combined_indices[c] for c in local_centers]
            else:
                current_centers = combined_indices

        return current_centers[:k]


class StratifiedSampler:
    """Maintain class balance while maximizing intra-class diversity."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def stratified_diverse_sample(self, data: np.ndarray,
                                   labels: np.ndarray,
                                   budget: int) -> List[int]:
        """Stratified sampling with intra-class diversity via facility location."""
        classes = np.unique(labels)
        n_classes = len(classes)
        per_class_budget = budget // n_classes
        remainder = budget - per_class_budget * n_classes

        selected: List[int] = []
        fl = FacilityLocation(self.metric)

        for c_idx, c in enumerate(classes):
            class_mask = labels == c
            class_indices = np.where(class_mask)[0]
            class_data = data[class_indices]

            class_budget = per_class_budget + (1 if c_idx < remainder else 0)
            class_budget = min(class_budget, len(class_indices))

            if class_budget <= 0:
                continue

            local_selected = fl.select(class_data, class_budget)
            selected.extend(class_indices[local_selected].tolist())

        return selected

    def class_balance_score(self, labels: np.ndarray,
                            selected_indices: List[int]) -> Dict[int, float]:
        """Compute class balance of selected subset."""
        selected_labels = labels[selected_indices]
        classes, counts = np.unique(selected_labels, return_counts=True)
        total = len(selected_indices)
        return {int(c): float(count / total) for c, count in zip(classes, counts)}

    def target_distribution_sample(self, data: np.ndarray,
                                    labels: np.ndarray,
                                    budget: int,
                                    target_dist: Dict[int, float]) -> List[int]:
        """Sample to match a target class distribution."""
        selected: List[int] = []
        fl = FacilityLocation(self.metric)

        for cls, fraction in target_dist.items():
            class_mask = labels == cls
            class_indices = np.where(class_mask)[0]
            class_data = data[class_indices]

            class_budget = max(1, int(budget * fraction))
            class_budget = min(class_budget, len(class_indices))

            if class_budget <= 0:
                continue

            local_selected = fl.select(class_data, class_budget)
            selected.extend(class_indices[local_selected].tolist())

        return selected


class DataDeduplicator:
    """Remove near-duplicates, keep diverse instances."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def deduplicate_threshold(self, data: np.ndarray,
                               threshold: float) -> DeduplicationResult:
        """Remove items within threshold distance of already-kept items."""
        n = len(data)
        kept: List[int] = []
        removed: List[int] = []
        kept_data: List[np.ndarray] = []

        for i in range(n):
            if not kept_data:
                kept.append(i)
                kept_data.append(data[i])
                continue

            dists = cdist(data[i:i + 1], np.array(kept_data), metric=self.metric).flatten()
            if np.min(dists) > threshold:
                kept.append(i)
                kept_data.append(data[i])
            else:
                removed.append(i)

        div_before = float(np.mean(pdist(data))) if n > 1 else 0.0
        div_after = float(np.mean(pdist(np.array(kept_data)))) if len(kept_data) > 1 else 0.0

        return DeduplicationResult(
            kept_indices=kept,
            removed_indices=removed,
            reduction_ratio=len(removed) / n,
            diversity_before=div_before,
            diversity_after=div_after
        )

    def deduplicate_clustering(self, data: np.ndarray,
                                n_clusters: int) -> DeduplicationResult:
        """Cluster and keep one representative per cluster (closest to centroid)."""
        n = len(data)
        if n_clusters >= n:
            return DeduplicationResult(
                kept_indices=list(range(n)),
                removed_indices=[],
                n_clusters=n
            )

        centroids, labels = kmeans2(data.astype(float), n_clusters,
                                    minit='points', seed=42)

        kept: List[int] = []
        for c in range(n_clusters):
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_data = data[cluster_indices]
            dists_to_centroid = cdist(cluster_data, centroids[c:c + 1],
                                     metric=self.metric).flatten()
            best_local = int(np.argmin(dists_to_centroid))
            kept.append(int(cluster_indices[best_local]))

        removed = [i for i in range(n) if i not in kept]

        div_before = float(np.mean(pdist(data))) if n > 1 else 0.0
        div_after = float(np.mean(pdist(data[kept]))) if len(kept) > 1 else 0.0

        return DeduplicationResult(
            kept_indices=kept,
            removed_indices=removed,
            n_clusters=n_clusters,
            reduction_ratio=len(removed) / n,
            diversity_before=div_before,
            diversity_after=div_after
        )

    def minhash_dedup(self, data: np.ndarray, n_hashes: int = 50,
                      threshold: float = 0.5, seed: int = 42) -> DeduplicationResult:
        """MinHash-based approximate deduplication."""
        rng = np.random.RandomState(seed)
        n, d = data.shape

        random_vectors = rng.randn(n_hashes, d)
        signatures = np.sign(data @ random_vectors.T)

        kept: List[int] = [0]
        removed: List[int] = []

        for i in range(1, n):
            is_duplicate = False
            for j in kept:
                agreement = np.mean(signatures[i] == signatures[j])
                if agreement > threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                removed.append(i)
            else:
                kept.append(i)

        return DeduplicationResult(
            kept_indices=kept,
            removed_indices=removed,
            reduction_ratio=len(removed) / n
        )


class DifficultyDiversityBalance:
    """Balance easy/hard distribution AND topical diversity."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def estimate_difficulty(self, data: np.ndarray, labels: np.ndarray,
                            method: str = 'knn') -> np.ndarray:
        """Estimate sample difficulty using k-NN disagreement."""
        n = len(data)
        k = min(10, n - 1)
        dists = squareform(pdist(data))
        np.fill_diagonal(dists, np.inf)

        difficulties = np.zeros(n)
        for i in range(n):
            neighbors = np.argsort(dists[i])[:k]
            neighbor_labels = labels[neighbors]
            disagreement = np.mean(neighbor_labels != labels[i])
            difficulties[i] = disagreement

        return difficulties

    def balanced_select(self, data: np.ndarray, labels: np.ndarray,
                        budget: int, difficulty_bins: int = 3,
                        diversity_weight: float = 0.5) -> List[int]:
        """Select balanced mix of easy/medium/hard examples with diversity."""
        difficulties = self.estimate_difficulty(data, labels)

        bin_edges = np.percentile(difficulties, np.linspace(0, 100, difficulty_bins + 1))
        bin_assignments = np.digitize(difficulties, bin_edges[1:-1])

        per_bin = budget // difficulty_bins
        remainder = budget - per_bin * difficulty_bins

        selected: List[int] = []
        fl = FacilityLocation()

        for b in range(difficulty_bins):
            bin_mask = bin_assignments == b
            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) == 0:
                continue

            bin_data = data[bin_indices]
            bin_budget = per_bin + (1 if b < remainder else 0)
            bin_budget = min(bin_budget, len(bin_indices))

            local = fl.select(bin_data, bin_budget)
            selected.extend(bin_indices[local].tolist())

        return selected


class ActiveDataSelector:
    """Select unlabeled examples that would most diversify training set."""

    def __init__(self, metric: str = 'euclidean', seed: int = 42):
        self.metric = metric
        self.rng = np.random.RandomState(seed)

    def select_most_diverse(self, labeled_data: np.ndarray,
                             unlabeled_data: np.ndarray,
                             budget: int) -> List[int]:
        """Select unlabeled examples that are most different from labeled set."""
        dists = cdist(unlabeled_data, labeled_data, metric=self.metric)
        min_dists = np.min(dists, axis=1)

        sorted_indices = np.argsort(min_dists)[::-1]
        return sorted_indices[:budget].tolist()

    def select_coverage_maximizing(self, labeled_data: np.ndarray,
                                    unlabeled_data: np.ndarray,
                                    budget: int,
                                    radius: float = 1.0) -> List[int]:
        """Select unlabeled examples that maximize coverage of feature space."""
        all_data = np.vstack([labeled_data, unlabeled_data])
        n_labeled = len(labeled_data)

        covered = np.zeros(len(all_data), dtype=bool)
        labeled_dists = cdist(all_data, labeled_data, metric=self.metric)
        covered = np.any(labeled_dists <= radius, axis=1)

        selected: List[int] = []

        for _ in range(budget):
            best_gain = -1
            best_idx = -1

            for i in range(len(unlabeled_data)):
                if i in selected:
                    continue
                candidate = unlabeled_data[i]
                dists = cdist(candidate.reshape(1, -1), all_data, metric=self.metric).flatten()
                new_covered = (~covered) & (dists <= radius)
                gain = np.sum(new_covered)

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
                candidate = unlabeled_data[best_idx]
                dists = cdist(candidate.reshape(1, -1), all_data, metric=self.metric).flatten()
                covered = covered | (dists <= radius)

        return selected

    def select_uncertainty_diverse(self, labeled_data: np.ndarray,
                                    labeled_labels: np.ndarray,
                                    unlabeled_data: np.ndarray,
                                    budget: int) -> List[int]:
        """Select by combining uncertainty (k-NN disagreement) and diversity."""
        k = min(5, len(labeled_data) - 1)
        dists = cdist(unlabeled_data, labeled_data, metric=self.metric)

        uncertainty = np.zeros(len(unlabeled_data))
        for i in range(len(unlabeled_data)):
            nn_indices = np.argsort(dists[i])[:k]
            nn_labels = labeled_labels[nn_indices]
            _, counts = np.unique(nn_labels, return_counts=True)
            probs = counts / counts.sum()
            uncertainty[i] = -np.sum(probs * np.log2(probs + 1e-15))

        diversity = np.min(dists, axis=1)

        u_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-15)
        d_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min() + 1e-15)

        scores = 0.5 * u_norm + 0.5 * d_norm
        return list(np.argsort(scores)[-budget:][::-1])


class AugmentationDiversity:
    """Assess which augmentations add genuinely diverse examples."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def augment_noise(self, data: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Add Gaussian noise."""
        return data + self.rng.randn(*data.shape) * scale

    def augment_dropout(self, data: np.ndarray, rate: float = 0.1) -> np.ndarray:
        """Random feature dropout."""
        mask = self.rng.binomial(1, 1 - rate, size=data.shape)
        return data * mask

    def augment_mixup(self, data: np.ndarray, labels: np.ndarray,
                      alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup augmentation."""
        n = len(data)
        indices = self.rng.permutation(n)
        lam = self.rng.beta(alpha, alpha, size=n)
        lam = np.maximum(lam, 1 - lam)

        mixed_data = data * lam.reshape(-1, 1) + data[indices] * (1 - lam.reshape(-1, 1))
        return mixed_data, labels

    def diversity_gain(self, original_data: np.ndarray,
                        augmented_data: np.ndarray) -> float:
        """How much diversity does augmentation add?"""
        orig_div = float(np.mean(pdist(original_data))) if len(original_data) > 1 else 0.0
        combined = np.vstack([original_data, augmented_data])
        combined_div = float(np.mean(pdist(combined)))
        return combined_div - orig_div

    def select_diverse_augmentations(self, data: np.ndarray,
                                      labels: np.ndarray,
                                      n_augmentations: int = 3) -> Dict[str, Any]:
        """Compare augmentation strategies by diversity gain."""
        results = {}

        noise_aug = self.augment_noise(data, scale=0.1)
        results['noise_0.1'] = {
            'diversity_gain': self.diversity_gain(data, noise_aug),
            'size': len(noise_aug)
        }

        noise_aug2 = self.augment_noise(data, scale=0.5)
        results['noise_0.5'] = {
            'diversity_gain': self.diversity_gain(data, noise_aug2),
            'size': len(noise_aug2)
        }

        dropout_aug = self.augment_dropout(data, rate=0.2)
        results['dropout_0.2'] = {
            'diversity_gain': self.diversity_gain(data, dropout_aug),
            'size': len(dropout_aug)
        }

        mixup_aug, _ = self.augment_mixup(data, labels)
        results['mixup'] = {
            'diversity_gain': self.diversity_gain(data, mixup_aug),
            'size': len(mixup_aug)
        }

        best_method = max(results.keys(), key=lambda k: results[k]['diversity_gain'])
        results['best_method'] = best_method

        return results


class CurriculumDiversity:
    """Main class: diverse training data selection."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.facility = FacilityLocation()
        self.coreset = CoresetConstruction()
        self.stratified = StratifiedSampler()
        self.deduplicator = DataDeduplicator()
        self.difficulty_balance = DifficultyDiversityBalance(seed)
        self.active_selector = ActiveDataSelector(seed=seed)
        self.augmentation = AugmentationDiversity(seed)

    def select(self, dataset: np.ndarray, budget: int,
               diversity_weight: float = 0.5,
               labels: Optional[np.ndarray] = None,
               method: str = 'facility_location') -> DataSubsetResult:
        """Select diverse training subset."""
        if method == 'facility_location':
            indices = self.facility.select(dataset, budget)
        elif method == 'coreset':
            indices = self.coreset.k_center_greedy(dataset, budget)
        elif method == 'stratified' and labels is not None:
            indices = self.stratified.stratified_diverse_sample(dataset, labels, budget)
        elif method == 'difficulty_balanced' and labels is not None:
            indices = self.difficulty_balance.balanced_select(
                dataset, labels, budget, diversity_weight=diversity_weight
            )
        else:
            indices = self.facility.select(dataset, budget)

        coverage = self.facility.coverage_score(dataset, indices)

        selected_data = dataset[indices]
        div = float(np.mean(pdist(selected_data))) if len(indices) > 1 else 0.0

        class_bal = {}
        if labels is not None:
            class_bal = self.stratified.class_balance_score(labels, indices)

        return DataSubsetResult(
            indices=indices,
            coverage=coverage,
            class_balance=class_bal,
            diversity_score=div,
            details={'method': method, 'budget': budget}
        )
