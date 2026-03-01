"""
Unified DiversitySelector interface.

Every diversity selection method in DivFlow implements this common interface
so users can swap algorithms with a single line change.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class DiversitySelector(ABC):
    """Common interface for all diversity selection algorithms."""

    @abstractmethod
    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        """Select k diverse items from candidates.

        Args:
            candidates: (n, d) feature matrix.
            k: number of items to select.

        Returns:
            (indices, metadata) where indices are the selected item indices
            and metadata contains algorithm-specific information.
        """
        raise NotImplementedError

    def spread(self, candidates: np.ndarray, indices: List[int]) -> float:
        """Compute spread (min pairwise distance) of selected subset."""
        if len(indices) < 2:
            return 0.0
        subset = candidates[indices]
        dists = np.linalg.norm(subset[:, None] - subset[None, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        return float(np.min(dists))

    def sum_distance(self, candidates: np.ndarray, indices: List[int]) -> float:
        """Compute sum of pairwise distances."""
        if len(indices) < 2:
            return 0.0
        subset = candidates[indices]
        dists = np.linalg.norm(subset[:, None] - subset[None, :], axis=-1)
        return float(np.sum(dists) / 2.0)


class DPPSelector(DiversitySelector):
    """DPP-based diversity selection."""

    def __init__(self, kernel: str = 'rbf', gamma: Optional[float] = None):
        self.kernel = kernel
        self.gamma = gamma

    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        from dpp_sampler import compute_kernel, DPPSampler
        K = compute_kernel(candidates, kernel=self.kernel, gamma=self.gamma)
        sampler = DPPSampler()
        sampler.fit(K)
        indices = sampler.sample_k(k)
        return list(indices), {'kernel': self.kernel, 'method': 'k-DPP'}


class MMRSelector(DiversitySelector):
    """Maximal Marginal Relevance selection."""

    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param

    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        from mmr_selector import MMRSelector as _MMR
        query = kwargs.get('query', np.mean(candidates, axis=0))
        mmr = _MMR()
        indices = mmr.select(items=candidates, query=query, k=k,
                             lambda_param=self.lambda_param)
        return list(indices), {'lambda': self.lambda_param, 'method': 'MMR'}


class SubmodularSelector(DiversitySelector):
    """Submodular optimization for diversity (sum-pairwise-distance objective)."""

    def __init__(self, method: str = 'greedy'):
        self.method = method

    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        from submodular_optimizer import SumPairwiseDistanceFunction, SubmodularOptimizer
        dist_matrix = np.linalg.norm(candidates[:, None] - candidates[None, :], axis=-1)
        f = SumPairwiseDistanceFunction(dist_matrix)
        opt = SubmodularOptimizer()
        if self.method == 'greedy':
            indices, val = opt.greedy(f, k)
        elif self.method == 'stochastic':
            indices, val = opt.stochastic_greedy(f, k)
        else:
            indices, val = opt.greedy(f, k)
        return indices, {'objective_value': val, 'method': f'submodular-{self.method}'}


class ClusteringSelector(DiversitySelector):
    """Cluster-based diversity: pick one representative per cluster."""

    def __init__(self, method: str = 'kmedoids'):
        self.method = method

    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        from clustering_diversity import ClusterDiversity
        cd = ClusterDiversity(method=self.method)
        indices = cd.select(candidates, k)
        return list(indices), {'method': f'clustering-{self.method}'}


class FarthestPointSelector(DiversitySelector):
    """Greedy farthest-point selection (maximin)."""

    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        n = len(candidates)
        k = min(k, n)
        rng = kwargs.get('rng', np.random.RandomState(42))
        selected = [int(rng.randint(n))]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            last = selected[-1]
            new_dists = np.linalg.norm(candidates - candidates[last], axis=1)
            dists = np.minimum(dists, new_dists)
            dists[selected] = -1
            next_idx = int(np.argmax(dists))
            selected.append(next_idx)
        return selected, {'method': 'farthest-point'}


class RandomSelector(DiversitySelector):
    """Random baseline."""

    def select(self, candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
        rng = kwargs.get('rng', np.random.RandomState(42))
        n = len(candidates)
        indices = rng.choice(n, size=min(k, n), replace=False).tolist()
        return indices, {'method': 'random'}


# Registry
SELECTORS = {
    'dpp': DPPSelector,
    'mmr': MMRSelector,
    'submodular': SubmodularSelector,
    'clustering': ClusteringSelector,
    'farthest_point': FarthestPointSelector,
    'random': RandomSelector,
}


def get_selector(name: str, **kwargs) -> DiversitySelector:
    """Factory function to get a selector by name."""
    if name not in SELECTORS:
        raise ValueError(f"Unknown selector: {name}. Choose from {list(SELECTORS.keys())}")
    return SELECTORS[name](**kwargs)
