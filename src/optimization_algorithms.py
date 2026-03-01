"""
Advanced diversity optimization algorithms for the Diversity Decoding Arena.

Implements mathematically rigorous diversity selection algorithms including
Determinantal Point Processes (DPP), facility location, max-sum dispersion,
p-dispersion, graph-based diversity, and streaming DPP. Also provides helper
classes for kernel construction, submodular optimization, distance matrix
computation, and complete DPP sampling.

References:
    - Kulesza & Taskar, "Determinantal Point Processes for Machine Learning", 2012
    - Nemhauser, Wolsey & Fisher, "An analysis of approximations for maximizing
      submodular set functions", 1978
    - Hassin, Rubinstein & Tamir, "Approximation algorithms for maximum
      dispersion", 1997
    - Cevallos, Eisenbrand & Zenklusen, "Max-Sum Diversification, Monotone
      Submodular Functions and Dynamic Updates", 2017
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
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
from numpy.linalg import det, eigh, inv, norm, slogdet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DPPResult:
    """Result of a Determinantal Point Process selection."""

    selected_indices: List[int]
    log_probability: float
    kernel_matrix_subset: np.ndarray
    diversity_score: float


@dataclass
class FacilityResult:
    """Result of facility location submodular maximization."""

    selected_indices: List[int]
    facility_cost: float
    assignment: Dict[int, int]
    marginal_gains: List[float]


@dataclass
class DispersionResult:
    """Result of a dispersion-based diversity selection."""

    selected_indices: List[int]
    min_distance: float
    avg_distance: float
    dispersion_value: float


# ---------------------------------------------------------------------------
# Helper: KernelBuilder
# ---------------------------------------------------------------------------


class KernelBuilder:
    """Build kernel (Gram) matrices from feature vectors.

    Supports RBF, cosine, polynomial, linear, and user-supplied kernels.
    All methods return symmetric positive semi-definite matrices suitable
    for use as L-ensemble kernels in DPP sampling.
    """

    @staticmethod
    def rbf_kernel(X: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """Radial-basis function (Gaussian) kernel.

        K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)

        Args:
            X: Feature matrix of shape ``(n, d)``.
            gamma: Kernel bandwidth.  Defaults to ``1 / d``.

        Returns:
            Kernel matrix of shape ``(n, n)``.
        """
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-gamma * sq_dists)

    @staticmethod
    def cosine_kernel(X: np.ndarray) -> np.ndarray:
        """Cosine similarity kernel.

        K(x_i, x_j) = (x_i · x_j) / (||x_i|| ||x_j||)

        Args:
            X: Feature matrix of shape ``(n, d)``.

        Returns:
            Kernel matrix of shape ``(n, n)`` with values in ``[-1, 1]``.
        """
        norms = norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_normed = X / norms
        return X_normed @ X_normed.T

    @staticmethod
    def polynomial_kernel(
        X: np.ndarray, degree: int = 3, coef0: float = 1.0
    ) -> np.ndarray:
        """Polynomial kernel.

        K(x_i, x_j) = (x_i · x_j + coef0)^degree

        Args:
            X: Feature matrix of shape ``(n, d)``.
            degree: Polynomial degree.
            coef0: Free coefficient.

        Returns:
            Kernel matrix of shape ``(n, n)``.
        """
        return (X @ X.T + coef0) ** degree

    @staticmethod
    def linear_kernel(X: np.ndarray) -> np.ndarray:
        """Linear (dot-product) kernel.

        K(x_i, x_j) = x_i · x_j

        Args:
            X: Feature matrix of shape ``(n, d)``.

        Returns:
            Kernel matrix of shape ``(n, n)``.
        """
        return X @ X.T

    @staticmethod
    def custom_kernel(X: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
        """Build a kernel matrix using an arbitrary pairwise function.

        Args:
            X: Feature matrix of shape ``(n, d)``.
            fn: Callable accepting two 1-d vectors and returning a scalar.

        Returns:
            Kernel matrix of shape ``(n, n)``.
        """
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = fn(X[i], X[j])
                K[i, j] = val
                K[j, i] = val
        return K


# ---------------------------------------------------------------------------
# Helper: DistanceMatrix
# ---------------------------------------------------------------------------


class DistanceMatrix:
    """Efficient pairwise distance computation with caching.

    Stores the full distance matrix and supports incremental updates and
    nearest-neighbour queries.
    """

    def __init__(self) -> None:
        self._dist: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None

    def compute(self, X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """Compute the full pairwise distance matrix.

        Args:
            X: Feature matrix of shape ``(n, d)``.
            metric: One of ``'euclidean'``, ``'cosine'``, ``'manhattan'``.

        Returns:
            Distance matrix of shape ``(n, n)``.
        """
        self._X = X.copy()
        if metric == "euclidean":
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            self._dist = np.sqrt(np.sum(diff ** 2, axis=2))
        elif metric == "cosine":
            norms = norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_n = X / norms
            self._dist = 1.0 - X_n @ X_n.T
        elif metric == "manhattan":
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            self._dist = np.sum(np.abs(diff), axis=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return self._dist

    def update(self, X_new: np.ndarray) -> np.ndarray:
        """Add new points and extend the distance matrix.

        Args:
            X_new: New points of shape ``(m, d)``.

        Returns:
            Updated distance matrix of shape ``(n+m, n+m)``.
        """
        if self._X is None or self._dist is None:
            return self.compute(X_new)
        combined = np.vstack([self._X, X_new])
        return self.compute(combined)

    def nearest_neighbors(self, i: int, k: int) -> List[Tuple[int, float]]:
        """Return the *k* nearest neighbours of item *i*.

        Args:
            i: Index of the query point.
            k: Number of neighbours.

        Returns:
            List of ``(index, distance)`` tuples sorted by distance.
        """
        if self._dist is None:
            raise RuntimeError("Call compute() before nearest_neighbors().")
        dists = self._dist[i].copy()
        dists[i] = np.inf  # exclude self
        indices = np.argsort(dists)[:k]
        return [(int(idx), float(dists[idx])) for idx in indices]

    @property
    def matrix(self) -> np.ndarray:
        """Return the cached distance matrix."""
        if self._dist is None:
            raise RuntimeError("Call compute() first.")
        return self._dist


# ---------------------------------------------------------------------------
# Helper: SubmodularOptimizer
# ---------------------------------------------------------------------------


class SubmodularOptimizer:
    """Generic (lazy) greedy algorithm for monotone submodular maximization.

    Provides the standard greedy and the *accelerated lazy greedy* algorithm
    of Minoux (1978).  Both achieve the optimal ``(1 - 1/e)`` approximation
    ratio for maximising a monotone submodular function subject to a
    cardinality constraint.

    Reference:
        Minoux, "Accelerated greedy algorithms for maximizing submodular set
        functions", 1978.
    """

    @staticmethod
    def maximize(
        f: Callable[[List[int]], float],
        ground_set: List[int],
        k: int,
    ) -> Tuple[List[int], List[float]]:
        """Standard greedy maximization.

        At each step pick the element with the largest marginal gain.

        Args:
            f: Monotone submodular set function.
            ground_set: List of candidate item indices.
            k: Number of items to select.

        Returns:
            Tuple of (selected indices, marginal gains per step).
        """
        selected: List[int] = []
        gains: List[float] = []
        remaining = set(ground_set)
        current_val = 0.0

        for _ in range(k):
            best_item = -1
            best_gain = -np.inf
            for item in remaining:
                candidate = selected + [item]
                gain = f(candidate) - current_val
                if gain > best_gain:
                    best_gain = gain
                    best_item = item
            if best_item == -1:
                break
            selected.append(best_item)
            remaining.discard(best_item)
            current_val += best_gain
            gains.append(best_gain)
        return selected, gains

    @staticmethod
    def lazy_greedy(
        f: Callable[[List[int]], float],
        ground_set: List[int],
        k: int,
    ) -> Tuple[List[int], List[float]]:
        """Lazy (accelerated) greedy maximization with a priority queue.

        Exploits submodularity: marginal gains are non-increasing, so we
        maintain upper bounds in a max-heap and only recompute when an
        element reaches the top.

        Args:
            f: Monotone submodular set function.
            ground_set: List of candidate item indices.
            k: Number of items to select.

        Returns:
            Tuple of (selected indices, marginal gains per step).
        """
        selected: List[int] = []
        gains: List[float] = []
        current_val = 0.0

        # Max-heap via negation.  Each entry is (-upper_bound, item, is_current).
        heap: List[Tuple[float, int, bool]] = []
        for item in ground_set:
            gain = f([item]) - current_val
            heapq.heappush(heap, (-gain, item, True))

        selected_set: Set[int] = set()

        for _ in range(k):
            while heap:
                neg_gain, item, is_current = heapq.heappop(heap)
                if item in selected_set:
                    continue
                if is_current:
                    # This gain is up-to-date — select the item.
                    selected.append(item)
                    selected_set.add(item)
                    gains.append(-neg_gain)
                    current_val += -neg_gain
                    break
                # Recompute gain and re-insert.
                new_gain = f(selected + [item]) - current_val
                heapq.heappush(heap, (-new_gain, item, True))
            else:
                break

            # Mark everything remaining as stale.
            refreshed: List[Tuple[float, int, bool]] = []
            for neg_g, itm, _ in heap:
                if itm not in selected_set:
                    refreshed.append((neg_g, itm, False))
            heap = refreshed
            heapq.heapify(heap)

        return selected, gains


# ---------------------------------------------------------------------------
# Helper: DPPSampler
# ---------------------------------------------------------------------------


class DPPSampler:
    """Complete toolkit for sampling from Determinantal Point Processes.

    Implements exact k-DPP sampling via eigendecomposition, MCMC-based
    approximate sampling, marginal probability computation, and conditional
    DPP sampling.

    Reference:
        Kulesza & Taskar, "Determinantal Point Processes for Machine
        Learning", Foundations and Trends in Machine Learning, 2012.
    """

    # ---- exact sampling ---------------------------------------------------

    @staticmethod
    def sample_exact(L: np.ndarray, k: int) -> List[int]:
        """Exact size-*k* DPP sampling via eigendecomposition.

        Algorithm (Kulesza & Taskar 2012, Algorithm 1 adapted for k-DPP):
            1. Eigendecompose *L* = V diag(λ) V^T.
            2. Sample a subset of eigenvectors of size *k* with the correct
               elementary symmetric polynomial probabilities.
            3. Sequentially sample items using the selected eigenvectors
               via orthogonal projection.

        Args:
            L: Positive semi-definite L-ensemble kernel of shape ``(n, n)``.
            k: Size of the desired subset.

        Returns:
            List of *k* selected indices.
        """
        n = L.shape[0]
        if k > n:
            raise ValueError(f"k={k} exceeds number of items n={n}.")

        eigenvalues, eigenvectors = eigh(L)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # --- Phase 1: sample k eigenvectors --------------------------------
        # Compute elementary symmetric polynomials e_k(lambda) using the
        # recurrence: E[l, j] = E[l, j-1] + lambda_j * E[l-1, j-1]
        # where E[l, j] = e_l({lambda_1, ..., lambda_j}).
        E = np.zeros((k + 1, n + 1))
        E[0, :] = 1.0
        for j in range(1, n + 1):
            for l in range(1, min(j, k) + 1):
                E[l, j] = E[l, j - 1] + eigenvalues[j - 1] * E[l - 1, j - 1]

        # Sample eigenvectors in reverse order.
        selected_eigvecs: List[int] = []
        remaining_k = k
        for j in range(n, 0, -1):
            if remaining_k == 0:
                break
            # Probability that eigenvector j is included.
            if E[remaining_k, j] == 0:
                continue
            prob = eigenvalues[j - 1] * E[remaining_k - 1, j - 1] / E[remaining_k, j]
            prob = min(max(prob, 0.0), 1.0)
            if np.random.rand() < prob:
                selected_eigvecs.append(j - 1)
                remaining_k -= 1

        if len(selected_eigvecs) < k:
            # Fallback: fill with highest remaining eigenvalue eigenvectors
            remaining = sorted(
                set(range(n)) - set(selected_eigvecs),
                key=lambda i: eigenvalues[i],
                reverse=True,
            )
            selected_eigvecs.extend(remaining[: k - len(selected_eigvecs)])

        # --- Phase 2: sample items using selected eigenvectors -------------
        V = eigenvectors[:, selected_eigvecs].copy()  # (n, k)
        selected_items: List[int] = []

        for i in range(k):
            # Compute sampling probabilities proportional to squared
            # projection lengths.
            probs = np.sum(V ** 2, axis=1)
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total < 1e-15:
                # Degenerate case: pick uniformly from remaining.
                remaining_items = list(set(range(n)) - set(selected_items))
                chosen = remaining_items[np.random.randint(len(remaining_items))]
            else:
                probs /= total
                chosen = int(np.random.choice(n, p=probs))

            selected_items.append(chosen)

            # Project out the component along the chosen item.
            if i < k - 1:
                b = V[chosen, :].copy()
                b_norm = norm(b)
                if b_norm > 1e-15:
                    b /= b_norm
                    V = V - np.outer(V @ b, b)
                # Zero out the chosen row so it cannot be picked again.
                V[chosen, :] = 0.0

        return selected_items

    # ---- MCMC sampling ----------------------------------------------------

    @staticmethod
    def sample_mcmc(
        L: np.ndarray, k: int, num_steps: int = 1000
    ) -> List[int]:
        """Approximate k-DPP sampling via MCMC (Metropolis–Hastings).

        Runs a Markov chain whose stationary distribution is the k-DPP.
        At each step, propose swapping one item in the current set with one
        outside it; accept with the DPP ratio.

        Reference:
            Anari, Gharan & Rezaei, "Monte Carlo Markov Chain Algorithms for
            Sampling Closely Related Distributions", COLT 2016.

        Args:
            L: PSD kernel of shape ``(n, n)``.
            k: Subset size.
            num_steps: Number of MCMC transitions.

        Returns:
            List of *k* selected indices after *num_steps* transitions.
        """
        n = L.shape[0]
        current = list(np.random.choice(n, size=k, replace=False))
        L_S = L[np.ix_(current, current)]
        current_det = max(det(L_S), 1e-300)

        for _ in range(num_steps):
            # Pick a random item to remove and a random item to add.
            idx_out = np.random.randint(k)
            remaining = list(set(range(n)) - set(current))
            if not remaining:
                break
            new_item = remaining[np.random.randint(len(remaining))]

            proposal = current.copy()
            proposal[idx_out] = new_item
            L_prop = L[np.ix_(proposal, proposal)]
            prop_det = max(det(L_prop), 1e-300)

            acceptance = min(1.0, prop_det / current_det)
            if np.random.rand() < acceptance:
                current = proposal
                current_det = prop_det

        return current

    # ---- marginal probabilities -------------------------------------------

    @staticmethod
    def marginal_probabilities(L: np.ndarray) -> np.ndarray:
        """Compute marginal inclusion probabilities for each item.

        For a DPP with L-ensemble kernel *L*, the marginal kernel is
        K = L (L + I)^{-1}, and K_{ii} is the marginal probability that
        item *i* is included in a sample.

        Args:
            L: PSD kernel of shape ``(n, n)``.

        Returns:
            Array of shape ``(n,)`` with marginal probabilities.
        """
        n = L.shape[0]
        K = L @ inv(L + np.eye(n))
        return np.diag(K)

    # ---- conditional DPP --------------------------------------------------

    @staticmethod
    def conditional_dpp(
        L: np.ndarray,
        fixed_items: List[int],
        k: int,
    ) -> List[int]:
        """Sample from a DPP conditioned on a fixed set of items.

        Uses the Schur complement to derive the conditional L-ensemble on the
        remaining items, then runs exact sampling to select *k* additional
        items.

        Args:
            L: PSD kernel of shape ``(n, n)``.
            fixed_items: Indices that must be included.
            k: Number of *additional* items to sample.

        Returns:
            Combined list of fixed + newly sampled indices.
        """
        n = L.shape[0]
        remaining = sorted(set(range(n)) - set(fixed_items))
        if k > len(remaining):
            raise ValueError("Not enough remaining items for conditional DPP.")

        if not fixed_items:
            sampled = DPPSampler.sample_exact(L, k)
            return sampled

        # Schur complement: L_cond = L_RR - L_RF L_FF^{-1} L_FR
        F = list(fixed_items)
        R = remaining
        L_FF = L[np.ix_(F, F)]
        L_FR = L[np.ix_(F, R)]
        L_RF = L[np.ix_(R, F)]
        L_RR = L[np.ix_(R, R)]

        # Regularise L_FF for numerical stability.
        L_FF_reg = L_FF + 1e-10 * np.eye(len(F))
        L_cond = L_RR - L_RF @ inv(L_FF_reg) @ L_FR

        # Ensure PSD by clamping eigenvalues.
        eigvals, eigvecs = eigh(L_cond)
        eigvals = np.maximum(eigvals, 0.0)
        L_cond = eigvecs @ np.diag(eigvals) @ eigvecs.T

        sampled_in_remaining = DPPSampler.sample_exact(L_cond, k)
        sampled_original = [R[i] for i in sampled_in_remaining]
        return list(fixed_items) + sampled_original


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def dpp_log_probability(kernel_matrix: np.ndarray, subset: List[int]) -> float:
    """Compute the log-probability of a subset under a DPP L-ensemble.

    log P(S) = log det(L_S) - log det(L + I)

    Args:
        kernel_matrix: L-ensemble kernel of shape ``(n, n)``.
        subset: Indices of the selected items.

    Returns:
        Log-probability of the subset.
    """
    n = kernel_matrix.shape[0]
    L_S = kernel_matrix[np.ix_(subset, subset)]
    sign_s, logdet_s = slogdet(L_S)
    sign_n, logdet_n = slogdet(kernel_matrix + np.eye(n))
    if sign_s <= 0:
        return -np.inf
    return logdet_s - logdet_n


def determinantal_point_process(
    kernel_matrix: np.ndarray, k: int
) -> DPPResult:
    """Exact k-DPP sampling via eigendecomposition.

    Implements the full Kulesza & Taskar (2012) algorithm:
        1. Eigendecompose the L-ensemble kernel.
        2. Sample *k* eigenvectors using elementary symmetric polynomials.
        3. Sequentially sample items via orthogonal projection.

    Args:
        kernel_matrix: Positive semi-definite L-ensemble kernel ``(n, n)``.
        k: Number of items to select.

    Returns:
        :class:`DPPResult` with selected indices, log-probability, the
        kernel sub-matrix, and a diversity score (log det L_S).

    Reference:
        Kulesza & Taskar, "Determinantal Point Processes for Machine
        Learning", Foundations and Trends in Machine Learning, 2012.
    """
    selected = DPPSampler.sample_exact(kernel_matrix, k)
    L_S = kernel_matrix[np.ix_(selected, selected)]
    log_prob = dpp_log_probability(kernel_matrix, selected)
    sign, logdet = slogdet(L_S)
    diversity = logdet if sign > 0 else -np.inf

    return DPPResult(
        selected_indices=selected,
        log_probability=log_prob,
        kernel_matrix_subset=L_S,
        diversity_score=float(diversity),
    )


def facility_location(items: np.ndarray, k: int) -> FacilityResult:
    """Greedy submodular maximization for facility location.

    Maximises f(S) = Σ_i max_{j ∈ S} sim(i, j) where sim is cosine
    similarity.  The greedy algorithm gives a ``(1 - 1/e)`` approximation
    for monotone submodular functions under cardinality constraints.

    Args:
        items: Feature matrix of shape ``(n, d)``.
        k: Number of facilities (items) to select.

    Returns:
        :class:`FacilityResult` with selected indices, total facility cost,
        assignment mapping, and per-step marginal gains.

    Reference:
        Nemhauser, Wolsey & Fisher, "An analysis of approximations for
        maximizing submodular set functions", Math. Programming, 1978.
    """
    n = items.shape[0]
    # Compute cosine similarity matrix.
    norms = norm(items, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_n = items / norms
    sim = X_n @ X_n.T  # (n, n)

    selected: List[int] = []
    marginal_gains: List[float] = []
    # max_sim[i] = max similarity of item i to any selected facility so far.
    max_sim = np.zeros(n)

    for _ in range(k):
        best_item = -1
        best_gain = -np.inf

        for j in range(n):
            if j in selected:
                continue
            # Marginal gain of adding j: Σ_i max(0, sim(i,j) - max_sim[i])
            improvements = np.maximum(sim[:, j] - max_sim, 0.0)
            gain = float(improvements.sum())
            if gain > best_gain:
                best_gain = gain
                best_item = j

        if best_item == -1:
            break
        selected.append(best_item)
        marginal_gains.append(best_gain)
        max_sim = np.maximum(max_sim, sim[:, best_item])

    # Build assignment: each item maps to its nearest selected facility.
    assignment: Dict[int, int] = {}
    if selected:
        sim_selected = sim[:, selected]  # (n, len(selected))
        best_facility_idx = np.argmax(sim_selected, axis=1)
        for i in range(n):
            assignment[i] = selected[int(best_facility_idx[i])]

    facility_cost = float(max_sim.sum()) if selected else 0.0
    return FacilityResult(
        selected_indices=selected,
        facility_cost=facility_cost,
        assignment=assignment,
        marginal_gains=marginal_gains,
    )


def max_sum_dispersion(items: np.ndarray, k: int) -> DispersionResult:
    """Maximize the sum of pairwise Euclidean distances among *k* items.

    Greedy algorithm:
        1. Start with the pair having maximum distance.
        2. Iteratively add the item maximizing the sum of distances to the
           current set.
    Followed by local-search refinement (swap-based 2-opt).

    Args:
        items: Feature matrix of shape ``(n, d)``.
        k: Number of items to select.

    Returns:
        :class:`DispersionResult` with selected indices and dispersion
        statistics.

    Reference:
        Hassin, Rubinstein & Tamir, "Approximation algorithms for maximum
        dispersion", Operations Research Letters, 1997.
    """
    n = items.shape[0]
    if k >= n:
        indices = list(range(n))
        dm = DistanceMatrix()
        dist = dm.compute(items)
        return _build_dispersion_result(indices, dist)

    dm = DistanceMatrix()
    dist = dm.compute(items)

    # --- Phase 1: greedy construction -------------------------------------
    # Find the pair with maximum distance.
    np.fill_diagonal(dist, -1.0)
    flat_idx = int(np.argmax(dist))
    i_start, j_start = divmod(flat_idx, n)
    np.fill_diagonal(dist, 0.0)

    selected: List[int] = [i_start, j_start]
    selected_set: Set[int] = {i_start, j_start}
    # dist_sum[i] = sum of distances from item i to all currently selected.
    dist_sum = dist[:, i_start] + dist[:, j_start]

    for _ in range(k - 2):
        # Mask already selected items.
        dist_sum_masked = dist_sum.copy()
        for s in selected_set:
            dist_sum_masked[s] = -np.inf
        best = int(np.argmax(dist_sum_masked))
        selected.append(best)
        selected_set.add(best)
        dist_sum += dist[:, best]

    # --- Phase 2: local search refinement (2-opt) -------------------------
    improved = True
    max_iter = n * k
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        current_obj = _sum_pairwise(selected, dist)
        for idx_pos in range(len(selected)):
            s_item = selected[idx_pos]
            for candidate in range(n):
                if candidate in selected_set:
                    continue
                trial = selected.copy()
                trial[idx_pos] = candidate
                trial_obj = _sum_pairwise(trial, dist)
                if trial_obj > current_obj + 1e-12:
                    selected_set.discard(s_item)
                    selected_set.add(candidate)
                    selected[idx_pos] = candidate
                    current_obj = trial_obj
                    improved = True
                    break
            if improved:
                break

    return _build_dispersion_result(selected, dist)


def p_dispersion(
    items: np.ndarray, k: int, p: float = 2.0
) -> DispersionResult:
    """Maximize the minimum Lp distance among *k* selected items.

    Greedy algorithm:
        1. Start with the pair having maximum Lp distance.
        2. Add the item maximizing the minimum distance to the current set.
    Optionally followed by 2-opt local search.

    Args:
        items: Feature matrix of shape ``(n, d)``.
        k: Number of items to select.
        p: Order of the Lp norm (default 2 = Euclidean).

    Returns:
        :class:`DispersionResult` with selected indices and dispersion
        statistics.

    Reference:
        Ravi, Rosenkrantz & Tayi, "Heuristic and special case algorithms
        for dispersion problems", Operations Research, 1994.
    """
    n = items.shape[0]
    if k >= n:
        indices = list(range(n))
        dist = _lp_distance_matrix(items, p)
        return _build_dispersion_result(indices, dist)

    dist = _lp_distance_matrix(items, p)

    # --- Phase 1: greedy construction -------------------------------------
    np.fill_diagonal(dist, -1.0)
    flat_idx = int(np.argmax(dist))
    i_start, j_start = divmod(flat_idx, n)
    np.fill_diagonal(dist, 0.0)

    selected: List[int] = [i_start, j_start]
    selected_set: Set[int] = {i_start, j_start}

    # min_dist_to_sel[i] = min distance from item i to any selected item.
    min_dist_to_sel = np.minimum(dist[:, i_start], dist[:, j_start])

    for _ in range(k - 2):
        min_dist_to_sel_masked = min_dist_to_sel.copy()
        for s in selected_set:
            min_dist_to_sel_masked[s] = -np.inf
        best = int(np.argmax(min_dist_to_sel_masked))
        selected.append(best)
        selected_set.add(best)
        min_dist_to_sel = np.minimum(min_dist_to_sel, dist[:, best])

    # --- Phase 2: 2-opt local search --------------------------------------
    improved = True
    max_iter = n * k
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        current_min = _min_pairwise(selected, dist)
        for idx_pos in range(len(selected)):
            s_item = selected[idx_pos]
            for candidate in range(n):
                if candidate in selected_set:
                    continue
                trial = selected.copy()
                trial[idx_pos] = candidate
                trial_min = _min_pairwise(trial, dist)
                if trial_min > current_min + 1e-12:
                    selected_set.discard(s_item)
                    selected_set.add(candidate)
                    selected[idx_pos] = candidate
                    current_min = trial_min
                    improved = True
                    break
            if improved:
                break

    return _build_dispersion_result(selected, dist)


def graph_based_diversity(
    similarity_graph: np.ndarray, k: int
) -> List[int]:
    """Select *k* nodes from a similarity graph to maximize diversity.

    Combines three complementary strategies and returns the best result:
        (a) Maximum independent set approximation on a thresholded graph.
        (b) Graph partitioning for balanced selection.
        (c) Spectral diversity using the Fiedler vector.

    Args:
        similarity_graph: Symmetric similarity / adjacency matrix ``(n, n)``.
        k: Number of nodes to select.

    Returns:
        List of *k* selected node indices.

    Reference:
        Fiedler, "Algebraic connectivity of graphs", 1973.
    """
    n = similarity_graph.shape[0]
    if k >= n:
        return list(range(n))

    candidates: List[Tuple[float, List[int]]] = []

    # (a) Max independent set on thresholded graph -------------------------
    mis_result = _max_independent_set_greedy(similarity_graph, k)
    mis_score = _diversity_score(mis_result, similarity_graph)
    candidates.append((mis_score, mis_result))

    # (b) Graph partitioning -----------------------------------------------
    part_result = _partition_selection(similarity_graph, k)
    part_score = _diversity_score(part_result, similarity_graph)
    candidates.append((part_score, part_result))

    # (c) Spectral diversity via Fiedler vector ----------------------------
    spec_result = _spectral_diversity(similarity_graph, k)
    spec_score = _diversity_score(spec_result, similarity_graph)
    candidates.append((spec_score, spec_result))

    # Return the solution with best diversity (lowest total similarity).
    best = min(candidates, key=lambda x: x[0])
    return best[1]


def streaming_dpp(
    stream: Iterator,
    k: int,
    budget: int,
    kernel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List:
    """Streaming diversity selection using a buffered DPP.

    Maintains a buffer of size *budget*.  When the buffer is full, runs a
    k-DPP on the buffer to identify the most diverse subset, then keeps
    those items and evicts the rest.  As new items arrive, they are added to
    the buffer and may replace existing items if they increase diversity.

    Args:
        stream: Iterator yielding feature vectors (1-d arrays).
        k: Number of items to ultimately select.
        budget: Buffer capacity (must be >= k).
        kernel_fn: Optional kernel function mapping ``(m, d)`` feature
            matrix to ``(m, m)`` kernel.  Defaults to RBF kernel.

    Returns:
        List of *k* selected feature vectors.
    """
    if budget < k:
        raise ValueError("budget must be >= k")

    if kernel_fn is None:
        kernel_fn = KernelBuilder.rbf_kernel

    buffer: List[np.ndarray] = []
    selected_vecs: List[np.ndarray] = []

    for item in stream:
        vec = np.asarray(item, dtype=float)
        buffer.append(vec)

        if len(buffer) >= budget:
            # Run k-DPP on the buffer.
            X = np.vstack(buffer)
            L = kernel_fn(X)
            result = determinantal_point_process(L, k)
            selected_vecs = [buffer[i] for i in result.selected_indices]
            # Keep selected items in the buffer, discard the rest.
            buffer = list(selected_vecs)

    # Final selection from remaining buffer.
    if len(buffer) <= k:
        return buffer

    X = np.vstack(buffer)
    L = kernel_fn(X)
    result = determinantal_point_process(L, min(k, len(buffer)))
    return [buffer[i] for i in result.selected_indices]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lp_distance_matrix(X: np.ndarray, p: float) -> np.ndarray:
    """Compute pairwise Lp distance matrix."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    if p == np.inf:
        return np.max(np.abs(diff), axis=2)
    return np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)


def _sum_pairwise(indices: List[int], dist: np.ndarray) -> float:
    """Sum of pairwise distances among *indices*."""
    total = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            total += dist[indices[i], indices[j]]
    return total


def _min_pairwise(indices: List[int], dist: np.ndarray) -> float:
    """Minimum pairwise distance among *indices*."""
    min_d = np.inf
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = dist[indices[i], indices[j]]
            if d < min_d:
                min_d = d
    return min_d


def _build_dispersion_result(
    indices: List[int], dist: np.ndarray
) -> DispersionResult:
    """Build a :class:`DispersionResult` from selected indices and dist matrix."""
    k = len(indices)
    if k < 2:
        return DispersionResult(
            selected_indices=indices,
            min_distance=0.0,
            avg_distance=0.0,
            dispersion_value=0.0,
        )
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            pairwise.append(dist[indices[i], indices[j]])
    pairwise_arr = np.array(pairwise)
    return DispersionResult(
        selected_indices=indices,
        min_distance=float(pairwise_arr.min()),
        avg_distance=float(pairwise_arr.mean()),
        dispersion_value=float(pairwise_arr.sum()),
    )


def _max_independent_set_greedy(
    sim: np.ndarray, k: int
) -> List[int]:
    """Greedy maximum independent set on a thresholded similarity graph.

    Threshold is the median off-diagonal similarity.  The greedy heuristic
    iteratively selects the node with fewest neighbours in the remaining
    thresholded graph, breaking ties by lowest total similarity.
    """
    n = sim.shape[0]
    # Threshold: edges where similarity exceeds the median.
    upper = sim[np.triu_indices(n, k=1)]
    threshold = float(np.median(upper)) if len(upper) > 0 else 0.5
    adj = (sim > threshold).astype(float)
    np.fill_diagonal(adj, 0.0)

    selected: List[int] = []
    available = set(range(n))

    while len(selected) < k and available:
        # Pick the node with the fewest neighbours among available nodes.
        best_node = -1
        best_deg = np.inf
        best_sim_sum = np.inf
        for node in available:
            deg = sum(1 for nb in available if nb != node and adj[node, nb] > 0)
            sim_sum = sum(sim[node, nb] for nb in available if nb != node)
            if deg < best_deg or (deg == best_deg and sim_sum < best_sim_sum):
                best_deg = deg
                best_sim_sum = sim_sum
                best_node = node
        if best_node == -1:
            break
        selected.append(best_node)
        # Remove best_node and its neighbours from the available set.
        to_remove = {best_node}
        for nb in list(available):
            if nb != best_node and adj[best_node, nb] > 0:
                to_remove.add(nb)
        available -= to_remove

    # If we don't have enough, fill greedily from remaining items.
    if len(selected) < k:
        remaining = sorted(
            set(range(n)) - set(selected),
            key=lambda i: sum(sim[i, s] for s in selected),
        )
        selected.extend(remaining[: k - len(selected)])

    return selected[:k]


def _partition_selection(sim: np.ndarray, k: int) -> List[int]:
    """Graph partitioning for balanced diversity selection.

    Uses recursive spectral bisection (via the Fiedler vector) to split
    nodes into *k* groups, then selects one representative per group (the
    node with the smallest total similarity to nodes in other groups).
    """
    n = sim.shape[0]
    indices = list(range(n))

    if k == 1:
        # Select the node with lowest total similarity.
        total_sim = sim.sum(axis=1)
        return [int(np.argmin(total_sim))]

    partitions = _recursive_bisect(sim, indices, k)

    selected: List[int] = []
    for part in partitions:
        if not part:
            continue
        # Select the node in *part* that is most dissimilar to nodes outside.
        outside = list(set(indices) - set(part))
        if not outside:
            selected.append(part[0])
            continue
        best_node = min(
            part,
            key=lambda node: sum(sim[node, o] for o in outside) / max(len(outside), 1),
        )
        selected.append(best_node)

    # Ensure exactly k items.
    if len(selected) < k:
        remaining = sorted(set(indices) - set(selected))
        selected.extend(remaining[: k - len(selected)])
    return selected[:k]


def _recursive_bisect(
    sim: np.ndarray, indices: List[int], num_parts: int
) -> List[List[int]]:
    """Recursively bisect *indices* using the Fiedler vector."""
    if num_parts <= 1 or len(indices) <= 1:
        return [indices]

    sub_sim = sim[np.ix_(indices, indices)]
    n_sub = sub_sim.shape[0]
    if n_sub <= 1:
        return [indices]

    # Laplacian.
    D = np.diag(sub_sim.sum(axis=1))
    L_mat = D - sub_sim

    eigenvalues, eigenvectors = eigh(L_mat)
    # Second smallest eigenvector (Fiedler vector).
    fiedler_idx = 1 if n_sub > 1 else 0
    fiedler = eigenvectors[:, fiedler_idx]
    median_val = float(np.median(fiedler))

    part_a = [indices[i] for i in range(n_sub) if fiedler[i] <= median_val]
    part_b = [indices[i] for i in range(n_sub) if fiedler[i] > median_val]
    if not part_a:
        part_a = [indices[0]]
        part_b = indices[1:]
    if not part_b:
        part_b = [indices[-1]]
        part_a = indices[:-1]

    left_parts = num_parts // 2
    right_parts = num_parts - left_parts
    return _recursive_bisect(sim, part_a, left_parts) + _recursive_bisect(
        sim, part_b, right_parts
    )


def _spectral_diversity(sim: np.ndarray, k: int) -> List[int]:
    """Select *k* diverse nodes using spectral embedding.

    Embed nodes using the bottom eigenvectors of the graph Laplacian, then
    greedily select the most spread-out nodes in the embedding space.
    """
    n = sim.shape[0]
    D = np.diag(sim.sum(axis=1))
    L_mat = D - sim

    num_vecs = min(k, n)
    eigenvalues, eigenvectors = eigh(L_mat)
    # Use the first *num_vecs* non-trivial eigenvectors (skip index 0).
    start = 1 if n > 1 else 0
    end = min(start + num_vecs, n)
    embedding = eigenvectors[:, start:end]  # (n, num_vecs)

    # Greedy farthest-point selection in embedding space.
    dm = DistanceMatrix()
    dist = dm.compute(embedding)
    np.fill_diagonal(dist, -1.0)
    flat_idx = int(np.argmax(dist))
    i0, j0 = divmod(flat_idx, n)
    np.fill_diagonal(dist, 0.0)

    selected: List[int] = [i0, j0]
    selected_set: Set[int] = {i0, j0}
    min_dist_to_sel = np.minimum(dist[:, i0], dist[:, j0])

    while len(selected) < k:
        masked = min_dist_to_sel.copy()
        for s in selected_set:
            masked[s] = -np.inf
        best = int(np.argmax(masked))
        selected.append(best)
        selected_set.add(best)
        min_dist_to_sel = np.minimum(min_dist_to_sel, dist[:, best])

    return selected[:k]


def _diversity_score(indices: List[int], sim: np.ndarray) -> float:
    """Total pairwise similarity among *indices* (lower is more diverse)."""
    total = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            total += sim[indices[i], indices[j]]
    return total
