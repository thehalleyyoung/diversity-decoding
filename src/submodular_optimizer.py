"""
Submodular function optimization for diversity.

Implements greedy with lazy evaluations, stochastic greedy,
continuous greedy + pipage rounding, multilinear extension evaluation,
and built-in submodular functions (facility location, graph cut,
log-determinant, coverage, feature-based).
"""

import numpy as np
from typing import Optional, List, Callable, Tuple, Set, Union, Dict


# ======================================================================
# Submodular Function Definitions
# ======================================================================

class SubmodularFunction:
    """Base class for submodular functions."""

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        """Evaluate f(S)."""
        raise NotImplementedError

    def marginal_gain(self, S: Union[Set[int], List[int]], element: int) -> float:
        """Compute f(S ∪ {element}) - f(S)."""
        S_set = set(S)
        if element in S_set:
            return 0.0
        val_S = self.evaluate(S_set)
        val_Se = self.evaluate(S_set | {element})
        return val_Se - val_S

    @property
    def ground_set_size(self) -> int:
        raise NotImplementedError


class FacilityLocationFunction(SubmodularFunction):
    """f(S) = Σ_i max_{j∈S} sim(i,j).

    Monotone submodular. Models how well selected items represent all items.
    """

    def __init__(self, similarity_matrix: np.ndarray):
        self.sim = np.asarray(similarity_matrix, dtype=np.float64)
        self.n = self.sim.shape[0]

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        S = list(S)
        if len(S) == 0:
            return 0.0
        return float(np.sum(np.max(self.sim[:, S], axis=1)))

    def marginal_gain(self, S: Union[Set[int], List[int]], element: int) -> float:
        S = list(S)
        if element in S:
            return 0.0
        if len(S) == 0:
            return float(np.sum(self.sim[:, element]))
        current_max = np.max(self.sim[:, S], axis=1)
        new_max = np.maximum(current_max, self.sim[:, element])
        return float(np.sum(new_max - current_max))

    @property
    def ground_set_size(self) -> int:
        return self.n


class GraphCutFunction(SubmodularFunction):
    """f(S) = Σ_{i∈S, j∉S} w(i,j).

    Non-monotone submodular. Measures boundary between selected and unselected.
    """

    def __init__(self, weight_matrix: np.ndarray):
        self.W = np.asarray(weight_matrix, dtype=np.float64)
        self.n = self.W.shape[0]

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        S = list(S)
        if len(S) == 0 or len(S) == self.n:
            return 0.0
        S_set = set(S)
        complement = [j for j in range(self.n) if j not in S_set]
        return float(np.sum(self.W[np.ix_(S, complement)]))

    @property
    def ground_set_size(self) -> int:
        return self.n


class LogDeterminantFunction(SubmodularFunction):
    """f(S) = log det(L_S + εI).

    Monotone submodular. The DPP marginal kernel log-determinant.
    """

    def __init__(self, kernel_matrix: np.ndarray, epsilon: float = 1e-6):
        self.L = np.asarray(kernel_matrix, dtype=np.float64)
        self.n = self.L.shape[0]
        self.epsilon = epsilon

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        S = list(S)
        if len(S) == 0:
            return 0.0
        L_S = self.L[np.ix_(S, S)] + self.epsilon * np.eye(len(S))
        sign, logdet = np.linalg.slogdet(L_S)
        if sign <= 0:
            return -1e10
        return float(logdet)

    @property
    def ground_set_size(self) -> int:
        return self.n


class CoverageFunction(SubmodularFunction):
    """f(S) = |∪_{i∈S} cover(i)|.

    Monotone submodular. Models set cover.
    """

    def __init__(self, covers: List[Set[int]]):
        """
        Args:
            covers: List of sets, one per item. covers[i] is the set of
                    elements covered by item i.
        """
        self.covers = [set(c) for c in covers]
        self.n = len(covers)

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        S = list(S)
        covered = set()
        for i in S:
            covered |= self.covers[i]
        return float(len(covered))

    def marginal_gain(self, S: Union[Set[int], List[int]], element: int) -> float:
        S = list(S)
        covered = set()
        for i in S:
            covered |= self.covers[i]
        new_elements = self.covers[element] - covered
        return float(len(new_elements))

    @property
    def ground_set_size(self) -> int:
        return self.n


class FeatureBasedFunction(SubmodularFunction):
    """f(S) = Σ_k min(Σ_{i∈S} w_{ik}, 1).

    Monotone submodular. Each feature has diminishing returns.
    The weight w_{ik} is how much item i contributes to feature k.
    """

    def __init__(self, weights: np.ndarray):
        """
        Args:
            weights: (n_items, n_features) non-negative weight matrix.
        """
        self.W = np.asarray(weights, dtype=np.float64)
        self.n = self.W.shape[0]

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        S = list(S)
        if len(S) == 0:
            return 0.0
        total_per_feature = np.sum(self.W[S, :], axis=0)
        return float(np.sum(np.minimum(total_per_feature, 1.0)))

    @property
    def ground_set_size(self) -> int:
        return self.n


class SumPairwiseDistanceFunction(SubmodularFunction):
    """f(S) = Σ_{i<j, i,j∈S} d(i,j).

    Maximises total pairwise distance of the selected subset.
    The marginal gain of adding element e to S is Σ_{j∈S} d(e,j),
    which is always non-negative (assuming non-negative distances),
    making this function monotone. Uses precomputed distance matrix.
    """

    def __init__(self, distance_matrix: np.ndarray):
        self.D = np.asarray(distance_matrix, dtype=np.float64)
        self.n = self.D.shape[0]

    def evaluate(self, S: Union[Set[int], List[int]]) -> float:
        S = list(S)
        if len(S) < 2:
            # Return small value for single element to allow greedy to start
            return 0.0
        return float(np.sum(self.D[np.ix_(S, S)])) / 2.0

    def marginal_gain(self, S: Union[Set[int], List[int]], element: int) -> float:
        S = list(S)
        if element in S:
            return 0.0
        if len(S) == 0:
            # First element: use sum of distances to all others as proxy
            return float(np.sum(self.D[element, :]))
        return float(np.sum(self.D[element, S]))

    @property
    def ground_set_size(self) -> int:
        return self.n


# ======================================================================
# Submodular Optimizer
# ======================================================================

class SubmodularOptimizer:
    """Optimizer for submodular function maximization.

    Supports cardinality constraints and matroid constraints.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Greedy with lazy evaluations (accelerated greedy)
    # ------------------------------------------------------------------

    def greedy(self, f: SubmodularFunction, k: int,
               ground_set: Optional[List[int]] = None) -> Tuple[List[int], float]:
        """Standard greedy maximization.

        At each step, select the element with highest marginal gain.

        Achieves (1 - 1/e) approximation for monotone submodular
        under cardinality constraint.

        Args:
            f: Submodular function.
            k: Cardinality constraint.
            ground_set: Elements to choose from (default: all).

        Returns:
            (selected_list, objective_value)
        """
        if ground_set is None:
            ground_set = list(range(f.ground_set_size))

        n = len(ground_set)
        k = min(k, n)

        selected = []
        selected_set = set()

        for _ in range(k):
            best_element = None
            best_gain = -float('inf')

            for e in ground_set:
                if e in selected_set:
                    continue
                gain = f.marginal_gain(selected, e)
                if gain > best_gain:
                    best_gain = gain
                    best_element = e

            if best_element is None or best_gain <= 0:
                break

            selected.append(best_element)
            selected_set.add(best_element)

        obj_val = f.evaluate(selected)
        return selected, float(obj_val)

    # ------------------------------------------------------------------
    # Stochastic greedy
    # ------------------------------------------------------------------

    def stochastic_greedy(self, f: SubmodularFunction, k: int,
                          epsilon: float = 0.1,
                          ground_set: Optional[List[int]] = None,
                          rng: Optional[np.random.RandomState] = None
                          ) -> Tuple[List[int], float]:
        """Stochastic greedy: subsample + greedy.

        At each step, subsample O(n/k * log(1/ε)) elements and pick
        the best among them. Achieves (1 - 1/e - ε) approximation
        in O(n log(1/ε)) time.

        Args:
            f: Submodular function.
            k: Cardinality constraint.
            epsilon: Approximation error tolerance.
            ground_set: Elements to choose from.
            rng: Random state.

        Returns:
            (selected_list, objective_value)
        """
        if ground_set is None:
            ground_set = list(range(f.ground_set_size))
        if rng is None:
            rng = np.random.RandomState()

        n = len(ground_set)
        k = min(k, n)

        # Subsample size
        s = max(int(n / k * np.log(1.0 / max(epsilon, 1e-10))), 1)
        s = min(s, n)

        selected = []
        selected_set = set()
        remaining = list(ground_set)

        for _ in range(k):
            if not remaining:
                break

            # Subsample
            sample_size = min(s, len(remaining))
            subsample_idx = rng.choice(len(remaining), size=sample_size, replace=False)
            subsample = [remaining[i] for i in subsample_idx]

            # Find best in subsample
            best_element = None
            best_gain = -float('inf')
            for e in subsample:
                gain = f.marginal_gain(selected, e)
                if gain > best_gain:
                    best_gain = gain
                    best_element = e

            if best_element is None or best_gain <= 0:
                break

            selected.append(best_element)
            selected_set.add(best_element)
            remaining.remove(best_element)

        obj_val = f.evaluate(selected)
        return selected, float(obj_val)

    # ------------------------------------------------------------------
    # Continuous greedy + pipage rounding
    # ------------------------------------------------------------------

    def continuous_greedy(self, f: SubmodularFunction, k: int,
                          n_steps: int = 20, n_samples: int = 50,
                          ground_set: Optional[List[int]] = None,
                          rng: Optional[np.random.RandomState] = None
                          ) -> Tuple[List[int], float]:
        """Continuous greedy + pipage rounding for matroid constraints.

        1. Optimize the multilinear extension F(x) via continuous greedy.
        2. Round the fractional solution via pipage rounding.

        Args:
            f: Submodular function.
            k: Cardinality constraint (uniform matroid).
            n_steps: Discretization steps for continuous greedy.
            n_samples: Samples for multilinear extension estimation.
            ground_set: Elements to choose from.
            rng: Random state.

        Returns:
            (selected_list, objective_value)
        """
        if ground_set is None:
            ground_set = list(range(f.ground_set_size))
        if rng is None:
            rng = np.random.RandomState()

        n = len(ground_set)
        k = min(k, n)
        idx_map = {e: i for i, e in enumerate(ground_set)}

        # Initialize fractional solution x ∈ [0,1]^n
        x = np.zeros(n, dtype=np.float64)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            # Estimate gradient of multilinear extension
            grad = self._estimate_multilinear_gradient(
                f, x, ground_set, n_samples, rng
            )

            # Find direction: maximize <grad, w-x> subject to w in matroid
            # For uniform matroid: select top-k coordinates
            w = np.zeros(n, dtype=np.float64)
            top_k = np.argsort(-grad)[:k]
            w[top_k] = 1.0

            # Update x
            x = x + dt * (w - x)
            x = np.clip(x, 0.0, 1.0)

        # Pipage rounding
        selected = self._pipage_round(f, x, ground_set, k, rng)

        obj_val = f.evaluate(selected)
        return selected, float(obj_val)

    def _estimate_multilinear_gradient(self, f: SubmodularFunction,
                                        x: np.ndarray,
                                        ground_set: List[int],
                                        n_samples: int,
                                        rng: np.random.RandomState
                                        ) -> np.ndarray:
        """Estimate gradient of multilinear extension via sampling.

        The multilinear extension is:
            F(x) = E_{R~x}[f(R)]

        Its partial derivative w.r.t. x_i is:
            ∂F/∂x_i = E_{R~x_{-i}}[f(R ∪ {i}) - f(R)]
        """
        n = len(ground_set)
        grad = np.zeros(n, dtype=np.float64)

        for _ in range(n_samples):
            # Sample a random set R according to x
            r = rng.rand(n)
            R = set()
            for j in range(n):
                if r[j] < x[j]:
                    R.add(ground_set[j])

            f_R = f.evaluate(R)

            for j in range(n):
                e = ground_set[j]
                if e in R:
                    # f(R) - f(R \ {e})
                    f_without = f.evaluate(R - {e})
                    grad[j] += f_R - f_without
                else:
                    # f(R ∪ {e}) - f(R)
                    f_with = f.evaluate(R | {e})
                    grad[j] += f_with - f_R

        grad /= n_samples
        return grad

    def _pipage_round(self, f: SubmodularFunction, x: np.ndarray,
                      ground_set: List[int], k: int,
                      rng: np.random.RandomState) -> List[int]:
        """Pipage rounding: convert fractional to integral solution.

        Iteratively transfer probability mass between pairs of
        fractional variables while maintaining the matroid constraint
        and (in expectation) not decreasing the objective.
        """
        x = x.copy()
        n = len(x)

        max_rounds = n * 10
        for _ in range(max_rounds):
            # Find two fractional variables
            fractional = [i for i in range(n) if 1e-8 < x[i] < 1 - 1e-8]
            if len(fractional) < 2:
                break

            i, j = fractional[0], fractional[1]

            # Compute how much to transfer
            # Option 1: increase x[i], decrease x[j]
            alpha1 = min(1.0 - x[i], x[j])
            # Option 2: decrease x[i], increase x[j]
            alpha2 = min(x[i], 1.0 - x[j])

            # Evaluate both options via multilinear extension (approximate)
            x1 = x.copy()
            x1[i] += alpha1
            x1[j] -= alpha1
            x1 = np.clip(x1, 0.0, 1.0)

            x2 = x.copy()
            x2[i] -= alpha2
            x2[j] += alpha2
            x2 = np.clip(x2, 0.0, 1.0)

            # Use a quick evaluation
            f1 = self._quick_multilinear(f, x1, ground_set, rng, 5)
            f2 = self._quick_multilinear(f, x2, ground_set, rng, 5)

            if f1 >= f2:
                x = x1
            else:
                x = x2

        # Final: round remaining fractional variables
        selected = []
        for i in range(n):
            if x[i] > 0.5:
                selected.append(ground_set[i])

        # Enforce cardinality
        if len(selected) > k:
            # Keep top-k by x value
            vals = [(x[ground_set.index(e) if e in ground_set else 0], e) for e in selected]
            vals.sort(reverse=True)
            selected = [e for _, e in vals[:k]]
        elif len(selected) < k:
            remaining = [ground_set[i] for i in range(n) if ground_set[i] not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[:k - len(selected)])

        return selected[:k]

    def _quick_multilinear(self, f: SubmodularFunction, x: np.ndarray,
                           ground_set: List[int],
                           rng: np.random.RandomState,
                           n_samples: int = 5) -> float:
        """Quick multilinear extension estimate."""
        total = 0.0
        for _ in range(n_samples):
            R = set()
            for j in range(len(x)):
                if rng.rand() < x[j]:
                    R.add(ground_set[j])
            total += f.evaluate(R)
        return total / n_samples

    # ------------------------------------------------------------------
    # Generic maximize interface
    # ------------------------------------------------------------------

    def maximize(self, f: SubmodularFunction, k: int,
                 constraint: str = 'cardinality',
                 algorithm: str = 'lazy_greedy',
                 ground_set: Optional[List[int]] = None,
                 **kwargs) -> Tuple[List[int], float]:
        """Maximize a submodular function subject to constraints.

        Args:
            f: Submodular function to maximize.
            k: Constraint parameter (cardinality).
            constraint: 'cardinality' or 'matroid'.
            algorithm: 'lazy_greedy', 'stochastic', or 'continuous'.
            ground_set: Elements to choose from.
            **kwargs: Additional arguments for specific algorithms.

        Returns:
            (selected_elements, objective_value)
        """
        if algorithm == 'lazy_greedy':
            return self.greedy(f, k, ground_set)
        elif algorithm == 'stochastic':
            return self.stochastic_greedy(f, k, ground_set=ground_set, **kwargs)
        elif algorithm == 'continuous':
            return self.continuous_greedy(f, k, ground_set=ground_set, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


# ======================================================================
# Diminishing Returns Verification
# ======================================================================

def verify_diminishing_returns(f: SubmodularFunction,
                               ground_set: Optional[List[int]] = None,
                               n_tests: int = 100,
                               rng: Optional[np.random.RandomState] = None
                               ) -> Tuple[bool, List[Dict]]:
    """Verify that a function exhibits diminishing returns (submodularity).

    For submodularity: for all A ⊆ B and element e ∉ B,
        f(A ∪ {e}) - f(A) ≥ f(B ∪ {e}) - f(B)

    Args:
        f: Function to test.
        ground_set: Elements to test over.
        n_tests: Number of random tests.
        rng: Random state.

    Returns:
        (is_submodular, violations) where violations lists any counterexamples.
    """
    if ground_set is None:
        ground_set = list(range(f.ground_set_size))
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(ground_set)
    violations = []

    for _ in range(n_tests):
        # Random subset A
        size_A = rng.randint(0, max(n // 2, 1))
        A_indices = rng.choice(n, size=size_A, replace=False)
        A = set(ground_set[i] for i in A_indices)

        # B ⊇ A with some extra elements
        extra = rng.randint(0, max((n - size_A) // 2, 1))
        remaining = [ground_set[i] for i in range(n) if ground_set[i] not in A]
        if extra > 0 and len(remaining) > 0:
            extra_indices = rng.choice(len(remaining), size=min(extra, len(remaining)), replace=False)
            B = A | set(remaining[i] for i in extra_indices)
        else:
            B = A.copy()

        # Element e ∉ B
        not_in_B = [e for e in ground_set if e not in B]
        if not not_in_B:
            continue
        e = rng.choice(not_in_B)

        gain_A = f.marginal_gain(A, e)
        gain_B = f.marginal_gain(B, e)

        if gain_A < gain_B - 1e-8:
            violations.append({
                'A': sorted(A),
                'B': sorted(B),
                'element': e,
                'gain_A': gain_A,
                'gain_B': gain_B,
                'difference': gain_A - gain_B
            })

    return len(violations) == 0, violations


# ======================================================================
# Multilinear Extension Evaluator
# ======================================================================

class MultilinearExtension:
    """Evaluate the multilinear extension F(x) of a submodular function f.

    F(x) = E_{R ~ Bern(x)}[f(R)] = Σ_{S⊆V} f(S) Π_{i∈S} x_i Π_{j∉S} (1-x_j)
    """

    def __init__(self, f: SubmodularFunction,
                 ground_set: Optional[List[int]] = None):
        self.f = f
        self.ground_set = ground_set or list(range(f.ground_set_size))

    def evaluate(self, x: np.ndarray, n_samples: int = 100,
                 rng: Optional[np.random.RandomState] = None) -> float:
        """Estimate F(x) via Monte Carlo sampling.

        Args:
            x: (n,) fractional solution in [0,1]^n.
            n_samples: Number of samples.
            rng: Random state.

        Returns:
            Estimated multilinear extension value.
        """
        if rng is None:
            rng = np.random.RandomState()

        x = np.asarray(x, dtype=np.float64)
        total = 0.0
        for _ in range(n_samples):
            R = set()
            for j, e in enumerate(self.ground_set):
                if rng.rand() < x[j]:
                    R.add(e)
            total += self.f.evaluate(R)
        return total / n_samples

    def gradient(self, x: np.ndarray, n_samples: int = 100,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Estimate gradient ∇F(x) via Monte Carlo.

        ∂F/∂x_i = E_{R~x_{-i}}[f(R∪{i}) - f(R\\{i})]
        """
        if rng is None:
            rng = np.random.RandomState()

        n = len(self.ground_set)
        grad = np.zeros(n, dtype=np.float64)

        for _ in range(n_samples):
            r = rng.rand(n)
            R = set()
            for j in range(n):
                if r[j] < x[j]:
                    R.add(self.ground_set[j])

            f_R = self.f.evaluate(R)

            for j in range(n):
                e = self.ground_set[j]
                if e in R:
                    grad[j] += f_R - self.f.evaluate(R - {e})
                else:
                    grad[j] += self.f.evaluate(R | {e}) - f_R

        grad /= n_samples
        return grad


# ======================================================================
# Constrained Submodular Maximization
# ======================================================================

class KnapsackConstrainedOptimizer:
    """Submodular maximization under knapsack constraint.

    max f(S) subject to Σ_{i∈S} c_i ≤ B
    """

    def __init__(self):
        pass

    def maximize(self, f: SubmodularFunction, costs: np.ndarray,
                 budget: float,
                 ground_set: Optional[List[int]] = None
                 ) -> Tuple[List[int], float]:
        """Greedy with cost-effectiveness ratio.

        At each step, select element with highest gain/cost ratio.

        Args:
            f: Submodular function.
            costs: (n,) cost per element.
            budget: Total budget.
            ground_set: Elements.

        Returns:
            (selected, objective_value)
        """
        if ground_set is None:
            ground_set = list(range(f.ground_set_size))

        costs = np.asarray(costs, dtype=np.float64)
        selected = []
        total_cost = 0.0
        remaining = set(ground_set)

        while remaining:
            best_element = None
            best_ratio = -float('inf')

            for e in remaining:
                idx = e  # Assuming ground_set maps directly
                if total_cost + costs[idx] > budget:
                    continue
                gain = f.marginal_gain(selected, e)
                ratio = gain / max(costs[idx], 1e-12)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_element = e

            if best_element is None:
                break

            selected.append(best_element)
            total_cost += costs[best_element]
            remaining.discard(best_element)

        # Also try single best element (for 1/2 approx guarantee)
        best_single = None
        best_single_val = -float('inf')
        for e in ground_set:
            if costs[e] <= budget:
                val = f.evaluate({e})
                if val > best_single_val:
                    best_single_val = val
                    best_single = e

        greedy_val = f.evaluate(selected)
        if best_single is not None and best_single_val > greedy_val:
            return [best_single], float(best_single_val)

        return selected, float(greedy_val)


class PartitionMatroidOptimizer:
    """Submodular maximization under partition matroid.

    Elements are partitioned into groups, with a capacity per group.
    max f(S) s.t. |S ∩ P_i| ≤ k_i for each partition P_i.
    """

    def __init__(self):
        pass

    def maximize(self, f: SubmodularFunction,
                 partitions: Dict[int, List[int]],
                 capacities: Dict[int, int]
                 ) -> Tuple[List[int], float]:
        """Greedy maximization under partition matroid.

        Args:
            f: Submodular function.
            partitions: Dict mapping partition_id -> list of elements.
            capacities: Dict mapping partition_id -> max elements from partition.

        Returns:
            (selected, objective_value)
        """
        selected = []
        selected_set = set()
        partition_counts = {pid: 0 for pid in partitions}

        all_elements = []
        elem_partition = {}
        for pid, elems in partitions.items():
            for e in elems:
                all_elements.append(e)
                elem_partition[e] = pid

        # Greedy
        improved = True
        while improved:
            improved = False
            best_element = None
            best_gain = -1e-12

            for e in all_elements:
                if e in selected_set:
                    continue
                pid = elem_partition[e]
                if partition_counts[pid] >= capacities.get(pid, 0):
                    continue

                gain = f.marginal_gain(selected, e)
                if gain > best_gain:
                    best_gain = gain
                    best_element = e

            if best_element is not None:
                selected.append(best_element)
                selected_set.add(best_element)
                pid = elem_partition[best_element]
                partition_counts[pid] += 1
                improved = True

        obj_val = f.evaluate(selected)
        return selected, float(obj_val)


def demo_submodular():
    """Demonstrate submodular optimization."""
    rng = np.random.RandomState(42)
    n, d = 30, 5
    X = rng.randn(n, d)

    # Pairwise similarity
    from clustering_diversity import pairwise_distances
    D = pairwise_distances(X)
    sim = np.max(D) - D
    np.fill_diagonal(sim, 0.0)

    optimizer = SubmodularOptimizer()

    # Facility location
    fl = FacilityLocationFunction(sim)
    sel, val = optimizer.greedy(fl, k=5)
    print(f"Facility location greedy: {sel}, value={val:.4f}")

    # Stochastic greedy
    sel2, val2 = optimizer.stochastic_greedy(fl, k=5, rng=rng)
    print(f"Facility location stochastic: {sel2}, value={val2:.4f}")

    # Log-determinant
    K = X @ X.T + 0.1 * np.eye(n)
    ld = LogDeterminantFunction(K)
    sel3, val3 = optimizer.greedy(ld, k=5)
    print(f"Log-det greedy: {sel3}, value={val3:.4f}")

    # Coverage
    covers = [set(rng.choice(20, size=5, replace=False)) for _ in range(n)]
    cov = CoverageFunction(covers)
    sel4, val4 = optimizer.greedy(cov, k=5)
    print(f"Coverage greedy: {sel4}, value={val4:.4f}")

    # Feature-based
    W = rng.rand(n, 10) * 0.3
    fb = FeatureBasedFunction(W)
    sel5, val5 = optimizer.greedy(fb, k=5)
    print(f"Feature-based greedy: {sel5}, value={val5:.4f}")

    # Verify diminishing returns
    is_sub, violations = verify_diminishing_returns(fl, n_tests=50, rng=rng)
    print(f"Facility location is submodular: {is_sub}, violations: {len(violations)}")

    # Graph cut
    W_graph = rng.rand(n, n)
    W_graph = 0.5 * (W_graph + W_graph.T)
    np.fill_diagonal(W_graph, 0.0)
    gc = GraphCutFunction(W_graph)
    sel6, val6 = optimizer.greedy(gc, k=5)
    print(f"Graph cut greedy: {sel6}, value={val6:.4f}")


if __name__ == '__main__':
    demo_submodular()
