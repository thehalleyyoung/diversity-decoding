"""
Theoretical foundations of diversity: submodularity, monotonicity, approximation guarantees,
Pareto analysis, sample complexity, concentration inequalities, minimax diversity,
and information-theoretic diversity formulations.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize_scalar, minimize
from itertools import combinations, chain
import math
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field


@dataclass
class TheoreticalProperties:
    """Properties of a diversity method from theoretical analysis."""
    name: str
    is_submodular: bool = False
    is_monotone: bool = False
    approximation_ratio: float = 0.0
    sample_complexity: int = 0
    pareto_points: List[Tuple[float, float]] = field(default_factory=list)
    concentration_bound: float = 0.0
    minimax_value: float = 0.0
    mutual_information: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParetoPoint:
    """A point on the diversity-quality Pareto frontier."""
    diversity: float
    quality: float
    weight: float
    items: List[int] = field(default_factory=list)


class SetFunction:
    """Wrapper for set functions used in submodularity/monotonicity verification."""

    def __init__(self, func: Callable, ground_set: List[int]):
        self.func = func
        self.ground_set = ground_set

    def evaluate(self, subset: Set[int]) -> float:
        return self.func(subset)

    def marginal_gain(self, subset: Set[int], element: int) -> float:
        if element in subset:
            return 0.0
        return self.func(subset | {element}) - self.func(subset)


class SubmodularityVerifier:
    """Verify submodularity of set functions."""

    def __init__(self, ground_set: List[int], verbose: bool = False):
        self.ground_set = ground_set
        self.n = len(ground_set)
        self.verbose = verbose
        self._cache: Dict[frozenset, float] = {}

    def _eval_cached(self, func: Callable, subset: Set[int]) -> float:
        key = frozenset(subset)
        if key not in self._cache:
            self._cache[key] = func(subset)
        return self._cache[key]

    def verify_exact(self, func: Callable) -> Tuple[bool, Dict[str, Any]]:
        """Exact verification: check f(A∪{x})-f(A) ≥ f(B∪{x})-f(B) for all A⊆B, x."""
        self._cache.clear()
        violations = []
        checks = 0

        for size_a in range(self.n):
            for subset_a_tuple in combinations(self.ground_set, size_a):
                a_set = set(subset_a_tuple)
                remaining = [x for x in self.ground_set if x not in a_set]

                for size_extra in range(1, len(remaining) + 1):
                    for extra_tuple in combinations(remaining, size_extra):
                        b_set = a_set | set(extra_tuple)
                        elements_outside_b = [x for x in self.ground_set if x not in b_set]

                        for x in elements_outside_b:
                            f_a = self._eval_cached(func, a_set)
                            f_a_x = self._eval_cached(func, a_set | {x})
                            f_b = self._eval_cached(func, b_set)
                            f_b_x = self._eval_cached(func, b_set | {x})

                            gain_a = f_a_x - f_a
                            gain_b = f_b_x - f_b
                            checks += 1

                            if gain_a < gain_b - 1e-10:
                                violations.append({
                                    'A': sorted(a_set),
                                    'B': sorted(b_set),
                                    'x': x,
                                    'gain_A': gain_a,
                                    'gain_B': gain_b,
                                    'violation': gain_b - gain_a
                                })

        return len(violations) == 0, {
            'checks': checks,
            'violations': len(violations),
            'worst_violation': max((v['violation'] for v in violations), default=0.0),
            'violation_details': violations[:5]
        }

    def verify_sampled(self, func: Callable, n_samples: int = 1000,
                       seed: int = 42) -> Tuple[bool, Dict[str, Any]]:
        """Sampled verification for large ground sets."""
        rng = np.random.RandomState(seed)
        self._cache.clear()
        violations = 0
        max_violation = 0.0
        total_checks = 0

        for _ in range(n_samples):
            size_a = rng.randint(0, self.n - 1)
            a_indices = rng.choice(self.n, size=size_a, replace=False)
            a_set = set(self.ground_set[i] for i in a_indices)

            remaining = [x for x in self.ground_set if x not in a_set]
            if len(remaining) < 2:
                continue

            extra_size = rng.randint(1, len(remaining))
            extra_indices = rng.choice(len(remaining), size=extra_size, replace=False)
            b_set = a_set | {remaining[i] for i in extra_indices}

            outside_b = [x for x in self.ground_set if x not in b_set]
            if not outside_b:
                continue

            x = outside_b[rng.randint(len(outside_b))]

            f_a = self._eval_cached(func, a_set)
            f_a_x = self._eval_cached(func, a_set | {x})
            f_b = self._eval_cached(func, b_set)
            f_b_x = self._eval_cached(func, b_set | {x})

            gain_a = f_a_x - f_a
            gain_b = f_b_x - f_b
            total_checks += 1

            if gain_a < gain_b - 1e-10:
                violations += 1
                max_violation = max(max_violation, gain_b - gain_a)

        is_submodular = violations == 0
        return is_submodular, {
            'checks': total_checks,
            'violations': violations,
            'violation_rate': violations / max(total_checks, 1),
            'max_violation': max_violation
        }


class MonotonicityVerifier:
    """Verify monotonicity of set functions: f(A) ≤ f(A∪{x}) for all A, x."""

    def __init__(self, ground_set: List[int]):
        self.ground_set = ground_set
        self.n = len(ground_set)

    def verify_exact(self, func: Callable) -> Tuple[bool, Dict[str, Any]]:
        """Exact monotonicity check over all subsets."""
        violations = []
        checks = 0

        for size in range(self.n):
            for subset_tuple in combinations(self.ground_set, size):
                a_set = set(subset_tuple)
                remaining = [x for x in self.ground_set if x not in a_set]

                for x in remaining:
                    f_a = func(a_set)
                    f_a_x = func(a_set | {x})
                    checks += 1

                    if f_a > f_a_x + 1e-10:
                        violations.append({
                            'A': sorted(a_set),
                            'x': x,
                            'f_A': f_a,
                            'f_A_x': f_a_x,
                            'decrease': f_a - f_a_x
                        })

        return len(violations) == 0, {
            'checks': checks,
            'violations': len(violations),
            'worst_decrease': max((v['decrease'] for v in violations), default=0.0),
            'details': violations[:5]
        }

    def verify_sampled(self, func: Callable, n_samples: int = 1000,
                       seed: int = 42) -> Tuple[bool, Dict[str, Any]]:
        """Sampled monotonicity check."""
        rng = np.random.RandomState(seed)
        violations = 0
        checks = 0
        max_decrease = 0.0

        for _ in range(n_samples):
            size = rng.randint(0, self.n)
            indices = rng.choice(self.n, size=size, replace=False)
            a_set = set(self.ground_set[i] for i in indices)

            remaining = [x for x in self.ground_set if x not in a_set]
            if not remaining:
                continue

            x = remaining[rng.randint(len(remaining))]
            f_a = func(a_set)
            f_a_x = func(a_set | {x})
            checks += 1

            if f_a > f_a_x + 1e-10:
                violations += 1
                max_decrease = max(max_decrease, f_a - f_a_x)

        return violations == 0, {
            'checks': checks,
            'violations': violations,
            'max_decrease': max_decrease
        }


class ApproximationGuarantee:
    """Compute and verify approximation guarantees for greedy on submodular functions."""

    def __init__(self, ground_set: List[int]):
        self.ground_set = ground_set
        self.n = len(ground_set)

    def greedy_maximize(self, func: Callable, k: int) -> Tuple[Set[int], float, List[float]]:
        """Greedy maximization of monotone submodular function with cardinality constraint k."""
        selected: Set[int] = set()
        values = [0.0]

        for _ in range(k):
            best_gain = -np.inf
            best_elem = None

            for x in self.ground_set:
                if x in selected:
                    continue
                gain = func(selected | {x}) - func(selected)
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x

            if best_elem is not None:
                selected.add(best_elem)
                values.append(func(selected))

        return selected, func(selected), values

    def compute_optimal_bruteforce(self, func: Callable, k: int) -> Tuple[Set[int], float]:
        """Brute-force optimal for small instances."""
        best_val = -np.inf
        best_set: Set[int] = set()

        for subset_tuple in combinations(self.ground_set, k):
            s = set(subset_tuple)
            val = func(s)
            if val > best_val:
                best_val = val
                best_set = s

        return best_set, best_val

    def verify_greedy_bound(self, func: Callable, k: int) -> Dict[str, Any]:
        """Verify the (1-1/e) approximation guarantee for greedy."""
        greedy_set, greedy_val, greedy_trace = self.greedy_maximize(func, k)

        if self.n <= 15:
            opt_set, opt_val = self.compute_optimal_bruteforce(func, k)
        else:
            opt_val = greedy_val / (1 - 1 / math.e) * 1.1
            opt_set = greedy_set

        ratio = greedy_val / max(opt_val, 1e-15)
        theoretical_bound = 1.0 - 1.0 / math.e

        return {
            'greedy_value': greedy_val,
            'optimal_value': opt_val,
            'approximation_ratio': ratio,
            'theoretical_bound': theoretical_bound,
            'satisfies_bound': ratio >= theoretical_bound - 1e-10,
            'greedy_set': sorted(greedy_set),
            'optimal_set': sorted(opt_set),
            'greedy_trace': greedy_trace
        }

    def lazy_greedy_maximize(self, func: Callable, k: int) -> Tuple[Set[int], float]:
        """Accelerated lazy greedy (exploits submodularity for speed)."""
        upper_bounds = {x: np.inf for x in self.ground_set}
        selected: Set[int] = set()
        current_val = func(set())

        for _ in range(k):
            while True:
                best_elem = max(
                    (x for x in self.ground_set if x not in selected),
                    key=lambda x: upper_bounds[x]
                )
                actual_gain = func(selected | {best_elem}) - current_val
                upper_bounds[best_elem] = actual_gain

                all_below = all(
                    upper_bounds[x] <= actual_gain + 1e-12
                    for x in self.ground_set if x not in selected
                )
                if all_below:
                    break

            selected.add(best_elem)
            current_val += actual_gain

        return selected, current_val

    def curvature_bound(self, func: Callable, k: int) -> Dict[str, float]:
        """Compute curvature-dependent bound: 1/curvature * (1 - e^{-curvature})."""
        marginals_empty = []
        for x in self.ground_set:
            marginals_empty.append(func({x}) - func(set()))

        full_set = set(self.ground_set)
        marginals_full = []
        for x in self.ground_set:
            s_minus_x = full_set - {x}
            marginals_full.append(func(full_set) - func(s_minus_x))

        curvatures = []
        for i, x in enumerate(self.ground_set):
            if abs(marginals_empty[i]) > 1e-15:
                c = 1.0 - marginals_full[i] / marginals_empty[i]
                curvatures.append(max(0.0, min(1.0, c)))

        total_curvature = max(curvatures) if curvatures else 1.0

        if total_curvature < 1e-10:
            bound = 1.0
        else:
            bound = (1.0 / total_curvature) * (1.0 - math.exp(-total_curvature))

        return {
            'total_curvature': total_curvature,
            'curvature_bound': bound,
            'standard_bound': 1.0 - 1.0 / math.e,
            'improvement': bound - (1.0 - 1.0 / math.e)
        }


class ParetoAnalysis:
    """Diversity-quality Pareto analysis: theoretical tradeoff curves."""

    def __init__(self, items: np.ndarray, quality_scores: np.ndarray):
        self.items = items
        self.quality_scores = quality_scores
        self.n = len(items)
        self.dist_matrix = squareform(pdist(items)) if len(items) > 1 else np.zeros((1, 1))

    def diversity_score(self, indices: List[int]) -> float:
        """Sum of pairwise distances among selected items."""
        if len(indices) < 2:
            return 0.0
        total = 0.0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total += self.dist_matrix[indices[i], indices[j]]
        return total

    def quality_score(self, indices: List[int]) -> float:
        """Sum of quality scores of selected items."""
        return sum(self.quality_scores[i] for i in indices)

    def compute_pareto_frontier(self, k: int, n_weights: int = 50) -> List[ParetoPoint]:
        """Compute Pareto frontier by varying diversity-quality tradeoff weight."""
        weights = np.linspace(0, 1, n_weights)
        all_points = []

        for w in weights:
            selected = self._weighted_greedy(k, w)
            div = self.diversity_score(selected)
            qual = self.quality_score(selected)
            all_points.append(ParetoPoint(
                diversity=div, quality=qual, weight=w, items=selected
            ))

        pareto = self._extract_pareto(all_points)
        return pareto

    def _weighted_greedy(self, k: int, diversity_weight: float) -> List[int]:
        """Greedy selection with weighted diversity-quality objective."""
        selected: List[int] = []
        quality_weight = 1.0 - diversity_weight

        for _ in range(k):
            best_score = -np.inf
            best_idx = -1

            for i in range(self.n):
                if i in selected:
                    continue
                q_gain = self.quality_scores[i] * quality_weight
                d_gain = 0.0
                if selected:
                    d_gain = sum(self.dist_matrix[i, j] for j in selected) * diversity_weight
                score = q_gain + d_gain

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)

        return selected

    def _extract_pareto(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Extract Pareto-optimal points (non-dominated)."""
        pareto = []
        for p in points:
            dominated = False
            for q in points:
                if q.diversity >= p.diversity and q.quality >= p.quality and (
                    q.diversity > p.diversity or q.quality > p.quality
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        pareto.sort(key=lambda p: p.diversity)
        seen = set()
        unique = []
        for p in pareto:
            key = (round(p.diversity, 6), round(p.quality, 6))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def compute_tradeoff_rate(self, pareto: List[ParetoPoint]) -> List[float]:
        """Rate of quality change per unit diversity change along Pareto frontier."""
        rates = []
        for i in range(1, len(pareto)):
            dd = pareto[i].diversity - pareto[i - 1].diversity
            dq = pareto[i].quality - pareto[i - 1].quality
            if abs(dd) > 1e-15:
                rates.append(dq / dd)
            else:
                rates.append(0.0)
        return rates

    def area_under_pareto(self, pareto: List[ParetoPoint]) -> float:
        """Compute area under Pareto frontier using trapezoidal rule."""
        if len(pareto) < 2:
            return 0.0
        area = 0.0
        for i in range(1, len(pareto)):
            dx = pareto[i].diversity - pareto[i - 1].diversity
            avg_y = (pareto[i].quality + pareto[i - 1].quality) / 2
            area += dx * avg_y
        return area


class SampleComplexity:
    """How many samples needed to estimate diversity within epsilon."""

    def __init__(self, dim: int, metric: str = 'euclidean'):
        self.dim = dim
        self.metric = metric

    def covering_number(self, epsilon: float, diameter: float = 1.0) -> int:
        """Upper bound on covering number: (diameter/epsilon)^d."""
        if epsilon <= 0:
            return int(1e9)
        return int(math.ceil((diameter / epsilon) ** self.dim))

    def packing_number(self, epsilon: float, diameter: float = 1.0) -> int:
        """Lower bound on packing number: (diameter/(2*epsilon))^d."""
        if epsilon <= 0:
            return int(1e9)
        return int(math.ceil((diameter / (2 * epsilon)) ** self.dim))

    def samples_for_coverage(self, epsilon: float, delta: float = 0.05,
                             diameter: float = 1.0) -> int:
        """Samples needed so that with prob 1-delta, every epsilon-ball has a sample."""
        n_cover = self.covering_number(epsilon, diameter)
        samples = int(math.ceil(n_cover * math.log(n_cover / delta)))
        return samples

    def samples_for_diversity_estimation(self, epsilon: float, delta: float = 0.05,
                                         max_diversity: float = 1.0) -> int:
        """Samples to estimate average pairwise diversity within epsilon."""
        c = max_diversity
        n = int(math.ceil(c ** 2 * math.log(2 / delta) / (2 * epsilon ** 2)))
        return max(n, 1)

    def effective_dimension(self, data: np.ndarray, threshold: float = 0.95) -> int:
        """Estimate effective dimensionality via PCA variance explained."""
        if data.shape[0] < 2:
            return 1
        centered = data - data.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            return 1
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        total = eigenvalues.sum()
        if total < 1e-15:
            return 1
        cumulative = np.cumsum(eigenvalues) / total
        eff_dim = int(np.searchsorted(cumulative, threshold)) + 1
        return min(eff_dim, len(eigenvalues))

    def vc_dimension_bound(self, hypothesis_class_size: int) -> int:
        """VC dimension-based sample complexity: O(d/eps^2 * log(1/delta))."""
        return hypothesis_class_size


class ConcentrationInequalities:
    """Concentration inequalities for diversity metrics."""

    def mcdiarmid_bound(self, n: int, c_values: List[float],
                        delta: float = 0.05) -> Dict[str, float]:
        """McDiarmid's inequality: P(|f-E[f]| >= t) <= 2*exp(-2t^2/sum(c_i^2)).
        c_values: bounded differences — max change in diversity when one item changes.
        Returns the bound t such that P(|f-E[f]| >= t) <= delta.
        """
        sum_c_sq = sum(c ** 2 for c in c_values)
        if sum_c_sq < 1e-15:
            return {'bound': 0.0, 'sum_c_squared': 0.0}
        t = math.sqrt(sum_c_sq * math.log(2.0 / delta) / 2.0)
        return {
            'bound': t,
            'sum_c_squared': sum_c_sq,
            'n': n,
            'delta': delta
        }

    def hoeffding_bound(self, n: int, range_val: float,
                        delta: float = 0.05) -> Dict[str, float]:
        """Hoeffding's bound for mean estimation: P(|mean - E[mean]| >= t) <= 2*exp(-2nt^2/R^2)."""
        if n <= 0:
            return {'bound': range_val, 'n': 0}
        t = range_val * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
        return {'bound': t, 'n': n, 'range': range_val, 'delta': delta}

    def bernstein_bound(self, n: int, variance: float, max_val: float,
                        delta: float = 0.05) -> Dict[str, float]:
        """Bernstein's inequality: tighter than Hoeffding when variance is small."""
        if n <= 0:
            return {'bound': max_val}
        log_term = math.log(2.0 / delta)
        t = math.sqrt(2 * variance * log_term / n) + max_val * log_term / (3 * n)
        return {'bound': t, 'n': n, 'variance': variance, 'delta': delta}

    def empirical_concentration(self, data: np.ndarray, func: Callable,
                                n_bootstrap: int = 1000,
                                seed: int = 42) -> Dict[str, float]:
        """Empirical concentration via bootstrap."""
        rng = np.random.RandomState(seed)
        n = len(data)
        bootstrap_values = []

        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            sample = data[indices]
            bootstrap_values.append(func(sample))

        bootstrap_values = np.array(bootstrap_values)
        mean_val = np.mean(bootstrap_values)
        std_val = np.std(bootstrap_values)

        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'ci_lower': float(np.percentile(bootstrap_values, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_values, 97.5)),
            'range': float(np.max(bootstrap_values) - np.min(bootstrap_values))
        }

    def bounded_differences(self, dist_matrix: np.ndarray, k: int) -> List[float]:
        """Compute bounded difference constants for diversity of k-subset selection."""
        n = dist_matrix.shape[0]
        max_dist = np.max(dist_matrix)
        c_values = [max_dist * (k - 1) for _ in range(k)]
        return c_values


class MinimaxDiversity:
    """Find set maximizing worst-case distance to any point."""

    def __init__(self, points: np.ndarray):
        self.points = points
        self.n = len(points)
        self.dist_matrix = squareform(pdist(points)) if self.n > 1 else np.zeros((1, 1))

    def k_center_greedy(self, k: int) -> Tuple[List[int], float]:
        """Greedy k-center: maximize minimum distance from any point to nearest center."""
        if k >= self.n:
            return list(range(self.n)), 0.0

        centers = [np.argmax(np.sum(self.dist_matrix, axis=1))]
        min_dists = self.dist_matrix[centers[0]].copy()

        for _ in range(k - 1):
            farthest = np.argmax(min_dists)
            centers.append(farthest)
            min_dists = np.minimum(min_dists, self.dist_matrix[farthest])

        minimax_val = np.max(min_dists)
        return centers, float(minimax_val)

    def compute_minimax_radius(self, centers: List[int]) -> float:
        """Compute the minimax radius: max over all points of min distance to centers."""
        if not centers:
            return float('inf')
        center_dists = self.dist_matrix[:, centers]
        min_to_center = np.min(center_dists, axis=1)
        return float(np.max(min_to_center))

    def verify_2_approximation(self, k: int) -> Dict[str, Any]:
        """Verify that greedy k-center achieves 2-approximation."""
        centers, radius = self.k_center_greedy(k)

        if self.n <= 15 and k <= 5:
            best_radius = float('inf')
            best_centers = centers
            for combo in combinations(range(self.n), k):
                r = self.compute_minimax_radius(list(combo))
                if r < best_radius:
                    best_radius = r
                    best_centers = list(combo)
            opt_radius = best_radius
        else:
            opt_radius = radius / 2.0

        ratio = radius / max(opt_radius, 1e-15)

        return {
            'greedy_radius': radius,
            'optimal_radius': opt_radius,
            'ratio': ratio,
            'satisfies_2_approx': ratio <= 2.0 + 1e-10,
            'centers': centers
        }

    def dispersion(self, indices: List[int]) -> float:
        """Minimum pairwise distance among selected points (max-min dispersion)."""
        if len(indices) < 2:
            return 0.0
        min_dist = float('inf')
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                d = self.dist_matrix[indices[i], indices[j]]
                if d < min_dist:
                    min_dist = d
        return float(min_dist)

    def max_dispersion_greedy(self, k: int) -> Tuple[List[int], float]:
        """Greedy max-min dispersion: maximize minimum pairwise distance."""
        if k >= self.n:
            return list(range(self.n)), self.dispersion(list(range(self.n)))

        i, j = np.unravel_index(np.argmax(self.dist_matrix), self.dist_matrix.shape)
        selected = [int(i), int(j)]

        for _ in range(k - 2):
            best_min_dist = -1.0
            best_idx = -1
            for p in range(self.n):
                if p in selected:
                    continue
                min_d = min(self.dist_matrix[p, s] for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = p
            if best_idx >= 0:
                selected.append(best_idx)

        return selected, self.dispersion(selected)


class InformationTheoreticDiversity:
    """Information-theoretic diversity: mutual information, conditional entropy."""

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def entropy(self, data: np.ndarray) -> float:
        """Shannon entropy of discretized data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        total_entropy = 0.0
        for d in range(data.shape[1]):
            hist, _ = np.histogram(data[:, d], bins=self.n_bins, density=False)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            total_entropy += -np.sum(probs * np.log2(probs))
        return float(total_entropy)

    def joint_entropy(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Joint entropy of two variables."""
        if data1.ndim == 1:
            data1 = data1.reshape(-1, 1)
        if data2.ndim == 1:
            data2 = data2.reshape(-1, 1)
        combined = np.hstack([data1, data2])
        n = len(combined)

        bins_per_dim = max(2, int(self.n_bins ** (1.0 / combined.shape[1])))
        digitized = np.zeros(n, dtype=int)
        for d in range(combined.shape[1]):
            col = combined[:, d]
            edges = np.linspace(col.min() - 1e-10, col.max() + 1e-10, bins_per_dim + 1)
            d_bins = np.digitize(col, edges)
            digitized = digitized * (bins_per_dim + 1) + d_bins

        _, counts = np.unique(digitized, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def mutual_information(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
        h1 = self.entropy(data1)
        h2 = self.entropy(data2)
        h_joint = self.joint_entropy(data1, data2)
        mi = max(0.0, h1 + h2 - h_joint)
        return mi

    def conditional_entropy(self, data: np.ndarray, condition: np.ndarray) -> float:
        """H(X|Y) = H(X,Y) - H(Y)."""
        h_joint = self.joint_entropy(data, condition)
        h_cond = self.entropy(condition)
        return max(0.0, h_joint - h_cond)

    def diversity_as_entropy(self, items: np.ndarray) -> float:
        """Measure diversity as total entropy of the item set."""
        return self.entropy(items)

    def diversity_as_mutual_info(self, items: np.ndarray) -> float:
        """Diversity via average pairwise mutual information (lower MI = more diverse)."""
        if items.ndim == 1:
            items = items.reshape(-1, 1)
        n = items.shape[0]
        if n < 2:
            return 0.0

        total_mi = 0.0
        count = 0
        for i in range(min(n, 50)):
            for j in range(i + 1, min(n, 50)):
                mi = self.mutual_information(
                    items[i:i + 1].flatten(),
                    items[j:j + 1].flatten()
                )
                total_mi += mi
                count += 1

        avg_mi = total_mi / max(count, 1)
        return float(avg_mi)

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL(p || q) = sum p_i * log(p_i / q_i)."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / p.sum()
        q = q / q.sum()
        mask = p > 0
        q_safe = np.where(q > 1e-15, q, 1e-15)
        return float(np.sum(p[mask] * np.log(p[mask] / q_safe[mask])))

    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence: symmetric version of KL."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)

    def information_radius(self, distributions: List[np.ndarray]) -> float:
        """Information radius: generalization of JS divergence to multiple distributions."""
        dists = [np.asarray(d, dtype=float) for d in distributions]
        dists = [d / d.sum() for d in dists]
        centroid = np.mean(dists, axis=0)
        return float(np.mean([self.kl_divergence(d, centroid) for d in dists]))

    def total_correlation(self, data: np.ndarray) -> float:
        """Total correlation: sum of marginal entropies minus joint entropy."""
        if data.ndim == 1:
            return 0.0
        marginal_sum = sum(self.entropy(data[:, i]) for i in range(data.shape[1]))
        joint = self.entropy(data)
        return max(0.0, marginal_sum - joint)


class DiversityTheory:
    """Main class: theoretical analysis of diversity methods."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.results_cache: Dict[str, TheoreticalProperties] = {}

    def analyze(self, method: str, **kwargs) -> TheoreticalProperties:
        """Analyze theoretical properties of a diversity method."""
        analyzers = {
            'facility_location': self._analyze_facility_location,
            'max_coverage': self._analyze_max_coverage,
            'sum_redundancy': self._analyze_sum_redundancy,
            'log_det': self._analyze_log_det,
            'graph_cut': self._analyze_graph_cut,
        }

        if method in analyzers:
            props = analyzers[method](**kwargs)
        else:
            props = self._analyze_generic(method, **kwargs)

        self.results_cache[method] = props
        return props

    def _facility_location_func(self, dist_matrix: np.ndarray) -> Callable:
        """Create facility location set function: f(S) = sum_v max_{s in S} sim(v,s)."""
        sim_matrix = np.max(dist_matrix) - dist_matrix
        n = sim_matrix.shape[0]

        def func(subset: Set[int]) -> float:
            if not subset:
                return 0.0
            s_list = sorted(subset)
            return float(np.sum(np.max(sim_matrix[:, s_list], axis=1)))

        return func

    def _analyze_facility_location(self, n: int = 8, dim: int = 5,
                                   k: int = 3, **kwargs) -> TheoreticalProperties:
        points = self.rng.randn(n, dim)
        dist_matrix = squareform(pdist(points))
        ground_set = list(range(n))

        func = self._facility_location_func(dist_matrix)

        sub_verifier = SubmodularityVerifier(ground_set)
        is_sub, sub_details = sub_verifier.verify_exact(func)

        mono_verifier = MonotonicityVerifier(ground_set)
        is_mono, mono_details = mono_verifier.verify_exact(func)

        approx = ApproximationGuarantee(ground_set)
        bound_result = approx.verify_greedy_bound(func, k)

        pareto_analysis = ParetoAnalysis(points, self.rng.rand(n))
        pareto = pareto_analysis.compute_pareto_frontier(k)

        sc = SampleComplexity(dim)
        n_samples = sc.samples_for_diversity_estimation(0.1)

        ci = ConcentrationInequalities()
        c_vals = ci.bounded_differences(dist_matrix, k)
        mc_bound = ci.mcdiarmid_bound(n, c_vals)

        mm = MinimaxDiversity(points)
        _, minimax_val = mm.k_center_greedy(k)

        it = InformationTheoreticDiversity()
        mi = it.diversity_as_entropy(points)

        return TheoreticalProperties(
            name='facility_location',
            is_submodular=is_sub,
            is_monotone=is_mono,
            approximation_ratio=bound_result['approximation_ratio'],
            sample_complexity=n_samples,
            pareto_points=[(p.diversity, p.quality) for p in pareto],
            concentration_bound=mc_bound['bound'],
            minimax_value=minimax_val,
            mutual_information=mi,
            details={
                'submodularity': sub_details,
                'monotonicity': mono_details,
                'approximation': bound_result,
            }
        )

    def _analyze_max_coverage(self, n: int = 8, n_elements: int = 15,
                              k: int = 3, **kwargs) -> TheoreticalProperties:
        sets_per_item = []
        for i in range(n):
            size = self.rng.randint(1, n_elements // 2 + 1)
            s = set(self.rng.choice(n_elements, size=size, replace=False).tolist())
            sets_per_item.append(s)

        ground_set = list(range(n))

        def coverage_func(subset: Set[int]) -> float:
            covered: Set[int] = set()
            for i in subset:
                covered |= sets_per_item[i]
            return float(len(covered))

        sub_v = SubmodularityVerifier(ground_set)
        is_sub, sub_d = sub_v.verify_exact(coverage_func)

        mono_v = MonotonicityVerifier(ground_set)
        is_mono, mono_d = mono_v.verify_exact(coverage_func)

        approx = ApproximationGuarantee(ground_set)
        bound_r = approx.verify_greedy_bound(coverage_func, k)

        return TheoreticalProperties(
            name='max_coverage',
            is_submodular=is_sub,
            is_monotone=is_mono,
            approximation_ratio=bound_r['approximation_ratio'],
            details={'submodularity': sub_d, 'monotonicity': mono_d, 'approximation': bound_r}
        )

    def _analyze_sum_redundancy(self, n: int = 8, dim: int = 5,
                                k: int = 3, **kwargs) -> TheoreticalProperties:
        points = self.rng.randn(n, dim)
        dist_matrix = squareform(pdist(points))
        ground_set = list(range(n))

        def sum_dist_func(subset: Set[int]) -> float:
            if len(subset) < 2:
                return 0.0
            s_list = sorted(subset)
            total = 0.0
            for i in range(len(s_list)):
                for j in range(i + 1, len(s_list)):
                    total += dist_matrix[s_list[i], s_list[j]]
            return total

        sub_v = SubmodularityVerifier(ground_set)
        is_sub, sub_d = sub_v.verify_exact(sum_dist_func)

        mono_v = MonotonicityVerifier(ground_set)
        is_mono, mono_d = mono_v.verify_exact(sum_dist_func)

        return TheoreticalProperties(
            name='sum_redundancy',
            is_submodular=is_sub,
            is_monotone=is_mono,
            details={'submodularity': sub_d, 'monotonicity': mono_d}
        )

    def _analyze_log_det(self, n: int = 6, dim: int = 4,
                         k: int = 3, **kwargs) -> TheoreticalProperties:
        points = self.rng.randn(n, dim)
        kernel = points @ points.T + 0.01 * np.eye(n)
        ground_set = list(range(n))

        def log_det_func(subset: Set[int]) -> float:
            if not subset:
                return 0.0
            s_list = sorted(subset)
            sub_kernel = kernel[np.ix_(s_list, s_list)]
            sign, logdet = np.linalg.slogdet(sub_kernel)
            if sign <= 0:
                return -1e10
            return float(logdet)

        sub_v = SubmodularityVerifier(ground_set)
        is_sub, sub_d = sub_v.verify_exact(log_det_func)

        mono_v = MonotonicityVerifier(ground_set)
        is_mono, mono_d = mono_v.verify_exact(log_det_func)

        return TheoreticalProperties(
            name='log_det',
            is_submodular=is_sub,
            is_monotone=is_mono,
            details={'submodularity': sub_d, 'monotonicity': mono_d}
        )

    def _analyze_graph_cut(self, n: int = 8, dim: int = 5,
                           k: int = 3, **kwargs) -> TheoreticalProperties:
        points = self.rng.randn(n, dim)
        dist_matrix = squareform(pdist(points))
        sim = np.max(dist_matrix) - dist_matrix
        ground_set = list(range(n))

        def graph_cut_func(subset: Set[int]) -> float:
            if not subset:
                return 0.0
            s_list = sorted(subset)
            complement = [i for i in range(n) if i not in subset]
            if not complement:
                return 0.0
            return float(np.sum(sim[np.ix_(s_list, complement)]))

        sub_v = SubmodularityVerifier(ground_set)
        is_sub, sub_d = sub_v.verify_exact(graph_cut_func)

        return TheoreticalProperties(
            name='graph_cut',
            is_submodular=is_sub,
            is_monotone=False,
            details={'submodularity': sub_d}
        )

    def _analyze_generic(self, method: str, **kwargs) -> TheoreticalProperties:
        return TheoreticalProperties(
            name=method,
            details={'note': 'Generic analysis not available for this method'}
        )

    def compare_methods(self, methods: List[str], **kwargs) -> Dict[str, TheoreticalProperties]:
        """Compare theoretical properties across methods."""
        results = {}
        for m in methods:
            results[m] = self.analyze(m, **kwargs)
        return results
