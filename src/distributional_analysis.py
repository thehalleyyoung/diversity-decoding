"""
Distributional analysis for diversity metrics.

Provides:
  - Bootstrap confidence intervals on Kendall τ estimates
  - Permutation tests for metric independence
  - Empirical metric algebra: equivalence classes under ≈ relation
  - Diagnostic tests for metric redundancy robustness
  - Submodularity verification for selection objectives
  - Berry-Esseen convergence rate for Kendall τ CLT
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """O(n²) Kendall τ."""
    n = len(x)
    conc, disc = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    d = conc + disc
    return (conc - disc) / d if d > 0 else 0.0


def berry_esseen_kendall_tau(n: int, tau: float = 0.0) -> Dict:
    r"""Berry-Esseen convergence rate for Kendall τ estimator.

    Under H₀: τ=0 (independence), the standardized Kendall τ statistic
        Z_n = τ̂ · √(n(n-1)/2) / σ_τ
    converges to N(0,1) at rate:
        sup_t |P(Z_n ≤ t) - Φ(t)| ≤ C / √n

    The Berry-Esseen bound gives:
        |P(Z_n ≤ t) - Φ(t)| ≤ C · E[|X|³] / (σ³ · √n)

    For Kendall τ with n items:
      - σ²(τ̂) = 2(2n+5) / (9n(n-1))  (under independence)
      - The Berry-Esseen constant C ≤ 0.4748 (Shevtsova 2011)

    Returns convergence rate bound and practical sample size guidance.
    """
    C_BE = 0.4748  # Best known Berry-Esseen constant (Shevtsova 2011)

    # Variance of τ̂ under H₀: independence
    var_tau = 2.0 * (2 * n + 5) / (9.0 * n * (n - 1)) if n > 1 else 1.0
    sigma_tau = math.sqrt(var_tau)

    # Third absolute moment bound (for i.i.d. summands underlying τ̂)
    # Each pair contributes ±1/(n choose 2); third moment ≤ 1/(n choose 2)
    pairs = n * (n - 1) / 2
    rho3 = 1.0 / pairs if pairs > 0 else 1.0

    # Berry-Esseen bound: sup |F_n(t) - Φ(t)| ≤ C·ρ₃/(σ³·√n)
    sigma3 = sigma_tau ** 3 if sigma_tau > 0 else 1.0
    be_bound = C_BE * rho3 / (sigma3 * math.sqrt(n)) if n > 0 else 1.0
    be_bound = min(be_bound, 1.0)  # Trivial bound

    # Practical: n needed for bound < 0.05 (Gaussian approximation reliable)
    # Solve C·ρ₃/(σ³·√n) < 0.05
    n_for_005 = None
    for n_test in range(4, 10001):
        v = 2.0 * (2 * n_test + 5) / (9.0 * n_test * (n_test - 1))
        s3 = v ** 1.5
        p = n_test * (n_test - 1) / 2
        r3 = 1.0 / p
        b = C_BE * r3 / (s3 * math.sqrt(n_test))
        if b < 0.05:
            n_for_005 = n_test
            break

    return {
        "n": n,
        "berry_esseen_bound": round(be_bound, 6),
        "sigma_tau": round(sigma_tau, 6),
        "variance_tau": round(var_tau, 6),
        "berry_esseen_constant": C_BE,
        "n_for_gaussian_reliable": n_for_005,
        "interpretation": (
            f"For n={n} items, the CDF of standardized τ̂ deviates from "
            f"Gaussian by at most {be_bound:.4f}. "
            f"Gaussian approximation reliable (bound<0.05) for n≥{n_for_005}."
        ),
    }


class MetricAlgebra:
    """Algebraic structure over diversity metrics.

    Defines equivalence classes M₁ ~ M₂ iff |τ(M₁,M₂)| ≥ 1 - δ
    and verifies transitivity.
    """

    def __init__(self, metric_values: Dict[str, np.ndarray],
                 delta: float = 0.3):
        """
        Args:
            metric_values: dict mapping metric name to array of values
                           (one per text group, same ordering).
            delta: equivalence threshold (|τ| ≥ 1-δ means equivalent).
        """
        self.metrics = metric_values
        self.delta = delta
        self.names = sorted(metric_values.keys())
        self.tau_matrix = self._compute_tau_matrix()

    def _compute_tau_matrix(self) -> Dict[str, float]:
        """Compute pairwise τ matrix."""
        result = {}
        for i, m1 in enumerate(self.names):
            for j, m2 in enumerate(self.names):
                if i >= j:
                    continue
                tau = _kendall_tau(self.metrics[m1], self.metrics[m2])
                result[f"{m1}_vs_{m2}"] = tau
        return result

    def equivalence_classes(self) -> List[List[str]]:
        """Compute equivalence classes under the ~ relation.

        Uses union-find to group metrics satisfying |τ| ≥ 1-δ.
        """
        parent = {m: m for m in self.names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, m1 in enumerate(self.names):
            for j, m2 in enumerate(self.names):
                if i >= j:
                    continue
                key = f"{m1}_vs_{m2}"
                if key in self.tau_matrix and abs(self.tau_matrix[key]) >= 1 - self.delta:
                    union(m1, m2)

        classes = defaultdict(list)
        for m in self.names:
            classes[find(m)].append(m)
        return list(classes.values())

    def verify_transitivity(self) -> Dict:
        """Verify transitivity of the equivalence relation.

        For all triples (A,B,C) where A~B and B~C, check if A~C.
        Reports any transitivity violations.
        """
        violations = []
        n = len(self.names)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    m1, m2, m3 = self.names[i], self.names[j], self.names[k]
                    tau12 = self.tau_matrix.get(f"{m1}_vs_{m2}", 0)
                    tau23 = self.tau_matrix.get(f"{m2}_vs_{m3}", 0)
                    tau13 = self.tau_matrix.get(f"{m1}_vs_{m3}", 0)

                    if abs(tau12) >= 1 - self.delta and abs(tau23) >= 1 - self.delta:
                        if abs(tau13) < 1 - self.delta:
                            violations.append({
                                'triple': (m1, m2, m3),
                                'tau_12': tau12,
                                'tau_23': tau23,
                                'tau_13': tau13,
                            })

        return {
            'is_transitive': len(violations) == 0,
            'n_violations': len(violations),
            'violations': violations,
            'n_triples_checked': n * (n-1) * (n-2) // 6,
        }

    def quotient_dimension(self) -> int:
        """Dimension of the quotient space M/~."""
        return len(self.equivalence_classes())

    def summary(self) -> Dict:
        """Full algebra summary."""
        classes = self.equivalence_classes()
        trans = self.verify_transitivity()
        return {
            'n_metrics': len(self.names),
            'n_equivalence_classes': len(classes),
            'equivalence_classes': classes,
            'quotient_dimension': len(classes),
            'delta': self.delta,
            'is_transitive': trans['is_transitive'],
            'transitivity_violations': trans['n_violations'],
            'tau_matrix': {k: round(v, 4) for k, v in self.tau_matrix.items()},
        }


class SubmodularityVerifier:
    """Verify submodularity of set functions on finite ground sets.

    For f: 2^V → R, submodularity means:
    f(A ∪ {e}) - f(A) ≥ f(B ∪ {e}) - f(B) for all A ⊆ B, e ∉ B.
    """

    def __init__(self, f: Callable, ground_set_size: int):
        self.f = f
        self.n = ground_set_size

    def verify_exact(self, max_subsets: int = 2000) -> Dict:
        """Verify submodularity by exhaustive or random subset checking.

        For small n (≤8), checks all relevant (A, B, e) triples.
        For large n, samples random triples with Clopper-Pearson CI
        on the violation rate.
        """
        rng = np.random.RandomState(42)
        violations = []
        n_checked = 0

        if self.n <= 8:
            # Exhaustive check
            for a_mask in range(1 << self.n):
                A = frozenset(i for i in range(self.n) if a_mask & (1 << i))
                for e in range(self.n):
                    if e in A:
                        continue
                    mg_A = self.f(A | {e}) - self.f(A)
                    # Check all supersets B ⊇ A with e ∉ B
                    for b_extra in range(1 << self.n):
                        B = A | frozenset(i for i in range(self.n)
                                          if b_extra & (1 << i) and i != e)
                        if B == A:
                            continue
                        mg_B = self.f(B | {e}) - self.f(B)
                        n_checked += 1
                        if mg_A < mg_B - 1e-10:
                            violations.append({
                                'A': list(A), 'B': list(B), 'e': e,
                                'mg_A': mg_A, 'mg_B': mg_B,
                                'gap': mg_B - mg_A,
                            })
                            if len(violations) >= 10:
                                break
                    if len(violations) >= 10:
                        break
                if len(violations) >= 10:
                    break
        else:
            # Random sampling for n > 8
            for _ in range(max_subsets):
                a_size = rng.randint(0, self.n - 1)
                A = frozenset(rng.choice(self.n, size=a_size, replace=False).tolist())
                remaining = [i for i in range(self.n) if i not in A]
                if not remaining:
                    continue
                e = remaining[rng.randint(len(remaining))]
                b_extra_size = rng.randint(0, len(remaining))
                B = A | frozenset(rng.choice(
                    [i for i in remaining if i != e],
                    size=min(b_extra_size, len(remaining) - 1),
                    replace=False
                ).tolist())
                mg_A = self.f(A | {e}) - self.f(A)
                mg_B = self.f(B | {e}) - self.f(B)
                n_checked += 1
                if mg_A < mg_B - 1e-10:
                    violations.append({
                        'A': list(A), 'B': list(B), 'e': e,
                        'mg_A': mg_A, 'mg_B': mg_B, 'gap': mg_B - mg_A,
                    })

        # Clopper-Pearson CI on violation rate for n > 8
        cp_ci = None
        if self.n > 8 and n_checked > 0:
            from scipy.stats import beta as beta_dist
            k_viol = len(violations)
            alpha_cp = 0.05
            if k_viol == 0:
                cp_lower = 0.0
                cp_upper = float(beta_dist.ppf(1 - alpha_cp / 2, 1, n_checked))
            else:
                cp_lower = float(beta_dist.ppf(alpha_cp / 2, k_viol, n_checked - k_viol + 1))
                cp_upper = float(beta_dist.ppf(1 - alpha_cp / 2, k_viol + 1, n_checked - k_viol))
            cp_ci = [round(cp_lower, 6), round(cp_upper, 6)]

        result = {
            'is_submodular': len(violations) == 0,
            'n_checked': n_checked,
            'n_violations': len(violations),
            'max_violation_gap': max((v['gap'] for v in violations), default=0.0),
            'violations': violations[:5],
            'ground_set_size': self.n,
            'method': 'exhaustive' if self.n <= 8 else f'random_sampling_{max_subsets}',
        }
        if cp_ci is not None:
            result['violation_rate_ci_95'] = cp_ci
        return result


class PermutationTest:
    """Permutation test for metric independence.

    Tests H₀: τ(M₁, M₂) = 0 against H₁: τ(M₁, M₂) ≠ 0
    using a permutation distribution.
    """

    def __init__(self, n_permutations: int = 10000, seed: int = 42):
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(seed)

    def test(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Run permutation test for Kendall τ independence.

        Returns:
            Dict with observed_tau, p_value, critical_values, permutation_std.
        """
        x, y = np.asarray(x), np.asarray(y)
        observed = _kendall_tau(x, y)

        perm_taus = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            y_perm = self.rng.permutation(y)
            perm_taus[i] = _kendall_tau(x, y_perm)

        p_value = np.mean(np.abs(perm_taus) >= np.abs(observed))

        return {
            'observed_tau': float(observed),
            'p_value': float(p_value),
            'permutation_mean': float(np.mean(perm_taus)),
            'permutation_std': float(np.std(perm_taus)),
            'critical_95': float(np.percentile(np.abs(perm_taus), 95)),
            'critical_99': float(np.percentile(np.abs(perm_taus), 99)),
            'n_permutations': self.n_permutations,
            'is_significant_05': float(p_value) < 0.05,
            'is_significant_01': float(p_value) < 0.01,
        }


def tightness_construction(n: int, eps1: float, eps2: float) -> Dict:
    """Construct instances achieving the tight bound in Theorem 4.1.

    For given ε₁, ε₂, constructs rankings R, M₁ = g₁∘R, M₂ = g₂∘R
    where g₁ and g₂ are ε-approximately monotone, achieving
    |τ(M₁, M₂)| = 1 - 2(ε₁ + ε₂ - ε₁ε₂).

    The construction uses disjoint adjacent-pair swaps so that
    D₁ ∩ D₂ = ∅ (the discordance sets are disjoint).

    Returns:
        Dict with construction details, achieved τ, and theoretical bound.
    """
    R = np.arange(n, dtype=float)
    total_pairs = n * (n - 1) // 2

    # Number of adjacent pairs to swap for each metric
    n_swap_1 = max(0, int(eps1 * total_pairs))
    n_swap_2 = max(0, int(eps2 * total_pairs))

    # Construct g₁: swap n_swap_1 adjacent pairs from the beginning
    M1 = R.copy()
    flipped = 0
    for i in range(0, n - 1, 2):
        if flipped >= n_swap_1:
            break
        M1[i], M1[i+1] = M1[i+1], M1[i]
        flipped += 1

    # Construct g₂: swap n_swap_2 adjacent pairs starting after g₁'s swaps
    M2 = R.copy()
    flipped = 0
    start_offset = (n_swap_1 * 2) + 1 if n_swap_1 * 2 + 1 < n else 1
    for i in range(start_offset, n - 1, 2):
        if flipped >= n_swap_2:
            break
        M2[i], M2[i+1] = M2[i+1], M2[i]
        flipped += 1

    achieved_tau = _kendall_tau(M1, M2)

    # Measure actual ε-approximate monotonicity
    actual_eps1 = 0
    actual_eps2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (M1[i] - M1[j]) * (R[i] - R[j]) < 0:
                actual_eps1 += 1
            if (M2[i] - M2[j]) * (R[i] - R[j]) < 0:
                actual_eps2 += 1

    actual_eps1 = actual_eps1 / total_pairs if total_pairs > 0 else 0
    actual_eps2 = actual_eps2 / total_pairs if total_pairs > 0 else 0

    theoretical_bound = 1 - 2 * (actual_eps1 + actual_eps2)

    return {
        'n': n,
        'eps1_target': eps1,
        'eps2_target': eps2,
        'eps1_actual': actual_eps1,
        'eps2_actual': actual_eps2,
        'achieved_tau': float(achieved_tau),
        'theoretical_bound': float(theoretical_bound),
        'gap': float(abs(achieved_tau) - theoretical_bound),
        'bound_is_tight': float(abs(achieved_tau) - theoretical_bound) < 0.05,
    }


# ---------------------------------------------------------------------------
# Information-theoretic baselines for metric comparison
# ---------------------------------------------------------------------------

def _discretize_rankings(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize continuous metric values into bins for MI computation."""
    bins = np.linspace(values.min() - 1e-10, values.max() + 1e-10, n_bins + 1)
    return np.digitize(values, bins) - 1


def normalized_mutual_information(x: np.ndarray, y: np.ndarray,
                                   n_bins: int = 10) -> float:
    """Normalized Mutual Information between two metric vectors.

    NMI = 2 * MI(X,Y) / (H(X) + H(Y)), ranges [0, 1].
    Provides an information-theoretic complement to Kendall τ
    that captures nonlinear dependencies.
    """
    import math
    xd = _discretize_rankings(x, n_bins)
    yd = _discretize_rankings(y, n_bins)
    n = len(x)

    # Joint distribution
    joint = {}
    for i in range(n):
        key = (int(xd[i]), int(yd[i]))
        joint[key] = joint.get(key, 0) + 1

    # Marginals
    px = {}
    py = {}
    for i in range(n):
        px[int(xd[i])] = px.get(int(xd[i]), 0) + 1
        py[int(yd[i])] = py.get(int(yd[i]), 0) + 1

    # Entropies
    h_x = -sum((c / n) * math.log2(c / n) for c in px.values())
    h_y = -sum((c / n) * math.log2(c / n) for c in py.values())

    # Mutual information
    mi = 0.0
    for (xi, yi), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((px[xi] / n) * (py[yi] / n)))

    denom = h_x + h_y
    return 2 * mi / denom if denom > 0 else 0.0


def variation_of_information(x: np.ndarray, y: np.ndarray,
                              n_bins: int = 10) -> float:
    """Variation of Information between two metric vectors.

    VI(X,Y) = H(X) + H(Y) - 2*MI(X,Y), a proper metric on partitions.
    Lower VI means more information shared between metrics.
    """
    import math
    xd = _discretize_rankings(x, n_bins)
    yd = _discretize_rankings(y, n_bins)
    n = len(x)

    joint = {}
    for i in range(n):
        key = (int(xd[i]), int(yd[i]))
        joint[key] = joint.get(key, 0) + 1

    px = {}
    py = {}
    for i in range(n):
        px[int(xd[i])] = px.get(int(xd[i]), 0) + 1
        py[int(yd[i])] = py.get(int(yd[i]), 0) + 1

    h_x = -sum((c / n) * math.log2(c / n) for c in px.values())
    h_y = -sum((c / n) * math.log2(c / n) for c in py.values())

    mi = 0.0
    for (xi, yi), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((px[xi] / n) * (py[yi] / n)))

    return h_x + h_y - 2 * mi


def info_theoretic_metric_comparison(
    metric_values: Dict[str, np.ndarray],
    n_bins: int = 10,
) -> Dict:
    """Compare metrics using NMI and VI alongside Kendall τ.

    Provides a richer comparison than τ alone by detecting
    nonlinear dependencies (NMI) and quantifying information
    distance (VI).
    """
    names = sorted(metric_values.keys())
    results = {
        'metrics': names,
        'comparisons': [],
        'nmi_matrix': {},
        'vi_matrix': {},
        'tau_matrix': {},
    }

    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i >= j:
                continue
            x = metric_values[m1]
            y = metric_values[m2]

            tau = _kendall_tau(x, y)
            nmi = normalized_mutual_information(x, y, n_bins)
            vi = variation_of_information(x, y, n_bins)

            key = f"{m1}_vs_{m2}"
            results['nmi_matrix'][key] = round(nmi, 4)
            results['vi_matrix'][key] = round(vi, 4)
            results['tau_matrix'][key] = round(tau, 4)
            results['comparisons'].append({
                'pair': key,
                'kendall_tau': round(tau, 4),
                'nmi': round(nmi, 4),
                'vi': round(vi, 4),
                'tau_indicates_redundant': abs(tau) > 0.7,
                'nmi_indicates_redundant': nmi > 0.5,
            })

    # Summary statistics
    nmi_vals = [c['nmi'] for c in results['comparisons']]
    vi_vals = [c['vi'] for c in results['comparisons']]
    tau_vals = [abs(c['kendall_tau']) for c in results['comparisons']]
    results['summary'] = {
        'n_pairs': len(results['comparisons']),
        'mean_nmi': round(float(np.mean(nmi_vals)), 4) if nmi_vals else 0,
        'mean_vi': round(float(np.mean(vi_vals)), 4) if vi_vals else 0,
        'mean_abs_tau': round(float(np.mean(tau_vals)), 4) if tau_vals else 0,
        'redundant_by_tau': sum(1 for t in tau_vals if t > 0.7),
        'redundant_by_nmi': sum(1 for n in nmi_vals if n > 0.5),
        'tau_nmi_agreement': sum(
            1 for c in results['comparisons']
            if c['tau_indicates_redundant'] == c['nmi_indicates_redundant']
        ),
    }
    return results
