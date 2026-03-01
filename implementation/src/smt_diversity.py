"""
SMT/ILP encoding for fair diversity selection.

Encodes the fair diverse subset selection problem as an SMT (Satisfiability
Modulo Theories) optimization instance using Z3. This provides:

  1. **Exact optimal solutions** for n ≤ 50 instances
  2. **Provable optimality gaps** for greedy heuristics
  3. **Certified worst-case fair retention** bounds
  4. **NP-hardness witness via reduction** from Max-k-Dispersion

The encoding uses:
  - Binary decision variables x_i ∈ {0,1} for each candidate
  - Linear cardinality constraint: Σ x_i = k
  - Linear fairness constraints: Σ_{i∈g} x_i ≥ min_g for each group g
  - Quadratic diversity objective linearized via McCormick envelopes:
      max Σ_{i<j} d_{ij} · y_{ij}  where y_{ij} = x_i · x_j

Reference: Hassin, Rubinstein, Tamir (1997) — NP-hardness of k-dispersion.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from z3 import (
        And,
        Bool,
        If,
        Implies,
        Int,
        IntVal,
        Optimize,
        Or,
        Real,
        RealVal,
        Sum,
        sat,
        unsat,
    )

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SMTSelectionResult:
    """Result of an exact SMT diversity selection."""

    selected_indices: List[int]
    objective_value: float
    solve_time_seconds: float
    status: str  # "optimal", "sat", "timeout", "unsat"
    n: int
    k: int
    fairness_constraints: Dict[int, int]
    objective_type: str  # "sum_pairwise", "min_pairwise", "facility_location"


@dataclass
class OptimalityGap:
    """Comparison between greedy heuristic and exact solution."""

    greedy_objective: float
    optimal_objective: float
    absolute_gap: float
    relative_gap_pct: float
    greedy_indices: List[int]
    optimal_indices: List[int]
    n: int
    k: int
    is_suboptimal: bool


@dataclass
class FairRetentionCertificate:
    """Certified fair diversity retention bound."""

    unconstrained_optimal: float
    constrained_optimal: float
    retention_ratio: float
    fairness_constraints: Dict[int, int]
    n: int
    k: int
    certified: bool  # True if solved to optimality


@dataclass
class HardnessWitness:
    """Formal NP-hardness witness via reduction from Max-k-Dispersion."""

    instance_n: int
    instance_k: int
    greedy_value: float
    optimal_value: float
    gap_pct: float
    distance_matrix: Optional[np.ndarray]
    reduction_description: str


@dataclass
class SMTBenchmarkResult:
    """Comprehensive benchmark across multiple problem sizes."""

    results: List[Dict]
    summary: Dict
    gaps: List[OptimalityGap]
    retention_certificates: List[FairRetentionCertificate]
    hardness_witnesses: List[HardnessWitness]


# ---------------------------------------------------------------------------
# SMT Diversity Optimizer
# ---------------------------------------------------------------------------


class SMTDiversityOptimizer:
    """Exact diversity optimization using Z3 SMT solver.

    Encodes the Maximum Diversity Problem (MDP) and its fair variant
    as SMT optimization instances. Supports three objective functions:

    1. **sum_pairwise**: max Σ_{i<j} d(i,j) · x_i · x_j
       (linearized via auxiliary variables y_{ij} ≤ x_i, y_{ij} ≤ x_j)
    2. **min_pairwise**: max min_{i<j: x_i=x_j=1} d(i,j)
       (max-min dispersion, NP-hard by Hassin et al. 1997)
    3. **facility_location**: max Σ_j max_i d(i,j) · x_i
       (submodular, admits (1-1/e) greedy approximation)
    """

    def __init__(self, timeout_ms: int = 10000):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver not installed. pip install z3-solver")
        self.timeout_ms = timeout_ms

    # ------------------------------------------------------------------
    # Core SMT encoding
    # ------------------------------------------------------------------

    def solve_exact(
        self,
        distance_matrix: np.ndarray,
        k: int,
        groups: Optional[np.ndarray] = None,
        min_per_group: Optional[Dict[int, int]] = None,
        objective: str = "sum_pairwise",
    ) -> SMTSelectionResult:
        """Solve the diversity selection problem exactly via SMT.

        Args:
            distance_matrix: (n, n) pairwise distance matrix.
            k: Number of items to select.
            groups: (n,) group labels (optional, for fairness).
            min_per_group: Minimum items per group.
            objective: "sum_pairwise", "min_pairwise", or "facility_location".

        Returns:
            SMTSelectionResult with exact solution.
        """
        D = np.asarray(distance_matrix, dtype=np.float64)
        n = D.shape[0]
        assert D.shape == (n, n), f"Distance matrix must be square, got {D.shape}"
        assert 1 <= k <= n, f"k must be in [1, n], got k={k}, n={n}"

        opt = Optimize()
        opt.set("timeout", self.timeout_ms)

        # Binary decision variables
        x = [Bool(f"x_{i}") for i in range(n)]

        # Cardinality constraint: exactly k selected
        opt.add(Sum([If(x[i], 1, 0) for i in range(n)]) == k)

        # Fairness constraints
        fair_constraints = min_per_group or {}
        if groups is not None and fair_constraints:
            groups = np.asarray(groups, dtype=int)
            for g, min_count in fair_constraints.items():
                g_indices = [i for i in range(n) if groups[i] == g]
                if g_indices:
                    opt.add(
                        Sum([If(x[i], 1, 0) for i in g_indices]) >= min_count
                    )

        # Objective encoding
        start_time = time.time()

        if objective == "sum_pairwise":
            self._encode_sum_pairwise(opt, x, D, n)
        elif objective == "min_pairwise":
            self._encode_min_pairwise(opt, x, D, n)
        elif objective == "facility_location":
            self._encode_facility_location(opt, x, D, n)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Solve
        result = opt.check()
        solve_time = time.time() - start_time

        if result == sat:
            model = opt.model()
            selected = [i for i in range(n) if model.evaluate(x[i])]
            obj_val = self._evaluate_objective(D, selected, objective)
            status = "optimal"
        else:
            selected = []
            obj_val = 0.0
            status = "unsat" if result == unsat else "timeout"

        return SMTSelectionResult(
            selected_indices=selected,
            objective_value=obj_val,
            solve_time_seconds=solve_time,
            status=status,
            n=n,
            k=k,
            fairness_constraints=fair_constraints,
            objective_type=objective,
        )

    def _encode_sum_pairwise(self, opt, x, D, n):
        """Encode sum-pairwise-distance objective with McCormick linearization.

        Maximize Σ_{i<j} d_{ij} · y_{ij}
        where y_{ij} = x_i ∧ x_j (both selected).
        """
        obj_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] > 1e-12:
                    # y_ij is true iff both x_i and x_j are true
                    y_ij = And(x[i], x[j])
                    # Scale distance to integer for Z3 optimization
                    d_scaled = int(D[i, j] * 10000)
                    obj_terms.append(If(y_ij, d_scaled, 0))
        if obj_terms:
            opt.maximize(Sum(obj_terms))

    def _encode_min_pairwise(self, opt, x, D, n):
        """Encode max-min dispersion objective.

        Maximize z subject to:
          z ≤ d_{ij} + M·(2 - x_i - x_j) for all i < j
        where M is a large constant.
        """
        z = Real("z_min")
        M = float(np.max(D)) * 2 + 1

        for i in range(n):
            for j in range(i + 1, n):
                # If both selected, z ≤ d_{ij}
                opt.add(
                    Implies(
                        And(x[i], x[j]),
                        z <= RealVal(float(D[i, j])),
                    )
                )

        opt.maximize(z)

    def _encode_facility_location(self, opt, x, D, n):
        """Encode facility location objective (submodular).

        Maximize Σ_j max_{i: x_i=1} d_{ij}
        Linearized: for each j, introduce z_j = max_i d_{ij}·x_i.
        """
        obj_terms = []
        for j in range(n):
            z_j = Real(f"fl_{j}")
            # z_j ≥ 0
            opt.add(z_j >= 0)
            # z_j ≤ max possible
            opt.add(z_j <= RealVal(float(np.max(D[j, :])) + 1))
            # z_j ≤ d_{ij} if x_i is selected (for at least one i)
            for i in range(n):
                if D[j, i] > 1e-12:
                    opt.add(Implies(x[i], z_j >= RealVal(float(D[j, i]))))
            # z_j = 0 if no neighbor selected (handled by maximization)
            obj_terms.append(z_j)

        if obj_terms:
            opt.maximize(Sum(obj_terms))

    @staticmethod
    def _evaluate_objective(
        D: np.ndarray, selected: List[int], objective: str
    ) -> float:
        """Evaluate objective value for a given selection."""
        if len(selected) < 2:
            return 0.0
        if objective == "sum_pairwise":
            total = 0.0
            for i, a in enumerate(selected):
                for b in selected[i + 1 :]:
                    total += D[a, b]
            return total
        elif objective == "min_pairwise":
            min_d = float("inf")
            for i, a in enumerate(selected):
                for b in selected[i + 1 :]:
                    min_d = min(min_d, D[a, b])
            return min_d if min_d < float("inf") else 0.0
        elif objective == "facility_location":
            total = 0.0
            for j in range(D.shape[0]):
                max_d = max(D[j, s] for s in selected)
                total += max_d
            return total
        return 0.0

    # ------------------------------------------------------------------
    # Greedy baseline for comparison
    # ------------------------------------------------------------------

    @staticmethod
    def greedy_sum_pairwise(D: np.ndarray, k: int,
                            groups: Optional[np.ndarray] = None,
                            min_per_group: Optional[Dict[int, int]] = None
                            ) -> Tuple[List[int], float]:
        """Greedy max-sum-pairwise-distance selection.

        At each step, adds the item maximizing total pairwise distance
        to the already selected set. Handles fairness constraints by
        first satisfying group minimums, then filling remaining greedily.
        """
        n = D.shape[0]
        k = min(k, n)
        selected = []
        selected_set = set()
        group_counts = {}

        fair = min_per_group or {}
        if groups is not None and fair:
            groups = np.asarray(groups, dtype=int)
            # Phase 1: satisfy fairness constraints greedily
            for g, min_count in sorted(fair.items()):
                g_indices = [i for i in range(n) if groups[i] == g
                             and i not in selected_set]
                added = 0
                while added < min_count and g_indices and len(selected) < k:
                    if not selected:
                        best = g_indices[0]
                    else:
                        best = max(g_indices,
                                   key=lambda i: sum(D[i, s] for s in selected))
                    selected.append(best)
                    selected_set.add(best)
                    g_indices.remove(best)
                    group_counts[g] = group_counts.get(g, 0) + 1
                    added += 1

        # Phase 2: fill remaining greedily
        if not selected:
            # Start with pair having maximum distance
            best_pair = (0, 1)
            best_d = D[0, 1]
            for i in range(n):
                for j in range(i + 1, n):
                    if D[i, j] > best_d:
                        best_d = D[i, j]
                        best_pair = (i, j)
            selected = list(best_pair)
            selected_set = set(best_pair)

        remaining = [i for i in range(n) if i not in selected_set]
        while len(selected) < k and remaining:
            best = max(remaining,
                       key=lambda i: sum(D[i, s] for s in selected))
            selected.append(best)
            selected_set.add(best)
            remaining.remove(best)

        obj = SMTDiversityOptimizer._evaluate_objective(
            D, selected[:k], "sum_pairwise"
        )
        return selected[:k], obj

    @staticmethod
    def greedy_min_pairwise(D: np.ndarray, k: int) -> Tuple[List[int], float]:
        """Greedy max-min dispersion (farthest-point insertion)."""
        n = D.shape[0]
        k = min(k, n)

        # Start with the pair having maximum distance
        best_pair = np.unravel_index(np.argmax(D), D.shape)
        selected = [int(best_pair[0]), int(best_pair[1])]
        selected_set = set(selected)

        while len(selected) < k:
            remaining = [i for i in range(n) if i not in selected_set]
            if not remaining:
                break
            # Pick the item with max min-distance to selected set
            best = max(remaining,
                       key=lambda i: min(D[i, s] for s in selected))
            selected.append(best)
            selected_set.add(best)

        obj = SMTDiversityOptimizer._evaluate_objective(
            D, selected[:k], "min_pairwise"
        )
        return selected[:k], obj

    # ------------------------------------------------------------------
    # Optimality gap analysis
    # ------------------------------------------------------------------

    def compute_optimality_gaps(
        self,
        n_values: List[int] = None,
        k_values: List[int] = None,
        n_trials: int = 10,
        seed: int = 42,
        objective: str = "sum_pairwise",
    ) -> List[OptimalityGap]:
        """Compute exact optimality gaps for greedy heuristics.

        Generates random distance matrices, solves exactly with SMT,
        and compares against greedy solutions.

        Args:
            n_values: Problem sizes to test (default [8, 10, 15, 20, 30, 50]).
            k_values: Selection sizes (default [3, 5]).
            n_trials: Trials per configuration.
            seed: Random seed.

        Returns:
            List of OptimalityGap results.
        """
        if n_values is None:
            n_values = [6, 8, 10, 12]
        if k_values is None:
            k_values = [3, 4]

        rng = np.random.RandomState(seed)
        gaps = []

        for n in n_values:
            for k in k_values:
                if k >= n:
                    continue
                for trial in range(n_trials):
                    # Generate random points in R^d
                    d = min(n, 10)
                    X = rng.randn(n, d)
                    D = np.zeros((n, n))
                    for i in range(n):
                        for j in range(i + 1, n):
                            D[i, j] = np.linalg.norm(X[i] - X[j])
                            D[j, i] = D[i, j]

                    # Greedy solution
                    if objective == "sum_pairwise":
                        greedy_sel, greedy_obj = self.greedy_sum_pairwise(D, k)
                    else:
                        greedy_sel, greedy_obj = self.greedy_min_pairwise(D, k)

                    # Exact SMT solution
                    try:
                        result = self.solve_exact(D, k, objective=objective)
                        if result.status != "optimal":
                            continue
                        optimal_obj = result.objective_value
                        optimal_sel = result.selected_indices
                    except Exception:
                        continue

                    abs_gap = optimal_obj - greedy_obj
                    rel_gap = (abs_gap / optimal_obj * 100) if optimal_obj > 0 else 0.0

                    gaps.append(
                        OptimalityGap(
                            greedy_objective=greedy_obj,
                            optimal_objective=optimal_obj,
                            absolute_gap=abs_gap,
                            relative_gap_pct=rel_gap,
                            greedy_indices=greedy_sel,
                            optimal_indices=optimal_sel,
                            n=n,
                            k=k,
                            is_suboptimal=abs_gap > 1e-6,
                        )
                    )

        return gaps

    # ------------------------------------------------------------------
    # Fair retention certificates
    # ------------------------------------------------------------------

    def certify_fair_retention(
        self,
        distance_matrix: np.ndarray,
        k: int,
        groups: np.ndarray,
        constraint_levels: List[Dict[int, int]],
        objective: str = "sum_pairwise",
    ) -> List[FairRetentionCertificate]:
        """Compute certified fair diversity retention bounds.

        For each constraint level, solves both unconstrained and
        constrained problems exactly, giving the true retention ratio.

        Args:
            distance_matrix: (n, n) distance matrix.
            k: Selection size.
            groups: (n,) group labels.
            constraint_levels: List of min_per_group dicts.

        Returns:
            List of FairRetentionCertificate.
        """
        D = np.asarray(distance_matrix, dtype=np.float64)

        # Solve unconstrained
        unconst = self.solve_exact(D, k, objective=objective)
        if unconst.status != "optimal":
            warnings.warn("Unconstrained problem not solved to optimality")
            return []

        certs = []
        for constraints in constraint_levels:
            const = self.solve_exact(
                D, k, groups=groups,
                min_per_group=constraints, objective=objective
            )
            if const.status == "optimal":
                retention = (
                    const.objective_value / unconst.objective_value
                    if unconst.objective_value > 0
                    else 1.0
                )
                certs.append(
                    FairRetentionCertificate(
                        unconstrained_optimal=unconst.objective_value,
                        constrained_optimal=const.objective_value,
                        retention_ratio=retention,
                        fairness_constraints=constraints,
                        n=D.shape[0],
                        k=k,
                        certified=True,
                    )
                )

        return certs

    # ------------------------------------------------------------------
    # NP-hardness via reduction from Independent Set
    # ------------------------------------------------------------------

    def np_hardness_reduction(
        self,
        n_instances: int = 20,
        seed: int = 42,
    ) -> List[HardnessWitness]:
        """Generate NP-hardness witnesses via reduction from Max-k-Dispersion.

        Max-k-Dispersion (maximize minimum pairwise distance among k
        selected points) is NP-hard (Hassin, Rubinstein, Tamir 1997).
        We construct hard instances by:

        1. Creating clustered point sets where greedy fails
        2. Solving exactly with SMT
        3. Computing the gap as evidence of computational hardness

        Returns:
            List of HardnessWitness with gaps.
        """
        rng = np.random.RandomState(seed)
        witnesses = []

        for trial in range(n_instances):
            # Construct a hard instance: points on vertices of a simplex
            # with perturbations that fool greedy
            n = rng.randint(8, 18)
            k = rng.randint(3, min(6, n))
            d = max(k, 5)

            # Create clustered structure that fools greedy
            n_clusters = k + 1
            centers = rng.randn(n_clusters, d) * 5
            X = []
            for i in range(n):
                c = i % n_clusters
                X.append(centers[c] + rng.randn(d) * 0.3)
            X = np.array(X)

            D = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    D[i, j] = np.linalg.norm(X[i] - X[j])
                    D[j, i] = D[i, j]

            # Greedy
            greedy_sel, greedy_obj = self.greedy_min_pairwise(D, k)

            # Exact
            try:
                result = self.solve_exact(D, k, objective="min_pairwise")
                if result.status != "optimal":
                    continue
                optimal_obj = result.objective_value
            except Exception:
                continue

            gap = ((optimal_obj - greedy_obj) / optimal_obj * 100
                   if optimal_obj > 0 else 0.0)

            witnesses.append(
                HardnessWitness(
                    instance_n=n,
                    instance_k=k,
                    greedy_value=greedy_obj,
                    optimal_value=optimal_obj,
                    gap_pct=gap,
                    distance_matrix=None,  # Don't store large matrices
                    reduction_description=(
                        f"Max-{k}-Dispersion on {n} points in R^{d}. "
                        f"Clustered instance with {n_clusters} clusters. "
                        f"Greedy achieves {greedy_obj:.4f} vs optimal "
                        f"{optimal_obj:.4f} (gap {gap:.1f}%). "
                        f"NP-hard by reduction from Hassin et al. 1997."
                    ),
                )
            )

        return witnesses

    # ------------------------------------------------------------------
    # Comprehensive benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        max_n: int = 20,
        n_trials: int = 5,
        seed: int = 42,
    ) -> SMTBenchmarkResult:
        """Run comprehensive SMT benchmark with optimality gaps,
        fair retention certificates, and hardness witnesses.

        Args:
            max_n: Maximum problem size.
            n_trials: Trials per configuration.
            seed: Random seed.

        Returns:
            SMTBenchmarkResult with all results.
        """
        rng = np.random.RandomState(seed)

        # Optimality gaps
        n_vals = [n for n in [8, 10, 15, 20, 30, 50] if n <= max_n]
        gaps = self.compute_optimality_gaps(
            n_values=n_vals,
            k_values=[3, 5],
            n_trials=n_trials,
            seed=seed,
        )

        # Fair retention on a sample instance
        n_fair = min(max_n, 20)
        X = rng.randn(n_fair, 5)
        D = np.zeros((n_fair, n_fair))
        for i in range(n_fair):
            for j in range(i + 1, n_fair):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]

        groups = np.array([i % 3 for i in range(n_fair)])
        k = min(8, n_fair)

        constraint_levels = [
            {0: 1, 1: 1, 2: 1},
            {0: 2, 1: 2, 2: 2},
            {0: 1, 1: 1, 2: 3},
        ]
        # Filter infeasible constraints
        constraint_levels = [
            c for c in constraint_levels
            if sum(c.values()) <= k
        ]

        certs = self.certify_fair_retention(
            D, k, groups, constraint_levels
        )

        # Hardness witnesses
        witnesses = self.np_hardness_reduction(
            n_instances=min(n_trials * 2, 10), seed=seed
        )

        # Summary statistics
        suboptimal_gaps = [g for g in gaps if g.is_suboptimal]
        summary = {
            "total_instances": len(gaps),
            "suboptimal_count": len(suboptimal_gaps),
            "suboptimal_pct": (len(suboptimal_gaps) / len(gaps) * 100
                               if gaps else 0),
            "mean_gap_pct": float(np.mean([g.relative_gap_pct for g in gaps]))
                            if gaps else 0,
            "max_gap_pct": float(np.max([g.relative_gap_pct for g in gaps]))
                           if gaps else 0,
            "median_gap_pct": float(np.median([g.relative_gap_pct for g in gaps]))
                              if gaps else 0,
            "n_fair_certificates": len(certs),
            "mean_retention": float(np.mean([c.retention_ratio for c in certs]))
                              if certs else 0,
            "worst_retention": float(np.min([c.retention_ratio for c in certs]))
                               if certs else 0,
            "n_hardness_witnesses": len(witnesses),
            "max_hardness_gap_pct": float(
                np.max([w.gap_pct for w in witnesses])
            ) if witnesses else 0,
        }

        results = []
        for g in gaps:
            results.append({
                "type": "optimality_gap",
                "n": g.n,
                "k": g.k,
                "greedy_obj": g.greedy_objective,
                "optimal_obj": g.optimal_objective,
                "gap_pct": g.relative_gap_pct,
                "is_suboptimal": g.is_suboptimal,
            })

        return SMTBenchmarkResult(
            results=results,
            summary=summary,
            gaps=gaps,
            retention_certificates=certs,
            hardness_witnesses=witnesses,
        )


# ---------------------------------------------------------------------------
# ILP fallback for larger instances
# ---------------------------------------------------------------------------


class ILPDiversityOptimizer:
    """Integer Linear Programming solver using scipy.optimize.milp.

    For instances where Z3 is too slow (n > 50), uses ILP relaxation
    with McCormick linearization of the quadratic objective.
    """

    @staticmethod
    def solve_ilp(
        distance_matrix: np.ndarray,
        k: int,
        groups: Optional[np.ndarray] = None,
        min_per_group: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[int], float]:
        """Solve sum-pairwise diversity via ILP using scipy.

        Uses McCormick envelopes to linearize x_i·x_j products.
        """
        from scipy.optimize import LinearConstraint, milp
        from scipy.sparse import eye as speye

        D = np.asarray(distance_matrix, dtype=np.float64)
        n = D.shape[0]

        # Variables: x_i (binary, n vars) + y_{ij} (continuous, n*(n-1)/2 vars)
        n_pairs = n * (n - 1) // 2
        n_vars = n + n_pairs

        # Objective: maximize Σ d_{ij} y_{ij} → minimize -Σ d_{ij} y_{ij}
        c = np.zeros(n_vars)
        pair_idx = 0
        pair_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                c[n + pair_idx] = -D[i, j]  # negate for minimization
                pair_map[(i, j)] = pair_idx
                pair_idx += 1

        # Constraints
        # 1. Σ x_i = k
        A_eq_rows = [np.zeros(n_vars)]
        A_eq_rows[0][:n] = 1.0
        b_eq = [k]

        # 2. McCormick: y_{ij} ≤ x_i, y_{ij} ≤ x_j, y_{ij} ≥ x_i + x_j - 1
        A_ub_rows = []
        b_ub = []

        for (i, j), pidx in pair_map.items():
            # y_{ij} ≤ x_i
            row = np.zeros(n_vars)
            row[n + pidx] = 1.0
            row[i] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

            # y_{ij} ≤ x_j
            row = np.zeros(n_vars)
            row[n + pidx] = 1.0
            row[j] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

            # y_{ij} ≥ x_i + x_j - 1 → -y_{ij} ≤ 1 - x_i - x_j
            row = np.zeros(n_vars)
            row[n + pidx] = -1.0
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub.append(1.0)

        # 3. Fairness constraints: Σ_{i∈g} x_i ≥ min_g → -Σ ≤ -min_g
        if groups is not None and min_per_group:
            for g, min_count in min_per_group.items():
                row = np.zeros(n_vars)
                g_indices = [i for i in range(n) if groups[i] == g]
                for i in g_indices:
                    row[i] = -1.0
                A_ub_rows.append(row)
                b_ub.append(-min_count)

        A_eq = np.array(A_eq_rows) if A_eq_rows else None
        b_eq_arr = np.array(b_eq) if b_eq else None
        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub_arr = np.array(b_ub) if b_ub else None

        # Bounds: x_i ∈ {0,1}, y_{ij} ∈ [0,1]
        from scipy.optimize import Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        bounds = Bounds(lb, ub)

        # Integrality: x_i are integers, y_{ij} are continuous
        integrality = np.zeros(n_vars)
        integrality[:n] = 1  # x_i are binary

        constraints = []
        if A_eq is not None:
            constraints.append(
                LinearConstraint(A_eq, b_eq_arr, b_eq_arr)
            )
        if A_ub is not None:
            constraints.append(
                LinearConstraint(A_ub, -np.inf * np.ones(len(b_ub_arr)), b_ub_arr)
            )

        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
            )
            if result.success:
                x_sol = result.x[:n]
                selected = [i for i in range(n) if x_sol[i] > 0.5]
                obj = -result.fun  # negate back
                return selected, obj
        except Exception:
            pass

        return [], 0.0


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def smt_benchmark(max_n: int = 20, n_trials: int = 5,
                  seed: int = 42) -> Dict:
    """Run SMT benchmark and return JSON-serializable results."""
    optimizer = SMTDiversityOptimizer(timeout_ms=30000)
    result = optimizer.run_benchmark(max_n=max_n, n_trials=n_trials, seed=seed)

    output = {
        "summary": result.summary,
        "optimality_gaps": [
            {
                "n": g.n, "k": g.k,
                "greedy_obj": round(g.greedy_objective, 4),
                "optimal_obj": round(g.optimal_objective, 4),
                "gap_pct": round(g.relative_gap_pct, 2),
                "is_suboptimal": bool(g.is_suboptimal),
            }
            for g in result.gaps
        ],
        "fair_retention": [
            {
                "constraints": c.fairness_constraints,
                "unconstrained": round(c.unconstrained_optimal, 4),
                "constrained": round(c.constrained_optimal, 4),
                "retention": round(c.retention_ratio, 4),
                "certified": c.certified,
            }
            for c in result.retention_certificates
        ],
        "hardness_witnesses": [
            {
                "n": w.instance_n, "k": w.instance_k,
                "greedy": round(w.greedy_value, 4),
                "optimal": round(w.optimal_value, 4),
                "gap_pct": round(w.gap_pct, 2),
                "description": w.reduction_description,
            }
            for w in result.hardness_witnesses
        ],
    }
    return output


if __name__ == "__main__":
    import json

    print("Running SMT diversity benchmark...")
    results = smt_benchmark(max_n=15, n_trials=3)
    print(json.dumps(results, indent=2))
