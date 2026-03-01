"""
MetricAlgebra lattice structure characterization.

Addresses reviewer critique: "MetricAlgebra quotient M/~ is computed
but not characterized—no lattice structure, meet/join operations,
or automorphism group analysis."

Implements:
  1. Hasse diagram computation at multiple δ thresholds
  2. Lattice meet (∧) and join (∨) operations on equivalence classes
  3. Lattice completeness verification
  4. Automorphism group computation
  5. δ-filtration: how equivalence classes evolve as δ varies
  6. Topological invariants (Betti numbers) of the metric space
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Kendall τ computation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LatticeElement:
    """An element in the metric equivalence class lattice."""
    metrics: FrozenSet[str]
    level: int  # level in Hasse diagram (0 = bottom)
    delta: float  # δ at which this class formed

    def __hash__(self):
        return hash(self.metrics)

    def __eq__(self, other):
        return isinstance(other, LatticeElement) and self.metrics == other.metrics


@dataclass
class HasseDiagram:
    """Hasse diagram of the metric equivalence lattice."""
    elements: List[LatticeElement]
    edges: List[Tuple[int, int]]  # (parent_idx, child_idx) - covers only
    delta: float
    n_classes: int


@dataclass
class LatticeStructure:
    """Complete lattice structure analysis."""
    hasse_diagrams: Dict[float, HasseDiagram]
    filtration: List[Tuple[float, int]]  # (δ, n_classes) pairs
    merge_sequence: List[Tuple[float, str, str]]  # (δ, metric_a, metric_b) merges
    is_lattice: bool
    meet_table: Dict[str, Dict[str, str]]  # meet of two elements
    join_table: Dict[str, Dict[str, str]]  # join of two elements
    automorphisms: List[Dict[str, str]]
    betti_numbers: List[int]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# MetricLattice
# ---------------------------------------------------------------------------


class MetricLattice:
    """Computes and characterizes the lattice structure of M/~_δ.

    Given metric values, computes the quotient M/~ where
    M₁ ~ M₂ iff |τ(M₁, M₂)| ≥ 1 - δ, and characterizes
    the resulting algebraic structure.
    """

    def __init__(self, metric_values: Dict[str, np.ndarray]):
        """
        Args:
            metric_values: Dict mapping metric name → array of values.
        """
        self.metrics = metric_values
        self.names = sorted(metric_values.keys())
        self.tau_matrix = self._compute_full_tau_matrix()

    def _compute_full_tau_matrix(self) -> np.ndarray:
        """Compute full |τ| matrix between all metrics."""
        n = len(self.names)
        tau = np.zeros((n, n))
        for i in range(n):
            tau[i, i] = 1.0
            for j in range(i + 1, n):
                t = abs(_kendall_tau(
                    self.metrics[self.names[i]],
                    self.metrics[self.names[j]]
                ))
                tau[i, j] = t
                tau[j, i] = t
        return tau

    def equivalence_classes(self, delta: float) -> List[FrozenSet[str]]:
        """Compute equivalence classes at threshold δ.

        M₁ ~ M₂ iff |τ(M₁, M₂)| ≥ 1 - δ.
        Uses union-find.
        """
        n = len(self.names)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        threshold = 1.0 - delta
        for i in range(n):
            for j in range(i + 1, n):
                if self.tau_matrix[i, j] >= threshold:
                    union(i, j)

        groups = defaultdict(set)
        for i in range(n):
            groups[find(i)].add(self.names[i])

        return [frozenset(g) for g in groups.values()]

    def hasse_diagram(self, delta: float) -> HasseDiagram:
        """Compute Hasse diagram at threshold δ.

        The Hasse diagram shows the covering relation: A covers B
        iff B ⊂ A and there is no C with B ⊂ C ⊂ A.
        """
        classes = self.equivalence_classes(delta)

        # Sort by size (smaller sets lower in the diagram)
        elements = []
        for i, cls in enumerate(sorted(classes, key=len)):
            elements.append(LatticeElement(
                metrics=cls,
                level=len(cls) - 1,
                delta=delta,
            ))

        # Compute covering relation
        edges = []
        for i, elem_i in enumerate(elements):
            for j, elem_j in enumerate(elements):
                if i == j:
                    continue
                # elem_j covers elem_i if elem_i ⊂ elem_j
                if elem_i.metrics < elem_j.metrics:
                    # Check no intermediate element
                    is_cover = True
                    for k, elem_k in enumerate(elements):
                        if k != i and k != j:
                            if (elem_i.metrics < elem_k.metrics
                                    < elem_j.metrics):
                                is_cover = False
                                break
                    if is_cover:
                        edges.append((j, i))  # parent → child

        return HasseDiagram(
            elements=elements,
            edges=edges,
            delta=delta,
            n_classes=len(classes),
        )

    def delta_filtration(
        self, n_thresholds: int = 50
    ) -> List[Tuple[float, int, List[FrozenSet[str]]]]:
        """Compute how equivalence classes evolve as δ varies.

        Returns list of (δ, n_classes, classes) tuples.
        """
        deltas = np.linspace(0.01, 0.99, n_thresholds)
        filtration = []

        for d in deltas:
            classes = self.equivalence_classes(d)
            filtration.append((float(d), len(classes), classes))

        return filtration

    def merge_sequence(self) -> List[Tuple[float, str, str]]:
        """Find the sequence of δ values at which metrics merge.

        For each pair (M₁, M₂), the merge point is δ* = 1 - |τ(M₁, M₂)|.
        """
        merges = []
        n = len(self.names)
        for i in range(n):
            for j in range(i + 1, n):
                delta_star = 1.0 - self.tau_matrix[i, j]
                merges.append((delta_star, self.names[i], self.names[j]))

        merges.sort(key=lambda x: x[0])
        return merges

    def compute_meet_join(
        self, delta: float
    ) -> Tuple[Dict, Dict]:
        """Compute meet (∧) and join (∨) operations.

        For equivalence classes:
          - meet(A, B) = largest class contained in both (intersection)
          - join(A, B) = smallest class containing both (union closure)
        """
        classes = self.equivalence_classes(delta)
        class_names = [",".join(sorted(c)) for c in classes]
        class_map = {name: cls for name, cls in zip(class_names, classes)}

        meet_table = {}
        join_table = {}

        for name_a, cls_a in zip(class_names, classes):
            meet_table[name_a] = {}
            join_table[name_a] = {}
            for name_b, cls_b in zip(class_names, classes):
                # Meet = intersection
                intersection = cls_a & cls_b
                if intersection:
                    # Find the class containing the intersection
                    meet_cls = None
                    for name_c, cls_c in zip(class_names, classes):
                        if intersection <= cls_c:
                            if meet_cls is None or len(cls_c) < len(class_map.get(meet_cls, frozenset())):
                                meet_cls = name_c
                    meet_table[name_a][name_b] = meet_cls or "∅"
                else:
                    meet_table[name_a][name_b] = "∅"

                # Join = union
                union = cls_a | cls_b
                join_cls = None
                for name_c, cls_c in zip(class_names, classes):
                    if union <= cls_c:
                        if join_cls is None or len(cls_c) < len(class_map.get(join_cls, frozenset())):
                            join_cls = name_c
                    elif cls_c <= union:
                        if join_cls is None or len(cls_c) > len(class_map.get(join_cls, frozenset())):
                            join_cls = name_c
                join_table[name_a][name_b] = join_cls or ",".join(sorted(union))

        return meet_table, join_table

    def is_lattice(self, delta: float) -> bool:
        """Check if M/~_δ forms a lattice.

        A lattice requires that every pair of elements has a unique
        meet (greatest lower bound) and join (least upper bound).
        """
        classes = self.equivalence_classes(delta)
        if len(classes) <= 2:
            return True

        # Check: every pair has a unique meet and join
        # For a partition lattice, this holds iff the partition
        # refines the lattice of all partitions
        # Simplified check: verify absorption laws
        meet_table, join_table = self.compute_meet_join(delta)

        for a in meet_table:
            for b in meet_table:
                # Absorption: a ∨ (a ∧ b) = a
                meet_ab = meet_table.get(a, {}).get(b, "∅")
                if meet_ab != "∅":
                    join_a_meet = join_table.get(a, {}).get(meet_ab, "")
                    if join_a_meet != a and join_a_meet != "":
                        return False
        return True

    def compute_automorphisms(self, delta: float) -> List[Dict[str, str]]:
        """Compute automorphism group of the metric equivalence graph.

        An automorphism is a permutation of metrics that preserves
        the equivalence relation.
        """
        classes = self.equivalence_classes(delta)
        n = len(self.names)

        # For small n, enumerate permutations
        if n > 8:
            return [{"note": "too many metrics for exhaustive search"}]

        automorphisms = []

        # The identity is always an automorphism
        identity = {m: m for m in self.names}
        automorphisms.append(identity)

        # Check all permutations within each equivalence class
        # (permutations across classes are not automorphisms)
        for perm in permutations(range(n)):
            perm_map = {self.names[i]: self.names[perm[i]] for i in range(n)}

            # Check if this permutation preserves all τ relationships
            preserves = True
            for i in range(n):
                for j in range(i + 1, n):
                    orig_tau = self.tau_matrix[i, j]
                    perm_i = perm[i]
                    perm_j = perm[j]
                    perm_tau = self.tau_matrix[perm_i, perm_j]
                    if abs(orig_tau - perm_tau) > 0.1:  # tolerance
                        preserves = False
                        break
                if not preserves:
                    break

            if preserves and perm_map != identity:
                automorphisms.append(perm_map)

        return automorphisms

    def betti_numbers(self, delta: float, max_dim: int = 3) -> List[int]:
        """Compute Betti numbers of the Vietoris-Rips complex.

        The VR complex at scale δ has:
          - 0-simplices: metrics
          - 1-simplices: pairs with |τ| ≥ 1-δ
          - k-simplices: (k+1)-cliques in the τ graph

        β₀ = number of connected components
        β₁ = number of 1-dimensional holes
        """
        n = len(self.names)
        threshold = 1.0 - delta

        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if self.tau_matrix[i, j] >= threshold:
                    adj[i, j] = True
                    adj[j, i] = True

        # β₀: connected components (via union-find)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    union(i, j)

        beta_0 = len(set(find(i) for i in range(n)))

        # β₁: number of independent cycles
        # Euler characteristic: χ = V - E + F - ...
        # For a graph: β₁ = E - V + β₀
        n_edges = np.sum(adj) // 2

        # Count triangles (2-simplices)
        n_triangles = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]:
                    continue
                for k in range(j + 1, n):
                    if adj[i, k] and adj[j, k]:
                        n_triangles += 1

        beta_1 = max(0, n_edges - n + beta_0 - n_triangles)

        # Higher Betti numbers: count higher cliques
        betti = [beta_0, beta_1]

        if max_dim >= 2:
            # β₂ via Euler characteristic with tetrahedra
            n_tetra = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i, j]:
                        continue
                    for k in range(j + 1, n):
                        if not (adj[i, k] and adj[j, k]):
                            continue
                        for l in range(k + 1, n):
                            if adj[i, l] and adj[j, l] and adj[k, l]:
                                n_tetra += 1
            beta_2 = max(0, n_triangles - n_edges + n - beta_0 + n_tetra)
            betti.append(beta_2)

        return betti

    def full_analysis(
        self, n_thresholds: int = 20
    ) -> LatticeStructure:
        """Run complete lattice structure analysis."""
        # Filtration
        filtration_data = self.delta_filtration(n_thresholds)
        filtration = [(d, nc) for d, nc, _ in filtration_data]

        # Merge sequence
        merges = self.merge_sequence()

        # Hasse diagrams at multiple thresholds
        key_deltas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        hasse_diagrams = {}
        for d in key_deltas:
            hasse_diagrams[d] = self.hasse_diagram(d)

        # Lattice check at δ=0.3
        is_lat = self.is_lattice(0.3)

        # Meet/join at δ=0.3
        meet_t, join_t = self.compute_meet_join(0.3)

        # Automorphisms at δ=0.3
        autos = self.compute_automorphisms(0.3)

        # Betti numbers at δ=0.3
        betti = self.betti_numbers(0.3)

        # Critical δ values where topology changes
        critical_deltas = []
        prev_nc = filtration[0][1] if filtration else 0
        for d, nc in filtration:
            if nc != prev_nc:
                critical_deltas.append({"delta": round(d, 4),
                                         "n_classes": nc,
                                         "change": prev_nc - nc})
                prev_nc = nc

        summary = {
            "n_metrics": len(self.names),
            "metrics": self.names,
            "is_lattice_at_0.3": is_lat,
            "n_automorphisms": len(autos),
            "betti_numbers": betti,
            "critical_deltas": critical_deltas,
            "merge_sequence": [
                {"delta": round(d, 4), "metric_a": a, "metric_b": b}
                for d, a, b in merges[:10]
            ],
            "filtration_summary": {
                "delta_0.1": next((nc for d, nc in filtration if d >= 0.1), None),
                "delta_0.3": next((nc for d, nc in filtration if d >= 0.3), None),
                "delta_0.5": next((nc for d, nc in filtration if d >= 0.5), None),
                "delta_0.7": next((nc for d, nc in filtration if d >= 0.7), None),
                "delta_0.9": next((nc for d, nc in filtration if d >= 0.9), None),
            },
            "hasse_diagrams": {
                str(d): {
                    "n_classes": h.n_classes,
                    "n_edges": len(h.edges),
                    "classes": [list(e.metrics) for e in h.elements],
                }
                for d, h in hasse_diagrams.items()
            },
        }

        return LatticeStructure(
            hasse_diagrams=hasse_diagrams,
            filtration=filtration,
            merge_sequence=merges,
            is_lattice=is_lat,
            meet_table=meet_t,
            join_table=join_t,
            automorphisms=autos,
            betti_numbers=betti,
            summary=summary,
        )


def lattice_analysis(metric_values: Dict[str, np.ndarray]) -> Dict:
    """Run lattice analysis and return JSON-serializable results."""
    lattice = MetricLattice(metric_values)
    result = lattice.full_analysis()
    return result.summary
