"""
Fairness-aware diverse selection.

Implements group-fair selection, intersectional fairness, proportional
representation, within-group diversity, Rooney rule constraints,
fairness metrics, and Pareto-optimal fairness-diversity tradeoff search.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union, NamedTuple
from collections import Counter


def pairwise_distances_l2(X: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distance matrix."""
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X ** 2, axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    return D


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    return Xn @ Xn.T


class FairnessMetrics(NamedTuple):
    """Container for fairness evaluation metrics."""
    demographic_parity: float
    representation_ratio: float
    diversity_index_per_group: Dict[int, float]
    overall_diversity: float
    min_group_representation: float
    max_group_representation: float
    representation_gap: float


class FairDiverseSelector:
    """Fairness-aware diverse selection.

    Selects items balancing overall diversity with fair representation
    across demographic or attribute groups.
    """

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    # ------------------------------------------------------------------
    # Main selection interface
    # ------------------------------------------------------------------

    def select(self, items: np.ndarray, groups: np.ndarray, k: int,
               min_per_group: Optional[Dict[int, int]] = None,
               strategy: str = 'group_fair') -> np.ndarray:
        """Select k diverse items with fairness constraints.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) integer group labels.
            k: Number to select.
            min_per_group: Minimum items per group (optional).
            strategy: 'group_fair', 'proportional', 'rooney', 'intersectional'.

        Returns:
            Array of k selected indices.
        """
        if strategy == 'group_fair':
            return self.group_fair_selection(items, groups, k, min_per_group)
        elif strategy == 'proportional':
            return self.proportional_representation(items, groups, k)
        elif strategy == 'rooney':
            return self.rooney_rule(items, groups, k)
        elif strategy == 'intersectional':
            # For intersectional, groups should encode combined attributes
            return self.group_fair_selection(items, groups, k, min_per_group)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    # Group-fair selection
    # ------------------------------------------------------------------

    def group_fair_selection(self, items: np.ndarray,
                              groups: np.ndarray, k: int,
                              min_per_group: Optional[Dict[int, int]] = None
                              ) -> np.ndarray:
        """Ensure minimum representation per group while maximizing diversity.

        Phase 1: Greedily fill minimum group quotas.
        Phase 2: Fill remaining slots with diversity-maximizing selection.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) group labels.
            k: Number to select.
            min_per_group: Minimum count per group.

        Returns:
            Selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.intp)
        n = items.shape[0]
        k = min(k, n)

        D = pairwise_distances_l2(items)

        selected = []
        selected_set = set()
        group_counts = Counter()

        if min_per_group is None:
            min_per_group = {}

        # Phase 1: satisfy minimum group constraints
        for group_id in sorted(min_per_group.keys()):
            needed = min_per_group[group_id]
            group_mask = groups == group_id
            group_indices = np.where(group_mask)[0]

            # Select most diverse items from this group
            available = [i for i in group_indices if i not in selected_set]
            while group_counts[group_id] < needed and available and len(selected) < k:
                if not selected:
                    # Pick the one closest to group center
                    group_items = items[available]
                    center = group_items.mean(axis=0)
                    dists = np.sqrt(np.sum((group_items - center) ** 2, axis=1))
                    best_local = np.argmin(dists)
                    best_idx = available[best_local]
                else:
                    # Pick the one farthest from already selected
                    sel_arr = np.array(list(selected_set))
                    min_dists = np.array([D[i, sel_arr].min() for i in available])
                    best_local = np.argmax(min_dists)
                    best_idx = available[best_local]

                selected.append(best_idx)
                selected_set.add(best_idx)
                group_counts[groups[best_idx]] += 1
                available.remove(best_idx)

        # Phase 2: fill remaining with max-min diversity
        remaining = [i for i in range(n) if i not in selected_set]
        while len(selected) < k and remaining:
            if not selected:
                # Pick arbitrary
                selected.append(remaining[0])
                selected_set.add(remaining[0])
                group_counts[groups[remaining[0]]] += 1
                remaining.remove(remaining[0])
            else:
                sel_arr = np.array(list(selected_set))
                min_dists = np.array([D[i, sel_arr].min() for i in remaining])
                best_local = np.argmax(min_dists)
                best_idx = remaining[best_local]
                selected.append(best_idx)
                selected_set.add(best_idx)
                group_counts[groups[best_idx]] += 1
                remaining.remove(best_idx)

        return np.array(selected[:k], dtype=np.intp)

    # ------------------------------------------------------------------
    # Intersectional fairness
    # ------------------------------------------------------------------

    @staticmethod
    def create_intersectional_groups(
            attribute_arrays: List[np.ndarray]) -> np.ndarray:
        """Create intersectional group labels from multiple attributes.

        Args:
            attribute_arrays: List of (n,) arrays, each encoding one attribute.

        Returns:
            (n,) array of combined group labels.
        """
        n = len(attribute_arrays[0])
        combined = {}
        labels = np.zeros(n, dtype=np.intp)

        for i in range(n):
            key = tuple(int(attr[i]) for attr in attribute_arrays)
            if key not in combined:
                combined[key] = len(combined)
            labels[i] = combined[key]

        return labels

    def intersectional_selection(self, items: np.ndarray,
                                  attribute_arrays: List[np.ndarray],
                                  k: int,
                                  min_per_intersection: int = 1
                                  ) -> np.ndarray:
        """Fairness-aware selection considering intersectional groups.

        Args:
            items: (n, d) feature matrix.
            attribute_arrays: List of attribute arrays.
            k: Number to select.
            min_per_intersection: Min items per intersectional group.

        Returns:
            Selected indices.
        """
        groups = self.create_intersectional_groups(attribute_arrays)
        unique_groups = np.unique(groups)
        min_per_group = {int(g): min_per_intersection for g in unique_groups}

        # Adjust if total minimums exceed k
        total_min = sum(min_per_group.values())
        if total_min > k:
            # Reduce proportionally
            factor = k / total_min
            min_per_group = {g: max(1, int(m * factor))
                            for g, m in min_per_group.items()}

        return self.group_fair_selection(items, groups, k, min_per_group)

    # ------------------------------------------------------------------
    # Proportional representation
    # ------------------------------------------------------------------

    def proportional_representation(self, items: np.ndarray,
                                     groups: np.ndarray,
                                     k: int) -> np.ndarray:
        """Select items matching group proportions to population.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) group labels.
            k: Number to select.

        Returns:
            Selected indices.
        """
        groups = np.asarray(groups, dtype=np.intp)
        n = len(groups)
        k = min(k, n)

        group_counts = Counter(groups.tolist())
        unique_groups = sorted(group_counts.keys())

        # Compute proportional quotas
        min_per_group = {}
        allocated = 0
        for g in unique_groups:
            proportion = group_counts[g] / n
            quota = int(np.floor(proportion * k))
            min_per_group[g] = quota
            allocated += quota

        # Distribute remaining slots to groups with highest fractional parts
        remaining = k - allocated
        fractional_parts = []
        for g in unique_groups:
            proportion = group_counts[g] / n
            frac = proportion * k - min_per_group[g]
            fractional_parts.append((frac, g))
        fractional_parts.sort(reverse=True)
        for i in range(min(remaining, len(fractional_parts))):
            min_per_group[fractional_parts[i][1]] += 1

        return self.group_fair_selection(items, groups, k, min_per_group)

    # ------------------------------------------------------------------
    # Diversity within groups
    # ------------------------------------------------------------------

    def diversity_within_groups(self, items: np.ndarray,
                                 groups: np.ndarray, k: int,
                                 k_per_group: Optional[Dict[int, int]] = None
                                 ) -> np.ndarray:
        """Ensure diversity within each group, not just across groups.

        Selects diverse representatives from each group independently,
        then combines.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) group labels.
            k: Total to select.
            k_per_group: Items per group. Default: proportional.

        Returns:
            Selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.intp)
        n = items.shape[0]
        k = min(k, n)

        unique_groups = sorted(set(groups.tolist()))
        group_counts_pop = Counter(groups.tolist())

        if k_per_group is None:
            # Proportional
            k_per_group = {}
            allocated = 0
            for g in unique_groups:
                quota = max(1, int(k * group_counts_pop[g] / n))
                k_per_group[g] = quota
                allocated += quota
            # Adjust if over-allocated
            while allocated > k:
                g_max = max(k_per_group, key=k_per_group.get)
                if k_per_group[g_max] > 1:
                    k_per_group[g_max] -= 1
                    allocated -= 1

        selected = []
        D = pairwise_distances_l2(items)

        for g in unique_groups:
            kg = k_per_group.get(g, 0)
            group_indices = np.where(groups == g)[0]
            if len(group_indices) == 0 or kg == 0:
                continue

            kg = min(kg, len(group_indices))

            # Max-min diversity within group
            group_selected = []
            remaining = list(group_indices)

            # Start with random
            group_selected.append(remaining[0])
            remaining.remove(remaining[0])

            while len(group_selected) < kg and remaining:
                sel_arr = np.array(group_selected)
                min_dists = np.array([D[i, sel_arr].min() for i in remaining])
                best_local = np.argmax(min_dists)
                group_selected.append(remaining[best_local])
                remaining.pop(best_local)

            selected.extend(group_selected)

        return np.array(selected[:k], dtype=np.intp)

    # ------------------------------------------------------------------
    # Rooney rule
    # ------------------------------------------------------------------

    def rooney_rule(self, items: np.ndarray, groups: np.ndarray,
                     k: int) -> np.ndarray:
        """Rooney rule: ensure at least 1 from each underrepresented group.

        Underrepresented groups are those with below-median population count.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) group labels.
            k: Number to select.

        Returns:
            Selected indices.
        """
        groups = np.asarray(groups, dtype=np.intp)
        group_counts = Counter(groups.tolist())
        unique_groups = sorted(group_counts.keys())

        if len(unique_groups) <= 1:
            return self.group_fair_selection(items, groups, k)

        median_count = np.median(list(group_counts.values()))
        underrepresented = [g for g in unique_groups if group_counts[g] <= median_count]

        min_per_group = {}
        for g in underrepresented:
            min_per_group[g] = 1

        return self.group_fair_selection(items, groups, k, min_per_group)

    # ------------------------------------------------------------------
    # Fairness metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fairness_metrics(items: np.ndarray,
                                  groups: np.ndarray,
                                  selected: np.ndarray) -> FairnessMetrics:
        """Compute fairness metrics for a selection.

        Args:
            items: (n, d) all items.
            groups: (n,) group labels.
            selected: Indices of selected items.

        Returns:
            FairnessMetrics.
        """
        items = np.asarray(items, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.intp)
        selected = np.asarray(selected, dtype=np.intp)
        n = len(groups)
        k = len(selected)

        if k == 0:
            return FairnessMetrics(
                demographic_parity=0, representation_ratio=0,
                diversity_index_per_group={}, overall_diversity=0,
                min_group_representation=0, max_group_representation=0,
                representation_gap=0
            )

        unique_groups = sorted(set(groups.tolist()))
        pop_proportions = {}
        sel_proportions = {}
        selected_groups = groups[selected]

        for g in unique_groups:
            pop_proportions[g] = np.sum(groups == g) / n
            sel_proportions[g] = np.sum(selected_groups == g) / k

        # Demographic parity: max |sel_proportion - pop_proportion|
        dp = max(abs(sel_proportions.get(g, 0) - pop_proportions.get(g, 0))
                 for g in unique_groups)

        # Representation ratio: min(sel_prop / pop_prop) across groups
        rep_ratios = []
        for g in unique_groups:
            pp = pop_proportions.get(g, 0)
            sp = sel_proportions.get(g, 0)
            if pp > 0:
                rep_ratios.append(sp / pp)
            else:
                rep_ratios.append(float('inf') if sp > 0 else 1.0)
        rep_ratio = min(rep_ratios) if rep_ratios else 0.0

        # Diversity index per group (mean pairwise distance within selected group members)
        D = pairwise_distances_l2(items)
        div_per_group = {}
        for g in unique_groups:
            group_sel = selected[selected_groups == g]
            if len(group_sel) <= 1:
                div_per_group[g] = 0.0
            else:
                D_g = D[np.ix_(group_sel, group_sel)]
                upper = D_g[np.triu_indices(len(group_sel), k=1)]
                div_per_group[g] = float(np.mean(upper)) if len(upper) > 0 else 0.0

        # Overall diversity
        if k <= 1:
            overall_div = 0.0
        else:
            D_sel = D[np.ix_(selected, selected)]
            upper = D_sel[np.triu_indices(k, k=1)]
            overall_div = float(np.mean(upper)) if len(upper) > 0 else 0.0

        # Representation stats
        rep_values = list(sel_proportions.values())
        min_rep = float(min(rep_values))
        max_rep = float(max(rep_values))

        return FairnessMetrics(
            demographic_parity=float(dp),
            representation_ratio=float(rep_ratio),
            diversity_index_per_group=div_per_group,
            overall_diversity=overall_div,
            min_group_representation=min_rep,
            max_group_representation=max_rep,
            representation_gap=float(max_rep - min_rep)
        )

    # ------------------------------------------------------------------
    # Pareto-optimal fairness-diversity tradeoff
    # ------------------------------------------------------------------

    def pareto_tradeoff(self, items: np.ndarray, groups: np.ndarray,
                        k: int, n_configs: int = 20
                        ) -> List[Dict]:
        """Search for Pareto-optimal fairness-diversity configurations.

        Tries different minimum-per-group constraints and evaluates
        the resulting diversity and fairness.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) group labels.
            k: Number to select.
            n_configs: Number of configurations to try.

        Returns:
            List of dicts with 'config', 'fairness', 'diversity', 'selected'.
        """
        items = np.asarray(items, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.intp)
        n = items.shape[0]
        k = min(k, n)

        unique_groups = sorted(set(groups.tolist()))
        n_groups = len(unique_groups)

        if n_groups == 0:
            return []

        results = []

        for trial in range(n_configs):
            # Generate random minimum constraints
            rng = np.random.RandomState(trial)
            max_per = max(k // n_groups, 1)
            mins = {}
            total = 0
            for g in unique_groups:
                m = rng.randint(0, max_per + 1)
                mins[g] = m
                total += m

            # Ensure total doesn't exceed k
            while total > k:
                g = rng.choice(unique_groups)
                if mins[g] > 0:
                    mins[g] -= 1
                    total -= 1

            selected = self.group_fair_selection(items, groups, k, mins)
            metrics = self.compute_fairness_metrics(items, groups, selected)

            results.append({
                'config': mins,
                'demographic_parity': metrics.demographic_parity,
                'representation_ratio': metrics.representation_ratio,
                'overall_diversity': metrics.overall_diversity,
                'representation_gap': metrics.representation_gap,
                'selected': selected.tolist()
            })

        # Filter to Pareto frontier (fairness vs diversity)
        # A point is Pareto-optimal if no other point dominates it
        pareto = []
        for i, r in enumerate(results):
            dominated = False
            for j, s in enumerate(results):
                if i == j:
                    continue
                # s dominates r if s is better in both objectives
                if (s['overall_diversity'] >= r['overall_diversity'] and
                    s['demographic_parity'] <= r['demographic_parity'] and
                    (s['overall_diversity'] > r['overall_diversity'] or
                     s['demographic_parity'] < r['demographic_parity'])):
                    dominated = True
                    break
            if not dominated:
                r['is_pareto'] = True
                pareto.append(r)
            else:
                r['is_pareto'] = False

        # Sort by diversity
        results.sort(key=lambda r: -r['overall_diversity'])
        return results


class GroupDiversityAnalyzer:
    """Analyze diversity patterns across groups."""

    @staticmethod
    def group_diversity_report(items: np.ndarray,
                                groups: np.ndarray) -> Dict:
        """Generate a comprehensive diversity report per group.

        Args:
            items: (n, d) feature matrix.
            groups: (n,) group labels.

        Returns:
            Dict with per-group and overall statistics.
        """
        items = np.asarray(items, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.intp)
        D = pairwise_distances_l2(items)

        unique_groups = sorted(set(groups.tolist()))
        report = {'groups': {}, 'overall': {}}

        # Per-group statistics
        for g in unique_groups:
            g_idx = np.where(groups == g)[0]
            n_g = len(g_idx)

            if n_g <= 1:
                report['groups'][g] = {
                    'size': n_g,
                    'mean_pairwise_dist': 0.0,
                    'spread': 0.0,
                    'centroid_dist_to_overall': 0.0
                }
                continue

            D_g = D[np.ix_(g_idx, g_idx)]
            upper = D_g[np.triu_indices(n_g, k=1)]

            centroid_g = items[g_idx].mean(axis=0)
            overall_centroid = items.mean(axis=0)
            centroid_dist = float(np.linalg.norm(centroid_g - overall_centroid))

            report['groups'][g] = {
                'size': n_g,
                'mean_pairwise_dist': float(np.mean(upper)),
                'spread': float(np.std(upper)),
                'centroid_dist_to_overall': centroid_dist
            }

        # Overall
        n = len(groups)
        if n > 1:
            upper = D[np.triu_indices(n, k=1)]
            report['overall'] = {
                'n_groups': len(unique_groups),
                'total_items': n,
                'mean_pairwise_dist': float(np.mean(upper)),
                'group_separation': _between_group_distance(items, groups)
            }
        else:
            report['overall'] = {
                'n_groups': len(unique_groups),
                'total_items': n,
                'mean_pairwise_dist': 0.0,
                'group_separation': 0.0
            }

        return report


def _between_group_distance(items: np.ndarray, groups: np.ndarray) -> float:
    """Average distance between group centroids."""
    unique_groups = sorted(set(groups.tolist()))
    centroids = []
    for g in unique_groups:
        g_idx = np.where(groups == g)[0]
        centroids.append(items[g_idx].mean(axis=0))

    if len(centroids) <= 1:
        return 0.0

    centroids = np.array(centroids)
    n = len(centroids)
    D = pairwise_distances_l2(centroids)
    upper = D[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def demo_fair_diversity():
    """Demonstrate fair diversity selection."""
    rng = np.random.RandomState(42)

    # Create items from 3 groups
    items_g0 = rng.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    items_g1 = rng.randn(20, 5) + np.array([3, 3, 0, 0, 0])
    items_g2 = rng.randn(10, 5) + np.array([0, 0, 3, 3, 0])

    items = np.vstack([items_g0, items_g1, items_g2])
    groups = np.array([0]*30 + [1]*20 + [2]*10)

    selector = FairDiverseSelector()

    # Group-fair selection
    sel1 = selector.select(items, groups, k=10,
                           min_per_group={0: 2, 1: 2, 2: 2},
                           strategy='group_fair')
    print(f"Group-fair selection: {sel1}")
    for g in range(3):
        cnt = np.sum(groups[sel1] == g)
        print(f"  Group {g}: {cnt}")

    # Proportional
    sel2 = selector.proportional_representation(items, groups, k=10)
    print(f"\nProportional selection: {sel2}")
    for g in range(3):
        cnt = np.sum(groups[sel2] == g)
        print(f"  Group {g}: {cnt}")

    # Rooney rule
    sel3 = selector.rooney_rule(items, groups, k=10)
    print(f"\nRooney rule selection: {sel3}")
    for g in range(3):
        cnt = np.sum(groups[sel3] == g)
        print(f"  Group {g}: {cnt}")

    # Within-group diversity
    sel4 = selector.diversity_within_groups(items, groups, k=10)
    print(f"\nWithin-group diverse selection: {sel4}")

    # Fairness metrics
    metrics = FairDiverseSelector.compute_fairness_metrics(items, groups, sel1)
    print(f"\nFairness metrics for group-fair selection:")
    print(f"  Demographic parity: {metrics.demographic_parity:.4f}")
    print(f"  Representation ratio: {metrics.representation_ratio:.4f}")
    print(f"  Overall diversity: {metrics.overall_diversity:.4f}")
    print(f"  Representation gap: {metrics.representation_gap:.4f}")
    print(f"  Per-group diversity: {metrics.diversity_index_per_group}")

    # Intersectional
    gender = np.array([0]*15 + [1]*15 + [0]*10 + [1]*10 + [0]*5 + [1]*5)
    sel5 = selector.intersectional_selection(items, [groups, gender], k=10)
    combined = FairDiverseSelector.create_intersectional_groups([groups, gender])
    print(f"\nIntersectional selection: {sel5}")
    for g in np.unique(combined):
        cnt = np.sum(combined[sel5] == g)
        print(f"  Intersection group {g}: {cnt}")

    # Pareto tradeoff
    pareto = selector.pareto_tradeoff(items, groups, k=10, n_configs=10)
    pareto_count = sum(1 for r in pareto if r.get('is_pareto', False))
    print(f"\nPareto frontier: {pareto_count} Pareto-optimal configurations")


if __name__ == '__main__':
    demo_fair_diversity()
