"""
Comprehensive evaluation of diversity methods: precision-recall for coverage,
nDCD, alpha-nDCG, ERR-IA, subtopic recall, diversity lift, statistical significance,
diversity-quality tradeoff curves.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import norm, rankdata
from typing import List, Tuple, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
import math


@dataclass
class EvalReport:
    """Comprehensive evaluation report for diversity methods."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ndcd: float = 0.0
    alpha_ndcg: float = 0.0
    err_ia: float = 0.0
    subtopic_recall: float = 0.0
    diversity_lift: float = 0.0
    pairwise_diversity: float = 0.0
    coverage: float = 0.0
    significance: Dict[str, Any] = field(default_factory=dict)
    tradeoff_curve: List[Tuple[float, float]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class CoveragePrecisionRecall:
    """Precision-recall for spatial coverage."""

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def coverage_precision(self, selected: np.ndarray,
                           reference: np.ndarray) -> float:
        """Precision: fraction of selected items that cover some reference point."""
        if len(selected) == 0 or len(reference) == 0:
            return 0.0
        dists = cdist(selected, reference)
        covers_something = np.any(dists <= self.radius, axis=1)
        return float(np.mean(covers_something))

    def coverage_recall(self, selected: np.ndarray,
                        reference: np.ndarray) -> float:
        """Recall: fraction of reference points covered by at least one selected item."""
        if len(selected) == 0 or len(reference) == 0:
            return 0.0
        dists = cdist(reference, selected)
        is_covered = np.any(dists <= self.radius, axis=1)
        return float(np.mean(is_covered))

    def coverage_f1(self, selected: np.ndarray,
                    reference: np.ndarray) -> float:
        """F1 score balancing precision and recall."""
        p = self.coverage_precision(selected, reference)
        r = self.coverage_recall(selected, reference)
        if p + r < 1e-15:
            return 0.0
        return 2 * p * r / (p + r)

    def multi_radius_analysis(self, selected: np.ndarray,
                               reference: np.ndarray,
                               radii: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """Compute precision/recall at multiple radii."""
        if radii is None:
            if len(reference) > 1:
                all_dists = pdist(reference)
                radii = np.percentile(all_dists, [10, 25, 50, 75, 90]).tolist()
            else:
                radii = [0.5, 1.0, 2.0]

        results = {'radii': radii, 'precision': [], 'recall': [], 'f1': []}
        for r in radii:
            self.radius = r
            results['precision'].append(self.coverage_precision(selected, reference))
            results['recall'].append(self.coverage_recall(selected, reference))
            results['f1'].append(self.coverage_f1(selected, reference))

        return results


class NormalizedDiscountedCumulativeDiversity:
    """nDCD: like nDCG but for diversity."""

    def dcd(self, selected: np.ndarray, position_discount: bool = True) -> float:
        """Discounted Cumulative Diversity: sum of diversity gains at each position."""
        if len(selected) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(selected)):
            min_dist = np.min(cdist(
                selected[i:i + 1], selected[:i]
            ))
            discount = 1.0 / np.log2(i + 1) if position_discount else 1.0
            total += min_dist * discount

        return float(total)

    def ideal_dcd(self, reference: np.ndarray, k: int,
                  position_discount: bool = True) -> float:
        """Ideal DCD: greedily select most diverse items from reference."""
        if len(reference) < 2 or k < 2:
            return 0.0

        dist_matrix = squareform(pdist(reference))
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        selected_indices = [int(i), int(j)]

        total = dist_matrix[i, j]
        if position_discount:
            total = total / np.log2(2)

        remaining = set(range(len(reference))) - set(selected_indices)

        for pos in range(2, min(k, len(reference))):
            best_min_dist = -1.0
            best_idx = -1

            for idx in remaining:
                min_d = min(dist_matrix[idx, s] for s in selected_indices)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.discard(best_idx)
                discount = 1.0 / np.log2(pos + 1) if position_discount else 1.0
                total += best_min_dist * discount

        return float(total)

    def ndcd(self, selected: np.ndarray, reference: np.ndarray) -> float:
        """Normalized DCD."""
        actual = self.dcd(selected)
        ideal = self.ideal_dcd(reference, len(selected))
        if ideal < 1e-15:
            return 0.0
        return min(actual / ideal, 1.0)


class AlphaNDCG:
    """alpha-nDCG: intent-aware diversity evaluation metric."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def compute(self, ranked_docs: List[int],
                doc_subtopics: Dict[int, List[int]],
                all_subtopics: Set[int],
                k: Optional[int] = None) -> float:
        """Compute alpha-nDCG.
        Redundancy-aware: each additional document covering same subtopic gets discounted.
        """
        if k is None:
            k = len(ranked_docs)

        subtopic_counts: Dict[int, int] = {s: 0 for s in all_subtopics}
        dcg = 0.0

        for rank, doc_id in enumerate(ranked_docs[:k]):
            doc_topics = doc_subtopics.get(doc_id, [])
            gain = 0.0
            for topic in doc_topics:
                if topic in subtopic_counts:
                    gain += (1 - self.alpha) ** subtopic_counts[topic]
                    subtopic_counts[topic] += 1

            discount = 1.0 / np.log2(rank + 2)
            dcg += gain * discount

        ideal_dcg = self._ideal_alpha_dcg(doc_subtopics, all_subtopics, k)

        if ideal_dcg < 1e-15:
            return 0.0
        return dcg / ideal_dcg

    def _ideal_alpha_dcg(self, doc_subtopics: Dict[int, List[int]],
                          all_subtopics: Set[int], k: int) -> float:
        """Compute ideal alpha-nDCG by greedily selecting best docs."""
        remaining_docs = list(doc_subtopics.keys())
        subtopic_counts: Dict[int, int] = {s: 0 for s in all_subtopics}
        ideal_dcg = 0.0

        for rank in range(min(k, len(remaining_docs))):
            best_gain = -1.0
            best_doc = remaining_docs[0] if remaining_docs else -1

            for doc_id in remaining_docs:
                doc_topics = doc_subtopics.get(doc_id, [])
                gain = 0.0
                for topic in doc_topics:
                    if topic in subtopic_counts:
                        gain += (1 - self.alpha) ** subtopic_counts[topic]
                if gain > best_gain:
                    best_gain = gain
                    best_doc = doc_id

            if best_doc >= 0:
                for topic in doc_subtopics.get(best_doc, []):
                    if topic in subtopic_counts:
                        subtopic_counts[topic] += 1

                discount = 1.0 / np.log2(rank + 2)
                ideal_dcg += best_gain * discount
                remaining_docs.remove(best_doc)

        return ideal_dcg


class ERRIA:
    """ERR-IA: Expected Reciprocal Rank, Intent-Aware."""

    def __init__(self, n_intents: int = 5):
        self.n_intents = n_intents

    def compute(self, ranked_docs: List[int],
                doc_intent_relevance: Dict[int, Dict[int, float]],
                intent_probs: np.ndarray) -> float:
        """Compute ERR-IA.
        doc_intent_relevance[doc_id][intent] = relevance grade (0-1)
        intent_probs: probability of each intent
        """
        err_ia = 0.0

        for intent in range(self.n_intents):
            p_intent = intent_probs[intent]
            trust = 1.0

            for rank, doc_id in enumerate(ranked_docs):
                relevance = doc_intent_relevance.get(doc_id, {}).get(intent, 0.0)
                err_ia += p_intent * trust * relevance / (rank + 1)
                trust *= (1 - relevance)

        return float(err_ia)

    def per_intent_err(self, ranked_docs: List[int],
                       doc_intent_relevance: Dict[int, Dict[int, float]],
                       intent_probs: np.ndarray) -> Dict[int, float]:
        """Compute ERR per intent."""
        results = {}
        for intent in range(self.n_intents):
            trust = 1.0
            err = 0.0
            for rank, doc_id in enumerate(ranked_docs):
                relevance = doc_intent_relevance.get(doc_id, {}).get(intent, 0.0)
                err += trust * relevance / (rank + 1)
                trust *= (1 - relevance)
            results[intent] = float(err)
        return results


class SubtopicRecallMetric:
    """Fraction of subtopics covered by selected set."""

    def compute(self, selected_subtopics: List[Set[int]],
                all_subtopics: Set[int]) -> float:
        """Compute subtopic recall."""
        covered = set()
        for topics in selected_subtopics:
            covered.update(topics)
        if not all_subtopics:
            return 1.0
        return len(covered & all_subtopics) / len(all_subtopics)

    def recall_at_k(self, ranked_subtopics: List[Set[int]],
                    all_subtopics: Set[int],
                    max_k: Optional[int] = None) -> List[float]:
        """Subtopic recall at each position."""
        if max_k is None:
            max_k = len(ranked_subtopics)

        covered = set()
        recalls = []
        for i in range(min(max_k, len(ranked_subtopics))):
            covered.update(ranked_subtopics[i])
            if not all_subtopics:
                recalls.append(1.0)
            else:
                recalls.append(len(covered & all_subtopics) / len(all_subtopics))

        return recalls

    def weighted_subtopic_recall(self, selected_subtopics: List[Set[int]],
                                  all_subtopics: Set[int],
                                  subtopic_weights: Dict[int, float]) -> float:
        """Weighted subtopic recall."""
        covered = set()
        for topics in selected_subtopics:
            covered.update(topics)

        total_weight = sum(subtopic_weights.get(t, 1.0) for t in all_subtopics)
        if total_weight < 1e-15:
            return 0.0

        covered_weight = sum(subtopic_weights.get(t, 1.0) for t in covered & all_subtopics)
        return covered_weight / total_weight


class DiversityLift:
    """Improvement of diversity over random selection."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def compute(self, selected: np.ndarray, pool: np.ndarray,
                k: Optional[int] = None,
                n_random_samples: int = 100) -> Dict[str, float]:
        """Compare diversity of selected vs random subsets of same size."""
        if k is None:
            k = len(selected)

        selected_diversity = self._pairwise_diversity(selected)

        random_diversities = []
        for _ in range(n_random_samples):
            indices = self.rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            random_subset = pool[indices]
            random_diversities.append(self._pairwise_diversity(random_subset))

        random_diversities = np.array(random_diversities)
        mean_random = float(np.mean(random_diversities))
        std_random = float(np.std(random_diversities))

        lift = (selected_diversity - mean_random) / max(std_random, 1e-15)
        percentile = float(np.mean(random_diversities <= selected_diversity))

        return {
            'selected_diversity': selected_diversity,
            'random_mean': mean_random,
            'random_std': std_random,
            'lift': selected_diversity - mean_random,
            'normalized_lift': lift,
            'percentile': percentile
        }

    def _pairwise_diversity(self, items: np.ndarray) -> float:
        if len(items) < 2:
            return 0.0
        return float(np.mean(pdist(items)))


class StatisticalSignificance:
    """Statistical significance of diversity improvements."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def paired_bootstrap(self, scores_a: np.ndarray, scores_b: np.ndarray,
                          n_bootstrap: int = 10000,
                          confidence: float = 0.95) -> Dict[str, Any]:
        """Paired bootstrap test for significance."""
        n = len(scores_a)
        diff = scores_a - scores_b
        observed_diff = float(np.mean(diff))

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            indices = self.rng.choice(n, size=n, replace=True)
            boot_diff = np.mean(diff[indices])
            bootstrap_diffs.append(float(boot_diff))

        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = float(np.mean(bootstrap_diffs <= 0))
        if observed_diff < 0:
            p_value = float(np.mean(bootstrap_diffs >= 0))

        alpha = 1 - confidence
        ci_lower = float(np.percentile(bootstrap_diffs, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100))

        return {
            'observed_diff': observed_diff,
            'p_value': min(p_value, 1.0 - p_value) * 2,
            'significant': p_value < alpha / 2 or p_value > 1 - alpha / 2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_bootstrap': n_bootstrap,
            'effect_size': observed_diff / max(float(np.std(diff)), 1e-15)
        }

    def permutation_test(self, scores_a: np.ndarray, scores_b: np.ndarray,
                          n_permutations: int = 10000) -> Dict[str, Any]:
        """Permutation test for significance."""
        observed_diff = float(np.mean(scores_a) - np.mean(scores_b))
        combined = np.concatenate([scores_a, scores_b])
        n_a = len(scores_a)

        count_extreme = 0
        for _ in range(n_permutations):
            perm = self.rng.permutation(combined)
            perm_diff = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_permutations + 1)

        return {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_permutations': n_permutations
        }


class DiversityQualityTradeoff:
    """Vary diversity weight, plot Pareto frontier."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def compute_tradeoff_curve(self, items: np.ndarray,
                                quality_scores: np.ndarray,
                                k: int,
                                n_weights: int = 20) -> List[Tuple[float, float]]:
        """Compute diversity-quality tradeoff by varying weight."""
        weights = np.linspace(0, 1, n_weights)
        curve = []

        for w in weights:
            selected = self._weighted_selection(items, quality_scores, k, w)
            diversity = self._compute_diversity(items[selected])
            quality = float(np.mean(quality_scores[selected]))
            curve.append((diversity, quality))

        return curve

    def _weighted_selection(self, items: np.ndarray,
                            quality_scores: np.ndarray,
                            k: int, diversity_weight: float) -> List[int]:
        """Greedy selection with diversity-quality tradeoff."""
        selected: List[int] = []
        remaining = set(range(len(items)))

        for _ in range(min(k, len(items))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                q = quality_scores[idx] * (1 - diversity_weight)
                d = 0.0
                if selected:
                    dists = cdist(items[idx:idx + 1], items[selected])
                    d = float(np.min(dists)) * diversity_weight
                score = q + d

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def _compute_diversity(self, items: np.ndarray) -> float:
        if len(items) < 2:
            return 0.0
        return float(np.mean(pdist(items)))

    def extract_pareto_frontier(self, curve: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Extract Pareto-optimal points from tradeoff curve."""
        pareto = []
        for p in curve:
            dominated = False
            for q in curve:
                if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        pareto.sort(key=lambda x: x[0])
        return pareto

    def area_under_tradeoff(self, curve: List[Tuple[float, float]]) -> float:
        """Area under the tradeoff curve (trapezoidal)."""
        if len(curve) < 2:
            return 0.0
        curve_sorted = sorted(curve, key=lambda x: x[0])
        area = 0.0
        for i in range(1, len(curve_sorted)):
            dx = curve_sorted[i][0] - curve_sorted[i - 1][0]
            avg_y = (curve_sorted[i][1] + curve_sorted[i - 1][1]) / 2
            area += dx * avg_y
        return area


class DiversityEvaluator:
    """Main class: comprehensive diversity evaluation."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.coverage_pr = CoveragePrecisionRecall()
        self.ndcd = NormalizedDiscountedCumulativeDiversity()
        self.alpha_ndcg = AlphaNDCG()
        self.err_ia = ERRIA()
        self.subtopic_recall = SubtopicRecallMetric()
        self.diversity_lift = DiversityLift(seed)
        self.significance = StatisticalSignificance(seed)
        self.tradeoff = DiversityQualityTradeoff(seed)

    def evaluate(self, selected: np.ndarray,
                 reference_set: np.ndarray,
                 subtopics: Optional[Dict[int, Set[int]]] = None,
                 quality_scores: Optional[np.ndarray] = None) -> EvalReport:
        """Comprehensive evaluation of a diverse selection."""
        p = self.coverage_pr.coverage_precision(selected, reference_set)
        r = self.coverage_pr.coverage_recall(selected, reference_set)
        f1 = self.coverage_pr.coverage_f1(selected, reference_set)

        ndcd_score = self.ndcd.ndcd(selected, reference_set)

        pairwise_div = float(np.mean(pdist(selected))) if len(selected) > 1 else 0.0

        lift_result = self.diversity_lift.compute(selected, reference_set)

        if subtopics is not None:
            all_st = set()
            for topics in subtopics.values():
                all_st.update(topics)
            selected_topics = [subtopics.get(i, set()) for i in range(len(selected))]
            st_recall = self.subtopic_recall.compute(selected_topics, all_st)

            doc_subtopics = {i: list(subtopics.get(i, set())) for i in range(len(selected))}
            a_ndcg = self.alpha_ndcg.compute(
                list(range(len(selected))), doc_subtopics, all_st
            )
        else:
            st_recall = 0.0
            a_ndcg = 0.0

        tradeoff_curve = []
        if quality_scores is not None:
            tradeoff_curve = self.tradeoff.compute_tradeoff_curve(
                reference_set, quality_scores, len(selected)
            )

        return EvalReport(
            precision=p,
            recall=r,
            f1=f1,
            ndcd=ndcd_score,
            alpha_ndcg=a_ndcg,
            subtopic_recall=st_recall,
            diversity_lift=lift_result['normalized_lift'],
            pairwise_diversity=pairwise_div,
            coverage=r,
            tradeoff_curve=tradeoff_curve,
            details={
                'lift_details': lift_result,
                'coverage_precision': p,
                'coverage_recall': r,
            }
        )

    def evaluate_ranking(self, ranked_items: np.ndarray,
                         reference_set: np.ndarray,
                         subtopics: Optional[Dict[int, Set[int]]] = None,
                         intent_relevance: Optional[Dict[int, Dict[int, float]]] = None,
                         intent_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate a ranking with all available metrics."""
        results = {}

        # Basic diversity
        if len(ranked_items) > 1:
            results['pairwise_diversity'] = float(np.mean(pdist(ranked_items)))
        else:
            results['pairwise_diversity'] = 0.0

        # Coverage
        results['precision'] = self.coverage_pr.coverage_precision(ranked_items, reference_set)
        results['recall'] = self.coverage_pr.coverage_recall(ranked_items, reference_set)
        results['f1'] = self.coverage_pr.coverage_f1(ranked_items, reference_set)

        # nDCD
        results['ndcd'] = self.ndcd.ndcd(ranked_items, reference_set)

        # Subtopic metrics
        if subtopics is not None:
            all_st = set()
            for topics in subtopics.values():
                all_st.update(topics)

            doc_subtopics = {i: list(subtopics.get(i, set())) for i in range(len(ranked_items))}
            results['alpha_ndcg'] = self.alpha_ndcg.compute(
                list(range(len(ranked_items))), doc_subtopics, all_st
            )

            selected_topics = [subtopics.get(i, set()) for i in range(len(ranked_items))]
            results['subtopic_recall'] = self.subtopic_recall.compute(selected_topics, all_st)

        # ERR-IA
        if intent_relevance is not None and intent_probs is not None:
            results['err_ia'] = self.err_ia.compute(
                list(range(len(ranked_items))), intent_relevance, intent_probs
            )

        # Diversity lift
        lift_result = self.diversity_lift.compute(ranked_items, reference_set)
        results['diversity_lift'] = lift_result['normalized_lift']
        results['lift_percentile'] = lift_result['percentile']

        return results

    def compare_methods(self, method_selections: Dict[str, np.ndarray],
                        reference_set: np.ndarray,
                        subtopics: Optional[Dict[int, Set[int]]] = None) -> Dict[str, EvalReport]:
        """Compare multiple diversity methods."""
        reports = {}
        for method_name, selected in method_selections.items():
            reports[method_name] = self.evaluate(selected, reference_set, subtopics)
        return reports
