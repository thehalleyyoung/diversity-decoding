"""
Active diverse selection: actively select diverse items using uncertainty,
Bayesian information gain, exploration-exploitation tradeoffs, contextual bandits,
and feedback integration.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import norm, entropy as sp_entropy
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
import math


@dataclass
class SelectionResult:
    """Result of an active diverse selection step."""
    selected_index: int
    score: float
    uncertainty: float = 0.0
    diversity_contribution: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSelectionResult:
    """Result of batch active diverse selection."""
    selected_indices: List[int]
    scores: List[float]
    total_diversity: float = 0.0
    total_uncertainty: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class UncertaintyEstimator:
    """Estimate model uncertainty for active selection."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def entropy_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Uncertainty via prediction entropy. predictions: (n_items, n_classes)."""
        predictions = np.clip(predictions, 1e-15, 1.0)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        return -np.sum(predictions * np.log2(predictions), axis=1)

    def margin_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Uncertainty via margin between top two predictions (lower margin = more uncertain)."""
        sorted_preds = np.sort(predictions, axis=1)[:, ::-1]
        margins = sorted_preds[:, 0] - sorted_preds[:, 1]
        return 1.0 - margins

    def least_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Uncertainty as 1 - max(prediction)."""
        return 1.0 - np.max(predictions, axis=1)

    def mc_dropout_uncertainty(self, features: np.ndarray, n_forward: int = 10,
                               dropout_rate: float = 0.3) -> np.ndarray:
        """Simulate MC dropout uncertainty with random linear models."""
        n, d = features.shape
        predictions = []

        for _ in range(n_forward):
            w = self.rng.randn(d, 5) * 0.5
            mask = self.rng.binomial(1, 1 - dropout_rate, size=w.shape)
            w_dropped = w * mask / (1 - dropout_rate)
            logits = features @ w_dropped
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            predictions.append(probs)

        predictions = np.stack(predictions, axis=0)
        mean_pred = predictions.mean(axis=0)
        epistemic = np.mean(np.var(predictions, axis=0), axis=1)
        aleatoric = np.mean(self.entropy_uncertainty(mean_pred))

        return epistemic

    def ensemble_uncertainty(self, features: np.ndarray,
                             n_models: int = 5) -> np.ndarray:
        """Uncertainty from ensemble disagreement."""
        n, d = features.shape
        predictions = []

        for _ in range(n_models):
            w = self.rng.randn(d, 5) * 0.3
            b = self.rng.randn(5) * 0.1
            logits = features @ w + b
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            predictions.append(probs)

        predictions = np.stack(predictions, axis=0)
        return np.mean(np.var(predictions, axis=0), axis=1)


class DiversityScorer:
    """Score items by their diversity contribution to already-selected set."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def min_distance_to_selected(self, pool: np.ndarray,
                                  selected: np.ndarray) -> np.ndarray:
        """Diversity as minimum distance to any selected item."""
        if len(selected) == 0:
            return np.ones(len(pool))
        dists = cdist(pool, selected, metric=self.metric)
        return np.min(dists, axis=1)

    def mean_distance_to_selected(self, pool: np.ndarray,
                                   selected: np.ndarray) -> np.ndarray:
        """Diversity as mean distance to selected items."""
        if len(selected) == 0:
            return np.ones(len(pool))
        dists = cdist(pool, selected, metric=self.metric)
        return np.mean(dists, axis=1)

    def max_distance_to_selected(self, pool: np.ndarray,
                                  selected: np.ndarray) -> np.ndarray:
        """Diversity as max distance to selected items (for coverage)."""
        if len(selected) == 0:
            return np.ones(len(pool))
        dists = cdist(pool, selected, metric=self.metric)
        return np.max(dists, axis=1)

    def determinantal_score(self, pool: np.ndarray,
                            selected: np.ndarray) -> np.ndarray:
        """DPP-inspired: volume contribution of each item."""
        if len(selected) == 0:
            return np.linalg.norm(pool, axis=1)

        scores = np.zeros(len(pool))
        for i, item in enumerate(pool):
            combined = np.vstack([selected, item.reshape(1, -1)])
            kernel = combined @ combined.T + 1e-6 * np.eye(len(combined))
            sign, logdet = np.linalg.slogdet(kernel)
            scores[i] = logdet if sign > 0 else -100
        return scores

    def coverage_gain(self, pool: np.ndarray, selected: np.ndarray,
                      all_points: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """How many new points would each candidate cover within radius?"""
        if len(selected) == 0:
            dists = cdist(pool, all_points, metric=self.metric)
            return np.sum(dists <= radius, axis=1).astype(float)

        covered = np.zeros(len(all_points), dtype=bool)
        sel_dists = cdist(selected, all_points, metric=self.metric)
        covered = np.any(sel_dists <= radius, axis=0)

        gains = np.zeros(len(pool))
        for i, item in enumerate(pool):
            d = cdist(item.reshape(1, -1), all_points, metric=self.metric).flatten()
            new_covered = (~covered) & (d <= radius)
            gains[i] = np.sum(new_covered)

        return gains


class ActiveDiverseSelector:
    """Actively select diverse items from a pool."""

    def __init__(self, diversity_weight: float = 0.5, uncertainty_method: str = 'entropy',
                 seed: int = 42):
        self.diversity_weight = diversity_weight
        self.uncertainty_method = uncertainty_method
        self.rng = np.random.RandomState(seed)
        self.uncertainty_estimator = UncertaintyEstimator(seed)
        self.diversity_scorer = DiversityScorer()
        self.feedback_history: List[Dict[str, Any]] = []
        self.learned_weights: Optional[np.ndarray] = None

    def select_next(self, pool: np.ndarray, selected: np.ndarray,
                    predictions: Optional[np.ndarray] = None,
                    query: Optional[np.ndarray] = None) -> SelectionResult:
        """Select next item balancing uncertainty and diversity."""
        if predictions is None:
            n_classes = 5
            logits = pool @ self.rng.randn(pool.shape[1], n_classes)
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            predictions = exp_l / exp_l.sum(axis=1, keepdims=True)

        uncertainty = self._compute_uncertainty(predictions)
        diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)

        u_norm = self._normalize(uncertainty)
        d_norm = self._normalize(diversity)

        w = self.diversity_weight
        scores = (1 - w) * u_norm + w * d_norm

        if query is not None:
            relevance = 1.0 / (1.0 + cdist(pool, query.reshape(1, -1)).flatten())
            scores = scores * relevance

        best_idx = int(np.argmax(scores))
        return SelectionResult(
            selected_index=best_idx,
            score=float(scores[best_idx]),
            uncertainty=float(uncertainty[best_idx]),
            diversity_contribution=float(diversity[best_idx])
        )

    def select_uncertainty_only(self, pool: np.ndarray,
                                predictions: np.ndarray) -> SelectionResult:
        """Pure uncertainty-based selection."""
        uncertainty = self._compute_uncertainty(predictions)
        best_idx = int(np.argmax(uncertainty))
        return SelectionResult(
            selected_index=best_idx,
            score=float(uncertainty[best_idx]),
            uncertainty=float(uncertainty[best_idx])
        )

    def select_diversity_weighted_uncertainty(self, pool: np.ndarray,
                                              selected: np.ndarray,
                                              predictions: np.ndarray,
                                              alpha: float = 0.5) -> SelectionResult:
        """Select by diversity-weighted uncertainty: u(x) * d(x)^alpha."""
        uncertainty = self._compute_uncertainty(predictions)
        diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)
        diversity = np.maximum(diversity, 1e-10)

        scores = uncertainty * (diversity ** alpha)
        best_idx = int(np.argmax(scores))
        return SelectionResult(
            selected_index=best_idx,
            score=float(scores[best_idx]),
            uncertainty=float(uncertainty[best_idx]),
            diversity_contribution=float(diversity[best_idx])
        )

    def batch_select(self, pool: np.ndarray, selected: np.ndarray,
                     k: int, predictions: Optional[np.ndarray] = None,
                     method: str = 'greedy') -> BatchSelectionResult:
        """Select k items at once considering diversity within the batch."""
        if method == 'greedy':
            return self._batch_greedy(pool, selected, k, predictions)
        elif method == 'determinantal':
            return self._batch_determinantal(pool, selected, k, predictions)
        elif method == 'clustering':
            return self._batch_clustering(pool, selected, k, predictions)
        else:
            return self._batch_greedy(pool, selected, k, predictions)

    def _batch_greedy(self, pool: np.ndarray, selected: np.ndarray,
                      k: int, predictions: Optional[np.ndarray]) -> BatchSelectionResult:
        """Greedy batch selection: sequentially pick items, updating diversity."""
        if predictions is None:
            n_classes = 5
            logits = pool @ self.rng.randn(pool.shape[1], n_classes)
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            predictions = exp_l / exp_l.sum(axis=1, keepdims=True)

        current_selected = selected.copy() if len(selected) > 0 else np.empty((0, pool.shape[1]))
        batch_indices: List[int] = []
        batch_scores: List[float] = []
        available = set(range(len(pool)))

        for _ in range(min(k, len(pool))):
            uncertainty = self._compute_uncertainty(predictions)
            diversity = self.diversity_scorer.min_distance_to_selected(
                pool, current_selected
            )

            u_norm = self._normalize(uncertainty)
            d_norm = self._normalize(diversity)
            scores = (1 - self.diversity_weight) * u_norm + self.diversity_weight * d_norm

            for idx in range(len(scores)):
                if idx not in available:
                    scores[idx] = -np.inf

            best_idx = int(np.argmax(scores))
            batch_indices.append(best_idx)
            batch_scores.append(float(scores[best_idx]))
            available.discard(best_idx)

            if len(current_selected) == 0:
                current_selected = pool[best_idx:best_idx + 1]
            else:
                current_selected = np.vstack([current_selected, pool[best_idx]])

        total_div = 0.0
        if len(batch_indices) > 1:
            batch_items = pool[batch_indices]
            dists = pdist(batch_items)
            total_div = float(np.sum(dists))

        return BatchSelectionResult(
            selected_indices=batch_indices,
            scores=batch_scores,
            total_diversity=total_div,
            total_uncertainty=float(np.mean([
                self._compute_uncertainty(predictions)[i] for i in batch_indices
            ]))
        )

    def _batch_determinantal(self, pool: np.ndarray, selected: np.ndarray,
                             k: int, predictions: Optional[np.ndarray]) -> BatchSelectionResult:
        """DPP-inspired batch selection for diversity."""
        n = len(pool)
        kernel = pool @ pool.T
        kernel = kernel / (np.max(np.abs(kernel)) + 1e-10)

        if predictions is not None:
            uncertainty = self._compute_uncertainty(predictions)
            quality = np.diag(uncertainty)
            kernel = quality @ kernel @ quality

        kernel += 1e-6 * np.eye(n)

        batch_indices: List[int] = []
        remaining = list(range(n))

        for _ in range(min(k, n)):
            if not remaining:
                break

            if not batch_indices:
                diag_vals = np.diag(kernel)
                best = remaining[int(np.argmax(diag_vals[remaining]))]
            else:
                best_score = -np.inf
                best = remaining[0]
                for idx in remaining:
                    test_set = batch_indices + [idx]
                    sub_kernel = kernel[np.ix_(test_set, test_set)]
                    sign, logdet = np.linalg.slogdet(sub_kernel)
                    score = logdet if sign > 0 else -100
                    if score > best_score:
                        best_score = score
                        best = idx

            batch_indices.append(best)
            remaining.remove(best)

        return BatchSelectionResult(
            selected_indices=batch_indices,
            scores=[float(kernel[i, i]) for i in batch_indices],
            total_diversity=float(np.sum(pdist(pool[batch_indices]))) if len(batch_indices) > 1 else 0.0
        )

    def _batch_clustering(self, pool: np.ndarray, selected: np.ndarray,
                          k: int, predictions: Optional[np.ndarray]) -> BatchSelectionResult:
        """Cluster pool into k groups, pick most uncertain from each."""
        from scipy.cluster.vq import kmeans2

        if len(pool) <= k:
            return BatchSelectionResult(
                selected_indices=list(range(len(pool))),
                scores=[1.0] * len(pool)
            )

        centroids, labels = kmeans2(pool.astype(float), k, minit='points',
                                    seed=self.rng.randint(10000))

        if predictions is None:
            n_classes = 5
            logits = pool @ self.rng.randn(pool.shape[1], n_classes)
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            predictions = exp_l / exp_l.sum(axis=1, keepdims=True)

        uncertainty = self._compute_uncertainty(predictions)

        batch_indices = []
        batch_scores = []
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue
            cluster_indices = np.where(cluster_mask)[0]
            cluster_uncert = uncertainty[cluster_indices]
            best_in_cluster = cluster_indices[np.argmax(cluster_uncert)]
            batch_indices.append(int(best_in_cluster))
            batch_scores.append(float(uncertainty[best_in_cluster]))

        return BatchSelectionResult(
            selected_indices=batch_indices,
            scores=batch_scores,
            total_diversity=float(np.sum(pdist(pool[batch_indices]))) if len(batch_indices) > 1 else 0.0
        )

    def bayesian_select(self, pool: np.ndarray, selected: np.ndarray,
                        predictions: np.ndarray,
                        prior_precision: float = 1.0) -> SelectionResult:
        """Bayesian active diverse selection: maximize expected information gain + diversity."""
        n, d = pool.shape
        n_classes = predictions.shape[1]

        pred_entropy = self._compute_uncertainty(predictions)

        posterior_var = np.zeros(n)
        for i in range(n):
            feature_norm = np.linalg.norm(pool[i])
            posterior_var[i] = 1.0 / (prior_precision + feature_norm ** 2 + 1e-10)

        info_gain = 0.5 * np.log(1.0 + posterior_var * np.sum(pool ** 2, axis=1))

        diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)

        ig_norm = self._normalize(info_gain)
        d_norm = self._normalize(diversity)

        scores = 0.5 * ig_norm + 0.5 * d_norm
        best_idx = int(np.argmax(scores))

        return SelectionResult(
            selected_index=best_idx,
            score=float(scores[best_idx]),
            uncertainty=float(info_gain[best_idx]),
            diversity_contribution=float(diversity[best_idx]),
            details={'info_gain': float(info_gain[best_idx]),
                     'posterior_var': float(posterior_var[best_idx])}
        )

    def ucb_select(self, pool: np.ndarray, selected: np.ndarray,
                   predictions: np.ndarray, t: int = 1,
                   exploration_coeff: float = 2.0,
                   diversity_bonus: float = 1.0) -> SelectionResult:
        """UCB-style selection with diversity bonus."""
        mean_quality = np.max(predictions, axis=1)
        uncertainty = self._compute_uncertainty(predictions)
        diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)

        exploration = exploration_coeff * np.sqrt(np.log(max(t, 1)) / (1 + uncertainty * 10))

        ucb_scores = mean_quality + exploration + diversity_bonus * self._normalize(diversity)

        best_idx = int(np.argmax(ucb_scores))
        return SelectionResult(
            selected_index=best_idx,
            score=float(ucb_scores[best_idx]),
            uncertainty=float(uncertainty[best_idx]),
            diversity_contribution=float(diversity[best_idx]),
            details={'exploration': float(exploration[best_idx]),
                     'mean_quality': float(mean_quality[best_idx])}
        )

    def contextual_bandit_select(self, pool: np.ndarray, selected: np.ndarray,
                                 context: np.ndarray,
                                 predictions: np.ndarray,
                                 context_weights: Optional[np.ndarray] = None) -> SelectionResult:
        """Contextual bandit: different diversity needs per context."""
        if context_weights is None:
            context_norm = np.linalg.norm(context)
            diversity_importance = 1.0 / (1.0 + context_norm)
        else:
            diversity_importance = float(np.dot(context, context_weights[:len(context)]))
            diversity_importance = 1.0 / (1.0 + np.exp(-diversity_importance))

        uncertainty = self._compute_uncertainty(predictions)
        diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)

        relevance = 1.0 / (1.0 + cdist(pool, context.reshape(1, -1)).flatten())

        scores = (
            (1 - diversity_importance) * self._normalize(uncertainty) * self._normalize(relevance) +
            diversity_importance * self._normalize(diversity)
        )

        best_idx = int(np.argmax(scores))
        return SelectionResult(
            selected_index=best_idx,
            score=float(scores[best_idx]),
            uncertainty=float(uncertainty[best_idx]),
            diversity_contribution=float(diversity[best_idx]),
            details={'diversity_importance': diversity_importance,
                     'relevance': float(relevance[best_idx])}
        )

    def integrate_feedback(self, item_idx: int, feedback_score: float,
                           pool: np.ndarray) -> None:
        """Learn from user feedback which items are truly diverse."""
        self.feedback_history.append({
            'item_idx': item_idx,
            'feedback': feedback_score,
            'features': pool[item_idx].copy()
        })

        if len(self.feedback_history) >= 5:
            self._update_weights(pool)

    def _update_weights(self, pool: np.ndarray) -> None:
        """Update diversity weights from feedback using simple linear regression."""
        if not self.feedback_history:
            return

        X = np.array([h['features'] for h in self.feedback_history])
        y = np.array([h['feedback'] for h in self.feedback_history])

        XtX = X.T @ X + 0.1 * np.eye(X.shape[1])
        Xty = X.T @ y
        self.learned_weights = np.linalg.solve(XtX, Xty)

    def select_with_learned_weights(self, pool: np.ndarray,
                                    selected: np.ndarray) -> SelectionResult:
        """Select using learned diversity weights from feedback."""
        if self.learned_weights is None:
            diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)
            best_idx = int(np.argmax(diversity))
            return SelectionResult(selected_index=best_idx, score=float(diversity[best_idx]))

        predicted_quality = pool @ self.learned_weights
        diversity = self.diversity_scorer.min_distance_to_selected(pool, selected)

        scores = self._normalize(predicted_quality) + self._normalize(diversity)
        best_idx = int(np.argmax(scores))

        return SelectionResult(
            selected_index=best_idx,
            score=float(scores[best_idx]),
            diversity_contribution=float(diversity[best_idx]),
            details={'predicted_quality': float(predicted_quality[best_idx])}
        )

    def _compute_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Compute uncertainty using configured method."""
        if self.uncertainty_method == 'entropy':
            return self.uncertainty_estimator.entropy_uncertainty(predictions)
        elif self.uncertainty_method == 'margin':
            return self.uncertainty_estimator.margin_uncertainty(predictions)
        elif self.uncertainty_method == 'least_confidence':
            return self.uncertainty_estimator.least_confidence(predictions)
        else:
            return self.uncertainty_estimator.entropy_uncertainty(predictions)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        vmin, vmax = np.min(values), np.max(values)
        if vmax - vmin < 1e-15:
            return np.ones_like(values) * 0.5
        return (values - vmin) / (vmax - vmin)


class ActiveDiverseSelectionSimulator:
    """Simulate active diverse selection over multiple rounds."""

    def __init__(self, pool: np.ndarray, selector: ActiveDiverseSelector,
                 seed: int = 42):
        self.pool = pool
        self.selector = selector
        self.rng = np.random.RandomState(seed)
        self.n = len(pool)

    def simulate(self, n_rounds: int, k_per_round: int = 1,
                 predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run active diverse selection simulation."""
        if predictions is None:
            n_classes = 5
            logits = self.pool @ self.rng.randn(self.pool.shape[1], n_classes) * 0.5
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            predictions = exp_l / exp_l.sum(axis=1, keepdims=True)

        selected_list: List[int] = []
        selected_features = np.empty((0, self.pool.shape[1]))
        round_results = []

        for round_idx in range(n_rounds):
            if k_per_round == 1:
                result = self.selector.select_next(
                    self.pool, selected_features, predictions
                )
                idx = result.selected_index
                if idx not in selected_list:
                    selected_list.append(idx)
                    selected_features = np.vstack([
                        selected_features, self.pool[idx:idx + 1]
                    ]) if len(selected_features) > 0 else self.pool[idx:idx + 1]
            else:
                result = self.selector.batch_select(
                    self.pool, selected_features, k_per_round, predictions
                )
                for idx in result.selected_indices:
                    if idx not in selected_list:
                        selected_list.append(idx)
                        selected_features = np.vstack([
                            selected_features, self.pool[idx:idx + 1]
                        ]) if len(selected_features) > 0 else self.pool[idx:idx + 1]

            coverage = self._compute_coverage(selected_features)
            diversity = self._compute_diversity(selected_features)

            round_results.append({
                'round': round_idx,
                'n_selected': len(selected_list),
                'coverage': coverage,
                'diversity': diversity,
            })

        return {
            'selected_indices': selected_list,
            'n_selected': len(selected_list),
            'final_coverage': round_results[-1]['coverage'] if round_results else 0.0,
            'final_diversity': round_results[-1]['diversity'] if round_results else 0.0,
            'round_results': round_results,
        }

    def _compute_coverage(self, selected: np.ndarray, radius: float = 1.5) -> float:
        """Fraction of pool covered by selected items within radius."""
        if len(selected) == 0:
            return 0.0
        dists = cdist(self.pool, selected)
        min_dists = np.min(dists, axis=1)
        covered = np.sum(min_dists <= radius)
        return float(covered) / self.n

    def _compute_diversity(self, selected: np.ndarray) -> float:
        """Average pairwise distance among selected."""
        if len(selected) < 2:
            return 0.0
        dists = pdist(selected)
        return float(np.mean(dists))
