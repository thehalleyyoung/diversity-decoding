"""
Diversity in recommendation systems: calibrated, serendipitous, novel, intra-list diverse,
coverage-optimized, fairness-aware, temporally diverse, multi-stakeholder recommendations.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform, cosine
from scipy.stats import entropy as sp_entropy
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import math


@dataclass
class RecommendationItem:
    """An item that can be recommended."""
    item_id: int
    features: np.ndarray
    categories: List[int] = field(default_factory=list)
    provider_id: int = 0
    popularity: float = 0.0
    quality: float = 0.0


@dataclass
class UserProfile:
    """User profile for personalized recommendations."""
    user_id: int
    preference_vector: np.ndarray
    history: List[int] = field(default_factory=list)
    category_distribution: Optional[np.ndarray] = None
    session_history: List[List[int]] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Result of a recommendation."""
    items: List[int]
    scores: List[float]
    diversity_score: float = 0.0
    calibration_score: float = 0.0
    novelty_score: float = 0.0
    serendipity_score: float = 0.0
    coverage_score: float = 0.0
    fairness_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class RelevanceScorer:
    """Compute relevance between users and items."""

    def __init__(self):
        pass

    def dot_product_relevance(self, user_vec: np.ndarray,
                               item_features: np.ndarray) -> np.ndarray:
        """Relevance as dot product between user preference and item features."""
        scores = item_features @ user_vec
        return scores

    def cosine_relevance(self, user_vec: np.ndarray,
                          item_features: np.ndarray) -> np.ndarray:
        """Cosine similarity relevance."""
        user_norm = np.linalg.norm(user_vec)
        if user_norm < 1e-15:
            return np.zeros(len(item_features))
        item_norms = np.linalg.norm(item_features, axis=1)
        item_norms = np.maximum(item_norms, 1e-15)
        return (item_features @ user_vec) / (item_norms * user_norm)

    def collaborative_relevance(self, user_vec: np.ndarray,
                                 item_features: np.ndarray,
                                 user_item_matrix: np.ndarray,
                                 user_idx: int) -> np.ndarray:
        """Simple collaborative filtering relevance."""
        user_sims = user_item_matrix @ user_item_matrix[user_idx]
        user_sims[user_idx] = 0
        user_sims_norm = user_sims / (np.sum(np.abs(user_sims)) + 1e-15)
        pred = user_sims_norm @ user_item_matrix
        return pred


class CalibratedRecommender:
    """Match topic/category distribution to user interests."""

    def __init__(self, n_categories: int = 10):
        self.n_categories = n_categories

    def compute_user_distribution(self, history_categories: List[List[int]]) -> np.ndarray:
        """Compute user's category distribution from history."""
        dist = np.zeros(self.n_categories)
        for cats in history_categories:
            for c in cats:
                if 0 <= c < self.n_categories:
                    dist[c] += 1
        total = dist.sum()
        if total > 0:
            dist = dist / total
        else:
            dist = np.ones(self.n_categories) / self.n_categories
        return dist

    def compute_list_distribution(self, items: List[RecommendationItem]) -> np.ndarray:
        """Compute category distribution of recommendation list."""
        dist = np.zeros(self.n_categories)
        for item in items:
            for c in item.categories:
                if 0 <= c < self.n_categories:
                    dist[c] += 1
        total = dist.sum()
        if total > 0:
            dist = dist / total
        return dist

    def calibration_score(self, target_dist: np.ndarray,
                          list_dist: np.ndarray) -> float:
        """KL divergence between target and list distributions (lower = better)."""
        target = np.clip(target_dist, 1e-10, 1.0)
        target = target / target.sum()
        list_d = np.clip(list_dist, 1e-10, 1.0)
        list_d = list_d / list_d.sum()
        return float(np.sum(target * np.log(target / list_d)))

    def calibrated_rerank(self, candidates: List[RecommendationItem],
                          relevance_scores: np.ndarray,
                          target_dist: np.ndarray, k: int,
                          lambda_cal: float = 0.5) -> List[int]:
        """Re-rank for calibration: greedily build list matching target distribution."""
        selected: List[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                test_selected = selected + [idx]
                test_items = [candidates[i] for i in test_selected]
                list_dist = self.compute_list_distribution(test_items)
                cal = self.calibration_score(target_dist, list_dist)

                score = (1 - lambda_cal) * relevance_scores[idx] - lambda_cal * cal
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected


class SerendipityOptimizer:
    """Recommend surprising but relevant items."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def surprise_score(self, item_features: np.ndarray,
                       user_history_features: np.ndarray) -> float:
        """How surprising is an item given user history? (distance from history centroid)."""
        if len(user_history_features) == 0:
            return 1.0
        centroid = np.mean(user_history_features, axis=0)
        dist = np.linalg.norm(item_features - centroid)
        return float(dist)

    def serendipity_score(self, item_features: np.ndarray,
                          user_history_features: np.ndarray,
                          relevance: float) -> float:
        """Serendipity = surprise * relevance."""
        surprise = self.surprise_score(item_features, user_history_features)
        return surprise * max(relevance, 0)

    def serendipitous_rerank(self, candidates: List[RecommendationItem],
                             relevance_scores: np.ndarray,
                             user_history_features: np.ndarray,
                             k: int, lambda_ser: float = 0.3) -> List[int]:
        """Re-rank for serendipity."""
        scores = np.zeros(len(candidates))
        for i, item in enumerate(candidates):
            surprise = self.surprise_score(item.features, user_history_features)
            scores[i] = (1 - lambda_ser) * relevance_scores[i] + lambda_ser * surprise * relevance_scores[i]

        return list(np.argsort(scores)[-k:][::-1])


class NoveltyScorer:
    """Penalize items similar to user's history."""

    def __init__(self):
        pass

    def item_novelty(self, item_features: np.ndarray,
                      history_features: np.ndarray) -> float:
        """Novelty: min distance to any item in history."""
        if len(history_features) == 0:
            return 1.0
        dists = cdist(item_features.reshape(1, -1), history_features).flatten()
        return float(np.min(dists))

    def popularity_novelty(self, popularity: float, max_popularity: float = 1.0) -> float:
        """Novelty as inverse popularity."""
        return 1.0 - (popularity / max(max_popularity, 1e-15))

    def combined_novelty(self, item: RecommendationItem,
                          history_features: np.ndarray,
                          alpha: float = 0.5) -> float:
        """Combined content and popularity novelty."""
        content_nov = self.item_novelty(item.features, history_features)
        pop_nov = self.popularity_novelty(item.popularity)
        return alpha * content_nov + (1 - alpha) * pop_nov

    def novelty_aware_rerank(self, candidates: List[RecommendationItem],
                              relevance_scores: np.ndarray,
                              history_features: np.ndarray,
                              k: int, lambda_nov: float = 0.3) -> List[int]:
        """Re-rank penalizing items similar to history."""
        scores = np.zeros(len(candidates))
        for i, item in enumerate(candidates):
            nov = self.combined_novelty(item, history_features)
            scores[i] = (1 - lambda_nov) * relevance_scores[i] + lambda_nov * nov

        return list(np.argsort(scores)[-k:][::-1])


class IntraListDiversity:
    """Ensure items in recommendation list are diverse from each other."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def ild_score(self, items: List[RecommendationItem]) -> float:
        """Intra-list diversity: average pairwise distance."""
        if len(items) < 2:
            return 0.0
        features = np.array([item.features for item in items])
        dists = pdist(features, metric=self.metric)
        return float(np.mean(dists))

    def mmr_rerank(self, candidates: List[RecommendationItem],
                   relevance_scores: np.ndarray, k: int,
                   lambda_mmr: float = 0.5) -> List[int]:
        """MMR re-ranking for intra-list diversity."""
        features = np.array([c.features for c in candidates])
        sim_matrix = 1.0 - squareform(pdist(features, metric='cosine'))

        selected: List[int] = []
        remaining = list(range(len(candidates)))

        first = int(np.argmax(relevance_scores))
        selected.append(first)
        remaining.remove(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                max_sim = max(sim_matrix[idx, s] for s in selected)
                score = lambda_mmr * relevance_scores[idx] - (1 - lambda_mmr) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def dpp_rerank(self, candidates: List[RecommendationItem],
                   relevance_scores: np.ndarray, k: int) -> List[int]:
        """DPP-based re-ranking for diversity."""
        features = np.array([c.features for c in candidates])
        quality = np.sqrt(np.maximum(relevance_scores, 0))

        kernel = features @ features.T
        kernel = np.diag(quality) @ kernel @ np.diag(quality)
        kernel += 1e-6 * np.eye(len(kernel))

        selected: List[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                test = selected + [idx]
                sub_k = kernel[np.ix_(test, test)]
                sign, logdet = np.linalg.slogdet(sub_k)
                score = logdet if sign > 0 else -100
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected


class CoverageOptimizer:
    """Recommend items from underrepresented categories."""

    def __init__(self, n_categories: int = 10):
        self.n_categories = n_categories

    def category_coverage(self, items: List[RecommendationItem]) -> float:
        """Fraction of categories covered."""
        covered = set()
        for item in items:
            covered.update(item.categories)
        return len(covered) / self.n_categories

    def coverage_rerank(self, candidates: List[RecommendationItem],
                        relevance_scores: np.ndarray, k: int,
                        lambda_cov: float = 0.3) -> List[int]:
        """Greedy re-rank for coverage."""
        selected: List[int] = []
        remaining = list(range(len(candidates)))
        covered: Set[int] = set()

        for _ in range(min(k, len(candidates))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                new_cats = set(candidates[idx].categories) - covered
                coverage_gain = len(new_cats) / self.n_categories
                score = (1 - lambda_cov) * relevance_scores[idx] + lambda_cov * coverage_gain
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                covered.update(candidates[best_idx].categories)
                remaining.remove(best_idx)

        return selected


class FairnessAwareRecommender:
    """Exposure fairness across item providers."""

    def __init__(self, n_providers: int = 5):
        self.n_providers = n_providers

    def provider_exposure(self, items: List[RecommendationItem],
                          position_discount: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute exposure per provider with position discount."""
        exposure = np.zeros(self.n_providers)
        for i, item in enumerate(items):
            discount = 1.0 / np.log2(i + 2) if position_discount is None else position_discount[i]
            if 0 <= item.provider_id < self.n_providers:
                exposure[item.provider_id] += discount
        return exposure

    def fairness_score(self, exposure: np.ndarray,
                       target_exposure: Optional[np.ndarray] = None) -> float:
        """Fairness as negative max deviation from target (uniform by default)."""
        if target_exposure is None:
            target_exposure = np.ones(self.n_providers) / self.n_providers

        total_exp = exposure.sum()
        if total_exp > 0:
            normalized = exposure / total_exp
        else:
            normalized = np.ones(self.n_providers) / self.n_providers

        return 1.0 - float(np.max(np.abs(normalized - target_exposure)))

    def fair_rerank(self, candidates: List[RecommendationItem],
                    relevance_scores: np.ndarray, k: int,
                    lambda_fair: float = 0.3) -> List[int]:
        """Re-rank for provider fairness."""
        selected: List[int] = []
        remaining = list(range(len(candidates)))
        provider_counts = np.zeros(self.n_providers)

        for pos in range(min(k, len(candidates))):
            best_score = -np.inf
            best_idx = -1

            total_so_far = provider_counts.sum() + 1
            target = total_so_far / self.n_providers

            for idx in remaining:
                pid = candidates[idx].provider_id
                if 0 <= pid < self.n_providers:
                    fairness_bonus = max(0, target - provider_counts[pid])
                else:
                    fairness_bonus = 0

                score = (1 - lambda_fair) * relevance_scores[idx] + lambda_fair * fairness_bonus
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                pid = candidates[best_idx].provider_id
                if 0 <= pid < self.n_providers:
                    provider_counts[pid] += 1
                remaining.remove(best_idx)

        return selected


class TemporalDiversity:
    """Avoid recommending similar items across sessions."""

    def __init__(self, decay: float = 0.8):
        self.decay = decay

    def temporal_penalty(self, item_features: np.ndarray,
                          session_history: List[List[np.ndarray]],
                          metric: str = 'euclidean') -> float:
        """Penalty based on similarity to items in recent sessions, with decay."""
        if not session_history:
            return 0.0
        total_penalty = 0.0
        for session_idx, session_items in enumerate(reversed(session_history)):
            weight = self.decay ** session_idx
            for past_features in session_items:
                dist = np.linalg.norm(item_features - past_features)
                sim = 1.0 / (1.0 + dist)
                total_penalty += weight * sim
        return total_penalty

    def temporal_rerank(self, candidates: List[RecommendationItem],
                        relevance_scores: np.ndarray,
                        session_history: List[List[np.ndarray]],
                        k: int, lambda_temp: float = 0.3) -> List[int]:
        """Re-rank penalizing items similar to recent sessions."""
        scores = np.zeros(len(candidates))
        for i, item in enumerate(candidates):
            penalty = self.temporal_penalty(item.features, session_history)
            scores[i] = relevance_scores[i] - lambda_temp * penalty

        return list(np.argsort(scores)[-k:][::-1])


class MultiStakeholderDiversity:
    """Balance user satisfaction, provider fairness, platform objectives."""

    def __init__(self, n_categories: int = 10, n_providers: int = 5):
        self.n_categories = n_categories
        self.n_providers = n_providers

    def multi_objective_score(self, candidates: List[RecommendationItem],
                              selected_indices: List[int],
                              relevance_scores: np.ndarray,
                              user_cat_dist: np.ndarray,
                              weights: Dict[str, float] = None) -> Dict[str, float]:
        """Compute multi-objective score for current selection."""
        if weights is None:
            weights = {'relevance': 0.4, 'diversity': 0.2,
                      'calibration': 0.15, 'fairness': 0.15, 'coverage': 0.1}

        selected_items = [candidates[i] for i in selected_indices]

        # Relevance
        rel = float(np.mean([relevance_scores[i] for i in selected_indices])) if selected_indices else 0.0

        # Intra-list diversity
        ild_scorer = IntraListDiversity()
        div = ild_scorer.ild_score(selected_items)

        # Calibration
        cal_rec = CalibratedRecommender(self.n_categories)
        list_dist = cal_rec.compute_list_distribution(selected_items)
        cal = 1.0 - cal_rec.calibration_score(user_cat_dist, list_dist) if np.any(list_dist > 0) else 0.0
        cal = max(0, cal)

        # Fairness
        fair_rec = FairnessAwareRecommender(self.n_providers)
        exposure = fair_rec.provider_exposure(selected_items)
        fair = fair_rec.fairness_score(exposure)

        # Coverage
        cov_opt = CoverageOptimizer(self.n_categories)
        cov = cov_opt.category_coverage(selected_items)

        total = (weights['relevance'] * rel + weights['diversity'] * div +
                 weights['calibration'] * cal + weights['fairness'] * fair +
                 weights['coverage'] * cov)

        return {
            'total': total,
            'relevance': rel,
            'diversity': div,
            'calibration': cal,
            'fairness': fair,
            'coverage': cov
        }

    def multi_stakeholder_rerank(self, candidates: List[RecommendationItem],
                                  relevance_scores: np.ndarray,
                                  user_cat_dist: np.ndarray,
                                  k: int,
                                  weights: Dict[str, float] = None) -> List[int]:
        """Greedy re-rank optimizing multi-stakeholder objectives."""
        selected: List[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                test_selected = selected + [idx]
                obj = self.multi_objective_score(
                    candidates, test_selected, relevance_scores,
                    user_cat_dist, weights
                )
                if obj['total'] > best_score:
                    best_score = obj['total']
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected


class DiverseRecommender:
    """Main class: generate diverse recommendations."""

    def __init__(self, n_categories: int = 10, n_providers: int = 5, seed: int = 42):
        self.n_categories = n_categories
        self.n_providers = n_providers
        self.rng = np.random.RandomState(seed)
        self.relevance_scorer = RelevanceScorer()
        self.calibrated = CalibratedRecommender(n_categories)
        self.serendipity = SerendipityOptimizer(seed)
        self.novelty = NoveltyScorer()
        self.ild = IntraListDiversity()
        self.coverage = CoverageOptimizer(n_categories)
        self.fairness = FairnessAwareRecommender(n_providers)
        self.temporal = TemporalDiversity()
        self.multi_stakeholder = MultiStakeholderDiversity(n_categories, n_providers)

    def recommend(self, user_profile: UserProfile,
                  item_pool: List[RecommendationItem], k: int,
                  method: str = 'multi_stakeholder') -> RecommendationResult:
        """Generate diverse recommendations."""
        relevance = self.relevance_scorer.cosine_relevance(
            user_profile.preference_vector,
            np.array([item.features for item in item_pool])
        )

        history_features = np.array([
            item_pool[i].features for i in user_profile.history
            if i < len(item_pool)
        ]) if user_profile.history else np.empty((0, item_pool[0].features.shape[0]))

        user_cat_dist = user_profile.category_distribution
        if user_cat_dist is None:
            history_cats = [item_pool[i].categories for i in user_profile.history
                          if i < len(item_pool)]
            user_cat_dist = self.calibrated.compute_user_distribution(history_cats)

        if method == 'calibrated':
            indices = self.calibrated.calibrated_rerank(
                item_pool, relevance, user_cat_dist, k
            )
        elif method == 'serendipity':
            indices = self.serendipity.serendipitous_rerank(
                item_pool, relevance, history_features, k
            )
        elif method == 'novelty':
            indices = self.novelty.novelty_aware_rerank(
                item_pool, relevance, history_features, k
            )
        elif method == 'mmr':
            indices = self.ild.mmr_rerank(item_pool, relevance, k)
        elif method == 'coverage':
            indices = self.coverage.coverage_rerank(item_pool, relevance, k)
        elif method == 'fairness':
            indices = self.fairness.fair_rerank(item_pool, relevance, k)
        elif method == 'multi_stakeholder':
            indices = self.multi_stakeholder.multi_stakeholder_rerank(
                item_pool, relevance, user_cat_dist, k
            )
        else:
            indices = list(np.argsort(relevance)[-k:][::-1])

        selected_items = [item_pool[i] for i in indices]

        list_dist = self.calibrated.compute_list_distribution(selected_items)
        cal_score = 1.0 - self.calibrated.calibration_score(user_cat_dist, list_dist)
        div_score = self.ild.ild_score(selected_items)
        cov_score = self.coverage.category_coverage(selected_items)
        exposure = self.fairness.provider_exposure(selected_items)
        fair_score = self.fairness.fairness_score(exposure)

        nov_scores = []
        ser_scores = []
        for i in indices:
            nov_scores.append(self.novelty.combined_novelty(
                item_pool[i], history_features
            ))
            ser_scores.append(self.serendipity.serendipity_score(
                item_pool[i].features, history_features, relevance[i]
            ))

        return RecommendationResult(
            items=indices,
            scores=[float(relevance[i]) for i in indices],
            diversity_score=div_score,
            calibration_score=max(0, cal_score),
            novelty_score=float(np.mean(nov_scores)) if nov_scores else 0.0,
            serendipity_score=float(np.mean(ser_scores)) if ser_scores else 0.0,
            coverage_score=cov_score,
            fairness_score=fair_score,
            details={'method': method}
        )

    def create_test_items(self, n_items: int, dim: int = 20) -> List[RecommendationItem]:
        """Create synthetic test items."""
        items = []
        for i in range(n_items):
            features = self.rng.randn(dim)
            n_cats = self.rng.randint(1, 4)
            cats = self.rng.choice(self.n_categories, size=n_cats, replace=False).tolist()
            items.append(RecommendationItem(
                item_id=i,
                features=features,
                categories=cats,
                provider_id=int(self.rng.randint(self.n_providers)),
                popularity=float(self.rng.uniform(0, 1)),
                quality=float(self.rng.uniform(0.3, 1.0))
            ))
        return items

    def create_test_user(self, dim: int = 20,
                          n_history: int = 10) -> UserProfile:
        """Create synthetic test user."""
        pref = self.rng.randn(dim)
        history = self.rng.choice(100, size=n_history, replace=False).tolist()
        cat_dist = self.rng.dirichlet(np.ones(self.n_categories))
        return UserProfile(
            user_id=0,
            preference_vector=pref,
            history=history,
            category_distribution=cat_dist
        )
