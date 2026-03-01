"""
Diversity-aware information retrieval: PM-2, xQuAD, IA-Select, MMR for retrieval,
subtopic coverage, novelty-biased ranking, diversified re-ranking.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform, cosine
from scipy.stats import entropy as sp_entropy
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import math


@dataclass
class Document:
    """A document in the corpus."""
    doc_id: int
    features: np.ndarray
    subtopics: List[int] = field(default_factory=list)
    relevance: float = 0.0
    text_hash: int = 0


@dataclass
class QueryAspect:
    """A query aspect or intent."""
    aspect_id: int
    weight: float = 1.0
    description: str = ""
    covered: bool = False


@dataclass
class RetrievalResult:
    """Result of diverse retrieval."""
    doc_ids: List[int]
    scores: List[float]
    diversity_score: float = 0.0
    subtopic_coverage: float = 0.0
    alpha_ndcg: float = 0.0
    err_ia: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class RelevanceModel:
    """Compute relevance between queries and documents."""

    def __init__(self):
        pass

    def cosine_relevance(self, query: np.ndarray,
                          doc_features: np.ndarray) -> np.ndarray:
        """Cosine similarity relevance."""
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-15:
            return np.zeros(len(doc_features))
        d_norms = np.linalg.norm(doc_features, axis=1)
        d_norms = np.maximum(d_norms, 1e-15)
        return (doc_features @ query) / (d_norms * q_norm)

    def bm25_relevance(self, query_terms: np.ndarray,
                        doc_term_matrix: np.ndarray,
                        doc_lengths: np.ndarray,
                        avg_dl: float, k1: float = 1.2,
                        b: float = 0.75) -> np.ndarray:
        """BM25-like relevance score."""
        n_docs = doc_term_matrix.shape[0]
        df = np.sum(doc_term_matrix > 0, axis=0) + 1
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        scores = np.zeros(n_docs)
        for d in range(n_docs):
            tf = doc_term_matrix[d]
            norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lengths[d] / avg_dl))
            scores[d] = float(np.sum(norm_tf * idf * query_terms))

        return scores


class PM2Diversifier:
    """PM-2: Proportional representation for search result diversification."""

    def __init__(self, n_aspects: int = 5):
        self.n_aspects = n_aspects

    def diversify(self, documents: List[Document],
                  relevance_scores: np.ndarray,
                  aspect_weights: np.ndarray,
                  doc_aspect_probs: np.ndarray,
                  k: int) -> List[int]:
        """PM-2 proportional diversification.
        doc_aspect_probs: (n_docs, n_aspects) probability of doc covering each aspect.
        """
        selected: List[int] = []
        remaining = set(range(len(documents)))
        slot_counts = np.zeros(self.n_aspects)

        for pos in range(min(k, len(documents))):
            quotients = aspect_weights / (2 * slot_counts + 1)
            target_aspect = int(np.argmax(quotients))

            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                coverage_gain = doc_aspect_probs[idx, target_aspect]
                other_gain = np.sum(doc_aspect_probs[idx]) - coverage_gain
                score = coverage_gain * (quotients[target_aspect]) + 0.5 * other_gain * np.mean(quotients)
                score += 0.1 * relevance_scores[idx]

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)
                slot_counts += doc_aspect_probs[best_idx]

        return selected


class XQuADDiversifier:
    """xQuAD: Explicit query aspect diversification."""

    def __init__(self, n_aspects: int = 5, lambda_xquad: float = 0.5):
        self.n_aspects = n_aspects
        self.lambda_xquad = lambda_xquad

    def diversify(self, documents: List[Document],
                  relevance_scores: np.ndarray,
                  aspect_weights: np.ndarray,
                  doc_aspect_probs: np.ndarray,
                  k: int) -> List[int]:
        """xQuAD diversification.
        Maximizes: (1-λ)*P(d|q) + λ * Σ_i P(q_i|q) * P(d|q_i) * Π_{d'∈S} (1-P(d'|q_i))
        """
        selected: List[int] = []
        remaining = set(range(len(documents)))
        coverage_product = np.ones(self.n_aspects)

        for _ in range(min(k, len(documents))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                diversity_score = 0.0
                for a in range(self.n_aspects):
                    diversity_score += (
                        aspect_weights[a] *
                        doc_aspect_probs[idx, a] *
                        coverage_product[a]
                    )

                score = ((1 - self.lambda_xquad) * relevance_scores[idx] +
                         self.lambda_xquad * diversity_score)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)
                for a in range(self.n_aspects):
                    coverage_product[a] *= (1 - doc_aspect_probs[best_idx, a])

        return selected


class IASelectDiversifier:
    """IA-Select: Intent-aware diversification."""

    def __init__(self, n_intents: int = 5):
        self.n_intents = n_intents

    def diversify(self, documents: List[Document],
                  relevance_scores: np.ndarray,
                  intent_probs: np.ndarray,
                  doc_intent_relevance: np.ndarray,
                  k: int) -> List[int]:
        """IA-Select: maximize probability of satisfying at least one intent.
        intent_probs: probability of each intent
        doc_intent_relevance: (n_docs, n_intents) relevance per intent
        """
        selected: List[int] = []
        remaining = set(range(len(documents)))
        unsatisfied_prob = np.copy(intent_probs)

        for _ in range(min(k, len(documents))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                marginal_utility = 0.0
                for intent in range(self.n_intents):
                    marginal_utility += (
                        unsatisfied_prob[intent] *
                        doc_intent_relevance[idx, intent]
                    )

                if marginal_utility > best_score:
                    best_score = marginal_utility
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)
                for intent in range(self.n_intents):
                    unsatisfied_prob[intent] *= (1 - doc_intent_relevance[best_idx, intent])

        return selected


class MMRRetriever:
    """MMR for retrieval: balance relevance and diversity."""

    def __init__(self, lambda_mmr: float = 0.5, metric: str = 'cosine'):
        self.lambda_mmr = lambda_mmr
        self.metric = metric

    def retrieve(self, query: np.ndarray, documents: List[Document],
                 k: int) -> List[int]:
        """MMR retrieval."""
        features = np.array([d.features for d in documents])
        relevance = RelevanceModel().cosine_relevance(query, features)

        if self.metric == 'cosine':
            sim_matrix = 1.0 - squareform(pdist(features, metric='cosine'))
        else:
            dists = squareform(pdist(features))
            max_d = np.max(dists) + 1e-10
            sim_matrix = 1.0 - dists / max_d

        selected: List[int] = []
        remaining = set(range(len(documents)))

        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                max_sim = max(sim_matrix[idx, s] for s in selected)
                score = self.lambda_mmr * relevance[idx] - (1 - self.lambda_mmr) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected


class SubtopicCoverage:
    """Ensure results cover all subtopics of the query."""

    def __init__(self, n_subtopics: int = 10):
        self.n_subtopics = n_subtopics

    def coverage_score(self, selected_docs: List[Document]) -> float:
        """Fraction of subtopics covered by selected documents."""
        covered = set()
        for doc in selected_docs:
            covered.update(doc.subtopics)
        return len(covered) / max(self.n_subtopics, 1)

    def subtopic_recall(self, selected_docs: List[Document],
                        all_subtopics: Set[int]) -> float:
        """Recall of subtopics."""
        covered = set()
        for doc in selected_docs:
            covered.update(doc.subtopics)
        if not all_subtopics:
            return 1.0
        return len(covered & all_subtopics) / len(all_subtopics)

    def greedy_subtopic_cover(self, documents: List[Document],
                               relevance_scores: np.ndarray,
                               k: int,
                               lambda_cov: float = 0.5) -> List[int]:
        """Greedily select documents to maximize subtopic coverage."""
        selected: List[int] = []
        remaining = set(range(len(documents)))
        covered: Set[int] = set()

        for _ in range(min(k, len(documents))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                new_topics = set(documents[idx].subtopics) - covered
                coverage_gain = len(new_topics) / max(self.n_subtopics, 1)
                score = (1 - lambda_cov) * relevance_scores[idx] + lambda_cov * coverage_gain
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                covered.update(documents[best_idx].subtopics)
                remaining.discard(best_idx)

        return selected

    def weighted_subtopic_cover(self, documents: List[Document],
                                 relevance_scores: np.ndarray,
                                 subtopic_weights: np.ndarray,
                                 k: int) -> List[int]:
        """Cover subtopics weighted by importance."""
        selected: List[int] = []
        remaining = set(range(len(documents)))
        covered_weight = 0.0
        covered: Set[int] = set()
        total_weight = float(np.sum(subtopic_weights))

        for _ in range(min(k, len(documents))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                new_topics = set(documents[idx].subtopics) - covered
                weight_gain = sum(
                    subtopic_weights[t] for t in new_topics
                    if t < len(subtopic_weights)
                ) / max(total_weight, 1e-10)

                score = 0.5 * relevance_scores[idx] + 0.5 * weight_gain
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                new_topics = set(documents[best_idx].subtopics) - covered
                for t in new_topics:
                    if t < len(subtopic_weights):
                        covered_weight += subtopic_weights[t]
                covered.update(documents[best_idx].subtopics)
                remaining.discard(best_idx)

        return selected


class NoveltyBiasedRanker:
    """Discount documents similar to higher-ranked ones."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def novelty_biased_rank(self, documents: List[Document],
                             relevance_scores: np.ndarray,
                             k: int, beta: float = 0.5) -> List[int]:
        """Rank with novelty discount: penalize similarity to already-ranked docs."""
        features = np.array([d.features for d in documents])
        sim_matrix = features @ features.T
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        sim_matrix = sim_matrix / (norms @ norms.T)

        selected: List[int] = []
        remaining = set(range(len(documents)))

        for _ in range(min(k, len(documents))):
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                novelty_penalty = 0.0
                if selected:
                    max_sim = max(sim_matrix[idx, s] for s in selected)
                    novelty_penalty = max_sim

                score = relevance_scores[idx] - beta * novelty_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def discount_factors(self, documents: List[Document],
                          ranking: List[int]) -> np.ndarray:
        """Compute novelty discount factors for a given ranking."""
        features = np.array([documents[i].features for i in ranking])
        discounts = np.ones(len(ranking))

        for i in range(1, len(ranking)):
            prev_features = features[:i]
            sims = prev_features @ features[i] / (
                np.linalg.norm(prev_features, axis=1) *
                np.linalg.norm(features[i]) + 1e-10
            )
            discounts[i] = 1.0 - float(np.max(sims))

        return discounts


class DiversifiedReRanker:
    """Take top-N by relevance, re-rank for diversity."""

    def __init__(self, top_n: int = 100, metric: str = 'euclidean'):
        self.top_n = top_n
        self.metric = metric

    def rerank(self, documents: List[Document],
               relevance_scores: np.ndarray, k: int,
               method: str = 'mmr', lambda_div: float = 0.5) -> List[int]:
        """Re-rank top-N documents for diversity."""
        top_n_indices = np.argsort(relevance_scores)[-self.top_n:][::-1]
        top_n_docs = [documents[i] for i in top_n_indices]
        top_n_rel = relevance_scores[top_n_indices]

        if method == 'mmr':
            local_indices = self._mmr_rerank(top_n_docs, top_n_rel, k, lambda_div)
        elif method == 'maxsum':
            local_indices = self._maxsum_rerank(top_n_docs, top_n_rel, k, lambda_div)
        elif method == 'maxmin':
            local_indices = self._maxmin_rerank(top_n_docs, top_n_rel, k, lambda_div)
        else:
            local_indices = list(range(min(k, len(top_n_indices))))

        return [int(top_n_indices[i]) for i in local_indices]

    def _mmr_rerank(self, docs: List[Document], rel: np.ndarray,
                    k: int, lambda_div: float) -> List[int]:
        """MMR-based re-ranking."""
        features = np.array([d.features for d in docs])
        sim = features @ features.T
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        sim = sim / (norms @ norms.T)

        selected: List[int] = []
        remaining = set(range(len(docs)))

        first = int(np.argmax(rel))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1
            for idx in remaining:
                max_sim = max(sim[idx, s] for s in selected)
                score = lambda_div * rel[idx] - (1 - lambda_div) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def _maxsum_rerank(self, docs: List[Document], rel: np.ndarray,
                       k: int, lambda_div: float) -> List[int]:
        """MaxSum diversity: maximize sum of pairwise distances."""
        features = np.array([d.features for d in docs])
        dist_matrix = squareform(pdist(features))

        selected: List[int] = []
        remaining = set(range(len(docs)))

        first = int(np.argmax(rel))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1
            for idx in remaining:
                div_gain = sum(dist_matrix[idx, s] for s in selected)
                score = lambda_div * rel[idx] + (1 - lambda_div) * div_gain
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def _maxmin_rerank(self, docs: List[Document], rel: np.ndarray,
                       k: int, lambda_div: float) -> List[int]:
        """MaxMin diversity: maximize minimum pairwise distance."""
        features = np.array([d.features for d in docs])
        dist_matrix = squareform(pdist(features))

        selected: List[int] = []
        remaining = set(range(len(docs)))

        first = int(np.argmax(rel))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1
            for idx in remaining:
                min_dist = min(dist_matrix[idx, s] for s in selected)
                score = lambda_div * rel[idx] + (1 - lambda_div) * min_dist
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected


class DiverseRetriever:
    """Main class: diversity-aware information retrieval."""

    def __init__(self, n_subtopics: int = 10, n_aspects: int = 5, seed: int = 42):
        self.n_subtopics = n_subtopics
        self.n_aspects = n_aspects
        self.rng = np.random.RandomState(seed)
        self.relevance_model = RelevanceModel()
        self.pm2 = PM2Diversifier(n_aspects)
        self.xquad = XQuADDiversifier(n_aspects)
        self.ia_select = IASelectDiversifier(n_aspects)
        self.mmr = MMRRetriever()
        self.subtopic_cov = SubtopicCoverage(n_subtopics)
        self.novelty_ranker = NoveltyBiasedRanker()
        self.reranker = DiversifiedReRanker()

    def retrieve(self, query: np.ndarray, corpus: List[Document], k: int,
                 method: str = 'xquad') -> RetrievalResult:
        """Retrieve diverse documents for a query."""
        features = np.array([d.features for d in corpus])
        relevance = self.relevance_model.cosine_relevance(query, features)

        doc_aspect_probs = self._estimate_aspect_probs(corpus)
        aspect_weights = np.ones(self.n_aspects) / self.n_aspects

        if method == 'pm2':
            indices = self.pm2.diversify(corpus, relevance, aspect_weights,
                                         doc_aspect_probs, k)
        elif method == 'xquad':
            indices = self.xquad.diversify(corpus, relevance, aspect_weights,
                                           doc_aspect_probs, k)
        elif method == 'ia_select':
            intent_probs = aspect_weights
            indices = self.ia_select.diversify(corpus, relevance, intent_probs,
                                              doc_aspect_probs, k)
        elif method == 'mmr':
            indices = self.mmr.retrieve(query, corpus, k)
        elif method == 'subtopic':
            indices = self.subtopic_cov.greedy_subtopic_cover(
                corpus, relevance, k
            )
        elif method == 'novelty':
            indices = self.novelty_ranker.novelty_biased_rank(
                corpus, relevance, k
            )
        elif method == 'rerank':
            indices = self.reranker.rerank(corpus, relevance, k)
        else:
            indices = list(np.argsort(relevance)[-k:][::-1])

        selected_docs = [corpus[i] for i in indices]
        cov = self.subtopic_cov.coverage_score(selected_docs)

        if len(indices) > 1:
            sel_features = features[indices]
            dists = pdist(sel_features)
            div = float(np.mean(dists))
        else:
            div = 0.0

        return RetrievalResult(
            doc_ids=indices,
            scores=[float(relevance[i]) for i in indices],
            diversity_score=div,
            subtopic_coverage=cov,
            details={'method': method}
        )

    def _estimate_aspect_probs(self, corpus: List[Document]) -> np.ndarray:
        """Estimate document-aspect probabilities from subtopics."""
        n = len(corpus)
        probs = np.zeros((n, self.n_aspects))
        for i, doc in enumerate(corpus):
            for subtopic in doc.subtopics:
                aspect = subtopic % self.n_aspects
                probs[i, aspect] = max(probs[i, aspect], 0.5 + 0.5 * self.rng.rand())
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        probs = probs / row_sums
        return probs

    def create_test_corpus(self, n_docs: int, dim: int = 20,
                           n_topics: int = 5) -> List[Document]:
        """Create synthetic test corpus with topic structure."""
        topic_centers = self.rng.randn(n_topics, dim)
        docs = []
        for i in range(n_docs):
            topic = i % n_topics
            features = topic_centers[topic] + self.rng.randn(dim) * 0.5
            subtopics_for_doc = [topic]
            if self.rng.rand() > 0.5:
                extra = self.rng.randint(0, self.n_subtopics)
                subtopics_for_doc.append(extra)
            docs.append(Document(
                doc_id=i,
                features=features,
                subtopics=subtopics_for_doc,
                relevance=float(self.rng.uniform(0.1, 1.0))
            ))
        return docs
