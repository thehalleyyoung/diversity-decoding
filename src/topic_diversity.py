"""
Topic-aware diversity analysis.

Extract topics from text collections, measure topic coverage, select
texts to maximise topic diversity, and identify coverage gaps. Uses
simple NMF-based topic modelling with TF-IDF — no external NLP deps.
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "was", "are", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "if", "then", "than", "that", "this", "these", "those", "as", "up",
    "out", "about", "into", "over", "after", "before", "between", "under",
    "again", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "too", "very", "just", "because", "through",
    "during", "while", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "his", "her", "its", "us",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \\1 ", text)
    tokens = _WS_RE.split(text)
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Simple TF-IDF
# ---------------------------------------------------------------------------


def _build_tfidf(texts: List[str], max_features: int = 500) -> Tuple[np.ndarray, List[str]]:
    """Build a TF-IDF matrix from texts. Returns (matrix, vocab)."""
    doc_tokens = [_tokenize(t) for t in texts]
    doc_freq: Counter = Counter()
    for tokens in doc_tokens:
        for tok in set(tokens):
            doc_freq[tok] += 1

    # Keep top features by document frequency
    min_df = max(2, len(texts) // 20) if len(texts) > 10 else 1
    vocab_items = [
        (tok, df) for tok, df in doc_freq.items() if df >= min_df
    ]
    vocab_items.sort(key=lambda x: -x[1])
    vocab_items = vocab_items[:max_features]
    vocab = [tok for tok, _ in vocab_items]
    tok2idx = {tok: i for i, tok in enumerate(vocab)}

    n_docs = len(texts)
    n_feat = len(vocab)
    if n_feat == 0:
        return np.zeros((n_docs, 1)), ["<empty>"]

    tfidf = np.zeros((n_docs, n_feat))
    for i, tokens in enumerate(doc_tokens):
        tf = Counter(tokens)
        for tok, count in tf.items():
            if tok in tok2idx:
                idx = tok2idx[tok]
                idf = math.log((n_docs + 1) / (doc_freq[tok] + 1)) + 1
                tfidf[i, idx] = count * idf

    # L2 normalize rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms

    return tfidf, vocab


# ---------------------------------------------------------------------------
# Simple NMF (multiplicative updates)
# ---------------------------------------------------------------------------


def _nmf(V: np.ndarray, n_components: int, max_iter: int = 200,
         seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorisation: V ≈ W @ H."""
    rng = np.random.RandomState(seed)
    n, m = V.shape
    k = min(n_components, n, m)
    W = rng.rand(n, k).astype(np.float64) + 1e-6
    H = rng.rand(k, m).astype(np.float64) + 1e-6
    eps = 1e-10

    for _ in range(max_iter):
        # Update H
        num_h = W.T @ V
        den_h = W.T @ W @ H + eps
        H *= num_h / den_h
        # Update W
        num_w = V @ H.T
        den_w = W @ H @ H.T + eps
        W *= num_w / den_w

    return W, H


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Topic:
    """A single extracted topic."""
    id: int
    top_words: List[str]
    weight: float  # proportion of corpus assigned to this topic
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = " / ".join(self.top_words[:3])


@dataclass
class TopicModel:
    """Result of topic extraction."""
    topics: List[Topic]
    doc_topic_matrix: np.ndarray  # (n_docs, n_topics)
    topic_word_matrix: np.ndarray  # (n_topics, n_words)
    vocab: List[str]
    n_docs: int

    def dominant_topic(self, doc_idx: int) -> int:
        return int(np.argmax(self.doc_topic_matrix[doc_idx]))

    def topic_distribution(self, doc_idx: int) -> np.ndarray:
        row = self.doc_topic_matrix[doc_idx]
        s = row.sum()
        return row / s if s > 0 else row


@dataclass
class CoverageResult:
    """Topic coverage analysis."""
    coverage_ratio: float  # fraction of target topics covered
    covered_topics: List[str]
    uncovered_topics: List[str]
    per_topic_count: Dict[str, int]
    total_texts: int


@dataclass
class GapAnalysis:
    """Topic gap analysis result."""
    underrepresented_topics: List[str]
    overrepresented_topics: List[str]
    ideal_distribution: Dict[str, float]
    actual_distribution: Dict[str, float]
    suggestions: List[str]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def extract_topics(
    texts: List[str],
    n_topics: int = 10,
    top_words: int = 8,
    max_features: int = 500,
) -> TopicModel:
    """Extract topics from a collection of texts using NMF.

    Parameters
    ----------
    texts : list of str
    n_topics : int
        Number of topics to extract.
    top_words : int
        Number of top words per topic.
    max_features : int
        Maximum vocabulary size for TF-IDF.
    """
    if not texts:
        return TopicModel([], np.zeros((0, 0)), np.zeros((0, 0)), [], 0)

    n_topics = min(n_topics, len(texts))
    tfidf, vocab = _build_tfidf(texts, max_features=max_features)
    W, H = _nmf(tfidf, n_topics)

    topics: List[Topic] = []
    col_sums = W.sum(axis=0)
    total_weight = col_sums.sum() if col_sums.sum() > 0 else 1.0

    for t in range(W.shape[1]):
        word_indices = np.argsort(H[t])[::-1][:top_words]
        tw = [vocab[i] for i in word_indices if i < len(vocab)]
        weight = float(col_sums[t] / total_weight)
        topics.append(Topic(id=t, top_words=tw, weight=weight))

    return TopicModel(
        topics=topics,
        doc_topic_matrix=W,
        topic_word_matrix=H,
        vocab=vocab,
        n_docs=len(texts),
    )


def topic_coverage(
    texts: List[str],
    target_topics: List[str],
    threshold: float = 0.1,
) -> CoverageResult:
    """Measure how well *texts* cover a set of *target_topics*.

    A target topic is considered "covered" if any text contains at least
    *threshold* fraction of the topic's keywords.
    """
    per_topic: Dict[str, int] = {t: 0 for t in target_topics}
    covered: Set[str] = set()

    for text in texts:
        text_tokens = set(_tokenize(text))
        for topic in target_topics:
            topic_words = set(_tokenize(topic))
            if not topic_words:
                covered.add(topic)
                per_topic[topic] += 1
                continue
            overlap = len(text_tokens & topic_words) / len(topic_words)
            if overlap >= threshold:
                covered.add(topic)
                per_topic[topic] += 1

    uncovered = [t for t in target_topics if t not in covered]
    ratio = len(covered) / len(target_topics) if target_topics else 1.0

    return CoverageResult(
        coverage_ratio=ratio,
        covered_topics=sorted(covered),
        uncovered_topics=uncovered,
        per_topic_count=per_topic,
        total_texts=len(texts),
    )


def diversify_by_topic(
    texts: List[str],
    k: int,
    topics: Optional[TopicModel] = None,
    n_topics: int = 10,
) -> List[str]:
    """Select *k* texts that maximise topic coverage.

    Uses a greedy algorithm: at each step, pick the text whose dominant
    topic is least represented in the current selection.
    """
    if not texts:
        return []
    k = min(k, len(texts))

    if topics is None:
        topics = extract_topics(texts, n_topics=n_topics)

    n_topics_actual = topics.doc_topic_matrix.shape[1]
    if n_topics_actual == 0:
        return texts[:k]

    # Assign each text to its dominant topic
    assignments = [
        int(np.argmax(topics.doc_topic_matrix[i]))
        for i in range(len(texts))
    ]

    selected: List[int] = []
    topic_counts: Counter = Counter()

    for _ in range(k):
        best_idx = -1
        best_score = -1.0
        for i in range(len(texts)):
            if i in selected:
                continue
            t = assignments[i]
            # Prefer topics with fewer selections
            score = 1.0 / (topic_counts[t] + 1)
            # Tie-break by topic weight (prefer rarer topics)
            score += topics.doc_topic_matrix[i, t] * 0.01
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        topic_counts[assignments[best_idx]] += 1

    return [texts[i] for i in selected]


def topic_gap_analysis(
    texts: List[str],
    n_topics: int = 10,
) -> GapAnalysis:
    """Analyse which topics are over/under-represented.

    Returns suggestions for what kinds of content to add.
    """
    if not texts:
        return GapAnalysis([], [], {}, {}, ["No texts provided."])

    model = extract_topics(texts, n_topics=n_topics)
    n_t = len(model.topics)
    if n_t == 0:
        return GapAnalysis([], [], {}, {}, ["Could not extract topics."])

    ideal = 1.0 / n_t
    ideal_dist = {t.label: ideal for t in model.topics}

    # Actual distribution: fraction of docs whose dominant topic is t
    assignments = [
        int(np.argmax(model.doc_topic_matrix[i]))
        for i in range(len(texts))
    ]
    counts = Counter(assignments)
    total = len(texts)
    actual_dist = {
        model.topics[t].label: counts.get(t, 0) / total
        for t in range(n_t)
    }

    under = [
        model.topics[t].label
        for t in range(n_t)
        if counts.get(t, 0) / total < ideal * 0.5
    ]
    over = [
        model.topics[t].label
        for t in range(n_t)
        if counts.get(t, 0) / total > ideal * 2.0
    ]

    suggestions = []
    for label in under:
        suggestions.append(f"Add more content about: {label}")
    for label in over:
        suggestions.append(f"Reduce redundancy in: {label}")

    if not under and not over:
        suggestions.append("Topic distribution is relatively balanced.")

    return GapAnalysis(
        underrepresented_topics=under,
        overrepresented_topics=over,
        ideal_distribution=ideal_dist,
        actual_distribution=actual_dist,
        suggestions=suggestions,
    )
