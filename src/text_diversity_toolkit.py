"""
Text-specific diversity toolkit.

Implements n-gram diversity, self-BLEU, semantic diversity (TF-IDF),
topic diversity (NMF), style diversity, cross-text novelty,
homogenization detection, and diversity-quality tradeoff curves.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, NamedTuple
import re
from collections import Counter


# ======================================================================
# Text preprocessing utilities
# ======================================================================

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def get_ngram_set(tokens: List[str], n: int) -> set:
    """Extract set of unique n-grams."""
    return set(get_ngrams(tokens, n))


# ======================================================================
# TF-IDF from scratch
# ======================================================================

def compute_tfidf(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute TF-IDF matrix.

    Args:
        texts: List of text strings.

    Returns:
        (tfidf_matrix, vocabulary) where tfidf_matrix is (n_texts, vocab_size).
    """
    docs = [tokenize(t) for t in texts]
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    vocab_list = sorted(vocab.keys(), key=lambda x: vocab[x])
    n = len(texts)
    v = len(vocab)

    if v == 0:
        return np.zeros((n, 1)), ['']

    tf = np.zeros((n, v), dtype=np.float64)
    for i, doc in enumerate(docs):
        for token in doc:
            tf[i, vocab[token]] += 1.0
        if len(doc) > 0:
            tf[i] /= len(doc)

    df = np.sum(tf > 0, axis=0).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    tfidf = tf * idf
    return tfidf, vocab_list


# ======================================================================
# NMF from scratch
# ======================================================================

def nmf(V: np.ndarray, n_components: int = 5, max_iter: int = 200,
        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Non-negative Matrix Factorization via multiplicative updates.

    V ≈ W H where V is (n, m), W is (n, k), H is (k, m).

    Args:
        V: (n, m) non-negative input matrix.
        n_components: Number of topics/components.
        max_iter: Maximum iterations.
        random_state: Random seed.

    Returns:
        (W, H)
    """
    rng = np.random.RandomState(random_state)
    n, m = V.shape
    k = n_components

    # Initialize with small random values
    W = rng.rand(n, k).astype(np.float64) + 1e-6
    H = rng.rand(k, m).astype(np.float64) + 1e-6

    for _ in range(max_iter):
        # Update H
        numerator_H = W.T @ V
        denominator_H = W.T @ W @ H + 1e-12
        H *= numerator_H / denominator_H

        # Update W
        numerator_W = V @ H.T
        denominator_W = W @ H @ H.T + 1e-12
        W *= numerator_W / denominator_W

    return W, H


# ======================================================================
# BLEU Score from scratch
# ======================================================================

def compute_bleu(candidate: List[str], references: List[List[str]],
                 max_n: int = 4) -> float:
    """Compute BLEU score of candidate against references.

    Args:
        candidate: Tokenized candidate.
        references: List of tokenized references.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score.
    """
    if len(candidate) == 0:
        return 0.0

    # Brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_len = min(ref_lens, key=lambda l: (abs(l - len(candidate)), l))
    if len(candidate) < closest_len:
        bp = np.exp(1 - closest_len / max(len(candidate), 1))
    else:
        bp = 1.0

    # Modified precision for each n-gram order
    log_precisions = []
    for n in range(1, min(max_n, len(candidate)) + 1):
        cand_ngrams = Counter(get_ngrams(candidate, n))

        # Clipped counts: for each n-gram, take min(count, max_ref_count)
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter(get_ngrams(ref, n))
            for ng, count in ref_ngrams.items():
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), count)

        clipped_count = 0
        total_count = 0
        for ng, count in cand_ngrams.items():
            clipped_count += min(count, max_ref_counts.get(ng, 0))
            total_count += count

        if total_count == 0:
            return 0.0

        precision = clipped_count / total_count
        if precision <= 0:
            return 0.0
        log_precisions.append(np.log(precision))

    if not log_precisions:
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(log_precisions) / len(log_precisions)
    return float(bp * np.exp(log_avg))


# ======================================================================
# TextDiversityReport
# ======================================================================

class TextDiversityReport(NamedTuple):
    """Container for text diversity analysis results."""
    distinct_1: float
    distinct_2: float
    distinct_3: float
    distinct_4: float
    self_bleu: float
    semantic_diversity: float
    topic_entropy: float
    n_topics: int
    style_diversity: Dict
    cross_text_novelty: float
    homogenization_alert: bool
    n_texts: int


# ======================================================================
# TextDiversityToolkit
# ======================================================================

class TextDiversityToolkit:
    """Comprehensive text diversity analysis.

    All methods work with raw text strings and use numpy only.
    """

    def __init__(self):
        pass

    def analyze(self, texts: List[str],
                n_topics: int = 5,
                homogenization_threshold: float = 0.3) -> TextDiversityReport:
        """Run full diversity analysis on a set of texts.

        Args:
            texts: List of text strings.
            n_topics: Number of topics for NMF.
            homogenization_threshold: Alert if diversity below this.

        Returns:
            TextDiversityReport with all metrics.
        """
        if not texts:
            return TextDiversityReport(
                distinct_1=0, distinct_2=0, distinct_3=0, distinct_4=0,
                self_bleu=0, semantic_diversity=0, topic_entropy=0,
                n_topics=0, style_diversity={}, cross_text_novelty=0,
                homogenization_alert=False, n_texts=0
            )

        d1 = self.distinct_n(texts, 1)
        d2 = self.distinct_n(texts, 2)
        d3 = self.distinct_n(texts, 3)
        d4 = self.distinct_n(texts, 4)
        sb = self.self_bleu(texts)
        sd = self.semantic_diversity(texts)
        te, nt = self.topic_diversity(texts, n_topics)
        style = self.style_diversity(texts)
        ctn = self.cross_text_novelty(texts)
        alert = self.homogenization_detection(
            texts, threshold=homogenization_threshold
        )

        return TextDiversityReport(
            distinct_1=d1, distinct_2=d2, distinct_3=d3, distinct_4=d4,
            self_bleu=sb, semantic_diversity=sd, topic_entropy=te,
            n_topics=nt, style_diversity=style, cross_text_novelty=ctn,
            homogenization_alert=alert, n_texts=len(texts)
        )

    # ------------------------------------------------------------------
    # N-gram diversity: distinct-n
    # ------------------------------------------------------------------

    @staticmethod
    def distinct_n(texts: List[str], n: int) -> float:
        """Compute distinct-n: ratio of unique n-grams to total n-grams.

        Distinct-n = |unique n-grams| / |total n-grams|

        Higher values indicate more lexical diversity.

        Args:
            texts: List of texts.
            n: N-gram order.

        Returns:
            Distinct-n ratio [0, 1].
        """
        total_ngrams = []
        for text in texts:
            tokens = tokenize(text)
            total_ngrams.extend(get_ngrams(tokens, n))

        if len(total_ngrams) == 0:
            return 0.0

        unique = set(total_ngrams)
        return float(len(unique)) / len(total_ngrams)

    # ------------------------------------------------------------------
    # Self-BLEU
    # ------------------------------------------------------------------

    @staticmethod
    def self_bleu(texts: List[str], max_n: int = 4) -> float:
        """Self-BLEU: average BLEU of each text against all others.

        Lower self-BLEU indicates higher diversity (texts are more
        different from each other).

        Args:
            texts: List of texts.
            max_n: Maximum n-gram order for BLEU.

        Returns:
            Average self-BLEU score [0, 1]. Lower = more diverse.
        """
        n = len(texts)
        if n <= 1:
            return 0.0

        tokenized = [tokenize(t) for t in texts]
        bleu_scores = []

        for i in range(n):
            refs = [tokenized[j] for j in range(n) if j != i]
            score = compute_bleu(tokenized[i], refs, max_n=max_n)
            bleu_scores.append(score)

        return float(np.mean(bleu_scores))

    # ------------------------------------------------------------------
    # Semantic diversity (TF-IDF based)
    # ------------------------------------------------------------------

    @staticmethod
    def semantic_diversity(texts: List[str]) -> float:
        """Semantic diversity: mean pairwise cosine distance of TF-IDF vectors.

        Higher values indicate more semantic diversity.

        Args:
            texts: List of texts.

        Returns:
            Mean pairwise cosine distance [0, 2].
        """
        if len(texts) <= 1:
            return 0.0

        tfidf, _ = compute_tfidf(texts)
        n = tfidf.shape[0]

        # Cosine similarity matrix
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tfidf_norm = tfidf / norms
        sim = tfidf_norm @ tfidf_norm.T
        np.clip(sim, -1.0, 1.0, out=sim)

        # Mean cosine distance (upper triangle)
        dist = 1.0 - sim
        upper_idx = np.triu_indices(n, k=1)
        return float(np.mean(dist[upper_idx]))

    # ------------------------------------------------------------------
    # Topic diversity (NMF-based)
    # ------------------------------------------------------------------

    @staticmethod
    def topic_diversity(texts: List[str],
                        n_topics: int = 5) -> Tuple[float, int]:
        """Topic diversity: entropy of topic distribution from NMF.

        1. Compute TF-IDF matrix.
        2. Run NMF to get topic proportions.
        3. Compute entropy of aggregate topic distribution.

        Args:
            texts: List of texts.
            n_topics: Number of topics.

        Returns:
            (topic_entropy, effective_n_topics)
        """
        if len(texts) <= 1:
            return 0.0, 0

        tfidf, _ = compute_tfidf(texts)
        n_topics = min(n_topics, tfidf.shape[0], tfidf.shape[1])

        if n_topics <= 0:
            return 0.0, 0

        # Ensure non-negative
        tfidf = np.maximum(tfidf, 0.0)
        if tfidf.sum() < 1e-12:
            return 0.0, 0

        W, H = nmf(tfidf, n_components=n_topics, max_iter=100)

        # Topic distribution per document: normalize W rows
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        topic_dist = W / row_sums

        # Aggregate topic distribution
        agg_dist = topic_dist.mean(axis=0)
        agg_dist = agg_dist / max(agg_dist.sum(), 1e-12)
        agg_dist = agg_dist[agg_dist > 1e-12]

        entropy = -np.sum(agg_dist * np.log(agg_dist))
        effective_topics = int(np.round(np.exp(entropy)))

        return float(entropy), effective_topics

    # ------------------------------------------------------------------
    # Style diversity
    # ------------------------------------------------------------------

    @staticmethod
    def style_diversity(texts: List[str]) -> Dict:
        """Style diversity: variance in stylistic features.

        Measures:
        - Sentence length variance
        - Vocabulary complexity variance
        - Punctuation density variance
        - Type-token ratio variance

        Args:
            texts: List of texts.

        Returns:
            Dict of style diversity metrics.
        """
        if len(texts) <= 1:
            return {
                'sentence_length_var': 0.0,
                'vocab_complexity_var': 0.0,
                'punctuation_density_var': 0.0,
                'type_token_ratio_var': 0.0,
                'overall_style_diversity': 0.0
            }

        sent_lengths = []
        vocab_complexities = []
        punct_densities = []
        ttrs = []

        for text in texts:
            tokens = tokenize(text)
            n_tokens = len(tokens)

            # Sentence length (approximate by counting sentence-ending punctuation)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            avg_sent_len = n_tokens / max(len(sentences), 1)
            sent_lengths.append(avg_sent_len)

            # Vocabulary complexity: average word length
            if n_tokens > 0:
                avg_word_len = np.mean([len(t) for t in tokens])
            else:
                avg_word_len = 0.0
            vocab_complexities.append(avg_word_len)

            # Punctuation density
            n_punct = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\')
            punct_densities.append(n_punct / max(len(text), 1))

            # Type-token ratio
            if n_tokens > 0:
                ttr = len(set(tokens)) / n_tokens
            else:
                ttr = 0.0
            ttrs.append(ttr)

        result = {
            'sentence_length_var': float(np.var(sent_lengths)),
            'vocab_complexity_var': float(np.var(vocab_complexities)),
            'punctuation_density_var': float(np.var(punct_densities)),
            'type_token_ratio_var': float(np.var(ttrs)),
        }

        # Overall style diversity: normalized sum
        values = list(result.values())
        result['overall_style_diversity'] = float(np.mean(values))

        return result

    # ------------------------------------------------------------------
    # Cross-text novelty
    # ------------------------------------------------------------------

    @staticmethod
    def cross_text_novelty(texts: List[str], n: int = 2) -> float:
        """Cross-text novelty: fraction of n-grams in each text
        that don't appear in any other text.

        Higher values mean texts are more novel relative to each other.

        Args:
            texts: List of texts.
            n: N-gram order.

        Returns:
            Average novelty fraction [0, 1].
        """
        if len(texts) <= 1:
            return 1.0

        tokenized = [tokenize(t) for t in texts]
        ngram_sets = [get_ngram_set(tokens, n) for tokens in tokenized]

        novelties = []
        for i in range(len(texts)):
            if len(ngram_sets[i]) == 0:
                continue
            # N-grams in text i not in any other text
            other_ngrams = set()
            for j in range(len(texts)):
                if j != i:
                    other_ngrams |= ngram_sets[j]

            novel = ngram_sets[i] - other_ngrams
            novelties.append(len(novel) / len(ngram_sets[i]))

        if not novelties:
            return 0.0
        return float(np.mean(novelties))

    # ------------------------------------------------------------------
    # Homogenization detection
    # ------------------------------------------------------------------

    def homogenization_detection(self, texts: List[str],
                                  threshold: float = 0.3) -> bool:
        """Detect if texts are becoming homogenized (low diversity).

        Alert if multiple diversity metrics fall below threshold.

        Args:
            texts: List of texts.
            threshold: Alert threshold.

        Returns:
            True if homogenization detected.
        """
        if len(texts) <= 1:
            return False

        d2 = self.distinct_n(texts, 2)
        novelty = self.cross_text_novelty(texts)
        sem_div = self.semantic_diversity(texts)

        # Alert if multiple metrics are low
        alerts = 0
        if d2 < threshold:
            alerts += 1
        if novelty < threshold:
            alerts += 1
        if sem_div < threshold:
            alerts += 1

        return alerts >= 2

    # ------------------------------------------------------------------
    # Diversity-quality tradeoff curve
    # ------------------------------------------------------------------

    @staticmethod
    def diversity_quality_tradeoff(texts: List[str],
                                    quality_scores: np.ndarray,
                                    n_points: int = 20
                                    ) -> Dict:
        """Compute the diversity-quality tradeoff (Pareto frontier).

        For different subset sizes, find the subset that maximizes
        a weighted combination of quality and diversity.

        Args:
            texts: List of texts.
            quality_scores: (n,) quality score per text.
            n_points: Number of points on the tradeoff curve.

        Returns:
            Dict with pareto frontier data.
        """
        n = len(texts)
        quality_scores = np.asarray(quality_scores, dtype=np.float64)

        # Compute pairwise diversity (distinct-2 based)
        tokenized = [tokenize(t) for t in texts]

        # For each subset size, estimate quality and diversity
        sizes = np.unique(np.linspace(2, min(n, 50), n_points).astype(int))
        curve_quality = []
        curve_diversity = []
        curve_sizes = []

        for k in sizes:
            k = int(k)
            if k > n:
                break

            # Greedy: pick top-k by quality
            top_k = np.argsort(-quality_scores)[:k]
            avg_quality = float(np.mean(quality_scores[top_k]))

            # Compute distinct-2 of the subset
            subset_texts = [texts[i] for i in top_k]
            all_bigrams = []
            for t in subset_texts:
                tokens = tokenize(t)
                all_bigrams.extend(get_ngrams(tokens, 2))
            if all_bigrams:
                div = len(set(all_bigrams)) / len(all_bigrams)
            else:
                div = 0.0

            curve_quality.append(avg_quality)
            curve_diversity.append(div)
            curve_sizes.append(k)

        # Compute Pareto frontier
        pareto_quality = []
        pareto_diversity = []
        if curve_quality:
            points = list(zip(curve_quality, curve_diversity))
            # Sort by quality descending
            points.sort(key=lambda p: -p[0])
            max_div = -1.0
            for q, d in points:
                if d > max_div:
                    pareto_quality.append(q)
                    pareto_diversity.append(d)
                    max_div = d

        return {
            'sizes': curve_sizes,
            'quality': curve_quality,
            'diversity': curve_diversity,
            'pareto_quality': pareto_quality,
            'pareto_diversity': pareto_diversity
        }


class TextSimilarityMatrix:
    """Compute text similarity matrices using various methods."""

    @staticmethod
    def ngram_overlap(texts: List[str], n: int = 2) -> np.ndarray:
        """Jaccard similarity based on n-gram overlap.

        Args:
            texts: List of texts.
            n: N-gram order.

        Returns:
            (n_texts, n_texts) similarity matrix.
        """
        ngram_sets = [get_ngram_set(tokenize(t), n) for t in texts]
        m = len(texts)
        sim = np.zeros((m, m), dtype=np.float64)

        for i in range(m):
            for j in range(i, m):
                if len(ngram_sets[i]) == 0 and len(ngram_sets[j]) == 0:
                    s = 1.0
                elif len(ngram_sets[i]) == 0 or len(ngram_sets[j]) == 0:
                    s = 0.0
                else:
                    intersection = len(ngram_sets[i] & ngram_sets[j])
                    union = len(ngram_sets[i] | ngram_sets[j])
                    s = intersection / union
                sim[i, j] = s
                sim[j, i] = s

        return sim

    @staticmethod
    def tfidf_cosine(texts: List[str]) -> np.ndarray:
        """Cosine similarity based on TF-IDF vectors.

        Args:
            texts: List of texts.

        Returns:
            (n_texts, n_texts) similarity matrix.
        """
        tfidf, _ = compute_tfidf(texts)
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tfidf_norm = tfidf / norms
        sim = tfidf_norm @ tfidf_norm.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return sim


def demo_text_diversity():
    """Demonstrate text diversity toolkit."""
    texts = [
        "The quick brown fox jumps over the lazy dog in the sunny meadow.",
        "A fast red fox leaps across the sleepy hound under bright sunshine.",
        "Machine learning algorithms process data to find patterns and insights.",
        "Deep neural networks transform input representations through multiple layers.",
        "The ocean waves crash against the rocky shoreline at dawn.",
        "Quantum computing leverages superposition and entanglement for computation.",
        "The cat sat on the mat and watched the birds fly by.",
        "Blockchain technology enables decentralized and transparent transactions.",
    ]

    toolkit = TextDiversityToolkit()
    report = toolkit.analyze(texts)

    print("=== Text Diversity Report ===")
    print(f"Number of texts: {report.n_texts}")
    print(f"Distinct-1: {report.distinct_1:.4f}")
    print(f"Distinct-2: {report.distinct_2:.4f}")
    print(f"Distinct-3: {report.distinct_3:.4f}")
    print(f"Distinct-4: {report.distinct_4:.4f}")
    print(f"Self-BLEU (lower=more diverse): {report.self_bleu:.4f}")
    print(f"Semantic diversity: {report.semantic_diversity:.4f}")
    print(f"Topic entropy: {report.topic_entropy:.4f}")
    print(f"Effective topics: {report.n_topics}")
    print(f"Cross-text novelty: {report.cross_text_novelty:.4f}")
    print(f"Homogenization alert: {report.homogenization_alert}")
    print(f"Style diversity: {report.style_diversity}")

    # Quality-diversity tradeoff
    quality = np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    tradeoff = TextDiversityToolkit.diversity_quality_tradeoff(texts, quality)
    print(f"\nTradeoff curve points: {len(tradeoff['sizes'])}")
    print(f"Pareto frontier points: {len(tradeoff['pareto_quality'])}")


if __name__ == '__main__':
    demo_text_diversity()
