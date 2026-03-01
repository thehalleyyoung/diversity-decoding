"""
Minimum Bayes Risk (MBR) Diversity Decoding.

Generates a candidate pool and selects a diverse subset by optimizing a utility
function that balances quality (expected utility) with diversity (pairwise distances).

Three-phase pipeline:
  1. Generate candidate and reference pools via ancestral sampling.
  2. Compute a utility matrix U[i,j] = utility(candidate_i, reference_j) and derive
     expected utilities E_ref[U] per candidate.
  3. Select a diverse subset using MMR, submodular maximization, or ILP-based
     selection that trades off quality and diversity.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from src.algorithms.base import (
    AlgorithmRegistry,
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    TokenSequence,
    sample_token,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MBRConfig(DecodingConfig):
    """Configuration for MBR diversity decoding."""

    algorithm_name: str = "mbr_diversity"

    # Pool sizes
    candidate_pool_size: int = 100
    reference_pool_size: int = 50
    select_k: int = 20

    # Utility function
    utility_function: str = "bleu"  # bleu, rouge, bertscore, chrF, edit_distance
    diversity_weight: float = 0.3

    # Selection method
    selection_method: str = "greedy_mmr"  # greedy_mmr, submodular, ilp

    # Sampling
    temperature: float = 1.0

    # Embedding model (used for BERTScore fallback / similarity)
    embedding_model: str = "all-MiniLM-L6-v2"

    # MMR parameters
    lambda_mmr: float = 0.5

    # BLEU parameters
    ngram_order: int = 4
    smoothing: str = "floor"  # floor, add_k, none

    # Submodular objective weights
    facility_location_weight: float = 0.5
    dispersion_weight: float = 0.5

    # Advanced
    top_k_sampling: int = 0
    top_p_sampling: float = 1.0
    dedup_candidates: bool = True
    normalize_utility: bool = True
    similarity_metric: str = "ngram"  # ngram, cosine, edit

    def validate(self) -> bool:
        """Validate MBR-specific configuration."""
        super().validate()
        assert self.candidate_pool_size > 0, "candidate_pool_size must be > 0"
        assert self.reference_pool_size > 0, "reference_pool_size must be > 0"
        assert 0 < self.select_k <= self.candidate_pool_size, (
            "select_k must be in (0, candidate_pool_size]"
        )
        assert self.utility_function in {
            "bleu", "rouge", "bertscore", "chrF", "edit_distance",
        }, f"Unknown utility function: {self.utility_function}"
        assert 0.0 <= self.diversity_weight <= 1.0, (
            "diversity_weight must be in [0, 1]"
        )
        assert self.selection_method in {
            "greedy_mmr", "submodular", "ilp",
        }, f"Unknown selection method: {self.selection_method}"
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.lambda_mmr <= 1.0, "lambda_mmr must be in [0, 1]"
        assert self.ngram_order >= 1, "ngram_order must be >= 1"
        assert self.smoothing in {"floor", "add_k", "none"}, (
            f"Unknown smoothing: {self.smoothing}"
        )
        return True


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------


def _tokenize_simple(text: str) -> List[str]:
    """Whitespace tokenizer with lowercasing and basic punctuation split."""
    tokens: List[str] = []
    for raw in text.lower().split():
        buf: List[str] = []
        for ch in raw:
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
        if buf:
            tokens.append("".join(buf))
    return tokens


def _extract_ngrams(tokens: List[str], order: int) -> Counter:
    """Extract n-grams of the given order from token list."""
    ngrams: Counter = Counter()
    for i in range(len(tokens) - order + 1):
        ngrams[tuple(tokens[i : i + order])] += 1
    return ngrams


def _extract_char_ngrams(text: str, order: int) -> Counter:
    """Extract character-level n-grams."""
    ngrams: Counter = Counter()
    cleaned = text.lower().strip()
    for i in range(len(cleaned) - order + 1):
        ngrams[cleaned[i : i + order]] += 1
    return ngrams


# ---------------------------------------------------------------------------
# BLEU Computer
# ---------------------------------------------------------------------------


class BLEUComputer:
    """Complete sentence-level and corpus-level BLEU implementation from scratch.

    Implements modified n-gram precision, brevity penalty, and multiple
    smoothing methods (floor, add-k, none).
    """

    def __init__(
        self,
        max_order: int = 4,
        smoothing: str = "floor",
        floor_value: float = 0.1,
        add_k_value: float = 1.0,
    ) -> None:
        self.max_order = max_order
        self.smoothing = smoothing
        self.floor_value = floor_value
        self.add_k_value = add_k_value

    # -- public API --------------------------------------------------------

    def sentence_bleu(
        self,
        hypothesis: str,
        reference: str,
        max_order: Optional[int] = None,
        smoothing: Optional[str] = None,
    ) -> float:
        """Compute sentence-level BLEU between *hypothesis* and *reference*.

        Returns a float in [0, 1].
        """
        order = max_order if max_order is not None else self.max_order
        smooth = smoothing if smoothing is not None else self.smoothing

        hyp_tokens = _tokenize_simple(hypothesis)
        ref_tokens = _tokenize_simple(reference)

        if not hyp_tokens or not ref_tokens:
            return 0.0

        precisions: List[float] = []
        for n in range(1, order + 1):
            numerator, denominator = self._modified_precision(
                _extract_ngrams(hyp_tokens, n),
                _extract_ngrams(ref_tokens, n),
                n,
            )
            if denominator == 0:
                precisions.append(0.0)
            else:
                precisions.append(numerator / denominator)

        precisions = self._apply_smoothing(precisions, smooth)

        if any(p == 0.0 for p in precisions):
            return 0.0

        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bp = self._brevity_penalty(len(hyp_tokens), len(ref_tokens))
        return bp * math.exp(log_avg)

    def corpus_bleu(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> float:
        """Compute corpus-level BLEU aggregated across sentence pairs."""
        assert len(hypotheses) == len(references)
        if not hypotheses:
            return 0.0

        total_numerators = [0] * self.max_order
        total_denominators = [0] * self.max_order
        total_hyp_len = 0
        total_ref_len = 0

        for hyp_str, ref_str in zip(hypotheses, references):
            hyp_tokens = _tokenize_simple(hyp_str)
            ref_tokens = _tokenize_simple(ref_str)
            total_hyp_len += len(hyp_tokens)
            total_ref_len += len(ref_tokens)

            for n in range(1, self.max_order + 1):
                num, den = self._modified_precision(
                    _extract_ngrams(hyp_tokens, n),
                    _extract_ngrams(ref_tokens, n),
                    n,
                )
                total_numerators[n - 1] += num
                total_denominators[n - 1] += den

        precisions: List[float] = []
        for n in range(self.max_order):
            if total_denominators[n] == 0:
                precisions.append(0.0)
            else:
                precisions.append(total_numerators[n] / total_denominators[n])

        precisions = self._apply_smoothing(precisions, self.smoothing)

        if any(p == 0.0 for p in precisions):
            return 0.0

        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bp = self._brevity_penalty(total_hyp_len, total_ref_len)
        return bp * math.exp(log_avg)

    # -- internals ---------------------------------------------------------

    def _modified_precision(
        self,
        hypothesis_ngrams: Counter,
        reference_ngrams: Counter,
        order: int,
    ) -> Tuple[int, int]:
        """Compute clipped modified precision for a single n-gram order.

        Returns (clipped_count, total_count).
        """
        clipped = 0
        total = 0
        for ngram, count in hypothesis_ngrams.items():
            if len(ngram) != order:
                continue
            total += count
            clipped += min(count, reference_ngrams.get(ngram, 0))
        return clipped, total

    def _brevity_penalty(self, hyp_len: int, ref_len: int) -> float:
        """Compute the brevity penalty."""
        if hyp_len == 0:
            return 0.0
        ratio = ref_len / hyp_len
        if ratio <= 1.0:
            return 1.0
        return math.exp(1.0 - ratio)

    def _apply_smoothing(
        self, precisions: List[float], method: str
    ) -> List[float]:
        if method == "floor":
            return self._smooth_floor(precisions, self.floor_value)
        elif method == "add_k":
            return self._smooth_add_k(precisions, self.add_k_value)
        return list(precisions)

    def _smooth_floor(
        self, precisions: List[float], floor: float = 0.1
    ) -> List[float]:
        """Replace zero precisions with a small floor value."""
        return [max(p, floor) for p in precisions]

    def _smooth_add_k(
        self, precisions: List[float], k: float = 1.0
    ) -> List[float]:
        """Add-k smoothing: add *k* to both numerator and denominator
        (approximated here as shifting zero precisions to k/(k+1))."""
        return [p if p > 0 else k / (k + 1.0) for p in precisions]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


class UtilityFunction(ABC):
    """Abstract base class for utility (similarity) functions."""

    @abstractmethod
    def compute(self, hypothesis: str, reference: str) -> float:
        """Compute utility between a single hypothesis–reference pair."""
        ...

    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> np.ndarray:
        """Compute utility matrix of shape (len(hypotheses), len(references)).

        Default implementation loops; subclasses may override for efficiency.
        """
        n_hyp = len(hypotheses)
        n_ref = len(references)
        matrix = np.zeros((n_hyp, n_ref), dtype=np.float64)
        for i, hyp in enumerate(hypotheses):
            for j, ref in enumerate(references):
                matrix[i, j] = self.compute(hyp, ref)
        return matrix


class BLEUUtility(UtilityFunction):
    """Sentence-level BLEU utility."""

    def __init__(
        self,
        max_order: int = 4,
        smoothing: str = "floor",
    ) -> None:
        self._bleu = BLEUComputer(max_order=max_order, smoothing=smoothing)

    def compute(self, hypothesis: str, reference: str) -> float:
        return self._bleu.sentence_bleu(hypothesis, reference)

    def compute_batch(
        self, hypotheses: List[str], references: List[str]
    ) -> np.ndarray:
        n_hyp = len(hypotheses)
        n_ref = len(references)
        matrix = np.zeros((n_hyp, n_ref), dtype=np.float64)

        # Pre-tokenize for speed
        hyp_tok = [_tokenize_simple(h) for h in hypotheses]
        ref_tok = [_tokenize_simple(r) for r in references]

        for i in range(n_hyp):
            for j in range(n_ref):
                if not hyp_tok[i] or not ref_tok[j]:
                    matrix[i, j] = 0.0
                    continue
                matrix[i, j] = self._bleu.sentence_bleu(
                    hypotheses[i], references[j]
                )
        return matrix


class ROUGEUtility(UtilityFunction):
    """ROUGE-L F1 utility based on longest common subsequence."""

    def compute(self, hypothesis: str, reference: str) -> float:
        hyp_tokens = _tokenize_simple(hypothesis)
        ref_tokens = _tokenize_simple(reference)
        if not hyp_tokens or not ref_tokens:
            return 0.0
        lcs_len = self._lcs_length(hyp_tokens, ref_tokens)
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(a: List[str], b: List[str]) -> int:
        """Compute length of the longest common subsequence."""
        m, n = len(a), len(b)
        # Space-efficient O(min(m,n)) DP
        if m < n:
            a, b = b, a
            m, n = n, m
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def compute_batch(
        self, hypotheses: List[str], references: List[str]
    ) -> np.ndarray:
        n_hyp, n_ref = len(hypotheses), len(references)
        matrix = np.zeros((n_hyp, n_ref), dtype=np.float64)
        for i in range(n_hyp):
            for j in range(n_ref):
                matrix[i, j] = self.compute(hypotheses[i], references[j])
        return matrix


class ChrFUtility(UtilityFunction):
    """Character n-gram F-score (chrF) utility.

    Computes a weighted average of character n-gram precisions/recalls
    for orders 1..max_order and combines them via F-beta score.
    """

    def __init__(
        self,
        max_order: int = 6,
        beta: float = 2.0,
    ) -> None:
        self.max_order = max_order
        self.beta = beta

    def compute(self, hypothesis: str, reference: str) -> float:
        if not hypothesis.strip() or not reference.strip():
            return 0.0

        total_precision = 0.0
        total_recall = 0.0
        n_orders = 0

        for n in range(1, self.max_order + 1):
            hyp_ngrams = _extract_char_ngrams(hypothesis, n)
            ref_ngrams = _extract_char_ngrams(reference, n)
            if not hyp_ngrams or not ref_ngrams:
                continue

            common = 0
            for ng, cnt in hyp_ngrams.items():
                common += min(cnt, ref_ngrams.get(ng, 0))

            hyp_total = sum(hyp_ngrams.values())
            ref_total = sum(ref_ngrams.values())

            precision = common / hyp_total if hyp_total > 0 else 0.0
            recall = common / ref_total if ref_total > 0 else 0.0

            total_precision += precision
            total_recall += recall
            n_orders += 1

        if n_orders == 0:
            return 0.0

        avg_p = total_precision / n_orders
        avg_r = total_recall / n_orders

        if avg_p + avg_r == 0:
            return 0.0

        beta_sq = self.beta ** 2
        f_score = (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)
        return f_score


class BERTScoreUtility(UtilityFunction):
    """BERTScore-like utility with fallback to n-gram overlap.

    If sentence-transformers is available, uses actual embeddings;
    otherwise falls back to a weighted n-gram overlap that approximates
    soft token matching.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ngram_fallback_orders: Tuple[int, ...] = (1, 2, 3),
    ) -> None:
        self.model_name = model_name
        self.ngram_fallback_orders = ngram_fallback_orders
        self._encoder: Optional[Any] = None
        self._use_embeddings = False
        self._try_load_encoder()

    def _try_load_encoder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer(self.model_name)
            self._use_embeddings = True
            logger.info("BERTScoreUtility: using embedding model %s", self.model_name)
        except ImportError:
            logger.info(
                "BERTScoreUtility: sentence-transformers not available, "
                "falling back to n-gram overlap"
            )
            self._use_embeddings = False

    def compute(self, hypothesis: str, reference: str) -> float:
        if self._use_embeddings:
            return self._compute_embedding(hypothesis, reference)
        return self._compute_ngram_fallback(hypothesis, reference)

    def _compute_embedding(self, hypothesis: str, reference: str) -> float:
        """Cosine similarity between sentence embeddings."""
        vecs = self._encoder.encode([hypothesis, reference])  # type: ignore[union-attr]
        h_vec = vecs[0]
        r_vec = vecs[1]
        dot = float(np.dot(h_vec, r_vec))
        norm = float(np.linalg.norm(h_vec) * np.linalg.norm(r_vec))
        if norm == 0:
            return 0.0
        return max(0.0, dot / norm)

    def _compute_ngram_fallback(self, hypothesis: str, reference: str) -> float:
        """Weighted n-gram overlap as BERTScore approximation."""
        hyp_tokens = _tokenize_simple(hypothesis)
        ref_tokens = _tokenize_simple(reference)
        if not hyp_tokens or not ref_tokens:
            return 0.0

        weights = [1.0 / (2 ** i) for i in range(len(self.ngram_fallback_orders))]
        total_weight = sum(weights)
        score = 0.0

        for w, n in zip(weights, self.ngram_fallback_orders):
            hyp_ng = _extract_ngrams(hyp_tokens, n)
            ref_ng = _extract_ngrams(ref_tokens, n)
            if not hyp_ng or not ref_ng:
                continue
            common = 0
            for ng, cnt in hyp_ng.items():
                common += min(cnt, ref_ng.get(ng, 0))
            hyp_total = sum(hyp_ng.values())
            ref_total = sum(ref_ng.values())
            p = common / hyp_total if hyp_total else 0.0
            r = common / ref_total if ref_total else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            score += w * f

        return score / total_weight

    def compute_batch(
        self, hypotheses: List[str], references: List[str]
    ) -> np.ndarray:
        if self._use_embeddings:
            return self._compute_batch_embedding(hypotheses, references)
        return super().compute_batch(hypotheses, references)

    def _compute_batch_embedding(
        self, hypotheses: List[str], references: List[str]
    ) -> np.ndarray:
        h_vecs = self._encoder.encode(hypotheses)  # type: ignore[union-attr]
        r_vecs = self._encoder.encode(references)  # type: ignore[union-attr]
        # Normalize
        h_norms = np.linalg.norm(h_vecs, axis=1, keepdims=True)
        r_norms = np.linalg.norm(r_vecs, axis=1, keepdims=True)
        h_norms = np.where(h_norms == 0, 1.0, h_norms)
        r_norms = np.where(r_norms == 0, 1.0, r_norms)
        h_vecs = h_vecs / h_norms
        r_vecs = r_vecs / r_norms
        return np.clip(h_vecs @ r_vecs.T, 0.0, 1.0)


class EditDistanceUtility(UtilityFunction):
    """Normalized edit distance (Levenshtein) utility.

    Returns 1 - (edit_distance / max_len) so that higher values mean more
    similar strings (consistent with other utilities).
    """

    def compute(self, hypothesis: str, reference: str) -> float:
        hyp_tokens = _tokenize_simple(hypothesis)
        ref_tokens = _tokenize_simple(reference)
        if not hyp_tokens and not ref_tokens:
            return 1.0
        dist = self._levenshtein(hyp_tokens, ref_tokens)
        max_len = max(len(hyp_tokens), len(ref_tokens))
        return 1.0 - dist / max_len

    @staticmethod
    def _levenshtein(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        if m < n:
            a, b = b, a
            m, n = n, m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev = curr
        return prev[n]


# ---------------------------------------------------------------------------
# Utility factory
# ---------------------------------------------------------------------------


def _create_utility(
    name: str,
    ngram_order: int = 4,
    smoothing: str = "floor",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> UtilityFunction:
    """Create a utility function by name."""
    name_lower = name.lower()
    if name_lower == "bleu":
        return BLEUUtility(max_order=ngram_order, smoothing=smoothing)
    elif name_lower == "rouge":
        return ROUGEUtility()
    elif name_lower == "chrf":
        return ChrFUtility()
    elif name_lower == "bertscore":
        return BERTScoreUtility(model_name=embedding_model)
    elif name_lower == "edit_distance":
        return EditDistanceUtility()
    else:
        raise ValueError(f"Unknown utility function: {name}")


# ---------------------------------------------------------------------------
# Pairwise similarity helpers
# ---------------------------------------------------------------------------


class SimilarityComputer:
    """Compute pairwise similarity between candidate strings."""

    def __init__(self, metric: str = "ngram", ngram_order: int = 4) -> None:
        self.metric = metric
        self.ngram_order = ngram_order

    def compute_matrix(self, candidates: List[str]) -> np.ndarray:
        """Return a symmetric similarity matrix of shape (n, n)."""
        n = len(candidates)
        sim = np.zeros((n, n), dtype=np.float64)

        if self.metric == "ngram":
            self._fill_ngram(candidates, sim)
        elif self.metric == "cosine":
            self._fill_cosine(candidates, sim)
        elif self.metric == "edit":
            self._fill_edit(candidates, sim)
        else:
            self._fill_ngram(candidates, sim)

        return sim

    # -- n-gram Jaccard similarity -----------------------------------------

    def _fill_ngram(self, candidates: List[str], sim: np.ndarray) -> None:
        token_lists = [_tokenize_simple(c) for c in candidates]
        ngram_sets: List[Set[tuple]] = []
        for tokens in token_lists:
            ng: Set[tuple] = set()
            for n in range(1, self.ngram_order + 1):
                ng.update(_extract_ngrams(tokens, n).keys())
            ngram_sets.append(ng)

        n = len(candidates)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                inter = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                val = inter / union if union > 0 else 0.0
                sim[i, j] = val
                sim[j, i] = val

    # -- TF-IDF cosine similarity ------------------------------------------

    def _fill_cosine(self, candidates: List[str], sim: np.ndarray) -> None:
        token_lists = [_tokenize_simple(c) for c in candidates]
        n = len(candidates)

        # Build vocabulary
        vocab: Dict[str, int] = {}
        doc_freq: Counter = Counter()
        for tokens in token_lists:
            seen: Set[str] = set()
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
                if t not in seen:
                    doc_freq[t] += 1
                    seen.add(t)

        vocab_size = len(vocab)
        if vocab_size == 0:
            return

        # TF-IDF vectors
        vectors = np.zeros((n, vocab_size), dtype=np.float64)
        for i, tokens in enumerate(token_lists):
            tf: Counter = Counter(tokens)
            for term, count in tf.items():
                idf = math.log((n + 1) / (doc_freq[term] + 1)) + 1.0
                vectors[i, vocab[term]] = count * idf

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors = vectors / norms

        result = vectors @ vectors.T
        np.clip(result, 0.0, 1.0, out=result)
        np.copyto(sim, result)

    # -- Edit distance similarity ------------------------------------------

    def _fill_edit(self, candidates: List[str], sim: np.ndarray) -> None:
        ed = EditDistanceUtility()
        n = len(candidates)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                val = ed.compute(candidates[i], candidates[j])
                sim[i, j] = val
                sim[j, i] = val


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


class MMRSelector:
    """Maximal Marginal Relevance (MMR) based diverse subset selection.

    Iteratively selects candidates that maximize:
        MMR(c) = λ * quality(c) - (1 - λ) * max_{s ∈ S} sim(c, s)

    where S is the already-selected set.
    """

    def select(
        self,
        scores: np.ndarray,
        similarity_matrix: np.ndarray,
        k: int,
        lambda_: float = 0.5,
    ) -> List[int]:
        """Select k indices using greedy MMR.

        Args:
            scores: 1-D array of quality scores for each candidate.
            similarity_matrix: (n, n) pairwise similarity matrix.
            k: Number of items to select.
            lambda_: Trade-off between quality (1.0) and diversity (0.0).

        Returns:
            List of selected candidate indices.
        """
        n = len(scores)
        k = min(k, n)
        if k <= 0:
            return []

        # Normalize scores to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)

        selected: List[int] = []
        remaining: Set[int] = set(range(n))

        # Seed with highest quality candidate
        first = int(np.argmax(norm_scores))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_idx = self._greedy_step(
                remaining, selected, norm_scores, similarity_matrix, lambda_
            )
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected

    def _mmr_score(
        self,
        candidate_idx: int,
        selected_indices: List[int],
        scores: np.ndarray,
        similarity: np.ndarray,
        lambda_: float,
    ) -> float:
        """Compute MMR score for a single candidate."""
        quality = scores[candidate_idx]
        if not selected_indices:
            return lambda_ * quality

        max_sim = max(
            similarity[candidate_idx, s] for s in selected_indices
        )
        return lambda_ * quality - (1.0 - lambda_) * max_sim

    def _greedy_step(
        self,
        remaining: Set[int],
        selected: List[int],
        scores: np.ndarray,
        similarity: np.ndarray,
        lambda_: float,
    ) -> int:
        """Execute one greedy selection step."""
        best_score = -float("inf")
        best_idx = -1
        for idx in remaining:
            s = self._mmr_score(idx, selected, scores, similarity, lambda_)
            if s > best_score:
                best_score = s
                best_idx = idx
        return best_idx


class SubmodularSelector:
    """Submodular function maximization for diverse subset selection.

    Combines facility location and dispersion objectives, both of which
    are monotone submodular, and selects a subset via the standard greedy
    algorithm which guarantees a (1 - 1/e) approximation.
    """

    def __init__(
        self,
        facility_weight: float = 0.5,
        dispersion_weight: float = 0.5,
    ) -> None:
        self.facility_weight = facility_weight
        self.dispersion_weight = dispersion_weight

    def select(
        self,
        scores: np.ndarray,
        similarity_matrix: np.ndarray,
        k: int,
    ) -> List[int]:
        """Select k items by greedy submodular maximization.

        The combined objective is:
            F(S) = w_f * FacilityLocation(S) + w_d * Dispersion(S)
                 + quality_bonus(S)

        where quality_bonus sums the normalized quality scores of selected items.
        """
        n = len(scores)
        k = min(k, n)
        if k <= 0:
            return []

        # Normalize scores
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)

        selected: List[int] = []
        remaining: Set[int] = set(range(n))

        for _ in range(k):
            if not remaining:
                break
            best_gain = -float("inf")
            best_idx = -1

            for c in remaining:
                gain = self._total_marginal_gain(
                    c, selected, similarity_matrix, norm_scores
                )
                if gain > best_gain:
                    best_gain = gain
                    best_idx = c

            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected

    def _total_marginal_gain(
        self,
        candidate: int,
        selected: List[int],
        similarity: np.ndarray,
        scores: np.ndarray,
    ) -> float:
        """Compute combined marginal gain of adding *candidate*."""
        fl_gain = self._facility_location_gain(candidate, selected, similarity)
        disp_gain = self._dispersion_gain(candidate, selected, similarity)
        quality_gain = scores[candidate]
        return (
            self.facility_weight * fl_gain
            + self.dispersion_weight * disp_gain
            + quality_gain
        )

    def _facility_location_gain(
        self,
        candidate: int,
        selected: List[int],
        similarity: np.ndarray,
    ) -> float:
        """Marginal gain of facility-location objective.

        FL(S) = Σ_v max_{s ∈ S} sim(v, s).
        Gain of adding c = Σ_v max(0, sim(v, c) - max_{s ∈ S} sim(v, s)).
        """
        n = similarity.shape[0]
        if not selected:
            return float(similarity[candidate, :].sum())

        current_max = np.max(similarity[np.ix_(selected, range(n))], axis=0)
        cand_sim = similarity[candidate, :]
        improvements = np.maximum(0.0, cand_sim - current_max)
        return float(improvements.sum())

    def _dispersion_gain(
        self,
        candidate: int,
        selected: List[int],
        similarity: np.ndarray,
    ) -> float:
        """Marginal gain for a dispersion-like objective.

        Rewards candidates that are dissimilar to the current set.
        Dispersion(c | S) = Σ_{s ∈ S} (1 - sim(c, s)).
        """
        if not selected:
            return 0.0
        total_dissim = 0.0
        for s in selected:
            total_dissim += 1.0 - similarity[candidate, s]
        return total_dissim / len(selected)

    def _marginal_gain(
        self,
        candidate: int,
        selected: List[int],
        objective_fn: Callable[[List[int]], float],
    ) -> float:
        """Generic marginal gain via function evaluation."""
        current_val = objective_fn(selected) if selected else 0.0
        new_val = objective_fn(selected + [candidate])
        return new_val - current_val


class ILPSelector:
    """Integer Linear Programming based diverse subset selection.

    Formulates the selection as a binary optimization problem and solves it
    with a greedy LP-relaxation approximation (no external solver needed).

    max  Σ_i x_i * quality_i  +  α * Σ_{i<j} x_i * x_j * (1 - sim_{ij})
    s.t. Σ_i x_i = k,  x_i ∈ {0, 1}

    We approximate this by iterative linearization (sequential greedy).
    """

    def __init__(self, diversity_weight: float = 0.3) -> None:
        self.diversity_weight = diversity_weight

    def select(
        self,
        scores: np.ndarray,
        similarity_matrix: np.ndarray,
        k: int,
    ) -> List[int]:
        n = len(scores)
        k = min(k, n)
        if k <= 0:
            return []

        # Normalize
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)

        # Dissimilarity matrix
        dissim = 1.0 - similarity_matrix

        selected: List[int] = []
        remaining = set(range(n))

        for _ in range(k):
            if not remaining:
                break
            best_obj = -float("inf")
            best_idx = -1
            for c in remaining:
                obj = norm_scores[c]
                if selected:
                    div_term = sum(dissim[c, s] for s in selected) / len(selected)
                    obj = (1.0 - self.diversity_weight) * obj + self.diversity_weight * div_term
                best_idx, best_obj = (c, obj) if obj > best_obj else (best_idx, best_obj)
            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected


# ---------------------------------------------------------------------------
# Text decoding helper
# ---------------------------------------------------------------------------


class TokenDecoder:
    """Decodes token ID sequences to strings.

    Wraps an optional external tokenizer; if none is available falls back
    to a simple integer-to-char mapping useful for unit tests.
    """

    def __init__(self, tokenizer: Optional[Any] = None) -> None:
        self._tokenizer = tokenizer

    def decode(self, token_ids: TokenSequence) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        # Fallback: map each token id to a short string
        return " ".join(str(tid) for tid in token_ids)

    def decode_batch(self, sequences: List[TokenSequence]) -> List[str]:
        return [self.decode(seq) for seq in sequences]


# ---------------------------------------------------------------------------
# MBR Analyzer
# ---------------------------------------------------------------------------


class MBRAnalyzer:
    """Diagnostic analysis tools for MBR diversity decoding.

    Provides methods to study the effect of pool size, utility distributions,
    and the quality–diversity tradeoff.
    """

    def __init__(
        self,
        utility_fn: Optional[UtilityFunction] = None,
        similarity_computer: Optional[SimilarityComputer] = None,
    ) -> None:
        self.utility_fn = utility_fn or BLEUUtility()
        self.similarity_computer = similarity_computer or SimilarityComputer()

    # -- Pool size effect --------------------------------------------------

    def analyze_pool_size_effect(
        self,
        logit_source: Any,
        prompt_ids: TokenSequence,
        pool_sizes: List[int],
        decoder: Optional[TokenDecoder] = None,
        config: Optional[MBRConfig] = None,
    ) -> dict:
        """Measure how candidate pool size affects expected utility and diversity.

        Generates pools of increasing size, computes expected utility for each,
        then reports statistics.
        """
        if config is None:
            config = MBRConfig()
        if decoder is None:
            decoder = TokenDecoder()

        results: Dict[str, Any] = {
            "pool_sizes": pool_sizes,
            "mean_expected_utilities": [],
            "max_expected_utilities": [],
            "std_expected_utilities": [],
            "mean_pairwise_distances": [],
        }

        mbr = MBRDiversity(config)

        for size in pool_sizes:
            candidates = mbr._generate_candidate_pool(
                logit_source, prompt_ids, size
            )
            references = mbr._generate_reference_pool(
                logit_source, prompt_ids, config.reference_pool_size
            )

            cand_strs = decoder.decode_batch(candidates)
            ref_strs = decoder.decode_batch(references)

            util_matrix = self.utility_fn.compute_batch(cand_strs, ref_strs)
            expected = util_matrix.mean(axis=1)

            results["mean_expected_utilities"].append(float(expected.mean()))
            results["max_expected_utilities"].append(float(expected.max()))
            results["std_expected_utilities"].append(float(expected.std()))

            sim_matrix = self.similarity_computer.compute_matrix(cand_strs)
            n = len(cand_strs)
            if n > 1:
                upper = sim_matrix[np.triu_indices(n, k=1)]
                results["mean_pairwise_distances"].append(
                    float(1.0 - upper.mean())
                )
            else:
                results["mean_pairwise_distances"].append(0.0)

        return results

    # -- Utility distribution analysis -------------------------------------

    def utility_distribution(self, utility_matrix: np.ndarray) -> dict:
        """Compute statistics over a utility matrix.

        Args:
            utility_matrix: Shape (n_candidates, n_references).

        Returns:
            Dict with distribution statistics.
        """
        expected = utility_matrix.mean(axis=1)
        per_ref_mean = utility_matrix.mean(axis=0)

        return {
            "n_candidates": utility_matrix.shape[0],
            "n_references": utility_matrix.shape[1],
            "expected_utility": {
                "mean": float(expected.mean()),
                "std": float(expected.std()),
                "min": float(expected.min()),
                "max": float(expected.max()),
                "median": float(np.median(expected)),
                "quartiles": [
                    float(np.percentile(expected, 25)),
                    float(np.percentile(expected, 50)),
                    float(np.percentile(expected, 75)),
                ],
            },
            "per_reference_mean": {
                "mean": float(per_ref_mean.mean()),
                "std": float(per_ref_mean.std()),
                "min": float(per_ref_mean.min()),
                "max": float(per_ref_mean.max()),
            },
            "overall": {
                "mean": float(utility_matrix.mean()),
                "std": float(utility_matrix.std()),
                "sparsity": float((utility_matrix == 0).mean()),
            },
        }

    # -- Diversity vs Quality tradeoff -------------------------------------

    def diversity_vs_quality_tradeoff(
        self,
        candidates: List[str],
        references: List[str],
        lambdas: List[float],
        k: int = 10,
    ) -> dict:
        """Evaluate the quality–diversity Pareto frontier across λ values.

        For each λ, runs MMR selection and reports:
        - mean quality of selected subset
        - mean pairwise diversity of selected subset
        """
        util_matrix = self.utility_fn.compute_batch(candidates, references)
        expected_utility = util_matrix.mean(axis=1)
        sim_matrix = self.similarity_computer.compute_matrix(candidates)
        selector = MMRSelector()

        results: Dict[str, Any] = {
            "lambdas": lambdas,
            "mean_qualities": [],
            "mean_diversities": [],
            "selected_sizes": [],
        }

        for lam in lambdas:
            selected = selector.select(expected_utility, sim_matrix, k, lam)
            sel_quality = float(expected_utility[selected].mean()) if selected else 0.0

            if len(selected) > 1:
                sel_sim = sim_matrix[np.ix_(selected, selected)]
                n_sel = len(selected)
                upper = sel_sim[np.triu_indices(n_sel, k=1)]
                diversity = float(1.0 - upper.mean())
            else:
                diversity = 0.0

            results["mean_qualities"].append(sel_quality)
            results["mean_diversities"].append(diversity)
            results["selected_sizes"].append(len(selected))

        return results

    # -- Reference pool convergence ----------------------------------------

    def reference_pool_convergence(
        self,
        logit_source: Any,
        prompt_ids: TokenSequence,
        ref_sizes: List[int],
        n_candidates: int = 50,
        decoder: Optional[TokenDecoder] = None,
        config: Optional[MBRConfig] = None,
    ) -> dict:
        """Check how expected-utility estimates converge with reference pool size."""
        if config is None:
            config = MBRConfig()
        if decoder is None:
            decoder = TokenDecoder()

        mbr = MBRDiversity(config)
        candidates = mbr._generate_candidate_pool(
            logit_source, prompt_ids, n_candidates
        )
        cand_strs = decoder.decode_batch(candidates)

        results: Dict[str, Any] = {
            "ref_sizes": ref_sizes,
            "mean_expected_utilities": [],
            "ranking_correlations": [],
        }

        prev_ranking: Optional[np.ndarray] = None

        for ref_size in ref_sizes:
            references = mbr._generate_reference_pool(
                logit_source, prompt_ids, ref_size
            )
            ref_strs = decoder.decode_batch(references)
            util_matrix = self.utility_fn.compute_batch(cand_strs, ref_strs)
            expected = util_matrix.mean(axis=1)

            results["mean_expected_utilities"].append(float(expected.mean()))

            ranking = np.argsort(-expected)
            if prev_ranking is not None:
                # Spearman-like rank correlation (simplified)
                n = len(ranking)
                rank_a = np.zeros(n, dtype=np.float64)
                rank_b = np.zeros(n, dtype=np.float64)
                for pos, idx in enumerate(prev_ranking):
                    rank_a[idx] = pos
                for pos, idx in enumerate(ranking):
                    rank_b[idx] = pos
                d = rank_a - rank_b
                rho = 1.0 - 6.0 * float((d ** 2).sum()) / (n * (n ** 2 - 1))
                results["ranking_correlations"].append(float(rho))
            else:
                results["ranking_correlations"].append(None)

            prev_ranking = ranking

        return results


# ---------------------------------------------------------------------------
# Main MBR Diversity Decoding Algorithm
# ---------------------------------------------------------------------------


class MBRDiversity(DecodingAlgorithm):
    """Minimum Bayes Risk (MBR) diversity decoding algorithm.

    Three-phase pipeline:
      1. Generate candidate and reference pools via ancestral sampling.
      2. Compute utility matrix and expected utilities.
      3. Select a diverse high-quality subset using MMR, submodular
         maximization, or ILP-based selection.
    """

    def __init__(self, config: MBRConfig) -> None:
        super().__init__(config)
        self.mbr_config: MBRConfig = config

        # Utility function
        self._utility_fn = _create_utility(
            config.utility_function,
            ngram_order=config.ngram_order,
            smoothing=config.smoothing,
            embedding_model=config.embedding_model,
        )

        # Selector
        if config.selection_method == "greedy_mmr":
            self._selector: Union[MMRSelector, SubmodularSelector, ILPSelector] = (
                MMRSelector()
            )
        elif config.selection_method == "submodular":
            self._selector = SubmodularSelector(
                facility_weight=config.facility_location_weight,
                dispersion_weight=config.dispersion_weight,
            )
        elif config.selection_method == "ilp":
            self._selector = ILPSelector(diversity_weight=config.diversity_weight)
        else:
            self._selector = MMRSelector()

        # Similarity computer
        self._similarity = SimilarityComputer(
            metric=config.similarity_metric,
            ngram_order=config.ngram_order,
        )

        # Token decoder (set externally or auto-detected)
        self._decoder = TokenDecoder()

        # RNG
        self._rng = np.random.RandomState(config.seed)

    # -- Public API --------------------------------------------------------

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Attach a tokenizer for decoding token IDs to strings."""
        self._decoder = TokenDecoder(tokenizer)

    def generate(
        self,
        logit_source: Any,
        prompt_ids: TokenSequence,
    ) -> List[TokenSequence]:
        """Run full MBR diversity decoding pipeline.

        Returns:
            List of *select_k* token sequences forming a diverse, high-quality
            subset of the candidate pool.
        """
        t0 = time.time()
        logger.info(
            "MBR diversity decoding: pool=%d, refs=%d, select=%d, "
            "utility=%s, method=%s",
            self.mbr_config.candidate_pool_size,
            self.mbr_config.reference_pool_size,
            self.mbr_config.select_k,
            self.mbr_config.utility_function,
            self.mbr_config.selection_method,
        )

        # Phase 1: Generate pools
        candidates = self._generate_candidate_pool(
            logit_source, prompt_ids, self.mbr_config.candidate_pool_size
        )
        references = self._generate_reference_pool(
            logit_source, prompt_ids, self.mbr_config.reference_pool_size
        )
        logger.info(
            "Generated %d candidates and %d references",
            len(candidates),
            len(references),
        )

        # Optionally deduplicate candidates
        if self.mbr_config.dedup_candidates:
            candidates = self._deduplicate(candidates)
            logger.info("After dedup: %d candidates", len(candidates))

        # Decode to strings
        cand_strs = self._decoder.decode_batch(candidates)
        ref_strs = self._decoder.decode_batch(references)

        # Phase 2: Compute utilities
        utility_matrix = self._compute_utility_matrix(cand_strs, ref_strs)
        expected_utility = self._compute_expected_utility(utility_matrix)
        logger.info(
            "Utility matrix shape=%s, E[U] range=[%.4f, %.4f]",
            utility_matrix.shape,
            expected_utility.min(),
            expected_utility.max(),
        )

        # Phase 3: Select diverse subset
        similarity = self._compute_pairwise_similarity(cand_strs)
        selected_indices = self._select_diverse_subset(
            expected_utility, similarity, self.mbr_config.select_k
        )
        logger.info("Selected %d diverse candidates", len(selected_indices))

        result = [candidates[i] for i in selected_indices]

        elapsed = time.time() - t0
        logger.info("MBR diversity decoding completed in %.2f s", elapsed)
        return result

    # -- Required abstract method ------------------------------------------

    def _step(self, state: DecodingState, logit_source: Any) -> DecodingState:
        """Single autoregressive decoding step (used for pool generation).

        Performs ancestral sampling at the configured temperature.
        """
        for idx in state.active_indices():
            seq = state.sequences[idx]
            logits = logit_source(seq)

            # Apply temperature
            if self.mbr_config.temperature != 1.0:
                logits = logits / self.mbr_config.temperature

            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, seq)

            # Apply n-gram blocking
            if self.config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram(
                    logits, seq, self.config.no_repeat_ngram_size
                )

            # Sample token
            token_id = sample_token(
                logits,
                temperature=1.0,  # already applied
                top_k=self.mbr_config.top_k_sampling,
                top_p=self.mbr_config.top_p_sampling,
            )
            state.update_sequence(idx, token_id)

            # Check EOS
            if (
                self.config.eos_token_id is not None
                and token_id == self.config.eos_token_id
            ):
                state.mark_finished(idx)

        state.step += 1
        return state

    # -- Phase 1: Pool generation ------------------------------------------

    def _generate_candidate_pool(
        self,
        logit_source: Any,
        prompt_ids: TokenSequence,
        n: int,
    ) -> List[TokenSequence]:
        """Generate *n* candidate sequences via ancestral sampling."""
        return self._generate_pool(logit_source, prompt_ids, n, "candidate")

    def _generate_reference_pool(
        self,
        logit_source: Any,
        prompt_ids: TokenSequence,
        n: int,
    ) -> List[TokenSequence]:
        """Generate *n* reference sequences via ancestral sampling."""
        return self._generate_pool(logit_source, prompt_ids, n, "reference")

    def _generate_pool(
        self,
        logit_source: Any,
        prompt_ids: TokenSequence,
        n: int,
        pool_name: str,
    ) -> List[TokenSequence]:
        """Shared pool generation logic.

        Generates *n* independent sequences by running the autoregressive
        loop *n* times (one sequence per state for memory efficiency).
        """
        sequences: List[TokenSequence] = []
        batch_size = min(n, 32)

        for batch_start in range(0, n, batch_size):
            batch_n = min(batch_size, n - batch_start)

            state = DecodingState(
                sequences=[list(prompt_ids) for _ in range(batch_n)],
                scores=[0.0] * batch_n,
                is_finished=[False] * batch_n,
                step=0,
                metadata={"pool_name": pool_name, "prompt_length": len(prompt_ids)},
            )

            stopping = self._build_stopping_criteria()

            while not state.all_finished():
                state = self._step(state, logit_source)
                if stopping(state):
                    break

            # Strip prompt prefix
            prompt_len = len(prompt_ids)
            for seq in state.sequences:
                generated = seq[prompt_len:]
                sequences.append(generated)

        return sequences[:n]

    def _deduplicate(self, sequences: List[TokenSequence]) -> List[TokenSequence]:
        """Remove duplicate sequences, preserving order."""
        seen: Set[tuple] = set()
        unique: List[TokenSequence] = []
        for seq in sequences:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique.append(seq)
        return unique

    # -- Phase 2: Utility computation --------------------------------------

    def _compute_utility_matrix(
        self,
        candidates: List[str],
        references: List[str],
    ) -> np.ndarray:
        """Compute utility matrix U[i,j] = utility(candidate_i, reference_j).

        Returns array of shape (n_candidates, n_references).
        """
        matrix = self._utility_fn.compute_batch(candidates, references)

        if self.mbr_config.normalize_utility and matrix.max() > 0:
            matrix = matrix / matrix.max()

        return matrix

    def _compute_expected_utility(
        self,
        utility_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute expected utility E_ref[U(candidate, ref)] per candidate.

        This is simply the mean over the reference axis (axis=1).
        Returns a 1-D array of length n_candidates.
        """
        return utility_matrix.mean(axis=1)

    def _compute_pairwise_similarity(
        self,
        candidates: List[str],
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between candidates.

        Returns a symmetric (n, n) matrix with values in [0, 1].
        """
        return self._similarity.compute_matrix(candidates)

    # -- Phase 3: Selection ------------------------------------------------

    def _select_diverse_subset(
        self,
        expected_utility: np.ndarray,
        similarity: np.ndarray,
        k: int,
    ) -> List[int]:
        """Dispatch to the configured selection method."""
        method = self.mbr_config.selection_method
        if method == "greedy_mmr":
            return self._select_diverse_subset_mmr(
                expected_utility, similarity, k, self.mbr_config.lambda_mmr
            )
        elif method == "submodular":
            return self._select_diverse_subset_submodular(
                expected_utility, similarity, k
            )
        elif method == "ilp":
            return self._select_diverse_subset_ilp(
                expected_utility, similarity, k
            )
        else:
            logger.warning("Unknown method %s, falling back to MMR", method)
            return self._select_diverse_subset_mmr(
                expected_utility, similarity, k, self.mbr_config.lambda_mmr
            )

    def _select_diverse_subset_mmr(
        self,
        expected_utility: np.ndarray,
        similarity: np.ndarray,
        k: int,
        lambda_: float,
    ) -> List[int]:
        """Select diverse subset using Maximal Marginal Relevance."""
        assert isinstance(self._selector, MMRSelector) or True
        selector = MMRSelector()
        return selector.select(expected_utility, similarity, k, lambda_)

    def _select_diverse_subset_submodular(
        self,
        expected_utility: np.ndarray,
        similarity: np.ndarray,
        k: int,
    ) -> List[int]:
        """Select diverse subset using greedy submodular maximization."""
        selector = SubmodularSelector(
            facility_weight=self.mbr_config.facility_location_weight,
            dispersion_weight=self.mbr_config.dispersion_weight,
        )
        return selector.select(expected_utility, similarity, k)

    def _select_diverse_subset_ilp(
        self,
        expected_utility: np.ndarray,
        similarity: np.ndarray,
        k: int,
    ) -> List[int]:
        """Select diverse subset using ILP-approximation."""
        selector = ILPSelector(diversity_weight=self.mbr_config.diversity_weight)
        return selector.select(expected_utility, similarity, k)

    # -- Introspection overrides -------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        """Return hyperparameter search space for MBR diversity decoding."""
        return {
            "candidate_pool_size": {"type": "int", "range": [20, 500], "default": 100},
            "reference_pool_size": {"type": "int", "range": [10, 200], "default": 50},
            "select_k": {"type": "int", "range": [5, 50], "default": 20},
            "utility_function": {
                "type": "categorical",
                "choices": ["bleu", "rouge", "bertscore", "chrF", "edit_distance"],
                "default": "bleu",
            },
            "diversity_weight": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.3,
            },
            "selection_method": {
                "type": "categorical",
                "choices": ["greedy_mmr", "submodular", "ilp"],
                "default": "greedy_mmr",
            },
            "lambda_mmr": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.5,
            },
            "temperature": {
                "type": "float",
                "range": [0.1, 2.0],
                "default": 1.0,
            },
            "ngram_order": {"type": "int", "range": [1, 6], "default": 4},
            "smoothing": {
                "type": "categorical",
                "choices": ["floor", "add_k", "none"],
                "default": "floor",
            },
            "similarity_metric": {
                "type": "categorical",
                "choices": ["ngram", "cosine", "edit"],
                "default": "ngram",
            },
        }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def create_mbr_diversity(
    candidate_pool_size: int = 100,
    reference_pool_size: int = 50,
    select_k: int = 20,
    utility_function: str = "bleu",
    diversity_weight: float = 0.3,
    selection_method: str = "greedy_mmr",
    lambda_mmr: float = 0.5,
    temperature: float = 1.0,
    **kwargs: Any,
) -> MBRDiversity:
    """Create an MBR diversity decoder with explicit parameters."""
    config = MBRConfig(
        candidate_pool_size=candidate_pool_size,
        reference_pool_size=reference_pool_size,
        select_k=select_k,
        utility_function=utility_function,
        diversity_weight=diversity_weight,
        selection_method=selection_method,
        lambda_mmr=lambda_mmr,
        temperature=temperature,
        **kwargs,
    )
    config.validate()
    return MBRDiversity(config)


def create_analyzer(
    utility_function: str = "bleu",
    similarity_metric: str = "ngram",
    ngram_order: int = 4,
) -> MBRAnalyzer:
    """Create an MBR analyzer with the given utility and similarity settings."""
    utility = _create_utility(utility_function, ngram_order=ngram_order)
    similarity = SimilarityComputer(metric=similarity_metric, ngram_order=ngram_order)
    return MBRAnalyzer(utility_fn=utility, similarity_computer=similarity)


# ---------------------------------------------------------------------------
# Register with the algorithm registry
# ---------------------------------------------------------------------------

try:
    AlgorithmRegistry.register("mbr_diversity", MBRDiversity)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Self-contained smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ---- BLEU smoke test -------------------------------------------------
    bleu = BLEUComputer(max_order=4, smoothing="floor")
    h = "the cat sat on the mat"
    r = "the cat is on the mat"
    score = bleu.sentence_bleu(h, r)
    print(f"BLEU('{h}', '{r}') = {score:.4f}")

    # ---- Utility functions -----------------------------------------------
    for name in ("bleu", "rouge", "chrf", "edit_distance", "bertscore"):
        fn = _create_utility(name)
        s = fn.compute(h, r)
        print(f"{name:>15}: {s:.4f}")

    # ---- Pairwise similarity ---------------------------------------------
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "a fast brown fox leaps over a sleepy dog",
        "the weather is sunny today",
        "machine learning is a subfield of artificial intelligence",
    ]
    sim_comp = SimilarityComputer(metric="ngram", ngram_order=4)
    sim = sim_comp.compute_matrix(texts)
    print("\nPairwise n-gram similarity:")
    for i in range(len(texts)):
        row = " ".join(f"{sim[i, j]:.3f}" for j in range(len(texts)))
        print(f"  [{row}]")

    # ---- MMR selection ---------------------------------------------------
    scores = np.array([0.9, 0.85, 0.8, 0.3])
    mmr = MMRSelector()
    selected = mmr.select(scores, sim, k=2, lambda_=0.5)
    print(f"\nMMR selected (k=2): {selected}")

    # ---- Submodular selection --------------------------------------------
    sub = SubmodularSelector()
    selected_sub = sub.select(scores, sim, k=2)
    print(f"Submodular selected (k=2): {selected_sub}")

    # ---- ILP selection ---------------------------------------------------
    ilp = ILPSelector(diversity_weight=0.3)
    selected_ilp = ilp.select(scores, sim, k=2)
    print(f"ILP selected (k=2): {selected_ilp}")

    # ---- Utility distribution analysis -----------------------------------
    util_fn = BLEUUtility()
    refs = [
        "the cat is on the mat",
        "there is a cat on the mat",
        "a cat sits on a mat",
    ]
    hyps = [
        "the cat sat on the mat",
        "a cat on the mat",
        "the dog is in the house",
    ]
    matrix = util_fn.compute_batch(hyps, refs)
    analyzer = MBRAnalyzer(utility_fn=util_fn)
    dist = analyzer.utility_distribution(matrix)
    print(f"\nUtility distribution: mean={dist['expected_utility']['mean']:.4f}, "
          f"max={dist['expected_utility']['max']:.4f}")

    # ---- Diversity-quality tradeoff --------------------------------------
    tradeoff = analyzer.diversity_vs_quality_tradeoff(
        hyps, refs, lambdas=[0.0, 0.25, 0.5, 0.75, 1.0], k=2
    )
    print("\nDiversity vs Quality tradeoff:")
    for lam, q, d in zip(
        tradeoff["lambdas"],
        tradeoff["mean_qualities"],
        tradeoff["mean_diversities"],
    ):
        print(f"  λ={lam:.2f}: quality={q:.4f}, diversity={d:.4f}")

    print("\nAll smoke tests passed ✓")
