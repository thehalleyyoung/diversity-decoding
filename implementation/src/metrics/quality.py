"""
Quality metrics module for the Diversity Decoding Arena.

Provides heuristic and statistical quality metrics for generated text,
with no external NLP library dependencies beyond numpy.
"""

import math
import re
import string
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    """Estimate the number of syllables in *word* using a vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    # Remove trailing silent-e
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


def word_tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    tokens = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\s]", text)
    return tokens


def sentence_split(text: str) -> List[str]:
    """Split *text* into sentences using punctuation heuristics."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or end
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        sentences = [text]
    return sentences


def flesch_reading_ease(text: str) -> float:
    """Compute the Flesch Reading Ease score for *text*."""
    sentences = sentence_split(text)
    words = word_tokenize(text)
    words_alpha = [w for w in words if w.isalpha()]
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words_alpha), 1)
    num_syllables = sum(count_syllables(w) for w in words_alpha)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return score


def flesch_kincaid_grade(text: str) -> float:
    """Compute the Flesch-Kincaid Grade Level for *text*."""
    sentences = sentence_split(text)
    words = word_tokenize(text)
    words_alpha = [w for w in words if w.isalpha()]
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words_alpha), 1)
    num_syllables = sum(count_syllables(w) for w in words_alpha)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    grade = 0.39 * asl + 11.8 * asw - 15.59
    return grade


def edit_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalised to [0, 1]."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return edit_distance(a, b) / max_len


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def tfidf_vectors(texts: List[str]) -> np.ndarray:
    """Build TF-IDF matrix (rows = documents) from *texts* using simple tokenisation.

    Returns an ``(n_docs, vocab_size)`` dense numpy array.
    """
    if not texts:
        return np.empty((0, 0))

    tokenised = [word_tokenize(t.lower()) for t in texts]
    vocab: Dict[str, int] = {}
    for tokens in tokenised:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    vocab_size = len(vocab)
    if vocab_size == 0:
        return np.zeros((len(texts), 1))

    n_docs = len(texts)
    tf = np.zeros((n_docs, vocab_size), dtype=np.float64)
    for i, tokens in enumerate(tokenised):
        for tok in tokens:
            tf[i, vocab[tok]] += 1
        row_sum = tf[i].sum()
        if row_sum > 0:
            tf[i] /= row_sum

    df = np.zeros(vocab_size, dtype=np.float64)
    for i, tokens in enumerate(tokenised):
        seen: Set[str] = set()
        for tok in tokens:
            if tok not in seen:
                df[vocab[tok]] += 1
                seen.add(tok)
    idf = np.log((n_docs + 1) / (df + 1)) + 1  # smoothed IDF

    tfidf = tf * idf[np.newaxis, :]
    return tfidf


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean of *values*.

    Returns ``(lower, upper)`` bounds.
    """
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class QualityMetric(ABC):
    """Abstract base class for all quality metrics."""

    @abstractmethod
    def compute(self, texts: List[str], prompt: Optional[str] = None) -> float:
        """Compute the aggregate metric over a list of texts."""
        ...

    @abstractmethod
    def compute_per_sample(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[float]:
        """Compute the metric independently for each text."""
        ...

    def compute_with_ci(
        self,
        texts: List[str],
        prompt: Optional[str] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, Tuple[float, float]]:
        """Return ``(mean, (lower, upper))`` with a bootstrap confidence interval."""
        per_sample = self.compute_per_sample(texts, prompt)
        mean_val = float(np.mean(per_sample)) if per_sample else 0.0
        ci = bootstrap_ci(per_sample, n_bootstrap=n_bootstrap, confidence=confidence)
        return (mean_val, ci)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        ...

    @property
    def description(self) -> str:
        return ""

    def validate_input(self, texts: List[str]) -> None:
        """Raise ``ValueError`` if *texts* is invalid."""
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"texts[{i}] is not a string: {type(t)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Constraint ABC + implementations
# ---------------------------------------------------------------------------

class Constraint(ABC):
    """Abstract base class for text constraints."""

    @abstractmethod
    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


@dataclass
class MinLengthConstraint(Constraint):
    """Text must have at least *min_words* words."""

    min_words: int = 10

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        words = word_tokenize(text)
        alpha_words = [w for w in words if w.isalpha()]
        return len(alpha_words) >= self.min_words

    @property
    def name(self) -> str:
        return f"min_length_{self.min_words}"

    @property
    def description(self) -> str:
        return f"Text must contain at least {self.min_words} words."


@dataclass
class MaxLengthConstraint(Constraint):
    """Text must have at most *max_words* words."""

    max_words: int = 500

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        words = word_tokenize(text)
        alpha_words = [w for w in words if w.isalpha()]
        return len(alpha_words) <= self.max_words

    @property
    def name(self) -> str:
        return f"max_length_{self.max_words}"

    @property
    def description(self) -> str:
        return f"Text must contain at most {self.max_words} words."


@dataclass
class ContainsKeywordConstraint(Constraint):
    """Text must contain **all** of the specified keywords (case-insensitive)."""

    keywords: List[str] = field(default_factory=list)

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        lower_text = text.lower()
        return all(kw.lower() in lower_text for kw in self.keywords)

    @property
    def name(self) -> str:
        return "contains_keywords"

    @property
    def description(self) -> str:
        return f"Text must contain keywords: {', '.join(self.keywords)}"


@dataclass
class DoesNotContainConstraint(Constraint):
    """Text must **not** contain any of the forbidden words (case-insensitive)."""

    forbidden: List[str] = field(default_factory=list)

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        lower_text = text.lower()
        return not any(fw.lower() in lower_text for fw in self.forbidden)

    @property
    def name(self) -> str:
        return "does_not_contain"

    @property
    def description(self) -> str:
        return f"Text must not contain: {', '.join(self.forbidden)}"


@dataclass
class SentenceCountConstraint(Constraint):
    """Text must have between *min_sentences* and *max_sentences* sentences."""

    min_sentences: int = 1
    max_sentences: int = 50

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        sents = sentence_split(text)
        return self.min_sentences <= len(sents) <= self.max_sentences

    @property
    def name(self) -> str:
        return f"sentence_count_{self.min_sentences}_{self.max_sentences}"

    @property
    def description(self) -> str:
        return (
            f"Text must have between {self.min_sentences} and "
            f"{self.max_sentences} sentences."
        )


@dataclass
class RepetitionConstraint(Constraint):
    """Fraction of repeated n-grams must not exceed *max_repeat_ratio*."""

    max_repeat_ratio: float = 0.3
    ngram_order: int = 3

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        tokens = word_tokenize(text.lower())
        if len(tokens) < self.ngram_order:
            return True
        ngrams = [
            tuple(tokens[i : i + self.ngram_order])
            for i in range(len(tokens) - self.ngram_order + 1)
        ]
        if not ngrams:
            return True
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        ratio = repeated / len(ngrams)
        return ratio <= self.max_repeat_ratio

    @property
    def name(self) -> str:
        return f"repetition_{self.max_repeat_ratio}"

    @property
    def description(self) -> str:
        return (
            f"Repeated {self.ngram_order}-gram ratio must be "
            f"<= {self.max_repeat_ratio}."
        )


@dataclass
class CoherenceConstraint(Constraint):
    """Minimum coherence score (computed as average adjacent-sentence similarity)."""

    min_coherence: float = 0.1

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        sents = sentence_split(text)
        if len(sents) < 2:
            return True
        embeddings = [_simple_sentence_embedding(s) for s in sents]
        sims: List[float] = []
        for i in range(len(embeddings) - 1):
            sims.append(cosine_similarity(embeddings[i], embeddings[i + 1]))
        avg_sim = float(np.mean(sims))
        return avg_sim >= self.min_coherence

    @property
    def name(self) -> str:
        return f"coherence_{self.min_coherence}"

    @property
    def description(self) -> str:
        return f"Average adjacent-sentence similarity must be >= {self.min_coherence}."


@dataclass
class FormatConstraint(Constraint):
    """Text must match the given regex *pattern*."""

    pattern: str = ".*"

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        try:
            return bool(re.search(self.pattern, text, re.DOTALL))
        except re.error:
            logger.warning("Invalid regex pattern in FormatConstraint: %s", self.pattern)
            return False

    @property
    def name(self) -> str:
        return "format_constraint"

    @property
    def description(self) -> str:
        return f"Text must match regex: {self.pattern}"


@dataclass
class ReadabilityConstraint(Constraint):
    """Flesch-Kincaid grade level must be within [*min_score*, *max_score*]."""

    min_score: float = 0.0
    max_score: float = 18.0

    def check(self, text: str, prompt: Optional[str] = None) -> bool:
        grade = flesch_kincaid_grade(text)
        return self.min_score <= grade <= self.max_score

    @property
    def name(self) -> str:
        return f"readability_{self.min_score}_{self.max_score}"

    @property
    def description(self) -> str:
        return (
            f"Flesch-Kincaid grade must be between "
            f"{self.min_score} and {self.max_score}."
        )


# ---------------------------------------------------------------------------
# Internal helper — sentence embedding for coherence / NLI
# ---------------------------------------------------------------------------

# A minimal set of English stop-words used for weighting.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "it", "its", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am",
}

# Common English words for frequency scoring (top-200 approximate)
_COMMON_WORDS: Set[str] = _STOP_WORDS | {
    "said", "one", "also", "like", "people", "new", "time", "way", "would",
    "make", "know", "get", "go", "see", "think", "say", "come", "want",
    "look", "use", "find", "give", "tell", "work", "call", "try", "ask",
    "need", "feel", "become", "leave", "put", "mean", "keep", "let",
    "begin", "seem", "help", "show", "hear", "play", "run", "move",
    "live", "believe", "bring", "happen", "write", "provide", "sit",
    "stand", "lose", "pay", "meet", "include", "continue", "set",
    "learn", "change", "lead", "understand", "watch", "follow", "stop",
    "create", "speak", "read", "allow", "add", "spend", "grow", "open",
    "walk", "win", "offer", "remember", "love", "consider", "appear",
    "buy", "wait", "serve", "die", "send", "expect", "build", "stay",
    "fall", "cut", "reach", "kill", "remain", "suggest", "raise", "pass",
    "sell", "require", "report", "decide", "pull",
}


def _simple_sentence_embedding(sentence: str, vocab: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Create a bag-of-words embedding for a sentence.

    If *vocab* is provided it maps words → indices and determines dimension.
    Otherwise a default 5000-bucket hash is used.
    """
    tokens = word_tokenize(sentence.lower())
    dim = 5000
    if vocab is not None:
        dim = max(len(vocab), 1)
    vec = np.zeros(dim, dtype=np.float64)
    for tok in tokens:
        if vocab is not None:
            idx = vocab.get(tok)
            if idx is not None:
                weight = 0.1 if tok in _STOP_WORDS else 1.0
                vec[idx] += weight
        else:
            idx = hash(tok) % dim
            weight = 0.1 if tok in _STOP_WORDS else 1.0
            vec[idx] += weight
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Perplexity metric
# ---------------------------------------------------------------------------

class Perplexity(QualityMetric):
    """N-gram language-model perplexity (lower is better).

    Builds an n-gram model from the provided texts (self-perplexity) or
    from an optional reference corpus.  Supports Laplace and interpolated
    Kneser-Ney smoothing.

    Parameters
    ----------
    order : int
        Maximum n-gram order (default 3).
    smoothing : str
        ``"laplace"`` or ``"kneser_ney"`` (default ``"laplace"``).
    alpha : float
        Laplace smoothing parameter (default 1.0).
    discount : float
        Kneser-Ney discount (default 0.75).
    reference_texts : list of str, optional
        If provided, the model is trained on these texts instead of the
        input texts.  This gives *cross*-perplexity.
    """

    def __init__(
        self,
        order: int = 3,
        smoothing: str = "laplace",
        alpha: float = 1.0,
        discount: float = 0.75,
        reference_texts: Optional[List[str]] = None,
    ) -> None:
        self._order = order
        self._smoothing = smoothing
        self._alpha = alpha
        self._discount = discount
        self._reference_texts = reference_texts
        self._model: Optional[Dict] = None
        self._vocab_size: int = 0

    # -- public interface ---------------------------------------------------

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def description(self) -> str:
        return (
            f"N-gram perplexity (order={self._order}, "
            f"smoothing={self._smoothing}). Lower is better."
        )

    def compute(self, texts: List[str], prompt: Optional[str] = None) -> float:
        self.validate_input(texts)
        if not texts:
            return 0.0
        per_sample = self.compute_per_sample(texts, prompt)
        return float(np.mean(per_sample))

    def compute_per_sample(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[float]:
        self.validate_input(texts)
        if not texts:
            return []
        training = self._reference_texts if self._reference_texts else texts
        model = self._build_ngram_model(training, self._order)
        self._model = model
        results: List[float] = []
        for text in texts:
            ppl = self._compute_ngram_perplexity(text, self._order, self._smoothing)
            results.append(ppl)
        return results

    # -- model building -----------------------------------------------------

    def _build_ngram_model(self, texts: List[str], order: int) -> Dict:
        """Build n-gram count tables up to *order* from *texts*.

        Returns a dict::

            {
                "counts": {n: Counter of (n-gram_tuple -> count)},
                "vocab": set of all tokens,
                "total_unigrams": int,
            }
        """
        vocab: Set[str] = set()
        counts: Dict[int, Counter] = {n: Counter() for n in range(1, order + 1)}

        for text in texts:
            tokens = ["<s>"] * (order - 1) + word_tokenize(text.lower()) + ["</s>"]
            vocab.update(tokens)
            for n in range(1, order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    counts[n][ngram] += 1

        self._vocab_size = len(vocab)
        return {
            "counts": counts,
            "vocab": vocab,
            "total_unigrams": sum(counts[1].values()),
        }

    # -- perplexity computation ---------------------------------------------

    def _compute_ngram_perplexity(
        self, text: str, order: int, smoothing: str
    ) -> float:
        """Compute perplexity of *text* under the stored model."""
        if self._model is None:
            return float("inf")
        ce = self._cross_entropy(text, self._model)
        ppl = math.pow(2, ce) if ce < 500 else float("inf")
        return ppl

    def _cross_entropy(self, text: str, model: Dict) -> float:
        """Compute the per-token cross-entropy of *text* under *model*."""
        order = self._order
        tokens = ["<s>"] * (order - 1) + word_tokenize(text.lower()) + ["</s>"]
        if len(tokens) <= order - 1:
            return 0.0

        counts = model["counts"]
        vocab_size = max(self._vocab_size, 1)

        log_prob_sum = 0.0
        n_tokens = 0

        if self._smoothing == "kneser_ney":
            smoothed_probs = self._kneser_ney_smoothing(
                counts[order], self._discount
            )
            for i in range(order - 1, len(tokens)):
                ngram = tuple(tokens[i - order + 1 : i + 1])
                prob = smoothed_probs.get(ngram, 1.0 / vocab_size)
                log_prob_sum += math.log2(max(prob, 1e-20))
                n_tokens += 1
        else:
            smoothed = self._laplace_smoothing(
                counts[order], vocab_size, self._alpha
            )
            context_counts: Counter = Counter()
            for ngram, c in counts[order].items():
                context_counts[ngram[:-1]] += c

            for i in range(order - 1, len(tokens)):
                ngram = tuple(tokens[i - order + 1 : i + 1])
                context = ngram[:-1]
                numerator = smoothed.get(ngram, self._alpha)
                denominator = context_counts.get(context, 0) + self._alpha * vocab_size
                prob = numerator / max(denominator, 1e-20)
                log_prob_sum += math.log2(max(prob, 1e-20))
                n_tokens += 1

        if n_tokens == 0:
            return 0.0
        return -log_prob_sum / n_tokens

    # -- smoothing methods --------------------------------------------------

    @staticmethod
    def _laplace_smoothing(
        counts: Counter, vocab_size: int, alpha: float = 1.0
    ) -> Dict:
        """Return a dict mapping each observed n-gram to its Laplace-smoothed count."""
        smoothed: Dict[tuple, float] = {}
        for ngram, c in counts.items():
            smoothed[ngram] = c + alpha
        return smoothed

    @staticmethod
    def _kneser_ney_smoothing(counts: Counter, discount: float = 0.75) -> Dict:
        """Return a dict mapping n-grams to Kneser-Ney smoothed probabilities.

        Uses the simplified (non-interpolated) absolute discounting variant.
        """
        if not counts:
            return {}

        # Group by context
        context_counts: Dict[tuple, float] = defaultdict(float)
        context_types: Dict[tuple, int] = defaultdict(int)
        for ngram, c in counts.items():
            ctx = ngram[:-1]
            context_counts[ctx] += c
            context_types[ctx] += 1

        total_types = len(counts)
        continuation_counts: Counter = Counter()
        for ngram in counts:
            continuation_counts[ngram[-1:]] += 1

        probs: Dict[tuple, float] = {}
        for ngram, c in counts.items():
            ctx = ngram[:-1]
            ctx_total = context_counts[ctx]
            n_types = context_types[ctx]
            first_term = max(c - discount, 0.0) / max(ctx_total, 1.0)
            lambda_ctx = (discount * n_types) / max(ctx_total, 1.0)
            p_continuation = continuation_counts.get(ngram[-1:], 1) / max(
                total_types, 1
            )
            probs[ngram] = first_term + lambda_ctx * p_continuation

        return probs

    @staticmethod
    def _interpolated_model(
        models: List[Dict], weights: List[float]
    ) -> Dict:
        """Linearly interpolate multiple n-gram probability tables.

        Each element of *models* is a dict mapping n-gram tuples to
        probabilities.  *weights* must sum to 1.
        """
        if not models:
            return {}
        combined: Dict[tuple, float] = defaultdict(float)
        all_keys: Set[tuple] = set()
        for m in models:
            all_keys.update(m.keys())
        for key in all_keys:
            for m, w in zip(models, weights):
                combined[key] += w * m.get(key, 0.0)
        return dict(combined)


# ---------------------------------------------------------------------------
# NLI Coherence metric
# ---------------------------------------------------------------------------

class NLICoherence(QualityMetric):
    """Measures internal coherence of generated text.

    Combines local (adjacent-sentence), global (all-pairs), entity, and
    topic coherence into a single score.  Optionally includes prompt
    relevance.

    Parameters
    ----------
    local_weight : float
        Weight for adjacent-sentence coherence (default 0.35).
    global_weight : float
        Weight for all-pairs coherence (default 0.20).
    entity_weight : float
        Weight for entity-grid coherence (default 0.20).
    topic_weight : float
        Weight for topic coherence (default 0.10).
    prompt_weight : float
        Weight for prompt relevance (default 0.15).  Ignored when no
        prompt is supplied.
    """

    def __init__(
        self,
        local_weight: float = 0.35,
        global_weight: float = 0.20,
        entity_weight: float = 0.20,
        topic_weight: float = 0.10,
        prompt_weight: float = 0.15,
    ) -> None:
        self._local_weight = local_weight
        self._global_weight = global_weight
        self._entity_weight = entity_weight
        self._topic_weight = topic_weight
        self._prompt_weight = prompt_weight

    @property
    def name(self) -> str:
        return "nli_coherence"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Composite coherence metric combining local, global, entity, "
            "and topic coherence with optional prompt relevance."
        )

    def compute(self, texts: List[str], prompt: Optional[str] = None) -> float:
        self.validate_input(texts)
        if not texts:
            return 0.0
        per_sample = self.compute_per_sample(texts, prompt)
        return float(np.mean(per_sample))

    def compute_per_sample(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[float]:
        self.validate_input(texts)
        results: List[float] = []
        for text in texts:
            sentences = self._sentence_split(text)
            if len(sentences) <= 1:
                score = 1.0 if sentences else 0.0
            else:
                local_c = self._compute_local_coherence(sentences)
                global_c = self._compute_global_coherence(sentences)
                entity_c = self._entity_coherence(sentences)
                topic_c = self._topic_coherence(sentences)

                if prompt:
                    prompt_rel = self._compute_prompt_relevance(text, prompt)
                    total_w = (
                        self._local_weight
                        + self._global_weight
                        + self._entity_weight
                        + self._topic_weight
                        + self._prompt_weight
                    )
                    score = (
                        self._local_weight * local_c
                        + self._global_weight * global_c
                        + self._entity_weight * entity_c
                        + self._topic_weight * topic_c
                        + self._prompt_weight * prompt_rel
                    ) / total_w
                else:
                    total_w = (
                        self._local_weight
                        + self._global_weight
                        + self._entity_weight
                        + self._topic_weight
                    )
                    score = (
                        self._local_weight * local_c
                        + self._global_weight * global_c
                        + self._entity_weight * entity_c
                        + self._topic_weight * topic_c
                    ) / total_w

            results.append(float(np.clip(score, 0.0, 1.0)))
        return results

    # -- sentence splitting -------------------------------------------------

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        return sentence_split(text)

    # -- coherence components -----------------------------------------------

    def _compute_local_coherence(self, sentences: List[str]) -> float:
        """Average cosine similarity between adjacent sentence embeddings."""
        if len(sentences) < 2:
            return 1.0
        embeddings = [self._simple_sentence_embedding(s) for s in sentences]
        sims: List[float] = []
        for i in range(len(embeddings) - 1):
            sims.append(self._cosine_similarity(embeddings[i], embeddings[i + 1]))
        return float(np.mean(sims))

    def _compute_global_coherence(self, sentences: List[str]) -> float:
        """Average cosine similarity across all pairs of sentence embeddings."""
        if len(sentences) < 2:
            return 1.0
        embeddings = [self._simple_sentence_embedding(s) for s in sentences]
        sims: List[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(self._cosine_similarity(embeddings[i], embeddings[j]))
        return float(np.mean(sims)) if sims else 0.0

    def _compute_prompt_relevance(self, text: str, prompt: str) -> float:
        """Cosine similarity between text and prompt embeddings."""
        emb_text = self._simple_sentence_embedding(text)
        emb_prompt = self._simple_sentence_embedding(prompt)
        return self._cosine_similarity(emb_text, emb_prompt)

    def _simple_sentence_embedding(self, sentence: str) -> np.ndarray:
        return _simple_sentence_embedding(sentence)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return cosine_similarity(a, b)

    def _entity_coherence(self, sentences: List[str]) -> float:
        """Approximate entity-grid coherence.

        Tracks capitalised words (as entity proxies) across sentences and
        measures how many entities persist from one sentence to the next.
        """
        if len(sentences) < 2:
            return 1.0

        entity_sets: List[Set[str]] = []
        for sent in sentences:
            tokens = word_tokenize(sent)
            entities: Set[str] = set()
            for tok in tokens:
                if tok[0].isupper() and tok.lower() not in _STOP_WORDS and len(tok) > 1:
                    entities.add(tok.lower())
            # Also include repeated nouns as entity proxies
            lower_tokens = [t.lower() for t in tokens if t.isalpha() and len(t) > 3]
            freq = Counter(lower_tokens)
            for w, c in freq.items():
                if c >= 2 and w not in _STOP_WORDS:
                    entities.add(w)
            entity_sets.append(entities)

        continuations: List[float] = []
        for i in range(len(entity_sets) - 1):
            current = entity_sets[i]
            next_set = entity_sets[i + 1]
            if not current and not next_set:
                continuations.append(0.5)
            elif not current or not next_set:
                continuations.append(0.0)
            else:
                overlap = len(current & next_set)
                union = len(current | next_set)
                continuations.append(overlap / union if union > 0 else 0.0)

        return float(np.mean(continuations)) if continuations else 0.0

    def _topic_coherence(self, sentences: List[str]) -> float:
        """Keyword overlap between consecutive sentences (excluding stop words)."""
        if len(sentences) < 2:
            return 1.0

        keyword_sets: List[Set[str]] = []
        for sent in sentences:
            tokens = word_tokenize(sent.lower())
            keywords = {
                t for t in tokens if t.isalpha() and t not in _STOP_WORDS and len(t) > 2
            }
            keyword_sets.append(keywords)

        overlaps: List[float] = []
        for i in range(len(keyword_sets) - 1):
            a = keyword_sets[i]
            b = keyword_sets[i + 1]
            if not a and not b:
                overlaps.append(0.5)
            elif not a or not b:
                overlaps.append(0.0)
            else:
                overlap = len(a & b)
                union = len(a | b)
                overlaps.append(overlap / union if union > 0 else 0.0)

        return float(np.mean(overlaps)) if overlaps else 0.0


# ---------------------------------------------------------------------------
# Constraint Satisfaction metric
# ---------------------------------------------------------------------------

class ConstraintSatisfaction(QualityMetric):
    """Fraction of texts that satisfy all registered constraints."""

    def __init__(self, constraints: Optional[List[Constraint]] = None) -> None:
        self._constraints: List[Constraint] = list(constraints or [])

    @property
    def name(self) -> str:
        return "constraint_satisfaction"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        names = [c.name for c in self._constraints]
        return f"Fraction of texts satisfying constraints: {names}"

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def remove_constraint(self, name: str) -> None:
        self._constraints = [c for c in self._constraints if c.name != name]

    def compute(self, texts: List[str], prompt: Optional[str] = None) -> float:
        self.validate_input(texts)
        if not texts:
            return 0.0
        per_sample = self.compute_per_sample(texts, prompt)
        return float(np.mean(per_sample))

    def compute_per_sample(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[float]:
        self.validate_input(texts)
        results: List[float] = []
        for text in texts:
            passed, _ = self._check_all_constraints(text, prompt)
            results.append(1.0 if passed else 0.0)
        return results

    def _check_all_constraints(
        self, text: str, prompt: Optional[str]
    ) -> Tuple[bool, Dict[str, bool]]:
        """Check all constraints and return ``(all_passed, per_constraint_dict)``."""
        detail: Dict[str, bool] = {}
        all_passed = True
        for constraint in self._constraints:
            result = constraint.check(text, prompt)
            detail[constraint.name] = result
            if not result:
                all_passed = False
        return all_passed, detail


# ---------------------------------------------------------------------------
# Fluency metric
# ---------------------------------------------------------------------------

class Fluency(QualityMetric):
    """Statistical fluency metric (higher is better).

    Combines word-frequency, sentence-length, punctuation, repetition,
    and vocabulary sophistication scores.

    Parameters
    ----------
    freq_weight : float
        Weight for the common-word frequency score.
    sent_len_weight : float
        Weight for the sentence-length score.
    punct_weight : float
        Weight for punctuation usage.
    rep_weight : float
        Weight for repetition penalty.
    vocab_weight : float
        Weight for vocabulary sophistication.
    """

    def __init__(
        self,
        freq_weight: float = 0.25,
        sent_len_weight: float = 0.20,
        punct_weight: float = 0.15,
        rep_weight: float = 0.20,
        vocab_weight: float = 0.20,
    ) -> None:
        self._freq_weight = freq_weight
        self._sent_len_weight = sent_len_weight
        self._punct_weight = punct_weight
        self._rep_weight = rep_weight
        self._vocab_weight = vocab_weight

    @property
    def name(self) -> str:
        return "fluency"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Composite fluency metric combining word-frequency, "
            "sentence-length, punctuation, repetition, and vocabulary scores."
        )

    def compute(self, texts: List[str], prompt: Optional[str] = None) -> float:
        self.validate_input(texts)
        if not texts:
            return 0.0
        per_sample = self.compute_per_sample(texts, prompt)
        return float(np.mean(per_sample))

    def compute_per_sample(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[float]:
        self.validate_input(texts)
        results: List[float] = []
        for text in texts:
            freq_s = self._compute_word_frequency_score(text)
            sent_s = self._compute_sentence_length_score(text)
            punct_s = self._compute_punctuation_score(text)
            rep_s = self._compute_repetition_score(text)
            vocab_s = self._compute_vocabulary_score(text)

            total_w = (
                self._freq_weight
                + self._sent_len_weight
                + self._punct_weight
                + self._rep_weight
                + self._vocab_weight
            )
            score = (
                self._freq_weight * freq_s
                + self._sent_len_weight * sent_s
                + self._punct_weight * punct_s
                + self._rep_weight * rep_s
                + self._vocab_weight * vocab_s
            ) / total_w

            results.append(float(np.clip(score, 0.0, 1.0)))
        return results

    # -- component scores ---------------------------------------------------

    @staticmethod
    def _compute_word_frequency_score(text: str) -> float:
        """Fraction of words that are in the common English word list."""
        tokens = word_tokenize(text.lower())
        alpha_tokens = [t for t in tokens if t.isalpha()]
        if not alpha_tokens:
            return 0.0
        common_count = sum(1 for t in alpha_tokens if t in _COMMON_WORDS)
        ratio = common_count / len(alpha_tokens)
        # A moderate ratio (0.4-0.7) is ideal; too high is robotic, too low
        # may indicate gibberish.
        if ratio < 0.2:
            return ratio / 0.2 * 0.5
        elif ratio <= 0.7:
            return 0.5 + 0.5 * ((ratio - 0.2) / 0.5)
        else:
            return 1.0 - 0.3 * ((ratio - 0.7) / 0.3)

    @staticmethod
    def _compute_sentence_length_score(text: str) -> float:
        """Score based on sentence length distribution.

        Penalises very short (<3 words) and very long (>50 words) sentences.
        """
        sentences = sentence_split(text)
        if not sentences:
            return 0.0
        scores: List[float] = []
        for sent in sentences:
            words = [w for w in word_tokenize(sent) if w.isalpha()]
            n = len(words)
            if n < 3:
                scores.append(0.3)
            elif n <= 30:
                scores.append(1.0)
            elif n <= 50:
                scores.append(1.0 - 0.5 * ((n - 30) / 20))
            else:
                scores.append(0.2)
        return float(np.mean(scores))

    @staticmethod
    def _compute_punctuation_score(text: str) -> float:
        """Score proper punctuation usage.

        Checks: sentences end with punctuation; commas/periods are followed
        by spaces; no double-spaces; balanced quotes and parentheses.
        """
        if not text.strip():
            return 0.0

        checks: List[float] = []

        # Sentence-ending punctuation
        sentences = sentence_split(text)
        ends_with_punct = sum(
            1 for s in sentences if s.strip() and s.strip()[-1] in ".!?;:"
        )
        checks.append(ends_with_punct / max(len(sentences), 1))

        # Punctuation followed by space or end
        bad_punct = len(re.findall(r'[,;:][^\s\n"\')]', text))
        total_punct = text.count(",") + text.count(";") + text.count(":")
        if total_punct > 0:
            checks.append(1.0 - bad_punct / total_punct)
        else:
            checks.append(0.8)

        # No excessive double spaces
        double_spaces = text.count("  ")
        total_spaces = text.count(" ")
        if total_spaces > 0:
            checks.append(1.0 - min(double_spaces / max(total_spaces, 1), 1.0))
        else:
            checks.append(0.5)

        # Balanced quotes
        quote_chars = ['"', "'"]
        for qc in quote_chars:
            cnt = text.count(qc)
            if cnt % 2 == 0:
                checks.append(1.0)
            else:
                checks.append(0.5)

        # Balanced parentheses
        open_p = text.count("(")
        close_p = text.count(")")
        if open_p == close_p:
            checks.append(1.0)
        else:
            checks.append(0.5)

        return float(np.mean(checks))

    @staticmethod
    def _compute_repetition_score(text: str) -> float:
        """Penalise excessive repetition of words and n-grams."""
        tokens = word_tokenize(text.lower())
        alpha_tokens = [t for t in tokens if t.isalpha()]
        if len(alpha_tokens) < 5:
            return 0.5

        # Word-level repetition
        freq = Counter(alpha_tokens)
        most_common_ratio = freq.most_common(1)[0][1] / len(alpha_tokens)

        # Bigram repetition
        bigrams = [
            (alpha_tokens[i], alpha_tokens[i + 1])
            for i in range(len(alpha_tokens) - 1)
        ]
        if bigrams:
            bi_freq = Counter(bigrams)
            bi_ratio = bi_freq.most_common(1)[0][1] / len(bigrams)
        else:
            bi_ratio = 0.0

        # Trigram repetition
        trigrams = [
            (alpha_tokens[i], alpha_tokens[i + 1], alpha_tokens[i + 2])
            for i in range(len(alpha_tokens) - 2)
        ]
        if trigrams:
            tri_freq = Counter(trigrams)
            tri_ratio = tri_freq.most_common(1)[0][1] / len(trigrams)
        else:
            tri_ratio = 0.0

        # Filter out stop-word dominated ratios
        top_word = freq.most_common(1)[0][0]
        if top_word in _STOP_WORDS:
            most_common_ratio *= 0.5

        penalty = most_common_ratio * 0.3 + bi_ratio * 0.3 + tri_ratio * 0.4
        score = 1.0 - min(penalty * 3.0, 1.0)
        return max(score, 0.0)

    @staticmethod
    def _compute_vocabulary_score(text: str) -> float:
        """Score vocabulary sophistication via type-token ratio and word length."""
        tokens = word_tokenize(text.lower())
        alpha_tokens = [t for t in tokens if t.isalpha()]
        if len(alpha_tokens) < 5:
            return 0.5

        # Type-token ratio (corrected for length)
        types = len(set(alpha_tokens))
        ttr = types / math.sqrt(len(alpha_tokens))  # Guiraud's TTR
        # Normalise to [0, 1] — typical range for Guiraud is 3-10
        ttr_score = min(ttr / 8.0, 1.0)

        # Average word length — very short avg indicates simple vocab
        avg_len = np.mean([len(w) for w in alpha_tokens])
        if avg_len < 3:
            len_score = 0.3
        elif avg_len <= 6:
            len_score = 0.5 + 0.5 * ((avg_len - 3) / 3)
        else:
            len_score = 1.0

        # Hapax legomena ratio (words appearing only once)
        freq = Counter(alpha_tokens)
        hapax = sum(1 for c in freq.values() if c == 1)
        hapax_ratio = hapax / max(types, 1)
        # High hapax ratio (~0.5+) suggests rich vocabulary
        hapax_score = min(hapax_ratio / 0.6, 1.0)

        return float(np.clip(
            0.4 * ttr_score + 0.3 * len_score + 0.3 * hapax_score, 0.0, 1.0
        ))


# ---------------------------------------------------------------------------
# Distinctiveness metric
# ---------------------------------------------------------------------------

class Distinctiveness(QualityMetric):
    """Measures how distinct generated text is from the prompt.

    Combines n-gram novelty, length ratio, and edit distance.

    Parameters
    ----------
    ngram_orders : list of int
        N-gram orders to check for novelty (default [1, 2, 3]).
    novelty_weight : float
        Weight for n-gram novelty (default 0.5).
    length_weight : float
        Weight for length ratio (default 0.2).
    edit_weight : float
        Weight for normalised edit distance (default 0.3).
    """

    def __init__(
        self,
        ngram_orders: Optional[List[int]] = None,
        novelty_weight: float = 0.5,
        length_weight: float = 0.2,
        edit_weight: float = 0.3,
    ) -> None:
        self._ngram_orders = ngram_orders or [1, 2, 3]
        self._novelty_weight = novelty_weight
        self._length_weight = length_weight
        self._edit_weight = edit_weight

    @property
    def name(self) -> str:
        return "distinctiveness"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Measures how distinct generated text is from the prompt "
            "using n-gram novelty, length ratio, and edit distance."
        )

    def compute(self, texts: List[str], prompt: Optional[str] = None) -> float:
        self.validate_input(texts)
        if not texts:
            return 0.0
        per_sample = self.compute_per_sample(texts, prompt)
        return float(np.mean(per_sample))

    def compute_per_sample(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[float]:
        self.validate_input(texts)
        if prompt is None:
            # Without a prompt, distinctiveness is undefined; return 1.0
            return [1.0] * len(texts)

        results: List[float] = []
        for text in texts:
            # N-gram novelty (averaged over orders)
            novelties: List[float] = []
            for n in self._ngram_orders:
                novelties.append(self._ngram_novelty(text, prompt, n))
            avg_novelty = float(np.mean(novelties)) if novelties else 1.0

            lr = self._length_ratio(text, prompt)
            ed = self._edit_distance_ratio(text, prompt)

            total_w = self._novelty_weight + self._length_weight + self._edit_weight
            score = (
                self._novelty_weight * avg_novelty
                + self._length_weight * lr
                + self._edit_weight * ed
            ) / total_w

            results.append(float(np.clip(score, 0.0, 1.0)))
        return results

    @staticmethod
    def _ngram_novelty(text: str, prompt: str, n: int = 2) -> float:
        """Fraction of n-grams in *text* that do not appear in *prompt*."""
        text_tokens = word_tokenize(text.lower())
        prompt_tokens = word_tokenize(prompt.lower())
        if len(text_tokens) < n:
            return 1.0

        text_ngrams = [
            tuple(text_tokens[i : i + n]) for i in range(len(text_tokens) - n + 1)
        ]
        prompt_ngrams = set(
            tuple(prompt_tokens[i : i + n])
            for i in range(len(prompt_tokens) - n + 1)
        )
        if not text_ngrams:
            return 1.0
        novel = sum(1 for ng in text_ngrams if ng not in prompt_ngrams)
        return novel / len(text_ngrams)

    @staticmethod
    def _length_ratio(text: str, prompt: str) -> float:
        """Score based on length ratio — near 1.0 is suspicious (copy)."""
        text_len = len(word_tokenize(text))
        prompt_len = len(word_tokenize(prompt))
        if prompt_len == 0:
            return 1.0
        ratio = text_len / prompt_len
        if 0.9 <= ratio <= 1.1:
            return 0.2  # suspiciously similar length
        elif ratio < 0.5 or ratio > 3.0:
            return 1.0
        else:
            return min(abs(ratio - 1.0) / 0.5, 1.0)

    @staticmethod
    def _edit_distance_ratio(text: str, prompt: str) -> float:
        """Normalised edit distance between text and prompt.

        Higher means more distinct.  For very long strings, operate on
        word sequences to keep computation feasible.
        """
        text_tokens = word_tokenize(text.lower())
        prompt_tokens = word_tokenize(prompt.lower())
        # Use word-level edit distance for efficiency
        t_str = " ".join(text_tokens)
        p_str = " ".join(prompt_tokens)
        if len(t_str) > 5000 or len(p_str) > 5000:
            # Approximate with token-set overlap
            t_set = set(text_tokens)
            p_set = set(prompt_tokens)
            if not t_set and not p_set:
                return 0.0
            jaccard = len(t_set & p_set) / max(len(t_set | p_set), 1)
            return 1.0 - jaccard
        return normalized_edit_distance(t_str, p_str)


# ---------------------------------------------------------------------------
# Quality Metric Suite
# ---------------------------------------------------------------------------

class QualityMetricSuite:
    """A collection of quality metrics that can be evaluated together.

    Parameters
    ----------
    metrics : list of QualityMetric
        Metrics to include in the suite.
    """

    def __init__(self, metrics: Optional[List[QualityMetric]] = None) -> None:
        self._metrics: List[QualityMetric] = list(metrics or [])

    def add_metric(self, metric: QualityMetric) -> None:
        self._metrics.append(metric)

    def compute_all(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute all metrics and return a name → value dict."""
        results: Dict[str, float] = {}
        for metric in self._metrics:
            try:
                results[metric.name] = metric.compute(texts, prompt)
            except Exception as exc:
                logger.error("Error computing %s: %s", metric.name, exc)
                results[metric.name] = float("nan")
        return results

    def compute_all_with_ci(
        self,
        texts: List[str],
        prompt: Optional[str] = None,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Tuple[float, Tuple[float, float]]]:
        """Compute all metrics with bootstrap confidence intervals."""
        results: Dict[str, Tuple[float, Tuple[float, float]]] = {}
        for metric in self._metrics:
            try:
                results[metric.name] = metric.compute_with_ci(
                    texts, prompt, n_bootstrap=n_bootstrap
                )
            except Exception as exc:
                logger.error("Error computing %s: %s", metric.name, exc)
                results[metric.name] = (float("nan"), (float("nan"), float("nan")))
        return results

    def summary(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> dict:
        """Return a human-readable summary dict."""
        all_vals = self.compute_all(texts, prompt)
        summary_dict: dict = {
            "n_texts": len(texts),
            "prompt_provided": prompt is not None,
            "metrics": {},
        }
        for metric in self._metrics:
            val = all_vals.get(metric.name, float("nan"))
            summary_dict["metrics"][metric.name] = {
                "value": val,
                "higher_is_better": metric.higher_is_better,
                "description": metric.description,
            }
        return summary_dict

    def __repr__(self) -> str:
        names = [m.name for m in self._metrics]
        return f"QualityMetricSuite(metrics={names})"

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self):
        return iter(self._metrics)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def default_quality_suite() -> QualityMetricSuite:
    """Create a :class:`QualityMetricSuite` with all default metrics."""
    return QualityMetricSuite(
        metrics=[
            Perplexity(),
            NLICoherence(),
            Fluency(),
            Distinctiveness(),
            ConstraintSatisfaction(
                constraints=[
                    MinLengthConstraint(min_words=5),
                    MaxLengthConstraint(max_words=2000),
                    RepetitionConstraint(max_repeat_ratio=0.4),
                ]
            ),
        ]
    )
