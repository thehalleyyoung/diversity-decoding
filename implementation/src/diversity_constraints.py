"""
Constrained diverse generation for the Diversity Decoding Arena.

Implements constraint types that enforce topic coverage, length bounds,
style variety, factual inclusion, exclusion lists, diversity thresholds,
tone matching, and composite constraints.  All text analysis is
self-contained — no external NLP libraries required.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
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

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")

_FORMAL_INDICATORS: List[str] = [
    "furthermore", "moreover", "therefore", "consequently", "nevertheless",
    "notwithstanding", "henceforth", "whereby", "herein", "thus",
    "accordingly", "whereas", "indeed", "hence", "subsequently",
]

_CASUAL_INDICATORS: List[str] = [
    "gonna", "wanna", "kinda", "sorta", "gotta", "ya", "yeah", "nah",
    "cool", "awesome", "stuff", "things", "basically", "literally",
    "lol", "omg", "btw", "tbh", "imo", "nope", "yep", "hey", "wow",
]

_ACADEMIC_INDICATORS: List[str] = [
    "hypothesis", "methodology", "empirical", "significant", "analysis",
    "correlation", "regression", "theoretical", "framework", "literature",
    "abstract", "conclusion", "findings", "demonstrates", "observations",
    "furthermore", "et al", "respectively", "paradigm", "discourse",
]

_PERSUASIVE_INDICATORS: List[str] = [
    "must", "should", "clearly", "undoubtedly", "certainly", "crucial",
    "essential", "imperative", "urgent", "critical", "vital", "important",
    "compelling", "convincing", "undeniable", "irrefutable", "paramount",
]

_TONE_INDICATORS: Dict[str, List[str]] = {
    "formal": _FORMAL_INDICATORS,
    "casual": _CASUAL_INDICATORS,
    "academic": _ACADEMIC_INDICATORS,
    "persuasive": _PERSUASIVE_INDICATORS,
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return [m.lower() for m in _WORD_RE.findall(text)]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    return sentences if sentences else [text.strip()] if text.strip() else []


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _fuzzy_match(needle: str, haystack: str, ngram_size: int = 3) -> float:
    """Fuzzy match via character n-gram Jaccard similarity."""
    needle_lower = needle.lower()
    haystack_lower = haystack.lower()

    # Exact substring check first
    if needle_lower in haystack_lower:
        return 1.0

    needle_ngrams = _char_ngrams(needle_lower, ngram_size)
    if not needle_ngrams:
        return 0.0

    # Slide a window of needle-length over the haystack and take the best
    best = 0.0
    window = len(needle_lower) + 4  # slight slack
    for start in range(0, max(1, len(haystack_lower) - window + 1)):
        chunk = haystack_lower[start : start + window]
        chunk_ngrams = _char_ngrams(chunk, ngram_size)
        sim = _jaccard_similarity(needle_ngrams, chunk_ngrams)
        if sim > best:
            best = sim
    return best


def _sigmoid(x: float, centre: float = 0.0, steepness: float = 1.0) -> float:
    """Standard sigmoid function with configurable centre and steepness."""
    z = steepness * (x - centre)
    z = max(-500.0, min(500.0, z))  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-z))


def _crude_stem(word: str) -> str:
    """Very crude suffix-stripping stemmer for English."""
    w = word.lower()
    for suffix in ("ingly", "tion", "sion", "ness", "ment", "ling",
                   "ious", "eous", "able", "ible", "ally", "edly",
                   "ing", "ous", "ful", "ive", "ize", "ise", "ess",
                   "ist", "ity", "ant", "ent", "ary", "ery", "ory",
                   "ate", "ure", "ial", "ual", "ual", "ly", "ed",
                   "er", "es", "en", "al", "ic"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def _distinct_n(tokens: List[str], n: int) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ConstraintResult:
    """Result of checking a single constraint against a response."""

    satisfied: bool
    score: float
    details: str
    violated_items: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.satisfied else "FAIL"
        return f"ConstraintResult({status}, score={self.score:.3f}, details={self.details!r})"


@dataclass
class ConstraintSet:
    """A named set of constraints combined with a logical operator."""

    constraints: List["Constraint"]
    logical_operator: str = "AND"  # "AND" or "OR"
    name: str = "unnamed"

    def check(self, response: str) -> ConstraintResult:
        """Evaluate all constraints in the set against a response."""
        results = [c.check(response) for c in self.constraints]
        if not results:
            return ConstraintResult(satisfied=True, score=1.0, details="empty set")

        scores = [r.score for r in results]
        all_violated: List[str] = []
        for r in results:
            all_violated.extend(r.violated_items)

        if self.logical_operator.upper() == "AND":
            satisfied = all(r.satisfied for r in results)
            agg_score = float(np.mean(scores))
        else:  # OR
            satisfied = any(r.satisfied for r in results)
            agg_score = float(np.max(scores))

        detail_parts = [f"{c.__class__.__name__}: {r.details}"
                        for c, r in zip(self.constraints, results)]
        return ConstraintResult(
            satisfied=satisfied,
            score=agg_score,
            details=f"{self.name}({self.logical_operator}): " + "; ".join(detail_parts),
            violated_items=all_violated,
        )


# ──────────────────────────────────────────────────────────────────────
# Constraint base class
# ──────────────────────────────────────────────────────────────────────


class Constraint(ABC):
    """Abstract base class for all generation constraints."""

    @abstractmethod
    def check(self, response: str) -> ConstraintResult:
        """Check whether *response* satisfies this constraint."""
        ...

    def batch_check(self, responses: List[str]) -> List[ConstraintResult]:
        """Check a batch of responses.  Subclasses may override for efficiency."""
        return [self.check(r) for r in responses]


# ──────────────────────────────────────────────────────────────────────
# Concrete constraint implementations
# ──────────────────────────────────────────────────────────────────────


class TopicConstraint(Constraint):
    """Ensure responses cover a specified set of topics.

    Detection uses both exact keyword matching and character n-gram
    overlap so that morphological variants and near-synonyms still
    register a partial match.
    """

    def __init__(self, topics: List[str], threshold: float = 0.5) -> None:
        self.topics = [t.lower().strip() for t in topics]
        self.threshold = threshold
        self._topic_ngrams = {t: _char_ngrams(t, 3) for t in self.topics}

    def check(self, response: str) -> ConstraintResult:
        """Return score = fraction of topics covered (keyword + n-gram)."""
        if not self.topics:
            return ConstraintResult(True, 1.0, "no topics specified")

        tokens = _tokenize(response)
        token_set = set(tokens)
        response_lower = response.lower()

        covered: List[str] = []
        missed: List[str] = []

        for topic in self.topics:
            topic_words = topic.split()
            # Direct keyword check
            if all(w in token_set for w in topic_words):
                covered.append(topic)
                continue
            # Substring check
            if topic in response_lower:
                covered.append(topic)
                continue
            # N-gram similarity fallback
            sim = _fuzzy_match(topic, response)
            if sim >= self.threshold:
                covered.append(topic)
            else:
                missed.append(topic)

        score = len(covered) / len(self.topics)
        satisfied = score >= 1.0 - 1e-9
        details = f"covered {len(covered)}/{len(self.topics)} topics"
        return ConstraintResult(satisfied, score, details, violated_items=missed)


class LengthConstraint(Constraint):
    """Enforce word-count length bounds with smooth sigmoid score falloff.

    The score is 1.0 inside the bounds and degrades smoothly outside via
    a sigmoid with configurable steepness.
    """

    def __init__(
        self,
        min_len: int = 0,
        max_len: int = 10_000,
        steepness: float = 0.15,
    ) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.steepness = steepness

    def check(self, response: str) -> ConstraintResult:
        """Score 1.0 within bounds, sigmoid falloff outside."""
        word_count = len(_tokenize(response))

        if self.min_len <= word_count <= self.max_len:
            return ConstraintResult(
                satisfied=True,
                score=1.0,
                details=f"word_count={word_count} in [{self.min_len}, {self.max_len}]",
            )

        if word_count < self.min_len:
            diff = self.min_len - word_count
            score = _sigmoid(-diff, centre=0.0, steepness=self.steepness)
            details = f"word_count={word_count} < min={self.min_len} (under by {diff})"
        else:
            diff = word_count - self.max_len
            score = _sigmoid(-diff, centre=0.0, steepness=self.steepness)
            details = f"word_count={word_count} > max={self.max_len} (over by {diff})"

        violated = [details]
        return ConstraintResult(satisfied=False, score=score, details=details,
                                violated_items=violated)


class StyleConstraint(Constraint):
    """Check that responses exhibit specific writing styles.

    Style detection uses lightweight linguistic features:
    * sentence length variance (complex vs simple)
    * vocabulary richness (type-token ratio)
    * punctuation density and variety
    * formality indicators
    """

    # Map style name → detector function name
    _STYLE_DETECTORS: Dict[str, str] = {
        "complex": "_detect_complex",
        "simple": "_detect_simple",
        "varied": "_detect_varied",
        "formal": "_detect_formal",
        "informal": "_detect_informal",
        "descriptive": "_detect_descriptive",
        "concise": "_detect_concise",
    }

    def __init__(self, styles: List[str], threshold: float = 0.4) -> None:
        self.styles = [s.lower().strip() for s in styles]
        self.threshold = threshold

    # ── Feature extraction ──────────────────────────────────────────

    @staticmethod
    def _sentence_length_variance(text: str) -> float:
        sentences = _split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        lengths = [len(_tokenize(s)) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        var = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        return math.sqrt(var)

    @staticmethod
    def _type_token_ratio(text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def _punctuation_density(text: str) -> float:
        if not text:
            return 0.0
        punct_count = sum(1 for ch in text if ch in string.punctuation)
        return punct_count / len(text)

    @staticmethod
    def _avg_sentence_length(text: str) -> float:
        sentences = _split_sentences(text)
        if not sentences:
            return 0.0
        return sum(len(_tokenize(s)) for s in sentences) / len(sentences)

    @staticmethod
    def _formality_score(text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        formal_count = sum(1 for t in tokens if t in _FORMAL_INDICATORS)
        casual_count = sum(1 for t in tokens if t in _CASUAL_INDICATORS)
        total = formal_count + casual_count
        if total == 0:
            return 0.5
        return formal_count / total

    # ── Style detectors ─────────────────────────────────────────────

    def _detect_complex(self, text: str) -> float:
        avg_len = self._avg_sentence_length(text)
        ttr = self._type_token_ratio(text)
        score = 0.0
        if avg_len > 15:
            score += min(1.0, (avg_len - 15) / 15)
        if ttr > 0.6:
            score += min(1.0, (ttr - 0.6) / 0.3) * 0.5
        return min(1.0, score)

    def _detect_simple(self, text: str) -> float:
        avg_len = self._avg_sentence_length(text)
        variance = self._sentence_length_variance(text)
        score = 0.0
        if avg_len < 12:
            score += min(1.0, (12 - avg_len) / 8)
        if variance < 4:
            score += 0.3
        return min(1.0, score)

    def _detect_varied(self, text: str) -> float:
        variance = self._sentence_length_variance(text)
        ttr = self._type_token_ratio(text)
        punct = self._punctuation_density(text)
        score = 0.0
        if variance > 5:
            score += min(1.0, variance / 12) * 0.5
        if ttr > 0.5:
            score += min(1.0, ttr) * 0.3
        if punct > 0.03:
            score += min(1.0, punct / 0.08) * 0.2
        return min(1.0, score)

    def _detect_formal(self, text: str) -> float:
        return self._formality_score(text)

    def _detect_informal(self, text: str) -> float:
        return 1.0 - self._formality_score(text)

    def _detect_descriptive(self, text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        # Heuristic: adjective-like words often end in -ly, -ous, -ful, -ive, -al
        adj_suffixes = ("ly", "ous", "ful", "ive", "al", "ish", "ent", "ant")
        adj_count = sum(1 for t in tokens if any(t.endswith(s) for s in adj_suffixes))
        ratio = adj_count / len(tokens)
        ttr = self._type_token_ratio(text)
        return min(1.0, ratio * 3 + ttr * 0.3)

    def _detect_concise(self, text: str) -> float:
        tokens = _tokenize(text)
        avg_len = self._avg_sentence_length(text)
        if not tokens:
            return 0.0
        word_lengths = [len(t) for t in tokens]
        avg_word = sum(word_lengths) / len(word_lengths)
        score = 0.0
        if avg_len < 10:
            score += 0.5
        if avg_word < 5:
            score += 0.3
        filler_words = {"very", "really", "just", "quite", "somewhat", "rather"}
        filler_ratio = sum(1 for t in tokens if t in filler_words) / len(tokens)
        score += max(0.0, 0.2 - filler_ratio * 2)
        return min(1.0, score)

    # ── Main interface ──────────────────────────────────────────────

    def check(self, response: str) -> ConstraintResult:
        """Score = fraction of required styles detected above threshold."""
        if not self.styles:
            return ConstraintResult(True, 1.0, "no styles specified")

        detected: List[str] = []
        missed: List[str] = []

        for style_name in self.styles:
            detector_name = self._STYLE_DETECTORS.get(style_name)
            if detector_name is not None:
                detector = getattr(self, detector_name)
                style_score = detector(response)
            else:
                # Fallback: treat style name as a keyword to look for
                style_score = 1.0 if style_name in response.lower() else 0.0

            if style_score >= self.threshold:
                detected.append(style_name)
            else:
                missed.append(style_name)

        score = len(detected) / len(self.styles)
        satisfied = len(missed) == 0
        details = f"detected styles: {detected}, missed: {missed}"
        return ConstraintResult(satisfied, score, details, violated_items=missed)


class FactualConstraint(Constraint):
    """Ensure specific facts are mentioned, allowing paraphrasing.

    Uses character n-gram Jaccard similarity so that reworded but
    semantically similar statements still match.
    """

    def __init__(
        self,
        facts: List[str],
        match_threshold: float = 0.35,
        ngram_size: int = 3,
    ) -> None:
        self.facts = facts
        self.match_threshold = match_threshold
        self.ngram_size = ngram_size

    def check(self, response: str) -> ConstraintResult:
        """Score = fraction of facts matched (fuzzy)."""
        if not self.facts:
            return ConstraintResult(True, 1.0, "no facts specified")

        matched: List[str] = []
        missed: List[str] = []

        sentences = _split_sentences(response)
        # Build a sentence-level index for faster matching
        sentence_ngrams = [_char_ngrams(s, self.ngram_size) for s in sentences]

        for fact in self.facts:
            fact_ngrams = _char_ngrams(fact.lower(), self.ngram_size)
            best_sim = 0.0

            # Check against whole response first
            whole_sim = _fuzzy_match(fact, response, self.ngram_size)
            best_sim = max(best_sim, whole_sim)

            # Check per sentence for better precision
            for s_ngrams in sentence_ngrams:
                sim = _jaccard_similarity(fact_ngrams, s_ngrams)
                if sim > best_sim:
                    best_sim = sim

            if best_sim >= self.match_threshold:
                matched.append(fact)
            else:
                missed.append(fact)

        score = len(matched) / len(self.facts)
        satisfied = len(missed) == 0
        details = f"matched {len(matched)}/{len(self.facts)} facts"
        return ConstraintResult(satisfied, score, details, violated_items=missed)


class ExclusionConstraint(Constraint):
    """Ensure forbidden words or phrases are absent.

    Matching is case-insensitive and uses crude stemming so that
    morphological variants of forbidden words are also caught.
    """

    def __init__(self, forbidden: List[str]) -> None:
        self.forbidden = [f.lower().strip() for f in forbidden]
        self._forbidden_stems = [
            [_crude_stem(w) for w in _tokenize(f)] for f in self.forbidden
        ]

    def check(self, response: str) -> ConstraintResult:
        """Score = 1 − (fraction of forbidden items found)."""
        if not self.forbidden:
            return ConstraintResult(True, 1.0, "no forbidden items")

        response_lower = response.lower()
        tokens = _tokenize(response)
        token_stems = [_crude_stem(t) for t in tokens]

        found: List[str] = []

        for phrase, stem_words in zip(self.forbidden, self._forbidden_stems):
            # Direct substring check
            if phrase in response_lower:
                found.append(phrase)
                continue

            # Stem-based check: all stem words present
            if stem_words and all(
                any(ts == sw for ts in token_stems) for sw in stem_words
            ):
                found.append(phrase)
                continue

        score = 1.0 - len(found) / len(self.forbidden)
        satisfied = len(found) == 0
        details = f"found {len(found)}/{len(self.forbidden)} forbidden items"
        return ConstraintResult(satisfied, score, details, violated_items=found)


class DiversityConstraint(Constraint):
    """Ensure a SET of responses meets a minimum diversity threshold.

    Computes distinct-n (n = 1, 2, 3) and pairwise Jaccard distance,
    then averages them into a single diversity score.

    Because this constraint evaluates across multiple responses, the
    single-response ``check`` method simply returns a neutral score.
    Use ``batch_check`` for the real evaluation.
    """

    def __init__(self, min_diversity: float = 0.3) -> None:
        self.min_diversity = min_diversity

    def check(self, response: str) -> ConstraintResult:
        """Single-response check is trivially satisfied."""
        return ConstraintResult(
            satisfied=True,
            score=1.0,
            details="diversity requires multiple responses; single response OK",
        )

    def batch_check(self, responses: List[str]) -> List[ConstraintResult]:
        """Compute set-level diversity and assign the same result to all."""
        if len(responses) < 2:
            result = ConstraintResult(True, 1.0, "fewer than 2 responses")
            return [result] * len(responses)

        # Tokenize all responses
        all_tokens = [_tokenize(r) for r in responses]

        # Distinct-n scores (averaged over all responses)
        distinct_scores: List[float] = []
        for n in (1, 2, 3):
            combined_tokens: List[str] = []
            for toks in all_tokens:
                combined_tokens.extend(toks)
            distinct_scores.append(_distinct_n(combined_tokens, n))

        # Pairwise Jaccard distance
        token_sets = [set(toks) for toks in all_tokens]
        pairwise_distances: List[float] = []
        for i, j in combinations(range(len(token_sets)), 2):
            sim = _jaccard_similarity(token_sets[i], token_sets[j])
            pairwise_distances.append(1.0 - sim)

        avg_distinct = float(np.mean(distinct_scores)) if distinct_scores else 0.0
        avg_distance = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0

        # Combined diversity score
        diversity_score = 0.5 * avg_distinct + 0.5 * avg_distance

        satisfied = diversity_score >= self.min_diversity
        details = (
            f"diversity={diversity_score:.3f} (distinct_avg={avg_distinct:.3f}, "
            f"jaccard_dist={avg_distance:.3f}), threshold={self.min_diversity}"
        )
        violated: List[str] = [] if satisfied else [
            f"diversity {diversity_score:.3f} < threshold {self.min_diversity}"
        ]

        result = ConstraintResult(satisfied, diversity_score, details, violated)
        return [result] * len(responses)


class ToneConstraint(Constraint):
    """Check that a response matches a target tone.

    Supported tones: formal, casual, academic, persuasive.  Detection
    is based on indicator-word density, sentence structure, and
    punctuation patterns.
    """

    def __init__(self, target_tone: str, threshold: float = 0.3) -> None:
        self.target_tone = target_tone.lower().strip()
        self.threshold = threshold

    def _tone_score(self, text: str, tone: str) -> float:
        """Compute a [0, 1] score for how well *text* matches *tone*."""
        tokens = _tokenize(text)
        if not tokens:
            return 0.0

        indicators = _TONE_INDICATORS.get(tone, [])
        indicator_set = set(indicators)
        indicator_count = sum(1 for t in tokens if t in indicator_set)
        indicator_density = indicator_count / len(tokens)

        # Sentence-structure features
        sentences = _split_sentences(text)
        avg_sent_len = (
            sum(len(_tokenize(s)) for s in sentences) / len(sentences)
            if sentences else 0.0
        )

        # Punctuation features
        exclamation_density = text.count("!") / max(1, len(text))
        question_density = text.count("?") / max(1, len(text))

        score = 0.0
        if tone == "formal":
            score += min(1.0, indicator_density * 20) * 0.4
            score += min(1.0, avg_sent_len / 25) * 0.3
            # Fewer exclamations = more formal
            score += max(0.0, 0.3 - exclamation_density * 100)
        elif tone == "casual":
            score += min(1.0, indicator_density * 15) * 0.4
            # Shorter sentences
            score += max(0.0, 1.0 - avg_sent_len / 20) * 0.3
            score += min(1.0, exclamation_density * 50) * 0.15
            score += min(1.0, question_density * 50) * 0.15
        elif tone == "academic":
            score += min(1.0, indicator_density * 15) * 0.4
            score += min(1.0, avg_sent_len / 20) * 0.3
            # Higher type-token ratio
            ttr = len(set(tokens)) / len(tokens) if tokens else 0
            score += min(1.0, ttr) * 0.3
        elif tone == "persuasive":
            score += min(1.0, indicator_density * 15) * 0.4
            score += min(1.0, exclamation_density * 30) * 0.2
            # Rhetorical questions
            score += min(1.0, question_density * 40) * 0.2
            # Strong/short sentences
            if sentences:
                short_count = sum(1 for s in sentences if len(_tokenize(s)) < 8)
                score += (short_count / len(sentences)) * 0.2
        else:
            # Unknown tone: fall back to keyword overlap
            tone_words = set(_tokenize(tone))
            overlap = len(tone_words & set(tokens))
            score = overlap / max(1, len(tone_words))

        return min(1.0, max(0.0, score))

    def check(self, response: str) -> ConstraintResult:
        """Score how well the response matches the target tone."""
        score = self._tone_score(response, self.target_tone)
        satisfied = score >= self.threshold
        details = f"tone '{self.target_tone}' score={score:.3f}, threshold={self.threshold}"
        violated: List[str] = [] if satisfied else [
            f"tone mismatch: {self.target_tone} ({score:.3f} < {self.threshold})"
        ]
        return ConstraintResult(satisfied, score, details, violated_items=violated)


class CompositeConstraint(Constraint):
    """Weighted combination of multiple constraints.

    The final score is the weighted average of sub-constraint scores.
    Satisfaction requires all sub-constraints to be individually satisfied,
    unless the weighted score exceeds an optional global threshold.
    """

    def __init__(
        self,
        constraints: List[Constraint],
        weights: Optional[List[float]] = None,
        satisfaction_threshold: float = 0.7,
    ) -> None:
        self.constraints = constraints
        if weights is None:
            self.weights = [1.0 / len(constraints)] * len(constraints)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights] if total > 0 else weights
        self.satisfaction_threshold = satisfaction_threshold

    def check(self, response: str) -> ConstraintResult:
        """Weighted aggregate of sub-constraint results."""
        if not self.constraints:
            return ConstraintResult(True, 1.0, "no sub-constraints")

        results = [c.check(response) for c in self.constraints]
        weighted_score = sum(w * r.score for w, r in zip(self.weights, results))

        all_violated: List[str] = []
        detail_parts: List[str] = []
        for c, r, w in zip(self.constraints, results, self.weights):
            detail_parts.append(
                f"{c.__class__.__name__}(w={w:.2f},s={r.score:.3f})"
            )
            all_violated.extend(r.violated_items)

        satisfied = weighted_score >= self.satisfaction_threshold
        details = f"weighted_score={weighted_score:.3f}: " + ", ".join(detail_parts)
        return ConstraintResult(satisfied, weighted_score, details, all_violated)

    def batch_check(self, responses: List[str]) -> List[ConstraintResult]:
        """Batch evaluation, respecting batch-aware sub-constraints."""
        if not self.constraints:
            return [ConstraintResult(True, 1.0, "no sub-constraints")] * len(responses)

        # Gather per-constraint batch results
        all_batch_results: List[List[ConstraintResult]] = []
        for c in self.constraints:
            all_batch_results.append(c.batch_check(responses))

        combined: List[ConstraintResult] = []
        for idx in range(len(responses)):
            sub_results = [batch[idx] for batch in all_batch_results]
            weighted_score = sum(
                w * r.score for w, r in zip(self.weights, sub_results)
            )
            all_violated: List[str] = []
            for r in sub_results:
                all_violated.extend(r.violated_items)

            satisfied = weighted_score >= self.satisfaction_threshold
            details = (
                f"weighted_score={weighted_score:.3f} across "
                f"{len(self.constraints)} constraints"
            )
            combined.append(
                ConstraintResult(satisfied, weighted_score, details, all_violated)
            )
        return combined


# ──────────────────────────────────────────────────────────────────────
# Module-level functions
# ──────────────────────────────────────────────────────────────────────


def generate_with_constraints(
    gen_fn: Callable[[], str],
    constraints: List[Constraint],
    n: int,
    max_retries: int = 50,
) -> List[str]:
    """Generate *n* outputs from *gen_fn* satisfying all *constraints*.

    Uses rejection sampling with progressive relaxation: after each
    failed round, the acceptance threshold is lowered so that the
    function always terminates and returns the best candidates found.

    Parameters
    ----------
    gen_fn:
        Zero-argument callable that returns a candidate string.
    constraints:
        List of constraints each candidate must satisfy.
    n:
        Number of outputs to return.
    max_retries:
        Maximum number of calls to *gen_fn* before giving up.

    Returns
    -------
    List of up to *n* strings that best satisfy the constraints.
    """
    if n <= 0:
        return []

    # Scored candidate pool: (score, text)
    pool: List[Tuple[float, str]] = []
    accepted: List[str] = []

    base_threshold = 1.0
    relaxation_rate = 0.85  # multiplicative relaxation per round

    attempts = 0
    current_threshold = base_threshold

    while len(accepted) < n and attempts < max_retries:
        try:
            candidate = gen_fn()
        except Exception:
            logger.warning("gen_fn raised an exception; skipping attempt")
            attempts += 1
            continue

        attempts += 1

        # Score against all constraints
        results = [c.check(candidate) for c in constraints]
        if not results:
            accepted.append(candidate)
            continue

        avg_score = sum(r.score for r in results) / len(results)
        pool.append((avg_score, candidate))

        if avg_score >= current_threshold:
            accepted.append(candidate)
        else:
            # Progressive relaxation every n attempts
            if attempts % max(1, max_retries // 5) == 0:
                current_threshold *= relaxation_rate
                logger.debug(
                    "Relaxed threshold to %.3f after %d attempts",
                    current_threshold, attempts,
                )

    # If we don't have enough accepted, fill from the pool by score
    if len(accepted) < n and pool:
        pool.sort(key=lambda x: x[0], reverse=True)
        seen = set(accepted)
        for score, text in pool:
            if text not in seen:
                accepted.append(text)
                seen.add(text)
            if len(accepted) >= n:
                break

    return accepted[:n]


def constraint_satisfaction_score(
    responses: List[str],
    constraints: List[Constraint],
) -> float:
    """Compute aggregate satisfaction score across responses and constraints.

    Returns the mean score over all (response, constraint) pairs.
    Batch-aware constraints (e.g., DiversityConstraint) are handled via
    ``batch_check``.
    """
    if not responses or not constraints:
        return 1.0

    total_score = 0.0
    count = 0

    for constraint in constraints:
        results = constraint.batch_check(responses)
        for r in results:
            total_score += r.score
            count += 1

    return total_score / count if count > 0 else 1.0


def find_constraint_violations(
    responses: List[str],
    constraints: List[Constraint],
) -> Dict[str, List[ConstraintResult]]:
    """Produce a detailed violation report per constraint.

    Returns a dict mapping constraint class name (with index for
    disambiguation) to a list of ``ConstraintResult`` objects, one per
    response.  Only violated results are included.
    """
    report: Dict[str, List[ConstraintResult]] = {}

    for idx, constraint in enumerate(constraints):
        key = f"{constraint.__class__.__name__}_{idx}"
        results = constraint.batch_check(responses)
        violations = [r for r in results if not r.satisfied]
        if violations:
            report[key] = violations

    return report


def optimize_for_constraints(
    candidates: List[str],
    constraints: List[Constraint],
    k: int,
) -> List[str]:
    """Select *k* candidates that best satisfy *constraints*.

    Uses greedy forward selection: at each step, the candidate that
    maximally improves the aggregate constraint score is added.  This
    naturally favours diverse selections when a
    ``DiversityConstraint`` is present.

    Parameters
    ----------
    candidates:
        Pool of candidate strings to choose from.
    constraints:
        Constraints to optimise for.
    k:
        Number of candidates to select.

    Returns
    -------
    List of *k* (or fewer if pool is small) selected candidates.
    """
    if k <= 0:
        return []
    if len(candidates) <= k:
        return list(candidates)

    # Pre-compute per-candidate individual scores (non-batch constraints)
    individual_scores: Dict[int, float] = {}
    for ci, cand in enumerate(candidates):
        scores: List[float] = []
        for constraint in constraints:
            r = constraint.check(cand)
            scores.append(r.score)
        individual_scores[ci] = float(np.mean(scores)) if scores else 1.0

    selected_indices: List[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(k):
        best_idx = -1
        best_score = -1.0

        for ci in remaining:
            trial = selected_indices + [ci]
            trial_texts = [candidates[i] for i in trial]

            # Evaluate batch-aware score
            batch_scores: List[float] = []
            for constraint in constraints:
                results = constraint.batch_check(trial_texts)
                batch_scores.extend(r.score for r in results)

            aggregate = float(np.mean(batch_scores)) if batch_scores else 0.0

            if aggregate > best_score:
                best_score = aggregate
                best_idx = ci

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected_indices]


def relax_constraints(
    constraints: List[Constraint],
    factor: float,
) -> List[Constraint]:
    """Create relaxed copies of constraints (wider bounds, lower thresholds).

    The *factor* controls relaxation strength: 0.0 = no change,
    1.0 = maximum relaxation.  Intermediate values interpolate.

    Parameters
    ----------
    constraints:
        Original constraints.
    factor:
        Relaxation factor in [0, 1].

    Returns
    -------
    New list of relaxed constraint objects (originals are not modified).
    """
    factor = max(0.0, min(1.0, factor))
    relaxed: List[Constraint] = []

    for c in constraints:
        rc = copy.deepcopy(c)

        if isinstance(rc, LengthConstraint):
            span = rc.max_len - rc.min_len
            expansion = int(span * factor * 0.5)
            rc.min_len = max(0, rc.min_len - expansion)
            rc.max_len = rc.max_len + expansion
            # Also soften the sigmoid
            rc.steepness = rc.steepness * (1.0 - factor * 0.5)

        elif isinstance(rc, TopicConstraint):
            rc.threshold = max(0.1, rc.threshold * (1.0 - factor * 0.6))

        elif isinstance(rc, FactualConstraint):
            rc.match_threshold = max(0.1, rc.match_threshold * (1.0 - factor * 0.5))

        elif isinstance(rc, StyleConstraint):
            rc.threshold = max(0.1, rc.threshold * (1.0 - factor * 0.5))

        elif isinstance(rc, DiversityConstraint):
            rc.min_diversity = max(0.0, rc.min_diversity * (1.0 - factor * 0.5))

        elif isinstance(rc, ToneConstraint):
            rc.threshold = max(0.05, rc.threshold * (1.0 - factor * 0.5))

        elif isinstance(rc, CompositeConstraint):
            rc.satisfaction_threshold = max(
                0.1, rc.satisfaction_threshold * (1.0 - factor * 0.4)
            )
            rc.constraints = relax_constraints(rc.constraints, factor)

        # ExclusionConstraint and unknown types: no numeric relaxation
        relaxed.append(rc)

    return relaxed
