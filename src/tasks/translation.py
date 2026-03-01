"""
Translation task domain for the Diversity Decoding Arena.

Implements machine translation with diversity across language pairs and
difficulty levels.  Includes from-scratch BLEU, chrF, and TER metrics,
adequacy / fluency estimators, terminology and register analysis, and a
curated prompt dataset of 40+ translation challenges spanning literal,
idiomatic, technical, literary, and ambiguous categories.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
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

from src.tasks.base import (
    GenerationTask,
    TaskConfig,
    TaskPrompt,
    TaskConstraint,
    PromptDataset,
    TaskEvaluator,
)
from src.types import TaskDomain


# ---------------------------------------------------------------------------
# Language pair
# ---------------------------------------------------------------------------

@dataclass
class LanguagePair:
    """An ordered source → target language pair."""

    source_lang: str
    target_lang: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.source_lang}-{self.target_lang}"

    def reversed(self) -> "LanguagePair":
        return LanguagePair(
            source_lang=self.target_lang,
            target_lang=self.source_lang,
        )

    def __str__(self) -> str:
        return self.name


# Pre-defined language pairs
EN_FR = LanguagePair("en", "fr", "English-French")
EN_DE = LanguagePair("en", "de", "English-German")
EN_ES = LanguagePair("en", "es", "English-Spanish")
EN_ZH = LanguagePair("en", "zh", "English-Chinese")
EN_JA = LanguagePair("en", "ja", "English-Japanese")
FR_EN = EN_FR.reversed()
DE_EN = EN_DE.reversed()
ES_EN = EN_ES.reversed()


# ---------------------------------------------------------------------------
# Difficulty enum
# ---------------------------------------------------------------------------

class TranslationDifficulty(Enum):
    """Difficulty tier for a translation prompt."""

    LITERAL = auto()
    IDIOMATIC = auto()
    TECHNICAL = auto()
    LITERARY = auto()
    AMBIGUOUS = auto()

    def __repr__(self) -> str:
        return f"TranslationDifficulty.{self.name}"


# ---------------------------------------------------------------------------
# Config & Prompt dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranslationConfig(TaskConfig):
    """Configuration specific to the translation task domain."""

    language_pair: LanguagePair = field(default_factory=lambda: EN_FR)
    difficulty: TranslationDifficulty = TranslationDifficulty.LITERAL
    preserve_register: bool = True
    formality_level: str = "neutral"
    domain_specific_terms: Dict[str, str] = field(default_factory=dict)


@dataclass
class TranslationPrompt(TaskPrompt):
    """A single translation prompt with source text and metadata."""

    source_text: str = ""
    source_lang: str = "en"
    target_lang: str = "fr"
    reference_translations: List[str] = field(default_factory=list)
    difficulty: TranslationDifficulty = TranslationDifficulty.LITERAL
    domain: str = "general"
    glossary: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS_EN: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their",
}

_FORMAL_MARKERS = {
    "fr": ["vous", "veuillez", "monsieur", "madame", "cordialement"],
    "de": ["Sie", "Ihnen", "bitte", "geehrte", "freundlichen"],
    "es": ["usted", "ustedes", "señor", "señora", "atentamente"],
}

_INFORMAL_MARKERS = {
    "fr": ["tu", "toi", "salut", "bisous", "mec"],
    "de": ["du", "dir", "dich", "hey", "tschüss"],
    "es": ["tú", "vos", "oye", "tío", "colega"],
}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    text = text.lower().strip()
    text = re.sub(r"([.!?,;:\"'])", r" \1 ", text)
    return [t for t in text.split() if t]


def _char_ngrams(text: str, n: int) -> List[str]:
    """Extract character n-grams from *text*."""
    text = text.strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _word_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract word n-grams from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein distance between two strings."""
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s[i - 1] == t[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _edit_distance_tokens(src: List[str], tgt: List[str]) -> int:
    """Token-level edit distance (for TER)."""
    m, n = len(src), len(tgt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if src[i - 1] == tgt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _content_words(tokens: List[str]) -> List[str]:
    """Filter out stop words, returning content words only."""
    return [t for t in tokens if t not in _STOP_WORDS_EN]


# ---------------------------------------------------------------------------
# TranslationTask
# ---------------------------------------------------------------------------

class TranslationTask(GenerationTask):
    """Machine translation task with diversity-aware evaluation.

    Provides 40+ curated prompts across English-French, English-German,
    English-Spanish, and ambiguous/multi-interpretation pairs.  Evaluation
    uses from-scratch BLEU, chrF, and TER together with adequacy, fluency,
    terminology, register, and diversity metrics.
    """

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        super().__init__(config or self.get_default_config())

    # -----------------------------------------------------------------
    # Default config
    # -----------------------------------------------------------------

    @classmethod
    def get_default_config(cls) -> TranslationConfig:
        return TranslationConfig(
            name="translation",
            domain=TaskDomain.TRANSLATION,
            num_prompts=50,
            max_length=256,
            min_length=1,
            temperature=0.7,
            evaluation_metrics=[
                "bleu", "chrf", "ter", "adequacy", "fluency",
                "diversity", "terminology",
            ],
            language_pair=EN_FR,
            difficulty=TranslationDifficulty.LITERAL,
            preserve_register=True,
            formality_level="neutral",
        )

    # =================================================================
    # Abstract interface implementation
    # =================================================================

    def load_prompts(self) -> PromptDataset:
        """Return a :class:`PromptDataset` with 40+ translation prompts."""
        prompts: List[TaskPrompt] = []
        prompts.extend(self._generate_en_fr_prompts())
        prompts.extend(self._generate_en_de_prompts())
        prompts.extend(self._generate_en_es_prompts())
        prompts.extend(self._generate_ambiguous_prompts())
        self._dataset = PromptDataset(prompts)
        return self._dataset

    def format_prompt(self, prompt: TaskPrompt) -> str:
        """Format a translation prompt for model consumption."""
        if not isinstance(prompt, TranslationPrompt):
            return self.config.prompt_template.format(text=prompt.text)

        tp: TranslationPrompt = prompt
        parts: List[str] = []

        parts.append(
            f"Translate the following text from {tp.source_lang.upper()} "
            f"to {tp.target_lang.upper()}."
        )

        if tp.difficulty == TranslationDifficulty.LITERARY:
            parts.append(
                "Preserve the literary style, tone, and rhetorical devices."
            )
        elif tp.difficulty == TranslationDifficulty.TECHNICAL:
            parts.append(
                "Use precise technical terminology appropriate for the domain."
            )
        elif tp.difficulty == TranslationDifficulty.IDIOMATIC:
            parts.append(
                "Translate idioms and expressions naturally in the target language; "
                "do not translate literally."
            )
        elif tp.difficulty == TranslationDifficulty.AMBIGUOUS:
            parts.append(
                "The source text is ambiguous. Provide a plausible translation "
                "and note any alternative interpretations."
            )

        if tp.glossary:
            glossary_lines = [f"  {k} → {v}" for k, v in tp.glossary.items()]
            parts.append("Glossary (use these terms):")
            parts.extend(glossary_lines)

        if tp.notes:
            parts.append(f"Note: {tp.notes}")

        cfg = self.config
        if isinstance(cfg, TranslationConfig):
            if cfg.formality_level != "neutral":
                parts.append(f"Formality level: {cfg.formality_level}.")
            if cfg.preserve_register:
                parts.append("Preserve the register of the source text.")

        parts.append("")
        parts.append(f"Source ({tp.source_lang.upper()}):")
        parts.append(tp.source_text)
        parts.append("")
        parts.append(f"Translation ({tp.target_lang.upper()}):")

        return "\n".join(parts)

    def evaluate(
        self,
        generations: List[str],
        prompts: List[TaskPrompt],
    ) -> Dict[str, float]:
        """Evaluate a batch of translations on quality and diversity metrics."""
        if not generations:
            return {}

        bleu_scores: List[float] = []
        chrf_scores: List[float] = []
        ter_scores: List[float] = []
        adequacy_scores: List[float] = []
        fluency_scores: List[float] = []
        length_ratios: List[float] = []
        term_scores: List[float] = []
        register_scores: List[float] = []
        coverage_scores: List[float] = []
        ne_scores: List[float] = []

        for gen, prompt in zip(generations, prompts):
            gen_clean = self.post_process(gen)
            tp = prompt if isinstance(prompt, TranslationPrompt) else None

            refs = (
                tp.reference_translations if tp and tp.reference_translations
                else prompt.reference_outputs
            )

            if refs:
                bleu_scores.append(self._bleu_score(gen_clean, refs))
                chrf_scores.append(self._chrf_score(gen_clean, refs))
                ter_scores.append(self._ter_score(gen_clean, refs[0]))
            else:
                bleu_scores.append(0.0)
                chrf_scores.append(0.0)
                ter_scores.append(1.0)

            src = tp.source_text if tp else prompt.text
            adequacy_scores.append(self._adequacy_score(gen_clean, src))
            fluency_scores.append(self._fluency_score(gen_clean))
            length_ratios.append(self._length_ratio(src, gen_clean))

            if tp and tp.glossary:
                term_scores.append(
                    self._terminology_accuracy(gen_clean, tp.glossary)
                )
            else:
                term_scores.append(1.0)

            cfg = self.config
            target_reg = "neutral"
            if isinstance(cfg, TranslationConfig):
                target_reg = cfg.formality_level
            register_scores.append(
                self._register_consistency(gen_clean, target_reg)
            )

            source_concepts = set(_content_words(_tokenize(src)))
            coverage_scores.append(
                self._coverage_score(gen_clean, source_concepts)
            )
            ne_scores.append(self._named_entity_preservation(src, gen_clean))

        diversity = self._diversity_of_translations(generations)

        results: Dict[str, float] = {
            "bleu": float(np.mean(bleu_scores)),
            "chrf": float(np.mean(chrf_scores)),
            "ter": float(np.mean(ter_scores)),
            "adequacy": float(np.mean(adequacy_scores)),
            "fluency": float(np.mean(fluency_scores)),
            "length_ratio": float(np.mean(length_ratios)),
            "terminology_accuracy": float(np.mean(term_scores)),
            "register_consistency": float(np.mean(register_scores)),
            "coverage": float(np.mean(coverage_scores)),
            "named_entity_preservation": float(np.mean(ne_scores)),
            "diversity": diversity,
        }

        if len(generations) >= 2:
            pair_qualities: List[float] = []
            for a, b in combinations(generations, 2):
                pair_qualities.append(
                    self._paraphrase_quality(
                        self.post_process(a), self.post_process(b)
                    )
                )
            results["paraphrase_quality"] = float(np.mean(pair_qualities))

        return results

    def get_constraints(self) -> List[TaskConstraint]:
        """Return translation-specific constraints."""
        from src.tasks.base import ConstraintType

        constraints: List[TaskConstraint] = []

        # Output must not be empty
        constraints.append(
            TaskConstraint(
                constraint_type=ConstraintType.LENGTH,
                parameters={"min": 1, "unit": "words"},
                required=True,
                weight=1.0,
            )
        )

        # Repetition guard
        constraints.append(
            TaskConstraint(
                constraint_type=ConstraintType.CONTENT,
                parameters={
                    "min_unique_words": 1,
                    "max_repetition_ratio": 0.5,
                },
                required=False,
                weight=0.8,
            )
        )

        return constraints

    # =================================================================
    # Translation metrics (implemented from scratch)
    # =================================================================

    # ----- BLEU (from scratch) -----

    def _n_gram_precision(
        self, hypothesis: str, reference: str, n: int
    ) -> float:
        """Compute clipped n-gram precision for a single reference."""
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        hyp_ngrams = _word_ngrams(hyp_tokens, n)
        ref_ngrams = _word_ngrams(ref_tokens, n)

        if not hyp_ngrams:
            return 0.0

        ref_counts: Counter = Counter(ref_ngrams)
        hyp_counts: Counter = Counter(hyp_ngrams)

        clipped = 0
        for ng, count in hyp_counts.items():
            clipped += min(count, ref_counts.get(ng, 0))

        return clipped / len(hyp_ngrams)

    def _brevity_penalty(
        self, hypothesis_len: int, reference_len: int
    ) -> float:
        """Compute the BLEU brevity penalty."""
        if hypothesis_len == 0:
            return 0.0
        if hypothesis_len >= reference_len:
            return 1.0
        return math.exp(1.0 - reference_len / hypothesis_len)

    def _bleu_score(
        self, hypothesis: str, references: List[str], max_n: int = 4
    ) -> float:
        """Compute corpus-level BLEU score from scratch (single segment).

        Uses uniform weights across 1..max_n n-grams and a brevity penalty
        computed against the closest reference length.
        """
        hyp_tokens = _tokenize(hypothesis)
        hyp_len = len(hyp_tokens)

        if hyp_len == 0:
            return 0.0

        # Closest reference length for brevity penalty
        ref_lens = [len(_tokenize(r)) for r in references]
        closest_ref_len = min(ref_lens, key=lambda rl: abs(rl - hyp_len))

        bp = self._brevity_penalty(hyp_len, closest_ref_len)

        log_avg = 0.0
        weight = 1.0 / max_n
        all_positive = True

        for n in range(1, max_n + 1):
            # Take maximum precision across references
            precisions = [
                self._n_gram_precision(hypothesis, ref, n) for ref in references
            ]
            p_n = max(precisions) if precisions else 0.0

            if p_n == 0:
                all_positive = False
                break
            log_avg += weight * math.log(p_n)

        if not all_positive:
            return 0.0

        return bp * math.exp(log_avg)

    # ----- chrF (character n-gram F-score) -----

    def _chrf_score(
        self,
        hypothesis: str,
        references: List[str],
        char_order: int = 6,
        beta: float = 2.0,
    ) -> float:
        """Compute chrF score (character n-gram F-score).

        Parameters
        ----------
        hypothesis : str
            The candidate translation.
        references : List[str]
            One or more reference translations.
        char_order : int
            Maximum character n-gram order.
        beta : float
            F-score weighting (beta > 1 favours recall).
        """
        best = 0.0
        for ref in references:
            score = self._chrf_single(hypothesis, ref, char_order, beta)
            best = max(best, score)
        return best

    def _chrf_single(
        self,
        hypothesis: str,
        reference: str,
        char_order: int = 6,
        beta: float = 2.0,
    ) -> float:
        """chrF against a single reference."""
        total_prec = 0.0
        total_rec = 0.0
        count = 0

        for n in range(1, char_order + 1):
            hyp_ngrams = Counter(_char_ngrams(hypothesis, n))
            ref_ngrams = Counter(_char_ngrams(reference, n))

            hyp_total = sum(hyp_ngrams.values())
            ref_total = sum(ref_ngrams.values())

            if hyp_total == 0 or ref_total == 0:
                continue

            common = 0
            for ng, cnt in hyp_ngrams.items():
                common += min(cnt, ref_ngrams.get(ng, 0))

            prec = common / hyp_total
            rec = common / ref_total
            total_prec += prec
            total_rec += rec
            count += 1

        if count == 0:
            return 0.0

        avg_prec = total_prec / count
        avg_rec = total_rec / count

        if avg_prec + avg_rec == 0:
            return 0.0

        beta_sq = beta * beta
        f_score = (
            (1 + beta_sq) * avg_prec * avg_rec
            / (beta_sq * avg_prec + avg_rec)
        )
        return f_score

    # ----- TER (Translation Edit Rate) -----

    def _ter_score(self, hypothesis: str, reference: str) -> float:
        """Compute Translation Edit Rate (lower is better).

        TER = (# edits) / (# reference tokens).
        """
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        if not ref_tokens:
            return 0.0 if not hyp_tokens else 1.0

        edits = _edit_distance_tokens(hyp_tokens, ref_tokens)
        return edits / len(ref_tokens)

    # =================================================================
    # Quality / adequacy / fluency estimators
    # =================================================================

    def _length_ratio(self, source: str, translation: str) -> float:
        """Ratio of translation length to source length (in tokens)."""
        src_len = max(len(_tokenize(source)), 1)
        tgt_len = len(_tokenize(translation))
        return tgt_len / src_len

    def _adequacy_score(self, translation: str, source: str) -> float:
        """Estimate meaning preservation via content-word overlap.

        This is a lightweight proxy: it extracts content words from source
        and translation, computes Jaccard similarity, and adjusts for
        length ratio.
        """
        src_tokens = _tokenize(source)
        tgt_tokens = _tokenize(translation)

        src_content = set(_content_words(src_tokens))
        tgt_content = set(tgt_tokens)

        if not src_content:
            return 1.0

        # Direct overlap (works for cognates / borrowed words)
        overlap = _jaccard(src_content, tgt_content)

        # Length ratio penalty
        ratio = self._length_ratio(source, translation)
        ratio_penalty = 1.0 - min(abs(1.0 - ratio), 1.0) * 0.3

        # Character-level similarity for cognate detection
        cognate_bonus = self._cognate_similarity(
            list(src_content), list(tgt_content)
        )

        score = 0.4 * overlap + 0.3 * ratio_penalty + 0.3 * cognate_bonus
        return min(max(score, 0.0), 1.0)

    def _cognate_similarity(
        self, src_words: List[str], tgt_words: List[str]
    ) -> float:
        """Detect cognates via normalised character edit distance."""
        if not src_words or not tgt_words:
            return 0.0

        matches = 0
        for sw in src_words:
            best_sim = 0.0
            for tw in tgt_words:
                max_len = max(len(sw), len(tw), 1)
                sim = 1.0 - _levenshtein(sw, tw) / max_len
                best_sim = max(best_sim, sim)
            if best_sim >= 0.6:
                matches += 1

        return matches / len(src_words)

    def _fluency_score(self, translation: str) -> float:
        """Estimate fluency from surface-level features."""
        if not translation.strip():
            return 0.0

        tokens = _tokenize(translation)
        if not tokens:
            return 0.0

        score = 1.0

        # Penalise very short translations
        if len(tokens) < 3:
            score *= 0.6

        # Repeated token penalty
        counts = Counter(tokens)
        if counts:
            most_common_ratio = counts.most_common(1)[0][1] / len(tokens)
            if most_common_ratio > 0.3:
                score *= 0.7

        # Punctuation balance
        open_parens = translation.count("(")
        close_parens = translation.count(")")
        if open_parens != close_parens:
            score *= 0.9

        open_quotes = translation.count('"')
        if open_quotes % 2 != 0:
            score *= 0.9

        # Sentence ending
        stripped = translation.strip()
        if stripped and stripped[-1] not in ".!?…»\"')":
            score *= 0.85

        # Capitalisation check (first char should be upper or special)
        if stripped and stripped[0].isalpha() and not stripped[0].isupper():
            score *= 0.9

        # Type-token ratio as a proxy for naturalness
        ttr = len(set(tokens)) / len(tokens)
        if ttr < 0.3:
            score *= 0.8

        return min(max(score, 0.0), 1.0)

    def _formality_consistency(
        self, translation: str, target_formality: str
    ) -> float:
        """Check whether the translation matches the desired formality level."""
        text_lower = translation.lower()
        cfg = self.config
        lang = "fr"
        if isinstance(cfg, TranslationConfig):
            lang = cfg.language_pair.target_lang

        formal = _FORMAL_MARKERS.get(lang, [])
        informal = _INFORMAL_MARKERS.get(lang, [])

        formal_count = sum(1 for m in formal if m.lower() in text_lower)
        informal_count = sum(1 for m in informal if m.lower() in text_lower)
        total = formal_count + informal_count

        if total == 0:
            return 0.8  # neutral by default

        if target_formality == "formal":
            return formal_count / total
        elif target_formality == "informal":
            return informal_count / total
        else:
            return 0.8  # neutral — small penalty if strongly one-sided

    def _terminology_accuracy(
        self, translation: str, glossary: Dict[str, str]
    ) -> float:
        """Fraction of glossary terms correctly used in the translation."""
        if not glossary:
            return 1.0

        text_lower = translation.lower()
        matches = 0
        for _src_term, tgt_term in glossary.items():
            if tgt_term.lower() in text_lower:
                matches += 1

        return matches / len(glossary)

    # =================================================================
    # Diversity metrics
    # =================================================================

    def _diversity_of_translations(
        self, translations: List[str]
    ) -> float:
        """Lexical diversity across a set of translations.

        Combines type-token ratio variance, pairwise Jaccard distance,
        and n-gram novelty.
        """
        if len(translations) < 2:
            return 0.0

        tokenised = [set(_tokenize(t)) for t in translations]

        # Pairwise Jaccard distance
        pairwise_dist: List[float] = []
        for i, j in combinations(range(len(tokenised)), 2):
            pairwise_dist.append(1.0 - _jaccard(tokenised[i], tokenised[j]))

        avg_dist = float(np.mean(pairwise_dist)) if pairwise_dist else 0.0

        # Unique bigrams across all translations
        all_bigrams: Set[Tuple[str, ...]] = set()
        per_trans_bigrams: List[Set[Tuple[str, ...]]] = []
        for t in translations:
            toks = _tokenize(t)
            bgs = set(_word_ngrams(toks, 2))
            per_trans_bigrams.append(bgs)
            all_bigrams.update(bgs)

        if all_bigrams:
            bigram_counts = [len(bg) for bg in per_trans_bigrams]
            total_unique = len(all_bigrams)
            novelty = 1.0 - (np.mean(bigram_counts) / max(total_unique, 1))
        else:
            novelty = 0.0

        # Surface form variance (normalised std of lengths)
        lengths = [len(_tokenize(t)) for t in translations]
        mean_len = np.mean(lengths)
        if mean_len > 0:
            len_var = float(np.std(lengths) / mean_len)
        else:
            len_var = 0.0

        diversity = 0.5 * avg_dist + 0.3 * float(novelty) + 0.2 * min(len_var, 1.0)
        return min(max(diversity, 0.0), 1.0)

    def _paraphrase_quality(self, trans_a: str, trans_b: str) -> float:
        """Estimate whether two translations convey the same meaning differently.

        High score = same meaning (high content overlap) + different surface
        form (low exact token overlap).
        """
        tokens_a = _tokenize(trans_a)
        tokens_b = _tokenize(trans_b)

        set_a = set(tokens_a)
        set_b = set(tokens_b)

        # Content overlap via Jaccard
        content_a = set(_content_words(tokens_a))
        content_b = set(_content_words(tokens_b))
        content_sim = _jaccard(content_a, content_b)

        # Surface-form distance (we want this to be high)
        surface_sim = _jaccard(set_a, set_b)
        surface_dist = 1.0 - surface_sim

        # Character-level similarity (cognate detection)
        char_sim = 0.0
        if trans_a and trans_b:
            max_len = max(len(trans_a), len(trans_b), 1)
            char_sim = 1.0 - _levenshtein(trans_a[:200], trans_b[:200]) / max_len

        # Good paraphrase = high content overlap + high surface distance
        quality = 0.4 * content_sim + 0.4 * surface_dist + 0.2 * (1.0 - char_sim)
        return min(max(quality, 0.0), 1.0)

    def _register_consistency(
        self, translation: str, target_register: str
    ) -> float:
        """Evaluate whether the translation preserves the target register."""
        return self._formality_consistency(translation, target_register)

    # =================================================================
    # Structural analysis
    # =================================================================

    def _word_order_analysis(
        self, source: str, translation: str
    ) -> Dict[str, Any]:
        """Analyse word-order divergence between source and translation.

        Returns a dictionary with monotonicity ratio, crossing count,
        and distortion score.
        """
        src_tokens = _tokenize(source)
        tgt_tokens = _tokenize(translation)

        if not src_tokens or not tgt_tokens:
            return {
                "monotonicity": 1.0,
                "crossings": 0,
                "distortion": 0.0,
                "alignment_count": 0,
            }

        # Build greedy alignment based on exact / cognate matches
        aligned: List[Tuple[int, int]] = []
        used_tgt: Set[int] = set()

        for i, sw in enumerate(src_tokens):
            best_j = -1
            best_sim = 0.0
            for j, tw in enumerate(tgt_tokens):
                if j in used_tgt:
                    continue
                if sw == tw:
                    best_j = j
                    best_sim = 1.0
                    break
                max_len = max(len(sw), len(tw), 1)
                sim = 1.0 - _levenshtein(sw, tw) / max_len
                if sim > best_sim and sim >= 0.6:
                    best_sim = sim
                    best_j = j

            if best_j >= 0:
                aligned.append((i, best_j))
                used_tgt.add(best_j)

        if len(aligned) < 2:
            return {
                "monotonicity": 1.0,
                "crossings": 0,
                "distortion": 0.0,
                "alignment_count": len(aligned),
            }

        # Count crossings (inversions)
        crossings = 0
        for idx in range(len(aligned) - 1):
            for jdx in range(idx + 1, len(aligned)):
                if aligned[idx][1] > aligned[jdx][1]:
                    crossings += 1

        max_crossings = len(aligned) * (len(aligned) - 1) / 2
        monotonicity = 1.0 - crossings / max(max_crossings, 1)

        # Average absolute distortion
        distortions = [
            abs(a[1] - a[0]) for a in aligned
        ]
        avg_distortion = float(np.mean(distortions))

        return {
            "monotonicity": monotonicity,
            "crossings": crossings,
            "distortion": avg_distortion,
            "alignment_count": len(aligned),
        }

    def _coverage_score(
        self, translation: str, source_concepts: Set[str]
    ) -> float:
        """Fraction of source concepts covered in the translation.

        Uses cognate / fuzzy matching since source and target languages
        differ.
        """
        if not source_concepts:
            return 1.0

        tgt_tokens = _tokenize(translation)
        if not tgt_tokens:
            return 0.0

        covered = 0
        for concept in source_concepts:
            # Exact match
            if concept in tgt_tokens:
                covered += 1
                continue
            # Fuzzy cognate match
            for tw in tgt_tokens:
                max_len = max(len(concept), len(tw), 1)
                if 1.0 - _levenshtein(concept, tw) / max_len >= 0.6:
                    covered += 1
                    break

        return covered / len(source_concepts)

    def _back_translation_similarity(
        self, translation: str, source: str
    ) -> float:
        """Estimate back-translation quality via surface similarity.

        Without an actual MT model, we approximate by measuring character
        and token overlap between source and translation (works partly for
        related language pairs with many cognates).
        """
        src_set = set(_tokenize(source))
        tgt_set = set(_tokenize(translation))
        token_overlap = _jaccard(src_set, tgt_set)

        # Character trigram overlap
        src_trigrams = set(_char_ngrams(source.lower(), 3))
        tgt_trigrams = set(_char_ngrams(translation.lower(), 3))
        char_overlap = _jaccard(src_trigrams, tgt_trigrams)

        return 0.5 * token_overlap + 0.5 * char_overlap

    def _idiom_handling_score(
        self, translation: str, source_idioms: List[str]
    ) -> float:
        """Score how well source idioms are handled.

        An idiom should NOT appear verbatim (literal translation) in the
        target; instead it should be conveyed by a natural equivalent.
        Score is higher when the literal form is absent but the translation
        is long enough to plausibly contain the meaning.
        """
        if not source_idioms:
            return 1.0

        tgt_lower = translation.lower()
        tgt_len = len(_tokenize(translation))
        scores: List[float] = []

        for idiom in source_idioms:
            idiom_lower = idiom.lower()
            # If idiom appears verbatim, that's a literal (bad) translation
            if idiom_lower in tgt_lower:
                scores.append(0.2)
            else:
                # Reward length proportional to idiom length (suggests expansion)
                idiom_words = len(idiom.split())
                expected_min = max(idiom_words - 1, 1)
                if tgt_len >= expected_min:
                    scores.append(0.9)
                else:
                    scores.append(0.5)

        return float(np.mean(scores))

    def _named_entity_preservation(
        self, source: str, translation: str
    ) -> float:
        """Measure preservation of named entities across translation.

        Uses simple heuristics: capitalised words in the source that are
        not sentence-initial are treated as named entities.
        """
        src_tokens = source.split()
        entities: Set[str] = set()

        for i, tok in enumerate(src_tokens):
            clean = tok.strip(string.punctuation)
            if not clean:
                continue
            if clean[0].isupper() and i > 0:
                entities.add(clean.lower())
            # Also catch all-caps acronyms
            if len(clean) >= 2 and clean.isupper():
                entities.add(clean.lower())

        if not entities:
            return 1.0

        tgt_lower = translation.lower()
        preserved = sum(1 for e in entities if e in tgt_lower)
        return preserved / len(entities)

    def _sentence_alignment(
        self, source: str, translation: str
    ) -> List[Tuple[str, str]]:
        """Align source and translation sentences by position.

        Falls back to sequential alignment when sentence counts differ.
        """
        src_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if s.strip()]
        tgt_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", translation) if s.strip()]

        if not src_sents:
            src_sents = [source.strip()]
        if not tgt_sents:
            tgt_sents = [translation.strip()]

        aligned: List[Tuple[str, str]] = []
        max_len = max(len(src_sents), len(tgt_sents))

        for i in range(max_len):
            src = src_sents[i] if i < len(src_sents) else ""
            tgt = tgt_sents[i] if i < len(tgt_sents) else ""
            aligned.append((src, tgt))

        return aligned

    # =================================================================
    # Post-processing
    # =================================================================

    def post_process(self, text: str) -> str:
        """Clean a raw translation output."""
        text = text.strip()

        # Remove common prefixes models add
        prefixes = [
            "Translation:", "Here is the translation:",
            "Here's the translation:", "Translated text:",
            "Output:", "Result:",
        ]
        for pfx in prefixes:
            if text.lower().startswith(pfx.lower()):
                text = text[len(pfx):].strip()

        # Remove surrounding quotes if present
        if len(text) >= 2:
            if (text[0] == '"' and text[-1] == '"') or \
               (text[0] == "'" and text[-1] == "'") or \
               (text[0] == "«" and text[-1] == "»"):
                text = text[1:-1].strip()

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix spacing before French punctuation
        text = re.sub(r"\s+([;:!?»])", r" \1", text)
        text = re.sub(r"([«])\s+", r"\1 ", text)

        return text.strip()

    # =================================================================
    # Prompt generators
    # =================================================================

    def _generate_en_fr_prompts(self) -> List[TranslationPrompt]:
        """Generate English → French translation prompts."""
        prompts: List[TranslationPrompt] = []

        # --- LITERAL ---
        prompts.append(TranslationPrompt(
            prompt_id="enfr-lit-01",
            text="Translate the following sentence to French.",
            source_text="The weather is beautiful today, and the children are playing in the park.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Le temps est magnifique aujourd'hui, et les enfants jouent dans le parc.",
                "Il fait beau aujourd'hui et les enfants jouent au parc.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-lit-02",
            text="Translate the following sentence to French.",
            source_text="She opened the window to let in some fresh air.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Elle a ouvert la fenêtre pour laisser entrer de l'air frais.",
                "Elle ouvrit la fenêtre pour faire entrer un peu d'air frais.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-lit-03",
            text="Translate to French.",
            source_text="The train arrives at the station every morning at eight o'clock.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Le train arrive à la gare chaque matin à huit heures.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        # --- IDIOMATIC ---
        prompts.append(TranslationPrompt(
            prompt_id="enfr-idm-01",
            text="Translate the idiom naturally into French.",
            source_text="It's raining cats and dogs, so we'd better stay inside and not beat around the bush.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Il pleut des cordes, alors on ferait mieux de rester à l'intérieur et de ne pas tourner autour du pot.",
                "Il tombe des hallebardes, alors restons à l'intérieur et ne tournons pas autour du pot.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
            notes="Contains two idioms: 'raining cats and dogs' and 'beat around the bush'.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-idm-02",
            text="Translate idiomatically to French.",
            source_text="He let the cat out of the bag about the surprise party.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Il a vendu la mèche au sujet de la fête surprise.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
            notes="'Let the cat out of the bag' = reveal a secret.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-idm-03",
            text="Translate idiomatically to French.",
            source_text="She was on cloud nine after receiving the good news.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Elle était aux anges après avoir reçu la bonne nouvelle.",
                "Elle était au septième ciel après avoir reçu la bonne nouvelle.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
        ))

        # --- TECHNICAL ---
        prompts.append(TranslationPrompt(
            prompt_id="enfr-tec-01",
            text="Translate the following technical text to French.",
            source_text=(
                "The neural network architecture employs a multi-head "
                "self-attention mechanism with residual connections and "
                "layer normalisation to achieve state-of-the-art performance "
                "on sequence-to-sequence tasks."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "L'architecture du réseau de neurones utilise un mécanisme "
                    "d'auto-attention multi-tête avec des connexions résiduelles "
                    "et une normalisation de couche pour atteindre des performances "
                    "de pointe sur les tâches de séquence à séquence."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="machine_learning",
            glossary={
                "neural network": "réseau de neurones",
                "self-attention": "auto-attention",
                "residual connections": "connexions résiduelles",
                "layer normalisation": "normalisation de couche",
            },
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-tec-02",
            text="Translate the medical text to French.",
            source_text=(
                "The patient presented with acute myocardial infarction and was "
                "immediately administered antiplatelet therapy along with "
                "percutaneous coronary intervention."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Le patient a présenté un infarctus aigu du myocarde et a "
                    "immédiatement reçu un traitement antiplaquettaire ainsi "
                    "qu'une intervention coronarienne percutanée."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="medical",
            glossary={
                "myocardial infarction": "infarctus du myocarde",
                "antiplatelet therapy": "traitement antiplaquettaire",
                "percutaneous coronary intervention": "intervention coronarienne percutanée",
            },
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-tec-03",
            text="Translate the legal text to French.",
            source_text=(
                "The undersigned parties hereby agree to indemnify and hold "
                "harmless the licensor against any claims arising from the "
                "licensee's use of the intellectual property."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Les parties soussignées conviennent par la présente d'indemniser "
                    "et de dégager de toute responsabilité le concédant de licence "
                    "contre toute réclamation découlant de l'utilisation de la "
                    "propriété intellectuelle par le preneur de licence."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="legal",
            glossary={
                "indemnify": "indemniser",
                "hold harmless": "dégager de toute responsabilité",
                "licensor": "concédant de licence",
                "licensee": "preneur de licence",
                "intellectual property": "propriété intellectuelle",
            },
        ))

        # --- LITERARY ---
        prompts.append(TranslationPrompt(
            prompt_id="enfr-lit-style-01",
            text="Translate preserving literary style.",
            source_text=(
                "The fog crept in on little cat feet, settling over the harbour "
                "and the city, muffling every sound until the world held its breath."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Le brouillard s'est glissé à pas de chat, se posant sur le port "
                    "et la ville, étouffant chaque son jusqu'à ce que le monde retienne "
                    "son souffle."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
            notes="Allusion to Carl Sandburg's 'Fog'. Preserve the personification.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enfr-lit-style-02",
            text="Translate preserving literary style.",
            source_text=(
                "In the twilight of the old world, empires crumbled like sandcastles "
                "before the tide, and from their dust rose the uncertain dawn of "
                "something altogether new."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Au crépuscule de l'ancien monde, les empires s'effondraient comme "
                    "des châteaux de sable devant la marée, et de leur poussière "
                    "s'élevait l'aube incertaine de quelque chose d'entièrement nouveau."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
        ))

        return prompts

    def _generate_en_de_prompts(self) -> List[TranslationPrompt]:
        """Generate English → German translation prompts."""
        prompts: List[TranslationPrompt] = []

        # --- LITERAL ---
        prompts.append(TranslationPrompt(
            prompt_id="ende-lit-01",
            text="Translate to German.",
            source_text="The new library was inaugurated last week in the centre of town.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Die neue Bibliothek wurde letzte Woche im Stadtzentrum eingeweiht.",
                "Letzte Woche wurde die neue Bibliothek im Zentrum der Stadt eröffnet.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="ende-lit-02",
            text="Translate to German.",
            source_text="We need to buy bread, milk, and eggs before the store closes.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Wir müssen Brot, Milch und Eier kaufen, bevor der Laden schließt.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="ende-lit-03",
            text="Translate to German.",
            source_text="The meeting has been postponed until next Thursday.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Die Besprechung wurde auf nächsten Donnerstag verschoben.",
                "Das Meeting wurde auf nächsten Donnerstag verschoben.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        # --- IDIOMATIC ---
        prompts.append(TranslationPrompt(
            prompt_id="ende-idm-01",
            text="Translate the idiom naturally into German.",
            source_text="Don't cry over spilt milk; what's done is done.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Wein nicht über verschüttete Milch; was geschehen ist, ist geschehen.",
                "Es hat keinen Sinn, über vergossene Milch zu weinen; was passiert ist, ist passiert.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="ende-idm-02",
            text="Translate idiomatically to German.",
            source_text="He's barking up the wrong tree if he thinks I'll agree.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Er ist auf dem Holzweg, wenn er denkt, dass ich zustimme.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
            notes="'Barking up the wrong tree' = auf dem Holzweg sein.",
        ))

        # --- TECHNICAL ---
        prompts.append(TranslationPrompt(
            prompt_id="ende-tec-01",
            text="Translate the engineering text to German.",
            source_text=(
                "The load-bearing capacity of the reinforced concrete beam was "
                "calculated using finite element analysis, taking into account "
                "both dead loads and live loads as specified in the Eurocode."
            ),
            source_lang="en", target_lang="de",
            reference_translations=[
                (
                    "Die Tragfähigkeit des Stahlbetonbalkens wurde mithilfe der "
                    "Finite-Elemente-Analyse berechnet, wobei sowohl Eigenlasten "
                    "als auch Verkehrslasten gemäß dem Eurocode berücksichtigt wurden."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="engineering",
            glossary={
                "load-bearing capacity": "Tragfähigkeit",
                "reinforced concrete": "Stahlbeton",
                "finite element analysis": "Finite-Elemente-Analyse",
                "dead loads": "Eigenlasten",
                "live loads": "Verkehrslasten",
            },
        ))

        prompts.append(TranslationPrompt(
            prompt_id="ende-tec-02",
            text="Translate the computer science text to German.",
            source_text=(
                "Garbage collection in managed runtimes reclaims heap-allocated "
                "memory that is no longer reachable from the root set, using "
                "either mark-and-sweep or generational algorithms."
            ),
            source_lang="en", target_lang="de",
            reference_translations=[
                (
                    "Die Garbage Collection in verwalteten Laufzeitumgebungen gibt "
                    "heap-allokierten Speicher frei, der vom Wurzelset aus nicht mehr "
                    "erreichbar ist, und verwendet dafür entweder Mark-and-Sweep- "
                    "oder generationelle Algorithmen."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="computer_science",
            glossary={
                "garbage collection": "Garbage Collection",
                "heap-allocated": "heap-allokiert",
                "root set": "Wurzelset",
                "mark-and-sweep": "Mark-and-Sweep",
            },
        ))

        # --- LITERARY ---
        prompts.append(TranslationPrompt(
            prompt_id="ende-lit-style-01",
            text="Translate preserving literary style.",
            source_text=(
                "Loneliness hung about him like a cloak, heavy and grey, "
                "and with every step he took the silence grew louder, "
                "pressing against his ears like water against a diver's mask."
            ),
            source_lang="en", target_lang="de",
            reference_translations=[
                (
                    "Einsamkeit hing um ihn wie ein Mantel, schwer und grau, "
                    "und mit jedem Schritt wurde die Stille lauter und drückte "
                    "gegen seine Ohren wie Wasser gegen die Maske eines Tauchers."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="ende-lit-style-02",
            text="Translate preserving literary style.",
            source_text=(
                "The river knew secrets that the mountains had forgotten, "
                "carrying them patiently to the sea where all stories end "
                "and begin again."
            ),
            source_lang="en", target_lang="de",
            reference_translations=[
                (
                    "Der Fluss kannte Geheimnisse, die die Berge vergessen hatten, "
                    "und trug sie geduldig zum Meer, wo alle Geschichten enden "
                    "und wieder beginnen."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
        ))

        return prompts

    def _generate_en_es_prompts(self) -> List[TranslationPrompt]:
        """Generate English → Spanish translation prompts."""
        prompts: List[TranslationPrompt] = []

        # --- LITERAL ---
        prompts.append(TranslationPrompt(
            prompt_id="enes-lit-01",
            text="Translate to Spanish.",
            source_text="The restaurant on the corner serves the best coffee in the neighbourhood.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "El restaurante de la esquina sirve el mejor café del barrio.",
                "El restaurante en la esquina sirve el mejor café del vecindario.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enes-lit-02",
            text="Translate to Spanish.",
            source_text="My grandmother used to tell us stories before bedtime every night.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Mi abuela solía contarnos historias antes de dormir cada noche.",
                "Mi abuela nos contaba cuentos antes de acostarnos todas las noches.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enes-lit-03",
            text="Translate to Spanish.",
            source_text="They decided to take the scenic route through the mountains.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Decidieron tomar la ruta panorámica por las montañas.",
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="general",
        ))

        # --- IDIOMATIC ---
        prompts.append(TranslationPrompt(
            prompt_id="enes-idm-01",
            text="Translate the idiom naturally into Spanish.",
            source_text="Break a leg tonight — I know your performance will be amazing!",
            source_lang="en", target_lang="es",
            reference_translations=[
                "¡Mucha mierda esta noche — sé que tu actuación será increíble!",
                "¡Mucha suerte esta noche — sé que tu actuación será asombrosa!",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
            notes="'Break a leg' = wish someone good luck (theatre).",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enes-idm-02",
            text="Translate idiomatically to Spanish.",
            source_text="She spilled the beans about the merger before the official announcement.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Ella soltó la sopa sobre la fusión antes del anuncio oficial.",
                "Ella reveló el secreto de la fusión antes del anuncio oficial.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="business",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enes-idm-03",
            text="Translate idiomatically to Spanish.",
            source_text="The project was a piece of cake once we got the right tools.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "El proyecto fue pan comido una vez que tuvimos las herramientas adecuadas.",
                "El proyecto fue coser y cantar una vez que conseguimos las herramientas correctas.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
        ))

        # --- TECHNICAL ---
        prompts.append(TranslationPrompt(
            prompt_id="enes-tec-01",
            text="Translate the scientific text to Spanish.",
            source_text=(
                "The double-blind randomised controlled trial demonstrated a "
                "statistically significant reduction in systolic blood pressure "
                "among participants receiving the experimental compound."
            ),
            source_lang="en", target_lang="es",
            reference_translations=[
                (
                    "El ensayo controlado aleatorizado doble ciego demostró una "
                    "reducción estadísticamente significativa de la presión arterial "
                    "sistólica entre los participantes que recibieron el compuesto experimental."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="medical",
            glossary={
                "double-blind": "doble ciego",
                "randomised controlled trial": "ensayo controlado aleatorizado",
                "systolic blood pressure": "presión arterial sistólica",
            },
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enes-tec-02",
            text="Translate the economics text to Spanish.",
            source_text=(
                "Quantitative easing involves the central bank purchasing "
                "government bonds to increase the money supply and lower "
                "long-term interest rates, thereby stimulating economic activity."
            ),
            source_lang="en", target_lang="es",
            reference_translations=[
                (
                    "La flexibilización cuantitativa implica que el banco central "
                    "compre bonos del gobierno para aumentar la oferta monetaria y "
                    "reducir las tasas de interés a largo plazo, estimulando así "
                    "la actividad económica."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="economics",
            glossary={
                "quantitative easing": "flexibilización cuantitativa",
                "central bank": "banco central",
                "government bonds": "bonos del gobierno",
                "money supply": "oferta monetaria",
            },
        ))

        # --- LITERARY ---
        prompts.append(TranslationPrompt(
            prompt_id="enes-lit-style-01",
            text="Translate preserving literary style.",
            source_text=(
                "She danced as if the music were woven from moonlight, "
                "each step a silver thread connecting earth to sky, "
                "and for a moment the audience forgot to breathe."
            ),
            source_lang="en", target_lang="es",
            reference_translations=[
                (
                    "Bailaba como si la música estuviera tejida de luz de luna, "
                    "cada paso un hilo de plata conectando la tierra con el cielo, "
                    "y por un momento el público olvidó respirar."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="enes-lit-style-02",
            text="Translate preserving literary style.",
            source_text=(
                "Memory is a garden where the dead still bloom, "
                "and every visit finds new flowers where there were none before."
            ),
            source_lang="en", target_lang="es",
            reference_translations=[
                (
                    "La memoria es un jardín donde los muertos aún florecen, "
                    "y cada visita encuentra nuevas flores donde antes no las había."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
        ))

        return prompts

    def _generate_ambiguous_prompts(self) -> List[TranslationPrompt]:
        """Generate prompts with ambiguous source text."""
        prompts: List[TranslationPrompt] = []

        prompts.append(TranslationPrompt(
            prompt_id="amb-01",
            text="Translate the ambiguous sentence to French.",
            source_text="I saw her duck.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "J'ai vu son canard.",
                "Je l'ai vue se baisser.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: 'duck' can be a noun (the bird) or a verb (to crouch).",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-02",
            text="Translate the ambiguous sentence to French.",
            source_text="Visiting relatives can be boring.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Rendre visite à des parents peut être ennuyeux.",
                "Les parents en visite peuvent être ennuyeux.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: the subject can be the act of visiting or the relatives themselves.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-03",
            text="Translate the ambiguous sentence to German.",
            source_text="The chicken is ready to eat.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Das Hähnchen ist fertig zum Essen.",
                "Das Huhn ist bereit zu fressen.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: the chicken is either cooked (ready to be eaten) or hungry (ready to eat something).",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-04",
            text="Translate the ambiguous sentence to Spanish.",
            source_text="They are flying planes.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Ellos están volando aviones.",
                "Son aviones que están volando.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: 'they' could be pilots flying, or the planes themselves are described as flying.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-05",
            text="Translate the ambiguous sentence to French.",
            source_text="The bank was steep.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "La berge était escarpée.",
                "La banque était raide.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: 'bank' can be a riverbank or a financial institution.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-06",
            text="Translate the ambiguous sentence to German.",
            source_text="Time flies like an arrow; fruit flies like a banana.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Die Zeit fliegt wie ein Pfeil; Fruchtfliegen mögen eine Banane.",
                "Die Zeit vergeht wie ein Pfeil; Obstfliegen mögen eine Banane.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Classic syntactic ambiguity: 'flies' and 'like' change meaning between clauses.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-07",
            text="Translate the ambiguous sentence to Spanish.",
            source_text="I left my glasses on the table.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Dejé mis gafas sobre la mesa.",
                "Dejé mis vasos sobre la mesa.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: 'glasses' could be eyeglasses or drinking glasses.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-08",
            text="Translate the ambiguous sentence to French.",
            source_text="He fed her cat food.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Il a donné de la nourriture pour chat à elle.",
                "Il lui a donné de la nourriture pour chat.",
                "Il a nourri son chat de nourriture.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Ambiguous: he fed her (cat food) or he fed (her cat) food.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-09",
            text="Translate the ambiguous sentence to German.",
            source_text="We saw the man with the telescope.",
            source_lang="en", target_lang="de",
            reference_translations=[
                "Wir sahen den Mann mit dem Teleskop.",
                "Wir sahen den Mann durch das Teleskop.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="PP-attachment ambiguity: the man has the telescope, or we used the telescope to see him.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-10",
            text="Translate the ambiguous passage to French.",
            source_text="The old man the boats while the young fish in the stream.",
            source_lang="en", target_lang="fr",
            reference_translations=[
                "Les vieux manœuvrent les bateaux tandis que les jeunes pêchent dans le ruisseau.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Garden-path sentence: 'old' and 'young' are nouns, 'man' and 'fish' are verbs.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="amb-11",
            text="Translate the ambiguous sentence to Spanish.",
            source_text="Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Los búfalos de Buffalo que los búfalos de Buffalo intimidan, intimidan a los búfalos de Buffalo.",
            ],
            difficulty=TranslationDifficulty.AMBIGUOUS,
            domain="general",
            notes="Famous ambiguous English sentence using 'buffalo' as noun, verb, and proper noun.",
        ))

        # Additional cross-language prompts
        prompts.append(TranslationPrompt(
            prompt_id="misc-01",
            text="Translate the following formal business email to French.",
            source_text=(
                "Dear Mr. Thompson, I am writing to confirm our meeting scheduled "
                "for March 15th at 2:00 PM. Please find attached the agenda and "
                "supporting documents. I look forward to a productive discussion. "
                "Best regards, Dr. Sarah Chen."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Cher Monsieur Thompson, je vous écris pour confirmer notre "
                    "réunion prévue le 15 mars à 14h00. Veuillez trouver ci-joint "
                    "l'ordre du jour et les documents justificatifs. J'attends avec "
                    "impatience une discussion productive. Cordialement, Dr. Sarah Chen."
                ),
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="business",
            notes="Formal register; preserve titles and date format.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="misc-02",
            text="Translate the dialogue to German.",
            source_text=(
                '"Have you ever been to Berlin?" she asked.\n'
                '"Once, a long time ago," he replied, staring out the window. '
                '"It was a different city then."'
            ),
            source_lang="en", target_lang="de",
            reference_translations=[
                (
                    '„Warst du schon einmal in Berlin?" fragte sie.\n'
                    '„Einmal, vor langer Zeit", antwortete er und starrte aus dem '
                    'Fenster. „Es war damals eine andere Stadt."'
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="literature",
            notes="Preserve dialogue formatting and German quotation marks.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="misc-03",
            text="Translate the proverb to Spanish.",
            source_text="A rolling stone gathers no moss.",
            source_lang="en", target_lang="es",
            reference_translations=[
                "Piedra movediza nunca moho la cobija.",
                "Piedra que rueda no cría musgo.",
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="general",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="misc-04",
            text="Translate the recipe instructions to French.",
            source_text=(
                "Preheat the oven to 180°C. Cream the butter and sugar together "
                "until light and fluffy. Beat in the eggs one at a time, then "
                "fold in the sifted flour and baking powder."
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Préchauffez le four à 180°C. Battez le beurre et le sucre "
                    "ensemble jusqu'à obtenir un mélange léger et mousseux. "
                    "Incorporez les œufs un à un, puis ajoutez délicatement "
                    "la farine tamisée et la levure chimique."
                ),
            ],
            difficulty=TranslationDifficulty.LITERAL,
            domain="cooking",
            glossary={
                "cream": "battre",
                "fold in": "incorporer délicatement",
                "baking powder": "levure chimique",
            },
        ))

        prompts.append(TranslationPrompt(
            prompt_id="misc-05",
            text="Translate the software documentation to German.",
            source_text=(
                "To install the package, run `pip install diversity-arena` in your "
                "terminal. You can verify the installation by running "
                "`python -c \"import diversity_arena; print(diversity_arena.__version__)\"`. "
                "For development, clone the repository and install in editable mode "
                "with `pip install -e '.[dev]'`."
            ),
            source_lang="en", target_lang="de",
            reference_translations=[
                (
                    "Um das Paket zu installieren, führen Sie `pip install diversity-arena` "
                    "in Ihrem Terminal aus. Sie können die Installation überprüfen, indem Sie "
                    "`python -c \"import diversity_arena; print(diversity_arena.__version__)\"` "
                    "ausführen. Für die Entwicklung klonen Sie das Repository und installieren "
                    "Sie es im bearbeitbaren Modus mit `pip install -e '.[dev]'`."
                ),
            ],
            difficulty=TranslationDifficulty.TECHNICAL,
            domain="software",
            notes="Preserve code snippets verbatim.",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="misc-06",
            text="Translate the philosophical text to Spanish.",
            source_text=(
                "To be is to be perceived, argued Berkeley, but if perception "
                "itself is an act of creation, then the observer and the observed "
                "are entangled in a dance neither can escape."
            ),
            source_lang="en", target_lang="es",
            reference_translations=[
                (
                    "Ser es ser percibido, argumentaba Berkeley, pero si la percepción "
                    "misma es un acto de creación, entonces el observador y lo observado "
                    "están entrelazados en una danza de la que ninguno puede escapar."
                ),
            ],
            difficulty=TranslationDifficulty.LITERARY,
            domain="philosophy",
        ))

        prompts.append(TranslationPrompt(
            prompt_id="misc-07",
            text="Translate the informal text message to French.",
            source_text=(
                "Hey, r u coming to the party tonite? It's gonna be lit!! "
                "Bring ur friends if u want. See u there 😊"
            ),
            source_lang="en", target_lang="fr",
            reference_translations=[
                (
                    "Salut, tu viens à la fête ce soir ? Ça va être trop bien !! "
                    "Amène tes potes si tu veux. À tout à l'heure 😊"
                ),
            ],
            difficulty=TranslationDifficulty.IDIOMATIC,
            domain="informal",
            notes="Very informal register; preserve SMS abbreviation style.",
        ))

        return prompts

    # =================================================================
    # Summary & metadata
    # =================================================================

    def summary(self) -> Dict[str, Any]:
        """Return a metadata summary of this task configuration."""
        cfg = self.config
        base: Dict[str, Any] = {
            "task_class": self.__class__.__name__,
            "name": cfg.name,
            "domain": cfg.domain.name,
            "num_prompts": cfg.num_prompts,
            "max_length": cfg.max_length,
            "evaluation_metrics": list(cfg.evaluation_metrics),
        }
        if isinstance(cfg, TranslationConfig):
            base.update({
                "language_pair": str(cfg.language_pair),
                "difficulty": cfg.difficulty.name,
                "formality_level": cfg.formality_level,
                "preserve_register": cfg.preserve_register,
                "domain_specific_terms": cfg.domain_specific_terms,
            })
        return base


# ---------------------------------------------------------------------------
# Private helper (avoid circular import for ConstraintType)
# ---------------------------------------------------------------------------

def _get_constraint_type(name: str):
    """Resolve a ConstraintType by name, avoiding top-level circular import."""
    from src.tasks.base import ConstraintType
    return ConstraintType[name]


# ---------------------------------------------------------------------------
# Translation Diversity Analyzer
# ---------------------------------------------------------------------------

class TranslationDiversityAnalyzer:
    """Comprehensive diversity analysis across a set of candidate translations.

    Provides fine-grained metrics covering lexical, structural, register, and
    cultural dimensions of translation diversity.  All public methods return
    scores normalised to [0, 1] unless otherwise noted.
    """

    # --- Formal / informal marker word lists --------------------------------

    _FORMAL_MARKERS: Set[str] = {
        "therefore", "furthermore", "moreover", "consequently", "nevertheless",
        "notwithstanding", "henceforth", "hereby", "herein", "therein",
        "accordingly", "thus", "hence", "whereas", "whereby", "shall",
        "whom", "one", "ought", "endeavour", "facilitate", "utilize",
        "ascertain", "commence", "terminate", "subsequent", "prior",
        "regarding", "pertaining", "pursuant", "aforementioned",
    }

    _INFORMAL_MARKERS: Set[str] = {
        "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep",
        "nope", "hey", "ok", "okay", "cool", "awesome", "stuff",
        "things", "like", "just", "really", "pretty", "super", "totally",
        "basically", "literally", "honestly", "anyway", "anyways",
        "btw", "lol", "omg", "dunno", "lemme", "gimme", "cuz",
    }

    _CULTURAL_IDIOM_PATTERNS: List[str] = [
        r"\b(piece of cake)\b",
        r"\b(break a leg)\b",
        r"\b(hit the nail on the head)\b",
        r"\b(cost an arm and a leg)\b",
        r"\b(under the weather)\b",
        r"\b(bite the bullet)\b",
        r"\b(once in a blue moon)\b",
        r"\b(spill the beans)\b",
        r"\b(burning the midnight oil)\b",
        r"\b(the ball is in your court)\b",
        r"\b(it takes two to tango)\b",
        r"\b(barking up the wrong tree)\b",
        r"\b(the last straw)\b",
        r"\b(a blessing in disguise)\b",
        r"\b(better late than never)\b",
    ]

    _CULTURAL_UNIT_PATTERNS: List[str] = [
        r"\b(miles?|yards?|feet|foot|inches?|gallons?|pounds?|ounces?|fahrenheit)\b",
        r"\b(kilometres?|kilometers?|metres?|meters?|litres?|liters?|celsius|kilograms?)\b",
        r"\$[\d,.]+",
        r"€[\d,.]+",
        r"£[\d,.]+",
        r"¥[\d,.]+",
    ]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def measure_translation_diversity(
        self, translations: List[str], source: str
    ) -> Dict[str, float]:
        """Compute an overall diversity profile for a set of translations.

        Parameters
        ----------
        translations : List[str]
            Two or more candidate translations of *source*.
        source : str
            The original source-language sentence.

        Returns
        -------
        Dict[str, float]
            Keys: ``lexical``, ``structural``, ``register``, ``cultural``,
            ``overall``.  Each value in [0, 1].
        """
        if len(translations) < 2:
            return {
                "lexical": 0.0,
                "structural": 0.0,
                "register": 0.0,
                "cultural": 0.0,
                "overall": 0.0,
            }

        lexical = self._compute_lexical_diversity(translations)
        structural = self._compute_structural_diversity(translations)
        register = self.measure_register_diversity(translations)
        cultural = self.measure_cultural_adaptation_diversity(
            translations, source
        )

        overall = (
            0.35 * lexical
            + 0.25 * structural
            + 0.20 * register
            + 0.20 * cultural
        )
        return {
            "lexical": lexical,
            "structural": structural,
            "register": register,
            "cultural": cultural,
            "overall": min(max(overall, 0.0), 1.0),
        }

    # -----------------------------------------------------------------------

    def measure_paraphrase_diversity(
        self, translations: List[str]
    ) -> float:
        """Measure lexical paraphrase variety across *translations*.

        Computes pairwise token-set Jaccard distance combined with bigram
        novelty so that translations sharing meaning but differing in
        surface form score highly.
        """
        if len(translations) < 2:
            return 0.0

        token_sets = [set(_tokenize(t)) for t in translations]

        # Pairwise Jaccard distance (surface-form dissimilarity)
        distances: List[float] = []
        for i, j in combinations(range(len(token_sets)), 2):
            distances.append(1.0 - _jaccard(token_sets[i], token_sets[j]))

        avg_distance = float(np.mean(distances))

        # Content-word overlap (should stay high for good paraphrases)
        content_sets = [
            set(_content_words(_tokenize(t))) for t in translations
        ]
        content_overlaps: List[float] = []
        for i, j in combinations(range(len(content_sets)), 2):
            content_overlaps.append(_jaccard(content_sets[i], content_sets[j]))

        avg_content = float(np.mean(content_overlaps)) if content_overlaps else 0.0

        # We want high surface distance *and* high content overlap
        score = 0.6 * avg_distance + 0.4 * avg_content
        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------

    def measure_register_diversity(
        self, translations: List[str]
    ) -> float:
        """Quantify variation in register / formality across *translations*.

        Returns a score in [0, 1] where 1 means the set spans the full
        formality spectrum and 0 means all translations share the same
        register.
        """
        if len(translations) < 2:
            return 0.0

        scores = [self._detect_register(t) for t in translations]
        score_range = max(scores) - min(scores)
        score_std = float(np.std(scores))

        # Combine range and standard deviation for robustness
        diversity = 0.6 * score_range + 0.4 * min(score_std * 2.0, 1.0)
        return min(max(diversity, 0.0), 1.0)

    # -----------------------------------------------------------------------

    def measure_cultural_adaptation_diversity(
        self, translations: List[str], source: str
    ) -> float:
        """Measure how differently translations handle cultural adaptation.

        Considers idiomatic expressions, measurement units, currency symbols,
        and other culturally specific choices.
        """
        if len(translations) < 2:
            return 0.0

        marker_sets = [self._extract_cultural_markers(t) for t in translations]

        # Pairwise Jaccard distance on marker sets
        distances: List[float] = []
        for i, j in combinations(range(len(marker_sets)), 2):
            distances.append(1.0 - _jaccard(marker_sets[i], marker_sets[j]))

        avg_distance = float(np.mean(distances)) if distances else 0.0

        # Bonus for translations that introduce markers absent from source
        source_markers = self._extract_cultural_markers(source)
        novel_count = 0
        total_markers = 0
        for ms in marker_sets:
            novel = ms - source_markers
            novel_count += len(novel)
            total_markers += len(ms)

        novelty_ratio = novel_count / max(total_markers, 1)

        score = 0.7 * avg_distance + 0.3 * min(novelty_ratio, 1.0)
        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------

    def compute_back_translation_consistency(
        self,
        translations: List[str],
        back_translations: List[str],
        source: str,
    ) -> Dict[str, float]:
        """Evaluate consistency of translations via back-translation.

        For each translation *t_i* we have a back-translation *bt_i*.  We
        measure how close each *bt_i* is to *source* and how the consistency
        varies across the set.

        Parameters
        ----------
        translations : List[str]
            Forward translations.
        back_translations : List[str]
            Corresponding back-translations (same length as *translations*).
        source : str
            Original source sentence.

        Returns
        -------
        Dict[str, float]
            ``mean_similarity`` – average source ↔ back-translation similarity.
            ``std_similarity``  – standard deviation of those similarities.
            ``consistency``     – 1 − normalised std (higher = more consistent).
        """
        if not translations or len(translations) != len(back_translations):
            return {
                "mean_similarity": 0.0,
                "std_similarity": 0.0,
                "consistency": 0.0,
            }

        source_tokens = set(_tokenize(source))
        source_content = set(_content_words(_tokenize(source)))

        similarities: List[float] = []
        for bt in back_translations:
            bt_tokens = set(_tokenize(bt))
            bt_content = set(_content_words(_tokenize(bt)))

            token_sim = _jaccard(source_tokens, bt_tokens)
            content_sim = _jaccard(source_content, bt_content)

            # Character-level similarity for short sentences
            if source and bt:
                max_len = max(len(source), len(bt), 1)
                char_sim = 1.0 - _levenshtein(
                    source[:300], bt[:300]
                ) / max_len
            else:
                char_sim = 0.0

            sim = 0.4 * content_sim + 0.4 * token_sim + 0.2 * max(char_sim, 0.0)
            similarities.append(sim)

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        consistency = 1.0 - min(std_sim * 2.0, 1.0)

        return {
            "mean_similarity": min(max(mean_sim, 0.0), 1.0),
            "std_similarity": std_sim,
            "consistency": min(max(consistency, 0.0), 1.0),
        }

    # -----------------------------------------------------------------------

    def compute_bleu_diversity(
        self, translations: List[str], reference: str, max_n: int = 4
    ) -> Dict[str, float]:
        """BLEU-based diversity metrics across *translations*.

        Computes per-translation BLEU against *reference*, then reports
        the mean, standard deviation, and a diversity score derived from
        pairwise BLEU between translations (lower pairwise → higher
        diversity).
        """
        if len(translations) < 2:
            return {"mean_bleu": 0.0, "std_bleu": 0.0, "bleu_diversity": 0.0}

        ref_tokens = _tokenize(reference)
        ref_len = len(ref_tokens)

        # Per-translation BLEU against reference
        bleu_scores: List[float] = []
        for t in translations:
            score = self._sentence_bleu(t, reference, max_n)
            bleu_scores.append(score)

        mean_bleu = float(np.mean(bleu_scores))
        std_bleu = float(np.std(bleu_scores))

        # Pairwise BLEU between translations (diversity proxy)
        pairwise: List[float] = []
        for i, j in combinations(range(len(translations)), 2):
            pw = self._sentence_bleu(translations[i], translations[j], max_n)
            pairwise.append(pw)

        avg_pairwise = float(np.mean(pairwise)) if pairwise else 1.0
        bleu_diversity = 1.0 - avg_pairwise

        return {
            "mean_bleu": min(max(mean_bleu, 0.0), 1.0),
            "std_bleu": std_bleu,
            "bleu_diversity": min(max(bleu_diversity, 0.0), 1.0),
        }

    # -----------------------------------------------------------------------

    def compute_chrf_diversity(
        self,
        translations: List[str],
        reference: str,
        char_order: int = 6,
        beta: float = 2.0,
    ) -> Dict[str, float]:
        """chrF-based diversity metrics across *translations*.

        Mirrors :meth:`compute_bleu_diversity` but uses character n-gram
        F-score, which is more robust for morphologically rich languages.
        """
        if len(translations) < 2:
            return {"mean_chrf": 0.0, "std_chrf": 0.0, "chrf_diversity": 0.0}

        # Per-translation chrF against reference
        chrf_scores: List[float] = []
        for t in translations:
            score = self._chrf_pair(t, reference, char_order, beta)
            chrf_scores.append(score)

        mean_chrf = float(np.mean(chrf_scores))
        std_chrf = float(np.std(chrf_scores))

        # Pairwise chrF between translations
        pairwise: List[float] = []
        for i, j in combinations(range(len(translations)), 2):
            pw = self._chrf_pair(
                translations[i], translations[j], char_order, beta
            )
            pairwise.append(pw)

        avg_pairwise = float(np.mean(pairwise)) if pairwise else 1.0
        chrf_diversity = 1.0 - avg_pairwise

        return {
            "mean_chrf": min(max(mean_chrf, 0.0), 1.0),
            "std_chrf": std_chrf,
            "chrf_diversity": min(max(chrf_diversity, 0.0), 1.0),
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_lexical_diversity(self, texts: List[str]) -> float:
        """Vocabulary variation across *texts*.

        Combines type-token ratio (TTR) variance, pairwise Jaccard distance,
        and unique unigram / bigram coverage.
        """
        if len(texts) < 2:
            return 0.0

        all_tokens: List[List[str]] = [_tokenize(t) for t in texts]
        token_sets = [set(toks) for toks in all_tokens]

        # --- TTR variance ---------------------------------------------------
        ttrs: List[float] = []
        for toks in all_tokens:
            if toks:
                ttrs.append(len(set(toks)) / len(toks))
            else:
                ttrs.append(0.0)
        ttr_var = float(np.std(ttrs))

        # --- Pairwise Jaccard distance on unigrams --------------------------
        distances: List[float] = []
        for i, j in combinations(range(len(token_sets)), 2):
            distances.append(1.0 - _jaccard(token_sets[i], token_sets[j]))
        avg_dist = float(np.mean(distances)) if distances else 0.0

        # --- Bigram novelty --------------------------------------------------
        all_bigrams: Set[Tuple[str, ...]] = set()
        per_text_bigrams: List[Set[Tuple[str, ...]]] = []
        for toks in all_tokens:
            bgs = set(_word_ngrams(toks, 2))
            per_text_bigrams.append(bgs)
            all_bigrams.update(bgs)

        if all_bigrams:
            mean_bg = float(np.mean([len(bg) for bg in per_text_bigrams]))
            novelty = 1.0 - mean_bg / max(len(all_bigrams), 1)
        else:
            novelty = 0.0

        score = 0.4 * avg_dist + 0.3 * float(novelty) + 0.3 * min(ttr_var * 4.0, 1.0)
        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------

    def _compute_structural_diversity(self, texts: List[str]) -> float:
        """Sentence-structure variation across *texts*.

        Captures differences in length distribution, word-order patterns,
        and punctuation usage.
        """
        if len(texts) < 2:
            return 0.0

        all_tokens: List[List[str]] = [_tokenize(t) for t in texts]

        # Length diversity (normalised std)
        lengths = np.array([len(toks) for toks in all_tokens], dtype=float)
        mean_len = float(np.mean(lengths))
        len_div = float(np.std(lengths) / max(mean_len, 1.0))

        # Word-order diversity
        wo_div = self._word_order_diversity(texts)

        # Punctuation profile diversity
        punct_div = self._punctuation_diversity(texts)

        score = 0.35 * min(len_div, 1.0) + 0.40 * wo_div + 0.25 * punct_div
        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------

    def _detect_register(self, text: str) -> float:
        """Estimate formality of *text* on a [0, 1] scale.

        0 → highly informal, 1 → highly formal.
        """
        tokens = _tokenize(text)
        if not tokens:
            return 0.5

        token_set = set(tokens)
        formal_hits = len(token_set & self._FORMAL_MARKERS)
        informal_hits = len(token_set & self._INFORMAL_MARKERS)

        total = formal_hits + informal_hits
        if total == 0:
            # Fall back to surface heuristics
            avg_word_len = np.mean([len(t) for t in tokens])
            # Longer words correlate with formality
            return min(max((avg_word_len - 3.0) / 5.0, 0.0), 1.0)

        formality = formal_hits / total

        # Sentence length bonus (formal texts tend to be longer)
        length_factor = min(len(tokens) / 30.0, 1.0) * 0.1
        return min(max(formality + length_factor, 0.0), 1.0)

    # -----------------------------------------------------------------------

    def _extract_cultural_markers(self, text: str) -> Set[str]:
        """Extract culturally specific markers from *text*.

        Returns a set of string labels identifying idioms, units,
        currency symbols, and other culture-dependent choices.
        """
        markers: Set[str] = set()
        lower = text.lower()

        for pattern in self._CULTURAL_IDIOM_PATTERNS:
            if re.search(pattern, lower):
                markers.add(f"idiom:{re.search(pattern, lower).group(1)}")

        for pattern in self._CULTURAL_UNIT_PATTERNS:
            for m in re.finditer(pattern, lower):
                markers.add(f"unit:{m.group(0).strip()}")

        # Date format detection
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
            markers.add("date:slash_format")
        if re.search(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", text):
            markers.add("date:dot_format")
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
            markers.add("date:iso_format")

        # Number formatting (comma vs. dot as thousands separator)
        if re.search(r"\b\d{1,3}(,\d{3})+\b", text):
            markers.add("number:comma_thousands")
        if re.search(r"\b\d{1,3}(\.\d{3})+\b", text):
            markers.add("number:dot_thousands")

        # Honorifics / titles
        for honorific in ("mr.", "mrs.", "ms.", "dr.", "prof.", "sir", "madam"):
            if honorific in lower:
                markers.add(f"honorific:{honorific}")

        return markers

    # -----------------------------------------------------------------------

    def _word_order_diversity(self, texts: List[str]) -> float:
        """Measure word-order variation across *texts*.

        Compares positional distributions of shared content words.  High
        score means translations arrange the same words in different orders.
        """
        if len(texts) < 2:
            return 0.0

        all_tokens: List[List[str]] = [_tokenize(t) for t in texts]
        content_lists: List[List[str]] = [
            _content_words(toks) for toks in all_tokens
        ]

        # Build a shared vocabulary from content words
        vocab: Set[str] = set()
        for cl in content_lists:
            vocab.update(cl)

        if not vocab:
            return 0.0

        # For each word in the shared vocabulary, record its relative
        # position in each translation that contains it.
        positions: Dict[str, List[float]] = defaultdict(list)
        for cl in content_lists:
            n = len(cl)
            for idx, word in enumerate(cl):
                positions[word].append(idx / max(n - 1, 1))

        # Variance of positions per word, averaged over vocabulary
        variances: List[float] = []
        for word, pos_list in positions.items():
            if len(pos_list) >= 2:
                variances.append(float(np.var(pos_list)))

        if not variances:
            return 0.0

        # Scale: a variance of 0.08+ is considered highly diverse
        avg_var = float(np.mean(variances))
        score = min(avg_var / 0.08, 1.0)
        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------
    # Metric primitives (sentence-level BLEU / chrF without external deps)
    # -----------------------------------------------------------------------

    def _sentence_bleu(
        self, hypothesis: str, reference: str, max_n: int = 4
    ) -> float:
        """Sentence-level BLEU between *hypothesis* and *reference*."""
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)
        hyp_len = len(hyp_tokens)
        ref_len = len(ref_tokens)

        if hyp_len == 0 or ref_len == 0:
            return 0.0

        # Brevity penalty
        if hyp_len < ref_len:
            bp = math.exp(1.0 - ref_len / hyp_len)
        else:
            bp = 1.0

        log_avg = 0.0
        weight = 1.0 / max_n

        for n in range(1, max_n + 1):
            hyp_ng = Counter(_word_ngrams(hyp_tokens, n))
            ref_ng = Counter(_word_ngrams(ref_tokens, n))

            hyp_total = sum(hyp_ng.values())
            if hyp_total == 0:
                return 0.0

            clipped = 0
            for ng, cnt in hyp_ng.items():
                clipped += min(cnt, ref_ng.get(ng, 0))

            precision = clipped / hyp_total
            if precision == 0:
                return 0.0

            log_avg += weight * math.log(precision)

        return bp * math.exp(log_avg)

    # -----------------------------------------------------------------------

    def _chrf_pair(
        self,
        hypothesis: str,
        reference: str,
        char_order: int = 6,
        beta: float = 2.0,
    ) -> float:
        """chrF between a single hypothesis–reference pair."""
        total_prec = 0.0
        total_rec = 0.0
        count = 0

        for n in range(1, char_order + 1):
            hyp_ngrams = Counter(_char_ngrams(hypothesis, n))
            ref_ngrams = Counter(_char_ngrams(reference, n))

            hyp_total = sum(hyp_ngrams.values())
            ref_total = sum(ref_ngrams.values())

            if hyp_total == 0 or ref_total == 0:
                continue

            common = sum(
                min(cnt, ref_ngrams.get(ng, 0))
                for ng, cnt in hyp_ngrams.items()
            )

            total_prec += common / hyp_total
            total_rec += common / ref_total
            count += 1

        if count == 0:
            return 0.0

        avg_prec = total_prec / count
        avg_rec = total_rec / count

        if avg_prec == 0.0 and avg_rec == 0.0:
            return 0.0

        beta_sq = beta ** 2
        f_score = (
            (1.0 + beta_sq) * avg_prec * avg_rec
            / (beta_sq * avg_prec + avg_rec)
        )
        return min(max(f_score, 0.0), 1.0)

    # -----------------------------------------------------------------------

    @staticmethod
    def _punctuation_diversity(texts: List[str]) -> float:
        """Diversity of punctuation profiles across *texts*."""
        if len(texts) < 2:
            return 0.0

        punct_chars = set(string.punctuation)

        profiles: List[Counter] = []
        for text in texts:
            profile: Counter = Counter()
            for ch in text:
                if ch in punct_chars:
                    profile[ch] += 1
            profiles.append(profile)

        # Normalise profiles to relative frequencies
        vectors: List[Dict[str, float]] = []
        for prof in profiles:
            total = sum(prof.values())
            if total > 0:
                vectors.append({k: v / total for k, v in prof.items()})
            else:
                vectors.append({})

        # Pairwise cosine distance
        all_keys: Set[str] = set()
        for v in vectors:
            all_keys.update(v.keys())

        if not all_keys:
            return 0.0

        ordered_keys = sorted(all_keys)
        np_vectors = np.array(
            [[v.get(k, 0.0) for k in ordered_keys] for v in vectors]
        )

        distances: List[float] = []
        for i, j in combinations(range(len(np_vectors)), 2):
            a = np_vectors[i]
            b = np_vectors[j]
            dot = float(np.dot(a, b))
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            if norm_a > 0 and norm_b > 0:
                cos_sim = dot / (norm_a * norm_b)
                distances.append(1.0 - cos_sim)
            else:
                distances.append(1.0)

        return min(max(float(np.mean(distances)), 0.0), 1.0)
