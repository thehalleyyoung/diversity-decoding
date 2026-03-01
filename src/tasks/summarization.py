"""
Summarization task domain for the Diversity Decoding Arena.

Implements diverse summary generation with ROUGE metrics computed from scratch,
faithfulness checking, coverage analysis, and structural quality evaluation.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.tasks.base import (
    GenerationTask,
    PromptDataset,
    TaskConfig,
    TaskConstraint,
    TaskEvaluator,
    TaskPrompt,
)
from src.types import TaskDomain


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SummaryType(Enum):
    """Types of summaries the system can generate."""
    EXTRACTIVE = auto()
    ABSTRACTIVE = auto()
    HEADLINE = auto()
    BULLET_POINTS = auto()
    TLDR = auto()
    EXECUTIVE = auto()


class SummaryLength(Enum):
    """Target length categories for summaries."""
    VERY_SHORT = auto()
    SHORT = auto()
    MEDIUM = auto()
    LONG = auto()

    @property
    def word_range(self) -> Tuple[int, int]:
        return {
            SummaryLength.VERY_SHORT: (10, 30),
            SummaryLength.SHORT: (30, 80),
            SummaryLength.MEDIUM: (80, 200),
            SummaryLength.LONG: (200, 500),
        }[self]


# ---------------------------------------------------------------------------
# Configs / Prompts
# ---------------------------------------------------------------------------

@dataclass
class SummarizationConfig(TaskConfig):
    """Configuration for the summarization task."""
    summary_type: SummaryType = SummaryType.ABSTRACTIVE
    target_length: SummaryLength = SummaryLength.MEDIUM
    compression_ratio: float = 0.2
    focus_aspects: List[str] = field(default_factory=list)
    preserve_entities: bool = True
    audience_level: str = "general"


@dataclass
class SummarizationPrompt(TaskPrompt):
    """A single summarization prompt with its source material."""
    source_document: str = ""
    summary_type: SummaryType = SummaryType.ABSTRACTIVE
    target_length: SummaryLength = SummaryLength.MEDIUM
    key_points: List[str] = field(default_factory=list)
    reference_summaries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stop-word list (small, self-contained)
# ---------------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "where", "when", "how", "not", "no", "nor", "as", "if",
    "then", "than", "too", "very", "just", "about", "above", "after",
    "again", "all", "also", "am", "any", "because", "before", "between",
    "both", "each", "few", "more", "most", "other", "own", "same", "so",
    "some", "such", "only", "into", "over", "out", "up", "down", "off",
    "once", "here", "there", "why", "s", "t", "d", "ll", "re", "ve",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenisation."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _syllable_count(word: str) -> int:
    """Heuristic syllable counter for Flesch-Kincaid."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    vowel_groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowel_groups))


def _simple_ner(text: str) -> List[str]:
    """Very lightweight named-entity extraction based on capitalisation."""
    entities: List[str] = []
    current: List[str] = []
    for tok in text.split():
        cleaned = tok.strip(string.punctuation)
        if not cleaned:
            if current:
                entities.append(" ".join(current))
                current = []
            continue
        if cleaned[0].isupper() and cleaned not in {"I", "A"}:
            current.append(cleaned)
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    return [e for e in entities if len(e) > 1]


# ---------------------------------------------------------------------------
# SummarizationTask
# ---------------------------------------------------------------------------

class SummarizationTask(GenerationTask):
    """Full implementation of the summarization task domain."""

    def __init__(self, config: Optional[SummarizationConfig] = None) -> None:
        self.config = config or SummarizationConfig()
        self._prompts: Optional[PromptDataset] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Load 40+ diverse summarization prompts."""
        if self._prompts is not None:
            return self._prompts

        prompts: List[SummarizationPrompt] = []
        prompts.extend(self._generate_news_summarization_prompts())
        prompts.extend(self._generate_scientific_summarization_prompts())
        prompts.extend(self._generate_document_summarization_prompts())

        self._prompts = PromptDataset(prompts=prompts)
        return self._prompts

    def format_prompt(self, prompt: TaskPrompt) -> str:
        """Format a *SummarizationPrompt* into a model-ready instruction."""
        if not isinstance(prompt, SummarizationPrompt):
            raise TypeError("Expected a SummarizationPrompt instance")

        stype = prompt.summary_type
        length = prompt.target_length

        type_instructions = {
            SummaryType.EXTRACTIVE: (
                "Create an extractive summary by selecting the most important "
                "sentences directly from the source text. Do not paraphrase."
            ),
            SummaryType.ABSTRACTIVE: (
                "Write an abstractive summary that captures the key information "
                "in your own words. Paraphrase and condense freely."
            ),
            SummaryType.HEADLINE: (
                "Write a concise headline (one sentence) that captures the core "
                "message of the source document."
            ),
            SummaryType.BULLET_POINTS: (
                "Summarize the source document as a bulleted list of key points. "
                "Each bullet should be one concise sentence."
            ),
            SummaryType.TLDR: (
                "Provide a TL;DR (too long; didn't read) summary – one to three "
                "sentences that give the gist of the document."
            ),
            SummaryType.EXECUTIVE: (
                "Write an executive summary suitable for a senior decision-maker. "
                "Include context, key findings, and recommended actions."
            ),
        }

        lo, hi = length.word_range
        length_instruction = f"Target length: {lo}–{hi} words."

        parts = [
            "=== SUMMARIZATION TASK ===",
            "",
            type_instructions.get(stype, type_instructions[SummaryType.ABSTRACTIVE]),
            length_instruction,
        ]

        if prompt.key_points:
            parts.append("")
            parts.append("Focus on the following key points:")
            for kp in prompt.key_points:
                parts.append(f"  - {kp}")

        if self.config.focus_aspects:
            parts.append("")
            parts.append("Pay special attention to these aspects:")
            for fa in self.config.focus_aspects:
                parts.append(f"  - {fa}")

        if self.config.audience_level != "general":
            parts.append(f"\nAudience level: {self.config.audience_level}")

        parts.append("")
        parts.append("=== SOURCE DOCUMENT ===")
        parts.append(prompt.source_document)
        parts.append("")
        parts.append("=== YOUR SUMMARY ===")

        return "\n".join(parts)

    def evaluate(
        self,
        generations: List[str],
        prompts: List[TaskPrompt],
    ) -> Dict[str, Any]:
        """Evaluate a batch of generated summaries against their prompts."""
        results: Dict[str, Any] = {
            "per_sample": [],
            "aggregate": {},
        }

        all_rouge1_f: List[float] = []
        all_rouge2_f: List[float] = []
        all_rougel_f: List[float] = []
        all_compression: List[float] = []
        all_density: List[float] = []
        all_coverage: List[float] = []
        all_faithfulness: List[float] = []
        all_coherence: List[float] = []
        all_redundancy: List[float] = []
        all_abstractiveness: List[float] = []
        all_entity_cov: List[float] = []
        all_readability: List[float] = []
        all_structural: List[float] = []

        for gen, prompt in zip(generations, prompts):
            if not isinstance(prompt, SummarizationPrompt):
                continue

            source = prompt.source_document
            summary = self.post_process(gen)

            rouge1 = self._rouge_n(summary, source, 1)
            rouge2 = self._rouge_n(summary, source, 2)
            rougel = self._rouge_l(summary, source)
            compression = self._compression_ratio(summary, source)
            density = self._information_density(summary)
            coverage = self._coverage_score(summary, prompt.key_points)
            faithfulness = self._faithfulness_score(summary, source)
            coherence = self._coherence_score(summary)
            redundancy = self._redundancy_score(summary)
            abstractiveness = self._abstractiveness_score(summary, source)
            entity_cov = self._entity_coverage(summary, source)
            readability = self._readability_score(summary)
            structural = self._structural_quality(summary, prompt.summary_type)

            # Reference-based ROUGE if references are available
            ref_rouge1: List[Dict[str, float]] = []
            ref_rouge2: List[Dict[str, float]] = []
            ref_rougel: List[Dict[str, float]] = []
            for ref in prompt.reference_summaries:
                ref_rouge1.append(self._rouge_n(summary, ref, 1))
                ref_rouge2.append(self._rouge_n(summary, ref, 2))
                ref_rougel.append(self._rouge_l(summary, ref))

            best_ref_r1 = max((r["f1"] for r in ref_rouge1), default=rouge1["f1"])
            best_ref_r2 = max((r["f1"] for r in ref_rouge2), default=rouge2["f1"])
            best_ref_rl = max((r["f1"] for r in ref_rougel), default=rougel["f1"])

            sample_result = {
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougel,
                "best_ref_rouge1_f1": best_ref_r1,
                "best_ref_rouge2_f1": best_ref_r2,
                "best_ref_rougeL_f1": best_ref_rl,
                "compression_ratio": compression,
                "information_density": density,
                "coverage": coverage,
                "faithfulness": faithfulness,
                "coherence": coherence,
                "redundancy": redundancy,
                "abstractiveness": abstractiveness,
                "entity_coverage": entity_cov,
                "readability": readability,
                "structural_quality": structural,
            }
            results["per_sample"].append(sample_result)

            all_rouge1_f.append(best_ref_r1)
            all_rouge2_f.append(best_ref_r2)
            all_rougel_f.append(best_ref_rl)
            all_compression.append(compression)
            all_density.append(density)
            all_coverage.append(coverage)
            all_faithfulness.append(faithfulness)
            all_coherence.append(coherence)
            all_redundancy.append(redundancy)
            all_abstractiveness.append(abstractiveness)
            all_entity_cov.append(entity_cov)
            all_readability.append(readability)
            all_structural.append(structural)

        def _safe_mean(vals: List[float]) -> float:
            return float(np.mean(vals)) if vals else 0.0

        def _safe_std(vals: List[float]) -> float:
            return float(np.std(vals)) if vals else 0.0

        results["aggregate"] = {
            "rouge1_f1_mean": _safe_mean(all_rouge1_f),
            "rouge1_f1_std": _safe_std(all_rouge1_f),
            "rouge2_f1_mean": _safe_mean(all_rouge2_f),
            "rouge2_f1_std": _safe_std(all_rouge2_f),
            "rougeL_f1_mean": _safe_mean(all_rougel_f),
            "rougeL_f1_std": _safe_std(all_rougel_f),
            "compression_ratio_mean": _safe_mean(all_compression),
            "information_density_mean": _safe_mean(all_density),
            "coverage_mean": _safe_mean(all_coverage),
            "faithfulness_mean": _safe_mean(all_faithfulness),
            "coherence_mean": _safe_mean(all_coherence),
            "redundancy_mean": _safe_mean(all_redundancy),
            "abstractiveness_mean": _safe_mean(all_abstractiveness),
            "entity_coverage_mean": _safe_mean(all_entity_cov),
            "readability_mean": _safe_mean(all_readability),
            "structural_quality_mean": _safe_mean(all_structural),
            "num_samples": len(all_rouge1_f),
        }

        # Diversity across all generated summaries
        if len(generations) > 1:
            results["aggregate"]["diversity"] = self._diversity_of_summaries(
                [self.post_process(g) for g in generations]
            )

        return results

    def get_constraints(self) -> List[TaskConstraint]:
        """Return constraints for summarization quality."""
        lo, hi = self.config.target_length.word_range

        constraints: List[TaskConstraint] = [
            TaskConstraint(
                name="length",
                description=f"Summary must be between {lo} and {hi} words.",
                check=lambda text: lo <= len(_tokenize(text)) <= hi,
            ),
            TaskConstraint(
                name="coverage",
                description="Summary should cover at least 40 % of source key points.",
                check=lambda text: True,  # evaluated post-hoc via coverage_score
            ),
            TaskConstraint(
                name="faithfulness",
                description="Summary must not introduce facts absent from the source.",
                check=lambda text: True,  # evaluated post-hoc via faithfulness_score
            ),
            TaskConstraint(
                name="non_empty",
                description="Summary must not be empty.",
                check=lambda text: len(text.strip()) > 0,
            ),
            TaskConstraint(
                name="compression",
                description="Summary should be significantly shorter than the source.",
                check=lambda text: True,
            ),
        ]

        if self.config.summary_type == SummaryType.BULLET_POINTS:
            constraints.append(
                TaskConstraint(
                    name="bullet_format",
                    description="Summary must use bullet-point formatting.",
                    check=lambda text: any(
                        line.strip().startswith(("-", "•", "*"))
                        for line in text.split("\n")
                        if line.strip()
                    ),
                )
            )

        if self.config.summary_type == SummaryType.HEADLINE:
            constraints.append(
                TaskConstraint(
                    name="headline_length",
                    description="Headline must be a single sentence of at most 20 words.",
                    check=lambda text: len(_tokenize(text)) <= 20
                    and text.count("\n") <= 1,
                )
            )

        return constraints

    # ------------------------------------------------------------------
    # ROUGE-N
    # ------------------------------------------------------------------

    def _rouge_n(
        self,
        hypothesis: str,
        reference: str,
        n: int,
    ) -> Dict[str, float]:
        """Compute ROUGE-N precision, recall, and F1 from scratch."""
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        if not hyp_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        if not hyp_ngrams or not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        hyp_counts: Counter = Counter(hyp_ngrams)
        ref_counts: Counter = Counter(ref_ngrams)

        overlap = 0
        for ng, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(ng, 0))

        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # ROUGE-L
    # ------------------------------------------------------------------

    def _rouge_l(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-L based on longest common subsequence."""
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        if not hyp_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        lcs = self._lcs_length(hyp_tokens, ref_tokens)

        precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            beta = precision / recall if recall > 0 else 1.0
            numerator = (1.0 + beta ** 2) * precision * recall
            denominator = (beta ** 2) * precision + recall
            f1 = numerator / denominator if denominator > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _compression_ratio(self, summary: str, source: str) -> float:
        """Ratio of summary length to source length (lower → more compressed)."""
        src_len = max(len(_tokenize(source)), 1)
        sum_len = len(_tokenize(summary))
        return sum_len / src_len

    def _information_density(self, summary: str) -> float:
        """Fraction of content words (non-stop-words) in the summary."""
        return self._content_word_ratio(summary)

    def _coverage_score(self, summary: str, source_key_points: List[str]) -> float:
        """How many of the key points are mentioned in the summary."""
        if not source_key_points:
            return 1.0

        summary_lower = summary.lower()
        summary_tokens = set(_tokenize(summary))
        covered = 0

        for kp in source_key_points:
            kp_tokens = set(_tokenize(kp))
            if not kp_tokens:
                covered += 1
                continue

            # Check substring match first
            if kp.lower() in summary_lower:
                covered += 1
                continue

            # Token overlap check
            overlap = len(kp_tokens & summary_tokens) / len(kp_tokens)
            if overlap >= 0.5:
                covered += 1

        return covered / len(source_key_points)

    def _faithfulness_score(self, summary: str, source: str) -> float:
        """Estimate faithfulness: penalise n-grams in summary absent from source.

        A simple proxy – real faithfulness checking would use NLI models.
        """
        sum_tokens = _tokenize(summary)
        src_tokens = _tokenize(source)

        if not sum_tokens:
            return 1.0

        src_bigrams = set(_ngrams(src_tokens, 2))
        src_unigrams = set(src_tokens)
        sum_bigrams = _ngrams(sum_tokens, 2)

        if not sum_bigrams:
            # Fall back to unigram check
            novel = [t for t in sum_tokens if t not in src_unigrams and t not in _STOP_WORDS]
            return 1.0 - min(len(novel) / max(len(sum_tokens), 1), 1.0)

        faithful_count = sum(1 for bg in sum_bigrams if bg in src_bigrams)
        bigram_faithfulness = faithful_count / len(sum_bigrams)

        # Unigram content-word check
        content_tokens = [t for t in sum_tokens if t not in _STOP_WORDS]
        if content_tokens:
            content_in_src = sum(1 for t in content_tokens if t in src_unigrams)
            unigram_faithfulness = content_in_src / len(content_tokens)
        else:
            unigram_faithfulness = 1.0

        return 0.4 * bigram_faithfulness + 0.6 * unigram_faithfulness

    def _coherence_score(self, summary: str) -> float:
        """Score logical flow by checking sentence connectivity."""
        sentences = self._sentence_segmentation(summary)
        if len(sentences) <= 1:
            return 1.0

        transition_words = {
            "however", "therefore", "moreover", "furthermore", "additionally",
            "consequently", "meanwhile", "nevertheless", "nonetheless",
            "thus", "hence", "also", "besides", "likewise", "similarly",
            "in addition", "as a result", "on the other hand", "in contrast",
            "for example", "for instance", "specifically", "in particular",
            "finally", "overall", "in conclusion", "to summarize",
            "first", "second", "third", "next", "then", "lastly",
        }

        transition_count = 0
        for sent in sentences[1:]:
            tokens = _tokenize(sent)
            first_few = " ".join(tokens[:4])
            if any(tw in first_few for tw in transition_words):
                transition_count += 1

        transition_score = min(transition_count / max(len(sentences) - 1, 1), 1.0)

        # Lexical overlap between consecutive sentences
        overlaps: List[float] = []
        for i in range(len(sentences) - 1):
            t1 = set(_tokenize(sentences[i])) - _STOP_WORDS
            t2 = set(_tokenize(sentences[i + 1])) - _STOP_WORDS
            if t1 and t2:
                overlaps.append(len(t1 & t2) / min(len(t1), len(t2)))
            else:
                overlaps.append(0.0)

        overlap_score = float(np.mean(overlaps)) if overlaps else 0.0

        return 0.4 * transition_score + 0.6 * min(overlap_score * 2.0, 1.0)

    def _redundancy_score(self, summary: str) -> float:
        """Penalise repetition. Returns 0 (no redundancy) to 1 (fully redundant)."""
        sentences = self._sentence_segmentation(summary)
        if len(sentences) <= 1:
            return 0.0

        # Pairwise sentence similarity
        max_sim = 0.0
        total_sim = 0.0
        pair_count = 0

        for i in range(len(sentences)):
            ti = set(_tokenize(sentences[i])) - _STOP_WORDS
            for j in range(i + 1, len(sentences)):
                tj = set(_tokenize(sentences[j])) - _STOP_WORDS
                if ti and tj:
                    sim = len(ti & tj) / max(len(ti | tj), 1)
                    max_sim = max(max_sim, sim)
                    total_sim += sim
                    pair_count += 1

        avg_sim = total_sim / pair_count if pair_count else 0.0

        # Repeated trigrams
        tokens = _tokenize(summary)
        trigrams = _ngrams(tokens, 3)
        if trigrams:
            trigram_counts = Counter(trigrams)
            repeated = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_rep = repeated / len(trigrams)
        else:
            trigram_rep = 0.0

        return min(0.5 * max_sim + 0.3 * avg_sim + 0.2 * trigram_rep, 1.0)

    def _abstractiveness_score(self, summary: str, source: str) -> float:
        """Measure how abstractive the summary is (1 = fully novel phrasing)."""
        fragments = self._extractive_fragments(summary, source)
        sum_tokens = _tokenize(summary)
        if not sum_tokens:
            return 1.0

        extractive_tokens = sum(len(_tokenize(f)) for f in fragments)
        extractive_ratio = extractive_tokens / len(sum_tokens)

        # Novel unigrams
        src_set = set(_tokenize(source))
        novel = [t for t in sum_tokens if t not in src_set and t not in _STOP_WORDS]
        novel_ratio = len(novel) / max(len([t for t in sum_tokens if t not in _STOP_WORDS]), 1)

        return 0.6 * (1.0 - extractive_ratio) + 0.4 * novel_ratio

    def _entity_coverage(self, summary: str, source: str) -> float:
        """Fraction of source named entities that appear in the summary."""
        src_entities = _simple_ner(source)
        if not src_entities:
            return 1.0

        summary_lower = summary.lower()
        covered = sum(1 for e in src_entities if e.lower() in summary_lower)
        return covered / len(src_entities)

    def _sentence_importance_ranking(self, source: str) -> List[Tuple[str, float]]:
        """Rank sentences in the source by importance (TF-based heuristic)."""
        sentences = self._sentence_segmentation(source)
        if not sentences:
            return []

        # Corpus-level token frequencies
        all_tokens = _tokenize(source)
        freq: Counter = Counter(all_tokens)
        max_freq = max(freq.values()) if freq else 1

        scored: List[Tuple[str, float]] = []
        for idx, sent in enumerate(sentences):
            tokens = _tokenize(sent)
            content = [t for t in tokens if t not in _STOP_WORDS]
            if not content:
                scored.append((sent, 0.0))
                continue

            # TF score
            tf_score = sum(freq[t] / max_freq for t in content) / len(content)

            # Position bias: first and last sentences are often important
            pos_score = 0.0
            if idx == 0:
                pos_score = 0.3
            elif idx == len(sentences) - 1:
                pos_score = 0.15
            elif idx <= 2:
                pos_score = 0.1

            # Length bonus – prefer medium-length sentences
            len_score = min(len(content) / 15.0, 1.0) * 0.1

            # Entity bonus
            ents = _simple_ner(sent)
            ent_score = min(len(ents) * 0.05, 0.2)

            total = tf_score + pos_score + len_score + ent_score
            scored.append((sent, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _extractive_fragments(self, summary: str, source: str) -> List[str]:
        """Find contiguous token spans from the source that appear in the summary."""
        sum_tokens = _tokenize(summary)
        src_tokens = _tokenize(source)

        if not sum_tokens or not src_tokens:
            return []

        fragments: List[str] = []
        i = 0
        while i < len(sum_tokens):
            best_len = 0
            for j in range(len(src_tokens)):
                k = 0
                while (
                    i + k < len(sum_tokens)
                    and j + k < len(src_tokens)
                    and sum_tokens[i + k] == src_tokens[j + k]
                ):
                    k += 1
                best_len = max(best_len, k)
            if best_len >= 3:
                fragments.append(" ".join(sum_tokens[i : i + best_len]))
                i += best_len
            else:
                i += 1

        return fragments

    def _diversity_of_summaries(self, summaries: List[str]) -> float:
        """Measure how different a set of summaries are from each other.

        Returns a value between 0 (identical) and 1 (maximally diverse).
        """
        if len(summaries) <= 1:
            return 0.0

        token_sets = [set(_tokenize(s)) for s in summaries]
        bigram_sets = [set(_ngrams(_tokenize(s), 2)) for s in summaries]

        pairwise_unigram: List[float] = []
        pairwise_bigram: List[float] = []

        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                # Jaccard distance for unigrams
                union_u = token_sets[i] | token_sets[j]
                inter_u = token_sets[i] & token_sets[j]
                if union_u:
                    pairwise_unigram.append(1.0 - len(inter_u) / len(union_u))
                else:
                    pairwise_unigram.append(0.0)

                # Jaccard distance for bigrams
                union_b = bigram_sets[i] | bigram_sets[j]
                inter_b = bigram_sets[i] & bigram_sets[j]
                if union_b:
                    pairwise_bigram.append(1.0 - len(inter_b) / len(union_b))
                else:
                    pairwise_bigram.append(0.0)

        avg_uni = float(np.mean(pairwise_unigram)) if pairwise_unigram else 0.0
        avg_bi = float(np.mean(pairwise_bigram)) if pairwise_bigram else 0.0

        # Length diversity
        lengths = [len(_tokenize(s)) for s in summaries]
        if max(lengths) > 0:
            len_cv = float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0.0
        else:
            len_cv = 0.0

        return 0.4 * avg_uni + 0.4 * avg_bi + 0.2 * min(len_cv, 1.0)

    def _focus_adherence(self, summary: str, focus_aspects: List[str]) -> float:
        """How well the summary addresses the requested focus aspects."""
        if not focus_aspects:
            return 1.0

        summary_lower = summary.lower()
        summary_tokens = set(_tokenize(summary))
        scores: List[float] = []

        for aspect in focus_aspects:
            aspect_tokens = set(_tokenize(aspect))
            if not aspect_tokens:
                scores.append(1.0)
                continue

            # Direct substring match
            if aspect.lower() in summary_lower:
                scores.append(1.0)
                continue

            # Token overlap
            overlap = len(aspect_tokens & summary_tokens)
            scores.append(min(overlap / len(aspect_tokens), 1.0))

        return float(np.mean(scores)) if scores else 1.0

    def _readability_score(self, text: str) -> float:
        """Flesch-Kincaid readability adapted to a 0-1 scale."""
        sentences = self._sentence_segmentation(text)
        words = _tokenize(text)

        if not words or not sentences:
            return 0.0

        total_syllables = sum(_syllable_count(w) for w in words)
        avg_sentence_len = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        # Flesch Reading Ease
        fre = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word

        # Clamp to [0, 100] then normalise
        fre = max(0.0, min(fre, 100.0))
        return fre / 100.0

    def _key_phrase_extraction(self, text: str) -> List[str]:
        """Extract key phrases using a simple frequency-based approach."""
        tokens = _tokenize(text)
        content = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
        freq: Counter = Counter(content)

        # Also consider bigrams of content words
        bigram_phrases: List[str] = []
        for i in range(len(tokens) - 1):
            if tokens[i] not in _STOP_WORDS and tokens[i + 1] not in _STOP_WORDS:
                bigram_phrases.append(f"{tokens[i]} {tokens[i + 1]}")

        bigram_freq: Counter = Counter(bigram_phrases)

        # Combine unigram and bigram candidates
        candidates: Dict[str, float] = {}
        for word, count in freq.most_common(30):
            candidates[word] = count
        for phrase, count in bigram_freq.most_common(20):
            candidates[phrase] = count * 1.5  # slight boost for bigrams

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_candidates[:15]]

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of the longest common subsequence via DP."""
        m, n = len(seq1), len(seq2)

        # Space-optimised: two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def _sentence_segmentation(self, text: str) -> List[str]:
        """Split text into sentences using regex heuristics."""
        text = text.strip()
        if not text:
            return []

        # Handle bullet points as separate 'sentences'
        lines = text.split("\n")
        bullets: List[str] = []
        non_bullet_text = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0] in "-•*" and len(stripped) > 2:
                bullets.append(stripped.lstrip("-•* ").strip())
            elif stripped:
                non_bullet_text.append(stripped)

        combined = " ".join(non_bullet_text)

        # Sentence boundary detection
        sent_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])"
            r"|(?<=[.!?])\s*$"
        )
        raw_sentences = sent_pattern.split(combined)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        return sentences + bullets

    def _content_word_ratio(self, text: str) -> float:
        """Fraction of tokens that are content (non-stop) words."""
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        content = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
        return len(content) / len(tokens)

    def _structural_quality(self, summary: str, summary_type: SummaryType) -> float:
        """Evaluate whether the summary follows the expected structural format."""
        score = 1.0

        if summary_type == SummaryType.BULLET_POINTS:
            lines = [l.strip() for l in summary.split("\n") if l.strip()]
            bullet_lines = [l for l in lines if l[0] in "-•*" if l]
            if not bullet_lines:
                score *= 0.2
            else:
                bullet_ratio = len(bullet_lines) / len(lines)
                score *= bullet_ratio
                # Each bullet should be concise
                long_bullets = [b for b in bullet_lines if len(_tokenize(b)) > 30]
                if long_bullets:
                    score *= max(0.5, 1.0 - len(long_bullets) / len(bullet_lines))

        elif summary_type == SummaryType.HEADLINE:
            sentences = self._sentence_segmentation(summary)
            word_count = len(_tokenize(summary))
            if len(sentences) > 1:
                score *= 0.3
            if word_count > 20:
                score *= max(0.3, 1.0 - (word_count - 20) / 20)
            if word_count < 3:
                score *= 0.2

        elif summary_type == SummaryType.TLDR:
            sentences = self._sentence_segmentation(summary)
            if len(sentences) > 3:
                score *= max(0.4, 1.0 - (len(sentences) - 3) / 5)
            word_count = len(_tokenize(summary))
            if word_count > 80:
                score *= max(0.4, 1.0 - (word_count - 80) / 80)

        elif summary_type == SummaryType.EXECUTIVE:
            sentences = self._sentence_segmentation(summary)
            if len(sentences) < 3:
                score *= 0.5
            word_count = len(_tokenize(summary))
            if word_count < 50:
                score *= 0.5

        elif summary_type == SummaryType.EXTRACTIVE:
            pass  # hard to check structurally without the source

        elif summary_type == SummaryType.ABSTRACTIVE:
            sentences = self._sentence_segmentation(summary)
            if not sentences:
                score *= 0.1

        return max(0.0, min(score, 1.0))

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self, text: str) -> str:
        """Clean and normalise a generated summary."""
        text = text.strip()

        # Remove common LLM artefacts
        prefixes_to_strip = [
            "Sure, here is a summary:",
            "Here is the summary:",
            "Summary:",
            "Here's a summary:",
            "SUMMARY:",
            "=== YOUR SUMMARY ===",
        ]
        for prefix in prefixes_to_strip:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()

        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalise whitespace within lines
        lines = text.split("\n")
        cleaned_lines = [re.sub(r"  +", " ", line) for line in lines]
        text = "\n".join(cleaned_lines)

        # Remove trailing incomplete sentence (no sentence-ending punctuation)
        sentences = self._sentence_segmentation(text)
        if sentences and not sentences[-1].rstrip().endswith((".", "!", "?", ":")):
            # Only remove if there are other complete sentences
            if len(sentences) > 1:
                # Find the last complete sentence in the raw text
                last_good = -1
                for punc in [".", "!", "?"]:
                    idx = text.rfind(punc)
                    if idx > last_good:
                        last_good = idx
                if last_good > 0:
                    text = text[: last_good + 1]

        return text.strip()

    # ------------------------------------------------------------------
    # Prompt generators
    # ------------------------------------------------------------------

    def _generate_news_summarization_prompts(self) -> List[SummarizationPrompt]:
        """Generate summarization prompts from news-style source documents."""
        prompts: List[SummarizationPrompt] = []

        # --- Prompt 1: Technology news ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A coalition of leading technology companies announced today a new "
                "open standard for artificial intelligence safety testing. The "
                "initiative, dubbed the AI Safety Benchmark Consortium, includes "
                "major firms such as several large cloud providers and prominent AI "
                "research labs. The consortium aims to create a unified framework "
                "for evaluating the safety and reliability of large language models "
                "before they are deployed in production environments. According to "
                "the press release, the framework will cover areas including "
                "hallucination detection, bias measurement, adversarial robustness, "
                "and privacy preservation. Industry analysts say the move reflects "
                "growing pressure from regulators worldwide who have called for "
                "more transparent AI development processes. The European Union, in "
                "particular, has been advancing its AI Act, which will require "
                "companies to demonstrate compliance with strict safety standards. "
                "Critics argue the consortium may move too slowly given the rapid "
                "pace of AI advancement, while supporters contend that industry-led "
                "standards tend to be more technically nuanced than government "
                "mandates. The first set of benchmark tests is expected to be "
                "released in the second quarter of next year."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "new open standard for AI safety",
                "multiple companies involved",
                "covers hallucination, bias, robustness, privacy",
                "response to regulatory pressure",
            ],
            reference_summaries=[
                "Major tech firms have formed the AI Safety Benchmark Consortium to "
                "create a unified safety testing framework for large language models, "
                "covering hallucination detection, bias, robustness, and privacy, in "
                "response to growing regulatory pressure."
            ],
        ))

        # --- Prompt 2: Climate change report ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A new report from the Global Climate Research Institute reveals "
                "that ocean temperatures have risen at an unprecedented rate over "
                "the past decade. The study, published in the journal Nature Climate "
                "Change, analyzed temperature data from over 3,800 monitoring "
                "stations across all major ocean basins. Researchers found that the "
                "average sea surface temperature increased by 0.12 degrees Celsius "
                "per year between 2014 and 2024, roughly double the rate observed "
                "in the previous decade. The warming has been particularly "
                "pronounced in the North Atlantic, where temperatures exceeded "
                "historical records by significant margins during the summer months "
                "of 2023 and 2024. Marine biologists warn that the accelerated "
                "warming threatens coral reef ecosystems, disrupts fish migration "
                "patterns, and may lead to more intense hurricane seasons. The "
                "report recommends immediate action to reduce greenhouse gas "
                "emissions and calls for expanded ocean monitoring infrastructure "
                "to better track future changes. Several governments have already "
                "cited the findings in support of new emissions reduction targets "
                "ahead of the upcoming climate summit."
            ),
            summary_type=SummaryType.TLDR,
            target_length=SummaryLength.VERY_SHORT,
            key_points=[
                "ocean temperatures rising faster than before",
                "0.12°C per year increase",
                "North Atlantic particularly affected",
                "threatens marine ecosystems",
            ],
            reference_summaries=[
                "Ocean temperatures have doubled their rate of increase over the "
                "past decade, with the North Atlantic especially hard hit, "
                "threatening coral reefs and fish populations."
            ],
        ))

        # --- Prompt 3: Economic news ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The Federal Reserve announced today that it will hold interest "
                "rates steady at the current range of 5.25 to 5.50 percent, "
                "marking the third consecutive meeting without a rate change. In "
                "a statement following the two-day policy meeting, the central "
                "bank noted that inflation remains above the 2 percent target but "
                "has shown signs of moderating in recent months. The consumer "
                "price index rose 3.1 percent year-over-year in the most recent "
                "reading, down from a peak of 9.1 percent in June 2022. The "
                "labor market continues to show resilience, with unemployment "
                "holding at 3.7 percent and job growth exceeding expectations in "
                "four of the past six months. Fed Chair Jerome Powell stated "
                "during the press conference that the committee is closely "
                "monitoring incoming data and remains prepared to adjust policy "
                "as needed. Financial markets reacted positively, with the S&P "
                "500 gaining 1.2 percent on the news. Bond yields fell slightly, "
                "suggesting investors expect rate cuts to begin in the coming "
                "months. Economists remain divided on the timing, with some "
                "forecasting the first cut in March and others predicting June."
            ),
            summary_type=SummaryType.HEADLINE,
            target_length=SummaryLength.VERY_SHORT,
            key_points=[
                "Fed holds rates steady",
                "inflation moderating",
                "markets react positively",
            ],
            reference_summaries=[
                "Federal Reserve Holds Rates Steady as Inflation Moderates, Markets Rally"
            ],
        ))

        # --- Prompt 4: Health/medical ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Researchers at the University of Cambridge have developed a novel "
                "blood test that can detect multiple types of cancer at early "
                "stages with over 90 percent accuracy. The test, described in a "
                "paper published in Science Translational Medicine, works by "
                "analyzing cell-free DNA fragments circulating in the bloodstream. "
                "These fragments carry distinctive methylation patterns that vary "
                "depending on the tissue of origin and the presence of cancerous "
                "cells. In a clinical trial involving 5,400 participants, the test "
                "successfully identified cancers of the lung, breast, colon, "
                "pancreas, and ovary, including many cases that had not yet been "
                "detected through conventional screening methods. The false "
                "positive rate was below 1 percent, a significant improvement "
                "over existing liquid biopsy technologies. Lead researcher Dr. "
                "Sarah Chen noted that the test could be particularly valuable "
                "for detecting pancreatic and ovarian cancers, which are often "
                "diagnosed at advanced stages due to the absence of early "
                "symptoms. The team plans to seek regulatory approval and hopes "
                "to make the test available within three years. Health policy "
                "experts suggest that widespread adoption could significantly "
                "reduce cancer mortality rates by enabling earlier treatment "
                "interventions."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "blood test for early cancer detection",
                "90 percent accuracy",
                "cell-free DNA analysis",
                "detects lung, breast, colon, pancreatic, ovarian cancers",
                "low false positive rate",
            ],
            reference_summaries=[
                "Cambridge researchers have created a blood test that detects five "
                "types of cancer at early stages with over 90% accuracy by analyzing "
                "cell-free DNA methylation patterns. In a trial of 5,400 patients, "
                "the test showed less than 1% false positives and caught cancers "
                "missed by conventional screening. Regulatory approval is expected "
                "within three years."
            ],
        ))

        # --- Prompt 5: Sports ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "In a dramatic finish to the championship final, the underdog "
                "team staged a remarkable comeback from a 21-point deficit in "
                "the fourth quarter to win by three points. The victory marks "
                "only the second time in tournament history that a team has "
                "overcome a 20-plus point deficit in the final period. Star "
                "guard Marcus Thompson led the charge with 18 points in the "
                "fourth quarter alone, including five three-pointers that "
                "silenced the previously raucous home crowd. Head coach Linda "
                "Martinez credited the team's defensive adjustments at halftime "
                "for limiting the opposition to just 12 points in the second "
                "half. The win also extended the team's remarkable unbeaten "
                "streak in playoff games to 14, a new franchise record. "
                "Post-game analysis highlighted the pivotal role of bench "
                "players, who contributed 35 points and provided fresh legs "
                "during the crucial final minutes. The championship trophy "
                "presentation was followed by celebrations that lasted well "
                "into the early hours. League commissioner David Park called "
                "it one of the greatest championship games in recent memory."
            ),
            summary_type=SummaryType.BULLET_POINTS,
            target_length=SummaryLength.SHORT,
            key_points=[
                "comeback from 21-point deficit",
                "Thompson scored 18 in fourth quarter",
                "defensive adjustments at halftime",
                "bench players contributed 35 points",
            ],
            reference_summaries=[
                "• Underdog team overcame a 21-point fourth-quarter deficit\n"
                "• Marcus Thompson scored 18 points in the final quarter\n"
                "• Defensive adjustments limited opponents to 12 second-half points\n"
                "• Bench players contributed 35 points in the comeback win"
            ],
        ))

        # --- Prompt 6: Space exploration ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "NASA's latest Mars rover mission has uncovered compelling "
                "evidence of ancient microbial life in rock samples collected "
                "from the Jezero Crater. The samples, drilled from sedimentary "
                "rock formations believed to be remnants of an ancient lake bed, "
                "contain organic molecules and mineral structures consistent "
                "with biological processes. The rover's onboard spectrometer "
                "identified complex carbon compounds, including amino acid "
                "precursors, embedded within the rock matrix. Scientists caution "
                "that while the findings are highly suggestive, definitive "
                "confirmation will require the samples to be returned to Earth "
                "for analysis in terrestrial laboratories. The sample return "
                "mission, a joint effort between NASA and the European Space "
                "Agency, is scheduled for the early 2030s. The discovery has "
                "reignited debate about the possibility of past life on Mars "
                "and its implications for astrobiology. Several independent "
                "research teams have already begun reanalyzing existing orbital "
                "data from the Jezero Crater region in light of the new findings."
            ),
            summary_type=SummaryType.EXECUTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "evidence of ancient microbial life on Mars",
                "organic molecules in Jezero Crater samples",
                "sample return mission planned for 2030s",
                "findings need Earth-based confirmation",
            ],
        ))

        # --- Prompt 7: Education policy ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The Department of Education released a comprehensive report "
                "evaluating the impact of remote learning during and after the "
                "pandemic on student academic performance. The report, based on "
                "standardized test scores from over 8 million students across "
                "all 50 states, found that average math scores declined by 7 "
                "percent and reading scores declined by 4 percent compared to "
                "pre-pandemic levels. The declines were most severe among students "
                "from low-income families and those in rural areas with limited "
                "internet access. Notably, the report also found that students "
                "who participated in hybrid learning models — combining in-person "
                "and online instruction — performed significantly better than "
                "those who were fully remote. The department recommended a set of "
                "interventions including expanded tutoring programs, summer "
                "enrichment initiatives, and increased investment in broadband "
                "infrastructure for underserved communities. Several states have "
                "already begun implementing recovery programs, with early results "
                "showing modest gains in math proficiency among targeted student "
                "populations."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "remote learning caused test score declines",
                "math down 7%, reading down 4%",
                "low-income and rural students most affected",
                "hybrid model better than fully remote",
            ],
        ))

        # --- Prompt 8: Cybersecurity ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A major international cybersecurity firm disclosed today that it "
                "has discovered a sophisticated supply chain attack affecting "
                "widely used enterprise software. The attack, which went "
                "undetected for approximately nine months, compromised a popular "
                "code library used by an estimated 18,000 organizations worldwide. "
                "The threat actors inserted a backdoor into the library during a "
                "routine software update, allowing them to gain remote access to "
                "systems that incorporated the compromised component. Initial "
                "forensic analysis suggests the attack originated from a state- "
                "sponsored group, though attribution remains preliminary. The "
                "cybersecurity firm has worked with affected vendors to issue "
                "emergency patches and is providing free incident response "
                "support to impacted organizations. Government agencies in "
                "multiple countries have issued alerts urging organizations to "
                "audit their software supply chains and apply the available "
                "patches immediately. Security researchers warn that the full "
                "extent of the breach may not be known for months as "
                "investigators continue to analyze compromised systems."
            ),
            summary_type=SummaryType.TLDR,
            target_length=SummaryLength.SHORT,
            key_points=[
                "supply chain attack on enterprise software",
                "18,000 organizations affected",
                "backdoor inserted during routine update",
                "likely state-sponsored",
            ],
        ))

        # --- Prompt 9: Environmental regulation ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The Environmental Protection Agency has finalized new regulations "
                "targeting per- and polyfluoroalkyl substances, commonly known as "
                "PFAS or forever chemicals, in drinking water. The rule sets "
                "legally enforceable maximum contaminant levels for six PFAS "
                "compounds at 4 parts per trillion, the lowest detectable "
                "threshold with current testing technology. Water utilities "
                "serving more than 10,000 customers will have three years to "
                "comply, while smaller systems will have five years. The EPA "
                "estimates the regulation will prevent approximately 10,000 "
                "cancer cases and reduce the incidence of other serious health "
                "conditions linked to PFAS exposure, including thyroid disease "
                "and developmental delays in children. The compliance cost for "
                "water utilities is estimated at $1.5 billion annually, which "
                "the EPA says will be partially offset by $9 billion in federal "
                "funding allocated through recent infrastructure legislation. "
                "Industry groups have pushed back, arguing the limits are "
                "technologically infeasible for many water systems, while "
                "environmental advocates have praised the rule as long overdue."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "EPA sets PFAS limits in drinking water",
                "4 parts per trillion for six compounds",
                "expected to prevent 10,000 cancer cases",
                "compliance cost $1.5 billion annually",
            ],
        ))

        # --- Prompt 10: Automotive industry ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Global electric vehicle sales surpassed 14 million units in 2023, "
                "representing a 35 percent increase over the previous year and "
                "accounting for approximately 18 percent of all new car sales "
                "worldwide. China remained the dominant market, with over 8 million "
                "EVs sold domestically and Chinese manufacturers increasingly "
                "exporting to Europe, Southeast Asia, and Latin America. European "
                "sales grew by 20 percent, driven largely by stricter emissions "
                "standards and generous government subsidies in Germany, France, "
                "and the Nordic countries. The United States saw a 40 percent "
                "increase in EV adoption, though from a smaller base, with the "
                "market share reaching 9 percent. Analysts attribute the growth "
                "to falling battery costs, which have declined by 14 percent "
                "year-over-year, along with expanding charging infrastructure and "
                "an increasingly diverse model lineup from both legacy automakers "
                "and new entrants. Challenges remain, including supply chain "
                "constraints for critical minerals such as lithium, cobalt, and "
                "nickel, as well as concerns about grid capacity in regions with "
                "rapid EV adoption. Industry forecasters predict EVs will "
                "represent 25 to 30 percent of global new car sales by 2026."
            ),
            summary_type=SummaryType.BULLET_POINTS,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "14 million EVs sold in 2023",
                "35% year-over-year growth",
                "China dominates with 8 million",
                "battery costs declined 14%",
            ],
        ))

        # --- Prompt 11: International diplomacy ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Representatives from 45 nations concluded a historic trade "
                "agreement after three years of negotiations. The pact eliminates "
                "tariffs on over 80 percent of goods traded between member "
                "countries and establishes common standards for digital commerce, "
                "intellectual property protection, and labor rights. Economists "
                "estimate the agreement will boost combined GDP of member nations "
                "by $900 billion over the next decade. The agreement also includes "
                "unprecedented environmental provisions, requiring signatories "
                "to meet specific carbon emission reduction targets as a condition "
                "of membership. Small and medium enterprises stand to benefit "
                "significantly from simplified customs procedures and reduced "
                "paperwork. However, agricultural sectors in several countries "
                "have expressed concerns about increased competition from lower- "
                "cost producers. The agreement will enter into force once "
                "ratified by at least 30 of the 45 signatory nations."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "45-nation trade agreement",
                "eliminates 80% of tariffs",
                "environmental provisions included",
                "$900 billion GDP boost estimated",
            ],
        ))

        # --- Prompt 12: Technology product launch ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A leading semiconductor company unveiled its next-generation "
                "processor architecture at the annual technology conference today. "
                "The new chip, fabricated using a 2-nanometer process node, "
                "delivers a 40 percent improvement in performance per watt "
                "compared to its predecessor. The architecture introduces a "
                "heterogeneous computing approach with dedicated AI acceleration "
                "cores alongside traditional CPU and GPU clusters. Memory "
                "bandwidth has been doubled through the adoption of a new high- "
                "bandwidth memory standard, enabling faster data processing for "
                "large-scale machine learning workloads. The company claims the "
                "chip can train large language models up to three times faster "
                "than current solutions while consuming 30 percent less power. "
                "Initial partners include several major cloud computing providers "
                "who plan to deploy the chips in their data centers by mid-2025. "
                "The announcement sent the company's stock price up 8 percent in "
                "after-hours trading."
            ),
            summary_type=SummaryType.HEADLINE,
            target_length=SummaryLength.VERY_SHORT,
            key_points=[
                "2nm processor unveiled",
                "40% better performance per watt",
                "dedicated AI cores",
                "3x faster LLM training",
            ],
        ))

        # --- Prompt 13: Legal / Regulatory ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The Supreme Court issued a landmark ruling today that will "
                "reshape how technology companies handle user data across state "
                "lines. In a 6-3 decision, the court held that a patchwork of "
                "conflicting state privacy laws creates an undue burden on "
                "interstate commerce and that a federal standard should preempt "
                "state regulations where they conflict. The ruling arose from a "
                "case brought by a coalition of technology firms challenging a "
                "California privacy law that imposed strict data localization "
                "requirements. Justice Williams, writing for the majority, "
                "argued that the digital economy requires uniform rules to "
                "function efficiently. The dissent, led by Justice Park, warned "
                "that the ruling could weaken consumer protections in states "
                "that had enacted stronger privacy measures. Privacy advocates "
                "expressed concern that Congress has not yet passed comprehensive "
                "federal privacy legislation, creating a potential regulatory "
                "gap. Legal scholars predict the decision will accelerate "
                "legislative efforts to pass a national privacy law."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "Supreme Court rules on data privacy",
                "federal standard preempts state laws",
                "6-3 decision",
                "may accelerate federal privacy legislation",
            ],
        ))

        # --- Prompt 14: Public health ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The World Health Organization declared an end to the global "
                "mpox health emergency after new case counts fell below critical "
                "thresholds in all monitored regions. At its peak, the outbreak "
                "affected 116 countries and resulted in over 90,000 confirmed "
                "cases. The WHO credited a combination of vaccination campaigns, "
                "public awareness efforts, and community engagement for bringing "
                "the outbreak under control. The organization cautioned, however, "
                "that surveillance must continue as the virus remains endemic "
                "in certain regions of Central and West Africa. Vaccination "
                "coverage in the most affected populations reached approximately "
                "70 percent, a level epidemiologists consider sufficient to "
                "prevent large-scale resurgences. The WHO also announced it "
                "would be releasing updated guidance on post-exposure "
                "prophylaxis protocols and called on member states to maintain "
                "adequate vaccine stockpiles. Public health experts noted that "
                "the response to the outbreak demonstrated the value of rapid "
                "international coordination but also highlighted persistent "
                "inequities in vaccine access between high- and low-income "
                "countries."
            ),
            summary_type=SummaryType.EXECUTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "WHO ends mpox emergency",
                "90,000 cases across 116 countries",
                "vaccination reached 70% in key populations",
                "continued surveillance recommended",
            ],
        ))

        # --- Prompt 15: Financial markets ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Cryptocurrency markets experienced a sharp correction this week "
                "as Bitcoin fell 15 percent from its all-time high, dragging "
                "the broader digital asset market down by over $300 billion in "
                "total market capitalization. The sell-off was triggered by "
                "reports that a major cryptocurrency exchange is under "
                "investigation by federal regulators for allegedly commingling "
                "customer funds with proprietary trading accounts. The exchange "
                "denied the allegations and stated it is fully cooperating with "
                "authorities. Other major cryptocurrencies including Ethereum "
                "and Solana experienced declines of 12 and 22 percent "
                "respectively. Analysts note that the correction comes after a "
                "period of rapid appreciation fueled by the approval of spot "
                "Bitcoin exchange-traded funds earlier this year. Despite the "
                "short-term volatility, institutional investors appear to be "
                "maintaining their positions, with net inflows into Bitcoin "
                "ETFs remaining positive for the week. Market commentators "
                "suggest the correction may represent a healthy consolidation "
                "rather than a fundamental shift in market sentiment."
            ),
            summary_type=SummaryType.TLDR,
            target_length=SummaryLength.SHORT,
            key_points=[
                "Bitcoin fell 15% from all-time high",
                "$300 billion market cap lost",
                "exchange investigation triggered sell-off",
                "institutional investors holding positions",
            ],
        ))

        return prompts

    def _generate_scientific_summarization_prompts(self) -> List[SummarizationPrompt]:
        """Generate summarization prompts from scientific/academic sources."""
        prompts: List[SummarizationPrompt] = []

        # --- Prompt 16: Neuroscience ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A groundbreaking study published in Nature Neuroscience has "
                "identified a previously unknown neural pathway that plays a "
                "critical role in the consolidation of long-term memories during "
                "sleep. Researchers at the Max Planck Institute used advanced "
                "optogenetic techniques to selectively activate and silence "
                "neurons in the hippocampus and prefrontal cortex of mice during "
                "different stages of sleep. They found that a specific population "
                "of inhibitory interneurons in the hippocampus generates "
                "precisely timed bursts of activity during slow-wave sleep that "
                "coordinate with sharp-wave ripples to transfer memory traces "
                "to the cortex. When these interneurons were silenced, mice "
                "showed significant impairments in spatial memory tasks the "
                "following day, despite sleeping for normal durations. "
                "Conversely, enhancing the activity of these neurons improved "
                "memory performance by approximately 25 percent. The findings "
                "suggest that the timing and coordination of inhibitory signals, "
                "rather than simply the amount of sleep, is crucial for memory "
                "consolidation. The researchers believe this pathway could be "
                "a therapeutic target for conditions involving memory impairment, "
                "including Alzheimer's disease and age-related cognitive decline."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "new neural pathway for memory consolidation",
                "inhibitory interneurons in hippocampus",
                "timing during slow-wave sleep is crucial",
                "potential therapeutic target for Alzheimer's",
            ],
        ))

        # --- Prompt 17: Quantum computing ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Physicists at a national laboratory have achieved a significant "
                "milestone in quantum error correction by demonstrating a logical "
                "qubit that maintains coherence for over 10 seconds, roughly 100 "
                "times longer than previous records. The team used a novel "
                "topological encoding scheme that distributes quantum information "
                "across 49 physical qubits arranged in a surface code "
                "configuration. Error syndromes are continuously monitored by "
                "ancilla qubits positioned at the boundaries of the code, "
                "allowing real-time correction of both bit-flip and phase-flip "
                "errors. The key innovation lies in a new decoding algorithm "
                "that processes syndrome measurements with a latency of just "
                "200 nanoseconds, fast enough to correct errors before they "
                "propagate through the system. The researchers demonstrated "
                "that the logical error rate decreases exponentially as the "
                "code distance increases, a critical threshold known as the "
                "break-even point that many experts considered years away. "
                "The result has been hailed as a turning point for practical "
                "quantum computing, as reliable error correction is widely "
                "regarded as the biggest obstacle to building fault-tolerant "
                "quantum machines capable of solving commercially relevant "
                "problems. Several quantum computing companies have already "
                "expressed interest in licensing the decoding algorithm."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.LONG,
            key_points=[
                "logical qubit coherent for 10 seconds",
                "100x improvement over previous records",
                "surface code with 49 physical qubits",
                "new decoding algorithm at 200ns latency",
                "exponential decrease in logical error rate",
            ],
        ))

        # --- Prompt 18: Materials science ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Engineers at MIT have developed a new class of biodegradable "
                "plastics derived entirely from plant-based feedstocks that "
                "match the mechanical properties of conventional petroleum- "
                "based polymers. The material, a modified form of polyhydroxy "
                "butyrate (PHB), has been engineered through a combination of "
                "metabolic pathway optimization in bacteria and a novel "
                "cross-linking chemistry that improves tensile strength by 60 "
                "percent over standard bioplastics. Crucially, the material "
                "decomposes completely in standard composting conditions within "
                "12 weeks, leaving no microplastic residues. The production "
                "process uses agricultural waste as a carbon source, making it "
                "both cost-competitive and carbon-negative when lifecycle "
                "emissions are considered. Initial testing shows the material "
                "is suitable for food packaging, single-use containers, and "
                "agricultural films. A startup spun out from the research lab "
                "has secured $45 million in funding to scale production and "
                "expects to begin commercial manufacturing within 18 months."
            ),
            summary_type=SummaryType.BULLET_POINTS,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "biodegradable plastic from plant feedstocks",
                "60% stronger than standard bioplastics",
                "decomposes in 12 weeks",
                "uses agricultural waste",
                "$45 million startup funding",
            ],
        ))

        # --- Prompt 19: Genetics ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A large-scale genome-wide association study involving over "
                "2 million participants has identified 127 new genetic variants "
                "associated with susceptibility to type 2 diabetes. The study, "
                "conducted by an international consortium of researchers, "
                "combined data from 23 biobank cohorts spanning diverse "
                "ancestral populations across five continents. Previous studies "
                "had identified approximately 400 diabetes-associated loci, "
                "but the new findings nearly double the number of known genetic "
                "signals. Functional annotation reveals that many of the newly "
                "identified variants are located in regulatory regions that "
                "control gene expression in pancreatic beta cells and adipose "
                "tissue. A polygenic risk score incorporating the new variants "
                "can now identify individuals in the top 10 percent of genetic "
                "risk with approximately 80 percent accuracy, compared to 65 "
                "percent with the previous model. The researchers also found "
                "significant differences in risk allele frequencies across "
                "populations, helping to explain observed disparities in "
                "diabetes prevalence between ethnic groups. The study provides "
                "a foundation for developing more targeted prevention strategies "
                "and may guide the development of new therapeutic approaches."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "127 new genetic variants for type 2 diabetes",
                "2 million participants across 23 biobanks",
                "polygenic risk score improved to 80% accuracy",
                "population-specific risk differences found",
            ],
        ))

        # --- Prompt 20: Ecology ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A decade-long ecological study conducted across 12 national "
                "parks in the western United States has documented a dramatic "
                "shift in the distribution of keystone species driven by "
                "changing climate conditions. The research, led by ecologists "
                "at the University of Montana, tracked populations of 43 "
                "mammalian species using a network of over 5,000 camera traps "
                "and GPS-collared individuals. Results show that the average "
                "range boundary for studied species has shifted approximately "
                "11 kilometers northward and 120 meters upward in elevation per "
                "decade. Species that depend on specific temperature ranges, "
                "such as the American pika and certain bat species, have "
                "experienced the most dramatic range contractions, with some "
                "populations declining by over 50 percent at their historical "
                "southern boundaries. The study also found cascading effects "
                "on plant communities, as herbivore migration patterns changed "
                "seed dispersal dynamics. The researchers recommend expanding "
                "wildlife corridors to facilitate species movement and adjusting "
                "conservation management plans to account for the ongoing "
                "redistribution of biodiversity."
            ),
            summary_type=SummaryType.EXECUTIVE,
            target_length=SummaryLength.LONG,
            key_points=[
                "species ranges shifting northward and upward",
                "11 km north and 120 m upward per decade",
                "American pika most affected",
                "cascading effects on plant communities",
                "wildlife corridors recommended",
            ],
        ))

        # --- Prompt 21: Artificial intelligence research ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Researchers have proposed a new training methodology for large "
                "language models that significantly reduces the amount of "
                "labeled data required for fine-tuning on downstream tasks. "
                "The approach, called Iterative Self-Refinement with Minimal "
                "Supervision (ISRMS), uses a three-stage pipeline in which the "
                "model first generates candidate responses, then critiques its "
                "own outputs using a small set of exemplar-based prompts, and "
                "finally refines its responses based on the self-generated "
                "feedback. In experiments across 14 natural language processing "
                "benchmarks, ISRMS achieved performance within 2 percent of "
                "fully supervised fine-tuning while using only 5 percent of the "
                "labeled training data. The method is particularly effective "
                "for tasks requiring structured reasoning, such as mathematical "
                "problem solving and logical inference, where the self-critique "
                "step provides the most substantial improvement. The researchers "
                "also demonstrated that the approach is complementary to existing "
                "techniques such as reinforcement learning from human feedback "
                "and can be combined with them for additional gains. The work "
                "has implications for deploying language models in specialized "
                "domains where labeled data is scarce and expensive to obtain."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "new training method ISRMS",
                "reduces labeled data to 5%",
                "within 2% of fully supervised performance",
                "self-critique improves structured reasoning",
            ],
        ))

        # --- Prompt 22: Astronomy ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The James Webb Space Telescope has captured detailed "
                "spectroscopic data from an exoplanet atmosphere that reveals "
                "unmistakable signatures of water vapor, carbon dioxide, and "
                "methane in a potentially habitable zone. The planet, designated "
                "K2-18b, orbits a red dwarf star approximately 120 light-years "
                "from Earth and has a mass roughly eight times that of Earth, "
                "placing it in the category of sub-Neptune exoplanets. The "
                "atmospheric composition is consistent with a hydrogen-rich "
                "atmosphere overlying a water ocean, a scenario that some "
                "astrobiologists consider favorable for the emergence of life. "
                "The detection of dimethyl sulfide, a molecule produced "
                "predominantly by biological processes on Earth, has generated "
                "particular excitement, though the research team emphasizes "
                "that the signal requires further confirmation. Additional "
                "observations are planned for the next observing cycle to "
                "constrain the abundance of these molecules more precisely "
                "and to search for additional biosignature gases. The findings "
                "represent the most detailed characterization of a potentially "
                "habitable exoplanet atmosphere to date."
            ),
            summary_type=SummaryType.HEADLINE,
            target_length=SummaryLength.VERY_SHORT,
            key_points=[
                "JWST detects water, CO2, methane on exoplanet",
                "K2-18b in habitable zone",
                "possible biosignature detected",
            ],
        ))

        # --- Prompt 23: Psychology ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A meta-analysis encompassing 312 studies and over 500,000 "
                "participants has challenged the widely held belief that "
                "multitasking always impairs cognitive performance. The "
                "analysis, published in Psychological Bulletin, found that the "
                "relationship between multitasking and performance is highly "
                "dependent on the nature and complexity of the tasks involved. "
                "For simple, well-practiced tasks with low cognitive demands, "
                "participants showed no significant performance decrement when "
                "multitasking, and in some cases even showed slight improvements "
                "attributed to increased arousal and engagement. However, for "
                "tasks requiring sustained attention, working memory, or "
                "creative problem-solving, performance declined by an average "
                "of 23 percent during multitasking conditions. The researchers "
                "also identified significant individual differences, with "
                "approximately 15 percent of participants consistently performing "
                "well across multitasking conditions, a group the researchers "
                "termed 'supertaskers.' Age was a moderating factor, with "
                "individuals over 55 showing larger performance decrements than "
                "younger adults. The findings have implications for workplace "
                "design, educational practices, and driving safety policies."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "multitasking effects depend on task complexity",
                "23% performance decline for complex tasks",
                "15% of people are supertaskers",
                "age is a moderating factor",
            ],
        ))

        # --- Prompt 24: Renewable energy ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A research team at Stanford University has announced a "
                "breakthrough in perovskite-silicon tandem solar cell efficiency, "
                "achieving a record 33.7 percent power conversion efficiency "
                "under standard test conditions. The cell combines a perovskite "
                "top layer that absorbs higher-energy photons with a silicon "
                "bottom cell that captures lower-energy light, allowing the "
                "tandem structure to harvest a broader spectrum of solar "
                "radiation than either material alone. The key advance was a "
                "new interface passivation technique that reduces recombination "
                "losses at the junction between the two materials by 80 percent "
                "compared to previous approaches. The researchers also addressed "
                "the long-standing stability problem of perovskite materials by "
                "incorporating a self-healing molecular layer that repairs "
                "degradation caused by moisture and ultraviolet exposure. "
                "Accelerated aging tests suggest the cells can maintain over "
                "95 percent of their initial efficiency after the equivalent "
                "of 25 years of outdoor operation. The efficiency record "
                "surpasses the theoretical single-junction silicon limit of "
                "approximately 29 percent, demonstrating that tandem "
                "architectures represent a viable path to next-generation "
                "photovoltaics."
            ),
            summary_type=SummaryType.TLDR,
            target_length=SummaryLength.SHORT,
            key_points=[
                "33.7% efficiency for perovskite-silicon tandem cell",
                "new interface passivation technique",
                "self-healing layer solves stability problem",
                "surpasses silicon single-junction limit",
            ],
        ))

        # --- Prompt 25: Biomedical engineering ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Biomedical engineers at Johns Hopkins University have developed "
                "a brain-computer interface that enables paralyzed patients to "
                "control a robotic arm with unprecedented precision using only "
                "their thoughts. The system uses a 1,024-electrode array "
                "implanted in the motor cortex to record neural signals at "
                "high spatial and temporal resolution. A custom deep learning "
                "algorithm decodes the neural activity in real time, translating "
                "intended movements into robotic arm commands with a latency of "
                "less than 50 milliseconds. In clinical trials involving 12 "
                "patients with spinal cord injuries, participants were able to "
                "perform complex manipulation tasks such as picking up small "
                "objects, pouring liquids, and using utensils with a success "
                "rate exceeding 95 percent. The system also demonstrated the "
                "ability to decode intended hand grasps and individual finger "
                "movements, a level of dexterity not previously achieved with "
                "implanted brain-computer interfaces. The device received "
                "breakthrough therapy designation from the FDA, which may "
                "expedite the regulatory approval process. The research team "
                "is now working on a wireless version that would eliminate the "
                "need for a percutaneous connector."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "brain-computer interface for paralyzed patients",
                "1,024-electrode array in motor cortex",
                "95% success rate in manipulation tasks",
                "individual finger movement decoding",
                "FDA breakthrough therapy designation",
            ],
        ))

        # --- Prompt 26: Climate modeling ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A new climate model developed by an international team of "
                "atmospheric scientists has achieved unprecedented resolution "
                "in simulating regional weather patterns and their interactions "
                "with global climate dynamics. The model, running on the world's "
                "fastest supercomputer, operates at a 1-kilometer grid spacing "
                "compared to the 25-100 kilometer grids used by conventional "
                "climate models. This resolution allows the model to explicitly "
                "simulate convective processes such as thunderstorms and tropical "
                "cyclones without relying on parameterization schemes that "
                "introduce significant uncertainties. Initial validation against "
                "satellite observations shows the high-resolution model reduces "
                "precipitation prediction errors by 40 percent in tropical "
                "regions and improves the representation of extreme rainfall "
                "events by a factor of three. The model has also revealed "
                "previously undetected feedback mechanisms between ocean "
                "mesoscale eddies and atmospheric boundary layer dynamics "
                "that may amplify regional warming trends. Computational costs "
                "remain a challenge, with a single century-long simulation "
                "requiring approximately 50 million core-hours."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "1-km resolution climate model",
                "explicitly simulates convection",
                "40% reduction in precipitation errors",
                "50 million core-hours per simulation",
            ],
        ))

        # --- Prompt 27: Pharmacology ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A phase III clinical trial has demonstrated that a new class "
                "of weight-loss medication produces sustained weight reduction "
                "of 22 percent of body weight over 72 weeks, the largest effect "
                "observed in any pharmacological obesity treatment to date. The "
                "drug, a dual GLP-1 and GIP receptor agonist administered as a "
                "once-weekly injection, was tested in a randomized, double-blind "
                "trial involving 3,700 participants with a BMI of 30 or higher. "
                "Participants receiving the active drug lost an average of 24 "
                "kilograms compared to 2.4 kilograms in the placebo group. "
                "The most common side effects were gastrointestinal in nature, "
                "including nausea (28 percent of participants), diarrhea (18 "
                "percent), and vomiting (12 percent), though these typically "
                "resolved within the first eight weeks of treatment. "
                "Cardiovascular risk markers including blood pressure, LDL "
                "cholesterol, and HbA1c also showed significant improvements. "
                "The manufacturer plans to submit the drug for regulatory "
                "approval in the first quarter of next year, with analysts "
                "projecting the global obesity drug market could reach $100 "
                "billion annually by 2030."
            ),
            summary_type=SummaryType.EXECUTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "22% body weight reduction over 72 weeks",
                "dual GLP-1 and GIP receptor agonist",
                "3,700-participant phase III trial",
                "gastrointestinal side effects common but transient",
                "regulatory submission planned next year",
            ],
        ))

        # --- Prompt 28: Computer science theory ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A team of theoretical computer scientists has proved that a "
                "certain class of optimization problems previously believed to "
                "require exponential time can be solved in polynomial time using "
                "a novel algorithmic framework based on semidefinite programming "
                "relaxations combined with rounding schemes inspired by algebraic "
                "topology. The result resolves a 30-year-old conjecture in "
                "computational complexity theory and has immediate practical "
                "implications for network design, resource allocation, and "
                "scheduling problems in operations research. The proof constructs "
                "an intricate sequence of reductions that transform the original "
                "problem into a series of convex optimization subproblems, each "
                "of which can be solved efficiently. The rounding scheme uses "
                "topological properties of high-dimensional polytopes to convert "
                "fractional solutions to integral solutions with a guaranteed "
                "approximation ratio of at most 1.01. The paper, which runs to "
                "over 200 pages, has been verified by three independent teams "
                "and is expected to receive significant attention at the upcoming "
                "symposium on theoretical computer science."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "polynomial-time algorithm for hard optimization problems",
                "resolves 30-year conjecture",
                "semidefinite programming with topological rounding",
                "1.01 approximation ratio",
            ],
        ))

        return prompts

    def _generate_document_summarization_prompts(self) -> List[SummarizationPrompt]:
        """Generate summarization prompts from documents and reports."""
        prompts: List[SummarizationPrompt] = []

        # --- Prompt 29: Business report ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The quarterly earnings report for the third quarter of fiscal "
                "year 2024 shows revenue of $42.3 billion, representing a 12 "
                "percent increase year-over-year and exceeding analyst consensus "
                "estimates by $1.8 billion. Cloud services revenue grew 29 "
                "percent to $18.7 billion, driven by strong enterprise adoption "
                "of generative AI workloads. The productivity and business "
                "processes segment generated $15.2 billion in revenue, up 8 "
                "percent, with particular strength in the commercial subscription "
                "tier. Operating income increased to $19.1 billion, yielding an "
                "operating margin of 45.2 percent, an improvement of 200 basis "
                "points from the prior year period. The company returned $9.8 "
                "billion to shareholders through dividends and share repurchases "
                "during the quarter. Management raised full-year revenue guidance "
                "by $2 billion to a range of $168 to $172 billion, citing "
                "continued momentum in AI-related services and a favorable "
                "foreign exchange environment. Capital expenditures totaled $11.2 "
                "billion, primarily directed toward expanding data center "
                "capacity to meet surging demand for AI inference and training "
                "infrastructure."
            ),
            summary_type=SummaryType.EXECUTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "$42.3B revenue, up 12% YoY",
                "cloud services grew 29%",
                "45.2% operating margin",
                "raised full-year guidance by $2B",
                "$11.2B capex for AI infrastructure",
            ],
        ))

        # --- Prompt 30: Policy white paper ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "This white paper examines the current state of housing "
                "affordability in major metropolitan areas and proposes a "
                "comprehensive framework for addressing the crisis. Median "
                "home prices have risen by 45 percent in the past five years, "
                "far outpacing wage growth of 18 percent over the same period. "
                "The ratio of median home price to median household income has "
                "reached 7.2 nationally, up from 4.5 two decades ago, with "
                "several coastal cities exceeding ratios of 12 to 1. The paper "
                "identifies three primary drivers of the affordability gap: "
                "restrictive zoning regulations that limit new housing "
                "construction, a shortage of skilled construction labor that has "
                "driven up building costs by 30 percent since 2019, and the "
                "increasing financialization of housing through institutional "
                "investor purchases that now account for 28 percent of single- "
                "family home transactions in some markets. Proposed solutions "
                "include reforming zoning laws to allow higher-density "
                "development, creating a national apprenticeship program for "
                "construction trades, implementing anti-speculation taxes on "
                "properties held for less than two years, and expanding the "
                "Low-Income Housing Tax Credit program. The paper estimates "
                "that a coordinated implementation of these measures could "
                "reduce the price-to-income ratio by 1.5 points within a decade."
            ),
            summary_type=SummaryType.BULLET_POINTS,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "home prices up 45% in 5 years",
                "price-to-income ratio at 7.2",
                "zoning, labor shortage, financialization as drivers",
                "proposed solutions include zoning reform and taxes",
            ],
        ))

        # --- Prompt 31: Technical documentation ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The distributed database system implements a novel consensus "
                "protocol that provides linearizable reads and writes with "
                "single-digit millisecond latency across geographically "
                "distributed data centers. The protocol extends the Raft "
                "consensus algorithm with a parallel commit optimization that "
                "allows transactions spanning multiple partitions to commit "
                "in a single round-trip when conflict detection indicates no "
                "contention. Under high-contention workloads, the system falls "
                "back to a two-phase commit protocol with an optimistic locking "
                "strategy. Benchmarks show the system achieves 2.1 million "
                "transactions per second on a 100-node cluster, with 99th "
                "percentile latency of 8.3 milliseconds for read-write "
                "transactions and 2.1 milliseconds for read-only transactions. "
                "Fault tolerance is maintained through synchronous replication "
                "to at least three replicas, with automatic leader election "
                "completing within 500 milliseconds of failure detection. The "
                "system supports online schema changes that do not require "
                "downtime and provides a SQL-compatible query interface with "
                "full support for secondary indexes, foreign keys, and stored "
                "procedures. Data is stored in a custom columnar format "
                "optimized for both OLTP and OLAP workloads, with automatic "
                "tiering between in-memory and disk-based storage based on "
                "access frequency."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "novel consensus protocol based on Raft",
                "2.1 million TPS on 100 nodes",
                "single-digit ms latency",
                "SQL-compatible with online schema changes",
            ],
        ))

        # --- Prompt 32: Historical analysis ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The industrial revolution in Britain during the late 18th "
                "and early 19th centuries fundamentally transformed economic "
                "production, social structures, and urban landscapes in ways "
                "that continue to shape the modern world. Prior to "
                "industrialization, the majority of the British population "
                "lived in rural areas and engaged in agricultural labor or "
                "cottage industries. The introduction of mechanized textile "
                "production, powered first by water wheels and later by steam "
                "engines, created a demand for concentrated labor forces near "
                "factories, driving unprecedented urbanization. Manchester's "
                "population grew from 25,000 in 1772 to over 300,000 by 1850. "
                "Working conditions in early factories were notoriously harsh, "
                "with 14-hour workdays common and child labor widespread. These "
                "conditions eventually gave rise to labor movements, trade "
                "unions, and progressive legislation including the Factory Acts "
                "of 1833 and 1844. The revolution also stimulated innovations "
                "in transportation, most notably the development of the railway "
                "network, which reduced travel times between major cities from "
                "days to hours and enabled the efficient distribution of goods "
                "across the country. Economists estimate that per capita income "
                "in Britain roughly doubled between 1780 and 1860, though the "
                "benefits were distributed unevenly, with factory owners and "
                "merchants accumulating significant wealth while many workers "
                "experienced declining living standards during the early decades "
                "of industrialization."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "mechanized production transformed Britain",
                "rapid urbanization",
                "harsh working conditions led to labor movements",
                "railway network development",
                "uneven distribution of economic gains",
            ],
        ))

        # --- Prompt 33: Environmental impact assessment ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The environmental impact assessment for the proposed offshore "
                "wind farm in the North Sea evaluates potential effects on "
                "marine ecosystems, bird populations, fishing activities, and "
                "visual amenity. The project would install 120 turbines with "
                "a combined capacity of 1.8 gigawatts across a 340 square "
                "kilometer area located 45 kilometers from the nearest "
                "coastline. Underwater noise modeling indicates that pile "
                "driving during the 18-month construction phase could disturb "
                "marine mammals, particularly harbor porpoises and grey seals, "
                "within a radius of approximately 15 kilometers. Mitigation "
                "measures including seasonal construction restrictions during "
                "peak calving periods, bubble curtains to reduce noise "
                "propagation, and dedicated marine mammal observers are "
                "proposed. Bird collision risk assessment, using radar tracking "
                "data from three years of baseline surveys, estimates an "
                "annual mortality of 200 to 400 seabirds, primarily gannets "
                "and lesser black-backed gulls. The assessment concludes that "
                "this level of additional mortality would not significantly "
                "affect population viability. Fishing displacement analysis "
                "shows approximately 12 percent of trawling effort in the "
                "region would need to be relocated, affecting 34 vessels. "
                "Compensation schemes have been developed in consultation "
                "with the fishing industry. Visual impact modeling demonstrates "
                "the turbines would be barely visible from shore under typical "
                "atmospheric conditions."
            ),
            summary_type=SummaryType.EXECUTIVE,
            target_length=SummaryLength.LONG,
            key_points=[
                "1.8 GW offshore wind farm, 120 turbines",
                "marine mammal disturbance during construction",
                "200-400 seabird deaths annually",
                "12% fishing effort displaced",
                "mitigation measures proposed",
            ],
        ))

        # --- Prompt 34: Philosophy/Ethics essay ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The ethics of artificial intelligence decision-making in "
                "criminal justice systems presents a profound tension between "
                "efficiency and fairness. Proponents argue that algorithmic "
                "risk assessment tools can reduce human bias in bail, "
                "sentencing, and parole decisions by applying consistent "
                "criteria to every case. Studies have shown that human judges "
                "are influenced by factors such as the time of day, their "
                "mood, and the order in which cases are presented, introducing "
                "an element of arbitrariness that algorithms could theoretically "
                "eliminate. However, critics point out that these algorithms "
                "are trained on historical data that reflects decades of "
                "systemic racial and socioeconomic disparities in policing "
                "and prosecution. When an algorithm learns from biased data, "
                "it inevitably reproduces and potentially amplifies those "
                "biases, creating a feedback loop of discriminatory outcomes. "
                "The opacity of many machine learning models compounds the "
                "problem, as defendants may be denied meaningful explanations "
                "of the factors that contributed to decisions affecting their "
                "liberty. Philosophers of technology have proposed various "
                "frameworks for navigating these tensions, including procedural "
                "fairness approaches that focus on the transparency and "
                "contestability of algorithmic decisions, and substantive "
                "fairness approaches that require algorithms to produce "
                "equitable outcomes across demographic groups regardless of "
                "the accuracy trade-offs involved."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "AI in criminal justice: efficiency vs fairness",
                "algorithms can reduce human bias",
                "but trained on historically biased data",
                "opacity limits accountability",
                "procedural and substantive fairness frameworks",
            ],
        ))

        # --- Prompt 35: Medical case study ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A 62-year-old male presented to the emergency department with "
                "acute onset chest pain radiating to the left arm, accompanied "
                "by diaphoresis and shortness of breath. The patient had a "
                "medical history significant for hypertension, type 2 diabetes "
                "mellitus, and hyperlipidemia, with a 30-pack-year smoking "
                "history. Initial electrocardiogram showed ST-segment elevation "
                "in leads V1 through V4, consistent with an acute anterior "
                "ST-elevation myocardial infarction. Troponin levels were "
                "elevated at 2.4 ng/mL. The patient was immediately treated "
                "with aspirin, clopidogrel, and heparin, and was taken for "
                "emergent percutaneous coronary intervention. Coronary "
                "angiography revealed a 95 percent occlusion of the left "
                "anterior descending artery, which was successfully treated "
                "with balloon angioplasty and placement of a drug-eluting "
                "stent. Post-procedure echocardiography showed an ejection "
                "fraction of 40 percent with hypokinesis of the anterior wall. "
                "The patient was started on dual antiplatelet therapy, a beta- "
                "blocker, an ACE inhibitor, and a high-intensity statin. "
                "Cardiac rehabilitation was recommended upon discharge. At "
                "three-month follow-up, the patient reported no recurrence of "
                "symptoms and echocardiography showed improvement in ejection "
                "fraction to 50 percent."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "62-year-old male with STEMI",
                "95% LAD occlusion",
                "successful PCI with drug-eluting stent",
                "EF improved from 40% to 50% at follow-up",
            ],
        ))

        # --- Prompt 36: Software architecture ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The microservices migration strategy for the legacy monolithic "
                "e-commerce platform is designed to be executed over four "
                "phases spanning 18 months. Phase one focuses on extracting "
                "the user authentication and authorization service, which "
                "currently handles 2.3 million login events daily and represents "
                "a critical dependency for all other system components. The "
                "extracted service will implement OAuth 2.0 with OpenID Connect "
                "and will be deployed on a Kubernetes cluster with auto-scaling "
                "capabilities. Phase two addresses the product catalog and "
                "inventory management subsystem, transitioning from a "
                "relational database to an event-sourced architecture with "
                "CQRS to improve read performance and enable real-time "
                "inventory tracking across multiple warehouses. Phase three "
                "tackles the order processing pipeline, introducing an "
                "asynchronous message queue to decouple order creation from "
                "payment processing and fulfillment operations. Phase four "
                "completes the migration by decomposing the remaining "
                "functionality including customer support, analytics, and "
                "recommendation services. Each phase includes a two-month "
                "stabilization period during which the old and new systems "
                "run in parallel with traffic gradually shifted to the new "
                "service. The estimated total cost is $4.2 million, with "
                "projected savings of $1.8 million annually in infrastructure "
                "and operational costs."
            ),
            summary_type=SummaryType.BULLET_POINTS,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "4-phase migration over 18 months",
                "auth service extracted first",
                "event-sourced architecture for catalog",
                "async message queue for orders",
                "$4.2M cost, $1.8M annual savings",
            ],
        ))

        # --- Prompt 37: Nutrition science ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A randomized controlled trial published in The Lancet has "
                "found that intermittent fasting produces comparable weight "
                "loss and metabolic improvements to traditional caloric "
                "restriction over a 12-month period, but with significantly "
                "higher adherence rates. The study enrolled 1,200 overweight "
                "adults and randomly assigned them to three groups: time- "
                "restricted eating with an 8-hour feeding window, alternate- "
                "day fasting with 500-calorie fast days, and a standard "
                "25-percent caloric restriction diet. At 12 months, all three "
                "groups achieved similar average weight loss of approximately "
                "7 to 8 percent of initial body weight. However, the "
                "adherence rate was 78 percent in the time-restricted eating "
                "group compared to 62 percent in the caloric restriction "
                "group and 55 percent in the alternate-day fasting group. "
                "Blood biomarkers showed comparable improvements in fasting "
                "glucose, insulin sensitivity, triglycerides, and inflammatory "
                "markers across all three groups among those who adhered to "
                "their assigned protocol. The researchers concluded that the "
                "choice of dietary approach should be guided primarily by "
                "individual preference and likelihood of long-term adherence "
                "rather than by the specific dietary mechanism."
            ),
            summary_type=SummaryType.TLDR,
            target_length=SummaryLength.VERY_SHORT,
            key_points=[
                "intermittent fasting vs caloric restriction",
                "similar weight loss (~7-8%)",
                "time-restricted eating had highest adherence (78%)",
                "individual preference should guide choice",
            ],
        ))

        # --- Prompt 38: Urban planning ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The city's comprehensive transportation plan proposes a $12 "
                "billion investment over 15 years to transform mobility "
                "infrastructure and reduce car dependency by 40 percent. The "
                "centerpiece is a 65-kilometer rapid transit network connecting "
                "the downtown core with six suburban centers, featuring "
                "electric buses operating in dedicated lanes with signal "
                "priority at intersections. The plan also calls for the "
                "construction of 200 kilometers of protected cycling lanes, "
                "a fourfold increase from the current network, along with "
                "secure bike parking at all transit stations and major "
                "employment centers. Pedestrian infrastructure improvements "
                "include widened sidewalks, enhanced street lighting, and the "
                "creation of 15 car-free zones in the city center totaling 8 "
                "square kilometers. A congestion pricing system is proposed "
                "for the central business district, with revenues dedicated "
                "to funding transit operations. The plan projects a reduction "
                "in transportation-related carbon emissions of 55 percent by "
                "2040 and estimates that the investments will generate $2.3 "
                "billion in annual economic benefits through reduced commute "
                "times, lower healthcare costs from improved air quality, and "
                "increased property values near transit corridors."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "$12 billion over 15 years",
                "65-km rapid transit network",
                "200 km protected cycling lanes",
                "congestion pricing proposed",
                "55% carbon emission reduction by 2040",
            ],
        ))

        # --- Prompt 39: Linguistics ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A comprehensive survey of the world's languages has found that "
                "the rate of language extinction is accelerating, with an "
                "estimated 40 percent of the approximately 7,000 languages "
                "currently spoken expected to fall silent by the end of this "
                "century. The study, conducted by an international team of "
                "linguists over a decade, documented 2,800 endangered languages "
                "across all inhabited continents. The most critically endangered "
                "languages are concentrated in regions experiencing rapid "
                "urbanization, including Papua New Guinea, sub-Saharan Africa, "
                "and the Amazon basin. The researchers identified several key "
                "factors driving language loss: migration to cities where "
                "dominant languages are required for economic participation, "
                "educational policies that do not support minority language "
                "instruction, and the increasing dominance of a small number "
                "of languages in digital media and technology. The study "
                "recommends a set of interventions including community-based "
                "language documentation programs, bilingual education "
                "initiatives, and the development of digital tools and "
                "resources in endangered languages. The researchers emphasize "
                "that each language represents a unique repository of cultural "
                "knowledge, ecological understanding, and cognitive diversity "
                "that cannot be recovered once lost."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=[
                "40% of 7,000 languages may disappear this century",
                "urbanization and digital dominance are key drivers",
                "Papua New Guinea, Africa, Amazon most affected",
                "language documentation and bilingual education recommended",
            ],
        ))

        # --- Prompt 40: Robotics ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A team of roboticists at ETH Zurich has developed a new class "
                "of soft robots inspired by the biomechanics of octopus "
                "tentacles that can navigate complex, unstructured environments "
                "with remarkable agility. The robots are constructed from a "
                "proprietary silicone elastomer embedded with a network of "
                "pneumatic actuators that allow continuous deformation and "
                "shape adaptation. Unlike rigid robots that rely on predefined "
                "joint configurations, the soft robots can squeeze through "
                "openings one-third of their nominal diameter, wrap around "
                "irregularly shaped objects, and distribute contact forces "
                "evenly to avoid damaging fragile items. Control is achieved "
                "through a novel reinforcement learning algorithm that "
                "processes inputs from 256 distributed pressure sensors "
                "embedded in the robot's skin, enabling tactile-guided "
                "manipulation without relying on visual feedback. In testing "
                "scenarios, the robots successfully performed search and rescue "
                "simulations in collapsed building environments, agricultural "
                "harvesting of delicate fruits, and minimally invasive medical "
                "procedures on phantom tissue models. The research team has "
                "filed patents on both the material composition and the control "
                "algorithm and is in discussions with several potential "
                "commercial partners."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "octopus-inspired soft robots",
                "can squeeze through small openings",
                "reinforcement learning with tactile sensors",
                "applications in rescue, agriculture, medicine",
            ],
        ))

        # --- Prompt 41: Energy storage ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Researchers at Argonne National Laboratory have demonstrated "
                "a solid-state battery technology that achieves an energy "
                "density of 500 watt-hours per kilogram, roughly double that "
                "of current lithium-ion batteries. The battery uses a lithium "
                "metal anode paired with a sulfide-based solid electrolyte that "
                "eliminates the flammable liquid electrolytes responsible for "
                "thermal runaway events in conventional batteries. The key "
                "breakthrough is a proprietary interface coating that prevents "
                "the formation of lithium dendrites during charging, a problem "
                "that has plagued solid-state battery research for decades. "
                "The prototype cells maintained 92 percent capacity retention "
                "after 1,000 charge-discharge cycles at room temperature, "
                "meeting automotive industry durability requirements. The "
                "researchers demonstrated that the battery can be charged to "
                "80 percent capacity in 12 minutes using fast-charging "
                "protocols. Manufacturing scalability remains the primary "
                "challenge, as the sulfide electrolyte requires processing "
                "in dry-room environments. A partnership with a major "
                "automotive manufacturer has been established to develop "
                "pilot production lines with the goal of producing battery "
                "packs for electric vehicles by 2028."
            ),
            summary_type=SummaryType.TLDR,
            target_length=SummaryLength.SHORT,
            key_points=[
                "500 Wh/kg solid-state battery",
                "double current lithium-ion density",
                "no dendrite formation",
                "92% capacity after 1000 cycles",
                "EV deployment targeted for 2028",
            ],
        ))

        # --- Prompt 42: Education technology ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "A large-scale study involving 15,000 university students "
                "across 42 institutions has evaluated the effectiveness of "
                "AI-powered tutoring systems in introductory STEM courses. "
                "Students who used the adaptive tutoring platform for at "
                "least three hours per week showed a 0.4 standard deviation "
                "improvement in final exam scores compared to students in "
                "traditional lecture-only sections, an effect size equivalent "
                "to moving from the 50th to the 66th percentile. The system "
                "uses a knowledge graph of over 12,000 concepts to model each "
                "student's understanding and generates personalized practice "
                "problems and explanations tailored to identified knowledge "
                "gaps. Notably, the achievement gap between students from "
                "underrepresented backgrounds and their peers narrowed by "
                "35 percent among those using the platform, suggesting that "
                "personalized instruction may help level the playing field. "
                "Student satisfaction surveys indicated high engagement, with "
                "82 percent of users reporting that the AI tutor was more "
                "responsive to their individual learning needs than human "
                "teaching assistants. Faculty adoption has been mixed, with "
                "some instructors embracing the technology as a complement "
                "to their teaching and others expressing concerns about over- "
                "reliance on automated systems and the potential erosion of "
                "student-instructor relationships."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "AI tutoring improved exam scores by 0.4 SD",
                "15,000 students across 42 institutions",
                "achievement gap narrowed by 35%",
                "82% student satisfaction",
                "mixed faculty adoption",
            ],
        ))

        # --- Prompt 43: Geopolitics ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "The annual Global Risk Report identifies five emerging threats "
                "that could fundamentally alter the international security "
                "landscape over the next decade. First, the weaponization of "
                "artificial intelligence in autonomous weapons systems has "
                "outpaced international regulatory frameworks, creating risks "
                "of unintended escalation in conflict zones. Second, the "
                "increasing concentration of critical mineral supply chains "
                "in a small number of countries creates strategic "
                "vulnerabilities for nations dependent on imported resources "
                "for their technology and defense industries. Third, the "
                "proliferation of deepfake technology threatens democratic "
                "processes by enabling the creation of highly convincing "
                "disinformation at scale. Fourth, the growing gap between "
                "cyber offensive and defensive capabilities leaves critical "
                "infrastructure, including power grids and financial systems, "
                "vulnerable to state-sponsored attacks. Fifth, the cascading "
                "effects of climate change on food and water security in "
                "vulnerable regions are projected to drive mass migration "
                "and resource competition, potentially triggering conflicts "
                "in already fragile states. The report recommends multilateral "
                "cooperation frameworks and early warning systems as essential "
                "tools for managing these interconnected risks."
            ),
            summary_type=SummaryType.BULLET_POINTS,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "five emerging global security threats",
                "AI weapons, mineral concentration, deepfakes",
                "cyber vulnerabilities, climate-driven migration",
                "multilateral cooperation recommended",
            ],
        ))

        # --- Prompt 44: Marine biology ---
        prompts.append(SummarizationPrompt(
            source_document=(
                "Marine biologists conducting deep-sea surveys in the Mariana "
                "Trench have discovered 23 previously unknown species living "
                "at depths exceeding 8,000 meters. The discoveries include a "
                "translucent snailfish that holds the record for the deepest- "
                "living vertebrate ever observed at 8,336 meters, several "
                "species of amphipods with novel bioluminescent capabilities, "
                "and a giant isopod measuring over 50 centimeters in length. "
                "The organisms display remarkable adaptations to extreme "
                "pressure, near-freezing temperatures, and complete darkness, "
                "including specialized cellular membranes with high "
                "concentrations of trimethylamine N-oxide, a compound that "
                "stabilizes proteins under crushing pressures. Genetic analysis "
                "of collected specimens reveals that many of the new species "
                "diverged from their shallow-water relatives between 15 and "
                "40 million years ago, suggesting that the hadal zone has "
                "served as a refuge for ancient lineages. The expedition also "
                "documented the presence of microplastic particles at the "
                "deepest sampling sites, providing further evidence that "
                "human pollution has reached the most remote environments "
                "on Earth. The research team plans to return next year with "
                "improved sampling equipment to collect live specimens for "
                "laboratory study."
            ),
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.MEDIUM,
            key_points=[
                "23 new species found in Mariana Trench",
                "deepest vertebrate at 8,336 meters",
                "novel adaptations to extreme pressure",
                "microplastics found at deepest sites",
            ],
        ))

        return prompts


# ---------------------------------------------------------------------------
# Summarization Diversity Analyzer
# ---------------------------------------------------------------------------


class SummarizationDiversityAnalyzer:
    """Analyse diversity across multiple generated summaries.

    Provides methods for measuring how differently a set of summaries
    abstract, compress, and cover source material. All metrics are
    computed from scratch using ``_tokenize`` and ``_ngrams`` helpers
    already defined in this module.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure_abstractive_diversity(
        self,
        summaries: List[str],
        source: str,
    ) -> Dict[str, Any]:
        """Measure how differently each summary abstracts the source.

        For every summary we compute a *novelty ratio* — the fraction of
        n-grams (uni-, bi-, tri-) that do not appear in the source — then
        report per-summary scores together with aggregate statistics.

        Args:
            summaries: List of generated summaries.
            source: The original source document.

        Returns:
            Dictionary with per-summary novelty ratios and aggregate
            statistics (mean, std, min, max, pairwise divergence).
        """
        if not summaries:
            return {"per_summary": [], "aggregate": {}}

        per_summary: List[Dict[str, float]] = []
        for summary in summaries:
            novelty = self._compute_novelty_ratio(summary, source)
            per_summary.append(novelty)

        # Aggregate across summaries for each n-gram order.
        aggregate: Dict[str, Any] = {}
        for key in ("unigram_novelty", "bigram_novelty", "trigram_novelty"):
            values = np.array([s[key] for s in per_summary])
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # Pairwise divergence: average absolute difference of novelty
        # ratios between all summary pairs.
        combined_novelties = np.array(
            [s["combined_novelty"] for s in per_summary]
        )
        n = len(combined_novelties)
        if n > 1:
            pair_diffs: List[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    pair_diffs.append(
                        abs(combined_novelties[i] - combined_novelties[j])
                    )
            aggregate["pairwise_divergence"] = float(np.mean(pair_diffs))
        else:
            aggregate["pairwise_divergence"] = 0.0

        return {"per_summary": per_summary, "aggregate": aggregate}

    # ------------------------------------------------------------------

    def compare_extractive_abstractive(
        self,
        summaries: List[str],
        source: str,
    ) -> Dict[str, Any]:
        """Classify each summary along the extractive–abstractive spectrum.

        A purely extractive summary re-uses source sentences verbatim; a
        purely abstractive one has no sentence overlap.  We score every
        summary on a 0–1 scale (0 = fully extractive, 1 = fully
        abstractive) using sentence-level overlap, then report how
        diverse the set is along this axis.

        Args:
            summaries: List of generated summaries.
            source: The original source document.

        Returns:
            Dictionary containing per-summary abstractiveness scores
            and aggregate statistics.
        """
        if not summaries:
            return {"per_summary": [], "aggregate": {}}

        source_sents = self._split_sentences(source)

        per_summary: List[Dict[str, float]] = []
        for summary in summaries:
            summary_sents = self._split_sentences(summary)
            if not summary_sents:
                per_summary.append({
                    "abstractiveness": 1.0,
                    "max_sentence_overlap": 0.0,
                    "mean_sentence_overlap": 0.0,
                })
                continue

            overlaps: List[float] = []
            for s_sent in summary_sents:
                best = 0.0
                for src_sent in source_sents:
                    best = max(best, self._sentence_overlap(s_sent, src_sent))
                overlaps.append(best)

            mean_overlap = float(np.mean(overlaps))
            per_summary.append({
                "abstractiveness": 1.0 - mean_overlap,
                "max_sentence_overlap": float(np.max(overlaps)),
                "mean_sentence_overlap": mean_overlap,
            })

        scores = np.array([s["abstractiveness"] for s in per_summary])
        aggregate = {
            "mean_abstractiveness": float(np.mean(scores)),
            "std_abstractiveness": float(np.std(scores)),
            "range": float(np.max(scores) - np.min(scores)),
            "fully_extractive_count": int(np.sum(scores < 0.1)),
            "fully_abstractive_count": int(np.sum(scores > 0.9)),
        }

        return {"per_summary": per_summary, "aggregate": aggregate}

    # ------------------------------------------------------------------

    def measure_coverage_diversity(
        self,
        summaries: List[str],
        source: str,
    ) -> Dict[str, Any]:
        """Measure which key information each summary covers.

        We extract key entities and important unigrams/bigrams from the
        source, then check which of them appear in each summary.  The
        resulting coverage vectors are compared to quantify how
        differently each summary selects information.

        Args:
            summaries: List of generated summaries.
            source: The original source document.

        Returns:
            Dictionary with per-summary coverage scores, entity
            coverage, union/intersection coverage, and diversity index.
        """
        if not summaries:
            return {"per_summary": [], "aggregate": {}}

        source_entities = self._extract_key_entities(source)
        source_tokens = _tokenize(source)
        source_bigrams = set(_ngrams(source_tokens, 2))

        # Identify "important" source tokens by frequency (top-25 %).
        freq = Counter(source_tokens)
        if freq:
            threshold = np.percentile(
                list(freq.values()), 75
            )
            important_tokens = {
                t for t, c in freq.items() if c >= threshold
            }
        else:
            important_tokens = set()

        per_summary: List[Dict[str, Any]] = []
        all_covered_entities: List[Set[str]] = []
        all_covered_tokens: List[Set[str]] = []

        for summary in summaries:
            cov = self._compute_coverage(summary, source)
            summary_tokens = set(_tokenize(summary))

            covered_entities = {
                e for e in source_entities if e.lower() in summary.lower()
            }
            covered_important = summary_tokens & important_tokens

            entity_cov = (
                len(covered_entities) / len(source_entities)
                if source_entities
                else 0.0
            )

            per_summary.append({
                "overall_coverage": cov,
                "entity_coverage": entity_cov,
                "covered_entity_count": len(covered_entities),
                "important_token_coverage": (
                    len(covered_important) / len(important_tokens)
                    if important_tokens
                    else 0.0
                ),
            })
            all_covered_entities.append(covered_entities)
            all_covered_tokens.append(covered_important)

        # Union / intersection analysis across summaries.
        if all_covered_entities:
            union_ent = set.union(*all_covered_entities) if all_covered_entities else set()
            inter_ent = set.intersection(*all_covered_entities) if all_covered_entities else set()
        else:
            union_ent = set()
            inter_ent = set()

        if all_covered_tokens:
            union_tok = set.union(*all_covered_tokens) if all_covered_tokens else set()
            inter_tok = set.intersection(*all_covered_tokens) if all_covered_tokens else set()
        else:
            union_tok = set()
            inter_tok = set()

        jaccard_entities = (
            len(inter_ent) / len(union_ent) if union_ent else 1.0
        )
        jaccard_tokens = (
            len(inter_tok) / len(union_tok) if union_tok else 1.0
        )

        # Diversity index: 1 − Jaccard (higher ⇒ more diverse).
        aggregate = {
            "entity_diversity_index": 1.0 - jaccard_entities,
            "token_diversity_index": 1.0 - jaccard_tokens,
            "union_entity_count": len(union_ent),
            "intersection_entity_count": len(inter_ent),
            "mean_coverage": float(
                np.mean([s["overall_coverage"] for s in per_summary])
            ),
            "std_coverage": float(
                np.std([s["overall_coverage"] for s in per_summary])
            ),
        }

        return {"per_summary": per_summary, "aggregate": aggregate}

    # ------------------------------------------------------------------

    def measure_compression_diversity(
        self,
        summaries: List[str],
        source: str,
    ) -> Dict[str, Any]:
        """Measure variation in compression ratios across summaries.

        Compression ratio is defined as
        ``len(summary_tokens) / len(source_tokens)``.  We report
        per-summary ratios and aggregate statistics describing how
        much the summaries differ in length relative to the source.

        Args:
            summaries: List of generated summaries.
            source: The original source document.

        Returns:
            Dictionary with per-summary compression ratios and
            aggregate statistics (mean, std, coefficient of variation).
        """
        if not summaries:
            return {"per_summary": [], "aggregate": {}}

        ratios: List[float] = []
        per_summary: List[Dict[str, float]] = []

        for summary in summaries:
            ratio = self._compression_ratio(summary, source)
            ratios.append(ratio)
            summary_tokens = _tokenize(summary)
            source_tokens = _tokenize(source)
            per_summary.append({
                "compression_ratio": ratio,
                "summary_word_count": len(summary_tokens),
                "source_word_count": len(source_tokens),
            })

        ratios_arr = np.array(ratios)
        mean_ratio = float(np.mean(ratios_arr))
        std_ratio = float(np.std(ratios_arr))
        cv = std_ratio / mean_ratio if mean_ratio > 0 else 0.0

        aggregate = {
            "mean_compression": mean_ratio,
            "std_compression": std_ratio,
            "coefficient_of_variation": cv,
            "min_compression": float(np.min(ratios_arr)),
            "max_compression": float(np.max(ratios_arr)),
            "range": float(np.max(ratios_arr) - np.min(ratios_arr)),
        }

        return {"per_summary": per_summary, "aggregate": aggregate}

    # ------------------------------------------------------------------

    def compute_faithfulness_diversity_tradeoff(
        self,
        summaries: List[str],
        source: str,
    ) -> Dict[str, Any]:
        """Analyse the tradeoff between faithfulness and diversity.

        More abstractive (diverse) summaries risk introducing factual
        errors.  We compute both a faithfulness score and a novelty
        score for each summary and report the Pearson correlation as
        a measure of the tradeoff.

        Args:
            summaries: List of generated summaries.
            source: The original source document.

        Returns:
            Dictionary with per-summary scores, Pearson correlation,
            Pareto-optimal set, and aggregate statistics.
        """
        if not summaries:
            return {"per_summary": [], "aggregate": {}}

        per_summary: List[Dict[str, float]] = []
        faithfulness_scores: List[float] = []
        novelty_scores: List[float] = []

        for summary in summaries:
            faith = self._compute_faithfulness(summary, source)
            novelty = self._compute_novelty_ratio(summary, source)[
                "combined_novelty"
            ]
            per_summary.append({
                "faithfulness": faith,
                "novelty": novelty,
            })
            faithfulness_scores.append(faith)
            novelty_scores.append(novelty)

        faith_arr = np.array(faithfulness_scores)
        nov_arr = np.array(novelty_scores)

        # Pearson correlation.
        if len(faith_arr) > 1 and np.std(faith_arr) > 0 and np.std(nov_arr) > 0:
            correlation = float(
                np.corrcoef(faith_arr, nov_arr)[0, 1]
            )
        else:
            correlation = 0.0

        # Pareto-optimal indices: summaries not dominated in both
        # faithfulness and novelty.
        pareto_indices: List[int] = []
        for i in range(len(summaries)):
            dominated = False
            for j in range(len(summaries)):
                if i == j:
                    continue
                if (
                    faithfulness_scores[j] >= faithfulness_scores[i]
                    and novelty_scores[j] >= novelty_scores[i]
                    and (
                        faithfulness_scores[j] > faithfulness_scores[i]
                        or novelty_scores[j] > novelty_scores[i]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        aggregate = {
            "pearson_correlation": correlation,
            "mean_faithfulness": float(np.mean(faith_arr)),
            "mean_novelty": float(np.mean(nov_arr)),
            "pareto_optimal_indices": pareto_indices,
            "pareto_optimal_count": len(pareto_indices),
        }

        return {"per_summary": per_summary, "aggregate": aggregate}

    # ------------------------------------------------------------------

    def measure_multi_document_diversity(
        self,
        summaries_per_doc: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Measure diversity of summaries across multiple source documents.

        For each document we compute intra-document diversity (how much
        summaries of the *same* document differ) and then report
        inter-document statistics.

        Args:
            summaries_per_doc: Mapping from document identifier to the
                list of summaries generated for that document.

        Returns:
            Dictionary with per-document diversity metrics and
            cross-document aggregate statistics.
        """
        if not summaries_per_doc:
            return {"per_document": {}, "aggregate": {}}

        per_document: Dict[str, Dict[str, float]] = {}
        all_intra: List[float] = []

        for doc_id, summaries in summaries_per_doc.items():
            if len(summaries) < 2:
                per_document[doc_id] = {
                    "intra_diversity": 0.0,
                    "mean_pairwise_jaccard": 1.0,
                    "summary_count": len(summaries),
                }
                all_intra.append(0.0)
                continue

            # Pairwise Jaccard distance on token sets.
            token_sets = [set(_tokenize(s)) for s in summaries]
            jaccard_dists: List[float] = []
            for i in range(len(token_sets)):
                for j in range(i + 1, len(token_sets)):
                    inter = len(token_sets[i] & token_sets[j])
                    union = len(token_sets[i] | token_sets[j])
                    jd = 1.0 - (inter / union if union else 1.0)
                    jaccard_dists.append(jd)

            mean_jd = float(np.mean(jaccard_dists))

            # Bigram-level diversity.
            bigram_sets = [
                set(_ngrams(_tokenize(s), 2)) for s in summaries
            ]
            bigram_dists: List[float] = []
            for i in range(len(bigram_sets)):
                for j in range(i + 1, len(bigram_sets)):
                    inter = len(bigram_sets[i] & bigram_sets[j])
                    union = len(bigram_sets[i] | bigram_sets[j])
                    bd = 1.0 - (inter / union if union else 1.0)
                    bigram_dists.append(bd)

            mean_bd = float(np.mean(bigram_dists)) if bigram_dists else 0.0

            intra_div = (mean_jd + mean_bd) / 2.0
            per_document[doc_id] = {
                "intra_diversity": intra_div,
                "mean_pairwise_jaccard": mean_jd,
                "mean_pairwise_bigram": mean_bd,
                "summary_count": len(summaries),
            }
            all_intra.append(intra_div)

        intra_arr = np.array(all_intra)
        aggregate = {
            "mean_intra_diversity": float(np.mean(intra_arr)),
            "std_intra_diversity": float(np.std(intra_arr)),
            "min_intra_diversity": float(np.min(intra_arr)),
            "max_intra_diversity": float(np.max(intra_arr)),
            "document_count": len(summaries_per_doc),
        }

        return {"per_document": per_document, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_novelty_ratio(
        self,
        summary: str,
        source: str,
    ) -> Dict[str, float]:
        """Compute the ratio of novel n-grams in a summary.

        A novel n-gram is one that does not appear anywhere in the
        source document.  We compute ratios for unigrams, bigrams,
        and trigrams, plus a combined average.

        Args:
            summary: The generated summary.
            source: The original source document.

        Returns:
            Dictionary mapping n-gram order to its novelty ratio and
            a ``combined_novelty`` key with the arithmetic mean.
        """
        summary_tokens = _tokenize(summary)
        source_tokens = _tokenize(source)

        results: Dict[str, float] = {}
        for n, label in [(1, "unigram"), (2, "bigram"), (3, "trigram")]:
            s_ngrams = _ngrams(summary_tokens, n)
            src_set = set(_ngrams(source_tokens, n))
            if not s_ngrams:
                results[f"{label}_novelty"] = 0.0
                continue
            novel = sum(1 for ng in s_ngrams if ng not in src_set)
            results[f"{label}_novelty"] = novel / len(s_ngrams)

        results["combined_novelty"] = float(
            np.mean([
                results["unigram_novelty"],
                results["bigram_novelty"],
                results["trigram_novelty"],
            ])
        )
        return results

    # ------------------------------------------------------------------

    def _compute_coverage(
        self,
        summary: str,
        source: str,
    ) -> float:
        """Compute information coverage of a summary w.r.t. the source.

        Coverage is estimated as the weighted fraction of important
        source unigrams and bigrams that are present in the summary.
        Unigrams contribute 40 % and bigrams 60 % of the score to
        reward phrase-level coverage more than single words.

        Args:
            summary: The generated summary.
            source: The original source document.

        Returns:
            Coverage score in [0, 1].
        """
        summary_tokens = _tokenize(summary)
        source_tokens = _tokenize(source)

        # Remove very common English stop-words for a cleaner signal.
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when",
            "that", "this", "these", "those", "it", "its", "they",
            "their", "them", "he", "she", "his", "her", "him", "we",
            "us", "our", "you", "your", "i", "me", "my",
        }

        src_uni = [t for t in source_tokens if t not in stop]
        sum_uni_set = set(summary_tokens)
        uni_covered = sum(1 for t in set(src_uni) if t in sum_uni_set)
        uni_total = len(set(src_uni))
        uni_cov = uni_covered / uni_total if uni_total else 0.0

        src_bi = set(_ngrams(source_tokens, 2))
        sum_bi = set(_ngrams(summary_tokens, 2))
        bi_covered = len(src_bi & sum_bi)
        bi_total = len(src_bi)
        bi_cov = bi_covered / bi_total if bi_total else 0.0

        return 0.4 * uni_cov + 0.6 * bi_cov

    # ------------------------------------------------------------------

    def _compute_faithfulness(
        self,
        summary: str,
        source: str,
    ) -> float:
        """Compute a lightweight faithfulness score for a summary.

        We approximate factual consistency by checking what fraction
        of the summary's content n-grams (bigrams and trigrams) can
        be found in the source.  A higher score indicates that the
        summary stays closer to the source wording and is less likely
        to hallucinate.

        Args:
            summary: The generated summary.
            source: The original source document.

        Returns:
            Faithfulness score in [0, 1].
        """
        summary_tokens = _tokenize(summary)
        source_tokens = _tokenize(source)

        scores: List[float] = []
        for n in (2, 3):
            s_ngrams = _ngrams(summary_tokens, n)
            src_set = set(_ngrams(source_tokens, n))
            if not s_ngrams:
                scores.append(1.0)
                continue
            found = sum(1 for ng in s_ngrams if ng in src_set)
            scores.append(found / len(s_ngrams))

        # Also check entity consistency: entities in the summary that
        # also appear in the source.
        summary_entities = self._extract_key_entities(summary)
        source_lower = source.lower()
        if summary_entities:
            entity_found = sum(
                1 for e in summary_entities if e.lower() in source_lower
            )
            entity_score = entity_found / len(summary_entities)
        else:
            entity_score = 1.0

        # Weighted combination: n-gram fidelity 70 %, entity fidelity 30 %.
        ngram_score = float(np.mean(scores))
        return 0.7 * ngram_score + 0.3 * entity_score

    # ------------------------------------------------------------------

    def _compression_ratio(self, summary: str, source: str) -> float:
        """Compute word-level compression ratio.

        Defined as ``len(summary_tokens) / len(source_tokens)``.  A
        value of 0.1 means the summary is 10 % as long as the source.

        Args:
            summary: The generated summary.
            source: The original source document.

        Returns:
            Compression ratio (float ≥ 0).
        """
        summary_tokens = _tokenize(summary)
        source_tokens = _tokenize(source)
        if not source_tokens:
            return 0.0
        return len(summary_tokens) / len(source_tokens)

    # ------------------------------------------------------------------

    def _extract_key_entities(self, text: str) -> List[str]:
        """Heuristic named-entity extraction.

        We identify capitalised multi-word spans that are likely proper
        nouns (names, places, organisations) as well as stand-alone
        capitalised words that are at least three characters long and
        are not sentence starters.

        Args:
            text: Input text.

        Returns:
            Deduplicated list of extracted entity strings.
        """
        # Multi-word capitalised spans (e.g. "United Nations").
        multi = re.findall(r"(?<!\. )(?<!\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)

        # Single capitalised words (≥3 chars) not at sentence start.
        singles: List[str] = []
        for match in re.finditer(r"(?<=\s)([A-Z][a-z]{2,})", text):
            start = match.start()
            # Skip if preceded by ". " or is at position 0.
            if start == 0:
                continue
            preceding = text[max(0, start - 2) : start]
            if preceding.rstrip().endswith("."):
                continue
            singles.append(match.group(1))

        # Numbers that look like quantities or years.
        numbers = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)

        seen: Set[str] = set()
        entities: List[str] = []
        for entity in multi + singles + numbers:
            normed = entity.strip()
            if normed.lower() not in seen:
                seen.add(normed.lower())
                entities.append(normed)

        return entities

    # ------------------------------------------------------------------

    def _sentence_overlap(self, sent_a: str, sent_b: str) -> float:
        """Compute token-level overlap between two sentences.

        The overlap is the size of the intersection of the two token
        sets divided by the size of the smaller set (Szymkiewicz–
        Simpson coefficient), so that a short sentence fully contained
        in a longer one receives a score of 1.0.

        Args:
            sent_a: First sentence.
            sent_b: Second sentence.

        Returns:
            Overlap score in [0, 1].
        """
        tokens_a = set(_tokenize(sent_a))
        tokens_b = set(_tokenize(sent_b))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        return intersection / min(len(tokens_a), len(tokens_b))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using a simple regex heuristic."""
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]
