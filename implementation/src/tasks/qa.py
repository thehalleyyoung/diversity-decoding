"""
Question Answering task domain for the Diversity Decoding Arena.

Implements diverse answer generation across multiple question types,
evaluation metrics for answer quality/diversity, and prompt datasets.
"""

import re
import math
import string
from enum import Enum, auto
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
from collections import Counter, defaultdict

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
# Enums
# ---------------------------------------------------------------------------

class QuestionType(Enum):
    """Classification of question intent."""
    FACTUAL = auto()
    OPEN_ENDED = auto()
    ANALYTICAL = auto()
    COMPARATIVE = auto()
    HYPOTHETICAL = auto()
    OPINION = auto()
    PROCEDURAL = auto()
    CLARIFICATION = auto()


class AnswerFormat(Enum):
    """Desired structure of the generated answer."""
    SHORT_ANSWER = auto()
    PARAGRAPH = auto()
    LIST = auto()
    STEP_BY_STEP = auto()
    DEBATE_STYLE = auto()
    MULTI_PERSPECTIVE = auto()


# ---------------------------------------------------------------------------
# Config & Prompt dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QAConfig(TaskConfig):
    """Configuration specific to the QA task domain."""
    question_type: QuestionType = QuestionType.FACTUAL
    answer_format: AnswerFormat = AnswerFormat.PARAGRAPH
    require_evidence: bool = False
    max_answer_length: int = 512
    allow_uncertainty: bool = True
    perspective_count: int = 1


@dataclass
class QAPrompt(TaskPrompt):
    """A single QA prompt with optional supporting material."""
    question: str = ""
    question_type: QuestionType = QuestionType.FACTUAL
    context_passage: Optional[str] = None
    reference_answers: List[str] = field(default_factory=list)
    evidence_passages: List[str] = field(default_factory=list)
    difficulty: float = 0.5
    required_perspectives: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "need", "dare", "ought", "used", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only",
    "own", "same", "than", "too", "very", "just", "because",
    "about", "that", "this", "these", "those", "it", "its",
}

_REASONING_MARKERS = [
    "because", "therefore", "thus", "hence", "consequently",
    "as a result", "due to", "since", "given that",
    "this means", "it follows", "implies", "leads to",
    "suggests", "indicates", "demonstrates", "proves",
    "for this reason", "on the basis of", "in light of",
    "considering", "taking into account", "assuming",
    "if we consider", "one reason", "another reason",
    "firstly", "secondly", "thirdly", "finally",
    "in conclusion", "to summarize", "overall",
]

_HEDGE_PHRASES = [
    "it is possible", "perhaps", "arguably", "it seems",
    "one could argue", "there is debate", "some believe",
    "it is unclear", "evidence suggests", "likely",
    "unlikely", "may", "might", "could", "possibly",
    "probably", "approximately", "roughly", "tends to",
    "in some cases", "not necessarily", "it depends",
    "to some extent", "partially", "in part",
    "on one hand", "on the other hand",
    "while some", "others contend", "alternatively",
]

_PERSPECTIVE_KEYWORDS = {
    "economic": ["economic", "economy", "financial", "cost", "market", "gdp", "trade", "fiscal"],
    "social": ["social", "society", "community", "cultural", "people", "population", "public"],
    "political": ["political", "government", "policy", "regulation", "law", "legislative"],
    "environmental": ["environmental", "climate", "ecology", "pollution", "sustainability", "green"],
    "ethical": ["ethical", "moral", "values", "rights", "justice", "fairness", "duty"],
    "technological": ["technology", "innovation", "digital", "technical", "computing", "ai"],
    "historical": ["history", "historical", "past", "tradition", "legacy", "precedent"],
    "scientific": ["science", "scientific", "research", "evidence", "empirical", "data"],
    "psychological": ["psychological", "mental", "cognitive", "emotional", "behavioral"],
    "philosophical": ["philosophy", "philosophical", "existential", "metaphysical", "epistemological"],
}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _content_tokens(text: str) -> List[str]:
    """Tokenize and remove stop-words."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# QuestionAnsweringTask
# ---------------------------------------------------------------------------

class QuestionAnsweringTask(GenerationTask):
    """Full implementation of the QA task domain."""

    DOMAIN = TaskDomain.QUESTION_ANSWERING if hasattr(TaskDomain, "QUESTION_ANSWERING") else "question_answering"

    def __init__(self, config: Optional[QAConfig] = None, **kwargs: Any) -> None:
        self.config: QAConfig = config or QAConfig()
        super().__init__(config=self.config, **kwargs)
        self._prompts: Optional[PromptDataset] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Build and return the full prompt dataset (50+ prompts)."""
        if self._prompts is not None:
            return self._prompts

        all_prompts: List[QAPrompt] = []
        all_prompts.extend(self._generate_factual_prompts())
        all_prompts.extend(self._generate_analytical_prompts())
        all_prompts.extend(self._generate_open_ended_prompts())
        all_prompts.extend(self._generate_comparative_prompts())
        all_prompts.extend(self._generate_hypothetical_prompts())
        all_prompts.extend(self._generate_opinion_prompts())
        all_prompts.extend(self._generate_procedural_prompts())
        all_prompts.extend(self._generate_clarification_prompts())

        self._prompts = PromptDataset(prompts=all_prompts)
        return self._prompts

    def format_prompt(self, prompt: QAPrompt) -> str:  # type: ignore[override]
        """Format a *QAPrompt* into a textual instruction for an LLM."""
        parts: List[str] = []

        # System-level preamble
        parts.append("You are a knowledgeable assistant. Answer the following question.")

        # Answer-format instruction
        fmt_map = {
            AnswerFormat.SHORT_ANSWER: "Provide a concise, short answer (one or two sentences).",
            AnswerFormat.PARAGRAPH: "Write a well-structured paragraph as your answer.",
            AnswerFormat.LIST: "Present your answer as a numbered or bulleted list.",
            AnswerFormat.STEP_BY_STEP: "Explain step by step.",
            AnswerFormat.DEBATE_STYLE: (
                "Present arguments for and against before giving your conclusion."
            ),
            AnswerFormat.MULTI_PERSPECTIVE: (
                f"Provide the answer from at least {self.config.perspective_count} "
                "different perspectives."
            ),
        }
        fmt_instruction = fmt_map.get(
            self.config.answer_format, "Provide a thorough answer."
        )
        parts.append(fmt_instruction)

        # Evidence instruction
        if self.config.require_evidence:
            parts.append(
                "Support your answer with specific evidence or citations from the "
                "provided context."
            )

        # Uncertainty allowance
        if self.config.allow_uncertainty:
            parts.append(
                "If you are unsure, express your uncertainty appropriately."
            )

        # Context passage
        if prompt.context_passage:
            parts.append(f"\n--- Context ---\n{prompt.context_passage}\n--- End Context ---")

        # Evidence passages
        if prompt.evidence_passages:
            parts.append("\n--- Evidence ---")
            for idx, ep in enumerate(prompt.evidence_passages, 1):
                parts.append(f"[{idx}] {ep}")
            parts.append("--- End Evidence ---")

        # Required perspectives
        if prompt.required_perspectives:
            perspectives_str = ", ".join(prompt.required_perspectives)
            parts.append(
                f"\nPlease address the following perspectives: {perspectives_str}."
            )

        # Max length
        if self.config.max_answer_length:
            parts.append(
                f"\nKeep your answer under {self.config.max_answer_length} tokens."
            )

        # The question itself
        parts.append(f"\nQuestion: {prompt.question}")
        parts.append("\nAnswer:")

        return "\n".join(parts)

    def evaluate(
        self,
        generations: List[str],
        prompts: List[QAPrompt],  # type: ignore[override]
    ) -> Dict[str, Any]:
        """Score a batch of generated answers against their prompts.

        Returns a dictionary with per-sample scores and aggregate metrics.
        """
        n = len(generations)
        assert n == len(prompts), "generations and prompts must be the same length"

        per_sample: List[Dict[str, float]] = []
        agg: Dict[str, List[float]] = defaultdict(list)

        for gen, prompt in zip(generations, prompts):
            cleaned = self.post_process(gen)
            scores: Dict[str, float] = {}

            # Reference-based metrics
            if prompt.reference_answers:
                scores["exact_match"] = max(
                    self._exact_match(cleaned, ref)
                    for ref in prompt.reference_answers
                )
                scores["f1"] = max(
                    self._f1_score(cleaned, ref)
                    for ref in prompt.reference_answers
                )
                scores["semantic_similarity"] = max(
                    self._semantic_similarity(cleaned, ref)
                    for ref in prompt.reference_answers
                )
            else:
                scores["exact_match"] = 0.0
                scores["f1"] = 0.0
                scores["semantic_similarity"] = 0.0

            # Content quality
            scores["completeness"] = self._answer_completeness(cleaned, prompt.question)
            scores["relevance"] = self._question_relevance(cleaned, prompt.question)
            scores["informativeness"] = self._informativeness(cleaned)
            scores["conciseness"] = self._conciseness_score(cleaned)
            scores["reasoning_depth"] = self._reasoning_depth(cleaned)
            scores["uncertainty"] = self._uncertainty_expression(cleaned)

            # Structure
            scores["structure"] = self._answer_structure_score(
                cleaned, self.config.answer_format
            )

            # Evidence grounding
            if prompt.evidence_passages:
                scores["evidence_grounding"] = self._evidence_grounding(
                    cleaned, prompt.evidence_passages
                )
            if prompt.context_passage:
                scores["factual_consistency"] = self._factual_consistency(
                    cleaned, prompt.context_passage
                )

            # Perspective coverage
            if prompt.required_perspectives:
                scores["perspective_coverage"] = self._perspective_coverage(
                    [cleaned], prompt.required_perspectives
                )

            per_sample.append(scores)
            for k, v in scores.items():
                agg[k].append(v)

        # Diversity across the batch
        diversity = self._answer_diversity(generations)

        # Contradiction detection across all pairs
        contradictions = self._contradiction_detection(generations)

        aggregated: Dict[str, float] = {
            k: float(np.mean(v)) for k, v in agg.items()
        }
        aggregated["diversity"] = diversity
        aggregated["contradiction_count"] = float(len(contradictions))

        return {
            "per_sample": per_sample,
            "aggregated": aggregated,
            "contradictions": contradictions,
        }

    def get_constraints(self) -> List[TaskConstraint]:
        """Return constraints derived from the current config."""
        constraints: List[TaskConstraint] = []

        # Answer format constraint
        constraints.append(
            TaskConstraint(
                name="answer_format",
                description=f"Answer must follow {self.config.answer_format.name} format.",
                check_fn=lambda ans: self._answer_structure_score(
                    ans, self.config.answer_format
                )
                > 0.3,
            )
        )

        # Length constraint
        constraints.append(
            TaskConstraint(
                name="max_length",
                description=(
                    f"Answer must not exceed {self.config.max_answer_length} tokens."
                ),
                check_fn=lambda ans: len(_tokenize(ans))
                <= self.config.max_answer_length,
            )
        )

        # Evidence constraint
        if self.config.require_evidence:
            constraints.append(
                TaskConstraint(
                    name="evidence_required",
                    description="Answer must reference provided evidence.",
                    check_fn=lambda ans: bool(
                        re.search(r"\[\d+\]", ans)
                        or re.search(r"according to|based on|as stated", ans, re.I)
                    ),
                )
            )

        # Perspective constraint
        if self.config.perspective_count > 1:
            constraints.append(
                TaskConstraint(
                    name="multi_perspective",
                    description=(
                        f"Answer must cover >= {self.config.perspective_count} perspectives."
                    ),
                    check_fn=lambda ans: self._count_perspectives(ans)
                    >= self.config.perspective_count,
                )
            )

        return constraints

    # ------------------------------------------------------------------
    # Reference-based metrics
    # ------------------------------------------------------------------

    def _exact_match(self, prediction: str, reference: str) -> float:
        """Normalised exact match (0 or 1)."""
        return float(
            self._normalize_answer(prediction) == self._normalize_answer(reference)
        )

    def _f1_score(self, prediction: str, reference: str) -> float:
        """Token-level F1 between prediction and reference."""
        pred_tokens = _tokenize(self._normalize_answer(prediction))
        ref_tokens = _tokenize(self._normalize_answer(reference))
        return self._token_f1(pred_tokens, ref_tokens)

    def _semantic_similarity(self, answer: str, reference: str) -> float:
        """Word-overlap based semantic similarity (Jaccard + weighted IDF)."""
        ans_tokens = set(_content_tokens(answer))
        ref_tokens = set(_content_tokens(reference))
        if not ans_tokens and not ref_tokens:
            return 1.0
        if not ans_tokens or not ref_tokens:
            return 0.0

        intersection = ans_tokens & ref_tokens
        union = ans_tokens | ref_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # Bigram overlap bonus
        ans_bigrams = set(_ngrams(_content_tokens(answer), 2))
        ref_bigrams = set(_ngrams(_content_tokens(reference), 2))
        if ans_bigrams and ref_bigrams:
            bigram_overlap = len(ans_bigrams & ref_bigrams) / len(
                ans_bigrams | ref_bigrams
            )
        else:
            bigram_overlap = 0.0

        return 0.6 * jaccard + 0.4 * bigram_overlap

    # ------------------------------------------------------------------
    # Content quality metrics
    # ------------------------------------------------------------------

    def _answer_completeness(self, answer: str, question: str) -> float:
        """How completely does the answer address the question?

        Heuristic: proportion of question content words present in answer,
        weighted by answer length adequacy.
        """
        q_tokens = set(_content_tokens(question))
        a_tokens = set(_content_tokens(answer))
        if not q_tokens:
            return 1.0
        topic_coverage = len(q_tokens & a_tokens) / len(q_tokens)

        # Length adequacy: penalise very short answers
        word_count = len(_tokenize(answer))
        length_factor = min(1.0, word_count / 20.0)

        # Sentence count bonus
        sentences = _sentence_split(answer)
        sentence_factor = min(1.0, len(sentences) / 2.0)

        return float(np.clip(
            0.5 * topic_coverage + 0.3 * length_factor + 0.2 * sentence_factor,
            0.0,
            1.0,
        ))

    def _evidence_grounding(self, answer: str, evidence: List[str]) -> float:
        """Proportion of evidence passages whose content is reflected in the
        answer, plus explicit citation detection."""
        if not evidence:
            return 0.0

        grounded_count = 0
        for idx, passage in enumerate(evidence, 1):
            ev_tokens = set(_content_tokens(passage))
            ans_tokens = set(_content_tokens(answer))
            overlap = len(ev_tokens & ans_tokens) / max(len(ev_tokens), 1)

            # Explicit citation pattern [1], [2], …
            cited = bool(re.search(rf"\[{idx}\]", answer))
            if overlap > 0.25 or cited:
                grounded_count += 1

        return grounded_count / len(evidence)

    def _factual_consistency(self, answer: str, context: str) -> float:
        """Rough check: fraction of answer content tokens that appear in the
        context (or are common function words)."""
        ans_tokens = _content_tokens(answer)
        ctx_tokens = set(_content_tokens(context))
        if not ans_tokens:
            return 1.0
        supported = sum(1 for t in ans_tokens if t in ctx_tokens)
        return supported / len(ans_tokens)

    def _answer_diversity(self, answers: List[str]) -> float:
        """Measure diversity within a set of answers.

        Combines:
        - pairwise dissimilarity (1 - avg Jaccard)
        - unique n-gram ratio
        - answer type variety
        """
        if len(answers) <= 1:
            return 0.0

        # Pairwise token Jaccard dissimilarity
        token_sets = [set(_content_tokens(a)) for a in answers]
        pair_dists: List[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                si, sj = token_sets[i], token_sets[j]
                union = si | sj
                if union:
                    pair_dists.append(1.0 - len(si & sj) / len(union))
                else:
                    pair_dists.append(0.0)
        avg_dissimilarity = float(np.mean(pair_dists)) if pair_dists else 0.0

        # Unique bigram ratio across all answers
        all_bigrams: List[Tuple[str, ...]] = []
        for a in answers:
            all_bigrams.extend(_ngrams(_content_tokens(a), 2))
        unique_ratio = len(set(all_bigrams)) / max(len(all_bigrams), 1)

        # Answer type variety
        types = set(self._answer_type_classification(a) for a in answers)
        type_diversity = len(types) / max(len(answers), 1)

        return float(np.clip(
            0.4 * avg_dissimilarity + 0.3 * unique_ratio + 0.3 * type_diversity,
            0.0, 1.0,
        ))

    def _perspective_coverage(
        self, answers: List[str], required_perspectives: List[str]
    ) -> float:
        """Fraction of required perspectives detected in the combined answers."""
        if not required_perspectives:
            return 1.0
        combined = " ".join(answers).lower()
        covered = 0
        for perspective in required_perspectives:
            keywords = _PERSPECTIVE_KEYWORDS.get(perspective.lower(), [perspective.lower()])
            if any(kw in combined for kw in keywords):
                covered += 1
        return covered / len(required_perspectives)

    def _reasoning_depth(self, answer: str) -> float:
        """Score the depth of reasoning chains in the answer."""
        lower = answer.lower()

        # Count reasoning markers
        marker_count = sum(1 for m in _REASONING_MARKERS if m in lower)

        # Count logical structure indicators
        numbered_steps = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", answer))
        bullets = len(re.findall(r"(?:^|\n)\s*[-•]\s", answer))
        structure_count = numbered_steps + bullets

        # Sentence count as proxy for elaboration
        sentences = _sentence_split(answer)
        sentence_factor = min(1.0, len(sentences) / 5.0)

        # Causal chain detection
        causal_pairs = len(re.findall(
            r"(because|since|due to|therefore|thus|hence|consequently)", lower
        ))
        causal_factor = min(1.0, causal_pairs / 3.0)

        raw = (
            0.3 * min(1.0, marker_count / 5.0)
            + 0.2 * min(1.0, structure_count / 3.0)
            + 0.25 * sentence_factor
            + 0.25 * causal_factor
        )
        return float(np.clip(raw, 0.0, 1.0))

    def _uncertainty_expression(self, answer: str) -> float:
        """Degree to which the answer expresses appropriate uncertainty."""
        lower = answer.lower()
        hedge_count = sum(1 for phrase in _HEDGE_PHRASES if phrase in lower)
        return float(np.clip(hedge_count / 4.0, 0.0, 1.0))

    def _answer_structure_score(
        self, answer: str, target_format: AnswerFormat
    ) -> float:
        """How well does the answer match the requested format?"""
        sentences = _sentence_split(answer)
        word_count = len(_tokenize(answer))

        if target_format == AnswerFormat.SHORT_ANSWER:
            if word_count <= 30:
                return 1.0
            elif word_count <= 60:
                return 0.6
            else:
                return max(0.1, 1.0 - (word_count - 30) / 200.0)

        if target_format == AnswerFormat.PARAGRAPH:
            has_paragraph = word_count >= 30
            few_bullets = len(re.findall(r"(?:^|\n)\s*[-•]\s", answer)) <= 1
            return 1.0 if has_paragraph and few_bullets else 0.5

        if target_format == AnswerFormat.LIST:
            list_items = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|[-•])\s", answer))
            if list_items >= 3:
                return 1.0
            elif list_items >= 1:
                return 0.5
            return 0.1

        if target_format == AnswerFormat.STEP_BY_STEP:
            steps = len(re.findall(
                r"(?:^|\n)\s*(?:step\s*\d+|^\d+[\.\)])", answer, re.I | re.M
            ))
            if steps >= 3:
                return 1.0
            elif steps >= 1:
                return 0.5
            return 0.1

        if target_format == AnswerFormat.DEBATE_STYLE:
            lower = answer.lower()
            has_for = any(
                p in lower
                for p in ["on one hand", "proponents", "in favor", "argument for", "supporters"]
            )
            has_against = any(
                p in lower
                for p in [
                    "on the other hand", "opponents", "against",
                    "argument against", "critics", "however",
                ]
            )
            has_conclusion = any(
                p in lower
                for p in ["in conclusion", "overall", "ultimately", "to summarize", "on balance"]
            )
            return (0.35 * has_for + 0.35 * has_against + 0.3 * has_conclusion)

        if target_format == AnswerFormat.MULTI_PERSPECTIVE:
            perspective_count = self._count_perspectives(answer)
            return float(np.clip(perspective_count / max(self.config.perspective_count, 1), 0.0, 1.0))

        return 0.5

    def _question_relevance(self, answer: str, question: str) -> float:
        """Proportion of question content words echoed in the answer, plus
        semantic-field bonus."""
        q_tokens = set(_content_tokens(question))
        a_tokens = set(_content_tokens(answer))
        if not q_tokens:
            return 1.0
        direct_overlap = len(q_tokens & a_tokens) / len(q_tokens)

        # Semantic-field bonus: check if question's topic words appear as
        # stems or substrings in the answer
        a_text = answer.lower()
        stem_hits = 0
        for qt in q_tokens:
            if len(qt) >= 4 and qt[:4] in a_text:
                stem_hits += 1
        stem_ratio = stem_hits / len(q_tokens) if q_tokens else 0.0

        return float(np.clip(0.7 * direct_overlap + 0.3 * stem_ratio, 0.0, 1.0))

    def _informativeness(self, answer: str) -> float:
        """Information content heuristic based on vocabulary richness and
        sentence variety."""
        tokens = _tokenize(answer)
        if not tokens:
            return 0.0

        # Type-token ratio
        ttr = len(set(tokens)) / len(tokens)

        # Content word ratio
        content = _content_tokens(answer)
        content_ratio = len(content) / max(len(tokens), 1)

        # Average sentence length variety
        sentences = _sentence_split(answer)
        if len(sentences) >= 2:
            lengths = [len(_tokenize(s)) for s in sentences]
            length_std = float(np.std(lengths))
            variety = min(1.0, length_std / 5.0)
        else:
            variety = 0.3

        # Unique bigrams relative to length
        bigrams = _ngrams(tokens, 2)
        bigram_ratio = len(set(bigrams)) / max(len(bigrams), 1)

        raw = 0.3 * ttr + 0.25 * content_ratio + 0.2 * variety + 0.25 * bigram_ratio
        return float(np.clip(raw, 0.0, 1.0))

    def _conciseness_score(self, answer: str) -> float:
        """Penalize unnecessary verbosity while rewarding compact, informative
        answers."""
        tokens = _tokenize(answer)
        if not tokens:
            return 0.0

        content = _content_tokens(answer)
        content_ratio = len(content) / len(tokens)

        # Repetition penalty
        token_counts = Counter(tokens)
        repeated = sum(1 for c in token_counts.values() if c > 2)
        repetition_penalty = min(1.0, repeated / 10.0)

        # Filler detection
        filler_phrases = [
            "it is important to note that",
            "it should be noted that",
            "as we can see",
            "in other words",
            "basically",
            "essentially",
            "actually",
            "as a matter of fact",
            "needless to say",
            "it goes without saying",
        ]
        lower = answer.lower()
        filler_count = sum(1 for f in filler_phrases if f in lower)
        filler_penalty = min(1.0, filler_count / 3.0)

        raw = content_ratio - 0.3 * repetition_penalty - 0.3 * filler_penalty
        return float(np.clip(raw, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Contradiction & claim analysis
    # ------------------------------------------------------------------

    def _contradiction_detection(
        self, answers: List[str]
    ) -> List[Tuple[int, int, float]]:
        """Detect potential contradictions between pairs of answers.

        Returns list of ``(idx_a, idx_b, contradiction_score)`` tuples.
        """
        results: List[Tuple[int, int, float]] = []
        claims_list = [self._extract_claims(a) for a in answers]

        negation_words = {"not", "no", "never", "neither", "nor", "cannot", "doesn't",
                          "isn't", "wasn't", "weren't", "won't", "wouldn't", "couldn't",
                          "shouldn't", "hardly", "scarcely", "barely"}

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                score = 0.0
                claims_a = claims_list[i]
                claims_b = claims_list[j]

                for ca in claims_a:
                    ca_tokens = set(_tokenize(ca))
                    ca_neg = bool(ca_tokens & negation_words)
                    for cb in claims_b:
                        cb_tokens = set(_tokenize(cb))
                        cb_neg = bool(cb_tokens & negation_words)

                        overlap = self._claim_overlap(
                            list(ca_tokens - negation_words),
                            list(cb_tokens - negation_words),
                        )
                        # High content overlap but different polarity
                        if overlap > 0.5 and ca_neg != cb_neg:
                            score = max(score, overlap)

                if score > 0.4:
                    results.append((i, j, score))

        return results

    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims (sentences that make assertions)."""
        sentences = _sentence_split(text)
        claims: List[str] = []
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            # Skip questions
            if s_stripped.endswith("?"):
                continue
            # Must have a verb-like word (simple heuristic)
            tokens = _tokenize(s_stripped)
            if len(tokens) >= 3:
                claims.append(s_stripped)
        return claims

    def _claim_overlap(self, claims_a: List[str], claims_b: List[str]) -> float:
        """Token-level overlap between two claim token lists."""
        set_a = set(claims_a)
        set_b = set(claims_b)
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        return len(intersection) / min(len(set_a), len(set_b))

    # ------------------------------------------------------------------
    # Classification & normalization helpers
    # ------------------------------------------------------------------

    def _answer_type_classification(self, answer: str) -> str:
        """Classify the rhetorical type of an answer."""
        lower = answer.lower()

        # Check for list
        list_items = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|[-•])\s", answer))
        if list_items >= 3:
            return "list"

        # Check for step-by-step
        if re.search(r"step\s*\d", lower) or (
            list_items >= 2
            and any(w in lower for w in ["first", "then", "next", "finally"])
        ):
            return "procedural"

        # Check for debate/comparison
        if any(
            p in lower
            for p in ["on one hand", "on the other hand", "however", "conversely"]
        ):
            return "comparative"

        # Check for opinion
        if any(
            p in lower
            for p in ["i believe", "in my opinion", "i think", "personally"]
        ):
            return "opinion"

        # Check for hedged/uncertain
        hedge_count = sum(1 for p in _HEDGE_PHRASES if p in lower)
        if hedge_count >= 3:
            return "hedged"

        # Check for short factual
        if len(_tokenize(answer)) <= 20:
            return "factual_short"

        # Check for analytical
        reasoning = sum(1 for m in _REASONING_MARKERS if m in lower)
        if reasoning >= 3:
            return "analytical"

        return "expository"

    def _normalize_answer(self, text: str) -> str:
        """Standard SQuAD-style answer normalization."""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove articles
        tokens = text.split()
        tokens = [t for t in tokens if t not in _ARTICLES]
        # Collapse whitespace
        text = " ".join(tokens)
        return text.strip()

    def _token_f1(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute token-level precision, recall, and F1."""
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self, text: str) -> str:
        """Clean up generated text for evaluation."""
        # Remove leading/trailing whitespace
        text = text.strip()

        # Strip common prefixes produced by LLMs
        for prefix in [
            "Answer:", "A:", "Response:", "Sure,", "Of course,",
            "Great question!", "That's a great question.",
            "Here is my answer:", "Here's the answer:",
        ]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces
        text = re.sub(r"  +", " ", text)

        # Truncate to max length if set
        if self.config.max_answer_length:
            tokens = text.split()
            if len(tokens) > self.config.max_answer_length:
                text = " ".join(tokens[: self.config.max_answer_length])
                # Try to end at a sentence boundary
                last_period = text.rfind(".")
                if last_period > len(text) * 0.7:
                    text = text[: last_period + 1]

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_perspectives(self, answer: str) -> int:
        """Count how many distinct perspectives are addressed in the answer."""
        lower = answer.lower()
        count = 0
        for _perspective, keywords in _PERSPECTIVE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                count += 1
        # Also detect explicit perspective markers
        explicit = len(re.findall(
            r"(?:from (?:a|an|the) \w+ perspective|from the standpoint of|"
            r"looking at .{1,30} from|considering the \w+ angle)",
            lower,
        ))
        return max(count, explicit)

    # ------------------------------------------------------------------
    # Prompt generators
    # ------------------------------------------------------------------

    def _generate_factual_prompts(self) -> List[QAPrompt]:
        """Factual / knowledge-recall questions."""
        prompts = [
            QAPrompt(
                question="What is the speed of light in a vacuum?",
                question_type=QuestionType.FACTUAL,
                reference_answers=[
                    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
                    "About 3 × 10^8 m/s.",
                ],
                difficulty=0.2,
            ),
            QAPrompt(
                question="Who developed the theory of general relativity?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["Albert Einstein"],
                difficulty=0.1,
            ),
            QAPrompt(
                question="What is the chemical formula for water?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["H2O"],
                difficulty=0.1,
            ),
            QAPrompt(
                question="In what year did the Berlin Wall fall?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["1989", "The Berlin Wall fell in 1989."],
                difficulty=0.2,
            ),
            QAPrompt(
                question="What is the largest organ in the human body?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["The skin", "Skin"],
                difficulty=0.2,
            ),
            QAPrompt(
                question="How many chromosomes do humans have?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["46", "Humans have 46 chromosomes (23 pairs)."],
                difficulty=0.2,
            ),
            QAPrompt(
                question="What is the capital of Australia?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["Canberra"],
                difficulty=0.2,
            ),
            QAPrompt(
                question=(
                    "According to the passage, what is the primary cause of ocean "
                    "acidification?"
                ),
                question_type=QuestionType.FACTUAL,
                context_passage=(
                    "Ocean acidification is the ongoing decrease in the pH of the "
                    "Earth's ocean, caused primarily by the uptake of carbon dioxide "
                    "(CO2) from the atmosphere. Since the beginning of the industrial "
                    "era, the ocean has absorbed about 30% of anthropogenic CO2 "
                    "emissions. When CO2 dissolves in seawater, it forms carbonic "
                    "acid, which releases hydrogen ions and lowers the pH. The average "
                    "ocean surface pH has decreased from approximately 8.2 to 8.1, "
                    "representing a 26% increase in acidity."
                ),
                reference_answers=[
                    "The primary cause is the uptake of carbon dioxide (CO2) from the atmosphere.",
                ],
                evidence_passages=[
                    "Ocean acidification is caused primarily by the uptake of carbon "
                    "dioxide (CO2) from the atmosphere.",
                ],
                difficulty=0.3,
            ),
            QAPrompt(
                question="What is the Pythagorean theorem?",
                question_type=QuestionType.FACTUAL,
                reference_answers=[
                    "In a right triangle, the square of the hypotenuse equals the sum "
                    "of the squares of the other two sides: a² + b² = c².",
                ],
                difficulty=0.2,
            ),
            QAPrompt(
                question="What is the primary function of mitochondria?",
                question_type=QuestionType.FACTUAL,
                reference_answers=[
                    "Mitochondria produce ATP through cellular respiration, serving as "
                    "the cell's primary energy source.",
                    "Energy production (ATP synthesis).",
                ],
                difficulty=0.3,
            ),
            QAPrompt(
                question="What programming language was created by Guido van Rossum?",
                question_type=QuestionType.FACTUAL,
                reference_answers=["Python"],
                difficulty=0.1,
            ),
            QAPrompt(
                question="What is the Heisenberg Uncertainty Principle?",
                question_type=QuestionType.FACTUAL,
                reference_answers=[
                    "It states that it is impossible to simultaneously know the exact "
                    "position and exact momentum of a particle with arbitrary precision.",
                ],
                difficulty=0.4,
            ),
        ]
        return prompts

    def _generate_analytical_prompts(self) -> List[QAPrompt]:
        """Questions requiring analysis, reasoning, or explanation."""
        prompts = [
            QAPrompt(
                question=(
                    "Why did the Roman Empire decline and eventually fall? Analyze the "
                    "key contributing factors."
                ),
                question_type=QuestionType.ANALYTICAL,
                required_perspectives=["political", "economic", "social"],
                difficulty=0.7,
            ),
            QAPrompt(
                question=(
                    "Explain how natural selection drives evolution. Use a specific "
                    "example to illustrate your explanation."
                ),
                question_type=QuestionType.ANALYTICAL,
                reference_answers=[
                    "Natural selection works through variation, inheritance, and "
                    "differential survival. Organisms with traits better suited to "
                    "their environment are more likely to survive and reproduce.",
                ],
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "What are the economic implications of an aging population in "
                    "developed countries?"
                ),
                question_type=QuestionType.ANALYTICAL,
                required_perspectives=["economic", "social", "political"],
                difficulty=0.7,
            ),
            QAPrompt(
                question=(
                    "Based on the following data, what conclusions can be drawn about "
                    "the relationship between education level and income?"
                ),
                question_type=QuestionType.ANALYTICAL,
                context_passage=(
                    "A longitudinal study tracked 10,000 individuals over 20 years. "
                    "Those with high school diplomas earned a median of $35,000/year, "
                    "bachelor's degree holders earned $55,000/year, master's degree "
                    "holders earned $70,000/year, and doctoral degree holders earned "
                    "$85,000/year. However, the study also found that individuals who "
                    "entered skilled trades earned a median of $52,000/year without a "
                    "college degree. Job satisfaction was highest among doctoral degree "
                    "holders (78%) and skilled trade workers (75%), and lowest among "
                    "high school diploma holders (45%)."
                ),
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "Why is biodiversity important for ecosystem stability? Explain "
                    "the underlying mechanisms."
                ),
                question_type=QuestionType.ANALYTICAL,
                required_perspectives=["scientific", "environmental"],
                difficulty=0.6,
            ),
            QAPrompt(
                question="Analyze the causes and consequences of the 2008 financial crisis.",
                question_type=QuestionType.ANALYTICAL,
                required_perspectives=["economic", "political", "social"],
                difficulty=0.8,
            ),
            QAPrompt(
                question=(
                    "How does the greenhouse effect work, and why is human activity "
                    "intensifying it?"
                ),
                question_type=QuestionType.ANALYTICAL,
                required_perspectives=["scientific", "environmental"],
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "What role does cognitive bias play in decision-making? Provide "
                    "examples of at least three biases."
                ),
                question_type=QuestionType.ANALYTICAL,
                required_perspectives=["psychological"],
                difficulty=0.6,
            ),
        ]
        return prompts

    def _generate_open_ended_prompts(self) -> List[QAPrompt]:
        """Open-ended questions that admit a variety of valid answers."""
        prompts = [
            QAPrompt(
                question="What does it mean to live a good life?",
                question_type=QuestionType.OPEN_ENDED,
                required_perspectives=["philosophical", "psychological", "social"],
                difficulty=0.7,
            ),
            QAPrompt(
                question=(
                    "How might artificial intelligence change the nature of work in "
                    "the next 20 years?"
                ),
                question_type=QuestionType.OPEN_ENDED,
                required_perspectives=["technological", "economic", "social"],
                difficulty=0.6,
            ),
            QAPrompt(
                question="What are the most important skills for the 21st century?",
                question_type=QuestionType.OPEN_ENDED,
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "How should society balance individual freedom with collective "
                    "well-being?"
                ),
                question_type=QuestionType.OPEN_ENDED,
                required_perspectives=["political", "ethical", "philosophical"],
                difficulty=0.8,
            ),
            QAPrompt(
                question="What makes a great leader?",
                question_type=QuestionType.OPEN_ENDED,
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "How can cities be redesigned to be more sustainable and "
                    "liveable?"
                ),
                question_type=QuestionType.OPEN_ENDED,
                required_perspectives=["environmental", "social", "technological"],
                difficulty=0.6,
            ),
            QAPrompt(
                question="What is the purpose of art in human society?",
                question_type=QuestionType.OPEN_ENDED,
                required_perspectives=["philosophical", "social", "historical"],
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "How can we ensure equitable access to education globally?"
                ),
                question_type=QuestionType.OPEN_ENDED,
                required_perspectives=["economic", "social", "political", "technological"],
                difficulty=0.7,
            ),
        ]
        return prompts

    def _generate_comparative_prompts(self) -> List[QAPrompt]:
        """Questions that ask the respondent to compare or contrast."""
        prompts = [
            QAPrompt(
                question=(
                    "Compare and contrast renewable energy sources (solar, wind) with "
                    "fossil fuels in terms of environmental impact, cost, and "
                    "reliability."
                ),
                question_type=QuestionType.COMPARATIVE,
                required_perspectives=["environmental", "economic", "technological"],
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "What are the key differences between supervised and unsupervised "
                    "machine learning?"
                ),
                question_type=QuestionType.COMPARATIVE,
                reference_answers=[
                    "Supervised learning uses labeled data and predicts known outputs, "
                    "while unsupervised learning finds hidden patterns in unlabeled data.",
                ],
                difficulty=0.4,
            ),
            QAPrompt(
                question=(
                    "Compare the parliamentary and presidential systems of government. "
                    "What are the advantages of each?"
                ),
                question_type=QuestionType.COMPARATIVE,
                required_perspectives=["political"],
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "How do classical and operant conditioning differ? In what "
                    "situations is each more effective?"
                ),
                question_type=QuestionType.COMPARATIVE,
                required_perspectives=["psychological"],
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "Compare the economic philosophies of Keynesianism and "
                    "monetarism."
                ),
                question_type=QuestionType.COMPARATIVE,
                required_perspectives=["economic", "political"],
                difficulty=0.7,
            ),
            QAPrompt(
                question=(
                    "What are the trade-offs between relational databases and NoSQL "
                    "databases?"
                ),
                question_type=QuestionType.COMPARATIVE,
                reference_answers=[
                    "Relational databases offer strong consistency and complex queries "
                    "via SQL, while NoSQL databases provide flexibility, horizontal "
                    "scalability, and are better suited for unstructured data.",
                ],
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "Compare the French and American revolutions. How were their "
                    "causes and outcomes similar and different?"
                ),
                question_type=QuestionType.COMPARATIVE,
                required_perspectives=["historical", "political", "social"],
                difficulty=0.7,
            ),
        ]
        return prompts

    def _generate_hypothetical_prompts(self) -> List[QAPrompt]:
        """Hypothetical / counterfactual questions."""
        prompts = [
            QAPrompt(
                question=(
                    "What would happen to Earth's climate if the Amazon rainforest "
                    "were completely destroyed?"
                ),
                question_type=QuestionType.HYPOTHETICAL,
                required_perspectives=["environmental", "scientific"],
                difficulty=0.7,
            ),
            QAPrompt(
                question=(
                    "If humans could photosynthesize, how would society and "
                    "agriculture change?"
                ),
                question_type=QuestionType.HYPOTHETICAL,
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "How might history have been different if the printing press had "
                    "never been invented?"
                ),
                question_type=QuestionType.HYPOTHETICAL,
                required_perspectives=["historical", "social", "technological"],
                difficulty=0.7,
            ),
            QAPrompt(
                question=(
                    "What challenges would arise if we discovered microbial life on "
                    "Mars?"
                ),
                question_type=QuestionType.HYPOTHETICAL,
                required_perspectives=["scientific", "ethical", "political"],
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "If a universal basic income were implemented worldwide, what "
                    "would the economic and social effects be?"
                ),
                question_type=QuestionType.HYPOTHETICAL,
                required_perspectives=["economic", "social", "political"],
                difficulty=0.7,
            ),
        ]
        return prompts

    def _generate_opinion_prompts(self) -> List[QAPrompt]:
        """Questions soliciting and evaluating well-reasoned opinions."""
        prompts = [
            QAPrompt(
                question=(
                    "Should genetic engineering be used to enhance human capabilities "
                    "beyond treating diseases?"
                ),
                question_type=QuestionType.OPINION,
                required_perspectives=["ethical", "scientific", "social"],
                difficulty=0.7,
            ),
            QAPrompt(
                question="Is social media a net positive or net negative for society?",
                question_type=QuestionType.OPINION,
                required_perspectives=["social", "psychological", "technological"],
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "Do you think space exploration is worth the investment? Why or "
                    "why not?"
                ),
                question_type=QuestionType.OPINION,
                required_perspectives=["scientific", "economic"],
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "Should governments regulate artificial intelligence, and if so, "
                    "how?"
                ),
                question_type=QuestionType.OPINION,
                required_perspectives=["political", "technological", "ethical"],
                difficulty=0.7,
            ),
            QAPrompt(
                question="Is it ethical to use animals in scientific research?",
                question_type=QuestionType.OPINION,
                required_perspectives=["ethical", "scientific"],
                difficulty=0.6,
            ),
        ]
        return prompts

    def _generate_procedural_prompts(self) -> List[QAPrompt]:
        """How-to / process-oriented questions."""
        prompts = [
            QAPrompt(
                question=(
                    "Explain the step-by-step process of how a bill becomes a law in "
                    "the United States."
                ),
                question_type=QuestionType.PROCEDURAL,
                difficulty=0.4,
            ),
            QAPrompt(
                question="How do you design and conduct a randomized controlled trial?",
                question_type=QuestionType.PROCEDURAL,
                required_perspectives=["scientific"],
                difficulty=0.6,
            ),
            QAPrompt(
                question=(
                    "Describe the process of photosynthesis from light absorption to "
                    "glucose production."
                ),
                question_type=QuestionType.PROCEDURAL,
                reference_answers=[
                    "Photosynthesis occurs in two stages: the light-dependent reactions "
                    "in the thylakoid membranes (which produce ATP and NADPH) and the "
                    "Calvin cycle in the stroma (which fixes CO2 into glucose).",
                ],
                difficulty=0.5,
            ),
            QAPrompt(
                question="How do you perform a code review effectively?",
                question_type=QuestionType.PROCEDURAL,
                difficulty=0.4,
            ),
            QAPrompt(
                question=(
                    "What is the process for training a neural network from scratch?"
                ),
                question_type=QuestionType.PROCEDURAL,
                required_perspectives=["technological"],
                difficulty=0.5,
            ),
        ]
        return prompts

    def _generate_clarification_prompts(self) -> List[QAPrompt]:
        """Questions that require disambiguating or clarifying concepts."""
        prompts = [
            QAPrompt(
                question=(
                    "What is the difference between correlation and causation? Why is "
                    "this distinction important?"
                ),
                question_type=QuestionType.CLARIFICATION,
                reference_answers=[
                    "Correlation means two variables are related or tend to occur "
                    "together, while causation means one directly causes the other. "
                    "Confusing them leads to incorrect conclusions.",
                ],
                difficulty=0.4,
            ),
            QAPrompt(
                question=(
                    "Clarify the difference between weather and climate. How are they "
                    "related?"
                ),
                question_type=QuestionType.CLARIFICATION,
                reference_answers=[
                    "Weather is the short-term atmospheric conditions at a specific "
                    "place and time, while climate is the average of weather patterns "
                    "over a long period (typically 30 years).",
                ],
                difficulty=0.3,
            ),
            QAPrompt(
                question=(
                    "What does 'statistical significance' mean, and why can it be "
                    "misleading?"
                ),
                question_type=QuestionType.CLARIFICATION,
                difficulty=0.5,
            ),
            QAPrompt(
                question=(
                    "Explain the distinction between machine learning, deep learning, "
                    "and artificial intelligence."
                ),
                question_type=QuestionType.CLARIFICATION,
                reference_answers=[
                    "AI is the broadest concept (machines mimicking intelligence), ML "
                    "is a subset (learning from data), and deep learning is a further "
                    "subset (using deep neural networks).",
                ],
                difficulty=0.4,
            ),
            QAPrompt(
                question=(
                    "What is the difference between Type I and Type II errors in "
                    "hypothesis testing?"
                ),
                question_type=QuestionType.CLARIFICATION,
                reference_answers=[
                    "A Type I error (false positive) occurs when you reject a true "
                    "null hypothesis. A Type II error (false negative) occurs when you "
                    "fail to reject a false null hypothesis.",
                ],
                difficulty=0.4,
            ),
        ]
        return prompts
