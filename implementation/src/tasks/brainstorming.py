"""
Brainstorming task domain for the Diversity Decoding Arena.

Evaluates diversity-promoting decoding algorithms on brainstorming tasks:
generating creative ideas, product concepts, solutions, strategies, and more.
Measures novelty, feasibility, specificity, diversity of ideas, relevance,
elaboration, actionability, impact, and creativity across generated outputs.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
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
# Enums
# ---------------------------------------------------------------------------


class BrainstormCategory(Enum):
    """Category of brainstorming task."""

    PRODUCT_IDEAS = auto()
    SOLUTIONS = auto()
    NAMES = auto()
    STRATEGIES = auto()
    CREATIVE_USES = auto()
    IMPROVEMENTS = auto()
    RESEARCH_QUESTIONS = auto()
    BUSINESS_MODELS = auto()

    def __repr__(self) -> str:
        return f"BrainstormCategory.{self.name}"

    @property
    def description(self) -> str:
        _desc = {
            BrainstormCategory.PRODUCT_IDEAS: "Generate novel product or feature ideas",
            BrainstormCategory.SOLUTIONS: "Brainstorm solutions to a given problem",
            BrainstormCategory.NAMES: "Come up with creative names or titles",
            BrainstormCategory.STRATEGIES: "Develop strategic approaches or plans",
            BrainstormCategory.CREATIVE_USES: "Find unconventional uses for an object or concept",
            BrainstormCategory.IMPROVEMENTS: "Suggest improvements to existing systems",
            BrainstormCategory.RESEARCH_QUESTIONS: "Formulate research questions or hypotheses",
            BrainstormCategory.BUSINESS_MODELS: "Design business model concepts",
        }
        return _desc[self]


class IdeaComplexity(Enum):
    """Expected complexity level for generated ideas."""

    SIMPLE = auto()
    MODERATE = auto()
    COMPLEX = auto()
    INNOVATIVE = auto()

    def __repr__(self) -> str:
        return f"IdeaComplexity.{self.name}"

    @property
    def min_word_count(self) -> int:
        return {
            IdeaComplexity.SIMPLE: 5,
            IdeaComplexity.MODERATE: 15,
            IdeaComplexity.COMPLEX: 30,
            IdeaComplexity.INNOVATIVE: 25,
        }[self]

    @property
    def elaboration_expectation(self) -> float:
        """Expected elaboration score 0-1 for this complexity."""
        return {
            IdeaComplexity.SIMPLE: 0.2,
            IdeaComplexity.MODERATE: 0.5,
            IdeaComplexity.COMPLEX: 0.8,
            IdeaComplexity.INNOVATIVE: 0.7,
        }[self]


# ---------------------------------------------------------------------------
# Keyword / heuristic lexicons
# ---------------------------------------------------------------------------

_FEASIBILITY_POSITIVE = frozenset({
    "using", "with", "by", "through", "via", "implement", "build", "create",
    "design", "deploy", "integrate", "leverage", "utilise", "utilize", "apply",
    "adopt", "install", "configure", "develop", "test", "measure", "track",
    "monitor", "automate", "streamline", "optimize", "reduce", "increase",
    "improve", "enhance", "scale", "extend", "modify", "adjust", "connect",
    "combine", "partner", "collaborate", "hire", "train", "launch", "release",
    "publish", "distribute", "market", "sell", "license", "open-source",
})

_FEASIBILITY_NEGATIVE = frozenset({
    "impossible", "unrealistic", "fantasy", "magic", "teleport", "teleportation",
    "perpetual", "infinite", "unlimited", "omniscient", "omnipotent",
    "time travel", "faster than light", "warp", "supernatural", "miracle",
})

_VAGUE_PHRASES = frozenset({
    "something", "somehow", "maybe", "perhaps", "kind of", "sort of",
    "stuff", "thing", "things", "whatever", "etc", "and so on",
    "various", "some kind", "do better", "improve things", "make it good",
    "be creative", "think outside the box", "innovative solution",
    "leverage synergies", "paradigm shift",
})

_IMPACT_KEYWORDS = frozenset({
    "revenue", "profit", "growth", "save", "cost", "efficiency", "productivity",
    "engagement", "retention", "acquisition", "conversion", "satisfaction",
    "safety", "health", "education", "sustainability", "environment", "climate",
    "equity", "access", "inclusion", "community", "scale", "global", "millions",
    "billions", "transform", "disrupt", "revolutionize", "breakthrough",
})

_CREATIVITY_MARKERS = frozenset({
    "combine", "merge", "fusion", "hybrid", "cross", "interdisciplinary",
    "unexpected", "surprising", "unconventional", "novel", "twist", "reimagine",
    "reinvent", "repurpose", "gamif", "reverse", "invert", "flip",
    "biomimicry", "analog", "metaphor", "inspired by", "crowdsourc",
})

_ACTION_VERBS = frozenset({
    "create", "build", "design", "develop", "implement", "launch", "deploy",
    "test", "measure", "analyze", "research", "survey", "interview",
    "prototype", "pilot", "iterate", "refine", "scale", "partner",
    "invest", "hire", "train", "automate", "integrate", "migrate",
    "restructure", "rebrand", "reposition", "diversify", "consolidate",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "or", "if", "that", "this", "it", "its", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BrainstormingConfig(TaskConfig):
    """Configuration for brainstorming task evaluation."""

    category: BrainstormCategory = BrainstormCategory.PRODUCT_IDEAS
    min_ideas: int = 5
    max_ideas: int = 20
    require_explanations: bool = True
    novelty_threshold: float = 0.3
    feasibility_weight: float = 0.15
    topic_constraint: Optional[str] = None

    # Scoring weights
    novelty_weight: float = 0.15
    specificity_weight: float = 0.10
    diversity_weight: float = 0.15
    relevance_weight: float = 0.10
    elaboration_weight: float = 0.10
    actionability_weight: float = 0.10
    impact_weight: float = 0.05
    creativity_weight: float = 0.10

    idea_complexity: IdeaComplexity = IdeaComplexity.MODERATE
    penalize_redundancy: bool = True
    redundancy_threshold: float = 0.6
    structural_bonus: float = 0.05
    cluster_diversity_bonus: float = 0.05

    def __post_init__(self) -> None:
        if self.min_ideas < 1:
            raise ValueError("min_ideas must be >= 1")
        if self.max_ideas < self.min_ideas:
            raise ValueError("max_ideas must be >= min_ideas")
        if not 0.0 <= self.novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be in [0, 1]")
        total = (
            self.novelty_weight
            + self.feasibility_weight
            + self.specificity_weight
            + self.diversity_weight
            + self.relevance_weight
            + self.elaboration_weight
            + self.actionability_weight
            + self.impact_weight
            + self.creativity_weight
        )
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Scoring weights should sum to ~1.0, got {total:.3f}"
            )

    @property
    def task_domain(self) -> TaskDomain:
        return TaskDomain.OPEN_ENDED_GENERATION


@dataclass
class BrainstormPrompt(TaskPrompt):
    """A single brainstorming prompt with metadata."""

    category: BrainstormCategory = BrainstormCategory.PRODUCT_IDEAS
    topic: str = ""
    context: str = ""
    constraints_text: str = ""
    target_audience: str = ""
    existing_solutions: List[str] = field(default_factory=list)

    # Internal fields populated during evaluation
    _expected_min_ideas: int = field(default=5, repr=False)
    _expected_complexity: IdeaComplexity = field(
        default=IdeaComplexity.MODERATE, repr=False
    )

    @property
    def full_context(self) -> str:
        parts = [self.topic]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.constraints_text:
            parts.append(f"Constraints: {self.constraints_text}")
        if self.target_audience:
            parts.append(f"Target audience: {self.target_audience}")
        if self.existing_solutions:
            parts.append(
                "Known solutions: " + "; ".join(self.existing_solutions)
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer with punctuation stripping."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [w for w in text.split() if len(w) > 1]


def _content_words(text: str) -> List[str]:
    """Return non-stopword tokens."""
    return [w for w in _tokenize(text) if w not in _STOPWORDS]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _cosine_bow(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Cosine similarity using bag-of-words vectors."""
    vocab: Set[str] = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0
    idx = {w: i for i, w in enumerate(sorted(vocab))}
    va = np.zeros(len(vocab))
    vb = np.zeros(len(vocab))
    for w in tokens_a:
        va[idx[w]] += 1.0
    for w in tokens_b:
        vb[idx[w]] += 1.0
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalised Levenshtein similarity (1 = identical)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        matrix[i, 0] = i
    for j in range(len2 + 1):
        matrix[0, j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,
                matrix[i, j - 1] + 1,
                matrix[i - 1, j - 1] + cost,
            )
    max_len = max(len1, len2)
    return 1.0 - matrix[len1, len2] / max_len


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _count_pattern_matches(text: str, patterns: frozenset) -> int:
    """Count how many patterns appear in text (case-insensitive)."""
    lower = text.lower()
    return sum(1 for p in patterns if p in lower)


# ---------------------------------------------------------------------------
# BrainstormingTask
# ---------------------------------------------------------------------------


class BrainstormingTask(GenerationTask):
    """
    Brainstorming task for evaluating how well decoding algorithms produce
    diverse, creative, and useful sets of ideas in response to brainstorming
    prompts.

    Scoring dimensions:
      - Novelty: How original are the ideas relative to known solutions?
      - Feasibility: Are the ideas practically implementable?
      - Specificity: Concrete details vs. vague hand-waving.
      - Diversity: Semantic spread across generated ideas.
      - Relevance: On-topic with respect to the prompt.
      - Elaboration: Depth of explanation for each idea.
      - Category coverage: Do ideas span different angles/themes?
      - Actionability: Can the ideas be acted upon directly?
      - Impact potential: Estimated real-world impact.
      - Creativity: Unexpected combinations and novel framings.
      - Redundancy penalty: Overlap reduction.
      - Structural quality: Organisation and readability bonus.
    """

    def __init__(
        self,
        config: Optional[BrainstormingConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.config: BrainstormingConfig = config or BrainstormingConfig()
        super().__init__(config=self.config, **kwargs)
        self._prompts: Optional[PromptDataset] = None
        self._rng = np.random.RandomState(self.config.seed if hasattr(self.config, "seed") else 42)

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Build and return the full prompt dataset (40+ prompts)."""
        if self._prompts is not None:
            return self._prompts

        all_prompts: List[BrainstormPrompt] = []
        all_prompts.extend(self._generate_product_prompts())
        all_prompts.extend(self._generate_solution_prompts())
        all_prompts.extend(self._generate_creative_use_prompts())
        all_prompts.extend(self._generate_strategy_prompts())
        all_prompts.extend(self._generate_name_prompts())
        all_prompts.extend(self._generate_improvement_prompts())
        all_prompts.extend(self._generate_research_prompts())
        all_prompts.extend(self._generate_business_model_prompts())

        self._prompts = PromptDataset(prompts=all_prompts)
        return self._prompts

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, prompt: BrainstormPrompt) -> str:  # type: ignore[override]
        """Format a BrainstormPrompt into a full instruction string."""
        lines: List[str] = []

        # Header based on category
        cat_instructions = {
            BrainstormCategory.PRODUCT_IDEAS: (
                "You are an innovative product designer. Brainstorm creative "
                "product ideas for the following topic."
            ),
            BrainstormCategory.SOLUTIONS: (
                "You are a creative problem solver. Brainstorm diverse solutions "
                "to the following challenge."
            ),
            BrainstormCategory.NAMES: (
                "You are a naming expert. Generate creative and memorable names "
                "for the following."
            ),
            BrainstormCategory.STRATEGIES: (
                "You are a strategic consultant. Develop varied strategies for "
                "the following scenario."
            ),
            BrainstormCategory.CREATIVE_USES: (
                "You are an inventive thinker. Come up with unconventional, "
                "creative uses for the following."
            ),
            BrainstormCategory.IMPROVEMENTS: (
                "You are a continuous-improvement specialist. Suggest meaningful "
                "improvements for the following."
            ),
            BrainstormCategory.RESEARCH_QUESTIONS: (
                "You are a research methodologist. Formulate insightful research "
                "questions about the following topic."
            ),
            BrainstormCategory.BUSINESS_MODELS: (
                "You are a business strategist. Design innovative business model "
                "concepts for the following."
            ),
        }

        lines.append(cat_instructions.get(
            prompt.category,
            "Brainstorm ideas for the following topic.",
        ))
        lines.append("")

        lines.append(f"Topic: {prompt.topic}")
        if prompt.context:
            lines.append(f"Background: {prompt.context}")
        if prompt.target_audience:
            lines.append(f"Target audience: {prompt.target_audience}")
        if prompt.constraints_text:
            lines.append(f"Constraints: {prompt.constraints_text}")
        if prompt.existing_solutions:
            lines.append(
                "Already known approaches (try to go beyond these): "
                + "; ".join(prompt.existing_solutions)
            )

        lines.append("")

        min_ideas = max(self.config.min_ideas, prompt._expected_min_ideas)
        lines.append(f"Generate at least {min_ideas} distinct ideas.")

        if self.config.require_explanations:
            lines.append(
                "For each idea, provide a brief explanation of how it works "
                "and why it could be effective."
            )

        lines.append(
            "Number your ideas (1., 2., 3., …) and make each one specific "
            "and actionable."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Constraint construction
    # ------------------------------------------------------------------

    def get_constraints(self) -> List[TaskConstraint]:
        """Return structural constraints for brainstorming outputs."""
        constraints: List[TaskConstraint] = []

        constraints.append(TaskConstraint(
            name="min_idea_count",
            description=f"At least {self.config.min_ideas} distinct ideas",
            check_fn=lambda text: self._count_ideas(text) >= self.config.min_ideas,
        ))

        constraints.append(TaskConstraint(
            name="max_idea_count",
            description=f"No more than {self.config.max_ideas} ideas",
            check_fn=lambda text: self._count_ideas(text) <= self.config.max_ideas,
        ))

        constraints.append(TaskConstraint(
            name="relevance_floor",
            description="Ideas must be relevant to the topic",
            check_fn=lambda text: True,  # evaluated via scoring
        ))

        if self.config.require_explanations:
            constraints.append(TaskConstraint(
                name="explanations_present",
                description="Each idea should include a brief explanation",
                check_fn=lambda text: self._check_explanations(text),
            ))

        if self.config.topic_constraint:
            topic = self.config.topic_constraint

            def _topic_check(text: str, _topic: str = topic) -> bool:
                return _topic.lower() in text.lower()

            constraints.append(TaskConstraint(
                name="topic_mentioned",
                description=f"Output must reference the topic '{topic}'",
                check_fn=_topic_check,
            ))

        return constraints

    def _check_explanations(self, text: str) -> bool:
        """Heuristic: each numbered idea should be >= 2 sentences."""
        ideas = self._parse_ideas(text)
        if not ideas:
            return False
        short_count = sum(
            1 for idea in ideas if len(_sentence_split(idea)) < 2
        )
        return short_count <= len(ideas) * 0.3

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generations: List[str],
        prompts: List[BrainstormPrompt],  # type: ignore[override]
    ) -> Dict[str, Any]:
        """
        Score a batch of generated brainstorming outputs.

        Returns per-sample scores and aggregate statistics.
        """
        if len(generations) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(generations)} generations vs "
                f"{len(prompts)} prompts."
            )

        per_sample: List[Dict[str, Any]] = []
        all_scores: List[float] = []

        for gen_text, prompt in zip(generations, prompts):
            ideas = self._parse_ideas(gen_text)
            topic = prompt.topic
            existing = prompt.existing_solutions

            # Compute individual dimension scores
            novelty = self._novelty_score(ideas, existing)
            feasibility = self._feasibility_score(ideas)
            specificity = self._specificity_score(ideas)
            diversity = self._diversity_of_ideas(ideas)
            relevance = self._relevance_score(ideas, topic)
            elaboration = self._elaboration_score(ideas)
            cat_cov = self._category_coverage(ideas)
            actionability = self._actionability_score(ideas)
            impact = self._impact_potential(ideas)
            creativity = self._creativity_index(ideas)

            # Penalties / bonuses
            redundancy_pen = self._redundancy_penalty(ideas)
            structural = self._structural_quality(gen_text)

            # Weighted composite
            composite = (
                self.config.novelty_weight * novelty
                + self.config.feasibility_weight * feasibility
                + self.config.specificity_weight * specificity
                + self.config.diversity_weight * diversity
                + self.config.relevance_weight * relevance
                + self.config.elaboration_weight * elaboration
                + self.config.actionability_weight * actionability
                + self.config.impact_weight * impact
                + self.config.creativity_weight * creativity
            )

            # Apply redundancy penalty
            if self.config.penalize_redundancy:
                composite *= (1.0 - 0.5 * redundancy_pen)

            # Structural bonus
            composite += self.config.structural_bonus * structural

            # Cluster diversity bonus
            composite += self.config.cluster_diversity_bonus * cat_cov

            composite = float(np.clip(composite, 0.0, 1.0))

            sample_result: Dict[str, Any] = {
                "composite_score": composite,
                "idea_count": len(ideas),
                "novelty": novelty,
                "feasibility": feasibility,
                "specificity": specificity,
                "diversity": diversity,
                "relevance": relevance,
                "elaboration": elaboration,
                "category_coverage": cat_cov,
                "actionability": actionability,
                "impact": impact,
                "creativity": creativity,
                "redundancy_penalty": redundancy_pen,
                "structural_quality": structural,
                "ideas": ideas,
                "clusters": self._cluster_ideas(ideas),
            }
            per_sample.append(sample_result)
            all_scores.append(composite)

        scores_arr = np.array(all_scores) if all_scores else np.array([0.0])

        return {
            "per_sample": per_sample,
            "mean_score": float(np.mean(scores_arr)),
            "std_score": float(np.std(scores_arr)),
            "median_score": float(np.median(scores_arr)),
            "min_score": float(np.min(scores_arr)),
            "max_score": float(np.max(scores_arr)),
            "num_samples": len(generations),
            "mean_idea_count": float(np.mean([
                s["idea_count"] for s in per_sample
            ])) if per_sample else 0.0,
        }

    # ------------------------------------------------------------------
    # Idea parsing
    # ------------------------------------------------------------------

    _IDEA_PATTERNS = [
        # "1. Idea text" or "1) Idea text"
        re.compile(r"^\s*(\d+)\s*[.)]\s*(.+)", re.MULTILINE),
        # "- Idea text" or "* Idea text"
        re.compile(r"^\s*[-*•]\s+(.+)", re.MULTILINE),
        # "Idea N: text"
        re.compile(r"^\s*(?:Idea|Option|Concept)\s*\d+\s*:\s*(.+)", re.MULTILINE | re.IGNORECASE),
    ]

    def _parse_ideas(self, text: str) -> List[str]:
        """
        Extract individual ideas from generated text.

        Tries numbered lists first, then bullet lists, then falls back to
        paragraph splitting.
        """
        if not text or not text.strip():
            return []

        # Strategy 1: numbered list  "1. … 2. …"
        numbered = re.findall(
            r"^\s*\d+\s*[.)]\s*(.+?)(?=\n\s*\d+\s*[.)]|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if len(numbered) >= 2:
            return [self._clean_idea(idea) for idea in numbered if idea.strip()]

        # Strategy 2: bullet list
        bullets = re.findall(
            r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if len(bullets) >= 2:
            return [self._clean_idea(idea) for idea in bullets if idea.strip()]

        # Strategy 3: "Idea N:" pattern
        labelled = re.findall(
            r"(?:Idea|Option|Concept)\s*\d+\s*:\s*(.+?)(?=(?:Idea|Option|Concept)\s*\d+\s*:|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if len(labelled) >= 2:
            return [self._clean_idea(idea) for idea in labelled if idea.strip()]

        # Strategy 4: paragraph splitting (fallback)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            return [self._clean_idea(p) for p in paragraphs]

        # Last resort: sentence-level split
        sentences = _sentence_split(text)
        if len(sentences) >= 3:
            return [self._clean_idea(s) for s in sentences]

        return [self._clean_idea(text)] if text.strip() else []

    @staticmethod
    def _clean_idea(text: str) -> str:
        """Normalise whitespace in a single idea."""
        text = re.sub(r"\s+", " ", text)
        return text.strip().rstrip(".")

    def _count_ideas(self, text: str) -> int:
        """Count distinct ideas in generated text."""
        return len(self._parse_ideas(text))

    # ------------------------------------------------------------------
    # Scoring dimensions
    # ------------------------------------------------------------------

    def _novelty_score(
        self, ideas: List[str], existing_solutions: List[str]
    ) -> float:
        """
        Measure how novel the ideas are compared to a list of known solutions.

        Uses bag-of-words cosine similarity; ideas with low max-similarity to
        any existing solution count as novel.
        """
        if not ideas:
            return 0.0
        if not existing_solutions:
            return 0.75  # no baseline → assume moderate novelty

        existing_tokens = [_content_words(sol) for sol in existing_solutions]
        novelty_scores: List[float] = []

        for idea in ideas:
            idea_tokens = _content_words(idea)
            if not idea_tokens:
                novelty_scores.append(0.5)
                continue
            max_sim = max(
                _cosine_bow(idea_tokens, et) for et in existing_tokens
            ) if existing_tokens else 0.0
            novelty_scores.append(1.0 - max_sim)

        raw = float(np.mean(novelty_scores))
        # Bonus for having at least some highly-novel ideas
        highly_novel = sum(1 for s in novelty_scores if s > 0.7)
        bonus = min(0.15, 0.05 * highly_novel)
        return float(np.clip(raw + bonus, 0.0, 1.0))

    def _feasibility_score(self, ideas: List[str]) -> float:
        """
        Heuristic feasibility: presence of concrete action words minus
        impossible/fantasy indicators.
        """
        if not ideas:
            return 0.0

        scores: List[float] = []
        for idea in ideas:
            lower = idea.lower()
            pos = _count_pattern_matches(idea, _FEASIBILITY_POSITIVE)
            neg = _count_pattern_matches(idea, _FEASIBILITY_NEGATIVE)
            word_count = len(lower.split())
            if word_count == 0:
                scores.append(0.0)
                continue
            # Normalise positive matches by word count
            pos_ratio = min(pos / max(word_count, 1) * 5.0, 1.0)
            neg_penalty = min(neg * 0.25, 0.8)
            score = max(pos_ratio - neg_penalty, 0.0)
            scores.append(score)

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _specificity_score(self, ideas: List[str]) -> float:
        """
        Concrete vs. vague: penalise vague phrases, reward numbers, proper
        nouns, and technical detail.
        """
        if not ideas:
            return 0.0

        scores: List[float] = []
        for idea in ideas:
            vague_count = _count_pattern_matches(idea, _VAGUE_PHRASES)
            # Reward specifics
            has_numbers = len(re.findall(r"\d+", idea))
            has_proper_nouns = len(re.findall(r"\b[A-Z][a-z]{2,}", idea))
            word_count = max(len(idea.split()), 1)
            detail_density = (has_numbers + has_proper_nouns) / word_count

            vague_ratio = vague_count / word_count
            score = 0.5 + detail_density * 3.0 - vague_ratio * 3.0
            scores.append(float(np.clip(score, 0.0, 1.0)))

        return float(np.mean(scores))

    def _diversity_of_ideas(self, ideas: List[str]) -> float:
        """
        Semantic spread: average pairwise dissimilarity (1 - cosine) among
        ideas.
        """
        if len(ideas) < 2:
            return 0.0

        tokenized = [_content_words(idea) for idea in ideas]
        dissimilarities: List[float] = []

        for i, j in combinations(range(len(ideas)), 2):
            sim = _cosine_bow(tokenized[i], tokenized[j])
            dissimilarities.append(1.0 - sim)

        raw = float(np.mean(dissimilarities))
        # Bonus for distinct clusters
        clusters = self._cluster_ideas(ideas)
        cluster_bonus = min(len(clusters) * 0.05, 0.2)
        return float(np.clip(raw + cluster_bonus, 0.0, 1.0))

    def _relevance_score(self, ideas: List[str], topic: str) -> float:
        """How relevant each idea is to the prompt topic."""
        if not ideas or not topic:
            return 0.0

        topic_tokens = set(_content_words(topic))
        if not topic_tokens:
            return 0.5

        scores: List[float] = []
        for idea in ideas:
            idea_tokens = set(_content_words(idea))
            if not idea_tokens:
                scores.append(0.0)
                continue
            overlap = len(idea_tokens & topic_tokens) / len(topic_tokens)
            # Softer metric: at least some connection
            jaccard = _jaccard(idea_tokens, topic_tokens)
            score = 0.6 * min(overlap, 1.0) + 0.4 * jaccard
            scores.append(score)

        raw = float(np.mean(scores))
        # Ensure minimum if ideas are at least loosely on-topic
        if raw < 0.1 and any(
            topic.lower().split()[0] in idea.lower() for idea in ideas
        ):
            raw = 0.15
        return float(np.clip(raw, 0.0, 1.0))

    def _elaboration_score(self, ideas: List[str]) -> float:
        """
        Depth of explanation: average word count per idea, normalised by
        expected complexity.
        """
        if not ideas:
            return 0.0

        expected_min = self.config.idea_complexity.min_word_count
        scores: List[float] = []
        for idea in ideas:
            wc = len(idea.split())
            ratio = wc / max(expected_min, 1)
            # Diminishing returns past 3x expected length
            score = min(ratio / 3.0, 1.0)
            scores.append(score)

        raw = float(np.mean(scores))
        # Penalise if all ideas are extremely short
        min_wc = min(len(idea.split()) for idea in ideas)
        if min_wc < 3:
            raw *= 0.8
        return float(np.clip(raw, 0.0, 1.0))

    def _category_coverage(self, ideas: List[str]) -> float:
        """
        Check whether ideas span different sub-categories / angles.
        Uses simple keyword clustering to detect distinct themes.
        """
        if not ideas:
            return 0.0

        clusters = self._cluster_ideas(ideas)
        n_clusters = len(clusters)
        n_ideas = len(ideas)

        if n_ideas <= 1:
            return 0.0

        # Ideal: roughly sqrt(n_ideas) distinct clusters
        ideal_clusters = max(2, int(math.sqrt(n_ideas)))
        ratio = n_clusters / ideal_clusters
        return float(np.clip(ratio, 0.0, 1.0))

    def _actionability_score(self, ideas: List[str]) -> float:
        """How actionable each idea is: presence of action verbs and steps."""
        if not ideas:
            return 0.0

        scores: List[float] = []
        for idea in ideas:
            tokens = set(_tokenize(idea))
            action_count = len(tokens & _ACTION_VERBS)
            # Check for step-like language
            has_steps = bool(re.search(
                r"(?:step|first|then|next|finally|start by|begin with)",
                idea,
                re.IGNORECASE,
            ))
            word_count = max(len(idea.split()), 1)
            score = min(action_count / max(word_count * 0.1, 1), 1.0)
            if has_steps:
                score = min(score + 0.15, 1.0)
            scores.append(score)

        return float(np.mean(scores))

    def _impact_potential(self, ideas: List[str]) -> float:
        """Estimated real-world impact based on impact keyword density."""
        if not ideas:
            return 0.0

        scores: List[float] = []
        for idea in ideas:
            hits = _count_pattern_matches(idea, _IMPACT_KEYWORDS)
            word_count = max(len(idea.split()), 1)
            density = hits / word_count
            score = min(density * 8.0, 1.0)
            scores.append(score)

        return float(np.mean(scores))

    def _creativity_index(self, ideas: List[str]) -> float:
        """
        Unexpected combinations and creative markers.

        Looks for cross-domain references, unusual adjective-noun pairs,
        and explicit creativity indicators.
        """
        if not ideas:
            return 0.0

        scores: List[float] = []
        for idea in ideas:
            markers = _count_pattern_matches(idea, _CREATIVITY_MARKERS)
            # Unusual bigrams: adjective-noun pairs that are rare
            tokens = _tokenize(idea)
            bigrams = _ngrams(tokens, 2)
            unusual_bigram_count = 0
            for bg in bigrams:
                combined = bg[0] + bg[1]
                # Heuristic: if the two words share no common substring > 3
                # they are likely from different domains
                if len(set(bg[0]) & set(bg[1])) <= 2:
                    unusual_bigram_count += 1

            word_count = max(len(tokens), 1)
            marker_score = min(markers * 0.2, 0.6)
            bigram_score = min(unusual_bigram_count / word_count * 2.0, 0.4)
            scores.append(marker_score + bigram_score)

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _redundancy_penalty(self, ideas: List[str]) -> float:
        """
        Fraction of idea pairs that are too similar (above
        config.redundancy_threshold).
        """
        if len(ideas) < 2:
            return 0.0

        tokenized = [_content_words(idea) for idea in ideas]
        n_pairs = 0
        redundant = 0

        for i, j in combinations(range(len(ideas)), 2):
            n_pairs += 1
            sim = _cosine_bow(tokenized[i], tokenized[j])
            if sim > self.config.redundancy_threshold:
                redundant += 1

        if n_pairs == 0:
            return 0.0
        return redundant / n_pairs

    def _structural_quality(self, text: str) -> float:
        """
        Assess the structural organisation of the output:
        numbered lists, headings, consistent formatting.
        """
        if not text.strip():
            return 0.0

        score = 0.0
        lines = text.strip().split("\n")

        # Numbered list detection
        numbered_lines = [
            l for l in lines if re.match(r"^\s*\d+\s*[.)]\s+", l)
        ]
        if len(numbered_lines) >= 3:
            score += 0.4

        # Consistent bullet formatting
        bullet_lines = [
            l for l in lines if re.match(r"^\s*[-*•]\s+", l)
        ]
        if len(bullet_lines) >= 3:
            score += 0.3

        # Section headings (markdown-style or bold)
        headings = [
            l for l in lines
            if re.match(r"^\s*#{1,3}\s+", l) or re.match(r"^\s*\*\*.+\*\*\s*$", l)
        ]
        if headings:
            score += 0.15

        # Consistent line length variance (lower is better)
        line_lengths = [len(l) for l in lines if l.strip()]
        if len(line_lengths) >= 3:
            cv = np.std(line_lengths) / max(np.mean(line_lengths), 1)
            consistency_bonus = max(0, 0.15 - cv * 0.1)
            score += consistency_bonus

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Higher-level analysis
    # ------------------------------------------------------------------

    def _mind_map_extraction(self, text: str) -> Dict[str, Any]:
        """
        Extract a tree-structured mind map from generated text.

        Returns a dict representing the root node with children.
        Each node: {"label": str, "children": List[node]}.
        """
        ideas = self._parse_ideas(text)
        if not ideas:
            return {"label": "root", "children": []}

        clusters = self._cluster_ideas(ideas)

        children: List[Dict[str, Any]] = []
        for theme, members in clusters.items():
            theme_node: Dict[str, Any] = {
                "label": theme,
                "children": [{"label": m, "children": []} for m in members],
            }
            children.append(theme_node)

        # Unclustered ideas
        clustered_ideas = {
            idea for members in clusters.values() for idea in members
        }
        for idea in ideas:
            if idea not in clustered_ideas:
                children.append({"label": idea, "children": []})

        return {"label": "root", "children": children}

    def _cluster_ideas(self, ideas: List[str]) -> Dict[str, List[str]]:
        """
        Group ideas by theme using keyword overlap clustering.

        Returns {theme_label: [ideas_in_cluster]}.
        """
        if not ideas:
            return {}

        tokenized = [set(_content_words(idea)) for idea in ideas]
        n = len(ideas)

        # Build similarity matrix
        sim_matrix = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            sim = _jaccard(tokenized[i], tokenized[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        # Simple agglomerative clustering via thresholding
        threshold = 0.15
        visited: Set[int] = set()
        clusters: Dict[str, List[str]] = {}

        for i in range(n):
            if i in visited:
                continue
            cluster_indices = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if sim_matrix[i, j] >= threshold:
                    cluster_indices.append(j)
                    visited.add(j)

            if len(cluster_indices) >= 2:
                # Label = most common content words in cluster
                all_words: List[str] = []
                for idx in cluster_indices:
                    all_words.extend(_content_words(ideas[idx]))
                if all_words:
                    common = Counter(all_words).most_common(2)
                    label = " ".join(w for w, _ in common)
                else:
                    label = f"cluster_{len(clusters)}"
                clusters[label] = [ideas[idx] for idx in cluster_indices]

        return clusters

    def _rank_ideas(
        self,
        ideas: List[str],
        criteria: str = "overall",
    ) -> List[Tuple[str, float]]:
        """
        Rank ideas by a composite quality score or a single criterion.

        Supported criteria: 'overall', 'feasibility', 'specificity',
        'creativity', 'actionability', 'impact'.
        """
        if not ideas:
            return []

        scoring_fn: Callable[[List[str]], float]
        if criteria == "feasibility":
            scoring_fn = self._feasibility_score
        elif criteria == "specificity":
            scoring_fn = self._specificity_score
        elif criteria == "creativity":
            scoring_fn = self._creativity_index
        elif criteria == "actionability":
            scoring_fn = self._actionability_score
        elif criteria == "impact":
            scoring_fn = self._impact_potential
        else:
            # Overall: average of all single-idea metrics
            def _overall_fn(idea_list: List[str]) -> float:
                return float(np.mean([
                    self._feasibility_score(idea_list),
                    self._specificity_score(idea_list),
                    self._creativity_index(idea_list),
                    self._actionability_score(idea_list),
                    self._impact_potential(idea_list),
                ]))
            scoring_fn = _overall_fn

        scored: List[Tuple[str, float]] = []
        for idea in ideas:
            s = scoring_fn([idea])
            scored.append((idea, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self, text: str) -> str:
        """Clean and normalise generated brainstorming output."""
        if not text:
            return ""

        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalise bullet characters to standard dash
        text = re.sub(r"^\s*[•◦▪▸►]\s+", "- ", text, flags=re.MULTILINE)

        # Normalise numbering: "1)" → "1."
        text = re.sub(r"^(\s*\d+)\)\s+", r"\1. ", text, flags=re.MULTILINE)

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Ensure final newline
        if not text.endswith("\n"):
            text += "\n"

        return text

    # ------------------------------------------------------------------
    # Prompt generators
    # ------------------------------------------------------------------

    def _generate_product_prompts(self) -> List[BrainstormPrompt]:
        """Generate product-idea brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="prod_01",
                text="Brainstorm product ideas for sustainable kitchen gadgets.",
                category=BrainstormCategory.PRODUCT_IDEAS,
                topic="Sustainable kitchen gadgets",
                context="Consumers increasingly prefer eco-friendly products. "
                        "The market for sustainable kitchenware has grown 25% YoY.",
                target_audience="Environmentally conscious home cooks aged 25-45",
                existing_solutions=["reusable beeswax wraps", "bamboo utensils", "compost bins"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="prod_02",
                text="Brainstorm wearable tech products for elderly care.",
                category=BrainstormCategory.PRODUCT_IDEAS,
                topic="Wearable technology for elderly care",
                context="Aging populations worldwide need accessible health monitoring.",
                target_audience="Elderly individuals 65+ and their caregivers",
                existing_solutions=["fall-detection watches", "heart rate monitors", "GPS trackers"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="prod_03",
                text="Brainstorm educational toy concepts for children aged 3-6.",
                category=BrainstormCategory.PRODUCT_IDEAS,
                topic="Educational toys for early childhood",
                context="STEM education demand is rising for younger age groups.",
                target_audience="Parents and early childhood educators",
                existing_solutions=["building blocks", "coding robots", "alphabet puzzles"],
                _expected_min_ideas=6,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="prod_04",
                text="Brainstorm smart home products for pet owners.",
                category=BrainstormCategory.PRODUCT_IDEAS,
                topic="Smart home devices for pet care",
                context="Pet ownership is at an all-time high; owners spend more on tech.",
                target_audience="Tech-savvy pet owners aged 25-50",
                existing_solutions=["automatic feeders", "pet cameras", "GPS collars"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="prod_05",
                text="Brainstorm mobile app ideas for mental health support.",
                category=BrainstormCategory.PRODUCT_IDEAS,
                topic="Mobile apps for mental health and wellbeing",
                context="Demand for accessible mental health tools is surging post-pandemic.",
                target_audience="Young adults aged 18-35 seeking mental wellness tools",
                existing_solutions=["meditation apps", "therapy chatbots", "mood trackers"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="prod_06",
                text="Brainstorm products for remote team collaboration beyond video calls.",
                category=BrainstormCategory.PRODUCT_IDEAS,
                topic="Next-generation remote collaboration tools",
                context="Remote work is permanent for many companies; video fatigue is real.",
                target_audience="Distributed teams in knowledge-work companies",
                existing_solutions=["Zoom", "Slack", "Miro", "Notion"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.INNOVATIVE,
            ),
        ]

    def _generate_solution_prompts(self) -> List[BrainstormPrompt]:
        """Generate problem-solving brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="sol_01",
                text="Brainstorm solutions to reduce food waste in restaurants.",
                category=BrainstormCategory.SOLUTIONS,
                topic="Reducing food waste in restaurants",
                context="Restaurants waste ~30% of food purchased. Margins are thin.",
                target_audience="Restaurant owners and managers",
                existing_solutions=["smaller portions", "composting", "donation programs"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="sol_02",
                text="Brainstorm ways to reduce traffic congestion in mid-size cities.",
                category=BrainstormCategory.SOLUTIONS,
                topic="Traffic congestion reduction for cities with 200K-1M population",
                context="Growing cities face worsening commute times without subway systems.",
                constraints_text="Budget under $50M; must be implementable within 3 years.",
                existing_solutions=["bus rapid transit", "bike lanes", "congestion pricing"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="sol_03",
                text="Brainstorm solutions for improving voter turnout in local elections.",
                category=BrainstormCategory.SOLUTIONS,
                topic="Increasing voter participation in local elections",
                context="Turnout in local elections averages under 25% in many regions.",
                target_audience="Municipal government officials and civic organisations",
                existing_solutions=["mail-in voting", "early voting", "civic education campaigns"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="sol_04",
                text="Brainstorm solutions for reducing ocean plastic pollution.",
                category=BrainstormCategory.SOLUTIONS,
                topic="Ocean plastic pollution mitigation",
                context="Over 8 million tonnes of plastic enter oceans annually.",
                target_audience="Environmental NGOs and policy makers",
                existing_solutions=["beach cleanups", "plastic bag bans", "recycling programs"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="sol_05",
                text="Brainstorm ways to combat loneliness among university students.",
                category=BrainstormCategory.SOLUTIONS,
                topic="Student loneliness and social isolation at universities",
                context="Loneliness affects ~40% of college students, impacting academic performance.",
                target_audience="University student affairs departments",
                existing_solutions=["orientation programs", "clubs", "counselling services"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="sol_06",
                text="Brainstorm approaches to bridge the digital divide in rural areas.",
                category=BrainstormCategory.SOLUTIONS,
                topic="Digital divide in rural communities",
                context="Many rural areas lack reliable broadband and digital literacy programs.",
                target_audience="Government broadband offices and rural community leaders",
                existing_solutions=["satellite internet", "mobile hotspots", "library programs"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
        ]

    def _generate_creative_use_prompts(self) -> List[BrainstormPrompt]:
        """Generate creative-use brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="cre_01",
                text="Brainstorm creative uses for old smartphones.",
                category=BrainstormCategory.CREATIVE_USES,
                topic="Repurposing old smartphones",
                context="Millions of smartphones are discarded yearly with working components.",
                existing_solutions=["security cameras", "media players", "alarm clocks"],
                _expected_min_ideas=10,
                _expected_complexity=IdeaComplexity.SIMPLE,
            ),
            BrainstormPrompt(
                prompt_id="cre_02",
                text="Brainstorm unusual uses for shipping containers.",
                category=BrainstormCategory.CREATIVE_USES,
                topic="Alternative uses for shipping containers",
                context="Surplus containers are cheap and structurally strong.",
                existing_solutions=["tiny homes", "pop-up shops", "storage units"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="cre_03",
                text="Brainstorm creative applications of drone technology beyond delivery.",
                category=BrainstormCategory.CREATIVE_USES,
                topic="Novel applications of consumer drones",
                context="Drone costs have dropped significantly; regulations are evolving.",
                existing_solutions=["aerial photography", "crop monitoring", "package delivery"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.INNOVATIVE,
            ),
            BrainstormPrompt(
                prompt_id="cre_04",
                text="Brainstorm creative uses for a large language model in education.",
                category=BrainstormCategory.CREATIVE_USES,
                topic="Creative LLM applications in education",
                context="LLMs can generate text, answer questions, and role-play scenarios.",
                target_audience="K-12 teachers and curriculum designers",
                existing_solutions=["tutoring chatbots", "essay feedback", "quiz generation"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="cre_05",
                text="Brainstorm unconventional uses for 3D printing technology.",
                category=BrainstormCategory.CREATIVE_USES,
                topic="Unconventional 3D printing applications",
                context="3D printers can now print in metal, ceramic, food, and bio-materials.",
                existing_solutions=["prototyping", "custom prosthetics", "architectural models"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.INNOVATIVE,
            ),
            BrainstormPrompt(
                prompt_id="cre_06",
                text="Brainstorm creative ways to use blockchain beyond cryptocurrency.",
                category=BrainstormCategory.CREATIVE_USES,
                topic="Non-financial blockchain applications",
                context="Blockchain provides immutable, decentralised record-keeping.",
                existing_solutions=["supply chain tracking", "digital identity", "NFTs"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
        ]

    def _generate_strategy_prompts(self) -> List[BrainstormPrompt]:
        """Generate strategy brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="str_01",
                text="Brainstorm strategies for a small bookstore to compete with Amazon.",
                category=BrainstormCategory.STRATEGIES,
                topic="Independent bookstore competitive strategy",
                context="Online giants dominate; local bookstores must find unique value.",
                target_audience="Small bookstore owners",
                existing_solutions=["author events", "curated selections", "loyalty programs"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="str_02",
                text="Brainstorm growth strategies for a SaaS startup with $1M ARR.",
                category=BrainstormCategory.STRATEGIES,
                topic="SaaS startup growth from $1M to $10M ARR",
                context="Product-market fit established; need to scale efficiently.",
                constraints_text="Limited budget; team of 15 people.",
                existing_solutions=["content marketing", "free tier", "partner integrations"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="str_03",
                text="Brainstorm strategies for a university to attract international students.",
                category=BrainstormCategory.STRATEGIES,
                topic="International student recruitment for mid-tier universities",
                context="Competition for international students is intensifying globally.",
                target_audience="University admissions and marketing teams",
                existing_solutions=["agent networks", "scholarship programs", "online programs"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="str_04",
                text="Brainstorm retention strategies for a mobile gaming company.",
                category=BrainstormCategory.STRATEGIES,
                topic="Player retention in mobile games",
                context="Average Day-30 retention in mobile games is under 5%.",
                target_audience="Mobile game product managers and designers",
                existing_solutions=["daily rewards", "social features", "seasonal events"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="str_05",
                text="Brainstorm strategies for reducing employee burnout in tech companies.",
                category=BrainstormCategory.STRATEGIES,
                topic="Employee burnout prevention in technology organisations",
                context="Tech worker burnout rates exceed 50% in surveys.",
                target_audience="HR leaders and engineering managers",
                existing_solutions=["flexible hours", "mental health benefits", "no-meeting days"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
        ]

    def _generate_name_prompts(self) -> List[BrainstormPrompt]:
        """Generate naming brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="nam_01",
                text="Brainstorm names for a new plant-based protein brand.",
                category=BrainstormCategory.NAMES,
                topic="Plant-based protein brand naming",
                context="Brand should convey natural, healthy, and modern vibes.",
                target_audience="Health-conscious millennials and Gen Z",
                constraints_text="Name should be 1-2 words, easy to pronounce globally.",
                _expected_min_ideas=15,
                _expected_complexity=IdeaComplexity.SIMPLE,
            ),
            BrainstormPrompt(
                prompt_id="nam_02",
                text="Brainstorm names for an AI-powered writing assistant.",
                category=BrainstormCategory.NAMES,
                topic="AI writing assistant product name",
                context="The tool helps professionals write emails, reports, and proposals.",
                constraints_text="Must not conflict with existing trademarks.",
                _expected_min_ideas=15,
                _expected_complexity=IdeaComplexity.SIMPLE,
            ),
            BrainstormPrompt(
                prompt_id="nam_03",
                text="Brainstorm names for a community co-working space in a small town.",
                category=BrainstormCategory.NAMES,
                topic="Community co-working space naming",
                context="Space serves freelancers, remote workers, and local entrepreneurs.",
                target_audience="Rural and small-town knowledge workers",
                _expected_min_ideas=12,
                _expected_complexity=IdeaComplexity.SIMPLE,
            ),
            BrainstormPrompt(
                prompt_id="nam_04",
                text="Brainstorm names for a podcast about the history of mathematics.",
                category=BrainstormCategory.NAMES,
                topic="Mathematics history podcast naming",
                context="Episodes cover stories of mathematicians and their discoveries.",
                target_audience="Curious adults who enjoy science communication",
                _expected_min_ideas=12,
                _expected_complexity=IdeaComplexity.SIMPLE,
            ),
        ]

    def _generate_improvement_prompts(self) -> List[BrainstormPrompt]:
        """Generate improvement brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="imp_01",
                text="Brainstorm improvements to the airport passenger experience.",
                category=BrainstormCategory.IMPROVEMENTS,
                topic="Airport passenger experience improvement",
                context="Airports are consistently rated as stressful by travellers.",
                target_audience="Airport operations managers",
                existing_solutions=["mobile boarding passes", "automated bag drop", "lounges"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="imp_02",
                text="Brainstorm improvements to online grocery shopping.",
                category=BrainstormCategory.IMPROVEMENTS,
                topic="Online grocery shopping UX improvements",
                context="Online grocery has grown but still frustrates many users.",
                target_audience="E-commerce product teams at grocery retailers",
                existing_solutions=["predictive reordering", "same-day delivery", "substitution choices"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="imp_03",
                text="Brainstorm improvements to public library services.",
                category=BrainstormCategory.IMPROVEMENTS,
                topic="Modernising public library services",
                context="Libraries compete with digital alternatives for community relevance.",
                target_audience="Public library directors and boards",
                existing_solutions=["e-book lending", "maker spaces", "community rooms"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.MODERATE,
            ),
            BrainstormPrompt(
                prompt_id="imp_04",
                text="Brainstorm improvements to the code review process in software teams.",
                category=BrainstormCategory.IMPROVEMENTS,
                topic="Software code review process improvements",
                context="Code reviews are valuable but often slow and inconsistent.",
                target_audience="Engineering leads and developer experience teams",
                existing_solutions=["PR templates", "automated linting", "review rotation"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
        ]

    def _generate_research_prompts(self) -> List[BrainstormPrompt]:
        """Generate research question brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="res_01",
                text="Brainstorm research questions about the impact of social media on teen mental health.",
                category=BrainstormCategory.RESEARCH_QUESTIONS,
                topic="Social media and adolescent mental health",
                context="Correlational studies exist but causal mechanisms remain unclear.",
                target_audience="Psychology and public health researchers",
                existing_solutions=["screen time surveys", "longitudinal cohort studies"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="res_02",
                text="Brainstorm research questions about AI alignment and safety.",
                category=BrainstormCategory.RESEARCH_QUESTIONS,
                topic="AI alignment research directions",
                context="As AI systems become more capable, alignment becomes critical.",
                target_audience="AI safety researchers",
                existing_solutions=["RLHF", "constitutional AI", "interpretability"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.INNOVATIVE,
            ),
            BrainstormPrompt(
                prompt_id="res_03",
                text="Brainstorm research questions about urban heat islands and climate adaptation.",
                category=BrainstormCategory.RESEARCH_QUESTIONS,
                topic="Urban heat island mitigation research",
                context="Cities are warming faster than surrounding areas; vulnerable populations are at risk.",
                target_audience="Urban planning and environmental science researchers",
                existing_solutions=["green roofs", "cool pavements", "urban tree canopy studies"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="res_04",
                text="Brainstorm research questions about the future of work and automation.",
                category=BrainstormCategory.RESEARCH_QUESTIONS,
                topic="Automation impact on labour markets",
                context="AI and robotics are transforming job markets across sectors.",
                target_audience="Labour economists and policy researchers",
                existing_solutions=["task-based analysis", "historical analogy studies"],
                _expected_min_ideas=8,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
        ]

    def _generate_business_model_prompts(self) -> List[BrainstormPrompt]:
        """Generate business model brainstorming prompts."""
        return [
            BrainstormPrompt(
                prompt_id="biz_01",
                text="Brainstorm business models for a hyperlocal news platform.",
                category=BrainstormCategory.BUSINESS_MODELS,
                topic="Hyperlocal news platform monetisation",
                context="Local journalism is declining; communities need local information.",
                target_audience="Media entrepreneurs and civic tech founders",
                existing_solutions=["subscription model", "ad-supported", "nonprofit grants"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="biz_02",
                text="Brainstorm business models for an open-source developer tools company.",
                category=BrainstormCategory.BUSINESS_MODELS,
                topic="Open-source developer tools monetisation",
                context="Building a sustainable business on top of an open-source project.",
                target_audience="Open-source project maintainers considering commercialisation",
                existing_solutions=["hosted SaaS", "enterprise support", "open-core"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.COMPLEX,
            ),
            BrainstormPrompt(
                prompt_id="biz_03",
                text="Brainstorm business models for a vertical farming startup.",
                category=BrainstormCategory.BUSINESS_MODELS,
                topic="Vertical farming business model innovation",
                context="Indoor farming can produce year-round but has high capex.",
                constraints_text="Must achieve unit economics within 3 years.",
                existing_solutions=["direct-to-consumer", "restaurant supply", "grocery wholesale"],
                _expected_min_ideas=7,
                _expected_complexity=IdeaComplexity.INNOVATIVE,
            ),
        ]

    # ------------------------------------------------------------------
    # Prompt retrieval utilities
    # ------------------------------------------------------------------

    def get_prompts_by_category(
        self, category: BrainstormCategory
    ) -> List[BrainstormPrompt]:
        """Return all prompts matching a given category."""
        dataset = self.load_prompts()
        return [
            p for p in dataset.prompts
            if isinstance(p, BrainstormPrompt) and p.category == category
        ]

    def get_prompt_by_id(self, prompt_id: str) -> Optional[BrainstormPrompt]:
        """Look up a single prompt by its ID."""
        dataset = self.load_prompts()
        for p in dataset.prompts:
            if isinstance(p, BrainstormPrompt) and p.prompt_id == prompt_id:
                return p
        return None

    def sample_prompts(
        self,
        n: int = 10,
        category: Optional[BrainstormCategory] = None,
    ) -> List[BrainstormPrompt]:
        """Randomly sample *n* prompts, optionally filtered by category."""
        pool = (
            self.get_prompts_by_category(category)
            if category
            else self.load_prompts().prompts
        )
        n = min(n, len(pool))
        indices = self._rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in indices]

    # ------------------------------------------------------------------
    # Aggregate analysis helpers
    # ------------------------------------------------------------------

    def compare_idea_sets(
        self,
        ideas_a: List[str],
        ideas_b: List[str],
    ) -> Dict[str, float]:
        """
        Compare two sets of ideas (e.g. from two different decoding strategies)
        and return similarity/divergence metrics.
        """
        tokens_a = [set(_content_words(i)) for i in ideas_a]
        tokens_b = [set(_content_words(i)) for i in ideas_b]

        # Cross-set similarity: average max-similarity from A→B and B→A
        def _avg_max_sim(
            src: List[Set[str]], tgt: List[Set[str]]
        ) -> float:
            if not src or not tgt:
                return 0.0
            sims: List[float] = []
            for s in src:
                max_sim = max(_jaccard(s, t) for t in tgt)
                sims.append(max_sim)
            return float(np.mean(sims))

        a_to_b = _avg_max_sim(tokens_a, tokens_b)
        b_to_a = _avg_max_sim(tokens_b, tokens_a)
        cross_sim = (a_to_b + b_to_a) / 2.0

        # Unique idea ratio: fraction of ideas in A not well-matched in B
        novel_a = sum(
            1 for s in tokens_a
            if all(_jaccard(s, t) < 0.3 for t in tokens_b)
        )
        novel_b = sum(
            1 for s in tokens_b
            if all(_jaccard(s, t) < 0.3 for t in tokens_a)
        )
        total = max(len(ideas_a) + len(ideas_b), 1)

        return {
            "cross_similarity": cross_sim,
            "unique_a_ratio": novel_a / max(len(ideas_a), 1),
            "unique_b_ratio": novel_b / max(len(ideas_b), 1),
            "combined_novelty_ratio": (novel_a + novel_b) / total,
            "diversity_a": self._diversity_of_ideas(ideas_a),
            "diversity_b": self._diversity_of_ideas(ideas_b),
        }

    def evaluate_single(
        self,
        generation: str,
        prompt: BrainstormPrompt,
    ) -> Dict[str, Any]:
        """Convenience: evaluate a single generation."""
        result = self.evaluate([generation], [prompt])
        return result["per_sample"][0]

    # ------------------------------------------------------------------
    # Detailed per-idea analysis
    # ------------------------------------------------------------------

    def analyze_ideas(
        self,
        text: str,
        prompt: BrainstormPrompt,
    ) -> List[Dict[str, Any]]:
        """
        Return a per-idea breakdown with all scoring dimensions.
        Useful for fine-grained comparison between decoding strategies.
        """
        ideas = self._parse_ideas(text)
        analysis: List[Dict[str, Any]] = []

        for idx, idea in enumerate(ideas):
            single = [idea]
            entry: Dict[str, Any] = {
                "index": idx,
                "text": idea,
                "word_count": len(idea.split()),
                "feasibility": self._feasibility_score(single),
                "specificity": self._specificity_score(single),
                "actionability": self._actionability_score(single),
                "impact": self._impact_potential(single),
                "creativity": self._creativity_index(single),
                "elaboration": self._elaboration_score(single),
                "relevance": self._relevance_score(single, prompt.topic),
            }
            # Novelty relative to known solutions
            if prompt.existing_solutions:
                entry["novelty"] = self._novelty_score(
                    single, prompt.existing_solutions
                )
            else:
                entry["novelty"] = None
            analysis.append(entry)

        return analysis

    def idea_quality_histogram(
        self,
        text: str,
        prompt: BrainstormPrompt,
        bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Generate histograms of per-idea scores for each dimension.
        Returns {dimension: counts_array}.
        """
        per_idea = self.analyze_ideas(text, prompt)
        dimensions = [
            "feasibility", "specificity", "actionability",
            "impact", "creativity", "elaboration", "relevance",
        ]
        histograms: Dict[str, np.ndarray] = {}
        for dim in dimensions:
            values = [
                entry[dim] for entry in per_idea
                if entry.get(dim) is not None
            ]
            if values:
                counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
                histograms[dim] = counts
            else:
                histograms[dim] = np.zeros(bins, dtype=int)
        return histograms

    # ------------------------------------------------------------------
    # Text statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _unique_word_ratio(text: str) -> float:
        words = _tokenize(text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def _avg_sentence_length(text: str) -> float:
        sentences = _sentence_split(text)
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return float(np.mean(lengths))

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary_report(
        self,
        results: Dict[str, Any],
    ) -> str:
        """
        Produce a human-readable summary of evaluation results.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("BRAINSTORMING TASK EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Samples evaluated : {results['num_samples']}")
        lines.append(f"Mean score        : {results['mean_score']:.4f}")
        lines.append(f"Std score         : {results['std_score']:.4f}")
        lines.append(f"Median score      : {results['median_score']:.4f}")
        lines.append(f"Min / Max         : {results['min_score']:.4f} / {results['max_score']:.4f}")
        lines.append(f"Mean idea count   : {results['mean_idea_count']:.1f}")
        lines.append("")

        # Per-dimension averages
        dims = [
            "novelty", "feasibility", "specificity", "diversity",
            "relevance", "elaboration", "category_coverage",
            "actionability", "impact", "creativity",
            "redundancy_penalty", "structural_quality",
        ]
        lines.append("Per-dimension averages:")
        lines.append("-" * 40)
        for dim in dims:
            values = [
                s[dim] for s in results["per_sample"]
                if dim in s
            ]
            if values:
                avg = float(np.mean(values))
                lines.append(f"  {dim:<25s}: {avg:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BrainstormingTask(category={self.config.category!r}, "
            f"min_ideas={self.config.min_ideas}, "
            f"max_ideas={self.config.max_ideas})"
        )
